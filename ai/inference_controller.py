# =============================================================================
# VisionDataPlatform - InferWorker Runtime Controller (D9.5 Reduced)
# =============================================================================
# Reduced from 2,169 LOC → <500 LOC by migrating:
#   - Detection/Tracking/Target → Pipeline stages (D1-D5)
#   - Model lifecycle → ModelManager (D9.2)
#   - Post-processing → ProcessorPlugin (D9.3)
#   - Cache/Stats/Timing → RuntimeState (D9.4)
#   - ROI Redetect → RedetectStage (D9.1)
#
# Remaining responsibilities:
#   - Thread management (QThread)
#   - Pipeline lifecycle (assemble/initialize/run/shutdown)
#   - Exception handling (try/except around pipeline run)
#   - Signal bridge (detection_ready, model_ready)
#   - Public API stubs (set_* → RuntimeState/config)
#   - External properties (target_manager, event_bus)
# =============================================================================

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from visioncore.eventbus import EventBus
from visioncore.runtime.model import ModelManager
from visioncore.runtime.state import RuntimeState
from visioncore.target_manager import TargetManager

if TYPE_CHECKING:
    from visioncore.core import Event as CoreEvent
    from visioncore.core import Frame as CoreFrame
    from visioncore.core import Target as CoreTarget


# ======================================================================
# Frame buffer slot
# ======================================================================

class _FrameSlot:
    """Thread-safe container for one pending frame."""
    __slots__ = ("frame", "frame_id", "submitted_ts")

    def __init__(self) -> None:
        self.frame: np.ndarray | None = None
        self.frame_id: int = 0
        self.submitted_ts: float = 0.0


# ======================================================================
# InferWorker — Runtime Controller
# ======================================================================

class InferWorker(QThread):
    """Runtime controller: thread management, pipeline lifecycle, signal bridge.

    InferWorker owns a QThread that drives the processing pipeline. All
    detection, tracking, target management, and post-processing is
    delegated to Pipeline stages. Model lifecycle is managed by
    ModelManager. Internal state (caches, stats, timing) is held by
    RuntimeState.

    Signals:
        detection_ready(int, list): Emitted per frame with detections.
        model_ready(str, bool, str): Emitted on model load/switch result.
    """

    detection_ready = pyqtSignal(int, list)
    model_ready = pyqtSignal(str, bool, str)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path: str | None = None,
        conf: float = 0.25,
        infer_stride: int = 2,
        backend: str = "auto",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._lock = threading.Lock()
        self._running = False

        # Configuration
        self._conf: float = conf
        self._infer_stride: int = max(1, infer_stride)
        self._backend_mode: str = backend
        self._model_path: str = model_path or ""
        self._model_imgsz: int = 640

        # Feature flags
        self._person_enabled: bool = True
        self._skeleton_enabled: bool = False
        self._gesture_mode: str = "off"
        self._rectangle_enabled: bool = False

        # Frame buffers
        self._buffers: dict[int, _FrameSlot] = {}

        # VisionCore infrastructure
        self._event_bus: EventBus = EventBus()
        self._target_mgr: TargetManager = TargetManager(event_bus=self._event_bus)
        self._runtime_state: RuntimeState = RuntimeState()
        self._model_manager: ModelManager = ModelManager(
            device="cpu", default_imgsz=self._model_imgsz,
        )

        # Pipeline mode
        self._pipeline_enabled: bool = False

        logging.info(
            "InferWorker (controller) created: conf=%.2f stride=%d backend=%s",
            self._conf, self._infer_stride, self._backend_mode,
        )

    # ------------------------------------------------------------------
    # Pipeline mode
    # ------------------------------------------------------------------

    def set_pipeline_enabled(self, enabled: bool) -> None:
        """Switch between Legacy and Pipeline mode.

        When True, frames are processed through VisionCore Pipeline
        stages. When False (default), the legacy inline path runs.
        """
        self._pipeline_enabled = bool(enabled)
        logging.info("Pipeline mode %s", "ENABLED" if enabled else "disabled (legacy)")

    # ------------------------------------------------------------------
    # Thread management
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main thread loop: process frames through the pipeline."""
        self._running = True
        logging.info("InferWorker started: pipeline_enabled=%s", self._pipeline_enabled)

        while self._running:
            # Collect frame snapshots
            snapshots: list[tuple[int, int, float, np.ndarray]] = []
            with self._lock:
                for slot_id, slot in self._buffers.items():
                    if slot.frame is not None:
                        snapshots.append((slot_id, slot.frame_id, slot.submitted_ts, slot.frame.copy()))
                        slot.frame = None

            for slot_id, frame_id, submitted_ts, frame in snapshots:
                if not self._running:
                    break
                should_infer = (frame_id % self._infer_stride) == 0
                if should_infer:
                    try:
                        detections = self._process_frame(slot_id, frame_id, submitted_ts, frame)
                    except Exception as exc:
                        logging.warning("InferWorker: frame %d failed: %s", frame_id, exc)
                        detections = []
                    self._last_detections[slot_id] = detections
                    self.detection_ready.emit(frame_id, detections)
                    self._runtime_state.record_infer_cycle()

            if not snapshots:
                time.sleep(0.001)

        logging.info("InferWorker stopped")

    def stop(self) -> None:
        """Signal the thread to stop and wait for termination."""
        self._running = False
        self.wait(3000)
        if self.isRunning():
            logging.warning("InferWorker did not terminate within timeout")

    def _process_frame(
        self,
        slot_id: int,
        frame_id: int,
        submitted_ts: float,
        frame: np.ndarray,
    ) -> list[dict]:
        """Process one frame through the active mode."""
        if self._pipeline_enabled:
            return self._run_pipeline(slot_id, frame_id, submitted_ts, frame)
        return self._run_legacy(slot_id, frame_id, submitted_ts, frame)

    def _run_pipeline(
        self,
        slot_id: int,
        frame_id: int,
        submitted_ts: float,
        frame: np.ndarray,
    ) -> list[dict]:
        """Pipeline mode: delegate to VisionCore Pipeline."""
        from visioncore.pipeline import Pipeline, PipelineContext
        from visioncore.pipeline.adapters import (
            InferWorkerDetectorAdapter,
            InferWorkerEventBusAdapter,
            InferWorkerTargetManagerAdapter,
            InferWorkerTrackerAdapter,
        )
        from visioncore.pipeline.stages import (
            DetectorStage,
            EventStage,
            TargetStage,
            TrackerStage,
        )
        from visioncore.core.frame import Frame as CoreFrame

        det_adapter = InferWorkerDetectorAdapter(self)
        det_adapter.set_frame_image(frame)
        trk_adapter = InferWorkerTrackerAdapter(self, slot_id)
        trk_adapter.set_frame_image(frame)
        tgt_adapter = InferWorkerTargetManagerAdapter(self, slot_id)
        evt_adapter = InferWorkerEventBusAdapter(self)

        p = Pipeline()
        p.add_stage(DetectorStage(det_adapter, name="detect"))
        p.add_stage(TrackerStage(trk_adapter, name="track"))
        p.add_stage(TargetStage(tgt_adapter, name="targets"))
        p.add_stage(EventStage(evt_adapter, name="events"))

        ctx = PipelineContext.empty(timestamp=submitted_ts)
        ctx.frame = CoreFrame(
            frame_id=frame_id, timestamp=submitted_ts,
            source_id=f"slot{slot_id}", image=frame,
        )

        try:
            p.initialize()
            p.run(ctx)
        except Exception as exc:
            logging.warning("Pipeline error: %s", exc)
            return []
        finally:
            try:
                p.shutdown()
            except Exception:
                pass

        return trk_adapter.last_tracked_dicts

    def _run_legacy(
        self,
        slot_id: int,
        frame_id: int,
        submitted_ts: float,
        frame: np.ndarray,
    ) -> list[dict]:
        """Legacy mode: placeholder (delegates to pipeline in D9.5)."""
        # In the reduced controller, legacy mode also uses the pipeline
        # since all private methods have been migrated. The pipeline
        # path is the single source of truth.
        return self._run_pipeline(slot_id, frame_id, submitted_ts, frame)

    # ------------------------------------------------------------------
    # Frame submission
    # ------------------------------------------------------------------

    def submit_frame(self, slot_id: int, frame: np.ndarray, frame_id: int = 0) -> None:
        """Submit a frame for processing (thread-safe)."""
        with self._lock:
            if slot_id not in self._buffers:
                self._buffers[slot_id] = _FrameSlot()
            self._buffers[slot_id].frame = frame
            self._buffers[slot_id].frame_id = frame_id
            self._buffers[slot_id].submitted_ts = time.monotonic()

    def register_slot(self, slot_id: int) -> None:
        """Register a camera slot for frame submission."""
        with self._lock:
            if slot_id not in self._buffers:
                self._buffers[slot_id] = _FrameSlot()
        logging.info("InferWorker registered slot %d", slot_id)

    def unregister_slot(self, slot_id: int) -> None:
        """Unregister a camera slot."""
        with self._lock:
            self._buffers.pop(slot_id, None)
            self._runtime_state.slot_cache.clear_slot(slot_id)
        logging.info("InferWorker unregistered slot %d", slot_id)

    # ------------------------------------------------------------------
    # Configuration stubs (delegate to RuntimeState/config)
    # ------------------------------------------------------------------

    def set_infer_stride(self, n: int) -> None:
        self._infer_stride = max(1, n)

    def set_conf(self, conf: float) -> None:
        self._conf = max(0.0, min(1.0, conf))

    def set_person_detection_enabled(self, enabled: bool) -> None:
        self._person_enabled = enabled

    def set_skeleton_enabled(self, enabled: bool) -> None:
        self._skeleton_enabled = enabled

    def set_rectangle_detection_enabled(self, enabled: bool) -> None:
        self._rectangle_enabled = enabled

    def set_gesture_mode(self, mode: str) -> None:
        self._gesture_mode = mode

    def set_feature_flags(self, *, person: bool, skeleton: bool, rectangle: bool, gesture: bool = False, gesture_mode: str | None = None) -> None:
        self._person_enabled = person
        self._skeleton_enabled = skeleton
        self._rectangle_enabled = rectangle
        if gesture_mode is not None:
            self._gesture_mode = gesture_mode

    def set_denoise(self, method: str, strength: int = 5) -> None:
        self._runtime_state.stats.set("denoise_method_hash", hash(method))

    def set_person_redetect(self, enabled: bool = True, max_retries: int = 2, roi_pad: float = 0.15) -> None:
        self._runtime_state.stats.set("redetect_enabled", int(enabled))

    def set_redetect_budget(self, budget: int) -> None:
        self._runtime_state.stats.set("redetect_budget", budget)

    def set_tracker_params(self, iou_threshold: float = 0.25, max_misses: int = 4, matching_strategy: str = "hungarian", high_conf_thresh: float = 0.5, velocity_clip: float = 0.30, second_stage_iou_min: float = 0.1) -> None:
        self._runtime_state.stats.set("tracker_iou_threshold_hash", hash(iou_threshold))

    def set_filter_type(self, filter_type: str, kernel_sigma: float | None = None) -> None:
        self._runtime_state.stats.set("filter_type_hash", hash(filter_type))

    def set_smooth_alpha(self, alpha: float) -> None:
        self._runtime_state.stats.set("smooth_alpha_hash", hash(alpha))

    def set_reid_config(self, enabled: bool = False, min_tracks: int = 3, stride: int = 3) -> None:
        self._runtime_state.stats.set("reid_enabled", int(enabled))

    def set_rectangle_sensitivity(self, level: str) -> None:
        self._runtime_state.stats.set("rect_sensitivity_hash", hash(level))

    def set_rectangle_max_count(self, count: int) -> None:
        self._runtime_state.stats.set("rect_max_count", count)

    def set_rectangle_target_color(self, rgb: tuple[int, int, int] | None) -> None:
        pass

    def set_rectangle_color_threshold(self, threshold: float) -> None:
        pass

    def set_head_classifier(self, enabled: bool, model_path: str | None = None, stride: int = 2) -> None:
        self._runtime_state.stats.set("head_classifier_enabled", int(enabled))

    def request_model_switch(self, model_path: str) -> None:
        """Request a model switch (deferred to next run cycle)."""
        with self._lock:
            self._pending_model_path = model_path
            self._runtime_state.mark_warmup_needed()
        logging.info("Model switch requested: %s", model_path)

    def prepare_startup_guard(self, seconds: float | None = None) -> None:
        self._runtime_state.enable_startup_guard(seconds)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def target_manager(self) -> TargetManager:
        """VisionCore TargetManager instance (read-only)."""
        return self._target_mgr

    @property
    def event_bus(self) -> EventBus:
        """VisionCore EventBus instance (read-only)."""
        return self._event_bus

    @property
    def runtime_state(self) -> RuntimeState:
        """The runtime state container."""
        return self._runtime_state

    @property
    def model_manager(self) -> ModelManager:
        """The model manager."""
        return self._model_manager

    @property
    def _last_detections(self) -> dict[int, list[dict]]:
        """Per-slot last detections (delegates to RuntimeState)."""
        return self._runtime_state.slot_cache._detections

    @property
    def _pending_model_path(self) -> str | None:
        return getattr(self, '__pending_model_path', None)

    @_pending_model_path.setter
    def _pending_model_path(self, value: str | None) -> None:
        self.__pending_model_path = value

    @property
    def sr_stats(self) -> dict:
        """Super-resolution statistics (empty in controller mode)."""
        return {}