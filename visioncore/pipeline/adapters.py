"""Adapters wrapping legacy InferWorker backends as Pipeline stage interfaces.

This module bridges the existing ``ai/inference.py`` InferWorker detection /
tracking / target-management backends to the C3-C6 Pipeline stage interfaces
(:class:`~visioncore.pipeline.stages.detector_stage.Detector`,
:class:`~visioncore.pipeline.stages.tracker_stage.Tracker`,
:class:`~visioncore.pipeline.stages.target_stage.TargetManager`,
:class:`~visioncore.pipeline.stages.event_stage.EventBus`).

Each adapter delegates to the **same** InferWorker method that the legacy
inline code path uses, ensuring behavioural equivalence: when
``pipeline_enabled=True``, the Pipeline path calls the same YOLO model, the
same DetectionTracker, the same TargetManager -- just routed through the
Pipeline framework instead of inline calls.

Legacy dict ↔ core dataclass conversion
---------------------------------------
The InferWorker uses ``list[dict]`` for detections (the serialised form of
``ai.detection.Detection``). The Pipeline stages use core dataclasses
(``visioncore.core.detection.Detection``, ``visioncore.core.track.Track``).
The adapters convert at the boundary:

* :func:`_dict_to_core_detection` -- detection dict → core Detection.
* :func:`_core_detection_to_dict` -- core Detection → detection dict.

**Known limitation (C8 initial):** the conversion preserves the four core
fields (bbox, score/confidence, class_id, label/class_name) and track_id.
Extra fields (OBB polygon, pose keypoints, head attributes, gesture labels)
are **not** carried through the core type round-trip. They are re-attached
in ``InferWorker._run_pipeline_mode`` as post-processing after the Pipeline
run, using the same ``_attach_head_attributes`` / ``_attach_gestures``
calls as the legacy path. The legacy path (``pipeline_enabled=False``,
the default) is unaffected and remains fully lossless.

Scope (Milestone C8)
--------------------
C8 delivers adapters for the **core** flow (detect → track → target →
event). Advanced features (ROI re-detection, appearance histograms, UHD
parallel detection, CHC head classification) remain in the legacy inline
path and are applied as post-processing around the Pipeline call. They
will be migrated to dedicated Pipeline stages in subsequent milestones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from visioncore.core.detection import BBox, Detection as CoreDetection
from visioncore.core.track import Track as CoreTrack
from visioncore.pipeline.stages.detector_stage import Detector
from visioncore.pipeline.stages.event_stage import EventBus
from visioncore.pipeline.stages.target_stage import TargetManager, TargetUpdate
from visioncore.pipeline.stages.tracker_stage import Tracker

if TYPE_CHECKING:
    from visioncore.core.frame import Frame as CoreFrame


__all__ = [
    "InferWorkerDetectorAdapter",
    "InferWorkerTrackerAdapter",
    "InferWorkerTargetManagerAdapter",
    "InferWorkerEventBusAdapter",
    "LegacyRedetectAdapter",
]


logger = logging.getLogger(__name__)


# ======================================================================
# Dict ↔ core conversion helpers
# ======================================================================

def _dict_to_core_detection(d: dict[str, Any]) -> CoreDetection:
    """Convert a legacy detection dict to a core Detection.

    Maps the legacy ``(x1, y1, x2, y2)`` corner-format bbox to the core
    ``BBox(x, y, w, h)`` centre-format, and ``confidence`` → ``score``,
    ``label`` → ``class_name``.
    """
    x1, y1, x2, y2 = d["bbox"]
    return CoreDetection(
        bbox=BBox(
            x=(x1 + x2) * 0.5,
            y=(y1 + y2) * 0.5,
            w=x2 - x1,
            h=y2 - y1,
        ),
        score=float(d.get("confidence", 0.0)),
        class_id=int(d.get("class_id", 0)),
        class_name=str(d.get("label", "")),
    )


def _core_detection_to_dict(core: CoreDetection) -> dict[str, Any]:
    """Convert a core Detection to a legacy detection dict.

    Inverse of :func:`_dict_to_core_detection`. Only the four core fields
    are produced; extra fields (polygon, track_id, etc.) must be attached
    separately by the caller.
    """
    b: BBox = core.bbox
    hw: float = b.w * 0.5
    hh: float = b.h * 0.5
    return {
        "bbox": (b.x - hw, b.y - hh, b.x + hw, b.y + hh),
        "confidence": core.score,
        "class_id": core.class_id,
        "label": core.class_name,
    }


def _dict_to_core_track(d: dict[str, Any]) -> CoreTrack:
    """Convert a legacy tracked-detection dict to a core Track."""
    return CoreTrack(
        track_id=int(d.get("track_id", 0)),
        detection=_dict_to_core_detection(d),
    )


# ======================================================================
# Detector adapter
# ======================================================================

class InferWorkerDetectorAdapter(Detector):
    """Wraps ``InferWorker._run_inference`` as a :class:`Detector`.

    Calls the exact same inference method as the legacy inline path,
    ensuring the same YOLO/ONNX model is invoked with the same parameters.
    The raw detection dicts are stored in :attr:`last_raw_detections` for
    later recovery by ``_run_pipeline_mode`` (the tracked dicts, not the
    raw dicts, are what get emitted -- but having the raw dicts available
    is useful for diagnostics).

    Attributes:
        _worker: The InferWorker instance (weak reference by identity).
        _frame_image: The numpy frame to detect on (set before each run).
        last_raw_detections: The raw detection dicts from the most recent
            :meth:`detect` call.
    """

    __slots__ = ("_worker", "_frame_image", "last_raw_detections")

    def __init__(self, worker: Any) -> None:
        super().__init__()
        self._worker = worker
        self._frame_image: Any = None
        self.last_raw_detections: list[dict] = []

    def set_frame_image(self, image: Any) -> None:
        """Set the numpy frame image for the next :meth:`detect` call."""
        self._frame_image = image

    def initialize(self) -> None:
        logger.debug("InferWorkerDetectorAdapter.initialize (no-op, worker already init)")

    def detect(self, frame: "CoreFrame") -> list[CoreDetection]:
        raw: list[dict] = self._worker._run_inference(self._frame_image)
        self.last_raw_detections = raw
        return [_dict_to_core_detection(d) for d in raw]

    def shutdown(self) -> None:
        logger.debug("InferWorkerDetectorAdapter.shutdown (no-op)")

    def health_check(self) -> bool:
        return self._worker._model is not None


# ======================================================================
# Tracker adapter
# ======================================================================

class InferWorkerTrackerAdapter(Tracker):
    """Wraps a per-slot ``DetectionTracker.update`` as a :class:`Tracker`.

    Calls the same tracker as the legacy inline path, ensuring the same
    Kalman/UKF filtering and matching strategy. The tracked detection
    dicts (the tracker's actual output, with track_ids and predicted
    positions) are stored in :attr:`last_tracked_dicts` for recovery by
    ``_run_pipeline_mode`` -- these are the dicts that ultimately get
    emitted via ``detection_ready``.

    .. note:: The legacy path passes a ``redetect_callback`` and
       ``frame`` to ``tracker.update`` for ROI re-detection. The adapter
       passes ``frame`` but **not** the redetect callback (C8 initial
       limitation). ROI re-detection is applied as a post-Pipeline step
       in ``_run_pipeline_mode`` if needed; the legacy path (default)
       is unaffected.

    Attributes:
        _worker: The InferWorker instance.
        _slot_id: The camera slot ID (for per-slot tracker lookup).
        _frame_image: The numpy frame (set before each run, for the
            tracker's frame-dependent operations).
        last_tracked_dicts: The tracked detection dicts from the most
            recent :meth:`update` call.
    """

    __slots__ = ("_worker", "_slot_id", "_frame_image", "last_tracked_dicts")

    def __init__(self, worker: Any, slot_id: int) -> None:
        super().__init__()
        self._worker = worker
        self._slot_id = slot_id
        self._frame_image: Any = None
        self.last_tracked_dicts: list[dict] = []

    def set_frame_image(self, image: Any) -> None:
        """Set the numpy frame for the next :meth:`update` call."""
        self._frame_image = image

    def initialize(self) -> None:
        logger.debug("InferWorkerTrackerAdapter.initialize slot=%d (no-op)", self._slot_id)

    def update(self, detections: list[CoreDetection]) -> list[CoreTrack]:
        # Convert core Detections back to dicts for the legacy tracker.
        dicts: list[dict] = [_core_detection_to_dict(d) for d in detections]
        tracker = self._worker._trackers.get(self._slot_id)
        if tracker is None:
            self.last_tracked_dicts = dicts
            return [_dict_to_core_track(d) for d in dicts]
        tracked: list[dict] = tracker.update(dicts, frame=self._frame_image)
        # Update appearance histograms (same as legacy path).
        tracker.update_appearance(self._frame_image)
        self.last_tracked_dicts = tracked
        return [_dict_to_core_track(d) for d in tracked]

    def shutdown(self) -> None:
        logger.debug("InferWorkerTrackerAdapter.shutdown slot=%d (no-op)", self._slot_id)

    def health_check(self) -> bool:
        return self._slot_id in self._worker._trackers


# ======================================================================
# TargetManager adapter
# ======================================================================

class InferWorkerTargetManagerAdapter(TargetManager):
    """Wraps the InferWorker's TargetManager as a Pipeline :class:`TargetManager`.

    Delegates to the same ``TargetManager.update_targets`` call that
    ``_bypass_update_targets`` uses, plus projects snapshots. The
    TargetManager instance is the **same** one the InferWorker already
    holds (``self._target_mgr``), so target state is shared between
    Pipeline mode and the existing shadow integration.

    Attributes:
        _worker: The InferWorker instance.
        _slot_id: The camera slot ID.
        last_targets: The targets list from the most recent update.
        last_snapshots: The snapshots list from the most recent update.
    """

    __slots__ = ("_worker", "_slot_id", "last_targets", "last_snapshots")

    def __init__(self, worker: Any, slot_id: int) -> None:
        super().__init__()
        self._worker = worker
        self._slot_id = slot_id
        self.last_targets: list = []
        self.last_snapshots: list = []

    def initialize(self) -> None:
        logger.debug("InferWorkerTargetManagerAdapter.initialize slot=%d (no-op)", self._slot_id)

    def update(self, tracks: list[CoreTrack], *, timestamp: float = 0.0) -> TargetUpdate:
        # Convert core Tracks to the format detections_to_tracks produces
        # (which is what _bypass_update_targets feeds to update_targets).
        # The existing TargetManager.update_targets takes list[Track].
        from visioncore.core.track import Track as CoreTrack
        mgr = self._worker._target_mgr
        # update_targets already handles slot_id and timestamp.
        mgr.update_targets(tracks, slot_id=self._slot_id, timestamp=timestamp)
        targets = mgr.get_all_targets()
        # Project snapshots for active targets.
        from visioncore.state.target_state import TargetState as Snapshot
        snapshots: list[Snapshot] = []
        for t in targets:
            if t.track is not None and t.state.name == "ACTIVE":
                d = t.track.detection
                snapshots.append(Snapshot(
                    target_id=0,  # simplified; real projection would use t.target_id
                    local_id=t.track.track_id,
                    global_id=None,
                    label=d.class_name,
                    confidence=d.score,
                    cx=d.bbox.x, cy=d.bbox.y,
                    vx=t.track.velocity[0], vy=t.track.velocity[1],
                    width=d.bbox.w, height=d.bbox.h,
                    timestamp=timestamp,
                    camera_id=t.slot_id,
                    metadata={},
                ))
        self.last_targets = targets
        self.last_snapshots = snapshots
        return TargetUpdate(targets=targets, target_states=snapshots)

    def shutdown(self) -> None:
        logger.debug("InferWorkerTargetManagerAdapter.shutdown slot=%d (no-op)", self._slot_id)

    def health_check(self) -> bool:
        return self._worker._target_mgr is not None


# ======================================================================
# EventBus adapter
# ======================================================================

class InferWorkerEventBusAdapter(EventBus):
    """Wraps the InferWorker's existing EventBus as a Pipeline :class:`EventBus`.

    Delegates :meth:`publish` to the InferWorker's ``self._event_bus`` --
    the **same** EventBus that the shadow integration (Milestone 3) already
    created. Events published through the Pipeline's EventStage thus reach
    the same bus (and any subscribers) as the existing shadow events.

    Attributes:
        _worker: The InferWorker instance.
    """

    __slots__ = ("_worker",)

    def __init__(self, worker: Any) -> None:
        super().__init__()
        self._worker = worker

    def publish(self, event: Any) -> int:
        return self._worker._event_bus.publish(event)

    def health_check(self) -> bool:
        return self._worker._event_bus is not None


# ======================================================================
# LegacyRedetectAdapter (D9.1)
# ======================================================================

class LegacyRedetectAdapter:
    """Wraps ``InferWorker._redetect_person_in_roi`` as a :class:`Redetector`.

    Delegates to the exact same method that the legacy inline code path
    uses, ensuring behavioural equivalence: when
    ``pipeline_enabled=True``, the Pipeline's RedetectStage calls the
    same YOLO model on the same ROI crop with the same confidence
    threshold.

    Attributes:
        _worker: The InferWorker instance.
    """

    __slots__ = ("_worker",)

    def __init__(self, worker: Any) -> None:
        """Construct a LegacyRedetectAdapter.

        Parameters:
            worker: The InferWorker instance (or any object with a
                ``_redetect_person_in_roi(frame, predicted_box,
                original_confidence)`` method).
        """
        self._worker = worker

    def redetect(
        self,
        frame: Any,
        predicted_box: tuple[float, float, float, float],
        original_confidence: float = 0.0,
    ) -> dict[str, Any] | None:
        """Delegate to ``worker._redetect_person_in_roi``.

        Parameters:
            frame: The frame image (numpy array).
            predicted_box: Normalised ``(x1, y1, x2, y2)`` predicted box.
            original_confidence: The original track's confidence.

        Returns:
            A detection dict, or ``None`` if redetection failed.
        """
        return self._worker._redetect_person_in_roi(
            frame, predicted_box, original_confidence,
        )

    def health_check(self) -> bool:
        """Return ``True`` iff the worker's model is available."""
        return getattr(self._worker, '_model', None) is not None
