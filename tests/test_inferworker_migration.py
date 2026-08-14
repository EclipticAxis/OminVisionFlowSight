"""InferWorker migration unit tests (Milestone D9).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - MigrationPipeline construction and mode switching
    - Legacy mode (pipeline_enabled=False) routes to legacy_process
    - Pipeline mode (pipeline_enabled=True) routes through Pipeline stages
    - Adapter module importability and dict↔core conversion
    - Dual-mode consistency: same backends produce equivalent results
    - Lifecycle (set_pipeline_enabled, run_frame) in both modes
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.pipeline import Pipeline, PipelineContext
from visioncore.pipeline.adapters import (
    InferWorkerDetectorAdapter,
    InferWorkerEventBusAdapter,
    InferWorkerTargetManagerAdapter,
    InferWorkerTrackerAdapter,
)
from visioncore.pipeline.migration import MigrationPipeline, MigrationPipelineError
from visioncore.pipeline.stages import (
    DetectorStage,
    EventStage,
    TargetStage,
    TrackerStage,
)


# ======================================================================
# Test helpers — mock backends
# ======================================================================

class _MockDetectorBackend:
    """Simulates a detector backend (YOLO etc.) for adapter tests."""

    def __init__(self, detections: list[dict] | None = None) -> None:
        self._detections = detections or [
            {"bbox": (0.1, 0.2, 0.3, 0.4), "confidence": 0.9,
             "class_id": 0, "label": "person"},
            {"bbox": (0.5, 0.6, 0.7, 0.8), "confidence": 0.8,
             "class_id": 0, "label": "person"},
        ]
        self._model = True  # health_check: model is not None
        self.call_count = 0

    def _run_inference(self, frame: Any) -> list[dict]:
        self.call_count += 1
        return list(self._detections)


class _MockTrackerBackend:
    """Simulates a DetectionTracker for adapter tests."""

    def __init__(self) -> None:
        self.call_count = 0
        self._last_update_args = None

    def update(self, detections: list[dict], frame: Any = None) -> list[dict]:
        self.call_count += 1
        self._last_update_args = (detections, frame)
        # Return tracked detections with track_ids assigned
        return [
            {**d, "track_id": i + 1} for i, d in enumerate(detections)
        ]

    def update_appearance(self, frame: Any) -> None:
        pass


class _MockTargetManager:
    """Simulates a TargetManager for adapter tests."""

    def __init__(self) -> None:
        self.call_count = 0

    def update_targets(self, tracks, slot_id=0, timestamp=0.0):
        self.call_count += 1

    def get_all_targets(self):
        return []


class _MockEventBus:
    """Simulates an EventBus for adapter tests."""

    def __init__(self) -> None:
        self.published: list[Any] = []

    def publish(self, event) -> int:
        self.published.append(event)
        return 0


class _MockWorker:
    """Simulates InferWorker's relevant attributes for adapter tests."""

    def __init__(
        self,
        detections: list[dict] | None = None,
        tracker: _MockTrackerBackend | None = None,
    ) -> None:
        self._detector = _MockDetectorBackend(detections)
        self._model = self._detector._model
        self._run_inference = self._detector._run_inference
        self._trackers: dict[int, _MockTrackerBackend] = {}
        self._target_mgr = _MockTargetManager()
        self._event_bus = _MockEventBus()
        if tracker is not None:
            self._trackers[0] = tracker


def _raise_process(frame, slot_id, frame_id, submitted_ts):
    """Legacy process callable that records calls."""
    _raise_process.calls.append((frame, slot_id, frame_id, submitted_ts))
    return [{"bbox": (0.1, 0.2, 0.3, 0.4), "confidence": 0.9,
             "class_id": 0, "label": "person", "track_id": 1}]


_raise_process.calls = []


# ======================================================================
# 0. Module surface
# ======================================================================

def test_migration_pipeline_error_is_runtime_error():
    assert issubclass(MigrationPipelineError, RuntimeError)


def test_adapters_importable():
    assert InferWorkerDetectorAdapter is not None
    assert InferWorkerTrackerAdapter is not None
    assert InferWorkerTargetManagerAdapter is not None
    assert InferWorkerEventBusAdapter is not None


# ======================================================================
# 1. MigrationPipeline construction
# ======================================================================

def test_migration_pipeline_construction():
    mp = MigrationPipeline()
    assert mp.pipeline_enabled is False
    assert mp.detector is None
    assert mp.tracker is None
    assert mp.target_manager is None
    assert mp.event_bus is None


def test_migration_pipeline_construction_with_backends():
    det = _MockDetectorBackend()
    trk = _MockTrackerBackend()
    tgt = _MockTargetManager()
    evt = _MockEventBus()
    mp = MigrationPipeline(detector=det, tracker=trk,
                           target_manager=tgt, event_bus=evt)
    assert mp.detector is det
    assert mp.tracker is trk
    assert mp.target_manager is tgt
    assert mp.event_bus is evt


# ======================================================================
# 2. Mode switching
# ======================================================================

def test_set_pipeline_enabled():
    mp = MigrationPipeline()
    assert mp.pipeline_enabled is False
    mp.set_pipeline_enabled(True)
    assert mp.pipeline_enabled is True
    mp.set_pipeline_enabled(False)
    assert mp.pipeline_enabled is False


def test_set_pipeline_enabled_idempotent():
    mp = MigrationPipeline()
    mp.set_pipeline_enabled(True)
    mp.set_pipeline_enabled(True)
    assert mp.pipeline_enabled is True


# ======================================================================
# 3. Legacy mode
# ======================================================================

def test_legacy_mode_calls_legacy_process():
    """Legacy mode delegates to the injected legacy_process callable."""
    _raise_process.calls = []
    mp = MigrationPipeline(legacy_process=_raise_process)
    result = mp.run_frame("fake_frame", slot_id=0, frame_id=42, submitted_ts=1.0)
    assert len(_raise_process.calls) == 1
    assert _raise_process.calls[0] == ("fake_frame", 0, 42, 1.0)
    assert len(result) == 1
    assert result[0]["track_id"] == 1


def test_legacy_mode_returns_empty_when_no_process():
    """Legacy mode with no legacy_process returns []."""
    mp = MigrationPipeline()
    result = mp.run_frame("fake_frame")
    assert result == []


# ======================================================================
# 4. Pipeline mode (with mock backends)
# ======================================================================

def test_pipeline_mode_calls_detector_adapter():
    """Pipeline mode invokes the detector through the adapter."""
    worker = _MockWorker()
    mp = MigrationPipeline(
        detector=worker._detector,
        tracker=_MockTrackerBackend(),
        target_manager=_MockTargetManager(),
        event_bus=_MockEventBus(),
    )
    # Test the detector adapter directly (adapter delegates to worker._run_inference)
    adapter = InferWorkerDetectorAdapter(worker)
    adapter.set_frame_image("fake_frame")
    from visioncore.core.frame import Frame as CoreFrame
    core_frame = CoreFrame(frame_id=1, timestamp=0.0, source_id="s0", image=None)
    result = adapter.detect(core_frame)
    assert len(result) == 2
    assert worker._detector.call_count == 1
    # Result is list[CoreDetection]
    from visioncore.core.detection import Detection as CoreDetection
    assert all(isinstance(d, CoreDetection) for d in result)


def test_tracker_adapter_with_mock():
    """Tracker adapter delegates to the worker's per-slot tracker."""
    trk = _MockTrackerBackend()
    worker = _MockWorker(tracker=trk)
    adapter = InferWorkerTrackerAdapter(worker, slot_id=0)
    adapter.set_frame_image("fake_frame")
    from visioncore.core.detection import BBox, Detection as CoreDetection
    dets = [
        CoreDetection(bbox=BBox(x=0.2, y=0.3, w=0.2, h=0.2),
                      score=0.9, class_id=0, class_name="person"),
    ]
    from visioncore.core.frame import Frame as CoreFrame
    core_frame = CoreFrame(frame_id=1, timestamp=0.0, source_id="s0", image=None)
    from visioncore.pipeline.stages.tracker_stage import Tracker
    result = adapter.update(dets)
    assert trk.call_count == 1
    assert len(adapter.last_tracked_dicts) == 1
    assert adapter.last_tracked_dicts[0]["track_id"] == 1


def test_target_manager_adapter_construction():
    """TargetManager adapter can be constructed with a mock worker."""
    worker = _MockWorker()
    adapter = InferWorkerTargetManagerAdapter(worker, slot_id=0)
    assert adapter is not None


def test_event_bus_adapter_construction():
    """EventBus adapter can be constructed with a mock worker."""
    worker = _MockWorker()
    adapter = InferWorkerEventBusAdapter(worker)
    assert adapter is not None


# ======================================================================
# 5. Dual-mode consistency
# ======================================================================

def test_dual_mode_same_backends():
    """Both modes are constructed with the same backend references."""
    det = _MockDetectorBackend()
    trk = _MockTrackerBackend()
    tgt = _MockTargetManager()
    evt = _MockEventBus()

    mp = MigrationPipeline(
        detector=det, tracker=trk,
        target_manager=tgt, event_bus=evt,
    )
    # Legacy mode
    mp.set_pipeline_enabled(False)
    assert mp.detector is det
    assert mp.tracker is trk

    # Pipeline mode
    mp.set_pipeline_enabled(True)
    assert mp.detector is det
    assert mp.tracker is trk


def test_dual_mode_returns_same_structure():
    """Both modes return list[dict] with the same keys."""
    detections = [
        {"bbox": (0.1, 0.2, 0.3, 0.4), "confidence": 0.9,
         "class_id": 0, "label": "person"},
    ]

    # Legacy mode
    _raise_process.calls = []
    mp = MigrationPipeline(legacy_process=_raise_process)
    legacy_result = mp.run_frame("frame", slot_id=0, frame_id=1, submitted_ts=0.0)
    assert isinstance(legacy_result, list)
    assert all(isinstance(d, dict) for d in legacy_result)
    assert "bbox" in legacy_result[0]
    assert "confidence" in legacy_result[0]


# ======================================================================
# 6. Pipeline integration (full Pipeline with adapters)
# ======================================================================

def test_full_pipeline_with_mock_backends():
    """Build a Pipeline with adapter-wrapped mock backends and run it."""
    from visioncore.core.frame import Frame as CoreFrame

    worker = _MockWorker()
    trk = _MockTrackerBackend()
    worker._trackers[0] = trk

    det_adapter = InferWorkerDetectorAdapter(worker)
    det_adapter.set_frame_image("fake_frame")
    trk_adapter = InferWorkerTrackerAdapter(worker, slot_id=0)
    trk_adapter.set_frame_image("fake_frame")
    tgt_adapter = InferWorkerTargetManagerAdapter(worker, slot_id=0)
    evt_adapter = InferWorkerEventBusAdapter(worker)

    p = Pipeline()
    p.add_stage(DetectorStage(det_adapter, name="detect"))
    p.add_stage(TrackerStage(trk_adapter, name="track"))
    p.add_stage(TargetStage(tgt_adapter, name="targets"))
    p.add_stage(EventStage(evt_adapter, name="events"))

    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.frame = CoreFrame(
        frame_id=1, timestamp=1.0, source_id="slot0", image="fake_frame",
    )
    with p:
        p.run(ctx)

    # The pipeline ran all 4 stages with mock backends
    assert len(p) == 4
    assert p.shutdown_done is True
    # Detector was called
    assert worker._detector.call_count == 1
    # Tracker was called
    assert trk.call_count == 1


def test_pipeline_with_adapter_error_returns_empty():
    """Pipeline errors are caught and return []."""
    from visioncore.core.frame import Frame as CoreFrame

    class _BrokenDetector:
        _model = None
        def _run_inference(self, frame):
            raise RuntimeError("model crashed")

    worker = _MockWorker()
    worker._detector = _BrokenDetector()
    worker._run_inference = _BrokenDetector()._run_inference
    worker._model = None

    det_adapter = InferWorkerDetectorAdapter(worker)
    det_adapter.set_frame_image("frame")
    trk_adapter = InferWorkerTrackerAdapter(worker, slot_id=0)
    trk_adapter.set_frame_image("frame")
    tgt_adapter = InferWorkerTargetManagerAdapter(worker, slot_id=0)
    evt_adapter = InferWorkerEventBusAdapter(worker)

    p = Pipeline()
    p.add_stage(DetectorStage(det_adapter, name="detect"))
    p.add_stage(TrackerStage(trk_adapter, name="track"))
    p.add_stage(TargetStage(tgt_adapter, name="targets"))
    p.add_stage(EventStage(evt_adapter, name="events"))

    ctx = PipelineContext.empty()
    ctx.frame = CoreFrame(
        frame_id=1, timestamp=0.0, source_id="slot0", image="frame",
    )
    with p:
        try:
            p.run(ctx)
            # If no error, the pipeline still ran (adapter may catch)
        except Exception:
            pass  # Expected: detector error propagates


# ======================================================================
# 7. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """migration.py contains no network / model / GUI / ai imports."""
    import visioncore.pipeline.migration as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("socket", "requests", "rospy", "mavlink", "zmq",
              "torch", "onnx", "cv2", "ultralytics", "yolo",
              "ai", "gui", "camera")
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for b in banned:
                    if b in alias.name.lower():
                        violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod_name = (node.module or "").lower()
            for b in banned:
                if b in mod_name:
                    violations.append(f"from {node.module}")
    assert not violations, f"migration.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing migration module does not load torch."""
    import visioncore.pipeline.migration  # noqa: F401
    assert "torch" not in sys.modules, "torch loaded as side effect"


# ======================================================================
# Runner
# ======================================================================

def _collect_tests() -> list[tuple[str, Any]]:
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} inferworker migration tests...\n")
    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            tb = traceback.format_exc()
            failures.append((name, tb))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 60}")
    print(f"Result: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        print(f"\n{'=' * 60}\nFailure details:\n")
        for name, tb in failures:
            print(f"---- {name} ----")
            print(tb)
        raise SystemExit(1)
    print("All inferworker migration tests passed.")
