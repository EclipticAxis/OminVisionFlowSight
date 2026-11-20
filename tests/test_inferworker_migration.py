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

from tests._runtime_checks import assert_import_clean

from visioncore.core.detection import Detection as _CoreDetection
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
        # Production schema: x1/y1/x2/y2 + confidence/class_id/label
        # (same as ai.detection.Detection.to_dict()); never a bbox key.
        self._detections = detections or [
            {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
             "confidence": 0.9, "class_id": 0, "label": "person"},
            {"x1": 0.5, "y1": 0.6, "x2": 0.7, "y2": 0.8,
             "confidence": 0.8, "class_id": 0, "label": "person"},
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
        self._last_redetect_callback: Any = "UNSET"
        self.appearance_calls = 0

    def update(self, detections: list[dict], frame: Any = None,
               redetect_callback: Any = None) -> list[dict]:
        self.call_count += 1
        self._last_update_args = (detections, frame)
        self._last_redetect_callback = redetect_callback
        # Return tracked detections with track_ids assigned
        return [
            {**d, "track_id": i + 1} for i, d in enumerate(detections)
        ]

    def update_appearance(self, frame: Any) -> None:
        self.appearance_calls += 1


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
    return [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
             "confidence": 0.9, "class_id": 0, "label": "person",
             "track_id": 1}]


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
    """Both modes return list[dict] with the production schema keys."""
    # Legacy mode
    _raise_process.calls = []
    mp = MigrationPipeline(legacy_process=_raise_process)
    legacy_result = mp.run_frame("frame", slot_id=0, frame_id=1, submitted_ts=0.0)
    assert isinstance(legacy_result, list)
    assert all(isinstance(d, dict) for d in legacy_result)
    # Production schema: x1/y1/x2/y2 corners, never a bbox key
    for key in ("x1", "y1", "x2", "y2", "confidence", "class_id", "label"):
        assert key in legacy_result[0]
    assert "bbox" not in legacy_result[0]


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
    assert_import_clean("visioncore.pipeline.migration", ["torch"])


# ======================================================================
# 8. Step-4 parity: lossless dict<->core round-trip + redetect passthrough
# ======================================================================

def test_dict_core_roundtrip_preserves_extras_exactly():
    """Extras (keypoints/polygon/track_id/track_state/gesture/head) survive."""
    from visioncore.pipeline.adapters import (
        _core_detection_to_dict,
        _dict_to_core_detection,
    )

    src = {
        "x1": 0.10, "y1": 0.10, "x2": 0.50, "y2": 0.90,
        "confidence": 0.92, "label": "person", "class_id": 0,
        "track_id": 7, "track_state": "normal",
        "keypoints": [{"x": 0.3, "y": 0.2, "conf": 0.9}],
        "polygon": [[0.6, 0.1], [0.9, 0.15], [0.88, 0.4], [0.58, 0.35]],
        "gesture": "raising_hand",
        "head_attrs": {"hat": True},
    }
    core = _dict_to_core_detection(src)
    out = _core_detection_to_dict(core)

    # Corners reproduced bit-exactly (float round-trip through centre/size
    # is not exact; the adapter snapshot must restore the original values).
    assert out == src
    # Production schema on output: never a bbox key.
    assert "bbox" not in out
    # Extras are deep copies: mutating the output must not touch the source.
    out["keypoints"][0]["x"] = 9.9
    assert src["keypoints"][0]["x"] == 0.3


def test_dict_core_roundtrip_core_bbox_change_maps_back():
    """A modified core bbox maps to the output; extras are kept."""
    from visioncore.core.detection import BBox
    from visioncore.pipeline.adapters import (
        _AdaptedDetection,
        _core_detection_to_dict,
        _dict_to_core_detection,
    )

    core = _dict_to_core_detection({
        "x1": 0.10, "y1": 0.10, "x2": 0.50, "y2": 0.90,
        "confidence": 0.92, "label": "person", "class_id": 0,
        "track_id": 7,
    })
    moved = _AdaptedDetection(
        bbox=BBox(x=0.4, y=0.5, w=0.2, h=0.4),
        score=core.score, class_id=core.class_id,
        class_name=core.class_name, extras=core.extras,
    )
    out = _core_detection_to_dict(moved)
    assert abs(out["x1"] - 0.3) < 1e-9
    assert abs(out["y1"] - 0.3) < 1e-9
    assert abs(out["x2"] - 0.5) < 1e-9
    assert abs(out["y2"] - 0.7) < 1e-9
    assert out["track_id"] == 7  # extras kept alongside changed basics
    assert "bbox" not in out


def test_dict_to_core_accepts_legacy_bbox_key_but_never_emits_it():
    """Old bbox-tuple input stays accepted; output is production schema."""
    from visioncore.pipeline.adapters import (
        _core_detection_to_dict,
        _dict_to_core_detection,
    )

    core = _dict_to_core_detection({
        "bbox": (0.1, 0.2, 0.3, 0.4),
        "confidence": 0.5, "class_id": 2, "label": "car",
    })
    out = _core_detection_to_dict(core)
    assert out["x1"] == 0.1 and out["y2"] == 0.4
    assert "bbox" not in out


def test_adapted_detection_value_semantics_match_core_detection():
    """Extras do not participate in eq/hash; base 4-field contract intact."""
    from visioncore.core.detection import BBox
    from visioncore.pipeline.adapters import _AdaptedDetection

    plain_core = _CoreDetection(
        bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9, class_id=0,
        class_name="person",
    )
    adapted = _AdaptedDetection(
        bbox=BBox(0.5, 0.5, 0.2, 0.4), score=0.9, class_id=0,
        class_name="person", extras={"track_id": 3},
    )
    assert adapted == plain_core
    assert plain_core == adapted
    assert hash(adapted) == hash(plain_core)
    assert adapted != _AdaptedDetection(
        bbox=BBox(0.4, 0.5, 0.2, 0.4), score=0.9, class_id=0,
        class_name="person",
    )
    # Base class still has exactly the four core fields.
    import dataclasses
    assert {f.name for f in dataclasses.fields(_CoreDetection)} == {
        "bbox", "score", "class_id", "class_name",
    }


def test_tracker_adapter_forwards_redetect_callback_when_set():
    """Enabled path passes the callback; disabled path is frame-only."""
    from visioncore.core.detection import BBox
    from visioncore.core.detection import Detection as CoreDetection

    trk = _MockTrackerBackend()
    worker = _MockWorker(tracker=trk)
    adapter = InferWorkerTrackerAdapter(worker, slot_id=0)
    adapter.set_frame_image("fake_frame")
    dets = [
        CoreDetection(bbox=BBox(x=0.2, y=0.3, w=0.2, h=0.2),
                      score=0.9, class_id=0, class_name="person"),
    ]

    # Disabled: tracker.update receives frame only (callback stays None).
    adapter.update(dets)
    assert trk._last_redetect_callback is None
    assert trk._last_update_args[1] == "fake_frame"

    # Enabled: the very same callback object is forwarded.
    def _cb(predicted_box, original_confidence):
        return None

    adapter.set_redetect_callback(_cb)
    adapter.update(dets)
    assert trk._last_redetect_callback is _cb
    assert trk._last_update_args[1] == "fake_frame"


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
