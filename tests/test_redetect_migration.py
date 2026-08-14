"""ROI Redetection migration unit tests (Milestone D9.1).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - RedetectConfig defaults and custom values
    - compute_roi: basic computation, boundary clamping, invalid dimensions
    - quality_gate: IoU, confidence, scale checks
    - RoiRedetector: success path, None frame, small crop, large crop,
      detector error, quality gate rejection
    - LegacyRedetectAdapter: wraps worker method
    - RedetectStage: config, passthrough, actual redetection
    - Shadow mode: Legacy vs Pipeline comparison
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.pipeline.stages.redetect import (
    RedetectConfig,
    Redetector,
    Roi,
    RoiRedetector,
    compute_roi,
    quality_gate,
)
from visioncore.pipeline.stages.redetect_stage import RedetectStage
from visioncore.pipeline.adapters import LegacyRedetectAdapter
from visioncore.pipeline import Pipeline, PipelineContext


# ======================================================================
# Test helpers
# ======================================================================

class _Raises:
    """Context manager asserting that a block raises a matching exception."""

    __slots__ = ("expected", "match", "caught")

    def __init__(self, expected: type[BaseException], match: str | None = None) -> None:
        self.expected = expected
        self.match = match
        self.caught = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"expected {self.expected.__name__}, but no exception was raised"
            )
        if not isinstance(exc_val, self.expected):
            raise AssertionError(
                f"expected {self.expected.__name__}, got "
                f"{type(exc_val).__name__}: {exc_val}"
            )
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(
                f"expected {self.match!r} in str, got {str(exc_val)!r}"
            )
        self.caught = exc_val
        return True


def raises(expected, match=None):
    return _Raises(expected, match=match)


def _mock_detector(roi_image, conf_threshold):
    """Mock detector: returns a person detection at center of ROI."""
    if roi_image is None:
        return []
    h, w = roi_image.shape[:2] if hasattr(roi_image, 'shape') else (100, 100)
    return [{
        "x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9,
        "confidence": 0.85, "label": "person",
    }]


def _mock_detector_empty(roi_image, conf_threshold):
    """Mock detector: returns no detections."""
    return []


def _mock_detector_error(roi_image, conf_threshold):
    """Mock detector: raises an error."""
    raise RuntimeError("model crashed")


def _mock_detector_low_conf_far(roi_image, conf_threshold):
    """Mock detector: returns a LOW-confidence detection FAR from ROI center."""
    return [{
        "x1": 0.0, "y1": 0.0, "x2": 0.2, "y2": 0.2,
        "confidence": 0.10, "label": "person",
    }]


# ======================================================================
# 0. Module surface
# ======================================================================

def test_redetect_config_defaults():
    cfg = RedetectConfig()
    assert cfg.enabled is True
    assert cfg.interval == 1
    assert cfg.region_expand == 0.15
    assert cfg.min_crop_pixels == 24
    assert cfg.max_crop_area_ratio == 0.20
    assert cfg.budget_per_frame == 1


def test_redetect_config_custom():
    cfg = RedetectConfig(enabled=False, interval=3, region_expand=0.25)
    assert cfg.enabled is False
    assert cfg.interval == 3
    assert cfg.region_expand == 0.25


def test_roi_namedtuple():
    roi = Roi(10, 20, 100, 200)
    assert roi.x1 == 10
    assert roi.y1 == 20
    assert roi.x2 == 100
    assert roi.y2 == 200
    assert roi.width == 90
    assert roi.height == 180
    assert roi.area == 90 * 180


def test_redetector_is_abc():
    assert hasattr(Redetector, 'redetect')


def test_iredetector_importable():
    assert RoiRedetector is not None
    assert LegacyRedetectAdapter is not None


# ======================================================================
# 1. compute_roi
# ======================================================================

def test_compute_roi_basic():
    """ROI computed from center box with padding."""
    roi = compute_roi((0.3, 0.4, 0.5, 0.6), frame_w=640, frame_h=480, pad=0.15)
    assert roi.x1 >= 0
    assert roi.y1 >= 0
    assert roi.x2 <= 640
    assert roi.y2 <= 480
    assert roi.width > 0
    assert roi.height > 0


def test_compute_roi_boundary_clamping():
    """ROI at frame edges is clamped to [0, frame_dim)."""
    roi = compute_roi((0.0, 0.0, 0.1, 0.1), frame_w=640, frame_h=480, pad=0.15)
    assert roi.x1 == 0
    assert roi.y1 == 0
    assert roi.x2 > 0
    assert roi.y2 > 0


def test_compute_roi_full_frame():
    """ROI covering full frame is clamped to frame dimensions."""
    roi = compute_roi((0.0, 0.0, 1.0, 1.0), frame_w=640, frame_h=480, pad=0.0)
    assert roi.x1 == 0
    assert roi.y1 == 0
    assert roi.x2 == 640
    assert roi.y2 == 480


def test_compute_roi_invalid_dimensions():
    with raises(ValueError, match="positive"):
        compute_roi((0.1, 0.2, 0.3, 0.4), frame_w=0, frame_h=480)


def test_compute_roi_no_padding():
    """With pad=0, ROI matches the box exactly."""
    roi = compute_roi((0.25, 0.25, 0.75, 0.75), frame_w=400, frame_h=400, pad=0.0)
    assert roi.x1 == 100
    assert roi.y1 == 100
    assert roi.x2 == 300
    assert roi.y2 == 300


# ======================================================================
# 2. quality_gate
# ======================================================================

def test_quality_gate_pass():
    """Refined box close to predicted with good confidence passes."""
    assert quality_gate(
        (0.3, 0.4, 0.5, 0.6),
        (0.31, 0.41, 0.51, 0.61),
        original_confidence=0.8,
        refined_confidence=0.85,
    ) is True


def test_quality_gate_fail_low_conf_low_iou():
    """Low confidence AND low IoU fails."""
    assert quality_gate(
        (0.1, 0.1, 0.3, 0.3),
        (0.7, 0.7, 0.9, 0.9),
        original_confidence=0.8,
        refined_confidence=0.10,
    ) is False


def test_quality_gate_fail_scale_mutation():
    """Refined box 3x wider fails scale check."""
    assert quality_gate(
        (0.3, 0.4, 0.5, 0.6),
        (0.1, 0.4, 0.9, 0.6),
        original_confidence=0.8,
        refined_confidence=0.85,
    ) is False


def test_quality_gate_pass_high_iou():
    """High IoU passes even with lower confidence."""
    assert quality_gate(
        (0.3, 0.4, 0.5, 0.6),
        (0.31, 0.41, 0.51, 0.61),
        original_confidence=0.8,
        refined_confidence=0.55,
    ) is True


# ======================================================================
# 3. RoiRedetector
# ======================================================================

def test_iredetect_success():
    """Successful redetection returns a detection dict."""
    rd = RoiRedetector(_mock_detector)
    # Use a numpy-like mock frame
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = rd.redetect(frame, (0.3, 0.4, 0.5, 0.6), original_confidence=0.7)
    assert result is not None
    assert "confidence" in result
    assert result["label"] == "person"


def test_iredetect_none_frame():
    """None frame returns None."""
    rd = RoiRedetector(_mock_detector)
    assert rd.redetect(None, (0.3, 0.4, 0.5, 0.6)) is None


def test_iredetect_small_crop():
    """Very small predicted box (below min_crop_pixels) returns None."""
    rd = RoiRedetector(_mock_detector, RedetectConfig(min_crop_pixels=100))
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Box is 1% of frame -> ROI ~10 pixels, below min_crop_pixels=100
    result = rd.redetect(frame, (0.49, 0.49, 0.50, 0.50))
    assert result is None


def test_iredetect_large_crop():
    """ROI > 20% of frame area returns None."""
    rd = RoiRedetector(_mock_detector, RedetectConfig(max_crop_area_ratio=0.10))
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Box covers 50% of frame
    result = rd.redetect(frame, (0.1, 0.1, 0.9, 0.9))
    assert result is None


def test_iredetect_empty_detections():
    """Detector returns no results -> None."""
    rd = RoiRedetector(_mock_detector_empty)
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert rd.redetect(frame, (0.3, 0.4, 0.5, 0.6)) is None


def test_iredetect_detector_error():
    """Detector raises -> None (caught, not propagated)."""
    rd = RoiRedetector(_mock_detector_error)
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert rd.redetect(frame, (0.3, 0.4, 0.5, 0.6)) is None


def test_iredetect_quality_gate_rejects():
    """Quality gate rejects distant result with low confidence."""
    rd = RoiRedetector(_mock_detector_low_conf_far, RedetectConfig(conf_floor=0.20))
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Detector returns (0.0,0.0,0.2,0.2) in ROI coords -> maps to far from center
    # original_confidence=0.8, refined=0.10 -> fails quality gate
    result = rd.redetect(frame, (0.3, 0.3, 0.7, 0.7), original_confidence=0.8)
    assert result is None


# ======================================================================
# 4. LegacyRedetectAdapter
# ======================================================================

class _MockWorker:
    """Simulates InferWorker with _redetect_person_in_roi."""
    def __init__(self, result=None):
        self._model = True
        self._redetect_result = result
        self.call_count = 0

    def _redetect_person_in_roi(self, frame, predicted_box, original_confidence=0.0):
        self.call_count += 1
        return self._redetect_result


def test_legacy_adapter_wraps_worker():
    """LegacyRedetectAdapter delegates to worker._redetect_person_in_roi."""
    worker = _MockWorker(result={"confidence": 0.8, "label": "person"})
    adapter = LegacyRedetectAdapter(worker)
    result = adapter.redetect("frame", (0.1, 0.2, 0.3, 0.4), 0.7)
    assert result is not None
    assert result["confidence"] == 0.8
    assert worker.call_count == 1


def test_legacy_adapter_returns_none():
    """LegacyRedetectAdapter returns None when worker returns None."""
    worker = _MockWorker(result=None)
    adapter = LegacyRedetectAdapter(worker)
    assert adapter.redetect("frame", (0.1, 0.2, 0.3, 0.4)) is None


def test_legacy_adapter_health_check():
    """LegacyRedetectAdapter health_check reflects worker model."""
    worker = _MockWorker()
    adapter = LegacyRedetectAdapter(worker)
    assert adapter.health_check() is True
    worker._model = None
    assert adapter.health_check() is False


# ======================================================================
# 5. RedetectStage
# ======================================================================

def test_redetect_stage_capability():
    stage = RedetectStage()
    assert stage.capability.name == "redetect"
    assert stage.capability.version == "0.2.0"
    assert "tracks" in stage.capability.required_context


def test_redetect_stage_config():
    cfg = RedetectConfig(enabled=False, region_expand=0.25)
    stage = RedetectStage(config=cfg)
    assert stage.config.enabled is False
    assert stage.config.region_expand == 0.25


def test_redetect_stage_passthrough_when_disabled():
    """Disabled config -> passthrough."""
    stage = RedetectStage(config=RedetectConfig(enabled=False))
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks = [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "confidence": 0.8}]
    stage.process(ctx)
    assert len(ctx.tracks) == 1


def test_redetect_stage_passthrough_when_no_redetector():
    """No redetector -> passthrough."""
    stage = RedetectStage(redetector=None)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks = [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "confidence": 0.8}]
    stage.process(ctx)
    assert len(ctx.tracks) == 1


def test_redetect_stage_passthrough_empty_tracks():
    """Empty tracks -> passthrough."""
    stage = RedetectStage(redetector=RoiRedetector(_mock_detector))
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks = []
    stage.process(ctx)
    assert len(ctx.tracks) == 0


def test_redetect_stage_passthrough_no_frame():
    """No frame -> passthrough."""
    stage = RedetectStage(redetector=RoiRedetector(_mock_detector))
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks = [{"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4, "confidence": 0.8}]
    stage.process(ctx)
    assert len(ctx.tracks) == 1


def test_redetect_stage_runs_redetection():
    """With frame + redetector + tracks -> redetection runs."""
    import numpy as np
    stage = RedetectStage(redetector=RoiRedetector(_mock_detector))
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = np.zeros((480, 640, 3), dtype=np.uint8)
    ctx.tracks = [{"x1": 0.3, "y1": 0.4, "x2": 0.5, "y2": 0.6, "confidence": 0.7}]
    stage.process(ctx)
    # Track should be updated with refined detection
    assert ctx.tracks[0]["confidence"] == 0.85


def test_redetect_stage_lifecycle():
    stage = RedetectStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


# ======================================================================
# 6. Shadow mode comparison
# ======================================================================

def test_shadow_mode_legacy_vs_pipeline():
    """LegacyRedetectAdapter and RoiRedetector produce same result on same input."""
    import numpy as np

    # Legacy path: mock worker with known result
    legacy_result = {"x1": 0.29, "y1": 0.39, "x2": 0.51, "y2": 0.61,
                     "confidence": 0.82, "label": "person"}
    worker = _MockWorker(result=legacy_result)
    legacy_adapter = LegacyRedetectAdapter(worker)

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (0.3, 0.4, 0.5, 0.6)
    legacy_out = legacy_adapter.redetect(frame, box, 0.7)

    # Pipeline path: RoiRedetector with matching detector
    def matching_detector(roi, conf):
        return [{"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9,
                 "confidence": 0.82, "label": "person"}]

    pipeline_redetector = RoiRedetector(matching_detector)
    pipeline_out = pipeline_redetector.redetect(frame, box, 0.7)

    # Both should return a result (or both None)
    assert (legacy_out is None) == (pipeline_out is None)
    if legacy_out is not None and pipeline_out is not None:
        assert abs(legacy_out["confidence"] - pipeline_out["confidence"]) < 0.01


def test_shadow_mode_timing():
    """Both modes complete without timing out."""
    import numpy as np
    import time

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (0.3, 0.4, 0.5, 0.6)

    # Legacy
    worker = _MockWorker(result={"confidence": 0.8, "label": "person"})
    adapter = LegacyRedetectAdapter(worker)
    t0 = time.perf_counter()
    for _ in range(100):
        adapter.redetect(frame, box, 0.7)
    legacy_time = time.perf_counter() - t0

    # Pipeline
    redetector = RoiRedetector(_mock_detector)
    t0 = time.perf_counter()
    for _ in range(100):
        redetector.redetect(frame, box, 0.7)
    pipeline_time = time.perf_counter() - t0

    # Both should complete (no strict timing requirement)
    assert legacy_time < 5.0
    assert pipeline_time < 5.0


# ======================================================================
# 7. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """redetect.py contains no network / model / GUI / ai imports."""
    import visioncore.pipeline.stages.redetect.redetect as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("socket", "requests", "rospy", "mavlink", "zmq",
              "torch", "onnx", "cv2", "ultralytics", "yolo",
              "ai", "gui", "camera")
    violations = []
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
    assert not violations, f"redetect.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing redetect modules does not load torch."""
    import visioncore.pipeline.stages.redetect  # noqa: F401
    assert "torch" not in sys.modules, "torch loaded as side effect"


# ======================================================================
# Runner
# ======================================================================

def _collect_tests():
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} redetect migration tests...\n")
    passed = 0
    failed = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:
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
    print("All redetect migration tests passed.")
