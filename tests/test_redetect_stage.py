"""RedetectStage unit tests (Milestone D3).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Construction (with/without redetector, default/custom name)
    - Capability declaration (name, version, required/provided context)
    - Lifecycle (initialize/process/shutdown/health_check callable, state
      transitions, idempotent init, safe shutdown)
    - Process passthrough (context.tracks list unchanged, identity
      preserved, empty tracks, check_context enforced)
    - Pipeline integration (RedetectStage in Pipeline, full chain with
      preceding stages)
    - No detector-model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from visioncore.core.detection import BBox, Detection
from visioncore.core.track import Track
from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages.advanced import AdvancedStage, AdvancedStageError
from visioncore.pipeline.stages.redetect_stage import RedetectStage


# ======================================================================
# Test helpers
# ======================================================================

class _Raises:
    """Context manager asserting that a block raises a matching exception."""

    __slots__ = ("expected", "match", "caught")

    def __init__(self, expected: type[BaseException], match: str | None = None) -> None:
        self.expected: type[BaseException] = expected
        self.match: str | None = match
        self.caught: BaseException | None = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"expected {self.expected.__name__}, but no exception was raised"
            )
        if not isinstance(exc_val, BaseException):
            raise AssertionError(f"expected {self.expected.__name__}, got {exc_val!r}")
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


def raises(expected: type[BaseException], match: str | None = None) -> _Raises:
    return _Raises(expected, match=match)


def _track(track_id: int = 1, class_name: str = "person") -> Track:
    """Build a synthetic Track for tests."""
    return Track(
        track_id=track_id,
        detection=Detection(BBox(0.5, 0.5, 0.2, 0.4), 0.9, 0, class_name),
    )


def _frame():
    """Build a synthetic Frame for pipeline integration tests."""
    from visioncore.core.frame import Frame
    return Frame(0, 0.0, "cam0", np.zeros((4, 4, 3), dtype=np.uint8))


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_redetect_stage_is_exported():
    """RedetectStage is importable from the redetect_stage module."""
    from visioncore.pipeline.stages.redetect_stage import RedetectStage as RS
    assert RS is RedetectStage


def test_redetect_stage_is_advanced_stage_subclass():
    assert issubclass(RedetectStage, AdvancedStage)


def test_redetect_stage_is_pipeline_stage_subclass():
    assert issubclass(RedetectStage, PipelineStage)


# ======================================================================
# 1. Construction
# ======================================================================

def test_construction_default():
    """Default constructor: redetector=None, name=RedetectStage."""
    stage = RedetectStage()
    assert stage.redetector is None
    assert stage.name == "RedetectStage"


def test_construction_with_redetector():
    """Constructor accepts an arbitrary redetector object."""
    redetector = object()
    stage = RedetectStage(redetector)
    assert stage.redetector is redetector


def test_construction_with_none_redetector():
    """Constructor accepts explicit None."""
    stage = RedetectStage(redetector=None)
    assert stage.redetector is None


def test_construction_custom_name():
    stage = RedetectStage(name="post-track-redetect")
    assert stage.name == "post-track-redetect"


def test_construction_name_keyword():
    """name is keyword-only (after redetector positional)."""
    stage = RedetectStage(None, name="custom")
    assert stage.name == "custom"


# ======================================================================
# 2. Capability declaration
# ======================================================================

def test_capability_name():
    assert RedetectStage().capability.name == "redetect"


def test_capability_version():
    assert RedetectStage().capability.version == "0.2.0"


def test_capability_required_context():
    assert RedetectStage().capability.required_context == ["tracks"]


def test_capability_provided_context():
    """D9.1 stage provides updated tracks to the context."""
    assert RedetectStage().capability.provided_context == ["tracks"]


def test_capability_description_mentions_redetection():
    assert "re-detection" in RedetectStage().capability.description.lower() or "redetect" in RedetectStage().capability.description.lower()


# ======================================================================
# 3. Lifecycle
# ======================================================================

def test_lifecycle_methods_callable():
    """All four lifecycle methods are callable."""
    stage = RedetectStage()
    assert callable(stage.initialize)
    assert callable(stage.process)
    assert callable(stage.shutdown)
    assert callable(stage.health_check)


def test_lifecycle_state_transitions():
    """healthy: False -> True -> False through init/process/shutdown."""
    stage = RedetectStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    ctx = PipelineContext.empty()
    ctx.tracks.append(_track())
    stage.process(ctx)
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_initialize_idempotent():
    stage = RedetectStage()
    stage.initialize()
    stage.initialize()
    assert stage.health_check() is True


def test_shutdown_never_raises():
    """shutdown() must not raise, even without prior initialize()."""
    stage = RedetectStage()
    stage.shutdown()
    assert stage.health_check() is False


def test_repr_reports_health():
    stage = RedetectStage(name="redetect")
    stage.initialize()
    r = repr(stage)
    assert "redetect" in r
    assert "healthy=True" in r


# ======================================================================
# 4. Process passthrough
# ======================================================================

def test_process_preserves_tracks_list_identity():
    """process() leaves context.tracks as the SAME list object."""
    stage = RedetectStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    original = ctx.tracks
    ctx.tracks.append(_track(1))
    ctx.tracks.append(_track(2))
    stage.process(ctx)
    assert ctx.tracks is original


def test_process_preserves_tracks_content():
    """process() does not alter tracks list contents."""
    stage = RedetectStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    t1 = _track(1, "person")
    t2 = _track(2, "car")
    ctx.tracks.extend([t1, t2])
    stage.process(ctx)
    assert len(ctx.tracks) == 2
    assert ctx.tracks[0] is t1
    assert ctx.tracks[1] is t2


def test_process_with_empty_tracks():
    """process() succeeds with empty tracks list (legitimate no-op)."""
    stage = RedetectStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    assert ctx.tracks == []
    stage.process(ctx)
    assert ctx.tracks == []


def test_process_preserves_track_fields():
    """process() does not alter any field on the Track objects."""
    stage = RedetectStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    t = _track(42, "vehicle")
    ctx.tracks.append(t)
    stage.process(ctx)
    assert ctx.tracks[0].track_id == 42
    assert ctx.tracks[0].detection.class_name == "vehicle"


def test_process_validates_context_contract():
    """process() calls check_context -- missing tracks raises."""
    stage = RedetectStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    del ctx.tracks
    with raises(AdvancedStageError):
        stage.process(ctx)


def test_process_multiple_calls():
    """process() can be called multiple times without side effects."""
    stage = RedetectStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    t = _track(1)
    ctx.tracks.append(t)
    stage.process(ctx)
    stage.process(ctx)
    stage.process(ctx)
    assert len(ctx.tracks) == 1
    assert ctx.tracks[0] is t


# ======================================================================
# 5. Pipeline integration
# ======================================================================

def test_pipeline_with_redetect_stage():
    """RedetectStage runs inside a Pipeline like any other stage."""
    stage = RedetectStage(name="redetect")
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty()
    ctx.tracks.append(_track(1))
    with p:
        p.run(ctx)
    assert len(ctx.tracks) == 1
    assert ctx.tracks[0].track_id == 1
    assert stage.health_check() is False  # shutdown called


def test_pipeline_full_chain_detect_track_redetect():
    """Full chain: DetectorStage -> TrackerStage -> RedetectStage.

    RedetectStage sits after the tracker and passes tracks through
    unchanged, proving the stage integrates into the core chain.
    """
    from visioncore.pipeline.stages.detector_stage import DetectorStage, DummyDetector
    from visioncore.pipeline.stages.tracker_stage import TrackerStage, DummyTracker

    detect = DetectorStage(DummyDetector(), name="detect")
    track = TrackerStage(DummyTracker(), name="track")
    redetect = RedetectStage(name="redetect")
    p = Pipeline()
    p.add_stage(detect)
    p.add_stage(track)
    p.add_stage(redetect)
    ctx = PipelineContext.empty()
    ctx.frame = _frame()
    with p:
        p.run(ctx)
    assert len(ctx.detections) == 1
    assert len(ctx.tracks) == 1
    assert ctx.tracks[0].track_id == 1
    assert redetect.health_check() is False


# ======================================================================
# 6. No detector-model / GUI / ai dependency
# ======================================================================

def test_no_yolo_detr_sam_in_source():
    """redetect_stage.py contains no detector-model imports (AST check)."""
    import visioncore.pipeline.stages.redetect_stage as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("yolo", "rt_detr", "rtdetr", "groundingdino", "sam",
              "ultralytics", "torch", "onnx")
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
    assert not violations, f"redetect_stage.py imports banned: {violations}"


def test_no_gui_ai_in_source():
    """redetect_stage.py contains no gui/ai imports (AST check)."""
    import visioncore.pipeline.stages.redetect_stage as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("gui", "PyQt5", "PyQt6", "inference", "inferworker")
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
    assert not violations, f"redetect_stage.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing redetect_stage does not load torch as side effect."""
    import visioncore.pipeline.stages.redetect_stage  # noqa: F401
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
    print(f"Running {len(tests)} redetect stage tests...\n")
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
    print("All redetect stage tests passed.")
