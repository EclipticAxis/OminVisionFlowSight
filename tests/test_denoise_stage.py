"""DenoiseStage unit tests (Milestone D2).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Construction (with/without denoiser, default/custom name)
    - Capability declaration (name, version, required/provided context)
    - Lifecycle (initialize/process/shutdown/health_check callable, state
      transitions, idempotent init, safe shutdown)
    - Process passthrough (context.frame identity preserved with Frame and
      with None, check_context enforced)
    - Pipeline integration (DenoiseStage in Pipeline, full chain)
    - No OpenCV / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from visioncore.core.frame import Frame
from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages.advanced import AdvancedStage, AdvancedStageError
from visioncore.pipeline.stages.denoise_stage import DenoiseStage


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


def _frame(frame_id: int = 0) -> Frame:
    """Build a synthetic Frame for tests."""
    return Frame(
        frame_id=frame_id,
        timestamp=float(frame_id),
        source_id="cam0",
        image=np.zeros((4, 4, 3), dtype=np.uint8),
    )


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_denoise_stage_is_exported():
    """DenoiseStage is importable from the stages package."""
    from visioncore.pipeline.stages.denoise_stage import DenoiseStage as DS
    assert DS is DenoiseStage


def test_denoise_stage_is_advanced_stage_subclass():
    assert issubclass(DenoiseStage, AdvancedStage)


def test_denoise_stage_is_pipeline_stage_subclass():
    assert issubclass(DenoiseStage, PipelineStage)


# ======================================================================
# 1. Construction
# ======================================================================

def test_construction_default():
    """Default constructor: denoiser=None, name=DenoiseStage."""
    stage = DenoiseStage()
    assert stage.denoiser is None
    assert stage.name == "DenoiseStage"


def test_construction_with_denoiser():
    """Constructor accepts an arbitrary denoiser object."""
    denoiser = object()
    stage = DenoiseStage(denoiser)
    assert stage.denoiser is denoiser


def test_construction_with_none_denoiser():
    """Constructor accepts explicit None."""
    stage = DenoiseStage(denoiser=None)
    assert stage.denoiser is None


def test_construction_custom_name():
    stage = DenoiseStage(name="pre-detect-denoise")
    assert stage.name == "pre-detect-denoise"


def test_construction_name_keyword():
    """name is keyword-only (after denoiser positional)."""
    stage = DenoiseStage(None, name="custom")
    assert stage.name == "custom"


# ======================================================================
# 2. Capability declaration
# ======================================================================

def test_capability_name():
    assert DenoiseStage().capability.name == "denoise"


def test_capability_version():
    assert DenoiseStage().capability.version == "0.1.0"


def test_capability_required_context():
    assert DenoiseStage().capability.required_context == ["frame"]


def test_capability_provided_context():
    """D2 passthrough writes nothing to the context."""
    assert DenoiseStage().capability.provided_context == []


def test_capability_description_mentions_placeholder():
    assert "placeholder" in DenoiseStage().capability.description.lower()


# ======================================================================
# 3. Lifecycle
# ======================================================================

def test_lifecycle_methods_callable():
    """All four lifecycle methods are callable."""
    stage = DenoiseStage()
    assert callable(stage.initialize)
    assert callable(stage.process)
    assert callable(stage.shutdown)
    assert callable(stage.health_check)


def test_lifecycle_state_transitions():
    """healthy: False -> True -> False through init/process/shutdown."""
    stage = DenoiseStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    ctx = PipelineContext.empty()
    ctx.frame = _frame()
    stage.process(ctx)
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_initialize_idempotent():
    stage = DenoiseStage()
    stage.initialize()
    stage.initialize()
    assert stage.health_check() is True


def test_shutdown_never_raises():
    """shutdown() must not raise, even without prior initialize()."""
    stage = DenoiseStage()
    stage.shutdown()
    assert stage.health_check() is False


def test_repr_reports_health():
    stage = DenoiseStage(name="denoise")
    stage.initialize()
    r = repr(stage)
    assert "denoise" in r
    assert "healthy=True" in r


# ======================================================================
# 4. Process passthrough
# ======================================================================

def test_process_preserves_frame_identity():
    """process() leaves context.frame as the SAME object (identity check)."""
    stage = DenoiseStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    original = _frame(42)
    ctx.frame = original
    stage.process(ctx)
    assert ctx.frame is original


def test_process_with_none_frame():
    """process() succeeds when context.frame is None (legitimate no-op)."""
    stage = DenoiseStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    assert ctx.frame is None
    stage.process(ctx)
    assert ctx.frame is None


def test_process_does_not_mutate_frame_fields():
    """process() does not alter any field on the Frame object."""
    stage = DenoiseStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _frame(7)
    stage.process(ctx)
    assert ctx.frame.frame_id == 7
    assert ctx.frame.timestamp == 7.0
    assert ctx.frame.source_id == "cam0"
    assert ctx.frame.image.shape == (4, 4, 3)


def test_process_validates_context_contract():
    """process() calls check_context -- missing frame raises."""
    stage = DenoiseStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    del ctx.frame
    with raises(AdvancedStageError):
        stage.process(ctx)


def test_process_multiple_calls():
    """process() can be called multiple times without side effects."""
    stage = DenoiseStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    f = _frame(1)
    ctx.frame = f
    stage.process(ctx)
    stage.process(ctx)
    stage.process(ctx)
    assert ctx.frame is f


# ======================================================================
# 5. Pipeline integration
# ======================================================================

def test_pipeline_with_denoise_stage():
    """DenoiseStage runs inside a Pipeline like any other stage."""
    stage = DenoiseStage(name="denoise")
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty()
    ctx.frame = _frame(0)
    with p:
        p.run(ctx)
    assert ctx.frame is not None
    assert ctx.frame.frame_id == 0
    assert stage.health_check() is False  # shutdown called


def test_pipeline_denoise_stage_full_chain():
    """DenoiseStage -> DummyDetector: frame passes through to detection."""
    from visioncore.pipeline.stages.detector_stage import DetectorStage, DummyDetector

    denoise = DenoiseStage(name="denoise")
    detect = DetectorStage(DummyDetector(), name="detect")
    p = Pipeline()
    p.add_stage(denoise)
    p.add_stage(detect)
    ctx = PipelineContext.empty()
    ctx.frame = _frame(0)
    with p:
        p.run(ctx)
    assert ctx.frame.frame_id == 0
    assert len(ctx.detections) == 1


# ======================================================================
# 6. No OpenCV / GUI / ai dependency
# ======================================================================

def test_no_opencv_in_source():
    """denoise_stage.py contains no cv2 import (AST check)."""
    import visioncore.pipeline.stages.denoise_stage as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if "cv2" in alias.name.lower():
                    violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod_name = (node.module or "").lower()
            if "cv2" in mod_name:
                violations.append(f"from {node.module}")
    assert not violations, f"denoise_stage.py imports cv2: {violations}"


def test_no_gui_ai_in_source():
    """denoise_stage.py contains no gui/ai imports (AST check)."""
    import visioncore.pipeline.stages.denoise_stage as mod
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
    assert not violations, f"denoise_stage.py imports banned: {violations}"


def test_no_cv2_loaded_at_runtime():
    """Importing denoise_stage does not load cv2 as side effect."""
    import visioncore.pipeline.stages.denoise_stage  # noqa: F401
    assert "cv2" not in sys.modules, "cv2 loaded as side effect"


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
    print(f"Running {len(tests)} denoise stage tests...\n")
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
    print("All denoise stage tests passed.")
