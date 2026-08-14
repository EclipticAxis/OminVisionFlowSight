"""AttributeStage / GestureStage unit tests (Milestone D4).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Construction (with/without estimator/recogniser, default/custom name)
    - Capability declaration (name, version, required/provided context)
    - Lifecycle (initialize/process/shutdown/health_check)
    - AttributeStage.process: sets attributes={} on each TargetState,
      preserves identity of target_states list, handles empty list
    - GestureStage.process: sets gesture=None on each TargetState,
      preserves identity of target_states list, handles empty list
    - Both stages use REPLACE semantics (clear+extend)
    - check_context enforced
    - Pipeline integration
    - No model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages.advanced import AdvancedStage, AdvancedStageError
from visioncore.pipeline.stages.attribute_gesture_stage import (
    AttributeStage,
    GestureStage,
)
from visioncore.state.target_state import TargetState


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


def _snapshot(
    target_id: int = 1,
    label: str = "person",
    attributes: dict[str, Any] | None = None,
    gesture: str | None = None,
) -> TargetState:
    """Build a synthetic TargetState snapshot for tests."""
    kwargs: dict[str, Any] = dict(
        target_id=target_id, local_id=1, global_id=None, label=label,
        confidence=0.9, cx=0.5, cy=0.5, vx=0.0, vy=0.0,
        width=0.2, height=0.4, timestamp=0.0, camera_id=0, metadata={},
    )
    if attributes is not None:
        kwargs["attributes"] = attributes
    if gesture is not None:
        kwargs["gesture"] = gesture
    return TargetState(**kwargs)


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_attribute_stage_is_advanced_stage_subclass():
    assert issubclass(AttributeStage, AdvancedStage)


def test_gesture_stage_is_advanced_stage_subclass():
    assert issubclass(GestureStage, AdvancedStage)


def test_attribute_stage_is_pipeline_stage_subclass():
    assert issubclass(AttributeStage, PipelineStage)


def test_gesture_stage_is_pipeline_stage_subclass():
    assert issubclass(GestureStage, PipelineStage)


# ======================================================================
# 1. Construction
# ======================================================================

def test_attribute_construction_default():
    stage = AttributeStage()
    assert stage.estimator is None
    assert stage.name == "AttributeStage"


def test_attribute_construction_with_estimator():
    estimator = object()
    stage = AttributeStage(estimator)
    assert stage.estimator is estimator


def test_attribute_construction_custom_name():
    stage = AttributeStage(name="attr")
    assert stage.name == "attr"


def test_gesture_construction_default():
    stage = GestureStage()
    assert stage.recogniser is None
    assert stage.name == "GestureStage"


def test_gesture_construction_with_recogniser():
    recogniser = object()
    stage = GestureStage(recogniser)
    assert stage.recogniser is recogniser


def test_gesture_construction_custom_name():
    stage = GestureStage(name="gest")
    assert stage.name == "gest"


# ======================================================================
# 2. Capability declaration
# ======================================================================

def test_attribute_capability():
    cap = AttributeStage().capability
    assert cap.name == "attribute"
    assert cap.version == "0.1.0"
    assert cap.required_context == ["target_states"]
    assert cap.provided_context == ["target_states"]
    assert "placeholder" in cap.description.lower()


def test_gesture_capability():
    cap = GestureStage().capability
    assert cap.name == "gesture"
    assert cap.version == "0.1.0"
    assert cap.required_context == ["target_states"]
    assert cap.provided_context == ["target_states"]
    assert "placeholder" in cap.description.lower()


# ======================================================================
# 3. Lifecycle
# ======================================================================

def test_attribute_lifecycle():
    stage = AttributeStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_gesture_lifecycle():
    stage = GestureStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_attribute_initialize_idempotent():
    stage = AttributeStage()
    stage.initialize()
    stage.initialize()
    assert stage.health_check() is True


def test_gesture_shutdown_never_raises():
    stage = GestureStage()
    stage.shutdown()
    assert stage.health_check() is False


# ======================================================================
# 4. AttributeStage.process
# ======================================================================

def test_attribute_sets_attributes_on_each_snapshot():
    """process() sets attributes={} on every TargetState."""
    stage = AttributeStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.extend([_snapshot(1), _snapshot(2), _snapshot(3)])
    stage.process(ctx)
    assert len(ctx.target_states) == 3
    for ts in ctx.target_states:
        assert hasattr(ts, "attributes")
        assert ts.attributes == {}


def test_attribute_preserves_target_states_list_identity():
    """process() mutates the list in place (clear+extend), not rebinds."""
    stage = AttributeStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    orig_list = ctx.target_states
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    assert ctx.target_states is orig_list


def test_attribute_preserves_snapshot_fields():
    """process() preserves all original fields on derived snapshots."""
    stage = AttributeStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(42, "car"))
    stage.process(ctx)
    ts = ctx.target_states[0]
    assert ts.target_id == 42
    assert ts.label == "car"
    assert ts.confidence == 0.9


def test_attribute_with_empty_target_states():
    """process() is a no-op on empty target_states."""
    stage = AttributeStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    assert ctx.target_states == []
    stage.process(ctx)
    assert ctx.target_states == []


def test_attribute_replaces_stale_snapshots():
    """process() uses REPLACE semantics -- old snapshots are gone."""
    stage = AttributeStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    old = _snapshot(1)
    ctx.target_states.append(old)
    stage.process(ctx)
    assert ctx.target_states[0] is not old


def test_attribute_check_context_enforced():
    """process() calls check_context -- missing target_states raises."""
    stage = AttributeStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    del ctx.target_states
    with raises(AdvancedStageError):
        stage.process(ctx)


def test_attribute_does_not_touch_other_context_fields():
    """process() only reads/writes target_states, nothing else."""
    stage = AttributeStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    ctx.metadata["before"] = True
    stage.process(ctx)
    assert ctx.metadata["before"] is True
    assert ctx.detections == []
    assert ctx.tracks == []


# ======================================================================
# 5. GestureStage.process
# ======================================================================

def test_gesture_sets_gesture_on_each_snapshot():
    """process() sets gesture=None on every TargetState."""
    stage = GestureStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.extend([_snapshot(1), _snapshot(2)])
    stage.process(ctx)
    assert len(ctx.target_states) == 2
    for ts in ctx.target_states:
        assert hasattr(ts, "gesture")
        assert ts.gesture is None


def test_gesture_preserves_target_states_list_identity():
    """process() mutates the list in place (clear+extend), not rebinds."""
    stage = GestureStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    orig_list = ctx.target_states
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    assert ctx.target_states is orig_list


def test_gesture_preserves_snapshot_fields():
    """process() preserves all original fields on derived snapshots."""
    stage = GestureStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(7, "head"))
    stage.process(ctx)
    ts = ctx.target_states[0]
    assert ts.target_id == 7
    assert ts.label == "head"


def test_gesture_with_empty_target_states():
    """process() is a no-op on empty target_states."""
    stage = GestureStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    stage.process(ctx)
    assert ctx.target_states == []


def test_gesture_replaces_stale_snapshots():
    """process() uses REPLACE semantics -- old snapshots are gone."""
    stage = GestureStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    old = _snapshot(1)
    ctx.target_states.append(old)
    stage.process(ctx)
    assert ctx.target_states[0] is not old


def test_gesture_check_context_enforced():
    """process() calls check_context -- missing target_states raises."""
    stage = GestureStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    del ctx.target_states
    with raises(AdvancedStageError):
        stage.process(ctx)


# ======================================================================
# 6. Chaining: AttributeStage -> GestureStage
# ======================================================================

def test_attribute_then_gesture_chain():
    """AttributeStage -> GestureStage: both fields set on each snapshot."""
    attr = AttributeStage(name="attr")
    gest = GestureStage(name="gest")
    attr.initialize()
    gest.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.extend([_snapshot(1), _snapshot(2)])
    attr.process(ctx)
    gest.process(ctx)
    assert len(ctx.target_states) == 2
    for ts in ctx.target_states:
        assert ts.attributes == {}
        assert ts.gesture is None
        assert ts.target_id in (1, 2)


def test_gesture_then_attribute_chain():
    """GestureStage -> AttributeStage: order doesn't matter for fields."""
    gest = GestureStage(name="gest")
    attr = AttributeStage(name="attr")
    gest.initialize()
    attr.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    gest.process(ctx)
    attr.process(ctx)
    ts = ctx.target_states[0]
    assert ts.gesture is None
    assert ts.attributes == {}


# ======================================================================
# 7. Pipeline integration
# ======================================================================

def test_pipeline_with_attribute_stage():
    stage = AttributeStage(name="attr")
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    with p:
        p.run(ctx)
    assert ctx.target_states[0].attributes == {}
    assert stage.health_check() is False


def test_pipeline_with_gesture_stage():
    stage = GestureStage(name="gest")
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    with p:
        p.run(ctx)
    assert ctx.target_states[0].gesture is None
    assert stage.health_check() is False


# ======================================================================
# 8. No model / GUI / ai dependency
# ======================================================================

def test_no_model_imports_in_source():
    """attribute_gesture_stage.py contains no model imports (AST check)."""
    import visioncore.pipeline.stages.attribute_gesture_stage as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("yolo", "rt_detr", "rtdetr", "groundingdino", "sam",
              "ultralytics", "torch", "onnx", "cv2")
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
    assert not violations, f"attribute_gesture_stage.py imports banned: {violations}"


def test_no_gui_ai_in_source():
    """attribute_gesture_stage.py contains no gui/ai imports (AST check)."""
    import visioncore.pipeline.stages.attribute_gesture_stage as mod
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
    assert not violations, f"attribute_gesture_stage.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing attribute_gesture_stage does not load torch."""
    import visioncore.pipeline.stages.attribute_gesture_stage  # noqa: F401
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
    print(f"Running {len(tests)} attribute/gesture stage tests...\n")
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
    print("All attribute/gesture stage tests passed.")
