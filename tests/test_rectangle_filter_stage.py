"""RectangleStage / FilterStage / HealthStage unit tests (Milestone D5).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Construction (with/without backend, default/custom name)
    - Capability declaration (name, version, required/provided context)
    - Lifecycle (initialize/process/shutdown/health_check)
    - RectangleStage.process: adds a rect=(x1,y1,x2,y2) field of the
      correct type to each TargetState, derived from cx/cy/width/height
    - FilterStage.process: evaluates the placeholder predicate but
      deletes nothing (callable, list untouched)
    - HealthStage.process: adds a numeric health=1.0 field to each
      TargetState
    - Backends held but never called
    - Chaining (Rectangle -> Filter -> Health)
    - Pipeline integration
    - check_context enforced
    - No model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import math
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages.advanced import AdvancedStage, AdvancedStageError
from visioncore.pipeline.stages.rectangle_filter_stage import (
    FilterStage,
    HealthStage,
    RectangleStage,
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


class _SpyBackend:
    """Test double: fails loudly if the stage ever calls its backend."""

    __slots__ = ("called",)

    def __init__(self) -> None:
        self.called: bool = False

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.called = True
        raise AssertionError(
            "D5 backends must be held, never called (spy was invoked)"
        )


def _snapshot(
    target_id: int = 1,
    label: str = "person",
    confidence: float = 0.9,
    rect: tuple[float, float, float, float] | None = None,
    health: float | None = None,
) -> TargetState:
    """Build a synthetic TargetState snapshot for tests."""
    kwargs: dict[str, Any] = dict(
        target_id=target_id, local_id=1, global_id=None, label=label,
        confidence=confidence, cx=0.5, cy=0.5, vx=0.0, vy=0.0,
        width=0.2, height=0.4, timestamp=0.0, camera_id=0, metadata={},
    )
    if rect is not None:
        kwargs["rect"] = rect
    if health is not None:
        kwargs["health"] = health
    return TargetState(**kwargs)


def _close(a: float, b: float) -> bool:
    """Float comparison with tight tolerances (rect is derived arithmetic)."""
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_rectangle_stage_is_advanced_stage_subclass():
    assert issubclass(RectangleStage, AdvancedStage)


def test_filter_stage_is_advanced_stage_subclass():
    assert issubclass(FilterStage, AdvancedStage)


def test_health_stage_is_advanced_stage_subclass():
    assert issubclass(HealthStage, AdvancedStage)


def test_rectangle_stage_is_pipeline_stage_subclass():
    assert issubclass(RectangleStage, PipelineStage)


def test_filter_stage_is_pipeline_stage_subclass():
    assert issubclass(FilterStage, PipelineStage)


def test_health_stage_is_pipeline_stage_subclass():
    assert issubclass(HealthStage, PipelineStage)


# ======================================================================
# 1. Construction
# ======================================================================

def test_rectangle_construction_default():
    stage = RectangleStage()
    assert stage.rect_builder is None
    assert stage.name == "RectangleStage"


def test_rectangle_construction_with_rect_builder():
    rect_builder = object()
    stage = RectangleStage(rect_builder)
    assert stage.rect_builder is rect_builder


def test_rectangle_construction_custom_name():
    stage = RectangleStage(name="rect")
    assert stage.name == "rect"


def test_filter_construction_default():
    stage = FilterStage()
    assert stage.filterer is None
    assert stage.name == "FilterStage"


def test_filter_construction_with_filterer():
    filterer = object()
    stage = FilterStage(filterer)
    assert stage.filterer is filterer


def test_filter_construction_custom_name():
    stage = FilterStage(name="filter")
    assert stage.name == "filter"


def test_health_construction_default():
    stage = HealthStage()
    assert stage.health_monitor is None
    assert stage.name == "HealthStage"


def test_health_construction_with_health_monitor():
    health_monitor = object()
    stage = HealthStage(health_monitor)
    assert stage.health_monitor is health_monitor


def test_health_construction_custom_name():
    stage = HealthStage(name="health")
    assert stage.name == "health"


# ======================================================================
# 2. Capability declaration
# ======================================================================

def test_rectangle_capability():
    cap = RectangleStage().capability
    assert cap.name == "rectangle"
    assert cap.version == "0.1.0"
    assert cap.required_context == ["target_states"]
    assert cap.provided_context == ["target_states"]
    assert "placeholder" in cap.description.lower()


def test_filter_capability():
    cap = FilterStage().capability
    assert cap.name == "filter"
    assert cap.version == "0.1.0"
    assert cap.required_context == ["target_states"]
    assert cap.provided_context == ["target_states"]
    assert "placeholder" in cap.description.lower()


def test_health_capability():
    cap = HealthStage().capability
    assert cap.name == "health"
    assert cap.version == "0.1.0"
    assert cap.required_context == ["target_states"]
    assert cap.provided_context == ["target_states"]
    assert "placeholder" in cap.description.lower()


# ======================================================================
# 3. Lifecycle
# ======================================================================

def test_rectangle_lifecycle():
    stage = RectangleStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_filter_lifecycle():
    stage = FilterStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_health_lifecycle():
    stage = HealthStage()
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


# ======================================================================
# 4. RectangleStage.process
# ======================================================================

def test_rectangle_adds_rect_field_with_correct_type():
    """process() adds a rect field of the correct type to every TargetState."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.extend([_snapshot(1), _snapshot(2), _snapshot(3)])
    stage.process(ctx)
    assert len(ctx.target_states) == 3
    for ts in ctx.target_states:
        assert hasattr(ts, "rect")
        assert isinstance(ts.rect, tuple)
        assert len(ts.rect) == 4
        for value in ts.rect:
            assert isinstance(value, float)


def test_rectangle_rect_values_match_corner_derivation():
    """rect equals (cx-w/2, cy-h/2, cx+w/2, cy+h/2) for each snapshot."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    # cx=0.5, cy=0.5, width=0.2, height=0.4 -> (0.4, 0.3, 0.6, 0.7)
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    x1, y1, x2, y2 = ctx.target_states[0].rect
    assert _close(x1, 0.4)
    assert _close(y1, 0.3)
    assert _close(x2, 0.6)
    assert _close(y2, 0.7)


def test_rectangle_rect_respects_per_snapshot_box():
    """Different boxes yield different rects, each derived from its own fields."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    a = _snapshot(1)
    b = _snapshot(2)
    b = b.copy_with(cx=0.25, cy=0.75, width=0.5, height=0.1)
    ctx.target_states.extend([a, b])
    stage.process(ctx)
    r1 = ctx.target_states[0].rect
    r2 = ctx.target_states[1].rect
    # a: (0.4, 0.3, 0.6, 0.7); b: (0.0, 0.7, 0.5, 0.8)
    assert _close(r1[0], 0.4) and _close(r1[2], 0.6)
    assert _close(r2[0], 0.0) and _close(r2[2], 0.5)
    assert _close(r2[1], 0.7) and _close(r2[3], 0.8)


def test_rectangle_preserves_target_states_list_identity():
    """process() mutates the list in place (clear+extend), not rebinds."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    orig_list = ctx.target_states
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    assert ctx.target_states is orig_list


def test_rectangle_preserves_snapshot_fields():
    """process() preserves all original fields on derived snapshots."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(42, "car"))
    stage.process(ctx)
    ts = ctx.target_states[0]
    assert ts.target_id == 42
    assert ts.label == "car"
    assert ts.confidence == 0.9
    assert _close(ts.cx, 0.5) and _close(ts.width, 0.2)


def test_rectangle_with_empty_target_states():
    """process() is a no-op on empty target_states."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    assert ctx.target_states == []
    stage.process(ctx)
    assert ctx.target_states == []


def test_rectangle_replaces_stale_snapshots():
    """process() uses REPLACE semantics -- old snapshots are gone."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    old = _snapshot(1)
    ctx.target_states.append(old)
    stage.process(ctx)
    assert ctx.target_states[0] is not old


def test_rectangle_check_context_enforced():
    """process() calls check_context -- missing target_states raises."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    del ctx.target_states
    with raises(AdvancedStageError):
        stage.process(ctx)


def test_rectangle_does_not_touch_other_context_fields():
    """process() only reads/writes target_states, nothing else."""
    stage = RectangleStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    ctx.metadata["before"] = True
    stage.process(ctx)
    assert ctx.metadata["before"] is True
    assert ctx.detections == []
    assert ctx.tracks == []


# ======================================================================
# 5. FilterStage.process
# ======================================================================

def test_filter_deletes_nothing_but_is_callable():
    """process() runs without error and deletes no targets."""
    stage = FilterStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    originals = [_snapshot(1), _snapshot(2, confidence=0.5), _snapshot(3, confidence=0.75)]
    ctx.target_states.extend(originals)
    orig_list = ctx.target_states
    stage.process(ctx)
    assert len(ctx.target_states) == 3
    # List identity AND element identity preserved -- nothing removed.
    assert ctx.target_states is orig_list
    for i, ts in enumerate(ctx.target_states):
        assert ts is originals[i]


def test_filter_keeps_anomalous_negative_confidence():
    """Even a confidence<0 snapshot is kept: the D5 predicate deletes nothing."""
    stage = FilterStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    weird = _snapshot(1, confidence=-0.5)
    ctx.target_states.append(weird)
    stage.process(ctx)
    assert len(ctx.target_states) == 1
    assert ctx.target_states[0] is weird


def test_filter_with_empty_target_states():
    """process() is callable on an empty list without error."""
    stage = FilterStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    stage.process(ctx)
    assert ctx.target_states == []


def test_filter_check_context_enforced():
    """process() calls check_context -- missing target_states raises."""
    stage = FilterStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    del ctx.target_states
    with raises(AdvancedStageError):
        stage.process(ctx)


def test_filter_does_not_touch_other_context_fields():
    """process() only reads target_states, nothing else."""
    stage = FilterStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    ctx.metadata["before"] = True
    stage.process(ctx)
    assert ctx.metadata["before"] is True
    assert ctx.detections == []
    assert ctx.tracks == []


# ======================================================================
# 6. HealthStage.process
# ======================================================================

def test_health_adds_health_numeric_field_on_each():
    """process() adds a numeric health field to every TargetState."""
    stage = HealthStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.extend([_snapshot(1), _snapshot(2)])
    stage.process(ctx)
    assert len(ctx.target_states) == 2
    for ts in ctx.target_states:
        assert hasattr(ts, "health")
        assert isinstance(ts.health, float)
        assert ts.health == 1.0


def test_health_preserves_target_states_list_identity():
    """process() mutates the list in place (clear+extend), not rebinds."""
    stage = HealthStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    orig_list = ctx.target_states
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    assert ctx.target_states is orig_list


def test_health_preserves_snapshot_fields():
    """process() preserves all original fields on derived snapshots."""
    stage = HealthStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(7, "head"))
    stage.process(ctx)
    ts = ctx.target_states[0]
    assert ts.target_id == 7
    assert ts.label == "head"
    assert ts.health == 1.0


def test_health_with_empty_target_states():
    """process() is a no-op on empty target_states."""
    stage = HealthStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    stage.process(ctx)
    assert ctx.target_states == []


def test_health_replaces_stale_snapshots():
    """process() uses REPLACE semantics -- old snapshots are gone."""
    stage = HealthStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    old = _snapshot(1)
    ctx.target_states.append(old)
    stage.process(ctx)
    assert ctx.target_states[0] is not old


def test_health_check_context_enforced():
    """process() calls check_context -- missing target_states raises."""
    stage = HealthStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    del ctx.target_states
    with raises(AdvancedStageError):
        stage.process(ctx)


def test_health_does_not_touch_other_context_fields():
    """process() only reads/writes target_states, nothing else."""
    stage = HealthStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    ctx.metadata["before"] = True
    stage.process(ctx)
    assert ctx.metadata["before"] is True
    assert ctx.detections == []
    assert ctx.tracks == []


# ======================================================================
# 7. Backends are held, never called
# ======================================================================

def test_rectangle_backend_not_called():
    stage = RectangleStage(_SpyBackend())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    assert stage.rect_builder.called is False


def test_filter_backend_not_called():
    stage = FilterStage(_SpyBackend())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    assert stage.filterer.called is False


def test_health_backend_not_called():
    stage = HealthStage(_SpyBackend())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    stage.process(ctx)
    assert stage.health_monitor.called is False


# ======================================================================
# 8. Chaining
# ======================================================================

def test_rectangle_filter_health_chain():
    """RectangleStage -> FilterStage -> HealthStage: rect + health present."""
    rect = RectangleStage(name="rect")
    filt = FilterStage(name="filter")
    health = HealthStage(name="health")
    rect.initialize()
    filt.initialize()
    health.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.extend([_snapshot(1), _snapshot(2)])
    rect.process(ctx)
    filt.process(ctx)
    health.process(ctx)
    assert len(ctx.target_states) == 2
    for ts in ctx.target_states:
        assert isinstance(ts.rect, tuple)
        assert ts.health == 1.0
        assert ts.target_id in (1, 2)


def test_health_then_rectangle_chain():
    """Order independence: rect and health coexist regardless of order."""
    health = HealthStage(name="health")
    rect = RectangleStage(name="rect")
    health.initialize()
    rect.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    health.process(ctx)
    rect.process(ctx)
    ts = ctx.target_states[0]
    assert ts.health == 1.0
    assert isinstance(ts.rect, tuple)


# ======================================================================
# 9. Pipeline integration
# ======================================================================

def test_pipeline_with_d5_stages():
    stage = RectangleStage(name="rect")
    filt = FilterStage(name="filter")
    health = HealthStage(name="health")
    p = Pipeline()
    p.add_stage(stage)
    p.add_stage(filt)
    p.add_stage(health)
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1))
    with p:
        p.run(ctx)
    ts = ctx.target_states[0]
    assert isinstance(ts.rect, tuple)
    assert len(ts.rect) == 4
    assert ts.health == 1.0
    assert stage.health_check() is False


# ======================================================================
# 10. No model / GUI / ai dependency
# ======================================================================

def test_no_model_imports_in_source():
    """rectangle_filter_stage.py contains no model imports (AST check)."""
    import visioncore.pipeline.stages.rectangle_filter_stage as mod
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
    assert not violations, f"rectangle_filter_stage.py imports banned: {violations}"


def test_no_gui_ai_in_source():
    """rectangle_filter_stage.py contains no gui/ai imports (AST check)."""
    import visioncore.pipeline.stages.rectangle_filter_stage as mod
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
    assert not violations, f"rectangle_filter_stage.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing rectangle_filter_stage does not load torch."""
    import visioncore.pipeline.stages.rectangle_filter_stage  # noqa: F401
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
    print(f"Running {len(tests)} rectangle/filter/health stage tests...\n")
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
    print("All rectangle/filter/health stage tests passed.")
