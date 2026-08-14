"""AdvancedStage / StageCapability unit tests (Milestone D1).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Package surface / exports
    - StageCapability dataclass (frozen, kw_only, fields, keyword-readable
      construction, "target_states" vocabulary)
    - AdvancedStage ABC (subclass of PipelineStage, same four-method
      lifecycle, capability property, not instantiable)
    - AdvancedStage.check_context contract validation
    - DummyAdvancedStage (lifecycle callability, counters, trace/read/write
      behaviour, raise_on_process, reset)
    - Context constraint: simulate PipelineContext with a restrictive
      tracing double -- process must touch ONLY declared fields
    - No GUI / InferWorker dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.pipeline import PipelineContext, PipelineStage
from visioncore.pipeline.stages.advanced import (
    AdvancedStage,
    AdvancedStageError,
    DummyAdvancedStage,
    StageCapability,
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


def _cap(
    required: list[str] | None = None,
    provided: list[str] | None = None,
    name: str = "test-plugin",
) -> StageCapability:
    """Build a keyword-constructed StageCapability for tests."""
    return StageCapability(
        name=name,
        version="1.2.3",
        description="Test capability built with keyword arguments.",
        required_context=required if required is not None else ["detections"],
        provided_context=provided if provided is not None else ["target_states"],
    )


def _snapshot() -> TargetState:
    """Build a synthetic TargetState snapshot."""
    return TargetState(
        target_id=1, local_id=1, global_id=None, label="person",
        confidence=0.9, cx=0.5, cy=0.5, vx=0.0, vy=0.0,
        width=0.2, height=0.4, timestamp=0.0, camera_id=0, metadata={},
    )


class _TracingContext:
    """Simulated PipelineContext: logs attribute access, blocks undeclared fields.

    A stand-in for :class:`PipelineContext` that makes the "only touches
    declared fields" property of a stage's ``process`` observable: any
    attribute outside the ``allowed`` set raises ``AttributeError``, and
    every access (declared or not) is recorded in ``_accessed``. If a
    stage's ``process`` succeeds against this double, it provably never
    read or wrote a field outside its declared contract.
    """

    __slots__ = ("_accessed", "_allowed", "detections", "tracks",
                 "targets", "target_states", "timestamp", "metadata")

    def __init__(self, allowed: set[str]) -> None:
        self._accessed: list[str] = []
        self._allowed: set[str] = set(allowed)
        self.detections: list[Any] = []
        self.tracks: list[Any] = []
        self.targets: list[Any] = []
        self.target_states: list[Any] = []
        self.timestamp: float = 0.0
        self.metadata: dict[str, Any] = {}

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        if name == "accessed":
            return set(object.__getattribute__(self, "_accessed"))
        object.__getattribute__(self, "_accessed").append(name)
        if name not in object.__getattribute__(self, "_allowed"):
            raise AttributeError(f"undeclared context field {name!r}")
        return object.__getattribute__(self, name)


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_package_exports_advanced_names():
    """The advanced package __all__ includes all four names."""
    import visioncore.pipeline.stages.advanced as pkg
    for name in ("AdvancedStage", "AdvancedStageError",
                 "DummyAdvancedStage", "StageCapability"):
        assert name in pkg.__all__, f"{name} missing from advanced __all__"


def test_advanced_error_is_runtime_error_subclass():
    assert issubclass(AdvancedStageError, RuntimeError)


def test_imports_do_not_break_existing_package():
    """Importing the advanced subpackage leaves the stages package intact."""
    import visioncore.pipeline.stages as stages_pkg
    assert "DetectorStage" in stages_pkg.__all__
    assert "TargetStage" in stages_pkg.__all__


# ======================================================================
# 1. StageCapability dataclass
# ======================================================================

def test_capability_is_frozen_kw_only_dataclass():
    assert dataclasses.is_dataclass(StageCapability)
    params = dataclasses.fields(StageCapability)
    assert {f.name for f in params} == {
        "name", "version", "description",
        "required_context", "provided_context",
    }
    assert StageCapability.__dataclass_params__.frozen is True
    sig = inspect.signature(StageCapability)
    assert all(
        p.kind is inspect.Parameter.KEYWORD_ONLY
        for p in sig.parameters.values()
    ), "every StageCapability field must be keyword-only"


def test_capability_kw_only_rejects_positional():
    """kw_only=True: positional construction must raise TypeError."""
    with raises(TypeError):
        StageCapability("a", "1.0.0", "d", [], [])  # type: ignore[misc]


def test_capability_keyword_construction_reads_clearly():
    """Keyword construction is self-documenting and fields are retained."""
    cap = StageCapability(
        name="denoise",
        version="1.0.0",
        description="Bilateral denoising of the frame before detection.",
        required_context=["frame"],
        provided_context=[],
    )
    assert cap.name == "denoise"
    assert cap.version == "1.0.0"
    assert cap.description.startswith("Bilateral denoising")
    assert cap.required_context == ["frame"]
    assert cap.provided_context == []


def test_capability_is_frozen():
    with raises(dataclasses.FrozenInstanceError):
        cap = _cap()
        cap.name = "renamed"  # type: ignore[misc]


def test_capability_provided_context_may_describe_target_states():
    """provided_context may carry the 'target_states' output vocabulary.

    ``provided_context=["target_states"]`` describes a plugin whose output
    is a list of TargetState snapshots.
    """
    cap = _cap(required=["targets"], provided=["target_states"])
    assert "target_states" in cap.provided_context
    assert cap.required_context == ["targets"]


def test_capability_keeps_passed_lists():
    """The declared lists are the exact lists passed by the caller."""
    req = ["frame", "detections"]
    prov = ["target_states"]
    cap = _cap(required=req, provided=prov)
    assert cap.required_context is req
    assert cap.provided_context is prov


def test_capability_may_contain_target_states_with_type_annotation_contract():
    """StageCapability docstring documents the List[TargetState] contract."""
    src = inspect.getsource(StageCapability)
    assert "target_states" in src
    assert "List[visioncore.state.target_state.TargetState]" in (
        inspect.getsource(
            __import__("visioncore.pipeline.stages.advanced.stage_capability",
                       fromlist=["x"])
        )
    )


# ======================================================================
# 2. AdvancedStage ABC
# ======================================================================

def test_advanced_stage_is_pipeline_stage_subclass():
    assert issubclass(AdvancedStage, PipelineStage)


def test_advanced_stage_is_abstract():
    with raises(TypeError):
        AdvancedStage()  # type: ignore[abstract]


def test_advanced_stage_same_four_method_lifecycle():
    """AdvancedStage keeps the same abstract lifecycle as PipelineStage."""
    base_methods = {
        name for name, _ in inspect.getmembers(
            PipelineStage, predicate=inspect.isfunction
        ) if not name.startswith("_")
    }
    advanced_methods = {
        name for name, _ in inspect.getmembers(
            AdvancedStage, predicate=inspect.isfunction
        ) if not name.startswith("_")
    }
    assert {"initialize", "process", "shutdown", "health_check"} <= base_methods
    assert {"initialize", "process", "shutdown", "health_check"} <= advanced_methods


def test_advanced_stage_subclass_missing_capability_fails():
    class NoCapability(AdvancedStage):
        def initialize(self) -> None: ...
        def process(self, context: PipelineContext) -> None: ...
        def shutdown(self) -> None: ...
        def health_check(self) -> bool:
            return True
    with raises(TypeError):
        NoCapability()  # type: ignore[abstract]


def test_advanced_stage_subclass_missing_lifecycle_fails():
    class NoLifecycle(AdvancedStage):
        @property
        def capability(self) -> StageCapability:
            return _cap()
    with raises(TypeError):
        NoLifecycle()  # type: ignore[abstract]


def test_advanced_stage_concrete_subclass_works():
    """A complete subclass instantiates, names itself, and reports capability."""
    class RefineStage(AdvancedStage):
        @property
        def capability(self) -> StageCapability:
            return _cap(name="refine")

        def initialize(self) -> None: ...
        def process(self, context: PipelineContext) -> None:
            self.check_context(context)
        def shutdown(self) -> None: ...
        def health_check(self) -> bool:
            return True

    stage = RefineStage()
    assert stage.name == "RefineStage"
    assert stage.capability.name == "refine"
    assert RefineStage(name="custom").name == "custom"


def test_advanced_stage_repr_reports_capability():
    stage = DummyAdvancedStage(name="denoise")
    stage.initialize()
    r = repr(stage)
    assert "AdvancedStage" in r or "DummyAdvancedStage" in r
    assert "name='denoise'" in r
    assert "capability=" in r
    assert "dummy-advanced" in r and "0.1.0" in r
    assert "healthy=True" in r


# ======================================================================
# 3. check_context contract validation
# ======================================================================

def test_check_context_passes_on_satisfied_context():
    stage = DummyAdvancedStage()
    ctx = PipelineContext.empty()
    stage.check_context(ctx)  # default contract (detections + target_states) ok


def test_check_context_missing_required_field():
    stage = DummyAdvancedStage()
    ctx = PipelineContext.empty()
    del ctx.detections
    with raises(AdvancedStageError, match="required_context"):
        stage.check_context(ctx)


def test_check_context_missing_provided_field():
    cap = _cap(required=[], provided=["target_states"])
    stage = DummyAdvancedStage(capability=cap)
    ctx = PipelineContext.empty()
    del ctx.target_states
    with raises(AdvancedStageError, match="provided_context"):
        stage.check_context(ctx)


def test_check_context_provided_field_not_list():
    cap = _cap(required=[], provided=["frame"])
    stage = DummyAdvancedStage(capability=cap)
    ctx = PipelineContext.empty()
    ctx.frame = "not a list"
    with raises(AdvancedStageError, match="must be a list"):
        stage.check_context(ctx)


def test_check_context_error_mentions_field_name():
    cap = _cap(required=["tracks"], provided=[])
    stage = DummyAdvancedStage(capability=cap)
    ctx = PipelineContext.empty()
    del ctx.tracks
    with raises(AdvancedStageError, match="'tracks'"):
        stage.check_context(ctx)


def test_check_context_multi_field_first_violation_wins():
    cap = _cap(required=["tracks", "targets"], provided=["target_states"])
    stage = DummyAdvancedStage(capability=cap)
    ctx = PipelineContext.empty()
    del ctx.tracks
    del ctx.targets
    with raises(AdvancedStageError, match="'tracks'"):
        stage.check_context(ctx)


# ======================================================================
# 4. DummyAdvancedStage
# ======================================================================

def test_dummy_default_capability():
    stage = DummyAdvancedStage()
    cap = stage.capability
    assert cap.name == "dummy-advanced"
    assert cap.version == "0.1.0"
    assert cap.required_context == ["detections"]
    assert cap.provided_context == ["target_states"]


def test_dummy_custom_capability():
    cap = _cap(required=["frame"], provided=["detections"])
    stage = DummyAdvancedStage(capability=cap)
    assert stage.capability is cap


def test_dummy_default_name():
    assert DummyAdvancedStage().name == "DummyAdvancedStage"


def test_dummy_custom_name():
    assert DummyAdvancedStage(name="denoise").name == "denoise"


def test_dummy_lifecycle_methods_are_callable():
    """initialize/process/shutdown/health_check are all callable and work."""
    stage = DummyAdvancedStage(name="probe")
    assert callable(stage.initialize)
    assert callable(stage.process)
    assert callable(stage.shutdown)
    assert callable(stage.health_check)
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.process(PipelineContext.empty())
    stage.shutdown()
    assert stage.health_check() is False
    assert (stage.initialize_count, stage.process_count, stage.shutdown_count) == (1, 1, 1)


def test_dummy_initialize_idempotent_health():
    stage = DummyAdvancedStage()
    stage.initialize()
    stage.initialize()
    assert stage.initialize_count == 2
    assert stage.health_check() is True


def test_dummy_process_traces_marker():
    stage = DummyAdvancedStage(name="denoise")
    stage.initialize()
    ctx = PipelineContext.empty()
    stage.process(ctx)
    assert ctx.metadata["advanced_trace"] == ["denoise"]


def test_dummy_process_reads_required_field():
    """process() reads the first required_context field (observable read)."""
    stage = DummyAdvancedStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.detections.append("d1")
    ctx.detections.append("d2")
    stage.process(ctx)
    assert ctx.metadata["advanced_reads"] == {"detections": 2}


def test_dummy_process_writes_provided_field():
    """process() appends its marker to every provided_context list."""
    stage = DummyAdvancedStage(name="denoise")
    stage.initialize()
    ctx = PipelineContext.empty()
    stage.process(ctx)
    assert ctx.target_states == ["denoise"]


def test_dummy_process_uses_write_value():
    cap = _cap(required=[], provided=["detections", "target_states"])
    stage = DummyAdvancedStage(capability=cap, write_value=_snapshot())
    stage.initialize()
    ctx = PipelineContext.empty()
    stage.process(ctx)
    assert ctx.detections == [_snapshot()]
    assert ctx.target_states == [_snapshot()]


def test_dummy_process_preserves_list_identity():
    stage = DummyAdvancedStage()
    stage.initialize()
    ctx = PipelineContext.empty()
    orig = ctx.target_states
    stage.process(ctx)
    assert ctx.target_states is orig
    assert len(orig) == 1


def test_dummy_process_no_required_field_skips_read_record():
    cap = _cap(required=[], provided=["target_states"])
    stage = DummyAdvancedStage(capability=cap)
    stage.initialize()
    ctx = PipelineContext.empty()
    stage.process(ctx)
    assert "advanced_reads" not in ctx.metadata
    assert len(ctx.target_states) == 1


def test_dummy_raise_on_process():
    stage = DummyAdvancedStage(raise_on_process=AdvancedStageError("boom"))
    stage.initialize()
    ctx = PipelineContext.empty()
    with raises(AdvancedStageError, match="boom"):
        stage.process(ctx)
    assert stage.process_count == 1
    assert ctx.metadata == {}  # nothing touched


def test_dummy_shutdown_never_raises():
    stage = DummyAdvancedStage()
    stage.initialize()
    stage.shutdown()
    assert stage.health_check() is False


def test_dummy_reset():
    stage = DummyAdvancedStage()
    stage.initialize()
    stage.process(PipelineContext.empty())
    stage.shutdown()
    stage.reset()
    assert (stage.initialize_count, stage.process_count, stage.shutdown_count) == (0, 0, 0)
    assert stage.health_check() is False


def test_dummy_capability_unchanged_by_reset():
    stage = DummyAdvancedStage()
    cap = stage.capability
    stage.reset()
    assert stage.capability is cap


# ======================================================================
# 5. Context constraint: process touches ONLY declared fields
# ======================================================================

def test_process_touches_only_declared_fields():
    """With a simulated PipelineContext that blocks undeclared fields,
    process() completes -- proving it only reads/writes its contract."""
    stage = DummyAdvancedStage(name="denoise")
    stage.initialize()
    allowed = {"detections", "target_states", "metadata", "timestamp"}
    ctx = _TracingContext(allowed)
    stage.process(ctx)
    # Every accessed name is inside the declared contract (+ metadata,
    # which is free-form by pipeline design).
    assert ctx.accessed <= allowed
    assert ctx.accessed >= {"detections", "target_states", "metadata"}


def test_process_blocks_undeclared_read():
    """A stage declaring a required field the context does not expose fails
    with AdvancedStageError -- the contract is enforced."""
    cap = _cap(required=["tracks"], provided=["target_states"])
    stage = DummyAdvancedStage(capability=cap)
    stage.initialize()
    ctx = _TracingContext({"detections", "target_states", "metadata", "timestamp"})
    with raises(AdvancedStageError, match="'tracks'"):
        stage.process(ctx)


def test_process_blocks_undeclared_write():
    """A stage declaring a provided field the context does not expose fails
    with AdvancedStageError before any write."""
    cap = _cap(required=[], provided=["targets"])
    stage = DummyAdvancedStage(capability=cap)
    stage.initialize()
    ctx = _TracingContext({"detections", "target_states", "metadata", "timestamp"})
    with raises(AdvancedStageError, match="'targets'"):
        stage.process(ctx)
    assert ctx.target_states == []  # nothing was written


def test_process_contract_honest_on_real_context():
    """On a real PipelineContext the full contract check runs end-to-end."""
    stage = DummyAdvancedStage()
    stage.initialize()
    ctx = PipelineContext.empty(timestamp=7.5)
    ctx.detections.append("d1")
    stage.process(ctx)
    assert ctx.metadata["advanced_reads"] == {"detections": 1}
    assert ctx.target_states == ["DummyAdvancedStage"]


# ======================================================================
# 6. Pipeline integration (advanced stage plugs in like any stage)
# ======================================================================

def test_advanced_stage_in_pipeline():
    """An advanced stage runs inside a Pipeline like a normal stage."""
    from visioncore.pipeline import Pipeline

    p = Pipeline()
    p.add_stage(DummyAdvancedStage(name="refine"))
    ctx = PipelineContext.empty()
    ctx.detections.append("d1")
    with p:
        p.run(ctx)
    assert ctx.metadata["advanced_trace"] == ["refine"]
    assert len(ctx.target_states) == 1


def test_advanced_stage_repr_in_pipeline_diagnostics():
    stage = DummyAdvancedStage(name="refine")
    r = repr(stage)
    assert "refine" in r and "dummy-advanced" in r


# ======================================================================
# 7. No GUI / InferWorker dependency
# ======================================================================

def test_no_gui_inferworker_in_source():
    """The advanced modules contain no GUI/InferWorker refs (AST check)."""
    import visioncore.pipeline.stages.advanced.advanced_stage as mod
    src = open(mod.__file__).read()
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
    assert not violations, f"advanced_stage.py imports banned: {violations}"


def test_no_gui_inferworker_loaded_at_runtime():
    """Importing the advanced package does not load GUI/InferWorker."""
    import sys
    import visioncore.pipeline.stages.advanced  # noqa: F401
    banned = ["PyQt6", "PyQt5"]
    loaded = [b for b in banned if b in sys.modules]
    assert not loaded, f"GUI loaded as side effect: {loaded}"
    assert "ai.inference" not in sys.modules, "ai.inference loaded as side effect"


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
    print(f"Running {len(tests)} advanced stage tests...\n")
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
    print("All advanced stage tests passed.")
