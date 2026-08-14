"""Pipeline foundation unit tests (Milestone C1).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary. A tiny ``raises``
context manager replaces ``pytest.raises`` for exception assertions.

Covers the four contractually-required areas plus the framework's own
invariants:

    1. Stage ordering          -- stages run in insertion order
    2. Context passing         -- stages share one context instance
    3. Exception propagation   -- a stage exception aborts the run
    4. Stage lifecycle         -- initialize/process/shutdown call counts + order

Plus:
    - Package exports + import surface
    - PipelineContext fields, factories, summary
    - PipelineStage ABC contract (cannot instantiate, subclass must implement)
    - Pipeline add_stage / remove_stage / registry invariants
    - Pipeline lifecycle state machine (NEW -> INITIALIZED -> SHUT_DOWN)
    - Pipeline health_check aggregation
    - Pipeline context manager (guaranteed teardown on exception)
    - DummyStage configuration (marker, raise_on_process, health)
"""
from __future__ import annotations

import contextlib
import traceback
from typing import Any

from visioncore.pipeline import (
    DummyStage,
    Pipeline,
    PipelineContext,
    PipelineStage,
    StageError,
)


# ======================================================================
# Test helpers (replace pytest dependencies)
# ======================================================================

class _Raises:
    """Context manager asserting that a block raises a matching exception.

    Replaces ``pytest.raises``. Usage::

        with raises(ValueError, match="boom"):
            raise ValueError("boom")

    Fails loudly (via ``AssertionError``) if no exception is raised, or if
    the raised exception is not an instance of ``expected``, or (when
    ``match`` is given) if ``match`` is not a substring of ``str(exc)``.
    """

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
                f"expected {self.expected.__name__} to be raised, "
                f"but no exception was raised"
            )
        if not isinstance(exc_val, BaseException):
            # Should not happen in practice, but be defensive.
            raise AssertionError(
                f"expected {self.expected.__name__}, got non-exception {exc_val!r}"
            )
        if not isinstance(exc_val, self.expected):
            raise AssertionError(
                f"expected {self.expected.__name__}, got "
                f"{type(exc_val).__name__}: {exc_val}"
            )
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(
                f"expected {self.match!r} in str({type(exc_val).__name__}), "
                f"got {str(exc_val)!r}"
            )
        self.caught = exc_val
        return True  # suppress the caught exception


def raises(
    expected: type[BaseException], match: str | None = None,
) -> _Raises:
    """Assert that the enclosed block raises ``expected``.

    Drop-in replacement for ``pytest.raises``. See :class:`_Raises`.
    """
    return _Raises(expected, match=match)


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_package_exports_public_api():
    """The package __all__ includes the five C1 framework names.

    C7 added shadow-runner exports to the same package; this test asserts
    the C1 framework names are present (subset), not that __all__ is
    exactly the C1 set -- the package legitimately grows as components
    are added.
    """
    import visioncore.pipeline as pkg
    c1_names = {
        "Pipeline", "PipelineStage", "PipelineContext",
        "DummyStage", "StageError",
    }
    assert c1_names.issubset(set(pkg.__all__)), (
        f"C1 framework names {c1_names} not all in pipeline __all__: {pkg.__all__}"
    )


def test_pipeline_stage_is_abstract():
    """PipelineStage is an ABC and cannot be instantiated directly."""
    with raises(TypeError):
        PipelineStage()  # type: ignore[abstract]


def test_stage_error_is_runtime_error_subclass():
    """StageError is a RuntimeError subclass (structured but not exotic)."""
    assert issubclass(StageError, RuntimeError)


def test_subclass_missing_methods_fails():
    """A subclass that omits any abstract method cannot be instantiated."""
    class Incomplete(PipelineStage):
        def initialize(self) -> None:
            pass
        # process / shutdown / health_check missing
    with raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ======================================================================
# 1. PipelineContext
# ======================================================================

def test_context_empty_factory_sets_defaults():
    """empty() produces a context with all collections empty and frame None."""
    ctx = PipelineContext.empty(timestamp=2.5)
    assert ctx.frame is None
    assert ctx.detections == []
    assert ctx.tracks == []
    assert ctx.targets == []
    assert ctx.target_states == []
    assert ctx.timestamp == 2.5
    assert ctx.metadata == {}


def test_context_empty_default_timestamp_is_zero():
    """empty() with no arg defaults timestamp to 0.0."""
    ctx = PipelineContext.empty()
    assert ctx.timestamp == 0.0


def test_context_metadata_is_independent_per_instance():
    """Two empty contexts do not share the same metadata dict."""
    a = PipelineContext.empty()
    b = PipelineContext.empty()
    a.metadata["k"] = "v"
    assert "k" not in b.metadata


def test_context_summary_reports_counts():
    """summary() reports frame presence and collection lengths."""
    ctx = PipelineContext.empty(timestamp=1.0)
    s = ctx.summary()
    assert "frame=no" in s
    assert "det=0" in s
    assert "trk=0" in s
    assert "tgt=0" in s
    assert "ts=0" in s  # target_states count, abbreviated
    assert "len=0" in s  # metadata length


def test_context_summary_shows_frame_yes_when_set():
    """summary() reports frame=yes when frame is truthy."""
    ctx = PipelineContext.empty()
    ctx.frame = object()  # pretend frame; type not enforced here
    assert "frame=yes" in ctx.summary()


def test_context_repr_is_compact():
    """__repr__ does not dump large collections; uses summary()."""
    ctx = PipelineContext.empty(timestamp=3.14)
    ctx.detections.extend(["d"] * 1000)  # type: ignore[list-item]
    r = repr(ctx)
    assert "PipelineContext(" in r
    assert "det=1000" in r
    # Must not contain 1000 copies of 'd'
    assert r.count("d") < 10


# ======================================================================
# 2. PipelineStage ABC contract
# ======================================================================

def test_pipeline_stage_name_defaults_to_class_name():
    """A stage with no explicit name uses type(self).__name__."""
    class MyStage(PipelineStage):
        def initialize(self) -> None:
            pass
        def process(self, context: PipelineContext) -> None:
            pass
        def shutdown(self) -> None:
            pass
        def health_check(self) -> bool:
            return True

    s = MyStage()
    assert s.name == "MyStage"


def test_pipeline_stage_explicit_name():
    """An explicit name overrides the class-name default."""
    class S(PipelineStage):
        def initialize(self) -> None:
            pass
        def process(self, context: PipelineContext) -> None:
            pass
        def shutdown(self) -> None:
            pass
        def health_check(self) -> bool:
            return True

    assert S(name="custom").name == "custom"


def test_pipeline_stage_repr_reports_health():
    """__repr__ calls health_check and reports it, without raising."""
    s = DummyStage("x")
    s.initialize()
    r = repr(s)
    assert "DummyStage" in r
    assert "name='x'" in r
    assert "healthy=True" in r


def test_pipeline_stage_repr_survives_health_check_exception():
    """A buggy health_check() that raises does not crash __repr__."""
    class BrokenHealth(PipelineStage):
        def initialize(self) -> None:
            pass
        def process(self, context: PipelineContext) -> None:
            pass
        def shutdown(self) -> None:
            pass
        def health_check(self) -> bool:
            raise RuntimeError("boom")

    s = BrokenHealth("broken")
    r = repr(s)  # must not raise
    assert "healthy=False" in r


# ======================================================================
# 3. DummyStage
# ======================================================================

def test_dummy_stage_defaults():
    """A freshly constructed DummyStage has zero counts and is unhealthy."""
    s = DummyStage("a")
    assert s.name == "a"
    assert s.marker == "a"
    assert s.raise_on_process is None
    assert s.healthy is False
    assert (s.initialize_count, s.process_count, s.shutdown_count) == (0, 0, 0)


def test_dummy_stage_marker_overrides_name():
    """marker can differ from name."""
    s = DummyStage("a", marker="MARK_A")
    assert s.name == "a"
    assert s.marker == "MARK_A"


def test_dummy_stage_initialize_makes_healthy():
    """initialize() flips healthy to True and increments the counter."""
    s = DummyStage()
    s.initialize()
    assert s.healthy is True
    assert s.initialize_count == 1


def test_dummy_stage_shutdown_makes_unhealthy():
    """shutdown() flips healthy to False and increments the counter."""
    s = DummyStage()
    s.initialize()
    s.shutdown()
    assert s.healthy is False
    assert s.shutdown_count == 1


def test_dummy_stage_process_stamps_context_order():
    """process() appends the marker to context.metadata['order']."""
    s = DummyStage("probe")
    s.initialize()
    ctx = PipelineContext.empty()
    s.process(ctx)
    assert ctx.metadata["order"] == ["probe"]
    assert s.process_count == 1


def test_dummy_stage_reset_clears_counters():
    """reset() zeroes counters and health, leaving marker intact."""
    s = DummyStage("a", marker="M")
    s.initialize()
    s.process(PipelineContext.empty())
    s.shutdown()
    s.reset()
    assert (s.initialize_count, s.process_count, s.shutdown_count) == (0, 0, 0)
    assert s.healthy is False
    assert s.marker == "M"  # marker preserved


# ======================================================================
# 4. Pipeline registry: add_stage / remove_stage
# ======================================================================

def test_add_stage_appends_in_order():
    """add_stage appends to the end; stages tuple preserves order."""
    p = Pipeline()
    a, b, c = DummyStage("a"), DummyStage("b"), DummyStage("c")
    p.add_stage(a)
    p.add_stage(b)
    p.add_stage(c)
    assert list(p.stages) == [a, b, c]


def test_add_stage_returns_self_for_chaining():
    """add_stage returns self so calls can be chained."""
    p = Pipeline()
    assert p.add_stage(DummyStage("a")) is p
    p.add_stage(DummyStage("b")).add_stage(DummyStage("c"))
    assert len(p) == 3


def test_add_stage_rejects_none_and_non_stage():
    """add_stage raises TypeError for None or non-PipelineStage objects."""
    p = Pipeline()
    with raises(TypeError):
        p.add_stage(None)  # type: ignore[arg-type]
    with raises(TypeError):
        p.add_stage("not a stage")  # type: ignore[arg-type]


def test_add_stage_after_initialize_raises():
    """Stages cannot be added after initialize() (lifecycle accounting)."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    p.initialize()
    with raises(RuntimeError):
        p.add_stage(DummyStage("late"))


def test_remove_stage_by_instance_identity():
    """remove_stage removes by identity (is), returns True/False."""
    p = Pipeline()
    a, b = DummyStage("a"), DummyStage("b")
    p.add_stage(a)
    p.add_stage(b)
    assert p.remove_stage(a) is True
    assert list(p.stages) == [b]
    assert p.remove_stage(a) is False  # already removed


def test_remove_stage_by_name():
    """remove_stage removes the first stage with a matching name."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    p.add_stage(DummyStage("b"))
    assert p.remove_stage("a") is True
    assert [s.name for s in p.stages] == ["b"]
    assert p.remove_stage("nonexistent") is False


def test_remove_stage_after_initialize_raises():
    """remove_stage is rejected after initialize()."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    p.initialize()
    with raises(RuntimeError):
        p.remove_stage("a")


def test_pipeline_len_and_repr():
    """__len__ reports stage count; __repr__ reports state."""
    p = Pipeline()
    assert len(p) == 0
    p.add_stage(DummyStage("a"))
    assert len(p) == 1
    r = repr(p)
    assert "Pipeline(" in r
    assert "state=NEW" in r
    p.initialize()
    assert "state=INITIALIZED" in repr(p)
    p.shutdown()
    assert "state=SHUT_DOWN" in repr(p)


# ======================================================================
# 5. Pipeline lifecycle state machine
# ======================================================================

def test_run_before_initialize_raises():
    """run() on a NEW pipeline raises RuntimeError."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    with raises(RuntimeError, match="initialize"):
        p.run(PipelineContext.empty())


def test_run_after_shutdown_raises():
    """run() on a SHUT_DOWN pipeline raises RuntimeError."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    p.initialize()
    p.shutdown()
    with raises(RuntimeError, match="shutdown"):
        p.run(PipelineContext.empty())


def test_initialize_is_idempotent():
    """Calling initialize() twice is a no-op (counter does not double)."""
    p = Pipeline()
    s = DummyStage("a")
    p.add_stage(s)
    p.initialize()
    p.initialize()  # no-op
    assert s.initialize_count == 1
    assert p.initialized is True


def test_shutdown_is_idempotent():
    """Calling shutdown() twice is a no-op."""
    p = Pipeline()
    s = DummyStage("a")
    p.add_stage(s)
    p.initialize()
    p.shutdown()
    p.shutdown()  # no-op
    assert s.shutdown_count == 1
    assert p.shutdown_done is True


def test_initialize_after_shutdown_raises():
    """initialize() is rejected in the terminal SHUT_DOWN state."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    p.initialize()
    p.shutdown()
    with raises(RuntimeError, match="terminal"):
        p.initialize()


def test_shutdown_without_initialize_still_works():
    """shutdown() is safe even if initialize() was never called."""
    p = Pipeline()
    s = DummyStage("a")
    p.add_stage(s)
    p.shutdown()  # must not raise
    assert p.shutdown_done is True
    assert s.shutdown_count == 1  # stage.shutdown called anyway


# ======================================================================
# === REQUIRED AREA 1: STAGE ORDERING ==================================
# ======================================================================

def test_stage_order_preserved_in_run():
    """Stages execute process() in insertion order."""
    p = Pipeline()
    p.add_stage(DummyStage("first"))
    p.add_stage(DummyStage("second"))
    p.add_stage(DummyStage("third"))
    with p:
        ctx = p.run(PipelineContext.empty())
    assert ctx.metadata["order"] == ["first", "second", "third"]


def test_stage_order_uses_marker_not_name():
    """The order list records markers (which may differ from names)."""
    p = Pipeline()
    p.add_stage(DummyStage("s1", marker="M1"))
    p.add_stage(DummyStage("s2", marker="M2"))
    with p:
        ctx = p.run(PipelineContext.empty())
    assert ctx.metadata["order"] == ["M1", "M2"]


def test_shutdown_runs_in_reverse_order():
    """shutdown() drives stages in reverse insertion order."""
    order: list[str] = []

    class OrderTrackingStage(DummyStage):
        def shutdown(self) -> None:  # noqa: D401
            order.append(self.name)
            super().shutdown()

    p = Pipeline()
    p.add_stage(OrderTrackingStage("a"))
    p.add_stage(OrderTrackingStage("b"))
    p.add_stage(OrderTrackingStage("c"))
    p.initialize()
    p.shutdown()
    assert order == ["c", "b", "a"]


# ======================================================================
# === REQUIRED AREA 2: CONTEXT PASSING =================================
# ======================================================================

def test_context_is_same_instance_across_stages():
    """Every stage receives the exact same context object."""
    seen: list[int] = []

    class CaptureIdStage(PipelineStage):
        def __init__(self, name: str) -> None:
            super().__init__(name)
        def initialize(self) -> None:
            pass
        def process(self, context: PipelineContext) -> None:
            seen.append(id(context))
            context.metadata.setdefault("ids", []).append(id(context))
        def shutdown(self) -> None:
            pass
        def health_check(self) -> bool:
            return True

    p = Pipeline()
    p.add_stage(CaptureIdStage("a"))
    p.add_stage(CaptureIdStage("b"))
    p.add_stage(CaptureIdStage("c"))
    with p:
        ctx = p.run(PipelineContext.empty())
    # All three stages saw the same id.
    assert len(set(seen)) == 1
    # And the context's recorded ids are all identical.
    assert len(set(ctx.metadata["ids"])) == 1


def test_context_downstream_reads_upstream_writes():
    """A later stage observes mutations an earlier stage made to the context."""
    class ProducerStage(PipelineStage):
        def initialize(self) -> None:
            pass
        def process(self, context: PipelineContext) -> None:
            context.metadata["produced"] = "hello"
            context.detections.append("d1")  # type: ignore[arg-type]
        def shutdown(self) -> None:
            pass
        def health_check(self) -> bool:
            return True

    class ConsumerStage(PipelineStage):
        def __init__(self) -> None:
            super().__init__("consumer")
            self.observed: list[Any] = []
        def initialize(self) -> None:
            pass
        def process(self, context: PipelineContext) -> None:
            self.observed.append(context.metadata.get("produced"))
            self.observed.append(list(context.detections))
        def shutdown(self) -> None:
            pass
        def health_check(self) -> bool:
            return True

    consumer = ConsumerStage()
    p = Pipeline()
    p.add_stage(ProducerStage())
    p.add_stage(consumer)
    with p:
        p.run(PipelineContext.empty())
    assert consumer.observed == ["hello", ["d1"]]


def test_run_returns_the_same_context_instance():
    """run() returns the context it was given (mutated), not a copy."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    with p:
        ctx = PipelineContext.empty()
        result = p.run(ctx)
    assert result is ctx


# ======================================================================
# === REQUIRED AREA 3: EXCEPTION PROPAGATION ===========================
# ======================================================================

def test_stage_exception_propagates_out_of_run():
    """An exception raised by a stage's process() propagates out of run()."""
    p = Pipeline()
    boom = ValueError("stage exploded")
    p.add_stage(DummyStage("a", raise_on_process=boom))
    p.initialize()
    with raises(ValueError, match="stage exploded"):
        p.run(PipelineContext.empty())


def test_stage_exception_aborts_subsequent_stages():
    """When a stage raises, later stages' process() is not called."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    p.add_stage(DummyStage("b", raise_on_process=RuntimeError("boom")))
    later = DummyStage("c")
    p.add_stage(later)
    p.initialize()
    with raises(RuntimeError):
        p.run(PipelineContext.empty())
    # 'a' ran, 'b' was entered (count incremented) but raised before stamping,
    # 'c' was never reached.
    a, b, c = p.stages
    assert a.process_count == 1   # type: ignore[attr-defined]
    assert b.process_count == 1   # type: ignore[attr-defined]
    assert c.process_count == 0   # type: ignore[attr-defined]


def test_stage_error_is_a_stage_exception():
    """StageError raised by a stage propagates as-is (not swallowed)."""
    p = Pipeline()
    p.add_stage(DummyStage("a", raise_on_process=StageError("model load failed")))
    p.initialize()
    with raises(StageError, match="model load failed"):
        p.run(PipelineContext.empty())


def test_exception_does_not_prevent_context_manager_teardown():
    """A run() exception inside 'with' still triggers shutdown()."""
    p = Pipeline()
    p.add_stage(DummyStage("a"))
    p.add_stage(DummyStage("b", raise_on_process=RuntimeError("boom")))
    with raises(RuntimeError):
        with p:
            p.run(PipelineContext.empty())
    # Even though run() raised, shutdown ran on every stage.
    a, b = p.stages
    assert a.shutdown_count == 1   # type: ignore[attr-defined]
    assert b.shutdown_count == 1   # type: ignore[attr-defined]
    assert p.shutdown_done is True


def test_failed_stage_does_not_stamp_context_captured():
    """A stage configured to raise does not append its marker (raises first)."""
    p = Pipeline()
    p.add_stage(DummyStage("ok"))
    p.add_stage(DummyStage("boom", raise_on_process=RuntimeError("x")))
    ctx = PipelineContext.empty()
    p.initialize()
    try:
        p.run(ctx)
    except RuntimeError:
        pass
    assert ctx.metadata.get("order") == ["ok"]  # 'boom' never stamped


# ======================================================================
# === REQUIRED AREA 4: STAGE LIFECYCLE =================================
# ======================================================================

def test_lifecycle_calls_each_method_exactly_once():
    """A clean run calls initialize/process/shutdown exactly once per stage."""
    p = Pipeline()
    a = DummyStage("a")
    b = DummyStage("b")
    p.add_stage(a)
    p.add_stage(b)
    with p:
        p.run(PipelineContext.empty())
    for s in (a, b):
        assert s.initialize_count == 1
        assert s.process_count == 1
        assert s.shutdown_count == 1


def test_lifecycle_initialize_before_process_before_shutdown():
    """Lifecycle order is initialize -> process -> shutdown."""
    trace: list[str] = []

    class TracingStage(PipelineStage):
        def __init__(self, name: str) -> None:
            super().__init__(name)
        def initialize(self) -> None:
            trace.append(f"{self.name}:init")
        def process(self, context: PipelineContext) -> None:
            trace.append(f"{self.name}:process")
        def shutdown(self) -> None:
            trace.append(f"{self.name}:shutdown")
        def health_check(self) -> bool:
            return True

    p = Pipeline()
    p.add_stage(TracingStage("a"))
    p.add_stage(TracingStage("b"))
    with p:
        p.run(PipelineContext.empty())
    assert trace == [
        "a:init", "b:init",          # initialize in order
        "a:process", "b:process",    # process in order
        "b:shutdown", "a:shutdown",  # shutdown in reverse
    ]


def test_lifecycle_multiple_runs_reuse_initialization():
    """One initialize() supports multiple run() calls; shutdown once at end."""
    p = Pipeline()
    s = DummyStage("a")
    p.add_stage(s)
    p.initialize()
    p.run(PipelineContext.empty())
    p.run(PipelineContext.empty())
    p.run(PipelineContext.empty())
    p.shutdown()
    assert s.initialize_count == 1
    assert s.process_count == 3
    assert s.shutdown_count == 1


def test_context_manager_is_single_use_due_to_terminal_shutdown():
    """Shutdown is terminal: re-entering 'with' after exit raises.

    The pipeline's lifecycle state machine treats ``shutdown()`` as a
    terminal transition (see ``test_initialize_after_shutdown_raises``).
    Because ``__enter__`` calls ``initialize()``, and ``initialize()``
    rejects a shut-down pipeline, the context manager is effectively
    single-use. Re-processing another context requires a fresh Pipeline
    instance -- this keeps resource lifecycles simple (no half-re-acquired
    state) and matches the "close a file" mental model documented in
    ``pipeline.py``.

    The supported multi-run pattern is a *single* ``with`` block wrapping
    multiple ``run()`` calls (see ``test_lifecycle_multiple_runs_reuse_initialization``).
    """
    p = Pipeline()
    s = DummyStage("a")
    p.add_stage(s)
    with p:
        p.run(PipelineContext.empty())
    assert p.shutdown_done is True
    assert s.shutdown_count == 1
    # Re-entering must raise -- shutdown is terminal.
    with raises(RuntimeError, match="terminal"):
        with p:
            p.run(PipelineContext.empty())


# ======================================================================
# 6. Pipeline health_check aggregation
# ======================================================================

def test_health_check_empty_pipeline_is_vacuously_healthy():
    """An empty pipeline is vacuously healthy (no stages to fail)."""
    p = Pipeline()
    assert p.health_check() is True


def test_health_check_all_healthy_returns_true():
    """All-healthy stages -> pipeline healthy."""
    p = Pipeline()
    a, b = DummyStage("a"), DummyStage("b")
    p.add_stage(a)
    p.add_stage(b)
    a.initialize()
    b.initialize()
    # NB: health_check does not require pipeline.initialize(); it just
    # probes each stage. This is intentional -- monitoring should work
    # even on a partially-initialised pipeline.
    assert p.health_check() is True


def test_health_check_one_unhealthy_returns_false():
    """One unhealthy stage -> pipeline unhealthy (short-circuit)."""
    p = Pipeline()
    a, b = DummyStage("a"), DummyStage("b")
    p.add_stage(a)
    p.add_stage(b)
    a.initialize()
    # b never initialised -> unhealthy
    assert p.health_check() is False


def test_health_check_survives_raising_stage():
    """A stage whose health_check() raises is treated as unhealthy."""
    class BadHealth(PipelineStage):
        def initialize(self) -> None:
            pass
        def process(self, context: PipelineContext) -> None:
            pass
        def shutdown(self) -> None:
            pass
        def health_check(self) -> bool:
            raise RuntimeError("probe broken")

    p = Pipeline()
    p.add_stage(BadHealth("bad"))
    assert p.health_check() is False  # must not raise


# ======================================================================
# 7. Empty pipeline edge cases
# ======================================================================

def test_empty_pipeline_run_is_noop():
    """Running an initialised empty pipeline returns the context unchanged."""
    p = Pipeline()
    p.initialize()
    ctx = PipelineContext.empty(timestamp=9.0)
    result = p.run(ctx)
    assert result is ctx
    assert result.timestamp == 9.0
    assert result.metadata == {}  # no stages touched it


def test_empty_pipeline_initialize_is_noop_but_marks_initialised():
    """Initialising an empty pipeline marks it initialised (no stages)."""
    p = Pipeline()
    p.initialize()
    assert p.initialized is True


# ======================================================================
# 8. End-to-end smoke test
# ======================================================================

def test_end_to_end_three_stage_pipeline():
    """A three-stage pipeline stamps order, returns context, tears down."""
    p = Pipeline()
    p.add_stage(DummyStage("capture"))
    p.add_stage(DummyStage("detect"))
    p.add_stage(DummyStage("track"))

    ctx = PipelineContext.empty(timestamp=42.0)
    with p:
        result = p.run(ctx)

    assert result is ctx
    assert result.metadata["order"] == ["capture", "detect", "track"]
    assert all(s.shutdown_count == 1 for s in p.stages)
    assert p.shutdown_done is True
    assert p.health_check() is False  # all stages unhealthy after shutdown


# ======================================================================
# Runner -- collects every test_* function and runs them.
# ======================================================================

def _collect_tests() -> list[tuple[str, Any]]:
    """Return [(name, func)] for every test_ function defined in this module."""
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} pipeline foundation tests...\n")
    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 -- runner catches all
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
    print("All pipeline foundation tests passed.")
