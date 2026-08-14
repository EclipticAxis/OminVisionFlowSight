"""EventStage unit tests (Milestone C6).

Functional test suite: no pytest, no test classes, ``__main__`` runner.

Covers:
    - EventBus ABC contract (abstract, subclass must implement publish)
    - DummyEventBus (records events, raise_on_publish, reset, convenience props)
    - EventStage construction (rejects None/non-EventBus, name, event_type)
    - EventStage lifecycle (initialize/shutdown no-ops, health_check delegates)
    - EventStage.process: reads target_states, one Event per snapshot
    - Sink semantics (does not modify context)
    - Empty target_states -> zero events (not an error)
    - Event payload carries the snapshot; timestamp from snapshot
    - event_type configurable
    - Exception propagation (bus.publish raises -> process raises)
    - Pipeline integration + end-to-end Detect->Track->Target->Event chain
    - No network / UDP / Protocol dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import traceback
from typing import Any

from visioncore.core.event import Event
from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages import (
    DetectorStage,
    DummyDetector,
    DummyEventBus,
    DummyTargetManager,
    DummyTracker,
    EventBus,
    EventStage,
    TargetStage,
    TrackerStage,
)
from visioncore.state.target_state import TargetState as TargetSnapshot


# ======================================================================
# Test helpers
# ======================================================================

class _Raises:
    __slots__ = ("expected", "match", "caught")

    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.caught = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"expected {self.expected.__name__}, but no exception was raised")
        if not isinstance(exc_val, self.expected):
            raise AssertionError(
                f"expected {self.expected.__name__}, got "
                f"{type(exc_val).__name__}: {exc_val}")
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(f"expected {self.match!r}, got {str(exc_val)!r}")
        self.caught = exc_val
        return True


def raises(expected, match=None):
    return _Raises(expected, match)


def _snapshot(target_id: int = 1, label: str = "person",
              timestamp: float = 1.0) -> TargetSnapshot:
    return TargetSnapshot(
        target_id=target_id, local_id=1, global_id=None, label=label,
        confidence=0.9, cx=0.5, cy=0.5, vx=0.0, vy=0.0,
        width=0.2, height=0.4, timestamp=timestamp, camera_id=0, metadata={},
    )


# ======================================================================
# 0. Package surface / ABC contract
# ======================================================================

def test_package_exports_event_names():
    import visioncore.pipeline.stages as pkg
    for name in ("EventBus", "EventStage", "DummyEventBus"):
        assert name in pkg.__all__, f"{name} missing from stages __all__"


def test_event_bus_is_abstract():
    with raises(TypeError):
        EventBus()  # type: ignore[abstract]


def test_event_bus_subclass_missing_publish_fails():
    class Incomplete(EventBus):
        pass  # publish not implemented
    with raises(TypeError):
        Incomplete()  # type: ignore[abstract]


def test_event_bus_health_check_default_is_true():
    """health_check has a concrete default returning True."""
    bus = DummyEventBus()
    assert bus.health_check() is True


# ======================================================================
# 1. DummyEventBus
# ======================================================================

def test_dummy_bus_records_published_events():
    bus = DummyEventBus()
    e1 = Event("a", 0.0)
    e2 = Event("b", 1.0)
    bus.publish(e1)
    bus.publish(e2)
    assert len(bus.published) == 2
    assert bus.published[0] is e1
    assert bus.published[1] is e2


def test_dummy_bus_publish_returns_one():
    bus = DummyEventBus()
    assert bus.publish(Event("x", 0.0)) == 1


def test_dummy_bus_publish_count_includes_raised():
    bus = DummyEventBus(raise_on_publish=ValueError("bad"))
    with raises(ValueError):
        bus.publish(Event("x", 0.0))
    assert bus.publish_count == 1  # incremented before raising
    assert len(bus.published) == 0  # not recorded (raised)


def test_dummy_bus_raise_persists_until_cleared():
    bus = DummyEventBus(raise_on_publish=RuntimeError("x"))
    with raises(RuntimeError):
        bus.publish(Event("a", 0.0))
    with raises(RuntimeError):
        bus.publish(Event("b", 0.0))
    bus.raise_on_publish = None
    bus.publish(Event("c", 0.0))
    assert len(bus.published) == 1


def test_dummy_bus_event_types_property():
    bus = DummyEventBus()
    bus.publish(Event("alpha", 0.0))
    bus.publish(Event("beta", 1.0))
    bus.publish(Event("alpha", 2.0))
    assert bus.event_types == ["alpha", "beta", "alpha"]


def test_dummy_bus_reset():
    bus = DummyEventBus()
    bus.publish(Event("a", 0.0))
    bus.publish(Event("b", 1.0))
    bus.reset()
    assert len(bus.published) == 0
    assert bus.publish_count == 0


# ======================================================================
# 2. EventStage construction
# ======================================================================

def test_stage_construction_wraps_bus():
    bus = DummyEventBus()
    stage = EventStage(bus)
    assert stage.event_bus is bus
    assert stage.health_check() is True  # bus health_check defaults True


def test_stage_default_name():
    assert EventStage(DummyEventBus()).name == "EventStage"


def test_stage_custom_name():
    stage = EventStage(DummyEventBus(), name="emitter")
    assert stage.name == "emitter"


def test_stage_default_event_type():
    stage = EventStage(DummyEventBus())
    assert stage.event_type == "target.snapshot"


def test_stage_custom_event_type():
    stage = EventStage(DummyEventBus(), event_type="target.active")
    assert stage.event_type == "target.active"


def test_stage_rejects_none_bus():
    with raises(TypeError, match="EventBus"):
        EventStage(None)  # type: ignore[arg-type]


def test_stage_rejects_non_bus():
    with raises(TypeError, match="EventBus"):
        EventStage("not a bus")  # type: ignore[arg-type]


def test_stage_rejects_empty_event_type():
    with raises(ValueError, match="event_type"):
        EventStage(DummyEventBus(), event_type="")


def test_stage_is_pipeline_stage():
    assert issubclass(EventStage, PipelineStage)


# ======================================================================
# 3. EventStage lifecycle
# ======================================================================

def test_stage_initialize_is_noop():
    """initialize() is a no-op (bus is ready on construction)."""
    stage = EventStage(DummyEventBus())
    stage.initialize()  # must not raise
    assert stage.health_check() is True


def test_stage_shutdown_is_noop():
    """shutdown() is a no-op (stage does not own the bus)."""
    bus = DummyEventBus()
    stage = EventStage(bus)
    stage.initialize()
    stage.shutdown()  # must not raise
    # Bus is still usable after stage shutdown (stage doesn't own it).
    assert bus.health_check() is True


def test_stage_health_check_delegates_to_bus():
    class SickBus(EventBus):
        def publish(self, event):
            return 0
        def health_check(self):
            return False
    stage = EventStage(SickBus())
    assert stage.health_check() is False


def test_stage_repr_reports_health():
    stage = EventStage(DummyEventBus(), name="events")
    stage.initialize()
    r = repr(stage)
    assert "EventStage" in r
    assert "name='events'" in r


# ======================================================================
# 4. EventStage.process data flow
# ======================================================================

def test_process_publishes_one_event_per_snapshot():
    bus = DummyEventBus()
    stage = EventStage(bus)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(1, "person"))
    ctx.target_states.append(_snapshot(2, "car"))
    stage.process(ctx)
    assert len(bus.published) == 2


def test_process_with_empty_target_states_publishes_zero():
    """Empty target_states -> zero events (not an error, not a skip)."""
    bus = DummyEventBus()
    stage = EventStage(bus)
    stage.initialize()
    ctx = PipelineContext.empty()  # target_states == []
    stage.process(ctx)
    assert len(bus.published) == 0
    assert stage.published_count == 0


def test_process_event_type_matches_stage_config():
    bus = DummyEventBus()
    stage = EventStage(bus, event_type="target.active")
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot())
    stage.process(ctx)
    assert bus.published[0].event_type == "target.active"


def test_process_event_timestamp_from_snapshot():
    """Event timestamp is the snapshot's timestamp (not wall-clock)."""
    bus = DummyEventBus()
    stage = EventStage(bus)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(timestamp=42.5))
    stage.process(ctx)
    assert bus.published[0].timestamp == 42.5


def test_process_event_payload_carries_snapshot():
    """The event payload carries the snapshot object directly."""
    bus = DummyEventBus()
    stage = EventStage(bus)
    stage.initialize()
    ctx = PipelineContext.empty()
    snap = _snapshot(target_id=7, label="dog")
    ctx.target_states.append(snap)
    stage.process(ctx)
    event = bus.published[0]
    assert event.payload["snapshot"] is snap  # exact object, no copy
    assert event.payload["snapshot"].label == "dog"


def test_process_increments_published_count():
    stage = EventStage(DummyEventBus())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot())
    stage.process(ctx)
    stage.process(ctx)
    stage.process(ctx)
    assert stage.published_count == 3


def test_process_does_not_modify_context():
    """EventStage is a sink: process reads but does not write context."""
    bus = DummyEventBus()
    stage = EventStage(bus)
    stage.initialize()
    ctx = PipelineContext.empty(timestamp=9.0)
    ctx.target_states.append(_snapshot())
    original_ts_count = len(ctx.target_states)
    stage.process(ctx)
    # Context unchanged: same frame, same detections, same tracks,
    # same targets, same target_states, same timestamp.
    assert ctx.frame is None
    assert ctx.detections == []
    assert ctx.tracks == []
    assert ctx.targets == []
    assert len(ctx.target_states) == original_ts_count
    assert ctx.timestamp == 9.0


# ======================================================================
# 5. Exception propagation
# ======================================================================

def test_process_propagates_bus_exception():
    """A bus.publish() exception propagates out of process()."""
    bus = DummyEventBus(raise_on_publish=RuntimeError("bus full"))
    stage = EventStage(bus)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot())
    with raises(RuntimeError, match="bus full"):
        stage.process(ctx)


def test_process_exception_stops_at_failing_snapshot():
    """If publish raises on snapshot N, snapshots N+1.. are not published."""
    bus = DummyEventBus(raise_on_publish=RuntimeError("boom"))
    stage = EventStage(bus)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.extend([_snapshot(1), _snapshot(2), _snapshot(3)])
    with raises(RuntimeError):
        stage.process(ctx)
    # Only the first publish was attempted (it raised); 2 and 3 not reached.
    assert len(bus.published) == 0
    assert bus.publish_count == 1


# ======================================================================
# 6. Pipeline integration
# ======================================================================

def test_pipeline_with_event_stage_end_to_end():
    bus = DummyEventBus()
    stage = EventStage(bus, name="events")
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.target_states.append(_snapshot(1, "person"))
    ctx.target_states.append(_snapshot(2, "car"))
    with p:
        p.run(ctx)
    assert len(bus.published) == 2
    assert bus.event_types == ["target.snapshot", "target.snapshot"]
    assert stage.published_count == 2


def test_pipeline_event_stage_lifecycle():
    """initialize/shutdown called; process emits events."""
    bus = DummyEventBus()
    stage = EventStage(bus)
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot())
    with p:
        p.run(ctx)
    assert len(bus.published) == 1


# ======================================================================
# 7. End-to-end: Detect -> Track -> Target -> Event (C3->C4->C5->C6)
# ======================================================================

def test_end_to_end_full_chain():
    """Full chain: Detector -> Tracker -> Target -> Event.

    C3 produces detections, C4 produces tracks, C5 produces targets +
    target_states, C6 emits one event per target_state. All four stages
    in one pipeline.
    """
    import numpy as np
    from visioncore.core.frame import Frame

    bus = DummyEventBus()
    det = DetectorStage(DummyDetector(), name="detect")
    trk = TrackerStage(DummyTracker(), name="track")
    tgt = TargetStage(DummyTargetManager(), name="targets")
    evt = EventStage(bus, name="events")

    p = Pipeline()
    p.add_stage(det)
    p.add_stage(trk)
    p.add_stage(tgt)
    p.add_stage(evt)

    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.frame = Frame(0, 0.0, "cam0", np.zeros((8, 8, 3), dtype=np.uint8))

    with p:
        p.run(ctx)

    # Full chain produced data at every stage.
    assert len(ctx.detections) == 1
    assert len(ctx.tracks) == 1
    assert len(ctx.targets) == 1
    assert len(ctx.target_states) == 1
    # Event stage emitted one event per snapshot.
    assert len(bus.published) == 1
    event = bus.published[0]
    assert event.event_type == "target.snapshot"
    assert event.payload["snapshot"].label == "person"
    # Event timestamp from the snapshot.
    assert event.timestamp == ctx.target_states[0].timestamp


# ======================================================================
# 8. No network / UDP / Protocol dependency
# ======================================================================

def test_no_network_udp_protocol_in_source():
    """The event_stage module source contains no banned transport refs.

    AST-parse and assert no import of socket, udp, protocol, http, or
    visioncore.protocol. The stage publishes to an in-process bus only.
    """
    import visioncore.pipeline.stages.event_stage as mod
    src = open(mod.__file__).read()
    tree = ast.parse(src)
    banned = ("socket", "udp", "protocol", "http", "requests", "urllib",
              "aiohttp", "zmq")
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
    assert not violations, f"event_stage.py imports banned: {violations}"


def test_no_protocol_loaded_at_runtime():
    """Importing the stages package does not load visioncore.protocol."""
    import sys
    import visioncore.pipeline.stages  # noqa: F401
    assert "visioncore.protocol" not in sys.modules, (
        "visioncore.protocol loaded as side effect of stages import"
    )
    assert "socket" not in sys.modules, (
        "socket loaded as side effect of stages import"
    )


def test_event_bus_interface_is_minimal():
    """The EventBus ABC has publish (abstract) + health_check (concrete)."""
    import inspect
    # publish is abstract
    abstract_methods = EventBus.__abstractmethods__
    assert abstract_methods == frozenset({"publish"})
    # health_check is concrete (not abstract)
    assert "health_check" in EventBus.__dict__


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
    print(f"Running {len(tests)} event stage tests...\n")
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
    print("All event stage tests passed.")
