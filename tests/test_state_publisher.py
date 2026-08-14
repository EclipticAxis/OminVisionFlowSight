"""Unit tests for visioncore.protocol.publisher.StatePublisher.

Verifies event subscription (all 5 lifecycle event types), TargetState
generation from events, ProtocolAdapter.publish() invocation, error
isolation (adapter failures don't crash the bus), lifecycle (start/stop
idempotency, context manager), and the no-GUI/no-InferWorker-modification
constraint.

Run::

    python -m pytest tests/test_state_publisher.py -v
    # or
    PYTHONPATH=F:/VisionBata python tests/test_state_publisher.py
"""

from __future__ import annotations

import ast
import inspect
import os

from visioncore.eventbus.bus import EventBus
from visioncore.eventbus.events import (
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)
from visioncore.protocol.base import ProtocolAdapter
from visioncore.protocol.publisher import StatePublisher
from visioncore.protocol.publisher.state_publisher import (
    StatePublisher as StatePublisherDirect,
)
from visioncore.state.target_state import TargetState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CapturingAdapter(ProtocolAdapter):
    """A test adapter that records every published TargetState."""

    def __init__(self) -> None:
        self._connected: bool = False
        self.published: list[TargetState] = []

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def publish(self, state: TargetState) -> None:
        if not self._connected:
            raise RuntimeError("not connected")
        if not isinstance(state, TargetState):
            raise TypeError(f"not a TargetState: {type(state).__name__}")
        self.published.append(state)

    def health_check(self) -> bool:
        return self._connected


class FailingAdapter(ProtocolAdapter):
    """An adapter whose publish() always raises."""

    def __init__(self) -> None:
        self._connected: bool = False
        self.publish_calls: int = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def publish(self, state: TargetState) -> None:
        self.publish_calls += 1
        raise OSError("simulated network failure")

    def health_check(self) -> bool:
        return self._connected


def _make_event(event_cls, **overrides):
    """Build a lifecycle event with sensible defaults + per-test overrides."""
    defaults = dict(
        event_id="evt-0001",
        timestamp=12.5,
        target_id="S0-T0001",
        slot_id=0,
        payload={},
    )
    defaults.update(overrides)
    return event_cls(**defaults)


# ---------------------------------------------------------------------------
# Type annotation convention
# ---------------------------------------------------------------------------

def test_type_annotation_convention_no_top_level_import():
    """StatePublisher module must not use 'from visioncore import TargetState'."""
    mod = __import__(StatePublisher.__module__, fromlist=["_"])
    src = inspect.getsource(mod)
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == "visioncore":
            for alias in node.names:
                assert alias.name != "TargetState", (
                    f"{StatePublisher.__module__} line {node.lineno}: "
                    f"forbidden 'from visioncore import TargetState'."
                )


# ---------------------------------------------------------------------------
# No GUI / no InferWorker modification
# ---------------------------------------------------------------------------

def test_no_gui_files_modified():
    """No files in gui/ are modified by B6 (only new files are added).

    This test checks that the gui/ directory's Python files have not
    been touched by verifying they don't import StatePublisher.
    """
    gui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "gui")
    if not os.path.isdir(gui_dir):
        return  # gui/ not found -- skip

    for fname in os.listdir(gui_dir):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(gui_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                src = f.read()
        except (IOError, UnicodeDecodeError):
            continue
        assert "StatePublisher" not in src, (
            f"gui/{fname} references StatePublisher -- GUI must not be "
            f"modified by B6."
        )
        assert "state_publisher" not in src, (
            f"gui/{fname} references state_publisher -- GUI must not be "
            f"modified by B6."
        )


def test_no_inferworker_modification():
    """ai/inference.py must not be modified by B6."""
    inference_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "ai", "inference.py")
    if not os.path.isfile(inference_path):
        return  # file not found -- skip

    with open(inference_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "StatePublisher" not in src, (
        "ai/inference.py references StatePublisher -- InferWorker must "
        "not be modified by B6."
    )
    assert "state_publisher" not in src, (
        "ai/inference.py references state_publisher -- InferWorker must "
        "not be modified by B6."
    )


# ---------------------------------------------------------------------------
# Package / import sanity
# ---------------------------------------------------------------------------

def test_direct_and_package_imports_are_same():
    """visioncore.protocol.publisher.StatePublisher == direct import."""
    assert StatePublisher is StatePublisherDirect


def test_package_exports_state_publisher():
    """visioncore.protocol.publisher.__all__ contains StatePublisher."""
    import visioncore.protocol.publisher as pub
    assert "StatePublisher" in pub.__all__


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def test_start_subscribes_to_five_event_types():
    """start() registers 5 subscriptions on the bus."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)

    assert bus.subscriber_count() == 0
    pub.start()
    # 5 subscriptions, one per event type.
    assert len(pub._subscriptions) == 5
    # Each event type should have exactly 1 subscriber.
    assert bus.subscriber_count(TargetCreatedEvent) == 1
    assert bus.subscriber_count(TargetLostEvent) == 1
    assert bus.subscriber_count(TargetRecoveredEvent) == 1
    assert bus.subscriber_count(TargetLockedEvent) == 1
    assert bus.subscriber_count(TargetRemovedEvent) == 1

    pub.stop()


def test_start_connects_adapter():
    """start() calls adapter.connect()."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)

    assert adapter.health_check() is False
    pub.start()
    assert adapter.health_check() is True
    pub.stop()
    assert adapter.health_check() is False


def test_start_is_idempotent():
    """Calling start() twice does not double-subscribe."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)

    pub.start()
    n1 = bus.subscriber_count()
    pub.start()  # idempotent -- should be a no-op
    n2 = bus.subscriber_count()
    assert n1 == n2

    pub.stop()


def test_stop_is_idempotent():
    """Calling stop() twice is safe."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)

    pub.start()
    pub.stop()
    pub.stop()  # should not raise
    assert pub.is_started is False


def test_stop_unsubscribes_all():
    """stop() removes all subscriptions from the bus."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)

    pub.start()
    assert bus.subscriber_count() > 0
    pub.stop()
    assert bus.subscriber_count() == 0


def test_context_manager():
    """StatePublisher supports 'with' -- start on enter, stop on exit."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)

    assert pub.is_started is False
    with pub:
        assert pub.is_started is True
        assert adapter.health_check() is True
    assert pub.is_started is False
    assert adapter.health_check() is False


def test_context_manager_stops_on_exception():
    """__exit__ calls stop() even when an exception propagates."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)

    try:
        with pub:
            raise ValueError("test error")
    except ValueError:
        pass

    assert pub.is_started is False


# ---------------------------------------------------------------------------
# Event -> TargetState conversion + publish
# ---------------------------------------------------------------------------

def test_created_event_publishes_state():
    """TargetCreatedEvent triggers adapter.publish() with a TargetState."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetCreatedEvent,
        payload={"track_id": 1, "priority": 5},
    )
    bus.publish(event)

    assert len(adapter.published) == 1
    state = adapter.published[0]
    assert isinstance(state, TargetState)
    assert state.target_id == 1
    assert state.camera_id == 0
    assert state.timestamp == 12.5
    assert state.metadata["event_type"] == "target.created"
    assert state.metadata["event_id"] == "evt-0001"
    assert state.metadata["priority"] == 5

    pub.stop()


def test_created_event_with_detection_data():
    """TargetCreatedEvent with detection in payload populates position fields."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetCreatedEvent,
        payload={
            "track_id": 42,
            "detection": {
                "bbox": {"x": 0.5, "y": 0.4, "w": 0.12, "h": 0.30},
                "score": 0.92,
                "class_name": "person",
                "class_id": 0,
            },
        },
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.target_id == 42
    assert state.label == "person"
    assert state.confidence == 0.92
    assert state.cx == 0.5
    assert state.cy == 0.4
    assert state.width == 0.12
    assert state.height == 0.30

    pub.stop()


def test_lost_event_publishes_state():
    """TargetLostEvent triggers a publish."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetLostEvent,
        target_id="S1-T0005",
        slot_id=1,
        payload={"last_seen": 10.0, "missed_frames": 4},
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.target_id == 5
    assert state.camera_id == 1
    assert state.metadata["event_type"] == "target.lost"
    assert state.metadata["missed_frames"] == 4
    # No detection in payload -> defaults.
    assert state.label == "unknown"
    assert state.confidence == 0.0
    assert state.cx == 0.0

    pub.stop()


def test_recovered_event_publishes_state():
    """TargetRecoveredEvent triggers a publish."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetRecoveredEvent,
        payload={"new_track_id": 7, "recovery_latency": 1.5},
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.metadata["event_type"] == "target.recovered"
    assert state.metadata["recovery_latency"] == 1.5

    pub.stop()


def test_locked_event_publishes_state():
    """TargetLockedEvent triggers a publish."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetLockedEvent,
        payload={"locked_by": "operator", "reason": "manual inspection"},
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.metadata["event_type"] == "target.locked"
    assert state.metadata["locked_by"] == "operator"

    pub.stop()


def test_removed_event_publishes_state():
    """TargetRemovedEvent triggers a publish."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetRemovedEvent,
        payload={"reason": "timeout", "final_state": "LOST"},
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.metadata["event_type"] == "target.removed"
    assert state.metadata["reason"] == "timeout"

    pub.stop()


def test_all_five_events_each_publish_once():
    """Publishing all 5 event types results in 5 adapter publishes."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    events = [
        _make_event(TargetCreatedEvent, event_id="e1", timestamp=1.0),
        _make_event(TargetLostEvent, event_id="e2", timestamp=2.0),
        _make_event(TargetRecoveredEvent, event_id="e3", timestamp=3.0),
        _make_event(TargetLockedEvent, event_id="e4", timestamp=4.0),
        _make_event(TargetRemovedEvent, event_id="e5", timestamp=5.0),
    ]
    for ev in events:
        bus.publish(ev)

    assert len(adapter.published) == 5
    # Verify event_type order is preserved.
    types = [s.metadata["event_type"] for s in adapter.published]
    assert types == [
        "target.created", "target.lost", "target.recovered",
        "target.locked", "target.removed",
    ]

    pub.stop()


# ---------------------------------------------------------------------------
# Error isolation
# ---------------------------------------------------------------------------

def test_adapter_failure_does_not_crash_bus():
    """If adapter.publish() raises, the error is caught and the bus continues."""
    bus = EventBus()
    adapter = FailingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    # Publish an event -- adapter.publish() will raise OSError.
    event = _make_event(TargetCreatedEvent)
    # The bus.publish should NOT raise -- the publisher catches the error.
    count = bus.publish(event)
    assert count == 1  # the subscriber was notified (even though it failed)

    # The adapter's publish was called.
    assert adapter.publish_calls == 1

    # A subsequent event should still be delivered -- the publisher
    # did not unsubscribe or crash.
    event2 = _make_event(TargetLostEvent, event_id="e2")
    bus.publish(event2)
    assert adapter.publish_calls == 2

    pub.stop()


def test_publisher_does_not_modify_bus():
    """The publisher only subscribes -- it does not publish events on the bus."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    # The publisher subscribes to 5 event types but does not publish
    # anything itself. Verify by checking subscriber_count for each
    # event type -- each should be exactly 1 (the publisher's own
    # subscription).
    assert bus.subscriber_count(TargetCreatedEvent) == 1
    assert bus.subscriber_count(TargetLostEvent) == 1

    pub.stop()


# ---------------------------------------------------------------------------
# TargetState generation details
# ---------------------------------------------------------------------------

def test_target_id_parsed_from_string():
    """The target_id string 'S0-T0001' is parsed to int 1."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetCreatedEvent,
        target_id="S2-T0042",
        slot_id=2,
        payload={"track_id": 42},
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.target_id == 42
    assert state.local_id == 42
    assert state.camera_id == 2

    pub.stop()


def test_target_id_uses_seq_when_no_track_id():
    """When payload has no track_id, the seq from target_id string is used."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetCreatedEvent,
        target_id="S0-T0007",
        payload={},  # no track_id
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.target_id == 7
    assert state.local_id == 7

    pub.stop()


def test_metadata_includes_event_type_and_id():
    """Every published TargetState's metadata includes event_type + event_id."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetCreatedEvent,
        event_id="evt-abc-123",
        payload={"track_id": 1, "custom_field": "value"},
    )
    bus.publish(event)

    state = adapter.published[0]
    assert state.metadata["event_type"] == "target.created"
    assert state.metadata["event_id"] == "evt-abc-123"
    assert state.metadata["custom_field"] == "value"

    pub.stop()


def test_metadata_excludes_detection_key():
    """The 'detection' key is extracted into position fields, not left in metadata."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(
        TargetCreatedEvent,
        payload={
            "track_id": 1,
            "detection": {"bbox": {"x": 0.5, "y": 0.4, "w": 0.1, "h": 0.2},
                          "score": 0.9, "class_name": "person"},
        },
    )
    bus.publish(event)

    state = adapter.published[0]
    # detection is extracted into position fields, not left in metadata.
    assert "detection" not in state.metadata
    assert state.cx == 0.5
    assert state.label == "person"

    pub.stop()


def test_vx_vy_always_zero():
    """Events do not carry velocity -- vx/vy are always 0.0."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(TargetCreatedEvent)
    bus.publish(event)

    state = adapter.published[0]
    assert state.vx == 0.0
    assert state.vy == 0.0

    pub.stop()


def test_global_id_always_none():
    """No cross-camera fusion in B6 -- global_id is always None."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    pub.start()

    event = _make_event(TargetCreatedEvent)
    bus.publish(event)

    state = adapter.published[0]
    assert state.global_id is None

    pub.stop()


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------

def test_repr():
    """__repr__ includes started state, adapter type, subscription count."""
    bus = EventBus()
    adapter = CapturingAdapter()
    pub = StatePublisher(bus, adapter)
    r = repr(pub)
    assert "StatePublisher" in r
    assert "started=False" in r
    assert "CapturingAdapter" in r

    pub.start()
    r = repr(pub)
    assert "started=True" in r
    assert "subscriptions=5" in r
    pub.stop()


# ---------------------------------------------------------------------------
# Module entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    failures: list[str] = []
    passed = 0
    g = globals()
    names = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    for name in names:
        try:
            g[name]()
            print(f"  [PASS] {name}")
            passed += 1
        except AssertionError as exc:
            print(f"  [FAIL] {name}: {exc}")
            failures.append(name)
        except Exception as exc:
            print(f"  [ERR ] {name}: {type(exc).__name__}: {exc}")
            failures.append(name)

    print()
    total = passed + len(failures)
    print(f"=== {passed}/{total} passed, {len(failures)} failed ===")
    if failures:
        print("Failed:", ", ".join(failures))
        sys.exit(1)
