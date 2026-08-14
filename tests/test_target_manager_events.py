"""Unit tests for TargetManager lifecycle event publication.

Verifies that TargetManager automatically publishes the correct typed
event on every successful lifecycle transition when an EventBus is
attached, that invalid / duplicate transitions emit nothing, that event
ordering is preserved, and that backward compatibility holds when no bus
is attached.

Run::

    python -m pytest tests/test_target_manager_events.py -v
    # or
    python tests/test_target_manager_events.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when the file is run directly.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from visioncore.core.detection import BBox, Detection  # noqa: E402
from visioncore.core.target import TargetState  # noqa: E402
from visioncore.core.track import Track  # noqa: E402
from visioncore.eventbus import EventBus  # noqa: E402
from visioncore.eventbus.events import (  # noqa: E402
    BaseEvent,
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)
from visioncore.target_manager import TargetManager  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(track_id: int = 1, score: float = 0.9) -> Track:
    """Create a minimal Track for testing."""
    return Track(
        track_id=track_id,
        detection=Detection(BBox(0.5, 0.5, 0.2, 0.4), score, 0, "person"),
    )


def _make_manager_with_bus() -> tuple[TargetManager, EventBus, list[BaseEvent]]:
    """Create a TargetManager with an attached EventBus capturing all events."""
    bus = EventBus()
    mgr = TargetManager(event_bus=bus)
    received: list[BaseEvent] = []
    bus.subscribe("*", received.append)
    return mgr, bus, received


def _event_types(events: list[BaseEvent]) -> list[str]:
    """Return the event_type strings of a captured event list."""
    return [e.event_type for e in events]


# ---------------------------------------------------------------------------
# 1. 状态切换 (each transition emits the correct event)
# ---------------------------------------------------------------------------

def test_create_target_emits_created_event():
    """create_target publishes a TargetCreatedEvent."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), slot_id=0, last_seen=1.0)
    assert len(received) == 1
    ev = received[0]
    assert isinstance(ev, TargetCreatedEvent)
    assert ev.target_id == t.target_id
    assert ev.slot_id == 0
    assert ev.payload["initial_state"] == "ACTIVE"
    assert ev.payload["track_id"] == 1


def test_create_target_no_track_emits_created_with_lost_state():
    """create_target(track=None) still emits Created (initial_state=LOST)."""
    mgr, _bus, received = _make_manager_with_bus()
    mgr.create_target(track=None, slot_id=2)
    assert len(received) == 1
    ev = received[0]
    assert isinstance(ev, TargetCreatedEvent)
    assert ev.payload["initial_state"] == "LOST"
    assert ev.payload["track_id"] is None
    assert ev.slot_id == 2


def test_mark_lost_emits_lost_event():
    """mark_lost publishes a TargetLostEvent."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    received.clear()
    assert mgr.mark_lost(t.target_id, timestamp=5.0) is True
    assert len(received) == 1
    ev = received[0]
    assert isinstance(ev, TargetLostEvent)
    assert ev.target_id == t.target_id
    assert ev.timestamp == 5.0
    assert ev.payload["last_seen"] == 5.0


def test_mark_recovered_emits_recovered_event():
    """mark_recovered publishes a TargetRecoveredEvent."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    received.clear()
    new_track = _make_track(2)
    assert mgr.mark_recovered(t.target_id, new_track, timestamp=3.0) is True
    assert len(received) == 1
    ev = received[0]
    assert isinstance(ev, TargetRecoveredEvent)
    assert ev.target_id == t.target_id
    assert ev.timestamp == 3.0
    assert ev.payload["new_track_id"] == 2
    # Recovery is a single event (the RECOVERED->ACTIVE step emits nothing).
    assert t.state == TargetState.ACTIVE


def test_lock_target_emits_locked_event():
    """lock_target publishes a TargetLockedEvent."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    received.clear()
    assert mgr.lock_target(t.target_id) is True
    assert len(received) == 1
    ev = received[0]
    assert isinstance(ev, TargetLockedEvent)
    assert ev.target_id == t.target_id


def test_mark_removed_emits_removed_event():
    """mark_removed publishes a TargetRemovedEvent."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    received.clear()
    assert mgr.mark_removed(t.target_id) is True
    assert len(received) == 1
    ev = received[0]
    assert isinstance(ev, TargetRemovedEvent)
    assert ev.target_id == t.target_id
    assert ev.payload["final_state"] == "LOST"


# ---------------------------------------------------------------------------
# 2. 重复状态 (duplicate transitions emit nothing)
# ---------------------------------------------------------------------------

def test_mark_lost_twice_emits_once():
    """mark_lost on an already-LOST target fails and emits nothing."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    assert mgr.mark_lost(t.target_id, timestamp=2.0) is True
    received.clear()
    # Second mark_lost: LOST -> LOST is invalid.
    assert mgr.mark_lost(t.target_id, timestamp=3.0) is False
    assert received == []


def test_lock_twice_emits_once():
    """lock on an already-LOCKED target fails and emits nothing."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    assert mgr.lock_target(t.target_id) is True
    received.clear()
    assert mgr.lock_target(t.target_id) is False
    assert received == []


def test_recover_active_target_emits_nothing():
    """mark_recovered on an ACTIVE target fails and emits nothing."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    received.clear()
    # ACTIVE -> RECOVERED is invalid.
    assert mgr.mark_recovered(t.target_id, _make_track(2)) is False
    assert received == []


# ---------------------------------------------------------------------------
# 3. 非法状态 (invalid transitions emit nothing)
# ---------------------------------------------------------------------------

def test_remove_from_active_emits_nothing():
    """mark_removed on an ACTIVE target is invalid and emits nothing."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    received.clear()
    # ACTIVE -> REMOVED is not allowed (must go via LOST).
    assert mgr.mark_removed(t.target_id) is False
    assert received == []


def test_lock_from_lost_emits_nothing():
    """lock on a LOST target is invalid and emits nothing."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    received.clear()
    assert mgr.lock_target(t.target_id) is False
    assert received == []


def test_recover_from_locked_emits_nothing():
    """mark_recovered on a LOCKED target is invalid and emits nothing."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.lock_target(t.target_id)
    received.clear()
    assert mgr.mark_recovered(t.target_id, _make_track(2)) is False
    assert received == []


def test_not_found_emits_nothing():
    """Operations on an unknown target_id emit nothing."""
    mgr, _bus, received = _make_manager_with_bus()
    assert mgr.mark_lost("ghost", timestamp=1.0) is False
    assert mgr.lock_target("ghost") is False
    assert mgr.mark_removed("ghost") is False
    assert mgr.mark_recovered("ghost", _make_track(1)) is False
    assert received == []


# ---------------------------------------------------------------------------
# 4. 事件顺序 (event ordering)
# ---------------------------------------------------------------------------

def test_full_lifecycle_event_sequence():
    """A full lifecycle produces events in the correct order."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.lock_target(t.target_id)              # ACTIVE -> LOCKED
    mgr.unlock_target(t.target_id)            # LOCKED -> ACTIVE (no event)
    mgr.mark_lost(t.target_id, timestamp=5.0)  # ACTIVE -> LOST
    mgr.mark_recovered(                        # LOST -> RECOVERED -> ACTIVE
        t.target_id, _make_track(2), timestamp=6.0,
    )
    mgr.mark_lost(t.target_id, timestamp=9.0)  # ACTIVE -> LOST
    mgr.mark_removed(t.target_id)              # LOST -> REMOVED

    assert _event_types(received) == [
        "target.created",
        "target.locked",
        "target.lost",
        "target.recovered",
        "target.lost",
        "target.removed",
    ]


def test_recover_sequence_emits_only_recovered():
    """mark_recovered emits exactly one Recovered event (not two)."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    received.clear()
    mgr.mark_recovered(t.target_id, _make_track(2), timestamp=3.0)
    # Only the Recovered event; the internal RECOVERED->ACTIVE emits nothing.
    assert _event_types(received) == ["target.recovered"]


def test_event_timestamps_preserve_order():
    """Event timestamps reflect the supplied transition timestamps."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=10.0)
    mgr.mark_lost(t.target_id, timestamp=20.0)
    mgr.mark_recovered(t.target_id, _make_track(2), timestamp=30.0)
    ts = [e.timestamp for e in received]
    # Created uses last_seen (10.0); Lost uses 20.0; Recovered uses 30.0.
    assert ts[0] <= ts[1] <= ts[2]


# ---------------------------------------------------------------------------
# 5. 事件字段校验 (event field validation)
# ---------------------------------------------------------------------------

def test_event_carries_target_id_and_slot():
    """Every event carries the originating target_id and slot_id."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), slot_id=3, last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    for ev in received:
        assert ev.target_id == t.target_id
        assert ev.slot_id == 3


def test_event_ids_are_unique_and_sequential():
    """Event ids are unique and monotonically increasing."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    mgr.mark_removed(t.target_id)
    ids = [e.event_id for e in received]
    assert len(set(ids)) == len(ids)  # unique
    assert ids == sorted(ids)         # sequential


def test_event_types_are_correct_classes():
    """Each published event is an instance of the expected BaseEvent subclass."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    mgr.mark_removed(t.target_id)
    assert isinstance(received[0], TargetCreatedEvent)
    assert isinstance(received[1], TargetLostEvent)
    assert isinstance(received[2], TargetRemovedEvent)
    for ev in received:
        assert isinstance(ev, BaseEvent)


# ---------------------------------------------------------------------------
# 6. 向后兼容 (no bus => no events, behaviour unchanged)
# ---------------------------------------------------------------------------

def test_no_event_bus_means_no_events():
    """Without an EventBus the manager publishes nothing."""
    mgr = TargetManager()  # no event_bus
    assert mgr.event_bus is None
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    mgr.mark_removed(t.target_id)
    # No bus to capture events, but operations still succeed.
    assert t.state == TargetState.REMOVED


def test_manager_works_without_bus_as_before():
    """Core lifecycle still functions identically with no bus."""
    mgr = TargetManager()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    assert t.state == TargetState.ACTIVE
    assert mgr.lock_target(t.target_id) is True
    assert t.state == TargetState.LOCKED
    assert mgr.unlock_target(t.target_id) is True
    assert t.state == TargetState.ACTIVE
    assert mgr.mark_lost(t.target_id, timestamp=2.0) is True
    assert t.state == TargetState.LOST


def test_event_bus_property_exposes_attached_bus():
    """The event_bus property returns the attached bus."""
    bus = EventBus()
    mgr = TargetManager(event_bus=bus)
    assert mgr.event_bus is bus


def test_repr_mentions_event_bus_state():
    """__repr__ reports whether a bus is attached."""
    on = TargetManager(event_bus=EventBus())
    off = TargetManager()
    assert "event_bus=on" in repr(on)
    assert "event_bus=off" in repr(off)


# ---------------------------------------------------------------------------
# 7. update_targets 集成 (batch operations emit events via sub-calls)
# ---------------------------------------------------------------------------

def test_update_targets_emits_created_events():
    """update_targets creating new targets emits Created events."""
    mgr, _bus, received = _make_manager_with_bus()
    mgr.update_targets([_make_track(1), _make_track(2)],
                       slot_id=0, timestamp=1.0)
    created = [e for e in received if isinstance(e, TargetCreatedEvent)]
    assert len(created) == 2


def test_update_targets_emits_lost_and_removed_on_sweep():
    """update_targets stale-sweep emits Lost then Removed events."""
    mgr, _bus, received = _make_manager_with_bus()
    # Create a target at t=1.0.
    mgr.update_targets([_make_track(1)], slot_id=0, timestamp=1.0)
    received.clear()
    # Frame 2 (t=10): empty tracks, age=9 >= stale -> ACTIVE -> LOST.
    mgr.update_targets(
        [], slot_id=0, timestamp=10.0, stale_threshold=5.0,
    )
    # Frame 3 (t=100): still empty, age=90 >= removal -> LOST -> REMOVED.
    mgr.update_targets(
        [], slot_id=0, timestamp=100.0,
        stale_threshold=5.0, removal_threshold=30.0,
    )
    types = _event_types(received)
    assert "target.lost" in types
    assert "target.removed" in types
    # Lost must come before Removed.
    assert types.index("target.lost") < types.index("target.removed")


def test_update_targets_recover_emits_recovered_event():
    """update_targets recovering a lost target emits a Recovered event."""
    mgr, _bus, received = _make_manager_with_bus()
    # Create then lose the target.
    mgr.update_targets([_make_track(1)], slot_id=0, timestamp=1.0)
    mgr.update_targets([], slot_id=0, timestamp=10.0, stale_threshold=5.0)
    received.clear()
    # Re-appear: recover.
    mgr.update_targets([_make_track(1)], slot_id=0, timestamp=12.0)
    types = _event_types(received)
    assert "target.recovered" in types


# ---------------------------------------------------------------------------
# 8. 事件发布异常隔离 (emit fault isolation -- audit P1 fix)
# ---------------------------------------------------------------------------

def test_emit_failure_does_not_break_transition():
    """A failure inside _emit_lifecycle_event must not affect the transition.

    The lifecycle transition is authoritative; event publication is a
    best-effort side channel. If event construction or publish raises,
    the transition method must still return its normal bool and the
    Target state must still be updated.
    """
    mgr, bus, _received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)

    # Sabotage the bus's dispatcher so publish() raises inside _emit.
    # EventBus uses __slots__, so we swap the _dispatcher instance (which
    # is a slot) rather than monkey-patching the publish method.
    original_dispatcher = bus._dispatcher

    class _BoomDispatcher:
        def dispatch(self, subscribers, event):
            raise RuntimeError("simulated publish failure")

    bus._dispatcher = _BoomDispatcher()  # type: ignore[assignment]
    try:
        # mark_lost must succeed (return True) despite the emit failure.
        result = mgr.mark_lost(t.target_id, timestamp=2.0)
        assert result is True
        # The lifecycle state change still took effect.
        assert t.state == TargetState.LOST
    finally:
        bus._dispatcher = original_dispatcher  # type: ignore[assignment]

    # After restoring the dispatcher, events flow again.
    received: list = []
    bus.subscribe("target.removed", received.append)
    assert mgr.mark_removed(t.target_id, timestamp=3.0) is True
    assert len(received) == 1


def test_lock_target_accepts_timestamp():
    """lock_target forwards its timestamp to the emitted event (audit P2)."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    received.clear()
    assert mgr.lock_target(t.target_id, timestamp=42.0) is True
    ev = received[0]
    assert ev.timestamp == 42.0


def test_mark_removed_accepts_timestamp():
    """mark_removed forwards its timestamp to the emitted event (audit P2)."""
    mgr, _bus, received = _make_manager_with_bus()
    t = mgr.create_target(track=_make_track(1), last_seen=1.0)
    mgr.mark_lost(t.target_id, timestamp=2.0)
    received.clear()
    assert mgr.mark_removed(t.target_id, timestamp=99.0) is True
    ev = received[0]
    assert ev.timestamp == 99.0


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    failures = 0
    g = globals()
    names = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    for name in names:
        try:
            g[name]()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(names) - failures}/{len(names)} tests passed")
    raise SystemExit(1 if failures else 0)
