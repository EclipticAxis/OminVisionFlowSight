"""Unit tests for visioncore.eventbus (EventBus + typed events).

Verifies subscription, unsubscription, event publishing, multi-subscriber
delivery, duplicate subscriptions, empty-subscriber publishing, wildcard
matching, exception isolation, basic thread safety, and -- in the
second half -- the typed event hierarchy (BaseEvent and the five
Target lifecycle events): object creation, serialisation, and field
validation, plus end-to-end delivery through the EventBus.

Run::

    python -m pytest tests/test_event_bus.py -v
    # or
    python tests/test_event_bus.py
"""

from __future__ import annotations

import dataclasses
import os
import sys
import threading

# Ensure the project root is importable when the file is run directly
# (``python tests/test_event_bus.py``) as well as under pytest run from the
# repository root.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from visioncore.core.event import Event  # noqa: E402
from visioncore.eventbus import (  # noqa: E402
    BaseEvent,
    Dispatcher,
    EventBus,
    EventListener,
    Subscriber,
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_event(event_type: str = "target.lost",
                timestamp: float = 1.0,
                payload: dict | None = None) -> Event:
    """Create a minimal Event for testing."""
    return Event(event_type, timestamp, payload if payload is not None else {})


def _make_bus() -> EventBus:
    """Create a fresh EventBus for each test."""
    return EventBus()


# ---------------------------------------------------------------------------
# 1. 订阅 (subscribe)
# ---------------------------------------------------------------------------

def test_subscribe_returns_subscriber():
    """subscribe() returns a Subscriber with a unique id and the event type."""
    bus = _make_bus()
    sub = bus.subscribe("target.created", lambda e: None)
    assert isinstance(sub, Subscriber)
    assert sub.event_type == "target.created"
    assert sub.id.startswith("sub-")
    assert sub.active is True
    assert bus.subscriber_count() == 1


def test_subscribe_assigns_sequential_ids():
    """Subscriber ids are monotonically increasing."""
    bus = _make_bus()
    s1 = bus.subscribe("a", lambda e: None)
    s2 = bus.subscribe("b", lambda e: None)
    s3 = bus.subscribe("c", lambda e: None)
    assert s1.id < s2.id < s3.id


def test_subscribe_none_callback_raises():
    """A None callback is rejected with TypeError."""
    bus = _make_bus()
    try:
        bus.subscribe("a", None)  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("TypeError not raised for None callback")


def test_subscribe_non_callable_raises():
    """A non-callable callback is rejected with TypeError."""
    bus = _make_bus()
    try:
        bus.subscribe("a", 123)  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("TypeError not raised for non-callable callback")


def test_subscribe_empty_event_type_raises():
    """An empty event_type string is rejected with ValueError."""
    bus = _make_bus()
    try:
        bus.subscribe("", lambda e: None)
    except ValueError:
        return
    raise AssertionError("ValueError not raised for empty event_type")


# ---------------------------------------------------------------------------
# 2. 事件发布 (publish)
# ---------------------------------------------------------------------------

def test_publish_delivers_to_matching_subscriber():
    """A published event reaches the matching subscriber's callback."""
    bus = _make_bus()
    received: list[Event] = []
    bus.subscribe("target.lost", received.append)
    ev = _make_event("target.lost", 1.0, {"target_id": "T-1"})
    invoked = bus.publish(ev)
    assert invoked == 1
    assert received == [ev]
    assert received[0].payload["target_id"] == "T-1"


def test_publish_returns_invocation_count():
    """publish() returns the number of callbacks actually invoked."""
    bus = _make_bus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("a", lambda e: None)
    assert bus.publish(_make_event("a")) == 2
    assert bus.publish(_make_event("b")) == 0


def test_publish_only_matching_type_receives():
    """Subscribers for a different event type do not receive the event."""
    bus = _make_bus()
    a_received: list[Event] = []
    b_received: list[Event] = []
    bus.subscribe("a", a_received.append)
    bus.subscribe("b", b_received.append)
    bus.publish(_make_event("a"))
    assert len(a_received) == 1
    assert b_received == []


# ---------------------------------------------------------------------------
# 3. 取消订阅 (unsubscribe)
# ---------------------------------------------------------------------------

def test_unsubscribe_stops_delivery():
    """After unsubscribe the callback is no longer invoked."""
    bus = _make_bus()
    received: list[Event] = []
    sub = bus.subscribe("a", received.append)
    assert bus.unsubscribe(sub) is True
    bus.publish(_make_event("a"))
    assert received == []
    assert bus.subscriber_count() == 0


def test_unsubscribe_by_id_string():
    """unsubscribe() accepts the subscriber id string as well as the token."""
    bus = _make_bus()
    received: list[Event] = []
    sub = bus.subscribe("a", received.append)
    assert bus.unsubscribe(sub.id) is True
    bus.publish(_make_event("a"))
    assert received == []


def test_unsubscribe_unknown_returns_false():
    """Unsubscribing an unknown id is a no-op returning False."""
    bus = _make_bus()
    assert bus.unsubscribe("sub-9999") is False


def test_unsubscribe_is_idempotent():
    """Unsubscribing the same token twice returns True then False."""
    bus = _make_bus()
    sub = bus.subscribe("a", lambda e: None)
    assert bus.unsubscribe(sub) is True
    assert bus.unsubscribe(sub) is False


def test_unsubscribe_marks_subscriber_inactive():
    """The Subscriber token is marked inactive after removal."""
    bus = _make_bus()
    sub = bus.subscribe("a", lambda e: None)
    assert sub.active is True
    bus.unsubscribe(sub)
    assert sub.active is False


# ---------------------------------------------------------------------------
# 4. 多订阅者 (multiple subscribers)
# ---------------------------------------------------------------------------

def test_multiple_subscribers_same_type_all_receive():
    """All subscribers for the same event type receive the event."""
    bus = _make_bus()
    r1: list[Event] = []
    r2: list[Event] = []
    r3: list[Event] = []
    bus.subscribe("a", r1.append)
    bus.subscribe("a", r2.append)
    bus.subscribe("a", r3.append)
    ev = _make_event("a")
    assert bus.publish(ev) == 3
    assert r1 == [ev]
    assert r2 == [ev]
    assert r3 == [ev]


def test_multiple_subscribers_delivered_in_subscription_order():
    """Subscribers are invoked in the order they were registered."""
    bus = _make_bus()
    order: list[str] = []
    bus.subscribe("a", lambda e: order.append("first"))
    bus.subscribe("a", lambda e: order.append("second"))
    bus.subscribe("a", lambda e: order.append("third"))
    bus.publish(_make_event("a"))
    assert order == ["first", "second", "third"]


def test_unsubscribe_one_keeps_others():
    """Removing one subscriber leaves the others intact."""
    bus = _make_bus()
    r1: list[Event] = []
    r2: list[Event] = []
    s1 = bus.subscribe("a", r1.append)
    bus.subscribe("a", r2.append)
    bus.unsubscribe(s1)
    ev = _make_event("a")
    assert bus.publish(ev) == 1
    assert r1 == []
    assert r2 == [ev]


# ---------------------------------------------------------------------------
# 5. 重复订阅 (duplicate subscriptions)
# ---------------------------------------------------------------------------

def test_duplicate_subscription_invoked_twice():
    """The same callback subscribed twice is invoked twice per event."""
    bus = _make_bus()
    received: list[Event] = []
    bus.subscribe("a", received.append)
    bus.subscribe("a", received.append)
    ev = _make_event("a")
    assert bus.publish(ev) == 2
    assert received == [ev, ev]


def test_duplicate_subscription_distinct_tokens():
    """Duplicate subscriptions yield distinct Subscriber tokens."""
    bus = _make_bus()
    cb: EventListener = lambda e: None  # noqa: E731
    s1 = bus.subscribe("a", cb)
    s2 = bus.subscribe("a", cb)
    assert s1 is not s2
    assert s1.id != s2.id
    assert bus.subscriber_count() == 2


def test_duplicate_subscription_unsubscribe_only_one():
    """Unsubscribing one duplicate leaves the other active."""
    bus = _make_bus()
    received: list[Event] = []
    s1 = bus.subscribe("a", received.append)
    bus.subscribe("a", received.append)
    bus.unsubscribe(s1)
    ev = _make_event("a")
    assert bus.publish(ev) == 1
    assert received == [ev]


# ---------------------------------------------------------------------------
# 6. 空订阅者 (publish with no subscribers)
# ---------------------------------------------------------------------------

def test_publish_with_no_subscribers_is_noop():
    """Publishing when nobody is subscribed returns 0 and raises nothing."""
    bus = _make_bus()
    assert bus.publish(_make_event("a")) == 0
    assert bus.publish(_make_event("anything")) == 0


def test_publish_with_no_matching_subscribers_is_noop():
    """Publishing an event type with only other-type subscribers is a no-op."""
    bus = _make_bus()
    received: list[Event] = []
    bus.subscribe("a", received.append)
    bus.publish(_make_event("b"))
    assert received == []


def test_publish_after_clear_is_noop():
    """After clear(), no subscribers receive events."""
    bus = _make_bus()
    received: list[Event] = []
    bus.subscribe("a", received.append)
    removed = bus.clear()
    assert removed == 1
    bus.publish(_make_event("a"))
    assert received == []


# ---------------------------------------------------------------------------
# 7. 通配符订阅 (wildcard)
# ---------------------------------------------------------------------------

def test_wildcard_receives_all_event_types():
    """A '*' subscriber receives every published event."""
    bus = _make_bus()
    received: list[Event] = []
    bus.subscribe("*", received.append)
    e1 = _make_event("a")
    e2 = _make_event("b")
    e3 = _make_event("target.lost")
    bus.publish(e1)
    bus.publish(e2)
    bus.publish(e3)
    assert received == [e1, e2, e3]


def test_wildcard_and_typed_both_receive():
    """Wildcard and typed subscribers both receive a matching event."""
    bus = _make_bus()
    wild: list[Event] = []
    typed: list[Event] = []
    bus.subscribe("*", wild.append)
    bus.subscribe("a", typed.append)
    ev = _make_event("a")
    assert bus.publish(ev) == 2
    assert wild == [ev]
    assert typed == [ev]


# ---------------------------------------------------------------------------
# 8. 异常隔离 (exception isolation)
# ---------------------------------------------------------------------------

def test_callback_exception_does_not_block_others():
    """An exception in one callback is logged and other callbacks still run."""
    bus = _make_bus()
    received: list[Event] = []

    def boom(_e: Event) -> None:
        raise RuntimeError("simulated failure")

    bus.subscribe("a", boom)
    bus.subscribe("a", received.append)
    ev = _make_event("a")
    # Both subscribers counted: the failing one still "invoked" (attempted).
    assert bus.publish(ev) == 2
    assert received == [ev]


def test_dispatcher_isolates_exceptions():
    """Dispatcher returns the count of attempted invocations despite errors."""
    d = Dispatcher()

    def boom(_e: Event) -> None:
        raise ValueError("fail")

    ok: list[Event] = []
    s1 = Subscriber(id="s1", event_type="a", callback=boom)
    s2 = Subscriber(id="s2", event_type="a", callback=ok.append)
    count = d.dispatch([s1, s2], _make_event("a"))
    assert count == 2
    assert len(ok) == 1


# ---------------------------------------------------------------------------
# 9. 回调内重入 (re-entrancy from callbacks)
# ---------------------------------------------------------------------------

def test_callback_can_unsubscribe_itself():
    """A callback may unsubscribe its own token without corrupting dispatch."""
    bus = _make_bus()
    received: list[Event] = []

    def self_unsub(_e: Event) -> None:
        received.append(_e)
        bus.unsubscribe(token)

    token = bus.subscribe("a", self_unsub)
    ev = _make_event("a")
    bus.publish(ev)
    assert received == [ev]
    # Subsequent publishes reach nobody.
    assert bus.publish(ev) == 0
    assert bus.subscriber_count() == 0


def test_callback_can_publish_nested_event():
    """A callback may publish another event; re-entrancy does not deadlock."""
    bus = _make_bus()
    outer: list[Event] = []
    inner: list[Event] = []

    def outer_cb(e: Event) -> None:
        outer.append(e)
        bus.publish(_make_event("inner"))

    bus.subscribe("outer", outer_cb)
    bus.subscribe("inner", inner.append)
    bus.publish(_make_event("outer"))
    assert len(outer) == 1
    assert len(inner) == 1


# ---------------------------------------------------------------------------
# 10. 内省与计数 (introspection)
# ---------------------------------------------------------------------------

def test_subscriber_count_by_type():
    """subscriber_count(type) counts only that type's subscribers."""
    bus = _make_bus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("a", lambda e: None)
    bus.subscribe("b", lambda e: None)
    bus.subscribe("*", lambda e: None)
    assert bus.subscriber_count() == 4
    assert bus.subscriber_count("a") == 2
    assert bus.subscriber_count("b") == 1
    assert bus.subscriber_count("*") == 1
    assert bus.subscriber_count("missing") == 0


def test_len_matches_total_count():
    """len(bus) equals the total subscriber count."""
    bus = _make_bus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("*", lambda e: None)
    assert len(bus) == 2


def test_repr_is_informative():
    """__repr__ reports type and subscriber counts without raising."""
    bus = _make_bus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("*", lambda e: None)
    text = repr(bus)
    assert "EventBus" in text
    assert "types=1" in text
    assert "wildcard_subscribers=1" in text


def test_clear_returns_count_and_resets():
    """clear() removes everything and returns the removed count."""
    bus = _make_bus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("b", lambda e: None)
    bus.subscribe("*", lambda e: None)
    assert bus.clear() == 3
    assert bus.subscriber_count() == 0
    assert len(bus) == 0


# ---------------------------------------------------------------------------
# 11. 线程安全 (thread safety)
# ---------------------------------------------------------------------------

def test_concurrent_publish_and_subscribe():
    """Concurrent publishers and subscribers do not corrupt the bus."""
    bus = _make_bus()
    counter_lock = threading.Lock()
    counts: list[int] = []

    def worker() -> None:
        local: list[Event] = []
        sub = bus.subscribe("a", local.append)
        for i in range(200):
            bus.publish(_make_event("a", float(i)))
        with counter_lock:
            counts.append(len(local))
        bus.unsubscribe(sub)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each worker subscribed before publishing, but other workers were
    # also publishing concurrently, so each receives >= its own 200
    # publishes. The exact total depends on interleaving; we only assert
    # the bus is empty afterwards and no exception propagated.
    assert bus.subscriber_count() == 0
    assert all(c >= 200 for c in counts)


def test_concurrent_publish_is_safe():
    """Many threads publishing to a shared subscriber do not lose events."""
    bus = _make_bus()
    received: list[int] = []
    lock = threading.Lock()

    def cb(e: Event) -> None:
        with lock:
            received.append(1)

    bus.subscribe("a", cb)

    def worker() -> None:
        for _ in range(500):
            bus.publish(_make_event("a"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(received) == 8 * 500


# ---------------------------------------------------------------------------
# 12. 事件对象创建 (typed event creation)
# ---------------------------------------------------------------------------

_TYPED_EVENT_CLASSES = [
    TargetCreatedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetLockedEvent,
    TargetRemovedEvent,
]

_TYPED_EVENT_TYPES = [
    "target.created",
    "target.lost",
    "target.recovered",
    "target.locked",
    "target.removed",
]


def test_typed_event_creation_basic():
    """Each typed event stores all common fields verbatim."""
    ev = TargetCreatedEvent(
        event_id="evt-1", timestamp=1.5,
        target_id="S0-T0001", slot_id=0,
        payload={"track_id": 1, "priority": 5},
    )
    assert ev.event_id == "evt-1"
    assert ev.timestamp == 1.5
    assert ev.target_id == "S0-T0001"
    assert ev.slot_id == 0
    assert ev.event_type == "target.created"
    assert ev.payload == {"track_id": 1, "priority": 5}


def test_typed_event_default_payload_is_empty_dict():
    """Omitting payload yields an independent empty dict per instance."""
    ev = TargetLostEvent(
        event_id="evt-2", timestamp=2.0,
        target_id="S0-T0002", slot_id=1,
    )
    assert ev.payload == {}
    assert isinstance(ev.payload, dict)


def test_each_typed_event_has_correct_event_type():
    """Every concrete subclass fixes its event_type to the right string."""
    for cls, expected in zip(_TYPED_EVENT_CLASSES, _TYPED_EVENT_TYPES):
        ev = cls(
            event_id="e", timestamp=0.0,
            target_id="T", slot_id=0,
        )
        assert ev.event_type == expected, (
            f"{cls.__name__}.event_type = {ev.event_type!r}, "
            f"expected {expected!r}"
        )


def test_typed_events_are_frozen():
    """Frozen dataclasses reject attribute assignment."""
    ev = TargetCreatedEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    try:
        ev.event_id = "mutated"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("FrozenInstanceError not raised on mutation")


def test_typed_events_use_slots():
    """Typed events are slotted: no __dict__, attribute storage only."""
    ev = TargetCreatedEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    assert not hasattr(ev, "__dict__")
    # Common fields live in __slots__.
    for attr in ("event_id", "timestamp", "target_id", "slot_id",
                 "event_type", "payload"):
        assert hasattr(ev, attr)


def test_typed_events_are_subclasses_of_base():
    """All concrete events inherit from BaseEvent."""
    for cls in _TYPED_EVENT_CLASSES:
        assert issubclass(cls, BaseEvent)


def test_typed_event_distinct_instances_are_independent():
    """Two events do not share mutable default payload (no aliasing)."""
    a = TargetCreatedEvent(
        event_id="a", timestamp=0.0, target_id="T", slot_id=0,
    )
    b = TargetCreatedEvent(
        event_id="b", timestamp=0.0, target_id="T", slot_id=0,
    )
    a.payload["k"] = "v"  # type: ignore[index]
    assert b.payload == {}


# ---------------------------------------------------------------------------
# 13. 事件序列化 (serialisation)
# ---------------------------------------------------------------------------

def test_typed_event_asdict_contains_all_fields():
    """dataclasses.asdict round-trips every field incl. event_type."""
    ev = TargetCreatedEvent(
        event_id="evt-1", timestamp=1.5,
        target_id="S0-T0001", slot_id=2,
        payload={"track_id": 7},
    )
    d = dataclasses.asdict(ev)
    assert d == {
        "event_id": "evt-1",
        "timestamp": 1.5,
        "target_id": "S0-T0001",
        "slot_id": 2,
        "event_type": "target.created",
        "payload": {"track_id": 7},
    }


def test_typed_event_asdict_payload_is_a_copy():
    """asdict returns a deep-copied payload dict (no aliasing)."""
    original = {"track_id": 1}
    ev = TargetCreatedEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
        payload=original,
    )
    d = dataclasses.asdict(ev)
    d["payload"]["track_id"] = 999
    assert original["track_id"] == 1
    assert ev.payload["track_id"] == 1


def test_typed_event_repr_is_informative():
    """__repr__ shows the concrete class name, event_type, and payload keys."""
    ev = TargetLostEvent(
        event_id="e1", timestamp=12.5,
        target_id="S0-T0001", slot_id=0,
        payload={"last_seen": 10.0, "missed_frames": 4},
    )
    text = repr(ev)
    assert text.startswith("TargetLostEvent(")
    assert "event_type='target.lost'" in text
    assert "target_id='S0-T0001'" in text
    assert "slot_id=0" in text
    assert "payload_keys={'last_seen', 'missed_frames'}" in text


def test_typed_event_repr_empty_payload():
    """__repr__ renders an empty payload as {}."""
    ev = TargetRemovedEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    assert "payload_keys={}" in repr(ev)


def test_typed_event_field_names_match_contract():
    """fields() exposes exactly the six required contract fields."""
    expected = {"event_id", "timestamp", "target_id",
                "slot_id", "event_type", "payload"}
    for cls in _TYPED_EVENT_CLASSES:
        names = {f.name for f in dataclasses.fields(cls)}
        assert names == expected, (
            f"{cls.__name__} fields = {names}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# 14. 事件字段校验 (field validation)
# ---------------------------------------------------------------------------

def test_event_type_is_non_empty_string_for_all_subclasses():
    """No concrete event leaves event_type as the empty base default."""
    for cls in _TYPED_EVENT_CLASSES:
        ev = cls(event_id="e", timestamp=0.0, target_id="T", slot_id=0)
        assert isinstance(ev.event_type, str)
        assert ev.event_type, f"{cls.__name__} has empty event_type"


def test_slot_id_accepts_full_range():
    """slot_id accepts the documented [0, 3] camera range."""
    for slot in range(4):
        ev = TargetCreatedEvent(
            event_id="e", timestamp=0.0, target_id="T", slot_id=slot,
        )
        assert ev.slot_id == slot


def test_event_type_constants_are_unique():
    """The five typed events carry five distinct event_type strings."""
    types = []
    for cls in _TYPED_EVENT_CLASSES:
        ev = cls(event_id="e", timestamp=0.0, target_id="T", slot_id=0)
        types.append(ev.event_type)
    assert len(set(types)) == len(types), (
        f"Duplicate event_type values: {types}"
    )


def test_event_type_matches_lifecycle_strings():
    """event_type strings follow the target.* dot convention."""
    for cls in _TYPED_EVENT_CLASSES:
        ev = cls(event_id="e", timestamp=0.0, target_id="T", slot_id=0)
        assert ev.event_type.startswith("target.")
        assert ev.event_type.count(".") == 1


def test_target_id_and_event_id_are_stored_as_strings():
    """Identifier fields preserve the caller-provided string values."""
    ev = TargetLockedEvent(
        event_id="evt-42", timestamp=3.0,
        target_id="S3-T0017", slot_id=3,
    )
    assert ev.event_id == "evt-42"
    assert ev.target_id == "S3-T0017"
    assert isinstance(ev.event_id, str)
    assert isinstance(ev.target_id, str)


def test_timestamp_is_float_compatible():
    """timestamp accepts int and float values."""
    ev_int = TargetCreatedEvent(
        event_id="e1", timestamp=1, target_id="T", slot_id=0,  # type: ignore[arg-type]
    )
    ev_float = TargetCreatedEvent(
        event_id="e2", timestamp=1.5, target_id="T", slot_id=0,
    )
    assert ev_int.timestamp == 1
    assert ev_float.timestamp == 1.5


# ---------------------------------------------------------------------------
# 15. 类型化事件与 EventBus 集成 (typed events through the bus)
# ---------------------------------------------------------------------------

def test_bus_dispatches_typed_event_by_event_type():
    """EventBus routes a TargetCreatedEvent to its event_type subscribers."""
    bus = _make_bus()
    received: list[TargetCreatedEvent] = []
    bus.subscribe("target.created", received.append)
    ev = TargetCreatedEvent(
        event_id="e1", timestamp=1.0,
        target_id="S0-T0001", slot_id=0,
        payload={"track_id": 1},
    )
    assert bus.publish(ev) == 1
    assert received == [ev]
    assert received[0].payload["track_id"] == 1


def test_bus_wildcard_receives_all_typed_events():
    """A '*' subscriber receives every typed event subclass."""
    bus = _make_bus()
    received: list[BaseEvent] = []
    bus.subscribe("*", received.append)
    events = [
        cls(event_id=f"e{i}", timestamp=float(i),
            target_id="T", slot_id=0)
        for i, cls in enumerate(_TYPED_EVENT_CLASSES)
    ]
    for ev in events:
        bus.publish(ev)
    assert received == events
    assert [e.event_type for e in received] == _TYPED_EVENT_TYPES


def test_bus_typed_events_isolated_by_type():
    """Subscribers for one event_type do not receive other typed events."""
    bus = _make_bus()
    created: list[BaseEvent] = []
    lost: list[BaseEvent] = []
    bus.subscribe("target.created", created.append)
    bus.subscribe("target.lost", lost.append)
    c = TargetCreatedEvent(
        event_id="c", timestamp=0.0, target_id="T", slot_id=0,
    )
    l = TargetLostEvent(
        event_id="l", timestamp=1.0, target_id="T", slot_id=0,
    )
    bus.publish(c)
    bus.publish(l)
    assert created == [c]
    assert lost == [l]


def test_bus_typed_event_and_core_event_coexist():
    """Typed events and core.Event coexist on the same bus (duck typing)."""
    bus = _make_bus()
    received: list = []
    bus.subscribe("target.removed", received.append)
    # core.Event
    core_ev = Event("target.removed", 1.0, {"note": "legacy"})
    # typed event
    typed_ev = TargetRemovedEvent(
        event_id="e1", timestamp=2.0, target_id="T", slot_id=0,
    )
    assert bus.publish(core_ev) == 1
    assert bus.publish(typed_ev) == 1
    assert received == [core_ev, typed_ev]


# ---------------------------------------------------------------------------
# 16. 类型化订阅 (subscribe by event class)
# ---------------------------------------------------------------------------

def test_subscribe_by_class_sets_event_class():
    """subscribe(EventClass, cb) returns a Subscriber with event_class set."""
    bus = _make_bus()
    sub = bus.subscribe(TargetLostEvent, lambda e: None)
    assert sub.event_class is TargetLostEvent
    # event_type derived from the class's default.
    assert sub.event_type == "target.lost"


def test_subscribe_by_class_publish_typed_event_received():
    """A class subscription receives its own typed event instances."""
    bus = _make_bus()
    received: list[TargetLostEvent] = []
    bus.subscribe(TargetLostEvent, received.append)
    ev = TargetLostEvent(
        event_id="e1", timestamp=1.0, target_id="T", slot_id=0,
    )
    assert bus.publish(ev) == 1
    assert received == [ev]


def test_subscribe_by_class_rejects_base_event():
    """Subscribing the abstract BaseEvent is rejected (empty event_type)."""
    bus = _make_bus()
    try:
        bus.subscribe(BaseEvent, lambda e: None)
    except ValueError:
        return
    raise AssertionError("ValueError not raised for BaseEvent subscription")


def test_subscribe_by_class_rejects_non_class_non_string():
    """A non-string, non-class key is rejected with TypeError."""
    bus = _make_bus()
    try:
        bus.subscribe(123, lambda e: None)  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("TypeError not raised for int key")


def test_subscribe_by_class_unsubscribe_works():
    """A class subscription can be unsubscribed by its token."""
    bus = _make_bus()
    received: list = []
    sub = bus.subscribe(TargetLostEvent, received.append)
    assert bus.unsubscribe(sub) is True
    bus.publish(TargetLostEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    ))
    assert received == []


# ---------------------------------------------------------------------------
# 17. 事件过滤 (event filtering -- type-safe isolation)
# ---------------------------------------------------------------------------

def test_class_subscription_does_not_receive_other_class():
    """A class subscription ignores typed events of a different class."""
    bus = _make_bus()
    lost: list = []
    bus.subscribe(TargetLostEvent, lost.append)
    bus.publish(TargetCreatedEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    ))
    assert lost == []


def test_class_subscription_does_not_receive_core_event_same_type():
    """A class subscription rejects a core.Event even if event_type matches."""
    bus = _make_bus()
    received: list = []
    bus.subscribe(TargetLostEvent, received.append)
    # core.Event carries event_type="target.lost" but is NOT a TargetLostEvent.
    bus.publish(Event("target.lost", 1.0))
    assert received == []


def test_string_subscription_receives_typed_event():
    """A string subscription receives typed events with matching event_type."""
    bus = _make_bus()
    received: list = []
    bus.subscribe("target.lost", received.append)
    ev = TargetLostEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    assert bus.publish(ev) == 1
    assert received == [ev]


def test_class_and_string_subscription_both_receive_typed():
    """Class and string subs for the same type both fire for typed events."""
    bus = _make_bus()
    cls_received: list = []
    str_received: list = []
    bus.subscribe(TargetLostEvent, cls_received.append)
    bus.subscribe("target.lost", str_received.append)
    ev = TargetLostEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    assert bus.publish(ev) == 2
    assert cls_received == [ev]
    assert str_received == [ev]


def test_string_subscription_does_not_receive_unrelated_typed_event():
    """A string sub for one type ignores a typed event of another type."""
    bus = _make_bus()
    received: list = []
    bus.subscribe("target.lost", received.append)
    bus.publish(TargetCreatedEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    ))
    assert received == []


# ---------------------------------------------------------------------------
# 18. 多事件类型 (multiple event types coexisting)
# ---------------------------------------------------------------------------

def test_multiple_class_subscriptions_isolated():
    """Each class subscription receives only its own events."""
    bus = _make_bus()
    created: list = []
    lost: list = []
    locked: list = []
    bus.subscribe(TargetCreatedEvent, created.append)
    bus.subscribe(TargetLostEvent, lost.append)
    bus.subscribe(TargetLockedEvent, locked.append)
    c = TargetCreatedEvent(
        event_id="c", timestamp=0.0, target_id="T", slot_id=0)
    l = TargetLostEvent(
        event_id="l", timestamp=1.0, target_id="T", slot_id=0)
    k = TargetLockedEvent(
        event_id="k", timestamp=2.0, target_id="T", slot_id=0)
    bus.publish(c)
    bus.publish(l)
    bus.publish(k)
    assert created == [c]
    assert lost == [l]
    assert locked == [k]


def test_wildcard_receives_all_typed_and_core_events():
    """A '*' subscription receives both typed and core.Event publishes."""
    bus = _make_bus()
    received: list = []
    bus.subscribe("*", received.append)
    typed = TargetLostEvent(
        event_id="t", timestamp=1.0, target_id="T", slot_id=0)
    core_ev = Event("anything", 2.0)
    bus.publish(typed)
    bus.publish(core_ev)
    assert received == [typed, core_ev]


def test_string_and_class_subscriptions_coexist_per_type():
    """String and class subscriptions for the same type coexist."""
    bus = _make_bus()
    s1 = bus.subscribe(TargetLostEvent, lambda e: None)
    s2 = bus.subscribe("target.lost", lambda e: None)
    assert s1 is not s2
    assert s1.event_class is TargetLostEvent
    assert s2.event_class is None
    assert bus.subscriber_count("target.lost") == 2


# ---------------------------------------------------------------------------
# 19. 未知事件类型 (publishing with no matching subscribers)
# ---------------------------------------------------------------------------

def test_publish_typed_event_no_subscribers_returns_zero():
    """Publishing a typed event with no subscribers returns 0."""
    bus = _make_bus()
    ev = TargetRemovedEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    assert bus.publish(ev) == 0


def test_publish_typed_event_only_other_class_subscribed_returns_zero():
    """A typed event with only an unrelated class subscription yields 0."""
    bus = _make_bus()
    bus.subscribe(TargetCreatedEvent, lambda e: None)
    ev = TargetLostEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    assert bus.publish(ev) == 0


def test_publish_typed_event_only_other_string_subscribed_returns_zero():
    """A typed event with only an unrelated string subscription yields 0."""
    bus = _make_bus()
    bus.subscribe("target.created", lambda e: None)
    ev = TargetLostEvent(
        event_id="e", timestamp=0.0, target_id="T", slot_id=0,
    )
    assert bus.publish(ev) == 0


def test_publish_event_without_event_type_attribute():
    """An object without event_type reaches only wildcard subscribers."""
    bus = _make_bus()
    wild: list = []
    typed_str: list = []
    cls_received: list = []
    bus.subscribe("*", wild.append)
    bus.subscribe("target.lost", typed_str.append)
    bus.subscribe(TargetLostEvent, cls_received.append)
    obj = object()  # no event_type attribute
    assert bus.publish(obj) == 1  # only wildcard
    assert wild == [obj]
    assert typed_str == []
    assert cls_received == []


# ---------------------------------------------------------------------------
# 20. 订阅计数 (subscriber_count with class keys)
# ---------------------------------------------------------------------------

def test_subscriber_count_accepts_class_key():
    """subscriber_count(EventClass) counts subscribers for that type."""
    bus = _make_bus()
    bus.subscribe(TargetLostEvent, lambda e: None)
    bus.subscribe("target.lost", lambda e: None)
    bus.subscribe(TargetCreatedEvent, lambda e: None)
    assert bus.subscriber_count(TargetLostEvent) == 2
    assert bus.subscriber_count(TargetCreatedEvent) == 1


def test_subscriber_count_rejects_invalid_key():
    """subscriber_count rejects non-str/non-class keys."""
    bus = _make_bus()
    try:
        bus.subscriber_count(123)  # type: ignore[arg-type]
    except TypeError:
        return
    raise AssertionError("TypeError not raised for invalid count key")


def test_clear_removes_class_subscriptions():
    """clear() removes class subscriptions too."""
    bus = _make_bus()
    bus.subscribe(TargetLostEvent, lambda e: None)
    bus.subscribe(TargetCreatedEvent, lambda e: None)
    assert bus.clear() == 2
    assert bus.subscriber_count() == 0


def test_repr_handles_class_subscriptions():
    """__repr__ does not raise when class subscriptions are present."""
    bus = _make_bus()
    bus.subscribe(TargetLostEvent, lambda e: None)
    bus.subscribe("a", lambda e: None)
    text = repr(bus)
    assert "EventBus" in text


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow running without pytest: collect and call every test_* function.
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
