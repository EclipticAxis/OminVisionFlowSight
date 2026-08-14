"""EventBus -- the central publish/subscribe hub for VisionCore.

The EventBus decouples event producers from event consumers. Producers
call :meth:`publish` with an event object; consumers register callbacks
via :meth:`subscribe`. The bus routes each published event to every
subscriber whose filter matches.

Subscription keys:
    :meth:`subscribe` accepts either of two filter forms:

    * **String key** -- an event-type string (e.g. ``"target.lost"``) or
      the wildcard ``"*"`` to receive every event. Matching is exact
      string equality against the published object's ``event_type``
      attribute; the wildcard is the only special token. This is the
      backward-compatible mode and works with both
      :class:`~visioncore.core.event.Event` and typed
      :class:`~visioncore.eventbus.events.BaseEvent` instances.
    * **Class key** -- a :class:`~visioncore.eventbus.events.BaseEvent`
      subclass (e.g. :class:`~visioncore.eventbus.events.TargetLostEvent`).
      The subscriber then matches only via ``isinstance`` -- it receives
      exactly the instances of that class (or its subclasses), regardless
      of ``event_type`` strings. This is type-safe: a class subscription
      for ``TargetLostEvent`` will not fire for a plain
      :class:`~visioncore.core.event.Event` even if its ``event_type``
      coincides.

    Both forms can be mixed freely on the same bus.

Scope of this milestone:
    This module provides the **core** bus only: subscribe, unsubscribe,
    publish, and introspection (subscriber counts). It deliberately does
    **not** wire into InferWorker, Tracker, the GUI, or TargetManager.
    Those integrations are subsequent milestones and are explicitly out
    of scope here.

Thread safety:
    All public methods acquire ``self._lock`` (an
    :class:`~threading.RLock`). The lock is re-entrant, so a callback
    running on the publishing thread may safely call ``subscribe`` /
    ``unsubscribe`` / ``publish`` again without deadlocking. Callback
    dispatch itself happens **outside** the lock: ``publish`` snapshots
    the matching subscribers under the lock, then releases it before
    invoking callbacks. This prevents long-running callbacks from
    blocking other publishers and avoids holding the lock across
    arbitrary user code.

Example:
    >>> from visioncore.core.event import Event
    >>> from visioncore.eventbus.bus import EventBus
    >>> from visioncore.eventbus.events import TargetLostEvent
    >>> bus = EventBus()
    >>> received = []
    >>> # string subscription (backward compatible)
    >>> sub = bus.subscribe("target.lost", received.append)
    >>> bus.publish(Event("target.lost", 1.0, {"target_id": "T-1"}))
    1
    >>> received[0].payload["target_id"]
    'T-1'
    >>> bus.unsubscribe(sub)
    True
    >>> # class subscription (typed)
    >>> typed = []
    >>> bus.subscribe(TargetLostEvent, typed.append)
    >>> bus.publish(TargetLostEvent(
    ...     event_id="e1", timestamp=2.0, target_id="T-1", slot_id=0))
    1
    >>> typed[0].event_type
    'target.lost'
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import threading
from typing import TYPE_CHECKING

from visioncore.eventbus.dispatcher import Dispatcher
from visioncore.eventbus.events import BaseEvent
from visioncore.eventbus.subscriber import EventListener, Subscriber

if TYPE_CHECKING:
    from visioncore.core.event import Event


class EventBus:
    """Thread-safe central publish/subscribe bus.

    Maintains a registry of Subscribers keyed by event type, plus a
    separate list for wildcard (``"*"``) subscribers. Each
    :meth:`publish` collects the matching subscribers under the bus lock,
    then dispatches them outside the lock via the shared
    :class:`~visioncore.eventbus.dispatcher.Dispatcher`.

    Attributes:
        _subscribers: Mapping ``event_type -> list[Subscriber]`` for
            typed subscriptions. Insertion order within each list is
            preserved (subscription order = delivery order).
        _wildcard_subscribers: Subscribers registered with ``"*"`` that
            receive every event.
        _lock: Re-entrant lock protecting the registry.
        _dispatcher: Shared synchronous dispatcher.
        _id_counter: Monotonic counter for Subscriber ids.
        _logger: Module-level logger.
    """

    __slots__ = (
        "_subscribers",
        "_wildcard_subscribers",
        "_lock",
        "_dispatcher",
        "_id_counter",
        "_logger",
    )

    def __init__(self) -> None:
        """Initialise an empty EventBus."""
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._wildcard_subscribers: list[Subscriber] = []
        self._lock: threading.RLock = threading.RLock()
        self._dispatcher: Dispatcher = Dispatcher()
        self._id_counter: itertools.count = itertools.count(1)
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.debug("EventBus initialised (empty)")

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------

    def subscribe(
        self,
        event_type: "str | type[BaseEvent]",
        callback: EventListener,
    ) -> Subscriber:
        """Register ``callback`` for events matching ``event_type``.

        The same callback may be subscribed multiple times (even for the
        same key); each registration yields a distinct Subscriber and
        will be invoked once per matching published event.

        Two key forms are accepted (see the module docstring for the
        full matching semantics):

        * **String key**: an event-type string (e.g. ``"target.lost"``)
          or the wildcard ``"*"``. The subscriber matches published
          objects by string equality on their ``event_type`` attribute
          (or matches everything for ``"*"``).
        * **Class key**: a :class:`~visioncore.eventbus.events.BaseEvent`
          subclass (e.g. :class:`~visioncore.eventbus.events.TargetLostEvent`).
          The subscriber matches via ``isinstance`` -- only instances of
          that class (or subclasses) are delivered. The class's
          ``event_type`` default is used internally for fast indexing.

        Parameters:
            event_type: The subscription key -- either a non-empty
                string (event type or ``"*"``) or a concrete
                :class:`~visioncore.eventbus.events.BaseEvent` subclass
                with a non-empty ``event_type`` default. The abstract
                :class:`BaseEvent` itself is rejected (use ``"*"`` to
                receive all events).
            callback: The callable invoked for each matching event.
                Signature: ``callback(event) -> None``. Must be callable
                and not ``None``.

        Returns:
            The created :class:`Subscriber`, which doubles as the
            unsubscribe token.

        Raises:
            ValueError: If ``event_type`` is an empty string, or a
                :class:`BaseEvent` subclass whose ``event_type`` default
                is empty/missing.
            TypeError: If ``callback`` is ``None`` or not callable, or
                ``event_type`` is neither a string nor a
                :class:`BaseEvent` subclass.

        Example:
            >>> bus = EventBus()
            >>> sub = bus.subscribe("target.created", lambda e: None)
            >>> sub.event_type
            'target.created'
            >>> from visioncore.eventbus.events import TargetLostEvent
            >>> tsub = bus.subscribe(TargetLostEvent, lambda e: None)
            >>> tsub.event_class is TargetLostEvent
            True
        """
        if callback is None or not callable(callback):
            raise TypeError(
                f"callback must be callable, got {type(callback).__name__}"
            )

        event_class: type[BaseEvent] | None = None
        if isinstance(event_type, str):
            if not event_type:
                raise ValueError(
                    f"event_type must be a non-empty string, got {event_type!r}"
                )
            key: str = event_type
        elif isinstance(event_type, type) and issubclass(event_type, BaseEvent):
            key = self._event_type_from_class(event_type)
            event_class = event_type
        else:
            raise TypeError(
                "event_type must be a non-empty str or a BaseEvent subclass, "
                f"got {type(event_type).__name__}"
            )

        with self._lock:
            sub_id: str = f"sub-{next(self._id_counter):04d}"
            sub: Subscriber = Subscriber(
                id=sub_id, event_type=key, callback=callback,
                event_class=event_class,
            )
            if key == "*" and event_class is None:
                self._wildcard_subscribers.append(sub)
            else:
                self._subscribers.setdefault(key, []).append(sub)
            self._logger.debug(
                "subscribe: id=%s event_type=%s class=%s",
                sub_id, key,
                event_class.__name__ if event_class is not None else "None",
            )
            return sub

    def unsubscribe(self, token: Subscriber | str) -> bool:
        """Remove a subscription.

        Idempotent: unsubscribing an already-removed or unknown
        subscription is a no-op and returns ``False``.

        Parameters:
            token: The :class:`Subscriber` instance returned by
                :meth:`subscribe`, or its string ``id``.

        Returns:
            ``True`` if a matching subscription was found and removed,
            ``False`` otherwise.

        Example:
            >>> bus = EventBus()
            >>> sub = bus.subscribe("x", lambda e: None)
            >>> bus.unsubscribe(sub)
            True
            >>> bus.unsubscribe(sub)        # idempotent
            False
            >>> bus.unsubscribe("sub-9999") # unknown id
            False
        """
        sub_id: str = token.id if isinstance(token, Subscriber) else token
        with self._lock:
            removed: Subscriber | None = self._remove_subscriber_by_id(sub_id)
            if removed is not None:
                self._logger.debug(
                    "unsubscribe: id=%s event_type=%s",
                    sub_id, removed.event_type,
                )
                return True
            self._logger.debug(
                "unsubscribe: id=%s not found (no-op)", sub_id,
            )
            return False

    def publish(self, event: "Event") -> int:
        """Publish ``event`` to all matching subscribers.

        Delivery happens on the calling thread, after the bus lock is
        released. The candidate subscriber list is snapshotted under the
        lock (indexed by the event's ``event_type``), then each
        candidate is re-checked with :meth:`Subscriber.matches` so that
        class subscriptions apply their ``isinstance`` filter. This means
        a class subscription for ``TargetLostEvent`` will not fire for a
        plain :class:`~visioncore.core.event.Event` even if both carry
        the same ``event_type`` string.

        Subscriptions added or removed during dispatch do not affect the
        in-flight delivery (a subscriber unsubscribed mid-dispatch is
        skipped via its ``active`` flag).

        Parameters:
            event: The event to publish. Any object with an
                ``event_type`` attribute is accepted; typed
                :class:`~visioncore.eventbus.events.BaseEvent` instances
                additionally satisfy class subscriptions.

        Returns:
            The number of subscribers whose callback was actually invoked
            (excluding inactive / already-unsubscribed ones and those
            filtered out by ``matches``).

        Example:
            >>> bus = EventBus()
            >>> bus.publish(__import__("visioncore.core.event",
            ...                        fromlist=["Event"]).Event("x", 0.0))
            0
        """
        with self._lock:
            et: object = getattr(event, "event_type", None)
            typed: list[Subscriber] = (
                self._subscribers.get(et, []) if isinstance(et, str) else []
            )
            # Snapshot the candidate list, then apply per-subscriber
            # matching (isinstance for class subs, already-satisfied for
            # string subs). A fresh list is built so iteration is immune
            # to concurrent subscribe/unsubscribe.
            candidates: list[Subscriber] = self._wildcard_subscribers + typed
            matching: list[Subscriber] = [
                s for s in candidates if s.matches(event)
            ]
        self._logger.debug(
            "publish: event_type=%s timestamp=%s candidates=%d matching=%d",
            et, getattr(event, "timestamp", "?"),
            len(candidates), len(matching),
        )
        return self._dispatcher.dispatch(matching, event)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def subscriber_count(
        self, event_type: "str | type[BaseEvent] | None" = None,
    ) -> int:
        """Return the number of registered subscribers.

        Parameters:
            event_type: If given, count only subscribers registered for
                that key -- either a string event type (``"*"`` counts
                wildcard subscribers) or a :class:`BaseEvent` subclass
                (counts subscribers for that class's ``event_type``
                string, both class and string subscriptions). If
                ``None``, count all subscribers across all event types.

        Returns:
            The subscriber count.

        Example:
            >>> bus = EventBus()
            >>> bus.subscribe("a", lambda e: None)
            >>> bus.subscribe("a", lambda e: None)
            >>> bus.subscribe("b", lambda e: None)
            >>> bus.subscribe("*", lambda e: None)
            >>> bus.subscriber_count()
            4
            >>> bus.subscriber_count("a")
            2
            >>> bus.subscriber_count("*")
            1
        """
        with self._lock:
            if event_type is None:
                total: int = len(self._wildcard_subscribers)
                total += sum(len(v) for v in self._subscribers.values())
                return total
            if isinstance(event_type, type) and issubclass(event_type, BaseEvent):
                key: str = self._event_type_from_class(event_type)
            elif isinstance(event_type, str):
                key = event_type
            else:
                raise TypeError(
                    "event_type must be str, BaseEvent subclass, or None, "
                    f"got {type(event_type).__name__}"
                )
            if key == "*":
                return len(self._wildcard_subscribers)
            return len(self._subscribers.get(key, []))

    def clear(self) -> int:
        """Remove all subscriptions.

        Returns:
            The number of subscriptions that were removed.

        Example:
            >>> bus = EventBus()
            >>> bus.subscribe("a", lambda e: None)
            >>> bus.clear()
            1
            >>> bus.subscriber_count()
            0
        """
        with self._lock:
            total: int = len(self._wildcard_subscribers)
            total += sum(len(v) for v in self._subscribers.values())
            # Mark every subscriber inactive so any externally held
            # reference becomes a no-op if dispatched.
            for sub in self._wildcard_subscribers:
                sub.active = False
            for subs in self._subscribers.values():
                for sub in subs:
                    sub.active = False
            self._wildcard_subscribers.clear()
            self._subscribers.clear()
            self._logger.info("clear: removed %d subscription(s)", total)
            return total

    # ------------------------------------------------------------------
    # Internal helpers (must be called under self._lock)
    # ------------------------------------------------------------------

    @staticmethod
    def _event_type_from_class(cls: type[BaseEvent]) -> str:
        """Derive the ``event_type`` string from a :class:`BaseEvent` subclass.

        Reads the default value of the ``event_type`` dataclass field on
        ``cls``. This is the string the bus uses to index class
        subscriptions for fast lookup.

        Parameters:
            cls: A concrete :class:`BaseEvent` subclass whose
                ``event_type`` field has a non-empty default value.

        Returns:
            The event-type string (e.g. ``"target.lost"``).

        Raises:
            ValueError: If ``cls`` has no ``event_type`` field or its
                default is missing/empty (e.g. the abstract
                :class:`BaseEvent` itself, whose default is ``""``).
        """
        for f in dataclasses.fields(cls):
            if f.name == "event_type":
                if f.default is dataclasses.MISSING or not f.default:
                    raise ValueError(
                        f"{cls.__name__} has no usable event_type default "
                        f"(empty or missing); subscribe to a concrete "
                        f"subclass or use the '*' wildcard"
                    )
                return f.default
        raise ValueError(f"{cls.__name__} has no event_type field")

    def _remove_subscriber_by_id(self, sub_id: str) -> Subscriber | None:
        """Remove and return the subscriber with ``sub_id``.

        Searches the wildcard list first, then every typed list. On
        removal the subscriber's ``active`` flag is flipped to ``False``
        so any snapshot captured by an in-flight dispatch skips it. Empty
        typed lists are deleted to avoid unbounded dict growth.

        Parameters:
            sub_id: The subscriber id to locate.

        Returns:
            The removed Subscriber, or ``None`` if not found.
        """
        # Wildcard list.
        for i, sub in enumerate(self._wildcard_subscribers):
            if sub.id == sub_id:
                removed: Subscriber = self._wildcard_subscribers.pop(i)
                removed.active = False
                return removed
        # Typed lists.
        for evt_type, subs in self._subscribers.items():
            for i, sub in enumerate(subs):
                if sub.id == sub_id:
                    removed = subs.pop(i)
                    removed.active = False
                    if not subs:
                        del self._subscribers[evt_type]
                    return removed
        return None

    # ------------------------------------------------------------------
    # Convenience dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the total number of registered subscribers."""
        return self.subscriber_count()

    def __repr__(self) -> str:
        """Return a concise representation showing subscriber counts."""
        with self._lock:
            type_count: int = len(self._subscribers)
            typed_total: int = sum(len(v) for v in self._subscribers.values())
            wildcard: int = len(self._wildcard_subscribers)
        return (
            f"EventBus(types={type_count}, typed_subscribers={typed_total}, "
            f"wildcard_subscribers={wildcard})"
        )
