"""Subscriber abstraction for the VisionCore EventBus.

A Subscriber binds a user-supplied callback to a single subscription,
which may be expressed either as an event-type string or as a typed
event class. It is the atomic unit of dispatch: when an event is
published, the EventBus hands the event to each matching Subscriber,
which invokes its callback in turn.

Subscribers are created exclusively by :meth:`EventBus.subscribe` -- they
are not intended to be constructed directly by application code. Each
Subscriber carries a unique, monotonically increasing id (assigned by the
bus) that doubles as the unsubscribe token.

Matching modes:
    * **String subscription** (``event_class is None``): the subscriber
      matches any published object whose ``event_type`` attribute equals
      the subscriber's ``event_type`` string, or matches everything when
      ``event_type == "*"`` (wildcard). This is the backward-compatible
      mode and works with both :class:`~visioncore.core.event.Event` and
      typed :class:`~visioncore.eventbus.events.BaseEvent` instances.
    * **Class subscription** (``event_class is not None``): the subscriber
      matches only when the published object is an ``isinstance`` of the
      registered class. This provides type-safe filtering -- a subscriber
      for ``TargetLostEvent`` will *not* receive a plain
      :class:`~visioncore.core.event.Event` even if its ``event_type``
      string happens to coincide.

Thread safety:
    Subscriber instances are effectively immutable after construction
    except for the ``active`` flag, which is flipped to ``False`` by
    :meth:`EventBus.unsubscribe`. That flag is read during dispatch, so a
    subscriber that is unsubscribed while a dispatch is in flight is
    skipped cleanly rather than delivering to a half-torn-down callback.
    The callback itself runs on the publishing thread; callbacks that
    touch shared state must perform their own synchronisation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from visioncore.core.event import Event
    from visioncore.eventbus.events import BaseEvent


# Type alias for the user-supplied event callback. Takes a single event
# object and returns nothing. Kept as a plain alias (not a Protocol) so
# that any callable with a compatible signature -- function, lambda,
# bound method, or a custom __call__ object -- is accepted without
# runtime checks.
EventListener = Callable[["Event"], None]


@dataclass(slots=True)
class Subscriber:
    """A single subscription on the EventBus.

    Binds a user callback to an event filter. The filter is either a
    string ``event_type`` (exact match or ``"*"`` wildcard) or a typed
    event ``event_class`` (``isinstance`` match). Exactly one of the two
    filtering modes is active per subscriber:

    * ``event_class is None``  -> string / wildcard matching.
    * ``event_class is not None`` -> ``isinstance`` matching (and
      ``event_type`` holds the class's type string, used by the bus for
      fast index lookup).

    Attributes:
        id: Unique, monotonically increasing subscription identifier
            (e.g. ``"sub-0001"``), assigned by the EventBus at subscribe
            time. Serves as the unsubscribe token.
        event_type: The event-type string this subscriber listens for,
            or ``"*"`` for all events. For class subscriptions this
            holds the class's ``event_type`` constant (e.g.
            ``"target.lost"``), used internally for indexing -- it is
            not used for matching when ``event_class`` is set.
        callback: The callable invoked when a matching event is
            published. Signature: ``callback(event: Event) -> None``.
        event_class: When not ``None``, this subscriber matches only
            events that are ``isinstance`` of this class (a
            :class:`~visioncore.eventbus.events.BaseEvent` subclass).
            When ``None``, string matching on ``event_type`` is used
            instead. Defaults to ``None`` (backward-compatible string
            mode).
        active: Whether the subscriber is still registered. Set to
            ``False`` by :meth:`EventBus.unsubscribe` so that stale
            references retained by the caller, or subscribers captured
            in an in-flight dispatch snapshot, become no-ops.

    Note:
        Instances are created by the EventBus, not by application code.
        The ``active`` flag makes unsubscribe idempotent and safe to call
        from within a callback that is currently being dispatched.
    """

    id: str
    event_type: str
    callback: EventListener
    event_class: "type[BaseEvent] | None" = None
    active: bool = True

    def matches(self, event: object) -> bool:
        """Return ``True`` if this subscriber should receive ``event``.

        Matching logic (first applicable rule wins):

        1. **Class subscription** (``event_class is not None``): return
           ``isinstance(event, event_class)``. This is type-safe -- only
           instances of the registered class (or its subclasses) match,
           regardless of ``event_type`` strings.
        2. **Wildcard** (``event_type == "*"``): return ``True`` -- the
           subscriber receives every published event.
        3. **String match**: return ``event_type == event.event_type``.
           The published object's ``event_type`` attribute is compared
           for string equality. Objects without an ``event_type``
           attribute never match.

        Parameters:
            event: The published event object.

        Returns:
            ``True`` if the subscriber should receive the event.

        Example:
            >>> s = Subscriber(id="sub-0001", event_type="target.lost",
            ...                callback=lambda e: None)
            >>> class E:
            ...     event_type = "target.lost"
            >>> s.matches(E())
            True
            >>> s.matches(type("X", (), {"event_type": "other"})())
            False
            >>> w = Subscriber(id="sub-0002", event_type="*",
            ...                callback=lambda e: None)
            >>> w.matches(object())
            True
        """
        if self.event_class is not None:
            return isinstance(event, self.event_class)
        if self.event_type == "*":
            return True
        return self.event_type == getattr(event, "event_type", None)

    def deliver(self, event: "Event", logger: logging.Logger) -> bool:
        """Invoke the callback for ``event``, isolating exceptions.

        The callback runs on the current (publishing) thread. Any
        exception raised by the callback is logged and swallowed so that
        one failing subscriber cannot abort delivery to the remaining
        subscribers.

        Inactive subscribers (already unsubscribed) are skipped silently
        and are not counted as invoked.

        Parameters:
            event: The event to deliver.
            logger: Logger used to record callback exceptions.

        Returns:
            ``True`` if the callback was actually invoked, ``False`` if
            the subscriber was inactive and skipped.
        """
        if not self.active:
            return False
        try:
            self.callback(event)
        except Exception:  # noqa: BLE001 -- intentional broad capture
            logger.exception(
                "Subscriber %s callback raised for event_type=%s",
                self.id, getattr(event, "event_type", "<unknown>"),
            )
        return True
