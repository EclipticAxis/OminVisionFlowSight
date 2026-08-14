"""Dispatcher for the VisionCore EventBus.

The Dispatcher is the delivery engine: given a list of matching Subscribers
and an Event, it invokes each subscriber's callback in subscription order,
isolating exceptions so one failing subscriber cannot starve the rest.

The current implementation is synchronous: callbacks run on the publishing
thread, one after another. This is the correct baseline for an event bus
that must preserve causal ordering (a callback that publishes a follow-up
event sees it processed inline before control returns to the original
publisher). Asynchronous / thread-pooled dispatch is deliberately out of
scope for this milestone -- the Dispatcher interface is shaped so a future
``AsyncDispatcher`` could be dropped in without touching the EventBus.

Thread safety:
    The Dispatcher is stateless apart from its logger; a single shared
    instance is held by the EventBus. :meth:`dispatch` performs no locking
    of its own -- the EventBus hands it an already-snapshotted subscriber
    list, and per-subscriber exception isolation is handled by
    :meth:`Subscriber.deliver`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visioncore.core.event import Event
    from visioncore.eventbus.subscriber import Subscriber


class Dispatcher:
    """Synchronous, exception-isolating event dispatcher.

    The dispatcher is stateless; a single shared instance is held by the
    EventBus. :meth:`dispatch` is the only public entry point.

    Design notes:
        * Subscribers are invoked in subscription (insertion) order. This
          gives deterministic delivery, which matters for tests and for
          ordered side-effects (e.g. a logging subscriber should observe
          an event before an alerting subscriber reacts to it).
        * Exceptions raised by a callback are logged and swallowed. The
          dispatcher never re-raises; an event bus whose delivery can be
          sabotaged by a single buggy subscriber is not robust.
        * The ``subscribers`` list is treated as read-only; the dispatcher
          does not mutate it. The EventBus passes a snapshot so that
          mutations during iteration (e.g. a callback unsubscribing a
          peer) cannot corrupt the loop.

    Example:
        >>> from visioncore.eventbus.dispatcher import Dispatcher
        >>> from visioncore.eventbus.subscriber import Subscriber
        >>> from visioncore.core.event import Event
        >>> d = Dispatcher()
        >>> seen = []
        >>> s = Subscriber(id="s1", event_type="x", callback=seen.append)
        >>> d.dispatch([s], Event("x", 0.0))
        1
        >>> seen[0].event_type
        'x'
    """

    __slots__ = ("_logger",)

    def __init__(self) -> None:
        """Initialise the dispatcher with a module-level logger."""
        self._logger: logging.Logger = logging.getLogger(__name__)

    def dispatch(self, subscribers: "list[Subscriber]", event: "Event") -> int:
        """Deliver ``event`` to each subscriber in order.

        Parameters:
            subscribers: The list of matching subscribers (already filtered
                by the EventBus). Treated as read-only; the caller is
                expected to pass a snapshot so that mutations during
                iteration (e.g. a callback unsubscribing) do not corrupt
                the loop.
            event: The Event to deliver.

        Returns:
            The number of subscribers whose callback was actually invoked
            (i.e. excluding inactive / already-unsubscribed ones).
        """
        invoked: int = 0
        for sub in subscribers:
            if sub.deliver(event, self._logger):
                invoked += 1
        return invoked
