"""EventStage -- the event-emission sink stage for the VisionCore pipeline.

This module defines three cohesive concerns, all in one file per the C6
specification:

* :class:`EventBus` -- the abstract event-bus **interface** that
  :class:`EventStage` depends on. Concrete buses (the existing
  :class:`visioncore.eventbus.EventBus`, or :class:`DummyEventBus` below)
  implement this interface.
* :class:`DummyEventBus` -- a test double that records every published
  event for assertion, with no real subscriber dispatch.
* :class:`EventStage` -- a :class:`~visioncore.pipeline.base.PipelineStage`
  that reads ``context.target_states``, generates one
  :class:`~visioncore.core.event.Event` per snapshot, and publishes each
  to the wrapped :class:`EventBus`.

Scope (Milestone C6)
--------------------
C6 delivers **only** the event stage + its bus interface + a test double.
It deliberately does **not** contain:

* any network / UDP / socket code -- the stage publishes to an in-process
  bus, never to a wire transport;
* any Protocol layer dependency (``visioncore.protocol``) -- the stage
  emits :class:`~visioncore.core.event.Event` objects to an
  :class:`EventBus`, not :class:`~visioncore.state.target_state.TargetState`
  snapshots to a :class:`~visioncore.protocol.base.ProtocolAdapter`;
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Sink stage (terminal)
---------------------
Unlike :class:`~visioncore.pipeline.stages.detector_stage.DetectorStage`,
:class:`~visioncore.pipeline.stages.tracker_stage.TrackerStage`, and
:class:`~visioncore.pipeline.stages.target_stage.TargetStage` (which read
one context field and write another), :class:`EventStage` is a **sink**:
it reads ``context.target_states`` and emits events, but writes **nothing**
back to the context. It is the terminal output stage of the pipeline's
event side. (The protocol / transport side -- shipping snapshots over UDP
to an external consumer -- is the job of a ProtocolAdapter wired in a
future milestone, not this stage.)

Naming note
-----------
The ``EventBus`` ABC defined here is the **pipeline-stage interface** (the
minimal ``publish`` contract). A concrete, full-featured ``EventBus``
already exists in :mod:`visioncore.eventbus` (with subscribe / unsubscribe
/ typed events / dispatcher); it conforms to this interface (it has
``publish``) and is adapted in a future milestone. The two share a name but
live in different modules -- use module-qualified imports::

    from visioncore.pipeline.stages.event_stage import EventBus  # ABC
    from visioncore.eventbus import EventBus as ConcreteEventBus

Data flow
---------
::

    context.target_states ──> EventStage.process
                                   │
                          for each snapshot:
                              Event(event_type, snapshot.timestamp,
                                    payload={"snapshot": snapshot})
                                   │
                                   ▼
                              bus.publish(event)

One event per snapshot, per process call. The event ``timestamp`` is taken
from the snapshot (so events are temporally consistent with the data they
describe, not with the wall-clock of the publish call). The snapshot
object is carried directly in the payload -- it is frozen (immutable), so
sharing the reference is safe.

Thread safety
-------------
:class:`EventStage` is not thread-safe. It is designed for the C1
pipeline's serial execution model. Concrete buses that share state across
threads must serialise access externally.

Example
-------
    >>> from visioncore.pipeline.stages.event_stage import (
    ...     EventStage, DummyEventBus,
    ... )
    >>> from visioncore.pipeline.context import PipelineContext
    >>> from visioncore.state.target_state import TargetState
    >>> bus = DummyEventBus()
    >>> stage = EventStage(bus)
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty()
    >>> ctx.target_states.append(TargetState(
    ...     1, 1, None, "person", 0.9, 0.5, 0.5, 0.0, 0.0,
    ...     0.2, 0.4, 1.0, 0, {}))
    >>> stage.process(ctx)
    >>> len(bus.published)
    1
    >>> bus.published[0].event_type
    'target.snapshot'
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from visioncore.core.event import Event
from visioncore.pipeline.base import PipelineStage

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = ["EventBus", "DummyEventBus", "EventStage"]


logger = logging.getLogger(__name__)


# ======================================================================
# EventBus interface (abstract)
# ======================================================================

class EventBus(ABC):
    """Abstract event-bus interface -- the publish contract.

    A :class:`EventBus` is a sink for :class:`~visioncore.core.event.Event`
    objects. Producers (like :class:`EventStage`) call :meth:`publish`;
    the bus routes the event to its subscribers (if any). The interface is
    deliberately minimal -- it exposes only :meth:`publish` (abstract) and
    :meth:`health_check` (concrete default). Subscribe / unsubscribe /
    typed-event machinery lives on the concrete
    :class:`visioncore.eventbus.EventBus`, not on this contract.

    Naming note
    -----------
    This is the **pipeline-stage interface**. A concrete, full-featured
    ``EventBus`` already exists in :mod:`visioncore.eventbus` (subscribe /
    unsubscribe / typed events / dispatcher). It conforms to this interface
    (it has ``publish``) and is the production bus. Use module-qualified
    imports to disambiguate.

    Subclassing
    -----------
    Subclasses **must** implement :meth:`publish`. :meth:`health_check` has
    a concrete default (returns ``True``) and may be overridden.
    """

    __slots__ = ()

    @abstractmethod
    def publish(self, event: Event) -> int:
        """Publish ``event`` to all matching subscribers.

        Parameters:
            event: The :class:`~visioncore.core.event.Event` to publish.
                Any object with ``event_type`` and ``timestamp``
                attributes is accepted (duck typing), matching the
                concrete bus's contract.

        Returns:
            The number of subscribers whose callback was actually invoked.
            A bus with no subscribers returns ``0``.

        Raises:
            Exception: Any exception from the bus's internal dispatch.
                The stage does not catch bus exceptions -- they propagate
                out of ``process()`` per the C1 "honest exceptions"
                contract.
        """

    def health_check(self) -> bool:
        """Return ``True`` iff the bus is ready to accept publishes.

        Concrete default: always ``True`` (an in-process bus is ready once
        constructed). Override for buses that may become unhealthy (e.g.
        a bus backed by a queue that can fill up). Must not raise; on
        internal error return ``False``.
        """
        return True


# ======================================================================
# DummyEventBus -- test double
# ======================================================================

class DummyEventBus(EventBus):
    """Test event-bus that records every published event, no dispatch.

    :class:`DummyEventBus` does no real subscriber dispatch -- it simply
    appends each published event to a list, making the bus's behaviour
    **observable** in tests. Use it to assert exactly which events an
    :class:`EventStage` emitted, in what order, with what payloads.

    What it is for
    --------------
    * **Stage testing** -- verify :class:`EventStage` emits the right
      events without wiring a real bus with subscribers.
    * **Pipeline integration** -- capture the event stream a pipeline
      produces for offline assertion.
    * **Benchmarks** -- measure publish overhead with zero dispatch cost.

    Attributes:
        published: The list of events published so far, in publish order.
            Each element is the exact :class:`~visioncore.core.event.Event`
            object passed to :meth:`publish` (not a copy).
        raise_on_publish: An exception instance to raise on the next
            :meth:`publish` call, or ``None`` (default) to publish
            normally. Persists until cleared. Used by exception-propagation
            tests.
        publish_count: Number of times :meth:`publish` was called
            (including calls that raised).

    Example:
        >>> bus = DummyEventBus()
        >>> from visioncore.core.event import Event
        >>> bus.publish(Event("x", 0.0))
        1
        >>> len(bus.published)
        1
        >>> bus.published[0].event_type
        'x'
    """

    __slots__ = ("published", "raise_on_publish", "publish_count")

    def __init__(
        self, *, raise_on_publish: BaseException | None = None,
    ) -> None:
        """Initialise an empty recording bus.

        Parameters:
            raise_on_publish: An exception instance to raise on the next
                :meth:`publish` call, or ``None`` (default) to publish
                normally.
        """
        self.published: list[Event] = []
        self.raise_on_publish: BaseException | None = raise_on_publish
        self.publish_count: int = 0
        logger.debug("DummyEventBus created")

    # ------------------------------------------------------------------
    # EventBus contract
    # ------------------------------------------------------------------

    def publish(self, event: Event) -> int:
        """Record ``event`` in :attr:`published`; return 1.

        If :attr:`raise_on_publish` is set, raise it instead (after
        incrementing :attr:`publish_count` but before recording).
        """
        self.publish_count += 1
        if self.raise_on_publish is not None:
            logger.debug("DummyEventBus.publish: RAISING %s",
                         type(self.raise_on_publish).__name__)
            raise self.raise_on_publish
        self.published.append(event)
        logger.debug("DummyEventBus.publish: count=%d total=%d",
                     self.publish_count, len(self.published))
        return 1

    def health_check(self) -> bool:
        """Always ``True`` -- a recording bus is always ready."""
        return True

    # ------------------------------------------------------------------
    # Test convenience
    # ------------------------------------------------------------------

    @property
    def event_count(self) -> int:
        """Number of events successfully recorded (excludes raised calls)."""
        return len(self.published)

    @property
    def event_types(self) -> list[str]:
        """The event_type strings of all recorded events, in order."""
        return [e.event_type for e in self.published]

    def reset(self) -> None:
        """Clear the recorded events and counters."""
        self.published.clear()
        self.publish_count = 0


# ======================================================================
# EventStage -- the pipeline stage
# ======================================================================

class EventStage(PipelineStage):
    """Pipeline stage that emits one Event per target_state snapshot.

    :class:`EventStage` is a **sink** stage: it reads
    ``context.target_states`` and publishes a
    :class:`~visioncore.core.event.Event` for each snapshot to the wrapped
    :class:`EventBus`, but writes **nothing** back to the context. It is
    the terminal output stage of the pipeline's event side.

    The stage is a thin adapter: it owns no event-generation logic beyond
    "one event per snapshot". The event ``event_type`` is configurable
    (default ``"target.snapshot"``); the ``timestamp`` is taken from the
    snapshot (temporal consistency with the data); the ``payload`` carries
    the snapshot object directly (frozen / immutable, safe to share).

    Data flow
    ---------
    On :meth:`process`:

    1. Iterate ``context.target_states``.
    2. For each snapshot, construct an :class:`~visioncore.core.event.Event`
       with ``event_type=self._event_type``,
       ``timestamp=snapshot.timestamp``, and
       ``payload={"snapshot": snapshot}``.
    3. Call ``bus.publish(event)`` for each.

    The stage does **not** skip when ``target_states`` is empty -- it
    simply publishes zero events (a legitimate "nothing to emit this
    frame" outcome, not an error).

    Lifecycle
    ---------
    * :meth:`initialize` -- no-op (the bus is ready on construction; no
      resources to acquire).
    * :meth:`process` -- generate + publish events (see above).
    * :meth:`shutdown` -- no-op (the bus is not owned by the stage; the
      caller manages the bus's lifetime).
    * :meth:`health_check` -- delegates to ``bus.health_check()``.

    Attributes:
        _bus: The wrapped :class:`EventBus` instance.
        _event_type: The event_type string stamped onto every emitted event.
        _published_count: Total events this stage has published (for
            diagnostics; survives across process calls).

    Example:
        >>> stage = EventStage(DummyEventBus(), name="events")
        >>> stage.name
        'events'
    """

    __slots__ = ("_bus", "_event_type", "_published_count")

    def __init__(
        self,
        event_bus: EventBus,
        *,
        name: str | None = None,
        event_type: str = "target.snapshot",
    ) -> None:
        """Construct an EventStage wrapping ``event_bus``.

        Parameters:
            event_bus: The :class:`EventBus` instance to publish to. Must
                not be ``None`` and must be an :class:`EventBus` (the
                interface is enforced). This is the dependency-inversion
                point: the stage depends on the interface, the concrete
                bus is injected.
            name: Optional stage name (defaults to ``"EventStage"``).
            event_type: The ``event_type`` string stamped onto every
                emitted event (default ``"target.snapshot"``). Configurable
                so different stages can emit different event kinds (e.g.
                ``"target.active"``, ``"target.lost"``).

        Raises:
            TypeError: If ``event_bus`` is ``None`` or not an
                :class:`EventBus`.
            ValueError: If ``event_type`` is empty.
        """
        super().__init__(name=name if name is not None else "EventStage")
        if event_bus is None or not isinstance(event_bus, EventBus):
            raise TypeError(
                f"event_bus must be an EventBus instance, got "
                f"{type(event_bus).__name__ if event_bus is not None else 'None'}"
            )
        if not event_type:
            raise ValueError("event_type must be a non-empty string")
        self._bus: EventBus = event_bus
        self._event_type: str = event_type
        self._published_count: int = 0
        logger.debug("EventStage created: name=%s event_type=%s",
                     self._name, self._event_type)

    # ------------------------------------------------------------------
    # PipelineStage lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """No-op. The bus is ready on construction; nothing to acquire."""
        logger.debug("EventStage.initialize: name=%s (no-op)", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Read ``context.target_states``; publish one Event per snapshot.

        Does not modify the context (sink stage). For each snapshot, an
        :class:`~visioncore.core.event.Event` is constructed with this
        stage's ``event_type``, the snapshot's ``timestamp``, and a
        payload carrying the snapshot object, then published to the bus.

        An empty ``target_states`` list publishes zero events (legitimate
        "nothing to emit", not an error).
        """
        snapshots: list[Any] = context.target_states
        for snapshot in snapshots:
            event: Event = Event(
                event_type=self._event_type,
                timestamp=snapshot.timestamp,
                payload={"snapshot": snapshot},
            )
            self._bus.publish(event)
            self._published_count += 1
        logger.debug(
            "EventStage.process: name=%s snapshots=%d published(total)=%d",
            self._name, len(snapshots), self._published_count,
        )

    def shutdown(self) -> None:
        """No-op. The bus is not owned by the stage; caller manages it."""
        logger.debug("EventStage.shutdown: name=%s (no-op)", self._name)

    def health_check(self) -> bool:
        """Delegate to ``bus.health_check()``."""
        return self._bus.health_check()

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def event_bus(self) -> EventBus:
        """The wrapped :class:`EventBus` instance."""
        return self._bus

    @property
    def event_type(self) -> str:
        """The event_type string stamped onto every emitted event."""
        return self._event_type

    @property
    def published_count(self) -> int:
        """Total events this stage has published across all process calls."""
        return self._published_count
