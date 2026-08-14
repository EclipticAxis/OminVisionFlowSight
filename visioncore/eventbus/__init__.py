"""EventBus package for VisionCore.

Provides:
    * :class:`EventBus` -- thread-safe central publish/subscribe hub.
    * :class:`Subscriber` -- subscription record (returned by
      :meth:`EventBus.subscribe`, doubles as the unsubscribe token).
    * :class:`Dispatcher` -- synchronous, exception-isolating delivery
      engine.
    * :class:`EventListener` -- type alias for the callback signature
      ``Callable[[Event], None]``.
    * Typed event classes (:class:`BaseEvent` and its subclasses) --
      frozen, slotted dataclasses mirroring the Target lifecycle, ready
      to publish on the bus.

Scope (Milestone 3):
    This package delivers the **core** EventBus infrastructure plus a
    typed event hierarchy. It is not yet wired into InferWorker, Tracker,
    the GUI, or TargetManager; those integrations are subsequent
    milestones and are out of scope here. No existing business logic is
    modified.

Design summary:
    * Producers call :meth:`EventBus.publish` with any object exposing
      ``event_type`` and ``timestamp`` attributes. Both the lightweight
      :class:`~visioncore.core.event.Event` and the typed
      :class:`BaseEvent` subclasses satisfy this contract (duck typing),
      so the bus accepts either without modification.
    * Consumers call :meth:`EventBus.subscribe` with an event type (or
      the ``"*"`` wildcard) and a callback, receiving a
      :class:`Subscriber` token.
    * The bus routes each published event to every matching subscriber,
      in subscription order, on the publishing thread.
    * Callback exceptions are logged and swallowed so one failing
      subscriber cannot abort delivery to the rest.
    * All registry mutations are guarded by a re-entrant lock; dispatch
      itself runs outside the lock so callbacks may safely re-enter
      the bus.

Quick start::

    from visioncore.eventbus import EventBus, TargetCreatedEvent

    bus = EventBus()
    received = []
    bus.subscribe("target.created", received.append)
    bus.publish(TargetCreatedEvent(
        event_id="evt-1", timestamp=1.0,
        target_id="S0-T0001", slot_id=0,
        payload={"track_id": 1},
    ))
"""

from __future__ import annotations

from visioncore.eventbus.bus import EventBus
from visioncore.eventbus.debug_logger import DebugEventLogger
from visioncore.eventbus.dispatcher import Dispatcher
from visioncore.eventbus.events import (
    BaseEvent,
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)
from visioncore.eventbus.subscriber import EventListener, Subscriber

__all__ = [
    # ---- Core bus infrastructure ----
    "EventBus",
    "Subscriber",
    "Dispatcher",
    "EventListener",
    # ---- Typed events ----
    "BaseEvent",
    "TargetCreatedEvent",
    "TargetLostEvent",
    "TargetRecoveredEvent",
    "TargetLockedEvent",
    "TargetRemovedEvent",
    # ---- Debug / observation ----
    "DebugEventLogger",
]
