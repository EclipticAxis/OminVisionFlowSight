"""Typed event classes for the VisionCore EventBus.

This module defines a small, closed hierarchy of typed events that flow
through :class:`~visioncore.eventbus.bus.EventBus`. Each event class is a
frozen, slotted dataclass carrying the common fields (``event_id``,
``timestamp``, ``target_id``, ``slot_id``, ``event_type``) plus an
extensible ``payload`` dict.

The hierarchy mirrors the :class:`~visioncore.core.target.TargetState`
lifecycle: every state transition that a Target can undergo has a
corresponding event class. Producers (typically the TargetManager, in a
later milestone) instantiate these events and publish them; consumers
(loggers, alerting, UI, persistence) subscribe by ``event_type``.

Design:
    * ``BaseEvent`` holds the common fields and a ``__repr__``. It is not
      meant to be instantiated directly -- use one of the concrete
      subclasses.
    * Each concrete subclass fixes its ``event_type`` via a default value,
      so callers normally omit it::

          ev = TargetCreatedEvent(
              event_id="evt-0001", timestamp=1.0,
              target_id="S0-T0001", slot_id=0,
              payload={"track_id": 1},
          )
          ev.event_type  # -> "target.created"

    * All events are frozen + slotted: hashable (when payload is hashable),
      memory-compact, and safe to share across threads -- consistent with
      :mod:`visioncore.core.event`.

Event type strings:
    ``target.created``   -- TargetCreatedEvent
    ``target.lost``      -- TargetLostEvent
    ``target.recovered`` -- TargetRecoveredEvent
    ``target.locked``    -- TargetLockedEvent
    ``target.removed``   -- TargetRemovedEvent

Lifecycle correspondence::

    (new)        --created-->   ACTIVE      [TargetCreatedEvent]
    ACTIVE       --lost------>  LOST        [TargetLostEvent]
    LOST         --recovered>   RECOVERED   [TargetRecoveredEvent]
    ACTIVE       --locked---->  LOCKED      [TargetLockedEvent]
    LOCKED       --released-->  ACTIVE      (no dedicated event; reuse
                                             TargetCreatedEvent? no -- release
                                             is a state change, see note)
    any          --removed----> REMOVED      [TargetRemovedEvent]

Note:
    The ``payload`` dict is technically mutable (Python has no frozen
    dict); by convention it is treated as read-only after construction,
    exactly as in :class:`~visioncore.core.event.Event`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True, repr=False)
class BaseEvent:
    """Abstract base for all VisionCore typed events.

    Carries the fields common to every event that concerns a Target.
    Concrete subclasses (:class:`TargetCreatedEvent`, ...) fix
    ``event_type`` to a constant string and are the ones actually
    instantiated. ``BaseEvent`` is not intended to be used directly.

    Attributes:
        event_id: Stable, globally unique identifier for this event
            instance (e.g. a UUID4 hex string or a monotonic counter).
            Producers are responsible for generation; this module does
            not assign ids.
        timestamp: Wall-clock or monotonic timestamp in seconds when the
            event occurred. Used for temporal ordering and latency
            measurement.
        target_id: The ``target_id`` of the Target this event concerns
            (e.g. ``"S0-T0001"``). Distinct from track_id.
        slot_id: Camera slot [0, 3] the Target belongs to. Enables
            multi-camera demultiplexing on the consumer side.
        event_type: A short, dot-separated string identifying the event
            kind (e.g. ``"target.created"``). Fixed per concrete subclass;
            callers normally do not pass it explicitly.
        payload: Structured, event-specific data. The schema is determined
            by ``event_type``; consumers negotiate expected keys. Defaults
            to an empty dict. Treated as read-only by convention.

    Example:
        >>> ev = TargetCreatedEvent(
        ...     event_id="evt-1", timestamp=1.0,
        ...     target_id="S0-T0001", slot_id=0,
        ...     payload={"track_id": 1},
        ... )
        >>> ev.event_type
        'target.created'
        >>> ev.target_id
        'S0-T0001'
    """

    event_id: str
    timestamp: float
    target_id: str
    slot_id: int
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return a concise, human-readable representation.

        The payload is summarised by its keys to keep the representation
        compact for large payloads. The concrete class name is used so
        subclasses are distinguishable in logs.
        """
        if self.payload:
            payload_keys: str = (
                "{" + ", ".join(repr(k) for k in self.payload) + "}"
            )
        else:
            payload_keys = "{}"
        return (
            f"{type(self).__name__}("
            f"event_id={self.event_id!r}, "
            f"event_type={self.event_type!r}, "
            f"timestamp={self.timestamp:.4f}, "
            f"target_id={self.target_id!r}, "
            f"slot_id={self.slot_id!r}, "
            f"payload_keys={payload_keys})"
        )


@dataclass(frozen=True, slots=True, repr=False)
class TargetCreatedEvent(BaseEvent):
    """Emitted when a new Target enters the system (-> ACTIVE).

    Corresponds to the ``TargetManager.create_target`` path: a fresh
    Target is registered with an initial Track and becomes ACTIVE.

    Typical payload:
        track_id (int): The initial track id associated with the target.
        priority (int): Initial processing priority [0, 255].
        detection (dict): Snapshot of the originating detection, if any.

    Attributes:
        event_type: Always ``"target.created"``.

    Example:
        >>> ev = TargetCreatedEvent(
        ...     event_id="e1", timestamp=0.0,
        ...     target_id="S0-T0001", slot_id=0,
        ... )
        >>> ev.event_type
        'target.created'
    """

    event_type: str = "target.created"


@dataclass(frozen=True, slots=True, repr=False)
class TargetLostEvent(BaseEvent):
    """Emitted when an ACTIVE Target's track exceeds the miss threshold.

    Corresponds to the ``ACTIVE -> LOST`` transition in
    :class:`~visioncore.core.target.TargetState`.

    Typical payload:
        last_seen (float): Timestamp of the last successful association.
        missed_frames (int): Consecutive frames without association.

    Attributes:
        event_type: Always ``"target.lost"``.
    """

    event_type: str = "target.lost"


@dataclass(frozen=True, slots=True, repr=False)
class TargetRecoveredEvent(BaseEvent):
    """Emitted when a LOST Target is re-associated with a new detection.

    Corresponds to the ``LOST -> RECOVERED`` transition. A subsequent
    :class:`TargetCreatedEvent` is not emitted -- the same target_id is
    retained; RECOVERED transitions back to ACTIVE on the next update.

    Typical payload:
        new_track_id (int): The track id that re-anchored the target.
        recovery_latency (float): Seconds lost -> recovered.

    Attributes:
        event_type: Always ``"target.recovered"``.
    """

    event_type: str = "target.recovered"


@dataclass(frozen=True, slots=True, repr=False)
class TargetLockedEvent(BaseEvent):
    """Emitted when a Target is selected for exclusive attention (-> LOCKED).

    Corresponds to the ``ACTIVE -> LOCKED`` transition (operator or
    programmatic lock). Locked targets receive priority processing.

    Typical payload:
        locked_by (str): Who/what issued the lock ("operator", "auto", ...).
        reason (str): Free-text reason for the lock.

    Attributes:
        event_type: Always ``"target.locked"``.
    """

    event_type: str = "target.locked"


@dataclass(frozen=True, slots=True, repr=False)
class TargetRemovedEvent(BaseEvent):
    """Emitted when a Target is permanently removed (-> REMOVED, terminal).

    Corresponds to the ``any -> REMOVED`` transition. REMOVED is
    terminal: no further events are emitted for this target_id.

    Typical payload:
        reason (str): Why the target was removed ("timeout", "manual", ...).
        final_state (str): The state the target was in before removal.

    Attributes:
        event_type: Always ``"target.removed"``.
    """

    event_type: str = "target.removed"


__all__ = [
    "BaseEvent",
    "TargetCreatedEvent",
    "TargetLostEvent",
    "TargetRecoveredEvent",
    "TargetLockedEvent",
    "TargetRemovedEvent",
]
