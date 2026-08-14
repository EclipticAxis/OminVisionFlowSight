"""StatePublisher -- bridges EventBus lifecycle events to a ProtocolAdapter.

Defines :class:`StatePublisher`, which subscribes to the five Target
lifecycle events on the :class:`~visioncore.eventbus.bus.EventBus`,
converts each event into a
:class:`~visioncore.state.target_state.TargetState` snapshot, and
publishes the snapshot via an injected
:class:`~visioncore.protocol.base.ProtocolAdapter`.

This module is the "glue" between the event-driven Target lifecycle
(EventBus) and the transport-driven snapshot stream (ProtocolAdapter).
It allows downstream consumers (UDP receivers, file loggers, console
monitors) to observe Target state changes without coupling to the
EventBus directly.

Architecture
------------
::

    TargetManager ──(lifecycle transition)──> EventBus
                                                    │
                                          StatePublisher subscribes
                                                    │
                                                    ▼
                                            TargetState (snapshot)
                                                    │
                                          ProtocolAdapter.publish()
                                                    │
                                                    ▼
                                        UDP / Console / File / ...

No modification of GUI or InferWorker
-------------------------------------
StatePublisher is a pure observer. It subscribes to events that are
already being published by the TargetManager (via the EventBus injected
into InferWorker in the B2 shadow integration). It does NOT modify:

* ``gui/`` -- no GUI files are touched
* ``ai/inference.py`` (InferWorker) -- no inference files are touched
* ``visioncore/target_manager/`` -- no target manager files are touched

It only reads from the EventBus and writes to the ProtocolAdapter.

Type annotation convention
--------------------------
This module imports ``TargetState`` via its fully-qualified module path::

    from visioncore.state.target_state import TargetState

This avoids collision with the ``visioncore.core.TargetState`` enum.

Example
-------
    >>> from visioncore.eventbus import EventBus
    >>> from visioncore.protocol import NullAdapter
    >>> from visioncore.protocol.publisher import StatePublisher
    >>> bus = EventBus()
    >>> adapter = NullAdapter()
    >>> publisher = StatePublisher(bus, adapter)
    >>> publisher.start()
    >>> # events published on the bus now flow to the adapter
    >>> publisher.stop()
"""

from __future__ import annotations

import logging
import re
from typing import Any

from visioncore.eventbus.bus import EventBus
from visioncore.eventbus.events import (
    BaseEvent,
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)
from visioncore.eventbus.subscriber import Subscriber
from visioncore.protocol.base import ProtocolAdapter
from visioncore.state.target_state import TargetState


__all__ = ["StatePublisher"]


logger = logging.getLogger(__name__)


# Pattern for parsing target_id strings like "S0-T0001" -> (slot=0, seq=1).
# The format is defined by visioncore.target_manager.target_id_factory.
_TARGET_ID_RE = re.compile(r"^S(\d+)-T(\d+)$")


class StatePublisher:
    """Bridges Target lifecycle events to a ProtocolAdapter.

    Subscribes to the five Target lifecycle events on an EventBus,
    converts each event to a :class:`TargetState` snapshot, and publishes
    the snapshot via the injected :class:`ProtocolAdapter`.

    The publisher is a passive observer -- it does not modify the
    EventBus, the TargetManager, or any Target. It only reads events and
    writes to the adapter. If the adapter's ``publish()`` raises, the
    error is caught and logged so that a single transport failure does
    not crash the event dispatch loop.

    Lifecycle
    ---------
    * :meth:`start` -- connects the adapter and subscribes to all five
      event types on the bus. Idempotent.
    * :meth:`stop` -- unsubscribes from the bus and disconnects the
      adapter. Idempotent.
    * Context manager -- ``with StatePublisher(bus, adapter) as p:``
      calls ``start()`` on enter and ``stop()`` on exit.

    Thread safety
    -------------
    StatePublisher itself is stateless aside from the subscription list.
    The EventBus handles callback dispatch safely (callbacks run on the
    publishing thread, but the bus's lock is released before dispatch).
    The adapter's thread-safety is the adapter's responsibility (see
    :class:`~visioncore.protocol.base.ProtocolAdapter`).

    Args:
        event_bus: The EventBus to subscribe to. Must be the same bus
            that the TargetManager publishes events on.
        adapter: The ProtocolAdapter to publish TargetState snapshots
            through. The publisher calls ``adapter.connect()`` on
            ``start()`` and ``adapter.disconnect()`` on ``stop()``.

    Example:
        >>> bus = EventBus()
        >>> adapter = NullAdapter()
        >>> publisher = StatePublisher(bus, adapter)
        >>> publisher.start()
        >>> # TargetManager emits TargetCreatedEvent -> publisher
        >>> # converts to TargetState -> adapter.publish(state)
        >>> publisher.stop()
    """

    def __init__(self, event_bus: EventBus, adapter: ProtocolAdapter) -> None:
        self._bus: EventBus = event_bus
        self._adapter: ProtocolAdapter = adapter
        self._subscriptions: list[Subscriber] = []
        self._started: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect the adapter and subscribe to all five event types.

        Idempotent -- calling ``start()`` on an already-started publisher
        is a no-op.

        Raises:
            Exception: If the adapter's ``connect()`` fails. In that case
                no subscriptions are registered and the publisher remains
                stopped.
        """
        if self._started:
            return

        # Connect the adapter first. If this fails, we don't want to
        # have registered subscriptions that would try to publish to a
        # disconnected adapter.
        self._adapter.connect()

        # Subscribe to each of the five lifecycle event types. Using
        # class-based subscription (not string-based) for type safety:
        # the handler receives the exact event subclass, not just any
        # event with a matching event_type string.
        event_classes: list[type[BaseEvent]] = [
            TargetCreatedEvent,
            TargetLostEvent,
            TargetRecoveredEvent,
            TargetLockedEvent,
            TargetRemovedEvent,
        ]
        for cls in event_classes:
            sub = self._bus.subscribe(cls, self._on_event)
            self._subscriptions.append(sub)

        self._started = True
        logger.info(
            "StatePublisher started: subscribed to %d event types, "
            "adapter=%s",
            len(self._subscriptions),
            type(self._adapter).__name__,
        )

    def stop(self) -> None:
        """Unsubscribe from the bus and disconnect the adapter.

        Idempotent -- safe to call on an already-stopped publisher.
        Never raises: adapter disconnect errors are caught and logged.
        """
        if not self._started:
            return

        # Unsubscribe first so no more events arrive while we're
        # tearing down.
        for sub in self._subscriptions:
            try:
                self._bus.unsubscribe(sub)
            except Exception:
                logger.warning(
                    "StatePublisher: error unsubscribing, ignoring",
                    exc_info=True,
                )
        self._subscriptions.clear()

        # Disconnect the adapter. Must not raise per the contract.
        try:
            self._adapter.disconnect()
        except Exception:
            logger.warning(
                "StatePublisher: error disconnecting adapter, ignoring",
                exc_info=True,
            )

        self._started = False
        logger.info("StatePublisher stopped")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "StatePublisher":
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Event handler (shared by all 5 event types)
    # ------------------------------------------------------------------

    def _on_event(self, event: BaseEvent) -> None:
        """Handle a lifecycle event: convert to TargetState and publish.

        This method is called by the EventBus for each of the five
        subscribed event types. It converts the event to a TargetState
        snapshot and calls ``adapter.publish()``. If the publish fails
        (adapter disconnected, network error, etc.), the error is caught
        and logged so it does not propagate back to the EventBus and
        crash the publisher thread.
        """
        try:
            state = self._event_to_state(event)
        except Exception:
            logger.exception(
                "StatePublisher: failed to convert event %s to TargetState",
                repr(event),
            )
            return

        try:
            self._adapter.publish(state)
        except Exception:
            # Catch ALL exceptions from publish -- the EventBus's
            # exception isolation would catch them anyway, but we log
            # here with more context for debugging.
            logger.warning(
                "StatePublisher: adapter.publish() failed for event %s "
                "(event_id=%s, target_id=%s), ignoring",
                event.event_type,
                event.event_id,
                event.target_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Event -> TargetState conversion
    # ------------------------------------------------------------------

    def _event_to_state(self, event: BaseEvent) -> TargetState:
        """Convert a lifecycle event to a TargetState snapshot.

        Extracts position/identity data from the event's payload (if
        present) and fills in defaults for fields the event does not
        carry. The event_type and event_id are always included in the
        TargetState's metadata so the receiver knows the lifecycle
        context.

        Field mapping:
            target_id  <- parsed from event.target_id ("S0-T0001" -> 1)
                          or payload.get("track_id") if available
            local_id   <- payload.get("track_id") or parsed seq
            global_id  <- None (no cross-camera fusion in B6)
            label      <- payload.detection.class_name or "unknown"
            confidence <- payload.detection.score or 0.0
            cx, cy     <- payload.detection.bbox.x/y or 0.0
            vx, vy     <- 0.0 (events do not carry velocity)
            width      <- payload.detection.bbox.w or 0.0
            height     <- payload.detection.bbox.h or 0.0
            timestamp  <- event.timestamp
            camera_id  <- event.slot_id
            metadata   <- {event_type, event_id, **event.payload}
        """
        # Parse the numeric parts from the target_id string.
        slot, seq = _parse_target_id(event.target_id)

        # Extract detection data from the payload if present.
        detection: dict[str, Any] = {}
        if "detection" in event.payload:
            det = event.payload["detection"]
            if isinstance(det, dict):
                detection = det

        # Extract bbox from detection dict. The bbox may be a nested
        # dict {"x":.., "y":.., "w":.., "h":..} or a flat set of keys.
        bbox: dict[str, Any] = {}
        if "bbox" in detection and isinstance(detection["bbox"], dict):
            bbox = detection["bbox"]

        # Use track_id from payload if available, else the parsed seq.
        track_id: int = event.payload.get("track_id", seq)

        # Build the metadata: always include event_type + event_id so
        # the receiver knows the lifecycle context. Merge in the rest
        # of the event payload (minus detection, which is already
        # extracted into position fields).
        metadata: dict[str, Any] = {
            "event_type": event.event_type,
            "event_id": event.event_id,
        }
        for key, value in event.payload.items():
            if key == "detection":
                continue  # already extracted into position fields
            metadata[key] = value

        return TargetState(
            target_id=track_id,
            local_id=track_id,
            global_id=None,
            label=detection.get("class_name", "unknown"),
            confidence=float(detection.get("score", 0.0)),
            cx=float(bbox.get("x", 0.0)),
            cy=float(bbox.get("y", 0.0)),
            vx=0.0,
            vy=0.0,
            width=float(bbox.get("w", 0.0)),
            height=float(bbox.get("h", 0.0)),
            timestamp=event.timestamp,
            camera_id=event.slot_id,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def is_started(self) -> bool:
        """Return ``True`` if the publisher is currently subscribed."""
        return self._started

    @property
    def adapter(self) -> ProtocolAdapter:
        """Return the injected ProtocolAdapter."""
        return self._adapter

    def __repr__(self) -> str:
        return (
            f"StatePublisher(started={self._started}, "
            f"adapter={type(self._adapter).__name__}, "
            f"subscriptions={len(self._subscriptions)})"
        )


# ---------------------------------------------------------------------------
# Helper: parse target_id string
# ---------------------------------------------------------------------------

def _parse_target_id(target_id: str) -> tuple[int, int]:
    """Parse a target_id string ``"S{slot}-T{seq}"`` into ``(slot, seq)``.

    The format is defined by
    :func:`visioncore.target_manager.target_id_factory.create_target_id`
    as ``"S{slot_id}-T{seq:04d}"`` (e.g. ``"S0-T0001"``).

    Args:
        target_id: The target_id string from an event.

    Returns:
        A ``(slot, seq)`` tuple of ints.

    Raises:
        ValueError: If the string does not match the expected format.
    """
    match = _TARGET_ID_RE.match(target_id)
    if match is None:
        raise ValueError(
            f"Cannot parse target_id {target_id!r}: expected format "
            f"'S{{slot}}-T{{seq}}' (e.g. 'S0-T0001')"
        )
    slot = int(match.group(1))
    seq = int(match.group(2))
    return slot, seq
