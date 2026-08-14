"""Debug event logger for the VisionCore EventBus.

Provides :class:`DebugEventLogger`, a drop-in observer that subscribes to
the ``"*"`` wildcard on an :class:`~visioncore.eventbus.bus.EventBus` and
formats every published event as a multi-line log record -- invaluable
for diagnosing lifecycle flow during development.

Design:
    * **Off by default.** The logger does not subscribe until
      :meth:`enable` is called, so merely constructing it (or leaving it
      disabled) costs nothing -- the bus never sees it.
    * **Toggleable.** :meth:`enable` / :meth:`disable` are idempotent and
      can be called repeatedly; they add / remove exactly one wildcard
      subscription.
    * **Logging, not printing.** All output goes through the standard
      :mod:`logging` module (no ``print``). The logger name is
      ``visioncore.eventbus.debug_logger``; configure level / handlers via
      the usual logging API.
    * **Performance-aware.** The callback short-circuits when the logger
      is not enabled for the :data:`logging.INFO` level, skipping the
      string formatting entirely. Disabled state has zero bus overhead
      (no subscription exists).

Output format (one :meth:`logging.Logger.info` call per event)::

    [EVENT]
    TargetLost
    event_id=evt-000001
    event_type=target.lost
    target=S0-T17
    slot=0
    timestamp=12.5000
    payload_keys={'last_seen'}

The second line is the event class name with a trailing ``Event`` suffix
striipped (``TargetLostEvent`` -> ``TargetLost``). Fields are emitted only
when present on the event object, so both typed
:class:`~visioncore.eventbus.events.BaseEvent` instances and the lighter
:class:`~visioncore.core.event.Event` are handled gracefully.

Example:
    >>> from visioncore.eventbus import EventBus, DebugEventLogger
    >>> from visioncore.eventbus.events import TargetLostEvent
    >>> bus = EventBus()
    >>> dbg = DebugEventLogger(bus)
    >>> dbg.enable()                       # start observing
    >>> bus.publish(TargetLostEvent(
    ...     event_id="e1", timestamp=1.0, target_id="S0-T1", slot_id=0))
    >>> dbg.disable()                      # stop observing
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visioncore.eventbus.bus import EventBus
    from visioncore.eventbus.subscriber import Subscriber


class DebugEventLogger:
    """Logs every published event on a bus for debugging.

    Subscribes to the ``"*"`` wildcard when enabled, formatting each
    event as a multi-line record emitted via :mod:`logging`. Disabled by
    default; toggling is idempotent and adds/removes exactly one
    subscription.

    Attributes:
        _bus: The :class:`EventBus` to observe.
        _token: The :class:`Subscriber` token for the wildcard
            subscription, or ``None`` when disabled.
        _logger: The :class:`logging.Logger` used for output.
        _enabled: Whether the wildcard subscription is currently active.
    """

    __slots__ = ("_bus", "_token", "_logger", "_enabled")

    def __init__(
        self,
        bus: "EventBus",
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise the logger in the **disabled** state.

        Parameters:
            bus: The :class:`EventBus` to observe.
            logger: Optional custom :class:`logging.Logger`. Defaults to
                ``logging.getLogger("visioncore.eventbus.debug_logger")``.
        """
        self._bus: EventBus = bus
        self._token: Subscriber | None = None
        self._logger: logging.Logger = (
            logger if logger is not None
            else logging.getLogger("visioncore.eventbus.debug_logger")
        )
        self._enabled: bool = False

    # ------------------------------------------------------------------
    # Toggle
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """Start observing the bus.

        Subscribes a single ``"*"`` wildcard callback. Idempotent:
        calling :meth:`enable` while already enabled is a no-op.
        """
        if self._enabled:
            return
        self._token = self._bus.subscribe("*", self._on_event)
        self._enabled = True
        self._logger.info("DebugEventLogger enabled (wildcard subscription)")

    def disable(self) -> None:
        """Stop observing the bus.

        Unsubscribes the wildcard callback. Idempotent: calling
        :meth:`disable` while already disabled is a no-op.
        """
        if not self._enabled:
            return
        self._bus.unsubscribe(self._token)  # type: ignore[arg-type]
        self._token = None
        self._enabled = False
        self._logger.info("DebugEventLogger disabled (subscription removed)")

    @property
    def enabled(self) -> bool:
        """Whether the logger is currently subscribed to the bus."""
        return self._enabled

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------

    def _on_event(self, event: object) -> None:
        """Wildcard callback invoked for every published event.

        Short-circuits when the logger is not enabled for :data:`INFO` so
        that raising the log level suppresses not just the output but
        also the formatting work -- keeping the disabled-but-subscribed
        path cheap (e.g. when a parent logger level is raised at runtime
        without calling :meth:`disable`).
        """
        if not self._logger.isEnabledFor(logging.INFO):
            return
        self._logger.info(self._format_event(event))

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_event(event: object) -> str:
        """Render ``event`` as a multi-line log string.

        Emits every field that is present on the event object (via
        :func:`getattr`), so typed :class:`BaseEvent` instances and the
        lightweight :class:`~visioncore.core.event.Event` are both
        handled without type checks.

        Parameters:
            event: The published event object.

        Returns:
            A newline-joined string beginning with ``[EVENT]``.
        """
        cls_name: str = type(event).__name__
        # Strip a trailing "Event" suffix for readability, but keep the
        # full name if stripping would yield an empty string.
        if cls_name.endswith("Event") and len(cls_name) > len("Event"):
            cls_name = cls_name[: -len("Event")]

        lines: list[str] = ["[EVENT]", cls_name]

        event_id: object = getattr(event, "event_id", None)
        if event_id is not None:
            lines.append(f"event_id={event_id}")

        event_type: object = getattr(event, "event_type", None)
        if event_type is not None:
            lines.append(f"event_type={event_type}")

        target_id: object = getattr(event, "target_id", None)
        if target_id is not None:
            lines.append(f"target={target_id}")

        slot_id: object = getattr(event, "slot_id", None)
        if slot_id is not None:
            lines.append(f"slot={slot_id}")

        timestamp: object = getattr(event, "timestamp", None)
        if isinstance(timestamp, (int, float)):
            lines.append(f"timestamp={timestamp:.4f}")
        elif timestamp is not None:
            lines.append(f"timestamp={timestamp}")

        payload: object = getattr(event, "payload", None)
        if isinstance(payload, dict) and payload:
            keys: str = "{" + ", ".join(repr(k) for k in payload) + "}"
        else:
            keys = "{}"
        lines.append(f"payload_keys={keys}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"DebugEventLogger(enabled={self._enabled}, "
            f"bus={self._bus!r})"
        )


__all__ = ["DebugEventLogger"]
