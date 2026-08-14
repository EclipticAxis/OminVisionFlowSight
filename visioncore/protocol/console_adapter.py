"""ConsoleAdapter -- a ProtocolAdapter that logs snapshots to a logger.

Each published TargetState is formatted and emitted at ``INFO`` level
via the standard ``logging`` framework. ``connect`` / ``disconnect``
events are also logged. Useful for debugging and development -- the
console is a human-readable sink for the snapshot stream.

Design notes
------------
* Uses ``logging`` (not bare ``print``) so output can be redirected,
  filtered, and formatted via standard logging configuration. This
  matches the convention established by
  :mod:`visioncore.eventbus.debug_logger`.
* The adapter is always "healthy" when connected -- the logging
  framework does not have a notion of connection failure. If the
  underlying logging handlers raise, Python's logging module silently
  swallows the error by default (call ``logging.raiseExceptions = True``
  to surface them).
* A custom logger can be injected at construction -- useful for routing
  ConsoleAdapter output to a specific handler in test fixtures.
"""

from __future__ import annotations

import logging

from visioncore.protocol.base import ProtocolAdapter
from visioncore.state.target_state import TargetState


__all__ = ["ConsoleAdapter"]


# Module-level logger. Subclasses or callers may override via the
# constructor's ``logger`` parameter.
_logger = logging.getLogger(__name__)


class ConsoleAdapter(ProtocolAdapter):
    """A protocol adapter that logs TargetState snapshots via ``logging``.

    Every ``publish()`` call logs a single-line, parseable record at
    ``INFO`` level. The format includes the most diagnostic fields
    (target_id, label, confidence, position, timestamp, camera_id) --
    metadata keys are omitted to keep the line short; the full snapshot
    is available via ``repr(state)`` if needed.

    Example output::

        INFO visioncore.protocol.console_adapter: [PUBLISH] id=42
        label='person' conf=0.920 pos=(0.5000,0.4000) ts=12.5000 cam=0

    Example:
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>> from visioncore.state.target_state import TargetState
        >>> from visioncore.protocol import ConsoleAdapter
        >>> with ConsoleAdapter():
        ...     adapter.publish(TargetState(1, 1, None, "person", 0.9,
        ...                                 0.5, 0.5, 0.0, 0.0, 0.1, 0.2,
        ...                                 0.0, 0, {}))
        INFO visioncore.protocol.console_adapter: ConsoleAdapter connected
        INFO visioncore.protocol.console_adapter: [PUBLISH] id=1 ...
        INFO visioncore.protocol.console_adapter: ConsoleAdapter disconnected
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Construct a ConsoleAdapter.

        Args:
            logger: Optional custom logger. If ``None``, uses the
                module-level logger ``visioncore.protocol.console_adapter``.
                Injecting a logger is useful in tests -- pass a
                ``logging.getLogger("test")`` with a captured handler
                to assert on logged output.
        """
        self._logger: logging.Logger = logger if logger is not None else _logger
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Mark the adapter as connected and log the event. Idempotent."""
        if not self._connected:
            self._connected = True
            self._logger.info("ConsoleAdapter connected")

    def disconnect(self) -> None:
        """Mark the adapter as disconnected and log the event. Idempotent."""
        if self._connected:
            self._connected = False
            self._logger.info("ConsoleAdapter disconnected")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, state: TargetState) -> None:
        """Log a single TargetState at INFO level.

        Raises:
            RuntimeError: If the adapter is not connected.
            TypeError: If ``state`` is not a TargetState.
        """
        if not self._connected:
            raise RuntimeError(
                "ConsoleAdapter.publish: adapter is not connected. "
                "Call connect() first."
            )
        if not isinstance(state, TargetState):
            raise TypeError(
                f"ConsoleAdapter.publish: state must be a TargetState, "
                f"got {type(state).__name__}"
            )
        # Lazy formatting via %-args -- the format string is only
        # expanded if the logger's level is INFO or below. This keeps
        # the adapter near-zero-cost when logging is disabled.
        self._logger.info(
            "[PUBLISH] id=%d label=%r conf=%.3f pos=(%.4f,%.4f) "
            "vel=(%.4f,%.4f) size=%.4fx%.4f ts=%.4f cam=%r "
            "metadata_keys=%s",
            state.target_id,
            state.label,
            state.confidence,
            state.cx,
            state.cy,
            state.vx,
            state.vy,
            state.width,
            state.height,
            state.timestamp,
            state.camera_id,
            list(state.metadata.keys()),
        )

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return the connection flag. Never raises."""
        return self._connected

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"ConsoleAdapter(connected={self._connected})"
