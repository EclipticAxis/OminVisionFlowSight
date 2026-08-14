"""NullAdapter -- a no-op ProtocolAdapter for testing and defaults.

All publish calls are silently discarded. Useful as:

* a default adapter when no real transport is configured (the "null
  object" pattern);
* a placeholder in test fixtures where a ProtocolAdapter slot must be
  filled but no actual delivery is expected;
* a baseline for performance comparison -- measures pure call overhead
  without any I/O.

NullAdapter is always "ready" once connected. It never fails, never
blocks, and never produces output. It is the safest possible adapter
and is recommended as the default in production code paths that have
not yet been wired to a real transport.
"""

from __future__ import annotations

from visioncore.protocol.base import ProtocolAdapter
from visioncore.state.target_state import TargetState


__all__ = ["NullAdapter"]


class NullAdapter(ProtocolAdapter):
    """A no-op protocol adapter that discards every published snapshot.

    ``connect()`` / ``disconnect()`` toggle an internal flag but perform
    no I/O. ``publish()`` / ``publish_many()`` accept any TargetState
    and discard it. ``health_check()`` returns the connection flag.

    The adapter enforces the full ProtocolAdapter contract (connection
    state check, type check) even though it does nothing with the data.
    This makes it useful for testing contract enforcement -- if code
    accidentally publishes to a disconnected adapter or passes a
    non-TargetState, NullAdapter will surface the error just like a
    real adapter would.

    Example:
        >>> from visioncore.state.target_state import TargetState
        >>> adapter = NullAdapter()
        >>> adapter.connect()
        >>> adapter.publish(TargetState(1, 1, None, "person", 0.9,
        ...                             0.5, 0.5, 0.0, 0.0, 0.1, 0.2,
        ...                             0.0, 0, {}))  # silently discarded
        >>> adapter.health_check()
        True
        >>> adapter.disconnect()
        >>> adapter.health_check()
        False
    """

    def __init__(self) -> None:
        # Track connection state so we can enforce the "must connect
        # before publish" contract. Start disconnected -- callers must
        # explicitly connect before use.
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Mark the adapter as connected. Idempotent."""
        # Idempotent: connecting an already-connected adapter is a no-op.
        self._connected = True

    def disconnect(self) -> None:
        """Mark the adapter as disconnected. Idempotent. Never raises."""
        # Idempotent: disconnecting an already-disconnected adapter is
        # a no-op. Must not raise under any circumstance.
        self._connected = False

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, state: TargetState) -> None:
        """Accept and discard a single TargetState.

        Raises:
            RuntimeError: If the adapter is not connected.
            TypeError: If ``state`` is not a TargetState.
        """
        if not self._connected:
            raise RuntimeError(
                "NullAdapter.publish: adapter is not connected. "
                "Call connect() first."
            )
        if not isinstance(state, TargetState):
            raise TypeError(
                f"NullAdapter.publish: state must be a TargetState, "
                f"got {type(state).__name__}"
            )
        # Discard -- this is the null sink.

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
        return f"NullAdapter(connected={self._connected})"
