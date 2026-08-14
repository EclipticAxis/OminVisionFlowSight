"""Protocol adapter abstraction for VisionCore.

Defines :class:`ProtocolAdapter`, the abstract base class for all protocol
adapters in the VisionCore protocol layer. A protocol adapter is a *sink*
for :class:`~visioncore.state.target_state.TargetState` snapshots -- it
receives published snapshots and forwards them to some downstream
consumer (console, network, file, message bus, etc.).

This module is deliberately transport-agnostic. It contains **no** UDP,
**no** ROS2, **no** MAVLink, **no** socket code. Concrete adapters
implement the actual transport; the base class only defines the contract.

Type annotation convention
--------------------------
All type annotations in this package reference ``TargetState`` via its
fully-qualified module path::

    from visioncore.state.target_state import TargetState

This is intentional and mandatory. The alternative
``from visioncore import TargetState`` would import the **lifecycle enum**
(``ACTIVE`` / ``LOST`` / ``LOCKED`` / ``RECOVERED`` / ``REMOVED``) that is
re-exported from the top-level ``visioncore`` package, not the **snapshot
dataclass** defined in ``visioncore.state``. The two types share a name
but are semantically distinct (see ``docs/targetstate-design.md``).

Thread safety
-------------
ProtocolAdapter itself does not enforce thread safety. Adapters are not
required to be safe for concurrent ``publish()`` calls from multiple
threads -- producers that share an adapter across threads must serialise
access (e.g. via a lock). However, ``TargetState`` instances are frozen
and safe to share between threads without copying.

Example
-------
    >>> from visioncore.state.target_state import TargetState
    >>> from visioncore.protocol import NullAdapter
    >>> adapter = NullAdapter()
    >>> with adapter:
    ...     adapter.publish(TargetState(1, 1, None, "person", 0.9,
    ...                                  0.5, 0.5, 0.0, 0.0, 0.1, 0.2,
    ...                                  0.0, 0, {}))
    ... # disconnect() called automatically on exit
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence

from visioncore.state.target_state import TargetState


__all__ = ["ProtocolAdapter"]


logger = logging.getLogger(__name__)


class ProtocolAdapter(ABC):
    """Abstract base class for protocol adapters.

    A ProtocolAdapter is a sink for TargetState snapshots. Producers
    (e.g. InferWorker, TargetManager) call ``publish()`` to push snapshots
    downstream; the adapter handles delivery to whatever consumer it
    wraps (console, network, file, etc.).

    Lifecycle contract
    ------------------
    * ``connect()`` must be called before ``publish()``. Calling
      ``publish()`` on a disconnected adapter raises ``RuntimeError``.
    * ``disconnect()`` is idempotent -- safe to call on an
      already-disconnected adapter, and must not raise.
    * ``connect()`` is also idempotent -- calling it on an already-connected
      adapter is a no-op.
    * ``health_check()`` returns ``True`` iff the adapter is connected and
      ready to accept publishes. It must not raise -- returns ``False`` on
      any internal error.

    Publishing contract
    -------------------
    * ``publish()`` accepts a single TargetState. ``None`` or non-TargetState
      values raise ``TypeError``.
    * ``publish_many()`` accepts a ``Sequence[TargetState]`` and preserves
      order. The default implementation loops ``publish()``; subclasses
      that wrap batch-optimised transports (e.g. a single UDP datagram
      carrying multiple snapshots) should override it.
    * Both methods must not block indefinitely. Network-backed adapters
      should enforce timeouts.

    Context manager
    ---------------
    ProtocolAdapter implements the context manager protocol so callers
    can ensure clean teardown::

        with adapter:
            adapter.publish(state)
        # disconnect() called automatically, even on exception

    Subclassing
    -----------
    Subclasses **must** implement: ``connect``, ``disconnect``,
    ``publish``, ``health_check``.

    Subclasses **may** override ``publish_many`` for batch-optimised
    transports. The default implementation type-checks eagerly (fails
    fast on the first bad element) then loops ``publish()``.
    """

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Establish the connection to the downstream consumer.

        Must be called before ``publish()``. Idempotent -- calling
        ``connect()`` on an already-connected adapter is a no-op.

        Raises:
            ConnectionError: If the connection cannot be established.
                (Subclasses may raise more specific subclasses.)
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection.

        Idempotent -- safe to call on an already-disconnected adapter.
        Must not raise. Releases any resources held by the adapter
        (sockets, file handles, etc.).
        """

    @abstractmethod
    def publish(self, state: TargetState) -> None:
        """Publish a single TargetState snapshot downstream.

        Args:
            state: The snapshot to publish. Must be a TargetState
                instance (not ``None``, not any other type).

        Raises:
            RuntimeError: If the adapter is not connected.
            TypeError: If ``state`` is not a TargetState.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the adapter is connected and ready.

        This method must be side-effect-free and must not raise. On any
        internal error, return ``False`` rather than propagating the
        exception. Used by health-monitoring infrastructure to decide
        whether to keep publishing or to fall back.

        Returns:
            ``True`` if the adapter can currently accept publishes,
            ``False`` otherwise.
        """

    # ------------------------------------------------------------------
    # Concrete methods -- subclasses MAY override
    # ------------------------------------------------------------------

    def publish_many(self, states: Sequence[TargetState]) -> None:
        """Publish a batch of TargetState snapshots.

        Default implementation loops ``publish()`` for each state.
        Subclasses that wrap batch-optimised transports (e.g. a single
        UDP datagram carrying multiple serialised snapshots) should
        override this to amortise per-message overhead.

        Type checking is done eagerly -- the entire sequence is scanned
        for non-TargetState elements before any publish happens. This
        ensures a batch either fully publishes or fully fails, with no
        partial delivery on a type error mid-batch.

        Args:
            states: Sequence of snapshots to publish. Order is preserved.
                An empty sequence is a no-op (and does not require the
                adapter to be connected).

        Raises:
            RuntimeError: If the adapter is not connected and the
                sequence is non-empty.
            TypeError: If any element is not a TargetState.
        """
        # Eager type-check: fail fast on the first bad element, before
        # publishing anything. This prevents partial-batch delivery.
        for i, s in enumerate(states):
            if not isinstance(s, TargetState):
                raise TypeError(
                    f"publish_many: states[{i}] must be a TargetState, "
                    f"got {type(s).__name__}"
                )
        # Publish in order. If the adapter is disconnected, the first
        # publish() call raises RuntimeError and the rest are skipped.
        for s in states:
            self.publish(s)

    # ------------------------------------------------------------------
    # Context manager -- concrete, NOT for override
    # ------------------------------------------------------------------

    def __enter__(self) -> "ProtocolAdapter":
        """Enter context: connect and return self."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit context: disconnect. Does not suppress exceptions."""
        self.disconnect()
        # Returning None (falsy) means exceptions are not suppressed.

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation with connection state.

        Calls :meth:`health_check` to report the current state. Because
        ``health_check`` is contractually side-effect-free and non-raising,
        this is safe to call at any time (e.g. from a debugger).
        """
        try:
            connected = self.health_check()
        except Exception:
            # Defensive: health_check is not supposed to raise, but if a
            # buggy subclass does, we must not crash repr().
            connected = False
        return f"{type(self).__name__}(connected={connected})"
