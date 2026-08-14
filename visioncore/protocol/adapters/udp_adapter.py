"""UDP adapter for the VisionCore protocol layer.

Defines :class:`UDPAdapter`, a :class:`~visioncore.protocol.base.ProtocolAdapter`
that sends serialized :class:`~visioncore.state.target_state.TargetState`
snapshots over UDP to a single destination (unicast or multicast).

The adapter is **serializer-agnostic** -- it accepts any object with a
``serialize(state) -> str | bytes`` method and sends whatever that method
returns. This means the same adapter works with:

* :class:`~visioncore.protocol.serialization.json_serializer.TargetStateSerializer`
  (JSON text, ``str`` output -- the adapter encodes to UTF-8 before sending)
* :class:`~visioncore.protocol.serialization.cbor_serializer.CBORSerializer`
  (CBOR binary, ``bytes`` output -- sent directly)
* Any future serializer that follows the same ``serialize()`` contract

The adapter never hardcodes JSON or any other specific format. Format
selection happens at construction time via the ``serializer`` parameter.

Type annotation convention
--------------------------
This module imports ``TargetState`` via its fully-qualified module path::

    from visioncore.state.target_state import TargetState

This avoids collision with the ``visioncore.core.TargetState`` enum.

MTU and fragmentation
---------------------
UDP datagrams have a maximum size of 65,507 bytes (IPv4). The typical
Ethernet MTU is 1500 bytes, leaving 1472 bytes for the UDP payload after
IP+UDP headers. Payloads exceeding the MTU are fragmented by the OS --
this works on localhost (loopback MTU is usually 65535) and on networks
that support fragmentation, but may fail on networks that drop fragments.

For production use, keep payloads under 1472 bytes. Use CBOR
(~32% smaller than JSON for large payloads) to fit more data in a single
datagram. The adapter does not enforce a size limit -- it relies on the
OS to handle fragmentation or raise ``OSError`` if the datagram is too
large.

Multicast
---------
The adapter supports multicast transparently -- pass a multicast group
address (e.g. ``"239.1.1.1"``) as ``host`` and the OS will send UDP
datagrams to the multicast group. No special send-side configuration is
needed; receivers must join the group to receive.

Example
-------
    >>> from visioncore.protocol import UDPAdapter
    >>> from visioncore.protocol.serialization import CBORSerializer
    >>> adapter = UDPAdapter(
    ...     serializer=CBORSerializer(),
    ...     host="127.0.0.1",
    ...     port=9999,
    ... )
    >>> with adapter:
    ...     adapter.publish(some_state)
    ... # socket closed on exit
"""

from __future__ import annotations

import logging
import socket
from typing import Any, Protocol, runtime_checkable

from visioncore.protocol.base import ProtocolAdapter
from visioncore.state.target_state import TargetState


__all__ = ["UDPAdapter"]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structural protocol for the serializer dependency.
# ---------------------------------------------------------------------------

@runtime_checkable
class _Serializer(Protocol):
    """Structural protocol that any serializer passed to UDPAdapter must satisfy.

    The adapter only calls ``serialize()`` -- it is a pure sink and never
    deserializes. The ``deserialize()`` method is listed here for
    documentation but is NOT called by the adapter; it exists on both
    :class:`TargetStateSerializer` and :class:`CBORSerializer` and is
    used by the receiving side.

    Using a ``Protocol`` (structural typing) rather than a base class
    keeps the adapter decoupled from any specific serializer hierarchy.
    """

    def serialize(self, state: TargetState) -> str | bytes:
        """Serialize a TargetState to a string or bytes payload."""
        ...


class UDPAdapter(ProtocolAdapter):
    """A protocol adapter that sends TargetState snapshots over UDP.

    The adapter opens a UDP socket on ``connect()``, sends each published
    snapshot as a single datagram via ``socket.send()``, and closes the
    socket on ``disconnect()``. The socket is ``connect()``-ed to the
    destination ``(host, port)`` so that ``send()`` (not ``sendto()``)
    can be used in the hot path -- this avoids re-resolving the address
    on every publish and lets the OS report ICMP errors (e.g.
    "destination unreachable") as ``OSError`` on subsequent sends.

    Serializer injection
    --------------------
    The adapter does NOT hardcode any serialization format. The
    ``serializer`` parameter accepts any object with a ``serialize()``
    method that returns ``str`` or ``bytes``:

    * If ``serialize()`` returns ``str`` (e.g. JSON), the adapter
      encodes it to UTF-8 before sending.
    * If ``serialize()`` returns ``bytes`` (e.g. CBOR), the adapter
      sends it directly with no additional encoding.

    This allows the caller to choose the wire format at construction
    time without modifying the adapter:

        >>> # JSON for debugging (human-readable)
        >>> adapter = UDPAdapter(TargetStateSerializer(), "127.0.0.1", 9999)
        >>> # CBOR for production (compact binary)
        >>> adapter = UDPAdapter(CBORSerializer(), "239.1.1.1", 5000)

    Thread safety
    -------------
    UDPAdapter is NOT thread-safe. The underlying socket is not protected
    by a lock. Producers that share an adapter across threads must
    serialise ``publish()`` calls externally.

    Args:
        serializer: An object with a ``serialize(state: TargetState) -> str | bytes``
            method. Typically :class:`TargetStateSerializer` (JSON) or
            :class:`CBORSerializer` (CBOR).
        host: Destination hostname or IP address. May be a unicast
            address (e.g. ``"127.0.0.1"``, ``"192.168.1.100"``) or a
            multicast group (e.g. ``"239.1.1.1"``).
        port: Destination UDP port (1-65535).

    Example:
        >>> from visioncore.protocol.serialization import CBORSerializer
        >>> adapter = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
        >>> adapter.connect()
        >>> adapter.publish(state)
        >>> adapter.disconnect()
    """

    def __init__(
        self,
        serializer: _Serializer,
        host: str,
        port: int,
    ) -> None:
        # Store the serializer -- we call serialize() on each publish.
        # The adapter never inspects or hardcodes the format.
        self._serializer: _Serializer = serializer
        self._host: str = host
        self._port: int = port

        # The socket is created lazily in connect() and destroyed in
        # disconnect(). Storing None here means health_check() returns
        # False before connect() is called.
        self._sock: socket.socket | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Create and connect the UDP socket.

        Creates a ``SOCK_DGRAM`` (UDP) socket and calls ``connect()``
        on it to set the default destination. For UDP, ``connect()``
        does NOT perform a handshake -- it only associates the
        destination address with the socket so that ``send()`` can be
        used without specifying the address each time.

        Idempotent: if already connected, this is a no-op.

        Raises:
            OSError: If the socket cannot be created or the destination
                address is invalid. This is propagated from the
                underlying ``socket`` module.
        """
        if self._sock is not None:
            # Already connected -- idempotent no-op.
            return

        # Create a UDP socket. AF_INET = IPv4. SOCK_DGRAM = UDP.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # connect() on a UDP socket sets the default destination --
        # no network traffic is generated. This lets us use send()
        # instead of sendto() in the hot path, and causes the OS to
        # deliver ICMP errors (e.g. port unreachable) as OSError on
        # subsequent send() calls.
        try:
            sock.connect((self._host, self._port))
        except OSError:
            # If connect fails, close the socket to avoid leaking it.
            sock.close()
            raise

        self._sock = sock
        logger.debug(
            "UDPAdapter connected to %s:%d (serializer=%s)",
            self._host, self._port, type(self._serializer).__name__,
        )

    def disconnect(self) -> None:
        """Close the UDP socket. Idempotent. Never raises.

        Closes the underlying socket if it is open. After disconnect,
        ``health_check()`` returns ``False`` and ``publish()`` raises
        ``RuntimeError``. A subsequent ``connect()`` creates a new
        socket.
        """
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                # disconnect() must not raise per the ProtocolAdapter
                # contract. Log the error and move on.
                logger.warning(
                    "UDPAdapter: error closing socket, ignoring: %s",
                    exc_info=True,
                )
            finally:
                self._sock = None
            logger.debug("UDPAdapter disconnected")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, state: TargetState) -> None:
        """Serialize and send a single TargetState as a UDP datagram.

        Calls ``self._serializer.serialize(state)`` to produce the
        payload, encodes to UTF-8 if the serializer returned ``str``,
        and sends the resulting bytes via ``socket.send()``.

        Args:
            state: The snapshot to publish.

        Raises:
            RuntimeError: If the adapter is not connected.
            TypeError: If ``state`` is not a TargetState (propagated
                from the serializer).
            OSError: If the send fails (e.g. destination unreachable,
                network error).
        """
        if self._sock is None:
            raise RuntimeError(
                "UDPAdapter.publish: adapter is not connected. "
                "Call connect() first."
            )
        if not isinstance(state, TargetState):
            raise TypeError(
                f"UDPAdapter.publish: state must be a TargetState, got "
                f"{type(state).__name__}"
            )

        # Serialize via the injected serializer. The adapter does NOT
        # hardcode any format -- it delegates entirely to the serializer.
        payload: str | bytes = self._serializer.serialize(state)

        # The serializer may return str (JSON) or bytes (CBOR). UDP
        # sockets require bytes. Encode str payloads as UTF-8.
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        elif not isinstance(payload, bytes):
            # Defensive: if the serializer returns something unexpected
            # (e.g. bytearray), convert to bytes.
            payload = bytes(payload)

        # Send as a single UDP datagram. If the payload exceeds the
        # path MTU, the OS will fragment it. On localhost this works
        # for payloads up to ~65507 bytes; on real networks, keep
        # payloads under 1472 bytes to avoid fragmentation.
        self._sock.send(payload)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return ``True`` if the socket is open and ready to send.

        This is a lightweight check -- it does not probe the network or
        verify that the destination is reachable. It only confirms that
        ``connect()`` has been called and ``disconnect()`` has not.
        """
        return self._sock is not None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UDPAdapter(serializer={type(self._serializer).__name__}, "
            f"host={self._host!r}, port={self._port!r}, "
            f"connected={self.health_check()})"
        )
