"""Protocol adapters subpackage for VisionCore.

Concrete :class:`~visioncore.protocol.base.ProtocolAdapter` implementations
that deliver serialized TargetState snapshots to specific transports
(network, file, etc.). Each adapter wraps a real I/O channel and delegates
serialization to an injected serializer (JSON, CBOR, or future formats).

Scope
-----
Currently the package exports:

    UDPAdapter  -- sends snapshots over UDP (unicast or multicast).

Future milestones may add sibling adapters (e.g. ``TcpAdapter``,
``FileAdapter``, ``SerialAdapter``). Each will subclass
:class:`~visioncore.protocol.base.ProtocolAdapter` and accept an injected
serializer.

Serializer injection
--------------------
Adapters in this package do NOT hardcode any serialization format. The
serializer is passed at construction time, allowing the caller to choose
the wire format:

    from visioncore.protocol.adapters import UDPAdapter
    from visioncore.protocol.serialization import CBORSerializer

    adapter = UDPAdapter(serializer=CBORSerializer(), host="127.0.0.1", port=9999)

This decouples transport (UDP) from format (CBOR), so the same adapter
works with any current or future serializer.

Type annotation convention
--------------------------
All modules here import ``TargetState`` via its fully-qualified module
path::

    from visioncore.state.target_state import TargetState

See ``docs/protocol-foundation.md`` for the full rationale.
"""

from __future__ import annotations

from visioncore.protocol.adapters.udp_adapter import UDPAdapter

__all__ = ["UDPAdapter"]
