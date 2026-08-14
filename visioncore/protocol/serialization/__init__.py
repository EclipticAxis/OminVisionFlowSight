"""Serialization subpackage for the VisionCore protocol layer.

Provides serializers that convert
:class:`~visioncore.state.target_state.TargetState` snapshots to and from
wire formats. Serializers are transport-agnostic -- they produce/consume
strings or bytes, leaving the actual network I/O to protocol adapters.

Scope
-----
The package exports two serializers with the same
``serialize()`` / ``deserialize()`` contract but different output formats:

    TargetStateSerializer  -- JSON text format (str output, str/bytes input).
                              Human-readable, standard-library only.
    CBORSerializer         -- CBOR binary format (bytes output, bytes input).
                              Compact, fast, requires the ``cbor2`` package.

Choose JSON for logging, debugging, and human-readable output. Choose
CBOR for network transport, high-volume file logging, and bandwidth-
sensitive links. Both produce semantically identical TargetState
snapshots on deserialize.

Future milestones may add sibling serializers (e.g.
``MessagePackSerializer``). Each will follow the same contract.

Type annotation convention
--------------------------
All modules in this package import ``TargetState`` via its
fully-qualified module path::

    from visioncore.state.target_state import TargetState

This is mandatory -- ``from visioncore import TargetState`` would
import the *lifecycle enum* (ACTIVE / LOST / LOCKED / RECOVERED /
REMOVED) re-exported from the top-level ``visioncore`` package, not the
*snapshot dataclass* defined in :mod:`visioncore.state`. See
``docs/protocol-foundation.md`` for the full rationale.
"""

from __future__ import annotations

from visioncore.protocol.serialization.cbor_serializer import CBORSerializer
from visioncore.protocol.serialization.json_serializer import TargetStateSerializer

__all__ = ["TargetStateSerializer", "CBORSerializer"]
