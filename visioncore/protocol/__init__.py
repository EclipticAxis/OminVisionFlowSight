"""VisionCore protocol layer -- abstract adapter contract + concrete sinks.

This package hosts the *protocol layer* of VisionCore: an abstraction over
the downstream delivery of :class:`~visioncore.state.target_state.TargetState`
snapshots. Producers (InferWorker, TargetManager, decision modules) push
snapshots into a :class:`ProtocolAdapter`; the adapter forwards them to
some consumer (console, network, file, message bus, ...).

Scope
-----
Currently the package exports three types:

    ProtocolAdapter  -- abstract base class (the contract).
    NullAdapter      -- no-op sink for testing and defaults.
    ConsoleAdapter   -- logging-based sink for debugging.

Future milestones will add network adapters (UDP, ...) as sibling modules
within this package. Each will subclass :class:`ProtocolAdapter` and
implement the five-method contract.

Type annotation convention
--------------------------
All modules in this package import ``TargetState`` via its fully-qualified
module path::

    from visioncore.state.target_state import TargetState

This is mandatory. The shortcut ``from visioncore import TargetState``
would import the *lifecycle enum* (ACTIVE / LOST / LOCKED / RECOVERED /
REMOVED) that is re-exported from the top-level ``visioncore`` package,
not the *snapshot dataclass* defined in :mod:`visioncore.state`. The two
types share a name but are semantically distinct.

Transport agnosticism
---------------------
This package contains **no** transport code -- no UDP, no ROS2, no
MAVLink, no sockets. The abstract base defines the contract; concrete
adapters implement delivery. ``NullAdapter`` and ``ConsoleAdapter`` are
both non-network (one discards, one logs). Future network adapters will
live in separate modules and will be the only place that imports
transport libraries.
"""

from __future__ import annotations

from visioncore.protocol.base import ProtocolAdapter
from visioncore.protocol.console_adapter import ConsoleAdapter
from visioncore.protocol.null_adapter import NullAdapter

__all__ = [
    "ProtocolAdapter",
    "NullAdapter",
    "ConsoleAdapter",
]
