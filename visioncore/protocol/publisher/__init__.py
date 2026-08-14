"""Publisher subpackage for the VisionCore protocol layer.

Provides bridges between the EventBus (event-driven) and the ProtocolAdapter
(transport-driven). Publishers subscribe to events on the bus, convert
them to :class:`~visioncore.state.target_state.TargetState` snapshots,
and publish the snapshots through an injected protocol adapter.

Scope
-----
Currently the package exports:

    StatePublisher       -- subscribes to the five Target lifecycle events
                            (Created/Lost/Recovered/Locked/Removed) and
                            publishes a TargetState snapshot for each.
    ShadowStatePublisher -- extends StatePublisher with debug metrics
                            (states/sec, success rate, avg latency).
                            Default NullAdapter = zero side effects.

Future milestones may add sibling publishers (e.g. ``FramePublisher``
for raw frame events, ``DetectionPublisher`` for detector output).

No modification of GUI or InferWorker
-------------------------------------
Publishers in this package are pure observers. They read from the
EventBus and write to a ProtocolAdapter. They do NOT modify:

* ``gui/`` -- no GUI files are touched
* ``ai/inference.py`` (InferWorker) -- no inference files are touched

The EventBus they subscribe to must already exist (created by the
caller, typically in InferWorker's shadow integration from B2).

Type annotation convention
--------------------------
All modules here import ``TargetState`` via its fully-qualified module
path::

    from visioncore.state.target_state import TargetState

See ``docs/protocol-foundation.md`` for the full rationale.
"""

from __future__ import annotations

from visioncore.protocol.publisher.shadow_publisher import ShadowStatePublisher
from visioncore.protocol.publisher.state_publisher import StatePublisher

__all__ = ["StatePublisher", "ShadowStatePublisher"]
