"""VisionCore state package -- immutable target-state snapshots.

This package hosts the *snapshot* layer of the VisionCore data model:
point-in-time observations of a target's identity, kinematics, and
provenance. Snapshots are immutable, framework-agnostic, and carry no
transport / protocol coupling.

Scope
-----
Currently the package exports a single type:

    TargetState  -- frozen dataclass snapshot of one target at one instant.

Naming note
-----------
A *different* ``TargetState`` already exists in
:mod:`visioncore.core.target` -- there it is a **lifecycle enum**
(ACTIVE / LOST / LOCKED / RECOVERED / REMOVED). The two types are
intentionally distinct and coexist:

    from visioncore.core import TargetState as TargetLifecycle   # enum
    from visioncore.state import TargetState                      # snapshot

To avoid collisions at the top-level ``visioncore`` namespace, the
snapshot type is **not** re-exported from ``visioncore/__init__.py``.
Consumers must import it explicitly from this package.

Future expansion
----------------
Subsequent milestones may add sibling snapshot types here (e.g.
``GlobalTargetState`` for cross-camera fused targets, ``ZoneState`` for
region-level aggregates). Each will follow the same frozen + slotted +
plain-primitive contract established by :class:`TargetState`.
"""

from __future__ import annotations

from visioncore.state.target_state import TargetState

__all__ = ["TargetState"]
