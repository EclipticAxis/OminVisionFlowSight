"""VisionCore plugin layer -- the plugin interface abstraction.

This package hosts the *plugin layer* of VisionCore (Milestone D6): the
interfaces that let future plugins (detectors, trackers, algorithms) be
integrated by implementing a small contract, plus two example plugins
and an in-memory plugin manager.

Scope (Milestones D6 + D7)
--------------------------
D6 delivered the **interface layer**:

    PluginInterface     -- abstract base class for all plugins
                           (load / run / shutdown contract).
    PluginManager       -- abstract base class for plugin registries
                           (register / unregister / get / list).
    MemoryPluginManager -- concrete in-memory reference implementation.
    NullPlugin          -- example plugin: no-op passthrough.
    ConsolePlugin       -- example plugin: logs lifecycle + snapshots.

D7 adds **dynamic discovery**:

    PluginRegistry      -- plugin registry with entry-point scanning
                           and dotted-path module loading.

It deliberately does **not** contain:

* any concrete detector / tracker / algorithm plugin (those arrive in
  later milestones as ``PluginInterface`` subclasses);
* any networking, ROS2, or MAVLink transport code;
* any modification to ``ai/``, ``gui/``, or ``camera/``.

Type annotation convention
--------------------------
Per the D6 spec, every interface signature that mentions a type uses its
**full path** in the annotation, e.g. ``run()`` declares the target
snapshot as ``"visioncore.state.target_state.TargetState"`` rather than
a bare ``TargetState``. This keeps the contract self-documenting and
independent of import aliasing.

Naming note
-----------
Mirroring ``visioncore.state``, the plugin layer is **not** re-exported
from ``visioncore/__init__.py`` -- consumers import it explicitly from
this package.
"""

from __future__ import annotations

from visioncore.plugin.base import (
    ConsolePlugin,
    MemoryPluginManager,
    NullPlugin,
    PluginError,
    PluginInterface,
    PluginManager,
)
from visioncore.plugin.registry import PluginRegistry

__all__ = [
    "PluginError",
    "PluginInterface",
    "PluginManager",
    "MemoryPluginManager",
    "PluginRegistry",
    "NullPlugin",
    "ConsolePlugin",
]
