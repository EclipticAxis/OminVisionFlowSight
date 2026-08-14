"""VisionCore runtime state -- cache, statistics, and timing state.

This package hosts the *runtime state layer* of VisionCore (Milestone D9.4):
the extraction of temporary caches, statistics counters, and timing state
from ``ai/inference.py``'s InferWorker into a clean, testable state
container.

Scope (Milestone D9.4)
----------------------
D9.4 delivers the **state layer only**:

    RuntimeState    -- state container for caches, statistics, timing.

It deliberately does **not** contain:

* any detection logic (filtering, NMS, box conversion);
* any model loading or inference code;
* any modification to ``ai/``, ``camera/``, or ``gui/``.
"""

from __future__ import annotations

from visioncore.runtime.state.runtime_state import RuntimeState

__all__ = ["RuntimeState"]
