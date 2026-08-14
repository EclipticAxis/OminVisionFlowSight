"""VisionCore runtime package -- model lifecycle management.

This package hosts the *model runtime layer* of VisionCore (Milestone D9.2):
the abstraction that separates model lifecycle (load / unload / warmup /
switch / health) from detection logic (NMS, box conversion, filtering).

Scope (Milestone D9.2)
----------------------
D9.2 delivers the **runtime layer only**:

    ModelRuntime    -- abstract base class for a single model lifecycle.
    ModelManager    -- manages multiple runtimes (device detection,
                       model switching, lifecycle coordination).

It deliberately does **not** contain:

* any detection logic (NMS, box conversion, label filtering);
* any YOLO / ONNX / torch imports at the top level (concrete runtimes
  are imported lazily);
* any modification to ``ai/``, ``camera/``, or ``gui/``.
"""

from __future__ import annotations

from visioncore.runtime.model.model_runtime import (
    ModelInfo,
    ModelManager,
    ModelManagerError,
    ModelRuntime,
)

__all__ = [
    "ModelInfo",
    "ModelManager",
    "ModelManagerError",
    "ModelRuntime",
]
