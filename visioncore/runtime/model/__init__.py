"""Model runtime subpackage (Milestone D9.2).

Exports from this subpackage:

    ModelInfo       -- model metadata dataclass.
    ModelManager    -- multi-model lifecycle manager.
    ModelManagerError -- structured exception.
    ModelRuntime    -- abstract base class for a single model.
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
