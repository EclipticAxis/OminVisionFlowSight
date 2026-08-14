"""VisionCore -- the future core architecture for VisionDataPlatform.

This top-level package will eventually host the decoupled, framework-
agnostic vision-processing engine. At present it contains only the core
data model (``visioncore.core``), which defines the immutable and mutable
data structures that will flow through the pipeline.

The existing VisionDataPlatform application (``ai/``, ``gui/``, ``camera/``,
``common/``) continues to operate unchanged. VisionCore is being introduced
**alongside** it; migration of existing logic onto these structures will
happen in later phases and is explicitly out of scope for this initial
scaffold.

Quick reference::

    from visioncore import Frame, Detection, BBox, Track, Target, TargetState, Event
"""

from __future__ import annotations

from visioncore.core import (
    BBox,
    Detection,
    Event,
    Frame,
    Target,
    TargetState,
    Track,
)

__all__ = [
    "Frame",
    "BBox",
    "Detection",
    "Track",
    "Target",
    "TargetState",
    "Event",
]

__version__ = "0.1.0"
