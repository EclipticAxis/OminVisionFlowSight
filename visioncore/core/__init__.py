"""Unified model export layer for VisionCore core data structures.

This package defines the foundational, framework-agnostic data model that
will underpin the future VisionCore vision-processing engine. All types
here are pure data containers -- no business logic, no I/O, no side effects.

Unified model export
--------------------
The six primary model types are the public API of this package::

    from visioncore.core import (
        Frame,       # immutable captured image + metadata
        Detection,   # immutable detector output (bbox + score + class)
        Track,       # mutable tracked object (id + detection + motion)
        Target,      # mutable high-level entity (id + track + state + attrs)
        Event,       # immutable system occurrence record
        TargetState, # lifecycle enum: ACTIVE / LOST / LOCKED / RECOVERED / REMOVED
    )

Star import (``from visioncore.core import *``) exports all names listed
in ``__all__`` below.

Additional exports:
    BBox             -- normalised bounding box used by Detection.
    to_core_detection / from_core_detection / ... -- adapter functions
                       bridging legacy ``ai.detection.Detection`` and the
                       core model. ``ai.detection`` is lazy-imported so
                       these are safe to star-import.

Submodules:
    frame:      Frame definition.
    detection:  BBox + Detection definitions.
    track:      Track definition.
    target:     Target + TargetState definitions.
    event:      Event definition.
    adapters:   Legacy <-> core Detection bridge functions.
"""

from __future__ import annotations

from visioncore.core.adapters import (
    extract_extras,
    from_core_detection,
    from_core_detections,
    from_core_with_extras,
    to_core_detection,
    to_core_detections,
)
from visioncore.core.detection import BBox, Detection
from visioncore.core.event import Event
from visioncore.core.frame import Frame
from visioncore.core.target import Target, TargetState
from visioncore.core.track import Track

# ---------------------------------------------------------------------------
# Unified model export -- ``from visioncore.core import *`` yields these names.
# The six primary model types are listed first; auxiliary types and adapter
# functions follow.
# ---------------------------------------------------------------------------
__all__ = [
    # ---- Primary model types (unified export) ----
    "Frame",
    "Detection",
    "Track",
    "Target",
    "Event",
    "TargetState",
    # ---- Auxiliary types ----
    "BBox",
    # ---- Adapter functions ----
    "to_core_detection",
    "from_core_detection",
    "extract_extras",
    "from_core_with_extras",
    "to_core_detections",
    "from_core_detections",
]
