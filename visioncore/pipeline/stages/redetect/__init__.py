"""ROI Redetection subpackage (Milestone D9.1).

Exports from this subpackage:

    RedetectConfig  -- redetection configuration dataclass.
    Roi             -- axis-aligned ROI in pixel coordinates.
    Redetector      -- abstract base class for redetection backends.
    RoiRedetector   -- concrete crop-and-rerun redetector.
    compute_roi     -- pure-function ROI computation.
    quality_gate    -- pure-function quality gate.
"""

from __future__ import annotations

from visioncore.pipeline.stages.redetect.redetect import (
    RedetectConfig,
    Redetector,
    Roi,
    RoiRedetector,
    compute_roi,
    quality_gate,
)

__all__ = [
    "RedetectConfig",
    "Redetector",
    "Roi",
    "RoiRedetector",
    "compute_roi",
    "quality_gate",
]
