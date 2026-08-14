"""Detection data structures for VisionCore.

Defines BBox (a normalised bounding box) and Detection (a single detector
output). Both are immutable value objects.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in normalised coordinates.

    All coordinates are expressed in the range [0.0, 1.0] relative to the
    source image dimensions. Normalised coordinates are source-resolution-
    independent, allowing detections to be reused across display surfaces
    and downstream processing stages without rescaling.

    Attributes:
        x: Centre x-coordinate (width axis), normalised.
        y: Centre y-coordinate (height axis), normalised.
        w: Box width, normalised.
        h: Box height, normalised.
    """

    x: float
    y: float
    w: float
    h: float

    def __repr__(self) -> str:
        """Return a concise representation with fixed-precision floats."""
        return (
            f"BBox(x={self.x:.4f}, y={self.y:.4f}, "
            f"w={self.w:.4f}, h={self.h:.4f})"
        )


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection produced by a detector backend.

    A Detection is an immutable snapshot of one detector output: where the
    object is (bbox), how confident the detector is (score), and what the
    object is classed as (class_id / class_name). Detections are consumed by
    the tracker to form Tracks, and may carry additional backend-specific
    metadata through the attributes dict in higher-level structures.

    Attributes:
        bbox: Normalised bounding box (centre + size) of the detected object.
        score: Detector confidence in the range [0.0, 1.0]. Higher values
            indicate greater certainty that the detection is correct.
        class_id: Integer class index as defined by the model's label set
            (e.g. COCO class indices for YOLO models).
        class_name: Human-readable class label corresponding to class_id
            (e.g. "person", "car"). Provided for convenience and logging;
            the authoritative identity is class_id.

    Example:
        >>> d = Detection(
        ...     bbox=BBox(0.5, 0.5, 0.2, 0.4),
        ...     score=0.92,
        ...     class_id=0,
        ...     class_name="person",
        ... )
        >>> d.class_name
        'person'
    """

    bbox: BBox
    score: float
    class_id: int
    class_name: str

    def __repr__(self) -> str:
        """Return a concise representation with formatted score."""
        return (
            f"Detection(bbox={self.bbox!r}, score={self.score:.4f}, "
            f"class_id={self.class_id!r}, class_name={self.class_name!r})"
        )
