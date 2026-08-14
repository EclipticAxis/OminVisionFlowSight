"""Frame data structure for VisionCore.

Defines the immutable Frame container that flows through the vision pipeline.
A Frame represents a single image capture from a video source at a specific
point in time, identified by a monotonically increasing frame_id.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class Frame:
    """A single image frame in the vision pipeline.

    Frame is the foundational data unit that enters the detection and tracking
    pipeline. It is immutable by design -- once captured, a frame's content
    and metadata never change. This immutability guarantees that multiple
    consumers (detectors, trackers, recorders) can safely share a reference
    to the same Frame instance without synchronization concerns.

    Attributes:
        frame_id: Monotonically increasing identifier assigned by the capture
            source. Uniquely identifies a frame within a single source stream.
            May reset when a new source session begins.
        timestamp: Capture timestamp in seconds, measured from the source's
            monotonic clock or wall-clock epoch. Used for temporal ordering
            across sources and for jitter / latency analysis.
        source_id: Identifier of the originating capture source (e.g. camera
            index, RTSP URL hash, or file path). Enables multi-source
            demultiplexing in downstream consumers.
        image: The raw pixel data as a NumPy array. Expected layout is
            HWC (height x width x channels) with uint8 dtype and BGR or RGB
            channel order, depending on the capture backend. The array is
            **not** copied on construction -- callers must not mutate it in
            place to preserve immutability semantics.

    Example:
        >>> import numpy as np
        >>> f = Frame(
        ...     frame_id=0,
        ...     timestamp=0.0,
        ...     source_id="cam0",
        ...     image=np.zeros((480, 640, 3), dtype=np.uint8),
        ... )
        >>> f.frame_id
        0
    """

    frame_id: int
    timestamp: float
    source_id: str
    image: np.ndarray

    def __repr__(self) -> str:
        """Return a concise, human-readable representation.

        The image array is summarised by its shape and dtype rather than
        printing every pixel value, keeping the representation compact even
        for high-resolution frames.
        """
        shape: object = getattr(self.image, "shape", "?")
        dtype: object = getattr(self.image, "dtype", "?")
        return (
            f"Frame(frame_id={self.frame_id!r}, "
            f"timestamp={self.timestamp!r}, "
            f"source_id={self.source_id!r}, "
            f"image=array(shape={shape}, dtype={dtype}))"
        )
