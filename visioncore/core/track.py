"""Track data structure for VisionCore.

Defines Track, a mutable container that associates a persistent identity
(track_id) with the most recent Detection and motion state.
"""

from __future__ import annotations

from dataclasses import dataclass

from visioncore.core.detection import Detection


@dataclass(slots=True)
class Track:
    """A tracked object spanning multiple frames.

    A Track associates a persistent identity (track_id) with the most recent
    Detection observed for that identity, along with motion state estimated
    by the tracker. Unlike Detection, a Track is **mutable** -- it is updated
    in place each frame as new detections are associated with it.

    Attributes:
        track_id: Globally unique identifier assigned when the track is
            created. Persists for the lifetime of the tracked object and
            is stable across short occlusions (up to max_misses).
        detection: The most recent Detection associated with this track.
            Represents the track's current spatial hypothesis.
        velocity: Estimated instantaneous velocity as (vx, vy) in normalised
            coordinates per frame. (0.0, 0.0) when no motion estimate is
            available (e.g. on the first frame).
        age: Number of frames since the track was created. Monotonically
            increasing; used for track maturation and filtering of
            short-lived tracks.
        lost_frames: Consecutive frames since the last successful detection
            association. Reset to 0 on a successful match; the track is
            removed when lost_frames exceeds the configured max_misses.

    Example:
        >>> from visioncore.core.detection import BBox, Detection
        >>> t = Track(
        ...     track_id=1,
        ...     detection=Detection(BBox(0.5, 0.5, 0.1, 0.2), 0.9, 0, "person"),
        ... )
        >>> t.track_id
        1
    """

    track_id: int
    detection: Detection
    velocity: tuple[float, float] = (0.0, 0.0)
    age: int = 0
    lost_frames: int = 0

    def __repr__(self) -> str:
        """Return a concise representation summarising key state.

        Includes the track id, detection score and class name, velocity
        components, age, and lost-frame count -- the fields most useful
        when debugging or logging tracker state.
        """
        vx: float
        vy: float
        vx, vy = self.velocity
        return (
            f"Track(track_id={self.track_id!r}, "
            f"score={self.detection.score:.4f}, "
            f"class_name={self.detection.class_name!r}, "
            f"velocity=({vx:.4f}, {vy:.4f}), "
            f"age={self.age!r}, "
            f"lost_frames={self.lost_frames!r})"
        )
