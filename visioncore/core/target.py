"""Target data structure and state enum for VisionCore.

Defines TargetState (lifecycle enum) and Target (the highest-level entity
in the VisionCore data model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from visioncore.core.track import Track


class TargetState(Enum):
    """Lifecycle state of a Target within the vision system.

    The state machine transitions are::

        ACTIVE --(lost too long)----> LOST
        LOST   --(re-detected)------> RECOVERED --> ACTIVE
        ACTIVE --(locked)-----------> LOCKED
        LOCKED --(released)---------> ACTIVE
        any    --(removed)----------> REMOVED  (terminal)

    Members:
        ACTIVE: The target is currently being tracked and has a recent
            detection. This is the normal operational state.
        LOST: The target's track has exceeded the missed-frame threshold
            but has not yet been removed. The target may be recovered if
            re-detected within the recovery window.
        LOCKED: The target has been manually or programmatically selected
            for exclusive attention (e.g. operator lock, region guard).
            Locked targets receive priority processing.
        RECOVERED: The target was lost and has just been re-associated
            with a new detection. A transient state that transitions back
            to ACTIVE on the next successful update.
        REMOVED: The target has been permanently removed from the system.
            This is a terminal state -- no further transitions occur.
    """

    ACTIVE = auto()
    LOST = auto()
    LOCKED = auto()
    RECOVERED = auto()
    REMOVED = auto()


@dataclass(slots=True)
class Target:
    """A semantically meaningful object under sustained observation.

    A Target is the highest-level entity in the VisionCore data model. While
    a Track answers *"what is moving and where"*, a Target answers *"what
    are we watching and why"*. Targets aggregate one or more Tracks over
    time, carry application-level priority and state, and hold extensible
    attributes for domain-specific metadata (e.g. person identity, vehicle
    plate, threat level).

    Attributes:
        target_id: Stable, globally unique identifier for this target.
            Distinct from track_id -- a single target may span multiple
            track lifetimes (e.g. after a track is lost and re-created).
            Recommended format: ``S{slot_id}-T{track_id}`` (e.g. ``S0-T1``).
        track: The current Track associated with this target, or None when
            the target is in LOST / REMOVED state with no active track.
        slot_id: Camera slot identifier [0, 3]. Disambiguates targets
            across multiple cameras that may produce overlapping track IDs.
            Default 0 (single-camera backward compatibility).
        priority: Processing priority in the range [0, 255] (higher = more
            important). Influences resource allocation in contested scenes.
            Default 0 (normal priority).
        state: Current lifecycle state (see :class:`TargetState`). Drives
            behaviour in downstream decision-making modules.
        attributes: Extensible key-value store for domain-specific metadata.
            Common keys might include "identity", "threat_level",
            "color_descriptor". Values are Any to avoid coupling the core
            data model to any single domain.
        last_seen: Timestamp (seconds) of the most recent frame in which
            this target had a successful detection association. Used for
            stale-target detection and recovery windowing.

    Example:
        >>> from visioncore.core.detection import BBox, Detection
        >>> from visioncore.core.track import Track
        >>> t = Target(
        ...     target_id="S0-T1",
        ...     track=Track(
        ...         track_id=1,
        ...         detection=Detection(BBox(0.5, 0.5, 0.1, 0.2), 0.9, 0, "person"),
        ...     ),
        ...     slot_id=0,
        ... )
        >>> t.state
        <TargetState.ACTIVE: 1>
        >>> t.slot_id
        0
    """

    target_id: str
    track: Track | None
    slot_id: int = 0
    priority: int = 0
    state: TargetState = TargetState.ACTIVE
    attributes: dict[str, Any] = field(default_factory=dict)
    last_seen: float = 0.0

    def __repr__(self) -> str:
        """Return a concise representation.

        The track is summarised by its id and score (or "None"). The
        attributes dict is summarised by its keys to keep the
        representation compact for large payloads.
        """
        if self.track is None:
            track_repr: str = "None"
        else:
            track_repr = (
                f"Track(id={self.track.track_id}, "
                f"score={self.track.detection.score:.4f})"
            )

        if self.attributes:
            attr_keys: str = (
                "{" + ", ".join(repr(k) for k in self.attributes) + "}"
            )
        else:
            attr_keys = "{}"

        return (
            f"Target(target_id={self.target_id!r}, "
            f"track={track_repr}, "
            f"slot_id={self.slot_id!r}, "
            f"priority={self.priority!r}, "
            f"state={self.state.name}, "
            f"attributes={attr_keys}, "
            f"last_seen={self.last_seen:.4f})"
        )
