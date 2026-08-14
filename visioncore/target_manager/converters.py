"""Track-to-Target converters for VisionCore.

Provides pure functions that translate :class:`~visioncore.core.track.Track`
objects into :class:`~visioncore.core.target.Target` objects, as well as
helpers that convert raw detection dicts (as emitted by
:class:`~ai.inference.InferWorker`) into :class:`Track` objects.

All converters are **stateless** -- they create new instances without
registering them in any store. The caller is responsible for passing the
resulting Target to :class:`~visioncore.target_manager.manager.TargetManager`
if persistence is needed.

Target ID generation is delegated to
:func:`~visioncore.target_manager.target_id_factory.create_target_id` so
that every code path produces IDs with the same format.

Field mapping (Track → Target)
------------------------------

    ====================  ================================================
    Track field           Target field
    ====================  ================================================
    ``track_id`` (int)    ``target_id`` (str, via ``create_target_id``)
    ``velocity`` (tuple)  ``attributes["velocity"]``
    *(timestamp param)*   ``last_seen`` (float)
    ``lost_frames``       ``state`` (0 → ACTIVE, >0 → LOST)
    ====================  ================================================

Additional fields set automatically:
    * ``track`` -- back-reference to the source Track object.
    * ``slot_id`` -- caller-specified camera slot [0, 3], default 0.
    * ``priority`` -- caller-specified, default 0.
    * ``attributes["age"]`` -- copied from Track.age for traceability.
    * ``attributes["lost_frames"]`` -- copied for traceability.

Detection dict → Track mapping
------------------------------

    =================  ================================================
    Dict key           Track / Detection field
    =================  ================================================
    ``track_id``       ``Track.track_id`` (skipped if absent)
    ``x1,y1,x2,y2``    ``BBox`` (converted to centre + size)
    ``confidence``     ``Detection.score``
    ``class_id``       ``Detection.class_id``
    ``label``          ``Detection.class_name``
    =================  ================================================

Thread safety:
    These functions are pure and stateless. They create new objects
    and do not touch any shared state. Safe to call from any thread.
"""

from __future__ import annotations

import logging
from typing import Any

from visioncore.core.detection import BBox, Detection
from visioncore.core.target import Target, TargetState
from visioncore.core.track import Track
from visioncore.target_manager.target_id_factory import create_target_id

_logger: logging.Logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single-track conversion
# ---------------------------------------------------------------------------


def track_to_target(
    track: Track,
    *,
    slot_id: int = 0,
    target_id: str | None = None,
    timestamp: float = 0.0,
    priority: int = 0,
) -> Target:
    """Convert a single :class:`Track` to a :class:`Target`.

    The mapping follows the Alpha-phase specification:

    * ``track_id`` → ``target_id``: The Track's integer id is formatted
      via :func:`create_target_id` as ``S{slot_id}-T{track_id:04d}``
      (e.g. slot 0, track 3 → ``"S0-T0003"``).
      Pass an explicit ``target_id`` to override.
    * ``velocity`` → ``attributes["velocity"]``: Stored as a tuple
      ``(vx, vy)`` in the Target's extensible attributes dict.
    * ``timestamp`` → ``last_seen``: Since :class:`Track` does not carry
      a wall-clock timestamp, the caller must supply one. Defaults to
      ``0.0`` (meaning "unknown").
    * ``lost_frames`` → ``state``: If ``lost_frames == 0`` the Track is
      actively tracked → :attr:`TargetState.ACTIVE`. If ``lost_frames > 0``
      the Track has missed recent detections → :attr:`TargetState.LOST`.

    The source Track is also stored as ``Target.track`` (back-reference)
    so downstream code can access the full track state, including the
    underlying Detection, without a separate lookup.

    Parameters:
        track: The source :class:`Track` to convert. Must have a valid
            ``track_id`` and ``detection``.
        target_id: Optional explicit Target id. If ``None`` (default),
            derived via :func:`create_target_id` from ``slot_id`` and
            ``track.track_id``.
        timestamp: Wall-clock or monotonic timestamp (seconds) at which
            the conversion occurs. Stored in ``Target.last_seen``.
        priority: Processing priority [0, 255]. Default 0 (normal).

    Returns:
        A new :class:`Target` with the mapped fields. The Target is
        **not** registered in any store -- the caller must pass it to
        :meth:`TargetManager.create_target` or
        :meth:`TargetStore.add_target` if persistence is needed.

    Example:
        >>> from visioncore.core.detection import BBox, Detection
        >>> from visioncore.core.track import Track
        >>> t = Track(track_id=7,
        ...           detection=Detection(BBox(0.5, 0.5, 0.2, 0.4),
        ...                                0.9, 0, "person"),
        ...           velocity=(0.1, -0.05), age=5, lost_frames=0)
        >>> target = track_to_target(t, timestamp=12.5, priority=3)
        >>> target.target_id
        'S0-T0007'
        >>> target.slot_id
        0
        >>> target.state.name
        'ACTIVE'
        >>> target.attributes["velocity"]
        (0.1, -0.05)
        >>> target.last_seen
        12.5
        >>> target.track is t
        True
    """
    # --- Derive target_id via the unified factory (unless overridden) ---
    tid: str = (
        target_id
        if target_id is not None
        else create_target_id(slot_id, track.track_id)
    )

    # --- Derive state from lost_frames ---
    state: TargetState = (
        TargetState.ACTIVE if track.lost_frames == 0
        else TargetState.LOST
    )

    # --- Build attributes dict with mapped + traceability fields ---
    attrs: dict[str, Any] = {
        "velocity": track.velocity,
        "age": track.age,
        "lost_frames": track.lost_frames,
    }

    target: Target = Target(
        target_id=tid,
        track=track,
        slot_id=slot_id,
        priority=priority,
        state=state,
        attributes=attrs,
        last_seen=timestamp,
    )

    _logger.debug(
        "track_to_target: track_id=%s -> target_id=%s state=%s "
        "vel=(%.4f, %.4f) ts=%.4f",
        track.track_id, tid, state.name,
        track.velocity[0], track.velocity[1], timestamp,
    )

    return target


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------


def tracks_to_targets(
    tracks: list[Track],
    *,
    slot_id: int = 0,
    timestamp: float = 0.0,
    priority: int = 0,
) -> list[Target]:
    """Convert a list of :class:`Track` objects to :class:`Target` objects.

    Each Track is converted independently via :func:`track_to_target`.
    The resulting Targets preserve the input order and each receives a
    ``target_id`` derived from its source Track's ``track_id`` via the
    unified :func:`create_target_id` factory.

    This function is equivalent to::

        [track_to_target(t, slot_id=slot_id, timestamp=timestamp,
                         priority=priority)
         for t in tracks]

    but is provided as a named entry point for readability at call sites
    that process track batches (e.g. the output of a tracker update step).

    Parameters:
        tracks: List of :class:`Track` objects to convert. Order is
            preserved in the output.
        slot_id: Camera slot identifier applied to all Targets.
        timestamp: Wall-clock timestamp (seconds) applied to all
            converted Targets. See :func:`track_to_target`.
        priority: Processing priority applied to all Targets.

    Returns:
        A new list of :class:`Target` objects, one per input Track,
        in the same order as the input.

    Example:
        >>> from visioncore.core.detection import BBox, Detection
        >>> from visioncore.core.track import Track
        >>> tracks = [
        ...     Track(track_id=1, detection=Detection(BBox(.1,.1,.2,.2),.9,0,"p")),
        ...     Track(track_id=2, detection=Detection(BBox(.3,.3,.4,.4),.8,0,"p")),
        ... ]
        >>> targets = tracks_to_targets(tracks, timestamp=5.0)
        >>> [t.target_id for t in targets]
        ['S0-T0001', 'S0-T0002']
        >>> len(targets)
        2
    """
    targets: list[Target] = [
        track_to_target(t, slot_id=slot_id, timestamp=timestamp, priority=priority)
        for t in tracks
    ]

    _logger.debug(
        "tracks_to_targets: converted %d track(s) -> %d target(s) ts=%.4f",
        len(tracks), len(targets), timestamp,
    )

    return targets


# ---------------------------------------------------------------------------
# Detection-dict → Track conversion (used by InferWorker bypass)
# ---------------------------------------------------------------------------


def detection_to_track(det: dict) -> Track | None:
    """Convert a single detection dict to a :class:`Track`.

    The detection dict is the format emitted by
    :class:`~ai.inference.InferWorker` via the ``detection_ready`` signal.
    Recognised keys:

        * ``track_id`` (int)    -- required; if absent, returns ``None``.
        * ``x1, y1, x2, y2``    -- bounding-box corners (normalised).
        * ``confidence`` (float)-- detection score.
        * ``class_id`` (int)    -- class index.
        * ``label`` (str)       -- class name.

    The bbox corners are converted to centre + size
    (``BBox(x=cx, y=cy, w=x2-x1, h=y2-y1)``) to match the
    :class:`~visioncore.core.detection.BBox` convention.

    Parameters:
        det: A detection dict.

    Returns:
        A new :class:`Track`, or ``None`` if ``track_id`` is absent.

    Example:
        >>> d = {"track_id": 5, "x1": 0.1, "y1": 0.2,
        ...      "x2": 0.3, "y2": 0.4, "confidence": 0.88,
        ...      "class_id": 0, "label": "person"}
        >>> t = detection_to_track(d)
        >>> t.track_id
        5
        >>> t.detection.score
        0.88
        >>> t.detection.bbox.x  # centre x
        0.2
    """
    tid = det.get("track_id")
    if tid is None:
        return None

    x1 = float(det.get("x1", 0.0))
    y1 = float(det.get("y1", 0.0))
    x2 = float(det.get("x2", 0.0))
    y2 = float(det.get("y2", 0.0))

    track: Track = Track(
        track_id=int(tid),
        detection=Detection(
            bbox=BBox(
                x=(x1 + x2) * 0.5,
                y=(y1 + y2) * 0.5,
                w=x2 - x1,
                h=y2 - y1,
            ),
            score=float(det.get("confidence", 0.0)),
            class_id=int(det.get("class_id", 0)),
            class_name=str(det.get("label", "")),
        ),
    )
    return track


def detections_to_tracks(detections: list[dict]) -> list[Track]:
    """Convert a list of detection dicts to :class:`Track` objects.

    Entries without a ``track_id`` are silently skipped (they represent
    untracked detections that cannot participate in target management).

    Parameters:
        detections: List of detection dicts (as emitted by InferWorker).

    Returns:
        A list of :class:`Track` objects, one per dict that had a
        ``track_id``. Order is preserved.

    Example:
        >>> dets = [
        ...     {"track_id": 1, "x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1,
        ...      "confidence": 0.9, "class_id": 0, "label": "p"},
        ...     {"track_id": None, "x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1},
        ...     {"track_id": 2, "x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.6,
        ...      "confidence": 0.8, "class_id": 0, "label": "p"},
        ... ]
        >>> tracks = detections_to_tracks(dets)
        >>> [t.track_id for t in tracks]
        [1, 2]
    """
    tracks: list[Track] = []
    for det in detections:
        track = detection_to_track(det)
        if track is not None:
            tracks.append(track)
    return tracks


__all__ = [
    "track_to_target",
    "tracks_to_targets",
    "detection_to_track",
    "detections_to_tracks",
]
