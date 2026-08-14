"""Adapters between legacy ``ai.detection.Detection`` and the new
``visioncore.core.detection.Detection``.

This module is the **only** bridge between the two detection models. It
allows the existing application (``ai/``, ``gui/``, ``camera/``) to keep
using its own Detection type unchanged, while new VisionCore code can
consume the same detections through the cleaner core model.

Design principles
-----------------
1. **No runtime dependency on ``ai`` at import time.**
   ``ai.detection`` is imported lazily inside ``from_core_detection`` so
   that ``visioncore`` remains importable in isolation.

2. **Duck-typed input for ``to_core_detection``.**
   Any object exposing ``bbox`` (4-tuple x1,y1,x2,y2), ``confidence``,
   ``class_id``, and ``label`` is accepted -- not just
   ``ai.detection.Detection``. This future-proofs the adapter against
   new detector backends.

3. **Lossless round-trip via extras.**
   The core Detection intentionally carries only four fields (bbox, score,
   class_id, class_name). OBB polygons, pose keypoints, and track metadata
   are preserved through ``extract_extras`` / ``from_core_with_extras`` so
   that OBB and Pose detections survive a full round-trip without data loss.

Compatibility
-------------
* **YOLO / YOLO26 (Detect)** -- straightforward bbox + score mapping.
* **OBB** -- bbox is derived from the axis-aligned bounding box; the
  oriented polygon is carried as an extra.
* **Pose** -- bbox is the person box; keypoints are carried as an extra.
* **Future extensions** -- new extra fields can be added to
  ``extract_extras`` and ``from_core_detection`` without breaking existing
  callers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from visioncore.core.detection import BBox, Detection as CoreDetection

if TYPE_CHECKING:  # pragma: no cover - type hints only, no runtime import
    from ai.detection import Detection as OldDetection
    from ai.detection import Keypoint


# ---------------------------------------------------------------------------
# Core conversion -- the two primary adapter functions
# ---------------------------------------------------------------------------


def to_core_detection(old: Any) -> CoreDetection:
    """Convert a legacy detection (or any compatible object) to a core
    :class:`~visioncore.core.detection.Detection`.

    The conversion maps the legacy ``(x1, y1, x2, y2)`` corner-format bbox
    to the core ``BBox(x, y, w, h)`` centre-format, and renames
    ``confidence`` → ``score``, ``label`` → ``class_name``.

    OBB polygons, pose keypoints, and track metadata are **not** carried
    over (the core model does not define them). Use
    :func:`extract_extras` to capture those separately for a lossless
    round-trip.

    Parameters:
        old: Any object with ``bbox`` (4-tuple), ``confidence``,
            ``class_id``, and ``label`` attributes. Typically an
            ``ai.detection.Detection`` instance.

    Returns:
        A new immutable :class:`CoreDetection`.

    Example:
        >>> from ai.detection import Detection as Old
        >>> old = Old(class_id=0, label="person", confidence=0.9,
        ...           bbox=(0.1, 0.2, 0.3, 0.6))
        >>> core = to_core_detection(old)
        >>> core.bbox
        BBox(x=0.2000, y=0.4000, w=0.2000, h=0.4000)
        >>> core.score
        0.9
    """
    x1: float
    y1: float
    x2: float
    y2: float
    x1, y1, x2, y2 = old.bbox

    return CoreDetection(
        bbox=BBox(
            x=(x1 + x2) * 0.5,
            y=(y1 + y2) * 0.5,
            w=x2 - x1,
            h=y2 - y1,
        ),
        score=float(old.confidence),
        class_id=int(old.class_id),
        class_name=str(old.label),
    )


def from_core_detection(
    core: CoreDetection,
    *,
    polygon: list[tuple[float, float]] | None = None,
    keypoints: list[Keypoint] | None = None,
    track_id: int | None = None,
    track_state: str = "normal",
) -> OldDetection:
    """Convert a core :class:`~visioncore.core.detection.Detection` back to
    a legacy :class:`ai.detection.Detection`.

    The core ``BBox(x, y, w, h)`` centre-format is mapped back to the
    legacy ``(x1, y1, x2, y2)`` corner-format. Optional keyword arguments
    restore OBB polygons, pose keypoints, and track metadata that were
    stripped during :func:`to_core_detection`.

    Parameters:
        core: The immutable core Detection to convert.
        polygon: Optional OBB polygon (4 corner points) to attach.
        keypoints: Optional pose keypoints list to attach.
        track_id: Optional tracker-assigned track id to attach.
        track_state: Optional track state string (default ``"normal"``).

    Returns:
        A new mutable :class:`ai.detection.Detection`.

    Note:
        ``ai.detection`` is imported lazily so that ``visioncore`` remains
        usable without the legacy application installed.

    Example:
        >>> from visioncore.core.detection import BBox, Detection as Core
        >>> core = Core(bbox=BBox(0.2, 0.4, 0.2, 0.4), score=0.9,
        ...             class_id=0, class_name="person")
        >>> old = from_core_detection(core)
        >>> old.bbox
        (0.1, 0.20000000000000004, 0.30000000000000004, 0.6)
        >>> old.label
        'person'
    """
    from ai.detection import Detection as OldDetection  # lazy import

    b: BBox = core.bbox
    half_w: float = b.w * 0.5
    half_h: float = b.h * 0.5

    return OldDetection(
        class_id=core.class_id,
        label=core.class_name,
        confidence=core.score,
        bbox=(b.x - half_w, b.y - half_h, b.x + half_w, b.y + half_h),
        polygon=polygon,
        keypoints=keypoints,
        track_id=track_id,
        track_state=track_state,
    )


# ---------------------------------------------------------------------------
# Extras handling -- lossless round-trip for OBB / Pose / track metadata
# ---------------------------------------------------------------------------


def extract_extras(old: Any) -> dict[str, Any]:
    """Extract OBB polygon, pose keypoints, and track metadata from a
    legacy detection into a plain dict.

    This complements :func:`to_core_detection` by capturing the fields
    that the core model does not represent. The returned dict can be
    passed to :func:`from_core_with_extras` to reconstruct a full legacy
    detection with no data loss.

    Parameters:
        old: Any object with optional ``polygon``, ``keypoints``,
            ``track_id``, and ``track_state`` attributes.

    Returns:
        A dict with keys ``"polygon"``, ``"keypoints"``, ``"track_id"``,
        and ``"track_state"``. Keys whose source value is ``None`` are
        still present with a ``None`` value, so the dict schema is stable.

    Example:
        >>> from ai.detection import Detection as Old
        >>> old = Old(class_id=0, label="box", confidence=0.8,
        ...           bbox=(0.1, 0.1, 0.3, 0.3),
        ...           polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)],
        ...           track_id=5)
        >>> extras = extract_extras(old)
        >>> sorted(extras.keys())
        ['keypoints', 'polygon', 'track_id', 'track_state']
        >>> extras["polygon"]
        [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    """
    extras: dict[str, Any] = {
        "polygon": None,
        "keypoints": None,
        "track_id": None,
        "track_state": "normal",
    }

    polygon_val: Any = getattr(old, "polygon", None)
    if polygon_val is not None:
        extras["polygon"] = list(polygon_val)

    keypoints_val: Any = getattr(old, "keypoints", None)
    if keypoints_val is not None:
        extras["keypoints"] = list(keypoints_val)

    track_id_val: Any = getattr(old, "track_id", None)
    if track_id_val is not None:
        extras["track_id"] = track_id_val

    extras["track_state"] = str(getattr(old, "track_state", "normal"))

    return extras


def from_core_with_extras(
    core: CoreDetection,
    extras: dict[str, Any] | None = None,
) -> OldDetection:
    """Convert a core Detection to a legacy Detection, restoring extras
    previously captured by :func:`extract_extras`.

    This is the inverse of the ``to_core_detection`` + ``extract_extras``
    pair, enabling a lossless round-trip for OBB and Pose detections.

    Parameters:
        core: The immutable core Detection.
        extras: Optional dict from :func:`extract_extras`. If ``None``,
            equivalent to an empty dict (all extras default).

    Returns:
        A new :class:`ai.detection.Detection` with all fields restored.

    Example:
        >>> from ai.detection import Detection as Old
        >>> from visioncore.core.detection import BBox, Detection as Core
        >>> old = Old(class_id=0, label="box", confidence=0.8,
        ...           bbox=(0.1, 0.1, 0.3, 0.3),
        ...           polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)])
        >>> core = to_core_detection(old)
        >>> extras = extract_extras(old)
        >>> restored = from_core_with_extras(core, extras)
        >>> restored.polygon == old.polygon
        True
    """
    extras = extras or {}

    return from_core_detection(
        core,
        polygon=extras.get("polygon"),
        keypoints=extras.get("keypoints"),
        track_id=extras.get("track_id"),
        track_state=extras.get("track_state", "normal"),
    )


# ---------------------------------------------------------------------------
# Batch helpers -- convenience for pipeline-style processing
# ---------------------------------------------------------------------------


def to_core_detections(old_list: list[Any]) -> list[CoreDetection]:
    """Convert a list of legacy detections to core Detections.

    Equivalent to ``[to_core_detection(d) for d in old_list]`` but
    semantically grouped for readability at call sites that process
    detection batches (e.g. the output of a detector backend).

    Parameters:
        old_list: List of objects compatible with :func:`to_core_detection`.

    Returns:
        List of :class:`CoreDetection`, preserving input order.
    """
    return [to_core_detection(d) for d in old_list]


def from_core_detections(
    core_list: list[CoreDetection],
) -> list[OldDetection]:
    """Convert a list of core Detections to legacy Detections.

    Equivalent to ``[from_core_detection(d) for d in core_list]``. Extras
    (polygon, keypoints, track) are not restored -- use
    :func:`from_core_with_extras` per-element if extras are needed.

    Parameters:
        core_list: List of core Detections.

    Returns:
        List of :class:`ai.detection.Detection`, preserving input order.
    """
    return [from_core_detection(d) for d in core_list]


__all__ = [
    "to_core_detection",
    "from_core_detection",
    "extract_extras",
    "from_core_with_extras",
    "to_core_detections",
    "from_core_detections",
]
