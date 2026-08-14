"""Processor plugins -- post-processing for detection results.

This module extracts post-processing logic from ``ai/inference.py`` into
configurable, composable plugins. It contains:

* :class:`ProcessorPlugin` -- abstract base class extending
  :class:`~visioncore.plugin.base.PluginInterface` with a
  ``process(detections, context)`` method.
* :class:`FilterProcessor` -- filters detections by label, confidence,
  and person visibility.
* :class:`BBoxProcessor` -- deduplicates detections by IoU, normalises
  bounding box coordinates.

Each processor is a ``PluginInterface`` subclass: it has ``name``,
``version``, ``load()``, ``run()``, ``shutdown()`` (from the plugin
contract) plus ``process()`` (the processor-specific method).

Scope (Milestone D9.3)
----------------------
D9.3 extracts post-processing from ``ai/inference.py``:

* ``_filter_display_detections`` → ``FilterProcessor``
* ``final_dedupe`` / ``_dict_iou`` → ``BBoxProcessor``

D9.3 deliberately does **not** contain:

* any detection model code;
* any head attribute / gesture attachment (those are model-dependent);
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Example
-------
    >>> from visioncore.plugin.processors import FilterProcessor, BBoxProcessor
    >>> fp = FilterProcessor()
    >>> fp.process([{"label": "person", "confidence": 0.9}], {})
    [{'label': 'person', 'confidence': 0.9}]
    >>> fp.process([{"label": "car", "confidence": 0.9}], {"person_enabled": False})
    [{'label': 'car', 'confidence': 0.9}]
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from visioncore.plugin.base import PluginInterface

__all__ = [
    "ProcessorPlugin",
    "FilterProcessor",
    "BBoxProcessor",
]


logger = logging.getLogger(__name__)


# ======================================================================
# ProcessorPlugin -- abstract base
# ======================================================================

class ProcessorPlugin(PluginInterface):
    """Abstract base class for processor plugins.

    A processor transforms a list of detection dicts. The
    ``process(detections, context)`` method is the processor-specific
    extension point; ``load()`` / ``run()`` / ``shutdown()`` are
    inherited from ``PluginInterface`` with default no-op behaviour.

    The ``context`` parameter is a plain dict (not a PipelineContext)
    for maximum composability. Typical keys:

    * ``"person_enabled"`` (bool) -- whether person detections are
      visible;
    * ``"frame"`` (np.ndarray) -- the current frame;
    * ``"frame_id"`` (int) -- the current frame number.

    Example:
        >>> class MyProcessor(ProcessorPlugin):
        ...     @property
        ...     def name(self): return "my"
        ...     @property
        ...     def version(self): return "0.1.0"
        ...     def process(self, detections, context=None):
        ...         return [d for d in detections if d.get("confidence", 0) > 0.5]
        >>> MyProcessor().process([{"confidence": 0.3}, {"confidence": 0.8}])
        [{'confidence': 0.8}]
    """

    @abstractmethod
    def process(
        self,
        detections: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Process a list of detection dicts.

        Parameters:
            detections: Input detections.
            context: Optional context dict with processing parameters.

        Returns:
            Processed detections (may be a subset or modified copy).
        """

    # ------------------------------------------------------------------
    # PluginInterface lifecycle (default no-ops)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """No-op. Subclasses may override to acquire resources."""
        pass

    def run(
        self,
        target_state: "visioncore.state.target_state.TargetState",
    ) -> "visioncore.state.target_state.TargetState":
        """Identity passthrough (PluginInterface contract)."""
        return target_state

    def shutdown(self) -> None:
        """No-op. Subclasses may override to release resources."""
        pass


# ======================================================================
# FilterProcessor -- label / confidence / person visibility filter
# ======================================================================

class FilterProcessor(ProcessorPlugin):
    """Filters detections by label, confidence, and person visibility.

    Extracted from ``ai/inference.py``'s ``_filter_display_detections``
    and ``_filter_inference_detections``. The processor:

    1. Removes person detections when ``person_enabled=False`` in
       context.
    2. Strips keypoints from person detections when skeleton/gesture
       features are disabled.
    3. Applies a minimum confidence floor.

    Attributes:
        _conf_floor: Minimum confidence floor (detections below this
            are dropped).

    Example:
    >>> fp = FilterProcessor(conf_floor=0.3)
    >>> fp.name
    'FilterProcessor'
        >>> fp.process([{"label": "person", "confidence": 0.9}], {})
        [{'label': 'person', 'confidence': 0.9}]
    """

    __slots__ = ("_conf_floor", "_loaded")

    def __init__(
        self,
        conf_floor: float = 0.0,
        *,
        name: str | None = None,
    ) -> None:
        """Construct a FilterProcessor.

        Parameters:
            conf_floor: Minimum confidence floor. Detections with
                confidence < conf_floor are dropped. 0.0 (default)
                means no confidence filtering.
            name: Optional plugin name (default ``"FilterProcessor"``).
        """
        self._name: str = name if name is not None else "FilterProcessor"
        self._conf_floor: float = conf_floor
        self._loaded: bool = False

    @property
    def name(self) -> str:
        """The plugin name."""
        return self._name

    @property
    def version(self) -> str:
        """The plugin version."""
        return "0.1.0"

    def load(self) -> None:
        self._loaded = True

    def shutdown(self) -> None:
        self._loaded = False

    def health_check(self) -> bool:
        return self._loaded

    def process(
        self,
        detections: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Filter detections by label, confidence, and person visibility.

        Parameters:
            detections: Input detections.
            context: Optional dict with keys:
                - ``"person_enabled"`` (bool, default True)
                - ``"skeleton_enabled"`` (bool, default False)

        Returns:
            Filtered detections.
        """
        ctx = context or {}
        person_enabled: bool = ctx.get("person_enabled", True)
        skeleton_enabled: bool = ctx.get("skeleton_enabled", False)

        filtered: list[dict[str, Any]] = []
        for det in detections:
            label = det.get("label", "")
            conf = float(det.get("confidence", 0.0))

            # Confidence floor
            if conf < self._conf_floor:
                continue

            # Person visibility
            if label == "person" and not person_enabled:
                continue

            # Strip keypoints when skeleton disabled
            if label == "person" and not skeleton_enabled and "keypoints" in det:
                det = dict(det)
                det["keypoints"] = []

            filtered.append(det)

        logger.debug("FilterProcessor.process: in=%d out=%d",
                     len(detections), len(filtered))
        return filtered


# ======================================================================
# BBoxProcessor -- deduplication and coordinate normalisation
# ======================================================================

class BBoxProcessor(ProcessorPlugin):
    """Deduplicates detections by IoU and normalises bounding boxes.

    Extracted from ``ai/inference.py``'s ``final_dedupe`` /
    ``_dict_iou`` / ``_tuple_iou``. The processor:

    1. Sorts detections by priority (track hits, confidence).
    2. Removes duplicates with IoU above a per-label threshold.
    3. Optionally normalises bbox coordinates from corner form
       ``(x1, y1, x2, y2)`` to centre form ``(cx, cy, w, h)``.

    Attributes:
        _iou_thresholds: Per-label IoU thresholds for deduplication.
        _normalise: Whether to normalise bbox coordinates.

    Example:
    >>> bp = BBoxProcessor()
    >>> bp.name
    'BBoxProcessor'
        >>> dets = [{"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}]
        >>> bp.process(dets)
        [{'label': 'person', 'confidence': 0.9, 'x1': 0.1, 'y1': 0.2, 'x2': 0.3, 'y2': 0.4}]
    """

    __slots__ = ("_iou_thresholds", "_normalise", "_loaded")

    def __init__(
        self,
        iou_thresholds: dict[str, float] | None = None,
        normalise: bool = False,
        *,
        name: str | None = None,
    ) -> None:
        """Construct a BBoxProcessor.

        Parameters:
            iou_thresholds: Per-label IoU thresholds for deduplication.
                Default: ``{"person": 0.55, "rectangle": 0.30,
                "hand_gesture": 0.35}``.
            normalise: If ``True``, convert bbox from corner form
                ``(x1, y1, x2, y2)`` to centre form ``(cx, cy, w, h)``.
            name: Optional plugin name (default ``"BBoxProcessor"``).
        """
        self._name: str = name if name is not None else "BBoxProcessor"
        self._iou_thresholds: dict[str, float] = (
            iou_thresholds if iou_thresholds is not None
            else {"person": 0.55, "rectangle": 0.30, "hand_gesture": 0.35}
        )
        self._normalise: bool = normalise
        self._loaded: bool = False

    @property
    def name(self) -> str:
        """The plugin name."""
        return self._name

    @property
    def version(self) -> str:
        """The plugin version."""
        return "0.1.0"

    def load(self) -> None:
        self._loaded = True

    def shutdown(self) -> None:
        self._loaded = False

    def health_check(self) -> bool:
        return self._loaded

    def process(
        self,
        detections: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Deduplicate detections and optionally normalise bboxes.

        Parameters:
            detections: Input detections with ``label``, ``confidence``,
                and ``x1/y1/x2/y2`` keys.
            context: Optional dict (unused by this processor).

        Returns:
            Deduplicated (and optionally normalised) detections.
        """
        # Sort by priority (track_hits descending, then confidence descending)
        ordered = sorted(detections, key=self._priority, reverse=True)

        kept: list[dict[str, Any]] = []
        for det in ordered:
            label = det.get("label", "")
            threshold = self._iou_thresholds.get(label)
            if threshold is not None:
                is_dup = any(
                    self._iou(det, existing) > threshold
                    for existing in kept
                    if existing.get("label") == label
                )
                if is_dup:
                    continue

            if self._normalise:
                det = self._to_centre_form(det)

            kept.append(det)

        logger.debug("BBoxProcessor.process: in=%d out=%d",
                     len(detections), len(kept))
        return kept

    @staticmethod
    def _priority(det: dict[str, Any]) -> tuple[float, float, float]:
        """Compute sort priority for a detection."""
        if det.get("label") == "rectangle":
            return (
                float(det.get("track_hits", 0)),
                -float(det.get("track_misses", 0)),
                float(det.get("confidence", 0.0)),
            )
        return (0.0, 0.0, float(det.get("confidence", 0.0)))

    @staticmethod
    def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
        """Compute IoU between two detections with x1/y1/x2/y2 keys."""
        ax1, ay1, ax2, ay2 = float(a.get("x1", 0)), float(a.get("y1", 0)), float(a.get("x2", 0)), float(a.get("y2", 0))
        bx1, by1, bx2, by2 = float(b.get("x1", 0)), float(b.get("y1", 0)), float(b.get("x2", 0)), float(b.get("y2", 0))
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _to_centre_form(det: dict[str, Any]) -> dict[str, Any]:
        """Convert bbox from corner form to centre form."""
        x1 = float(det.get("x1", 0))
        y1 = float(det.get("y1", 0))
        x2 = float(det.get("x2", 0))
        y2 = float(det.get("y2", 0))
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = x2 - x1
        h = y2 - y1
        result = dict(det)
        result["cx"] = cx
        result["cy"] = cy
        result["w"] = w
        result["h"] = h
        return result