"""ROI Redetection -- crop-and-rerun logic for the VisionCore pipeline.

This module implements the core ROI re-detection algorithm extracted from
``ai/inference.py``'s ``_redetect_person_in_roi`` / ``_redetect_quality_gate``
methods. It provides:

* :class:`RedetectConfig` -- configuration for redetection behavior.
* :class:`Redetector` -- abstract base class (the injection point for the
  stage).
* :class:`RoiRedetector` -- concrete implementation that computes an ROI
  from a predicted bounding box, crops the frame, delegates to an injected
  detector callable, and applies quality gating.
* :func:`compute_roi` -- pure-function ROI computation (frame-size
  independent, testable).
* :func:`quality_gate` -- pure-function quality gate (IoU + confidence +
  scale checks, testable).

Scope (Milestone D9.1)
----------------------
D9.1 migrates the ROI re-detection logic from ``ai/inference.py``'s inline
code into a testable, config-driven Pipeline stage. It deliberately does
**not** contain:

* any YOLO / RT-DETR / torch / onnx model code (the detector callable is
  injected);
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Example
-------
    >>> from visioncore.pipeline.stages.redetect import (
    ...     RedetectConfig, compute_roi, quality_gate,
    ... )
    >>> cfg = RedetectConfig()
    >>> roi = compute_roi((0.3, 0.4, 0.5, 0.6), frame_w=640, frame_h=480, pad=cfg.region_expand)
    >>> roi.x1 >= 0
    True
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, NamedTuple

__all__ = [
    "RedetectConfig",
    "Roi",
    "Redetector",
    "RoiRedetector",
    "compute_roi",
    "quality_gate",
]


logger = logging.getLogger(__name__)


# ======================================================================
# Configuration
# ======================================================================

@dataclass(frozen=True)
class RedetectConfig:
    """Configuration for ROI re-detection.

    Attributes:
        enabled: Master switch. When ``False``, redetection is skipped
            entirely.
        interval: Run redetection every N frames (1 = every frame).
        region_expand: Fraction of the predicted box size used as padding
            around the ROI (0.15 = 15% padding on each side).
        min_crop_pixels: Minimum crop dimension in pixels (smaller crops
            are skipped).
        max_crop_area_ratio: Maximum crop area as a fraction of frame area
            (larger crops are skipped -- they degenerate to full-frame
            inference).
        conf_floor: Minimum confidence floor for redetection inference.
        conf_ratio: Minimum ratio of refined confidence to original
            confidence.
        iou_floor: Minimum IoU between predicted and refined box when both
            confidence and IoU are low.
        scale_min: Minimum allowed scale ratio (refined / predicted width
            or height).
        scale_max: Maximum allowed scale ratio.
        budget_per_frame: Maximum number of tracks to redetect per frame.
    """

    enabled: bool = True
    interval: int = 1
    region_expand: float = 0.15
    min_crop_pixels: int = 24
    max_crop_area_ratio: float = 0.20
    conf_floor: float = 0.15
    conf_ratio: float = 0.6
    iou_floor: float = 0.35
    scale_min: float = 0.5
    scale_max: float = 1.6
    budget_per_frame: int = 1


# ======================================================================
# ROI computation
# ======================================================================

class Roi(NamedTuple):
    """An axis-aligned region of interest in pixel coordinates.

    Attributes:
        x1: Left edge (inclusive, pixels).
        y1: Top edge (inclusive, pixels).
        x2: Right edge (exclusive, pixels).
        y2: Bottom edge (exclusive, pixels).
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """ROI width in pixels."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """ROI height in pixels."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        """ROI area in square pixels."""
        return max(0, self.width * self.height)


def compute_roi(
    box: tuple[float, float, float, float],
    *,
    frame_w: int,
    frame_h: int,
    pad: float = 0.15,
) -> Roi:
    """Compute a padded ROI from a normalised bounding box.

    The box is ``(x1, y1, x2, y2)`` in normalised ``[0, 1]`` coordinates.
    The ROI is expanded by ``pad * box_dimension`` on each side, then
    clamped to ``[0, frame_w) × [0, frame_h)``.

    Parameters:
        box: Normalised ``(x1, y1, x2, y2)`` bounding box.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        pad: Padding fraction (0.15 = 15% of box width/height on each
            side).

    Returns:
        A :class:`Roi` in pixel coordinates.

    Raises:
        ValueError: If ``frame_w`` or ``frame_h`` is non-positive.
    """
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"frame dimensions must be positive: {frame_w}x{frame_h}")

    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    bw = x2 - x1
    bh = y2 - y1

    px_cx = cx * frame_w
    px_cy = cy * frame_h
    px_bw = bw * frame_w
    px_bh = bh * frame_h
    pad_x = px_bw * pad
    pad_y = px_bh * pad

    roi_x1 = max(0, int(px_cx - px_bw * 0.5 - pad_x))
    roi_y1 = max(0, int(px_cy - px_bh * 0.5 - pad_y))
    roi_x2 = min(frame_w, int(px_cx + px_bw * 0.5 + pad_x))
    roi_y2 = min(frame_h, int(px_cy + px_bh * 0.5 + pad_y))

    return Roi(roi_x1, roi_y1, roi_x2, roi_y2)


# ======================================================================
# Quality gate
# ======================================================================

def _box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute IoU between two ``(x1, y1, x2, y2)`` boxes."""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def quality_gate(
    predicted_box: tuple[float, float, float, float],
    refined_box: tuple[float, float, float, float],
    original_confidence: float,
    *,
    conf_floor: float = 0.15,
    conf_ratio: float = 0.6,
    iou_floor: float = 0.35,
    scale_min: float = 0.5,
    scale_max: float = 1.6,
    refined_confidence: float = 0.0,
) -> bool:
    """Apply quality gating to a redetection result.

    Mirrors ``ai/inference.py``'s ``_redetect_quality_gate``: checks IoU,
    confidence ratio, and scale mutation.

    Parameters:
        predicted_box: The original predicted ``(x1, y1, x2, y2)``.
        refined_box: The refined ``(x1, y1, x2, y2)`` from redetection.
        original_confidence: The original track's confidence.
        conf_floor: Absolute confidence floor.
        conf_ratio: Minimum ratio of refined to original confidence.
        iou_floor: Minimum IoU when both confidences are low.
        scale_min: Minimum scale ratio (refined / predicted).
        scale_max: Maximum scale ratio (refined / predicted).
        refined_confidence: The refined detection's confidence.

    Returns:
        ``True`` if the refined box passes quality checks.
    """
    iou = _box_iou(predicted_box, refined_box)
    conf = refined_confidence

    # IoU + confidence dual threshold
    if conf < max(conf_floor, original_confidence * conf_ratio) and iou < iou_floor:
        return False

    # Scale mutation check
    pred_w = max(1e-5, predicted_box[2] - predicted_box[0])
    pred_h = max(1e-5, predicted_box[3] - predicted_box[1])
    ref_w = max(1e-5, refined_box[2] - refined_box[0])
    ref_h = max(1e-5, refined_box[3] - refined_box[1])

    if ref_w < pred_w * scale_min or ref_w > pred_w * scale_max:
        return False
    if ref_h < pred_h * scale_min or ref_h > pred_h * scale_max:
        return False

    return True


# ======================================================================
# Redetector ABC
# ======================================================================

class Redetector(ABC):
    """Abstract base class for redetection backends.

    A redetector takes a frame and a predicted bounding box, crops an ROI,
    runs detection on the crop, and returns the best refined detection
    (or ``None`` if redetection fails).

    Example:
        >>> class MyRedetector(Redetector):
        ...     def redetect(self, frame, predicted_box, original_confidence=0.0):
        ...         return None
        >>> MyRedetector().redetect(None, (0.1, 0.2, 0.3, 0.4)) is None
        True
    """

    @abstractmethod
    def redetect(
        self,
        frame: Any,
        predicted_box: tuple[float, float, float, float],
        original_confidence: float = 0.0,
    ) -> dict[str, Any] | None:
        """Run redetection on the ROI around ``predicted_box``.

        Parameters:
            frame: The frame data (numpy array or Pipeline frame object).
            predicted_box: Normalised ``(x1, y1, x2, y2)`` predicted box.
            original_confidence: The original track's confidence.

        Returns:
            A detection dict with ``bbox``/``confidence``/``label`` keys,
            or ``None`` if redetection failed.
        """

    def redetect_batch(
        self,
        frame: Any,
        entries: list[tuple[tuple[float, float, float, float], float]],
    ) -> list[dict[str, Any] | None]:
        """Run redetection on multiple predicted boxes.

        Default implementation calls :meth:`redetect` for each entry.
        Subclasses may override for batch-optimised inference.

        Parameters:
            frame: The frame data.
            entries: List of ``(predicted_box, original_confidence)`` tuples.

        Returns:
            List of detection dicts (or ``None``) aligned with ``entries``.
        """
        return [
            self.redetect(frame, box, conf) for box, conf in entries
        ]


# ======================================================================
# RoiRedetector -- concrete implementation
# ======================================================================

class RoiRedetector(Redetector):
    """Crop-and-rerun redetector using an injected detector callable.

    :class:`RoiRedetector` extracts the ROI redetection logic from
    ``ai/inference.py``'s ``_redetect_person_in_roi`` and makes it
    testable and config-driven.

    The detector callable signature:
    ``(roi_image: np.ndarray, conf_threshold: float) -> list[dict]``

    Each returned dict must contain at least ``confidence`` and ``label``
    keys; ``x1/y1/x2/y2`` in ROI-normalised coordinates are used for
    coordinate remapping.

    Attributes:
        _detector: The injected detector callable.
        _config: Redetection configuration.

    Example:
        >>> def mock_detector(roi, conf):
        ...     return [{"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9,
        ...              "confidence": 0.85, "label": "person"}]
        >>> rd = RoiRedetector(mock_detector)
        >>> rd.config.enabled
        True
    """

    __slots__ = ("_detector", "_config")

    def __init__(
        self,
        detector: Any,
        config: RedetectConfig | None = None,
    ) -> None:
        """Construct a RoiRedetector.

        Parameters:
            detector: Callable ``(roi_image, conf_threshold) -> list[dict]``.
            config: Redetection configuration. ``None`` uses defaults.
        """
        self._detector = detector
        self._config = config if config is not None else RedetectConfig()

    @property
    def config(self) -> RedetectConfig:
        """The redetection configuration."""
        return self._config

    def redetect(
        self,
        frame: Any,
        predicted_box: tuple[float, float, float, float],
        original_confidence: float = 0.0,
    ) -> dict[str, Any] | None:
        """Run ROI redetection.

        1. Compute ROI from ``predicted_box`` + ``config.region_expand``.
        2. Validate crop size.
        3. Crop frame to ROI.
        4. Run detector on crop with lowered confidence threshold.
        5. Find best person detection.
        6. Remap coordinates from ROI to frame space.
        7. Apply quality gate.

        Returns:
            A detection dict in frame-normalised coordinates, or ``None``.
        """
        cfg = self._config

        if frame is None:
            return None

        # Get frame dimensions (numpy array or Pipeline frame)
        if hasattr(frame, 'shape'):
            frame_h, frame_w = frame.shape[:2]
        elif hasattr(frame, 'image') and hasattr(frame.image, 'shape'):
            frame_h, frame_w = frame.image.shape[:2]
            frame = frame.image
        else:
            return None

        if frame_w <= 0 or frame_h <= 0:
            return None

        # Compute ROI
        roi = compute_roi(
            predicted_box,
            frame_w=frame_w,
            frame_h=frame_h,
            pad=cfg.region_expand,
        )

        # Validate crop size
        if roi.width < cfg.min_crop_pixels or roi.height < cfg.min_crop_pixels:
            return None
        if roi.area > (frame_w * frame_h) * cfg.max_crop_area_ratio:
            return None

        # Crop frame
        roi_image = frame[roi.y1:roi.y2, roi.x1:roi.x2]
        if roi_image.size == 0:
            return None

        # Run detector with lowered confidence
        redetect_conf = max(cfg.conf_floor, original_confidence * cfg.conf_ratio)
        try:
            results = self._detector(roi_image, redetect_conf)
        except Exception as exc:
            logger.warning("RoiRedetector: detector failed: %s: %s",
                           type(exc).__name__, exc)
            return None

        if not results:
            return None

        # Find best person detection
        best_det = None
        best_conf = 0.0

        for det in results:
            label = det.get("label", "")
            if label != "person":
                continue
            conf = float(det.get("confidence", 0.0))
            if conf <= best_conf:
                continue
            best_conf = conf

            # Remap ROI-normalised coords to frame-normalised coords
            det_x1 = det.get("x1", det.get("bbox", (0,))[0] if "bbox" in det else 0)
            det_y1 = det.get("y1", det.get("bbox", (0, 0))[1] if "bbox" in det else 0)
            det_x2 = det.get("x2", det.get("bbox", (0, 0, 0))[2] if "bbox" in det else 0)
            det_y2 = det.get("y2", det.get("bbox", (0, 0, 0, 0))[3] if "bbox" in det else 0)

            best_det = {
                "x1": float(det_x1 * roi.width / frame_w + roi.x1 / frame_w),
                "y1": float(det_y1 * roi.height / frame_h + roi.y1 / frame_h),
                "x2": float(det_x2 * roi.width / frame_w + roi.x1 / frame_w),
                "y2": float(det_y2 * roi.height / frame_h + roi.y1 / frame_h),
                "confidence": conf,
                "label": "person",
            }

        # Quality gate
        if best_det is not None:
            refined_box = (best_det["x1"], best_det["y1"], best_det["x2"], best_det["y2"])
            if not quality_gate(
                predicted_box,
                refined_box,
                original_confidence,
                conf_floor=cfg.conf_floor,
                conf_ratio=cfg.conf_ratio,
                iou_floor=cfg.iou_floor,
                scale_min=cfg.scale_min,
                scale_max=cfg.scale_max,
                refined_confidence=best_conf,
            ):
                best_det = None

        return best_det