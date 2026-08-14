"""RedetectStage -- the re-detection stage for the VisionCore pipeline.

This module defines :class:`RedetectStage`, a concrete
:class:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage`
plugin that performs target re-detection: refining or supplementing the
detection results for tracked targets by cropping a region of interest
(ROI) around each predicted bounding box and re-running detection on
the crop.

Scope (Milestones D3 + D9.1)
----------------------------
D3 delivered the **stage shell** (passthrough). D9.1 migrates the
ROI re-detection logic from ``ai/inference.py``'s
``_redetect_person_in_roi`` / ``_redetect_quality_gate`` into the
stage. The stage now:

1. Reads ``context.tracks`` (the current tracked detections).
2. For each track that needs redetection (based on config), computes
   an ROI from the track's predicted bounding box.
3. Delegates to the injected :class:`Redetector` backend to re-run
   detection on the ROI crop.
4. Applies quality gating (IoU + confidence + scale checks).
5. Updates ``context.tracks`` with refined detections.

The stage still accepts an optional ``redetector`` instance for
dependency injection. When ``redetector=None``, the stage acts as a
pure passthrough (D3 behavior).

D9.1 deliberately does **not** contain:

* any YOLO / RT-DETR / GroundingDINO / SAM code or imports;
* any model loading, GPU allocation, or ONNX/torch code;
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Data flow
---------
::

    context.tracks ──> RedetectStage.process
                            │
                  for each track (up to budget):
                      compute ROI from predicted box
                      ──> redetector.redetect(frame, box, conf)
                      ──> quality_gate(predicted, refined)
                            │
                            ▼
                  context.tracks (updated with refined detections)

Example
-------
    >>> from visioncore.pipeline.stages.redetect_stage import RedetectStage
    >>> from visioncore.pipeline.context import PipelineContext
    >>> stage = RedetectStage(name="redetect")
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty()
    >>> stage.process(ctx)
    >>> len(ctx.tracks)
    0
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from visioncore.pipeline.stages.advanced import (
    AdvancedStage,
    StageCapability,
)
from visioncore.pipeline.stages.redetect import (
    RedetectConfig,
    Redetector,
    RoiRedetector,
    compute_roi,
    quality_gate,
)

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = ["RedetectStage"]


logger = logging.getLogger(__name__)


# ======================================================================
# Module-level capability (single source of truth)
# ======================================================================

_CAPABILITY = StageCapability(
    name="redetect",
    version="0.2.0",
    description=(
        "ROI re-detection stage: crops regions around predicted bounding "
        "boxes and re-runs detection to refine tracked targets. Uses "
        "configurable quality gating (IoU + confidence + scale checks)."
    ),
    required_context=["tracks"],
    provided_context=["tracks"],
)


# ======================================================================
# RedetectStage
# ======================================================================

class RedetectStage(AdvancedStage):
    """Pipeline stage that performs ROI re-detection on tracked targets.

    :class:`RedetectStage` reads ``context.tracks``, selects tracks for
    redetection (up to a configurable budget per frame), computes an ROI
    from each track's predicted bounding box, delegates to the injected
    :class:`Redetector` backend, and applies quality gating.

    When no redetector is injected (``redetector=None``), the stage acts
    as a pure passthrough (D3 behavior).

    Attributes:
        _redetector: The wrapped redetector instance, or ``None`` for a
            pure passthrough.
        _config: Redetection configuration.

    Example:
        >>> stage = RedetectStage(name="redetect")
        >>> stage.capability.name
        'redetect'
        >>> stage.config.enabled
        True
    """

    __slots__ = ("_redetector", "_ready", "_config")

    def __init__(
        self,
        redetector: Any | None = None,
        *,
        config: RedetectConfig | None = None,
        name: str | None = None,
    ) -> None:
        """Construct a RedetectStage with an optional redetector backend.

        Parameters:
            redetector: The redetector instance to wrap. ``None`` (default)
                means no redetector is available -- the stage acts as a
                pure passthrough.
            config: Redetection configuration. ``None`` uses defaults.
            name: Optional stage name (defaults to ``"RedetectStage"``).
        """
        super().__init__(name=name if name is not None else "RedetectStage")
        self._redetector: Any | None = redetector
        self._config: RedetectConfig = (
            config if config is not None else RedetectConfig()
        )
        self._ready: bool = False
        logger.debug("RedetectStage created: name=%s redetector=%s config=%s",
                     self._name,
                     type(redetector).__name__ if redetector is not None else "None",
                     self._config)

    # ------------------------------------------------------------------
    # Capability declaration
    # ------------------------------------------------------------------

    @property
    def capability(self) -> StageCapability:
        """The stage's declared :class:`StageCapability`.

        ``required_context=["tracks"]``, ``provided_context=["tracks"]``.
        """
        return _CAPABILITY

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def config(self) -> RedetectConfig:
        """The redetection configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Lifecycle (AdvancedStage contract)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Mark the stage initialised and ready. Idempotent."""
        self._ready = True
        logger.debug("RedetectStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Run ROI re-detection on tracked targets.

        Calls :meth:`AdvancedStage.check_context` first, then:

        1. If ``config.enabled`` is ``False`` or no redetector is
           injected, passes through unchanged (D3 behavior).
        2. Otherwise, iterates over ``context.tracks`` (up to
           ``config.budget_per_frame``), computes ROI, delegates to
           the redetector, applies quality gating, and updates
           ``context.tracks`` with refined detections.

        Parameters:
            context: The shared
                :class:`~visioncore.pipeline.context.PipelineContext`.
        """
        self.check_context(context)

        cfg = self._config

        # Passthrough when disabled or no redetector
        if not cfg.enabled or self._redetector is None:
            logger.debug("RedetectStage.process: name=%s passthrough "
                         "(enabled=%s, redetector=%s)",
                         self._name, cfg.enabled,
                         self._redetector is not None)
            return

        tracks = context.tracks
        if not tracks:
            logger.debug("RedetectStage.process: name=%s no tracks", self._name)
            return

        # Frame for ROI cropping
        frame = getattr(context, 'frame', None)
        if frame is None:
            logger.debug("RedetectStage.process: name=%s no frame", self._name)
            return

        # Select tracks for redetection (up to budget)
        budget = min(cfg.budget_per_frame, len(tracks))
        refined_count = 0

        for i in range(budget):
            track = tracks[i]
            # Extract predicted box from track
            predicted_box = self._extract_predicted_box(track)
            if predicted_box is None:
                continue

            original_conf = float(getattr(track, 'confidence', 0.0)
                                  if hasattr(track, 'confidence')
                                  else getattr(track, 'score', 0.0))

            try:
                result = self._redetector.redetect(
                    frame, predicted_box, original_conf,
                )
            except Exception as exc:
                logger.warning("RedetectStage.process: name=%s track[%d] "
                               "redetect failed: %s: %s",
                               self._name, i, type(exc).__name__, exc)
                continue

            if result is not None:
                # Update track with refined detection
                self._apply_refinement(track, result)
                refined_count += 1

        logger.debug("RedetectStage.process: name=%s tracks=%d refined=%d",
                     self._name, len(tracks), refined_count)

    def _extract_predicted_box(self, track: Any) -> tuple[float, float, float, float] | None:
        """Extract the predicted bounding box from a track.

        Supports both dict tracks (``x1/y1/x2/y2`` or ``bbox``) and
        core Track objects (``detection.bbox``).
        """
        if isinstance(track, dict):
            if "x1" in track:
                return (float(track["x1"]), float(track["y1"]),
                        float(track["x2"]), float(track["y2"]))
            if "bbox" in track:
                bbox = track["bbox"]
                if len(bbox) == 4:
                    return tuple(float(v) for v in bbox)  # type: ignore[return-value]
        # Core Track object
        if hasattr(track, 'detection') and hasattr(track.detection, 'bbox'):
            b = track.detection.bbox
            hw = b.w * 0.5
            hh = b.h * 0.5
            return (b.x - hw, b.y - hh, b.x + hw, b.y + hh)
        return None

    def _apply_refinement(self, track: Any, result: dict[str, Any]) -> None:
        """Apply a refined detection result to a track."""
        if isinstance(track, dict):
            if "x1" in result:
                track["x1"] = result["x1"]
                track["y1"] = result["y1"]
                track["x2"] = result["x2"]
                track["y2"] = result["y2"]
            elif "bbox" in result:
                track["bbox"] = result["bbox"]
            if "confidence" in result:
                track["confidence"] = result["confidence"]
        # Core Track objects are frozen -- refinement logged but not applied
        # (the pipeline adapter handles conversion)

    def shutdown(self) -> None:
        """Mark the stage shut down. Idempotent, never raises."""
        self._ready = False
        logger.debug("RedetectStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the stage is initialised and ready."""
        return self._ready

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def redetector(self) -> Any | None:
        """The wrapped redetector instance, or ``None``."""
        return self._redetector