"""Migration wrapper for Legacy → Pipeline mode transition (Milestone D9).

This module provides :class:`MigrationPipeline`, a reusable wrapper that
encapsulates the dual-mode (Legacy vs Pipeline) execution pattern used by
``ai/inference.py``'s :class:`InferWorker`. It wraps the same backends
(detector, tracker, target manager, event bus) through Pipeline adapters
and stages, making the migration pattern testable and reusable without the
full ``InferWorker`` QThread machinery.

Scope (Milestone D9)
--------------------
D9 delivers the **migration infrastructure**: a reusable wrapper, tests
verifying dual-mode consistency, and documentation. It deliberately does
**not** contain:

* any modification to ``ai/inference.py`` (InferWorker is unchanged);
* any modification to ``camera/`` or ``gui/``;
* any concrete model loading (backends are injected).

The migration strategy is:
1. Legacy mode (``pipeline_enabled=False``): the existing inline code path
   in ``InferWorker.run()`` runs unchanged.
2. Pipeline mode (``pipeline_enabled=True``): frames flow through
   ``Pipeline(DetectorStage → TrackerStage → TargetStage → EventStage)``,
   using adapters that delegate to the **same** backend methods as Legacy.
3. Post-processing (head attributes, gestures, rectangle detection, health
   monitor) is applied identically in both modes.
4. Results are ``list[dict]`` in both cases, consumed by
   ``detection_ready.emit`` and ``_bypass_update_targets``.

Example
-------
    >>> from visioncore.pipeline.migration import MigrationPipeline
    >>> mp = MigrationPipeline()
    >>> mp.pipeline_enabled
    False
    >>> mp.set_pipeline_enabled(True)
    >>> mp.pipeline_enabled
    True
"""

from __future__ import annotations

import logging
from typing import Any

from visioncore.pipeline.base import PipelineStage

__all__ = ["MigrationPipeline", "MigrationPipelineError"]


logger = logging.getLogger(__name__)


class MigrationPipelineError(RuntimeError):
    """Structured exception signalling a migration pipeline failure."""


class MigrationPipeline:
    """Reusable dual-mode (Legacy / Pipeline) execution wrapper.

    :class:`MigrationPipeline` holds references to the backends that
    ``InferWorker`` uses (detector, tracker, target manager, event bus)
    and provides a ``run_frame()`` method that routes through either the
    Legacy inline path or the Pipeline path, depending on
    :attr:`pipeline_enabled`.

    In Pipeline mode, the same backends are wrapped by
    ``visioncore.pipeline.adapters`` and run through
    ``Pipeline(DetectorStage → TrackerStage → TargetStage → EventStage)``.
    In Legacy mode, a caller-provided ``legacy_process`` callable is
    invoked directly.

    Attributes:
        _detector: The detector backend (e.g. YOLO model wrapper).
        _tracker: The tracker backend (e.g. DetectionTracker).
        _target_manager: The target manager instance.
        _event_bus: The event bus instance.
        _pipeline_enabled: Whether Pipeline mode is active.
        _legacy_process: Callable ``(frame, slot_id, frame_id, submitted_ts) -> list[dict]``
            that runs the Legacy inline path.

    Example:
        >>> mp = MigrationPipeline(
        ...     detector=None, tracker=None,
        ...     target_manager=None, event_bus=None,
        ... )
        >>> mp.pipeline_enabled
        False
        >>> mp.set_pipeline_enabled(True)
        >>> mp.pipeline_enabled
        True
    """

    __slots__ = (
        "_detector",
        "_tracker",
        "_target_manager",
        "_event_bus",
        "_pipeline_enabled",
        "_legacy_process",
    )

    def __init__(
        self,
        detector: Any = None,
        tracker: Any = None,
        target_manager: Any = None,
        event_bus: Any = None,
        legacy_process: Any = None,
    ) -> None:
        """Construct a MigrationPipeline with backend references.

        Parameters:
            detector: The detector backend instance.
            tracker: The tracker backend instance.
            target_manager: The target manager instance.
            event_bus: The event bus instance.
            legacy_process: Callable implementing the Legacy inline path.
                Signature: ``(frame, slot_id, frame_id, submitted_ts) -> list[dict]``.
        """
        self._detector = detector
        self._tracker = tracker
        self._target_manager = target_manager
        self._event_bus = event_bus
        self._pipeline_enabled: bool = False
        self._legacy_process = legacy_process
        logger.debug("MigrationPipeline created: pipeline_enabled=%s",
                     self._pipeline_enabled)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def pipeline_enabled(self) -> bool:
        """Whether Pipeline mode is active."""
        return self._pipeline_enabled

    def set_pipeline_enabled(self, enabled: bool) -> None:
        """Switch between Legacy and Pipeline mode.

        Parameters:
            enabled: ``True`` for Pipeline mode, ``False`` for Legacy.
        """
        self._pipeline_enabled = bool(enabled)
        logger.info("MigrationPipeline: pipeline mode %s",
                     "ENABLED" if enabled else "disabled (legacy)")

    # ------------------------------------------------------------------
    # Backend access (for introspection / testing)
    # ------------------------------------------------------------------

    @property
    def detector(self) -> Any:
        """The detector backend instance."""
        return self._detector

    @property
    def tracker(self) -> Any:
        """The tracker backend instance."""
        return self._tracker

    @property
    def target_manager(self) -> Any:
        """The target manager instance."""
        return self._target_manager

    @property
    def event_bus(self) -> Any:
        """The event bus instance."""
        return self._event_bus

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_frame(
        self,
        frame: Any,
        slot_id: int = 0,
        frame_id: int = 0,
        submitted_ts: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Process one frame through the active mode.

        Parameters:
            frame: The frame data (numpy array or Pipeline frame object).
            slot_id: Camera slot identifier.
            frame_id: Frame sequence number.
            submitted_ts: Frame submission timestamp.

        Returns:
            ``list[dict]`` of detection results (same format in both modes).
        """
        if self._pipeline_enabled:
            return self._run_pipeline(frame, slot_id, frame_id, submitted_ts)
        return self._run_legacy(frame, slot_id, frame_id, submitted_ts)

    def _run_legacy(
        self,
        frame: Any,
        slot_id: int,
        frame_id: int,
        submitted_ts: float,
    ) -> list[dict[str, Any]]:
        """Run the Legacy inline path via the injected callable."""
        if self._legacy_process is None:
            logger.warning(
                "MigrationPipeline: legacy_process not set, returning []"
            )
            return []
        return self._legacy_process(frame, slot_id, frame_id, submitted_ts)

    def _run_pipeline(
        self,
        frame: Any,
        slot_id: int,
        frame_id: int,
        submitted_ts: float,
    ) -> list[dict[str, Any]]:
        """Run the Pipeline path using adapters + Pipeline stages."""
        from visioncore.pipeline import Pipeline, PipelineContext
        from visioncore.pipeline.adapters import (
            InferWorkerDetectorAdapter,
            InferWorkerEventBusAdapter,
            InferWorkerTargetManagerAdapter,
            InferWorkerTrackerAdapter,
        )
        from visioncore.pipeline.stages import (
            DetectorStage,
            EventStage,
            TargetStage,
            TrackerStage,
        )

        # Build adapters wrapping the same backends
        det_adapter = InferWorkerDetectorAdapter(self)
        det_adapter.set_frame_image(frame)
        trk_adapter = InferWorkerTrackerAdapter(self, slot_id)
        trk_adapter.set_frame_image(frame)
        tgt_adapter = InferWorkerTargetManagerAdapter(self, slot_id)
        evt_adapter = InferWorkerEventBusAdapter(self)

        # Build Pipeline
        p = Pipeline()
        p.add_stage(DetectorStage(det_adapter, name="detect"))
        p.add_stage(TrackerStage(trk_adapter, name="track"))
        p.add_stage(TargetStage(tgt_adapter, name="targets"))
        p.add_stage(EventStage(evt_adapter, name="events"))

        # Build Context and run
        ctx = PipelineContext.empty(timestamp=submitted_ts)
        ctx.metadata["slot_id"] = slot_id
        ctx.metadata["frame_id"] = frame_id

        try:
            p.initialize()
            p.run(ctx)
        except Exception as exc:
            logger.warning(
                "MigrationPipeline: pipeline error (falling back to []): "
                "%s: %s", type(exc).__name__, exc,
            )
            return []
        finally:
            try:
                p.shutdown()
            except Exception:
                pass

        # Extract tracked detections from the tracker adapter
        detections: list[dict[str, Any]] = trk_adapter.last_tracked_dicts
        logger.debug(
            "MigrationPipeline._run_pipeline: slot=%d frame=%d "
            "detections=%d", slot_id, frame_id, len(detections),
        )
        return detections