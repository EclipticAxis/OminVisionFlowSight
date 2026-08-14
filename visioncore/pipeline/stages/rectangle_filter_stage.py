"""RectangleStage / FilterStage / HealthStage -- box normalisation and health stages.

This module defines three cohesive
:class:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage`
plugins that run after the target-management stage and post-process each
:class:`~visioncore.state.target_state.TargetState` snapshot:

* :class:`RectangleStage` -- derives an axis-aligned bounding box in
  corner form ``rect=(x1, y1, x2, y2)`` from the centre/size
  representation (``cx``/``cy``/``width``/``height``) and writes it into
  ``TargetState.rect``.
* :class:`FilterStage` -- the extension point for **target filtering**.
  It evaluates a predicate over ``context.target_states`` (the D5 spec
  example drops snapshots with ``confidence < 0``) but, as a D5
  placeholder, deliberately deletes nothing. Real thresholding /
  suppression logic arrives in a later milestone with an injected
  filterer.
* :class:`HealthStage` -- writes a placeholder health score
  ``health=1.0`` into ``TargetState.health`` for every snapshot, the
  extension point for checking whether a target track is healthy.

Scope (Milestone D5)
--------------------
D5 delivers **only** the stage shells -- no actual filtering or health
monitoring algorithm. Rectangle/Filter/Health each accept an algorithm
instance in their constructor (**held, never called**). As with D2-D4,
real backends arrive in later milestones.

D5 deliberately does **not** contain:

* any filtering / box-normalisation / health-monitoring model;
* any call to a concrete image-processing function;
* any model loading, GPU allocation, or ONNX/torch code;
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Data flow
---------
::

    context.target_states ──> RectangleStage.process
                                    │
                          for each snapshot:
                              x1,y2 = cx ∓ width/2 ; y1,y2 = cy ∓ height/2
                              snapshot.copy_with(rect=(x1, y1, x2, y2))
                                    │
                                    ▼
                          context.target_states (REPLACE)

    context.target_states ──> FilterStage.process
                                    │
                          predicate evaluated (confidence < 0)
                          ── D5 placeholder: nothing deleted ──

    context.target_states ──> HealthStage.process
                                    │
                          for each snapshot:
                              snapshot.copy_with(health=1.0)
                                    │
                                    ▼
                          context.target_states (REPLACE)

``RectangleStage`` and ``HealthStage`` use **REPLACE** semantics on
``context.target_states`` (clear + extend with derived snapshots),
consistent with
:class:`~visioncore.pipeline.stages.target_stage.TargetStage`. All three
declare ``required_context=["target_states"]`` /
``provided_context=["target_states"]``.

Example
-------
    >>> from visioncore.pipeline.stages.rectangle_filter_stage import (
    ...     RectangleStage, FilterStage, HealthStage,
    ... )
    >>> from visioncore.pipeline.context import PipelineContext
    >>> rectangle = RectangleStage()
    >>> rectangle.initialize()
    >>> ctx = PipelineContext.empty()
    >>> rectangle.process(ctx)
    >>> len(ctx.target_states)
    0
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from visioncore.pipeline.stages.advanced import (
    AdvancedStage,
    StageCapability,
)
from visioncore.state.target_state import TargetState

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = ["RectangleStage", "FilterStage", "HealthStage"]


logger = logging.getLogger(__name__)


# ======================================================================
# Module-level capabilities (single source of truth)
# ======================================================================

_RECTANGLE_CAPABILITY = StageCapability(
    name="rectangle",
    version="0.1.0",
    description=(
        "Bounding-box normalisation stage placeholder: derives "
        "rect=(x1, y1, x2, y2) from cx/cy/width/height on each target "
        "snapshot until a real box builder is injected."
    ),
    required_context=["target_states"],
    provided_context=["target_states"],
)

_FILTER_CAPABILITY = StageCapability(
    name="filter",
    version="0.1.0",
    description=(
        "Target-filtering stage placeholder: evaluates a predicate over "
        "context.target_states (drops targets with confidence < 0) but "
        "deletes nothing until a real filter is injected."
    ),
    required_context=["target_states"],
    provided_context=["target_states"],
)

_HEALTH_CAPABILITY = StageCapability(
    name="health",
    version="0.1.0",
    description=(
        "Health-check stage placeholder: sets TargetState.health=1.0 on "
        "each target snapshot until a real health monitor is injected."
    ),
    required_context=["target_states"],
    provided_context=["target_states"],
)


# ======================================================================
# RectangleStage
# ======================================================================

class RectangleStage(AdvancedStage):
    """Pipeline stage that normalises a target's box into corner form.

    :class:`RectangleStage` iterates over ``context.target_states`` and
    replaces each snapshot with a derived copy carrying a ``rect`` tuple
    ``(x1, y1, x2, y2)`` computed from the existing centre/size fields:

    * ``x1 = cx - width / 2``
    * ``y1 = cy - height / 2``
    * ``x2 = cx + width / 2``
    * ``y2 = cy + height / 2``

    This is pure arithmetic -- **no** concrete image-processing function
    is invoked. In D5 the conversion is a placeholder example: when a
    real box builder is injected, the stage will call it on each
    snapshot and write the result into ``rect``.

    Attributes:
        _rect_builder: The wrapped box-builder instance, or ``None``
            for a placeholder-only stage.

    Example:
        >>> stage = RectangleStage(name="rect")
        >>> stage.capability.name
        'rectangle'
    """

    __slots__ = ("_rect_builder", "_ready")

    def __init__(
        self,
        rect_builder: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Construct a RectangleStage with an optional box builder.

        Parameters:
            rect_builder: The box-builder instance. ``None`` (default)
                means no builder is available -- the stage derives the
                rect with the placeholder arithmetic.
            name: Optional stage name (defaults to ``"RectangleStage"``).
        """
        super().__init__(name=name if name is not None else "RectangleStage")
        self._rect_builder: Any | None = rect_builder
        self._ready: bool = False
        logger.debug("RectangleStage created: name=%s rect_builder=%s",
                     self._name,
                     type(rect_builder).__name__ if rect_builder is not None else "None")

    @property
    def capability(self) -> StageCapability:
        """``required_context=["target_states"]``, ``provided_context=["target_states"]``."""
        return _RECTANGLE_CAPABILITY

    def initialize(self) -> None:
        """Mark the stage initialised and ready. Idempotent."""
        self._ready = True
        logger.debug("RectangleStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Set ``rect=(x1, y1, x2, y2)`` on every ``TargetState``.

        Calls :meth:`AdvancedStage.check_context` first, then iterates
        over ``context.target_states``, replacing each snapshot with a
        derived copy carrying the corner-form box computed by
        :meth:`_derive_rect`. Uses REPLACE semantics (clear + extend in
        place).
        """
        self.check_context(context)
        originals: list[TargetState] = list(context.target_states)
        enriched: list[TargetState] = [
            ts.copy_with(rect=self._derive_rect(ts)) for ts in originals
        ]
        context.target_states.clear()
        context.target_states.extend(enriched)
        logger.debug("RectangleStage.process: name=%s snapshots=%d",
                     self._name, len(enriched))

    def _derive_rect(
        self, ts: TargetState,
    ) -> tuple[float, float, float, float]:
        """Convert centre/size box representation to corner form.

        Args:
            ts: A target snapshot carrying ``cx``/``cy``/``width``/
                ``height`` in normalised coordinates.

        Returns:
            ``(x1, y1, x2, y2)`` with ``x1 = cx - width/2``,
            ``y1 = cy - height/2``, ``x2 = cx + width/2``,
            ``y2 = cy + height/2``.
        """
        x1: float = ts.cx - ts.width / 2.0
        y1: float = ts.cy - ts.height / 2.0
        x2: float = ts.cx + ts.width / 2.0
        y2: float = ts.cy + ts.height / 2.0
        return (x1, y1, x2, y2)

    def shutdown(self) -> None:
        """Mark the stage shut down. Idempotent, never raises."""
        self._ready = False
        logger.debug("RectangleStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the stage is initialised and ready."""
        return self._ready

    @property
    def rect_builder(self) -> Any | None:
        """The wrapped box-builder instance, or ``None``."""
        return self._rect_builder


# ======================================================================
# FilterStage
# ======================================================================

class FilterStage(AdvancedStage):
    """Pipeline stage that filters ``context.target_states`` by a predicate.

    :class:`FilterStage` is the extension point for discarding noisy /
    low-quality targets. It evaluates the D5 placeholder predicate --
    drop a snapshot when ``target.confidence < 0`` -- over
    ``context.target_states`` and logs how many would be dropped.

    Because ``confidence`` is defined in ``[0.0, 1.0]``, the spec's
    example drop-condition can never fire on valid data, so in D5 the
    stage **deletes nothing**: the target list is left untouched
    (no reordering, no removal). When a real filterer is injected, this
    same evaluation point becomes the actual suppression pass.

    Attributes:
        _filterer: The wrapped filter instance, or ``None`` for a
            placeholder-only stage.

    Example:
        >>> stage = FilterStage(name="filter")
        >>> stage.capability.name
        'filter'
    """

    __slots__ = ("_filterer", "_ready")

    def __init__(
        self,
        filterer: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Construct a FilterStage with an optional filter.

        Parameters:
            filterer: The filter instance. ``None`` (default) means no
                filterer is available -- the stage evaluates the
                placeholder predicate but deletes nothing.
            name: Optional stage name (defaults to ``"FilterStage"``).
        """
        super().__init__(name=name if name is not None else "FilterStage")
        self._filterer: Any | None = filterer
        self._ready: bool = False
        logger.debug("FilterStage created: name=%s filterer=%s",
                     self._name,
                     type(filterer).__name__ if filterer is not None else "None")

    @property
    def capability(self) -> StageCapability:
        """``required_context=["target_states"]``, ``provided_context=["target_states"]``."""
        return _FILTER_CAPABILITY

    def initialize(self) -> None:
        """Mark the stage initialised and ready. Idempotent."""
        self._ready = True
        logger.debug("FilterStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Evaluate the filter predicate but delete nothing.

        Calls :meth:`AdvancedStage.check_context` first, then computes
        the set of snapshots the placeholder predicate would drop
        (``target.confidence < 0``) and logs the count. The D5 shell
        deliberately leaves ``context.target_states`` untouched so no
        target is ever removed from the pipeline.
        """
        self.check_context(context)
        pending_drop: list[TargetState] = [
            ts for ts in context.target_states if ts.confidence < 0.0
        ]
        logger.debug(
            "FilterStage.process: name=%s evaluated=%d pending_drop=%d "
            "(D5 placeholder: deletes nothing)",
            self._name, len(context.target_states), len(pending_drop),
        )

    def shutdown(self) -> None:
        """Mark the stage shut down. Idempotent, never raises."""
        self._ready = False
        logger.debug("FilterStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the stage is initialised and ready."""
        return self._ready

    @property
    def filterer(self) -> Any | None:
        """The wrapped filter instance, or ``None``."""
        return self._filterer


# ======================================================================
# HealthStage
# ======================================================================

class HealthStage(AdvancedStage):
    """Pipeline stage that scores each target's track health.

    :class:`HealthStage` iterates over ``context.target_states`` and
    replaces each snapshot with a derived copy carrying a ``health``
    score. In D5 the score is always ``1.0`` (placeholder). When a real
    health monitor is injected, the stage will call the monitor on each
    snapshot and write the result into ``health``.

    Attributes:
        _health_monitor: The wrapped health-monitor instance, or
            ``None`` for a placeholder-only stage.

    Example:
        >>> stage = HealthStage(name="health")
        >>> stage.capability.name
        'health'
    """

    __slots__ = ("_health_monitor", "_ready")

    def __init__(
        self,
        health_monitor: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Construct a HealthStage with an optional health monitor.

        Parameters:
            health_monitor: The health-monitor instance. ``None``
                (default) means no monitor is available -- the stage
                writes a placeholder ``1.0``.
            name: Optional stage name (defaults to ``"HealthStage"``).
        """
        super().__init__(name=name if name is not None else "HealthStage")
        self._health_monitor: Any | None = health_monitor
        self._ready: bool = False
        logger.debug("HealthStage created: name=%s health_monitor=%s",
                     self._name,
                     type(health_monitor).__name__ if health_monitor is not None else "None")

    @property
    def capability(self) -> StageCapability:
        """``required_context=["target_states"]``, ``provided_context=["target_states"]``."""
        return _HEALTH_CAPABILITY

    def initialize(self) -> None:
        """Mark the stage initialised and ready. Idempotent."""
        self._ready = True
        logger.debug("HealthStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Set ``health=1.0`` on every ``TargetState`` in the context.

        Calls :meth:`AdvancedStage.check_context` first, then iterates
        over ``context.target_states``, replacing each snapshot with a
        derived copy carrying ``health=1.0``. Uses REPLACE semantics
        (clear + extend in place).
        """
        self.check_context(context)
        originals: list[TargetState] = list(context.target_states)
        enriched: list[TargetState] = [
            ts.copy_with(health=1.0) for ts in originals
        ]
        context.target_states.clear()
        context.target_states.extend(enriched)
        logger.debug("HealthStage.process: name=%s snapshots=%d",
                     self._name, len(enriched))

    def shutdown(self) -> None:
        """Mark the stage shut down. Idempotent, never raises."""
        self._ready = False
        logger.debug("HealthStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the stage is initialised and ready."""
        return self._ready

    @property
    def health_monitor(self) -> Any | None:
        """The wrapped health-monitor instance, or ``None``."""
        return self._health_monitor