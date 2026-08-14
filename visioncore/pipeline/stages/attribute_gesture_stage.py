"""AttributeStage / GestureStage -- target enrichment stages for the VisionCore pipeline.

This module defines two cohesive
:class:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage`
plugins that sit after the target-management stage and enrich each
:class:`~visioncore.state.target_state.TargetState` snapshot with
additional fields:

* :class:`AttributeStage` -- populates ``TargetState.attributes`` with
  a placeholder ``{}`` (or a value produced by an injected attribute
  estimator).
* :class:`GestureStage` -- populates ``TargetState.gesture`` with a
  placeholder ``None`` (or a label produced by an injected gesture
  recogniser).

Scope (Milestone D4)
--------------------
D4 delivers **only** the stage shells -- no actual attribute-estimation
or gesture-recognition algorithm. Each stage iterates over
``context.target_states`` and uses
:meth:`~visioncore.state.target_state.TargetState.copy_with` to produce
a derived snapshot with the placeholder field set. The constructors
accept optional estimator / recogniser instances, but they are held and
not invoked. Real backends arrive in later milestones.

D4 deliberately does **not** contain:

* any attribute-estimation or gesture-recognition model;
* any model loading, GPU allocation, or ONNX/torch code;
* any modification to ``ai/``, ``camera/``, or ``gui/``.

Data flow
---------
::

    context.target_states ──> AttributeStage.process
                                    │
                          for each snapshot:
                              snapshot.copy_with(attributes={...})
                                    │
                                    ▼
                          context.target_states (REPLACE)

    context.target_states ──> GestureStage.process
                                    │
                          for each snapshot:
                              snapshot.copy_with(gesture="...")
                                    │
                                    ▼
                          context.target_states (REPLACE)

Both stages use **REPLACE** semantics on ``context.target_states``
(clear + extend with derived snapshots), consistent with
:class:`~visioncore.pipeline.stages.target_stage.TargetStage`.

Example
-------
    >>> from visioncore.pipeline.stages.attribute_gesture_stage import (
    ...     AttributeStage, GestureStage,
    ... )
    >>> from visioncore.pipeline.context import PipelineContext
    >>> stage = AttributeStage()
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty()
    >>> stage.process(ctx)
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


__all__ = ["AttributeStage", "GestureStage"]


logger = logging.getLogger(__name__)


# ======================================================================
# Module-level capabilities (single source of truth)
# ======================================================================

_ATTRIBUTE_CAPABILITY = StageCapability(
    name="attribute",
    version="0.1.0",
    description=(
        "Attribute-estimation stage placeholder: sets "
        "TargetState.attributes={} on each snapshot until a real "
        "attribute estimator is injected."
    ),
    required_context=["target_states"],
    provided_context=["target_states"],
)

_GESTURE_CAPABILITY = StageCapability(
    name="gesture",
    version="0.1.0",
    description=(
        "Gesture-recognition stage placeholder: sets "
        "TargetState.gesture=None on each snapshot until a real "
        "gesture recogniser is injected."
    ),
    required_context=["target_states"],
    provided_context=["target_states"],
)


# ======================================================================
# AttributeStage
# ======================================================================

class AttributeStage(AdvancedStage):
    """Pipeline stage that populates ``TargetState.attributes``.

    :class:`AttributeStage` iterates over ``context.target_states`` and
    replaces each snapshot with a derived copy carrying an ``attributes``
    dict. In D4 the dict is always ``{}`` (placeholder). When a real
    attribute estimator is injected, the stage will call the estimator
    on each snapshot and write the result into ``attributes``.

    Attributes:
        _estimator: The wrapped attribute-estimator instance, or
            ``None`` for a placeholder-only stage.

    Example:
        >>> stage = AttributeStage(name="attr")
        >>> stage.capability.name
        'attribute'
    """

    __slots__ = ("_estimator", "_ready")

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Construct an AttributeStage with an optional estimator.

        Parameters:
            estimator: The attribute-estimator instance. ``None``
                (default) means no estimator is available -- the stage
                writes a placeholder ``{}``.
            name: Optional stage name (defaults to ``"AttributeStage"``).
        """
        super().__init__(name=name if name is not None else "AttributeStage")
        self._estimator: Any | None = estimator
        self._ready: bool = False
        logger.debug("AttributeStage created: name=%s estimator=%s",
                     self._name,
                     type(estimator).__name__ if estimator is not None else "None")

    @property
    def capability(self) -> StageCapability:
        """``required_context=["target_states"]``, ``provided_context=["target_states"]``."""
        return _ATTRIBUTE_CAPABILITY

    def initialize(self) -> None:
        """Mark the stage initialised and ready. Idempotent."""
        self._ready = True
        logger.debug("AttributeStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Set ``attributes={}`` on every ``TargetState`` in the context.

        Calls :meth:`AdvancedStage.check_context` first, then iterates
        over ``context.target_states``, replacing each snapshot with a
        derived copy carrying ``attributes={}``. Uses REPLACE semantics
        (clear + extend in place) consistent with
        :class:`~visioncore.pipeline.stages.target_stage.TargetStage`.
        """
        self.check_context(context)
        originals: list[TargetState] = list(context.target_states)
        enriched: list[TargetState] = [
            ts.copy_with(attributes={}) for ts in originals
        ]
        context.target_states.clear()
        context.target_states.extend(enriched)
        logger.debug("AttributeStage.process: name=%s snapshots=%d",
                     self._name, len(enriched))

    def shutdown(self) -> None:
        """Mark the stage shut down. Idempotent, never raises."""
        self._ready = False
        logger.debug("AttributeStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the stage is initialised and ready."""
        return self._ready

    @property
    def estimator(self) -> Any | None:
        """The wrapped attribute-estimator instance, or ``None``."""
        return self._estimator


# ======================================================================
# GestureStage
# ======================================================================

class GestureStage(AdvancedStage):
    """Pipeline stage that populates ``TargetState.gesture``.

    :class:`GestureStage` iterates over ``context.target_states`` and
    replaces each snapshot with a derived copy carrying a ``gesture``
    label. In D4 the label is always ``None`` (placeholder). When a real
    gesture recogniser is injected, the stage will call the recogniser
    on each snapshot and write the result into ``gesture``.

    Attributes:
        _recogniser: The wrapped gesture-recogniser instance, or
            ``None`` for a placeholder-only stage.

    Example:
        >>> stage = GestureStage(name="gest")
        >>> stage.capability.name
        'gesture'
    """

    __slots__ = ("_recogniser", "_ready")

    def __init__(
        self,
        recogniser: Any | None = None,
        *,
        name: str | None = None,
    ) -> None:
        """Construct a GestureStage with an optional recogniser.

        Parameters:
            recogniser: The gesture-recogniser instance. ``None``
                (default) means no recogniser is available -- the stage
                writes a placeholder ``None``.
            name: Optional stage name (defaults to ``"GestureStage"``).
        """
        super().__init__(name=name if name is not None else "GestureStage")
        self._recogniser: Any | None = recogniser
        self._ready: bool = False
        logger.debug("GestureStage created: name=%s recogniser=%s",
                     self._name,
                     type(recogniser).__name__ if recogniser is not None else "None")

    @property
    def capability(self) -> StageCapability:
        """``required_context=["target_states"]``, ``provided_context=["target_states"]``."""
        return _GESTURE_CAPABILITY

    def initialize(self) -> None:
        """Mark the stage initialised and ready. Idempotent."""
        self._ready = True
        logger.debug("GestureStage.initialize: name=%s", self._name)

    def process(self, context: "PipelineContext") -> None:
        """Set ``gesture=None`` on every ``TargetState`` in the context.

        Calls :meth:`AdvancedStage.check_context` first, then iterates
        over ``context.target_states``, replacing each snapshot with a
        derived copy carrying ``gesture=None``. Uses REPLACE semantics
        (clear + extend in place) consistent with
        :class:`~visioncore.pipeline.stages.target_stage.TargetStage`.
        """
        self.check_context(context)
        originals: list[TargetState] = list(context.target_states)
        enriched: list[TargetState] = [
            ts.copy_with(gesture=None) for ts in originals
        ]
        context.target_states.clear()
        context.target_states.extend(enriched)
        logger.debug("GestureStage.process: name=%s snapshots=%d",
                     self._name, len(enriched))

    def shutdown(self) -> None:
        """Mark the stage shut down. Idempotent, never raises."""
        self._ready = False
        logger.debug("GestureStage.shutdown: name=%s", self._name)

    def health_check(self) -> bool:
        """Return ``True`` iff the stage is initialised and ready."""
        return self._ready

    @property
    def recogniser(self) -> Any | None:
        """The wrapped gesture-recogniser instance, or ``None``."""
        return self._recogniser
