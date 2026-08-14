"""Advanced processing stages -- the Milestone D1 abstraction layer.

This subpackage hosts :class:`AdvancedStage`, the base class for the
*advanced* processing stages (denoising, re-detection, attribute
estimation, ...) that will layer on top of the core detect -> track ->
target-manage chain. Every advanced stage:

* subclasses :class:`AdvancedStage` (which in turn subclasses
  :class:`~visioncore.pipeline.base.PipelineStage`), so it plugs into the
  standard ``Pipeline`` exactly like any other stage;
* declares its data contract via a :class:`StageCapability` describing
  which ``PipelineContext`` fields it reads (``required_context``) and
  which it writes (``provided_context``);
* calls :meth:`AdvancedStage.check_context` at the start of ``process``
  to verify the context satisfies that contract before doing any work.

Scope (Milestone D1)
--------------------
D1 delivers the **abstraction layer only** -- no concrete advanced
plugins. It contains:

    StageCapability   -- frozen kw-only dataclass describing a stage's
                         data contract (name / version / description /
                         required_context / provided_context).
    AdvancedStage     -- abstract base class for advanced stages; same
                         four-method lifecycle as PipelineStage plus a
                         capability description and context-contract
                         checking.
    AdvancedStageError -- structured exception for contract violations
                         and recoverable advanced-stage failures.
    DummyAdvancedStage -- configurable test double implementing the
                         contract (reads its required fields, writes its
                         provided fields, traces through metadata).

Concrete advanced plugins (denoising, re-detection, attribute estimation,
... ) arrive in later milestones as ``AdvancedStage`` subclasses, exactly
like the detector / tracker / target stages did for their interfaces.

No modification is made to ``ai/``, ``camera/``, ``gui/`` or any existing
pipeline stage.
"""

from __future__ import annotations

from visioncore.pipeline.stages.advanced.advanced_stage import (
    AdvancedStage,
    AdvancedStageError,
    DummyAdvancedStage,
)
from visioncore.pipeline.stages.advanced.stage_capability import StageCapability

__all__ = [
    "AdvancedStage",
    "AdvancedStageError",
    "DummyAdvancedStage",
    "StageCapability",
]
