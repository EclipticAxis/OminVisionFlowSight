"""VisionCore pipeline layer -- the ordered stage-execution framework.

This package hosts the *pipeline layer* of VisionCore: an ordered,
sequential executor (:class:`Pipeline`) that drives a list of
:class:`PipelineStage` instances against a shared
:class:`PipelineContext`. It is the future backbone that will replace the
monolithic ``InferWorker`` with a composable chain of single-purpose
stages (capture -> detect -> track -> target-manage -> project).

Scope (Milestone C1 -- Pipeline Architecture Foundation)
-------------------------------------------------------
C1 delivers the **framework only**. The package exports four public types:

    PipelineStage    -- abstract base class (the four-method lifecycle contract).
    PipelineContext  -- mutable per-frame data envelope (the shared blackboard).
    Pipeline         -- ordered, sequential stage executor.
    DummyStage       -- configurable no-op stage for testing the machinery.

It deliberately does **not** contain:

* any detector, tracker, or target-manager wiring (that is C2+);
* any threading, async, or queue-based execution (stages run serially);
* any I/O, transport, or serialisation;
* any modification to ``ai/``, ``gui/``, or ``camera/`` -- the existing
  VisionDataPlatform application continues to run unchanged. VisionCore
  is being introduced **alongside** it.

Design summary
--------------
* **Pure scheduler**: the pipeline knows about stages and order, and
  nothing else. Real stages arrive in C2+ as ``PipelineStage`` subclasses;
  the pipeline itself never changes.
* **Shared blackboard**: a single ``PipelineContext`` flows stage-by-stage;
  stages read upstream fields and write downstream fields on the same
  instance.
* **Strict lifecycle**: every stage obeys ``initialize -> process ->
  shutdown``; the pipeline drives ``initialize`` in order and ``shutdown``
  in reverse, with guaranteed teardown via the context manager.
* **Honest exceptions**: a stage's ``process`` exception propagates out
  of ``run`` unchanged -- the framework never silently swallows stage
  failures. ``shutdown`` is the only best-effort path (teardown must
  complete regardless).

Quick start::

    from visioncore.pipeline import (
        Pipeline, PipelineStage, PipelineContext, DummyStage,
    )

    p = Pipeline()
    p.add_stage(DummyStage("capture"))
    p.add_stage(DummyStage("detect"))

    with p:                                # initialise
        ctx = p.run(PipelineContext.empty(timestamp=0.0))
    # shutdown() called automatically on exit

    print(ctx.metadata["order"])           # ['capture', 'detect']

Future milestones (C2+)
-----------------------
C2+ will add real stages as sibling modules / subpackages under
``visioncore.pipeline``: a capture stage, a detection stage, a tracking
stage, a target-management stage, and a projection stage (targets ->
``TargetState`` snapshots, bridging to the protocol layer of Milestone B2).
Each will subclass ``PipelineStage`` and obey the same contract; the
``Pipeline`` and ``PipelineContext`` defined here will not change.
"""

from __future__ import annotations

from visioncore.pipeline.base import PipelineStage, StageError
from visioncore.pipeline.context import PipelineContext
from visioncore.pipeline.pipeline import Pipeline
from visioncore.pipeline.shadow import ShadowPipelineRunner, ShadowStats, StageStats
from visioncore.pipeline.stage import DummyStage

__all__ = [
    # ---- Core framework ----
    "Pipeline",
    "PipelineStage",
    "PipelineContext",
    # ---- Test doubles ----
    "DummyStage",
    # ---- Exceptions ----
    "StageError",
    # ---- Shadow runner (C7) ----
    "ShadowPipelineRunner",
    "ShadowStats",
    "StageStats",
]
