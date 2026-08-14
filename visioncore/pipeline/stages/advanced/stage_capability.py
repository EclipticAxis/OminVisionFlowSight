"""StageCapability -- the declarative data contract of an advanced stage.

A :class:`StageCapability` is a frozen, keyword-only dataclass that
describes *what* an :class:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage`
plugin needs and *what* it produces. It is the metadata a stage carries
about itself, consumable by pipelines, configuration GUIs, and automated
schedulers without instantiating or running the stage.

Design rationale
----------------
* **Frozen + keyword-only**: a capability is an immutable declaration.
  ``kw_only=True`` forces callers to spell out every field by name, which
  keeps construction self-documenting -- a capability like
  ``StageCapability(name="denoise", version="1.0.0", ...)`` reads far
  better than five positional strings.
* **Plain lists of strings**: the context fields are named by the
  ``PipelineContext`` attribute they map to (e.g. ``"target_states"``,
  ``"detections"``). No introspection, no object references -- just names,
  so a capability stays serialisable and framework-agnostic.
* **No behaviour**: a capability is data only. The stage's logic lives in
  the stage; the capability exists so *other* code can reason about the
  stage's inputs and outputs up front.

The ``provided_context`` vocabulary
-----------------------------------
The write fields are described by string names, so any pipeline context
attribute can be declared. A convention is emerging for the most common
advanced output -- the projected target snapshot list::

    provided_context=["target_states"]   # List[visioncore.state.target_state.TargetState]

A plugin that emits snapshots (re-detection refinement, attribute
estimation, fusion) declares ``"target_states"`` here. The full list of
today's pipeline fields (see :mod:`visioncore.pipeline.context`):

    "frame", "detections", "tracks", "targets", "target_states"

Field semantics
---------------
* ``name`` -- plugin identifier (e.g. ``"denoise"``, ``"re-detect"``,
  ``"attribute-estimate"``). Stable across versions; used for logging and
  capability lookup.
* ``version`` -- plugin semantic version (e.g. ``"1.2.0"``). Lets
  pipelines detect incompatible plugin upgrades.
* ``description`` -- human-readable one-liner describing what the plugin
  does.
* ``required_context`` -- ``PipelineContext`` field names the stage **must
  read** before doing work. The stage's ``process`` validates these exist
  before touching any of them.
* ``provided_context`` -- ``PipelineContext`` field names the stage
  **writes**. Conventionally list-typed fields that the stage appends to
  / replaces. ``process`` validates each is present and mutable before
  writing.

Example
-------
    >>> from visioncore.pipeline.stages.advanced import StageCapability
    >>> cap = StageCapability(
    ...     name="denoise",
    ...     version="1.0.0",
    ...     description="Bilateral denoising of the frame before detection.",
    ...     required_context=["frame"],
    ...     provided_context=[],
    ... )
    >>> cap.name, cap.version
    ('denoise', '1.0.0')
    >>> cap.required_context
    ['frame']
    >>> cap.provided_context
    []
"""

from __future__ import annotations

from dataclasses import dataclass


__all__ = ["StageCapability"]


@dataclass(frozen=True, kw_only=True)
class StageCapability:
    """Immutable declaration of an advanced stage's inputs and outputs.

    A :class:`StageCapability` names the stage (``name`` / ``version`` /
    ``description``) and its data contract with the pipeline: the
    :class:`~visioncore.pipeline.context.PipelineContext` fields it reads
    (``required_context``) and the fields it writes
    (``provided_context``).

    All fields are keyword-only: positional construction is a ``TypeError``.
    The lists are read-only by convention (the dataclass is frozen, but
    Python cannot freeze list contents); treat them as immutable after
    construction.

    Attributes:
        name: Stable plugin identifier, e.g. ``"denoise"``. Used for
            logging, diagnostics, and capability lookup. Should not change
            between versions.
        version: Plugin semantic version, e.g. ``"1.2.0"``. Lets
            pipelines detect incompatible plugin upgrades.
        description: Human-readable one-liner describing what the plugin
            does.
        required_context: Names of :class:`~visioncore.pipeline.context.PipelineContext`
            fields the stage reads before doing work. Must be present on
            the context passed to ``process`` -- validated by
            :meth:`~visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStage.check_context`.
        provided_context: Names of
            :class:`~visioncore.pipeline.context.PipelineContext` fields
            the stage writes. May include ``"target_states"`` to describe
            an output target-state snapshot list
            (:class:`~visioncore.state.target_state.TargetState`). Each
            must be present on the context and list-typed -- also
            validated by ``check_context``.

    Example:
        >>> cap = StageCapability(
        ...     name="re-detect",
        ...     version="0.3.1",
        ...     description="Re-runs detection on low-confidence regions.",
        ...     required_context=["frame", "detections"],
        ...     provided_context=["target_states"],
        ... )
        >>> cap.description.startswith("Re-runs")
        True
        >>> "target_states" in cap.provided_context
        True
        >>> cap.required_context == ["frame", "detections"]
        True
    """

    name: str
    version: str
    description: str
    required_context: list[str]
    provided_context: list[str]
