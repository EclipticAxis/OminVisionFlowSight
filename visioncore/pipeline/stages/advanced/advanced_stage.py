"""AdvancedStage -- the abstract base class for advanced processing plugins.

This module defines the *advanced processing stage* abstraction (Milestone
D1): a layer above the core detect -> track -> target-manage chain for
plugins such as denoising, re-detection, and attribute estimation. It
contains three cohesive concerns:

* :class:`AdvancedStageError` -- structured exception for contract
  violations and recoverable advanced-stage failures.
* :class:`AdvancedStage` -- the abstract base class. It subclasses
  :class:`~visioncore.pipeline.base.PipelineStage`, keeps the exact same
  four-method lifecycle (initialize / process / shutdown / health_check),
  and adds two things: a :attr:`~AdvancedStage.capability` declaration
  (:class:`~visioncore.pipeline.stages.advanced.stage_capability.StageCapability`)
  and :meth:`AdvancedStage.check_context`, the context-contract validator.
* :class:`DummyAdvancedStage` -- a configurable test double that obeys
  the contract: it reads its declared required fields, writes its
  declared provided fields, and records its lifecycle calls.

Scope (Milestone D1)
--------------------
D1 delivers the **abstraction layer only**. It deliberately does **not**
contain:

* any concrete advanced plugin (denoising, re-detection, attribute
  estimation, fusion, ...) -- those arrive in later milestones as
  ``AdvancedStage`` subclasses, exactly as the detector / tracker /
  target stages did for their interfaces;
* any modification to ``ai/``, ``camera/``, or ``gui/``;
* any modification to existing pipeline stages or business logic.

Relationship with PipelineStage
-------------------------------
:class:`AdvancedStage` **is a** :class:`PipelineStage`. It inherits the
name identity, ``__repr__``, and the full lifecycle contract unchanged --
a pipeline treats an advanced stage exactly like any other stage. The
only additions are the *capability* (what the stage declares it needs /
produces) and the *context contract check* (runtime validation that the
shared :class:`~visioncore.pipeline.context.PipelineContext` satisfies
that declaration before any work happens).

Where advanced stages plug in
-----------------------------
Advanced stages are *post-target* processors: they run **after** the
core chain (detect -> track -> target-manage) has populated the context,
and they refine or annotate its outputs. Typical examples:

* **Denoising** -- reads ``frame``, writes a cleaned ``frame`` (or
  ``metadata`` annotations) before detection;
* **Re-detection** -- reads ``frame`` + ``detections``, appends refined
  detections;
* **Attribute estimation** -- reads ``targets`` / ``target_states``,
  writes enriched snapshots to ``target_states`` with attribute data in
  each snapshot's ``metadata``.

The ``provided_context=["target_states"]`` convention describes a plugin
that emits the output target-state snapshot list
(:class:`~visioncore.state.target_state.TargetState`).

The context contract
--------------------
Every advanced stage declares its contract up front via
:attr:`AdvancedStage.capability`:

* ``required_context`` -- field names the stage reads;
* ``provided_context`` -- field names the stage writes (conventionally
  list-typed fields it appends to / replaces).

:meth:`AdvancedStage.check_context` validates the contract at runtime:

1. every ``required_context`` name exists on the context (readable);
2. every ``provided_context`` name exists and is list-typed (writable).

It raises :class:`AdvancedStageError` on the first violation. Stages
**must** call it at the top of ``process()`` before touching the context;
:class:`DummyAdvancedStage` demonstrates the convention. The contract is
declarative -- ``check_context`` does not instrument reads/writes, so the
"only touches declared fields" property is enforced by discipline and
verified in tests with a restrictive context double.

Thread safety
-------------
:class:`AdvancedStage` is not thread-safe, matching the serial pipeline
execution model of :class:`~visioncore.pipeline.pipeline.Pipeline`.

Example
-------
    >>> from visioncore.pipeline.stages.advanced import (
    ...     AdvancedStage, StageCapability, DummyAdvancedStage,
    ... )
    >>> from visioncore.pipeline.context import PipelineContext
    >>> stage = DummyAdvancedStage(name="denoise")
    >>> stage.initialize()
    >>> ctx = PipelineContext.empty()
    >>> stage.process(ctx)
    >>> ctx.metadata["advanced_trace"]
    ['denoise']
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import TYPE_CHECKING, Any

from visioncore.pipeline.base import PipelineStage
from visioncore.pipeline.stages.advanced.stage_capability import StageCapability

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = [
    "AdvancedStageError",
    "AdvancedStage",
    "DummyAdvancedStage",
]


logger = logging.getLogger(__name__)


# ======================================================================
# Exception
# ======================================================================

class AdvancedStageError(RuntimeError):
    """Structured exception signalling an advanced-stage failure.

    Raised by :meth:`AdvancedStage.check_context` when the shared
    :class:`~visioncore.pipeline.context.PipelineContext` violates the
    stage's declared :class:`StageCapability` contract (a required field
    is missing, or a provided field is missing / not list-typed).
    Concrete plugins may also raise it for their own recoverable failures.

    Like :class:`~visioncore.pipeline.base.StageError`, it is **not**
    caught by the pipeline -- it propagates out of ``run()`` unchanged, so
    callers can wrap the run for retry / fallback semantics.

    Example:
        >>> raise AdvancedStageError("required field 'frame' missing")  # doctest: +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
            ...
        visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStageError: required field 'frame' missing
    """


# ======================================================================
# AdvancedStage -- the abstract base class
# ======================================================================

class AdvancedStage(PipelineStage):
    """Abstract base class for advanced processing stage plugins.

    :class:`AdvancedStage` inherits the full
    :class:`~visioncore.pipeline.base.PipelineStage` contract: the
    four-method lifecycle (initialize / process / shutdown / health_check)
    is **unchanged**, so a pipeline schedules an advanced stage exactly
    like any other stage. On top of the base contract it adds:

    * :attr:`capability` -- an abstract property returning the stage's
      :class:`StageCapability` declaration (name / version / description /
      required_context / provided_context);
    * :meth:`check_context` -- a concrete helper validating that a
      :class:`~visioncore.pipeline.context.PipelineContext` satisfies the
      declared contract before any work begins.

    Subclassing
    -----------
    Subclasses **must** implement:

    * the four abstract lifecycle methods (same as
      :class:`~visioncore.pipeline.base.PipelineStage`);
    * :attr:`capability`, returning a :class:`StageCapability` instance.

    Subclasses **should** call :meth:`check_context` at the top of
    ``process()`` (see :class:`DummyAdvancedStage` for the convention),
    and **should not** read or write context fields outside the declared
    ``required_context`` / ``provided_context``.

    Attributes:
        capability: The stage's declared
            :class:`~visioncore.pipeline.stages.advanced.stage_capability.StageCapability`.
            Implemented as an abstract property -- each plugin describes
            itself.

    Example:
        >>> class AttributeStage(AdvancedStage):
        ...     @property
        ...     def capability(self):
        ...         return StageCapability(
        ...             name="attribute",
        ...             version="0.1.0",
        ...             description="Estimates target attributes.",
        ...             required_context=["target_states"],
        ...             provided_context=["target_states"],
        ...         )
        ...     def initialize(self): pass
        ...     def process(self, context): self.check_context(context)
        ...     def shutdown(self): pass
        ...     def health_check(self): return True
        >>> AttributeStage().name
        'AttributeStage'
        >>> AttributeStage().capability.version
        '0.1.0'
    """

    __slots__ = ()

    # ------------------------------------------------------------------
    # Capability declaration (abstract -- subclasses MUST implement)
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def capability(self) -> StageCapability:
        """Return the stage's declared :class:`StageCapability`.

        The capability names the plugin (``name`` / ``version`` /
        ``description``) and its data contract with the pipeline
        (``required_context`` / ``provided_context``). It must be
        available **before** ``initialize()`` -- pipeline tooling (logs,
        capability lookup, config GUIs) inspects it without running the
        stage.

        Returns:
            A :class:`~visioncore.pipeline.stages.advanced.stage_capability.StageCapability`
            describing this stage.
        """

    # ------------------------------------------------------------------
    # Lifecycle (PipelineStage contract -- unchanged, abstract)
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Acquire resources before the first ``process()`` call.

        Same contract as
        :meth:`~visioncore.pipeline.base.PipelineStage.initialize`:
        called once, idempotent, must leave the stage ready for
        :meth:`process`.
        """

    @abstractmethod
    def process(self, context: "PipelineContext") -> None:
        """Process one pipeline run, mutating ``context`` in place.

        Same contract as
        :meth:`~visioncore.pipeline.base.PipelineStage.process`.
        Subclasses **must** call :meth:`check_context` first and then
        confine their reads to ``required_context`` fields and their
        writes to ``provided_context`` fields (plus the freely-shared
        ``context.metadata`` / ``context.timestamp``).
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources acquired in :meth:`initialize`.

        Same contract as
        :meth:`~visioncore.pipeline.base.PipelineStage.shutdown`:
        idempotent, must not raise.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the stage is ready to accept ``process()``.

        Same contract as
        :meth:`~visioncore.pipeline.base.PipelineStage.health_check`:
        side-effect-free, must not raise.
        """

    # ------------------------------------------------------------------
    # Context contract validation (concrete helper)
    # ------------------------------------------------------------------

    def check_context(self, context: "PipelineContext") -> None:
        """Validate ``context`` against the stage's declared capability.

        Verifies the data contract of :attr:`capability` against a
        :class:`~visioncore.pipeline.context.PipelineContext`:

        1. **Required fields** -- every name in
           ``capability.required_context`` must exist on ``context``
           (readable).
        2. **Provided fields** -- every name in
           ``capability.provided_context`` must exist on ``context`` and
           be list-typed (writable; stages append to / replace these
           lists).

        On the first violation an :class:`AdvancedStageError` is raised
        naming the offending field. Stages call this at the top of
        :meth:`process`; it is also the hook tests use to verify the
        declarative contract in isolation.

        Parameters:
            context: The shared
                :class:`~visioncore.pipeline.context.PipelineContext`.

        Raises:
            AdvancedStageError: If a required field is missing, or a
                provided field is missing or not list-typed.

        Example:
            >>> from visioncore.pipeline.stages.advanced import DummyAdvancedStage
            >>> from visioncore.pipeline.context import PipelineContext
            >>> stage = DummyAdvancedStage()
            >>> ctx = PipelineContext.empty()
            >>> stage.check_context(ctx)  # default contract satisfied
            >>> ctx2 = PipelineContext.empty()
            >>> del ctx2.detections  # violate required_context
            >>> stage.check_context(ctx2)  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            visioncore.pipeline.stages.advanced.advanced_stage.AdvancedStageError: ...
        """
        cap: StageCapability = self.capability
        for name in cap.required_context:
            if not hasattr(context, name):
                raise AdvancedStageError(
                    f"{self._name}: required_context field {name!r} is "
                    f"missing from context"
                )
        for name in cap.provided_context:
            if not hasattr(context, name):
                raise AdvancedStageError(
                    f"{self._name}: provided_context field {name!r} is "
                    f"missing from context"
                )
            field: Any = getattr(context, name)
            if not isinstance(field, list):
                raise AdvancedStageError(
                    f"{self._name}: provided_context field {name!r} must "
                    f"be a list for writing, got "
                    f"{type(field).__name__}"
                )
        logger.debug("AdvancedStage.check_context: name=%s capability=%s",
                     self._name, cap.name)

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation with name, capability, health.

        Extends :meth:`~visioncore.pipeline.base.PipelineStage.__repr__`
        with the declared capability name and version, so logs and
        debugger output identify which plugin a stage hosts. Both
        ``capability`` and ``health_check`` are contractually
        side-effect-free and non-raising; defensive fallbacks are kept
        for robustness.
        """
        try:
            cap = self.capability
            capability_tag: str = f"{cap.name} {cap.version}"
        except Exception:  # noqa: BLE001 -- defensive: repr must not raise
            capability_tag = "<unknown>"
        try:
            healthy: bool = self.health_check()
        except Exception:  # noqa: BLE001 -- defensive: health_check must not raise
            healthy = False
        return (
            f"{type(self).__name__}(name={self._name!r}, "
            f"capability={capability_tag!r}, healthy={healthy})"
        )


# ======================================================================
# DummyAdvancedStage -- configurable test double
# ======================================================================

def _default_capability() -> StageCapability:
    """Build the default StageCapability for DummyAdvancedStage."""
    return StageCapability(
        name="dummy-advanced",
        version="0.1.0",
        description=(
            "Test placeholder advanced stage: reads its required "
            "context fields and writes a marker to its provided fields."
        ),
        required_context=["detections"],
        provided_context=["target_states"],
    )


class DummyAdvancedStage(AdvancedStage):
    """Configurable no-op advanced stage for testing the D1 abstraction.

    :class:`DummyAdvancedStage` does no real vision work. Its purpose is
    to make the advanced-stage abstraction **observable and testable**:
    it implements the full lifecycle, declares a
    :class:`StageCapability`, validates the context contract on every
    ``process()``, and records its behaviour so tests can assert on it.

    What it does on each lifecycle method
    -------------------------------------
    * :meth:`initialize` -- increments :attr:`initialize_count`; sets
      :attr:`healthy` to ``True``.
    * :meth:`process` -- increments :attr:`process_count`; calls
      :meth:`AdvancedStage.check_context` to validate the declared
      contract; appends :attr:`marker` to ``context.metadata["advanced_trace"]``;
      records the first required field's length in
      ``context.metadata["advanced_reads"]`` (demonstrating a read of
      declared input); appends :attr:`write_value` to every declared
      provided context list (demonstrating a write of declared output).
      If :attr:`raise_on_process` is set, raises it instead.
    * :meth:`shutdown` -- increments :attr:`shutdown_count`; sets
      :attr:`healthy` to ``False``. Never raises.
    * :meth:`health_check` -- returns :attr:`healthy`.

    Attributes:
        marker: Value appended to ``context.metadata["advanced_trace"]``
            and (when :attr:`write_value` is ``None``) to every provided
            context list on each ``process()``. Defaults to the stage
            name.
        write_value: Value appended to every provided context list. When
            ``None`` (default), :attr:`marker` is appended instead.
        raise_on_process: An exception instance to raise on the next
            ``process()`` call, or ``None`` (default) to process
            normally. Raised **after** :attr:`process_count` is
            incremented and **before** :meth:`check_context` runs.
        initialize_count / process_count / shutdown_count: Lifecycle
            call counters.

    Example:
        >>> from visioncore.pipeline.context import PipelineContext
        >>> s = DummyAdvancedStage(name="denoise")
        >>> s.initialize()
        >>> ctx = PipelineContext.empty()
        >>> s.process(ctx)
        >>> ctx.metadata["advanced_trace"]
        ['denoise']
        >>> len(ctx.target_states)  # provided field was written
        1
        >>> (s.initialize_count, s.process_count, s.shutdown_count)
        (1, 1, 0)
    """

    __slots__ = (
        "_capability",
        "marker",
        "write_value",
        "raise_on_process",
        "_healthy",
        "initialize_count",
        "process_count",
        "shutdown_count",
    )

    def __init__(
        self,
        name: str | None = None,
        *,
        capability: StageCapability | None = None,
        marker: Any | None = None,
        write_value: Any | None = None,
        raise_on_process: BaseException | None = None,
        healthy: bool = False,
    ) -> None:
        """Initialise a DummyAdvancedStage with optional test behaviour.

        Parameters:
            name: Stage name. Defaults to ``"DummyAdvancedStage"``. Also
                used as the default ``marker`` when ``marker`` is
                ``None``.
            capability: The :class:`StageCapability` the stage declares.
                When ``None`` (default), a standard placeholder
                capability is used: ``required_context=["detections"]``,
                ``provided_context=["target_states"]``.
            marker: Value appended to
                ``context.metadata["advanced_trace"]`` on each
                ``process()``. Defaults to ``name``.
            write_value: Value appended to every provided context list on
                each ``process()``. ``None`` (default) means use
                :attr:`marker`.
            raise_on_process: An exception instance to raise on
                ``process()`` instead of processing. ``None`` (default)
                means process normally.
            healthy: Initial value of :attr:`healthy`. Defaults to
                ``False`` (unhealthy until :meth:`initialize`).

        Example:
            >>> s = DummyAdvancedStage("a")
            >>> s.name, s.marker
            ('a', 'a')
            >>> s.capability.provided_context
            ['target_states']
        """
        super().__init__(name=name)
        resolved_name: str = self._name
        self._capability: StageCapability = (
            capability if capability is not None else _default_capability()
        )
        self.marker: Any = marker if marker is not None else resolved_name
        self.write_value: Any = write_value
        self.raise_on_process: BaseException | None = raise_on_process
        self._healthy: bool = healthy
        self.initialize_count: int = 0
        self.process_count: int = 0
        self.shutdown_count: int = 0
        logger.debug("DummyAdvancedStage created: name=%s capability=%s",
                     resolved_name, self._capability.name)

    # ------------------------------------------------------------------
    # Capability declaration
    # ------------------------------------------------------------------

    @property
    def capability(self) -> StageCapability:
        """The stage's declared :class:`StageCapability`."""
        return self._capability

    # ------------------------------------------------------------------
    # Lifecycle (AdvancedStage contract)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Mark the stage initialised and healthy. Idempotent."""
        self.initialize_count += 1
        self._healthy = True
        logger.debug("DummyAdvancedStage.initialize: name=%s count=%d",
                     self._name, self.initialize_count)

    def process(self, context: "PipelineContext") -> None:
        """Validate the contract, read required fields, write provided fields.

        If :attr:`raise_on_process` is set, raise it instead (after
        incrementing :attr:`process_count` but before any context
        interaction), so tests can confirm the stage was reached and that
        the run aborted before processing.

        Otherwise:

        1. :meth:`AdvancedStage.check_context` -- validate the declared
           required / provided fields against ``context``.
        2. Trace -- append :attr:`marker` to
           ``context.metadata["advanced_trace"]``.
        3. Read -- record the first required field's length in
           ``context.metadata["advanced_reads"]`` (a concrete read of a
           declared input; skipped when ``required_context`` is empty).
        4. Write -- append :attr:`write_value` (or :attr:`marker` when
           ``None``) to every declared provided context list.

        Parameters:
            context: The shared
                :class:`~visioncore.pipeline.context.PipelineContext`.
                Mutated: ``metadata["advanced_trace"]``,
                ``metadata["advanced_reads"]``, and each declared
                provided list.

        Raises:
            AdvancedStageError: If ``context`` violates the declared
                capability contract.
        """
        self.process_count += 1
        if self.raise_on_process is not None:
            logger.debug("DummyAdvancedStage.process: name=%s RAISING %s",
                         self._name, type(self.raise_on_process).__name__)
            raise self.raise_on_process
        self.check_context(context)

        cap: StageCapability = self._capability

        trace: list[Any] = context.metadata.setdefault("advanced_trace", [])
        trace.append(self.marker)

        if cap.required_context:
            reads: list[Any] = context.metadata.setdefault(
                "advanced_reads", {}
            )
            reads[cap.required_context[0]] = len(
                getattr(context, cap.required_context[0])
            )

        value: Any = self.write_value if self.write_value is not None else self.marker
        for name in cap.provided_context:
            getattr(context, name).append(value)

        logger.debug(
            "DummyAdvancedStage.process: name=%s count=%d required=%s provided=%s",
            self._name, self.process_count,
            cap.required_context, cap.provided_context,
        )

    def shutdown(self) -> None:
        """Mark the stage shut down and unhealthy. Never raises."""
        self.shutdown_count += 1
        self._healthy = False
        logger.debug("DummyAdvancedStage.shutdown: name=%s count=%d",
                     self._name, self.shutdown_count)

    def health_check(self) -> bool:
        """Return the stage's current health flag. Never raises."""
        return self._healthy

    # ------------------------------------------------------------------
    # Test convenience
    # ------------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """Current health flag (backed by ``_healthy``)."""
        return self._healthy

    def reset(self) -> None:
        """Reset all counters and the health flag to post-construction state.

        Does **not** reset :attr:`marker`, :attr:`write_value`,
        :attr:`raise_on_process`, or :attr:`capability` -- only the
        lifecycle counters and health.
        """
        self._healthy = False
        self.initialize_count = 0
        self.process_count = 0
        self.shutdown_count = 0
