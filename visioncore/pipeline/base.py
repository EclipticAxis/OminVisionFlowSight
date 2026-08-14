"""PipelineStage -- the abstract processing unit contract for a Pipeline.

Defines :class:`PipelineStage`, the abstract base class that every processing
unit in a :class:`~visioncore.pipeline.pipeline.Pipeline` must implement. A
stage is a single step in the detection -> tracking -> target-management ->
projection chain: it reads upstream fields from a
:class:`~visioncore.pipeline.context.PipelineContext`, does its work, and
writes downstream fields back into the same context.

Scope of this milestone (C1)
---------------------------
This module defines the **contract only**. It contains:

* :class:`PipelineStage` -- the ABC with a four-method lifecycle.
* :class:`StageError` -- a structured exception that stages *may* raise to
  signal a recoverable stage-specific failure.

It deliberately does **not** contain:

* any detector, tracker, or target-manager wiring (that is C2+);
* any threading or async execution (stages run synchronously, in order);
* any I/O or transport code;
* any concrete stage implementations (see :mod:`visioncore.pipeline.stage`
  for :class:`~visioncore.pipeline.stage.DummyStage`, the test double).

Lifecycle contract
------------------
Every stage obeys a strict four-method lifecycle, driven by the enclosing
:class:`~visioncore.pipeline.pipeline.Pipeline`:

1. ``initialize()`` -- called **once** before the first ``process()``. Used
   to acquire resources (model loading, allocator warmup, config reads).
   Must be idempotent: calling it twice is a no-op.
2. ``process(context)`` -- called **per pipeline run** with the shared
   :class:`~visioncore.pipeline.context.PipelineContext`. This is where the
   stage does its real work. Exceptions raised here propagate out of
   :meth:`~visioncore.pipeline.pipeline.Pipeline.run` and abort the run.
3. ``shutdown()`` -- called **once** at teardown. Must be idempotent and
   **must not raise** -- it releases resources acquired in ``initialize()``.
   The pipeline calls stages' ``shutdown()`` in reverse insertion order so
   that dependant stages are torn down before the stages they depend on.
4. ``health_check()`` -- side-effect-free probe returning ``True`` iff the
   stage is initialised and ready to accept ``process()`` calls. Must not
   raise; on internal error return ``False``.

Naming
------
Each stage carries a ``name`` (defaults to the class name) used in logs,
``__repr__``, and pipeline diagnostics. Two stages of the same class may
carry different names -- this is essential for pipelines that chain, say,
two projection stages.

Thread safety
-------------
Stages are not required to be thread-safe. A pipeline runs its stages
serially on the calling thread. Stages that internally fan out to worker
threads are responsible for their own synchronisation and for not mutating
the shared :class:`~visioncore.pipeline.context.PipelineContext` from
multiple threads concurrently.

Example
-------
    >>> from visioncore.pipeline.base import PipelineStage
    >>> from visioncore.pipeline.context import PipelineContext
    >>> class NoOpStage(PipelineStage):
    ...     def initialize(self) -> None:
    ...         pass
    ...     def process(self, context: PipelineContext) -> None:
    ...         context.metadata.setdefault("trace", []).append(self.name)
    ...     def shutdown(self) -> None:
    ...         pass
    ...     def health_check(self) -> bool:
    ...         return True
    >>> stage = NoOpStage(name="noop")
    >>> stage.name
    'noop'
    >>> ctx = PipelineContext.empty()
    >>> stage.initialize()
    >>> stage.process(ctx)
    >>> ctx.metadata["trace"]
    ['noop']
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = ["PipelineStage", "StageError"]


logger = logging.getLogger(__name__)


class StageError(RuntimeError):
    """Structured exception signalling a recoverable stage failure.

    Stages *may* raise ``StageError`` (or any subclass) to indicate a
    failure that is specific to the stage's domain -- e.g. a model failed
    to load, a tracker diverged, a projection produced no snapshots.

    Raising ``StageError`` instead of a bare ``RuntimeError`` lets
    upstream callers distinguish "the stage deliberately reported a
    failure" from "an unexpected bug crashed the stage". The enclosing
    :class:`~visioncore.pipeline.pipeline.Pipeline` does **not** catch
    ``StageError`` specially -- it propagates out of ``run()`` exactly
    like any other exception. Callers that want retry / fallback
    semantics wrap the ``run()`` call in their own ``try/except``.

    Example:
        >>> raise StageError("model 'yolo26n' failed to load")
        Traceback (most recent call last):
            ...
        visioncore.pipeline.base.StageError: model 'yolo26n' failed to load
    """


class PipelineStage(ABC):
    """Abstract base class for all pipeline processing stages.

    A stage is one step in a pipeline. It receives a shared
    :class:`~visioncore.pipeline.context.PipelineContext`, reads upstream
    fields, performs its work, and writes downstream fields. The four-method
    lifecycle (initialize / process / shutdown / health_check) is driven by
    the enclosing :class:`~visioncore.pipeline.pipeline.Pipeline`.

    Subclassing
    -----------
    Subclasses **must** implement all four abstract methods:
    :meth:`initialize`, :meth:`process`, :meth:`shutdown`, :meth:`health_check`.

    Subclasses **may** override the ``name`` property or pass a ``name`` to
    ``super().__init__()`` to give the stage a distinct identity (useful
    when a pipeline chains multiple stages of the same class).

    Attributes:
        _name: The stage's display name. Defaults to the class name when
            ``None`` is passed to ``__init__``. Stored so that two instances
            of the same subclass can be told apart in logs.
    """

    __slots__ = ("_name",)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, name: str | None = None) -> None:
        """Initialise the stage with an optional display name.

        Parameters:
            name: Human-readable stage name. When ``None`` (default), the
                class name (``type(self).__name__``) is used. Two stages of
                the same class may carry different names -- this is how a
                pipeline distinguishes, e.g., two projection stages in logs
                and diagnostics.

        Example:
            >>> class MyStage(PipelineStage):
            ...     def initialize(self): pass
            ...     def process(self, ctx): pass
            ...     def shutdown(self): pass
            ...     def health_check(self): return True
            >>> MyStage().name
            'MyStage'
            >>> MyStage(name="custom").name
            'custom'
        """
        self._name: str = name if name is not None else type(self).__name__
        logger.debug("PipelineStage created: name=%s cls=%s",
                     self._name, type(self).__name__)

    # ------------------------------------------------------------------
    # Abstract lifecycle -- subclasses MUST implement
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self) -> None:
        """Acquire resources before the first ``process()`` call.

        Called **once** by the pipeline before any processing begins.
        Typical work: load model weights, warm up allocators, read config,
        open files. Must be idempotent -- a second call is a no-op.

        Raises:
            StageError: If initialisation fails and the stage cannot
                proceed. The stage must leave itself in a state where
                :meth:`shutdown` is still safe to call.
        """

    @abstractmethod
    def process(self, context: "PipelineContext") -> None:
        """Process one pipeline run, mutating ``context`` in place.

        This is the stage's real work. Read upstream fields from
        ``context`` (e.g. ``context.frame``), do the work, and write
        downstream fields back (e.g. ``context.detections.append(...)``).

        Exceptions raised here propagate out of
        :meth:`~visioncore.pipeline.pipeline.Pipeline.run` and abort the
        current run. The pipeline does not catch stage exceptions -- if a
        stage fails, the run fails. Callers wanting retry / fallback wrap
        ``run()`` themselves.

        Parameters:
            context: The shared :class:`~visioncore.pipeline.context.PipelineContext`.
                Mutated in place; the stage must not replace the instance.

        Raises:
            StageError: For deliberate, recoverable stage failures.
            Exception: Any other exception indicates an unexpected bug and
                propagates unchanged.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Release resources acquired in :meth:`initialize`.

        Called **once** by the pipeline at teardown, in **reverse** stage
        insertion order so dependant stages are torn down first. Must be
        idempotent and **must not raise** -- a failing ``shutdown`` would
        mask the real error that triggered teardown. Log internally
        instead.

        The pipeline guarantees this is called even if a prior stage's
        ``process`` raised. Best-effort: each stage's ``shutdown`` is
        isolated from the others, so one stage's failure does not skip
        the rest.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` iff the stage is ready to accept ``process()``.

        Side-effect-free probe. Must not raise; on internal error return
        ``False``. Used by the pipeline's aggregate
        :meth:`~visioncore.pipeline.pipeline.Pipeline.health_check` and by
        external monitoring to decide whether to keep running or fall back.

        Returns:
            ``True`` if the stage is initialised and healthy, ``False``
            otherwise.
        """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The stage's display name (defaults to the class name)."""
        return self._name

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a concise representation with name and health.

        Calls :meth:`health_check` to report readiness. Because
        ``health_check`` is contractually side-effect-free and non-raising,
        this is safe to call at any time (e.g. from a debugger).
        """
        try:
            healthy: bool = self.health_check()
        except Exception:  # noqa: BLE001 -- defensive: health_check must not raise
            healthy = False
        return f"{type(self).__name__}(name={self._name!r}, healthy={healthy})"
