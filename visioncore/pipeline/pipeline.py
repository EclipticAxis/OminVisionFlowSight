"""Pipeline -- the ordered, sequential stage executor for VisionCore.

Defines :class:`Pipeline`, the top-level orchestrator that runs an ordered
list of :class:`~visioncore.pipeline.base.PipelineStage` instances against a
single :class:`~visioncore.pipeline.context.PipelineContext`.

Scope of this milestone (C1)
---------------------------
This module delivers the **core pipeline engine only**:

* :class:`Pipeline` -- add/remove stages, initialise / run / shutdown.
* Aggregate :meth:`~Pipeline.health_check` over all stages.
* Context-manager protocol for guaranteed teardown.

It deliberately does **not** contain:

* any :class:`~visioncore.core.detection.Detection`-producing detector;
* any :class:`~visioncore.core.track.Track`-producing tracker;
* any :class:`~visioncore.target_manager.manager.TargetManager`;
* any threading, async, or queue-based execution;
* any I/O, transport, or serialisation.

The pipeline is a **pure scheduler**: it knows about stages and order, and
nothing else. Real detection / tracking / target-management stages arrive
in C2+ as :class:`~visioncore.pipeline.base.PipelineStage` subclasses;
the pipeline itself never changes.

Execution model
---------------
Stages run **synchronously, in insertion order**, on the calling thread.
There is no parallelism within a run -- stage N+1 starts only after stage
N's ``process()`` returns. This keeps the C1 contract trivially correct
and easy to reason about. Future milestones may add a parallel branch
executor, but the serial executor remains the reference.

Exception semantics
-------------------
If a stage's ``process()`` raises, the run aborts **immediately**:

* The exception propagates out of :meth:`run` unchanged.
* Stages after the failing one are **not** called for this run.
* :meth:`shutdown` is **still** called on every initialised stage, in
  **reverse** insertion order, so resources are released. ``shutdown`` is
  best-effort: a failing ``shutdown`` is logged and does not prevent the
  remaining stages from shutting down.

The recommended pattern is to drive a pipeline through the context
manager so that ``shutdown`` is guaranteed regardless of what ``run``
does::

    with pipeline:
        pipeline.run(ctx)
    # shutdown() called on exit, even if run() raised

Lifecycle state machine
-----------------------
A pipeline has three states, tracked by an internal flag:

* **NEW** (``_initialized == False``, ``_shutdown == False``): freshly
  constructed. ``run()`` raises :class:`RuntimeError` -- call
  :meth:`initialize` first.
* **INITIALIZED** (``_initialized == True``, ``_shutdown == False``):
  ready to ``run()``. May be called multiple times across contexts.
* **SHUT_DOWN** (``_shutdown == True``): terminal. ``run()`` and
  :meth:`initialize` both raise :class:`RuntimeError`.

:meth:`initialize` and :meth:`shutdown` are idempotent within their
transitions (calling ``initialize`` twice is a no-op; calling ``shutdown``
twice is a no-op).

Thread safety
-------------
A :class:`Pipeline` instance is **not** thread-safe. It is designed to be
driven from a single thread (the pipeline owner). The internal state flag
is not guarded by a lock -- concurrent ``run()`` calls from multiple
threads would corrupt the lifecycle state. Owners that need to share a
pipeline across threads must serialise access externally.

Example
-------
    >>> from visioncore.pipeline.pipeline import Pipeline
    >>> from visioncore.pipeline.stage import DummyStage
    >>> from visioncore.pipeline.context import PipelineContext
    >>> p = Pipeline()
    >>> p.add_stage(DummyStage("capture"))
    >>> p.add_stage(DummyStage("detect"))
    >>> with p:
    ...     ctx = p.run(PipelineContext.empty(timestamp=1.0))
    >>> ctx.metadata["order"]
    ['capture', 'detect']
    >>> p.health_check()
    False
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from visioncore.pipeline.base import PipelineStage

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext
    from visioncore.plugin.registry import PluginRegistry


__all__ = ["Pipeline"]


logger = logging.getLogger(__name__)


class Pipeline:
    """Ordered, sequential executor of :class:`PipelineStage` instances.

    A pipeline owns an ordered list of stages. ``initialize()`` prepares
    them in order, ``run(context)`` drives them in order against a shared
    :class:`~visioncore.pipeline.context.PipelineContext`, and
    ``shutdown()`` tears them down in reverse order.

    Attributes:
        _stages: Ordered list of stages. Insertion order is preserved;
            ``process`` and ``initialize`` run in this order, ``shutdown``
            runs in reverse.
        _initialized: Whether :meth:`initialize` has been called.
        _shutdown: Whether :meth:`shutdown` has been called (terminal).
        _logger: Module-level logger.
    """

    __slots__ = ("_stages", "_initialized", "_shutdown", "_logger")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        """Initialise an empty pipeline in the NEW state.

        The pipeline starts with no stages and in the NEW lifecycle state
        (neither initialised nor shut down). Add stages with
        :meth:`add_stage`, then :meth:`initialize`, then :meth:`run`.
        """
        self._stages: list[PipelineStage] = []
        self._initialized: bool = False
        self._shutdown: bool = False
        self._logger: logging.Logger = logging.getLogger(__name__)
        self._logger.debug("Pipeline created (empty, NEW)")

    # ------------------------------------------------------------------
    # Stage registry
    # ------------------------------------------------------------------

    def add_stage(self, stage: PipelineStage) -> "Pipeline":
        """Append ``stage`` to the end of the pipeline.

        Stages may only be added while the pipeline is in the NEW state
        (before :meth:`initialize`). Adding a stage after initialisation
        would skip its ``initialize()`` and corrupt lifecycle accounting,
        so it is rejected.

        Parameters:
            stage: A concrete :class:`PipelineStage` instance. Must not be
                ``None`` and must be a :class:`PipelineStage` subclass
                instance.

        Returns:
            ``self``, to enable fluent chaining::

                p.add_stage(A()).add_stage(B()).add_stage(C())

        Raises:
            TypeError: If ``stage`` is ``None`` or not a
                :class:`PipelineStage`.
            RuntimeError: If the pipeline has already been initialised
                (lifecycle state is not NEW).

        Example:
            >>> p = Pipeline()
            >>> p.add_stage(DummyStage("a")) is p
            True
            >>> len(p)
            1
        """
        if stage is None or not isinstance(stage, PipelineStage):
            raise TypeError(
                f"stage must be a PipelineStage instance, got "
                f"{type(stage).__name__ if stage is not None else 'None'}"
            )
        if self._initialized or self._shutdown:
            raise RuntimeError(
                "cannot add_stage after initialize(); the pipeline is "
                "no longer in the NEW state"
            )
        self._stages.append(stage)
        self._logger.debug("add_stage: name=%s total=%d",
                           stage.name, len(self._stages))
        return self

    def remove_stage(self, stage: "PipelineStage | str") -> bool:
        """Remove a stage by instance identity or by name.

        Parameters:
            stage: Either a :class:`PipelineStage` instance (matched by
                identity, ``is``) or a stage name string (matched by
                ``stage.name`` equality). When matching by name, the
                **first** stage with that name is removed.

        Returns:
            ``True`` if a stage was found and removed, ``False`` otherwise.

        Raises:
            RuntimeError: If the pipeline has already been initialised
                (lifecycle state is not NEW). Removing stages after
                initialisation would desync lifecycle accounting.

        Example:
            >>> p = Pipeline()
            >>> s = DummyStage("x")
            >>> p.add_stage(s)
            >>> p.remove_stage(s)
            True
            >>> p.remove_stage("nonexistent")
            False
        """
        if self._initialized or self._shutdown:
            raise RuntimeError(
                "cannot remove_stage after initialize(); the pipeline is "
                "no longer in the NEW state"
            )
        if isinstance(stage, str):
            for i, s in enumerate(self._stages):
                if s.name == stage:
                    removed: PipelineStage = self._stages.pop(i)
                    self._logger.debug("remove_stage(by name): %s",
                                       removed.name)
                    return True
            return False
        # Match by identity.
        for i, s in enumerate(self._stages):
            if s is stage:
                removed = self._stages.pop(i)
                self._logger.debug("remove_stage(by identity): %s",
                                   removed.name)
                return True
        return False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialise every stage in insertion order.

        Idempotent: a second call is a no-op (the pipeline remembers it
        has been initialised). Stages' own ``initialize()`` methods are
        contractually idempotent too, so a double call at the pipeline
        level would be harmless even without the guard -- but the guard
        avoids the redundant work.

        If a stage's ``initialize()`` raises, the pipeline aborts: the
        failing stage and all stages **after** it are left uninitialised,
        and stages **before** it are **not** shut down automatically (the
        caller must invoke :meth:`shutdown` to clean up, or use the
        context manager which does so on exit). The exception propagates
        out of ``initialize()``.

        Raises:
            RuntimeError: If the pipeline has already been shut down
                (terminal state).

        Example:
            >>> p = Pipeline()
            >>> p.initialize()  # no stages -> no-op, but marks initialised
            >>> p._initialized
            True
        """
        if self._shutdown:
            raise RuntimeError(
                "cannot initialize() after shutdown(); the pipeline is "
                "in the terminal SHUT_DOWN state"
            )
        if self._initialized:
            self._logger.debug("initialize: already initialised (no-op)")
            return
        self._logger.info("initialize: %d stage(s) in order",
                          len(self._stages))
        for stage in self._stages:
            stage.initialize()
            self._logger.debug("initialize: stage '%s' done", stage.name)
        self._initialized = True

    def shutdown(self) -> None:
        """Shut down every stage in **reverse** insertion order.

        Idempotent: a second call is a no-op. Best-effort: each stage's
        ``shutdown()`` is isolated in ``try/except``; a failing stage is
        logged at ``WARNING`` level and does **not** prevent the remaining
        stages from shutting down. The first exception encountered (if
        any) is recorded but **not** re-raised -- teardown must complete
        regardless. This matches the established pattern in
        :class:`~visioncore.eventbus.dispatcher.Dispatcher` (swallow +
        log).

        Safe to call even if :meth:`initialize` was never called or
        raised partway through: only stages that were actually initialised
        are shut down. (In C1, the pipeline does not track per-stage
        init status, so ``shutdown`` calls every registered stage's
        ``shutdown()``; stages are contractually tolerant of being shut
        down without having been initialised.)

        Example:
            >>> p = Pipeline()
            >>> p.add_stage(DummyStage("a"))
            >>> p.initialize()
            >>> p.shutdown()
            >>> p._shutdown
            True
        """
        if self._shutdown:
            self._logger.debug("shutdown: already shut down (no-op)")
            return
        self._logger.info("shutdown: %d stage(s) in reverse order",
                          len(self._stages))
        # Reverse insertion order so dependant stages tear down first.
        for stage in reversed(self._stages):
            try:
                stage.shutdown()
                self._logger.debug("shutdown: stage '%s' done", stage.name)
            except Exception:  # noqa: BLE001 -- teardown must not abort
                self._logger.warning(
                    "shutdown: stage '%s' raised -- continuing teardown",
                    stage.name, exc_info=True,
                )
        self._shutdown = True
        self._initialized = False

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, context: "PipelineContext") -> "PipelineContext":
        """Run every stage's ``process()`` against ``context``, in order.

        The context is mutated in place by each stage and returned for
        convenience (callers may also just keep their own reference).
        Stages run synchronously; stage N+1 starts only after stage N
        returns.

        If a stage raises, the run aborts immediately: subsequent stages
        are not called for this run, and the exception propagates out.
        :meth:`shutdown` is **not** called automatically by ``run`` --
        use the context manager (``with pipeline: ...``) to guarantee
        teardown on exception.

        Parameters:
            context: The shared :class:`~visioncore.pipeline.context.PipelineContext`.
                Mutated in place; the stage must not replace the instance.

        Returns:
            The same ``context`` instance (mutated), for chaining.

        Raises:
            RuntimeError: If called before :meth:`initialize` (NEW state)
                or after :meth:`shutdown` (SHUT_DOWN state).
            Exception: Any exception raised by a stage's ``process()``,
                propagated unchanged.

        Example:
            >>> p = Pipeline()
            >>> p.add_stage(DummyStage("a"))
            >>> p.initialize()
            >>> ctx = p.run(PipelineContext.empty())
            >>> ctx.metadata["order"]
            ['a']
        """
        if self._shutdown:
            raise RuntimeError(
                "cannot run() after shutdown(); the pipeline is in the "
                "terminal SHUT_DOWN state"
            )
        if not self._initialized:
            raise RuntimeError(
                "cannot run() before initialize(); call initialize() "
                "first (or use the context manager: 'with pipeline: ...')"
            )
        self._logger.debug("run: %d stage(s) in order, ctx=%r",
                           len(self._stages), context)
        for stage in self._stages:
            stage.process(context)
            self._logger.debug("run: stage '%s' done, ctx=%s",
                               stage.name, context.summary())
        return context

    # ------------------------------------------------------------------
    # Aggregate health
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Return ``True`` iff **every** stage reports healthy.

        Short-circuits on the first unhealthy stage. Each stage's
        ``health_check()`` is contractually non-raising, but this method
        defends against buggy subclasses by wrapping each call in
        ``try/except`` (an exception is treated as unhealthy).

        Returns:
            ``True`` if all stages are healthy, ``False`` otherwise.
            An empty pipeline returns ``True`` (vacuously healthy).

        Example:
            >>> p = Pipeline()
            >>> p.health_check()  # empty pipeline -> vacuously healthy
            True
        """
        for stage in self._stages:
            try:
                if not stage.health_check():
                    return False
            except Exception:  # noqa: BLE001 -- defensive
                self._logger.warning(
                    "health_check: stage '%s' raised -- treating as unhealthy",
                    stage.name, exc_info=True,
                )
                return False
        return True

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        """A read-only tuple view of the registered stages (insertion order)."""
        return tuple(self._stages)

    @property
    def initialized(self) -> bool:
        """Whether :meth:`initialize` has been called and not since undone."""
        return self._initialized

    @property
    def shutdown_done(self) -> bool:
        """Whether :meth:`shutdown` has been called (terminal state)."""
        return self._shutdown

    def __len__(self) -> int:
        """Return the number of registered stages."""
        return len(self._stages)

    def __repr__(self) -> str:
        """Return a concise representation with stage count and lifecycle state."""
        state: str
        if self._shutdown:
            state = "SHUT_DOWN"
        elif self._initialized:
            state = "INITIALIZED"
        else:
            state = "NEW"
        names: str = ", ".join(s.name for s in self._stages)
        return (
            f"Pipeline(stages=[{names}], count={len(self._stages)}, "
            f"state={state})"
        )

    # ------------------------------------------------------------------
    # Context manager -- guaranteed teardown
    # ------------------------------------------------------------------

    def __enter__(self) -> "Pipeline":
        """Enter context: initialise the pipeline and return self.

        Calling ``__enter__`` on an already-initialised pipeline is a
        no-op (``initialize`` is idempotent). This lets a pipeline be
        reused across multiple ``with`` blocks.
        """
        self.initialize()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        """Exit context: shut the pipeline down. Does not suppress exceptions.

        ``shutdown`` is called regardless of whether the ``with`` body
        raised. The exception (if any) is **not** suppressed -- it
        propagates after teardown completes. This is the same contract
        as :class:`~visioncore.protocol.base.ProtocolAdapter.__exit__`.
        """
        self.shutdown()
        # Returning None (falsy) means exceptions are not suppressed.

    # ------------------------------------------------------------------
    # D8: Config-driven construction
    # ------------------------------------------------------------------

    @classmethod
    def load_config(
        cls,
        path: str,
        extra_stages: dict[str, type[PipelineStage]] | None = None,
        plugin_registry: "PluginRegistry | None" = None,
    ) -> "Pipeline":
        """Build a pipeline from a JSON / YAML config file.

        Convenience classmethod that delegates to
        :class:`~visioncore.pipeline.config.pipeline_config.PipelineConfig`.
        Reads the config at ``path``, resolves each stage's ``type``
        name, and returns a fully-assembled :class:`Pipeline` with
        stages added in config order.

        The pipeline is returned in the **NEW** state (not yet
        initialised) -- call ``initialize()`` or use ``with`` as usual.

        Parameters:
            path: Path to the config file (``.json`` / ``.yaml`` /
                ``.yml``).
            extra_stages: Additional stage classes keyed by name,
                merged into the built-in registry for this load.
            plugin_registry: Optional
                :class:`~visioncore.plugin.registry.PluginRegistry`
                for plugin-backed stage resolution.

        Returns:
            A configured :class:`Pipeline` in the NEW state.

        Raises:
            PipelineConfigError: If the config is invalid or unknown
                stage types are referenced.
        """
        from visioncore.pipeline.config.pipeline_config import PipelineConfig
        return PipelineConfig.build_from_file(
            path,
            extra_stages=extra_stages,
            plugin_registry=plugin_registry,
        )
