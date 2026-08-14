"""Concrete pipeline stages -- test doubles and reference implementations.

This module hosts stages that are useful for testing the pipeline
machinery itself, independent of any real detector / tracker /
target-manager. The flagship export is :class:`DummyStage`, a configurable
no-op stage that records its lifecycle calls and stamps the shared
context's metadata so tests can assert on stage ordering.

Scope (Milestone C1)
--------------------
C1 delivers **only** :class:`DummyStage`. Real stages (a capture stage, a
detection stage, a tracking stage, a target-management stage, a projection
stage) are introduced in subsequent milestones (C2+) as sibling modules or
subpackages under :mod:`visioncore.pipeline`. They will all subclass
:class:`~visioncore.pipeline.base.PipelineStage` and obey the same
four-method lifecycle.

Why a DummyStage?
-----------------
A pipeline framework is only as useful as its ability to be tested in
isolation. :class:`DummyStage` lets tests verify:

* **Stage ordering** -- append a marker to ``context.metadata["order"]``
  on each ``process()`` and assert the list matches insertion order.
* **Context passing** -- mutate the shared context and have the next stage
  observe the mutation, proving stages share one instance.
* **Exception propagation** -- raise on ``process()`` and assert the run
  aborts and subsequent stages are never called.
* **Lifecycle call counts** -- assert ``initialize_count``,
  ``process_count``, ``shutdown_count`` are exactly 1/1/1 after a clean
  run.

All of this without depending on numpy, model files, or any real vision
logic.

Thread safety
-------------
:class:`DummyStage` is not thread-safe. It is designed for serial pipeline
execution (the only mode the C1 pipeline supports). Its counters are plain
ints mutated on the calling thread.

Example
-------
    >>> from visioncore.pipeline.stage import DummyStage
    >>> from visioncore.pipeline.pipeline import Pipeline
    >>> from visioncore.pipeline.context import PipelineContext
    >>> p = Pipeline()
    >>> p.add_stage(DummyStage("a"))
    >>> p.add_stage(DummyStage("b"))
    >>> with p:
    ...     ctx = p.run(PipelineContext.empty())
    ... # initialize() x2, process() x2, shutdown() x2 called
    >>> ctx.metadata["order"]
    ['a', 'b']
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from visioncore.pipeline.base import PipelineStage

if TYPE_CHECKING:
    from visioncore.pipeline.context import PipelineContext


__all__ = ["DummyStage"]


logger = logging.getLogger(__name__)


class DummyStage(PipelineStage):
    """Configurable no-op stage for testing pipeline scheduling and lifecycle.

    :class:`DummyStage` does no real vision work. Its sole purpose is to
    make the pipeline's behaviour **observable** in tests: it counts its
    lifecycle calls and stamps the shared context's ``metadata`` so tests
    can assert on ordering, context sharing, and exception propagation.

    What it does on each lifecycle method
    --------------------------------------
    * :meth:`initialize` -- increments :attr:`initialize_count`; sets
      :attr:`healthy` to ``True`` (unless configured otherwise).
    * :meth:`process` -- increments :attr:`process_count` and appends
      :attr:`marker` to ``context.metadata.setdefault("order", [])``.
      If :attr:`raise_on_process` is set, raises it instead (after
      incrementing the count, so tests can confirm the stage was reached).
    * :meth:`shutdown` -- increments :attr:`shutdown_count`; sets
      :attr:`healthy` to ``False``. Never raises, even if configured to
      fail -- shutdown is contractually non-raising.
    * :meth:`health_check` -- returns :attr:`healthy`.

    Attributes:
        marker: The value appended to ``context.metadata["order"]`` on each
            ``process()``. Defaults to the stage's name, so the order list
            is human-readable out of the box.
        raise_on_process: An exception instance to raise on the next
            ``process()`` call, or ``None`` (default) to process normally.
            Used by exception-propagation tests. The exception is raised
            **after** the count is incremented and **before** the marker is
            appended, so tests can confirm the stage was entered.
        _healthy: Backing field for :attr:`healthy`. Toggled by
            :meth:`initialize` (to ``True``) and :meth:`shutdown`
            (to ``False``). May be set directly by tests to simulate an
            unhealthy stage.
        initialize_count: Number of times :meth:`initialize` was called.
        process_count: Number of times :meth:`process` was called.
        shutdown_count: Number of times :meth:`shutdown` was called.

    Example:
        >>> s = DummyStage("probe")
        >>> s.initialize()
        >>> ctx = PipelineContext.empty()
        >>> s.process(ctx)
        >>> ctx.metadata["order"]
        ['probe']
        >>> (s.initialize_count, s.process_count, s.shutdown_count)
        (1, 1, 0)
    """

    __slots__ = (
        "marker",
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
        marker: Any | None = None,
        raise_on_process: BaseException | None = None,
        healthy: bool = False,
    ) -> None:
        """Initialise a DummyStage with optional test behaviour.

        Parameters:
            name: Stage name. Defaults to ``"DummyStage"``. Also used as
                the default ``marker`` when ``marker`` is ``None``.
            marker: Value appended to ``context.metadata["order"]`` on
                each ``process()``. Defaults to ``name``.
            raise_on_process: An exception instance to raise on
                ``process()`` instead of stamping the context. ``None``
                (default) means process normally.
            healthy: Initial value of :attr:`healthy`. Defaults to
                ``False`` (stage is unhealthy until :meth:`initialize`
                runs, which sets it to ``True``).

        Example:
            >>> s = DummyStage("a")
            >>> s.name, s.marker
            ('a', 'a')
            >>> s.healthy
            False
        """
        super().__init__(name=name)
        resolved_name: str = self._name
        self.marker: Any = marker if marker is not None else resolved_name
        self.raise_on_process: BaseException | None = raise_on_process
        self._healthy: bool = healthy
        self.initialize_count: int = 0
        self.process_count: int = 0
        self.shutdown_count: int = 0
        logger.debug("DummyStage created: name=%s marker=%s",
                     resolved_name, self.marker)

    # ------------------------------------------------------------------
    # Lifecycle (PipelineStage contract)
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Mark the stage initialised and healthy.

        Idempotent: repeated calls only increment the counter (they do not
        toggle ``healthy`` back and forth). A stage that was configured
        unhealthy via the ``healthy=False`` constructor default becomes
        healthy after the first ``initialize()``.
        """
        self.initialize_count += 1
        self._healthy = True
        logger.debug("DummyStage.initialize: name=%s count=%d",
                     self._name, self.initialize_count)

    def process(self, context: "PipelineContext") -> None:
        """Stamp the context's ``metadata["order"]`` with this stage's marker.

        If :attr:`raise_on_process` is set, raise it instead (after
        incrementing :attr:`process_count` but before stamping), so tests
        can confirm the stage was reached and that the run aborted before
        stamping.

        Parameters:
            context: The shared :class:`~visioncore.pipeline.context.PipelineContext`.
                Mutated: ``context.metadata["order"]`` gets ``self.marker``
                appended (unless raising).
        """
        self.process_count += 1
        if self.raise_on_process is not None:
            logger.debug("DummyStage.process: name=%s RAISING %s",
                         self._name, type(self.raise_on_process).__name__)
            raise self.raise_on_process
        order: list[Any] = context.metadata.setdefault("order", [])
        order.append(self.marker)
        logger.debug("DummyStage.process: name=%s count=%d order=%s",
                     self._name, self.process_count, order)

    def shutdown(self) -> None:
        """Mark the stage shut down and unhealthy. Never raises."""
        self.shutdown_count += 1
        self._healthy = False
        logger.debug("DummyStage.shutdown: name=%s count=%d",
                     self._name, self.shutdown_count)

    def health_check(self) -> bool:
        """Return the stage's current health flag. Never raises."""
        return self._healthy

    # ------------------------------------------------------------------
    # Test convenience
    # ------------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        """Current health flag (backed by ``_healthy``).

        Read-only as a property; tests that want to simulate an unhealthy
        stage set ``stage._healthy = False`` directly (the backing field
        is exposed via ``__slots__``).
        """
        return self._healthy

    def reset(self) -> None:
        """Reset all counters and the health flag to their post-construction state.

        Convenience for tests that reuse a stage across multiple runs
        without re-instantiating. Does **not** reset :attr:`marker` or
        :attr:`raise_on_process` -- only the lifecycle counters and health.
        """
        self._healthy = False
        self.initialize_count = 0
        self.process_count = 0
        self.shutdown_count = 0
