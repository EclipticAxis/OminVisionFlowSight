"""ShadowPipelineRunner -- runs a Pipeline in shadow mode with statistics.

Defines :class:`ShadowPipelineRunner`, a standalone runner that drives a
:class:`~visioncore.pipeline.pipeline.Pipeline` against a
:class:`~visioncore.source.base.FrameSource`, collecting per-stage timing,
FPS, context-lifecycle, and exception statistics. The runner is a **shadow**:
it produces no output that reaches the GUI, the Recorder, or the existing
:class:`~visioncore.pipeline.base.PipelineStage`-ignorant ``InferWorker``.
Its sole output is the accumulated :class:`ShadowStats`.

Scope (Milestone C7 -- Pipeline Shadow Integration)
----------------------------------------------------
C7 delivers the shadow runner + statistics. The existing
``ai/inference.py`` ``InferWorker`` is **not modified** -- the shadow
pipeline runs alongside it, fed by its own
:class:`~visioncore.source.base.FrameSource`, and its results are discarded
(only stats are kept). This is the validation scaffold that lets the new
Pipeline architecture be benchmarked against the legacy InferWorker without
risking the live system.

What "shadow" means
-------------------
* **No output side effects**: the runner writes nothing to the GUI, the
  Recorder, or any external consumer. The
  :class:`~visioncore.pipeline.context.PipelineContext` it creates is
  local and discarded after stats are extracted.
* **No InferWorker modification**: the legacy ``InferWorker`` continues
  to drive detection / tracking / GUI updates exactly as before. The
  shadow runner is a separate, self-contained component.
* **Statistics only**: the runner's product is a :class:`ShadowStats`
  object -- FPS, per-stage timing, context lifecycle, exception counts.
  This is the data that informs the eventual Pipeline-vs-InferWorker
  cutover decision.

Statistics collected
--------------------
* **Pipeline FPS**: ``stats.fps`` -- frames processed per second.
* **Stage timing**: ``stats.stage_stats[name]`` -- per-stage call count,
  total / min / max / avg time, error count.
* **Context lifecycle**: ``stats.context_count`` -- number of
  :class:`~visioncore.pipeline.context.PipelineContext` instances created
  (one per frame; equals successful frame count).
* **Exception statistics**: ``stats.exceptions`` -- list of
  ``(stage_name, exception_type_name)`` tuples; ``stats.error_frame_count``
  -- frames with at least one stage error; ``stats.error_rate`` -- ratio.

Error handling
--------------
The shadow runner **absorbs** stage exceptions (unlike
:class:`~visioncore.pipeline.pipeline.Pipeline.run`, which propagates
them). When a stage's ``process()`` raises:

1. The exception is recorded in :class:`ShadowStats`.
2. The stage's error count is incremented.
3. Remaining stages for that frame are **skipped** (the context is in an
   unknown state after a failure).
4. The runner advances to the next frame.

This lets the shadow collect meaningful statistics across many frames even
if some fail -- the whole point of a validation scaffold is to see what
breaks and how often, not to abort on the first error.

Thread safety
-------------
:class:`ShadowPipelineRunner` is not thread-safe. For a live shadow that
runs alongside ``InferWorker`` (which lives on its own QThread), wrap the
runner in a dedicated ``QThread`` and drive it via signals. The runner
itself is a synchronous, single-threaded component.

Example
-------
    >>> from visioncore.pipeline import Pipeline
    >>> from visioncore.pipeline.shadow import ShadowPipelineRunner
    >>> from visioncore.source import DummyFrameSource
    >>> from visioncore.pipeline.stages import (
    ...     DetectorStage, DummyDetector,
    ...     TrackerStage, DummyTracker,
    ...     TargetStage, DummyTargetManager,
    ...     EventStage, DummyEventBus,
    ... )
    >>> p = Pipeline()
    >>> p.add_stage(DetectorStage(DummyDetector()))
    >>> p.add_stage(TrackerStage(DummyTracker()))
    >>> p.add_stage(TargetStage(DummyTargetManager()))
    >>> p.add_stage(EventStage(DummyEventBus()))
    >>> runner = ShadowPipelineRunner(p, DummyFrameSource(loop=False))
    >>> runner.initialize()
    >>> stats = runner.run_batch(max_frames=10)
    >>> stats.frame_count  # 1 frame in the dummy source (loop=False)
    1
    >>> stats.fps > 0
    True
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from math import inf
from typing import TYPE_CHECKING

from visioncore.pipeline.context import PipelineContext
from visioncore.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from visioncore.source.base import FrameSource


__all__ = ["StageStats", "ShadowStats", "ShadowPipelineRunner"]


logger = logging.getLogger(__name__)


# ======================================================================
# Statistics accumulators
# ======================================================================

@dataclass
class StageStats:
    """Per-stage timing and error accumulator.

    Accumulated across all ``process()`` calls for one stage within a
    shadow run. Mutable -- updated in place by
    :class:`ShadowPipelineRunner`.

    Attributes:
        name: The stage's display name (``stage.name``).
        call_count: Number of ``process()`` calls attempted (including
            those that raised).
        total_time: Cumulative wall-clock time in seconds across all calls.
        min_time: Shortest single ``process()`` time in seconds.
        max_time: Longest single ``process()`` time in seconds.
        error_count: Number of ``process()`` calls that raised.
        last_error: The most recent exception raised by this stage, or
            ``None`` if none. Kept for diagnostics; not exhaustive.
    """

    name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = inf
    max_time: float = 0.0
    error_count: int = 0
    last_error: BaseException | None = None

    @property
    def avg_time(self) -> float:
        """Mean ``process()`` time in seconds (0.0 if no calls)."""
        return self.total_time / self.call_count if self.call_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Fraction of calls that raised (0.0 if no calls)."""
        return self.error_count / self.call_count if self.call_count > 0 else 0.0

    def __repr__(self) -> str:
        """Compact representation with key metrics."""
        return (
            f"StageStats(name={self.name!r}, calls={self.call_count}, "
            f"avg={self.avg_time * 1e6:.1f}us, "
            f"min={self.min_time * 1e6:.1f}us, "
            f"max={self.max_time * 1e6:.1f}us, "
            f"errors={self.error_count})"
        )


@dataclass
class ShadowStats:
    """Aggregate statistics for a shadow pipeline run.

    Accumulated across all :meth:`ShadowPipelineRunner.run_once` calls.
    Mutable -- updated in place by the runner.

    Attributes:
        frame_count: Number of frames fully or partially processed
            (includes frames where a stage raised).
        skip_count: Number of frames skipped because the source returned
            ``None`` (exhaustion).
        error_frame_count: Frames where at least one stage raised.
        total_frame_time: Cumulative wall-clock time (seconds) for all
            processed frames (``time.perf_counter`` delta per frame).
        context_count: Number of :class:`~visioncore.pipeline.context.PipelineContext`
            instances created (one per processed frame; equals
            ``frame_count``).
        stage_stats: Per-stage accumulators keyed by stage name.
        exceptions: List of ``(stage_name, exception_type_name)`` tuples
            for every stage exception, in occurrence order. May contain
            duplicates (one per occurrence).
    """

    frame_count: int = 0
    skip_count: int = 0
    error_frame_count: int = 0
    total_frame_time: float = 0.0
    context_count: int = 0
    stage_stats: dict[str, StageStats] = field(default_factory=dict)
    exceptions: list[tuple[str, str]] = field(default_factory=list)

    @property
    def fps(self) -> float:
        """Pipeline throughput in frames per second (0.0 if no frames)."""
        return self.frame_count / self.total_frame_time if self.total_frame_time > 0 else 0.0

    @property
    def avg_frame_time(self) -> float:
        """Mean per-frame processing time in seconds (0.0 if no frames)."""
        return self.total_frame_time / self.frame_count if self.frame_count > 0 else 0.0

    @property
    def error_rate(self) -> float:
        """Fraction of processed frames with at least one stage error."""
        return self.error_frame_count / self.frame_count if self.frame_count > 0 else 0.0

    def stage_stat(self, name: str) -> StageStats:
        """Return the accumulator for stage ``name`` (creating if absent)."""
        if name not in self.stage_stats:
            self.stage_stats[name] = StageStats(name=name)
        return self.stage_stats[name]

    def summary(self) -> str:
        """Return a multi-line human-readable summary for reports/logs.

        Includes FPS, frame/error counts, per-stage timing, and the first
        few exception types. Designed for inclusion in a benchmark report.
        """
        lines: list[str] = []
        lines.append(f"frames={self.frame_count}  skips={self.skip_count}  "
                     f"errors={self.error_frame_count}  "
                     f"error_rate={self.error_rate:.2%}")
        lines.append(f"total_time={self.total_frame_time * 1e3:.2f}ms  "
                     f"avg_frame={self.avg_frame_time * 1e6:.1f}us  "
                     f"fps={self.fps:.1f}")
        lines.append(f"contexts_created={self.context_count}")
        if self.stage_stats:
            lines.append("per-stage:")
            for s in self.stage_stats.values():
                lines.append(f"  {s!r}")
        if self.exceptions:
            # Summarise exception types (count per type).
            counts: dict[str, int] = {}
            for _stage, exc_type in self.exceptions:
                key = f"{exc_type}"
                counts[key] = counts.get(key, 0) + 1
            lines.append(f"exceptions ({len(self.exceptions)} total):")
            for exc_type, cnt in sorted(counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {exc_type}: {cnt}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ShadowStats(frames={self.frame_count}, fps={self.fps:.1f}, "
            f"errors={self.error_frame_count})"
        )


# ======================================================================
# ShadowPipelineRunner
# ======================================================================

class ShadowPipelineRunner:
    """Runs a Pipeline in shadow mode, collecting statistics only.

    The runner drives a :class:`~visioncore.pipeline.pipeline.Pipeline`
    against a :class:`~visioncore.source.base.FrameSource`, frame by
    frame, timing each stage and recording exceptions. It produces **no**
    output that reaches the GUI, the Recorder, or the legacy
    ``InferWorker`` -- the
    :class:`~visioncore.pipeline.context.PipelineContext` is local and
    discarded after stats extraction. This is the validation scaffold for
    benchmarking the new Pipeline architecture against the existing
    InferWorker without risking the live system.

    The runner iterates ``pipeline.stages`` manually (rather than calling
    :meth:`~visioncore.pipeline.pipeline.Pipeline.run`) so it can time
    each stage individually and absorb per-stage exceptions without
    aborting the batch.

    Attributes:
        _pipeline: The wrapped :class:`~visioncore.pipeline.pipeline.Pipeline`.
        _source: The :class:`~visioncore.source.base.FrameSource` providing
            frames.
        _name: A label for this runner (for logging / diagnostics).
        _stats: The accumulated :class:`ShadowStats`.
        _initialized: Whether :meth:`initialize` has been called.

    Example:
        >>> runner = ShadowPipelineRunner(pipeline, source, name="shadow-1")
        >>> runner.name
        'shadow-1'
    """

    __slots__ = ("_pipeline", "_source", "_name", "_stats", "_initialized")

    def __init__(
        self,
        pipeline: Pipeline,
        frame_source: "FrameSource",
        *,
        name: str = "shadow",
    ) -> None:
        """Construct a ShadowPipelineRunner.

        Parameters:
            pipeline: An **initialised-capable** :class:`~visioncore.pipeline.pipeline.Pipeline`
                whose stages form the shadow chain (typically DetectorStage
                -> TrackerStage -> TargetStage -> EventStage). The runner
                calls ``pipeline.initialize()`` / ``pipeline.shutdown()``.
                Stages must be added before passing the pipeline here (the
                pipeline rejects ``add_stage`` after ``initialize``).
            frame_source: The :class:`~visioncore.source.base.FrameSource`
                providing frames. The runner calls ``source.open()`` /
                ``source.close()``. For a live shadow, this is a source
                tapping the same camera feed as ``InferWorker``; for
                benchmarks, a :class:`~visioncore.source.dummy_source.DummyFrameSource`.
            name: A label for this runner (default ``"shadow"``), used in
                logs and diagnostics.

        Raises:
            TypeError: If ``pipeline`` is not a
                :class:`~visioncore.pipeline.pipeline.Pipeline` or
                ``frame_source`` is not a
                :class:`~visioncore.source.base.FrameSource`.
        """
        if not isinstance(pipeline, Pipeline):
            raise TypeError(
                f"pipeline must be a Pipeline, got {type(pipeline).__name__}"
            )
        # Lazy import to avoid a hard runtime dependency on the source
        # package at module load (the source package is a sibling; this
        # keeps the import graph clean and matches the TYPE_CHECKING
        # convention used elsewhere).
        from visioncore.source.base import FrameSource
        if not isinstance(frame_source, FrameSource):
            raise TypeError(
                f"frame_source must be a FrameSource, got "
                f"{type(frame_source).__name__}"
            )
        self._pipeline: Pipeline = pipeline
        self._source: "FrameSource" = frame_source
        self._name: str = name
        self._stats: ShadowStats = ShadowStats()
        self._initialized: bool = False
        logger.debug("ShadowPipelineRunner created: name=%s stages=%d",
                     self._name, len(self._pipeline))

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """The runner's label."""
        return self._name

    @property
    def stats(self) -> ShadowStats:
        """The accumulated :class:`ShadowStats` (live, mutable)."""
        return self._stats

    @property
    def initialized(self) -> bool:
        """Whether :meth:`initialize` has been called."""
        return self._initialized

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialise the pipeline and open the source.

        Idempotent: a second call is a no-op. Calls
        ``pipeline.initialize()`` then ``source.open()`` so the source is
        ready before the first frame read.
        """
        if self._initialized:
            logger.debug("ShadowPipelineRunner.initialize: already (no-op)")
            return
        self._pipeline.initialize()
        self._source.open()
        self._initialized = True
        logger.info("ShadowPipelineRunner initialised: name=%s stages=%d",
                    self._name, len(self._pipeline))

    def shutdown(self) -> None:
        """Shut the pipeline down and close the source.

        Idempotent. Calls ``pipeline.shutdown()`` (best-effort, reverse
        order) then ``source.close()`` (idempotent, non-raising).
        """
        if not self._initialized:
            logger.debug("ShadowPipelineRunner.shutdown: not initialised (no-op)")
            return
        self._pipeline.shutdown()
        self._source.close()
        self._initialized = False
        logger.info("ShadowPipelineRunner shut down: name=%s", self._name)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_once(self) -> PipelineContext:
        """Run one frame through the pipeline; record stats; return context.

        Creates a fresh :class:`~visioncore.pipeline.context.PipelineContext`,
        reads a frame from the source, and drives each stage's ``process()``
        in order with per-stage timing. If a stage raises, the exception is
        recorded, remaining stages are skipped, and the frame is counted as
        an error frame -- but the runner does **not** re-raise (shadow
        absorbs errors to keep collecting stats).

        If the source returns ``None`` (exhausted), the frame is counted
        as a skip and no stages run.

        Returns:
            The :class:`~visioncore.pipeline.context.PipelineContext` for
            this frame (may be partially populated if a stage raised).

        Raises:
            RuntimeError: If called before :meth:`initialize`.
        """
        if not self._initialized:
            raise RuntimeError(
                "ShadowPipelineRunner.run_once before initialize(); "
                "call initialize() first"
            )
        ctx: PipelineContext = PipelineContext.empty()
        self._stats.context_count += 1

        # Read frame.
        frame = self._source.read()
        if frame is None:
            self._stats.skip_count += 1
            logger.debug("run_once: source exhausted (skip)")
            return ctx
        ctx.frame = frame
        ctx.timestamp = frame.timestamp

        # Drive stages with per-stage timing.
        frame_t0: float = time.perf_counter()
        had_error: bool = False
        for stage in self._pipeline.stages:
            ss: StageStats = self._stats.stage_stat(stage.name)
            ss.call_count += 1
            stage_t0: float = time.perf_counter()
            try:
                stage.process(ctx)
            except Exception as exc:  # noqa: BLE001 -- shadow absorbs
                ss.error_count += 1
                ss.last_error = exc
                self._stats.exceptions.append(
                    (stage.name, type(exc).__name__)
                )
                had_error = True
                logger.warning(
                    "shadow stage '%s' raised %s: %s (absorbed, skipping rest)",
                    stage.name, type(exc).__name__, exc,
                )
                break  # skip remaining stages for this frame
            finally:
                dt: float = time.perf_counter() - stage_t0
                ss.total_time += dt
                if dt < ss.min_time:
                    ss.min_time = dt
                if dt > ss.max_time:
                    ss.max_time = dt

        frame_dt: float = time.perf_counter() - frame_t0
        self._stats.total_frame_time += frame_dt
        self._stats.frame_count += 1
        if had_error:
            self._stats.error_frame_count += 1

        logger.debug(
            "run_once: name=%s frame_id=%s stages=%d time=%.1fus errors=%s",
            self._name, getattr(frame, "frame_id", "?"),
            len(self._pipeline), frame_dt * 1e6, had_error,
        )
        return ctx

    def run_batch(
        self,
        max_frames: int = 1000,
        *,
        stop_on_error_rate: float | None = None,
    ) -> ShadowStats:
        """Run up to ``max_frames`` frames; return accumulated stats.

        Repeatedly calls :meth:`run_once` until one of:
        * ``max_frames`` frames have been processed;
        * the source is exhausted (``read()`` returns ``None``);
        * ``stop_on_error_rate`` (if set) is exceeded by the current
          error rate.

        Parameters:
            max_frames: Upper bound on processed frames (default 1000).
                Skips (source exhaustion) do not count toward this limit
                -- the first skip ends the batch.
            stop_on_error_rate: If set (e.g. ``0.5``), abort the batch
                once the error rate exceeds this fraction. Useful for
                stopping a benchmark when the pipeline is clearly broken.

        Returns:
            The accumulated :class:`ShadowStats` (also available as
            ``self.stats``).
        """
        processed: int = 0
        while processed < max_frames:
            ctx: PipelineContext = self.run_once()
            # A skip means the source is exhausted -- end the batch.
            if self._stats.skip_count > 0 and ctx.frame is None:
                # This frame was a skip; check if it was the first skip.
                # (run_once already incremented skip_count.)
                break
            processed += 1
            if (stop_on_error_rate is not None
                    and self._stats.error_rate > stop_on_error_rate
                    and self._stats.frame_count >= 10):
                logger.info(
                    "run_batch: stopping early, error_rate=%.2f > %.2f",
                    self._stats.error_rate, stop_on_error_rate,
                )
                break
        logger.info(
            "run_batch: name=%s processed=%d frames=%d skips=%d errors=%d fps=%.1f",
            self._name, processed, self._stats.frame_count,
            self._stats.skip_count, self._stats.error_frame_count,
            self._stats.fps,
        )
        return self._stats

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "ShadowPipelineRunner":
        self.initialize()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.shutdown()
        # Does not suppress exceptions.

    def __repr__(self) -> str:
        state: str = "INITIALIZED" if self._initialized else "NEW"
        return (
            f"ShadowPipelineRunner(name={self._name!r}, "
            f"stages={len(self._pipeline)}, state={state}, "
            f"frames={self._stats.frame_count})"
        )
