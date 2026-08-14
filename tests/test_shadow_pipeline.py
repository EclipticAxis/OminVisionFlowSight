"""ShadowPipelineRunner unit tests (Milestone C7).

Functional test suite: no pytest, no test classes, ``__main__`` runner.

Covers:
    - Construction & type validation (rejects non-Pipeline / non-FrameSource)
    - Lifecycle (initialize idempotent, shutdown idempotent, context manager)
    - run_once: creates context, reads frame, drives stages, records stats
    - run_once with exhausted source: skip, no stages run
    - run_batch: processes N frames, stops on exhaustion
    - Per-stage timing: stage_stats has call_count / total / min / max
    - Exception absorption: stage error recorded, remaining skipped, batch continues
    - Statistics: fps, error_rate, context_count, exceptions list
    - No output side effects (context is local, not shared)
    - End-to-end: DummyFrameSource -> Detector -> Tracker -> Target -> Event
"""
from __future__ import annotations

import traceback
from typing import Any

import numpy as np

from visioncore.core.detection import BBox, Detection
from visioncore.core.frame import Frame
from visioncore.core.track import Track
from visioncore.pipeline import (
    Pipeline,
    PipelineContext,
    ShadowPipelineRunner,
    ShadowStats,
    StageStats,
)
from visioncore.pipeline.stages import (
    DetectorStage,
    DummyDetector,
    DummyEventBus,
    DummyTargetManager,
    DummyTracker,
    EventStage,
    TargetStage,
    TrackerStage,
)
from visioncore.source import DummyFrameSource


# ======================================================================
# Test helpers
# ======================================================================

class _Raises:
    __slots__ = ("expected", "match", "caught")

    def __init__(self, expected, match=None):
        self.expected = expected
        self.match = match
        self.caught = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(
                f"expected {self.expected.__name__}, but no exception was raised")
        if not isinstance(exc_val, self.expected):
            raise AssertionError(
                f"expected {self.expected.__name__}, got "
                f"{type(exc_val).__name__}: {exc_val}")
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(f"expected {self.match!r}, got {str(exc_val)!r}")
        self.caught = exc_val
        return True


def raises(expected, match=None):
    return _Raises(expected, match=match)


def _make_frame(frame_id: int = 0) -> Frame:
    return Frame(frame_id, float(frame_id), "cam0",
                 np.zeros((8, 8, 3), dtype=np.uint8))


def _make_pipeline() -> Pipeline:
    """Build the standard 4-stage shadow pipeline with Dummy backends."""
    p = Pipeline()
    p.add_stage(DetectorStage(DummyDetector(), name="detect"))
    p.add_stage(TrackerStage(DummyTracker(), name="track"))
    p.add_stage(TargetStage(DummyTargetManager(), name="targets"))
    p.add_stage(EventStage(DummyEventBus(), name="events"))
    return p


# ======================================================================
# 1. Construction
# ======================================================================

def test_construction_stores_pipeline_and_source():
    p = _make_pipeline()
    src = DummyFrameSource()
    runner = ShadowPipelineRunner(p, src, name="s1")
    assert runner.name == "s1"
    assert runner.initialized is False
    assert runner.stats.frame_count == 0


def test_construction_rejects_non_pipeline():
    src = DummyFrameSource()
    with raises(TypeError, match="Pipeline"):
        ShadowPipelineRunner("not a pipeline", src)  # type: ignore[arg-type]


def test_construction_rejects_non_frame_source():
    p = _make_pipeline()
    with raises(TypeError, match="FrameSource"):
        ShadowPipelineRunner(p, "not a source")  # type: ignore[arg-type]


def test_default_name_is_shadow():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    assert runner.name == "shadow"


# ======================================================================
# 2. Lifecycle
# ======================================================================

def test_initialize_pipeline_and_source():
    p = _make_pipeline()
    src = DummyFrameSource()
    runner = ShadowPipelineRunner(p, src)
    runner.initialize()
    assert runner.initialized is True
    assert src.is_open() is True
    # Pipeline stages initialised.
    for stage in p.stages:
        assert stage.health_check() is True


def test_initialize_is_idempotent():
    p = _make_pipeline()
    src = DummyFrameSource()
    runner = ShadowPipelineRunner(p, src)
    runner.initialize()
    runner.initialize()  # no-op
    assert src.open_count == 1


def test_shutdown_closes_source():
    p = _make_pipeline()
    src = DummyFrameSource()
    runner = ShadowPipelineRunner(p, src)
    runner.initialize()
    runner.shutdown()
    assert src.is_open() is False
    assert runner.initialized is False


def test_shutdown_is_idempotent():
    p = _make_pipeline()
    src = DummyFrameSource()
    runner = ShadowPipelineRunner(p, src)
    runner.initialize()
    runner.shutdown()
    runner.shutdown()  # no-op, must not raise
    assert src.close_count == 1


def test_context_manager():
    p = _make_pipeline()
    src = DummyFrameSource()
    with ShadowPipelineRunner(p, src) as runner:
        assert runner.initialized is True
    assert runner.initialized is False
    assert src.is_open() is False


def test_run_once_before_initialize_raises():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    with raises(RuntimeError, match="initialize"):
        runner.run_once()


# ======================================================================
# 3. run_once
# ======================================================================

def test_run_once_processes_one_frame():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    ctx = runner.run_once()
    assert ctx.frame is not None
    assert len(ctx.detections) == 1
    assert len(ctx.tracks) == 1
    assert len(ctx.targets) == 1
    assert len(ctx.target_states) == 1
    assert runner.stats.frame_count == 1


def test_run_once_increments_context_count():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    runner.run_once()
    runner.run_once()
    runner.run_once()
    assert runner.stats.context_count == 3
    assert runner.stats.frame_count == 3


def test_run_once_with_exhausted_source_skips():
    src = DummyFrameSource(loop=False)  # 1 frame, then None
    runner = ShadowPipelineRunner(_make_pipeline(), src)
    runner.initialize()
    # First read consumes the single frame.
    runner.run_once()
    assert runner.stats.frame_count == 1
    assert runner.stats.skip_count == 0
    # Second read: source exhausted -> skip.
    runner.run_once()
    assert runner.stats.frame_count == 1  # unchanged
    assert runner.stats.skip_count == 1


def test_run_once_records_stage_timing():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    runner.run_once()
    stats = runner.stats
    # All 4 stages should have stats.
    assert set(stats.stage_stats.keys()) == {"detect", "track", "targets", "events"}
    for ss in stats.stage_stats.values():
        assert ss.call_count == 1
        assert ss.total_time > 0.0
        assert ss.min_time > 0.0
        assert ss.max_time >= ss.min_time
        assert ss.error_count == 0


def test_run_once_does_not_share_context_across_calls():
    """Each run_once creates a fresh context (no cross-frame leakage)."""
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    ctx1 = runner.run_once()
    ctx2 = runner.run_once()
    assert ctx1 is not ctx2  # different instances
    # ctx1's detections should not accumulate into ctx2.
    assert len(ctx2.detections) == 1  # fresh, not 2


# ======================================================================
# 4. run_batch
# ======================================================================

def test_run_batch_processes_multiple_frames():
    src = DummyFrameSource()  # loop=True -> infinite
    runner = ShadowPipelineRunner(_make_pipeline(), src)
    runner.initialize()
    stats = runner.run_batch(max_frames=50)
    assert stats.frame_count == 50
    assert stats.skip_count == 0
    assert stats.context_count == 50


def test_run_batch_stops_on_exhaustion():
    src = DummyFrameSource(loop=False)  # 1 frame
    runner = ShadowPipelineRunner(_make_pipeline(), src)
    runner.initialize()
    stats = runner.run_batch(max_frames=100)
    assert stats.frame_count == 1
    assert stats.skip_count == 1  # the exhausted read


def test_run_batch_returns_stats():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    stats = runner.run_batch(max_frames=10)
    assert stats is runner.stats  # same object


def test_run_batch_stop_on_error_rate():
    """run_batch aborts early when error_rate exceeds threshold."""
    # Build a pipeline where the tracker always fails.
    p = Pipeline()
    p.add_stage(DetectorStage(DummyDetector(), name="detect"))
    p.add_stage(TrackerStage(
        DummyTracker(raise_on_update=RuntimeError("boom")), name="track"))
    runner = ShadowPipelineRunner(p, DummyFrameSource())
    runner.initialize()
    stats = runner.run_batch(max_frames=100, stop_on_error_rate=0.5)
    # Every frame errors (tracker always raises). Should stop early.
    assert stats.error_frame_count == stats.frame_count
    assert stats.frame_count < 100  # stopped early
    assert stats.error_rate == 1.0


# ======================================================================
# 5. Exception absorption
# ======================================================================

def test_stage_exception_is_absorbed_and_recorded():
    p = Pipeline()
    p.add_stage(DetectorStage(DummyDetector(), name="detect"))
    p.add_stage(TrackerStage(
        DummyTracker(raise_on_update=ValueError("track fail")), name="track"))
    p.add_stage(TargetStage(DummyTargetManager(), name="targets"))
    runner = ShadowPipelineRunner(p, DummyFrameSource())
    runner.initialize()
    ctx = runner.run_once()
    # detect ran, track raised, targets skipped.
    assert len(ctx.detections) == 1
    assert ctx.tracks == []  # track raised before writing
    assert ctx.targets == []  # targets stage skipped
    # Stats recorded the error.
    assert runner.stats.error_frame_count == 1
    assert runner.stats.exceptions == [("track", "ValueError")]
    track_ss = runner.stats.stage_stats["track"]
    assert track_ss.error_count == 1
    assert isinstance(track_ss.last_error, ValueError)


def test_exception_skips_remaining_stages():
    """When a stage raises, subsequent stages don't run for that frame."""
    p = Pipeline()
    p.add_stage(DetectorStage(DummyDetector(), name="detect"))
    p.add_stage(TrackerStage(
        DummyTracker(raise_on_update=RuntimeError("x")), name="track"))
    tgt = DummyTargetManager()
    p.add_stage(TargetStage(tgt, name="targets"))
    runner = ShadowPipelineRunner(p, DummyFrameSource())
    runner.initialize()
    runner.run_once()
    # targets stage never called (track raised first).
    assert tgt.update_count == 0


def test_batch_continues_after_exception():
    """A stage error on one frame doesn't abort the batch."""
    p = Pipeline()
    p.add_stage(DetectorStage(DummyDetector(), name="detect"))
    # Tracker raises on first call, then works.
    trk = DummyTracker(raise_on_update=RuntimeError("transient"))
    p.add_stage(TrackerStage(trk, name="track"))
    runner = ShadowPipelineRunner(p, DummyFrameSource())
    runner.initialize()
    runner.run_once()  # track raises
    trk.raise_on_update = None  # clear
    runner.run_once()  # track works
    assert runner.stats.frame_count == 2
    assert runner.stats.error_frame_count == 1
    assert runner.stats.exceptions == [("track", "RuntimeError")]


# ======================================================================
# 6. Statistics
# ======================================================================

def test_fps_is_positive_after_frames():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    runner.run_batch(max_frames=20)
    assert runner.stats.fps > 0.0


def test_error_rate_zero_on_clean_run():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    runner.run_batch(max_frames=10)
    assert runner.stats.error_rate == 0.0
    assert runner.stats.error_frame_count == 0


def test_stage_stats_avg_time():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    runner.run_batch(max_frames=10)
    for ss in runner.stats.stage_stats.values():
        assert ss.call_count == 10
        assert ss.avg_time > 0.0
        assert ss.min_time <= ss.avg_time <= ss.max_time


def test_stats_summary_is_readable():
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    runner.run_batch(max_frames=5)
    s = runner.stats.summary()
    assert "frames=5" in s
    assert "fps=" in s
    assert "per-stage:" in s
    assert "detect" in s


def test_stats_summary_includes_exceptions():
    p = Pipeline()
    p.add_stage(DetectorStage(DummyDetector(), name="detect"))
    p.add_stage(TrackerStage(
        DummyTracker(raise_on_update=ValueError("x")), name="track"))
    runner = ShadowPipelineRunner(p, DummyFrameSource())
    runner.initialize()
    runner.run_batch(max_frames=3)
    s = runner.stats.summary()
    assert "exceptions" in s
    assert "ValueError" in s


# ======================================================================
# 7. No output side effects (shadow guarantee)
# ======================================================================

def test_shadow_does_not_modify_shared_state():
    """The shadow runner's context is local -- it doesn't leak into any
    shared structure. Here we verify the DummyEventBus (the terminal sink)
    records events but nothing else is touched."""
    bus = DummyEventBus()
    p = Pipeline()
    p.add_stage(DetectorStage(DummyDetector(), name="detect"))
    p.add_stage(TrackerStage(DummyTracker(), name="track"))
    p.add_stage(TargetStage(DummyTargetManager(), name="targets"))
    p.add_stage(EventStage(bus, name="events"))
    runner = ShadowPipelineRunner(p, DummyFrameSource())
    runner.initialize()
    runner.run_batch(max_frames=5)
    # The bus recorded 5 events (one per frame). Nothing else was touched.
    assert len(bus.published) == 5
    # The runner's stats are the only other output.
    assert runner.stats.frame_count == 5


def test_shadow_context_is_discarded():
    """The context returned by run_once is not retained by the runner."""
    runner = ShadowPipelineRunner(_make_pipeline(), DummyFrameSource())
    runner.initialize()
    ctx = runner.run_once()
    # The runner doesn't keep a reference to past contexts.
    assert not hasattr(runner, "_last_ctx") or getattr(runner, "_last_ctx", None) is None


# ======================================================================
# 8. End-to-end with multi-frame source
# ======================================================================

def test_end_to_end_multi_frame_pipeline():
    """Run 100 frames through the full 4-stage shadow pipeline."""
    import numpy as np
    frames = [Frame(i, float(i) / 30.0, "cam0", np.zeros((8, 8, 3), dtype=np.uint8))
              for i in range(100)]
    src = DummyFrameSource(frames=frames, loop=False)
    runner = ShadowPipelineRunner(_make_pipeline(), src)
    runner.initialize()
    stats = runner.run_batch(max_frames=200)
    assert stats.frame_count == 100
    assert stats.skip_count == 1  # exhaustion
    assert stats.error_frame_count == 0
    assert stats.fps > 0.0
    # Each stage called 100 times.
    for ss in stats.stage_stats.values():
        assert ss.call_count == 100
    runner.shutdown()


# ======================================================================
# Runner
# ======================================================================

def _collect_tests():
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} shadow pipeline tests...\n")
    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            tb = traceback.format_exc()
            failures.append((name, tb))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 60}")
    print(f"Result: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        print(f"\n{'=' * 60}\nFailure details:\n")
        for name, tb in failures:
            print(f"---- {name} ----")
            print(tb)
        raise SystemExit(1)
    print("All shadow pipeline tests passed.")
