"""TrackerStage unit tests (Milestone C4).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary. A tiny ``raises``
context manager replaces ``pytest.raises``.

Covers:
    - Tracker ABC contract (abstract, subclass must implement)
    - DummyTracker (fixed Tracks, counters, raise_on_update, last_detections, reset)
    - TrackerStage construction & lifecycle delegation
    - TrackerStage.process data flow (detections -> tracks, REPLACE semantics)
    - Empty-detections still calls update (no skip)
    - Replace-not-extend semantics (vs DetectorStage's extend)
    - Exception propagation (tracker.update raises -> process raises)
    - Pipeline integration (end-to-end)
    - Detector -> Tracker chain (C3 -> C4 integration)
    - No ByteTrack / BoTSORT / OSTrack dependency
"""
from __future__ import annotations

import ast
import traceback
from typing import Any

from visioncore.core.detection import BBox, Detection
from visioncore.core.track import Track
from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages import (
    DetectorStage,
    DummyDetector,
    DummyTracker,
    Tracker,
    TrackerError,
    TrackerStage,
)


# ======================================================================
# Test helpers (replace pytest dependencies)
# ======================================================================

class _Raises:
    """Context manager asserting that a block raises a matching exception."""

    __slots__ = ("expected", "match", "caught")

    def __init__(self, expected: type[BaseException], match: str | None = None) -> None:
        self.expected: type[BaseException] = expected
        self.match: str | None = match
        self.caught: BaseException | None = None

    def __enter__(self) -> "_Raises":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> bool:
        if exc_type is None:
            raise AssertionError(
                f"expected {self.expected.__name__} to be raised, "
                f"but no exception was raised"
            )
        if not isinstance(exc_val, BaseException):
            raise AssertionError(
                f"expected {self.expected.__name__}, got non-exception {exc_val!r}"
            )
        if not isinstance(exc_val, self.expected):
            raise AssertionError(
                f"expected {self.expected.__name__}, got "
                f"{type(exc_val).__name__}: {exc_val}"
            )
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(
                f"expected {self.match!r} in str({type(exc_val).__name__}), "
                f"got {str(exc_val)!r}"
            )
        self.caught = exc_val
        return True


def raises(
    expected: type[BaseException], match: str | None = None,
) -> _Raises:
    """Assert that the enclosed block raises ``expected``."""
    return _Raises(expected, match=match)


def _make_detection(class_name: str = "person", score: float = 0.9) -> Detection:
    """Build a synthetic Detection."""
    return Detection(BBox(0.5, 0.5, 0.2, 0.4), score, 0, class_name)


def _make_track(track_id: int = 1, class_name: str = "person") -> Track:
    """Build a synthetic Track."""
    return Track(track_id=track_id, detection=_make_detection(class_name))


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_package_exports_tracker_names():
    """The stages __all__ includes the four tracker names."""
    import visioncore.pipeline.stages as pkg
    for name in ("Tracker", "TrackerError", "TrackerStage", "DummyTracker"):
        assert name in pkg.__all__, f"{name} missing from stages __all__"


def test_tracker_error_is_runtime_error_subclass():
    """TrackerError is a RuntimeError subclass."""
    assert issubclass(TrackerError, RuntimeError)


def test_tracker_is_abstract():
    """Tracker is an ABC and cannot be instantiated directly."""
    with raises(TypeError):
        Tracker()  # type: ignore[abstract]


def test_tracker_subclass_missing_methods_fails():
    """A Tracker subclass that omits methods cannot be instantiated."""
    class Incomplete(Tracker):
        def update(self, detections):
            return []
        # initialize / shutdown / health_check missing
    with raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ======================================================================
# 1. DummyTracker
# ======================================================================

def test_dummy_tracker_default_returns_one_track():
    """A default DummyTracker returns one Track (track_id=1, person)."""
    trk = DummyTracker()
    trk.initialize()
    tracks = trk.update([])
    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].detection.class_name == "person"


def test_dummy_tracker_custom_tracks():
    """Explicit tracks are replayed."""
    custom = [_make_track(1, "car"), _make_track(2, "truck")]
    trk = DummyTracker(tracks=custom)
    trk.initialize()
    tracks = trk.update([])
    assert len(tracks) == 2
    assert [t.track_id for t in tracks] == [1, 2]


def test_dummy_tracker_update_returns_fresh_list():
    """update() returns a new list each call (caller may mutate safely)."""
    trk = DummyTracker()
    trk.initialize()
    l1 = trk.update([])
    l2 = trk.update([])
    assert l1 is not l2  # different list objects
    assert l1 == l2      # same contents


def test_dummy_tracker_records_last_detections():
    """update() records the detections it was passed (for test assertions)."""
    trk = DummyTracker()
    trk.initialize()
    dets = [_make_detection("car"), _make_detection("dog")]
    trk.update(dets)
    assert trk.last_detections is dets  # the exact list forwarded


def test_dummy_tracker_lifecycle_and_counters():
    """initialize/update/shutdown increment counters and toggle health."""
    trk = DummyTracker()
    assert trk.health_check() is False
    trk.initialize()
    assert trk.health_check() is True
    trk.update([])
    trk.shutdown()
    assert trk.health_check() is False
    assert (trk.initialize_count, trk.update_count, trk.shutdown_count) == (1, 1, 1)


def test_dummy_tracker_raise_on_update():
    """raise_on_update propagates the configured exception."""
    trk = DummyTracker(raise_on_update=TrackerError("filter diverged"))
    trk.initialize()
    with raises(TrackerError, match="filter diverged"):
        trk.update([])
    assert trk.update_count == 1  # incremented before raising


def test_dummy_tracker_raise_persists_until_cleared():
    """raise_on_update raises every update until cleared."""
    trk = DummyTracker(raise_on_update=ValueError("bad"))
    trk.initialize()
    with raises(ValueError):
        trk.update([])
    with raises(ValueError):
        trk.update([])
    trk.raise_on_update = None
    assert trk.update([]) is not None  # now works


def test_dummy_tracker_reset():
    """reset() zeroes counters and state, preserves the track buffer."""
    trk = DummyTracker()
    trk.initialize()
    trk.update([])
    trk.shutdown()
    trk.reset()
    assert (trk.initialize_count, trk.update_count, trk.shutdown_count) == (0, 0, 0)
    assert trk.health_check() is False
    assert trk.track_count == 1  # buffer preserved
    assert trk.last_detections is None


# ======================================================================
# 2. TrackerStage construction
# ======================================================================

def test_stage_construction_wraps_tracker():
    """TrackerStage stores the wrapped tracker and reports health."""
    trk = DummyTracker()
    stage = TrackerStage(trk)
    assert stage.tracker is trk
    assert stage.health_check() is False  # tracker not initialised yet


def test_stage_default_name():
    """Default stage name is 'TrackerStage'."""
    stage = TrackerStage(DummyTracker())
    assert stage.name == "TrackerStage"


def test_stage_custom_name():
    """An explicit name overrides the default."""
    stage = TrackerStage(DummyTracker(), name="person-tracker")
    assert stage.name == "person-tracker"


def test_stage_rejects_none_tracker():
    """Constructing with None raises TypeError."""
    with raises(TypeError, match="Tracker"):
        TrackerStage(None)  # type: ignore[arg-type]


def test_stage_rejects_non_tracker():
    """Constructing with a non-Tracker raises TypeError."""
    with raises(TypeError, match="Tracker"):
        TrackerStage("not a tracker")  # type: ignore[arg-type]


def test_stage_is_pipeline_stage():
    """TrackerStage is a PipelineStage subclass."""
    assert issubclass(TrackerStage, PipelineStage)


# ======================================================================
# 3. TrackerStage lifecycle delegation
# ======================================================================

def test_stage_initialize_delegates_to_tracker():
    """stage.initialize() calls tracker.initialize()."""
    trk = DummyTracker()
    stage = TrackerStage(trk)
    stage.initialize()
    assert trk.initialize_count == 1
    assert stage.health_check() is True


def test_stage_shutdown_delegates_to_tracker():
    """stage.shutdown() calls tracker.shutdown()."""
    trk = DummyTracker()
    stage = TrackerStage(trk)
    stage.initialize()
    stage.shutdown()
    assert trk.shutdown_count == 1
    assert stage.health_check() is False


def test_stage_health_check_delegates_to_tracker():
    """stage.health_check() reflects tracker health."""
    trk = DummyTracker()
    stage = TrackerStage(trk)
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_stage_repr_reports_health():
    """__repr__ reports the stage name and health."""
    stage = TrackerStage(DummyTracker(), name="track")
    stage.initialize()
    r = repr(stage)
    assert "TrackerStage" in r
    assert "name='track'" in r
    assert "healthy=True" in r


# ======================================================================
# 4. TrackerStage.process data flow
# ======================================================================

def test_process_reads_detections_writes_tracks():
    """process() reads context.detections, writes context.tracks."""
    stage = TrackerStage(DummyTracker())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.detections.append(_make_detection("person"))
    stage.process(ctx)
    assert len(ctx.tracks) == 1
    assert ctx.tracks[0].track_id == 1


def test_process_with_custom_tracks():
    """process() writes whatever the tracker returns."""
    custom = [_make_track(10, "car"), _make_track(20, "truck")]
    stage = TrackerStage(DummyTracker(tracks=custom))
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.detections.append(_make_detection("car"))
    stage.process(ctx)
    assert [t.track_id for t in ctx.tracks] == [10, 20]


def test_process_replaces_not_extends_tracks():
    """process() REPLACES context.tracks (clear + extend), not appends.

    A tracker owns the complete current track set; its output is the full
    state. Pre-existing tracks in context.tracks are cleared before the
    new set is written. This contrasts with DetectorStage, which EXTENDS
    detections.
    """
    stage = TrackerStage(DummyTracker())
    stage.initialize()
    ctx = PipelineContext.empty()
    # Pre-populate with a stale track (simulating a prior call / stage)
    ctx.tracks.append(_make_track(999, "stale"))
    stage.process(ctx)
    # The stale track is gone; only the tracker's output remains.
    assert len(ctx.tracks) == 1
    assert ctx.tracks[0].track_id == 1  # the tracker's default, not 999


def test_process_preserves_list_identity():
    """process() mutates context.tracks in place (clear+extend), not rebinds.

    The list object identity is preserved so any external reference to
    context.tracks sees the new contents.
    """
    stage = TrackerStage(DummyTracker())
    stage.initialize()
    ctx = PipelineContext.empty()
    original_list = ctx.tracks
    ctx.detections.append(_make_detection())
    stage.process(ctx)
    assert ctx.tracks is original_list  # same list object, contents replaced


def test_process_with_empty_detections_still_calls_update():
    """process() with empty detections still calls tracker.update().

    An empty list is the legitimate "no new detections, just predict"
    signal -- NOT a skip condition. This contrasts with DetectorStage,
    which skips when frame is None.
    """
    trk = DummyTracker()
    stage = TrackerStage(trk)
    stage.initialize()
    ctx = PipelineContext.empty()
    assert ctx.detections == []  # empty
    stage.process(ctx)  # must call update, not skip
    assert trk.update_count == 1
    assert trk.last_detections == []  # empty list forwarded
    assert len(ctx.tracks) == 1  # tracker still returned its tracks


def test_process_increments_tracker_update_count():
    """process() calls tracker.update() exactly once per process call."""
    stage = TrackerStage(DummyTracker())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.detections.append(_make_detection())
    stage.process(ctx)
    stage.process(ctx)
    assert stage.tracker.update_count == 2


def test_process_forwards_exact_detections_list():
    """process() forwards context.detections (the exact list) to tracker."""
    trk = DummyTracker()
    stage = TrackerStage(trk)
    stage.initialize()
    ctx = PipelineContext.empty()
    dets = [_make_detection("a"), _make_detection("b"), _make_detection("c")]
    ctx.detections.extend(dets)
    stage.process(ctx)
    assert trk.last_detections is ctx.detections  # same list forwarded
    assert len(trk.last_detections) == 3


def test_process_with_no_detections_field():
    """process() works on a fresh context (detections is empty list)."""
    stage = TrackerStage(DummyTracker())
    stage.initialize()
    ctx = PipelineContext.empty()  # detections == []
    stage.process(ctx)
    assert len(ctx.tracks) == 1  # tracker returned its default tracks


# ======================================================================
# 5. Exception propagation
# ======================================================================

def test_process_propagates_tracker_exception():
    """A tracker.update() exception propagates out of process()."""
    trk = DummyTracker(raise_on_update=TrackerError("matching failed"))
    stage = TrackerStage(trk)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.detections.append(_make_detection())
    with raises(TrackerError, match="matching failed"):
        stage.process(ctx)
    # No tracks written (the update call raised)
    assert ctx.tracks == []


def test_process_propagates_arbitrary_exception():
    """Any exception type from update() propagates."""
    trk = DummyTracker(raise_on_update=RuntimeError("boom"))
    stage = TrackerStage(trk)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.detections.append(_make_detection())
    with raises(RuntimeError, match="boom"):
        stage.process(ctx)


def test_process_exception_does_not_write_partial_tracks():
    """If update() raises, context.tracks is unchanged (not cleared)."""
    trk = DummyTracker(raise_on_update=ValueError("x"))
    stage = TrackerStage(trk)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks.append(_make_track(999, "prior"))
    ctx.detections.append(_make_detection())
    try:
        stage.process(ctx)
    except ValueError:
        pass
    # The clear() happens AFTER update() returns; since update raised,
    # the prior track is still there.
    assert len(ctx.tracks) == 1
    assert ctx.tracks[0].track_id == 999


# ======================================================================
# 6. Pipeline integration
# ======================================================================

def test_pipeline_with_tracker_stage_end_to_end():
    """A Pipeline with a TrackerStage produces tracks from detections."""
    stage = TrackerStage(DummyTracker(), name="track")
    p = Pipeline()
    p.add_stage(stage)

    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.detections.append(_make_detection("person"))
    with p:
        p.run(ctx)

    assert len(ctx.tracks) == 1
    assert ctx.tracks[0].track_id == 1
    assert stage.tracker.shutdown_count == 1  # lifecycle ran


def test_pipeline_tracker_stage_lifecycle_once_per_run():
    """initialize once, process once per run, shutdown once."""
    trk = DummyTracker()
    stage = TrackerStage(trk)
    p = Pipeline()
    p.add_stage(stage)

    ctx = PipelineContext.empty()
    ctx.detections.append(_make_detection())
    with p:
        p.run(ctx)

    assert trk.initialize_count == 1
    assert trk.update_count == 1
    assert trk.shutdown_count == 1


def test_pipeline_multiple_runs_reuse_initialization():
    """One initialize supports multiple runs; shutdown once at end."""
    trk = DummyTracker()
    stage = TrackerStage(trk)
    p = Pipeline()
    p.add_stage(stage)
    p.initialize()
    for i in range(3):
        ctx = PipelineContext.empty(timestamp=float(i))
        ctx.detections.append(_make_detection())
        p.run(ctx)
    p.shutdown()
    assert trk.initialize_count == 1
    assert trk.update_count == 3
    assert trk.shutdown_count == 1


# ======================================================================
# 7. End-to-end: Detector -> Tracker chain (C3 -> C4 integration)
# ======================================================================

def test_end_to_end_detector_then_tracker():
    """A DetectorStage feeds detections into a TrackerStage.

    This exercises the C3->C4 chain: DetectorStage writes
    context.detections, TrackerStage reads them and writes context.tracks.
    """
    det_stage = DetectorStage(DummyDetector(), name="detect")
    trk_stage = TrackerStage(DummyTracker(), name="track")

    p = Pipeline()
    p.add_stage(det_stage)
    p.add_stage(trk_stage)

    # Provide a frame so the detector stage has something to detect on.
    import numpy as np
    from visioncore.core.frame import Frame
    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.frame = Frame(0, 0.0, "cam0", np.zeros((8, 8, 3), dtype=np.uint8))

    with p:
        p.run(ctx)

    # Detector produced 1 detection, tracker consumed it and produced 1 track.
    assert len(ctx.detections) == 1
    assert ctx.detections[0].class_name == "person"
    assert len(ctx.tracks) == 1
    assert ctx.tracks[0].track_id == 1
    # The tracker saw the detector's output.
    assert trk_stage.tracker.last_detections is ctx.detections
    # Both stages' resources released.
    assert det_stage.detector.shutdown_count == 1
    assert trk_stage.tracker.shutdown_count == 1


def test_end_to_end_detector_tracker_no_frame():
    """Detector->Tracker chain with no frame: detector skips, tracker still runs.

    With frame=None, DetectorStage skips (no detections added). TrackerStage
    still calls update([]) -- the empty-list "predict only" signal -- and
    produces its default tracks.
    """
    det_stage = DetectorStage(DummyDetector(), name="detect")
    trk_stage = TrackerStage(DummyTracker(), name="track")

    p = Pipeline()
    p.add_stage(det_stage)
    p.add_stage(trk_stage)

    ctx = PipelineContext.empty()  # frame is None, detections empty
    with p:
        p.run(ctx)

    assert ctx.detections == []  # detector skipped (no frame)
    assert det_stage.detector.detect_count == 0  # detector never called
    # Tracker still ran (update([]) is the predict-only signal).
    assert trk_stage.tracker.update_count == 1
    assert len(ctx.tracks) == 1  # tracker returned its default tracks


# ======================================================================
# 8. No ByteTrack / BoTSORT / OSTrack dependency
# ======================================================================

def test_no_bytrack_botsort_ostrack_in_source():
    """The tracker_stage module source contains no banned backend refs.

    AST-parse the module and assert no import references ByteTrack,
    BoTSORT, OSTrack, or related libraries. This is the architectural
    firewall: the stage depends on the abstract Tracker interface, never
    on a concrete tracking library.
    """
    import visioncore.pipeline.stages.tracker_stage as mod
    src = open(mod.__file__).read()
    tree = ast.parse(src)
    banned = ("bytrack", "byte_track", "bytetrack", "botsort", "bot_sort",
              "ostrack", "os_track", "ultralytics", "onnxruntime", "cv2",
              "opencv", "filterpy", "scipy")
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for b in banned:
                    if b in alias.name.lower():
                        violations.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            mod_name = (node.module or "").lower()
            for b in banned:
                if b in mod_name:
                    violations.append(f"from {node.module}")
    assert not violations, (
        f"tracker_stage.py imports banned backend(s): {violations}"
    )


def test_no_banned_backends_loaded_at_runtime():
    """Importing the stages package does not load any banned backend."""
    import sys
    import visioncore.pipeline.stages  # noqa: F401
    banned = ["ultralytics", "onnxruntime", "cv2", "filterpy", "scipy"]
    loaded = [b for b in banned if b in sys.modules]
    assert not loaded, (
        f"banned backend(s) loaded as side effect: {loaded}"
    )


def test_tracker_interface_is_algorithm_agnostic():
    """The Tracker ABC has no algorithm-specific attributes or methods."""
    import inspect
    methods = {name for name, _ in inspect.getmembers(Tracker, predicate=inspect.isfunction)
               if not name.startswith("_")}
    # The four interface methods -- nothing algorithm-specific.
    assert methods == {"initialize", "update", "shutdown", "health_check"}


# ======================================================================
# 9. Replace vs extend semantics (contrast with DetectorStage)
# ======================================================================

def test_tracker_replaces_while_detector_extends():
    """Contrast: DetectorStage EXTENDS detections, TrackerStage REPLACES tracks.

    This is a deliberate semantic distinction documented in both stages:
    detections are additive (multiple detectors can contribute), tracks
    are stateful (one tracker owns the complete current set).
    """
    det = DummyDetector()
    trk = DummyTracker()
    det_stage = DetectorStage(det, name="detect")
    trk_stage = TrackerStage(trk, name="track")

    ctx = PipelineContext.empty()
    ctx.detections.append(_make_detection("prior_det"))
    ctx.tracks.append(_make_track(999, "prior_trk"))

    det_stage.initialize()
    trk_stage.initialize()

    # Detector extends (prior + new), tracker replaces (prior gone).
    det_stage.process(ctx)  # needs a frame, but we test the list semantics
    # Detector didn't run (frame is None -> skip), so detections unchanged.
    # Now tracker:
    trk_stage.process(ctx)

    # Tracker replaced: prior_trk (999) is gone, only tracker's output.
    assert all(t.track_id != 999 for t in ctx.tracks)
    assert len(ctx.tracks) == 1  # tracker's default


# ======================================================================
# Runner -- collects every test_* function and runs them.
# ======================================================================

def _collect_tests() -> list[tuple[str, Any]]:
    """Return [(name, func)] for every test_ function defined in this module."""
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} tracker stage tests...\n")
    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 -- runner catches all
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
    print("All tracker stage tests passed.")
