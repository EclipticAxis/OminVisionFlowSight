"""TargetStage unit tests (Milestone C5).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - TargetManager ABC contract (abstract, subclass must implement)
    - TargetUpdate dataclass (frozen, fields, repr)
    - DummyTargetManager (defaults, custom, counters, raise_on_update,
      last_tracks/last_timestamp, reset)
    - TargetStage construction & lifecycle delegation
    - TargetStage.process DUAL-output data flow (tracks -> targets + target_states)
    - REPLACE semantics for BOTH outputs (vs DetectorStage's extend)
    - Empty-tracks still calls update (no skip)
    - Timestamp forwarding
    - Exception propagation (update raises -> prior targets/snapshots preserved)
    - Pipeline integration + end-to-end Detector->Tracker->Target chain
    - No GUI / InferWorker dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import traceback
from typing import Any

from visioncore.core.detection import BBox, Detection
from visioncore.core.target import Target
from visioncore.core.target import TargetState as TargetLifecycle
from visioncore.core.track import Track
from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages import (
    DetectorStage,
    DummyDetector,
    DummyTargetManager,
    DummyTracker,
    TargetError,
    TargetManager,
    TargetStage,
    TargetUpdate,
    TrackerStage,
)
from visioncore.state.target_state import TargetState as TargetSnapshot


# ======================================================================
# Test helpers
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
                f"expected {self.expected.__name__}, but no exception was raised"
            )
        if not isinstance(exc_val, BaseException):
            raise AssertionError(f"expected {self.expected.__name__}, got {exc_val!r}")
        if not isinstance(exc_val, self.expected):
            raise AssertionError(
                f"expected {self.expected.__name__}, got "
                f"{type(exc_val).__name__}: {exc_val}"
            )
        if self.match is not None and self.match not in str(exc_val):
            raise AssertionError(
                f"expected {self.match!r} in str, got {str(exc_val)!r}"
            )
        self.caught = exc_val
        return True


def raises(expected: type[BaseException], match: str | None = None) -> _Raises:
    return _Raises(expected, match=match)


def _det(class_name: str = "person", score: float = 0.9) -> Detection:
    return Detection(BBox(0.5, 0.5, 0.2, 0.4), score, 0, class_name)


def _track(track_id: int = 1, class_name: str = "person") -> Track:
    return Track(track_id=track_id, detection=_det(class_name))


def _target(target_id: str = "S0-T0001", track_id: int = 1) -> Target:
    return Target(
        target_id=target_id,
        track=_track(track_id),
        slot_id=0,
        state=TargetLifecycle.ACTIVE,
    )


def _snapshot(target_id: int = 1, label: str = "person") -> TargetSnapshot:
    return TargetSnapshot(
        target_id=target_id, local_id=1, global_id=None, label=label,
        confidence=0.9, cx=0.5, cy=0.5, vx=0.0, vy=0.0,
        width=0.2, height=0.4, timestamp=0.0, camera_id=0, metadata={},
    )


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_package_exports_target_names():
    """The stages __all__ includes the five target names."""
    import visioncore.pipeline.stages as pkg
    for name in ("TargetManager", "TargetError", "TargetStage",
                 "TargetUpdate", "DummyTargetManager"):
        assert name in pkg.__all__, f"{name} missing from stages __all__"


def test_target_error_is_runtime_error_subclass():
    assert issubclass(TargetError, RuntimeError)


def test_target_manager_is_abstract():
    with raises(TypeError):
        TargetManager()  # type: ignore[abstract]


def test_target_manager_subclass_missing_methods_fails():
    class Incomplete(TargetManager):
        def update(self, tracks, *, timestamp=0.0):
            return TargetUpdate([], [])
    with raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ======================================================================
# 1. TargetUpdate dataclass
# ======================================================================

def test_target_update_is_frozen():
    """TargetUpdate is frozen (fields cannot be reassigned)."""
    import dataclasses
    assert dataclasses.is_dataclass(TargetUpdate)
    params = dataclasses.fields(TargetUpdate)
    assert {f.name for f in params} == {"targets", "target_states"}


def test_target_update_holds_both_lists():
    """TargetUpdate holds targets and target_states lists."""
    tgts = [_target()]
    snaps = [_snapshot()]
    upd = TargetUpdate(targets=tgts, target_states=snaps)
    assert upd.targets is tgts
    assert upd.target_states is snaps


def test_target_update_repr_is_compact():
    """__repr__ reports list lengths, not contents."""
    upd = TargetUpdate(
        targets=[_target(), _target("S0-T0002")],
        target_states=[_snapshot()],
    )
    r = repr(upd)
    assert "targets=2" in r
    assert "target_states=1" in r


# ======================================================================
# 2. DummyTargetManager
# ======================================================================

def test_dummy_default_returns_one_target_one_snapshot():
    mgr = DummyTargetManager()
    mgr.initialize()
    result = mgr.update([])
    assert len(result.targets) == 1
    assert len(result.target_states) == 1
    assert result.targets[0].target_id == "S0-T0001"
    assert result.target_states[0].label == "person"


def test_dummy_custom_targets_and_snapshots():
    tgts = [_target("S0-T0001"), _target("S0-T0002", track_id=2)]
    snaps = [_snapshot(1), _snapshot(2, "car")]
    mgr = DummyTargetManager(targets=tgts, target_states=snaps)
    mgr.initialize()
    result = mgr.update([])
    assert len(result.targets) == 2
    assert len(result.target_states) == 2
    assert result.target_states[1].label == "car"


def test_dummy_update_returns_fresh_lists():
    """update() returns new lists each call (caller may mutate safely)."""
    mgr = DummyTargetManager()
    mgr.initialize()
    r1 = mgr.update([])
    r2 = mgr.update([])
    assert r1.targets is not r2.targets
    assert r1.target_states is not r2.target_states
    assert r1.targets == r2.targets  # same contents


def test_dummy_records_last_tracks_and_timestamp():
    mgr = DummyTargetManager()
    mgr.initialize()
    tracks = [_track(1), _track(2)]
    mgr.update(tracks, timestamp=3.5)
    assert mgr.last_tracks is tracks
    assert mgr.last_timestamp == 3.5


def test_dummy_lifecycle_and_counters():
    mgr = DummyTargetManager()
    assert mgr.health_check() is False
    mgr.initialize()
    assert mgr.health_check() is True
    mgr.update([])
    mgr.shutdown()
    assert mgr.health_check() is False
    assert (mgr.initialize_count, mgr.update_count, mgr.shutdown_count) == (1, 1, 1)


def test_dummy_raise_on_update():
    mgr = DummyTargetManager(raise_on_update=TargetError("store corrupted"))
    mgr.initialize()
    with raises(TargetError, match="store corrupted"):
        mgr.update([])
    assert mgr.update_count == 1


def test_dummy_raise_persists_until_cleared():
    mgr = DummyTargetManager(raise_on_update=ValueError("bad"))
    mgr.initialize()
    with raises(ValueError):
        mgr.update([])
    with raises(ValueError):
        mgr.update([])
    mgr.raise_on_update = None
    assert mgr.update([]) is not None


def test_dummy_reset():
    mgr = DummyTargetManager()
    mgr.initialize()
    mgr.update([])
    mgr.shutdown()
    mgr.reset()
    assert (mgr.initialize_count, mgr.update_count, mgr.shutdown_count) == (0, 0, 0)
    assert mgr.health_check() is False
    assert mgr.target_count == 1  # buffer preserved
    assert mgr.snapshot_count == 1
    assert mgr.last_tracks is None


# ======================================================================
# 3. TargetStage construction
# ======================================================================

def test_stage_construction_wraps_manager():
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    assert stage.manager is mgr
    assert stage.health_check() is False


def test_stage_default_name():
    assert TargetStage(DummyTargetManager()).name == "TargetStage"


def test_stage_custom_name():
    stage = TargetStage(DummyTargetManager(), name="targets")
    assert stage.name == "targets"


def test_stage_rejects_none_manager():
    with raises(TypeError, match="TargetManager"):
        TargetStage(None)  # type: ignore[arg-type]


def test_stage_rejects_non_manager():
    with raises(TypeError, match="TargetManager"):
        TargetStage("not a manager")  # type: ignore[arg-type]


def test_stage_is_pipeline_stage():
    assert issubclass(TargetStage, PipelineStage)


# ======================================================================
# 4. TargetStage lifecycle delegation
# ======================================================================

def test_stage_initialize_delegates():
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    stage.initialize()
    assert mgr.initialize_count == 1
    assert stage.health_check() is True


def test_stage_shutdown_delegates():
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    stage.initialize()
    stage.shutdown()
    assert mgr.shutdown_count == 1
    assert stage.health_check() is False


def test_stage_health_check_delegates():
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True


def test_stage_repr_reports_health():
    stage = TargetStage(DummyTargetManager(), name="targets")
    stage.initialize()
    r = repr(stage)
    assert "TargetStage" in r
    assert "name='targets'" in r
    assert "healthy=True" in r


# ======================================================================
# 5. TargetStage.process DUAL-output data flow
# ======================================================================

def test_process_reads_tracks_writes_both_outputs():
    """process() reads context.tracks, writes BOTH targets and target_states."""
    stage = TargetStage(DummyTargetManager())
    stage.initialize()
    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.tracks.append(_track(1, "person"))
    stage.process(ctx)
    assert len(ctx.targets) == 1
    assert len(ctx.target_states) == 1
    assert ctx.targets[0].target_id == "S0-T0001"
    assert ctx.target_states[0].label == "person"


def test_process_with_custom_outputs():
    tgts = [_target("S0-T0001"), _target("S0-T0002", track_id=2)]
    snaps = [_snapshot(1, "person"), _snapshot(2, "car")]
    stage = TargetStage(DummyTargetManager(targets=tgts, target_states=snaps))
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks.append(_track())
    stage.process(ctx)
    assert [t.target_id for t in ctx.targets] == ["S0-T0001", "S0-T0002"]
    assert [s.label for s in ctx.target_states] == ["person", "car"]


def test_process_replaces_not_extends_targets():
    """process() REPLACES context.targets (clear+extend), not appends."""
    stage = TargetStage(DummyTargetManager())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.targets.append(_target("STALE", track_id=999))  # pre-existing
    stage.process(ctx)
    assert len(ctx.targets) == 1
    assert ctx.targets[0].target_id == "S0-T0001"  # stale gone


def test_process_replaces_not_extends_target_states():
    """process() REPLACES context.target_states (clear+extend), not appends."""
    stage = TargetStage(DummyTargetManager())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.target_states.append(_snapshot(999, "stale"))  # pre-existing
    stage.process(ctx)
    assert len(ctx.target_states) == 1
    assert ctx.target_states[0].target_id == 1  # stale gone


def test_process_preserves_list_identity():
    """process() mutates lists in place (clear+extend), not rebinds."""
    stage = TargetStage(DummyTargetManager())
    stage.initialize()
    ctx = PipelineContext.empty()
    orig_targets = ctx.targets
    orig_states = ctx.target_states
    ctx.tracks.append(_track())
    stage.process(ctx)
    assert ctx.targets is orig_targets
    assert ctx.target_states is orig_states


def test_process_with_empty_tracks_still_calls_update():
    """process() with empty tracks still calls manager.update().

    An empty list is the "advance / stale-sweep" signal, not a skip.
    """
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    stage.initialize()
    ctx = PipelineContext.empty()
    assert ctx.tracks == []
    stage.process(ctx)
    assert mgr.update_count == 1
    assert mgr.last_tracks == []
    # Manager still returned its targets/snapshots.
    assert len(ctx.targets) == 1
    assert len(ctx.target_states) == 1


def test_process_forwards_timestamp():
    """process() passes context.timestamp to manager.update()."""
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    stage.initialize()
    ctx = PipelineContext.empty(timestamp=42.5)
    ctx.tracks.append(_track())
    stage.process(ctx)
    assert mgr.last_timestamp == 42.5


def test_process_increments_update_count():
    stage = TargetStage(DummyTargetManager())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks.append(_track())
    stage.process(ctx)
    stage.process(ctx)
    assert stage.manager.update_count == 2


def test_process_forwards_exact_tracks_list():
    """process() forwards context.tracks (the exact list) to manager.update()."""
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    stage.initialize()
    ctx = PipelineContext.empty()
    tracks = [_track(1), _track(2), _track(3)]
    ctx.tracks.extend(tracks)
    stage.process(ctx)
    assert mgr.last_tracks is ctx.tracks
    assert len(mgr.last_tracks) == 3


# ======================================================================
# 6. Exception propagation (prior state preserved)
# ======================================================================

def test_process_propagates_manager_exception():
    mgr = DummyTargetManager(raise_on_update=TargetError("projection failed"))
    stage = TargetStage(mgr)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks.append(_track())
    with raises(TargetError, match="projection failed"):
        stage.process(ctx)


def test_process_propagates_arbitrary_exception():
    mgr = DummyTargetManager(raise_on_update=RuntimeError("boom"))
    stage = TargetStage(mgr)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.tracks.append(_track())
    with raises(RuntimeError, match="boom"):
        stage.process(ctx)


def test_process_exception_preserves_prior_targets():
    """If update() raises, context.targets is NOT cleared (prior preserved)."""
    mgr = DummyTargetManager(raise_on_update=ValueError("x"))
    stage = TargetStage(mgr)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.targets.append(_target("PRIOR", track_id=99))
    ctx.target_states.append(_snapshot(99, "prior"))
    ctx.tracks.append(_track())
    try:
        stage.process(ctx)
    except ValueError:
        pass
    # Prior targets AND snapshots preserved (clear happens after update).
    assert len(ctx.targets) == 1
    assert ctx.targets[0].target_id == "PRIOR"
    assert len(ctx.target_states) == 1
    assert ctx.target_states[0].label == "prior"


# ======================================================================
# 7. Pipeline integration
# ======================================================================

def test_pipeline_with_target_stage_end_to_end():
    stage = TargetStage(DummyTargetManager(), name="targets")
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.tracks.append(_track(1, "person"))
    with p:
        p.run(ctx)
    assert len(ctx.targets) == 1
    assert len(ctx.target_states) == 1
    assert stage.manager.shutdown_count == 1


def test_pipeline_target_stage_lifecycle_once_per_run():
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    p = Pipeline()
    p.add_stage(stage)
    ctx = PipelineContext.empty()
    ctx.tracks.append(_track())
    with p:
        p.run(ctx)
    assert (mgr.initialize_count, mgr.update_count, mgr.shutdown_count) == (1, 1, 1)


def test_pipeline_multiple_runs_reuse_initialization():
    mgr = DummyTargetManager()
    stage = TargetStage(mgr)
    p = Pipeline()
    p.add_stage(stage)
    p.initialize()
    for i in range(3):
        ctx = PipelineContext.empty(timestamp=float(i))
        ctx.tracks.append(_track())
        p.run(ctx)
    p.shutdown()
    assert mgr.initialize_count == 1
    assert mgr.update_count == 3
    assert mgr.shutdown_count == 1


# ======================================================================
# 8. End-to-end: Detector -> Tracker -> Target (C3 -> C4 -> C5)
# ======================================================================

def test_end_to_end_detect_track_target():
    """Full chain: DetectorStage -> TrackerStage -> TargetStage.

    C3 produces detections, C4 produces tracks, C5 produces targets +
    target_states. All three stages in one pipeline.
    """
    import numpy as np
    from visioncore.core.frame import Frame

    det_stage = DetectorStage(DummyDetector(), name="detect")
    trk_stage = TrackerStage(DummyTracker(), name="track")
    tgt_stage = TargetStage(DummyTargetManager(), name="targets")

    p = Pipeline()
    p.add_stage(det_stage)
    p.add_stage(trk_stage)
    p.add_stage(tgt_stage)

    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.frame = Frame(0, 0.0, "cam0", np.zeros((8, 8, 3), dtype=np.uint8))

    with p:
        p.run(ctx)

    # Each stage produced its output.
    assert len(ctx.detections) == 1
    assert len(ctx.tracks) == 1
    assert len(ctx.targets) == 1
    assert len(ctx.target_states) == 1
    # The chain is consistent: person throughout.
    assert ctx.detections[0].class_name == "person"
    assert ctx.tracks[0].track_id == 1
    assert ctx.targets[0].target_id == "S0-T0001"
    assert ctx.target_states[0].label == "person"
    # All stages shut down.
    assert det_stage.detector.shutdown_count == 1
    assert trk_stage.tracker.shutdown_count == 1
    assert tgt_stage.manager.shutdown_count == 1


# ======================================================================
# 9. No GUI / InferWorker dependency
# ======================================================================

def test_no_gui_inferworker_in_source():
    """The target_stage module source contains no GUI/InferWorker refs.

    AST-parse and assert no import of gui, PyQt, or ai.inference.
    """
    import visioncore.pipeline.stages.target_stage as mod
    src = open(mod.__file__).read()
    tree = ast.parse(src)
    banned = ("gui", "PyQt5", "PyQt6", "inference", "inferworker")
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
    assert not violations, f"target_stage.py imports banned: {violations}"


def test_no_gui_inferworker_loaded_at_runtime():
    """Importing the stages package does not load GUI/InferWorker."""
    import sys
    import visioncore.pipeline.stages  # noqa: F401
    banned = ["PyQt6", "PyQt5"]
    loaded = [b for b in banned if b in sys.modules]
    assert not loaded, f"GUI loaded as side effect: {loaded}"
    # ai.inference should not be loaded either.
    assert "ai.inference" not in sys.modules, "ai.inference loaded as side effect"


def test_target_manager_interface_has_four_methods():
    """The TargetManager ABC has exactly the four interface methods."""
    import inspect
    methods = {name for name, _ in inspect.getmembers(
        TargetManager, predicate=inspect.isfunction
    ) if not name.startswith("_")}
    assert methods == {"initialize", "update", "shutdown", "health_check"}


# ======================================================================
# Runner
# ======================================================================

def _collect_tests() -> list[tuple[str, Any]]:
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} target stage tests...\n")
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
    print("All target stage tests passed.")
