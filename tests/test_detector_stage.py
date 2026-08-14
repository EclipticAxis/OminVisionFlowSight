"""DetectorStage unit tests (Milestone C3).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary. A tiny ``raises``
context manager replaces ``pytest.raises``.

Covers:
    - Detector ABC contract (abstract, subclass must implement)
    - DummyDetector (fixed Detections, counters, raise_on_detect, reset)
    - DetectorStage construction & lifecycle delegation
    - DetectorStage.process data flow (frame -> detections)
    - frame=None skip semantics
    - Exception propagation (detector.detect raises -> process raises)
    - Pipeline integration (end-to-end source -> frame -> detect)
    - No YOLO / RT-DETR / GroundingDINO / SAM dependency
"""
from __future__ import annotations

import ast
import traceback
from typing import Any

import numpy as np

from visioncore.core.detection import BBox, Detection
from visioncore.core.frame import Frame
from visioncore.pipeline import Pipeline, PipelineContext, PipelineStage
from visioncore.pipeline.stages import (
    Detector,
    DetectorError,
    DetectorStage,
    DummyDetector,
)
from visioncore.source import DummyFrameSource


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


def _make_frame(frame_id: int = 0, source_id: str = "cam0") -> Frame:
    """Build a small synthetic Frame for tests."""
    return Frame(frame_id, float(frame_id), source_id,
                 np.zeros((8, 8, 3), dtype=np.uint8))


def _make_detection(class_name: str = "person", score: float = 0.9) -> Detection:
    """Build a synthetic Detection."""
    return Detection(BBox(0.5, 0.5, 0.2, 0.4), score, 0, class_name)


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_package_exports_public_api():
    """The stages __all__ includes the four detector names.

    C4 added tracker exports to the same package; this test asserts the
    detector names are present (subset), not that __all__ is exactly the
    detector set -- the package legitimately grows as stages are added.
    """
    import visioncore.pipeline.stages as pkg
    detector_names = {
        "Detector", "DetectorError", "DetectorStage", "DummyDetector",
    }
    assert detector_names.issubset(set(pkg.__all__)), (
        f"detector names {detector_names} not all in stages __all__: {pkg.__all__}"
    )


def test_detector_error_is_runtime_error_subclass():
    """DetectorError is a RuntimeError subclass."""
    assert issubclass(DetectorError, RuntimeError)


def test_detector_is_abstract():
    """Detector is an ABC and cannot be instantiated directly."""
    with raises(TypeError):
        Detector()  # type: ignore[abstract]


def test_detector_subclass_missing_methods_fails():
    """A Detector subclass that omits methods cannot be instantiated."""
    class Incomplete(Detector):
        def detect(self, frame):
            return []
        # initialize / shutdown / health_check missing
    with raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ======================================================================
# 1. DummyDetector
# ======================================================================

def test_dummy_detector_default_returns_one_person():
    """A default DummyDetector returns one person Detection."""
    det = DummyDetector()
    det.initialize()
    dets = det.detect(_make_frame())
    assert len(dets) == 1
    assert dets[0].class_name == "person"
    assert dets[0].score == 0.9


def test_dummy_detector_custom_detections():
    """Explicit detections are replayed."""
    custom = [_make_detection("car", 0.8), _make_detection("dog", 0.7)]
    det = DummyDetector(detections=custom)
    det.initialize()
    dets = det.detect(_make_frame())
    assert len(dets) == 2
    assert dets[0].class_name == "car"
    assert dets[1].class_name == "dog"


def test_dummy_detector_detect_returns_fresh_list():
    """detect() returns a new list each call (caller may mutate safely)."""
    det = DummyDetector()
    det.initialize()
    f = _make_frame()
    l1 = det.detect(f)
    l2 = det.detect(f)
    assert l1 is not l2  # different list objects
    assert l1 == l2      # same contents


def test_dummy_detector_shares_immutable_detection_instances():
    """The Detection instances themselves are shared (frozen, safe)."""
    det = DummyDetector()
    det.initialize()
    f = _make_frame()
    l1 = det.detect(f)
    l2 = det.detect(f)
    assert l1[0] is l2[0]  # same frozen Detection instance


def test_dummy_detector_lifecycle_and_counters():
    """initialize/detect/shutdown increment counters and toggle health."""
    det = DummyDetector()
    assert det.health_check() is False
    det.initialize()
    assert det.health_check() is True
    det.detect(_make_frame())
    det.shutdown()
    assert det.health_check() is False
    assert (det.initialize_count, det.detect_count, det.shutdown_count) == (1, 1, 1)


def test_dummy_detector_raise_on_detect():
    """raise_on_detect propagates the configured exception."""
    det = DummyDetector(raise_on_detect=DetectorError("model OOM"))
    det.initialize()
    with raises(DetectorError, match="model OOM"):
        det.detect(_make_frame())
    assert det.detect_count == 1  # incremented before raising


def test_dummy_detector_raise_persists_until_cleared():
    """raise_on_detect raises every detect until cleared."""
    det = DummyDetector(raise_on_detect=ValueError("bad"))
    det.initialize()
    with raises(ValueError):
        det.detect(_make_frame())
    with raises(ValueError):
        det.detect(_make_frame())
    det.raise_on_detect = None
    assert det.detect(_make_frame()) is not None  # now works


def test_dummy_detector_reset():
    """reset() zeroes counters and state, preserves the detection buffer."""
    det = DummyDetector()
    det.initialize()
    det.detect(_make_frame())
    det.shutdown()
    det.reset()
    assert (det.initialize_count, det.detect_count, det.shutdown_count) == (0, 0, 0)
    assert det.health_check() is False
    assert det.detection_count == 1  # buffer preserved


# ======================================================================
# 2. DetectorStage construction
# ======================================================================

def test_stage_construction_wraps_detector():
    """DetectorStage stores the wrapped detector and reports health."""
    det = DummyDetector()
    stage = DetectorStage(det)
    assert stage.detector is det
    assert stage.health_check() is False  # detector not initialised yet


def test_stage_default_name():
    """Default stage name is 'DetectorStage'."""
    stage = DetectorStage(DummyDetector())
    assert stage.name == "DetectorStage"


def test_stage_custom_name():
    """An explicit name overrides the default."""
    stage = DetectorStage(DummyDetector(), name="person-detector")
    assert stage.name == "person-detector"


def test_stage_rejects_none_detector():
    """Constructing with None raises TypeError."""
    with raises(TypeError, match="Detector"):
        DetectorStage(None)  # type: ignore[arg-type]


def test_stage_rejects_non_detector():
    """Constructing with a non-Detector raises TypeError."""
    with raises(TypeError, match="Detector"):
        DetectorStage("not a detector")  # type: ignore[arg-type]


def test_stage_is_pipeline_stage():
    """DetectorStage is a PipelineStage subclass."""
    assert issubclass(DetectorStage, PipelineStage)


# ======================================================================
# 3. DetectorStage lifecycle delegation
# ======================================================================

def test_stage_initialize_delegates_to_detector():
    """stage.initialize() calls detector.initialize()."""
    det = DummyDetector()
    stage = DetectorStage(det)
    stage.initialize()
    assert det.initialize_count == 1
    assert stage.health_check() is True  # detector now healthy


def test_stage_shutdown_delegates_to_detector():
    """stage.shutdown() calls detector.shutdown()."""
    det = DummyDetector()
    stage = DetectorStage(det)
    stage.initialize()
    stage.shutdown()
    assert det.shutdown_count == 1
    assert stage.health_check() is False


def test_stage_health_check_delegates_to_detector():
    """stage.health_check() reflects detector health."""
    det = DummyDetector()
    stage = DetectorStage(det)
    assert stage.health_check() is False
    stage.initialize()
    assert stage.health_check() is True
    stage.shutdown()
    assert stage.health_check() is False


def test_stage_repr_reports_health():
    """__repr__ reports the stage name and health."""
    stage = DetectorStage(DummyDetector(), name="detect")
    stage.initialize()
    r = repr(stage)
    assert "DetectorStage" in r
    assert "name='detect'" in r
    assert "healthy=True" in r


# ======================================================================
# 4. DetectorStage.process data flow
# ======================================================================

def test_process_reads_frame_writes_detections():
    """process() reads context.frame, writes context.detections."""
    stage = DetectorStage(DummyDetector())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    stage.process(ctx)
    assert len(ctx.detections) == 1
    assert ctx.detections[0].class_name == "person"


def test_process_with_custom_detections():
    """process() writes whatever the detector returns."""
    custom = [_make_detection("car"), _make_detection("truck")]
    stage = DetectorStage(DummyDetector(detections=custom))
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    stage.process(ctx)
    assert [d.class_name for d in ctx.detections] == ["car", "truck"]


def test_process_extends_existing_detections():
    """process() extends (not replaces) context.detections."""
    stage = DetectorStage(DummyDetector())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    # Pre-populate with a prior detection (simulating an earlier stage)
    ctx.detections.append(_make_detection("preexisting"))
    stage.process(ctx)
    assert len(ctx.detections) == 2  # 1 preexisting + 1 from detector
    assert ctx.detections[0].class_name == "preexisting"
    assert ctx.detections[1].class_name == "person"


def test_process_with_none_frame_skips():
    """process() with frame=None skips detection (no error, no detections)."""
    stage = DetectorStage(DummyDetector())
    stage.initialize()
    ctx = PipelineContext.empty()
    assert ctx.frame is None
    stage.process(ctx)  # must not raise
    assert ctx.detections == []  # nothing added
    assert stage.detector.detect_count == 0  # detector never called


def test_process_increments_detector_detect_count():
    """process() calls detector.detect() exactly once per frame."""
    stage = DetectorStage(DummyDetector())
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    stage.process(ctx)
    stage.process(ctx)
    assert stage.detector.detect_count == 2


def test_process_does_not_mutate_frame():
    """process() reads frame but does not mutate it (Frame is frozen)."""
    stage = DetectorStage(DummyDetector())
    stage.initialize()
    ctx = PipelineContext.empty()
    frame = _make_frame(frame_id=42)
    ctx.frame = frame
    stage.process(ctx)
    assert ctx.frame is frame  # same instance, untouched
    assert ctx.frame.frame_id == 42


# ======================================================================
# 5. Exception propagation
# ======================================================================

def test_process_propagates_detector_exception():
    """A detector.detect() exception propagates out of process()."""
    det = DummyDetector(raise_on_detect=DetectorError("inference failed"))
    stage = DetectorStage(det)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    with raises(DetectorError, match="inference failed"):
        stage.process(ctx)
    # No detections written (the detect call raised)
    assert ctx.detections == []


def test_process_propagates_arbitrary_exception():
    """Any exception type from detect() propagates."""
    det = DummyDetector(raise_on_detect=RuntimeError("boom"))
    stage = DetectorStage(det)
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    with raises(RuntimeError, match="boom"):
        stage.process(ctx)


def test_process_exception_does_not_write_partial_detections():
    """If detect() raises, context.detections is unchanged."""
    stage = DetectorStage(
        DummyDetector(raise_on_detect=ValueError("x"))
    )
    stage.initialize()
    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    ctx.detections.append(_make_detection("prior"))
    try:
        stage.process(ctx)
    except ValueError:
        pass
    assert len(ctx.detections) == 1  # only the prior, nothing added


# ======================================================================
# 6. Pipeline integration (end-to-end)
# ======================================================================

def test_pipeline_with_detector_stage_end_to_end():
    """A Pipeline with a DetectorStage produces detections from a frame."""
    stage = DetectorStage(DummyDetector(), name="detect")
    p = Pipeline()
    p.add_stage(stage)

    ctx = PipelineContext.empty(timestamp=1.0)
    ctx.frame = _make_frame(source_id="cam0")
    with p:
        p.run(ctx)

    assert len(ctx.detections) == 1
    assert ctx.detections[0].class_name == "person"
    assert stage.detector.shutdown_count == 1  # lifecycle ran


def test_pipeline_detector_stage_lifecycle_once_per_run():
    """initialize once, process once per run, shutdown once."""
    det = DummyDetector()
    stage = DetectorStage(det)
    p = Pipeline()
    p.add_stage(stage)

    ctx = PipelineContext.empty()
    ctx.frame = _make_frame()
    with p:
        p.run(ctx)

    assert det.initialize_count == 1
    assert det.detect_count == 1
    assert det.shutdown_count == 1


def test_pipeline_multiple_runs_reuse_initialization():
    """One initialize supports multiple runs; shutdown once at end."""
    det = DummyDetector()
    stage = DetectorStage(det)
    p = Pipeline()
    p.add_stage(stage)
    p.initialize()
    for i in range(3):
        ctx = PipelineContext.empty(timestamp=float(i))
        ctx.frame = _make_frame(frame_id=i)
        p.run(ctx)
    p.shutdown()
    assert det.initialize_count == 1
    assert det.detect_count == 3
    assert det.shutdown_count == 1


def test_pipeline_no_frame_skips_detection():
    """A pipeline run with frame=None skips detection gracefully."""
    stage = DetectorStage(DummyDetector())
    p = Pipeline()
    p.add_stage(stage)
    with p:
        ctx = p.run(PipelineContext.empty())  # frame is None
    assert ctx.detections == []
    assert stage.detector.detect_count == 0


# ======================================================================
# 7. End-to-end: FrameSource -> frame -> DetectorStage -> detections
# ======================================================================

def test_end_to_end_source_feeds_detector():
    """A DummyFrameSource yields a Frame that DetectorStage detects on.

    This exercises the C1->C2->C3 chain: source produces Frame (C2),
    a feeder stage puts it in context.frame, DetectorStage (C3) detects.
    """
    # A tiny feeder stage that reads from a FrameSource into context.frame
    class FeederStage(PipelineStage):
        def __init__(self, source):
            super().__init__(name="feeder")
            self._source = source
        def initialize(self):
            self._source.open()
        def process(self, context):
            context.frame = self._source.read()
        def shutdown(self):
            self._source.close()
        def health_check(self):
            return self._source.health_check()

    src = DummyFrameSource(source_id="cam0")
    feeder = FeederStage(src)
    detector = DummyDetector()
    detect_stage = DetectorStage(detector, name="detect")

    p = Pipeline()
    p.add_stage(feeder)
    p.add_stage(detect_stage)

    with p:
        ctx = p.run(PipelineContext.empty(timestamp=1.0))

    assert ctx.frame is not None
    assert ctx.frame.source_id == "cam0"
    assert len(ctx.detections) == 1
    assert ctx.detections[0].class_name == "person"
    # Both stages' resources released
    assert detector.shutdown_count == 1
    assert src.close_count == 1


# ======================================================================
# 8. No YOLO / RT-DETR / GroundingDINO / SAM dependency
# ======================================================================

def test_no_yolo_rt_detr_groundingdino_sam_in_source():
    """The detector_stage module source contains no banned backend refs.

    AST-parse the module and assert no import references YOLO, RT-DETR,
    GroundingDINO, or SAM. This is the architectural firewall: the stage
    depends on the abstract Detector interface, never on a concrete model
    family.
    """
    import visioncore.pipeline.stages.detector_stage as mod
    src = open(mod.__file__).read()
    tree = ast.parse(src)
    banned = ("yolo", "rt_detr", "rtdetr", "rt-detr", "groundingdino",
              "grounding_dino", "grounding-dino", "sam", "ultralytics",
              "onnxruntime", "cv2", "opencv")
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
        f"detector_stage.py imports banned backend(s): {violations}"
    )


def test_no_yolo_rt_detr_groundingdino_sam_loaded_at_runtime():
    """Importing the stages package does not load any banned backend."""
    import sys
    # Ensure a clean-ish view: the banned names must not be present after
    # importing the stages package.
    import visioncore.pipeline.stages  # noqa: F401
    banned = ["ultralytics", "onnxruntime", "cv2"]
    loaded = [b for b in banned if b in sys.modules]
    assert not loaded, (
        f"banned backend(s) loaded as side effect: {loaded}"
    )


def test_detector_interface_is_model_agnostic():
    """The Detector ABC has no model-specific attributes or methods."""
    import inspect
    methods = {name for name, _ in inspect.getmembers(Detector, predicate=inspect.isfunction)
               if not name.startswith("_")}
    # The four interface methods -- nothing model-specific.
    assert methods == {"initialize", "detect", "shutdown", "health_check"}


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
    print(f"Running {len(tests)} detector stage tests...\n")
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
    print("All detector stage tests passed.")
