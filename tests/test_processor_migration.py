"""Processor migration unit tests (Milestone D9.3).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - ProcessorPlugin ABC: subclassing, process() abstract
    - FilterProcessor: label filter, confidence floor, person visibility,
      keypoints stripping
    - BBoxProcessor: deduplication by IoU, priority sorting, coordinate
      normalisation
    - Plugin contract: name, version, load, shutdown, health_check
    - Input/output consistency: same input produces expected output
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.plugin.processors import (
    BBoxProcessor,
    FilterProcessor,
    ProcessorPlugin,
)
from visioncore.plugin.base import PluginInterface


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
                f"expected {self.expected.__name__}, but no exception was raised"
            )
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


def raises(expected, match=None):
    return _Raises(expected, match=match)


# ======================================================================
# 0. Module surface
# ======================================================================

def test_processor_plugin_is_plugin_interface_subclass():
    assert issubclass(ProcessorPlugin, PluginInterface)


def test_filter_processor_is_processor_plugin_subclass():
    assert issubclass(FilterProcessor, ProcessorPlugin)


def test_bbox_processor_is_processor_plugin_subclass():
    assert issubclass(BBoxProcessor, ProcessorPlugin)


def test_processor_plugin_has_process_method():
    assert hasattr(ProcessorPlugin, 'process')


# ======================================================================
# 1. FilterProcessor -- plugin contract
# ======================================================================

def test_filter_processor_defaults():
    fp = FilterProcessor()
    assert fp.name == "FilterProcessor"
    assert fp.version == "0.1.0"


def test_filter_processor_custom_name():
    fp = FilterProcessor(name="my-filter")
    assert fp.name == "my-filter"


def test_filter_processor_lifecycle():
    fp = FilterProcessor()
    assert fp.health_check() is False
    fp.load()
    assert fp.health_check() is True
    fp.shutdown()
    assert fp.health_check() is False


# ======================================================================
# 2. FilterProcessor -- filtering logic
# ======================================================================

def test_filter_passes_all_when_enabled():
    """All detections pass when person_enabled=True and conf_floor=0."""
    fp = FilterProcessor()
    dets = [
        {"label": "person", "confidence": 0.9},
        {"label": "car", "confidence": 0.8},
    ]
    result = fp.process(dets)
    assert len(result) == 2


def test_filter_removes_person_when_disabled():
    """Person detections removed when person_enabled=False."""
    fp = FilterProcessor()
    dets = [
        {"label": "person", "confidence": 0.9},
        {"label": "car", "confidence": 0.8},
    ]
    result = fp.process(dets, {"person_enabled": False})
    assert len(result) == 1
    assert result[0]["label"] == "car"


def test_filter_confidence_floor():
    """Detections below conf_floor are dropped."""
    fp = FilterProcessor(conf_floor=0.5)
    dets = [
        {"label": "person", "confidence": 0.9},
        {"label": "person", "confidence": 0.3},
        {"label": "car", "confidence": 0.4},
    ]
    result = fp.process(dets)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_filter_strips_keypoints_when_skeleton_disabled():
    """Keypoints stripped from person detections when skeleton_enabled=False."""
    fp = FilterProcessor()
    dets = [
        {"label": "person", "confidence": 0.9, "keypoints": [{"x": 0.5, "y": 0.5}]},
    ]
    result = fp.process(dets, {"skeleton_enabled": False})
    assert result[0]["keypoints"] == []


def test_filter_keeps_keypoints_when_skeleton_enabled():
    """Keypoints preserved when skeleton_enabled=True."""
    fp = FilterProcessor()
    dets = [
        {"label": "person", "confidence": 0.9, "keypoints": [{"x": 0.5, "y": 0.5}]},
    ]
    result = fp.process(dets, {"skeleton_enabled": True})
    assert result[0]["keypoints"] == [{"x": 0.5, "y": 0.5}]


def test_filter_empty_input():
    fp = FilterProcessor()
    assert fp.process([]) == []


def test_filter_no_context():
    """No context dict -> all defaults (person_enabled=True, etc.)."""
    fp = FilterProcessor()
    dets = [{"label": "person", "confidence": 0.9}]
    result = fp.process(dets)
    assert len(result) == 1


# ======================================================================
# 3. BBoxProcessor -- plugin contract
# ======================================================================

def test_bbox_processor_defaults():
    bp = BBoxProcessor()
    assert bp.name == "BBoxProcessor"
    assert bp.version == "0.1.0"


def test_bbox_processor_lifecycle():
    bp = BBoxProcessor()
    assert bp.health_check() is False
    bp.load()
    assert bp.health_check() is True
    bp.shutdown()
    assert bp.health_check() is False


# ======================================================================
# 4. BBoxProcessor -- deduplication
# ======================================================================

def test_bbox_dedup_removes_overlapping():
    """Two detections with high IoU -> one kept."""
    bp = BBoxProcessor(iou_thresholds={"person": 0.3})
    dets = [
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
        {"label": "person", "confidence": 0.8, "x1": 0.11, "y1": 0.11, "x2": 0.31, "y2": 0.31},
    ]
    result = bp.process(dets)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_bbox_dedup_keeps_non_overlapping():
    """Two non-overlapping detections -> both kept."""
    bp = BBoxProcessor(iou_thresholds={"person": 0.3})
    dets = [
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2},
        {"label": "person", "confidence": 0.8, "x1": 0.7, "y1": 0.7, "x2": 0.9, "y2": 0.9},
    ]
    result = bp.process(dets)
    assert len(result) == 2


def test_bbox_dedup_different_labels():
    """Detections with different labels are not compared."""
    bp = BBoxProcessor(iou_thresholds={"person": 0.3})
    dets = [
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
        {"label": "car", "confidence": 0.8, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
    ]
    result = bp.process(dets)
    assert len(result) == 2


def test_bbox_priority_sorting():
    """Higher confidence detection wins over lower when IoU is high."""
    bp = BBoxProcessor(iou_thresholds={"person": 0.1})
    dets = [
        {"label": "person", "confidence": 0.5, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
    ]
    result = bp.process(dets)
    assert len(result) == 1
    assert result[0]["confidence"] == 0.9


def test_bbox_normalise():
    """normalise=True converts bbox to centre form."""
    bp = BBoxProcessor(normalise=True)
    dets = [
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.6},
    ]
    result = bp.process(dets)
    assert abs(result[0]["cx"] - 0.2) < 1e-6
    assert abs(result[0]["cy"] - 0.4) < 1e-6
    assert abs(result[0]["w"] - 0.2) < 1e-6
    assert abs(result[0]["h"] - 0.4) < 1e-6


def test_bbox_no_normalise():
    """normalise=False preserves original keys."""
    bp = BBoxProcessor(normalise=False)
    dets = [
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.6},
    ]
    result = bp.process(dets)
    assert "cx" not in result[0]
    assert result[0]["x1"] == 0.1


def test_bbox_empty_input():
    bp = BBoxProcessor()
    assert bp.process([]) == []


def test_bbox_custom_thresholds():
    """Custom thresholds change deduplication behavior."""
    bp = BBoxProcessor(iou_thresholds={"person": 0.99})  # very strict
    dets = [
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
        {"label": "person", "confidence": 0.8, "x1": 0.11, "y1": 0.11, "x2": 0.31, "y2": 0.31},
    ]
    result = bp.process(dets)
    # With strict threshold, both pass (IoU ~0.86 < 0.99)
    assert len(result) == 2


# ======================================================================
# 5. Input/output consistency
# ======================================================================

def test_filter_input_not_mutated():
    """FilterProcessor does not mutate the input list."""
    fp = FilterProcessor()
    dets = [{"label": "person", "confidence": 0.9, "keypoints": [1, 2, 3]}]
    original_len = len(dets)
    fp.process(dets, {"skeleton_enabled": False})
    assert len(dets) == original_len
    assert dets[0]["keypoints"] == [1, 2, 3]  # original unchanged


def test_bbox_input_not_mutated():
    """BBoxProcessor does not mutate the input list."""
    bp = BBoxProcessor()
    dets = [{"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3}]
    original = dict(dets[0])
    bp.process(dets)
    assert dets[0] == original


def test_chained_processors():
    """FilterProcessor -> BBoxProcessor produces consistent output."""
    fp = FilterProcessor(conf_floor=0.3)
    bp = BBoxProcessor(iou_thresholds={"person": 0.5})
    dets = [
        {"label": "person", "confidence": 0.9, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
        {"label": "person", "confidence": 0.2, "x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.3},
        {"label": "person", "confidence": 0.8, "x1": 0.11, "y1": 0.11, "x2": 0.31, "y2": 0.31},
    ]
    filtered = fp.process(dets)
    deduped = bp.process(filtered)
    assert len(filtered) == 2  # 0.2 dropped by conf_floor
    assert len(deduped) == 1   # remaining two overlap


# ======================================================================
# 6. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """processor.py contains no network / model / GUI / ai imports."""
    import visioncore.plugin.processors.processor as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("socket", "requests", "rospy", "mavlink", "zmq",
              "torch", "onnx", "cv2", "ultralytics", "yolo",
              "ai", "gui", "camera")
    violations = []
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
    assert not violations, f"processor.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing processors does not load torch."""
    import visioncore.plugin.processors  # noqa: F401
    assert "torch" not in sys.modules, "torch loaded as side effect"


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
    print(f"Running {len(tests)} processor migration tests...\n")
    passed = 0
    failed = 0
    failures = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:
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
    print("All processor migration tests passed.")
