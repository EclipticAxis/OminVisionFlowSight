"""Runtime state unit tests (Milestone D9.4).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Stats: increment, get, set, reset, snapshot
    - SlotCache: set/get detections, timestamps, clear, clear_slot
    - HeadAttrCache: set/get with stride expiry, clear
    - RuntimeState: warmup, startup guard, redetect stats, infer stats, reset
    - No business logic (no detection/filtering code)
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import time
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.runtime.state import RuntimeState
from visioncore.runtime.state.runtime_state import (
    HeadAttrCache,
    SlotCache,
    Stats,
)


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

def test_runtime_state_importable():
    assert RuntimeState is not None


def test_stats_importable():
    assert Stats is not None


def test_slot_cache_importable():
    assert SlotCache is not None


def test_head_attr_cache_importable():
    assert HeadAttrCache is not None


# ======================================================================
# 1. Stats
# ======================================================================

def test_stats_increment():
    s = Stats()
    s.increment("infer_cycles")
    assert s.get("infer_cycles") == 1


def test_stats_increment_amount():
    s = Stats()
    s.increment("count", 5)
    assert s.get("count") == 5


def test_stats_increment_accumulates():
    s = Stats()
    s.increment("count")
    s.increment("count", 3)
    assert s.get("count") == 4


def test_stats_get_unknown_returns_zero():
    s = Stats()
    assert s.get("missing") == 0


def test_stats_set():
    s = Stats()
    s.set("count", 42)
    assert s.get("count") == 42


def test_stats_reset_single():
    s = Stats()
    s.increment("a")
    s.increment("b")
    s.reset("a")
    assert s.get("a") == 0
    assert s.get("b") == 1


def test_stats_reset_all():
    s = Stats()
    s.increment("a")
    s.increment("b")
    s.reset()
    assert s.get("a") == 0
    assert s.get("b") == 0


def test_stats_snapshot():
    s = Stats()
    s.increment("a", 3)
    s.increment("b", 5)
    snap = s.snapshot()
    assert snap == {"a": 3, "b": 5}
    # Snapshot is a copy
    snap["c"] = 99
    assert s.get("c") == 0


def test_stats_float_values():
    s = Stats()
    s.increment("conf_gain", 0.15)
    s.increment("conf_gain", 0.10)
    assert abs(s.get("conf_gain") - 0.25) < 1e-9


# ======================================================================
# 2. SlotCache
# ======================================================================

def test_slot_cache_set_get_detections():
    cache = SlotCache()
    dets = [{"label": "person", "confidence": 0.9}]
    cache.set_detections(0, dets)
    assert cache.get_detections(0) == dets


def test_slot_cache_get_unknown_returns_empty():
    cache = SlotCache()
    assert cache.get_detections(99) == []


def test_slot_cache_set_get_timestamp():
    cache = SlotCache()
    cache.set_timestamp(0, 1.5)
    assert cache.get_timestamp(0) == 1.5


def test_slot_cache_get_unknown_timestamp_returns_none():
    cache = SlotCache()
    assert cache.get_timestamp(99) is None


def test_slot_cache_clear():
    cache = SlotCache()
    cache.set_detections(0, [{"label": "person"}])
    cache.set_timestamp(0, 1.0)
    cache.clear()
    assert cache.get_detections(0) == []
    assert cache.get_timestamp(0) is None


def test_slot_cache_clear_slot():
    cache = SlotCache()
    cache.set_detections(0, [{"label": "person"}])
    cache.set_detections(1, [{"label": "car"}])
    cache.clear_slot(0)
    assert cache.get_detections(0) == []
    assert cache.get_detections(1) == [{"label": "car"}]


def test_slot_cache_slot_ids():
    cache = SlotCache()
    cache.set_detections(0, [])
    cache.set_detections(2, [])
    assert set(cache.slot_ids) == {0, 2}


def test_slot_cache_multiple_slots():
    cache = SlotCache()
    cache.set_detections(0, [{"label": "person"}])
    cache.set_detections(1, [{"label": "car"}])
    cache.set_detections(2, [{"label": "head"}])
    assert len(cache.slot_ids) == 3
    assert cache.get_detections(1) == [{"label": "car"}]


# ======================================================================
# 3. HeadAttrCache
# ======================================================================

def test_head_attr_cache_set_get():
    cache = HeadAttrCache(stride=2)
    cache.set(1, 10, {"hat": True})
    assert cache.get(1, 10) == {"hat": True}


def test_head_attr_cache_within_stride():
    cache = HeadAttrCache(stride=5)
    cache.set(1, 10, {"hat": True})
    assert cache.get(1, 12) == {"hat": True}  # frame 12, stride 5, cached at 10
    assert cache.get(1, 14) == {"hat": True}  # frame 14, still within stride


def test_head_attr_cache_expired():
    cache = HeadAttrCache(stride=2)
    cache.set(1, 10, {"hat": True})
    assert cache.get(1, 13) is None  # frame 13, stride 2, expired (13-10=3 >= 2)


def test_head_attr_cache_missing_track():
    cache = HeadAttrCache(stride=2)
    assert cache.get(99, 10) is None


def test_head_attr_cache_clear():
    cache = HeadAttrCache(stride=2)
    cache.set(1, 10, {"hat": True})
    cache.clear()
    assert cache.get(1, 10) is None


def test_head_attr_cache_size():
    cache = HeadAttrCache(stride=2)
    cache.set(1, 10, {"hat": True})
    cache.set(2, 10, {"mask": False})
    assert cache.size == 2


# ======================================================================
# 4. RuntimeState
# ======================================================================

def test_runtime_state_defaults():
    state = RuntimeState()
    assert state.warmup_done is False
    assert state.startup_guard_until == 0.0
    assert state.startup_guard_seconds == 3.0


def test_runtime_state_custom_guard():
    state = RuntimeState(startup_guard_seconds=5.0)
    assert state.startup_guard_seconds == 5.0


def test_runtime_state_warmup():
    state = RuntimeState()
    assert state.warmup_done is False
    state.mark_warmup_done()
    assert state.warmup_done is True
    state.mark_warmup_needed()
    assert state.warmup_done is False


def test_runtime_state_startup_guard():
    state = RuntimeState()
    assert state.is_startup_guard_active() is False
    state.enable_startup_guard(0.1)  # 100ms
    assert state.is_startup_guard_active() is True
    time.sleep(0.15)
    assert state.is_startup_guard_active() is False


def test_runtime_state_startup_guard_default_seconds():
    state = RuntimeState(startup_guard_seconds=0.1)
    state.enable_startup_guard()
    assert state.is_startup_guard_active() is True
    time.sleep(0.15)
    assert state.is_startup_guard_active() is False


def test_runtime_state_redetect_stats():
    state = RuntimeState()
    state.record_redetect_attempt()
    state.record_redetect_attempt()
    state.record_redetect_success()
    state.record_redetect_failure()
    stats = state.redetect_stats
    assert stats["attempts"] == 2
    assert stats["successes"] == 1
    assert stats["failures"] == 1


def test_runtime_state_infer_cycle():
    state = RuntimeState()
    assert state.infer_cycle_count == 0
    state.record_infer_cycle()
    state.record_infer_cycle()
    assert state.infer_cycle_count == 2


def test_runtime_state_stats_integration():
    state = RuntimeState()
    state.stats.increment("debug_infer_reports", 10)
    state.stats.increment("debug_submit_reports", 5)
    snap = state.stats.snapshot()
    assert snap["debug_infer_reports"] == 10
    assert snap["debug_submit_reports"] == 5


def test_runtime_state_slot_cache_integration():
    state = RuntimeState()
    state.slot_cache.set_detections(0, [{"label": "person"}])
    state.slot_cache.set_timestamp(0, 1.5)
    assert state.slot_cache.get_detections(0) == [{"label": "person"}]
    assert state.slot_cache.get_timestamp(0) == 1.5


def test_runtime_state_head_attr_cache_integration():
    state = RuntimeState()
    state.head_attr_cache.set(1, 10, {"hat": True})
    assert state.head_attr_cache.get(1, 11) == {"hat": True}


def test_runtime_state_reset():
    state = RuntimeState()
    state.stats.increment("a", 5)
    state.slot_cache.set_detections(0, [{"label": "person"}])
    state.head_attr_cache.set(1, 10, {"hat": True})
    state.mark_warmup_done()
    state.enable_startup_guard(10.0)

    state.reset()

    assert state.stats.get("a") == 0
    assert state.slot_cache.get_detections(0) == []
    assert state.head_attr_cache.get(1, 10) is None
    assert state.warmup_done is False
    assert state.startup_guard_until == 0.0


# ======================================================================
# 5. No business logic
# ======================================================================

def test_no_business_logic_in_source():
    """runtime_state.py contains no detection/filtering/NMS code."""
    import visioncore.runtime.state.runtime_state as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    # Check for function definitions that suggest detection logic
    detection_funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            if any(term in name for term in ("nms", "non_max", "iou", "filter_detection", "detect_person")):
                detection_funcs.append(node.name)
    assert not detection_funcs, f"runtime_state.py has detection functions: {detection_funcs}"


# ======================================================================
# 6. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """runtime_state.py contains no network / model / GUI / ai imports."""
    import visioncore.runtime.state.runtime_state as mod
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
    assert not violations, f"runtime_state.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing runtime state does not load torch."""
    import visioncore.runtime.state  # noqa: F401
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
    print(f"Running {len(tests)} runtime state tests...\n")
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
    print("All runtime state tests passed.")
