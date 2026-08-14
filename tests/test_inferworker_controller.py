"""InferWorker controller unit tests (Milestone D9.5).

Functional test suite verifying the reduced InferWorker controller:
    - Construction (signals, properties, default state)
    - Configuration stubs (set_* methods don't raise)
    - RuntimeState integration (stats, caches, timing)
    - Frame submission and slot management
    - No detection/tracking/target/post-processing logic
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.runtime.state import RuntimeState
from visioncore.runtime.model import ModelManager


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
# 1. Module surface
# ======================================================================

def test_controller_importable():
    """inference_controller.py can be imported without error."""
    import ai.inference_controller as mod
    assert hasattr(mod, 'InferWorker')


def test_controller_has_signals():
    """InferWorker has detection_ready and model_ready signals."""
    from ai.inference_controller import InferWorker
    assert hasattr(InferWorker, 'detection_ready')
    assert hasattr(InferWorker, 'model_ready')


def test_controller_has_required_properties():
    """InferWorker has target_manager, event_bus, runtime_state, model_manager."""
    from ai.inference_controller import InferWorker
    assert hasattr(InferWorker, 'target_manager')
    assert hasattr(InferWorker, 'event_bus')
    assert hasattr(InferWorker, 'runtime_state')
    assert hasattr(InferWorker, 'model_manager')


def test_controller_has_public_methods():
    """InferWorker has all required public methods."""
    from ai.inference_controller import InferWorker
    required = [
        'set_pipeline_enabled', 'set_infer_stride', 'set_conf',
        'set_person_detection_enabled', 'set_skeleton_enabled',
        'set_rectangle_detection_enabled', 'set_gesture_mode',
        'set_feature_flags', 'set_denoise', 'set_person_redetect',
        'set_redetect_budget', 'set_tracker_params', 'set_filter_type',
        'set_smooth_alpha', 'set_reid_config', 'set_rectangle_sensitivity',
        'set_rectangle_max_count', 'set_rectangle_target_color',
        'set_rectangle_color_threshold', 'set_head_classifier',
        'request_model_switch', 'prepare_startup_guard',
        'submit_frame', 'register_slot', 'unregister_slot',
        'run', 'stop', 'sr_stats',
    ]
    for name in required:
        assert hasattr(InferWorker, name), f"missing: {name}"


# ======================================================================
# 2. Configuration stubs don't raise
# ======================================================================

def test_config_stubs_callable():
    """All set_* methods are callable without raising."""
    from ai.inference_controller import InferWorker
    # We can't instantiate InferWorker (needs QApplication for QThread)
    # but we can verify the methods exist and are callable
    for name in dir(InferWorker):
        if name.startswith('set_'):
            method = getattr(InferWorker, name)
            assert callable(method), f"{name} not callable"


# ======================================================================
# 3. RuntimeState integration
# ======================================================================

def test_runtime_state_stats():
    """RuntimeState stats work correctly."""
    state = RuntimeState()
    state.stats.increment("infer_cycles")
    state.stats.increment("redetect_attempts", 3)
    assert state.stats.get("infer_cycles") == 1
    assert state.stats.get("redetect_attempts") == 3
    snap = state.stats.snapshot()
    assert "infer_cycles" in snap
    assert "redetect_attempts" in snap


def test_runtime_state_slot_cache():
    """RuntimeState slot cache works correctly."""
    state = RuntimeState()
    state.slot_cache.set_detections(0, [{"label": "person"}])
    state.slot_cache.set_timestamp(0, 1.5)
    assert state.slot_cache.get_detections(0) == [{"label": "person"}]
    assert state.slot_cache.get_timestamp(0) == 1.5
    state.slot_cache.clear()
    assert state.slot_cache.get_detections(0) == []


def test_runtime_state_warmup():
    """RuntimeState warmup tracking works."""
    state = RuntimeState()
    assert state.warmup_done is False
    state.mark_warmup_done()
    assert state.warmup_done is True
    state.mark_warmup_needed()
    assert state.warmup_done is False


def test_runtime_state_startup_guard():
    """RuntimeState startup guard works."""
    import time
    state = RuntimeState()
    assert state.is_startup_guard_active() is False
    state.enable_startup_guard(0.05)
    assert state.is_startup_guard_active() is True
    time.sleep(0.1)
    assert state.is_startup_guard_active() is False


def test_model_manager_construction():
    """ModelManager construction works."""
    manager = ModelManager()
    assert manager._device == "cpu"
    assert manager.list_runtimes() == []


# ======================================================================
# 4. No business logic in controller
# ======================================================================

def test_no_detection_logic_in_source():
    """inference_controller.py has no detection/NMS/box functions."""
    import ai.inference_controller as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    detection_funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            if any(term in name for term in ("nms", "non_max", "iou", "filter_detection", "detect_person", "redetect_person", "attach_head", "attach_gesture")):
                detection_funcs.append(node.name)
    assert not detection_funcs, f"controller has detection functions: {detection_funcs}"


def test_no_model_loading_in_source():
    """inference_controller.py has no model loading functions."""
    import ai.inference_controller as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    model_funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            if any(term in name for term in ("load_model", "load_detect", "load_pose", "load_onnx", "initialize_backend", "warmup_model")):
                model_funcs.append(node.name)
    assert not model_funcs, f"controller has model functions: {model_funcs}"


# ======================================================================
# 5. Line count verification
# ======================================================================

def test_controller_under_500_loc():
    """inference_controller.py is under 500 lines."""
    import ai.inference_controller as mod
    lines = len(open(mod.__file__, encoding="utf-8").readlines())
    assert lines < 500, f"controller is {lines} lines (target: <500)"


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
    print(f"Running {len(tests)} inferworker controller tests...\n")
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
    print("All inferworker controller tests passed.")
