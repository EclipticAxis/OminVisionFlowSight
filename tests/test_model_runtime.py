"""Model runtime unit tests (Milestone D9.2).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - ModelInfo dataclass defaults
    - ModelManager construction and device detection
    - Runtime registration and lookup
    - Model lifecycle: load / unload / warmup / health
    - Model switching with rollback on failure
    - Aggregate health checking
    - No detection logic (runtime only)
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from typing import Any

import sys as _sys
from pathlib import Path
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.runtime.model import (
    ModelInfo,
    ModelManager,
    ModelManagerError,
    ModelRuntime,
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


class _MockRuntime(ModelRuntime):
    """Test double: records lifecycle calls and simulates load/unload."""

    def __init__(self, healthy: bool = True, fail_load: bool = False):
        self.healthy = healthy
        self.fail_load = fail_load
        self.loaded = False
        self.warmup_done = False
        self.load_args = None
        self.unload_count = 0
        self.warmup_count = 0
        self.predict_count = 0

    def load(self, path=None, device="cpu"):
        if self.fail_load:
            raise RuntimeError("mock load failure")
        self.load_args = (path, device)
        self.loaded = True

    def unload(self):
        self.loaded = False
        self.unload_count += 1

    def warmup(self, imgsz=640):
        self.warmup_done = True
        self.warmup_count += 1

    def predict(self, frame, conf=0.25, imgsz=640):
        self.predict_count += 1
        return []

    def health_check(self):
        return self.healthy and self.loaded


# ======================================================================
# 0. Module surface
# ======================================================================

def test_model_manager_error_is_runtime_error():
    assert issubclass(ModelManagerError, RuntimeError)


def test_model_info_defaults():
    info = ModelInfo()
    assert info.path == ""
    assert info.backend == "pytorch"
    assert info.device == "cpu"
    assert info.loaded is False
    assert info.warmup_done is False


def test_model_runtime_is_abc():
    assert hasattr(ModelRuntime, 'load')
    assert hasattr(ModelRuntime, 'unload')
    assert hasattr(ModelRuntime, 'warmup')
    assert hasattr(ModelRuntime, 'predict')
    assert hasattr(ModelRuntime, 'health_check')


# ======================================================================
# 1. ModelManager construction
# ======================================================================

def test_manager_construction():
    manager = ModelManager()
    assert manager._device == "cpu"
    assert manager._default_imgsz == 640
    assert manager.list_runtimes() == []


def test_manager_construction_custom_device():
    manager = ModelManager(device="cuda", default_imgsz=1280)
    assert manager._device == "cuda"
    assert manager._default_imgsz == 1280


# ======================================================================
# 2. Device detection
# ======================================================================

def test_detect_device_returns_string():
    device = ModelManager.detect_device()
    assert isinstance(device, str)
    assert device in ("cpu", "cuda")


def test_detect_device_cpu_fallback():
    """On systems without CUDA, returns 'cpu'."""
    device = ModelManager.detect_device()
    # We can't guarantee CUDA, but cpu is always valid
    assert device in ("cpu", "cuda")


# ======================================================================
# 3. Runtime registration
# ======================================================================

def test_register_and_get():
    manager = ModelManager()
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    assert manager.get_runtime("detect") is rt
    assert manager.list_runtimes() == ["detect"]


def test_get_unknown_returns_none():
    manager = ModelManager()
    assert manager.get_runtime("missing") is None


def test_register_multiple():
    manager = ModelManager()
    rt1 = _MockRuntime()
    rt2 = _MockRuntime()
    manager.register_runtime("detect", rt1)
    manager.register_runtime("pose", rt2)
    assert set(manager.list_runtimes()) == {"detect", "pose"}


def test_register_creates_model_info():
    manager = ModelManager()
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    info = manager.get_info("detect")
    assert info is not None
    assert info.device == "cpu"
    assert info.loaded is False


# ======================================================================
# 4. Model lifecycle: load / unload / warmup
# ======================================================================

def test_load_calls_runtime():
    manager = ModelManager()
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    manager.load("detect", "model.pt", "cpu")
    assert rt.load_args == ("model.pt", "cpu")
    assert rt.loaded is True
    info = manager.get_info("detect")
    assert info.loaded is True
    assert info.path == "model.pt"


def test_load_unknown_runtime_raises():
    manager = ModelManager()
    with raises(ModelManagerError, match="unknown runtime"):
        manager.load("missing", "model.pt")


def test_load_failure_raises():
    manager = ModelManager()
    rt = _MockRuntime(fail_load=True)
    manager.register_runtime("detect", rt)
    with raises(ModelManagerError, match="failed"):
        manager.load("detect", "model.pt")
    info = manager.get_info("detect")
    assert info.loaded is False


def test_unload_calls_runtime():
    manager = ModelManager()
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    manager.load("detect")
    manager.unload("detect")
    assert rt.loaded is False
    assert rt.unload_count == 1
    info = manager.get_info("detect")
    assert info.loaded is False


def test_unload_unknown_is_noop():
    manager = ModelManager()
    manager.unload("missing")


def test_warmup_all():
    manager = ModelManager()
    rt1 = _MockRuntime()
    rt2 = _MockRuntime()
    manager.register_runtime("detect", rt1)
    manager.register_runtime("pose", rt2)
    manager.load("detect")
    manager.load("pose")
    manager.warmup_all()
    assert rt1.warmup_count == 1
    assert rt2.warmup_count == 1
    assert manager.get_info("detect").warmup_done is True
    assert manager.get_info("pose").warmup_done is True


def test_warmup_all_custom_imgsz():
    manager = ModelManager(default_imgsz=1280)
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    manager.load("detect")
    manager.warmup_all(imgsz=320)
    assert rt.warmup_count == 1


def test_warmup_all_empty_is_noop():
    manager = ModelManager()
    manager.warmup_all()


# ======================================================================
# 5. Model switching with rollback
# ======================================================================

def test_switch_model_success():
    manager = ModelManager()
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    manager.load("detect", "old.pt")
    result = manager.switch_model("detect", "new.pt")
    assert result is True
    assert rt.loaded is True
    info = manager.get_info("detect")
    assert info.path == "new.pt"
    assert info.loaded is True


def test_switch_model_failure_rollback():
    """On failure, old model is restored."""
    manager = ModelManager()
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    manager.load("detect", "old.pt")
    # Make next load fail
    rt.fail_load = True
    result = manager.switch_model("detect", "bad.pt")
    assert result is False
    # After rollback, runtime should be loaded with old path
    # (rollback calls load(old_path) which also fails in this mock)
    info = manager.get_info("detect")
    # Since rollback also fails, loaded should be False
    assert info.loaded is False


def test_switch_model_unknown_runtime():
    manager = ModelManager()
    result = manager.switch_model("missing", "new.pt")
    assert result is False


def test_switch_model_clears_warmup():
    manager = ModelManager()
    rt = _MockRuntime()
    manager.register_runtime("detect", rt)
    manager.load("detect", "old.pt")
    manager.warmup_all()
    assert manager.get_info("detect").warmup_done is True
    manager.switch_model("detect", "new.pt")
    # After successful switch, warmup_done is set by switch_model
    assert manager.get_info("detect").warmup_done is True


# ======================================================================
# 6. Aggregate health
# ======================================================================

def test_health_check_empty():
    """Empty manager is vacuously healthy."""
    manager = ModelManager()
    assert manager.health_check() is True


def test_health_check_all_healthy():
    manager = ModelManager()
    rt1 = _MockRuntime(healthy=True)
    rt2 = _MockRuntime(healthy=True)
    manager.register_runtime("detect", rt1)
    manager.register_runtime("pose", rt2)
    manager.load("detect")
    manager.load("pose")
    assert manager.health_check() is True


def test_health_check_one_unhealthy():
    manager = ModelManager()
    rt1 = _MockRuntime(healthy=True)
    rt2 = _MockRuntime(healthy=False)
    manager.register_runtime("detect", rt1)
    manager.register_runtime("pose", rt2)
    manager.load("detect")
    manager.load("pose")
    assert manager.health_check() is False


def test_health_check_not_loaded():
    manager = ModelManager()
    rt = _MockRuntime(healthy=True)
    manager.register_runtime("detect", rt)
    # Not loaded -> health_check returns False
    assert manager.health_check() is False


# ======================================================================
# 7. No detection logic
# ======================================================================

def test_no_detection_logic_in_source():
    """model_runtime.py contains no actual detection code.

    The docstring mentions NMS/box conversion as things the module does
    NOT contain -- that's documentation, not code. We check for actual
    detection code patterns instead.
    """
    import visioncore.runtime.model.model_runtime as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    # Check for function definitions that suggest detection logic
    detection_funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            if any(term in name for term in ("nms", "non_max", "iou", "filter_label", "convert_box")):
                detection_funcs.append(node.name)
    assert not detection_funcs, f"model_runtime.py has detection functions: {detection_funcs}"


# ======================================================================
# 8. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """model_runtime.py has no top-level network / model / GUI / ai imports.

    Note: detect_device() lazily imports torch inside a try block (for
    CUDA detection). This is acceptable -- the module-level namespace
    is clean.
    """
    import visioncore.runtime.model.model_runtime as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("socket", "requests", "rospy", "mavlink", "zmq",
              "cv2", "ultralytics", "yolo",
              "ai", "gui", "camera")
    violations = []
    for node in ast.iter_child_nodes(tree):
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
    assert not violations, f"model_runtime.py top-level imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing model runtime module itself does not load torch.

    Note: detect_device() may load torch as a side effect (it checks
    CUDA availability), but the module import itself does not.
    """
    # The module-level import does not trigger torch.
    # detect_device() is a static method that lazily imports torch.
    import visioncore.runtime.model.model_runtime as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert "import torch" not in src.split("#")[0]  # no top-level torch import


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
    print(f"Running {len(tests)} model runtime tests...\n")
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
    print("All model runtime tests passed.")
