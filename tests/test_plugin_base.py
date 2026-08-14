"""Plugin interface layer unit tests (Milestone D6).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Package surface (PluginInterface / PluginManager / example plugins)
    - Abstractness (interface + manager cannot be instantiated)
    - NullPlugin: basic methods callable, run() is an identity passthrough
    - ConsolePlugin: basic methods callable, logs without network I/O
    - PluginManager interface: register / unregister / get / list /
      names / contains / load_all / shutdown_all raise nothing and work
    - Registration validation (non-plugin rejected, duplicates rejected)
    - Full-path type annotations (visioncore.state.target_state.TargetState)
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
from abc import ABC
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visioncore.plugin import (
    ConsolePlugin,
    MemoryPluginManager,
    NullPlugin,
    PluginError,
    PluginInterface,
    PluginManager,
)
from visioncore.state.target_state import TargetState


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


def _snapshot(target_id: int = 1, label: str = "person") -> TargetState:
    """Build a synthetic TargetState snapshot for tests."""
    return TargetState(
        target_id=target_id, local_id=1, global_id=None, label=label,
        confidence=0.9, cx=0.5, cy=0.5, vx=0.0, vy=0.0,
        width=0.2, height=0.4, timestamp=0.0, camera_id=0, metadata={},
    )


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_plugin_interface_is_abc():
    assert issubclass(PluginInterface, ABC)


def test_plugin_manager_is_abc():
    assert issubclass(PluginManager, ABC)


def test_null_plugin_is_plugin_interface_subclass():
    assert issubclass(NullPlugin, PluginInterface)


def test_console_plugin_is_plugin_interface_subclass():
    assert issubclass(ConsolePlugin, PluginInterface)


def test_memory_manager_is_plugin_manager_subclass():
    assert issubclass(MemoryPluginManager, PluginManager)


def test_plugin_error_is_runtime_error():
    assert issubclass(PluginError, RuntimeError)


# ======================================================================
# 1. Abstractness
# ======================================================================

def test_plugin_interface_cannot_be_instantiated():
    """PluginInterface is abstract -- direct instantiation raises TypeError."""
    with raises(TypeError, match="abstract"):
        PluginInterface()


def test_plugin_manager_cannot_be_instantiated():
    """PluginManager is abstract -- direct instantiation raises TypeError."""
    with raises(TypeError, match="abstract"):
        PluginManager()


def test_incomplete_plugin_cannot_be_instantiated():
    """A subclass that skips part of the contract stays abstract."""
    class Incomplete(PluginInterface):
        pass

    with raises(TypeError, match="abstract"):
        Incomplete()


# ======================================================================
# 2. NullPlugin -- basic methods callable
# ======================================================================

def test_null_plugin_defaults():
    plugin = NullPlugin()
    assert plugin.name == "null"
    assert plugin.version == "0.1.0"


def test_null_plugin_custom_name():
    plugin = NullPlugin("my-null")
    assert plugin.name == "my-null"


def test_null_plugin_load_callable():
    """load() raises nothing and marks the plugin loaded."""
    plugin = NullPlugin()
    assert plugin.health_check() is False
    plugin.load()
    assert plugin.health_check() is True


def test_null_plugin_run_returns_input():
    """run() is callable and returns the exact same snapshot (identity)."""
    plugin = NullPlugin()
    ts = _snapshot(42)
    result = plugin.run(ts)
    assert result is ts
    assert isinstance(result, TargetState)


def test_null_plugin_run_preserves_snapshot():
    """run() leaves the snapshot untouched."""
    plugin = NullPlugin()
    ts = _snapshot(42, "car")
    result = plugin.run(ts)
    assert result.target_id == 42
    assert result.label == "car"
    assert result.confidence == 0.9


def test_null_plugin_shutdown_callable():
    """shutdown() raises nothing and marks the plugin unloaded."""
    plugin = NullPlugin()
    plugin.load()
    plugin.shutdown()
    assert plugin.health_check() is False


def test_null_plugin_full_lifecycle():
    """load -> run -> shutdown completes without raising."""
    plugin = NullPlugin()
    plugin.load()
    plugin.run(_snapshot(1))
    plugin.shutdown()


# ======================================================================
# 3. ConsolePlugin -- example logging plugin
# ======================================================================

def test_console_plugin_defaults():
    plugin = ConsolePlugin()
    assert plugin.name == "console"
    assert plugin.version == "0.1.0"


def test_console_plugin_custom_name():
    plugin = ConsolePlugin("my-console")
    assert plugin.name == "my-console"


def test_console_plugin_load_callable():
    plugin = ConsolePlugin()
    assert plugin.health_check() is False
    plugin.load()
    assert plugin.health_check() is True


def test_console_plugin_run_returns_input():
    """run() is callable, returns the snapshot, and does no I/O."""
    plugin = ConsolePlugin()
    ts = _snapshot(7)
    result = plugin.run(ts)
    assert result is ts


def test_console_plugin_shutdown_callable():
    plugin = ConsolePlugin()
    plugin.load()
    plugin.shutdown()
    assert plugin.health_check() is False


def test_console_plugin_full_lifecycle():
    """load -> run -> shutdown completes without raising."""
    plugin = ConsolePlugin()
    plugin.load()
    plugin.run(_snapshot(1))
    plugin.shutdown()


# ======================================================================
# 4. PluginManager interface -- register / unregister / get / list
# ======================================================================

def test_manager_register_then_get():
    manager = MemoryPluginManager()
    plugin = NullPlugin("a")
    manager.register(plugin)
    assert manager.get("a") is plugin


def test_manager_get_unknown_returns_none():
    """Querying an unregistered name returns None -- no exception."""
    manager = MemoryPluginManager()
    assert manager.get("missing") is None


def test_manager_contains():
    manager = MemoryPluginManager()
    manager.register(NullPlugin("a"))
    assert "a" in manager
    assert "missing" not in manager
    assert 42 not in manager


def test_manager_unregister_removes():
    manager = MemoryPluginManager()
    manager.register(NullPlugin("a"))
    manager.unregister("a")
    assert manager.get("a") is None


def test_manager_unregister_unknown_is_noop():
    """Unregistering an unknown name does not raise."""
    manager = MemoryPluginManager()
    manager.unregister("never-registered")


def test_manager_list_in_registration_order():
    manager = MemoryPluginManager()
    a = NullPlugin("a")
    b = NullPlugin("b")
    c = NullPlugin("c")
    manager.register(a)
    manager.register(b)
    manager.register(c)
    assert manager.list() == [a, b, c]


def test_manager_names():
    manager = MemoryPluginManager()
    manager.register(NullPlugin("a"))
    manager.register(ConsolePlugin("b"))
    assert manager.names() == ["a", "b"]


def test_manager_load_all_and_shutdown_all():
    """load_all()/shutdown_all() drive every plugin's lifecycle."""
    manager = MemoryPluginManager()
    p1 = NullPlugin("a")
    p2 = NullPlugin("b")
    manager.register(p1)
    manager.register(p2)
    assert p1.health_check() is False
    manager.load_all()
    assert p1.health_check() is True
    assert p2.health_check() is True
    manager.shutdown_all()
    assert p1.health_check() is False
    assert p2.health_check() is False


def test_manager_empty_registry_helpers_noop():
    """load_all()/shutdown_all()/names() are no-ops on an empty registry."""
    manager = MemoryPluginManager()
    manager.load_all()
    manager.shutdown_all()
    assert manager.names() == []
    assert manager.list() == []


# ======================================================================
# 5. Registration validation
# ======================================================================

def test_manager_register_non_plugin_raises():
    manager = MemoryPluginManager()
    with raises(PluginError, match="PluginInterface"):
        manager.register(object())


def test_manager_register_duplicate_raises():
    manager = MemoryPluginManager()
    manager.register(NullPlugin("dup"))
    with raises(PluginError, match="already registered"):
        manager.register(NullPlugin("dup"))


def test_manager_register_after_unregister_ok():
    """Unregistering frees the name for re-registration."""
    manager = MemoryPluginManager()
    manager.register(NullPlugin("a"))
    manager.unregister("a")
    manager.register(NullPlugin("a"))
    assert manager.get("a") is not None


# ======================================================================
# 6. Full-path type annotations (D6 spec)
# ======================================================================

def test_run_annotations_use_full_target_state_path():
    """run() declares the snapshot as the full path
    'visioncore.state.target_state.TargetState'."""
    import visioncore.plugin.base as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert 'target_state: "visioncore.state.target_state.TargetState"' in src
    assert ') -> "visioncore.state.target_state.TargetState":' in src


def test_manager_annotations_use_full_plugin_paths():
    """Manager contract methods use full-path PluginInterface annotations."""
    import visioncore.plugin.base as mod
    src = open(mod.__file__, encoding="utf-8").read()
    assert 'plugin: "visioncore.plugin.base.PluginInterface"' in src
    assert '-> "visioncore.plugin.base.PluginInterface" | None:' in src
    assert 'list["visioncore.plugin.base.PluginInterface"]' in src


def test_plugin_run_works_with_state_target_state_type():
    """A plugin's run() accepts/returns visioncore.state.target_state.TargetState."""
    from visioncore.state.target_state import TargetState as Snapshot
    plugin = NullPlugin()
    ts = _snapshot(3)
    result = plugin.run(ts)
    assert isinstance(result, Snapshot)
    assert type(result).__module__ == "visioncore.state.target_state"


# ======================================================================
# 7. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """base.py contains no network / model / GUI / ai imports (AST check)."""
    import visioncore.plugin.base as mod
    src = open(mod.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    banned = ("socket", "requests", "rospy", "mavlink", "zmq",
              "torch", "onnx", "cv2", "ultralytics", "yolo",
              "ai", "gui", "camera")
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
    assert not violations, f"visioncore/plugin/base.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing the plugin layer does not load torch."""
    import visioncore.plugin  # noqa: F401
    assert "torch" not in sys.modules, "torch loaded as side effect"


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
    print(f"Running {len(tests)} plugin interface tests...\n")
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
    print("All plugin interface tests passed.")
