"""Plugin registry unit tests (Milestone D7).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary.

Covers:
    - Package surface (PluginRegistry inherits PluginManager)
    - Construction (initially empty)
    - Register / get_plugin / list / unregister / contains / names
    - Registration validation (non-plugin rejected, duplicates rejected)
    - load_package: dotted-path module scanning discovers instances
    - load_entrypoints: no entries gracefully returns empty list
    - Manager convenience (load_all / shutdown_all)
    - No network / model / GUI / ai dependency (AST + runtime)
"""
from __future__ import annotations

import ast
import sys
import traceback
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
    PluginRegistry,
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


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_registry_is_plugin_manager_subclass():
    assert issubclass(PluginRegistry, PluginManager)


def test_registry_importable_from_package():
    from visioncore.plugin import PluginRegistry as PR
    assert PR is PluginRegistry


# ======================================================================
# 1. Construction
# ======================================================================

def test_registry_construction_default():
    registry = PluginRegistry()
    assert isinstance(registry, PluginManager)
    assert isinstance(registry, PluginRegistry)


def test_registry_initially_empty():
    registry = PluginRegistry()
    assert len(registry) == 0
    assert registry.names() == []
    assert registry.list() == []


# ======================================================================
# 2. Register / get_plugin / list
# ======================================================================

def test_registry_register_then_get_plugin():
    registry = PluginRegistry()
    plugin = NullPlugin("null")
    registry.register(plugin)
    assert registry.get_plugin("null") is plugin


def test_registry_get_plugin_unknown_returns_none():
    registry = PluginRegistry()
    assert registry.get_plugin("missing") is None


def test_registry_register_multiple():
    registry = PluginRegistry()
    a = NullPlugin("a")
    b = NullPlugin("b")
    c = ConsolePlugin("c")
    registry.register(a)
    registry.register(b)
    registry.register(c)
    assert len(registry) == 3
    assert registry.get_plugin("a") is a
    assert registry.get_plugin("b") is b
    assert registry.get_plugin("c") is c


def test_registry_list_order():
    registry = PluginRegistry()
    a = NullPlugin("a")
    b = NullPlugin("b")
    c = NullPlugin("c")
    registry.register(a)
    registry.register(b)
    registry.register(c)
    assert registry.list() == [a, b, c]


# ======================================================================
# 3. Unregister
# ======================================================================

def test_registry_unregister_removes():
    registry = PluginRegistry()
    registry.register(NullPlugin("a"))
    registry.unregister("a")
    assert registry.get_plugin("a") is None
    assert len(registry) == 0


def test_registry_unregister_unknown_is_noop():
    registry = PluginRegistry()
    registry.unregister("never-registered")


def test_registry_register_after_unregister_ok():
    registry = PluginRegistry()
    registry.register(NullPlugin("a"))
    registry.unregister("a")
    registry.register(NullPlugin("a"))
    assert registry.get_plugin("a") is not None


# ======================================================================
# 4. Manager convenience (inherited)
# ======================================================================

def test_registry_names():
    registry = PluginRegistry()
    registry.register(NullPlugin("a"))
    registry.register(ConsolePlugin("b"))
    assert registry.names() == ["a", "b"]


def test_registry_contains():
    registry = PluginRegistry()
    registry.register(NullPlugin("a"))
    assert "a" in registry
    assert "missing" not in registry


def test_registry_load_all_shutdown_all():
    registry = PluginRegistry()
    p1 = NullPlugin("a")
    p2 = NullPlugin("b")
    registry.register(p1)
    registry.register(p2)
    assert p1.health_check() is False
    registry.load_all()
    assert p1.health_check() is True
    assert p2.health_check() is True
    registry.shutdown_all()
    assert p1.health_check() is False
    assert p2.health_check() is False


# ======================================================================
# 5. Validation
# ======================================================================

def test_registry_register_non_plugin_raises():
    registry = PluginRegistry()
    with raises(PluginError, match="PluginInterface"):
        registry.register(object())


def test_registry_register_duplicate_raises():
    registry = PluginRegistry()
    registry.register(NullPlugin("dup"))
    with raises(PluginError, match="already registered"):
        registry.register(NullPlugin("dup"))


# ======================================================================
# 6. load_package (dotted-path module scanning)
# ======================================================================

def test_registry_load_package_real_plugin():
    """load_package discovers instances from a real module."""
    registry = PluginRegistry()
    loaded = registry.load_package("tests._plugin_fixtures")
    names = [p.name for p in loaded]
    assert "alpha" in names
    assert "beta" in names
    assert registry.get_plugin("alpha").name == "alpha"
    assert registry.get_plugin("beta").name == "beta"


def test_registry_load_package_no_plugins():
    """load_package on a module with no PluginInterface instances returns []."""
    registry = PluginRegistry()
    loaded = registry.load_package("os")  # stdlib module -- no plugins
    assert loaded == []
    assert len(registry) == 0


def test_registry_load_package_nonexistent_raises():
    """load_package on a missing module raises PluginError."""
    registry = PluginRegistry()
    with raises(PluginError, match="failed to import"):
        registry.load_package("nonexistent.fake.module.path")


def test_registry_load_package_skips_duplicates():
    """Second load_package on the same module skips already-registered names."""
    registry = PluginRegistry()
    registry.load_package("tests._plugin_fixtures")
    loaded2 = registry.load_package("tests._plugin_fixtures")
    assert loaded2 == []
    assert len(registry) == 2


# ======================================================================
# 7. load_entrypoints (discovery)
# ======================================================================

def test_registry_load_entrypoints_no_entries():
    """load_entrypoints on a non-existent group returns [] -- no exception."""
    registry = PluginRegistry()
    loaded = registry.load_entrypoints("visioncore.plugin.test.nonexistent")
    assert loaded == []
    assert len(registry) == 0


def test_registry_load_entrypoints_importlib_available():
    """importlib.metadata.entry_points is available on this Python."""
    from importlib.metadata import entry_points
    assert callable(entry_points)


# ======================================================================
# 8. No network / model / GUI / ai dependency
# ======================================================================

def test_no_network_model_gui_imports_in_source():
    """registry.py contains no network / model / GUI / ai imports (AST check)."""
    import visioncore.plugin.registry as mod
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
    assert not violations, f"registry.py imports banned: {violations}"


def test_no_torch_loaded_at_runtime():
    """Importing the plugin registry does not load torch."""
    import visioncore.plugin.registry  # noqa: F401
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
    print(f"Running {len(tests)} plugin registry tests...\n")
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
    print("All plugin registry tests passed.")
