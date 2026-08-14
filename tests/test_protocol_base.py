"""Unit tests for visioncore.protocol.

Verifies the ProtocolAdapter ABC contract, NullAdapter and ConsoleAdapter
concrete implementations, the default publish_many behaviour, the context
manager protocol, type enforcement, and the mandatory type-annotation
convention (from visioncore.state.target_state import TargetState).

Run::

    python -m pytest tests/test_protocol_base.py -v
    # or
    PYTHONPATH=F:/VisionBata python tests/test_protocol_base.py
"""

from __future__ import annotations

import io
import logging
from abc import ABC

from visioncore.protocol import ConsoleAdapter, NullAdapter, ProtocolAdapter
from visioncore.protocol.base import ProtocolAdapter as ProtocolAdapterDirect
from visioncore.protocol.console_adapter import ConsoleAdapter as ConsoleAdapterDirect
from visioncore.protocol.null_adapter import NullAdapter as NullAdapterDirect
from visioncore.state.target_state import TargetState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(**overrides) -> TargetState:
    """Build a TargetState with sensible defaults + per-test overrides."""
    defaults: dict = dict(
        target_id=42,
        local_id=1,
        global_id=None,
        label="person",
        confidence=0.92,
        cx=0.5,
        cy=0.4,
        vx=0.01,
        vy=-0.002,
        width=0.12,
        height=0.30,
        timestamp=12.5,
        camera_id=0,
        metadata={},
    )
    defaults.update(overrides)
    return TargetState(**defaults)


def _make_states(n: int = 3) -> list[TargetState]:
    """Build a list of n distinct TargetState instances."""
    return [
        _make_state(target_id=i, cx=0.1 * i, timestamp=1.0 * i)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Type annotation convention (mandatory per B2 spec)
# ---------------------------------------------------------------------------

def test_type_annotation_convention_no_top_level_import():
    """No module in visioncore.protocol may use 'from visioncore import TargetState'.

    The top-level visioncore package re-exports the TargetState *enum*
    (lifecycle), not the *snapshot dataclass*. All protocol modules must
    use the fully-qualified path: from visioncore.state.target_state
    import TargetState.

    This test parses each module's AST and checks actual ``ImportFrom``
    nodes -- docstrings that *mention* the forbidden pattern (e.g. while
    explaining why it should not be used) are not flagged.
    """
    import ast
    import inspect

    modules_to_check = [
        ProtocolAdapter.__module__,
        NullAdapter.__module__,
        ConsoleAdapter.__module__,
        "visioncore.protocol",
    ]

    for mod_name in modules_to_check:
        mod = __import__(mod_name, fromlist=["_"])
        src = inspect.getsource(mod)
        tree = ast.parse(src)

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            # 'from visioncore import TargetState' has module='visioncore'
            # and names containing 'TargetState' (no `.module` qualifier).
            if node.module == "visioncore":
                for alias in node.names:
                    if alias.name == "TargetState":
                        raise AssertionError(
                            f"{mod_name} line {node.lineno}: forbidden "
                            f"'from visioncore import TargetState'. Use "
                            f"'from visioncore.state.target_state import "
                            f"TargetState' instead."
                        )


def test_protocol_uses_snapshot_not_enum():
    """ProtocolAdapter.publish must accept the snapshot dataclass, not the enum.

    This guards against a regression where someone changes the import to
    'from visioncore import TargetState' (the enum) -- the snapshot and
    the enum are different types.
    """
    from visioncore.core import TargetState as TargetLifecycleEnum

    # The snapshot is a dataclass; the enum is not.
    import dataclasses
    assert dataclasses.is_dataclass(TargetState)
    assert not dataclasses.is_dataclass(TargetLifecycleEnum)
    assert TargetState is not TargetLifecycleEnum

    # NullAdapter.publish must accept the snapshot.
    adapter = NullAdapter()
    adapter.connect()
    adapter.publish(_make_state())  # no error
    adapter.disconnect()

    # NullAdapter.publish must reject the enum.
    try:
        adapter.connect()
        adapter.publish(TargetLifecycleEnum.ACTIVE)  # type: ignore[arg-type]
        raise AssertionError(
            "publish should reject the TargetState enum (only the "
            "snapshot dataclass is accepted)"
        )
    except TypeError:
        pass
    finally:
        adapter.disconnect()


# ---------------------------------------------------------------------------
# Package export sanity
# ---------------------------------------------------------------------------

def test_package_exports_three_types():
    """visioncore.protocol exports ProtocolAdapter, NullAdapter, ConsoleAdapter."""
    import visioncore.protocol as proto
    exported = set(proto.__all__)
    required = {"ProtocolAdapter", "NullAdapter", "ConsoleAdapter"}
    assert required.issubset(exported), f"Missing: {required - exported}"


def test_direct_and_package_imports_are_same():
    """Direct module imports == package-level imports."""
    assert ProtocolAdapter is ProtocolAdapterDirect
    assert NullAdapter is NullAdapterDirect
    assert ConsoleAdapter is ConsoleAdapterDirect


# ---------------------------------------------------------------------------
# ProtocolAdapter ABC contract
# ---------------------------------------------------------------------------

def test_protocol_adapter_is_abc():
    """ProtocolAdapter is an ABC and cannot be instantiated directly."""
    assert issubclass(ProtocolAdapter, ABC)
    try:
        ProtocolAdapter()  # type: ignore[abstract]
        raise AssertionError("ProtocolAdapter should be abstract")
    except TypeError:
        pass


def test_abstract_methods_defined():
    """ProtocolAdapter declares connect/disconnect/publish/health_check abstract."""
    # __abstractmethods__ is populated by ABCMeta for methods marked @abstractmethod.
    abstract = ProtocolAdapter.__abstractmethods__
    expected = {"connect", "disconnect", "publish", "health_check"}
    assert expected.issubset(abstract), (
        f"Missing abstract methods: {expected - abstract}"
    )


def test_publish_many_is_concrete_default():
    """publish_many is NOT abstract -- it has a default loop implementation."""
    abstract = ProtocolAdapter.__abstractmethods__
    assert "publish_many" not in abstract, (
        "publish_many should be concrete (default implementation loops "
        "publish). It must not be abstract."
    )


def test_context_manager_methods_exist():
    """ProtocolAdapter implements __enter__ / __exit__."""
    assert hasattr(ProtocolAdapter, "__enter__")
    assert hasattr(ProtocolAdapter, "__exit__")


# ---------------------------------------------------------------------------
# Subclass contract enforcement
# ---------------------------------------------------------------------------

def test_subclass_missing_method_fails():
    """A subclass missing any abstract method cannot be instantiated."""

    # Missing publish.
    class MissingPublish(ProtocolAdapter):
        def connect(self): pass
        def disconnect(self): pass
        def health_check(self): return True

    try:
        MissingPublish()  # type: ignore[abstract]
        raise AssertionError("MissingPublish should not be instantiable")
    except TypeError:
        pass

    # Missing health_check.
    class MissingHealth(ProtocolAdapter):
        def connect(self): pass
        def disconnect(self): pass
        def publish(self, state): pass

    try:
        MissingHealth()  # type: ignore[abstract]
        raise AssertionError("MissingHealth should not be instantiable")
    except TypeError:
        pass


def test_subclass_with_all_methods_works():
    """A subclass implementing all 4 abstract methods can use default publish_many."""

    class MinimalAdapter(ProtocolAdapter):
        def __init__(self):
            self._connected = False
            self.published: list[TargetState] = []
        def connect(self): self._connected = True
        def disconnect(self): self._connected = False
        def publish(self, state):
            if not self._connected:
                raise RuntimeError("not connected")
            self.published.append(state)
        def health_check(self): return self._connected

    adapter = MinimalAdapter()
    adapter.connect()
    states = _make_states(3)
    adapter.publish_many(states)
    assert adapter.published == states
    assert len(adapter.published) == 3


# ---------------------------------------------------------------------------
# NullAdapter
# ---------------------------------------------------------------------------

def test_null_adapter_is_protocol_adapter():
    """NullAdapter is a subclass of ProtocolAdapter."""
    assert issubclass(NullAdapter, ProtocolAdapter)


def test_null_adapter_lifecycle():
    """NullAdapter connect/disconnect are idempotent and toggle health_check."""
    a = NullAdapter()
    assert a.health_check() is False  # starts disconnected

    a.connect()
    assert a.health_check() is True

    # Idempotent connect.
    a.connect()
    assert a.health_check() is True

    a.disconnect()
    assert a.health_check() is False

    # Idempotent disconnect.
    a.disconnect()
    assert a.health_check() is False


def test_null_adapter_publish_when_disconnected_raises():
    """publish on a disconnected NullAdapter raises RuntimeError."""
    a = NullAdapter()
    try:
        a.publish(_make_state())
        raise AssertionError("publish should raise RuntimeError when disconnected")
    except RuntimeError as exc:
        assert "not connected" in str(exc).lower() or "connect" in str(exc).lower()


def test_null_adapter_publish_accepts_target_state():
    """publish on a connected NullAdapter accepts and discards a TargetState."""
    a = NullAdapter()
    a.connect()
    a.publish(_make_state())  # no error
    a.publish(_make_state(target_id=99, label="vehicle"))  # no error


def test_null_adapter_publish_rejects_non_target_state():
    """publish rejects None and non-TargetState values with TypeError."""
    a = NullAdapter()
    a.connect()
    bad_values = [None, "string", 42, 3.14, [1, 2], {"a": 1}, object()]
    for bad in bad_values:
        try:
            a.publish(bad)  # type: ignore[arg-type]
            raise AssertionError(
                f"publish should reject {bad!r} ({type(bad).__name__})"
            )
        except TypeError as exc:
            assert "TargetState" in str(exc), (
                f"TypeError message should mention TargetState for {bad!r}"
            )


def test_null_adapter_publish_many_preserves_order():
    """publish_many delivers states in order (even though they're discarded)."""
    a = NullAdapter()
    a.connect()
    states = _make_states(5)
    a.publish_many(states)  # no error


def test_null_adapter_publish_many_empty_sequence():
    """publish_many with an empty sequence is a no-op (and works when disconnected)."""
    a = NullAdapter()  # not connected
    a.publish_many([])  # no error -- empty batch doesn't require connection


def test_null_adapter_publish_many_when_disconnected_raises():
    """publish_many on a disconnected adapter raises on the first element."""
    a = NullAdapter()  # not connected
    try:
        a.publish_many(_make_states(2))
        raise AssertionError("publish_many should raise when disconnected")
    except RuntimeError:
        pass


def test_null_adapter_publish_many_rejects_non_target_state():
    """publish_many eagerly type-checks before publishing anything."""
    a = NullAdapter()
    a.connect()
    states: list = _make_states(2) + ["not a state"]  # type: ignore[list-item]
    try:
        a.publish_many(states)
        raise AssertionError("publish_many should reject non-TargetState element")
    except TypeError as exc:
        assert "states[2]" in str(exc) or "TargetState" in str(exc)


def test_null_adapter_context_manager():
    """NullAdapter supports 'with' -- connect on enter, disconnect on exit."""
    a = NullAdapter()
    assert a.health_check() is False
    with a:
        assert a.health_check() is True
        a.publish(_make_state())
    assert a.health_check() is False


def test_null_adapter_context_manager_disconnects_on_exception():
    """__exit__ calls disconnect even when an exception propagates."""
    a = NullAdapter()
    try:
        with a:
            assert a.health_check() is True
            raise ValueError("test error")
    except ValueError:
        pass
    assert a.health_check() is False


def test_null_adapter_context_manager_does_not_suppress_exceptions():
    """__exit__ returns None (falsy) so exceptions propagate."""
    a = NullAdapter()
    raised = False
    try:
        with a:
            raise ValueError("test")
    except ValueError:
        raised = True
    assert raised


def test_null_adapter_repr():
    """NullAdapter repr includes connection state."""
    a = NullAdapter()
    assert "NullAdapter" in repr(a)
    assert "connected=False" in repr(a)
    a.connect()
    assert "connected=True" in repr(a)


# ---------------------------------------------------------------------------
# ConsoleAdapter
# ---------------------------------------------------------------------------

def test_console_adapter_is_protocol_adapter():
    """ConsoleAdapter is a subclass of ProtocolAdapter."""
    assert issubclass(ConsoleAdapter, ProtocolAdapter)


def _make_console_adapter_with_capture() -> tuple[ConsoleAdapter, list[str]]:
    """Build a ConsoleAdapter whose logger output is captured to a list."""
    captured: list[str] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record.getMessage())

    log = logging.getLogger("test_console_adapter_capture")
    log.handlers.clear()
    log.addHandler(CaptureHandler())
    log.setLevel(logging.INFO)
    log.propagate = False

    adapter = ConsoleAdapter(logger=log)
    return adapter, captured


def test_console_adapter_lifecycle_logs():
    """connect/disconnect log INFO messages."""
    a, captured = _make_console_adapter_with_capture()
    a.connect()
    assert any("connected" in m for m in captured)
    a.disconnect()
    assert any("disconnected" in m for m in captured)


def test_console_adapter_idempotent_lifecycle():
    """Repeated connect/disconnect only logs once each."""
    a, captured = _make_console_adapter_with_capture()
    a.connect()
    a.connect()  # idempotent -- should not log again
    assert sum("connected" in m for m in captured) == 1
    a.disconnect()
    a.disconnect()  # idempotent
    assert sum("disconnected" in m for m in captured) == 1


def test_console_adapter_publish_logs_state():
    """publish logs the target_id, label, and key fields."""
    a, captured = _make_console_adapter_with_capture()
    a.connect()
    state = _make_state(target_id=77, label="vehicle", confidence=0.5)
    a.publish(state)
    publish_logs = [m for m in captured if "PUBLISH" in m]
    assert len(publish_logs) == 1
    msg = publish_logs[0]
    assert "id=77" in msg
    assert "label='vehicle'" in msg
    assert "conf=0.500" in msg


def test_console_adapter_publish_many_logs_each():
    """publish_many logs one PUBLISH line per state, in order."""
    a, captured = _make_console_adapter_with_capture()
    a.connect()
    states = _make_states(3)
    a.publish_many(states)
    publish_logs = [m for m in captured if "PUBLISH" in m]
    assert len(publish_logs) == 3
    # Order preserved: target_ids 0, 1, 2.
    assert "id=0" in publish_logs[0]
    assert "id=1" in publish_logs[1]
    assert "id=2" in publish_logs[2]


def test_console_adapter_publish_when_disconnected_raises():
    """publish on a disconnected ConsoleAdapter raises RuntimeError."""
    a, _ = _make_console_adapter_with_capture()
    try:
        a.publish(_make_state())
        raise AssertionError("publish should raise when disconnected")
    except RuntimeError:
        pass


def test_console_adapter_publish_rejects_non_target_state():
    """publish rejects non-TargetState values with TypeError."""
    a, _ = _make_console_adapter_with_capture()
    a.connect()
    for bad in [None, "string", 42, object()]:
        try:
            a.publish(bad)  # type: ignore[arg-type]
            raise AssertionError(f"publish should reject {bad!r}")
        except TypeError:
            pass


def test_console_adapter_health_check():
    """health_check reflects connection state."""
    a, _ = _make_console_adapter_with_capture()
    assert a.health_check() is False
    a.connect()
    assert a.health_check() is True
    a.disconnect()
    assert a.health_check() is False


def test_console_adapter_context_manager():
    """ConsoleAdapter supports 'with' and logs connect/disconnect."""
    a, captured = _make_console_adapter_with_capture()
    with a:
        a.publish(_make_state())
    assert any("connected" in m for m in captured)
    assert any("disconnected" in m for m in captured)
    assert a.health_check() is False


def test_console_adapter_default_logger():
    """ConsoleAdapter without explicit logger uses the module logger."""
    a = ConsoleAdapter()
    import visioncore.protocol.console_adapter as mod
    assert a._logger is mod._logger or a._logger.name == mod._logger.name


def test_console_adapter_repr():
    """ConsoleAdapter repr includes connection state."""
    a, _ = _make_console_adapter_with_capture()
    assert "ConsoleAdapter" in repr(a)
    assert "connected=False" in repr(a)
    a.connect()
    assert "connected=True" in repr(a)


# ---------------------------------------------------------------------------
# ProtocolAdapter.__repr__ (base class)
# ---------------------------------------------------------------------------

def test_base_repr_calls_health_check():
    """The base __repr__ calls health_check and includes the result."""
    a = NullAdapter()
    a.connect()
    r = repr(a)
    assert "NullAdapter" in r
    assert "connected=True" in r


def test_base_repr_survives_health_check_exception():
    """__repr__ does not crash if a buggy subclass's health_check raises."""
    class BrokenHealth(ProtocolAdapter):
        def connect(self): pass
        def disconnect(self): pass
        def publish(self, state): pass
        def health_check(self):
            raise RuntimeError("broken")
    a = BrokenHealth()
    # repr must not raise even though health_check does.
    r = repr(a)
    assert "BrokenHealth" in r
    assert "connected=False" in r  # falls back to False on exception


# ---------------------------------------------------------------------------
# Module entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import inspect
    import sys

    failures: list[str] = []
    passed = 0
    for name, obj in sorted(inspect.getmembers(sys.modules[__name__])):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"  [PASS] {name}")
                passed += 1
            except AssertionError as exc:
                print(f"  [FAIL] {name}: {exc}")
                failures.append(name)
            except Exception as exc:
                print(f"  [ERR ] {name}: {type(exc).__name__}: {exc}")
                failures.append(name)

    print()
    total = passed + len(failures)
    print(f"=== {passed}/{total} passed, {len(failures)} failed ===")
    if failures:
        print("Failed:", ", ".join(failures))
        sys.exit(1)
