"""FrameSource unit tests (Milestone C2).

Functional test suite following project convention: no pytest dependency,
no test classes, each test is a function, and a ``__main__`` runner at the
bottom calls them all and prints a pass/fail summary. A tiny ``raises``
context manager replaces ``pytest.raises`` for exception assertions.

Covers the contractually-required areas:

    1. Lifecycle          -- open / close / is_open idempotency + ordering
    2. read()             -- Frame|None return, fixed output, exhaustion
    3. close()            -- idempotent, never raises
    4. Exception recovery -- raise_on_read, close+open recovery, closed-read raises

Plus:
    - Package exports + import surface
    - FrameSource ABC contract (cannot instantiate, subclass must implement)
    - health_check semantics (open vs closed, non-raising)
    - Context manager (guaranteed teardown on exception)
    - Frame type enforcement (read returns Frame, never bare ndarray)
    - Factory / registry (register / create / available / overwrite)
    - DummyFrameSource configuration (frames, loop, source_id, counters, reset)
"""
from __future__ import annotations

import traceback
from typing import Any

from visioncore.core.frame import Frame
from visioncore.source import (
    DummyFrameSource,
    FrameSource,
    FrameSourceError,
    available_frame_sources,
    create_frame_source,
    get_frame_source_class,
    register_frame_source,
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
        return True  # suppress the caught exception


def raises(
    expected: type[BaseException], match: str | None = None,
) -> _Raises:
    """Assert that the enclosed block raises ``expected``."""
    return _Raises(expected, match=match)


# ======================================================================
# 0. Package surface / imports
# ======================================================================

def test_package_exports_public_api():
    """The package __all__ exposes the documented names."""
    import visioncore.source as pkg
    assert set(pkg.__all__) == {
        "FrameSource", "FrameSourceError", "DummyFrameSource",
        "create_frame_source", "register_frame_source",
        "available_frame_sources", "get_frame_source_class",
    }


def test_frame_source_is_abstract():
    """FrameSource is an ABC and cannot be instantiated directly."""
    with raises(TypeError):
        FrameSource()  # type: ignore[abstract]


def test_frame_source_error_is_runtime_error_subclass():
    """FrameSourceError is a RuntimeError subclass."""
    assert issubclass(FrameSourceError, RuntimeError)


def test_subclass_missing_methods_fails():
    """A subclass that omits any abstract method cannot be instantiated."""
    class Incomplete(FrameSource):
        def open(self) -> None:
            pass
        # is_open / read / close / health_check missing
    with raises(TypeError):
        Incomplete()  # type: ignore[abstract]


# ======================================================================
# 1. DummyFrameSource construction & identity
# ======================================================================

def test_dummy_default_source_id():
    """A DummyFrameSource with no source_id defaults to 'dummy'."""
    src = DummyFrameSource()
    assert src.source_id == "dummy"


def test_dummy_explicit_source_id():
    """An explicit source_id is honoured."""
    src = DummyFrameSource(source_id="cam0")
    assert src.source_id == "cam0"


def test_dummy_default_loop_is_true():
    """loop defaults to True (infinite replay)."""
    src = DummyFrameSource()
    assert src.loop is True


def test_dummy_default_counters_are_zero_and_closed():
    """A fresh source has zero counters and is closed/unhealthy."""
    src = DummyFrameSource()
    assert src.open_count == 0
    assert src.read_count == 0
    assert src.close_count == 0
    assert src.is_open() is False
    assert src.health_check() is False


def test_dummy_auto_generates_one_frame():
    """With no frames supplied, exactly one synthetic frame is generated."""
    src = DummyFrameSource()
    assert src.frame_buffer_size == 1


def test_dummy_explicit_frames_preserved():
    """Explicitly supplied frames are stored in order."""
    f1 = Frame(1, 0.0, "s", image=__import__("numpy").zeros((2, 2, 3)))
    f2 = Frame(2, 1.0, "s", image=__import__("numpy").zeros((2, 2, 3)))
    src = DummyFrameSource(frames=[f1, f2])
    assert src.frame_buffer_size == 2


# ======================================================================
# === REQUIRED AREA 1: LIFECYCLE =======================================
# ======================================================================

def test_open_makes_source_open_and_healthy():
    """open() flips is_open and health_check to True."""
    src = DummyFrameSource()
    src.open()
    assert src.is_open() is True
    assert src.health_check() is True
    assert src.open_count == 1


def test_open_is_idempotent():
    """Calling open() twice is allowed; the source stays open."""
    src = DummyFrameSource()
    src.open()
    src.open()
    assert src.is_open() is True
    assert src.open_count == 2  # counter tracks calls; state unchanged


def test_close_makes_source_closed_and_unhealthy():
    """close() flips is_open and health_check to False."""
    src = DummyFrameSource()
    src.open()
    src.close()
    assert src.is_open() is False
    assert src.health_check() is False
    assert src.close_count == 1


def test_close_is_idempotent():
    """Calling close() twice is allowed; the source stays closed."""
    src = DummyFrameSource()
    src.open()
    src.close()
    src.close()
    assert src.is_open() is False
    assert src.close_count == 2


def test_close_without_open_is_safe():
    """close() on a never-opened source is a no-op (does not raise)."""
    src = DummyFrameSource()
    src.close()  # must not raise
    assert src.is_open() is False


def test_repr_reports_open_state():
    """__repr__ reports the source_id and open state."""
    src = DummyFrameSource(source_id="cam0")
    assert "DummyFrameSource" in repr(src)
    assert "source_id='cam0'" in repr(src)
    assert "open=False" in repr(src)
    src.open()
    assert "open=True" in repr(src)


def test_repr_survives_is_open_exception():
    """A buggy is_open() that raises does not crash __repr__."""
    class BrokenIsOpen(FrameSource):
        def open(self) -> None:
            pass
        def is_open(self) -> bool:
            raise RuntimeError("boom")
        def read(self):
            return None
        def close(self) -> None:
            pass
        def health_check(self) -> bool:
            return True

    src = BrokenIsOpen("broken")
    r = repr(src)  # must not raise
    assert "open=False" in r


# ======================================================================
# === REQUIRED AREA 2: read() ==========================================
# ======================================================================

def test_read_returns_frame_when_open():
    """read() on an open source returns a Frame (not None)."""
    src = DummyFrameSource(source_id="cam0")
    src.open()
    f = src.read()
    assert f is not None
    assert isinstance(f, Frame)


def test_read_returns_frame_not_ndarray():
    """read() returns a Frame envelope, never a bare numpy ndarray."""
    import numpy as np
    src = DummyFrameSource()
    src.open()
    f = src.read()
    assert isinstance(f, Frame)  # the envelope
    assert not isinstance(f, np.ndarray)  # never a bare array
    # The Frame's image field DOES hold an ndarray internally -- that is
    # the data model's design, and is allowed. The prohibition is on
    # returning a bare array as the read() result, not on the Frame
    # containing one.
    assert isinstance(f.image, np.ndarray)


def test_read_stamps_source_id_on_auto_frame():
    """An auto-generated frame carries the source's source_id."""
    src = DummyFrameSource(source_id="cam0")
    src.open()
    f = src.read()
    assert f is not None
    assert f.source_id == "cam0"


def test_read_returns_fixed_frame_by_default():
    """With one frame and loop=True, read() returns the same Frame each time."""
    src = DummyFrameSource()
    src.open()
    f1 = src.read()
    f2 = src.read()
    f3 = src.read()
    assert f1 is f2 is f3  # same instance (fixed output, looped)


def test_read_increments_counter():
    """read_count tracks the number of read() calls."""
    src = DummyFrameSource()
    src.open()
    src.read()
    src.read()
    assert src.read_count == 2


def test_read_returns_none_when_exhausted_no_loop():
    """With loop=False, read() returns None after the buffer is drained."""
    import numpy as np
    single = Frame(7, 1.0, "s", np.zeros((2, 2, 3)))
    src = DummyFrameSource(frames=[single], loop=False)
    src.open()
    f1 = src.read()
    f2 = src.read()
    assert f1 is single
    assert f2 is None  # exhausted


def test_read_loops_through_multiple_frames():
    """With multiple frames and loop=True, read() cycles through them."""
    import numpy as np
    frames = [Frame(i, float(i), "s", np.zeros((2, 2, 3))) for i in range(3)]
    src = DummyFrameSource(frames=frames, loop=True)
    src.open()
    seq = [src.read() for _ in range(7)]
    # Cycle: 0,1,2,0,1,2,0
    assert [f.frame_id for f in seq] == [0, 1, 2, 0, 1, 2, 0]


def test_read_replays_explicit_frames_in_order():
    """Explicit frames are replayed in the supplied order."""
    import numpy as np
    frames = [Frame(10, 0.0, "x", np.zeros((2, 2, 3))),
              Frame(11, 1.0, "x", np.zeros((2, 2, 3)))]
    src = DummyFrameSource(frames=frames, loop=False)
    src.open()
    assert src.read().frame_id == 10
    assert src.read().frame_id == 11
    assert src.read() is None


# ======================================================================
# === REQUIRED AREA 3: close() =========================================
# ======================================================================

def test_close_never_raises_even_after_failure():
    """close() must not raise even if the source is in a failed state."""
    src = DummyFrameSource()
    src.open()
    src.raise_on_read = FrameSourceError("device dropped")
    try:
        src.read()
    except FrameSourceError:
        pass
    # close() must succeed despite the failed read
    src.close()
    assert src.is_open() is False


def test_close_after_close_does_not_revive_source():
    """A second close() keeps the source closed (does not reopen)."""
    src = DummyFrameSource()
    src.open()
    src.close()
    src.close()
    assert src.is_open() is False
    assert src.health_check() is False


def test_close_resets_health():
    """close() flips health_check to False."""
    src = DummyFrameSource()
    src.open()
    assert src.health_check() is True
    src.close()
    assert src.health_check() is False


# ======================================================================
# === REQUIRED AREA 4: EXCEPTION RECOVERY ==============================
# ======================================================================

def test_read_on_closed_source_raises_runtime_error():
    """read() on a closed source raises RuntimeError (contract violation)."""
    src = DummyFrameSource()
    # never opened
    with raises(RuntimeError, match="closed source"):
        src.read()
    # opened then closed
    src.open()
    src.close()
    with raises(RuntimeError, match="closed source"):
        src.read()


def test_raise_on_read_propagates_frame_source_error():
    """raise_on_read=FrameSourceError propagates out of read()."""
    src = DummyFrameSource()
    src.open()
    src.raise_on_read = FrameSourceError("stream timed out")
    with raises(FrameSourceError, match="stream timed out"):
        src.read()
    assert src.read_count == 1  # counter incremented before raising


def test_raise_on_read_propagates_arbitrary_exception():
    """raise_on_read accepts any exception type."""
    src = DummyFrameSource()
    src.open()
    src.raise_on_read = ValueError("bad payload")
    with raises(ValueError, match="bad payload"):
        src.read()


def test_recovery_via_close_and_open():
    """After a failure, close()+open() recovers the source; read() works again.

    This is the core exception-recovery pattern: a transient source failure
    is cleared, the source is recycled, and reads succeed again.
    """
    src = DummyFrameSource(source_id="cam0")
    src.open()
    # Simulate a transient failure
    src.raise_on_read = FrameSourceError("device dropped")
    with raises(FrameSourceError):
        src.read()
    # Clear the failure and recycle the source
    src.raise_on_read = None
    src.close()
    assert src.is_open() is False
    src.open()
    assert src.is_open() is True
    # Recovery: read() now succeeds
    f = src.read()
    assert f is not None
    assert f.source_id == "cam0"
    assert src.read_count == 2  # one failed + one successful


def test_raise_on_read_persists_until_cleared():
    """raise_on_read raises on every read until explicitly cleared."""
    src = DummyFrameSource()
    src.open()
    src.raise_on_read = RuntimeError("persistent fault")
    with raises(RuntimeError):
        src.read()
    with raises(RuntimeError):
        src.read()  # still failing
    assert src.read_count == 2
    src.raise_on_read = None  # clear
    assert src.read() is not None  # now works


def test_read_failure_does_not_corrupt_frame_buffer():
    """A failed read does not advance the read index or drop frames."""
    import numpy as np
    frames = [Frame(0, 0.0, "s", np.zeros((2, 2, 3))),
              Frame(1, 1.0, "s", np.zeros((2, 2, 3)))]
    src = DummyFrameSource(frames=frames, loop=False)
    src.open()
    assert src.read().frame_id == 0  # consume frame 0
    src.raise_on_read = RuntimeError("transient")
    with raises(RuntimeError):
        src.read()  # fails before consuming frame 1
    src.raise_on_read = None
    assert src.read().frame_id == 1  # frame 1 still there, in order


# ======================================================================
# 5. health_check semantics
# ======================================================================

def test_health_check_false_before_open():
    """health_check() is False on a fresh (unopened) source."""
    src = DummyFrameSource()
    assert src.health_check() is False


def test_health_check_true_after_open():
    """health_check() is True after open()."""
    src = DummyFrameSource()
    src.open()
    assert src.health_check() is True


def test_health_check_false_after_close():
    """health_check() is False after close()."""
    src = DummyFrameSource()
    src.open()
    src.close()
    assert src.health_check() is False


# ======================================================================
# 6. Context manager
# ======================================================================

def test_context_manager_opens_and_closes():
    """'with source' opens on entry and closes on exit."""
    src = DummyFrameSource()
    assert src.is_open() is False
    with src:
        assert src.is_open() is True
    assert src.is_open() is False
    assert src.open_count == 1
    assert src.close_count == 1


def test_context_manager_closes_on_exception():
    """close() runs even if the with-body raises."""
    src = DummyFrameSource()
    with raises(ValueError):
        with src:
            raise ValueError("body exploded")
    assert src.is_open() is False  # close() still ran
    assert src.close_count == 1


def test_context_manager_does_not_suppress_exception():
    """The with-body exception propagates after close()."""
    src = DummyFrameSource()
    caught = False
    try:
        with src:
            raise ValueError("propagate me")
    except ValueError as e:
        caught = True
        assert "propagate me" in str(e)
    assert caught


def test_context_manager_read_inside():
    """read() works inside the with-block and returns a Frame."""
    src = DummyFrameSource(source_id="cam0")
    with src:
        f = src.read()
    assert f is not None
    assert isinstance(f, Frame)
    assert f.source_id == "cam0"


# ======================================================================
# 7. Factory / registry
# ======================================================================

def test_dummy_is_registered_as_dummy_kind():
    """The 'dummy' kind is registered to DummyFrameSource."""
    assert get_frame_source_class("dummy") is DummyFrameSource
    assert "dummy" in available_frame_sources()


def test_create_frame_source_dummy():
    """create_frame_source('dummy', ...) builds a DummyFrameSource."""
    src = create_frame_source("dummy", source_id="cam0")
    assert isinstance(src, DummyFrameSource)
    assert src.source_id == "cam0"


def test_create_frame_source_unknown_kind_raises():
    """An unknown kind raises ValueError listing available kinds."""
    with raises(ValueError, match="unknown frame source kind"):
        create_frame_source("nonexistent")


def test_create_frame_source_end_to_end():
    """A source built via the factory reads a Frame end-to-end."""
    src = create_frame_source("dummy", source_id="cam0")
    with src:
        f = src.read()
    assert f is not None
    assert f.source_id == "cam0"


def test_register_rejects_empty_name():
    """register_frame_source rejects an empty name."""
    with raises(ValueError, match="non-empty"):
        register_frame_source("", DummyFrameSource)


def test_register_rejects_non_subclass():
    """register_frame_source rejects a non-FrameSource class."""
    with raises(ValueError, match="FrameSource subclass"):
        register_frame_source("bad", object)  # type: ignore[arg-type]


def test_register_rejects_abstract_base_itself():
    """register_frame_source rejects the abstract FrameSource itself."""
    with raises(ValueError, match="abstract"):
        register_frame_source("base", FrameSource)


def test_register_refuses_shadowing_different_class():
    """Re-registering a name with a DIFFERENT class raises (no overwrite)."""
    class OtherDummy(FrameSource):
        def open(self) -> None:
            pass
        def is_open(self) -> bool:
            return False
        def read(self):
            return None
        def close(self) -> None:
            pass
        def health_check(self) -> bool:
            return False
    # 'dummy' is already registered to DummyFrameSource
    with raises(ValueError, match="already registered"):
        register_frame_source("dummy", OtherDummy)


def test_register_same_class_is_idempotent():
    """Re-registering the same (name, class) pair is a no-op."""
    register_frame_source("dummy", DummyFrameSource)  # no raise
    assert get_frame_source_class("dummy") is DummyFrameSource


def test_register_overwrite_replaces():
    """overwrite=True replaces an existing registration."""
    class Replacement(FrameSource):
        def open(self) -> None:
            pass
        def is_open(self) -> bool:
            return False
        def read(self):
            return None
        def close(self) -> None:
            pass
        def health_check(self) -> bool:
            return False
    register_frame_source("dummy", Replacement, overwrite=True)
    assert get_frame_source_class("dummy") is Replacement
    # Restore the original so other tests are not affected
    register_frame_source("dummy", DummyFrameSource, overwrite=True)
    assert get_frame_source_class("dummy") is DummyFrameSource


# ======================================================================
# 8. reset() and misc
# ======================================================================

def test_reset_clears_counters_and_state():
    """reset() zeroes counters, rewinds the index, and closes the source."""
    src = DummyFrameSource()
    src.open()
    src.read()
    src.read()
    src.close()
    src.reset()
    assert src.open_count == 0
    assert src.read_count == 0
    assert src.close_count == 0
    assert src.is_open() is False
    assert src.health_check() is False


def test_reset_preserves_frame_buffer_and_loop():
    """reset() does not alter the frame buffer or loop flag."""
    import numpy as np
    frames = [Frame(0, 0.0, "s", np.zeros((2, 2, 3)))]
    src = DummyFrameSource(frames=frames, loop=False)
    src.reset()
    assert src.frame_buffer_size == 1
    assert src.loop is False


def test_explicit_frames_accept_non_array_image():
    """The source does not inspect/require the image to be an ndarray.

    The source contract is to return a Frame envelope; it does not enforce
    that ``Frame.image`` is a numpy array (that is the data model's concern,
    not the source's). A Frame built with a plain-object image is replayed
    unchanged. This confirms the source is image-agnostic.
    """
    fake_frame = Frame(frame_id=0, timestamp=0.0, source_id="test", image="not-an-array")
    src = DummyFrameSource(frames=[fake_frame])
    src.open()
    f = src.read()
    assert f is fake_frame
    assert f.image == "not-an-array"


# ======================================================================
# 9. End-to-end: source feeds a Frame into a pipeline-like consumer
# ======================================================================

def test_end_to_end_source_feeds_consumer():
    """A DummyFrameSource yields Frames that a consumer reads in a loop."""
    src = create_frame_source("dummy", source_id="cam0")
    collected: list[Frame] = []
    with src:
        for _ in range(5):
            f = src.read()
            if f is not None:
                collected.append(f)
    assert len(collected) == 5
    assert all(isinstance(f, Frame) for f in collected)
    assert all(f.source_id == "cam0" for f in collected)
    assert src.read_count == 5
    assert src.is_open() is False  # closed by context manager


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
    print(f"Running {len(tests)} frame source tests...\n")
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
    print("All frame source tests passed.")
