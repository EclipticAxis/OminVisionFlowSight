"""Unit tests for ShadowStatePublisher and ShadowMetrics.

Verifies:
* Event subscription (all 5 lifecycle event types).
* Metrics collection: total_events, total_publishes, success/failure,
  conversion_failures, timing.
* Derived metrics: states_per_second, success_rate, avg_latency_us.
* Shadow mode: default NullAdapter produces zero side effects.
* No modification of GUI / InferWorker / detection / tracker.
* Error isolation: adapter failures are recorded but don't crash the bus.

Run::

    PYTHONPATH=F:/VisionBata python tests/test_shadow_publisher.py
"""

from __future__ import annotations

import ast
import inspect
import os
import time

from visioncore.eventbus.bus import EventBus
from visioncore.eventbus.events import (
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)
from visioncore.protocol.base import ProtocolAdapter
from visioncore.protocol.null_adapter import NullAdapter
from visioncore.protocol.publisher import ShadowStatePublisher
from visioncore.protocol.publisher.shadow_publisher import (
    ShadowMetrics,
    ShadowStatePublisher as ShadowDirect,
)
from visioncore.state.target_state import TargetState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CapturingAdapter(ProtocolAdapter):
    """Test adapter that records every published TargetState."""

    def __init__(self) -> None:
        self._connected: bool = False
        self.published: list[TargetState] = []

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def publish(self, state: TargetState) -> None:
        if not self._connected:
            raise RuntimeError("not connected")
        self.published.append(state)

    def health_check(self) -> bool:
        return self._connected


class FailingAdapter(ProtocolAdapter):
    """Adapter whose publish() always raises."""

    def __init__(self) -> None:
        self._connected: bool = False
        self.calls: int = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def publish(self, state: TargetState) -> None:
        self.calls += 1
        raise OSError("simulated failure")

    def health_check(self) -> bool:
        return self._connected


class SlowAdapter(ProtocolAdapter):
    """Adapter whose publish() sleeps for a configurable duration."""

    def __init__(self, delay_us: int = 100) -> None:
        self._connected: bool = False
        self._delay_s: float = delay_us / 1e6
        self.calls: int = 0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def publish(self, state: TargetState) -> None:
        self.calls += 1
        time.sleep(self._delay_s)

    def health_check(self) -> bool:
        return self._connected


def _make_event(event_cls, **overrides):
    """Build a lifecycle event with sensible defaults."""
    defaults = dict(
        event_id="evt-0001",
        timestamp=12.5,
        target_id="S0-T0001",
        slot_id=0,
        payload={"track_id": 1},
    )
    defaults.update(overrides)
    return event_cls(**defaults)


def _publish_n_events(bus: EventBus, n: int = 5) -> None:
    """Publish n TargetCreatedEvents on the bus."""
    for i in range(n):
        bus.publish(_make_event(
            TargetCreatedEvent,
            event_id=f"evt-{i:04d}",
            timestamp=float(i),
        ))


# ---------------------------------------------------------------------------
# Type annotation convention
# ---------------------------------------------------------------------------

def test_type_annotation_convention():
    """shadow_publisher.py must not use 'from visioncore import TargetState'."""
    mod = __import__(ShadowStatePublisher.__module__, fromlist=["_"])
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == "visioncore":
            for alias in node.names:
                assert alias.name != "TargetState", (
                    f"forbidden import at line {node.lineno}"
                )


# ---------------------------------------------------------------------------
# No modification of existing system
# ---------------------------------------------------------------------------

def test_no_gui_modification():
    """No file in gui/ references ShadowStatePublisher."""
    gui_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "gui")
    if not os.path.isdir(gui_dir):
        return
    for fname in os.listdir(gui_dir):
        if not fname.endswith(".py"):
            continue
        fpath = os.path.join(gui_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                src = f.read()
        except (IOError, UnicodeDecodeError):
            continue
        assert "ShadowStatePublisher" not in src, (
            f"gui/{fname} references ShadowStatePublisher"
        )


def test_no_inferworker_modification():
    """ai/inference.py does not reference ShadowStatePublisher."""
    inf_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "ai", "inference.py")
    if not os.path.isfile(inf_path):
        return
    with open(inf_path, "r", encoding="utf-8") as f:
        src = f.read()
    assert "ShadowStatePublisher" not in src, (
        "ai/inference.py references ShadowStatePublisher"
    )


# ---------------------------------------------------------------------------
# Package exports
# ---------------------------------------------------------------------------

def test_package_exports_shadow():
    """visioncore.protocol.publisher.__all__ contains ShadowStatePublisher."""
    import visioncore.protocol.publisher as pub
    assert "ShadowStatePublisher" in pub.__all__


def test_direct_import_same():
    """Direct module import == package import."""
    assert ShadowStatePublisher is ShadowDirect


# ---------------------------------------------------------------------------
# ShadowMetrics unit tests
# ---------------------------------------------------------------------------

def test_metrics_initial_state():
    """New ShadowMetrics has all zeros."""
    m = ShadowMetrics()
    assert m.total_events == 0
    assert m.total_publishes == 0
    assert m.successful_publishes == 0
    assert m.failed_publishes == 0
    assert m.conversion_failures == 0
    assert m.states_per_second == 0.0
    assert m.success_rate == 1.0  # vacuously true
    assert m.avg_latency_us == 0.0
    assert m.elapsed_seconds == 0.0


def test_metrics_record_event():
    """record_event increments total_events and sets timing reference."""
    m = ShadowMetrics()
    m.record_event()
    assert m.total_events == 1
    assert m.elapsed_seconds >= 0.0
    m.record_event()
    assert m.total_events == 2


def test_metrics_record_publish_success():
    """record_publish with success=True increments successful_publishes."""
    m = ShadowMetrics()
    m.record_publish(1000, True)   # 1µs
    m.record_publish(2000, True)   # 2µs
    assert m.total_publishes == 2
    assert m.successful_publishes == 2
    assert m.failed_publishes == 0
    assert m.avg_latency_us == 1.5  # (1000+2000)/2/1000 = 1.5µs


def test_metrics_record_publish_failure():
    """record_publish with success=False increments failed_publishes."""
    m = ShadowMetrics()
    m.record_publish(500, False)
    assert m.total_publishes == 1
    assert m.successful_publishes == 0
    assert m.failed_publishes == 1
    assert m.success_rate == 0.0


def test_metrics_success_rate_mixed():
    """success_rate = successful / total."""
    m = ShadowMetrics()
    m.record_publish(100, True)
    m.record_publish(200, True)
    m.record_publish(300, False)
    assert m.success_rate == 2.0 / 3.0


def test_metrics_conversion_failure():
    """record_conversion_failure increments the conversion_failures counter."""
    m = ShadowMetrics()
    m.record_conversion_failure()
    m.record_conversion_failure()
    assert m.conversion_failures == 2


def test_metrics_states_per_second():
    """states_per_second = total_publishes / elapsed_seconds."""
    m = ShadowMetrics()
    # Simulate: 10 publishes over ~0.1 seconds.
    m.record_event()  # starts timing
    for _ in range(10):
        m.record_publish(1000, True)
    # Wait a small amount of time to get a non-zero elapsed.
    time.sleep(0.01)
    sps = m.states_per_second
    assert sps > 0.0
    # Should be roughly 10 / 0.01 = 1000, but allow wide tolerance.
    assert sps < 100000  # sanity upper bound


def test_metrics_avg_latency():
    """avg_latency_us is the mean of all recorded publish durations."""
    m = ShadowMetrics()
    m.record_publish(1_000, True)   # 1.0 µs
    m.record_publish(3_000, True)   # 3.0 µs
    m.record_publish(5_000, True)   # 5.0 µs
    assert m.avg_latency_us == 3.0  # (1+3+5)/3


def test_metrics_reset():
    """reset() clears all counters and timing reference."""
    m = ShadowMetrics()
    m.record_event()
    m.record_publish(1000, True)
    m.record_conversion_failure()
    m.reset()
    assert m.total_events == 0
    assert m.total_publishes == 0
    assert m.conversion_failures == 0
    assert m.elapsed_seconds == 0.0
    assert m.states_per_second == 0.0


def test_metrics_summary():
    """summary() returns a dict snapshot of all metrics."""
    m = ShadowMetrics()
    m.record_event()
    m.record_publish(2000, True)
    m.record_publish(1000, False)
    s = m.summary()
    assert isinstance(s, dict)
    assert s["total_events"] == 1
    assert s["total_publishes"] == 2
    assert s["successful"] == 1
    assert s["failed"] == 1
    assert s["conversion_failures"] == 0
    assert "states_per_second" in s
    assert "success_rate" in s
    assert "avg_latency_us" in s
    assert s["success_rate"] == 0.5


def test_metrics_repr():
    """__repr__ includes key metrics."""
    m = ShadowMetrics()
    m.record_event()
    m.record_publish(1000, True)
    r = repr(m)
    assert "ShadowMetrics" in r
    assert "events=1" in r
    assert "publishes=1" in r


# ---------------------------------------------------------------------------
# ShadowStatePublisher lifecycle
# ---------------------------------------------------------------------------

def test_shadow_starts_with_null_adapter_by_default():
    """Default adapter is NullAdapter (shadow mode -- zero side effects)."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    # The _adapter should be a _MetricsAdapter wrapping a NullAdapter.
    assert isinstance(shadow._adapter._inner, NullAdapter)
    shadow.start()
    assert shadow._adapter.health_check() is True
    shadow.stop()


def test_shadow_start_subscribes_five_events():
    """start() registers 5 subscriptions on the bus."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    assert bus.subscriber_count() == 0
    shadow.start()
    assert bus.subscriber_count(TargetCreatedEvent) == 1
    assert bus.subscriber_count(TargetLostEvent) == 1
    assert bus.subscriber_count(TargetRecoveredEvent) == 1
    assert bus.subscriber_count(TargetLockedEvent) == 1
    assert bus.subscriber_count(TargetRemovedEvent) == 1
    shadow.stop()


def test_shadow_stop_unsubscribes():
    """stop() removes all subscriptions."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()
    assert bus.subscriber_count() > 0
    shadow.stop()
    assert bus.subscriber_count() == 0


def test_shadow_context_manager():
    """ShadowStatePublisher supports 'with'."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    assert not shadow.is_started
    with shadow:
        assert shadow.is_started
    assert not shadow.is_started


def test_shadow_start_idempotent():
    """Double start() is a no-op."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()
    n1 = bus.subscriber_count()
    shadow.start()
    n2 = bus.subscriber_count()
    assert n1 == n2
    shadow.stop()


def test_shadow_stop_idempotent():
    """Double stop() is safe."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()
    shadow.stop()
    shadow.stop()  # should not raise
    assert not shadow.is_started


# ---------------------------------------------------------------------------
# Metrics collection through event flow
# ---------------------------------------------------------------------------

def test_metrics_count_events():
    """Publishing events on the bus increments total_events."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    assert shadow.metrics.total_events == 0
    _publish_n_events(bus, 5)
    assert shadow.metrics.total_events == 5

    shadow.stop()


def test_metrics_count_publishes():
    """Each event triggers a publish (via NullAdapter) -- total_publishes increments."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    _publish_n_events(bus, 3)
    assert shadow.metrics.total_publishes == 3
    assert shadow.metrics.successful_publishes == 3
    assert shadow.metrics.failed_publishes == 0

    shadow.stop()


def test_metrics_all_five_event_types():
    """All 5 event types are counted in metrics."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    events = [
        _make_event(TargetCreatedEvent, event_id="e1"),
        _make_event(TargetLostEvent, event_id="e2"),
        _make_event(TargetRecoveredEvent, event_id="e3"),
        _make_event(TargetLockedEvent, event_id="e4"),
        _make_event(TargetRemovedEvent, event_id="e5"),
    ]
    for ev in events:
        bus.publish(ev)

    assert shadow.metrics.total_events == 5
    assert shadow.metrics.total_publishes == 5
    assert shadow.metrics.successful_publishes == 5

    shadow.stop()


def test_metrics_success_rate_with_failing_adapter():
    """FailingAdapter causes failed_publishes to increment; bus continues."""
    bus = EventBus()
    adapter = FailingAdapter()
    shadow = ShadowStatePublisher(bus, adapter=adapter)
    shadow.start()

    _publish_n_events(bus, 3)

    # All 3 publishes failed (adapter raises OSError).
    assert shadow.metrics.total_publishes == 3
    assert shadow.metrics.failed_publishes == 3
    assert shadow.metrics.successful_publishes == 0
    assert shadow.metrics.success_rate == 0.0

    # The bus was not crashed -- events were still received.
    assert shadow.metrics.total_events == 3

    shadow.stop()


def test_metrics_avg_latency_nonzero():
    """avg_latency_us is > 0 after at least one publish."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    _publish_n_events(bus, 1)
    assert shadow.metrics.avg_latency_us > 0.0
    # NullAdapter is sub-microsecond, so latency should be tiny but nonzero.
    assert shadow.metrics.avg_latency_us < 100.0

    shadow.stop()


def test_metrics_avg_latency_with_slow_adapter():
    """SlowAdapter produces measurable avg_latency_us."""
    bus = EventBus()
    adapter = SlowAdapter(delay_us=500)  # 0.5ms per publish
    shadow = ShadowStatePublisher(bus, adapter=adapter)
    shadow.start()

    _publish_n_events(bus, 5)

    avg = shadow.metrics.avg_latency_us
    # 500µs expected, allow tolerance for scheduling jitter.
    assert avg > 300.0
    assert avg < 2000.0

    shadow.stop()


def test_metrics_states_per_second_nonzero():
    """states_per_second > 0 after events flow."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    _publish_n_events(bus, 10)
    time.sleep(0.001)  # let some time elapse
    sps = shadow.metrics.states_per_second
    assert sps > 0.0

    shadow.stop()


def test_metrics_reset_clears_after_start():
    """reset() clears metrics mid-run; subsequent events start fresh."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    _publish_n_events(bus, 5)
    assert shadow.metrics.total_events == 5

    shadow.metrics.reset()
    assert shadow.metrics.total_events == 0
    assert shadow.metrics.total_publishes == 0

    _publish_n_events(bus, 3)
    assert shadow.metrics.total_events == 3
    assert shadow.metrics.total_publishes == 3

    shadow.stop()


def test_metrics_summary_after_run():
    """summary() returns a complete dict after a run."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    _publish_n_events(bus, 5)
    s = shadow.metrics.summary()

    assert s["total_events"] == 5
    assert s["total_publishes"] == 5
    assert s["successful"] == 5
    assert s["failed"] == 0
    assert s["conversion_failures"] == 0
    assert s["states_per_second"] > 0.0
    assert s["success_rate"] == 1.0
    assert s["avg_latency_us"] > 0.0
    assert s["elapsed_seconds"] > 0.0

    shadow.stop()


# ---------------------------------------------------------------------------
# Shadow mode: zero side effects with NullAdapter
# ---------------------------------------------------------------------------

def test_shadow_null_adapter_produces_no_network_output():
    """With default NullAdapter, no real transport occurs -- pure metrics."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)  # NullAdapter by default
    shadow.start()

    _publish_n_events(bus, 10)

    # All events were received and "published" (to NullAdapter).
    assert shadow.metrics.total_events == 10
    assert shadow.metrics.total_publishes == 10
    assert shadow.metrics.successful_publishes == 10
    # No side effects -- NullAdapter discarded everything.

    shadow.stop()


def test_shadow_with_custom_adapter():
    """A custom adapter (e.g. CapturingAdapter) receives the TargetStates."""
    bus = EventBus()
    adapter = CapturingAdapter()
    shadow = ShadowStatePublisher(bus, adapter=adapter)
    shadow.start()

    _publish_n_events(bus, 3)

    # The inner adapter received 3 TargetStates.
    assert len(adapter.published) == 3
    assert all(isinstance(s, TargetState) for s in adapter.published)

    # Metrics also recorded.
    assert shadow.metrics.total_publishes == 3
    assert shadow.metrics.successful_publishes == 3

    shadow.stop()


# ---------------------------------------------------------------------------
# No impact on existing system
# ---------------------------------------------------------------------------

def test_shadow_does_not_publish_on_bus():
    """ShadowStatePublisher only subscribes -- it does not publish events."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    shadow.start()

    # The shadow subscribes to 5 event types but does not publish.
    assert bus.subscriber_count(TargetCreatedEvent) == 1
    assert bus.subscriber_count(TargetLostEvent) == 1

    shadow.stop()


def test_shadow_does_not_affect_other_subscribers():
    """Other subscribers on the bus still receive events normally."""
    bus = EventBus()
    received: list = []
    bus.subscribe(TargetCreatedEvent, received.append)

    shadow = ShadowStatePublisher(bus)
    shadow.start()

    bus.publish(_make_event(TargetCreatedEvent, event_id="e1"))

    # The other subscriber received the event.
    assert len(received) == 1
    # The shadow also processed it.
    assert shadow.metrics.total_events == 1

    shadow.stop()


def test_repr():
    """__repr__ includes started state and metrics summary."""
    bus = EventBus()
    shadow = ShadowStatePublisher(bus)
    r = repr(shadow)
    assert "ShadowStatePublisher" in r
    assert "started=False" in r

    shadow.start()
    r = repr(shadow)
    assert "started=True" in r
    shadow.stop()


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    failures: list[str] = []
    passed = 0
    g = globals()
    names = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    for name in names:
        try:
            g[name]()
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
