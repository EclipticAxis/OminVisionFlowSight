"""ShadowStatePublisher -- metrics-collecting shadow observer.

Defines :class:`ShadowStatePublisher`, a drop-in replacement for
:class:`~visioncore.protocol.publisher.state_publisher.StatePublisher`
that adds debug metrics collection without affecting the existing system.

"Shadow mode" means:
    * The publisher subscribes to the EventBus exactly like StatePublisher.
    * By default it uses a :class:`~visioncore.protocol.NullAdapter` --
      events are converted to TargetState and "published" but the payload
      is silently discarded. Zero network overhead, zero side effects.
    * Every event received and every publish attempted is timed and
      counted, producing real-time debug statistics.
    * A custom adapter can be injected for real transport (e.g. UDPAdapter)
      while still collecting metrics.

Metrics collected:
    * **total_events** -- every event received from the bus (before conversion).
    * **total_publishes** -- every ``adapter.publish()`` call attempted.
    * **successful_publishes** -- publish calls that did not raise.
    * **failed_publishes** -- publish calls that raised.
    * **total_publish_time_ns** -- cumulative publish wall time.
    * **conversion_failures** -- events that could not be converted to TargetState.

Derived metrics:
    * **states_per_second** -- total_publishes / elapsed_seconds.
    * **success_rate** -- successful_publishes / total_publishes.
    * **avg_latency_us** -- average publish time in microseconds.

No modification of existing system
----------------------------------
ShadowStatePublisher is a pure observer. It does NOT modify:

* ``ai/`` (InferWorker, Tracker, detection) -- no inference files touched.
* ``gui/`` -- no GUI files touched.
* ``camera/`` -- no camera files touched.
* ``visioncore/target_manager/`` -- no target manager files touched.
* ``visioncore/eventbus/`` -- no event bus files touched.

It only reads from the EventBus (via subscription) and writes to the
injected adapter (NullAdapter by default). The existing detection
pipeline, tracker, and GUI continue to operate exactly as before.

Example
-------
    >>> from visioncore.eventbus import EventBus
    >>> from visioncore.protocol.publisher import ShadowStatePublisher
    >>> bus = EventBus()
    >>> shadow = ShadowStatePublisher(bus)  # NullAdapter by default
    >>> shadow.start()
    >>> # ... TargetManager publishes events on bus ...
    >>> shadow.metrics.summary()
    {'total_events': 42, 'total_publishes': 42, 'successful': 42,
     'failed': 0, 'conversion_failures': 0, 'states_per_second': 10.5,
     'success_rate': 1.0, 'avg_latency_us': 3.2}
    >>> shadow.stop()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from visioncore.eventbus.events import BaseEvent
from visioncore.protocol.base import ProtocolAdapter
from visioncore.protocol.null_adapter import NullAdapter
from visioncore.protocol.publisher.state_publisher import StatePublisher
from visioncore.state.target_state import TargetState


__all__ = ["ShadowStatePublisher", "ShadowMetrics"]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ShadowMetrics -- debug statistics collector
# ---------------------------------------------------------------------------

class ShadowMetrics:
    """Collects debug metrics for shadow publishing.

    Thread-safe: all counter updates are protected by an internal lock.
    The lock is held only during counter increments (nanoseconds), never
    during the actual publish call -- so contention is negligible even
    under high event rates.

    The metrics are "eventually consistent" -- a snapshot via
    :meth:`summary` may miss an in-flight update, but the values converge
    to the true counts once the publisher is stopped.
    """

    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()

        # Raw counters.
        self._total_events: int = 0
        self._total_publishes: int = 0
        self._successful_publishes: int = 0
        self._failed_publishes: int = 0
        self._conversion_failures: int = 0
        self._total_publish_time_ns: int = 0

        # Timing reference for states_per_second.
        self._first_event_time: float = 0.0
        self._started: bool = False

    # ------------------------------------------------------------------
    # Recording (called by ShadowStatePublisher / _MetricsAdapter)
    # ------------------------------------------------------------------

    def record_event(self) -> None:
        """Record that an event was received from the bus."""
        with self._lock:
            if not self._started:
                self._first_event_time = time.perf_counter()
                self._started = True
            self._total_events += 1

    def record_conversion_failure(self) -> None:
        """Record that an event could not be converted to TargetState."""
        with self._lock:
            self._conversion_failures += 1

    def record_publish(self, duration_ns: int, success: bool) -> None:
        """Record a publish attempt with its duration and outcome.

        Args:
            duration_ns: Wall-clock duration of the publish call in
                nanoseconds (use ``time.perf_counter_ns()`` delta).
            success: ``True`` if the publish did not raise, ``False``
                if it raised.
        """
        with self._lock:
            self._total_publishes += 1
            self._total_publish_time_ns += duration_ns
            if success:
                self._successful_publishes += 1
            else:
                self._failed_publishes += 1

    # ------------------------------------------------------------------
    # Derived metrics (read-only properties)
    # ------------------------------------------------------------------

    @property
    def total_events(self) -> int:
        """Total events received from the bus."""
        with self._lock:
            return self._total_events

    @property
    def total_publishes(self) -> int:
        """Total ``adapter.publish()`` calls attempted."""
        with self._lock:
            return self._total_publishes

    @property
    def successful_publishes(self) -> int:
        """Publish calls that did not raise."""
        with self._lock:
            return self._successful_publishes

    @property
    def failed_publishes(self) -> int:
        """Publish calls that raised."""
        with self._lock:
            return self._failed_publishes

    @property
    def conversion_failures(self) -> int:
        """Events that could not be converted to TargetState."""
        with self._lock:
            return self._conversion_failures

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since the first event was received.

        Returns 0.0 if no events have been received yet.
        """
        with self._lock:
            if not self._started:
                return 0.0
            return time.perf_counter() - self._first_event_time

    @property
    def states_per_second(self) -> float:
        """Average states published per second since the first event.

        Returns 0.0 if no events have been received or if less than 1
        microsecond has elapsed (to avoid division by near-zero).
        """
        with self._lock:
            if self._total_publishes == 0 or not self._started:
                return 0.0
            elapsed = time.perf_counter() - self._first_event_time
            if elapsed < 1e-6:
                return 0.0
            return self._total_publishes / elapsed

    @property
    def success_rate(self) -> float:
        """Fraction of publish calls that succeeded (0.0 to 1.0).

        Returns 1.0 if no publishes have been attempted (vacuously true).
        """
        with self._lock:
            total = self._total_publishes
            if total == 0:
                return 1.0
            return self._successful_publishes / total

    @property
    def avg_latency_us(self) -> float:
        """Average publish call duration in microseconds.

        Returns 0.0 if no publishes have been attempted.
        """
        with self._lock:
            if self._total_publishes == 0:
                return 0.0
            return (self._total_publish_time_ns / self._total_publishes) / 1000.0

    # ------------------------------------------------------------------
    # Snapshot / reset
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a snapshot of all metrics as a dict.

        The dict is a point-in-time copy -- mutating it does not affect
        the live counters. Useful for logging, JSON serialisation, or
        passing to a monitoring dashboard.
        """
        with self._lock:
            elapsed = 0.0
            sps = 0.0
            if self._started:
                elapsed = time.perf_counter() - self._first_event_time
                if elapsed > 1e-6:
                    sps = self._total_publishes / elapsed
            total_pubs = self._total_publishes
            avg_us = (
                (self._total_publish_time_ns / total_pubs) / 1000.0
                if total_pubs > 0 else 0.0
            )
            sr = (
                self._successful_publishes / total_pubs
                if total_pubs > 0 else 1.0
            )
            return {
                "total_events": self._total_events,
                "total_publishes": total_pubs,
                "successful": self._successful_publishes,
                "failed": self._failed_publishes,
                "conversion_failures": self._conversion_failures,
                "elapsed_seconds": round(elapsed, 6),
                "states_per_second": round(sps, 2),
                "success_rate": round(sr, 4),
                "avg_latency_us": round(avg_us, 2),
            }

    def reset(self) -> None:
        """Reset all counters to zero and clear the timing reference.

        After reset, :attr:`states_per_second` will be 0.0 until the
        next event is received (which establishes a new timing reference).
        """
        with self._lock:
            self._total_events = 0
            self._total_publishes = 0
            self._successful_publishes = 0
            self._failed_publishes = 0
            self._conversion_failures = 0
            self._total_publish_time_ns = 0
            self._first_event_time = 0.0
            self._started = False

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ShadowMetrics(events={s['total_events']}, "
            f"publishes={s['total_publishes']}, "
            f"ok={s['successful']}, fail={s['failed']}, "
            f"sps={s['states_per_second']}, "
            f"sr={s['success_rate']}, "
            f"lat_us={s['avg_latency_us']})"
        )


# ---------------------------------------------------------------------------
# _MetricsAdapter -- wraps a real adapter, records publish metrics
# ---------------------------------------------------------------------------

class _MetricsAdapter(ProtocolAdapter):
    """A ProtocolAdapter wrapper that records publish timing and success.

    Delegates all operations to the wrapped (inner) adapter, but wraps
    ``publish()`` with high-resolution timing and outcome tracking. The
    metrics are recorded into the provided :class:`ShadowMetrics` instance.

    This adapter is internal to the shadow publisher module -- it is not
    part of the public API.
    """

    def __init__(self, inner: ProtocolAdapter, metrics: ShadowMetrics) -> None:
        self._inner: ProtocolAdapter = inner
        self._metrics: ShadowMetrics = metrics

    def connect(self) -> None:
        self._inner.connect()

    def disconnect(self) -> None:
        self._inner.disconnect()

    def publish(self, state: TargetState) -> None:
        # Time the publish call. perf_counter_ns is monotonic and
        # high-resolution -- suitable for sub-microsecond timing.
        start_ns: int = time.perf_counter_ns()
        success: bool = True
        try:
            self._inner.publish(state)
        except Exception:
            success = False
            raise  # Re-raise so StatePublisher._on_event can catch it.
        finally:
            duration_ns: int = time.perf_counter_ns() - start_ns
            self._metrics.record_publish(duration_ns, success)

    def health_check(self) -> bool:
        return self._inner.health_check()

    def __repr__(self) -> str:
        return f"_MetricsAdapter(inner={type(self._inner).__name__})"


# ---------------------------------------------------------------------------
# ShadowStatePublisher -- extends StatePublisher with metrics
# ---------------------------------------------------------------------------

class ShadowStatePublisher(StatePublisher):
    """A StatePublisher that collects debug metrics in shadow mode.

    Extends :class:`StatePublisher` with automatic metrics collection.
    By default, uses a :class:`~visioncore.protocol.NullAdapter` (the
    "shadow" -- events are converted and "published" but the payload is
    discarded, producing zero side effects). A custom adapter can be
    injected for real transport while still collecting metrics.

    Shadow guarantees
    -----------------
    * **No impact on detection**: the publisher subscribes to the EventBus
      and converts events; it does not touch the detector, tracker, or
      any inference code.
    * **No impact on Tracker**: the publisher reads events that have
      already been published; it does not modify the TargetManager or
      any Target.
    * **No impact on GUI**: no GUI files are modified; the publisher's
      metrics are accessed programmatically, not through the GUI.
    * **NullAdapter default**: with the default adapter, every publish
      is a no-op -- the only overhead is event-to-TargetState conversion
      and metrics recording (sub-microsecond).

    Metrics access
    --------------
    The :attr:`metrics` property returns the :class:`ShadowMetrics`
    instance, which provides real-time access to counts, rates, and
    latency::

        >>> shadow.metrics.total_events
        42
        >>> shadow.metrics.success_rate
        1.0
        >>> shadow.metrics.summary()
        {'total_events': 42, ...}

    Args:
        event_bus: The EventBus to subscribe to.
        adapter: Optional ProtocolAdapter for real transport. If ``None``
            (default), uses :class:`NullAdapter` -- shadow mode with
            zero network overhead.

    Example:
        >>> bus = EventBus()
        >>> shadow = ShadowStatePublisher(bus)
        >>> shadow.start()
        >>> # ... events flow ...
        >>> print(shadow.metrics.summary())
        >>> shadow.stop()
    """

    def __init__(
        self,
        event_bus: Any,  # EventBus, typed as Any to avoid circular import
        adapter: ProtocolAdapter | None = None,
    ) -> None:
        # Create the metrics instance first -- the _MetricsAdapter needs it.
        self._shadow_metrics: ShadowMetrics = ShadowMetrics()

        # Wrap the user's adapter (or NullAdapter) in a _MetricsAdapter
        # so every publish call is timed and counted.
        inner_adapter: ProtocolAdapter = adapter if adapter is not None else NullAdapter()
        metrics_adapter: _MetricsAdapter = _MetricsAdapter(inner_adapter, self._shadow_metrics)

        # Initialise the parent StatePublisher with the metrics-wrapped
        # adapter. StatePublisher will call connect()/disconnect()/
        # publish() on the _MetricsAdapter, which delegates to the inner
        # adapter while recording metrics.
        super().__init__(event_bus, metrics_adapter)

    # ------------------------------------------------------------------
    # Override _on_event to add event counting + conversion failure tracking
    # ------------------------------------------------------------------

    def _on_event(self, event: BaseEvent) -> None:
        """Handle a lifecycle event with metrics collection.

        Wraps the parent's ``_on_event`` with:
        1. Event counting (``record_event``).
        2. Conversion failure tracking (``record_conversion_failure``).

        The publish timing and success tracking is handled by the
        :class:`_MetricsAdapter` that wraps the inner adapter.
        """
        self._shadow_metrics.record_event()

        # Check if conversion will succeed by attempting it first.
        # If conversion fails, record it and return (don't attempt publish).
        try:
            state = self._event_to_state(event)
        except Exception:
            self._shadow_metrics.record_conversion_failure()
            logger.exception(
                "ShadowStatePublisher: failed to convert event %s",
                repr(event),
            )
            return

        # Conversion succeeded -- publish via the metrics-wrapped adapter.
        # The _MetricsAdapter records timing + success/failure.
        try:
            self._adapter.publish(state)
        except Exception:
            logger.warning(
                "ShadowStatePublisher: publish failed for event %s "
                "(event_id=%s), ignoring",
                event.event_type,
                event.event_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # Public metrics access
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> ShadowMetrics:
        """Return the :class:`ShadowMetrics` instance for this publisher.

        The metrics object is live -- reading its properties reflects
        the current state of the counters at the moment of the read.
        """
        return self._shadow_metrics

    def __repr__(self) -> str:
        return (
            f"ShadowStatePublisher(started={self._started}, "
            f"adapter={type(self._adapter).__name__}, "
            f"metrics={self._shadow_metrics!r})"
        )
