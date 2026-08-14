"""Unit tests for visioncore.eventbus.debug_logger.DebugEventLogger.

Verifies enable/disable toggling, log output format, performance
short-circuiting, idempotency, and graceful handling of both typed
BaseEvent instances and the lightweight core.Event.

Run::

    python -m pytest tests/test_debug_logger.py -v
    # or
    python tests/test_debug_logger.py
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure the project root is importable when the file is run directly.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from visioncore.core.event import Event  # noqa: E402
from visioncore.eventbus import (  # noqa: E402
    BaseEvent,
    DebugEventLogger,
    EventBus,
    TargetCreatedEvent,
    TargetLockedEvent,
    TargetLostEvent,
    TargetRecoveredEvent,
    TargetRemovedEvent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CaptureHandler(logging.Handler):
    """A logging handler that records every emitted LogRecord."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    @property
    def messages(self) -> list[str]:
        return [self.format(r) for r in self.records]


def _make_logger() -> tuple[logging.Logger, _CaptureHandler]:
    """Create an isolated logger + capture handler for a single test."""
    logger = logging.getLogger(f"test.debug_logger.{id(object())}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    handler = _CaptureHandler()
    logger.addHandler(handler)
    return logger, handler


def _make_bus() -> EventBus:
    return EventBus()


def _make_typed_event() -> TargetLostEvent:
    return TargetLostEvent(
        event_id="evt-000001", timestamp=12.5,
        target_id="S0-T17", slot_id=0,
        payload={"last_seen": 10.0, "missed_frames": 4},
    )


# ---------------------------------------------------------------------------
# 1. 默认状态 (default state)
# ---------------------------------------------------------------------------

def test_default_disabled():
    """A fresh DebugEventLogger is disabled and does not subscribe."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    assert dbg.enabled is False
    assert bus.subscriber_count() == 0


def test_repr_reflects_state():
    """__repr__ reports the enabled state without raising."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    assert "enabled=False" in repr(dbg)
    dbg.enable()
    assert "enabled=True" in repr(dbg)


# ---------------------------------------------------------------------------
# 2. enable / disable
# ---------------------------------------------------------------------------

def test_enable_adds_wildcard_subscription():
    """enable() adds exactly one '*' subscription."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    dbg.enable()
    assert dbg.enabled is True
    assert bus.subscriber_count() == 1
    assert bus.subscriber_count("*") == 1


def test_disable_removes_subscription():
    """disable() removes the wildcard subscription."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    dbg.enable()
    dbg.disable()
    assert dbg.enabled is False
    assert bus.subscriber_count() == 0


def test_enable_is_idempotent():
    """Calling enable() twice adds only one subscription."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    dbg.enable()
    dbg.enable()
    dbg.enable()
    assert bus.subscriber_count() == 1


def test_disable_is_idempotent():
    """Calling disable() when already disabled is a no-op."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    dbg.disable()  # never enabled -- should not raise
    dbg.enable()
    dbg.disable()
    dbg.disable()
    assert dbg.enabled is False
    assert bus.subscriber_count() == 0


def test_reenable_after_disable():
    """enable -> disable -> enable works and re-subscribes."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    dbg.enable()
    dbg.disable()
    dbg.enable()
    assert dbg.enabled is True
    assert bus.subscriber_count() == 1


# ---------------------------------------------------------------------------
# 3. 日志输出 (log output)
# ---------------------------------------------------------------------------

def test_event_is_logged_when_enabled():
    """A published event produces exactly one INFO log record."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(_make_typed_event())
    event_records = [r for r in handler.records if r.levelno == logging.INFO
                     and "[EVENT]" in r.getMessage()]
    assert len(event_records) == 1


def test_no_log_when_disabled():
    """A published event produces no log when the logger is disabled."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    # never enabled
    bus.publish(_make_typed_event())
    event_records = [r for r in handler.records if "[EVENT]" in r.getMessage()]
    assert event_records == []


def test_log_format_contains_required_fields():
    """The log message matches the documented multi-line format."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(_make_typed_event())
    msg = handler.records[-1].getMessage()
    lines = msg.split("\n")
    assert lines[0] == "[EVENT]"
    assert lines[1] == "TargetLost"            # class name, Event suffix stripped
    assert any(ln.startswith("event_id=evt-000001") for ln in lines)
    assert any(ln == "event_type=target.lost" for ln in lines)
    assert any(ln == "target=S0-T17" for ln in lines)
    assert any(ln == "slot=0" for ln in lines)
    assert any(ln.startswith("timestamp=12.5000") for ln in lines)
    assert any(ln.startswith("payload_keys=") for ln in lines)


def test_log_format_strips_event_suffix():
    """TargetCreatedEvent is rendered as 'TargetCreated'."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(TargetCreatedEvent(
        event_id="e1", timestamp=1.0, target_id="S0-T1", slot_id=0,
    ))
    assert handler.records[-1].getMessage().split("\n")[1] == "TargetCreated"


def test_log_includes_payload_keys():
    """Non-empty payload is summarised by its keys."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(_make_typed_event())
    msg = handler.records[-1].getMessage()
    assert "payload_keys={'last_seen', 'missed_frames'}" in msg


def test_log_empty_payload():
    """An empty payload is rendered as payload_keys={}."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(TargetRemovedEvent(
        event_id="e1", timestamp=1.0, target_id="S0-T1", slot_id=0,
    ))
    msg = handler.records[-1].getMessage()
    assert "payload_keys={}" in msg


# ---------------------------------------------------------------------------
# 4. 兼容 core.Event (lightweight event without BaseEvent fields)
# ---------------------------------------------------------------------------

def test_core_event_is_formatted():
    """core.Event (no event_id/target_id/slot_id) is logged gracefully."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(Event("target.lost", 1.0, {"note": "legacy"}))
    msg = handler.records[-1].getMessage()
    lines = msg.split("\n")
    assert lines[0] == "[EVENT]"
    assert lines[1] == "Event"               # core.Event -> "Event" (suffix strip leaves empty -> kept)
    assert any(ln == "event_type=target.lost" for ln in lines)
    assert any(ln.startswith("timestamp=1.0000") for ln in lines)
    # core.Event has no event_id / target_id / slot_id -- those lines absent.
    assert not any(ln.startswith("event_id=") for ln in lines)
    assert not any(ln.startswith("target=") for ln in lines)
    assert not any(ln.startswith("slot=") for ln in lines)


# ---------------------------------------------------------------------------
# 5. 性能短路 (performance short-circuit)
# ---------------------------------------------------------------------------

def test_no_formatting_when_level_above_info():
    """When the logger level is above INFO, formatting is skipped."""
    bus = _make_bus()
    logger, handler = _make_logger()
    logger.setLevel(logging.WARNING)  # INFO records suppressed
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(_make_typed_event())
    event_records = [r for r in handler.records if "[EVENT]" in r.getMessage()]
    assert event_records == []   # no formatting happened, no record emitted


def test_disabled_bus_has_zero_overhead():
    """A disabled logger adds no subscribers to the bus."""
    bus = _make_bus()
    DebugEventLogger(bus)  # not enabled
    assert bus.subscriber_count() == 0
    # Publishing still works and returns 0 (no subscribers).
    assert bus.publish(_make_typed_event()) == 0


# ---------------------------------------------------------------------------
# 6. 多事件连续观察 (multiple events)
# ---------------------------------------------------------------------------

def test_multiple_events_logged_in_order():
    """Successive events produce log records in publish order."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(TargetCreatedEvent(
        event_id="e1", timestamp=1.0, target_id="S0-T1", slot_id=0))
    bus.publish(TargetLostEvent(
        event_id="e2", timestamp=2.0, target_id="S0-T1", slot_id=0))
    bus.publish(TargetRemovedEvent(
        event_id="e3", timestamp=3.0, target_id="S0-T1", slot_id=0))
    event_msgs = [r.getMessage() for r in handler.records
                  if r.getMessage().startswith("[EVENT]")]
    assert len(event_msgs) == 3
    assert event_msgs[0].split("\n")[1] == "TargetCreated"
    assert event_msgs[1].split("\n")[1] == "TargetLost"
    assert event_msgs[2].split("\n")[1] == "TargetRemoved"


def test_all_typed_event_classes_render():
    """Every typed event subclass renders without error."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    for cls in (TargetCreatedEvent, TargetLostEvent, TargetRecoveredEvent,
                TargetLockedEvent, TargetRemovedEvent):
        bus.publish(cls(
            event_id="e", timestamp=1.0, target_id="T", slot_id=0,
        ))
    event_msgs = [r for r in handler.records
                  if r.getMessage().startswith("[EVENT]")]
    assert len(event_msgs) == 5


# ---------------------------------------------------------------------------
# 7. 自定义 logger (custom logger injection)
# ---------------------------------------------------------------------------

def test_custom_logger_is_used():
    """A logger passed to the constructor is used for output."""
    bus = _make_bus()
    logger, handler = _make_logger()
    dbg = DebugEventLogger(bus, logger=logger)
    dbg.enable()
    bus.publish(_make_typed_event())
    assert any("[EVENT]" in r.getMessage() for r in handler.records)


def test_default_logger_name():
    """The default logger is named visioncore.eventbus.debug_logger."""
    bus = _make_bus()
    dbg = DebugEventLogger(bus)
    assert dbg._logger.name == "visioncore.eventbus.debug_logger"


# ---------------------------------------------------------------------------
# Module entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    failures = 0
    g = globals()
    names = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    for name in names:
        try:
            g[name]()
            print(f"PASS {name}")
        except Exception:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    print(f"\n{len(names) - failures}/{len(names)} tests passed")
    raise SystemExit(1 if failures else 0)
