"""Shadow Integration tests -- Tracker -> TargetManager -> EventBus pipeline.

Verifies that the event flow pipeline assembled in Milestone 3 works end
to end without any consumer subscribing:

    Tracker (detections)
        |  InferWorker._bypass_update_targets
        v
    detections_to_tracks -> list[Track]
        |  TargetManager.update_targets
        v
    _emit_lifecycle_event
        |  EventBus.publish
        v
    EventBus (no subscribers -- pipeline validated, nothing consumed)

InferWorker and the GUI do **not** subscribe to events. This test suite
exercises the same data path the InferWorker bypass uses
(:func:`detections_to_tracks` + :meth:`TargetManager.update_targets`) and
confirms events reach the bus. A lightweight import-level check confirms
:class:`~ai.inference.InferWorker` exposes the ``event_bus`` property
without instantiating the worker (which would load ML models).

Run::

    python -m pytest tests/test_shadow_integration.py -v
    # or
    python tests/test_shadow_integration.py
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is importable when the file is run directly.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from visioncore.eventbus import EventBus  # noqa: E402
from visioncore.eventbus.events import (  # noqa: E402
    BaseEvent,
    TargetCreatedEvent,
    TargetLostEvent,
    TargetRemovedEvent,
)
from visioncore.target_manager import TargetManager, detections_to_tracks  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers -- detection dicts in the same format InferWorker emits
# ---------------------------------------------------------------------------

def _det(track_id: int, label: str = "person", conf: float = 0.9) -> dict:
    """Build a detection dict identical in shape to InferWorker's output."""
    return {
        "track_id": track_id,
        "x1": 0.4, "y1": 0.3, "x2": 0.6, "y2": 0.7,
        "confidence": conf,
        "class_id": 0,
        "label": label,
    }


def _make_pipeline() -> tuple[TargetManager, EventBus, list[BaseEvent]]:
    """Assemble the shadow pipeline: EventBus -> TargetManager -> capture.

    Returns (manager, bus, captured_events). A wildcard subscriber records
    every event so tests can assert on the flow.
    """
    bus = EventBus()
    mgr = TargetManager(event_bus=bus)
    captured: list[BaseEvent] = []
    bus.subscribe("*", captured.append)
    return mgr, bus, captured


def _bypass(mgr: TargetManager, slot_id: int, detections: list[dict],
            timestamp: float) -> dict[str, int]:
    """Replicate InferWorker._bypass_update_targets core logic.

    This is exactly what the InferWorker bypass does: convert detection
    dicts to Tracks, then feed them to update_targets. Reusing the real
    converter keeps the test faithful to the production data path.
    """
    tracks = detections_to_tracks(detections)
    return mgr.update_targets(tracks, slot_id=slot_id, timestamp=timestamp)


# ---------------------------------------------------------------------------
# 1. EventBus 创建并注入 TargetManager
# ---------------------------------------------------------------------------

def test_event_bus_created_and_injected():
    """TargetManager(event_bus=bus) exposes the same bus instance."""
    bus = EventBus()
    mgr = TargetManager(event_bus=bus)
    assert mgr.event_bus is bus


def test_default_manager_has_no_bus():
    """Without explicit injection the manager has no bus (backward compat)."""
    mgr = TargetManager()
    assert mgr.event_bus is None


# ---------------------------------------------------------------------------
# 2. 数据流：Tracker -> TargetManager -> EventBus
# ---------------------------------------------------------------------------

def test_detections_flow_to_event_bus_as_created():
    """Detection dicts flow through the pipeline and emit Created events."""
    mgr, _bus, captured = _make_pipeline()
    counts = _bypass(mgr, slot_id=0, detections=[_det(1), _det(2)],
                     timestamp=1.0)
    assert counts["created"] == 2
    created = [e for e in captured if isinstance(e, TargetCreatedEvent)]
    assert len(created) == 2
    # Each created event carries the slot and a payload with track_id.
    for ev in created:
        assert ev.slot_id == 0
        assert ev.event_type == "target.created"


def test_empty_detections_no_new_events():
    """An empty detection list for a fresh slot emits nothing."""
    mgr, _bus, captured = _make_pipeline()
    _bypass(mgr, slot_id=0, detections=[], timestamp=1.0)
    assert captured == []


def test_stale_sweep_emits_lost_then_removed():
    """Stale ACTIVE targets emit Lost, then later Removed on timeout."""
    mgr, _bus, captured = _make_pipeline()
    # Create a target at t=1.0.
    _bypass(mgr, slot_id=0, detections=[_det(1)], timestamp=1.0)
    captured.clear()
    # t=10: empty -> ACTIVE goes stale (age 9 >= 5) -> LOST.
    _bypass(mgr, slot_id=0, detections=[], timestamp=10.0)
    # t=100: still empty -> LOST times out (age 90 >= 30) -> REMOVED.
    _bypass(mgr, slot_id=0, detections=[], timestamp=100.0)
    types = [e.event_type for e in captured]
    assert "target.lost" in types
    assert "target.removed" in types
    assert types.index("target.lost") < types.index("target.removed")


def test_reappear_emits_recovered():
    """A lost target re-appearing emits a Recovered event."""
    mgr, _bus, captured = _make_pipeline()
    _bypass(mgr, slot_id=0, detections=[_det(1)], timestamp=1.0)
    _bypass(mgr, slot_id=0, detections=[], timestamp=10.0)  # -> LOST
    captured.clear()
    _bypass(mgr, slot_id=0, detections=[_det(1)], timestamp=12.0)  # recover
    assert any(e.event_type == "target.recovered" for e in captured)


# ---------------------------------------------------------------------------
# 3. 多槽位隔离（事件携带正确 slot_id）
# ---------------------------------------------------------------------------

def test_multi_slot_events_carry_correct_slot():
    """Events from different slots carry their originating slot_id."""
    mgr, _bus, captured = _make_pipeline()
    _bypass(mgr, slot_id=0, detections=[_det(1)], timestamp=1.0)
    _bypass(mgr, slot_id=2, detections=[_det(1)], timestamp=1.0)
    created = [e for e in captured if isinstance(e, TargetCreatedEvent)]
    slots = sorted(ev.slot_id for ev in created)
    assert slots == [0, 2]


# ---------------------------------------------------------------------------
# 4. 无订阅者管道仍正常（shadow 不影响功能）
# ---------------------------------------------------------------------------

def test_no_subscribers_pipeline_runs_without_error():
    """With zero subscribers the pipeline still runs and returns counts."""
    bus = EventBus()
    mgr = TargetManager(event_bus=bus)
    # No subscribe() calls -- bus has zero subscribers.
    counts = _bypass(mgr, slot_id=0, detections=[_det(1), _det(2)],
                     timestamp=1.0)
    assert counts["created"] == 2
    assert bus.subscriber_count() == 0


def test_bus_injection_does_not_change_counts():
    """Injecting a bus yields identical update_targets counts vs no bus."""
    det = [_det(1), _det(2), _det(3)]
    mgr_with = TargetManager(event_bus=EventBus())
    mgr_without = TargetManager()
    c_with = _bypass(mgr_with, slot_id=1, detections=det, timestamp=1.0)
    c_without = _bypass(mgr_without, slot_id=1, detections=det, timestamp=1.0)
    assert c_with == c_without


def test_bus_injection_does_not_change_target_state():
    """Target state transitions are identical with/without a bus."""
    mgr_with = TargetManager(event_bus=EventBus())
    mgr_without = TargetManager()
    _bypass(mgr_with, slot_id=0, detections=[_det(1)], timestamp=1.0)
    _bypass(mgr_without, slot_id=0, detections=[_det(1)], timestamp=1.0)
    t_w = mgr_with.get_all_targets()[0]
    t_wo = mgr_without.get_all_targets()[0]
    assert t_w.state == t_wo.state
    assert t_w.slot_id == t_wo.slot_id


# ---------------------------------------------------------------------------
# 5. InferWorker 集成验证（导入级，不实例化模型）
# ---------------------------------------------------------------------------

def test_inferworker_exposes_event_bus_property():
    """InferWorker class defines an event_bus property (shadow wiring).

    The worker is not instantiated (that would load ML models); instead we
    introspect the class to confirm the shadow-integration wiring exists.
    """
    try:
        from ai.inference import InferWorker
    except Exception as exc:  # noqa: BLE001 -- heavy ML deps may be absent
        import pytest
        pytest.skip(f"InferWorker import unavailable: {exc!r}")
    assert hasattr(InferWorker, "event_bus")
    assert hasattr(InferWorker, "target_manager")
    # The property must be a descriptor on the class.
    assert isinstance(
        type(InferWorker).__dict__.get("event_bus")
        or InferWorker.__dict__.get("event_bus"),
        property,
    )


# ---------------------------------------------------------------------------
# 6. 端到端事件流序列（完整生命周期）
# ---------------------------------------------------------------------------

def test_end_to_end_event_sequence():
    """Full shadow pipeline produces the expected event sequence."""
    mgr, _bus, captured = _make_pipeline()
    # Create.
    _bypass(mgr, slot_id=0, detections=[_det(1)], timestamp=1.0)
    # Lose.
    _bypass(mgr, slot_id=0, detections=[], timestamp=10.0)
    # Recover.
    _bypass(mgr, slot_id=0, detections=[_det(1)], timestamp=12.0)
    # Lose again + remove.
    _bypass(mgr, slot_id=0, detections=[], timestamp=20.0)
    _bypass(mgr, slot_id=0, detections=[], timestamp=200.0)

    types = [e.event_type for e in captured]
    # created, lost, recovered, lost, removed
    assert types == [
        "target.created",
        "target.lost",
        "target.recovered",
        "target.lost",
        "target.removed",
    ]


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
