"""Unit tests for visioncore.target_manager.TargetManager.

Verifies Target creation, deletion, state transitions (lock/unlock/recover),
and batch reconciliation (update_targets).

Run::

    python -m pytest tests/test_target_manager.py -v
    # or
    python tests/test_target_manager.py
"""

from __future__ import annotations

import inspect
import sys
import threading

from visioncore.core.detection import BBox, Detection
from visioncore.core.target import Target, TargetState
from visioncore.core.track import Track
from visioncore.target_manager import TargetManager, TargetStore
from visioncore.target_manager.lifecycle import TargetLifecycleManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(track_id: int = 1, score: float = 0.9) -> Track:
    """Create a minimal Track for testing."""
    return Track(
        track_id=track_id,
        detection=Detection(BBox(0.5, 0.5, 0.2, 0.4), score, 0, "person"),
    )


def _make_manager() -> TargetManager:
    """Create a fresh TargetManager for each test."""
    return TargetManager()


# ---------------------------------------------------------------------------
# 1. 创建 Target
# ---------------------------------------------------------------------------

def test_create_target_basic():
    """create_target returns a Target with sequential id and ACTIVE state."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert t.target_id == "S0-T0001"
    assert t.state == TargetState.ACTIVE
    assert t.priority == 0
    assert t.track is not None
    assert t.track.track_id == 1
    assert mgr.target_count == 1


def test_create_target_no_track_is_lost():
    """create_target with track=None starts in LOST state."""
    mgr = _make_manager()
    t = mgr.create_target(track=None)
    assert t.state == TargetState.LOST
    assert t.track is None


def test_create_target_with_attributes():
    """create_target stores a shallow copy of attributes."""
    mgr = _make_manager()
    attrs = {"identity": "alice", "threat": "low"}
    t = mgr.create_target(track=_make_track(1), attributes=attrs)
    assert t.attributes["identity"] == "alice"
    assert t.attributes["threat"] == "low"
    # Mutating original dict must not affect the target
    attrs["identity"] = "bob"
    assert t.attributes["identity"] == "alice"


def test_create_target_sequential_ids():
    """IDs are sequential: S0-T0001, S0-T0002, S0-T0003."""
    mgr = _make_manager()
    t1 = mgr.create_target(track=_make_track(1))
    t2 = mgr.create_target(track=_make_track(2))
    t3 = mgr.create_target(track=_make_track(3))
    assert t1.target_id == "S0-T0001"
    assert t2.target_id == "S0-T0002"
    assert t3.target_id == "S0-T0003"


def test_create_target_priority():
    """create_target stores the given priority."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), priority=42)
    assert t.priority == 42


def test_create_target_last_seen():
    """create_target stores the given last_seen timestamp."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), last_seen=12.5)
    assert t.last_seen == 12.5


def test_create_target_added_to_store():
    """After creation, the target is retrievable from the store."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.get_target(t.target_id) is t
    assert t.target_id in mgr.store


# ---------------------------------------------------------------------------
# 2. 删除 Target
# ---------------------------------------------------------------------------

def test_mark_removed_from_lost():
    """mark_removed transitions LOST -> REMOVED and deletes from store."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.mark_lost(t.target_id)
    assert mgr.mark_removed(t.target_id) is True
    assert mgr.get_target(t.target_id) is None
    assert mgr.target_count == 0


def test_mark_removed_from_active_fails():
    """mark_removed on an ACTIVE target is rejected by the lifecycle."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.mark_removed(t.target_id) is False
    assert mgr.get_target(t.target_id) is not None
    assert t.state == TargetState.ACTIVE


def test_mark_removed_from_locked_fails():
    """mark_removed on a LOCKED target is rejected."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.lock_target(t.target_id)
    assert mgr.mark_removed(t.target_id) is False
    assert t.state == TargetState.LOCKED


def test_remove_target_alias():
    """remove_target is an alias for mark_removed."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.mark_lost(t.target_id)
    assert mgr.remove_target(t.target_id) is True
    assert mgr.get_target(t.target_id) is None


def test_mark_removed_not_found():
    """mark_removed on a nonexistent id returns False."""
    mgr = _make_manager()
    assert mgr.mark_removed("nonexistent") is False


def test_mark_removed_double_fails():
    """Calling mark_removed twice returns False the second time."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.mark_lost(t.target_id)
    assert mgr.mark_removed(t.target_id) is True
    assert mgr.mark_removed(t.target_id) is False


# ---------------------------------------------------------------------------
# 3. 状态切换
# ---------------------------------------------------------------------------

def test_mark_lost_active_to_lost():
    """mark_lost transitions ACTIVE -> LOST."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.mark_lost(t.target_id) is True
    assert t.state == TargetState.LOST


def test_mark_lost_with_timestamp():
    """mark_lost stores the timestamp in last_seen."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.mark_lost(t.target_id, timestamp=42.0)
    assert t.last_seen == 42.0


def test_mark_lost_from_locked_fails():
    """mark_lost on a LOCKED target is rejected."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.lock_target(t.target_id)
    assert mgr.mark_lost(t.target_id) is False
    assert t.state == TargetState.LOCKED


def test_mark_lost_not_found():
    """mark_lost on a nonexistent id returns False."""
    mgr = _make_manager()
    assert mgr.mark_lost("nonexistent") is False


def test_mark_lost_double_fails():
    """Calling mark_lost twice returns False the second time."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.mark_lost(t.target_id) is True
    assert mgr.mark_lost(t.target_id) is False


def test_full_recovery_cycle():
    """ACTIVE -> LOST -> RECOVERED -> ACTIVE round-trip."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.mark_lost(t.target_id) is True
    assert t.state == TargetState.LOST
    assert mgr.mark_recovered(t.target_id, _make_track(2)) is True
    assert t.state == TargetState.ACTIVE


def test_mark_recovered_updates_track():
    """mark_recovered replaces the target's track."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.mark_lost(t.target_id)
    new_track = _make_track(99)
    mgr.mark_recovered(t.target_id, new_track)
    assert t.track is new_track
    assert t.track.track_id == 99


def test_mark_recovered_with_timestamp():
    """mark_recovered updates last_seen."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), last_seen=0.0)
    mgr.mark_lost(t.target_id)
    mgr.mark_recovered(t.target_id, _make_track(2), timestamp=15.0)
    assert t.last_seen == 15.0


def test_mark_recovered_from_active_fails():
    """mark_recovered on an ACTIVE target is rejected."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.mark_recovered(t.target_id, _make_track(2)) is False
    assert t.state == TargetState.ACTIVE


def test_mark_recovered_not_found():
    """mark_recovered on a nonexistent id returns False."""
    mgr = _make_manager()
    assert mgr.mark_recovered("nonexistent", _make_track(1)) is False


# ---------------------------------------------------------------------------
# 4. 锁定目标
# ---------------------------------------------------------------------------

def test_lock_target_active_to_locked():
    """lock_target transitions ACTIVE -> LOCKED."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.lock_target(t.target_id) is True
    assert t.state == TargetState.LOCKED


def test_lock_target_from_lost_fails():
    """lock_target on a LOST target is rejected."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.mark_lost(t.target_id)
    assert mgr.lock_target(t.target_id) is False
    assert t.state == TargetState.LOST


def test_lock_target_double_fails():
    """Locking an already-locked target is rejected."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.lock_target(t.target_id)
    assert mgr.lock_target(t.target_id) is False


def test_lock_target_not_found():
    """lock_target on a nonexistent id returns False."""
    mgr = _make_manager()
    assert mgr.lock_target("nonexistent") is False


def test_unlock_target_locked_to_active():
    """unlock_target transitions LOCKED -> ACTIVE."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.lock_target(t.target_id)
    assert mgr.unlock_target(t.target_id) is True
    assert t.state == TargetState.ACTIVE


def test_unlock_target_from_active_fails():
    """unlock_target on an ACTIVE target is rejected."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.unlock_target(t.target_id) is False
    assert t.state == TargetState.ACTIVE


def test_unlock_target_not_found():
    """unlock_target on a nonexistent id returns False."""
    mgr = _make_manager()
    assert mgr.unlock_target("nonexistent") is False


def test_release_target_alias():
    """release_target is an alias for unlock_target."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.lock_target(t.target_id)
    assert mgr.release_target(t.target_id) is True
    assert t.state == TargetState.ACTIVE


def test_lock_unlock_cycle():
    """Lock then unlock returns to ACTIVE."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    for _ in range(3):
        assert mgr.lock_target(t.target_id) is True
        assert t.state == TargetState.LOCKED
        assert mgr.unlock_target(t.target_id) is True
        assert t.state == TargetState.ACTIVE


# ---------------------------------------------------------------------------
# 5. 恢复目标
# ---------------------------------------------------------------------------

def test_recover_lost_target():
    """A LOST target can be recovered to ACTIVE."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.mark_lost(t.target_id)
    assert mgr.mark_recovered(t.target_id, _make_track(2)) is True
    assert t.state == TargetState.ACTIVE
    assert t.track.track_id == 2


def test_recover_after_long_loss():
    """Recovery works even after many frames lost."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), last_seen=0.0)
    mgr.mark_lost(t.target_id, timestamp=1.0)
    # Simulate time passing
    mgr.update_last_seen(t.target_id, 100.0)
    assert mgr.mark_recovered(t.target_id, _make_track(3), timestamp=101.0) is True
    assert t.state == TargetState.ACTIVE
    assert t.last_seen == 101.0


def test_recover_preserves_priority():
    """Recovery does not change the target's priority."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), priority=7)
    mgr.mark_lost(t.target_id)
    mgr.mark_recovered(t.target_id, _make_track(2))
    assert t.priority == 7


def test_recover_preserves_attributes():
    """Recovery does not clear the target's attributes."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), attributes={"id": "abc"})
    mgr.mark_lost(t.target_id)
    mgr.mark_recovered(t.target_id, _make_track(2))
    assert t.attributes["id"] == "abc"


def test_recover_from_locked_fails():
    """Cannot recover a LOCKED target (must be in LOST state)."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    mgr.lock_target(t.target_id)
    assert mgr.mark_recovered(t.target_id, _make_track(2)) is False
    assert t.state == TargetState.LOCKED


# ---------------------------------------------------------------------------
# 6. 批量更新
# ---------------------------------------------------------------------------

def test_update_targets_create_new():
    """update_targets creates targets for unmatched tracks."""
    mgr = _make_manager()
    counts = mgr.update_targets([_make_track(1), _make_track(2)], timestamp=1.0)
    assert counts["created"] == 2
    assert counts["updated"] == 0
    assert mgr.target_count == 2


def test_update_targets_update_existing():
    """update_targets updates ACTIVE targets that match input tracks."""
    mgr = _make_manager()
    mgr.create_target(track=_make_track(1), last_seen=0.0)
    counts = mgr.update_targets([_make_track(1)], timestamp=5.0)
    assert counts["updated"] == 1
    assert counts["created"] == 0
    t = mgr.get_target("S0-T0001")
    assert t.last_seen == 5.0


def test_update_targets_recover_lost():
    """update_targets recovers LOST targets that match input tracks."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), last_seen=0.0)
    mgr.mark_lost(t.target_id, timestamp=1.0)
    counts = mgr.update_targets([_make_track(1)], timestamp=10.0)
    assert counts["recovered"] == 1
    assert t.state == TargetState.ACTIVE


def test_update_targets_mark_stale_lost():
    """update_targets marks ACTIVE targets as LOST when stale."""
    mgr = _make_manager()
    mgr.create_target(track=_make_track(1), last_seen=0.0)
    # Feed only track 2 — track 1 is unmatched and stale
    counts = mgr.update_targets(
        [_make_track(2)], timestamp=10.0, stale_threshold=5.0,
    )
    assert counts["lost"] == 1
    assert counts["created"] == 1  # track 2 is new
    t = mgr.get_target("S0-T0001")
    assert t.state == TargetState.LOST


def test_update_targets_remove_old_lost():
    """update_targets removes LOST targets beyond removal_threshold."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), last_seen=0.0)
    mgr.mark_lost(t.target_id, timestamp=1.0)
    # Feed only track 2 — track 1 is LOST and old
    counts = mgr.update_targets(
        [_make_track(2)], timestamp=100.0,
        stale_threshold=5.0, removal_threshold=30.0,
    )
    assert counts["removed"] == 1
    assert mgr.get_target("S0-T0001") is None


def test_update_targets_full_scenario():
    """Multi-frame scenario: create, update, lose, recover, remove."""
    mgr = _make_manager()

    # Frame 1: tracks 1, 2, 3 → create 3 targets
    r1 = mgr.update_targets([_make_track(i) for i in (1, 2, 3)], timestamp=0.0)
    assert r1["created"] == 3
    assert mgr.target_count == 3

    # Frame 2 (t=10): tracks 1, 2 → track 3 stale, track 4 new
    r2 = mgr.update_targets(
        [_make_track(i) for i in (1, 2, 4)], timestamp=10.0, stale_threshold=5.0,
    )
    assert r2["updated"] == 2
    assert r2["lost"] == 1
    assert r2["created"] == 1
    assert mgr.target_count == 4

    # Frame 3 (t=20): tracks 2, 3, 4 → track 1 stale, track 3 recovered
    r3 = mgr.update_targets(
        [_make_track(i) for i in (2, 3, 4)], timestamp=20.0, stale_threshold=5.0,
    )
    assert r3["recovered"] == 1
    assert r3["lost"] == 1  # track 1 goes stale
    assert r3["updated"] == 2

    # Frame 4 (t=100): tracks 2, 3, 4 → track 1 LOST and old → removed
    r4 = mgr.update_targets(
        [_make_track(i) for i in (2, 3, 4)], timestamp=100.0,
        stale_threshold=5.0, removal_threshold=30.0,
    )
    assert r4["removed"] == 1
    assert mgr.get_target("S0-T0001") is None
    assert mgr.target_count == 3


def test_update_targets_empty_input():
    """update_targets with empty list returns all-zero counts."""
    mgr = _make_manager()
    counts = mgr.update_targets([], timestamp=0.0)
    assert counts == {"created": 0, "updated": 0, "recovered": 0,
                      "lost": 0, "removed": 0}


def test_update_targets_returns_counts():
    """update_targets returns a dict with all 5 count keys."""
    mgr = _make_manager()
    counts = mgr.update_targets([_make_track(1)], timestamp=0.0)
    assert set(counts.keys()) == {"created", "updated", "recovered",
                                   "lost", "removed"}


def test_update_targets_locked_target_not_lost():
    """A LOCKED target is not marked lost even if unmatched."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), last_seen=0.0)
    mgr.lock_target(t.target_id)
    counts = mgr.update_targets([], timestamp=100.0, stale_threshold=5.0)
    assert counts["lost"] == 0  # LOCKED targets are not auto-lost
    assert t.state == TargetState.LOCKED


# ---------------------------------------------------------------------------
# 7. 查询方法
# ---------------------------------------------------------------------------

def test_get_targets_by_state():
    """get_targets_by_state filters correctly."""
    mgr = _make_manager()
    t1 = mgr.create_target(track=_make_track(1))
    t2 = mgr.create_target(track=_make_track(2))
    mgr.lock_target(t2.target_id)
    t3 = mgr.create_target(track=None)  # LOST

    active = mgr.get_active_targets()
    locked = mgr.get_locked_targets()
    lost = mgr.get_lost_targets()
    assert len(active) == 1
    assert len(locked) == 1
    assert len(lost) == 1
    assert active[0].target_id == t1.target_id
    assert locked[0].target_id == t2.target_id
    assert lost[0].target_id == t3.target_id


def test_get_all_targets():
    """get_all_targets returns all targets in the store."""
    mgr = _make_manager()
    mgr.create_target(track=_make_track(1))
    mgr.create_target(track=_make_track(2))
    all_t = mgr.get_all_targets()
    assert len(all_t) == 2


def test_get_target_not_found():
    """get_target returns None for nonexistent id."""
    mgr = _make_manager()
    assert mgr.get_target("nonexistent") is None


# ---------------------------------------------------------------------------
# 8. 更新方法
# ---------------------------------------------------------------------------

def test_update_last_seen():
    """update_last_seen sets the timestamp."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    assert mgr.update_last_seen(t.target_id, 42.0) is True
    assert t.last_seen == 42.0


def test_update_last_seen_not_found():
    """update_last_seen returns False for nonexistent id."""
    mgr = _make_manager()
    assert mgr.update_last_seen("nonexistent", 1.0) is False


def test_update_track():
    """update_track replaces the target's track."""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1))
    new_track = _make_track(99)
    assert mgr.update_track(t.target_id, new_track) is True
    assert t.track is new_track


def test_update_track_not_found():
    """update_track returns False for nonexistent id."""
    mgr = _make_manager()
    assert mgr.update_track("nonexistent", _make_track(1)) is False


# ---------------------------------------------------------------------------
# 9. Properties 和 __repr__
# ---------------------------------------------------------------------------

def test_target_count_property():
    """target_count reflects the number of stored targets."""
    mgr = _make_manager()
    assert mgr.target_count == 0
    mgr.create_target(track=_make_track(1))
    assert mgr.target_count == 1
    mgr.create_target(track=_make_track(2))
    assert mgr.target_count == 2


def test_store_property():
    """store property returns the underlying TargetStore."""
    mgr = _make_manager()
    assert isinstance(mgr.store, TargetStore)


def test_lifecycle_property():
    """lifecycle property returns the TargetLifecycleManager."""
    mgr = _make_manager()
    assert isinstance(mgr.lifecycle, TargetLifecycleManager)


def test_repr():
    """__repr__ includes target count and next id."""
    mgr = _make_manager()
    mgr.create_target(track=_make_track(1))
    r = repr(mgr)
    assert "TargetManager(" in r
    assert "target_count=1" in r


def test_tick_noop():
    """tick does not raise and does not change state."""
    mgr = _make_manager()
    mgr.create_target(track=_make_track(1))
    mgr.tick(timestamp=1.0)  # should not raise
    assert mgr.target_count == 1


# ---------------------------------------------------------------------------
# 10. 线程安全
# ---------------------------------------------------------------------------

def test_thread_safety_concurrent_create():
    """Concurrent create_target calls do not corrupt state."""
    mgr = _make_manager()
    errors: list[str] = []

    def worker(thread_id: int) -> None:
        try:
            for i in range(20):
                mgr.create_target(track=_make_track(thread_id * 100 + i))
        except Exception as exc:
            errors.append(f"thread-{thread_id}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"Thread errors: {errors}"
    assert mgr.target_count == 80  # 4 threads * 20 targets


def test_thread_safety_concurrent_lifecycle():
    """Concurrent lock/unlock on different targets is safe."""
    mgr = _make_manager()
    targets = [mgr.create_target(track=_make_track(i)) for i in range(20)]
    errors: list[str] = []

    def worker(idx: int) -> None:
        try:
            t = targets[idx]
            for _ in range(10):
                mgr.lock_target(t.target_id)
                mgr.unlock_target(t.target_id)
        except Exception as exc:
            errors.append(f"worker-{idx}: {exc}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"Thread errors: {errors}"
    for t in targets:
        assert t.state == TargetState.ACTIVE


# ---------------------------------------------------------------------------
# 11. Store 注入
# ---------------------------------------------------------------------------

def test_store_injection():
    """Two managers sharing the same store see each other's targets."""
    shared = TargetStore()
    mgr_a = TargetManager(store=shared)
    mgr_b = TargetManager(store=shared)
    t = mgr_a.create_target(track=_make_track(1))
    assert mgr_b.get_target(t.target_id) is t


# ---------------------------------------------------------------------------
# 12. 多 Slot 隔离 (Multi-Slot Isolation)
# ---------------------------------------------------------------------------
# 审计问题：多个摄像头 slot 共用一个 TargetManager 时，slot0 track_id=1
# 与 slot1 track_id=1 会产生身份冲突。以下测试验证 (slot_id, track_id)
# 二元组索引 + S{slot_id}-T{NNNN} 命名能正确隔离不同 slot 的目标。
# ---------------------------------------------------------------------------


def test_multislot_same_track_id_no_collision():
    """两个 slot 各自 track_id=1 不会冲突，target_id 不同且共存。"""
    mgr = _make_manager()
    t0 = mgr.create_target(track=_make_track(1), slot_id=0)
    t1 = mgr.create_target(track=_make_track(1), slot_id=1)
    assert t0.target_id != t1.target_id
    assert t0.target_id.startswith("S0-")
    assert t1.target_id.startswith("S1-")
    assert t0.slot_id == 0
    assert t1.slot_id == 1
    assert mgr.target_count == 2
    # 两个 target 都能独立查到
    assert mgr.get_target(t0.target_id) is t0
    assert mgr.get_target(t1.target_id) is t1


def test_multislot_update_does_not_cross_contaminate():
    """slot0 的 update_targets 不会影响 slot1 的目标。"""
    mgr = _make_manager()
    # slot0 和 slot1 都有 track_id=1
    t0 = mgr.create_target(track=_make_track(1), slot_id=0, last_seen=0.0)
    t1 = mgr.create_target(track=_make_track(1), slot_id=1, last_seen=0.0)

    # 只更新 slot0 的 track_id=1，时间戳推进
    counts = mgr.update_targets([_make_track(1)], slot_id=0, timestamp=100.0)
    assert counts["updated"] == 1

    # slot0 的 last_seen 被更新
    assert t0.last_seen == 100.0
    # slot1 的 last_seen 不受影响
    assert t1.last_seen == 0.0
    # slot1 仍为 ACTIVE（未被误标为 LOST）
    assert t1.state == TargetState.ACTIVE


def test_multislot_stale_only_affects_own_slot():
    """slot 的过期清理只作用于本 slot，不影响其他 slot 的 ACTIVE 目标。"""
    mgr = _make_manager()
    # slot0 有 track 1，slot1 也有 track 1
    t0 = mgr.create_target(track=_make_track(1), slot_id=0, last_seen=0.0)
    t1 = mgr.create_target(track=_make_track(1), slot_id=1, last_seen=0.0)

    # slot0 喂入空列表 + 大时间戳 → slot0 的 track1 应被标记 LOST
    counts = mgr.update_targets(
        [], slot_id=0, timestamp=100.0, stale_threshold=5.0,
    )
    assert counts["lost"] == 1
    assert t0.state == TargetState.LOST
    # slot1 不受影响
    assert t1.state == TargetState.ACTIVE


def test_multislot_empty_input_triggers_cleanup():
    """空 detections 仍触发该 slot 的过期清理（bypass if-tracks 修复回归）。"""
    mgr = _make_manager()
    t = mgr.create_target(track=_make_track(1), slot_id=2, last_seen=0.0)
    # 模拟 slot2 断流：喂入空列表
    counts = mgr.update_targets(
        [], slot_id=2, timestamp=50.0, stale_threshold=5.0,
    )
    assert counts["lost"] == 1
    assert t.state == TargetState.LOST


def test_multislot_recover_is_slot_scoped():
    """LOST 目标的恢复按 (slot_id, track_id) 匹配，不跨 slot 误恢复。"""
    mgr = _make_manager()
    # slot0 track1 LOST, slot1 track1 ACTIVE
    t0 = mgr.create_target(track=_make_track(1), slot_id=0, last_seen=0.0)
    t1 = mgr.create_target(track=_make_track(1), slot_id=1, last_seen=0.0)
    mgr.mark_lost(t0.target_id, timestamp=1.0)
    assert t0.state == TargetState.LOST
    assert t1.state == TargetState.ACTIVE

    # slot1 收到 track1 → 更新 slot1 的目标，不恢复 slot0 的
    counts = mgr.update_targets([_make_track(1)], slot_id=1, timestamp=10.0)
    assert counts["updated"] == 1
    assert counts["recovered"] == 0
    assert t0.state == TargetState.LOST  # slot0 仍 LOST
    assert t1.state == TargetState.ACTIVE


def test_multislot_full_scenario():
    """双 slot 并行场景：各自独立创建/更新/丢失/恢复。"""
    mgr = _make_manager()

    # Frame1: slot0 有 track1, slot1 有 track1 和 track2
    r0 = mgr.update_targets([_make_track(1)], slot_id=0, timestamp=0.0)
    r1 = mgr.update_targets(
        [_make_track(1), _make_track(2)], slot_id=1, timestamp=0.0,
    )
    assert r0["created"] == 1
    assert r1["created"] == 2
    assert mgr.target_count == 3  # S0-T0001, S1-T0002, S1-T0003

    # Frame2 (t=100): slot0 断流(空), slot1 仍有 track1,track2
    r0b = mgr.update_targets(
        [], slot_id=0, timestamp=100.0, stale_threshold=5.0,
    )
    r1b = mgr.update_targets(
        [_make_track(1), _make_track(2)], slot_id=1, timestamp=100.0,
    )
    assert r0b["lost"] == 1  # slot0 的目标过期
    assert r1b["updated"] == 2  # slot1 的两个目标正常更新

    # slot0 的目标 LOST，slot1 的两个目标仍 ACTIVE
    slot0_targets = [t for t in mgr.get_all_targets() if t.slot_id == 0]
    slot1_targets = [t for t in mgr.get_all_targets() if t.slot_id == 1]
    assert all(t.state == TargetState.LOST for t in slot0_targets)
    assert all(t.state == TargetState.ACTIVE for t in slot1_targets)
    assert len(slot1_targets) == 2


def test_track_to_target_slot_id_in_id():
    """track_to_target 生成的 target_id 包含 slot_id 前缀。"""
    from visioncore.target_manager import track_to_target
    t0 = track_to_target(_make_track(5), slot_id=0)
    t1 = track_to_target(_make_track(5), slot_id=3)
    assert t0.target_id == "S0-T0005"
    assert t1.target_id == "S3-T0005"
    assert t0.slot_id == 0
    assert t1.slot_id == 3
    assert t0.target_id != t1.target_id  # 不同 slot，同 track_id，不冲突


def test_tracks_to_targets_slot_scoped():
    """tracks_to_targets 批量转换时 slot_id 一致应用到所有目标。"""
    from visioncore.target_manager import tracks_to_targets
    tracks = [_make_track(1), _make_track(2), _make_track(3)]
    targets = tracks_to_targets(tracks, slot_id=2, timestamp=5.0)
    assert len(targets) == 3
    assert [t.target_id for t in targets] == ["S2-T0001", "S2-T0002", "S2-T0003"]
    assert all(t.slot_id == 2 for t in targets)
    assert all(t.last_seen == 5.0 for t in targets)


def test_multislot_monotonic_id_global_unique():
    """跨 slot 的单调计数器保证 target_id 全局唯一。"""
    mgr = _make_manager()
    ids = set()
    for slot in range(4):
        for tid in range(3):
            t = mgr.create_target(track=_make_track(tid), slot_id=slot)
            assert t.target_id not in ids, f"Duplicate id: {t.target_id}"
            ids.add(t.target_id)
    assert len(ids) == 12
    assert mgr.target_count == 12


# ---------------------------------------------------------------------------
# 13. ID 工厂统一性 + 检测转换器
# ---------------------------------------------------------------------------


def test_create_target_id_format():
    """create_target_id 生成 S{slot}-T{NNNN} 格式，4 位补零。"""
    from visioncore.target_manager import create_target_id
    assert create_target_id(0, 1) == "S0-T0001"
    assert create_target_id(1, 1) == "S1-T0001"
    assert create_target_id(3, 99) == "S3-T0099"
    assert create_target_id(0, 10000) == "S0-T10000"  # 超过 4 位不截断


def test_id_factory_shared_by_converter_and_manager():
    """converter 和 manager 生成同格式 ID（统一性验证）。"""
    from visioncore.target_manager import create_target_id, track_to_target
    mgr = _make_manager()
    # manager 路径：create_target → _generate_id → create_target_id
    t_mgr = mgr.create_target(track=_make_track(1), slot_id=0)
    # converter 路径：track_to_target → create_target_id
    t_conv = track_to_target(_make_track(1), slot_id=0)
    # 两者的 local_id 都是 1，格式一致
    assert t_mgr.target_id == create_target_id(0, 1)
    assert t_conv.target_id == create_target_id(0, 1)
    # 格式完全相同（仅 local_id 来源不同：monotonic vs track_id）
    assert t_mgr.target_id == "S0-T0001"
    assert t_conv.target_id == "S0-T0001"


def test_detection_to_track_basic():
    """detection_to_track 正确转换 dict → Track。"""
    from visioncore.target_manager import detection_to_track
    det = {
        "track_id": 5, "x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4,
        "confidence": 0.88, "class_id": 0, "label": "person",
    }
    t = detection_to_track(det)
    assert t is not None
    assert t.track_id == 5
    assert t.detection.score == 0.88
    assert t.detection.class_name == "person"
    # 中心坐标 = (0.1+0.3)/2 = 0.2
    assert abs(t.detection.bbox.x - 0.2) < 1e-9
    assert abs(t.detection.bbox.w - 0.2) < 1e-9


def test_detection_to_track_no_track_id():
    """detection_to_track 对无 track_id 的 dict 返回 None。"""
    from visioncore.target_manager import detection_to_track
    assert detection_to_track({"x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1}) is None
    assert detection_to_track({"track_id": None}) is None


def test_detections_to_tracks_skips_untracked():
    """detections_to_tracks 跳过无 track_id 的条目。"""
    from visioncore.target_manager import detections_to_tracks
    dets = [
        {"track_id": 1, "x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1,
         "confidence": 0.9, "class_id": 0, "label": "p"},
        {"track_id": None, "x1": 0, "y1": 0, "x2": 0.1, "y2": 0.1},
        {"track_id": 2, "x1": 0.5, "y1": 0.5, "x2": 0.6, "y2": 0.6,
         "confidence": 0.8, "class_id": 0, "label": "p"},
    ]
    tracks = detections_to_tracks(dets)
    assert len(tracks) == 2
    assert [t.track_id for t in tracks] == [1, 2]


def test_detections_to_tracks_empty():
    """detections_to_tracks 对空列表返回空列表。"""
    from visioncore.target_manager import detections_to_tracks
    assert detections_to_tracks([]) == []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
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
