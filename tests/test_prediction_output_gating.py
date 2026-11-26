from __future__ import annotations

import pytest

from ai.tracker import DetectionTracker


def _det(x1: float = 0.30, y1: float = 0.30, x2: float = 0.50, y2: float = 0.50) -> dict:
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": 0.9,
        "label": "person",
    }


def _tracker(backend: str, **kwargs):
    if backend == "rust":
        pytest.importorskip("omintrack")
    return DetectionTracker(
        backend=backend,
        matching_strategy="hungarian",
        redetect_enabled=False,
        reid_backend=None,
        **kwargs,
    )


def _internal_track_count(tracker, backend: str) -> int:
    if backend == "python":
        return len(tracker._tracks)
    return tracker._core.track_count()


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_hidden_prediction_keeps_internal_track_and_reappears_with_same_id(backend: str):
    tracker = _tracker(backend, max_misses=4)

    tid = tracker.update([_det()])[0]["track_id"]
    assert len(tracker.predict_step()) == 1

    # The first unmatched inference update suppresses output, not the track.
    assert tracker.update([]) == []
    assert _internal_track_count(tracker, backend) == 1
    assert tracker.predict_step() == []
    assert _internal_track_count(tracker, backend) == 1

    visible = tracker.update([_det(x1=0.305, x2=0.505)])
    assert len(visible) == 1
    assert visible[0]["track_id"] == tid
    assert visible[0]["predicted_only"] is False
    assert _internal_track_count(tracker, backend) == 1


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_legacy_output_limit_preserves_previous_visibility_and_expiry(backend: str):
    tracker = _tracker(backend, max_misses=4, max_output_misses=4)

    tid = tracker.update([_det()])[0]["track_id"]
    missed = tracker.update([])
    assert len(missed) == 1
    assert missed[0]["track_id"] == tid
    assert missed[0]["predicted_only"] is True
    assert len(tracker.predict_step()) == 1

    for _ in range(4):
        tracker.update([])
    assert _internal_track_count(tracker, backend) == 0
    assert tracker.update([]) == []


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_output_limit_runtime_setter_is_independent_of_track_retention(backend: str):
    tracker = _tracker(backend, max_misses=4, max_output_misses=1)
    tracker.update([_det()])

    assert len(tracker.update([])) == 1  # misses=1 remains visible
    assert tracker.update([]) == []      # misses=2 is output-gated
    assert _internal_track_count(tracker, backend) == 1

    tracker.set_tracker_params(max_output_misses=4)
    assert len(tracker.update([])) == 1  # misses=3 is visible under the relaxed limit

    tracker.set_tracker_params(max_output_misses=0)
    assert tracker.update([]) == []      # misses=4 is hidden but retained
    assert _internal_track_count(tracker, backend) == 1

    tracker.update([])                   # misses=5 expires the internal track
    assert _internal_track_count(tracker, backend) == 0


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_negative_max_output_misses_is_clamped_to_zero(backend: str):
    tracker = _tracker(backend, max_misses=4, max_output_misses=-3)

    tid = tracker.update([_det()])[0]["track_id"]

    # 构造期负数被钳制为 0：首次 miss 即隐藏，内部 track 仍保留。
    assert tracker.update([]) == []
    assert _internal_track_count(tracker, backend) == 1
    assert tracker.predict_step() == []
    assert _internal_track_count(tracker, backend) == 1

    visible = tracker.update([_det(x1=0.305, x2=0.505)])
    assert len(visible) == 1
    assert visible[0]["track_id"] == tid

    # 运行时 setter 的负数同样钳制为 0。
    tracker.set_tracker_params(max_output_misses=2)
    assert len(tracker.update([])) == 1  # misses=1 在放宽限值下可见
    tracker.set_tracker_params(max_output_misses=-1)
    assert tracker.update([]) == []      # misses=2 被钳制后的 0 门控隐藏
    assert _internal_track_count(tracker, backend) == 1


@pytest.mark.parametrize("backend", ["python", "rust"])
def test_output_gating_interacts_with_min_hits_to_show(backend: str):
    tracker = _tracker(
        backend, max_misses=4, max_output_misses=0, min_hits_to_show=2,
    )

    # hits=1 未达输出阈值：即使 misses=0 也不输出，但 track 已内部建立。
    assert tracker.update([_det()]) == []
    assert _internal_track_count(tracker, backend) == 1

    # hits=2 达到输出阈值：开始可见。
    first_visible = tracker.update([_det()])
    assert len(first_visible) == 1
    tid = first_visible[0]["track_id"]

    # 首次 miss：max_output_misses=0 立即隐藏，内部保留。
    assert tracker.update([]) == []
    assert _internal_track_count(tracker, backend) == 1
    assert tracker.predict_step() == []

    # correct 命中后恢复原 ID 可见。
    visible = tracker.update([_det(x1=0.305, x2=0.505)])
    assert len(visible) == 1
    assert visible[0]["track_id"] == tid
    assert visible[0]["predicted_only"] is False
