#!/usr/bin/env python3
"""匹配策略单元测试：验证匈牙利/级联/贪婪匹配的正确性。

运行: PYTHONPATH=F:/VisionBata venv/Scripts/python.exe tests/test_matching.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.tracker import DetectionTracker


def make_det(x1, y1, x2, y2, conf=0.8, label="person"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf, "label": label, "keypoints": []}


def test_hungarian_basic_tracking():
    """匈牙利匹配：两目标稳定跟踪，ID 跨帧保持。"""
    t = DetectionTracker(matching_strategy="hungarian", max_misses=4)
    dets = [make_det(0.1, 0.1, 0.3, 0.3, conf=0.9),
            make_det(0.5, 0.5, 0.7, 0.7, conf=0.85)]
    out1 = t.update(dets)
    ids1 = {d["track_id"] for d in out1}
    assert len(out1) == 2

    # 第二帧轻微移动
    dets2 = [make_det(0.12, 0.12, 0.32, 0.32, conf=0.88),
             make_det(0.52, 0.52, 0.72, 0.72, conf=0.82)]
    out2 = t.update(dets2)
    ids2 = {d["track_id"] for d in out2}
    assert ids1 == ids2, f"ID 应保持不变: {ids1} != {ids2}"
    print("  test_hungarian_basic_tracking: PASS")


def test_cascade_suppresses_low_conf_noise():
    """级联匹配：低置信度检测不创建新 track。"""
    t = DetectionTracker(matching_strategy="cascade", high_conf_thresh=0.5, max_misses=4)
    # 第一帧：高置信度目标
    t.update([make_det(0.3, 0.3, 0.4, 0.4, conf=0.85)])
    assert len(t._tracks) == 1

    # 第二帧：高置信度目标 + 低置信度噪声
    t.update([
        make_det(0.3, 0.3, 0.4, 0.4, conf=0.85),
        make_det(0.7, 0.7, 0.75, 0.75, conf=0.3),  # 低置信度
    ])
    assert len(t._tracks) == 1, f"低置信度不应创建新track: {len(t._tracks)}"
    print("  test_cascade_suppresses_low_conf_noise: PASS")


def test_cascade_low_conf_associates_existing():
    """级联匹配：低置信度检测可关联已有 track。"""
    t = DetectionTracker(matching_strategy="cascade", high_conf_thresh=0.5, max_misses=4)
    t.update([make_det(0.3, 0.3, 0.4, 0.4, conf=0.85)])
    tid1 = list(t._tracks.keys())[0]

    # 第二帧：目标置信度降到 0.3（低于 high_conf_thresh）
    t.update([make_det(0.31, 0.31, 0.41, 0.41, conf=0.3)])
    assert len(t._tracks) == 1, "低置信度应关联已有 track"
    # track 仍存在（未被删除）
    assert tid1 in t._tracks
    print("  test_cascade_low_conf_associates_existing: PASS")


def test_greedy_fallback():
    """贪婪策略作为回退仍正常工作。"""
    t = DetectionTracker(matching_strategy="greedy", max_misses=4)
    out = t.update([make_det(0.1, 0.1, 0.3, 0.3, conf=0.9)])
    assert len(out) == 1
    print("  test_greedy_fallback: PASS")


def test_appearance_similarity_clamp():
    """外观相似度 clamp 到 [0,1]。"""
    t = DetectionTracker(matching_strategy="hungarian", max_misses=4)
    # 无直方图时返回 0.5
    from ai.tracker import _Track
    import numpy as np
    trk = _Track(1, "person", np.array([0.1, 0.1, 0.3, 0.3]), 0.9, [])
    sim = trk.appearance_similarity(None)
    assert 0.0 <= sim <= 1.0
    # 设置一个直方图
    trk.color_hist = np.ones(256, dtype=np.float32) * 0.5
    other = np.zeros(256, dtype=np.float32)
    sim2 = trk.appearance_similarity(other)
    assert 0.0 <= sim2 <= 1.0, f"相似度应 clamp 到 [0,1]: {sim2}"
    print("  test_appearance_similarity_clamp: PASS")


def test_occlusion_recovery():
    """遮挡恢复：max_misses=4 时，遮挡 3 帧后目标重新出现 ID 保持。"""
    t = DetectionTracker(matching_strategy="hungarian", max_misses=4, iou_threshold=0.15)
    t.update([make_det(0.3, 0.3, 0.4, 0.4, conf=0.85)])
    tid1 = list(t._tracks.keys())[0]

    # 遮挡 3 帧（无检测）
    for _ in range(3):
        t.update([])

    # 目标重新出现
    t.update([make_det(0.31, 0.31, 0.41, 0.41, conf=0.80)])
    assert tid1 in t._tracks, f"遮挡后 ID 应保持: tid={tid1}, tracks={list(t._tracks.keys())}"
    print("  test_occlusion_recovery: PASS")


def test_velocity_clip_configurable():
    """速度限幅可配置。"""
    import numpy as np
    from ai.tracker import _DetectionUKF
    # 默认 0.30
    ukf1 = _DetectionUKF(np.array([0.1, 0.1, 0.2, 0.2]))
    assert ukf1._velocity_clip == 0.30
    # 自定义 0.50
    ukf2 = _DetectionUKF(np.array([0.1, 0.1, 0.2, 0.2]), velocity_clip=0.50)
    assert ukf2._velocity_clip == 0.50
    print("  test_velocity_clip_configurable: PASS")


def test_set_tracker_params():
    """set_tracker_params 运行时更新参数。"""
    t = DetectionTracker(matching_strategy="hungarian", iou_threshold=0.25, max_misses=4)
    t.set_tracker_params(iou_threshold=0.40, max_misses=6, matching_strategy="cascade")
    assert t._iou_threshold == 0.40
    assert t._max_misses == 6
    assert t._matching_strategy == "cascade"
    print("  test_set_tracker_params: PASS")


if __name__ == "__main__":
    print("Running matching strategy tests...")
    test_hungarian_basic_tracking()
    test_cascade_suppresses_low_conf_noise()
    test_cascade_low_conf_associates_existing()
    test_greedy_fallback()
    test_appearance_similarity_clamp()
    test_occlusion_recovery()
    test_velocity_clip_configurable()
    test_set_tracker_params()
    print("\nAll matching tests passed.")
