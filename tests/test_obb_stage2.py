"""OBB Tracker 第二阶段快速验证脚本。"""
from __future__ import annotations

import numpy as np

from ai.detection import Detection, polygon_iou, polygon_center
from ai.tracker import DetectionTracker


def test_polygon_iou_identical():
    sq = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    iou = polygon_iou(sq, sq)
    assert abs(iou - 1.0) < 1e-6, f"identical polygon IoU should be 1, got {iou}"


def test_polygon_iou_half_overlap():
    a = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    b = [(0.2, 0.1), (0.4, 0.1), (0.4, 0.3), (0.2, 0.3)]
    iou = polygon_iou(a, b)
    # 交集 0.1*0.2=0.02, 并集 0.04*2-0.02=0.06, iou=1/3
    assert abs(iou - 1.0 / 3.0) < 1e-6, f"expected ~0.333, got {iou}"


def test_polygon_center():
    sq = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
    cx, cy = polygon_center(sq)
    assert abs(cx - 0.2) < 1e-6 and abs(cy - 0.2) < 1e-6


def test_detection_polygon_methods():
    d1 = Detection(class_id=0, label="person", confidence=0.9,
                   bbox=(0.1, 0.1, 0.3, 0.3),
                   polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)])
    d2 = Detection(class_id=0, label="person", confidence=0.9,
                   bbox=(0.2, 0.1, 0.4, 0.3),
                   polygon=[(0.2, 0.1), (0.4, 0.1), (0.4, 0.3), (0.2, 0.3)])
    assert d1.has_polygon()
    assert abs(d1.polygon_iou_with(d2) - 1.0 / 3.0) < 1e-6


def test_tracker_obb_association():
    tracker = DetectionTracker(iou_threshold=0.25, max_misses=2)
    d1 = Detection(class_id=0, label="person", confidence=0.9,
                 bbox=(0.1, 0.1, 0.3, 0.3),
                 polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)])
    out = tracker.update([d1])
    assert len(out) == 1
    tid1 = out[0]["track_id"]

    # 同一目标向右小幅度移动，polygon IoU 仍高于阈值
    d2 = Detection(class_id=0, label="person", confidence=0.9,
                 bbox=(0.18, 0.1, 0.38, 0.3),
                 polygon=[(0.18, 0.1), (0.38, 0.1), (0.38, 0.3), (0.18, 0.3)])
    out = tracker.update([d2])
    assert len(out) == 1
    assert out[0]["track_id"] == tid1, f"expected same track {tid1}, got {out[0]['track_id']}"

    # 预测 polygon 应跟随 bbox 平移
    assert "polygon" in out[0]
    poly = out[0]["polygon"]
    assert len(poly) == 4
    assert all(0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0 for p in poly)


def test_tracker_obb_vs_bbox():
    """OBB 检测能更好区分相邻目标：第一帧 bbox 重叠但 polygon 分离，第二帧保持分离。"""
    tracker = DetectionTracker(iou_threshold=0.5, max_misses=2)
    # 两个相邻的竖直条，bbox 在水平方向有重叠但 polygon 不重叠
    d1 = Detection(class_id=0, label="person", confidence=0.9,
                 bbox=(0.1, 0.1, 0.3, 0.6),
                 polygon=[(0.2, 0.1), (0.3, 0.1), (0.3, 0.6), (0.2, 0.6)])
    d2 = Detection(class_id=0, label="person", confidence=0.85,
                 bbox=(0.2, 0.1, 0.4, 0.6),
                 polygon=[(0.3, 0.1), (0.4, 0.1), (0.4, 0.6), (0.3, 0.6)])
    out = tracker.update([d1, d2])
    assert len(out) == 2
    tids = {o["track_id"] for o in out}
    assert len(tids) == 2

    # 第二帧两个目标都稍微左移，但 bbox 仍然重叠；polygon 明确分离应维持两个 track
    d1_next = Detection(class_id=0, label="person", confidence=0.9,
                      bbox=(0.08, 0.1, 0.28, 0.6),
                      polygon=[(0.18, 0.1), (0.28, 0.1), (0.28, 0.6), (0.18, 0.6)])
    d2_next = Detection(class_id=0, label="person", confidence=0.85,
                      bbox=(0.28, 0.1, 0.48, 0.6),
                      polygon=[(0.28, 0.1), (0.38, 0.1), (0.38, 0.6), (0.28, 0.6)])
    out = tracker.update([d1_next, d2_next])
    assert len(out) == 2, f"expected 2 tracks, got {len(out)}"
    # 两个 track_id 应保持不变
    assert {o["track_id"] for o in out} == tids


def test_predict_step_polygon():
    tracker = DetectionTracker(iou_threshold=0.25, max_misses=2)
    d1 = Detection(class_id=0, label="person", confidence=0.9,
                 bbox=(0.1, 0.1, 0.3, 0.3),
                 polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)])
    tracker.update([d1])
    out = tracker.predict_step()
    assert len(out) == 1
    assert "polygon" in out[0] and len(out[0]["polygon"]) == 4


if __name__ == "__main__":
    test_polygon_iou_identical()
    test_polygon_iou_half_overlap()
    test_polygon_center()
    test_detection_polygon_methods()
    test_tracker_obb_association()
    test_tracker_obb_vs_bbox()
    test_predict_step_polygon()
    print("All OBB stage-2 tests passed.")
