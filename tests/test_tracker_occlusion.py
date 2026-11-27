#!/usr/bin/env python3
"""跟踪增强批次 3（ORU + OCR + TCM）测试（docs/tracker-oru-ocr-tcm.md §3）。

覆盖：
  - ORU：丢失 N 帧恢复后回滚重放，状态与纯外推不同且更接近真实恢复观测；
    无快照/未丢失时不触发；
  - OCR：预测已偏离但最后观测与剩余检测重叠的 track 经二次关联恢复；
    阈值门控生效；OCR 先于 ROI 重检测；
  - TCM：置信度一致性奖励符号与量级；高/低分阶段 conf_pred 来源（滤波预测 vs
    线性外推）；第二阶段不施加；默认关闭零影响；
  - 双后端等价（开启三特性及全特性，坐标 1e-6、状态机精确）。

运行: venv/Scripts/python.exe -m pytest tests/test_tracker_occlusion.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import omintrack  # noqa: E402
from ai.tracker import DetectionTracker, _box_iou  # noqa: E402
from tests.test_tracker_rust_equivalence import _run_pair  # noqa: E402
from tests.test_tracker_snapshot import _det  # noqa: E402

BATCH3_KW = dict(oru_enabled=True, ocr_enabled=True, tcm_enabled=True)
ALL_KW = dict(nsa_enabled=True, ocm_enabled=True, hmiou_enabled=True, **BATCH3_KW)


def make_det(x1, y1, x2, y2, conf=0.9, label="person"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf, "label": label, "keypoints": []}


def det_cx(cx, conf=0.9):
    """w=0.1, h=0.4, y=[0.1,0.5] 的居中检测。"""
    return make_det(cx - 0.05, 0.10, cx + 0.05, 0.50, conf=conf)


# ---------------------------------------------------------------------------
# ORU（§2.1）
# ---------------------------------------------------------------------------

def _oru_scenario(oru_on: bool):
    """静止 5 帧 → 丢失 3 帧（目标以 0.03/帧右移）→ 恢复观测在真实位置。"""
    t = DetectionTracker(backend="python", oru_enabled=oru_on)
    for _ in range(5):
        t.update([det_cx(0.30)])
    for _ in range(3):
        t.update([])                       # 丢失（目标实际已右移 0.09）
    out = t.update([det_cx(0.39)])         # 恢复观测 = 真实位置
    return t, out


def test_oru_replay_differs_and_closer_to_recovery():
    t_off, _ = _oru_scenario(False)
    t_on, _ = _oru_scenario(True)
    trk_off = next(iter(t_off._tracks.values()))
    trk_on = next(iter(t_on._tracks.values()))
    # 回滚重放后 KF 状态与纯外推不同
    assert not np.allclose(trk_on.kalman.x, trk_off.kalman.x)
    # 且更接近真实恢复观测（cx=0.39）
    cx_off = (trk_off.box[0] + trk_off.box[2]) * 0.5
    cx_on = (trk_on.box[0] + trk_on.box[2]) * 0.5
    assert abs(cx_on - 0.39) < abs(cx_off - 0.39)


def test_oru_snapshot_saved_after_correct():
    t = DetectionTracker(backend="python", oru_enabled=True)
    trk = None
    t.update([det_cx(0.30)])
    trk = next(iter(t._tracks.values()))
    assert trk.kf_snapshot is None          # 创建不是 correct，无快照
    t.update([det_cx(0.31)])
    assert trk.kf_snapshot is not None      # correct 后保存 (x,P,Q,R)
    x, p, q, r = trk.kf_snapshot
    assert x.shape == (8, 1) and p.shape == (8, 8) and q.shape == (8, 8) and r.shape == (4, 4)


def test_oru_not_triggered_without_snapshot():
    """丢失前从未 correct（无快照）→ ORU 不触发，与关闭逐位一致。"""
    def run(oru_on):
        t = DetectionTracker(backend="python", oru_enabled=oru_on)
        t.update([det_cx(0.30)])            # 创建（非 correct）
        t.update([])
        t.update([])
        return t.update([det_cx(0.30)])     # 恢复：misses>0 但无快照
    o_off = run(False)
    o_on = run(True)
    assert len(o_off) == len(o_on) == 1
    for a, b in zip(o_off, o_on):
        assert a == b


def test_oru_not_triggered_when_not_lost():
    """misses==0 的正常匹配不触发 ORU（与关闭逐位一致）。"""
    def run(oru_on):
        t = DetectionTracker(backend="python", oru_enabled=oru_on)
        outs = []
        for f in range(5):
            outs.append(t.update([det_cx(0.30 + 0.01 * f)]))
        return outs
    for o_off, o_on in zip(run(False), run(True)):
        for a, b in zip(o_off, o_on):
            for k in ("x1", "y1", "x2", "y2", "confidence"):
                assert a[k] == b[k]


# ---------------------------------------------------------------------------
# OCR（§2.2）
# ---------------------------------------------------------------------------

def _ocr_scenario(ocr_on: bool, **kw):
    """右移（0.05/帧，逐帧可匹配）→ 丢失 3 帧（预测外推远离）→ 检测停在最后观测处。"""
    t = DetectionTracker(backend="python", ocr_enabled=ocr_on, **kw)
    for f in range(4):
        t.update([det_cx(0.30 + 0.05 * f)])   # 0.30→0.45，逐帧 IoU 可匹配
    for _ in range(3):
        t.update([])                           # 丢失：预测框外推至 ~0.59
    out = t.update([det_cx(0.45)])             # 目标停在 last_observation 处
    return t, out


def test_ocr_recovers_drifted_track():
    t_off, o_off = _ocr_scenario(False)
    t_on, o_on = _ocr_scenario(True)
    # 关闭：主匹配门控拒绝（预测已偏离）→ 检测新建 track
    assert len(t_off._tracks) == 2
    assert 2 in {o["track_id"] for o in o_off}
    # 开启：last_observation 与检测 IoU≈1.0 > 0.3 → 二次关联恢复原 ID
    assert len(t_on._tracks) == 1
    assert [o["track_id"] for o in o_on] == [1]


def test_ocr_threshold_gate():
    """IoU ≤ ocr_threshold 的剩余检测不被 OCR 关联（注入式确定性）。"""
    t = DetectionTracker(backend="python", ocr_enabled=True, ocr_threshold=0.3)
    t.update([det_cx(0.60)])
    trk = next(iter(t._tracks.values()))
    trk.misses = 2
    trk.last_observation = np.array([0.55, 0.10, 0.65, 0.50])
    # IoU = 0.25 < 0.3：不关联
    low = [make_det(0.61, 0.10, 0.71, 0.50)]
    assert _box_iou(trk.last_observation, np.array([0.61, 0.10, 0.71, 0.50])) < 0.3
    matched_tids: set[int] = set()
    matched_dis: set[int] = set()
    t._match_ocr(low, np.array([[0.61, 0.10, 0.71, 0.50]]), matched_tids, matched_dis)
    assert not matched_tids and not matched_dis
    # IoU ≈ 1.0 > 0.3：关联
    high = [make_det(0.55, 0.10, 0.65, 0.50)]
    t._match_ocr(high, np.array([[0.55, 0.10, 0.65, 0.50]]), matched_tids, matched_dis)
    assert matched_tids == {trk.track_id} and matched_dis == {0}


def test_ocr_only_lost_tracks():
    """活跃 track（misses==0）不参与 OCR。"""
    t = DetectionTracker(backend="python", ocr_enabled=True)
    t.update([det_cx(0.60)])
    trk = next(iter(t._tracks.values()))
    assert trk.misses == 0
    matched_tids: set[int] = set()
    matched_dis: set[int] = set()
    t._match_ocr([det_cx(0.60)], np.array([[0.55, 0.10, 0.65, 0.50]]),
                 matched_tids, matched_dis)
    assert not matched_tids


def test_ocr_runs_before_roi_redetect():
    """OCR 在 ROI 重检测之前：剩余检测可恢复时走 OCR（normal），不走重检测。

    丢失帧不带 callback（纯 miss 累积）；恢复帧同时提供 OCR 可用剩余检测与
    必成功的 redetect callback——若 ROI 在先会走 redetected，OCR 在先则 normal。
    """
    def redetect_cb(box, conf):
        return make_det(box[0], box[1], box[2], box[3], conf=conf)  # 总能"找回"

    t = DetectionTracker(
        backend="python", ocr_enabled=True,
        redetect_enabled=True, redetect_max_retries=2,
    )
    for f in range(4):
        t.update([det_cx(0.30 + 0.05 * f)])
    for _ in range(3):
        t.update([])                               # 无 callback：纯 miss
    out = t.update([det_cx(0.45)], redetect_callback=redetect_cb)
    trk = next(iter(t._tracks.values()))
    # OCR 用剩余检测恢复 → track_state 为 normal（非 redetected）
    assert [o["track_id"] for o in out] == [1]
    assert trk.track_state == "normal"


# ---------------------------------------------------------------------------
# TCM（§2.3）
# ---------------------------------------------------------------------------

def test_tcm_conf_pred_sources():
    """conf_pred 来源：高阶段=滤波一步预测（est+vel）；低阶段=两点线性外推。"""
    t = DetectionTracker(backend="python", tcm_enabled=True)
    t.update([det_cx(0.30, conf=0.8)])
    trk = next(iter(t._tracks.values()))
    assert trk.tcm_conf_est == pytest.approx(0.8)
    assert trk.tcm_conf_pred_high() == pytest.approx(0.8)   # vel=0
    assert trk.tcm_conf_pred_low() == pytest.approx(0.8)    # 无 prev → est
    t.update([det_cx(0.31, conf=0.9)])
    # vel = 0.7*0 + 0.3*(0.9-0.8) = 0.03
    assert trk.tcm_conf_vel == pytest.approx(0.03)
    assert trk.tcm_conf_pred_high() == pytest.approx(0.93)
    # 线性外推：2*0.9 - 0.8 = 1.0
    assert trk.tcm_conf_pred_low() == pytest.approx(1.0)


def test_tcm_consistency_reward_prefers_matching_conf():
    """一致性奖励：conf 一致的检测凭 TCM 项逆转几何劣势胜出（严格）。

    候选 A（索引 0）：几何更优但 conf=0.3（与轨迹 conf≈0.9 不一致）；
    候选 B（索引 1）：几何稍差但 conf=0.9（一致）。TCM 关闭时 A 凭几何胜出，
    开启时一致性奖励（|0.3-0.9|·0.5=0.3 > 几何差）使 B 逆转。
    """
    def run(tcm_on):
        t = DetectionTracker(backend="python", tcm_enabled=tcm_on, tcm_weight=0.5)
        for _ in range(3):
            t.update([det_cx(0.35, conf=0.9)])
        dets = [det_cx(0.355, conf=0.3), det_cx(0.37, conf=0.9)]
        boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
        matched_tids, matched_dis = t._match_hungarian(
            dets, boxes, [None, None], np.array([0.3, 0.9]),
        )
        return matched_dis

    assert run(True) == {1}, "TCM 应使 conf 一致的检测逆转几何劣势胜出"
    assert run(False) == {0}, "TCM 关闭时几何更优者胜出"


def test_tcm_second_stage_not_applied():
    """cascade 第二阶段不施加 TCM 项（tcm_first_stage=False 时与关闭一致）。"""
    t = DetectionTracker(backend="python", tcm_enabled=True, tcm_weight=0.5)
    for _ in range(3):
        t.update([det_cx(0.35, conf=0.9)])
    # 同 test_tcm_consistency_reward：A 几何优/conf 差，B 几何差/conf 一致
    dets = [det_cx(0.355, conf=0.3), det_cx(0.37, conf=0.9)]
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
    matched_tids, matched_dis = t._match_hungarian(
        dets, boxes, [None, None], np.array([0.3, 0.9]),
        tcm_first_stage=False,
    )
    assert matched_dis == {0}, "第二阶段不施加 TCM：几何更优者胜出（与关闭一致）"


def test_tcm_default_off_zero_effect():
    """默认关闭 ⇒ 显式关闭与不传参数逐位一致。"""
    t1 = DetectionTracker(backend="python")
    t2 = DetectionTracker(backend="python", oru_enabled=False, ocr_enabled=False,
                          tcm_enabled=False)
    for f in range(6):
        dets = [] if f == 3 else [det_cx(0.30 + 0.01 * f, conf=0.8 + 0.02 * f)]
        o1 = t1.update(dets)
        o2 = t2.update(dets)
        assert len(o1) == len(o2)
        for a, b in zip(o1, o2):
            assert a == b


# ---------------------------------------------------------------------------
# 双后端等价（批次 3 开启 / 全特性开启）
# ---------------------------------------------------------------------------

def _occlusion_seq(n=60):
    """双目标 + 遮挡（连续空帧）+ 置信度起伏 + 恢复 的确定性序列。"""
    seq = []
    for f in range(n):
        dets = []
        # 目标 1：中段连续遮挡 4 帧后恢复
        if not (20 <= f < 24):
            dets.append(_det(0.2 + 0.006 * f, 0.4, 0.08, 0.16,
                             conf=0.55 + 0.4 * ((f % 5) / 4.0)))
        # 目标 2：周期性单帧 miss
        if f % 9 != 4:
            dets.append(_det(0.8 - 0.006 * f, 0.42, 0.08, 0.16,
                             conf=0.6 + 0.35 * ((f % 3) / 2.0)))
        seq.append(dets)
    return seq


def test_dual_backend_batch3_equivalent():
    """ORU+OCR+TCM 开启：python vs rust 逐帧等价（坐标 1e-6、状态机精确）。"""
    _run_pair(dict(BATCH3_KW), dict(BATCH3_KW), _occlusion_seq(), predict_steps=3)


def test_dual_backend_all_features_equivalent():
    """批次 1+2+3 全特性开启：python vs rust 逐帧等价。"""
    _run_pair(dict(ALL_KW), dict(ALL_KW), _occlusion_seq(), predict_steps=3)


def test_dual_backend_batch3_cascade_equivalent():
    kw = dict(matching_strategy="cascade", high_conf_thresh=0.6, **BATCH3_KW)
    _run_pair(dict(kw), dict(kw), _occlusion_seq(), predict_steps=2)


def test_dual_backend_set_tracker_params_batch3():
    """set_tracker_params 运行时开启批次 3，双端一致。"""
    py_trk = DetectionTracker(backend="python")
    rs_trk = omintrack.RustTracker()
    seq = _occlusion_seq(30)
    for dets in seq[:15]:
        py_trk.update(dets)
        rs_trk.update(dets)
    py_trk.set_tracker_params(**BATCH3_KW)
    rs_trk.set_tracker_params(**BATCH3_KW)
    for dets in seq[15:]:
        po = py_trk.update(dets)
        ro = rs_trk.update(dets)
        assert len(po) == len(ro)
        for a, b in zip(po, ro):
            assert a["track_id"] == b["track_id"]
            for k in ("x1", "y1", "x2", "y2", "confidence"):
                assert abs(a[k] - b[k]) <= 1e-6
