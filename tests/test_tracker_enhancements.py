#!/usr/bin/env python3
"""跟踪增强（NSA-KF + OCM + HMIoU）测试（docs/tracker-nsa-ocm-hmiou.md §3）。

覆盖：
  - NSA：R_eff 乘子单调性 / nsa_min 截断 / conf 缺省中性 / 默认关闭逐位一致；
  - OCM：angle_term 符号与量级 / velocity=None 时 ocm_cost=0 / 观测仅在匹配时写入；
  - HMIoU：hmiou ≤ IoU / 高度不重叠 h_iou=0 / 门控替换生效 / polygon 对不适用；
  - 双后端：三特性开启的等价测试（复用 equivalence 框架，坐标 1e-6、状态机精确）。

运行: venv/Scripts/python.exe -m pytest tests/test_tracker_enhancements.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import omintrack  # noqa: E402
from ai.tracker import (  # noqa: E402
    DetectionTracker,
    _DetectionUKF,
    _height_iou,
    _speed_direction,
)
from tests.test_tracker_rust_equivalence import _run_pair  # noqa: E402
from tests.test_tracker_snapshot import _det  # noqa: E402

ENH_KW = dict(nsa_enabled=True, ocm_enabled=True, hmiou_enabled=True)


def make_det(x1, y1, x2, y2, conf=0.9, label="person"):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "confidence": conf, "label": label, "keypoints": []}


def moving_seq(n=12, conf=0.85):
    """单目标匀速右移序列。"""
    return [[_det(0.2 + 0.01 * f, 0.4, 0.08, 0.16, conf=conf)] for f in range(n)]


# ---------------------------------------------------------------------------
# NSA-KF（§2.1）
# ---------------------------------------------------------------------------

def _nsa_filter(enabled=True, interval=1.0, nsa_min=0.1):
    kf = _DetectionUKF(np.array([0.1, 0.1, 0.3, 0.4]))
    kf.nsa_enabled = enabled
    kf.nsa_interval = interval
    kf.nsa_min = nsa_min
    return kf


def test_nsa_multiplier_segmented_semantics():
    """上游 HybridSORT 分段语义（P0-2 修正）。"""
    kf = _nsa_filter()
    # 高置信度分支（conf > 0.6）：(1-conf)·interval，缩 R
    assert kf._nsa_multiplier(0.9) == pytest.approx(0.1)
    assert kf._nsa_multiplier(0.9) < 1.0
    assert kf._nsa_multiplier(0.61) == pytest.approx(0.39)
    # 低置信度分支（conf <= 0.6）：nsa_interval_sec=2.0，放大 R 降权观测
    assert kf._nsa_multiplier(0.3) == pytest.approx(2.0)
    assert kf._nsa_multiplier(0.0) == pytest.approx(2.0)
    # 边界 conf=0.6 走 interval_sec（严格大于才进高置信度分支）
    assert kf._nsa_multiplier(0.6) == pytest.approx(2.0)
    # nsa_min 仅在高置信度分支截断：conf→1 乘子不退化到 0（保护性）
    assert kf._nsa_multiplier(1.0) == pytest.approx(0.1)
    assert kf._nsa_multiplier(0.999) == pytest.approx(0.1)
    # 低置信度分支不受 nsa_min 影响
    kf2 = _nsa_filter(nsa_min=0.5)
    assert kf2._nsa_multiplier(0.3) == pytest.approx(2.0)
    # interval 缩放（高置信度分支）
    kf3 = _nsa_filter(interval=2.0)
    assert kf3._nsa_multiplier(0.9) == pytest.approx(0.2)


def test_nsa_multiplier_none_is_neutral():
    """conf=None → 1.0（无 NSA，真正中性，缺省安全）。"""
    kf = _nsa_filter()
    assert kf._nsa_multiplier(None) == 1.0


def test_nsa_disabled_multiplier_is_one():
    kf = _nsa_filter(enabled=False)
    assert kf._nsa_multiplier(0.0) == 1.0
    assert kf._nsa_multiplier(0.7) == 1.0
    assert kf._nsa_multiplier(None) == 1.0


def test_nsa_conf_default_neutral():
    """conf 缺省 None（真正中性）：correct(box) 与关闭 NSA 逐位一致。"""
    box = np.array([0.12, 0.11, 0.32, 0.42])
    kf1 = _nsa_filter()                    # NSA 开启，conf=None
    kf2 = _nsa_filter(enabled=False)       # NSA 关闭基线
    kf1.predict(); kf2.predict()
    out1 = kf1.correct(box)                # conf 缺省 None
    out2 = kf2.correct(box, 0.3)           # 关闭时 conf 不影响
    np.testing.assert_array_equal(out1, out2)
    # 显式 None 与缺省一致（用未更新过的新滤波器）
    kf3 = _nsa_filter()
    kf3.predict()
    np.testing.assert_array_equal(kf3.correct(box, None), out1)


def test_nsa_low_conf_trusts_observation_less():
    """与无 NSA 基线对比：低 conf 时 R 放大（更新更小），高 conf 时 R 缩小（更新更大）。"""
    box = np.array([0.20, 0.20, 0.40, 0.50])  # 相对预测明显偏移的观测

    def update_magnitude(nsa_on, conf):
        kf = _nsa_filter(enabled=nsa_on)
        kf.predict()
        pred = kf.to_box().copy()
        return float(np.abs(kf.correct(box, conf) - pred).sum())

    base = update_magnitude(False, 0.5)          # 无 NSA 基线
    low = update_magnitude(True, 0.3)            # R×2.0 → 更信预测
    high = update_magnitude(True, 0.9)           # R×0.1 → 更信观测
    assert 0.0 < low < base < high


def test_nsa_default_off_bit_identical():
    """默认关闭 ⇒ 显式关闭与不传参数逐位一致。"""
    t1 = DetectionTracker(backend="python")
    t2 = DetectionTracker(backend="python", nsa_enabled=False, ocm_enabled=False,
                          hmiou_enabled=False)
    for dets in moving_seq(10):
        o1 = t1.update(dets)
        o2 = t2.update(dets)
        assert len(o1) == len(o2)
        for a, b in zip(o1, o2):
            assert a == b  # dict 逐值精确相等


def test_nsa_mcukf_manifold_accept_conf():
    """MCUKF / Manifold 同一 correct 入口接受 conf；缺省 None 与关闭逐位一致。"""
    from ai.tracker import _DetectionMCUKF, _DetectionManifoldUKF
    box = np.array([0.12, 0.11, 0.32, 0.42])
    for cls in (_DetectionMCUKF, _DetectionManifoldUKF):
        # NSA 开启 + conf=None（缺省）== NSA 关闭
        kf1, kf2 = cls(np.array([0.1, 0.1, 0.3, 0.4])), cls(np.array([0.1, 0.1, 0.3, 0.4]))
        kf1.nsa_enabled = True
        kf1.predict(); kf2.predict()
        np.testing.assert_array_equal(kf1.correct(box), kf2.correct(box, 0.5))
        # 开启 NSA：高置信度更新幅度 > 低置信度（R 缩放方向正确）
        kf3, kf4 = cls(np.array([0.1, 0.1, 0.3, 0.4])), cls(np.array([0.1, 0.1, 0.3, 0.4]))
        for kf in (kf3, kf4):
            kf.nsa_enabled = True
        kf3.predict(); kf4.predict()
        pred3 = kf3.to_box().copy()
        pred4 = kf4.to_box().copy()
        far = np.abs(kf3.correct(np.array([0.2, 0.2, 0.4, 0.5]), 0.9) - pred3).sum()
        near = np.abs(kf4.correct(np.array([0.2, 0.2, 0.4, 0.5]), 0.3) - pred4).sum()
        assert far > near > 0.0


# ---------------------------------------------------------------------------
# OCM（§2.2）
# ---------------------------------------------------------------------------

def test_ocm_speed_direction():
    prev = np.array([0.0, 0.0, 0.1, 0.1])
    cur_right = np.array([0.1, 0.0, 0.2, 0.1])
    vy, vx = _speed_direction(prev, cur_right)
    assert (vy, vx) == (pytest.approx(0.0), pytest.approx(1.0))
    cur_down = np.array([0.0, 0.1, 0.1, 0.2])
    vy, vx = _speed_direction(prev, cur_down)
    assert (vy, vx) == (pytest.approx(1.0), pytest.approx(0.0))
    # 零位移 → None（保护性处理）
    assert _speed_direction(prev, prev.copy()) is None


def test_ocm_observations_only_on_match():
    """观测仅在匹配成功时写入；纯 miss/predict 帧不写入。"""
    t = DetectionTracker(backend="python", ocm_enabled=True)
    t.update([make_det(0.1, 0.1, 0.2, 0.3)])
    trk = next(iter(t._tracks.values()))
    assert trk.observations == {}          # 创建帧不计观测（仅初始化 last_observation）
    assert trk.velocity is None
    t.update([make_det(0.11, 0.1, 0.21, 0.3)])
    assert len(trk.observations) == 1      # 第一次匹配
    assert trk.velocity is not None        # 由 last_observation → 本次观测得到方向
    obs_snapshot = dict(trk.observations)
    t.update([])                           # miss 帧：不写入
    assert trk.observations == obs_snapshot
    t.predict_step()                       # 跳帧预测：不写入
    assert trk.observations == obs_snapshot


def test_ocm_observation_window_pruned():
    t = DetectionTracker(backend="python", ocm_enabled=True, ocm_delta_t=2)
    for f in range(8):
        t.update([make_det(0.1 + 0.01 * f, 0.1, 0.2 + 0.01 * f, 0.3)])
    trk = next(iter(t._tracks.values()))
    ages = sorted(trk.observations)
    assert len(ages) <= 3                  # 仅保留最近 delta_t+1 个 age
    assert ages[-1] == trk.age             # 最新观测在当前 age


def test_ocm_angle_term_sign_and_magnitude():
    """angle_term = (π/2 - |θ|)/π：同向 +0.5、垂直 0、反向 -0.5。"""
    def angle_term(v, d):
        cos_sim = max(-1.0, min(1.0, v[0] * d[0] + v[1] * d[1]))
        return (math.pi * 0.5 - abs(math.acos(cos_sim))) / math.pi

    assert angle_term((0.0, 1.0), (0.0, 1.0)) == pytest.approx(0.5)
    assert angle_term((0.0, 1.0), (1.0, 0.0)) == pytest.approx(0.0)
    assert angle_term((0.0, 1.0), (0.0, -1.0)) == pytest.approx(-0.5)


def test_ocm_prefers_direction_consistent_detection():
    """OCM 开启后，方向一致的检测获得更高 score 并被匈牙利选中（注入式确定性场景）。

    track box=[0.30,0.40]、velocity 向右、age-3 观测中心 0.37；
    两个对称候选（IoU 相同）：右侧 det（D 同向，angle_term=+0.5）与
    左侧 det（D 反向，angle_term=-0.5）→ OCM 使右侧严格胜出。
    """
    t = DetectionTracker(backend="python", ocm_enabled=True, ocm_weight=0.5)
    t.update([make_det(0.30, 0.10, 0.40, 0.50, conf=0.9)])
    trk = next(iter(t._tracks.values()))
    trk.age = 5
    trk.velocity = (0.0, 1.0)  # 向右
    trk.observations = {2: np.array([0.32, 0.10, 0.42, 0.50])}  # 中心 0.37
    trk.last_observation = np.array([0.32, 0.10, 0.42, 0.50])
    det_left = make_det(0.27, 0.10, 0.37, 0.50, conf=0.9)   # cx 0.32 < 0.37 → 反向
    det_right = make_det(0.33, 0.10, 0.43, 0.50, conf=0.9)  # cx 0.38 > 0.37 → 同向
    dets = [det_left, det_right]
    boxes = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets])
    matched_tids, matched_dis = t._match_hungarian(
        dets, boxes, [None, None], np.array([0.9, 0.9]),
    )
    assert matched_tids == {trk.track_id}
    assert matched_dis == {1}, "OCM 应使方向一致的右侧检测胜出（严格）"


def test_ocm_negative_cost_pair_still_accepted():
    """P0-1 回归：OCM 负代价使 cost>0.999 时，门控通过的合法 pair 不得被丢。

    构造 iou≈0.08、速度反平行（θ=π → angle_term=-0.5）、det_conf=0.95、
    ocm_weight=1.0 的 track×det：score≈0.16-0.475<0 ⇒ cost>0.999，
    旧的 cost<0.999 判据会误丢；修复后以 accepted 掩码判定仍接受。
    """
    t = DetectionTracker(backend="python", ocm_enabled=True, ocm_weight=1.0)
    t.update([make_det(0.30, 0.10, 0.40, 0.50, conf=0.95)])  # track box=[0.30,0.10,0.40,0.50]
    trk = next(iter(t._tracks.values()))
    # 注入确定性 OCM 状态：速度向右，age-3 观测中心在 det 右侧（反平行）
    trk.age = 5
    trk.velocity = (0.0, 1.0)
    trk.observations = {2: np.array([0.32, 0.10, 0.42, 0.50])}
    trk.last_observation = np.array([0.32, 0.10, 0.42, 0.50])
    # det：中心 0.2648（prev_obs 中心 0.37 左侧 → θ=π）；
    # 与 track box iou≈0.08（>0.05 且中心距 0.085<0.125 过门控回退）
    det = make_det(0.2148, 0.10, 0.3148, 0.50, conf=0.95)
    dets = [det]
    boxes = np.array([[det["x1"], det["y1"], det["x2"], det["y2"]]])
    matched_tids, matched_dis = t._match_hungarian(dets, boxes, [None], np.array([0.95]))
    assert matched_tids == {trk.track_id}, "OCM 负代价的合法 pair 被误丢（P0-1 回归）"
    assert matched_dis == {0}


def test_ocm_negative_cost_pair_accepted_dual_backend():
    """P0-1 双端：匀速右移后接反平行低 IoU 检测，双端均保持接受（单 track、同 ID）。"""
    def det(cx, conf=0.95):
        return make_det(cx - 0.05, 0.10, cx + 0.05, 0.50, conf=conf)

    def run(backend):
        t = DetectionTracker(backend=backend, ocm_enabled=True, ocm_weight=1.0)
        for f in range(5):
            t.update([det(0.30 + 0.02 * f)])
        out = t.update([det(0.31)])  # 反平行 + iou≈0.08
        return t, out

    tp, op = run("python")
    tr, orr = run("rust")
    assert len(tp._tracks) == 1, "python：反平行 det 应匹配既有 track 而非新建"
    assert [o["track_id"] for o in op] == [1]
    assert [o["track_id"] for o in orr] == [1]
    for a, b in zip(op, orr):
        for k in ("x1", "y1", "x2", "y2", "confidence"):
            assert abs(a[k] - b[k]) <= 1e-6


def test_ocm_velocity_none_cost_zero():
    """velocity=None 时 OCM 项为 0：首帧匹配不受 OCM 影响（双跑逐位一致）。"""
    def run():
        t = DetectionTracker(backend="python", ocm_enabled=True)
        t.update([make_det(0.1, 0.1, 0.2, 0.3)])
        t.update([make_det(0.11, 0.1, 0.21, 0.3)])   # 首次匹配时 velocity 尚为 None
        return next(iter(t._tracks.values())).box.copy()

    t_ref = DetectionTracker(backend="python", ocm_enabled=False)
    t_ref.update([make_det(0.1, 0.1, 0.2, 0.3)])
    t_ref.update([make_det(0.11, 0.1, 0.21, 0.3)])
    np.testing.assert_array_equal(run(), next(iter(t_ref._tracks.values())).box)


# ---------------------------------------------------------------------------
# HMIoU（§2.3）
# ---------------------------------------------------------------------------

def test_height_iou_properties():
    a = np.array([0.1, 0.1, 0.2, 0.5])
    # 完全相同 → 1
    assert _height_iou(a, a.copy()) == pytest.approx(1.0)
    # 高度不重叠 → 0（且 max(0,·) 防负）
    b = np.array([0.1, 0.6, 0.2, 0.9])
    assert _height_iou(a, b) == 0.0
    # hmiou ≤ IoU
    from ai.tracker import _box_iou
    c = np.array([0.1, 0.3, 0.2, 0.7])
    assert _box_iou(a, c) * _height_iou(a, c) <= _box_iou(a, c) + 1e-12


def test_hmiou_gate_replacement():
    """hmiou ≤ IoU ⇒ 同等阈值下门控更严格：2D IoU 过门控而 hmiou 不过。

    a=[0.1,0.1,0.2,0.5] vs b=[0.1,0.3,0.2,0.7]：2D IoU≈0.333 > 0.25 门控，
    hmiou≈0.111 < 0.25 且中心距回退不满足（dist 0.2 > 0.5*avg_size 0.125）。
    """
    a = make_det(0.1, 0.1, 0.2, 0.5)
    b = make_det(0.1, 0.3, 0.2, 0.7)

    t_off = DetectionTracker(backend="python", hmiou_enabled=False)
    t_off.update([a])
    t_off.update([b])
    assert len(t_off._tracks) == 1          # 2D IoU 匹配成功

    t_on = DetectionTracker(backend="python", hmiou_enabled=True)
    t_on.update([a])
    t_on.update([b])
    assert len(t_on._tracks) == 2           # hmiou 门控拒绝 → 新建 track


def test_hmiou_polygon_pair_unaffected():
    """OBB polygon 对不适用 HMIoU：polygon 检测对保持 2D polygon IoU。"""
    from tests.test_tracker_snapshot import _rot_box
    poly = _rot_box(0.3, 0.3, 0.05, 0.1, 0.3)
    d1 = make_det(0.25, 0.2, 0.35, 0.4)
    d1["polygon"] = poly
    d2 = make_det(0.25, 0.2, 0.35, 0.4)
    d2["polygon"] = [list(p) for p in poly]

    t_off = DetectionTracker(backend="python", hmiou_enabled=False)
    t_on = DetectionTracker(backend="python", hmiou_enabled=True)
    for t in (t_off, t_on):
        t.update([d1])
        t.update([d2])
    box_off = next(iter(t_off._tracks.values())).box
    box_on = next(iter(t_on._tracks.values())).box
    np.testing.assert_array_equal(box_off, box_on)  # polygon 对行为不变


# ---------------------------------------------------------------------------
# 双后端等价（三特性开启，复用 equivalence 框架）
# ---------------------------------------------------------------------------

def _enhanced_seq(n=40):
    """双目标交叉 + 置信度起伏 + 周期 miss 的确定性序列。"""
    seq = []
    for f in range(n):
        dets = []
        if f % 7 != 3:  # 周期 miss
            dets.append(_det(0.2 + 0.008 * f, 0.4, 0.08, 0.16,
                             conf=0.55 + 0.4 * ((f % 5) / 4.0)))
        if f % 11 != 5:
            dets.append(_det(0.8 - 0.008 * f, 0.42, 0.08, 0.16,
                             conf=0.6 + 0.35 * ((f % 3) / 2.0)))
        seq.append(dets)
    return seq


@pytest.mark.parametrize("filter_type", ["ukf", "mcukf", "manifold_ukf"])
def test_dual_backend_enhancements_equivalent(filter_type):
    """NSA+OCM+HMIoU 全开：python vs rust 逐帧等价（坐标 1e-6、状态机精确）。"""
    kw = dict(filter_type=filter_type, **ENH_KW)
    _run_pair(dict(kw), dict(kw), _enhanced_seq(), predict_steps=3)


def test_dual_backend_enhancements_cascade_equivalent():
    kw = dict(matching_strategy="cascade", high_conf_thresh=0.6, **ENH_KW)
    _run_pair(dict(kw), dict(kw), _enhanced_seq(), predict_steps=2)


def test_dual_backend_auto_filter_enhancements_equivalent():
    """AUTO 模式 + 三特性：滤波器热切换后 NSA 配置保留、双端一致。"""
    from tests.test_tracker_snapshot import _auto_dets
    kw = dict(filter_type="auto", **ENH_KW)
    py_trk, rs_trk, _ = _run_pair(dict(kw), dict(kw), _auto_dets(120), predict_steps=2)
    # NSA 配置在 AUTO 热切换后仍保留（双端）
    for trk in py_trk._tracks.values():
        assert trk.kalman.nsa_enabled is True


def test_dual_backend_set_tracker_params_enhancements():
    """set_tracker_params 运行时开启三特性，双端一致。"""
    py_trk = DetectionTracker(backend="python")
    rs_trk = omintrack.RustTracker()
    seq = _enhanced_seq(20)
    for dets in seq[:10]:
        py_trk.update(dets)
        rs_trk.update(dets)
    py_trk.set_tracker_params(**ENH_KW)
    rs_trk.set_tracker_params(**ENH_KW)
    for f, dets in enumerate(seq[10:]):
        po = py_trk.update(dets)
        ro = rs_trk.update(dets)
        assert len(po) == len(ro)
        for a, b in zip(po, ro):
            assert a["track_id"] == b["track_id"]
            for k in ("x1", "y1", "x2", "y2", "confidence"):
                assert abs(a[k] - b[k]) <= 1e-6
