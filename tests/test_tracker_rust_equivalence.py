r"""Rust 扩展 omintrack 与 Python DetectionTracker 的逐帧等价性测试。

- 场景构建器复用 tests/test_tracker_snapshot.py 的确定性辅助函数
  (_det/_rot_box/_keypoints_for/_auto_dets)，序列逻辑与快照测试一一对应
  （同一场景在快照测试中已被 golden 冻结，因此这里的 Python 侧行为有基线）。
- 同一序列分别喂给 Python DetectionTracker 与 omintrack.RustTracker，
  逐帧比较：track_id 集合 / track_state / predicted_only / label 精确相等；
  浮点字段（x1..y2、confidence、keypoints、polygon）在容差内相等。
- 容差 TOL = 1e-6：NumPy(LAPACK) 与 nalgebra 的浮点实现差异在单次
  UKF 运算中为 1e-15 量级，经 330 帧 EMA/UKF 递推放大后实测仍 < 1e-9
  （见测试输出中的 max_abs 统计），1e-6 留出 3 个数量级余量。
  ManifoldUKF 的 eigh 路径与 NumPy eigh 特征向量符号/排序约定一致
  （均为升序、V diag V^T 形式，与顺序无关），实测最大偏差同样 < 1e-9，
  无需放宽到 1e-4。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import omintrack  # noqa: E402
from ai.tracker import DetectionTracker  # noqa: E402
from tests.test_tracker_snapshot import (  # noqa: E402
    _auto_dets,
    _det,
    _keypoints_for,
    _rot_box,
)

TOL = 1e-6


# ======================================================================
# 比较基础设施
# ======================================================================

def _note(stats: dict, diff: float) -> None:
    stats["max_abs"] = max(stats["max_abs"], abs(float(diff)))


def _cmp_float(stats: dict, a, b, what: str) -> None:
    diff = abs(float(a) - float(b))
    _note(stats, diff)
    assert diff <= TOL, f"{what}: {a!r} vs {b!r} (diff={diff})"


def _compare_outputs(py_out, rs_out, stats: dict, frame_no) -> None:
    assert len(py_out) == len(rs_out), f"frame {frame_no}: output count differs"
    py_map = {o["track_id"]: o for o in py_out}
    rs_map = {o["track_id"]: o for o in rs_out}
    assert set(py_map) == set(rs_map), (
        f"frame {frame_no}: track_id set differs: {sorted(py_map)} vs {sorted(rs_map)}"
    )
    for tid, po in py_map.items():
        ro = rs_map[tid]
        assert po["track_state"] == ro["track_state"], f"frame {frame_no} tid {tid}"
        assert bool(po["predicted_only"]) == bool(ro["predicted_only"]), f"frame {frame_no} tid {tid}"
        assert po["label"] == ro["label"], f"frame {frame_no} tid {tid}"
        for k in ("x1", "y1", "x2", "y2", "confidence"):
            _cmp_float(stats, po[k], ro[k], f"frame {frame_no} tid {tid} {k}")
        assert len(po["keypoints"]) == len(ro["keypoints"]), f"frame {frame_no} tid {tid} keypoints len"
        for j, (pk, rk) in enumerate(zip(po["keypoints"], ro["keypoints"])):
            for k in ("x", "y", "conf"):
                _cmp_float(stats, pk[k], rk[k], f"frame {frame_no} tid {tid} kp{j}.{k}")
        assert ("polygon" in po) == ("polygon" in ro), f"frame {frame_no} tid {tid} polygon presence"
        if "polygon" in po:
            for pa, ra in zip(po["polygon"], ro["polygon"]):
                _cmp_float(stats, pa[0], ra[0], f"frame {frame_no} tid {tid} polygon x")
                _cmp_float(stats, pa[1], ra[1], f"frame {frame_no} tid {tid} polygon y")


def _check_introspection(py_trk, rs_trk, frame_no) -> None:
    assert rs_trk.track_count() == len(py_trk._tracks), f"frame {frame_no}: track_count"
    py_persons = sum(1 for t in py_trk._tracks.values() if t.label == "person")
    assert rs_trk.person_track_count() == py_persons, f"frame {frame_no}: person_track_count"
    py_adaptor = py_trk._adaptor.current_type if py_trk._adaptor is not None else None
    assert rs_trk.adaptor_current_type() == py_adaptor, f"frame {frame_no}: adaptor type"
    # facade ROI：label==person 且 misses==0 的 track
    py_rois = {
        tid: [float(v) for v in trk.box]
        for tid, trk in py_trk._tracks.items()
        if trk.label == "person" and trk.misses == 0
    }
    rs_rois = {int(tid): box for tid, box in rs_trk.appearance_update_rois().items()}
    assert set(rs_rois) == set(py_rois), f"frame {frame_no}: appearance rois keys"
    for tid, box in py_rois.items():
        for a, b in zip(box, rs_rois[tid]):
            assert abs(a - float(b)) <= TOL, f"frame {frame_no}: roi {tid}"


def _run_pair(py_kwargs, rs_kwargs, det_seq, callbacks=None, frames=None,
              rs_det_hists=None, rs_det_embs=None, predict_steps=0):
    """同一场景双跑：Python DetectionTracker vs omintrack.RustTracker。"""
    py_trk = DetectionTracker(backend="python", **py_kwargs)
    rs_trk = omintrack.RustTracker(**rs_kwargs)
    stats = {"max_abs": 0.0}
    n = len(det_seq)
    callbacks = callbacks if callbacks is not None else [None] * n
    frames = frames if frames is not None else [None] * n
    rs_det_hists = rs_det_hists if rs_det_hists is not None else [None] * n
    rs_det_embs = rs_det_embs if rs_det_embs is not None else [None] * n
    for f, (dets, cb, frame, dh, de) in enumerate(
        zip(det_seq, callbacks, frames, rs_det_hists, rs_det_embs)
    ):
        py_out = py_trk.update(dets, redetect_callback=cb, frame=frame)
        rs_out = rs_trk.update(dets, det_hist=dh, det_emb=de, redetect_callback=cb)
        _compare_outputs(py_out, rs_out, stats, f)
        _check_introspection(py_trk, rs_trk, f)
    for _ in range(predict_steps):
        py_out = py_trk.predict_step()
        rs_out = rs_trk.predict_step()
        _compare_outputs(py_out, rs_out, stats, "predict_step")
        _check_introspection(py_trk, rs_trk, "predict_step")
    return py_trk, rs_trk, stats

# ======================================================================
# 数值组件对拍：Bhattacharyya vs cv2，匈牙利 vs scipy
# ======================================================================

def test_bhattacharyya_matches_cv2():
    """Rust Bhattacharyya 相似度必须精确复刻 cv2.compareHist(HISTCMP_BHATTACHARYYA)。"""
    rng = np.random.default_rng(42)
    max_diff_f64 = 0.0
    max_diff_cv2 = 0.0
    for _ in range(200):
        length = int(rng.choice([520, 256, 64]))
        h1 = rng.random(length)
        h2 = rng.random(length)
        h1 /= np.linalg.norm(h1)
        h2 /= np.linalg.norm(h2)
        # float64 公式为真值基准（cv2 内部 float32 舍入 ~3e-9，仅作交叉验证）
        bc = float(np.sum(np.sqrt(h1 * h2)))
        mean1 = float(np.mean(h1))
        mean2 = float(np.mean(h2))
        dist_f64 = float(np.sqrt(max(1 - bc / np.sqrt(mean1 * mean2 * length**2), 0)))
        sim_f64 = max(0.0, min(1.0, 1.0 - dist_f64))
        sim_rs = omintrack._bhattacharyya_similarity(h1.tolist(), h2.tolist())
        max_diff_f64 = max(max_diff_f64, abs(sim_rs - sim_f64))
        sim_cv2 = max(0.0, min(1.0, 1.0 - cv2_compare_hist(h1, h2)))
        max_diff_cv2 = max(max_diff_cv2, abs(sim_rs - sim_cv2))
    assert max_diff_f64 <= 1e-12, f"vs float64 formula max diff {max_diff_f64}"
    assert max_diff_cv2 <= 1e-8, f"vs cv2(float32) max diff {max_diff_cv2}"
    print(f"[bhattacharyya] vs float64={max_diff_f64:.3e} vs cv2={max_diff_cv2:.3e}")


def cv2_compare_hist(h1: np.ndarray, h2: np.ndarray) -> float:
    import cv2

    return float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                                 cv2.HISTCMP_BHATTACHARYYA))


def test_hungarian_matches_scipy():
    """Rust O(n^3) 匈牙利与 scipy.linear_sum_assignment 总代价一致（100 组随机）。"""
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(7)
    for _ in range(100):
        n_r = int(rng.integers(1, 9))
        n_c = int(rng.integers(1, 9))
        cost = rng.random((n_r, n_c)) * 2.0
        cost[rng.random((n_r, n_c)) < 0.3] = 1e6  # 随机门控项
        r1, c1 = linear_sum_assignment(cost)
        tot_ref = float(cost[r1, c1].sum())
        pairs = omintrack._hungarian_pairs(cost.tolist())
        tot_rs = float(sum(cost[r, c] for r, c in pairs))
        assert abs(tot_ref - tot_rs) <= 1e-9 * max(1.0, abs(tot_ref))


# ======================================================================
# 场景等价性
# ======================================================================

def test_single_uniform_all_filters():
    """场景 1：单目标匀速直线运动 x40 帧（ukf / mcukf / manifold_ukf）。"""
    for filter_type in ("ukf", "mcukf", "manifold_ukf"):
        seq = [[_det(0.20 + 0.008 * f, 0.30 + 0.004 * f, 0.12, 0.16, conf=0.9)]
               for f in range(40)]
        kwargs = {"filter_type": filter_type}
        _, _, stats = _run_pair(kwargs, dict(kwargs), seq)
        assert stats["max_abs"] <= TOL
        print(f"[single_uniform/{filter_type}] max_abs={stats['max_abs']:.3e}")


def test_auto_outlier_switch():
    """场景 2：filter_type='auto'，前 30 帧正常 + 300 帧周期性缺失，驱动适配器切换。"""
    kwargs = {"filter_type": "auto"}
    py_trk, rs_trk, stats = _run_pair(kwargs, dict(kwargs), _auto_dets())
    assert stats["max_abs"] <= TOL
    assert py_trk._adaptor.current_type == rs_trk.adaptor_current_type()
    print(f"[auto_outlier] max_abs={stats['max_abs']:.3e} adaptor={rs_trk.adaptor_current_type()}")


def test_cross_hungarian_ids():
    """场景 3：双目标交叉穿越 x50 帧，hungarian 匹配。"""
    kwargs = {"matching_strategy": "hungarian"}
    seq = [[_det(0.15 + 0.012 * f, 0.40, 0.16, 0.20, conf=0.90),
            _det(0.75 - 0.012 * f, 0.40, 0.10, 0.12, conf=0.85)]
           for f in range(50)]
    _, _, stats = _run_pair(kwargs, dict(kwargs), seq)
    assert stats["max_abs"] <= TOL
    print(f"[cross_hungarian] max_abs={stats['max_abs']:.3e}")


@pytest.mark.parametrize("strategy", ["greedy", "cascade"])
def test_mixed_conf(strategy):
    """场景 4：混合高/低置信度检测（低置信度框远离既有 track）。"""
    kwargs = {"matching_strategy": strategy}
    seq = []
    for f in range(3):
        seq.append([_det(0.20 + 0.005 * f, 0.30, 0.12, 0.16, conf=0.90),
                    _det(0.70, 0.60, 0.10, 0.12, conf=0.88)])
    seq.append([_det(0.235, 0.30, 0.12, 0.16, conf=0.90),
                _det(0.50, 0.50, 0.10, 0.12, conf=0.30)])
    seq.append([_det(0.240, 0.30, 0.12, 0.16, conf=0.90),
                _det(0.70, 0.60, 0.10, 0.12, conf=0.88)])
    _, _, stats = _run_pair(kwargs, dict(kwargs), seq)
    assert stats["max_abs"] <= TOL
    print(f"[mixed_conf/{strategy}] max_abs={stats['max_abs']:.3e}")

def test_disappear_and_delete():
    """场景 5：目标消失 -> predicted -> 删除 -> 重现得到新 track_id。"""
    kwargs = {}
    seq = [[_det(0.30 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(3)]
    seq += [[] for _ in range(6)]
    seq.append([_det(0.30 + 0.008, 0.40, 0.12, 0.16, conf=0.9)])
    _, _, stats = _run_pair(kwargs, dict(kwargs), seq)
    assert stats["max_abs"] <= TOL
    print(f"[disappear] max_abs={stats['max_abs']:.3e}")


def test_predict_step():
    """场景 6：update 后连续 2 次 predict_step（跳帧外推）。"""
    kwargs = {}
    seq = [[_det(0.20 + 0.006 * f, 0.30, 0.12, 0.16, conf=0.9)] for f in range(6)]
    _, _, stats = _run_pair(kwargs, dict(kwargs), seq, predict_steps=2)
    assert stats["max_abs"] <= TOL
    print(f"[predict_step] max_abs={stats['max_abs']:.3e}")


def test_obb_polygon():
    """场景 7：带 OBB polygon 的检测 x20 帧（旋转 + 平移）。"""
    kwargs = {}
    seq = []
    for f in range(20):
        cx, cy, ang = 0.30 + 0.005 * f, 0.40 + 0.002 * f, 0.05 * f
        poly = _rot_box(cx, cy, 0.06, 0.03, ang)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        d = _det(cx, cy, max(xs) - min(xs), max(ys) - min(ys), conf=0.9)
        d["polygon"] = [[float(x), float(y)] for x, y in poly]
        seq.append([d])
    _, _, stats = _run_pair(kwargs, dict(kwargs), seq)
    assert stats["max_abs"] <= TOL
    print(f"[obb] max_abs={stats['max_abs']:.3e}")


def test_keypoints_ema():
    """场景 8：带 17 关键点的 person x15 帧（EMA 平滑）。"""
    kwargs = {}
    seq = []
    for f in range(15):
        cx, cy = 0.25 + 0.004 * f, 0.35
        d = _det(cx, cy, 0.12, 0.16, conf=0.9)
        d["keypoints"] = _keypoints_for(cx, cy, 0.12, 0.16, conf=0.8)
        seq.append([d])
    _, _, stats = _run_pair(kwargs, dict(kwargs), seq)
    assert stats["max_abs"] <= TOL
    print(f"[keypoints] max_abs={stats['max_abs']:.3e}")


def test_redetect_success():
    """场景 9a：redetect_enabled，回调返回与预测框重叠的框 -> redetected。"""
    kwargs = {"redetect_enabled": True}
    seq = [[_det(0.30 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(3)] + [[]]
    calls_py, calls_rs = [], []

    def make_cb(store):
        def cb(box, confidence):
            store.append([round(float(v), 9) for v in box])
            return {"x1": float(box[0]), "y1": float(box[1]),
                    "x2": float(box[2]), "y2": float(box[3]),
                    "confidence": 0.60, "label": "person"}
        return cb

    py_trk, rs_trk, stats = _run_pair(
        kwargs, dict(kwargs), seq, callbacks=[None, None, None, make_cb(calls_py)]
    )
    # Rust 侧需要自己的回调实例（同一回调会被两个 tracker 共享，分别记录）
    # 注意上面 _run_pair 把同一回调同时传给两边 —— 这里改为分开跑
    calls_py.clear()
    py_trk2 = DetectionTracker(backend="python", **kwargs)
    for dets, cb in zip(seq, [None, None, None, make_cb(calls_py)]):
        py_trk2.update(dets, redetect_callback=cb)
    calls_rs.clear()
    rs_trk2 = omintrack.RustTracker(**kwargs)
    for dets, cb in zip(seq, [None, None, None, make_cb(calls_rs)]):
        rs_trk2.update(dets, redetect_callback=cb)
    assert stats["max_abs"] <= TOL
    assert len(calls_py) == len(calls_rs) and len(calls_py) > 0
    for cp, cr in zip(calls_py, calls_rs):
        for a, b in zip(cp, cr):
            assert abs(a - b) <= TOL
    print(f"[redetect_success] max_abs={stats['max_abs']:.3e} calls={len(calls_rs)}")

def test_redetect_failure():
    """场景 9b：回调持续返回 None -> attempts 达上限后不再调用。"""
    kwargs = {"redetect_enabled": True}
    seq = [[_det(0.30 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(3)] + [[]] * 5

    def make_cb(store):
        def cb(box, confidence):
            store.append(1)
            return None
        return cb

    calls_py, calls_rs = [], []
    py_trk = DetectionTracker(backend="python", **kwargs)
    for dets, cb in zip(seq, [None] * 3 + [make_cb(calls_py)] * 5):
        py_trk.update(dets, redetect_callback=cb)
    rs_trk = omintrack.RustTracker(**kwargs)
    for dets, cb in zip(seq, [None] * 3 + [make_cb(calls_rs)] * 5):
        rs_trk.update(dets, redetect_callback=cb)
    assert calls_py == calls_rs and len(calls_rs) > 0
    assert rs_trk.track_count() == len(py_trk._tracks)
    print(f"[redetect_failure] calls={len(calls_rs)} (python == rust)")


def _appearance_frames(n: int, person_specs, fh: int = 360, fw: int = 480):
    """生成与快照场景 10 同风格的图案帧 + 检测序列。

    person_specs: list of (color_bgr, cx_fn) —— 每个行人一个固定色块。
    """
    seq, frames = [], []
    for f in range(n):
        frame = np.zeros((fh, fw, 3), dtype=np.uint8)
        frame[:, : fw // 2] = (40, 40, 40)
        frame[:, fw // 2:] = (200, 60, 30)
        dets = []
        for color, cx_fn in person_specs:
            cx = cx_fn(f)
            x1 = int((cx - 0.06) * fw)
            x2 = int((cx + 0.06) * fw)
            y1, y2 = int(0.35 * fh), int(0.55 * fh)
            frame[y1:y2, x1:x2] = color
            dets.append(_det(cx, 0.45, 0.12, 0.20, conf=0.9))
        frames.append(frame)
        seq.append(dets)
    return seq, frames


def _det_hists_for_frame(py_trk, frame, dets):
    """Python 侧预处理：为每个 person 检测预计算 520 维 HSV 直方图（未来 facade 行为）。"""
    return [
        py_trk._compute_color_hist_for_box(frame, det) if det.get("label") == "person" else None
        for det in dets
    ]


def test_frame_appearance():
    """场景 10：传入固定图案 BGR 帧，person 检测 x10 帧（HSV 外观路径）。

    Python 侧 update(frame=frame) 内部计算 det_hist；Rust 侧由 Python 预计算后
    以 det_hist 传入（模拟未来 facade 行为）。
    """
    kwargs = {}
    seq, frames = _appearance_frames(10, [((30, 30, 220), lambda f: 0.25 + 0.01 * f)])
    py_trk = DetectionTracker(backend="python", **kwargs)
    rs_trk = omintrack.RustTracker(**kwargs)
    stats = {"max_abs": 0.0}
    for f, (dets, frame) in enumerate(zip(seq, frames)):
        py_out = py_trk.update(dets, frame=frame)
        rs_out = rs_trk.update(dets, det_hist=_det_hists_for_frame(py_trk, frame, dets))
        _compare_outputs(py_out, rs_out, stats, f)
        _check_introspection(py_trk, rs_trk, f)
    assert stats["max_abs"] <= TOL
    print(f"[frame_appearance] max_abs={stats['max_abs']:.3e}")


def _hist_for_roi(frame: np.ndarray, box) -> np.ndarray | None:
    """按 `_Track.update_color_hist` 的 ROI 规则（中心 50% 区域）从帧算 HSV 直方图。"""
    from ai.tracker import _compute_hsv_hist

    fh, fw = frame.shape[:2]
    x1 = int(min(max(float(box[0]), 0.0), 1.0) * fw)
    y1 = int(min(max(float(box[1]), 0.0), 1.0) * fh)
    x2 = int(min(max(float(box[2]), 0.0), 1.0) * fw)
    y2 = int(min(max(float(box[3]), 0.0), 1.0) * fh)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w2, h2 = max(2, (x2 - x1) // 4), max(2, (y2 - y1) // 4)
    roi = frame[max(0, cy - h2):min(fh, cy + h2), max(0, cx - w2):min(fw, cx + w2)]
    if roi.size == 0:
        return None
    return _compute_hsv_hist(roi)


def test_update_color_hists_facade():
    """facade 闭环等价性：update 后 Python 走 update_appearance(frame)，
    Rust 走 appearance_update_rois() + update_color_hists()，后续匹配输出必须一致。"""
    kwargs = {}
    seq, frames = _appearance_frames(12, [((30, 30, 220), lambda f: 0.20 + 0.006 * f),
                                          ((30, 220, 30), lambda f: 0.70 - 0.006 * f)])
    py_trk = DetectionTracker(backend="python", **kwargs)
    rs_trk = omintrack.RustTracker(**kwargs)
    stats = {"max_abs": 0.0}
    for f, (dets, frame) in enumerate(zip(seq, frames)):
        py_out = py_trk.update(dets, frame=frame)
        rs_out = rs_trk.update(dets, det_hist=_det_hists_for_frame(py_trk, frame, dets))
        # Python：内部 appearance 更新路径
        py_trk.update_appearance(frame)
        # Rust：facade 写回路径
        rois = rs_trk.appearance_update_rois()
        hists = {}
        for tid, box in rois.items():
            hist = _hist_for_roi(frame, box)
            if hist is not None:
                hists[int(tid)] = hist.tolist()
        rs_trk.update_color_hists(hists)
        _compare_outputs(py_out, rs_out, stats, f)
        _check_introspection(py_trk, rs_trk, f)
    assert stats["max_abs"] <= TOL
    print(f"[update_color_hists] max_abs={stats['max_abs']:.3e}")

def test_appearance_assisted_matching():
    """外观辅助匹配：两个不同颜色的行人近距离并行，appearance bonus 影响
    greedy 打分（0.15 权重），Python/Rust 的匹配决策必须一致。"""
    kwargs = {"matching_strategy": "greedy"}
    # 两行人同向运动且距离较近（0.24 与 0.31 起始，宽度 0.12/0.20 → IoU 接近 0）
    seq, frames = _appearance_frames(
        15,
        [((30, 30, 220), lambda f: 0.24 + 0.004 * f),
         ((30, 220, 30), lambda f: 0.31 + 0.004 * f)],
    )
    py_trk = DetectionTracker(backend="python", **kwargs)
    rs_trk = omintrack.RustTracker(**kwargs)
    stats = {"max_abs": 0.0}
    for f, (dets, frame) in enumerate(zip(seq, frames)):
        py_out = py_trk.update(dets, frame=frame)
        rs_out = rs_trk.update(dets, det_hist=_det_hists_for_frame(py_trk, frame, dets))
        _compare_outputs(py_out, rs_out, stats, f)
    assert stats["max_abs"] <= TOL
    print(f"[appearance_matching] max_abs={stats['max_abs']:.3e}")


class _FakeReidBackend:
    """确定性 fake ReID backend（5 维 embedding，由 ROI 均值色构造）。"""

    def extract(self, roi):
        if roi is None or roi.size == 0:
            return None
        b, g, r = roi[..., 0].mean(), roi[..., 1].mean(), roi[..., 2].mean()
        h = roi.shape[0]
        v = np.array([b, g, r, (b + g + r) / 3.0 + 0.01 * h, 0.5], dtype=np.float64)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v


def _reid_emb_for_det(frame, det, backend):
    """复刻 update() 内 ReID embedding 的裁剪逻辑。"""
    fh, fw = frame.shape[:2]
    x1 = int(min(max(float(det["x1"]), 0.0), 1.0) * fw)
    y1 = int(min(max(float(det["y1"]), 0.0), 1.0) * fh)
    x2 = int(min(max(float(det["x2"]), 0.0), 1.0) * fw)
    y2 = int(min(max(float(det["y2"]), 0.0), 1.0) * fh)
    if x2 > x1 and y2 > y1:
        return backend.extract(frame[y1:y2, x1:x2])
    return None


def test_reid_embedding_matching():
    """ReID 分支：reid_active 时 hungarian 走 0.4*IoU + 0.4*cosine + 0.2*center，
    匹配后 track embedding 做 EMA 0.8/0.2 + L2 归一化。同 backend 分别喂两边。"""
    backend = _FakeReidBackend()
    py_kwargs = {"matching_strategy": "hungarian", "reid_backend": backend,
                 "reid_min_tracks": 3, "reid_stride": 3}
    rs_kwargs = {"matching_strategy": "hungarian"}
    seq, frames = _appearance_frames(
        18,
        [((30, 30, 220), lambda f: 0.15 + 0.003 * f),
         ((30, 220, 30), lambda f: 0.50 + 0.002 * f),
         ((220, 30, 30), lambda f: 0.80 - 0.003 * f)],
    )
    py_trk = DetectionTracker(backend="python", **py_kwargs)
    rs_trk = omintrack.RustTracker(**rs_kwargs)
    stats = {"max_abs": 0.0}
    for f, (dets, frame) in enumerate(zip(seq, frames)):
        py_out = py_trk.update(dets, frame=frame)
        # 与 Python 内部 gating 完全一致：person_count>=1 且 track_person_count>=3
        # 且帧计数 % stride == 0；facade 负责该 gating，这里手动复刻
        track_person_count = py_trk and sum(
            1 for t in py_trk._tracks.values() if t.label == "person"
        )
        reid_active = (
            sum(1 for d in dets if d.get("label") == "person") >= 1
            and track_person_count >= 3
            and f % 3 == 0
        )
        if reid_active:
            embs = [
                _reid_emb_for_det(frame, det, backend) if det.get("label") == "person" else None
                for det in dets
            ]
            embs_arg = [None if e is None else e.tolist() for e in embs]
        else:
            reid_active = False
            embs_arg = None
        rs_out = rs_trk.update(dets, det_emb=embs_arg, reid_active=reid_active)
        _compare_outputs(py_out, rs_out, stats, f)
        _check_introspection(py_trk, rs_trk, f)
    assert stats["max_abs"] <= TOL
    print(f"[reid] max_abs={stats['max_abs']:.3e}")


def test_min_hits_and_setters():
    """min_hits_to_show 输出门控 + 运行时 setter 等价性。"""
    kwargs = {"min_hits_to_show": 2}
    seq = [[_det(0.30 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(4)]
    py_trk = DetectionTracker(backend="python", **kwargs)
    rs_trk = omintrack.RustTracker(**kwargs)
    stats = {"max_abs": 0.0}
    for f, dets in enumerate(seq):
        py_out = py_trk.update(dets)
        rs_out = rs_trk.update(dets)
        _compare_outputs(py_out, rs_out, stats, f)
        _check_introspection(py_trk, rs_trk, f)
    # 运行时 setter（含已有 track 的 velocity_clip 同步）
    py_trk.set_smooth_alpha(0.7)
    rs_trk.set_smooth_alpha(0.7)
    py_trk.set_redetect_config(enabled=True, max_retries=3, iou_threshold=0.15)
    rs_trk.set_redetect_config(enabled=True, max_retries=3, iou_threshold=0.15)
    py_trk.set_filter_type("mcukf", kernel_sigma=0.3)
    rs_trk.set_filter_type("mcukf", kernel_sigma=0.3)
    py_trk.set_tracker_params(iou_threshold=0.2, max_misses=3, matching_strategy="greedy",
                              high_conf_thresh=0.6, velocity_clip=0.2, second_stage_iou_min=0.15)
    rs_trk.set_tracker_params(iou_threshold=0.2, max_misses=3, matching_strategy="greedy",
                              high_conf_thresh=0.6, velocity_clip=0.2, second_stage_iou_min=0.15)
    seq2 = [[_det(0.32 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(6)]
    for f, dets in enumerate(seq2):
        py_out = py_trk.update(dets)
        rs_out = rs_trk.update(dets)
        _compare_outputs(py_out, rs_out, stats, f"setter{f}")
        _check_introspection(py_trk, rs_trk, f"setter{f}")
    # reset 等价
    py_trk.reset()
    rs_trk.reset()
    assert rs_trk.track_count() == 0
    assert rs_trk.person_track_count() == 0
    assert rs_trk.adaptor_current_type() is None
    print(f"[min_hits_setters] max_abs={stats['max_abs']:.3e}")

