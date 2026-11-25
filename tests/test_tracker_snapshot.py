r"""Golden-master snapshot tests for ai.tracker.DetectionTracker.

冻结当前 Python 跟踪器的行为，作为后续 Rust 重写的回归基准。

- 所有输入由解析轨迹构造（确定性，无随机数）。
- 每个场景的完整输出 + 关键内部状态被序列化（浮点 round 到 9 位小数）
  并与 tests/golden/tracker/<name>.json 精确比较。
- golden 再生成方式（二选一）：
    1) venv\Scripts\python.exe tests\golden\tracker\_generate.py
    2) venv\Scripts\python.exe -m pytest tests/test_tracker_snapshot.py --update-golden
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.tracker import DetectionTracker

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "tracker"


# ======================================================================
# 序列化 / golden 基础设施
# ======================================================================

def _serialize(obj):
    """递归把 tracker 输出转成 JSON 安全结构，浮点统一 round 到 9 位小数。"""
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return round(float(obj), 9)
    return obj


def _capture_meta(tracker: DetectionTracker) -> dict:
    """抓取每帧内部状态（misses / attempts / 滤波器类型 / adaptor 类型）。"""
    tracks = {}
    for tid, trk in tracker._tracks.items():
        tracks[str(tid)] = {
            "misses": int(trk.misses),
            "hits": int(trk.hits),
            "attempts": int(trk.redetect_attempts),
            "state": trk.track_state,
            "kalman": type(trk.kalman).__name__.replace("_Detection", "").lower(),
        }
    return {
        "tracks": tracks,
        "adaptor_type": tracker._adaptor.current_type if tracker._adaptor is not None else None,
    }


def _run(tracker, det_seq, callbacks=None, frames=None) -> dict:
    """逐帧 update 并收集输出快照 + 内部状态快照。"""
    callbacks = callbacks if callbacks is not None else [None] * len(det_seq)
    frames = frames if frames is not None else [None] * len(det_seq)
    data = {"frames": [], "meta": []}
    for dets, cb, frame in zip(det_seq, callbacks, frames):
        results = tracker.update(dets, redetect_callback=cb, frame=frame)
        data["frames"].append([_serialize(o) for o in results])
        data["meta"].append(_capture_meta(tracker))
    return data


def _first_diff(expected, actual, path="$"):
    """返回两个字典结构的第一个差异路径（供断言消息使用）。"""
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return f"{path}.{key}: missing in golden"
            if key not in actual:
                return f"{path}.{key}: missing in actual"
            diff = _first_diff(expected[key], actual[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path}: len {len(expected)} != {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual)):
            diff = _first_diff(e, a, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return None


def _check_golden(name: str, actual: dict, update: bool) -> None:
    path = GOLDEN_DIR / f"{name}.json"
    if update or not path.exists():
        if not path.exists():
            raise AssertionError(
                f"golden file missing: {path}; run "
                f"venv\\Scripts\\python.exe tests\\golden\\tracker\\_generate.py"
            )
        path.write_text(
            json.dumps(actual, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"[golden] regenerated {path.name}")
    expected = json.loads(path.read_text(encoding="utf-8"))
    diff = _first_diff(expected, actual)
    assert diff is None, (
        f"golden mismatch for '{name}': {diff}\n"
        f"(若行为变更是有意为之，请用 --update-golden 或 _generate.py 重新基线化)"
    )


@pytest.fixture(scope="module")
def update_golden(request):
    return bool(request.config.getoption("--update-golden"))


# ======================================================================
# 检测构造辅助
# ======================================================================

def _det(cx, cy, w, h, conf=0.9, label="person") -> dict:
    return {
        "x1": cx - w / 2.0, "y1": cy - h / 2.0,
        "x2": cx + w / 2.0, "y2": cy + h / 2.0,
        "confidence": conf, "label": label,
    }


def _rot_box(cx, cy, hw, hh, ang) -> list:
    """旋转矩形 4 角点（顺序与 tracker 约定一致）。"""
    pts = []
    for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        pts.append([
            cx + dx * math.cos(ang) - dy * math.sin(ang),
            cy + dx * math.sin(ang) + dy * math.cos(ang),
        ])
    return pts


def _keypoints_for(cx, cy, w, h, conf=0.8) -> list:
    """确定性 17 关键点骨架（固定相对 box 的偏移图案）。"""
    kpts = []
    for j in range(17):
        ox = ((j * 7) % 5 - 2) / 2.0
        oy = ((j * 11) % 5 - 2) / 2.0
        kpts.append({"x": cx + ox * w * 0.4, "y": cy + oy * h * 0.4, "conf": conf})
    return kpts


# ======================================================================
# 场景构建器（测试与 _generate.py 共用）
# ======================================================================

def scenario_single_uniform(filter_type: str) -> dict:
    """场景 1：单目标匀速直线运动 x40 帧。"""
    tracker = DetectionTracker(backend="python", filter_type=filter_type)
    seq = [[_det(0.20 + 0.008 * f, 0.30 + 0.004 * f, 0.12, 0.16, conf=0.9)]
           for f in range(40)]
    data = _run(tracker, seq)
    data["adaptor_type"] = tracker._adaptor.current_type if tracker._adaptor else None
    return data


def _auto_dets(n: int = 330) -> list:
    """场景 2 输入：前 30 帧正常运动，之后 [匹配, 空, 空] 周期注入 miss。

    miss_rate -> 2/3 > 0.5，驱动 AUTO 适配器在 miss_rate 统计上切换滤波器。
    （实测：纯观测离群无法把 NIS 推到 8.0 以上，见 _FilterAdaptor._NIS_THRESHOLD。）
    """
    seq = []
    for f in range(n):
        if f < 30:
            seq.append([_det(0.20 + 0.001 * f, 0.30, 0.12, 0.16, conf=0.9)])
        else:
            seq.append([_det(0.20 + 0.001 * f, 0.30, 0.12, 0.16, conf=0.9)]
                       if (f - 30) % 3 == 0 else [])
    return seq


def scenario_auto_outlier() -> dict:
    """场景 2：filter_type='auto'，前 30 帧正常 + 300 帧异常（周期性缺失）。"""
    tracker = DetectionTracker(backend="python", filter_type="auto")
    data = _run(tracker, _auto_dets())
    data["adaptor_type"] = tracker._adaptor.current_type
    return data


def scenario_cross() -> dict:
    """场景 3：双目标交叉穿越 x50 帧，hungarian 匹配。"""
    tracker = DetectionTracker(backend="python", matching_strategy="hungarian")
    seq = [[_det(0.15 + 0.012 * f, 0.40, 0.16, 0.20, conf=0.90),
            _det(0.75 - 0.012 * f, 0.40, 0.10, 0.12, conf=0.85)]
           for f in range(50)]
    return _run(tracker, seq)


def scenario_mixed_conf(strategy: str) -> dict:
    """场景 4：混合高/低置信度检测（低置信度框远离既有 track）。"""
    tracker = DetectionTracker(backend="python", matching_strategy=strategy)
    seq = []
    for f in range(3):
        seq.append([_det(0.20 + 0.005 * f, 0.30, 0.12, 0.16, conf=0.90),
                    _det(0.70, 0.60, 0.10, 0.12, conf=0.88)])
    # 第 4 帧：track1 的高置信度检测 + 远处孤立的低置信度检测
    seq.append([_det(0.235, 0.30, 0.12, 0.16, conf=0.90),
                _det(0.50, 0.50, 0.10, 0.12, conf=0.30)])
    # 第 5 帧：两个 track 都以高置信度回归
    seq.append([_det(0.240, 0.30, 0.12, 0.16, conf=0.90),
                _det(0.70, 0.60, 0.10, 0.12, conf=0.88)])
    return _run(tracker, seq)


def scenario_disappear() -> dict:
    """场景 5：目标消失 -> predicted -> 删除 -> 重现得到新 track_id。"""
    tracker = DetectionTracker(backend="python", )  # max_misses=4（默认）
    seq = [[_det(0.30 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(3)]
    seq += [[] for _ in range(6)]
    seq.append([_det(0.30 + 0.008, 0.40, 0.12, 0.16, conf=0.9)])
    return _run(tracker, seq)


def scenario_predict_step() -> dict:
    """场景 6：update 后连续 2 次 predict_step（跳帧外推）。"""
    tracker = DetectionTracker(backend="python", )
    seq = [[_det(0.20 + 0.006 * f, 0.30, 0.12, 0.16, conf=0.9)] for f in range(6)]
    data = _run(tracker, seq)
    for _ in range(2):
        preds = tracker.predict_step()
        data["frames"].append([_serialize(o) for o in preds])
        data["meta"].append(_capture_meta(tracker))
    return data


def scenario_obb() -> dict:
    """场景 7：带 OBB polygon 的检测 x20 帧（旋转 + 平移）。"""
    tracker = DetectionTracker(backend="python", )
    seq = []
    for f in range(20):
        cx, cy, ang = 0.30 + 0.005 * f, 0.40 + 0.002 * f, 0.05 * f
        poly = _rot_box(cx, cy, 0.06, 0.03, ang)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        d = _det(cx, cy, max(xs) - min(xs), max(ys) - min(ys), conf=0.9)
        d["polygon"] = [[float(x), float(y)] for x, y in poly]
        seq.append([d])
    return _run(tracker, seq)


def scenario_keypoints() -> dict:
    """场景 8：带 17 关键点的 person x15 帧（EMA 平滑）。"""
    tracker = DetectionTracker(backend="python", )
    seq = []
    for f in range(15):
        cx, cy = 0.25 + 0.004 * f, 0.35
        d = _det(cx, cy, 0.12, 0.16, conf=0.9)
        d["keypoints"] = _keypoints_for(cx, cy, 0.12, 0.16, conf=0.8)
        seq.append([d])
    return _run(tracker, seq)


def scenario_redetect_success() -> dict:
    """场景 9a：redetect_enabled，回调返回与预测框重叠的框 -> redetected。"""
    tracker = DetectionTracker(backend="python", redetect_enabled=True)
    seq = [[_det(0.30 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(3)]
    calls = []

    def cb(box, confidence):
        calls.append([round(float(v), 9) for v in box])
        return {"x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
                "confidence": 0.60, "label": "person"}

    data = _run(tracker, seq + [[]], callbacks=[None, None, None, cb])
    data["redetect_calls"] = calls
    return data


def scenario_redetect_failure() -> dict:
    """场景 9b：回调持续返回 None -> attempts 达上限后不再调用。"""
    tracker = DetectionTracker(backend="python", redetect_enabled=True)
    seq = [[_det(0.30 + 0.004 * f, 0.40, 0.12, 0.16, conf=0.9)] for f in range(3)]
    calls = []

    def cb(box, confidence):
        calls.append(1)
        return None

    data = _run(tracker, seq + [[]] * 5, callbacks=[None] * 3 + [cb] * 5)
    data["redetect_calls"] = calls
    return data


def scenario_frame_appearance() -> dict:
    """场景 10：传入固定图案 BGR 帧，person 检测 x10 帧（HSV 外观路径）。"""
    tracker = DetectionTracker(backend="python", )
    fh, fw = 360, 480
    seq, frames = [], []
    for f in range(10):
        frame = np.zeros((fh, fw, 3), dtype=np.uint8)
        frame[:, : fw // 2] = (40, 40, 40)
        frame[:, fw // 2:] = (200, 60, 30)
        cx = 0.25 + 0.01 * f
        x1 = int((cx - 0.06) * fw)
        x2 = int((cx + 0.06) * fw)
        y1, y2 = int(0.35 * fh), int(0.55 * fh)
        frame[y1:y2, x1:x2] = (30, 30, 220)  # 固定颜色的“行人”色块
        frames.append(frame)
        seq.append([_det(cx, 0.45, 0.12, 0.20, conf=0.9)])
    return _run(tracker, seq, frames=frames)


SCENARIOS = {
    "single_uniform_ukf": lambda: scenario_single_uniform("ukf"),
    "single_uniform_mcukf": lambda: scenario_single_uniform("mcukf"),
    "single_uniform_manifold_ukf": lambda: scenario_single_uniform("manifold_ukf"),
    "auto_outlier": scenario_auto_outlier,
    "cross_hungarian": scenario_cross,
    "mixed_conf_greedy": lambda: scenario_mixed_conf("greedy"),
    "mixed_conf_cascade": lambda: scenario_mixed_conf("cascade"),
    "disappear_delete": scenario_disappear,
    "predict_step": scenario_predict_step,
    "obb_polygon": scenario_obb,
    "keypoints_ema": scenario_keypoints,
    "redetect_success": scenario_redetect_success,
    "redetect_failure": scenario_redetect_failure,
    "frame_appearance": scenario_frame_appearance,
}


# ======================================================================
# 场景 1：单目标匀速直线运动 x40 帧（ukf / mcukf / manifold_ukf）
# ======================================================================

def _assert_single_uniform(data):
    assert len(data["frames"]) == 40
    for fr in data["frames"]:
        assert len(fr) == 1
        o = fr[0]
        assert o["track_id"] == 1
        assert o["track_state"] == "normal"
        assert o["predicted_only"] is False
        assert o["label"] == "person"
        assert "polygon" not in o
        assert o["keypoints"] == []
        assert 0.0 <= o["x1"] < o["x2"] <= 1.0


def test_single_uniform_ukf(update_golden):
    data = scenario_single_uniform("ukf")
    _assert_single_uniform(data)
    assert data["adaptor_type"] is None
    kalman_seen = {t["kalman"] for m in data["meta"] for t in m["tracks"].values()}
    assert kalman_seen == {"ukf"}
    _check_golden("single_uniform_ukf", data, update_golden)


def test_single_uniform_mcukf(update_golden):
    data = scenario_single_uniform("mcukf")
    # 新 IRLS-MCC UKF 契约（默认 sigma=1.0、R 白化逐通道相关熵、真实 IRLS）
    # 下，匀速场景必须与 UKF 场景行为一致：40 帧单 track、无 predicted
    # 幽灵。旧实现的周期性 ID churn 是核重加权 sigma 矩缺陷，不再作为期望。
    # golden 已按上述新契约重新基线。
    _assert_single_uniform(data)
    for fr in data["frames"]:
        assert all(o["track_id"] == 1 for o in fr)
        assert all(o["track_state"] == "normal" for o in fr)  # 无 predicted 输出
    assert data["adaptor_type"] is None
    kalman_seen = {t["kalman"] for m in data["meta"] for t in m["tracks"].values()}
    assert kalman_seen == {"mcukf"}
    _check_golden("single_uniform_mcukf", data, update_golden)


def test_single_uniform_manifold_ukf(update_golden):
    data = scenario_single_uniform("manifold_ukf")
    _assert_single_uniform(data)
    kalman_seen = {t["kalman"] for m in data["meta"] for t in m["tracks"].values()}
    assert kalman_seen == {"manifoldukf"}
    _check_golden("single_uniform_manifold_ukf", data, update_golden)


# ======================================================================
# 场景 2：filter_type="auto"，异常注入 -> 适配器切到非 ukf 类型
# ======================================================================

def test_auto_outlier_switch(update_golden):
    data = scenario_auto_outlier()
    # 全程单 track，id 稳定
    for fr in data["frames"]:
        assert len(fr) == 1
        assert fr[0]["track_id"] == 1
    # 适配器最终切换到非 ukf 类型（实测经由 miss_rate > 0.5 触发 mcukf）
    assert data["adaptor_type"] in ("mcukf", "manifold_ukf"), data["adaptor_type"]
    kalman_seen = {t["kalman"] for m in data["meta"] for t in m["tracks"].values()}
    assert "mcukf" in kalman_seen
    # 中段确实发生了 ukf -> mcukf 的热切换（meta 中可见滤波器类型变化）
    types_per_frame = [next(iter(m["tracks"].values()))["kalman"]
                       for m in data["meta"] if m["tracks"]]
    assert "ukf" in types_per_frame
    assert "mcukf" in types_per_frame
    first_mcukf = types_per_frame.index("mcukf")
    assert first_mcukf > 30  # 前 30 帧正常期保持 ukf
    _check_golden("auto_outlier", data, update_golden)


# ======================================================================
# 场景 3：双目标交叉穿越 x50 帧（hungarian），ID 不互换
# ======================================================================

def test_cross_hungarian_ids(update_golden):
    data = scenario_cross()
    assert len(data["frames"]) == 50
    for fr in data["frames"]:
        assert sorted(o["track_id"] for o in fr) == [1, 2]
    last_pos = {o["track_id"]: (o["x1"] + o["x2"]) / 2.0 for o in data["frames"][-1]}
    # track 1（左侧进入、大尺寸）交叉后到达右侧；track 2 反之
    assert last_pos[1] > 0.6
    assert last_pos[2] < 0.3
    assert last_pos[1] > last_pos[2]
    _check_golden("cross_hungarian", data, update_golden)


# ======================================================================
# 场景 4：greedy vs cascade 混合置信度
# ======================================================================

def test_mixed_conf_greedy(update_golden):
    data = scenario_mixed_conf("greedy")
    # greedy：远处低置信度检测不关联既有 track -> 新建 track 3
    assert sorted(o["track_id"] for o in data["frames"][3]) == [1, 2, 3]
    assert sorted(o["track_id"] for o in data["frames"][4]) == [1, 2, 3]
    _check_golden("mixed_conf_greedy", data, update_golden)


def test_mixed_conf_cascade(update_golden):
    data = scenario_mixed_conf("cascade")
    # cascade：低置信度检测仅关联不新建 -> track 数保持 2
    fr3 = data["frames"][3]
    assert sorted(o["track_id"] for o in fr3) == [1, 2]
    states = {o["track_id"]: o["track_state"] for o in fr3}
    assert states[2] == "predicted"  # 未匹配 -> predicted
    # 第 5 帧两 track 以原 ID 恢复
    assert sorted(o["track_id"] for o in data["frames"][4]) == [1, 2]
    _check_golden("mixed_conf_cascade", data, update_golden)


# ======================================================================
# 场景 5：消失与删除（max_misses=4）
# ======================================================================

def test_disappear_and_delete(update_golden):
    data = scenario_disappear()
    frames = data["frames"]
    assert len(frames) == 10
    # 前 3 帧正常
    for fr in frames[:3]:
        assert len(fr) == 1 and fr[0]["track_state"] == "normal"
    # 接下来 4 个空帧：misses=1..4 <= max_misses -> track 以 predicted 存活
    for fr in frames[3:7]:
        assert len(fr) == 1
        assert fr[0]["track_state"] == "predicted"
        assert fr[0]["predicted_only"] is True
    # 第 5 个连续 miss（misses=5 > max_misses=4）-> 删除，输出为空
    assert frames[7] == []
    assert frames[8] == []
    # 重现 -> 全新 track_id（_next_id 不复用）
    assert frames[9][0]["track_id"] == 2
    _check_golden("disappear_delete", data, update_golden)


# ======================================================================
# 场景 6：predict_step 跳帧外推
# ======================================================================

def test_predict_step(update_golden):
    data = scenario_predict_step()
    frames = data["frames"]
    assert len(frames) == 8  # 6 次 update + 2 次 predict_step
    last_upd, p1, p2 = frames[5], frames[6], frames[7]
    for fr in (p1, p2):
        assert len(fr) == 1
        assert fr[0]["track_id"] == 1
        assert fr[0]["predicted_only"] is True
        assert fr[0]["track_state"] == "predicted"
    # 沿 +x 速度方向外推
    assert p1[0]["x1"] > last_upd[0]["x1"]
    assert p2[0]["x1"] > p1[0]["x1"]
    # predict_step 不增加 misses
    assert data["meta"][7]["tracks"]["1"]["misses"] == 0
    _check_golden("predict_step", data, update_golden)


# ======================================================================
# 场景 7：OBB polygon（旋转 + 平移）
# ======================================================================

def test_obb_polygon(update_golden):
    data = scenario_obb()
    assert len(data["frames"]) == 20
    for fr in data["frames"]:
        assert len(fr) == 1
        o = fr[0]
        assert o["track_id"] == 1
        assert "polygon" in o
        assert len(o["polygon"]) == 4
        assert all(0.0 <= c <= 1.0 for pt in o["polygon"] for c in pt)
    _check_golden("obb_polygon", data, update_golden)


# ======================================================================
# 场景 8：17 关键点 EMA
# ======================================================================

def test_keypoints_ema(update_golden):
    data = scenario_keypoints()
    assert len(data["frames"]) == 15
    for fr in data["frames"]:
        assert len(fr) == 1
        o = fr[0]
        assert o["track_id"] == 1
        assert len(o["keypoints"]) == 17
        for kp in o["keypoints"]:
            assert set(kp.keys()) == {"x", "y", "conf"}
            assert 0.0 <= kp["x"] <= 1.0 and 0.0 <= kp["y"] <= 1.0
    _check_golden("keypoints_ema", data, update_golden)


# ======================================================================
# 场景 9：redetect_callback
# ======================================================================

def test_redetect_success(update_golden):
    data = scenario_redetect_success()
    assert len(data["redetect_calls"]) == 1
    fr3 = data["frames"][3]
    assert len(fr3) == 1
    assert fr3[0]["track_state"] == "redetected"
    assert fr3[0]["predicted_only"] is False
    # misses 复位、attempts 清零
    assert data["meta"][3]["tracks"]["1"]["misses"] == 0
    assert data["meta"][3]["tracks"]["1"]["attempts"] == 0
    _check_golden("redetect_success", data, update_golden)


def test_redetect_failure(update_golden):
    data = scenario_redetect_failure()
    # 恰好调用 max_retries=2 次，之后不再调用
    assert len(data["redetect_calls"]) == 2
    # 3 帧正常 + 4 帧 predicted（misses=1..4）+ 第 5 帧删除 + 1 帧空
    assert [len(fr) for fr in data["frames"]] == [1, 1, 1, 1, 1, 1, 1, 0]
    assert data["frames"][3][0]["track_state"] == "predicted"
    # attempts 递增过程被冻结在 meta 中
    assert data["meta"][4]["tracks"]["1"]["attempts"] == 2
    _check_golden("redetect_failure", data, update_golden)


# ======================================================================
# 场景 10：frame 外观路径（HSV 颜色直方图辅助匹配）
# ======================================================================

def test_frame_appearance(update_golden):
    data = scenario_frame_appearance()
    assert len(data["frames"]) == 10
    for fr in data["frames"]:
        assert len(fr) == 1
        assert fr[0]["track_id"] == 1
        assert fr[0]["track_state"] == "normal"
        assert fr[0]["predicted_only"] is False
    _check_golden("frame_appearance", data, update_golden)
