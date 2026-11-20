"""InferWorker Legacy 行为基线契约测试（第 3 步，Pipeline 对拍准备）。

直接调用真实 ``ai.inference.InferWorker._process_slot_frame``（第 2 步机械提取），
锁定 Legacy 命中帧 / 跳帧路径的当前行为契约，为下一步"同 scenario 对拍
Legacy ↔ Pipeline"提供可复用的假件与 scenario 工厂（``LegacyScenario`` /
``build_worker``）。

确定性保障：
    - monkeypatch 禁掉 ``_initialize_backend``（不加载任何模型，backend=cpu）；
    - backend / tracker / 后处理器 / health monitor 全部为确定性 fake，
      由 ``LegacyScenario`` 数据驱动；
    - ``_debug_emit_reports = 20`` 禁用 debug HTTP 上报；
    - 不启动 QThread、不访问摄像头 / 网络 / GUI。

被测方法 ``_process_slot_frame`` 本身绝不 mock；只对组件边界（模型推理、
tracker、后处理器、health monitor、ROI 重检测）做 fake / 插桩。

坐标 schema 说明：生产 ``Detection.to_dict()`` / ``DetectionTracker`` 输出
使用 ``x1/y1/x2/y2``（归一化）+ ``confidence`` / ``label`` / ``class_id``，
**不含** ``bbox`` 键。本基线全部使用生产 schema，供对拍双方共用。
"""
from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ai.inference as inference_mod
from ai.inference import InferWorker


FRAME = np.zeros((64, 64, 3), dtype=np.uint8)

# 17 点姿态关键点（确定性数据；手腕/肘置信度满足 hand ROI 构建阈值）
KEYPOINTS_17 = [
    {"x": 0.30 + i * 0.01, "y": 0.20 + i * 0.01, "conf": 0.9}
    for i in range(17)
]
KEYPOINTS_17[9]["conf"] = 0.85   # 左手腕
KEYPOINTS_17[10]["conf"] = 0.85  # 右手腕
KEYPOINTS_17[7]["conf"] = 0.80   # 左肘
KEYPOINTS_17[8]["conf"] = 0.80   # 右肘

OBB_POLYGON = [[0.60, 0.10], [0.90, 0.15], [0.88, 0.40], [0.58, 0.35]]


def person_det(**over: Any) -> dict:
    """生产 schema 的 person 检测 dict（x1/y1/x2/y2，无 bbox 键）。"""
    det = {
        "x1": 0.10, "y1": 0.10, "x2": 0.50, "y2": 0.90,
        "confidence": 0.92, "label": "person", "class_id": 0,
        "track_id": 1, "track_state": "normal",
    }
    det.update(over)
    return det


def obb_det(**over: Any) -> dict:
    det = {
        "x1": 0.58, "y1": 0.10, "x2": 0.90, "y2": 0.40,
        "confidence": 0.81, "label": "obb", "class_id": 5,
        "track_id": 2, "track_state": "normal",
        "polygon": [list(p) for p in OBB_POLYGON],
    }
    det.update(over)
    return det


def rectangle_raw_det(**over: Any) -> dict:
    det = {
        "x1": 0.05, "y1": 0.55, "x2": 0.30, "y2": 0.85,
        "confidence": 0.88, "label": "rectangle", "class_id": 80,
        "polygon": [[0.05, 0.55], [0.30, 0.55], [0.30, 0.85], [0.05, 0.85]],
    }
    det.update(over)
    return det


def hand_gesture_det(**over: Any) -> dict:
    det = {
        "x1": 0.62, "y1": 0.55, "x2": 0.78, "y2": 0.75,
        "confidence": 0.77, "label": "hand_gesture", "class_id": 9,
        "gesture": "open_palm",
    }
    det.update(over)
    return det


# ======================================================================
# Scenario —— 下一步 Legacy ↔ Pipeline 对拍复用的数据入口
# ======================================================================


@dataclass
class LegacyScenario:
    """数据驱动的单帧 scenario。对拍时同一实例同时喂给 Legacy 与 Pipeline。

    所有字段均为"组件边界返回值"，被测编排逻辑（``_process_slot_frame``
    及 Pipeline 等价路径）必须原样消费并保留。
    """

    raw_detections: list[dict] = field(default_factory=list)   # backend._run_inference 输出
    tracker_output: list[dict] = field(default_factory=list)   # tracker.update 输出（含 track_id）
    predicted_output: list[dict] = field(default_factory=list)  # tracker.predict_step 输出（跳帧）
    redetect_result: dict | None = None                        # _redetect_person_in_roi 返回值
    head_attrs: dict | None = None                             # head classifier predict 输出
    body_gestures: list[str] = field(default_factory=list)     # gesture recognizer 输出
    rectangle_raw: list[dict] = field(default_factory=list)    # rectangle detector 输出
    hand_raw: list[dict] = field(default_factory=list)         # hand gesture recognizer 输出
    call_redetect: bool = False  # tracker.update 是否模拟"存在未匹配 track"

# ======================================================================
# 确定性 fakes（记录事件顺序 + 返回 scenario 数据）
# ======================================================================


class FakeBackend:
    """模型推理边界 fake：记录调用与收到的帧，返回 raw_detections。"""

    def __init__(self, scenario: LegacyScenario, events: list[str]) -> None:
        self._scenario = scenario
        self._events = events
        self.calls = 0
        self.frames: list[np.ndarray] = []

    def __call__(self, frame: np.ndarray) -> list[dict]:
        self.calls += 1
        self.frames.append(frame)
        self._events.append("inference")
        return copy.deepcopy(self._scenario.raw_detections)


class FakeTracker:
    """DetectionTracker 边界 fake：记录调用顺序与 callback 注入。"""

    def __init__(self, scenario: LegacyScenario, events: list[str]) -> None:
        self._scenario = scenario
        self._events = events
        self.update_calls = 0
        self.appearance_calls = 0
        self.predict_calls = 0
        self.last_redetect_callback: Any = "UNSET"
        self.last_update_detections: Any = None
        self.last_frame: Any = "UNSET"
        self.appearance_frames: list[np.ndarray] = []
        self.redetect_config: dict = {}

    def update(self, detections, redetect_callback=None, frame=None):
        self.update_calls += 1
        self.last_redetect_callback = redetect_callback
        self.last_update_detections = detections
        self.last_frame = frame
        self._events.append("tracker.update")
        if redetect_callback is not None and self._scenario.call_redetect:
            # 模拟"存在未匹配 track"：tracker 用预测框 + 原置信度回调重检测
            self._events.append("redetect_cb")
            redetect_callback((0.20, 0.20, 0.60, 0.80), 0.50)
        return copy.deepcopy(self._scenario.tracker_output)

    def update_appearance(self, frame):
        self.appearance_calls += 1
        self.appearance_frames.append(frame)
        self._events.append("tracker.update_appearance")

    def predict_step(self):
        self.predict_calls += 1
        self._events.append("tracker.predict_step")
        return copy.deepcopy(self._scenario.predicted_output)

    def set_redetect_config(self, **kwargs):
        self.redetect_config = dict(kwargs)

    def set_filter_type(self, filter_type, kernel_sigma=None):
        self.filter_type = filter_type

    def reset(self):
        pass


class FakeRectangleTracker:
    """RectangleTracker 边界 fake：直通并附加 track 字段（min_hits=1）。"""

    def update(self, detections: list[dict]) -> list[dict]:
        out = []
        for det in detections:
            d = dict(det)
            d.setdefault("track_id", 1)
            d.setdefault("track_hits", 3)
            d.setdefault("track_misses", 0)
            out.append(d)
        return out

    def reset(self):
        pass


class FakeHeadClassifier:
    def __init__(self, scenario: LegacyScenario, events: list[str]) -> None:
        self._scenario = scenario
        self._events = events
        self.calls = 0

    def predict(self, frame, det):
        self.calls += 1
        self._events.append("head_attrs")
        return copy.deepcopy(self._scenario.head_attrs)


class FakeGestureRecognizer:
    def __init__(self, scenario: LegacyScenario, events: list[str]) -> None:
        self._scenario = scenario
        self._events = events

    def detect(self, det):
        self._events.append("body_gestures")
        return list(self._scenario.body_gestures)


class FakeHandGestureRecognizer:
    def __init__(self, scenario: LegacyScenario, events: list[str]) -> None:
        self._scenario = scenario
        self._events = events
        self.roi_calls = 0
        self.full_calls = 0

    def detect_in_rois(self, frame, rois):
        self.roi_calls += 1
        self._events.append("hand_gestures")
        return copy.deepcopy(self._scenario.hand_raw)

    def detect(self, frame):
        self.full_calls += 1
        self._events.append("hand_gestures")
        return copy.deepcopy(self._scenario.hand_raw)

    def close(self):
        pass


class FakeRectangleDetector:
    def __init__(self, scenario: LegacyScenario, events: list[str]) -> None:
        self._scenario = scenario
        self._events = events
        self.max_rectangles = 3

    def detect(self, frame):
        self._events.append("rectangle")
        return copy.deepcopy(self._scenario.rectangle_raw)

    def set_max_rectangles(self, count):
        self.max_rectangles = max(1, min(10, int(count)))


class FakeHealthMonitor:
    """健康监控边界 fake：恒健康、不恢复；记录是否已见 source 标记。"""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.updates = 0
        self.saw_source_fields = False
        self.last_detections: Any = None

    def update(self, detections):
        self.updates += 1
        self.last_detections = detections
        if any("source_frame_id" in d or "source_ts" in d or "latency_ms" in d for d in detections):
            self.saw_source_fields = True
        self._events.append("health")

    def is_healthy(self, threshold: float = 0.35) -> bool:
        return True

    def check_recovery(self) -> bool:
        return False

# ======================================================================
# 工厂 —— 构造不加载模型的真实 InferWorker 并注入确定性组件
# ======================================================================


def build_worker(
    monkeypatch: pytest.MonkeyPatch,
    scenario: LegacyScenario,
    *,
    slot_id: int = 0,
    stride: int = 1,
    denoise: bool = False,
    head: bool = False,
    body_gesture: bool = False,
    hand_gesture: bool = False,
    rectangle: bool = False,
    redetect: bool = True,
    events: list[str] | None = None,
    pipeline_enabled: bool = False,
) -> SimpleNamespace:
    """构造真实 InferWorker 并注入 fake 组件（Legacy / Pipeline 双模式）。

    ``pipeline_enabled=False``（默认）为 Legacy 基线；``True`` 时走
    ``_run_pipeline_mode`` 对拍路径。两种模式共用同一 scenario、
    事件名单与全部断言（双模式参数化测试）。

    返回 namespace：worker / backend / tracker / health / events /
    redetect_calls。对拍时 Pipeline 侧复用同一 scenario 与事件名单。
    """
    events = events if events is not None else []
    monkeypatch.setattr(
        InferWorker, "_initialize_backend", lambda self, model_path=None: None
    )
    worker = InferWorker(model_path=None, conf=0.25, infer_stride=stride, backend="cpu")
    worker._debug_emit_reports = 20  # 禁用 debug HTTP 上报
    worker._model = object()  # 满足 Pipeline 适配器 health_check（不加载模型）
    worker.set_pipeline_enabled(pipeline_enabled)
    worker.register_slot(slot_id)

    backend = FakeBackend(scenario, events)
    monkeypatch.setattr(worker, "_run_inference", backend)
    tracker = FakeTracker(scenario, events)
    worker._trackers[slot_id] = tracker
    worker._rectangle_trackers[slot_id] = FakeRectangleTracker()

    if not redetect:
        worker.set_person_redetect(enabled=False)  # 同步配置到 fake tracker

    if denoise:
        worker.set_denoise("bilateral", 5)
        real_denoise = InferWorker._denoise_frame
        holder: dict[str, Any] = {}

        def _recording_denoise(frame):
            out = real_denoise(worker, frame)
            events.append("denoise")
            holder["frame"] = out
            return out

        monkeypatch.setattr(worker, "_denoise_frame", _recording_denoise)
        backend.expected_denoised = holder

    if head:
        worker._head_classifier_enabled = True
        worker._head_classifier = FakeHeadClassifier(scenario, events)
    if body_gesture or hand_gesture:
        mode = "all" if (body_gesture and hand_gesture) else (
            "body" if body_gesture else "hand"
        )
        worker.set_gesture_mode(mode)
    if body_gesture:
        worker._gesture_recognizer = FakeGestureRecognizer(scenario, events)
    if hand_gesture:
        worker._hand_gesture_recognizer = FakeHandGestureRecognizer(scenario, events)
    if rectangle:
        worker.set_rectangle_detection_enabled(True)
        worker._rectangle_detector = FakeRectangleDetector(scenario, events)

    # 记录 dedupe/display 过滤的位置（包一层纯函数 final_dedupe）
    real_dedupe = inference_mod.final_dedupe

    def _recording_dedupe(detections):
        events.append("dedupe")
        return real_dedupe(detections)

    monkeypatch.setattr(inference_mod, "final_dedupe", _recording_dedupe)

    health = FakeHealthMonitor(events)
    worker._health_monitor = health

    # ROI 重检测插桩：记录委托并返回 scenario 结果（验证计数与委托链）
    redetect_calls: list[tuple] = []

    def _recording_redetect(frame, predicted_box, original_confidence=0.0):
        events.append("redetect_roi")
        redetect_calls.append((frame, predicted_box, original_confidence))
        return copy.deepcopy(scenario.redetect_result)

    monkeypatch.setattr(worker, "_redetect_person_in_roi", _recording_redetect)

    return SimpleNamespace(
        worker=worker,
        backend=backend,
        tracker=tracker,
        health=health,
        events=events,
        redetect_calls=redetect_calls,
        slot_id=slot_id,
    )

# ======================================================================
# 1. 命中帧处理顺序
# ======================================================================


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_hit_frame_minimal_order_and_cache(monkeypatch, pipeline_enabled):
    """全可选功能关闭时，Legacy 命中帧的核心处理顺序与缓存更新。"""
    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
    )
    env = build_worker(monkeypatch, scenario, pipeline_enabled=pipeline_enabled)
    result = env.worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    assert env.events == [
        "inference",
        "tracker.update",
        "tracker.update_appearance",
        "dedupe",
        "health",
    ]
    # 最终输出 = tracker 输出 + source 标记（契约第 7 项在专例中断言）
    stripped = [
        {k: v for k, v in d.items()
         if k not in ("source_frame_id", "source_ts", "latency_ms")}
        for d in result
    ]
    assert stripped == scenario.tracker_output
    # 命中帧路径：处理结果被原样缓存为 _last_detections（同一列表对象）
    assert env.worker._last_detections[env.slot_id] is result
    # health 在 source 标记之前执行
    assert env.health.saw_source_fields is False
    # tracker 收到 backend 原始输出与当前帧
    assert env.tracker.last_update_detections == scenario.raw_detections
    assert env.tracker.last_frame is not None


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_hit_frame_full_feature_order(monkeypatch, pipeline_enabled):
    """全开（去噪/头部属性/手势/矩形/手部/重检测）时的完整处理顺序。"""
    scenario = LegacyScenario(
        raw_detections=[person_det(keypoints=copy.deepcopy(KEYPOINTS_17))],
        tracker_output=[
            person_det(keypoints=copy.deepcopy(KEYPOINTS_17)),
            obb_det(),
        ],
        redetect_result=person_det(x1=0.21, y1=0.21, x2=0.59, y2=0.79),
        head_attrs={"hat": False, "mask": False, "sunglass": False},
        body_gestures=["standing"],
        rectangle_raw=[rectangle_raw_det()],
        hand_raw=[hand_gesture_det()],
        call_redetect=True,
    )
    env = build_worker(
        monkeypatch,
        scenario,
        pipeline_enabled=pipeline_enabled,
        denoise=True,
        head=True,
        body_gesture=True,
        hand_gesture=True,
        rectangle=True,
        redetect=True,
    )
    result = env.worker._process_slot_frame(env.slot_id, 4, time.monotonic(), FRAME)

    assert env.events == [
        "denoise",
        "inference",
        "tracker.update",
        "redetect_cb",
        "redetect_roi",
        "tracker.update_appearance",
        "head_attrs",
        "body_gestures",
        "rectangle",
        "hand_gestures",
        "dedupe",
        "health",
    ]
    # 去噪输出是推理输入（同一 ndarray 对象）
    assert env.backend.frames[0] is env.backend.expected_denoised["frame"]
    # 重检测回调委托 _redetect_person_in_roi，且使用与主推理相同的输入帧
    assert env.redetect_calls[0][0] is env.backend.frames[0]
    assert env.redetect_calls[0][1] == (0.20, 0.20, 0.60, 0.80)
    assert env.redetect_calls[0][2] == 0.50
    # 成功路径计数
    assert env.worker._redetect_attempt_count == 1
    assert env.worker._redetect_success_count == 1
    assert env.worker._redetect_failed_count == 0
    # 各阶段产物都在最终输出中
    labels = {d["label"] for d in result}
    assert labels == {"person", "obb", "rectangle", "hand_gesture"}
    # health 先于 source 标记
    assert env.health.saw_source_fields is False
    assert env.worker._last_detections[env.slot_id] is result


# ======================================================================
# 2. redetect 回调契约
# ======================================================================


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_redetect_enabled_counts_success(monkeypatch, pipeline_enabled):
    """redetect_enabled=True：tracker 收到非 None callback，成功计数正确。"""
    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
        redetect_result=person_det(),
        call_redetect=True,
    )
    env = build_worker(monkeypatch, scenario, redetect=True, pipeline_enabled=pipeline_enabled)
    env.worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    assert env.tracker.last_redetect_callback is not None
    assert callable(env.tracker.last_redetect_callback)
    assert env.worker._redetect_attempt_count == 1
    assert env.worker._redetect_success_count == 1
    assert env.worker._redetect_failed_count == 0


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_redetect_enabled_counts_failure(monkeypatch, pipeline_enabled):
    """重检测返回 None 时 failed 计数 +1，success 不变。"""
    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
        redetect_result=None,  # _redetect_person_in_roi 未找回目标
        call_redetect=True,
    )
    env = build_worker(monkeypatch, scenario, redetect=True, pipeline_enabled=pipeline_enabled)
    env.worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    assert env.tracker.last_redetect_callback is not None
    assert env.worker._redetect_attempt_count == 1
    assert env.worker._redetect_success_count == 0
    assert env.worker._redetect_failed_count == 1


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_redetect_disabled_passes_none(monkeypatch, pipeline_enabled):
    """redetect_enabled=False：tracker 收到 None callback，计数保持 0。"""
    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
        call_redetect=True,  # 即便有未匹配 track 也不应拿到 callback
    )
    env = build_worker(monkeypatch, scenario, redetect=False, pipeline_enabled=pipeline_enabled)
    env.worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    assert env.tracker.last_redetect_callback is None
    assert "redetect_roi" not in env.events
    assert env.worker._redetect_attempt_count == 0
    assert env.worker._redetect_success_count == 0
    assert env.worker._redetect_failed_count == 0


# ======================================================================
# 3. 跳帧预测路径
# ======================================================================


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_skipped_frame_uses_prediction_and_last_non_person(monkeypatch, pipeline_enabled):
    """跳帧：predict_step 的 person + 上次非 person；不再调用推理。"""
    bag_det = {
        "x1": 0.60, "y1": 0.60, "x2": 0.75, "y2": 0.80,
        "confidence": 0.70, "label": "bag", "class_id": 25,
        "track_id": 5, "track_state": "normal",
    }
    predicted_person = person_det(
        x1=0.12, y1=0.12, track_state="predicted", predicted_only=True,
    )
    scenario = LegacyScenario(
        raw_detections=[person_det(), dict(bag_det)],
        tracker_output=[person_det(), dict(bag_det)],
        predicted_output=[predicted_person],
    )
    env = build_worker(monkeypatch, scenario, stride=2, pipeline_enabled=pipeline_enabled)
    ts = time.monotonic()

    hit = env.worker._process_slot_frame(env.slot_id, 0, ts, FRAME)
    assert {d["label"] for d in hit} == {"person", "bag"}
    assert env.backend.calls == 1

    events_before = list(env.events)
    skipped = env.worker._process_slot_frame(env.slot_id, 1, time.monotonic(), FRAME)

    # 推理 / tracker.update / health / dedupe 均未再执行
    assert env.events == events_before + ["tracker.predict_step"]
    assert env.backend.calls == 1
    assert env.tracker.update_calls == 1
    assert env.health.updates == 1

    # person 来自 predict_step；非 person 原样复用上次结果（同一对象）
    assert skipped[0] == predicted_person
    bag_items = [d for d in skipped if d["label"] == "bag"]
    assert len(bag_items) == 1
    assert bag_items[0] is hit[1]
    # 跳帧输出被缓存为 _last_detections
    assert env.worker._last_detections[env.slot_id] is skipped


# ======================================================================
# 4. 扩展字段保留 + source/latency 契约（生产 schema，无 bbox 键）
# ======================================================================


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_extension_fields_preserved(monkeypatch, pipeline_enabled):
    """keypoints / OBB polygon / gesture / head_attrs 等扩展字段不丢失。"""
    kpts = copy.deepcopy(KEYPOINTS_17)
    polygon = [list(p) for p in OBB_POLYGON]
    scenario = LegacyScenario(
        raw_detections=[person_det(keypoints=copy.deepcopy(kpts)), obb_det()],
        tracker_output=[
            person_det(keypoints=copy.deepcopy(kpts)),
            obb_det(),
        ],
        head_attrs={"hat": True, "mask": False},
        body_gestures=["raising_hand"],
    )
    env = build_worker(monkeypatch, scenario, head=True, body_gesture=True, pipeline_enabled=pipeline_enabled)
    result = env.worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    by_label = {d["label"]: d for d in result}
    assert set(by_label) == {"person", "obb"}

    person = by_label["person"]
    assert person["keypoints"] == kpts            # 17 点关键点原样保留
    assert person["head_attrs"] == {"hat": True, "mask": False}
    assert person["gestures"] == ["raising_hand"]

    obb = by_label["obb"]
    assert obb["polygon"] == polygon              # OBB 四边形保留

    # 生产 schema：x1..y2 存在，绝不含 bbox 键
    for det in result:
        assert "bbox" not in det
        for key in ("x1", "y1", "x2", "y2", "confidence", "label", "class_id"):
            assert key in det


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_source_mark_and_latency_contract(monkeypatch, pipeline_enabled):
    """source_frame_id / source_ts / latency_ms 存在、同帧一致且数值合理。

    时间字段归一化策略：source_ts / source_frame_id 做精确相等断言；
    latency_ms 不做精确 wall-clock 断言，而是断言"同一帧内所有检测
    共享同一 latency 值" + "落在 [调用前, 调用后] 的单调区间内"。
    """
    scenario = LegacyScenario(
        raw_detections=[person_det(), obb_det()],
        tracker_output=[person_det(), obb_det()],
    )
    env = build_worker(monkeypatch, scenario, pipeline_enabled=pipeline_enabled)
    frame_id = 42
    submitted_ts = time.monotonic() - 0.25  # 预老化 250ms，使 latency 可区分

    t_before = time.monotonic()
    result = env.worker._process_slot_frame(env.slot_id, frame_id, submitted_ts, FRAME)
    t_after = time.monotonic()

    latency_values = {d["latency_ms"] for d in result}
    assert len(latency_values) == 1, "同帧所有检测应共享同一 latency_ms"
    latency_ms = latency_values.pop()

    for det in result:
        assert det["source_frame_id"] == frame_id        # 精确断言
        assert det["source_ts"] == submitted_ts          # 精确断言
        assert latency_ms == det["latency_ms"]

    # 数值合理：落在单调时钟区间内（250ms 预老化 + 本帧处理耗时）
    assert latency_ms >= (t_before - submitted_ts) * 1000.0
    assert latency_ms <= (t_after - submitted_ts) * 1000.0


# ======================================================================
# 5. 共享后处理（架构第 4 步抽取）：单点实现 + 健康回退/恢复文案与行为
# ======================================================================


class ScriptedHealthMonitor:
    """可脚本化的健康监控 fake：按设定回应 unhealthy / recovery。

    与 FakeHealthMonitor（恒健康）互补，用于精确驱动共享后处理中的
    健康回退/恢复分支，并记录 trigger_fallback 收到的快照内容。
    """

    def __init__(self, events: list[str], *, healthy: bool = True, recovery: bool = False) -> None:
        self._events = events
        self.healthy = healthy
        self.recovery = recovery
        self.updates = 0
        self.saw_source_fields = False
        self.last_detections: Any = None
        self._fallback_active = False
        self.fallback_snapshots: list[dict] = []
        self.restore_calls = 0
        self.snapshot_to_restore: dict | None = None

    def update(self, detections):
        self.updates += 1
        self.last_detections = detections
        if any("source_frame_id" in d for d in detections):
            self.saw_source_fields = True
        self._events.append("health")

    def is_healthy(self, threshold: float = 0.35):
        return self.healthy

    def check_recovery(self):
        return self.recovery

    def get_recent_mean(self):
        return 0.123

    def trigger_fallback(self, snapshot):
        self._fallback_active = True
        self.fallback_snapshots.append(snapshot)

    def restore_snapshot(self):
        self.restore_calls += 1
        return self.snapshot_to_restore


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_shared_postprocess_single_implementation(monkeypatch, pipeline_enabled):
    """双模式命中帧都恰好经共享方法 _postprocess_slot_detections 一次。"""
    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
    )
    env = build_worker(monkeypatch, scenario, pipeline_enabled=pipeline_enabled)
    worker = env.worker
    calls: list[str] = []
    real = InferWorker._postprocess_slot_detections

    def _spy(slot_id, frame_id, submitted_ts, detections, infer_frame, original_frame, *, mode):
        calls.append(mode)
        return real(
            worker, slot_id, frame_id, submitted_ts, detections,
            infer_frame, original_frame, mode=mode,
        )

    monkeypatch.setattr(worker, "_postprocess_slot_detections", _spy)
    result = worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    expected_mode = "pipeline" if pipeline_enabled else "legacy"
    assert calls == [expected_mode]
    # 缓存语义不变：_last_detections 与返回值为同一列表对象
    assert worker._last_detections[env.slot_id] is result


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_shared_postprocess_health_fallback_behavior_and_wording(
    monkeypatch, pipeline_enabled, caplog,
):
    """不健康帧触发回退：快照内容、三项回退生效、文案按模式区分。"""
    import logging

    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
    )
    env = build_worker(monkeypatch, scenario, pipeline_enabled=pipeline_enabled)
    worker = env.worker
    health = ScriptedHealthMonitor(env.events, healthy=False)
    worker._health_monitor = health
    worker.set_denoise("bilateral", 5)  # 使回退触发条件成立
    filter_before = worker._filter_type
    redetect_before = worker._person_redetect_enabled

    with caplog.at_level(logging.WARNING):
        result = worker._process_slot_frame(env.slot_id, 7, time.monotonic(), FRAME)

    # 行为：快照精确保存回退前配置，随后三项回退全部生效
    assert health.fallback_snapshots == [{
        "denoise": ("bilateral", 5),
        "filter": filter_before,
        "redetect": redetect_before,
    }]
    assert health._fallback_active is True
    assert worker._denoise_method == "none"
    assert worker._filter_type == "ukf"
    assert worker._person_redetect_enabled is False
    # 健康监控先于 source 标记；source 标记在回退后仍执行
    assert health.saw_source_fields is False
    assert all(d["source_frame_id"] == 7 for d in result)
    assert len(result) == 1

    # 文案：legacy 无 "(pipeline mode)" 且含三项说明；pipeline 反之
    messages = [
        r.getMessage() for r in caplog.records
        if "Health monitor triggered fallback" in r.getMessage()
    ]
    assert len(messages) == 1
    if pipeline_enabled:
        assert "(pipeline mode)" in messages[0]
        assert "disabling denoise" not in messages[0]
    else:
        assert "(pipeline mode)" not in messages[0]
        assert "disabling denoise" in messages[0]


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_shared_postprocess_health_recovery_behavior_and_wording(
    monkeypatch, pipeline_enabled, caplog,
):
    """健康且可恢复帧：快照配置精确还原、文案按模式区分。"""
    import logging

    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
    )
    env = build_worker(monkeypatch, scenario, pipeline_enabled=pipeline_enabled)
    worker = env.worker
    health = ScriptedHealthMonitor(env.events, healthy=True, recovery=True)
    health.snapshot_to_restore = {
        "denoise": ("bilateral", 7),
        "filter": "mcukf",
        "redetect": True,
    }
    worker._health_monitor = health

    with caplog.at_level(logging.INFO):
        result = worker._process_slot_frame(env.slot_id, 3, time.monotonic(), FRAME)

    # 行为：快照被取回一次，三项配置精确还原
    assert health.restore_calls == 1
    assert worker._denoise_method == "bilateral"
    assert worker._denoise_strength == 7
    assert worker._filter_type == "mcukf"
    assert worker._person_redetect_enabled is True
    assert health.fallback_snapshots == []
    # source 标记仍执行
    assert all(d["source_frame_id"] == 3 for d in result)

    # 文案：legacy 记录还原详情；pipeline 仅简短模式标记
    messages = [
        r.getMessage() for r in caplog.records
        if "Health monitor recovery" in r.getMessage()
    ]
    assert len(messages) == 1
    if pipeline_enabled:
        assert messages[0] == "Health monitor recovery (pipeline mode)"
    else:
        assert "restoring user config" in messages[0]
        assert "denoise=bilateral" in messages[0]


# ======================================================================
# 9. Pipeline runtime 复用与生命周期收口（0.4.0）
# ======================================================================


def _spy_pipeline_lifecycle(monkeypatch):
    """包装 visioncore Pipeline 的 initialize/shutdown/run，返回调用记录。

    以 spy/计数断言生命周期契约，不依赖 sleep；Pipeline 类级 patch 对
    ``_SlotPipelineRuntime.create`` 内部 ``pipeline.initialize()`` 等调用
    天然生效（同一类对象）。
    """
    from visioncore.pipeline import Pipeline

    inits: list = []
    shutdowns: list = []
    runs: list = []
    real_init = Pipeline.initialize
    real_shutdown = Pipeline.shutdown
    real_run = Pipeline.run

    def _counting_init(self):
        inits.append(self)
        return real_init(self)

    def _counting_shutdown(self):
        shutdowns.append(self)
        return real_shutdown(self)

    def _counting_run(self, ctx):
        runs.append(ctx)
        return real_run(self, ctx)

    monkeypatch.setattr(Pipeline, "initialize", _counting_init)
    monkeypatch.setattr(Pipeline, "shutdown", _counting_shutdown)
    monkeypatch.setattr(Pipeline, "run", _counting_run)
    return inits, shutdowns, runs


def _pipeline_env(monkeypatch, **kwargs):
    """pipeline 模式的单帧 scenario 环境（默认带一条 person 检测）。"""
    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
    )
    kwargs.setdefault("pipeline_enabled", True)
    return build_worker(monkeypatch, scenario, **kwargs)


def test_pipeline_runtime_reused_across_frames_same_slot(monkeypatch):
    """同 slot 连续两帧复用同一 Pipeline/adapters，initialize 只一次，成功帧无 shutdown。"""
    env = _pipeline_env(monkeypatch)
    worker = env.worker
    inits, shutdowns, runs = _spy_pipeline_lifecycle(monkeypatch)

    worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)
    rt1 = worker._pipeline_runtimes[env.slot_id]
    worker._process_slot_frame(env.slot_id, 1, time.monotonic(), FRAME)
    rt2 = worker._pipeline_runtimes[env.slot_id]

    # 同一 runtime 对象复用（pipeline + 4 adapters 全部同一实例）
    assert rt1 is rt2
    assert inits == [rt1.pipeline]  # initialize 恰好一次
    assert len(runs) == 2  # 两帧各 run 一次
    assert shutdowns == []  # 成功帧不得 shutdown
    # set_pipeline_enabled(False) 保留 cache（Legacy 行为不变，不触碰 runtime）
    worker.set_pipeline_enabled(False)
    assert worker._pipeline_runtimes[env.slot_id] is rt1
    assert shutdowns == []
    worker.set_pipeline_enabled(True)
    worker._process_slot_frame(env.slot_id, 2, time.monotonic(), FRAME)
    assert worker._pipeline_runtimes[env.slot_id] is rt1
    assert inits == [rt1.pipeline]  # 复用期间不再 initialize
    assert shutdowns == []


def test_pipeline_runtime_independent_across_slots(monkeypatch):
    """不同 slot 各自独立 runtime（互不共享 pipeline/adapters）。"""
    env = _pipeline_env(monkeypatch)
    worker = env.worker
    worker.register_slot(1)
    inits, shutdowns, _ = _spy_pipeline_lifecycle(monkeypatch)

    worker._process_slot_frame(0, 0, time.monotonic(), FRAME)
    worker._process_slot_frame(1, 0, time.monotonic(), FRAME)

    rt0 = worker._pipeline_runtimes[0]
    rt1 = worker._pipeline_runtimes[1]
    assert rt0 is not rt1
    assert rt0.pipeline is not rt1.pipeline
    assert rt0.trk_adapter is not rt1.trk_adapter
    assert rt0.tgt_adapter is not rt1.tgt_adapter
    assert len(inits) == 2  # 每 slot 各 initialize 一次
    assert shutdowns == []


def test_pipeline_redetect_callback_reset_each_frame(monkeypatch):
    """redetect 回调逐帧重设：enabled 每帧新回调，disabled 必须清 None（旧回调不残留）。"""
    env = _pipeline_env(monkeypatch, redetect=True)
    worker = env.worker

    worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)
    rt = worker._pipeline_runtimes[env.slot_id]
    cb_first = rt.trk_adapter._redetect_callback
    assert cb_first is not None

    # enabled：第二帧挂的是**新**回调（不复用上一帧对象）
    worker._process_slot_frame(env.slot_id, 1, time.monotonic(), FRAME)
    cb_second = rt.trk_adapter._redetect_callback
    assert cb_second is not None
    assert cb_second is not cb_first

    # disabled：必须清 None，旧回调不得残留
    worker.set_person_redetect(enabled=False)
    worker._process_slot_frame(env.slot_id, 2, time.monotonic(), FRAME)
    assert rt.trk_adapter._redetect_callback is None
    # 关闭后 FakeTracker 走无回调路径（事件序列无 redetect 委托亦无异常）
    assert env.events.count("tracker.update") == 3


def test_pipeline_failing_runtime_shutdown_evicted_and_rebuilt(monkeypatch):
    """run 异常：记 warning 返回 []，runtime shutdown+淘汰，下一帧重建并重新 initialize。"""
    from visioncore.pipeline import Pipeline

    env = _pipeline_env(monkeypatch)
    worker = env.worker
    inits, shutdowns, _ = _spy_pipeline_lifecycle(monkeypatch)

    # 第一帧 run 抛出（模拟任一 stage 处理异常）
    real_run = Pipeline.run
    state = {"boom": True}

    def _flaky_run(self, ctx):
        if state["boom"]:
            state["boom"] = False
            raise RuntimeError("pipeline stage boom")
        return real_run(self, ctx)

    monkeypatch.setattr(Pipeline, "run", _flaky_run)

    result = worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)
    assert result == []  # 既有行为：异常帧返回 []
    assert worker._last_detections[env.slot_id] == []
    # 异常 runtime 被 shutdown 且从 cache 淘汰
    assert worker._pipeline_runtimes == {}
    assert len(shutdowns) == 1
    # 单写契约：TargetStage 未完成 -> _pipeline_target_updated 为 False
    assert worker._pipeline_target_updated is False

    # 下一帧重建：新 runtime + 重新 initialize 一次，新 runtime 不被 shutdown
    worker._process_slot_frame(env.slot_id, 1, time.monotonic(), FRAME)
    assert env.slot_id in worker._pipeline_runtimes
    rt2 = worker._pipeline_runtimes[env.slot_id]
    assert rt2.pipeline not in shutdowns
    assert len(inits) == 2 and inits[1] is rt2.pipeline  # 重建时重新 initialize
    assert shutdowns == [inits[0]]  # 仅异常 runtime 被 shutdown，无额外关闭
    assert worker._pipeline_target_updated is True  # 成功帧 TargetStage 单写置位


def test_pipeline_legacy_mode_creates_no_runtime(monkeypatch):
    """Legacy 模式不创建任何 Pipeline runtime。"""
    scenario = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
    )
    env = build_worker(monkeypatch, scenario, pipeline_enabled=False)
    worker = env.worker
    inits, shutdowns, _ = _spy_pipeline_lifecycle(monkeypatch)

    worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    assert worker._pipeline_runtimes == {}
    assert inits == []
    assert shutdowns == []


def test_unregister_not_running_shuts_down_runtime_immediately(monkeypatch):
    """worker 未运行时 unregister_slot 直接 shutdown + 移除 runtime（每个至多一次）。"""
    env = _pipeline_env(monkeypatch)
    worker = env.worker
    _, shutdowns, _ = _spy_pipeline_lifecycle(monkeypatch)

    worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)
    rt = worker._pipeline_runtimes[env.slot_id]

    worker.unregister_slot(env.slot_id)
    assert worker._pipeline_runtimes == {}
    assert shutdowns == [rt.pipeline]
    rt.shutdown_once()  # 幂等：重复清理调用不触发第二次 shutdown
    assert shutdowns == [rt.pipeline]


def test_reregister_cancels_retirement_and_keeps_runtime(monkeypatch):
    """运行中 unregister 标记 retirement；重新注册同 slot 取消 retirement 并保留 runtime。"""
    env = _pipeline_env(monkeypatch)
    worker = env.worker
    _, shutdowns, _ = _spy_pipeline_lifecycle(monkeypatch)

    worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)
    rt = worker._pipeline_runtimes[env.slot_id]

    # 模拟运行中（不启动线程，仅走 running 分支的标记语义）
    worker._running = True
    try:
        worker.unregister_slot(env.slot_id)
        # 运行中：不直接 shutdown/移除，仅标记 retirement
        assert worker._pipeline_runtimes[env.slot_id] is rt
        assert env.slot_id in worker._retired_pipeline_slots
        assert shutdowns == []

        # 重新注册同 slot：取消 retirement，runtime 保留复用
        worker.register_slot(env.slot_id)
        assert env.slot_id not in worker._retired_pipeline_slots
        assert worker._pipeline_runtimes[env.slot_id] is rt
        assert shutdowns == []

        worker._process_slot_frame(env.slot_id, 1, time.monotonic(), FRAME)
        assert worker._pipeline_runtimes[env.slot_id] is rt
        assert shutdowns == []
    finally:
        worker._running = False


def test_pipeline_runtime_retired_during_creation_shutdown_outside_lock(monkeypatch):
    """创建期出现 retirement：锁外 shutdown、返回 []，半成品不得进入 cache。"""
    env = _pipeline_env(monkeypatch)
    worker = env.worker
    inits, shutdowns, runs = _spy_pipeline_lifecycle(monkeypatch)
    from visioncore.pipeline import Pipeline

    counting_init = Pipeline.initialize

    def _retire_during_init(pipeline):
        worker._retired_pipeline_slots.add(env.slot_id)
        return counting_init(pipeline)

    counting_shutdown = Pipeline.shutdown
    lock_states: list[bool] = []

    def _check_shutdown_lock(pipeline):
        lock_states.append(worker._lock.locked())
        return counting_shutdown(pipeline)

    monkeypatch.setattr(Pipeline, "initialize", _retire_during_init)
    monkeypatch.setattr(Pipeline, "shutdown", _check_shutdown_lock)

    result = worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)

    assert result == []
    assert worker._pipeline_runtimes == {}
    assert worker._pipeline_target_updated is False
    assert len(inits) == 1
    assert runs == []
    assert len(shutdowns) == 1
    assert lock_states == [False], "retired runtime 必须在锁外 shutdown"


def test_pipeline_postprocess_exception_retains_runtime_for_next_frame(monkeypatch):
    """后处理异常不淘汰已成功的 Pipeline runtime，下一帧继续复用。"""
    env = _pipeline_env(monkeypatch)
    worker = env.worker
    inits, shutdowns, _ = _spy_pipeline_lifecycle(monkeypatch)

    worker._process_slot_frame(env.slot_id, 0, time.monotonic(), FRAME)
    runtime = worker._pipeline_runtimes[env.slot_id]
    real_postprocess = worker._postprocess_slot_detections
    calls = {"count": 0}

    def _flaky_postprocess(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("postprocess boom")
        return real_postprocess(*args, **kwargs)

    monkeypatch.setattr(worker, "_postprocess_slot_detections", _flaky_postprocess)
    with pytest.raises(RuntimeError, match="postprocess boom"):
        worker._process_slot_frame(env.slot_id, 1, time.monotonic(), FRAME)

    assert worker._pipeline_runtimes[env.slot_id] is runtime
    assert shutdowns == []

    recovered = worker._process_slot_frame(env.slot_id, 2, time.monotonic(), FRAME)
    assert recovered
    assert worker._pipeline_runtimes[env.slot_id] is runtime
    assert inits == [runtime.pipeline]
    assert shutdowns == []
