# =============================================================================
# VisionDataPlatform - YOLOv8 自适应推理引擎
# =============================================================================
# 安装依赖：
#   pip install ultralytics>=8.0.0
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   (若需 CUDA 加速，请访问 https://pytorch.org/get-started/locally/ 选择对应版本)
#
# 强制锁定 CPU 方法：
#   在 _detect_device() 方法中，将 return "cuda" 改为 return "cpu"
# =============================================================================

from __future__ import annotations

import logging
import os
import sys
import threading
import time
import json
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import cv2
import torch
from PyQt6.QtCore import QThread, pyqtSignal
from ai.detection import Detection
from ai.gesture_recognizer import GestureRecognizer
from ai.hand_gesture_recognizer import HandGestureRecognizer, normalize_gesture_mode
from ai.head_classifier import HeadClassificationBackend
from ai.model_task import ModelTask
from ai.onnx_yolo_backend import OnnxYoloBackend
from ai.rectangle_detector import RectangleDetector, RectangleTracker, normalize_rectangle_sensitivity
from ai.tracker import DetectionTracker, _box_iou
from ai.uhd_backend import UhdBackend
from ai.yolo26_backend import Yolo26Backend
from ultralytics import YOLO

# VisionCore TargetManager — 旁路维护目标状态（不影响现有检测输出）
from visioncore.eventbus import EventBus
from visioncore.target_manager import TargetManager, detections_to_tracks

if TYPE_CHECKING:
    # VisionCore 类型引用 —— 仅用于类型检查，不影响运行时。
    from visioncore.core import Event as CoreEvent
    from visioncore.core import Frame as CoreFrame
    from visioncore.core import Target as CoreTarget


def _is_uhd_model(model_path: str | None) -> bool:
    """通过文件名识别 UHD 模型（支持计划中的短名与仓库原始长名）。"""
    if not model_path:
        return False
    name = Path(model_path).name.lower()
    return "uhd" in name or "ultratinyod" in name


def _is_yolo26_model(model_path: str | None) -> bool:
    """通过文件名识别 YOLO26 模型。"""
    if not model_path:
        return False
    name = Path(model_path).name.lower()
    return "yolo26" in name or "yolov10" in name


def _infer_model_task_from_path(model_path: str | None) -> ModelTask:
    """根据文件名推断模型任务类型。"""
    if not model_path:
        return ModelTask.DETECT
    name = Path(model_path).name.lower()
    if "obb" in name:
        return ModelTask.OBB
    if "pose" in name:
        return ModelTask.POSE
    return ModelTask.DETECT


def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    try:
        debug_server_url = "http://127.0.0.1:7777/event"
        debug_session_id = "yolo-detection-overlay"
        with open(".dbg/yolo-detection-overlay.env", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_server_url = line.split("=", 1)[1]
                elif line.startswith("DEBUG_SESSION_ID="):
                    debug_session_id = line.split("=", 1)[1]
        urllib.request.urlopen(
            urllib.request.Request(
                debug_server_url,
                data=json.dumps(
                    {
                        "sessionId": debug_session_id,
                        "runId": "post-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": f"[DEBUG] {msg}",
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.5,
        ).read()
    except Exception:
        pass


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent.parent


def _resolve_artifact_path(*candidate_names: str) -> Path | None:
    root = _project_root()
    candidates = []
    for name in candidate_names:
        candidates.append(root / "models" / name)
        candidates.append(root / name)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def final_dedupe(detections: list[dict]) -> list[dict]:
    thresholds = {"person": 0.55, "rectangle": 0.30, "hand_gesture": 0.35}
    ordered = sorted(detections, key=_dedupe_priority, reverse=True)
    kept: list[dict] = []
    for det in ordered:
        label = det.get("label")
        threshold = thresholds.get(label)
        if threshold is not None and any(existing.get("label") == label and _dict_iou(det, existing) > threshold for existing in kept):
            continue
        kept.append(det)
    return kept


def _dedupe_priority(det: dict) -> tuple[float, float, float]:
    if det.get("label") == "rectangle":
        return (float(det.get("track_hits", 0)), -float(det.get("track_misses", 0)), float(det.get("confidence", 0.0)))
    return (0.0, 0.0, float(det.get("confidence", 0.0)))


def _merge_overlapping_rois(rois: list[tuple[float, float, float, float]], iou_threshold: float) -> list[tuple[float, float, float, float]]:
    merged: list[tuple[float, float, float, float]] = []
    for roi in rois:
        if any(_tuple_iou(roi, existing) > iou_threshold for existing in merged):
            continue
        merged.append(roi)
    return merged


def _dict_iou(a: dict, b: dict) -> float:
    return _tuple_iou((float(a["x1"]), float(a["y1"]), float(a["x2"]), float(a["y2"])), (float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])))


def _tuple_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rgb_to_hsv(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_RGB2HSV)[0, 0]
    return (int(hsv[0]) * 2, int(hsv[1]), int(hsv[2]))


def _lab_distance(a_rgb: tuple[int, int, int], b_rgb: tuple[int, int, int]) -> float:
    pixels = np.array(
        [[
            [a_rgb[0], a_rgb[1], a_rgb[2]],
            [b_rgb[0], b_rgb[1], b_rgb[2]],
        ]],
        dtype=np.uint8,
    )
    lab = cv2.cvtColor(pixels, cv2.COLOR_RGB2LAB).astype(np.float32)[0]
    diff = lab[0] - lab[1]
    return float(np.sqrt(np.sum(diff * diff)))


class _HealthMonitor:
    """运行时健康监控：连续 N 秒人物置信度低于阈值则触发自动回退，恢复后自动还原。"""

    def __init__(self, history_seconds: float = 3.0, fps: float = 30.0):
        self._history_len = int(history_seconds * fps)
        self._conf_history: list[float] = []
        # 恢复机制
        self._fallback_active = False
        self._config_snapshot: dict | None = None
        self._recovery_frames = 0
        self._recovery_threshold = 0.55
        self._recovery_required = 60  # 持续 60 帧（~2秒）健康才恢复

    def update(self, detections: list[dict]) -> None:
        person_confs = [d.get("confidence", 0.0) for d in detections if d.get("label") == "person"]
        mean_conf = float(np.mean(person_confs)) if person_confs else 0.0
        self._conf_history.append(mean_conf)
        if len(self._conf_history) > self._history_len:
            self._conf_history.pop(0)

    def is_healthy(self, threshold: float = 0.35) -> bool:
        if len(self._conf_history) < self._history_len:
            return True
        recent_mean = float(np.mean(self._conf_history[-self._history_len:]))
        return recent_mean >= threshold

    def get_recent_mean(self) -> float:
        if not self._conf_history:
            return 0.0
        return float(np.mean(self._conf_history[-self._history_len:]))

    def trigger_fallback(self, snapshot: dict) -> None:
        """触发回退，保存当前用户配置快照。"""
        self._config_snapshot = snapshot
        self._fallback_active = True
        self._recovery_frames = 0

    def check_recovery(self) -> bool:
        """检查是否满足恢复条件：持续 N 帧置信度 >= 恢复阈值。"""
        if not self._fallback_active:
            return False
        if self.get_recent_mean() >= self._recovery_threshold:
            self._recovery_frames += 1
        else:
            self._recovery_frames = 0
        return self._recovery_frames >= self._recovery_required

    def restore_snapshot(self) -> dict | None:
        """返回快照并重置回退状态。"""
        snap = self._config_snapshot
        self._fallback_active = False
        self._recovery_frames = 0
        self._config_snapshot = None
        return snap


class SuperResEngine:

    def __init__(self, model_path: str | None = None, scale: int = 4):
        self._scale = scale

        if model_path is None:
            resolved = _resolve_artifact_path(f"ESPCN_x{scale}.pb")
            model_path = str(resolved) if resolved is not None else ""

        if not os.path.exists(model_path):
            logging.warning("Super-res model not found at %s, disabling upscaler", model_path)
            self._sr = None
            return

        self._sr = cv2.dnn_superres.DnnSuperResImpl_create()
        self._sr.readModel(model_path)
        self._sr.setModel("espcn", scale)
        logging.info("SuperResEngine loaded: %s (x%d)", model_path, scale)

    def upscale(self, roi_img: np.ndarray) -> np.ndarray | None:
        if self._sr is None:
            return None
        try:
            return self._sr.upsample(roi_img)
        except Exception as e:
            logging.error("Super-resolution failed: %s", e)
            return None

    @property
    def available(self) -> bool:
        return self._sr is not None


class _FrameSlot:

    __slots__ = ("frame", "frame_id", "submitted_ts")

    def __init__(self):
        self.frame: np.ndarray | None = None
        self.frame_id: int = 0
        self.submitted_ts: float = 0.0


class InferWorker(QThread):
    # VisionCore 未来入口: detection_ready 未来可发射 list[CoreDetection]，
    # 通过 visioncore.core.adapters.to_core_detection() 转换。

    detection_ready = pyqtSignal(int, list)
    model_ready = pyqtSignal(str, bool, str)

    def __init__(self, model_path: str | None = None, conf: float = 0.25, infer_stride: int = 2, backend: str = "auto", parent=None):
        super().__init__(parent)
        self._buffers: dict[int, _FrameSlot] = {}
        self._lock = threading.Lock()
        self._running = False
        self._conf = conf
        self._infer_stride = max(1, infer_stride)  # 1 = 每帧，2 = 每2帧
        self._debug_infer_reports = 0
        self._debug_submit_reports = 0
        self._debug_emit_reports = 0
        self._sr_attempt_count = 0
        self._sr_candidates_total = 0
        self._sr_refined_count = 0
        self._sr_refined_conf_gain_sum = 0.0
        self._sr_refined_person_only_hit_count = 0
        self._sr_refine_failed_count = 0
        self._sr_skipped_budget_count = 0
        self._infer_cycle_count = 0
        self._warmup_done = False
        self._startup_guard_until = 0.0
        self._startup_guard_seconds = 3.0
        self._startup_ai_interval = 0.0
        self._target_ai_interval = 0.0
        self._model_imgsz = 640
        self._model_path = model_path or ""
        self._backend_mode = self._normalize_backend_mode(backend)
        self._runtime_backend = "pytorch"
        self._ai_disabled = False
        self._pending_model_path: str | None = None
        self._model_switching = False
        self._person_enabled = True
        self._skeleton_enabled = False
        self._gesture_mode = "off"
        self._rectangle_enabled = False
        self._rectangle_sensitivity = "low"
        self._rectangle_max_count = 3
        self._rectangle_target_rgb: tuple[int, int, int] | None = None
        self._rectangle_color_threshold = 30.0
        self._rectangle_detector = RectangleDetector.from_preset(self._rectangle_sensitivity)
        self._rectangle_detector.set_max_rectangles(self._rectangle_max_count)
        self._gesture_recognizer = GestureRecognizer()
        self._hand_gesture_recognizer: HandGestureRecognizer | None = None
        self._rectangle_detected_total = 0

        self._device = "cpu" if self._backend_mode == "cpu" else self._detect_device()
        self._custom_model_path = model_path or ""
        self._pose_model = None
        self._pose_model_path = ""
        self._detect_model = None
        self._detect_model_path = ""
        self._model = None
        self._initialize_backend(model_path)
        self._super_res = SuperResEngine()
        self._max_sr_rois = 1 if self._device == "cpu" else 2

        # 每 slot 一个跟踪器 + 缓存上次结果
        self._trackers: dict[int, DetectionTracker] = {}
        self._rectangle_trackers: dict[int, RectangleTracker] = {}
        self._last_detections: dict[int, list[dict]] = {}
        self._last_submit_ts_by_slot: dict[int, float] = {}

        # ROI 重检测配置（基于卡尔曼预测框）
        self._person_redetect_enabled = True
        self._person_redetect_max_retries = 2
        self._person_redetect_roi_pad = 0.15
        self._redetect_attempt_count = 0
        self._redetect_success_count = 0
        self._redetect_failed_count = 0
        self._redetect_budget_per_frame = 1  # 每帧最多重检测的 track 数
        # ReID 配置
        self._reid_backend = None  # 懒加载
        self._reid_enabled = False
        self._reid_min_tracks = 3
        self._reid_stride = 3
        # 滤波器类型配置
        self._filter_type = "ukf"
        self._mcukf_kernel_sigma = 0.25
        # 跟踪器匹配策略参数
        self._iou_threshold = 0.25
        self._max_misses = 4
        self._matching_strategy = "hungarian"
        self._high_conf_thresh = 0.5
        self._velocity_clip = 0.30
        self._second_stage_iou_min = 0.1
        # 去噪配置
        self._denoise_method = "none"
        self._denoise_strength = 5
        # 运行时健康监控
        self._health_monitor = _HealthMonitor(history_seconds=3.0, fps=30.0)
        # 人头属性分类器（CHC）
        self._head_classifier: HeadClassificationBackend | None = None
        self._head_classifier_enabled = False
        self._head_classifier_path = ""
        self._head_class_stride = 2
        self._head_attr_cache: dict[int, tuple[int, dict]] = {}

        # VisionCore TargetManager — 旁路维护目标状态，不影响现有检测输出。
        # Shadow Integration (Milestone 3)：创建 EventBus 并注入 TargetManager，
        # 使生命周期事件（target.created/lost/recovered/locked/removed）自动
        # 流入总线。InferWorker 自身不订阅、不消费事件；GUI 也不消费。总线
        # 此刻仅作为接收端，验证 "Tracker -> TargetManager -> EventBus" 事件
        # 流动管道畅通。注入 bus 不改变 TargetManager 的核心逻辑与现有检测
        # 输出——无订阅者时 publish 是 O(1) 空操作。
        self._event_bus: EventBus = EventBus()
        self._target_mgr: TargetManager = TargetManager(event_bus=self._event_bus)

        # Milestone C8 — Pipeline 模式开关（默认 False = Legacy Mode）。
        # 当 True 时，推理帧经 Pipeline（DetectorStage → TrackerStage →
        # TargetStage → EventStage）处理，使用 visioncore.pipeline.adapters
        # 中的适配器包装现有后端。Legacy 代码路径完全不变。
        self._pipeline_enabled: bool = False

    def set_pipeline_enabled(self, enabled: bool) -> None:
        """切换 Pipeline 模式（C8）。

        当 ``enabled=True`` 时，推理帧经 VisionCore Pipeline（DetectorStage →
        TrackerStage → TargetStage → EventStage）处理，使用
        ``visioncore.pipeline.adapters`` 中的适配器包装现有 YOLO/Tracker/
        TargetManager 后端。当 ``False``（默认）时，现有 Legacy 代码路径
        原样运行，业务行为完全不变。

        两种模式调用**相同的后端方法**（同一个 YOLO 模型、同一个
        DetectionTracker、同一个 TargetManager），仅组织方式不同。Pipeline
        模式的初始版本（C8）不支持 ROI 重检测回调（redetect_callback），
        该功能在后续里程碑迁移到 Pipeline Stage。
        """
        self._pipeline_enabled = bool(enabled)
        logging.info("Pipeline mode %s", "ENABLED" if enabled else "disabled (legacy)")

    def _run_pipeline_mode(
        self,
        slot_id: int,
        frame_id: int,
        submitted_ts: float,
        frame: np.ndarray,
    ) -> list[dict]:
        """Pipeline 模式处理一帧（C8）。

        构建 Pipeline（DetectorStage → TrackerStage → TargetStage →
        EventStage），使用适配器包装现有后端，运行后提取跟踪检测结果
        并应用与 Legacy 路径相同的后处理（头部属性、手势、矩形检测、
        健康监控、延迟标记）。

        返回 ``list[dict]``（与 Legacy 路径格式完全一致），供
        ``detection_ready.emit`` 和 ``_bypass_update_targets`` 使用。
        """
        from visioncore.pipeline import Pipeline, PipelineContext
        from visioncore.pipeline.adapters import (
            InferWorkerDetectorAdapter,
            InferWorkerTrackerAdapter,
            InferWorkerTargetManagerAdapter,
            InferWorkerEventBusAdapter,
        )
        from visioncore.pipeline.stages import (
            DetectorStage, TrackerStage, TargetStage, EventStage,
        )
        from visioncore.core.frame import Frame as CoreFrame

        original_frame = frame
        infer_frame = (
            self._denoise_frame(original_frame)
            if self._denoise_method != "none"
            else original_frame
        )

        # 构建适配器（每次帧创建新适配器以设置当前帧引用）
        det_adapter = InferWorkerDetectorAdapter(self)
        det_adapter.set_frame_image(infer_frame)
        trk_adapter = InferWorkerTrackerAdapter(self, slot_id)
        trk_adapter.set_frame_image(infer_frame)
        tgt_adapter = InferWorkerTargetManagerAdapter(self, slot_id)
        evt_adapter = InferWorkerEventBusAdapter(self)

        # 构建 Pipeline
        p = Pipeline()
        p.add_stage(DetectorStage(det_adapter, name="detect"))
        p.add_stage(TrackerStage(trk_adapter, name="track"))
        p.add_stage(TargetStage(tgt_adapter, name="targets"))
        p.add_stage(EventStage(evt_adapter, name="events"))

        # 构建 Context 并运行
        ctx = PipelineContext.empty(timestamp=submitted_ts)
        ctx.frame = CoreFrame(
            frame_id=frame_id,
            timestamp=submitted_ts,
            source_id=f"slot{slot_id}",
            image=infer_frame,
        )

        try:
            p.initialize()
            p.run(ctx)
        except Exception as e:
            logging.warning("Pipeline mode error (falling back to empty): %s", e)
            return []
        finally:
            try:
                p.shutdown()
            except Exception:
                pass

        # 从 Tracker 适配器恢复跟踪后的检测 dicts（含 track_id 等）
        detections = trk_adapter.last_tracked_dicts

        # --- 后处理（与 Legacy 路径一致）---
        if self._should_run_yolo_inference():
            detections = self._attach_head_attributes(detections, infer_frame, frame_id)
            detections = self._attach_gestures(detections)
        else:
            detections = []
        rectangle_detections = self._run_rectangle_detection(slot_id, original_frame)
        if rectangle_detections:
            detections = detections + rectangle_detections
        hand_gesture_detections = self._run_hand_gesture_detection(original_frame, detections)
        if hand_gesture_detections:
            detections = detections + hand_gesture_detections
        detections = self._filter_display_detections(final_dedupe(detections))

        # 健康监控
        self._health_monitor.update(detections)
        if not self._health_monitor.is_healthy(threshold=0.35):
            if not self._health_monitor._fallback_active and (
                self._denoise_method != "none" or self._filter_type != "ukf" or self._person_redetect_enabled
            ):
                logging.warning(
                    "Health monitor triggered fallback (pipeline mode): person_conf=%.3f",
                    self._health_monitor.get_recent_mean(),
                )
                self._health_monitor.trigger_fallback({
                    "denoise": (self._denoise_method, self._denoise_strength),
                    "filter": self._filter_type,
                    "redetect": self._person_redetect_enabled,
                })
                self.set_denoise("none", 5)
                self.set_filter_type("ukf")
                self.set_person_redetect(enabled=False)
        elif self._health_monitor.check_recovery():
            snap = self._health_monitor.restore_snapshot()
            if snap:
                logging.info("Health monitor recovery (pipeline mode)")
                self.set_denoise(snap["denoise"][0], snap["denoise"][1])
                self.set_filter_type(snap["filter"])
                self.set_person_redetect(enabled=snap["redetect"])

        # 延迟标记
        latency_ms = (time.monotonic() - submitted_ts) * 1000.0
        for det in detections:
            det["source_frame_id"] = frame_id
            det["source_ts"] = submitted_ts
            det["latency_ms"] = latency_ms

        return detections

    def set_head_classifier(self, enabled: bool, model_path: str | None = None, stride: int = 2) -> None:
        """运行时启停或切换头部属性分类模型。"""
        self._head_classifier_enabled = bool(enabled)
        if model_path is not None:
            self._head_classifier_path = str(model_path)
        self._head_class_stride = max(1, int(stride))
        if not self._head_classifier_enabled:
            self._head_classifier = None
            self._head_attr_cache.clear()
        else:
            # 模型在首次需要时懒加载；路径变化时重置
            if self._head_classifier is not None and model_path is not None:
                if self._head_classifier.model_path != str(model_path):
                    self._head_classifier = None
        logging.info(
            "Head classifier config: enabled=%s path=%s stride=%d",
            self._head_classifier_enabled, self._head_classifier_path or "<auto>", self._head_class_stride,
        )

    def _load_head_classifier(self) -> HeadClassificationBackend | None:
        if not self._head_classifier_enabled:
            return None
        if self._head_classifier is not None:
            return self._head_classifier
        path = self._head_classifier_path
        if not path:
            resolved = _resolve_artifact_path("chc_s_wo_fiqa.onnx")
            path = str(resolved) if resolved is not None else ""
        if not path or not os.path.exists(path):
            logging.warning("Head classifier model not found at %s, disabling head classification", path)
            self._head_classifier_enabled = False
            return None
        try:
            self._head_classifier = HeadClassificationBackend(path, provider_mode=self._backend_mode)
        except Exception as e:
            logging.error("Failed to load head classifier: %s", e)
            self._head_classifier_enabled = False
            return None
        return self._head_classifier

    def set_infer_stride(self, n: int) -> None:
        self._infer_stride = max(1, n)
        self._clear_slot_caches()

    def set_conf(self, conf: float) -> None:
        self._conf = float(conf)
        # 置信度变化会改变检测集合，重置跟踪器避免旧 ID 与滞留框干扰
        for tracker in self._trackers.values():
            tracker.reset()
        self._clear_slot_caches()

    def set_smooth_alpha(self, alpha: float) -> None:
        for tracker in self._trackers.values():
            tracker.set_smooth_alpha(alpha)
        self._clear_slot_caches()

    def set_person_redetect(
        self,
        *,
        enabled: bool | None = None,
        max_retries: int | None = None,
        roi_pad: float | None = None,
    ) -> None:
        """运行时更新 person ROI 重检测配置，对所有现有 tracker 生效。"""
        if enabled is not None:
            self._person_redetect_enabled = bool(enabled)
        if max_retries is not None:
            self._person_redetect_max_retries = max(0, min(5, int(max_retries)))
        if roi_pad is not None:
            self._person_redetect_roi_pad = float(max(0.0, min(0.5, roi_pad)))
        for tracker in self._trackers.values():
            tracker.set_redetect_config(
                enabled=self._person_redetect_enabled,
                max_retries=self._person_redetect_max_retries,
            )
        self._clear_slot_caches()
        logging.info(
            "Person redetect config: enabled=%s retries=%d roi_pad=%.2f",
            self._person_redetect_enabled, self._person_redetect_max_retries, self._person_redetect_roi_pad,
        )

    def set_redetect_budget(self, budget: int) -> None:
        """设置每帧重检测预算（最多同时重检测几个 track）。"""
        self._redetect_budget_per_frame = max(0, min(5, int(budget)))
        for tracker in self._trackers.values():
            tracker._redetect_budget = self._redetect_budget_per_frame
        logging.info("Redetect budget set to: %d", self._redetect_budget_per_frame)

    def set_reid_config(
        self,
        enabled: bool = False,
        min_tracks: int = 3,
        stride: int = 3,
        model_path: str | None = None,
    ) -> None:
        """配置 ReID 插件。enabled 时懒加载 ReIDBackend。"""
        self._reid_enabled = bool(enabled)
        self._reid_min_tracks = max(2, int(min_tracks))
        self._reid_stride = max(1, int(stride))
        if enabled:
            if self._reid_backend is None:
                from ai.reid_backend import ReIDBackend
                self._reid_backend = ReIDBackend(model_path)
                logging.info("ReID backend loaded: mock=%s", self._reid_backend._mock)
        else:
            self._reid_backend = None
        # 更新已有 tracker
        for tracker in self._trackers.values():
            tracker._reid_backend = self._reid_backend
            tracker._reid_min_tracks = self._reid_min_tracks
            tracker._reid_stride = self._reid_stride
        logging.info("ReID config: enabled=%s min_tracks=%d stride=%d", enabled, self._reid_min_tracks, self._reid_stride)

    def set_filter_type(self, filter_type: str, kernel_sigma: float | None = None) -> None:
        """运行时切换滤波器类型。已有 track 保持旧滤波器，新 track 使用新类型。"""
        self._filter_type = filter_type
        if kernel_sigma is not None:
            self._mcukf_kernel_sigma = float(kernel_sigma)
        for tracker in self._trackers.values():
            tracker.set_filter_type(filter_type, kernel_sigma=self._mcukf_kernel_sigma)
        self._clear_slot_caches()
        logging.info("Filter type set to: %s (kernel_sigma=%.2f)", filter_type, self._mcukf_kernel_sigma)

    def set_tracker_params(
        self,
        *,
        iou_threshold: float | None = None,
        max_misses: int | None = None,
        matching_strategy: str | None = None,
        high_conf_thresh: float | None = None,
        velocity_clip: float | None = None,
        second_stage_iou_min: float | None = None,
    ) -> None:
        """批量更新跟踪器匹配参数，对所有槽位生效。"""
        if iou_threshold is not None:
            self._iou_threshold = float(iou_threshold)
        if max_misses is not None:
            self._max_misses = int(max_misses)
        if matching_strategy is not None:
            self._matching_strategy = str(matching_strategy)
        if high_conf_thresh is not None:
            self._high_conf_thresh = float(high_conf_thresh)
        if velocity_clip is not None:
            self._velocity_clip = float(velocity_clip)
        if second_stage_iou_min is not None:
            self._second_stage_iou_min = float(second_stage_iou_min)
        for tracker in self._trackers.values():
            tracker.set_tracker_params(
                iou_threshold=self._iou_threshold,
                max_misses=self._max_misses,
                matching_strategy=self._matching_strategy,
                high_conf_thresh=self._high_conf_thresh,
                velocity_clip=self._velocity_clip,
                second_stage_iou_min=self._second_stage_iou_min,
            )
        self._clear_slot_caches()
        logging.info(
            "Tracker params: strategy=%s iou=%.2f misses=%d vel_clip=%.2f",
            self._matching_strategy, self._iou_threshold, self._max_misses, self._velocity_clip,
        )

    def set_denoise(self, method: str, strength: int = 5) -> None:
        """运行时设置去噪方法和强度。

        method: "none" / "bilateral" / "nl_means"
        strength: 1-10，控制去噪强度
        """
        self._denoise_method = method if method in ("none", "bilateral", "nl_means") else "none"
        self._denoise_strength = max(1, min(10, int(strength)))
        logging.info("Denoise set to: %s (strength=%d)", self._denoise_method, self._denoise_strength)

    def _denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """对帧做去噪预处理，返回去噪后的帧。

        参数已保守化：优先保留边缘与纹理，避免人物/手势/矩形特征丢失。
        strength 1~10 映射到非线性保守区间。
        """
        method = self._denoise_method
        strength = self._denoise_strength

        if method == "bilateral":
            # 保守映射：d 不超过 7，sigmaColor 13~40，sigmaSpace 7~25
            d = 5 if strength <= 4 else 7
            sigma_color = 10.0 + strength * 3.0
            sigma_space = 5.0 + strength * 2.0
            return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

        if method == "nl_means":
            # 保守映射：h 2.8~10
            h = 2.0 + strength * 0.8
            template_size = 5 if frame.shape[0] > 720 else 7
            search_size = 15 if frame.shape[0] > 720 else 21
            # 仅在 strength>=7 且 >720p 时降采样 1.5x，减少细节损失
            if frame.shape[0] > 720 and strength >= 7:
                new_w = int(frame.shape[1] / 1.5)
                new_h = int(frame.shape[0] / 1.5)
                small = cv2.resize(frame, (new_w, new_h))
                denoised = cv2.fastNlMeansDenoisingColored(small, None, h, h, template_size, search_size)
                return cv2.resize(denoised, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            return cv2.fastNlMeansDenoisingColored(frame, None, h, h, template_size, search_size)

        return frame

    def set_rectangle_detection_enabled(self, enabled: bool) -> None:
        self._rectangle_enabled = bool(enabled)
        for tracker in self._rectangle_trackers.values():
            tracker.reset()
        self._clear_slot_caches()

    def set_person_detection_enabled(self, enabled: bool) -> None:
        self._person_enabled = bool(enabled)
        for tracker in self._trackers.values():
            tracker.reset()
        self._clear_slot_caches()

    def set_skeleton_enabled(self, enabled: bool) -> None:
        self._skeleton_enabled = bool(enabled)
        for tracker in self._trackers.values():
            tracker.reset()
        self._clear_slot_caches()

    def set_feature_flags(self, *, person: bool, skeleton: bool, rectangle: bool, gesture: bool = False, gesture_mode: str | None = None) -> None:
        person_changed = self._person_enabled != bool(person)
        skeleton_changed = self._skeleton_enabled != bool(skeleton)
        requested_gesture_mode = normalize_gesture_mode(gesture_mode if gesture_mode is not None else ("body" if gesture else "off"))
        if not bool(person) and requested_gesture_mode == "body":
            requested_gesture_mode = "off"
        elif not bool(person) and requested_gesture_mode == "all":
            requested_gesture_mode = "hand"
        gesture_changed = self._gesture_mode != requested_gesture_mode
        rectangle_changed = self._rectangle_enabled != bool(rectangle)
        self._person_enabled = bool(person)
        self._skeleton_enabled = bool(skeleton) and self._person_enabled
        self._gesture_mode = requested_gesture_mode
        self._rectangle_enabled = bool(rectangle)
        if person_changed or skeleton_changed or gesture_changed:
            for tracker in self._trackers.values():
                tracker.reset()
        if rectangle_changed:
            for tracker in self._rectangle_trackers.values():
                tracker.reset()
        if person_changed or skeleton_changed or gesture_changed or rectangle_changed:
            self._clear_slot_caches()

    def set_rectangle_sensitivity(self, level: str) -> None:
        normalized = normalize_rectangle_sensitivity(level)
        if self._rectangle_sensitivity == normalized:
            return
        self._rectangle_sensitivity = normalized
        self._rectangle_detector = RectangleDetector.from_preset(normalized)
        self._rectangle_detector.set_max_rectangles(self._rectangle_max_count)
        for slot_id in list(self._rectangle_trackers.keys()):
            self._rectangle_trackers[slot_id] = RectangleTracker.from_preset(normalized)
        self._clear_slot_caches()
        logging.info("Rectangle sensitivity changed: %s", normalized)

    def set_rectangle_max_count(self, count: int) -> None:
        normalized = max(1, min(10, int(count)))
        if self._rectangle_max_count == normalized:
            return
        self._rectangle_max_count = normalized
        self._rectangle_detector.set_max_rectangles(normalized)
        for tracker in self._rectangle_trackers.values():
            tracker.reset()
        self._clear_slot_caches()
        logging.info("Rectangle max count changed: %d", normalized)

    def set_rectangle_target_color(self, rgb: tuple[int, int, int] | None) -> None:
        normalized = None if rgb is None else tuple(max(0, min(255, int(value))) for value in rgb)
        if self._rectangle_target_rgb == normalized:
            return
        self._rectangle_target_rgb = normalized
        self._clear_slot_caches()
        logging.info("Rectangle target color changed: %s", normalized)

    def set_rectangle_color_threshold(self, threshold: float) -> None:
        normalized = max(1.0, min(100.0, float(threshold)))
        if abs(self._rectangle_color_threshold - normalized) < 1e-6:
            return
        self._rectangle_color_threshold = normalized
        self._clear_slot_caches()
        logging.info("Rectangle color threshold changed: %.1f", normalized)

    def set_gesture_mode(self, mode: str) -> None:
        normalized = normalize_gesture_mode(mode)
        if not self._person_enabled and normalized == "body":
            normalized = "off"
        elif not self._person_enabled and normalized == "all":
            normalized = "hand"
        if self._gesture_mode == normalized:
            return
        self._gesture_mode = normalized
        for tracker in self._trackers.values():
            tracker.reset()
        self._clear_slot_caches()
        logging.info("Gesture mode changed: %s", normalized)

    def _clear_slot_caches(self) -> None:
        """同步清空各 slot 的检测结果缓存，防止跳帧时回放过期数据造成闪烁。"""
        self._last_detections.clear()

    def request_model_switch(self, model_path: str) -> None:
        normalized_path = str(model_path or "")
        with self._lock:
            if normalized_path == self._model_path and self._pending_model_path is None:
                self.model_ready.emit(normalized_path, True, "模型未变化")
                return
            self._pending_model_path = normalized_path
            self._model_switching = True
            self._clear_frame_buffers_locked()
            self._last_detections.clear()
            for tracker in self._trackers.values():
                tracker.reset()
        logging.info("InferWorker model switch requested: %s", normalized_path or "<auto>")

    def _clear_frame_buffers_locked(self) -> None:
        for slot in self._buffers.values():
            slot.frame = None

    def prepare_startup_guard(self, seconds: float | None = None) -> None:
        self._startup_guard_until = time.monotonic() + (self._startup_guard_seconds if seconds is None else seconds)
        for tracker in self._trackers.values():
            tracker.reset()
        self._clear_slot_caches()
        logging.info("InferWorker startup guard enabled for %.2fs", self._startup_guard_until - time.monotonic())

    def _is_startup_guard_active(self) -> bool:
        return time.monotonic() < self._startup_guard_until

    @staticmethod
    def _normalize_backend_mode(mode: str) -> str:
        normalized = str(mode or "auto").lower()
        return normalized if normalized in {"auto", "directml", "pytorch", "cpu"} else "auto"

    def _initialize_backend(self, model_path: str | None = None) -> None:
        self._detect_model = None
        self._pose_model = None
        self._detect_model_path = ""
        self._pose_model_path = ""
        self._ai_disabled = False
        # UHD 仅支持 ONNX，即使在 cpu 模式下也强制走 ONNX CPU provider
        force_onnx_for_uhd = _is_uhd_model(model_path)
        if not force_onnx_for_uhd and self._backend_mode in {"pytorch", "cpu"}:
            self._runtime_backend = "pytorch"
            self._model = self._load_detect_model(model_path)
            return
        try:
            self._load_onnx_detect_model(model_path)
            self._runtime_backend = "onnx"
            self._model = self._detect_model
            logging.info("InferWorker using ONNX backend mode=%s uhd=%s", self._backend_mode, force_onnx_for_uhd)
        except Exception as err:
            if self._backend_mode == "directml":
                self._runtime_backend = "disabled"
                self._ai_disabled = True
                self._model = None
                logging.error("DirectML backend initialization failed, AI inference disabled: %s", err)
                self.model_ready.emit(model_path or "", False, f"DirectML 初始化失败：{err}")
                return
            logging.warning("ONNX backend initialization failed, falling back to PyTorch: %s", err)
            self._runtime_backend = "pytorch"
            self._model = self._load_pytorch_detect_model(model_path)

    def _warmup(self) -> None:
        if self._warmup_done:
            return
        try:
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            for _ in range(2):
                model = self._select_inference_model()
                if model is not None:
                    self._predict_model(model, dummy, self._conf)
            if self._super_res.available:
                roi = np.zeros((32, 32, 3), dtype=np.uint8)
                self._super_res.upscale(roi)
            logging.info("InferWorker warmup completed")
        except Exception as e:
            logging.warning("InferWorker warmup failed: %s", e)
        finally:
            self._warmup_done = True

    def _switch_model_if_needed(self) -> bool:
        with self._lock:
            pending = self._pending_model_path
            self._pending_model_path = None

        if pending is None:
            return False

        old_model = self._detect_model
        old_model_path = self._detect_model_path
        old_custom_model_path = self._custom_model_path
        label = pending or "<auto>"
        try:
            logging.info("InferWorker switching model to %s", label)
            self._initialize_backend(pending or None)
            self._model_path = pending
            self._custom_model_path = pending
            self._warmup_done = False
            self._warmup()
            with self._lock:
                self._clear_frame_buffers_locked()
                self._last_detections.clear()
                for tracker in self._trackers.values():
                    tracker.reset()
                self._model_switching = False
            self.model_ready.emit(pending, True, f"模型已切换：{label}")
        except Exception as e:
            self._detect_model = old_model
            self._detect_model_path = old_model_path
            self._model_path = old_custom_model_path
            self._custom_model_path = old_custom_model_path
            with self._lock:
                self._clear_frame_buffers_locked()
                self._last_detections.clear()
                for tracker in self._trackers.values():
                    tracker.reset()
                self._model_switching = False
            logging.error("InferWorker model switch failed: %s", e)
            self.model_ready.emit(pending, False, f"模型切换失败：{e}")
        return True

    def _detect_device(self) -> str:
        try:
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                logging.info("CUDA available: %s", device_name)
                return "cuda"
        except RuntimeError as e:
            logging.warning("CUDA detection failed (driver mismatch?): %s, falling back to CPU", e)
        logging.info("Using CPU for inference")
        return "cpu"

    def _load_pose_model(self):
        if self._runtime_backend == "onnx":
            return self._load_onnx_pose_model()
        return self._load_pytorch_pose_model()

    def _load_pytorch_pose_model(self) -> YOLO:
        resolved = _resolve_artifact_path("yolov8n-pose.pt")
        model_path = str(resolved) if resolved is not None else ""
        if self._pose_model is not None and self._pose_model_path == model_path:
            return self._pose_model

        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            logging.warning("Local pose model not found at %s. Auto-downloading yolov8n-pose.pt...", model_path)
            model = YOLO("yolov8n-pose.pt")

        self._pose_model = model
        self._pose_model_path = model_path
        logging.info("YOLOv8 Pose model loaded on device: %s path=%s", self._device, model_path or "yolov8n-pose.pt")
        return model

    def _load_detect_model(self, model_path: str | None = None):
        if model_path and _is_yolo26_model(model_path):
            return self._load_yolo26_model(model_path)
        if self._runtime_backend == "onnx":
            return self._load_onnx_detect_model(model_path)
        return self._load_pytorch_detect_model(model_path)

    def _load_yolo26_model(self, model_path: str) -> Yolo26Backend:
        if self._detect_model is not None and self._detect_model_path == model_path:
            return self._detect_model
        task = _infer_model_task_from_path(model_path)
        model = Yolo26Backend(model_path, task=task)
        self._detect_model = model
        self._detect_model_path = model_path
        logging.info("YOLO26 model loaded path=%s task=%s", model_path, task.name)
        return model

    def _load_pytorch_detect_model(self, model_path: str | None = None) -> YOLO:
        if model_path is None:
            resolved = _resolve_artifact_path("yolov8n.pt")
            model_path = str(resolved) if resolved is not None else ""
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        if self._detect_model is not None and self._detect_model_path == model_path:
            return self._detect_model

        if model_path and os.path.exists(model_path):
            model = YOLO(model_path)
        else:
            logging.warning("Local detect model not found at %s. Auto-downloading yolov8n.pt...", model_path)
            model = YOLO("yolov8n.pt")
        self._detect_model = model
        self._detect_model_path = model_path
        _debug_report(
            "E",
            "ai/inference.py:_load_detect_model",
            "model loaded",
            {
                "requested_model_path": model_path,
                "device": self._device,
                "task": getattr(model, "task", None),
                "names_count": len(getattr(model, "names", {}) or {}),
            },
        )
        logging.info("YOLOv8 Detect model loaded on device: %s path=%s", self._device, model_path or "yolov8n.pt")
        return model

    def _load_onnx_detect_model(self, model_path: str | None = None) -> OnnxYoloBackend | UhdBackend:
        if model_path is None:
            resolved = _resolve_artifact_path("yolov8n.onnx")
            model_path = str(resolved) if resolved is not None else ""
        elif not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        if self._detect_model is not None and self._detect_model_path == model_path:
            return self._detect_model
        if _is_uhd_model(model_path):
            model = UhdBackend(model_path, provider_mode=self._backend_mode)
        else:
            provider_mode = "cpu" if self._backend_mode == "cpu" else "directml"
            task = ModelTask.DETECT
            model = OnnxYoloBackend(model_path, task=task, provider_mode=provider_mode, imgsz=self._model_imgsz)
        self._detect_model = model
        self._detect_model_path = model_path
        logging.info("ONNX Detect model loaded path=%s providers=%s", model_path, model.providers)
        return model

    def _load_onnx_pose_model(self) -> OnnxYoloBackend:
        resolved = _resolve_artifact_path("yolov8n-pose.onnx")
        model_path = str(resolved) if resolved is not None else ""
        if self._pose_model is not None and self._pose_model_path == model_path:
            return self._pose_model
        provider_mode = "cpu" if self._backend_mode == "cpu" else "directml"
        model = OnnxYoloBackend(model_path, task=ModelTask.POSE, provider_mode=provider_mode, imgsz=self._model_imgsz)
        self._pose_model = model
        self._pose_model_path = model_path
        logging.info("YOLOv8 ONNX Pose model loaded path=%s providers=%s", model_path, model.providers)
        return model

    def _select_inference_model(self):
        if self._ai_disabled:
            return None
        # 用户显式选择的自定义模型（UHD / YOLO26）优先使用；姿态关键点由 _run_inference 按需融合
        if _is_uhd_model(self._custom_model_path or "") or _is_yolo26_model(self._custom_model_path or ""):
            return self._load_detect_model(self._custom_model_path or None)
        if self._rectangle_enabled or self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled():
            return self._load_pose_model()
        return self._load_detect_model(self._custom_model_path or None)

    def _predict_model(self, model, frame: np.ndarray, conf: float):
        """调用模型并返回原始结果。

        - Backend（ONNX / UHD / YOLO26）返回 list[Detection]。
        - PyTorch YOLOv8 .pt 返回 ultralytics Results 迭代器，由 _run_inference 进一步解析。
        """
        if isinstance(model, (OnnxYoloBackend, UhdBackend, Yolo26Backend)):
            return model.predict(frame, conf=conf)
        return model(frame, verbose=False, conf=conf, imgsz=self._model_imgsz, device=self._device)

    def _merge_detections_with_pose(
        self,
        base_dets: list[dict],
        pose_dets: list[dict],
    ) -> list[dict]:
        """将 pose 关键点按鼻子点落入 bbox 的原则融合到基础检测框。"""
        if not pose_dets:
            return base_dets

        merged = []
        for udet in base_dets:
            best_kpts = []
            best_score = -1.0
            for pdet in pose_dets:
                kpts = pdet.get("keypoints") or []
                if len(kpts) != 17:
                    continue
                nose = kpts[0]
                if (
                    udet["x1"] <= nose["x"] <= udet["x2"]
                    and udet["y1"] <= nose["y"] <= udet["y2"]
                ):
                    score = sum(kp.get("conf", 0.0) for kp in kpts) / 17
                    if score > best_score:
                        best_score = score
                        best_kpts = kpts
            udet = dict(udet)
            udet["keypoints"] = best_kpts
            merged.append(udet)
        return merged

    @staticmethod
    def _detections_to_dicts(detections: list[Detection]) -> list[dict]:
        """将统一 Detection 类型转换为现有模块消费的 dict schema。"""
        return [d.to_dict() for d in detections]

    def _run_inference(self, frame: np.ndarray) -> list[dict]:
        try:
            startup_guard_active = self._is_startup_guard_active()
            output_conf = self._conf
            model = self._select_inference_model()
            if model is None:
                return []
            if isinstance(model, OnnxYoloBackend):
                self._infer_cycle_count += 1
                detections = self._detections_to_dicts(
                    self._filter_detections(model.predict(frame, conf=self._conf))
                )
                if self._debug_infer_reports < 10:
                    _debug_report(
                        "B",
                        "ai/inference.py:_run_inference",
                        "onnx inference completed",
                        {
                            "detections_count": len(detections),
                            "startup_guard_active": startup_guard_active,
                            "output_conf": output_conf,
                            "runtime_backend": self._runtime_backend,
                            "sample_detection": detections[0] if detections else None,
                        },
                    )
                    self._debug_infer_reports += 1
                return detections

            if isinstance(model, Yolo26Backend):
                self._infer_cycle_count += 1
                y26_dets = self._filter_detections(model.predict(frame, conf=self._conf))
                # 若启用了姿态/手势，额外用 YOLO pose 提取关键点并融合
                if self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled():
                    pose_model = self._load_pose_model()
                    pose_dets = self._detections_to_dicts(
                        self._filter_detections(pose_model.predict(frame, conf=self._conf))
                    )
                    y26_dicts = self._detections_to_dicts(y26_dets)
                    detections = self._merge_detections_with_pose(y26_dicts, pose_dets)
                else:
                    detections = self._detections_to_dicts(y26_dets)
                if self._debug_infer_reports < 10:
                    _debug_report(
                        "B",
                        "ai/inference.py:_run_inference",
                        "yolo26 inference completed",
                        {
                            "detections_count": len(detections),
                            "yolo26_count": len(y26_dets),
                            "startup_guard_active": startup_guard_active,
                            "output_conf": output_conf,
                            "runtime_backend": self._runtime_backend,
                            "sample_detection": detections[0] if detections else None,
                        },
                    )
                    self._debug_infer_reports += 1
                return detections

            if isinstance(model, UhdBackend):
                self._infer_cycle_count += 1
                uhd_dets = self._filter_inference_detections(model.predict(frame, conf=self._conf))
                # 若启用了姿态/手势，额外用 YOLO pose 提取关键点并融合
                if self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled():
                    pose_model = self._load_onnx_pose_model()
                    pose_dets = self._detections_to_dicts(
                        self._filter_detections(pose_model.predict(frame, conf=self._conf))
                    )
                    detections = self._merge_detections_with_pose(uhd_dets, pose_dets)
                else:
                    detections = uhd_dets
                if self._debug_infer_reports < 10:
                    _debug_report(
                        "B",
                        "ai/inference.py:_run_inference",
                        "uhd inference completed",
                        {
                            "detections_count": len(detections),
                            "uhd_count": len(uhd_dets),
                            "startup_guard_active": startup_guard_active,
                            "output_conf": output_conf,
                            "runtime_backend": self._runtime_backend,
                            "sample_detection": detections[0] if detections else None,
                        },
                    )
                    self._debug_infer_reports += 1
                return detections

            coarse_results = self._predict_model(model, frame, self._conf)
            detections = []
            coarse_entries = []
            self._infer_cycle_count += 1

            for r in coarse_results:
                boxes = r.boxes
                keypoints = r.keypoints if hasattr(r, 'keypoints') else None

                for i, box in enumerate(boxes):
                    b = box.xyxyn[0].cpu().numpy()
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    label = r.names[cls_id]

                    if conf < 0.15:
                        continue

                    box_w = b[2] - b[0]
                    box_h = b[3] - b[1]
                    area = box_w * box_h
                    is_tiny = box_w < 0.035 and box_h < 0.035
                    is_low_conf = 0.18 <= conf < 0.42
                    sr_candidate = (
                        label == 'person'
                        and self._person_enabled
                        and self._skeleton_enabled
                        and is_tiny
                        and is_low_conf
                        and self._super_res.available
                    )

                    coarse_entries.append(
                        {
                            "result": r,
                            "keypoints": keypoints,
                            "index": i,
                            "b": b,
                            "conf": conf,
                            "label": label,
                            "sr_candidate": sr_candidate,
                            "area": area,
                        }
                    )

            sr_candidates = [entry for entry in coarse_entries if entry["sr_candidate"]]
            sr_candidates.sort(key=lambda entry: (-entry["conf"], entry["area"]))
            selected_sr = {id(entry) for entry in sr_candidates[: self._max_sr_rois]}
            skipped_budget = max(0, len(sr_candidates) - len(selected_sr))
            self._sr_candidates_total += len(sr_candidates)
            self._sr_skipped_budget_count += skipped_budget
            self._sr_attempt_count += len(selected_sr)

            for entry in coarse_entries:
                r = entry["result"]
                keypoints = entry["keypoints"]
                i = entry["index"]
                b = entry["b"]
                conf = entry["conf"]
                label = entry["label"]

                if label == 'person' and not self._should_keep_person_for_features():
                    continue

                if id(entry) in selected_sr:
                    refined = self._refine_with_super_res(frame, b, frame.shape[1], frame.shape[0])
                    if refined is not None and refined.get("label") == "person":
                        detections.append(refined)
                        self._sr_refined_count += 1
                        self._sr_refined_conf_gain_sum += float(refined["confidence"]) - float(conf)
                        self._sr_refined_person_only_hit_count += 1
                        continue
                    self._sr_refine_failed_count += 1

                if conf < output_conf:
                    continue

                det = {
                    "x1": float(b[0]), "y1": float(b[1]),
                    "x2": float(b[2]), "y2": float(b[3]),
                    "confidence": float(conf),
                    "label": label,
                    "keypoints": []
                }

                if label == 'person' and (self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled()):
                    det["keypoints"] = self._extract_keypoints(keypoints, i)

                detections.append(det)

            # #region debug-point B:model-output
            if self._debug_infer_reports < 10:
                sample_detection = detections[0] if detections else None
                _debug_report(
                    "B",
                    "ai/inference.py:_run_inference",
                    "inference completed",
                    {
                        "detections_count": len(detections),
                        "sr_candidates": len(sr_candidates),
                        "sr_selected": len(selected_sr),
                        "startup_guard_active": startup_guard_active,
                        "output_conf": output_conf,
                        "sr_stats": self.sr_stats(),
                        "sample_detection": sample_detection,
                    },
                )
                self._debug_infer_reports += 1

            if self._infer_cycle_count % 50 == 0:
                stats = self.sr_stats()
                logging.info(
                    "Infer stats: cycles=%d sr_candidates_total=%d sr_attempts=%d sr_refined=%d "
                    "sr_refined_conf_gain_avg=%.4f sr_refined_person_only_hit_rate=%.3f "
                    "sr_refine_failed_count=%d sr_skipped_budget=%d",
                    self._infer_cycle_count,
                    stats["sr_candidates_total"],
                    stats["sr_attempts_total"],
                    stats["sr_refined_total"],
                    stats["sr_refined_conf_gain_avg"],
                    stats["sr_refined_person_only_hit_rate"],
                    stats["sr_refine_failed_count"],
                    stats["sr_skipped_budget_total"],
                )
            # #endregion
            return detections

        except Exception as e:
            logging.error("YOLOv8 inference failed: %s", e)
            return []

    def sr_stats(self) -> dict:
        refined_count = max(1, self._sr_refined_count)
        attempt_count = max(1, self._sr_attempt_count)
        return {
            "sr_candidates_total": self._sr_candidates_total,
            "sr_attempts_total": self._sr_attempt_count,
            "sr_refined_total": self._sr_refined_count,
            "sr_refined_conf_gain_avg": self._sr_refined_conf_gain_sum / refined_count,
            "sr_refined_person_only_hit_rate": self._sr_refined_person_only_hit_count / attempt_count,
            "sr_refine_failed_count": self._sr_refine_failed_count,
            "sr_skipped_budget_total": self._sr_skipped_budget_count,
        }

    def _run_rectangle_detection(self, slot_id: int, frame: np.ndarray) -> list[dict]:
        if not self._rectangle_enabled:
            return []
        try:
            raw_detections = self._rectangle_detector.detect(frame)
            filter_threshold = self._rectangle_output_threshold()
            filtered_count = sum(1 for det in raw_detections if float(det.get("confidence", 0.0)) >= filter_threshold)
            detections = [det for det in raw_detections if float(det.get("confidence", 0.0)) >= filter_threshold]
            tracker = self._rectangle_trackers.get(slot_id)
            if tracker is not None:
                detections = tracker.update(detections)
            detections = [self._annotate_rectangle_color(frame, det) for det in detections]
            detections = sorted(
                detections,
                key=_dedupe_priority,
                reverse=True,
            )[: self._rectangle_detector.max_rectangles]
            self._rectangle_detected_total += len(detections)
            if self._infer_cycle_count % 50 == 0:
                logging.info(
                    "Rectangle stats: total=%d raw=%d filtered=%d current=%d tracks=%d threshold=%.2f tracker_stats=%s track_summary=%s",
                    self._rectangle_detected_total,
                    len(raw_detections),
                    filtered_count,
                    len(detections),
                    len(getattr(tracker, "_tracks", {})) if tracker is not None else 0,
                    filter_threshold,
                    getattr(tracker, "_last_stats", {}) if tracker is not None else {},
                    self._rectangle_track_summary(tracker),
                )
            return detections
        except Exception as e:
            logging.error("Rectangle detection failed: %s", e)
            return []

    def _rectangle_output_threshold(self) -> float:
        return {"low": 0.60, "medium": 0.55, "high": 0.50}.get(self._rectangle_sensitivity, 0.55)

    @staticmethod
    def _rectangle_track_summary(tracker) -> list[dict]:
        if tracker is None:
            return []
        tracks = getattr(tracker, "_tracks", {})
        summary = []
        for track_id, track in list(tracks.items())[:6]:
            summary.append(
                {
                    "id": int(track_id),
                    "hits": int(getattr(track, "hits", 0)),
                    "misses": int(getattr(track, "misses", 0)),
                    "conf": round(float(getattr(track, "det", {}).get("confidence", 0.0)), 3),
                }
            )
        return summary

    def _annotate_rectangle_color(self, frame: np.ndarray, det: dict) -> dict:
        if self._rectangle_target_rgb is None or det.get("label") != "rectangle":
            return det
        sample_rgb = self._sample_rectangle_rgb(frame, det)
        if sample_rgb is None:
            return det
        annotated = dict(det)
        annotated["sample_rgb"] = list(sample_rgb)
        annotated["sample_hsv"] = list(_rgb_to_hsv(sample_rgb))
        distance = _lab_distance(self._rectangle_target_rgb, sample_rgb)
        annotated["color_distance"] = float(distance)
        annotated["color_selected"] = bool(distance <= self._rectangle_color_threshold)
        annotated["target_rgb"] = list(self._rectangle_target_rgb)
        return annotated

    def _sample_rectangle_rgb(self, frame: np.ndarray, det: dict) -> tuple[int, int, int] | None:
        if frame is None or frame.size == 0:
            return None
        frame_h, frame_w = frame.shape[:2]
        polygon = det.get("polygon") or []
        if len(polygon) >= 3:
            points = np.array(
                [
                    [
                        max(0, min(frame_w - 1, int(round(float(point["x"]) * frame_w)))),
                        max(0, min(frame_h - 1, int(round(float(point["y"]) * frame_h)))),
                    ]
                    for point in polygon
                ],
                dtype=np.int32,
            )
            mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
            cv2.fillPoly(mask, [points], 255)
            values = frame[mask > 0]
            if values.size:
                mean = np.mean(values, axis=0)
                return tuple(int(round(float(value))) for value in mean[:3])

        x1 = max(0, min(frame_w - 1, int(round(float(det["x1"]) * frame_w))))
        y1 = max(0, min(frame_h - 1, int(round(float(det["y1"]) * frame_h))))
        x2 = max(0, min(frame_w, int(round(float(det["x2"]) * frame_w))))
        y2 = max(0, min(frame_h, int(round(float(det["y2"]) * frame_h))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        pad_x = int((x2 - x1) * 0.25)
        pad_y = int((y2 - y1) * 0.25)
        cx1 = min(x2 - 1, x1 + pad_x)
        cy1 = min(y2 - 1, y1 + pad_y)
        cx2 = max(cx1 + 1, x2 - pad_x)
        cy2 = max(cy1 + 1, y2 - pad_y)
        roi = frame[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return None
        mean = np.mean(roi.reshape(-1, 3), axis=0)
        return tuple(int(round(float(value))) for value in mean[:3])

    def _run_hand_gesture_detection(self, frame: np.ndarray, pose_detections: list[dict] | None = None) -> list[dict]:
        if not self._hand_gesture_enabled():
            return []
        if self._hand_gesture_recognizer is None:
            self._hand_gesture_recognizer = HandGestureRecognizer()
        try:
            rois = self._build_hand_rois(pose_detections or [], frame.shape[1], frame.shape[0])
            detections = self._hand_gesture_recognizer.detect_in_rois(frame, rois) if rois else []
            if not detections:
                detections = self._hand_gesture_recognizer.detect(frame)
            if self._infer_cycle_count % 30 == 0 or detections:
                logging.info(
                    "Hand gesture stats: mode=%s rois=%d hands=%d",
                    self._gesture_mode,
                    len(rois),
                    len(detections),
                )
            return detections
        except Exception as e:
            logging.error("Hand gesture detection failed: %s", e)
            return []

    def _build_hand_rois(self, detections: list[dict], frame_w: int, frame_h: int) -> list[tuple[float, float, float, float]]:
        _ = frame_w, frame_h
        rois: list[tuple[float, float, float, float]] = []
        for det in detections:
            if det.get("label") != "person":
                continue
            kpts = det.get("keypoints") or []
            if len(kpts) != 17:
                continue
            person_w = max(0.0, float(det.get("x2", 0.0)) - float(det.get("x1", 0.0)))
            person_h = max(0.0, float(det.get("y2", 0.0)) - float(det.get("y1", 0.0)))
            side = max(person_w, person_h) * 0.32
            if side <= 0.0:
                continue
            min_side = max(64.0 / max(frame_w, 1), 64.0 / max(frame_h, 1))
            if side < min_side:
                continue
            for wrist_index in (9, 10):
                wrist = kpts[wrist_index]
                if float(wrist.get("conf", 0.0)) < 0.40:  # 提高 wrist 置信度阈值，减少误 ROI
                    continue
                # 增加肘部约束：手腕点不应孤立存在
                elbow_index = wrist_index - 2
                if elbow_index < 0 or elbow_index >= len(kpts):
                    continue
                elbow = kpts[elbow_index]
                if float(elbow.get("conf", 0.0)) < 0.30:
                    continue
                # 限制 ROI 大小，避免手部区域过大引入背景
                side = max(0.12, min(side, 0.35))
                cx = float(wrist["x"])
                cy = float(wrist["y"])
                rois.append((_clamp01(cx - side * 0.5), _clamp01(cy - side * 0.5), _clamp01(cx + side * 0.5), _clamp01(cy + side * 0.5)))
        return _merge_overlapping_rois(rois, 0.55)

    def _attach_gestures(self, detections: list[dict]) -> list[dict]:
        if not self._body_gesture_enabled():
            return detections
        for det in detections:
            if det.get("label") == "person":
                det["gestures"] = self._gesture_recognizer.detect(det)
        return detections

    def _attach_head_attributes(
        self,
        detections: list[dict],
        frame: np.ndarray,
        frame_id: int,
    ) -> list[dict]:
        """为 person 检测框附加头部属性（hat/mask/sunglass/eye/mouth）。"""
        classifier = self._load_head_classifier()
        if classifier is None:
            return detections

        out = []
        for det in detections:
            if det.get("label") != "person":
                out.append(det)
                continue
            track_id = det.get("track_id")
            attrs = None
            if track_id is not None:
                cached = self._head_attr_cache.get(track_id)
                if cached is not None and (frame_id - cached[0]) < self._head_class_stride:
                    attrs = cached[1]
            if attrs is None:
                attrs = classifier.predict(frame, det)
                if attrs is not None and track_id is not None:
                    self._head_attr_cache[track_id] = (frame_id, attrs)
                    # 防止缓存无限增长：超过阈值时清理最早的一半
                    if len(self._head_attr_cache) > 500:
                        sorted_items = sorted(self._head_attr_cache.items(), key=lambda item: item[1][0])
                        self._head_attr_cache = dict(sorted_items[-250:])
            if attrs is not None:
                det = dict(det)
                det["head_attrs"] = attrs
            out.append(det)
        return out

    def _body_gesture_enabled(self) -> bool:
        return self._gesture_mode in {"body", "all"} and self._person_enabled

    def _hand_gesture_enabled(self) -> bool:
        return self._gesture_mode in {"hand", "all"}

    def _should_keep_person_for_features(self) -> bool:
        return self._person_enabled or self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled()

    def _should_run_yolo_inference(self) -> bool:
        return self._should_keep_person_for_features()

    def _filter_detections(self, detections: list[Detection]) -> list[Detection]:
        """对 Detection 列表做与 _filter_inference_detections 等价的过滤。"""
        filtered: list[Detection] = []
        for det in detections:
            if det.label == "person" and not self._should_keep_person_for_features():
                continue
            if det.label == "person" and not (self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled()):
                det.keypoints = None
            filtered.append(det)
        return filtered

    def _filter_inference_detections(self, detections: list[dict]) -> list[dict]:
        filtered = []
        for det in detections:
            if det.get("label") == "person" and not self._should_keep_person_for_features():
                continue
            if det.get("label") == "person" and not (self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled()):
                det = dict(det)
                det["keypoints"] = []
            filtered.append(det)
        return filtered

    def _filter_display_detections(self, detections: list[dict]) -> list[dict]:
        if self._person_enabled:
            return detections
        return [det for det in detections if det.get("label") != "person"]

    def _extract_keypoints(self, keypoints, index: int) -> list[dict]:
        if keypoints is None or keypoints.xyn is None:
            return []

        try:
            kpts_norm = keypoints.xyn[index].cpu().numpy()
            kpts_conf = keypoints.conf[index].cpu().numpy() if keypoints.conf is not None else [1.0] * len(kpts_norm)
        except Exception:
            return []

        extracted = []
        for j in range(len(kpts_norm)):
            extracted.append({
                "x": float(kpts_norm[j][0]),
                "y": float(kpts_norm[j][1]),
                "conf": float(kpts_conf[j]),
            })
        return extracted

    def _redetect_quality_gate(self, predicted_box, refined_box: dict, original_confidence: float) -> bool:
        """重检测结果质量门控：防止 ROI 跑偏或召回低置信度误检。"""
        try:
            pred = np.asarray(predicted_box, dtype=np.float64).flatten()
            ref = np.array(
                [refined_box["x1"], refined_box["y1"], refined_box["x2"], refined_box["y2"]],
                dtype=np.float64,
            )
        except Exception:
            return False

        if pred.shape[0] < 4 or ref.shape[0] < 4:
            return False

        iou = _box_iou(pred, ref)
        conf = float(refined_box.get("confidence", 0.0))

        # IoU 与置信度双阈值：避免召回误检
        if conf < max(0.15, original_confidence * 0.6) and iou < 0.35:
            return False

        # 尺度突变检查：防止重检测框突然放大/缩小
        pred_w = max(1e-5, pred[2] - pred[0])
        pred_h = max(1e-5, pred[3] - pred[1])
        ref_w = max(1e-5, ref[2] - ref[0])
        ref_h = max(1e-5, ref[3] - ref[1])
        if ref_w < pred_w * 0.5 or ref_w > pred_w * 1.6:
            return False
        if ref_h < pred_h * 0.5 or ref_h > pred_h * 1.6:
            return False

        return True

    def _redetect_person_in_roi(self, frame: np.ndarray, predicted_box, original_confidence: float = 0.0) -> dict | None:
        """用卡尔曼预测框做 ROI 裁剪 + YOLO 重检测（不做超分辨率）。

        输入：归一化预测框 [x1, y1, x2, y2]，以及原 track 置信度（用于质量门控）
        输出：归一化坐标的 detection dict 或 None
        """
        if frame is None or frame.size == 0:
            return None
        # 启动期保护：避免噪声误触发
        if self._is_startup_guard_active():
            return None

        frame_h, frame_w = frame.shape[:2]
        if frame_w <= 0 or frame_h <= 0:
            return None

        try:
            box = np.asarray(predicted_box, dtype=np.float64).flatten()
            if box.shape[0] < 4:
                return None
        except Exception:
            return None

        pad = self._person_redetect_roi_pad
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        bw = box[2] - box[0]
        bh = box[3] - box[1]

        px_cx = cx * frame_w
        px_cy = cy * frame_h
        px_bw = bw * frame_w
        px_bh = bh * frame_h
        pad_x = px_bw * pad
        pad_y = px_bh * pad

        crop_x1 = max(0, int(px_cx - px_bw * 0.5 - pad_x))
        crop_y1 = max(0, int(px_cy - px_bh * 0.5 - pad_y))
        crop_x2 = min(frame_w, int(px_cx + px_bw * 0.5 + pad_x))
        crop_y2 = min(frame_h, int(px_cy + px_bh * 0.5 + pad_y))

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        if crop_w < 24 or crop_h < 24:
            return None
        # ROI 过大则放弃（退化成全图推理无意义）
        if (crop_w * crop_h) > (frame_w * frame_h) * 0.20:
            return None

        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]

        try:
            model = self._select_inference_model()
            if model is None:
                return None
            # UHD 输入为 64x64，ROI 重检测时若 ROI 过小会过度上采样，质量下降
            if isinstance(model, UhdBackend) and (crop_w < 48 or crop_h < 48):
                return None
            # 预测框附近大概率有目标，用稍低 conf 多召回，再由 tracker 的 IoU 阈值过滤
            redetect_conf = max(0.15, self._conf * 0.6)
            results = self._predict_model(model, roi, redetect_conf)
        except Exception:
            return None

        best_det = None
        best_conf = 0.0

        if isinstance(model, (OnnxYoloBackend, UhdBackend, Yolo26Backend)):
            for det in results:
                if isinstance(det, Detection):
                    det_dict = det.to_dict()
                else:
                    det_dict = det
                if det_dict.get("label") != "person":
                    continue
                conf = float(det_dict.get("confidence", 0.0))
                if conf <= best_conf:
                    continue
                best_conf = conf
                best_det = {
                    "x1": float(det_dict["x1"] * crop_w / frame_w + crop_x1 / frame_w),
                    "y1": float(det_dict["y1"] * crop_h / frame_h + crop_y1 / frame_h),
                    "x2": float(det_dict["x2"] * crop_w / frame_w + crop_x1 / frame_w),
                    "y2": float(det_dict["y2"] * crop_h / frame_h + crop_y1 / frame_h),
                    "confidence": conf,
                    "label": "person",
                    "keypoints": [],
                }
        else:
            for r in results:
                for box_item in r.boxes:
                    cls_id = int(box_item.cls[0].item())
                    label = r.names[cls_id]
                    if label != "person":
                        continue
                    conf = box_item.conf[0].item()
                    if conf <= best_conf:
                        continue
                    best_conf = conf
                    b = box_item.xyxyn[0].cpu().numpy()
                    best_det = {
                        "x1": float(b[0] * crop_w / frame_w + crop_x1 / frame_w),
                        "y1": float(b[1] * crop_h / frame_h + crop_y1 / frame_h),
                        "x2": float(b[2] * crop_w / frame_w + crop_x1 / frame_w),
                        "y2": float(b[3] * crop_h / frame_h + crop_y1 / frame_h),
                        "confidence": float(conf),
                        "label": "person",
                        "keypoints": [],
                    }

        # 质量门控：过滤掉与预测框偏差过大或尺度突变的召回结果
        if best_det is not None and not self._redetect_quality_gate(predicted_box, best_det, original_confidence):
            best_det = None

        return best_det

    def _refine_with_super_res(self, frame: np.ndarray, coarse_box, frame_w: int, frame_h: int) -> dict | None:
        scale = self._super_res._scale
        padding = 20

        cx = int((coarse_box[0] + coarse_box[2]) / 2 * frame_w)
        cy = int((coarse_box[1] + coarse_box[3]) / 2 * frame_h)
        bw = int((coarse_box[2] - coarse_box[0]) * frame_w)
        bh = int((coarse_box[3] - coarse_box[1]) * frame_h)

        crop_x1 = max(0, cx - bw // 2 - padding)
        crop_y1 = max(0, cy - bh // 2 - padding)
        crop_x2 = min(frame_w, cx + bw // 2 + padding)
        crop_y2 = min(frame_h, cy + bh // 2 + padding)

        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        roi_area = crop_w * crop_h
        frame_area = frame_w * frame_h
        aspect_ratio = crop_w / max(crop_h, 1)

        if crop_w < 12 or crop_h < 12:
            return None
        if roi_area > frame_area * 0.08:
            return None
        if aspect_ratio < 0.35 or aspect_ratio > 3.5:
            return None

        roi = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        upscaled = self._super_res.upscale(roi)

        if upscaled is None:
            return None

        upscale_h, upscale_w = upscaled.shape[:2]

        try:
            refine_model = self._load_detect_model(self._custom_model_path or None)
            refine_results = self._predict_model(refine_model, upscaled, self._conf)
        except Exception:
            return None

        if isinstance(refine_model, OnnxYoloBackend):
            best_det = None
            best_conf = 0.0
            for det in refine_results:
                conf = float(det.get("confidence", 0.0))
                if conf <= best_conf:
                    continue
                best_conf = conf
                best_det = {
                    "x1": float(det["x1"] * crop_w / frame_w + crop_x1 / frame_w),
                    "y1": float(det["y1"] * crop_h / frame_h + crop_y1 / frame_h),
                    "x2": float(det["x2"] * crop_w / frame_w + crop_x1 / frame_w),
                    "y2": float(det["y2"] * crop_h / frame_h + crop_y1 / frame_h),
                    "confidence": conf,
                    "label": det.get("label", "person"),
                    "keypoints": [],
                }
            return best_det

        best_det = None
        best_conf = 0.0

        for r in refine_results:
            for box in r.boxes:
                b = box.xyxyn[0].cpu().numpy()
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())

                if conf > best_conf:
                    best_conf = conf

                    # 修正坐标映射：上采样图像 → ROI坐标系 → 原图归一化坐标
                    # 关键：需要用原始crop尺寸而非上采样后的尺寸，以抵消上采样倍数
                    orig_x1 = (b[0] * crop_w + crop_x1) / frame_w
                    orig_y1 = (b[1] * crop_h + crop_y1) / frame_h
                    orig_x2 = (b[2] * crop_w + crop_x1) / frame_w
                    orig_y2 = (b[3] * crop_h + crop_y1) / frame_h

                    best_det = {
                        "x1": float(orig_x1), "y1": float(orig_y1),
                        "x2": float(orig_x2), "y2": float(orig_y2),
                        "confidence": float(conf),
                        "label": r.names[cls_id],
                        "keypoints": []
                    }

                    if self._skeleton_enabled and hasattr(r, 'keypoints'):
                        roi_keypoints = self._extract_keypoints(r.keypoints, 0)
                        for kp in roi_keypoints:
                            orig_kx = (kp["x"] * crop_w + crop_x1) / frame_w
                            orig_ky = (kp["y"] * crop_h + crop_y1) / frame_h
                            best_det["keypoints"].append({
                                "x": float(orig_kx),
                                "y": float(orig_ky),
                                "conf": float(kp["conf"]),
                            })

        return best_det

    def submit_frame(self, slot_id: int, frame: np.ndarray) -> None:
        now = time.monotonic()
        interval = self._startup_ai_interval if self._is_startup_guard_active() else self._target_ai_interval
        with self._lock:
            last_submit_ts = self._last_submit_ts_by_slot.get(slot_id, 0.0)
            if interval > 0.0 and now - last_submit_ts < interval:
                return
            self._last_submit_ts_by_slot[slot_id] = now

            slot = self._buffers.get(slot_id)
            if slot is None:
                slot = _FrameSlot()
                self._buffers[slot_id] = slot
            slot.frame = frame
            slot.frame_id += 1
            slot.submitted_ts = now
            # #region debug-point C:frame-buffer
            if self._debug_submit_reports < 10:
                _debug_report(
                    "C",
                    "ai/inference.py:submit_frame",
                    "frame submitted",
                    {
                        "slot_id": slot_id,
                        "frame_id": slot.frame_id,
                        "frame_shape": list(frame.shape),
                        "ai_interval": interval,
                        "startup_guard_active": self._is_startup_guard_active(),
                    },
                )
                self._debug_submit_reports += 1
            # #endregion

    def register_slot(self, slot_id: int) -> None:
        with self._lock:
            if slot_id not in self._buffers:
                self._buffers[slot_id] = _FrameSlot()
                self._trackers[slot_id] = DetectionTracker(
                    redetect_enabled=self._person_redetect_enabled,
                    redetect_max_retries=self._person_redetect_max_retries,
                    filter_type=self._filter_type,
                    mcukf_kernel_sigma=self._mcukf_kernel_sigma,
                    iou_threshold=self._iou_threshold,
                    max_misses=self._max_misses,
                    matching_strategy=self._matching_strategy,
                    high_conf_thresh=self._high_conf_thresh,
                    velocity_clip=self._velocity_clip,
                    second_stage_iou_min=self._second_stage_iou_min,
                    redetect_budget=self._redetect_budget_per_frame,
                    reid_backend=self._reid_backend,
                    reid_min_tracks=self._reid_min_tracks,
                    reid_stride=self._reid_stride,
                )
                self._rectangle_trackers[slot_id] = RectangleTracker.from_preset(self._rectangle_sensitivity)
                self._last_submit_ts_by_slot.pop(slot_id, None)
                logging.info("InferWorker registered slot %d", slot_id)

    def unregister_slot(self, slot_id: int) -> None:
        with self._lock:
            self._buffers.pop(slot_id, None)
            self._trackers.pop(slot_id, None)
            self._rectangle_trackers.pop(slot_id, None)
            self._last_detections.pop(slot_id, None)
            self._last_submit_ts_by_slot.pop(slot_id, None)
            logging.info("InferWorker unregistered slot %d", slot_id)

    def run(self) -> None:
        self._running = True
        logging.info("InferWorker started on device: %s, stride=%d", self._device, self._infer_stride)
        self._warmup()

        while self._running:
            if self._switch_model_if_needed():
                continue

            snapshots: list[tuple[int, int, float, np.ndarray]] = []

            with self._lock:
                for slot_id, slot in self._buffers.items():
                    if slot.frame is not None:
                        snapshots.append((slot_id, slot.frame_id, slot.submitted_ts, slot.frame.copy()))
                        slot.frame = None

            for slot_id, frame_id, submitted_ts, frame in snapshots:
                if not self._running:
                    break

                # --- 跳帧调度：只有命中帧才跑推理，跳帧用卡尔曼预测填充 ---
                should_infer = (frame_id % self._infer_stride) == 0
                if self._pipeline_enabled and should_infer:
                    # --- Pipeline Mode (C8) ---
                    detections = self._run_pipeline_mode(slot_id, frame_id, submitted_ts, frame)
                    self._last_detections[slot_id] = detections
                elif should_infer:
                    # 保留原始帧，不同任务按需选择输入
                    original_frame = frame
                    if self._should_run_yolo_inference():
                        # 人物/姿态检测：可选去噪
                        infer_frame = (
                            self._denoise_frame(original_frame)
                            if self._denoise_method != "none"
                            else original_frame
                        )
                        raw_detections = self._run_inference(infer_frame)
                        tracker = self._trackers.get(slot_id)
                        if tracker is not None:
                            if self._person_redetect_enabled:
                                # 注入重检测回调（使用与主推理相同的输入）
                                def _redetect_cb(predicted_box, original_confidence, _frame=infer_frame):
                                    self._redetect_attempt_count += 1
                                    det = self._redetect_person_in_roi(_frame, predicted_box, original_confidence)
                                    if det is not None:
                                        self._redetect_success_count += 1
                                    else:
                                        self._redetect_failed_count += 1
                                    return det
                                detections = tracker.update(raw_detections, redetect_callback=_redetect_cb, frame=infer_frame)
                            else:
                                detections = tracker.update(raw_detections, frame=infer_frame)
                            # 更新所有 track 的外观直方图
                            tracker.update_appearance(infer_frame)
                        else:
                            detections = raw_detections
                        detections = self._attach_head_attributes(detections, infer_frame, frame_id)
                        detections = self._attach_gestures(detections)
                    else:
                        detections = []
                    # 矩形检测与手势识别使用原始帧，保留边缘细节
                    rectangle_detections = self._run_rectangle_detection(slot_id, original_frame)
                    if rectangle_detections:
                        detections = detections + rectangle_detections
                    hand_gesture_detections = self._run_hand_gesture_detection(original_frame, detections)
                    if hand_gesture_detections:
                        detections = detections + hand_gesture_detections
                    detections = self._filter_display_detections(final_dedupe(detections))
                    # 运行时健康监控：人物置信度持续低迷时自动回退到保守配置
                    self._health_monitor.update(detections)
                    if not self._health_monitor.is_healthy(threshold=0.35):
                        if not self._health_monitor._fallback_active and (
                            self._denoise_method != "none" or self._filter_type != "ukf" or self._person_redetect_enabled
                        ):
                            logging.warning(
                                "Health monitor triggered fallback: person_conf=%.3f; "
                                "disabling denoise, reverting to ukf, disabling redetect",
                                self._health_monitor.get_recent_mean(),
                            )
                            # 保存快照后回退
                            self._health_monitor.trigger_fallback({
                                "denoise": (self._denoise_method, self._denoise_strength),
                                "filter": self._filter_type,
                                "redetect": self._person_redetect_enabled,
                            })
                            self.set_denoise("none", 5)
                            self.set_filter_type("ukf")
                            self.set_person_redetect(enabled=False)
                    elif self._health_monitor.check_recovery():
                        # 恢复用户原始配置
                        snap = self._health_monitor.restore_snapshot()
                        if snap:
                            logging.info(
                                "Health monitor recovery: restoring user config denoise=%s filter=%s redetect=%s",
                                snap["denoise"][0], snap["filter"], snap["redetect"],
                            )
                            self.set_denoise(snap["denoise"][0], snap["denoise"][1])
                            self.set_filter_type(snap["filter"])
                            self.set_person_redetect(enabled=snap["redetect"])
                    latency_ms = (time.monotonic() - submitted_ts) * 1000.0
                    for det in detections:
                        det["source_frame_id"] = frame_id
                        det["source_ts"] = submitted_ts
                        det["latency_ms"] = latency_ms
                    self._last_detections[slot_id] = detections
                else:
                    # 跳帧路径：用卡尔曼预测更新 person 位置，非 person 项复用上次结果
                    tracker = self._trackers.get(slot_id)
                    if tracker is not None:
                        person_predicted = tracker.predict_step()
                        last = self._last_detections.get(slot_id, [])
                        non_person = [d for d in last if d.get("label") != "person"]
                        detections = person_predicted + non_person
                        self._last_detections[slot_id] = detections
                    else:
                        detections = self._last_detections.get(slot_id, [])

                # #region debug-point A:infer-emit
                if self._debug_emit_reports < 20:
                    _debug_report(
                        "A",
                        "ai/inference.py:run",
                        "detection emitted",
                        {
                            "slot_id": slot_id,
                            "frame_id": frame_id,
                            "inferred": should_infer,
                            "detections_count": len(detections),
                        },
                    )
                    self._debug_emit_reports += 1
                # #endregion
                self.detection_ready.emit(slot_id, detections)
                # VisionCore 旁路：更新 TargetManager（不影响现有输出）
                self._bypass_update_targets(slot_id, detections, submitted_ts)

            if not snapshots:
                time.sleep(0.001)

        logging.info("InferWorker stopped")

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
        if self.isRunning():
            logging.warning("InferWorker did not terminate within timeout")
        if self._hand_gesture_recognizer is not None:
            self._hand_gesture_recognizer.close()

    # ------------------------------------------------------------------
    # VisionCore 旁路：TargetManager 集成
    # ------------------------------------------------------------------

    @property
    def target_manager(self) -> TargetManager:
        """VisionCore TargetManager 实例（只读），供外部检查目标状态。"""
        return self._target_mgr

    @property
    def event_bus(self) -> EventBus:
        """VisionCore EventBus 实例（只读），供未来订阅生命周期事件。

        Shadow Integration：当前 InferWorker 与 GUI 均不消费事件，此属性
        仅暴露总线以便后续里程碑接入订阅者（日志、告警、UI 高亮等）。
        """
        return self._event_bus

    def _bypass_update_targets(
        self,
        slot_id: int,
        detections: list[dict],
        timestamp: float,
    ) -> None:
        """将检测结果旁路同步到 TargetManager。

        在 ``detection_ready.emit`` 之后调用，不影响现有检测输出。
        仅更新 TargetManager 内部状态。任何异常被静默捕获，不影响推理流程。

        多 Slot 隔离：``slot_id`` 作为目标身份的一部分参与匹配。
        TargetManager 内部以 ``(slot_id, track_id)`` 二元组建立索引，
        因此 slot0 的 track_id=1 与 slot1 的 track_id=1 不会冲突，
        生成的 target_id 形如 ``S0-T0001`` / ``S1-T0002``。

        即使当前帧该 slot 无任何检测结果（``detections`` 为空），
        仍会调用 ``update_targets`` 以便将过期的 ACTIVE 目标标记为 LOST、
        将超时的 LOST 目标移除。这保证某个 slot 短暂断流时其目标不会
        永久残留。

        检测 dict → Track 的转换复用
        :func:`~visioncore.target_manager.detections_to_tracks`，避免在此
        处手工构造 ``CoreTrack`` / ``CoreDetection`` / ``CoreBBox``。Target
        ID 的生成统一由 :func:`create_target_id` 工厂负责（``track_to_target``
        与 ``TargetManager.create_target`` 共用同一格式 ``S{slot}-T{NNNN}``）。

        Parameters:
            slot_id: 摄像头槽位 id，用于多摄像头目标隔离。
            detections: 原始检测结果列表（dict 格式，与 emit 的完全相同）。
            timestamp: 帧提交时间戳，用于 ``last_seen`` 和过期判断。
        """
        try:
            # 复用 converters.detections_to_tracks，避免手工构造 CoreTrack
            tracks = detections_to_tracks(detections)
            # 即使 tracks 为空也调用 update_targets，触发该 slot 的过期清理
            self._target_mgr.update_targets(tracks, slot_id=slot_id, timestamp=timestamp)
        except Exception:
            pass  # 旁路失败不影响推理流程
