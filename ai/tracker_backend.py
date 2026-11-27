# =============================================================================
# VisionDataPlatform - Rust 后端跟踪器适配层
# =============================================================================
# RustDetectionTracker 与 ai.tracker.DetectionTracker 的公开 API 完全一致：
#   update / update_appearance / predict_step / reset / set_smooth_alpha /
#   set_redetect_config / set_filter_type / set_tracker_params
#
# 分工：跟踪核心（UKF 滤波族、AUTO 自适应、匹配引擎、重检测、生命周期）在
# Rust 扩展 omintrack 内完成；帧相关预处理（HSV 颜色直方图、ReID embedding
# 提取）保留在 Python 侧（cv2 / ReID 后端模型），以 det_hist / det_emb 参数
# 传入 Rust 核心。redetect_callback 仍为 Python callable，由 Rust 回调。
# =============================================================================

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from ai.detection import Detection
# 复用 Python 版的纯函数/方法，保证外观计算路径与 Python 实现逐位一致
from ai.tracker import DetectionTracker as _PyTracker
from ai.tracker import _compute_hsv_hist, _clamp01


class RustDetectionTracker:
    """DetectionTracker 的 Rust 后端版本（公开 API 逐方法对齐）。"""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_misses: int = 4,
        smooth_alpha: float = 0.85,
        min_hits_to_show: int = 1,
        redetect_enabled: bool = False,
        redetect_max_retries: int = 2,
        redetect_iou_threshold: float = 0.20,
        filter_type: str = "ukf",
        mcukf_kernel_sigma: float = 1.0,
        matching_strategy: str = "hungarian",
        high_conf_thresh: float = 0.5,
        velocity_clip: float = 0.30,
        second_stage_iou_min: float = 0.1,
        redetect_budget: int = 1,
        reid_backend=None,
        reid_min_tracks: int = 3,
        reid_stride: int = 3,
        max_output_misses: int = 0,
        # 跟踪增强（docs/tracker-nsa-ocm-hmiou.md，默认全部关闭 opt-in）
        nsa_enabled: bool = False,
        nsa_interval: float = 1.0,
        nsa_min: float = 0.1,
        nsa_conf_thresh: float = 0.6,
        nsa_interval_sec: float = 2.0,
        ocm_enabled: bool = False,
        ocm_delta_t: int = 3,
        ocm_weight: float = 0.2,
        hmiou_enabled: bool = False,
        # 批次 3（docs/tracker-oru-ocr-tcm.md，默认全部关闭 opt-in）
        oru_enabled: bool = False,
        ocr_enabled: bool = False,
        ocr_threshold: float = 0.3,
        tcm_enabled: bool = False,
        tcm_weight: float = 0.2,
    ):
        import omintrack  # 延迟导入：未安装扩展时不影响 Python 后端

        self._core = omintrack.RustTracker(
            iou_threshold=iou_threshold,
            max_misses=max_misses,
            smooth_alpha=smooth_alpha,
            min_hits_to_show=min_hits_to_show,
            redetect_enabled=redetect_enabled,
            redetect_max_retries=redetect_max_retries,
            redetect_iou_threshold=redetect_iou_threshold,
            filter_type=filter_type,
            mcukf_kernel_sigma=mcukf_kernel_sigma,
            matching_strategy=matching_strategy,
            high_conf_thresh=high_conf_thresh,
            velocity_clip=velocity_clip,
            second_stage_iou_min=second_stage_iou_min,
            redetect_budget=redetect_budget,
            max_output_misses=max_output_misses,
            nsa_enabled=nsa_enabled,
            nsa_interval=nsa_interval,
            nsa_min=nsa_min,
            nsa_conf_thresh=nsa_conf_thresh,
            nsa_interval_sec=nsa_interval_sec,
            ocm_enabled=ocm_enabled,
            ocm_delta_t=ocm_delta_t,
            ocm_weight=ocm_weight,
            hmiou_enabled=hmiou_enabled,
            oru_enabled=oru_enabled,
            ocr_enabled=ocr_enabled,
            ocr_threshold=ocr_threshold,
            tcm_enabled=tcm_enabled,
            tcm_weight=tcm_weight,
        )
        self._reid_backend = reid_backend
        self._reid_min_tracks = int(reid_min_tracks)
        self._reid_stride = max(1, int(reid_stride))
        self._reid_frame_counter = 0

    # ------------------------------------------------------------------
    # 帧相关预处理（与 Python 版 update() 中的逻辑逐步对应）
    # ------------------------------------------------------------------

    def _compute_color_hist_for_box(self, frame: np.ndarray, det: dict) -> np.ndarray | None:
        # Python 版同名方法不访问任何实例状态，直接复用保证行为一致
        return _PyTracker._compute_color_hist_for_box(self, frame, det)

    def _prepare_frame_inputs(
        self,
        detections: list[dict],
        frame: np.ndarray | None,
    ) -> tuple[list, list, bool]:
        """计算 det_hist / det_emb / reid_active（无 frame 时全为 None/False）。"""
        n = len(detections)
        det_hist = [None] * n
        det_emb = [None] * n
        reid_active = False

        if frame is not None and frame.size > 0:
            for i, det in enumerate(detections):
                if det.get("label") == "person":
                    det_hist[i] = self._compute_color_hist_for_box(frame, det)

            # mock 止损门控：mock backend 注入的是无语义噪声，ReID 匹配不得激活
            # （getattr 默认 False 兼容无 is_mock 属性的自定义 backend，如测试 fake）
            if self._reid_backend is not None \
                    and not getattr(self._reid_backend, "is_mock", False):
                person_count = sum(1 for d in detections if d.get("label") == "person")
                track_person_count = self._core.person_track_count()
                if person_count >= 1 and track_person_count >= self._reid_min_tracks \
                        and self._reid_frame_counter % self._reid_stride == 0:
                    reid_active = True
                    frame_h, frame_w = frame.shape[:2]
                    for i, det in enumerate(detections):
                        if det.get("label") == "person":
                            x1 = int(_clamp01(float(det["x1"])) * frame_w)
                            y1 = int(_clamp01(float(det["y1"])) * frame_h)
                            x2 = int(_clamp01(float(det["x2"])) * frame_w)
                            y2 = int(_clamp01(float(det["y2"])) * frame_h)
                            if x2 > x1 and y2 > y1:
                                det_emb[i] = self._reid_backend.extract(frame[y1:y2, x1:x2])
        return det_hist, det_emb, reid_active

    # ------------------------------------------------------------------
    # 与 DetectionTracker 对齐的公开 API
    # ------------------------------------------------------------------

    def update(
        self,
        detections: Sequence[dict] | Sequence[Detection],
        redetect_callback: Callable[[np.ndarray, float], dict | None] | None = None,
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        detections = [d.to_dict() if isinstance(d, Detection) else d for d in detections]
        det_hist, det_emb, reid_active = self._prepare_frame_inputs(detections, frame)

        # Rust 侧的回调桥：Python 版约定 callback(box: np.ndarray, confidence)
        # Rust 传入 list[float]，在此包一层转回 ndarray
        cb = None
        if redetect_callback is not None:
            def cb(box, confidence):
                return redetect_callback(np.asarray(box, dtype=np.float64), confidence)

        result = self._core.update(
            detections,
            det_hist=det_hist,
            det_emb=det_emb,
            reid_active=reid_active,
            redetect_callback=cb,
        )
        self._reid_frame_counter += 1
        return result

    def update_appearance(self, frame: np.ndarray) -> None:
        """为所有活跃 person track 更新外观颜色直方图（中心 50% 区域）。"""
        if frame is None or frame.size == 0:
            return
        rois = self._core.appearance_update_rois()
        if not rois:
            return
        frame_h, frame_w = frame.shape[:2]
        hists: dict[int, np.ndarray] = {}
        for tid, box in rois.items():
            # 与 _Track.update_color_hist 的 ROI 提取完全一致
            x1 = int(_clamp01(box[0]) * frame_w)
            y1 = int(_clamp01(box[1]) * frame_h)
            x2 = int(_clamp01(box[2]) * frame_w)
            y2 = int(_clamp01(box[3]) * frame_h)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            w2, h2 = max(2, (x2 - x1) // 4), max(2, (y2 - y1) // 4)
            cx1, cy1 = max(0, cx - w2), max(0, cy - h2)
            cx2, cy2 = min(frame_w, cx + w2), min(frame_h, cy + h2)
            roi = frame[cy1:cy2, cx1:cx2]
            if roi.size == 0:
                continue
            hist = _compute_hsv_hist(roi)
            if hist is not None:
                hists[int(tid)] = hist
        if hists:
            self._core.update_color_hists(hists)

    def predict_step(self) -> list[dict]:
        return self._core.predict_step()

    def reset(self) -> None:
        self._core.reset()
        self._reid_frame_counter = 0

    def set_smooth_alpha(self, alpha: float) -> None:
        self._core.set_smooth_alpha(alpha)

    def set_redetect_config(
        self,
        *,
        enabled: bool | None = None,
        max_retries: int | None = None,
        iou_threshold: float | None = None,
    ) -> None:
        self._core.set_redetect_config(
            enabled=enabled, max_retries=max_retries, iou_threshold=iou_threshold,
        )

    def set_filter_type(self, filter_type: str, kernel_sigma: float | None = None) -> None:
        self._core.set_filter_type(filter_type, kernel_sigma=kernel_sigma)

    def set_tracker_params(
        self,
        *,
        iou_threshold: float | None = None,
        max_misses: int | None = None,
        matching_strategy: str | None = None,
        high_conf_thresh: float | None = None,
        velocity_clip: float | None = None,
        second_stage_iou_min: float | None = None,
        max_output_misses: int | None = None,
        nsa_enabled: bool | None = None,
        nsa_interval: float | None = None,
        nsa_min: float | None = None,
        nsa_conf_thresh: float | None = None,
        nsa_interval_sec: float | None = None,
        ocm_enabled: bool | None = None,
        ocm_delta_t: int | None = None,
        ocm_weight: float | None = None,
        hmiou_enabled: bool | None = None,
        oru_enabled: bool | None = None,
        ocr_enabled: bool | None = None,
        ocr_threshold: float | None = None,
        tcm_enabled: bool | None = None,
        tcm_weight: float | None = None,
    ) -> None:
        self._core.set_tracker_params(
            iou_threshold=iou_threshold,
            max_misses=max_misses,
            matching_strategy=matching_strategy,
            high_conf_thresh=high_conf_thresh,
            velocity_clip=velocity_clip,
            second_stage_iou_min=second_stage_iou_min,
            max_output_misses=max_output_misses,
            nsa_enabled=nsa_enabled,
            nsa_interval=nsa_interval,
            nsa_min=nsa_min,
            nsa_conf_thresh=nsa_conf_thresh,
            nsa_interval_sec=nsa_interval_sec,
            ocm_enabled=ocm_enabled,
            ocm_delta_t=ocm_delta_t,
            ocm_weight=ocm_weight,
            hmiou_enabled=hmiou_enabled,
            oru_enabled=oru_enabled,
            ocr_enabled=ocr_enabled,
            ocr_threshold=ocr_threshold,
            tcm_enabled=tcm_enabled,
            tcm_weight=tcm_weight,
        )

    # ------------------------------------------------------------------
    # 测试/调试内省（直通 Rust 核心）
    # ------------------------------------------------------------------

    def track_count(self) -> int:
        return self._core.track_count()

    def adaptor_current_type(self) -> str | None:
        return self._core.adaptor_current_type()
