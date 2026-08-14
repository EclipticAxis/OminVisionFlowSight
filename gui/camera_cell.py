from __future__ import annotations

import random
import time
import json
import logging
import urllib.request
import threading
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QEasingCurve, QPointF, QRectF, Qt, QTimer, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QImage, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    # VisionCore 未来统一模型入口 —— 当前仅用于类型检查，不影响运行时。
    # CameraCell 的绘制逻辑保持不变；这些类型为未来迁移做准备。
    from visioncore.core import Detection as CoreDetection
    from visioncore.core import Frame as CoreFrame

_NO_SIGNAL_SUBTEXTS = [
    "no input. very reasonable.",
    "camera is absent today.",
    "signal line is taking a break.",
    "waiting for a quiet frame.",
]

_SKELETON_CONNECTIONS = [
    # 躯干核心
    (5, 6),     # 左肩-右肩
    (5, 11),    # 左肩-左髋
    (6, 12),    # 右肩-右髋
    (11, 12),   # 左髋-右髋
    # 脊柱（虚拟中轴线）
    (5, 6), (11, 12),
    # 头部
    (0, 1), (0, 2), (1, 3), (2, 4),
    # 左臂
    (5, 7), (7, 9),
    # 右臂
    (6, 8), (8, 10),
    # 左腿
    (11, 13), (13, 15),
    # 右腿
    (12, 14), (14, 16),
]

# COCO 关键点索引定义
_KP_NOSE = 0
_KP_LEFT_EYE = 1
_KP_RIGHT_EYE = 2
_KP_LEFT_EAR = 3
_KP_RIGHT_EAR = 4
_KP_LEFT_SHOULDER = 5
_KP_RIGHT_SHOULDER = 6
_KP_LEFT_ELBOW = 7
_KP_RIGHT_ELBOW = 8
_KP_LEFT_WRIST = 9
_KP_RIGHT_WRIST = 10
_KP_LEFT_HIP = 11
_KP_RIGHT_HIP = 12
_KP_LEFT_KNEE = 13
_KP_RIGHT_KNEE = 14
_KP_LEFT_ANKLE = 15
_KP_RIGHT_ANKLE = 16

# 骨架分区定义（用于分区着色）
_SKELETON_ZONES = {
    "head": [
        (0, 1), (0, 2), (1, 3), (2, 4),
    ],
    "torso": [
        (5, 6), (5, 11), (6, 12), (11, 12),
    ],
    "left_arm": [
        (5, 7), (7, 9),
    ],
    "right_arm": [
        (6, 8), (8, 10),
    ],
    "left_leg": [
        (11, 13), (13, 15),
    ],
    "right_leg": [
        (12, 14), (14, 16),
    ],
}

# 为不同人体实例生成区分色的调色板
_INSTANCE_COLORS = [
    (111, 168, 255),   # 冷蓝
    (143, 224, 255),   # 冰蓝
    (127, 149, 214),   # 静谧蓝灰
    (111, 163, 140),   # 低饱和灰绿
    (167, 139, 250),   # 靛紫
    (185, 163, 106),   # 冷金
    (96, 125, 180),    # 深钢蓝
    (216, 106, 122),   # 克制玫红
]


def _get_instance_color(index: int) -> tuple[int, int, int]:
    """为第 index 个人体实例分配颜色"""
    return _INSTANCE_COLORS[index % len(_INSTANCE_COLORS)]


def _rectangle_display_color(det: dict) -> QColor:
    if det.get("color_selected"):
        rgb = det.get("target_rgb") or det.get("sample_rgb")
        if isinstance(rgb, (list, tuple)) and len(rgb) == 3:
            try:
                return QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
            except Exception:
                pass
    return QColor("#F4B860")


def _get_pose_bbox(keypoints: list[dict], conf_thresh: float = 0.2) -> tuple[float, float, float, float] | None:
    """根据关键点计算姿态包围盒，比检测框更贴合人体"""
    valid_x = [kp["x"] for kp in keypoints if kp["conf"] > conf_thresh]
    valid_y = [kp["y"] for kp in keypoints if kp["conf"] > conf_thresh]
    if not valid_x or not valid_y:
        return None
    return (min(valid_x), min(valid_y), max(valid_x), max(valid_y))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    """异步发送debug报告，避免阻塞UI线程"""
    def _send_report():
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

    thread = threading.Thread(target=_send_report, daemon=True)
    thread.start()


class CameraCell(QWidget):
    double_clicked = pyqtSignal(int)

    _UI_RADIUS = 5
    _BBOX_TTL = 2.0
    _MIN_REPAINT_INTERVAL = 1.0 / 30.0
    _MAX_DETECTION_FRAME_AGE = 0.28

    def __init__(self, slot_id, label, parent=None):
        super().__init__(parent)
        self.slot_id = slot_id
        self._label_text = label
        self._current_frame: QImage | None = None
        self._current_frame_ts = 0.0
        self._bbox_cache: list[dict] = []
        self._last_detect_time = 0.0
        self._idle_subtext = _NO_SIGNAL_SUBTEXTS[0]
        self._fps_counter = 0
        self._fps_last_time = time.monotonic()
        self._current_fps = 0.0
        self._is_connected = False
        self._is_ai_enabled = False
        self._is_recording = False
        self._frame_flipped = False
        self._device_name = ""
        self._hover_progress = 0.0
        self._record_breathe = 0.0
        self._debug_frame_reports = 0
        self._debug_detection_reports = 0
        self._debug_paint_reports = 0
        self._video_rect_cache: QRectF | None = None
        self._cache_frame_size: tuple[int, int] | None = None
        self._last_paint_ts = 0.0
        self._overlay_cache: QImage | None = None
        self._overlay_cache_size: tuple[int, int] | None = None
        self._overlay_cache_rect: tuple[float, float, float, float] | None = None
        self._overlay_dirty = True
        self._background_cache: QImage | None = None
        self._background_cache_size: tuple[int, int] | None = None
        self._background_cache_rect: tuple[float, float, float, float] | None = None
        self._text_width_cache: dict[tuple[str, int, int], float] = {}
        self._bbox_signature: tuple | None = None
        self._placeholder_cache: QImage | None = None
        self._placeholder_cache_size: tuple[int, int] | None = None
        self._placeholder_cache_key: tuple[str, str] | None = None
        self._repaint_count_total: int = 0
        self._repaint_count_frame: int = 0
        self._repaint_count_detection: int = 0
        self._repaint_count_hover: int = 0
        self._repaint_count_breathe: int = 0
        self._repaint_count_state: int = 0
        self._repaint_count_resize: int = 0
        self._repaint_count_other: int = 0
        self._last_repaint_stats_log_ts = time.monotonic()
        self._pill_font = QFont("Segoe UI Variable Text", 9, QFont.Weight.DemiBold)
        self._placeholder_title_font = QFont("Segoe UI Variable Text", 16, QFont.Weight.Bold)
        self._placeholder_sub_font = QFont("Segoe UI Variable Text", 10)

        self.setMouseTracking(True)
        self.setMinimumSize(320, 240)

        # 录制呼吸灯动画
        self._record_breathe_anim = QVariantAnimation(self)
        self._record_breathe_anim.setDuration(2000)
        self._record_breathe_anim.setStartValue(0.0)
        self._record_breathe_anim.setEndValue(1.0)
        self._record_breathe_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._record_breathe_anim.valueChanged.connect(self._on_breathe_changed)
        self._record_breathe_anim.finished.connect(self._toggle_breathe_direction)

        # 悬停动画
        self._hover_animation = QVariantAnimation(self)
        self._hover_animation.setDuration(220)
        self._hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._hover_animation.valueChanged.connect(self._on_hover_value_changed)

    def _toggle_breathe_direction(self):
        if self._record_breathe_anim.direction() == QVariantAnimation.Direction.Forward:
            self._record_breathe_anim.setDirection(QVariantAnimation.Direction.Backward)
        else:
            self._record_breathe_anim.setDirection(QVariantAnimation.Direction.Forward)
        self._record_breathe_anim.start()

    def _on_breathe_changed(self, value):
        self._record_breathe = float(value)
        if self._is_recording:
            self._record_repaint("breathe")
            self.update()

    def clear_stream(self) -> None:
        self._current_frame = None
        self._current_frame_ts = 0.0
        self._bbox_cache = []
        self._last_detect_time = 0.0
        self._fps_counter = 0
        self._current_fps = 0.0
        self._idle_subtext = random.choice(_NO_SIGNAL_SUBTEXTS)
        self._is_connected = False
        self._device_name = ""
        self._record_repaint("state")
        self._overlay_cache = None
        self._overlay_cache_size = None
        self._overlay_cache_rect = None
        self._overlay_dirty = True
        self._background_cache = None
        self._background_cache_size = None
        self._background_cache_rect = None
        self._bbox_signature = None
        self._placeholder_cache = None
        self._placeholder_cache_size = None
        self._placeholder_cache_key = None
        self.update()

    def set_stream_state(
        self,
        *,
        connected: bool | None = None,
        device_name: str | None = None,
        ai_enabled: bool | None = None,
        recording: bool | None = None,
    ) -> None:
        if connected is not None:
            self._is_connected = connected
        if device_name is not None:
            self._device_name = device_name
        if ai_enabled is not None:
            self._is_ai_enabled = ai_enabled
        if recording is not None:
            self._is_recording = recording
            if recording:
                if not self._record_breathe_anim.isActive():
                    self._record_breathe_anim.start()
            else:
                self._record_breathe_anim.stop()
                self._record_breathe = 0.0
        self._overlay_dirty = True
        self._record_repaint("state")
        self.update()

    def has_input_signal(self) -> bool:
        return self._is_connected and self._current_frame is not None

    def set_frame_flipped(self, flipped: bool) -> None:
        if self._frame_flipped == flipped:
            return
        self._frame_flipped = flipped
        self._overlay_cache = None
        self._overlay_cache_size = None
        self._overlay_cache_rect = None
        self._overlay_dirty = True
        self.update()

    def _record_repaint(self, reason: str) -> None:
        if reason == "frame":
            self._repaint_count_frame += 1
        elif reason == "detection":
            self._repaint_count_detection += 1
        elif reason == "hover":
            self._repaint_count_hover += 1
        elif reason == "breathe":
            self._repaint_count_breathe += 1
        elif reason == "state":
            self._repaint_count_state += 1
        elif reason == "resize":
            self._repaint_count_resize += 1
        else:
            self._repaint_count_other += 1

        self._repaint_count_total += 1
        self._log_repaint_stats_if_due()

    def repaint_stats(self) -> dict:
        return {
            "slot_id": self.slot_id,
            "repaint_total": self._repaint_count_total,
            "repaint_frame": self._repaint_count_frame,
            "repaint_detection": self._repaint_count_detection,
            "repaint_hover": self._repaint_count_hover,
            "repaint_breathe": self._repaint_count_breathe,
            "repaint_state": self._repaint_count_state,
            "repaint_resize": self._repaint_count_resize,
            "repaint_other": self._repaint_count_other,
        }

    def _log_repaint_stats_if_due(self) -> None:
        now = time.monotonic()
        if now - self._last_repaint_stats_log_ts < 10.0:
            return
        self._last_repaint_stats_log_ts = now
        stats = self.repaint_stats()
        logging.info(
            "CameraCell[%d] repaint stats: total=%d frame=%d detection=%d hover=%d "
            "breathe=%d state=%d resize=%d other=%d",
            stats["slot_id"],
            stats["repaint_total"],
            stats["repaint_frame"],
            stats["repaint_detection"],
            stats["repaint_hover"],
            stats["repaint_breathe"],
            stats["repaint_state"],
            stats["repaint_resize"],
            stats["repaint_other"],
        )

    def _should_repaint(self) -> bool:
        now = time.monotonic()
        if now - self._last_paint_ts >= self._MIN_REPAINT_INTERVAL:
            self._last_paint_ts = now
            return True
        return False

    def update_frame(self, slot_id: int, frame: np.ndarray) -> None:
        # VisionCore 未来入口: frame 参数未来可标注为 CoreFrame
        if slot_id != self.slot_id:
            return

        self._fps_counter += 1
        current_time = time.monotonic()
        elapsed = current_time - self._fps_last_time
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_last_time = current_time

        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._current_frame = qimg.copy()
        self._current_frame_ts = current_time
        self._is_connected = True

        if self._debug_frame_reports < 10:
            _debug_report(
                "C",
                "gui/camera_cell.py:update_frame",
                "frame received by cell",
                {
                    "slot_id": self.slot_id,
                    "frame_shape": list(frame.shape),
                    "image_size": [self._current_frame.width(), self._current_frame.height()],
                },
            )
            self._debug_frame_reports += 1
        if self._should_repaint():
            self._record_repaint("frame")
            self.update()

    def update_detections(self, slot_id: int, detections: list[dict]) -> None:
        # VisionCore 未来入口: detections 未来可标注为 list[CoreDetection]
        if slot_id != self.slot_id:
            return

        source_ts = max((float(det.get("source_ts", 0.0)) for det in detections), default=0.0)
        if source_ts > 0.0 and self._current_frame_ts - source_ts > self._MAX_DETECTION_FRAME_AGE:
            if self._bbox_cache:
                self._bbox_cache = []
                self._bbox_signature = None
                self._overlay_dirty = True
                self._record_repaint("detection")
                self.update()
            return

        # 无论有没有结果，都更新时间戳（防止漏检一帧就清空）
        self._last_detect_time = time.monotonic()

        incoming_signature = self._build_detection_signature(detections)
        changed = incoming_signature != self._bbox_signature
        if detections:
            self._bbox_cache = detections
            if changed:
                self._bbox_signature = incoming_signature
                self._overlay_dirty = True
        else:
            if self._bbox_cache or self._bbox_signature is not None:
                self._bbox_cache = []
                self._bbox_signature = None
                self._overlay_dirty = True

        if self._debug_detection_reports < 10:
            _debug_report(
                "D",
                "gui/camera_cell.py:update_detections",
                "detections delivered to cell",
                {
                    "slot_id": self.slot_id,
                    "detections_count": len(detections),
                    "sample_detection": detections[0] if detections else None,
                },
            )
            self._debug_detection_reports += 1

        if self._should_repaint():
            self._record_repaint("detection")
            self.update()

    def _on_hover_value_changed(self, value: float) -> None:
        self._hover_progress = float(value)
        self._record_repaint("hover")
        self.update()

    def _animate_hover(self, target: float) -> None:
        self._hover_animation.stop()
        self._hover_animation.setStartValue(self._hover_progress)
        self._hover_animation.setEndValue(target)
        self._hover_animation.start()

    def enterEvent(self, event) -> None:
        self._animate_hover(1.0)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._animate_hover(0.0)
        super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            if self.has_input_signal():
                self.double_clicked.emit(self.slot_id)
                event.accept()
            else:
                event.ignore()
            return
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:
        # 控件尺寸变化后，视频映射矩形必须重建，否则归一化坐标会映射到旧矩形导致检测框偏移
        self._video_rect_cache = None
        self._cache_frame_size = None
        self._overlay_cache = None
        self._overlay_cache_size = None
        self._overlay_cache_rect = None
        self._overlay_dirty = True
        self._background_cache = None
        self._background_cache_size = None
        self._background_cache_rect = None
        self._placeholder_cache = None
        self._placeholder_cache_size = None
        self._placeholder_cache_key = None
        self._record_repaint("resize")
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        card_rect = QRectF(self.rect()).adjusted(6, 6, -6, -6)
        self._paint_card_background(painter, card_rect)

        inner_rect = card_rect.adjusted(10, 10, -10, -10)
        frame_size = (
            (self._current_frame.width(), self._current_frame.height())
            if self._current_frame is not None
            else None
        )
        # 缓存命中条件：已有缓存 + 控件未 resize + 帧尺寸未变（分辨率切换也要重建）
        if self._video_rect_cache is None or self._cache_frame_size != frame_size:
            self._video_rect_cache = self._fit_rect(inner_rect)
            self._cache_frame_size = frame_size
            self._overlay_cache = None
            self._overlay_cache_size = None
            self._overlay_cache_rect = None
            self._overlay_dirty = True
        video_rect = self._video_rect_cache

        if self._current_frame is None:
            self._paint_placeholder(painter, inner_rect)
        else:
            self._paint_frame(painter, video_rect)
            self._paint_bboxes(painter, video_rect)

        self._paint_overlay_info(painter, inner_rect)
        painter.end()

    def _paint_card_background(self, painter: QPainter, rect: QRectF) -> None:
        rect_key = (round(rect.x(), 2), round(rect.y(), 2), round(rect.width(), 2), round(rect.height(), 2))
        size_key = (self.width(), self.height())
        if (
            self._background_cache is None
            or self._background_cache_size != size_key
            or self._background_cache_rect != rect_key
        ):
            self._rebuild_background_cache(rect)

        if self._background_cache is not None:
            painter.drawImage(0, 0, self._background_cache)

        base_border = QColor("#27304A")
        hover_border = QColor("#32405F")
        accent_border = QColor("#6FA8FF")

        if self._hover_progress > 0.0:
            if self._hover_progress < 0.5:
                t = self._hover_progress * 2.0
                border_color = QColor(
                    int(base_border.red() + (hover_border.red() - base_border.red()) * t),
                    int(base_border.green() + (hover_border.green() - base_border.green()) * t),
                    int(base_border.blue() + (hover_border.blue() - base_border.blue()) * t),
                )
            else:
                t = (self._hover_progress - 0.5) * 2.0
                border_color = QColor(
                    int(hover_border.red() + (accent_border.red() - hover_border.red()) * t),
                    int(hover_border.green() + (accent_border.green() - hover_border.green()) * t),
                    int(hover_border.blue() + (accent_border.blue() - hover_border.blue()) * t),
                )

            painter.setPen(QPen(border_color, 1.4))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, self._UI_RADIUS, self._UI_RADIUS)

            glow_pen = QPen(QColor(111, 168, 255, int(32 * self._hover_progress)), 1)
            painter.setPen(glow_pen)
            glow_rect = rect.adjusted(-1, -1, 1, 1)
            painter.drawRoundedRect(glow_rect, self._UI_RADIUS, self._UI_RADIUS)

    def _rebuild_background_cache(self, rect: QRectF) -> None:
        if self.width() <= 0 or self.height() <= 0:
            self._background_cache = None
            self._background_cache_size = None
            self._background_cache_rect = None
            return

        rect_key = (round(rect.x(), 2), round(rect.y(), 2), round(rect.width(), 2), round(rect.height(), 2))
        size_key = (self.width(), self.height())
        background = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        background.fill(Qt.GlobalColor.transparent)
        bg_painter = QPainter(background)
        bg_painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        bg_gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        bg_gradient.setColorAt(0.0, QColor("#1A2238"))
        bg_gradient.setColorAt(0.5, QColor("#12182B"))
        bg_gradient.setColorAt(1.0, QColor("#0B1020"))

        shadow_rect = rect.adjusted(2, 2, 2, 2)
        shadow_gradient = QLinearGradient(shadow_rect.topLeft(), shadow_rect.bottomRight())
        shadow_gradient.setColorAt(0.0, QColor(7, 10, 18, 40))
        shadow_gradient.setColorAt(1.0, QColor(7, 10, 18, 0))
        bg_painter.setPen(Qt.PenStyle.NoPen)
        bg_painter.setBrush(shadow_gradient)
        bg_painter.drawRoundedRect(shadow_rect, self._UI_RADIUS, self._UI_RADIUS)

        bg_painter.setPen(QPen(QColor("#27304A"), 1.5))
        bg_painter.setBrush(bg_gradient)
        bg_painter.drawRoundedRect(rect, self._UI_RADIUS, self._UI_RADIUS)
        bg_painter.end()

        self._background_cache = background
        self._background_cache_size = size_key
        self._background_cache_rect = rect_key

    def _fit_rect(self, bounds: QRectF) -> QRectF:
        if self._current_frame is None:
            return bounds

        frame_w = self._current_frame.width()
        frame_h = self._current_frame.height()
        if frame_w <= 0 or frame_h <= 0:
            return bounds

        scale = min(bounds.width() / frame_w, bounds.height() / frame_h)
        draw_w = frame_w * scale
        draw_h = frame_h * scale
        x = bounds.x() + (bounds.width() - draw_w) / 2
        y = bounds.y() + (bounds.height() - draw_h) / 2
        return QRectF(x, y, draw_w, draw_h)

    def _paint_frame(self, painter: QPainter, rect: QRectF) -> None:
        clip = QPainterPath()
        clip.addRoundedRect(rect, self._UI_RADIUS, self._UI_RADIUS)
        painter.save()
        painter.setClipPath(clip)
        if self._frame_flipped:
            painter.translate(rect.left() + rect.width(), rect.top())
            painter.scale(-1, 1)
            painter.drawImage(QRectF(0, 0, rect.width(), rect.height()), self._current_frame)
        else:
            painter.drawImage(rect, self._current_frame)
        self._paint_bass_strings(painter, rect)
        painter.restore()

    def _paint_placeholder(self, painter: QPainter, rect: QRectF) -> None:
        size_key = (self.width(), self.height())
        content_key = (self._label_text, self._idle_subtext)
        if (
            self._placeholder_cache is None
            or self._placeholder_cache_size != size_key
            or self._placeholder_cache_key != content_key
        ):
            self._rebuild_placeholder_cache(rect, size_key, content_key)

        if self._placeholder_cache is not None:
            painter.drawImage(0, 0, self._placeholder_cache)

    def _rebuild_placeholder_cache(
        self,
        rect: QRectF,
        size_key: tuple[int, int],
        content_key: tuple[str, str],
    ) -> None:
        placeholder = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        placeholder.fill(Qt.GlobalColor.transparent)
        cache_painter = QPainter(placeholder)
        cache_painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        gradient = QLinearGradient(rect.topLeft(), rect.bottomRight())
        gradient.setColorAt(0.0, QColor("#1A2238"))
        gradient.setColorAt(0.5, QColor("#12182B"))
        gradient.setColorAt(1.0, QColor("#0B1020"))
        cache_painter.setPen(Qt.PenStyle.NoPen)
        cache_painter.setBrush(gradient)
        cache_painter.drawRoundedRect(rect, self._UI_RADIUS, self._UI_RADIUS)

        grid_pen = QPen(QColor(50, 64, 95, 34), 1)
        cache_painter.setPen(grid_pen)
        for i in range(0, int(rect.width()), 20):
            cache_painter.drawLine(int(rect.left()) + i, int(rect.top()), int(rect.left()) + i, int(rect.bottom()))
        for i in range(0, int(rect.height()), 20):
            cache_painter.drawLine(int(rect.left()), int(rect.top()) + i, int(rect.right()), int(rect.top()) + i)

        cache_painter.setPen(QColor("#EAF1FF"))
        cache_painter.setFont(self._placeholder_title_font)
        cache_painter.drawText(
            rect.adjusted(0, -16, 0, 0),
            Qt.AlignmentFlag.AlignCenter,
            self._label_text,
        )

        cache_painter.setPen(QColor("#9AA8C7"))
        cache_painter.setFont(self._placeholder_sub_font)
        cache_painter.drawText(
            rect.adjusted(40, 40, -40, 70),
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
            self._idle_subtext,
        )

        cache_painter.end()
        self._placeholder_cache = placeholder
        self._placeholder_cache_size = size_key
        self._placeholder_cache_key = content_key

    def _paint_bass_strings(self, painter: QPainter, rect: QRectF) -> None:
        string_widths = [4, 3, 2, 1]
        string_color = QColor(111, 168, 255, 36)
        start_x = rect.right() - 64

        for index, width in enumerate(string_widths):
            pen = QPen(string_color, width)
            painter.setPen(pen)
            x_pos = start_x + (index * 14)
            painter.drawLine(
                int(x_pos),
                int(rect.top()),
                int(x_pos),
                int(rect.bottom()),
            )

    def _paint_bboxes(self, painter: QPainter, rect: QRectF) -> None:
        if time.monotonic() - self._last_detect_time > self._BBOX_TTL:
            if self._bbox_cache:
                self._bbox_cache = []
                self._overlay_dirty = True
            self._overlay_cache = None
            self._overlay_cache_size = None
            self._overlay_cache_rect = None
            return

        if not self._bbox_cache:
            self._overlay_cache = None
            self._overlay_cache_size = None
            self._overlay_cache_rect = None
            return

        rect_key = (round(rect.left(), 2), round(rect.top(), 2), round(rect.width(), 2), round(rect.height(), 2))
        size_key = (self.width(), self.height())
        if (
            self._overlay_dirty
            or self._overlay_cache is None
            or self._overlay_cache_size != size_key
            or self._overlay_cache_rect != rect_key
        ):
            self._rebuild_overlay_cache(rect)

        if self._overlay_cache is not None:
            painter.drawImage(0, 0, self._overlay_cache)

    @staticmethod
    def _build_detection_signature(detections: list[dict]) -> tuple:
        signature = []
        for index, det in enumerate(detections):
            track_id = det.get("track_id", index)
            kpts = det.get("keypoints") or []
            keypoint_count = len(kpts)
            polygon = det.get("polygon") or []
            polygon_sig = tuple(
                (
                    round(float(point.get("x", 0.0)), 3),
                    round(float(point.get("y", 0.0)), 3),
                )
                for point in polygon
            )
            signature.append(
                (
                    track_id,
                    det.get("label", ""),
                    round(float(det.get("confidence", 0.0)), 3),
                    round(float(det.get("x1", 0.0)), 3),
                    round(float(det.get("y1", 0.0)), 3),
                    round(float(det.get("x2", 0.0)), 3),
                    round(float(det.get("y2", 0.0)), 3),
                    keypoint_count,
                    polygon_sig,
                )
            )
        return tuple(signature)

    def _rebuild_overlay_cache(self, rect: QRectF) -> None:
        if self.width() <= 0 or self.height() <= 0 or not self._bbox_cache:
            self._overlay_cache = None
            self._overlay_cache_size = None
            self._overlay_cache_rect = None
            self._overlay_dirty = False
            return

        rect_key = (round(rect.left(), 2), round(rect.top(), 2), round(rect.width(), 2), round(rect.height(), 2))
        size_key = (self.width(), self.height())

        overlay = QImage(self.size(), QImage.Format.Format_ARGB32_Premultiplied)
        overlay.fill(Qt.GlobalColor.transparent)
        painter = QPainter(overlay)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self._debug_paint_reports < 10:
            _debug_report(
                "A",
                "gui/camera_cell.py:_paint_bboxes",
                "painting detections",
                {
                    "slot_id": self.slot_id,
                    "rect": {
                        "x": round(rect.left(), 2),
                        "y": round(rect.top(), 2),
                        "w": round(rect.width(), 2),
                        "h": round(rect.height(), 2),
                    },
                    "frame_size": [
                        self._current_frame.width() if self._current_frame is not None else None,
                        self._current_frame.height() if self._current_frame is not None else None,
                    ],
                    "detections_count": len(self._bbox_cache),
                    "sample_detection": self._bbox_cache[0] if self._bbox_cache else None,
                },
            )
            self._debug_paint_reports += 1

        for instance_index, det in enumerate(self._bbox_cache):
            conf = det.get("confidence", 0.0)
            if conf < 0.1:
                continue

            alpha_ratio = 1.0 if conf >= 0.2 else (conf - 0.1) / 0.1
            alpha_ratio = max(0.3, alpha_ratio)

            track_id = det.get("track_id", instance_index)
            if det.get("label") == "rectangle":
                base_color = _rectangle_display_color(det)
            elif det.get("label") == "hand_gesture":
                base_color = QColor("#8FE0FF")
            else:
                color_rgb = _get_instance_color(track_id)
                base_color = QColor(int(color_rgb[0]), int(color_rgb[1]), int(color_rgb[2]))

            kpts = det.get("keypoints")
            has_keypoints = kpts and len(kpts) == 17

            display_bbox = self._build_display_bbox(det, kpts if has_keypoints else None)
            display_bbox = self._map_display_bbox(display_bbox)
            x1 = rect.left() + display_bbox[0] * rect.width()
            y1 = rect.top() + display_bbox[1] * rect.height()
            x2 = rect.left() + display_bbox[2] * rect.width()
            y2 = rect.top() + display_bbox[3] * rect.height()

            bbox_rect = QRectF(x1, y1, x2 - x1, y2 - y1)

            polygon_path = self._build_detection_polygon_path(det, rect)
            is_obb = polygon_path is not None and det.get("label") != "rectangle"

            if polygon_path is not None and not is_obb:
                # 矩形识别：沿用原有仅绘制多边形的逻辑
                fill_alpha = int(18 * alpha_ratio)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), fill_alpha))
                painter.drawPath(polygon_path)

                glow_pen = QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), int(46 * alpha_ratio)), 6)
                glow_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(glow_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(polygon_path)

                border_pen = QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), int(215 * alpha_ratio)), 2.5)
                border_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(border_pen)
                painter.drawPath(polygon_path)

                label_text = f"矩形  {det['confidence']:.2f}"
                if det.get("color_distance") is not None:
                    label_text = f"{label_text}  色差 {float(det['color_distance']):.1f}"
                self._draw_label_chip(
                    painter,
                    x1,
                    max(rect.top() + 8, y1 - 32),
                    label_text,
                    QColor(11, 16, 32, 228),
                    base_color,
                )
                continue

            # 预测/重检框视觉降级：填充与发光更淡，边框改虚线
            track_state = det.get("track_state", "normal")
            is_predicted_state = track_state in ("predicted", "redetected")

            fill_alpha = int((22 if not is_predicted_state else 12) * alpha_ratio)
            fill_color = QColor(base_color.red(), base_color.green(), base_color.blue(), fill_alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawRoundedRect(bbox_rect, self._UI_RADIUS, self._UI_RADIUS)

            glow_alpha = int((50 if not is_predicted_state else 26) * alpha_ratio)
            glow_color = QColor(base_color.red(), base_color.green(), base_color.blue(), glow_alpha)
            glow_pen = QPen(glow_color, 6)
            painter.setPen(glow_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(bbox_rect, self._UI_RADIUS, self._UI_RADIUS)

            border_alpha = int((200 if not is_predicted_state else 150) * alpha_ratio)
            border_color = QColor(base_color.red(), base_color.green(), base_color.blue(), border_alpha)
            pen = QPen(border_color, 2.5)
            if is_predicted_state:
                pen.setStyle(Qt.PenStyle.DashLine)
                pen.setDashPattern([6, 4])
            painter.setPen(pen)
            painter.drawRoundedRect(bbox_rect, self._UI_RADIUS, self._UI_RADIUS)

            # OBB：在 bbox 之上叠加绘制旋转多边形
            if is_obb and polygon_path is not None:
                obb_pen = QPen(QColor(base_color.red(), base_color.green(), base_color.blue(), int(215 * alpha_ratio)), 2.0)
                obb_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
                painter.setPen(obb_pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawPath(polygon_path)

            if has_keypoints:
                self._paint_pose_fill(painter, rect, kpts, base_color, alpha_ratio)
                self._paint_skeleton_zones(painter, rect, kpts, base_color, alpha_ratio)
                self._paint_keypoints(painter, rect, kpts, base_color, alpha_ratio)
                self._paint_instance_label(painter, rect, kpts, track_id, base_color)
            else:
                label_text = self._build_detection_label(det, track_id)
                self._draw_label_chip(
                    painter,
                    x1,
                    max(rect.top() + 8, y1 - 32),
                    label_text,
                    QColor(11, 16, 32, 228),
                    base_color,
                )

        painter.end()
        self._overlay_cache = overlay
        self._overlay_cache_size = size_key
        self._overlay_cache_rect = rect_key
        self._overlay_dirty = False

    @staticmethod
    def _build_detection_label(det: dict, track_id: int) -> str:
        gestures = det.get("gestures") or []
        attrs = det.get("head_attrs")
        attr_labels: list[str] = []
        if attrs is not None:
            if attrs.get("hat", 0.0) > 0.5:
                attr_labels.append("hat")
            if attrs.get("sunglass", 0.0) > 0.5:
                attr_labels.append("sunglass")
            if attrs.get("masked", 0.0) > 0.5:
                attr_labels.append("mask")
            eye_open = attrs.get("eye_open", [0.0, 0.0])
            if isinstance(eye_open, (list, tuple)) and len(eye_open) >= 2:
                if eye_open[0] > 0.5 and eye_open[1] > 0.5:
                    attr_labels.append("eye_open")
                elif eye_open[0] <= 0.5 and eye_open[1] <= 0.5:
                    attr_labels.append("eye_closed")
            if attrs.get("mouth_open", 0.0) > 0.5:
                attr_labels.append("mouth_open")

        if det.get("label") == "person" and gestures:
            names = [str(gesture.get("name", "")) for gesture in gestures[:2] if gesture.get("name")]
            if names:
                base = f"{' / '.join(names)}  {det.get('confidence', 0.0):.2f}"
            else:
                base = f"{det['label']} #{track_id}  {det.get('confidence', 0.0):.2f}"
        elif det.get("label") == "hand_gesture":
            name = str(det.get("gesture_name") or "手部")
            base = f"{name}  {det.get('confidence', 0.0):.2f}"
        else:
            base = f"{det['label']} #{track_id}  {det.get('confidence', 0.0):.2f}"

        if attr_labels:
            base = f"{base}  [{'|'.join(attr_labels)}]"

        # 预测/重检状态前缀
        state = det.get("track_state")
        if state == "predicted":
            base = f"[预测] {base}"
        elif state == "redetected":
            base = f"[重检] {base}"
        return base

    def _build_detection_polygon_path(self, det: dict, rect: QRectF) -> QPainterPath | None:
        polygon = det.get("polygon") or []
        if len(polygon) != 4:
            return None

        path = QPainterPath()
        for index, point in enumerate(polygon):
            try:
                x = rect.left() + self._map_display_x(float(point["x"])) * rect.width()
                y = rect.top() + float(point["y"]) * rect.height()
            except Exception:
                return None
            if index == 0:
                path.moveTo(QPointF(x, y))
            else:
                path.lineTo(QPointF(x, y))
        path.closeSubpath()
        return path

    @staticmethod
    def _build_display_bbox(det: dict, kpts: list[dict] | None) -> tuple[float, float, float, float]:
        x1 = float(det["x1"])
        y1 = float(det["y1"])
        x2 = float(det["x2"])
        y2 = float(det["y2"])

        pose_bbox = _get_pose_bbox(kpts, conf_thresh=0.15) if kpts else None
        if pose_bbox is not None:
            x1 = min(x1, pose_bbox[0])
            y1 = min(y1, pose_bbox[1])
            x2 = max(x2, pose_bbox[2])
            y2 = max(y2, pose_bbox[3])

        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        return (
            _clamp01(x1 - width * 0.06),
            _clamp01(y1 - height * 0.10),
            _clamp01(x2 + width * 0.06),
            _clamp01(y2 + height * 0.08),
        )

    def _map_display_x(self, x: float) -> float:
        return 1.0 - x if self._frame_flipped else x

    def _map_display_bbox(self, bbox: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        if not self._frame_flipped:
            return bbox
        return (1.0 - bbox[2], bbox[1], 1.0 - bbox[0], bbox[3])

    def _paint_pose_fill(
        self,
        painter: QPainter,
        rect: QRectF,
        kpts: list[dict],
        base_color: QColor,
        alpha_ratio: float = 1.0,
    ) -> None:
        """绘制人体姿态填充区域（半透明多边形）"""
        conf_thresh = 0.15

        def _kp_point(idx: int) -> tuple[float, float] | None:
            kp = kpts[idx]
            if kp["conf"] <= conf_thresh:
                return None
            return (
                rect.left() + self._map_display_x(kp["x"]) * rect.width(),
                rect.top() + kp["y"] * rect.height(),
            )

        def _build_polygon(indices: list[int]) -> QPainterPath:
            path = QPainterPath()
            points = []
            for idx in indices:
                pt = _kp_point(idx)
                if pt is not None:
                    points.append(pt)
            if len(points) < 3:
                return path
            path.moveTo(points[0][0], points[0][1])
            for pt in points[1:]:
                path.lineTo(pt[0], pt[1])
            path.closeSubpath()
            return path

        # 躯干填充（肩-肩-髋-髋）
        torso_indices = [5, 6, 12, 11]  # 左肩 -> 右肩 -> 右髋 -> 左髋
        torso_path = _build_polygon(torso_indices)
        if not torso_path.isEmpty():
            torso_fill = QColor(base_color.red(), base_color.green(), base_color.blue(), int(35 * alpha_ratio))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(torso_fill)
            painter.drawPath(torso_path)

        # 头部填充（鼻子-眼-耳-耳-眼）
        head_indices = [0, 1, 3, 4, 2]  # 鼻子 -> 左眼 -> 左耳 -> 右耳 -> 右目
        head_path = _build_polygon(head_indices)
        if not head_path.isEmpty():
            head_fill = QColor(base_color.red(), base_color.green(), base_color.blue(), int(30 * alpha_ratio))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(head_fill)
            painter.drawPath(head_path)

        # 左臂填充（肩-肘-腕）
        left_arm_indices = [5, 7, 9]
        left_arm_path = _build_polygon(left_arm_indices)
        if not left_arm_path.isEmpty():
            arm_fill = QColor(base_color.red(), base_color.green(), base_color.blue(), int(25 * alpha_ratio))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(arm_fill)
            painter.drawPath(left_arm_path)

        # 右臂填充
        right_arm_indices = [6, 8, 10]
        right_arm_path = _build_polygon(right_arm_indices)
        if not right_arm_path.isEmpty():
            arm_fill = QColor(base_color.red(), base_color.green(), base_color.blue(), int(25 * alpha_ratio))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(arm_fill)
            painter.drawPath(right_arm_path)

        # 左腿填充
        left_leg_indices = [11, 13, 15]
        left_leg_path = _build_polygon(left_leg_indices)
        if not left_leg_path.isEmpty():
            leg_fill = QColor(base_color.red(), base_color.green(), base_color.blue(), int(25 * alpha_ratio))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(leg_fill)
            painter.drawPath(left_leg_path)

        # 右腿填充
        right_leg_indices = [12, 14, 16]
        right_leg_path = _build_polygon(right_leg_indices)
        if not right_leg_path.isEmpty():
            leg_fill = QColor(base_color.red(), base_color.green(), base_color.blue(), int(25 * alpha_ratio))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(leg_fill)
            painter.drawPath(right_leg_path)

    def _paint_instance_label(
        self,
        painter: QPainter,
        rect: QRectF,
        kpts: list[dict],
        instance_index: int,
        base_color: QColor,
    ) -> None:
        """在头顶上方绘制实例编号标签"""
        # 使用鼻子（0）或左眼（1）或右眼（2）作为头顶参考
        for head_idx in [0, 1, 2]:
            kp = kpts[head_idx]
            if kp["conf"] > 0.2:
                x = rect.left() + self._map_display_x(kp["x"]) * rect.width()
                y = rect.top() + kp["y"] * rect.height()
                # 标签位置：头顶上方
                label_y = y - 28
                label_x = x - 12
                label_text = f"{instance_index}"
                # 绘制标签背景（小方块）
                bg_rect = QRectF(label_x, label_y, 24, 24)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(base_color.red(), base_color.green(), base_color.blue(), 200))
                painter.drawRoundedRect(bg_rect, self._UI_RADIUS, self._UI_RADIUS)
                # 绘制文字
                painter.setPen(QColor(255, 255, 255, 240))
                painter.setFont(QFont("Segoe UI Variable Text", 10, QFont.Weight.Bold))
                painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, label_text)
                break

    def _paint_skeleton_zones(
        self,
        painter: QPainter,
        rect: QRectF,
        kpts: list[dict],
        base_color: QColor,
        alpha_ratio: float = 1.0,
    ) -> None:
        """按人体分区绘制骨架，不同区域用不同粗细/透明度"""
        conf_thresh = 0.2

        # 定义各区域的绘制参数：线条粗细、透明度比例
        zone_params = {
            "torso": (3.5, 1.0),      # 躯干最粗最实
            "head": (2.5, 0.9),       # 头部中等
            "left_arm": (2.5, 0.8),
            "right_arm": (2.5, 0.8),
            "left_leg": (3.0, 0.85),
            "right_leg": (3.0, 0.85),
        }

        for zone_name, connections in _SKELETON_ZONES.items():
            width, base_alpha = zone_params.get(zone_name, (2.0, 0.7))
            line_color = QColor(
                base_color.red(),
                base_color.green(),
                base_color.blue(),
                int(200 * base_alpha * alpha_ratio),
            )
            pen = QPen(line_color, width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)

            for p1, p2 in connections:
                kp1, kp2 = kpts[p1], kpts[p2]
                if kp1["conf"] > conf_thresh and kp2["conf"] > conf_thresh:
                    x1_k = rect.left() + self._map_display_x(kp1["x"]) * rect.width()
                    y1_k = rect.top() + kp1["y"] * rect.height()
                    x2_k = rect.left() + self._map_display_x(kp2["x"]) * rect.width()
                    y2_k = rect.top() + kp2["y"] * rect.height()
                    painter.drawLine(round(x1_k), round(y1_k), round(x2_k), round(y2_k))

    def _paint_keypoints(
        self,
        painter: QPainter,
        rect: QRectF,
        kpts: list[dict],
        base_color: QColor,
        alpha_ratio: float = 1.0,
    ) -> None:
        """绘制方块关键点，参考图片中的效果"""
        conf_thresh = 0.15

        # 核心关节索引
        core_joints = {5, 6, 11, 12}  # 双肩 + 双髋
        head_joints = {0, 1, 2, 3, 4}  # 头部
        end_joints = {9, 10, 15, 16}  # 手腕 + 脚踝

        for idx, kp in enumerate(kpts):
            if kp["conf"] <= conf_thresh:
                continue

            x_k = rect.left() + self._map_display_x(kp["x"]) * rect.width()
            y_k = rect.top() + kp["y"] * rect.height()

            # 根据关节重要性调整方块大小
            if idx in core_joints:
                size = 8.0
                alpha = 240
            elif idx in head_joints:
                size = 6.5
                alpha = 200
            elif idx in end_joints:
                size = 7.0
                alpha = 220
            else:
                size = 5.5
                alpha = 180

            alpha = int(alpha * alpha_ratio)
            half = size / 2
            box_rect = QRectF(x_k - half, y_k - half, size, size)

            # 外框（白色边框）
            border_pen = QPen(QColor(255, 255, 255, int(180 * alpha_ratio)), 1.5)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(box_rect, 3, 3)

            # 实心方块（半透明填充）
            fill_color = QColor(base_color.red(), base_color.green(), base_color.blue(), alpha)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(fill_color)
            painter.drawRoundedRect(box_rect.adjusted(1.5, 1.5, -1.5, -1.5), 2, 2)

            # 中心小白点
            center_rect = QRectF(x_k - 1.5, y_k - 1.5, 3, 3)
            painter.setBrush(QColor(255, 255, 255, int(220 * alpha_ratio)))
            painter.drawRoundedRect(center_rect, 1, 1)

    def _paint_overlay_info(self, painter: QPainter, rect: QRectF) -> None:
        top_y = rect.top() + 12
        left_x = rect.left() + 14
        right_x = rect.right() - 14

        # 主标签
        self._draw_label_chip(
            painter,
            left_x,
            top_y,
            self._label_text,
            QColor(11, 16, 32, 220),
            QColor("#EAF1FF"),
        )

        # 设备名
        if self._device_name:
            self._draw_label_chip(
                painter,
                left_x,
                top_y + 32,
                self._device_name,
                QColor(11, 16, 32, 196),
                QColor("#9AA8C7"),
            )

        badges: list[tuple[str, QColor, QColor]] = []
        badges.append(
            ("在线" if self._is_connected else "离线", QColor("#1E2B26"), QColor("#6FA38C"))
            if self._is_connected
            else ("离线", QColor("#351D2A"), QColor("#D86A7A"))
        )
        if self._is_ai_enabled:
            badges.append(("AI", QColor("#18243D"), QColor("#6FA8FF")))
        if self._is_recording:
            # 呼吸灯效果
            breathe_alpha = int(120 + 110 * self._record_breathe)
            badges.append(("REC", QColor(216, 106, 122, breathe_alpha), QColor("#EAF1FF")))

        current_right = right_x
        for text, bg, fg in reversed(badges):
            width = self._measure_pill_width(text)
            x = current_right - width
            self._draw_label_chip(painter, x, top_y, text, bg, fg)
            current_right = x - 10

        # 底部信息
        bottom_y = rect.bottom() - 38
        self._draw_label_chip(
            painter,
            left_x,
            bottom_y,
            f"FPS {self._current_fps:.1f}",
            QColor(11, 16, 32, 220),
            QColor("#EAF1FF"),
        )

        if self._bbox_cache:
            self._draw_label_chip(
                painter,
                left_x + self._measure_pill_width(f"FPS {self._current_fps:.1f}") + 10,
                bottom_y,
                f"目标 {len(self._bbox_cache)}",
                QColor(11, 16, 32, 220),
                QColor("#8FE0FF"),
            )

    def _measure_pill_width(self, text: str) -> float:
        cache_key = (text, self._pill_font.pointSize(), int(self._pill_font.weight()))
        cached = self._text_width_cache.get(cache_key)
        if cached is not None:
            return cached
        metrics = QFontMetrics(self._pill_font)
        width = metrics.horizontalAdvance(text) + 24
        self._text_width_cache[cache_key] = width
        return width

    def _draw_label_chip(
        self,
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        bg_color: QColor,
        fg_color: QColor,
    ) -> None:
        painter.setFont(self._pill_font)
        width = self._measure_pill_width(text)
        rect = QRectF(x, y, width, 26)

        # 绘制背景
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg_color)
        painter.drawRoundedRect(rect, self._UI_RADIUS, self._UI_RADIUS)

        # 绘制文字
        painter.setPen(fg_color)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
