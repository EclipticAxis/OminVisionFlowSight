"""SentinelTrack / VDP - 最小可运行 Demo。

单路摄像头实时视觉闭环：
  摄像头采集 → 实时显示 → YOLO 检测 → 目标跟踪 → 结果叠加显示

运行方式：
  python demo.py
  python demo.py --camera "HD Webcam"
  python demo.py --model models/yolov8n.pt --conf 0.5

验收标准：
  1. 程序可启动
  2. 摄像头可打开
  3. 画面可实时显示
  4. 检测结果显示
  5. 跟踪结果显示
  6. 单帧异常不崩溃
  7. 代码结构清晰
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QImage, QPainter, QColor, QPen, QFont, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QStatusBar,
)

# 项目模块
from common import Frame, Detection, DetectionTracker
from common.pyav_capture import PyAVCapture
from common.yolo_detector import YOLODetector

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("demo")

# 检测框颜色调色板（冷蓝系）
_BOX_COLORS = [
    (100, 180, 255),   # 浅蓝
    (120, 220, 180),   # 青绿
    (180, 160, 255),   # 淡紫
    (255, 180, 120),   # 橙
    (255, 220, 120),   # 黄
    (160, 255, 200),   # 薄荷
    (200, 200, 255),   # 淡蓝白
    (255, 200, 180),   # 桃
]


# ---------------------------------------------------------------------------
# 采集线程
# ---------------------------------------------------------------------------

class CaptureThread(QThread):
    """摄像头采集线程：持续读取帧并通过信号发送。"""

    frame_ready = pyqtSignal(object)   # Frame
    error_occurred = pyqtSignal(str)

    def __init__(self, source: str, parent=None):
        super().__init__(parent)
        self._source = source
        self._capture = PyAVCapture()
        self._running = False

    def run(self):
        self._running = True

        if not self._capture.open(self._source, resolution=(640, 480), fps=30):
            self.error_occurred.emit(f"无法打开摄像头: {self._source}")
            return

        while self._running:
            frame = self._capture.read()
            if frame is not None:
                self.frame_ready.emit(frame)
            else:
                # 读帧失败，等待重试
                self.msleep(10)

        self._capture.close()

    def stop(self):
        self._running = False
        self.wait(3000)


# ---------------------------------------------------------------------------
# 推理线程
# ---------------------------------------------------------------------------

class InferenceThread(QThread):
    """推理 + 跟踪线程：接收帧，执行检测和跟踪，输出结果。"""

    result_ready = pyqtSignal(object)  # Frame (with detections in metadata)
    error_occurred = pyqtSignal(str)

    def __init__(self, model_path: str, conf: float = 0.5,
                 infer_stride: int = 1, parent=None):
        super().__init__(parent)
        self._detector = YOLODetector()
        self._tracker = DetectionTracker(iou_threshold=0.3, max_misses=30)
        self._model_path = model_path
        self._conf = conf
        self._stride = max(1, infer_stride)

        self._frame_queue: list[Frame] = []
        self._running = False
        self._frame_count = 0

        # 性能统计
        self._fps_counter = 0
        self._fps_timer = time.time()
        self._current_fps = 0.0

    def load(self):
        """加载模型（在主线程调用，避免 QThread 启动前的阻塞）。"""
        self._detector.load_model(self._model_path, device="cpu")

    def submit(self, frame: Frame):
        """提交帧到推理队列。"""
        self._frame_queue.append(frame)
        # 队列过长时丢弃旧帧（保持实时性）
        if len(self._frame_queue) > 3:
            self._frame_queue = self._frame_queue[-3:]

    def run(self):
        self._running = True

        while self._running:
            if not self._frame_queue:
                self.msleep(5)
                continue

            frame = self._frame_queue.pop(0)
            self._frame_count += 1

            try:
                # 跳帧控制
                if self._frame_count % self._stride == 0:
                    # 推理帧：完整检测 + 跟踪
                    detections = self._detector.predict(frame.data, conf=self._conf)
                    tracks = self._tracker.update(detections)
                else:
                    # 跳帧：只用卡尔曼预测
                    tracks = self._tracker.predict_step()

                # 将结果附加到 frame metadata
                frame.metadata["detections"] = tracks
                frame.metadata["infer_fps"] = self._current_fps

                # FPS 统计
                self._fps_counter += 1
                elapsed = time.time() - self._fps_timer
                if elapsed >= 1.0:
                    self._current_fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_timer = time.time()

                self.result_ready.emit(frame)

            except Exception as e:
                logger.error("Inference error: %s", e, exc_info=True)
                # 不崩溃，继续处理下一帧
                frame.metadata["detections"] = []
                frame.metadata["infer_fps"] = self._current_fps
                self.result_ready.emit(frame)

    def stop(self):
        self._running = False
        self.wait(3000)


# ---------------------------------------------------------------------------
# 显示画布
# ---------------------------------------------------------------------------

class VideoCanvas(QWidget):
    """视频显示画布：绘制画面 + 检测框 + 跟踪 ID + FPS。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(QSize(640, 480))
        self.setStyleSheet("background-color: #1a1a2e;")

        self._frame: Frame | None = None
        self._qimage: QImage | None = None
        self._display_fps = 0.0
        self._fps_counter = 0
        self._fps_timer = time.time()

    def update_frame(self, frame: Frame):
        """更新显示帧（来自推理线程的结果）。"""
        self._frame = frame

        # 转换为 QImage
        rgb = frame.data
        h, w = rgb.shape[:2]
        # QImage 需要 ARGB 格式
        self._qimage = QImage(
            rgb.data, w, h, w * 3, QImage.Format.Format_RGB888
        ).copy()  # copy 确保数据生命周期

        # FPS 统计
        self._fps_counter += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._display_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_timer = time.time()

        self.update()  # 触发重绘

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景
        painter.fillRect(self.rect(), QColor("#1a1a2e"))

        if self._qimage is None:
            painter.setPen(QColor("#888888"))
            font = QFont("Segoe UI", 12)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             "等待摄像头...")
            return

        # 绘制视频帧（保持宽高比）
        img_w = self._qimage.width()
        img_h = self._qimage.height()
        canvas_w = self.width()
        canvas_h = self.height()

        scale = min(canvas_w / img_w, canvas_h / img_h)
        draw_w = int(img_w * scale)
        draw_h = int(img_h * scale)
        draw_x = (canvas_w - draw_w) // 2
        draw_y = (canvas_h - draw_h) // 2

        painter.drawImage(draw_x, draw_y, self._qimage.scaled(
            draw_w, draw_h, Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

        # 绘制检测框 + 跟踪 ID
        detections = self._frame.metadata.get("detections", []) if self._frame else []
        if detections:
            self._draw_detections(painter, detections, draw_x, draw_y, draw_w, draw_h)

        # 绘制 HUD（FPS / 目标数）
        self._draw_hud(painter, len(detections))

    def _draw_detections(self, painter: QPainter, detections: list[dict],
                          ox: int, oy: int, w: int, h: int):
        """绘制检测框和标签。"""
        label_font = QFont("Segoe UI", 9)
        painter.setFont(label_font)

        for det in detections:
            # 归一化坐标 → 像素坐标
            x1 = ox + det.get("x1", 0) * w
            y1 = oy + det.get("y1", 0) * h
            x2 = ox + det.get("x2", 1) * w
            y2 = oy + det.get("y2", 1) * h

            # 颜色按 track_id 分配
            track_id = det.get("track_id", 0)
            color_idx = track_id % len(_BOX_COLORS)
            color = _BOX_COLORS[color_idx]

            # 检测框
            pen = QPen(QColor(*color), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # 标签
            label = det.get("label", "?")
            conf = det.get("confidence", 0.0)
            state = det.get("track_state", "")

            text = f"#{track_id} {label} {conf:.2f}"
            if state == "predicted":
                text += " [pred]"

            # 标签背景
            fm = painter.fontMetrics()
            text_w = fm.horizontalAdvance(text) + 8
            text_h = fm.height() + 2

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(*color, 200))
            painter.drawRect(int(x1), int(y1 - text_h), text_w, text_h)

            painter.setPen(QColor(255, 255, 255))
            painter.drawText(int(x1 + 4), int(y1 - 4), text)

    def _draw_hud(self, painter: QPainter, target_count: int):
        """绘制 HUD 信息。"""
        hud_font = QFont("Consolas", 10)
        painter.setFont(hud_font)

        infer_fps = self._frame.metadata.get("infer_fps", 0) if self._frame else 0
        lines = [
            f"Display: {self._display_fps:.1f} fps",
            f"Infer:   {infer_fps:.1f} fps",
            f"Targets: {target_count}",
        ]

        painter.setPen(QColor(180, 255, 180))
        for i, line in enumerate(lines):
            painter.drawText(10, 20 + i * 16, line)


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class DemoWindow(QMainWindow):
    """Demo 主窗口。"""

    def __init__(self, model_path: str, conf: float, stride: int):
        super().__init__()
        self.setWindowTitle("SentinelTrack Demo - 单路视觉闭环")
        self.resize(800, 600)

        self._model_path = model_path
        self._conf = conf
        self._stride = stride

        self._capture_thread: CaptureThread | None = None
        self._inference_thread: InferenceThread | None = None

        self._build_ui()
        self._scan_cameras()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # 顶部控制栏
        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)

        self._cam_combo = QComboBox()
        self._cam_combo.setMinimumWidth(200)
        self._cam_combo.setPlaceholderText("选择摄像头...")
        ctrl.addWidget(QLabel("摄像头:"))
        ctrl.addWidget(self._cam_combo)

        self._refresh_btn = QPushButton("刷新")
        self._refresh_btn.clicked.connect(self._scan_cameras)
        ctrl.addWidget(self._refresh_btn)

        self._start_btn = QPushButton("启动")
        self._start_btn.clicked.connect(self._on_start)
        self._start_btn.setStyleSheet("background-color: #2d5f2d; color: white; font-weight: bold;")
        ctrl.addWidget(self._start_btn)

        self._stop_btn = QPushButton("停止")
        self._stop_btn.clicked.connect(self._on_stop)
        self._stop_btn.setEnabled(False)
        ctrl.addWidget(self._stop_btn)

        ctrl.addStretch()

        self._status_label = QLabel("就绪")
        ctrl.addWidget(self._status_label)

        layout.addLayout(ctrl)

        # 视频画布
        self._canvas = VideoCanvas()
        layout.addWidget(self._canvas, stretch=1)

        # 状态栏
        self.setStatusBar(QStatusBar())

    def _scan_cameras(self):
        """扫描可用摄像头。"""
        self._cam_combo.clear()
        self._status_label.setText("扫描摄像头中...")

        try:
            from camera.enumerator import scan_cameras
            cameras = scan_cameras()
        except Exception as e:
            logger.error("Camera scan failed: %s", e)
            cameras = []

        if not cameras:
            self._cam_combo.addItem("（未找到摄像头）", None)
            self._status_label.setText("未找到摄像头")
            return

        for cam in cameras:
            label = f"[{cam.index}] {cam.name}"
            spec = cam.device_spec or cam.name
            self._cam_combo.addItem(label, spec)

        self._status_label.setText(f"找到 {len(cameras)} 个摄像头")

    def _on_start(self):
        """启动采集 + 推理。"""
        idx = self._cam_combo.currentIndex()
        if idx < 0:
            return
        source = self._cam_combo.itemData(idx)
        if source is None:
            self._status_label.setText("请先选择摄像头")
            return

        # 初始化推理线程（加载模型）
        if self._inference_thread is None:
            self._status_label.setText("加载模型中...")
            QApplication.processEvents()
            try:
                self._inference_thread = InferenceThread(
                    self._model_path, self._conf, self._stride
                )
                self._inference_thread.load()
                self._inference_thread.result_ready.connect(self._on_result)
                self._inference_thread.start()
            except Exception as e:
                self._status_label.setText(f"模型加载失败: {e}")
                self._inference_thread = None
                return

        # 启动采集线程
        self._capture_thread = CaptureThread(source)
        self._capture_thread.frame_ready.connect(self._on_frame)
        self._capture_thread.error_occurred.connect(self._on_error)
        self._capture_thread.start()

        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_label.setText("运行中")

    def _on_stop(self):
        """停止采集 + 推理。"""
        if self._capture_thread:
            self._capture_thread.stop()
            self._capture_thread = None

        # 推理线程保持运行（模型已加载），只是不再收到帧
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_label.setText("已停止")

    def _on_frame(self, frame: Frame):
        """采集线程发来新帧：送到显示和推理。"""
        # 直接送推理（推理线程内部有队列）
        if self._inference_thread and self._inference_thread.isRunning():
            self._inference_thread.submit(frame)

        # 如果推理慢，直接显示原始帧（不等待检测结果）
        # 这里用 50ms 的容忍度：如果推理结果 50ms 内没回来，先显示原始帧
        # 但为了简化，我们只显示带检测结果的帧（来自推理线程）

    def _on_result(self, frame: Frame):
        """推理线程返回结果：更新显示。"""
        self._canvas.update_frame(frame)

    def _on_error(self, msg: str):
        self._status_label.setText(f"错误: {msg}")
        self._on_stop()

    def closeEvent(self, event):
        """窗口关闭时清理资源。"""
        if self._capture_thread:
            self._capture_thread.stop()
        if self._inference_thread:
            self._inference_thread.stop()
        event.accept()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SentinelTrack Demo")
    parser.add_argument("--camera", default=None, help="摄像头名称（不指定则下拉选择）")
    parser.add_argument("--model", default="models/yolov8n.pt", help="YOLO 模型路径")
    parser.add_argument("--conf", type=float, default=0.5, help="置信度阈值")
    parser.add_argument("--stride", type=int, default=1, help="推理跳帧间隔 (1=每帧)")
    args = parser.parse_args()

    # 检查模型文件
    if not Path(args.model).exists():
        logger.error("模型文件不存在: %s", args.model)
        logger.info("可用模型:")
        models_dir = Path("models")
        if models_dir.exists():
            for f in models_dir.glob("*.pt"):
                logger.info("  %s", f)
        sys.exit(1)

    # 高 DPI
    import os
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("SentinelTrack")
    app.setStyle("Fusion")

    # 暗色主题（简化版）
    from PyQt6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#1a1a2e"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#16213e"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#16213e"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#e0e0e0"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#0f3460"))
    app.setPalette(palette)

    window = DemoWindow(args.model, args.conf, args.stride)

    # 如果指定了摄像头，直接启动
    if args.camera:
        idx = window._cam_combo.findData(args.camera)
        if idx >= 0:
            window._cam_combo.setCurrentIndex(idx)
            window._on_start()

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
