"""深度视频显示与主窗口路由契约（offscreen，不加载真实模型）。"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from PyQt6.QtCore import QObject, Qt, pyqtSignal

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _render_center(cell) -> tuple[int, int, int]:
    from PyQt6.QtGui import QColor, QImage

    image = QImage(cell.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(QColor("white"))
    cell.render(image)
    color = image.pixelColor(image.width() // 2, image.height() // 2)
    return color.red(), color.green(), color.blue()


def test_cell_replaces_whole_frame_and_restores_raw(qapp) -> None:
    from gui.camera_cell import CameraCell

    cell = CameraCell(0, "camera")
    cell.resize(320, 240)
    raw = np.zeros((48, 64, 3), dtype=np.uint8)
    raw[:] = (240, 10, 20)
    depth = np.zeros((24, 32, 3), dtype=np.uint8)
    depth[:] = (5, 15, 230)

    cell.update_frame(0, raw)
    assert _render_center(cell) == (240, 10, 20)

    cell.set_depth_mode(True)
    cell.update_depth_frame(0, depth)
    # 精确等于深度源颜色：不是原图叠加、alpha 混合或检测框着色。
    assert _render_center(cell) == (5, 15, 230)

    cell.set_depth_mode(False)
    assert _render_center(cell) == (240, 10, 20)
    cell.deleteLater()


def test_depth_mode_never_paints_detection_overlay(qapp, monkeypatch) -> None:
    from gui.camera_cell import CameraCell

    cell = CameraCell(0, "camera")
    cell.resize(320, 240)
    cell.update_frame(0, np.full((48, 64, 3), 80, dtype=np.uint8))
    cell.set_depth_mode(True)
    cell.update_detections(0, [{"x1": 0, "y1": 0, "x2": 1, "y2": 1}])
    assert cell._bbox_cache == []

    def unexpected_paint(*_args):
        raise AssertionError("深度模式不得调用检测框绘制")

    monkeypatch.setattr(cell, "_paint_bboxes", unexpected_paint)
    assert _render_center(cell) == (80, 80, 80)
    cell.deleteLater()


def test_depth_frame_requires_matching_slot_and_enabled_mode(qapp) -> None:
    from gui.camera_cell import CameraCell

    cell = CameraCell(0, "camera")
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    cell.update_depth_frame(0, frame)
    assert cell._depth_frame is None
    cell.set_depth_mode(True)
    cell.update_depth_frame(1, frame)
    assert cell._depth_frame is None
    cell.update_depth_frame(0, frame)
    assert cell._depth_frame is not None
    non_contiguous = np.zeros((8, 16, 3), dtype=np.uint8)[:, ::2, :]
    assert not non_contiguous.flags["C_CONTIGUOUS"]
    cell.update_depth_frame(0, non_contiguous)
    assert cell._depth_frame is not None
    cell.clear_stream()
    assert cell._depth_frame is None
    cell.deleteLater()


def test_depth_scale_tracks_range_and_colormap(qapp, monkeypatch) -> None:
    from gui.camera_cell import CameraCell
    from ai.depth_worker import colorize_depth

    cell = CameraCell(0, "camera")
    cell.resize(320, 240)
    cell.set_depth_mode(True)
    cell.update_depth_frame(0, np.zeros((48, 64, 3), dtype=np.uint8))
    cell.set_depth_colormap("turbo")
    cell.update_depth_range(0, 0.64, 1.72)
    assert cell._depth_range == (0.64, 1.72)

    _render_center(cell)
    assert cell._depth_palette is not None
    palette = cell._depth_palette
    near = palette.pixelColor(1, 0)
    far = palette.pixelColor(palette.width() - 1, 0)
    assert near.red() > near.blue()  # 近处暖色
    assert far.blue() >= far.red()   # 远处蓝紫

    # 色标上的每个采样点必须与同一米制深度在画面中的颜色完全一致。
    for index in (1, 64, 128, 192, 255):
        depth = float(np.linspace(0.64, 1.72, 256, dtype=np.float32)[index])
        expected = colorize_depth(
            np.array([[depth]], dtype=np.float32),
            vmin=0.64,
            vmax=1.72,
            cmap="turbo",
        )[0, 0]
        actual = palette.pixelColor(index, 0)
        assert (actual.red(), actual.green(), actual.blue()) == tuple(int(v) for v in expected)

    captured_bounds = []
    original_paint = cell._paint_depth_scale

    def capture_bounds(painter, bounds):
        captured_bounds.append(bounds)
        original_paint(painter, bounds)

    monkeypatch.setattr(cell, "_paint_depth_scale", capture_bounds)
    _render_center(cell)
    assert captured_bounds[-1] == cell._video_rect_cache  # 锚定视频区而非卡片黑边

    cell.update_depth_range(0, 0.0, 3.0)
    changed_palette = cell._get_depth_palette()
    assert cell._depth_palette_key == ("turbo", 0.0, 3.0)
    depth = float(np.linspace(0.0, 3.0, 256, dtype=np.float32)[128])
    expected = colorize_depth(
        np.array([[depth]], dtype=np.float32), vmin=0.0, vmax=3.0, cmap="turbo"
    )[0, 0]
    actual = changed_palette.pixelColor(128, 0)
    assert (actual.red(), actual.green(), actual.blue()) == tuple(int(v) for v in expected)

    cell.set_depth_colormap("inferno")
    assert cell._depth_palette is None
    cell.update_depth_range(1, 0.0, 3.0)
    assert cell._depth_range == (0.0, 3.0)
    cell.update_depth_range(0, float("nan"), 3.0)
    assert cell._depth_range == (0.0, 3.0)
    cell.set_depth_mode(False)
    assert cell._depth_range is None
    cell.deleteLater()


class _CameraWorkerStub(QObject):
    frame_ready = pyqtSignal(int, object)

    def __init__(self) -> None:
        super().__init__()


class _InferWorkerStub:
    def __init__(self) -> None:
        self.registered: list[int] = []
        self.unregistered: list[int] = []
        self.submit_calls: list[tuple[int, object]] = []

    def submit_frame(self, slot_id, frame) -> None:
        self.submit_calls.append((slot_id, frame))

    def register_slot(self, slot_id) -> None:
        self.registered.append(slot_id)

    def unregister_slot(self, slot_id) -> None:
        self.unregistered.append(slot_id)


class _DepthWorkerStub:
    def __init__(self) -> None:
        self.registered: list[int] = []
        self.unregistered: list[int] = []
        self.enabled: list[bool] = []
        self.running = False
        self.starts = 0
        self.submit_calls: list[tuple[int, object]] = []

    def submit_frame(self, slot_id, frame) -> None:
        self.submit_calls.append((slot_id, frame))

    def register_slot(self, slot_id) -> None:
        self.registered.append(slot_id)

    def unregister_slot(self, slot_id) -> None:
        self.unregistered.append(slot_id)

    def set_enabled(self, enabled) -> None:
        self.enabled.append(enabled)

    def isRunning(self) -> bool:
        return self.running

    def start(self) -> None:
        self.starts += 1
        self.running = True


class _CellStub:
    def __init__(self) -> None:
        self.depth_modes: list[bool] = []

    def set_depth_mode(self, enabled) -> None:
        self.depth_modes.append(enabled)

    def update_detections(self, _slot_id, _detections) -> None:
        pass


class _WindowHarness:
    def __init__(self) -> None:
        from gui.main_window import MainWindow

        self._safe_direct_connect = MainWindow._safe_direct_connect
        self._disconnect_all = MainWindow._disconnect_all
        self._is_ai_enabled = True
        self._ai_depth_enabled = False
        self._cells = {0: _CellStub()}
        self.camera = _CameraWorkerStub()
        self._camera_capture = SimpleNamespace(
            get_all_workers=lambda: [(0, self.camera)]
        )
        self._infer_worker = _InferWorkerStub()
        self._depth_worker = _DepthWorkerStub()

    def apply(self) -> None:
        from gui.main_window import MainWindow

        MainWindow._apply_depth_state(self)


def test_main_window_switches_inference_route_without_duplicate_connections(qapp) -> None:
    harness = _WindowHarness()
    camera = harness.camera
    infer = harness._infer_worker
    depth = harness._depth_worker

    harness.apply()
    camera.frame_ready.emit(0, "raw-0")
    assert len(infer.submit_calls) == 1
    assert depth.submit_calls == []

    harness._ai_depth_enabled = True
    harness.apply()
    camera.frame_ready.emit(0, "depth-0")
    assert len(infer.submit_calls) == 1
    assert len(depth.submit_calls) == 1
    assert infer.unregistered == []  # 深度预览暂停提交，不销毁 tracker。
    assert depth.enabled[-1] is True and depth.starts == 1
    assert harness._cells[0].depth_modes[-1] is True

    for index in range(4):
        harness.apply()
        camera.frame_ready.emit(0, f"depth-{index + 1}")
    assert len(depth.submit_calls) == 5
    assert len(infer.submit_calls) == 1
    assert depth.starts == 1

    harness._ai_depth_enabled = False
    harness.apply()
    camera.frame_ready.emit(0, "raw-1")
    assert len(infer.submit_calls) == 2
    assert len(depth.submit_calls) == 5
    assert depth.enabled[-1] is False
    assert harness._cells[0].depth_modes[-1] is False

    for index in range(3):
        harness.apply()
        camera.frame_ready.emit(0, f"raw-{index + 2}")
    assert len(infer.submit_calls) == 5
    assert len(depth.submit_calls) == 5


def test_master_ai_off_disables_depth_without_starting_detection(qapp) -> None:
    harness = _WindowHarness()
    harness._is_ai_enabled = False
    harness._ai_depth_enabled = True
    harness.apply()

    harness.camera.frame_ready.emit(0, "ignored")
    assert harness._infer_worker.submit_calls == []
    assert harness._depth_worker.submit_calls == []
    assert harness._depth_worker.enabled == [False]
    assert harness._depth_worker.starts == 0
    assert harness._cells[0].depth_modes == [False]


def test_safe_direct_connect_removes_preexisting_qt_duplicates(qapp) -> None:
    from gui.main_window import MainWindow

    camera = _CameraWorkerStub()
    calls: list[object] = []

    def slot(_slot_id, frame) -> None:
        calls.append(frame)

    for _ in range(3):
        camera.frame_ready.connect(slot, Qt.ConnectionType.DirectConnection)
    MainWindow._safe_direct_connect(camera.frame_ready, slot)
    camera.frame_ready.emit(0, "one")
    assert calls == ["one"]

    MainWindow._disconnect_all(camera.frame_ready, slot)
    camera.frame_ready.emit(0, "ignored")
    assert calls == ["one"]


def test_depth_settings_default_off_and_roundtrip(monkeypatch) -> None:
    from gui import settings_dialog as module

    values: dict[str, object] = {}

    class SettingsStub:
        def __init__(self, *_args) -> None:
            pass

        def value(self, key, default):
            return values.get(key, default)

        def setValue(self, key, value) -> None:
            values[key] = value

    monkeypatch.setattr(module, "QSettings", SettingsStub)
    settings = module.SettingsDialog
    assert settings.load_ai_depth_enabled() is False
    assert settings.load_ai_depth_colormap() == "turbo"
    assert settings.load_ai_depth_range_mode() == "auto"
    settings.save_ai_depth_enabled(True)
    settings.save_ai_depth_colormap("plasma")
    settings.save_ai_depth_range_mode("0-8")
    assert values[settings._KEY_AI_DEPTH_ENABLED] is True
    assert settings.load_ai_depth_enabled() is True
    assert settings.load_ai_depth_colormap() == "plasma"
    assert settings.load_ai_depth_range_mode() == "0-8"


class _ErrorWindowHarness:
    _DEPTH_ERROR_NOTICE_INTERVAL = 1.0

    def __init__(self) -> None:
        self._is_ai_enabled = True
        self._ai_depth_enabled = True
        self.messages: list[str] = []
        self.sync_calls = 0
        self.clear_calls = 0
        self._depth_error_notice_ts: dict[int, float] = {}

    def _refresh_dashboard(self, message) -> None:
        self.messages.append(message)

    def _sync_feature_toggle_states(self) -> None:
        self.sync_calls += 1

    def _clear_all_detections(self) -> None:
        self.clear_calls += 1

    def error(self, slot_id, message) -> None:
        from gui.main_window import MainWindow

        MainWindow._on_depth_error(self, slot_id, message)


def test_transient_depth_error_keeps_feature_running() -> None:
    window = _ErrorWindowHarness()

    window.error(0, "temporary CUDA error")

    assert window._ai_depth_enabled is True
    assert window.sync_calls == 0  # 不得保存关闭状态或影响其他采集路。
    assert window.clear_calls == 0
    assert "跳过" in window.messages[-1]

    window.error(0, "another transient error")
    assert len(window.messages) == 1  # dashboard 采用时间窗节流。


def test_model_depth_error_falls_back_and_ignores_stale_error() -> None:
    window = _ErrorWindowHarness()

    window.error(-1, "model missing")

    assert window._ai_depth_enabled is False
    assert window.sync_calls == 1
    assert window.clear_calls == 1

    window.error(-1, "stale queued error")
    assert window.sync_calls == 1  # 已关闭后重复投递不再保存/回退。
    assert window.clear_calls == 1
