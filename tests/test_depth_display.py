"""深度视频显示与主窗口路由契约（offscreen，不加载真实模型）。"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

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
    cell.clear_stream()
    assert cell._depth_frame is None
    cell.deleteLater()


class _SignalStub:
    def __init__(self) -> None:
        self.slots: list[object] = []

    def connect(self, slot, *_args) -> None:
        if slot not in self.slots:
            self.slots.append(slot)

    def disconnect(self, slot) -> None:
        if slot not in self.slots:
            raise TypeError("not connected")
        self.slots.remove(slot)


class _CameraWorkerStub:
    def __init__(self) -> None:
        self.frame_ready = _SignalStub()


class _InferWorkerStub:
    def __init__(self) -> None:
        self.registered: list[int] = []
        self.unregistered: list[int] = []

    def submit_frame(self, _slot_id, _frame) -> None:
        pass

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

    def submit_frame(self, _slot_id, _frame) -> None:
        pass

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


def test_main_window_switches_inference_route_without_duplicate_connections() -> None:
    harness = _WindowHarness()
    camera = harness.camera
    infer = harness._infer_worker
    depth = harness._depth_worker

    harness.apply()
    assert camera.frame_ready.slots == [infer.submit_frame]

    harness._ai_depth_enabled = True
    harness.apply()
    assert camera.frame_ready.slots == [depth.submit_frame]
    assert infer.unregistered == [0]
    assert depth.enabled[-1] is True and depth.starts == 1
    assert harness._cells[0].depth_modes[-1] is True

    harness.apply()
    assert camera.frame_ready.slots == [depth.submit_frame]
    assert depth.starts == 1

    harness._ai_depth_enabled = False
    harness.apply()
    assert camera.frame_ready.slots == [infer.submit_frame]
    assert depth.enabled[-1] is False
    assert harness._cells[0].depth_modes[-1] is False


def test_master_ai_off_disables_depth_without_starting_detection() -> None:
    harness = _WindowHarness()
    harness._is_ai_enabled = False
    harness._ai_depth_enabled = True
    harness.apply()

    assert harness.camera.frame_ready.slots == []
    assert harness._depth_worker.enabled == [False]
    assert harness._depth_worker.starts == 0
    assert harness._cells[0].depth_modes == [False]


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
    settings.save_ai_depth_enabled(True)
    assert values[settings._KEY_AI_DEPTH_ENABLED] is True
    assert settings.load_ai_depth_enabled() is True


class _ErrorWindowHarness:
    def __init__(self) -> None:
        self._is_ai_enabled = True
        self._ai_depth_enabled = True
        self.messages: list[str] = []
        self.sync_calls = 0
        self.clear_calls = 0

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


def test_model_depth_error_falls_back_and_ignores_stale_error() -> None:
    window = _ErrorWindowHarness()

    window.error(-1, "model missing")

    assert window._ai_depth_enabled is False
    assert window.sync_calls == 1
    assert window.clear_calls == 1

    window.error(-1, "stale queued error")
    assert window.sync_calls == 1  # 已关闭后重复投递不再保存/回退。
    assert window.clear_calls == 1
