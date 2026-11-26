"""SettingsDialog ``max_output_misses``（预测输出保留帧数）GUI 链路契约测试（offscreen）。

契约（QSettings 键 ``ai/max_output_misses``）：
- 旧配置缺键时 loader 默认 0（向后兼容），读取范围钳制到 0..10；
- ``get_ai_settings`` 透出新键，``_save_ai_settings`` 写键后可回读；
- ``MainWindow._apply_ai_settings`` 将其透传到
  ``InferWorker.set_tracker_params(max_output_misses=...)``，缺键时默认 0。
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def isolated_settings(qapp, tmp_path_factory):
    """把 QSettings 重定向到临时 ini，避免污染真实用户配置。"""
    from PyQt6.QtCore import QSettings

    tmp = tmp_path_factory.mktemp("qsettings")
    old_format = QSettings.defaultFormat()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    from gui.settings_dialog import SettingsDialog

    old_path = str(Path(QSettings(
        SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP,
    ).fileName()).parent)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp))
    yield tmp
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, old_path)
    QSettings.setDefaultFormat(old_format)


@pytest.fixture()
def dialog(qapp, isolated_settings):
    from gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    yield dlg
    dlg.close()
    dlg.deleteLater()


def test_loader_defaults_to_zero_when_key_missing(isolated_settings) -> None:
    from gui.settings_dialog import SettingsDialog

    # 旧配置无 ai/max_output_misses 键 → 默认 0（向后兼容）
    assert SettingsDialog.load_ai_max_output_misses() == 0


def test_save_then_load_roundtrip(dialog) -> None:
    from gui.settings_dialog import SettingsDialog

    dialog._save_ai_settings(0.25, 2, 0.85, 1500, max_output_misses=7)
    assert SettingsDialog.load_ai_max_output_misses() == 7


def test_confirm_persists_current_spin_values(dialog, monkeypatch) -> None:
    from gui.settings_dialog import SettingsDialog

    dialog._max_misses_spin.setValue(5)
    dialog._max_output_misses_spin.setValue(6)
    monkeypatch.setattr(dialog, "_animate_close", lambda _decision: None)

    dialog._on_confirm()

    assert SettingsDialog.load_ai_max_misses() == 5
    assert SettingsDialog.load_ai_max_output_misses() == 6


def test_restore_recommended_resets_both_limits(dialog) -> None:
    dialog._max_misses_spin.setValue(7)
    dialog._max_output_misses_spin.setValue(6)

    dialog._restore_recommended_ai_params()

    assert dialog._max_misses_spin.value() == 4
    assert dialog._max_output_misses_spin.value() == 0


def test_loader_clamps_out_of_range(dialog, isolated_settings) -> None:
    from PyQt6.QtCore import QSettings

    from gui.settings_dialog import SettingsDialog

    dialog._save_ai_settings(0.25, 2, 0.85, 1500, max_output_misses=99)
    assert SettingsDialog.load_ai_max_output_misses() == 10

    settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
    settings.setValue(SettingsDialog._KEY_AI_MAX_OUTPUT_MISSES, -3)
    settings.sync()
    assert SettingsDialog.load_ai_max_output_misses() == 0


def test_get_ai_settings_exposes_new_key(dialog) -> None:
    dialog._max_output_misses_spin.setValue(6)
    cfg = dialog.get_ai_settings()
    assert cfg["max_output_misses"] == 6
    # 既有键不受影响
    assert "max_misses" in cfg


def test_spins_have_tooltips(dialog) -> None:
    assert "内部轨迹" in dialog._max_misses_spin.toolTip()
    assert "预测" in dialog._max_output_misses_spin.toolTip()
    assert "0" in dialog._max_output_misses_spin.toolTip()


class _WorkerStub:
    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _WindowHarness:
    """只承载 _apply_ai_settings 所需属性，避免创建额外真实 MainWindow。"""

    def __init__(self) -> None:
        self._infer_worker = _WorkerStub()
        self._ai_model_path = ""
        self._cells = {}

    def _apply_ai_settings(self, cfg: dict) -> None:
        from gui.main_window import MainWindow

        MainWindow._apply_ai_settings(self, cfg)

    def _apply_feature_flags_to_worker(self) -> None:
        pass

    def _resolve_special_model_path(self, path: str) -> str:
        return path

    def _refresh_dashboard(self, _message: str) -> None:
        pass


@pytest.fixture()
def window():
    return _WindowHarness()


def _base_cfg(window) -> dict:
    return {
        "conf": 0.25,
        "stride": 2,
        "smooth": 0.85,
        "hide_ms": 1500,
        # 与当前模型一致，避免触发模型切换分支
        "model_path": window._ai_model_path,
    }


def test_apply_ai_settings_passes_max_output_misses(window, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        window._infer_worker, "set_tracker_params", lambda **kw: captured.update(kw)
    )

    window._apply_ai_settings({**_base_cfg(window), "max_output_misses": 5})
    assert captured["max_output_misses"] == 5


def test_apply_ai_settings_defaults_to_zero_when_key_absent(window, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr(
        window._infer_worker, "set_tracker_params", lambda **kw: captured.update(kw)
    )

    window._apply_ai_settings(_base_cfg(window))
    assert captured["max_output_misses"] == 0
