"""默认行为变更契约测试（设计见 docs/default-backends.md）：

1. DetectionTracker 缺省 backend="auto"：omintrack 可导入 → Rust；
   ImportError / OSError（.pyd DLL 加载失败）→ 回退 Python。
2. 环境变量 OMINTRACK_BACKEND 优先级（显式参数 > 环境变量 > 缺省 auto）。
3. ReID 模型路径三态默认加载：QSettings 键不存在 → 默认模型发现；
   键存在（含 ""）→ 尊重存储值。
4. GUI offscreen 冒烟：跟踪后端行项集合 / auto 默认值 / Rust 项可用性门控。

QSettings 隔离：本文件 monkeypatch gui.settings_dialog.QSettings 为内存假实现，
严禁写真实用户配置。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

RUST_AVAILABLE = importlib.util.find_spec("omintrack") is not None


@pytest.fixture(autouse=True)
def _clean_backend_env(monkeypatch):
    """隔离外部 OMINTRACK_BACKEND 环境变量（各用例自行 setenv 覆盖）。"""
    monkeypatch.delenv("OMINTRACK_BACKEND", raising=False)


# ======================================================================
# 1. auto 三态
# ======================================================================

def test_auto_default_uses_rust_when_omintrack_available():
    """本机 omintrack 可用：无参构造 → RustDetectionTracker。"""
    pytest.importorskip("omintrack")
    from ai.tracker import DetectionTracker
    from ai.tracker_backend import RustDetectionTracker

    tracker = DetectionTracker()
    assert isinstance(tracker, RustDetectionTracker)


def test_auto_falls_back_to_python_on_import_error(monkeypatch):
    """模拟 omintrack 缺失（sys.modules 置 None → ImportError）→ 回退 Python。"""
    from ai.tracker import DetectionTracker

    monkeypatch.setitem(sys.modules, "omintrack", None)
    tracker = DetectionTracker()
    assert type(tracker) is DetectionTracker


def test_auto_falls_back_to_python_on_oserror(monkeypatch):
    """模拟 .pyd DLL 加载失败（OSError）→ 回退 Python。"""
    import builtins

    from ai.tracker import DetectionTracker

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "omintrack":
            raise OSError("DLL load failed: 找不到指定的模块")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    tracker = DetectionTracker()
    assert type(tracker) is DetectionTracker


def test_explicit_python_always_python():
    """显式 backend="python" 恒为 Python 实现（即使 omintrack 可用）。"""
    from ai.tracker import DetectionTracker

    tracker = DetectionTracker(backend="python")
    assert type(tracker) is DetectionTracker


def test_explicit_rust_behavior_unchanged():
    """显式 backend="rust" 行为不变：可用 → Rust 实例。"""
    pytest.importorskip("omintrack")
    from ai.tracker import DetectionTracker
    from ai.tracker_backend import RustDetectionTracker

    tracker = DetectionTracker(backend="rust")
    assert isinstance(tracker, RustDetectionTracker)


# ======================================================================
# 2. 环境变量优先级
# ======================================================================

def test_env_var_python_beats_auto_default(monkeypatch):
    """OMINTRACK_BACKEND=python 时无参构造为 Python。"""
    from ai.tracker import DetectionTracker

    monkeypatch.setenv("OMINTRACK_BACKEND", "python")
    assert type(DetectionTracker()) is DetectionTracker


def test_env_var_auto_matches_default(monkeypatch):
    """OMINTRACK_BACKEND=auto 时与缺省一致。"""
    from ai.tracker import DetectionTracker

    monkeypatch.setenv("OMINTRACK_BACKEND", "auto")
    if RUST_AVAILABLE:
        from ai.tracker_backend import RustDetectionTracker

        assert isinstance(DetectionTracker(), RustDetectionTracker)
    else:
        assert type(DetectionTracker()) is DetectionTracker


def test_explicit_param_beats_env_var(monkeypatch):
    """显式参数 > 环境变量：env=python 但显式 rust 仍走 Rust。"""
    pytest.importorskip("omintrack")
    from ai.tracker import DetectionTracker
    from ai.tracker_backend import RustDetectionTracker

    monkeypatch.setenv("OMINTRACK_BACKEND", "python")
    assert isinstance(DetectionTracker(backend="rust"), RustDetectionTracker)


# ======================================================================
# 3. ReID 三态默认加载（QSettings 内存隔离）
# ======================================================================

@pytest.fixture
def fake_settings(monkeypatch):
    """以内存假 QSettings 替换 gui.settings_dialog.QSettings，严禁写真实配置。"""
    from gui import settings_dialog as sd

    store: dict = {}

    class _FakeQSettings:
        def __init__(self, *args, **kwargs):
            pass

        def value(self, key, default=None):
            return store.get(key, default)

        def setValue(self, key, val):
            store[key] = val

        def contains(self, key):
            return key in store

    monkeypatch.setattr(sd, "QSettings", _FakeQSettings)
    return store


DEFAULT_MODEL = "models/osnet_x0_25_256x128_msmt17.onnx"


def test_reid_default_enabled_when_key_absent_and_model_exists(fake_settings, monkeypatch):
    """无键 + 模型存在 → 默认模型路径（默认启用）。"""
    from gui.settings_dialog import SettingsDialog

    monkeypatch.setattr(
        SettingsDialog, "_default_reid_model_path", staticmethod(lambda: DEFAULT_MODEL)
    )
    assert SettingsDialog.load_ai_reid_model_path() == DEFAULT_MODEL


def test_reid_respects_user_disabled_empty_string(fake_settings, monkeypatch):
    """键存在且为 ""（用户主动关闭）→ 尊重关闭，不做默认发现。"""
    from gui.settings_dialog import SettingsDialog

    def _boom():
        raise AssertionError("键存在时不应触发默认模型发现")

    monkeypatch.setattr(SettingsDialog, "_default_reid_model_path", staticmethod(_boom))
    fake_settings[SettingsDialog._KEY_AI_REID_MODEL_PATH] = ""
    assert SettingsDialog.load_ai_reid_model_path() == ""


def test_reid_respects_stored_custom_path(fake_settings):
    """键存在为非空自定义路径 → 原样返回。"""
    from gui.settings_dialog import SettingsDialog

    fake_settings[SettingsDialog._KEY_AI_REID_MODEL_PATH] = "models/custom_osnet.onnx"
    assert SettingsDialog.load_ai_reid_model_path() == "models/custom_osnet.onnx"


def test_reid_default_empty_when_no_model(fake_settings, monkeypatch):
    """无键 + 无模型 → ""。（默认关闭，不炸应用）"""
    from gui.settings_dialog import SettingsDialog

    monkeypatch.setattr(SettingsDialog, "_default_reid_model_path", staticmethod(lambda: ""))
    assert SettingsDialog.load_ai_reid_model_path() == ""


def test_reid_real_repo_default_discovery():
    """真实仓库默认发现：tests 从仓库根运行，模型随仓库分发。"""
    from gui.settings_dialog import SettingsDialog

    path = SettingsDialog._default_reid_model_path()
    assert path == str(Path.cwd() / DEFAULT_MODEL)
    assert Path(path).exists()


# ======================================================================
# 4. GUI offscreen 冒烟
# ======================================================================

@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_settings_dialog_tracker_backend_row(qapp, fake_settings):
    """跟踪后端行：auto 默认选中；Rust 项仅在 omintrack 可导入时出现。"""
    from gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    try:
        combo = dlg._tracker_backend_combo
        modes = [combo.itemData(i) for i in range(combo.count())]
        assert modes[0] == "auto"
        assert "python" in modes
        if RUST_AVAILABLE:
            assert "rust" in modes
        else:
            assert "rust" not in modes
        assert combo.currentData() == "auto"
        # _collect（get_ai_settings）补全 tracker_backend
        assert dlg.get_ai_settings()["tracker_backend"] == "auto"
    finally:
        dlg.deleteLater()


def test_settings_dialog_reid_default_selected(qapp, fake_settings):
    """无键首次启动：ReID 模型行选中默认模型（默认启用）。"""
    from gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    try:
        expected = SettingsDialog._default_reid_model_path()
        assert expected != ""
        assert dlg._reid_model_combo.currentData() == expected
    finally:
        dlg.deleteLater()


def test_restore_recommended_reselects_repo_default_model(qapp, fake_settings):
    """推荐恢复：ReID 走默认模型发现（推荐 = 启用随仓库模型，不退化为关闭）。"""
    from gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    try:
        dlg._reid_model_combo.setCurrentIndex(0)  # "关闭"
        dlg._restore_recommended_ai_params()
        assert dlg._reid_model_combo.currentData() == SettingsDialog._default_reid_model_path()
        # 跟踪后端推荐恢复为 auto
        assert dlg._tracker_backend_combo.currentData() == "auto"
    finally:
        dlg.deleteLater()
