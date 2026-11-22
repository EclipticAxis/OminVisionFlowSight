"""侧栏 AI 子选项面板折叠/展开动画契约测试（offscreen）。

契约见 docs/sidebar-options-animation.md：
- 展开/收起为"瞬时几何 + 透明度渐变"（禁止逐帧 maximumHeight 动画）；
- 动画结束后落在稳定终态（可见性、maximumHeight、无残留 graphicsEffect、句柄清空）；
- 已处目标状态时重复调用为 no-op（不重启动画）。
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qapp):
    """共享 MainWindow；不启动 InferWorker 线程（动画契约与推理无关，避免模型加载拖慢套件）。"""
    from gui import main_window as mw

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mw.InferWorker, "start", lambda self: None)
        win = mw.MainWindow()
    yield win
    win.close()
    win.deleteLater()


def _settle(ms: int = 400) -> None:
    from PyQt6.QtTest import QTest

    QTest.qWait(ms)


def test_expand_reaches_stable_state(window) -> None:
    panel = window._rectangle_options_panel
    window._set_rectangle_options_visible(False, animated=False)
    assert panel.isHidden() and panel.maximumHeight() == 0

    window._set_rectangle_options_visible(True, animated=True)
    # 瞬时几何：调用后立即可见且高度一次到位（不再逐帧生长）
    assert not panel.isHidden()
    assert panel.maximumHeight() > 0

    _settle()
    assert not panel.isHidden() and panel.maximumHeight() > 0
    assert panel.graphicsEffect() is None
    assert window._rectangle_options_animation is None


def test_collapse_reaches_stable_state(window) -> None:
    panel = window._rectangle_options_panel
    window._set_rectangle_options_visible(True, animated=False)

    window._set_rectangle_options_visible(False, animated=True)
    _settle()

    assert panel.isHidden()
    assert panel.maximumHeight() == 0
    assert panel.graphicsEffect() is None
    assert window._rectangle_options_animation is None


def test_repeated_same_state_call_is_noop(window) -> None:
    # _sync_feature_toggle_states 每次开关变化都会调用两个面板的可见性方法，
    # 已处目标状态时必须 no-op，否则会反复重启动画（卡顿来源之一）。
    window._set_rectangle_options_visible(True, animated=False)
    window._set_rectangle_options_visible(True, animated=True)
    assert window._rectangle_options_animation is None

    window._set_rectangle_options_visible(False, animated=False)
    window._set_rectangle_options_visible(False, animated=True)
    assert window._rectangle_options_animation is None


def test_gesture_panel_parity(window) -> None:
    panel = window._gesture_options_panel
    window._set_gesture_options_visible(True, animated=True)
    _settle()
    assert not panel.isHidden() and panel.maximumHeight() > 0
    assert panel.graphicsEffect() is None
    assert window._gesture_options_animation is None

    window._set_gesture_options_visible(False, animated=True)
    _settle()
    assert panel.isHidden() and panel.maximumHeight() == 0
    assert window._gesture_options_animation is None


def test_expand_not_squeezed_by_stale_parent_clamp(window) -> None:
    """回归：AI 面板在选项收起状态下展开后，再展开选项不得被旧 maxHeight 钳制挤压。

    复现路径：_animate_ai_feature_panel 曾把 AI 面板钳制在展开瞬间的 sizeHint，
    之后选项面板展开时无处生长，各行被压成几像素高（视觉上文字重叠）。
    修复后两级面板展开均为 QWIDGETSIZE_MAX（取消钳制，布局自适应）。
    """
    from PyQt6.QtWidgets import QWIDGETSIZE_MAX

    aip = window._ai_feature_panel
    rect = window._rectangle_options_panel

    window._set_rectangle_options_visible(False, animated=False)
    window._animate_ai_feature_panel(True)
    _settle()

    window._set_rectangle_options_visible(True, animated=True)
    _settle()

    assert aip.maximumHeight() == QWIDGETSIZE_MAX
    assert rect.maximumHeight() == QWIDGETSIZE_MAX

    window.show()
    try:
        _settle(300)
        # 真实布局后每行都分得完整 sizeHint 高度（允许少量取整误差）
        assert rect.height() >= rect.sizeHint().height() - 12
    finally:
        window.hide()

    # 收尾复位，避免影响同模块其他用例
    window._set_rectangle_options_visible(False, animated=False)
    window._animate_ai_feature_panel(False)
    _settle()
