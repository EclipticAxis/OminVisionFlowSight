"""squircle 基础设施测试：纯数学路径测试（无需 QApplication）+ offscreen 冒烟。"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_squircle_path_closed_structure():
    """路径结构：1 moveTo + 4 段角弧折线（每段 segments 个 lineTo）+ 3 条边 lineTo。"""
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QPainterPath
    from gui.squircle import DEFAULT_SEGMENTS, squircle_path

    path = squircle_path(QRectF(0, 0, 100, 40), 10.0)
    # 1 move + 4*segs 角弧点 + 3 条直边 = 4*segs + 4
    assert path.elementCount() == 4 * DEFAULT_SEGMENTS + 4
    assert path.elementAt(0).type == QPainterPath.ElementType.MoveToElement
    for i in range(1, path.elementCount()):
        assert path.elementAt(i).type == QPainterPath.ElementType.LineToElement
    # 闭合：末元素回到起点（左上角弧终点 = moveTo 点）
    last = path.elementAt(path.elementCount() - 1)
    first = path.elementAt(0)
    assert last.x == pytest.approx(first.x)
    assert last.y == pytest.approx(first.y)


def test_squircle_path_corner_geometry():
    """角几何：45° 采样点应落在真实 n=4 超椭圆弧上（而非圆弧）。"""
    import math

    from PyQt6.QtCore import QRectF
    from gui.squircle import squircle_path

    path = squircle_path(QRectF(0, 0, 100, 100), 10.0)
    # 右上角弧第 5 个采样点（segments=10 -> s=0.5, θ=π/4）
    # TR 角点正方形中心 (90, 10)，45° 点 = (90 + r*cos(π/4)^0.5, 10 - r*sin(π/4)^0.5)
    expect = 10.0 * math.cos(math.pi / 4) ** 0.5
    el = path.elementAt(5)  # moveTo=0, TR 弧 j=1..10 -> 元素 1..10
    assert el.x == pytest.approx(90.0 + expect, abs=1e-6)
    assert el.y == pytest.approx(10.0 - expect, abs=1e-6)
    # 关键：squircle 的 45° 点在 0.8409*r 处，圆弧只在 0.7071*r —— 必须更"鼓"
    assert expect > 10.0 * 0.8


def test_squircle_path_bounding_within_rect():
    from PyQt6.QtCore import QRectF
    from gui.squircle import squircle_path

    for rect in (QRectF(0, 0, 100, 40), QRectF(-5.5, 7.25, 33.3, 12.1)):
        path = squircle_path(rect, 8.0)
        br = path.boundingRect()
        assert br.left() >= rect.left() - 1e-6
        assert br.top() >= rect.top() - 1e-6
        assert br.right() <= rect.right() + 1e-6
        assert br.bottom() <= rect.bottom() + 1e-6


def test_squircle_path_radius_zero_degenerates_to_rect():
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QPainterPath
    from gui.squircle import squircle_path

    rect = QRectF(0, 0, 50, 30)
    path = squircle_path(rect, 0.0)
    # addRect 产生 1 move + 4 line（末点回到起点闭合）= 5 个元素，且无曲线
    assert path.elementCount() == 5
    assert path.elementAt(0).type == QPainterPath.ElementType.MoveToElement
    for i in range(1, 5):
        assert path.elementAt(i).type == QPainterPath.ElementType.LineToElement
    assert path.boundingRect() == rect


def test_squircle_path_clamps_radius_and_tiny_rect():
    from PyQt6.QtCore import QRectF
    from gui.squircle import squircle_path

    # radius 超过 min(w,h)/2 时应被 clamp，不崩溃
    path = squircle_path(QRectF(0, 0, 20, 10), 999.0)
    br = path.boundingRect()
    assert br.width() <= 20.0 + 1e-6 and br.height() <= 10.0 + 1e-6
    # 极小 rect 也不崩溃（零长线段会被 QPainterPath 去重，元素数只增不减地检查）
    path = squircle_path(QRectF(0, 0, 0.5, 0.25), 1.0)
    assert path.elementCount() > 0
    br = path.boundingRect()
    assert br.width() <= 0.5 + 1e-6 and br.height() <= 0.25 + 1e-6


def test_squircle_asset_url():
    from gui.squircle import asset_file, squircle_asset_url

    import urllib.parse

    url = squircle_asset_url("PILL", 5)
    assert url.startswith("file:///")
    # as_uri() 会对 '@' 等字符做百分号编码，解码后比较文件名
    assert urllib.parse.unquote(url).endswith("pill_5px@4x.png")
    assert url == asset_file("PILL", 5).resolve().as_uri()


@pytest.fixture(scope="module")
def qapp():
    try:
        from PyQt6.QtGui import QGuiApplication
        app = QGuiApplication([])
    except Exception as exc:  # 无显示环境等
        pytest.skip(f"无法创建 QGuiApplication: {exc}")
    yield app


def test_squircle_tile_smoke(qapp, tmp_path):
    from PyQt6.QtGui import QPixmap
    from gui.squircle import render_tile_to, squircle_tile

    pm = squircle_tile("#1A2238", "#32405F", radius=5)
    assert isinstance(pm, QPixmap)
    assert not pm.isNull()
    assert pm.width() > 0 and pm.height() > 0

    out = tmp_path / "tile.png"
    render_tile_to(out, "#1A2238", "#32405F", 5)
    assert out.exists()
    assert out.stat().st_size > 100


# === 阶段 2：胶囊控件全面 squircle 化后的 STYLESHEET 契约测试（无需 QApplication） ===

def test_stylesheet_has_no_border_radius_except_scrollbar():
    """胶囊控件均为 squircle border-image；border-radius 只允许出现在 QScrollBar 规则
    （5px 细滚动条用 squircle 条带会把滑块挤成 1px 不可见细线，故保留纯色+小圆角）。"""
    from gui.theme import get_stylesheet

    ss = get_stylesheet()
    selector = ""
    for line in ss.splitlines():
        stripped = line.strip()
        if "{" in stripped:
            selector = stripped.split("{")[0].strip()
        if "border-radius" in stripped:
            assert selector.startswith("QScrollBar"), (
                f"非滚动条规则残留 border-radius: [{selector}] {stripped}"
            )


def test_squircle_asset_pngs_exist():
    """SQUIRCLE_ASSETS 中每个 token 的 PNG 资产都已生成在磁盘上。"""
    from gui.squircle import asset_file
    from gui.theme import RADIUS, SQUIRCLE_ASSETS

    assert len(SQUIRCLE_ASSETS) >= 50
    for token, spec in SQUIRCLE_ASSETS.items():
        f = asset_file(token, RADIUS[spec["radius"]])
        assert f.exists(), f"缺少 squircle 资产: {f}"
        assert f.stat().st_size > 100, f"资产文件异常: {f}"


def test_stylesheet_border_image_urls_exist():
    """STYLESHEET 中每个 border-image 的 url() 指向的 PNG 文件都存在。"""
    import re
    from gui.theme import get_stylesheet

    ss = get_stylesheet()
    urls = re.findall(r'border-image: url\("([^"]+)"\)', ss)
    assert len(urls) > 0
    for u in urls:
        assert Path(u).exists(), f"border-image 资产不存在: {u}"
