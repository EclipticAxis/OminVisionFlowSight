"""squircle（连续曲率圆角）基础设施。

QSS 的 border-radius 只能画普通圆弧（曲率在直边连接点从 0 突跳到 1/r，
非 G2 连续）。本模块用 n=4 超椭圆（|x/a|^4+|y/b|^4=1）构造真正的连续曲率
圆角：每个角是角点局部 2r×2r 正方形内的超椭圆弧，弧上采样为折线
（QPainterPath lineTo）。超椭圆在直边连接点处曲率自然趋于 0，随后平滑
升至 45° 处最大再回落，整条路径 G2 连续——即 iOS/macOS 的 continuous
corner（squircle）造型。

本模块为纯模块：import 时不依赖 QApplication（QPixmap 等仅在函数体内使用）。"""
from __future__ import annotations

from pathlib import Path

# 超椭圆指数 n（n=4 即经典 squircle；x=sgn(cos)|cos|^(2/n) 参数式）。
SUPERELLIPSE_N = 4
# 每角采样段数：段数越多折线越贴合真实超椭圆。10 段时任意采样点与真实曲线
# 的偏差 < 0.1% * r（r=5px 时 < 0.005px，远低于像素精度）。
DEFAULT_SEGMENTS = 10

# 资产生成契约（阶段契约，单一定义源）：由 gui/theme.py 的 SQUIRCLE_ASSETS 提供。
ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "squircle"


def squircle_path(rect, radius: float, segments: int = DEFAULT_SEGMENTS):
    """构造覆盖 rect 的 squircle 闭合路径（QPainterPath）。

    每个角是角点局部 2r×2r 正方形内 n=4 超椭圆的 1/4 弧，弧按 segments
    段采样为折线；角与角之间以直线段相连。超椭圆弧在直边连接点曲率趋于
    0，整条路径曲率连续（G2），即连续曲率圆角。
    radius 先 clamp 到 min(w, h) / 2（达到后即"跑道形"端点）；r <= 0 或
    矩形退化时退化为矩形。width/height 极小时同样安全。
    """
    import math

    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QPainterPath

    if not isinstance(rect, QRectF):
        rect = QRectF(rect)
    w = rect.width()
    h = rect.height()
    if w <= 0.0 or h <= 0.0:
        path = QPainterPath()
        path.addRect(rect)
        return path

    r = float(radius)
    half = min(w, h) / 2.0
    if r > half:
        r = half
    if r <= 0.0:
        path = QPainterPath()
        path.addRect(rect)
        return path

    segs = max(2, int(segments))
    exp = 2.0 / SUPERELLIPSE_N  # n=4 -> 0.5
    x0, y0 = rect.x(), rect.y()
    x1, y1 = rect.right(), rect.bottom()

    def arc_pts(cx, cy, sx, sy, t0, t1, end):
        """角点局部正方形中心 (cx,cy)，符号 (sx,sy) 指向矩形内部，θ 从 t0 扫到 t1。

        末点用解析锚点 end（精确坐标）而不是 θ=t1 的浮点计算值——
        cos(pi/2) 并非精确 0，会引入 ~1e-8 误差导致路径无法精确闭合。
        """
        pts = []
        for j in range(1, segs + 1):
            f = j / segs
            th = t0 + (t1 - t0) * f
            u = r * (abs(math.cos(th)) ** exp)
            v = r * (abs(math.sin(th)) ** exp)
            pts.append((cx + sx * u, cy + sy * v))
        pts[-1] = end
        return pts

    path = QPainterPath()
    path.moveTo(x0 + r, y0)
    # 右上：(x1-r, y0) -> (x1, y0+r)
    for px, py in arc_pts(x1 - r, y0 + r, +1, -1, math.pi / 2, 0.0, (x1, y0 + r)):
        path.lineTo(px, py)
    path.lineTo(x1, y1 - r)
    # 右下：(x1, y1-r) -> (x1-r, y1)
    for px, py in arc_pts(x1 - r, y1 - r, +1, +1, 0.0, math.pi / 2, (x1 - r, y1)):
        path.lineTo(px, py)
    path.lineTo(x0 + r, y1)
    # 左下：(x0+r, y1) -> (x0, y1-r)
    for px, py in arc_pts(x0 + r, y1 - r, -1, +1, math.pi / 2, 0.0, (x0, y1 - r)):
        path.lineTo(px, py)
    path.lineTo(x0, y0 + r)
    # 左上：(x0, y0+r) -> (x0+r, y0)
    for px, py in arc_pts(x0 + r, y0 + r, -1, -1, 0.0, math.pi / 2, (x0 + r, y0)):
        path.lineTo(px, py)
    path.closeSubpath()
    return path


def _coerce_color(value) -> "QColor":
    """接受 QColor / '#RRGGBB' / '#AARRGGBB' / 'rgba(r,g,b,a)'，返回 QColor。"""
    from PyQt6.QtGui import QColor

    if isinstance(value, QColor):
        return QColor(value)
    s = str(value).strip()
    if s.lower().startswith("rgba("):
        # Qt 的 setNamedColor 不支持 CSS rgba() 语法，手工解析。
        inner = s[s.index("(") + 1 : s.rindex(")")]
        parts = [float(p.strip()) for p in inner.split(",")]
        if len(parts) == 4:
            a = parts[3]
            if a <= 1.0:  # 归一化 alpha -> 0-255
                a = a * 255.0
            return QColor(int(parts[0]), int(parts[1]), int(parts[2]), int(round(a)))
        return QColor(int(parts[0]), int(parts[1]), int(parts[2]))
    c = QColor()
    c.setNamedColor(s)
    return c


def squircle_tile(
    bg,
    border,
    radius: int,
    border_width: int = 1,
    pad: int = 8,
    scale: int = 4,
    gradient: tuple | None = None,
):
    """绘制可九宫格切片（border-image）的 squircle 瓦片，返回 QPixmap。

    画布为 (2 * (radius + pad) * scale) 的方形；squircle 主体居中，
    四角外区域透明，可被阶段 2 的 QSS border-image 方案切片使用。
    bg 默认实底填充；gradient=(c1, c2) 时改为对角 QLinearGradient；
    border 非 None 时沿路径描边 border_width px（已乘 scale）。
    """
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtGui import QLinearGradient, QPainter, QPen, QPixmap

    size = 2 * (radius + pad) * scale
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)  # 全透明底

    rect = QRectF(
        float(pad * scale), float(pad * scale),
        float(size - 2 * pad * scale), float(size - 2 * pad * scale),
    )
    path = squircle_path(rect, float(radius * scale))

    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    if gradient is not None:
        grad = QLinearGradient(QPointF(rect.topLeft()), QPointF(rect.bottomRight()))
        grad.setColorAt(0.0, _coerce_color(gradient[0]))
        grad.setColorAt(1.0, _coerce_color(gradient[1]))
        painter.setBrush(grad)
    else:
        painter.setBrush(_coerce_color(bg))
    painter.drawPath(path)

    if border is not None:
        pen = QPen(_coerce_color(border))
        pen.setWidthF(float(border_width * scale))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)
    painter.end()
    return pm


def asset_file(key: str, radius: int, scale: int = 4) -> Path:
    """按契约生成资产文件名：<key>_<radius>px@<scale>x.png。"""
    return ASSETS_DIR / f"{key.lower()}_{radius}px@{scale}x.png"


def squircle_asset_url(key: str, radius: int) -> str:
    """返回资产的 file:/// 绝对 URL（纯字符串操作，不读文件）。"""
    return asset_file(key, radius).resolve().as_uri()


def render_tile_to(
    path: Path,
    bg,
    border,
    radius,
    border_width: int = 1,
    pad: int = 8,
    scale: int = 4,
    gradient: tuple | None = None,
) -> None:
    """画一张瓦片并存盘（供阶段 2 运行时动态生成，如颜色样本按钮）。

    父目录自动创建；已存在则覆盖。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pm = squircle_tile(
        bg, border, int(radius),
        border_width=border_width, pad=pad, scale=scale, gradient=gradient,
    )
    pm.save(str(path), "PNG")
