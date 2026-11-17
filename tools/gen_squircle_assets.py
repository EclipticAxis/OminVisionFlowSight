"""squircle 资产生成器（阶段 1）。

遍历 gui.theme.SQUIRCLE_ASSETS，按 bg/border/radius 解析颜色并调用
gui.squircle.render_tile_to 生成 PNG 到 gui/assets/squircle/。
可重复运行（覆盖写）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 必须在使用 Qt 之前设置 offscreen 平台。
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtGui import QGuiApplication  # noqa: E402

from gui.squircle import asset_file, render_tile_to  # noqa: E402
from gui.theme import (  # noqa: E402
    COLORS,
    RADIUS,
    SQ_TILE_PAD,
    SQ_TILE_SCALE,
    SQUIRCLE_ASSETS,
)


def resolve_color(name_or_value: str):
    """把 COLORS 键名字符串解析为 QColor；支持 '#RRGGBB' 与 rgba() 字符串。"""
    from PyQt6.QtGui import QColor

    raw = COLORS.get(name_or_value, name_or_value)
    s = str(raw).strip()
    if s.lower().startswith("rgba("):
        # QColor.setNamedColor 不支持 CSS rgba()，手工解析。
        inner = s[s.index("(") + 1 : s.rindex(")")]
        parts = [float(p.strip()) for p in inner.split(",")]
        r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
        if len(parts) == 4:
            a = parts[3]
            if a <= 1.0:  # 归一化 alpha -> 0-255
                a = a * 255.0
            return QColor(r, g, b, int(round(a)))
        return QColor(r, g, b)
    c = QColor()
    c.setNamedColor(s)
    return c


def main() -> int:
    app = QGuiApplication(sys.argv)  # noqa: F841  QPixmap 需要 GUI 应用实例
    print(f"生成目录: {asset_file('x', 1).parent}")
    print(f"瓦片参数: pad={SQ_TILE_PAD} scale={SQ_TILE_SCALE}（与 gui.theme 一致）")
    for key, spec in SQUIRCLE_ASSETS.items():
        bg = resolve_color(spec["bg"])
        border = resolve_color(spec["border"]) if spec["border"] is not None else None
        radius = RADIUS[spec["radius"]]
        border_width = spec.get("border_width", 1)
        gradient = None
        if spec.get("gradient") is not None:
            gradient = tuple(resolve_color(g) for g in spec["gradient"])
        out = asset_file(key, radius, scale=SQ_TILE_SCALE)
        render_tile_to(
            out, bg, border, radius,
            border_width=border_width, pad=SQ_TILE_PAD,
            scale=SQ_TILE_SCALE, gradient=gradient,
        )
        size = out.stat().st_size
        extra = f" gradient={spec['gradient']}" if gradient else ""
        print(f"  {out.name}  ({size} bytes)  bg={spec['bg']} border={spec['border']} "
              f"radius={radius}px bw={border_width}{extra}")
    print(f"共生成 {len(SQUIRCLE_ASSETS)} 个资产。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
