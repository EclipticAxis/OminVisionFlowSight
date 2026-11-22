r"""精确凸多边形 IoU 双实现测试（Sutherland-Hodgman + shoelace）。

- Python 单测：轴对齐矩形 / 旋转 45° 正方形 / 无交集 / 包含 / 退化零面积 /
  顺逆时针输入等价 / 入口校验。
- shapely 交叉对拍（skip-if-missing）：随机凸四边形/旋转矩形数百组，
  自实现 vs shapely Polygon.intersection 参考值，误差 < 1e-9。
- 双側对拍：omintrack._polygon_iou（Rust）vs ai.detection._precise_polygon_iou
  （Python）同输入逐点比对，误差 < 1e-9。

shapely 2.2.0 仅作测试参考值，生产路径（ai/detection.py）不依赖它。
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.detection import polygon_iou, _precise_polygon_iou  # noqa: E402

omintrack = pytest.importorskip(
    "omintrack", reason="Rust 扩展未构建：先执行 maturin develop"
)

try:
    from shapely.geometry import Polygon as _ShapelyPolygon
except ImportError:  # pragma: no cover - 本机 venv 已装，缺失时跳过对拍
    _ShapelyPolygon = None

TOL_CROSS = 1e-9
N_RANDOM = 400


def _rot_rect(
    cx: float, cy: float, w: float, h: float, ang: float
) -> list[tuple[float, float]]:
    """旋转矩形四顶点（凸，逆时针起点无关，环绕方向由 ang 符号决定）。"""
    ca, sa = math.cos(ang), math.sin(ang)
    pts = []
    for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)):
        pts.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return pts


def _convex_hull(pts: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """单调链凸包（返回 CCW 顶点，不含重复端点）。"""
    ps = sorted(set(pts))
    if len(ps) <= 2:
        return ps
    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    lower = []
    for p in ps:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(ps):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _random_convex_quad(rng: random.Random) -> list[tuple[float, float]]:
    """随机严格凸四边形：随机点取凸包，重采样直到恰好 4 个包顶点。"""
    while True:
        pts = [(rng.uniform(0, 10), rng.uniform(0, 10)) for _ in range(6)]
        hull = _convex_hull(pts)
        if len(hull) == 4:
            return hull


def _shapely_iou_ref(a, b) -> float:
    pa, pb = _ShapelyPolygon(a), _ShapelyPolygon(b)
    inter = pa.intersection(pb).area
    union = pa.area + pb.area - inter
    return inter / union if union > 1e-8 else 0.0


# ======================================================================
# Python 单测
# ======================================================================

def test_axis_aligned_rect_matches_bbox_iou():
    a = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    b = [(1.0, 0.5), (3.0, 0.5), (3.0, 2.0), (1.0, 2.0)]
    # 交 1x0.5=0.5，a 面积 2，b 面积 3
    assert polygon_iou(a, b) == pytest.approx(0.5 / 4.5, abs=1e-12)


def test_rotated_45_diamond_self_overlap():
    # 菱形 |x|+|y|<=1 面积 2；x 方向平移 0.5 后交集 1.125（几何性质：可手算）
    d1 = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    d2 = [(0.5, 1.0), (1.5, 0.0), (0.5, -1.0), (-0.5, 0.0)]
    assert polygon_iou(d1, d2) == pytest.approx(1.125 / (4.0 - 1.125), abs=1e-12)


def test_disjoint_polygons_zero_iou():
    a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    b = [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]
    assert polygon_iou(a, b) == 0.0


def test_containment_iou_is_area_ratio():
    a = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    c = [(-1.0, -1.0), (5.0, -1.0), (5.0, 3.0), (-1.0, 3.0)]
    assert polygon_iou(a, c) == pytest.approx(2.0 / 24.0, abs=1e-12)
    assert polygon_iou(c, a) == pytest.approx(2.0 / 24.0, abs=1e-12)


def test_degenerate_zero_area_polygon():
    a = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    line = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]  # 共线，零面积
    assert polygon_iou(a, line) == 0.0
    assert polygon_iou(line, line) == 0.0


def test_winding_order_equivalence():
    a = _rot_rect(1.0, 1.0, 2.0, 1.0, 0.4)
    b = _rot_rect(1.5, 1.2, 1.5, 2.0, 1.1)
    ccw = polygon_iou(a, b)
    cw = polygon_iou(list(reversed(a)), list(reversed(b)))
    assert cw == pytest.approx(ccw, abs=1e-15)


def test_entry_validation_returns_zero():
    assert polygon_iou([], [(0, 0), (1, 0), (1, 1)]) == 0.0
    assert polygon_iou([(0, 0), (1, 0)], [(0, 0), (1, 0), (1, 1)]) == 0.0


def test_identical_polygon_iou_one():
    a = _rot_rect(3.0, 2.0, 2.0, 1.0, 0.7)
    assert polygon_iou(a, a) == pytest.approx(1.0, abs=1e-12)


# ======================================================================
# shapely 交叉对拍（skip-if-missing）
# ======================================================================

@pytest.mark.skipif(_ShapelyPolygon is None, reason="shapely 未安装（仅测试参考）")
def test_cross_check_shapely_random_rotated_rects():
    rng = random.Random(20261122)
    max_diff = 0.0
    for _ in range(N_RANDOM):
        a = _rot_rect(
            rng.uniform(0, 10), rng.uniform(0, 10),
            rng.uniform(0.5, 4), rng.uniform(0.5, 4), rng.uniform(0, math.pi),
        )
        b = _rot_rect(
            rng.uniform(0, 10), rng.uniform(0, 10),
            rng.uniform(0.5, 4), rng.uniform(0.5, 4), rng.uniform(0, math.pi),
        )
        mine = _precise_polygon_iou(a, b)
        ref = _shapely_iou_ref(a, b)
        max_diff = max(max_diff, abs(mine - ref))
    assert max_diff < TOL_CROSS, f"max |self - shapely| = {max_diff}"


@pytest.mark.skipif(_ShapelyPolygon is None, reason="shapely 未安装（仅测试参考）")
def test_cross_check_shapely_random_convex_quads():
    rng = random.Random(1122)
    max_diff = 0.0
    for _ in range(N_RANDOM):
        a = _random_convex_quad(rng)
        b = _random_convex_quad(rng)
        mine = _precise_polygon_iou(a, b)
        ref = _shapely_iou_ref(a, b)
        max_diff = max(max_diff, abs(mine - ref))
    assert max_diff < TOL_CROSS, f"max |self - shapely| = {max_diff}"


# ======================================================================
# 双側对拍（Rust omintrack._polygon_iou vs Python _precise_polygon_iou）
# ======================================================================

def _assert_bilateral(a, b) -> float:
    py_val = _precise_polygon_iou(a, b)
    rs_val = omintrack._polygon_iou([[p[0], p[1]] for p in a], [[p[0], p[1]] for p in b])
    assert abs(py_val - rs_val) < TOL_CROSS, (
        f"bilateral diff {abs(py_val - rs_val)}: py={py_val} rust={rs_val}"
    )
    return abs(py_val - rs_val)


def test_bilateral_fixed_cases():
    a = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0)]
    b = [(1.0, 0.5), (3.0, 0.5), (3.0, 2.0), (1.0, 2.0)]
    assert _assert_bilateral(a, b) == 0.0  # 同算法应逐位一致
    d1 = [(0.0, 1.0), (1.0, 0.0), (0.0, -1.0), (-1.0, 0.0)]
    d2 = [(0.5, 1.0), (1.5, 0.0), (0.5, -1.0), (-0.5, 0.0)]
    assert _assert_bilateral(d1, d2) < 1e-15
    assert _assert_bilateral(list(reversed(d1)), list(reversed(d2))) < 1e-15
    assert _assert_bilateral(a, [(5.0, 5.0), (6.0, 5.0), (6.0, 6.0), (5.0, 6.0)]) == 0.0


def test_bilateral_random_rotated_rects():
    rng = random.Random(7701)
    for _ in range(N_RANDOM):
        a = _rot_rect(
            rng.uniform(0, 10), rng.uniform(0, 10),
            rng.uniform(0.5, 4), rng.uniform(0.5, 4), rng.uniform(0, math.pi),
        )
        b = _rot_rect(
            rng.uniform(0, 10), rng.uniform(0, 10),
            rng.uniform(0.5, 4), rng.uniform(0.5, 4), rng.uniform(0, math.pi),
        )
        _assert_bilateral(a, b)


def test_bilateral_random_convex_quads():
    rng = random.Random(9009)
    for _ in range(N_RANDOM):
        a = _random_convex_quad(rng)
        b = _random_convex_quad(rng)
        _assert_bilateral(a, b)
