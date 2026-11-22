from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # VisionCore 类型标注（仅 TYPE_CHECKING，不影响运行时）。
    # 运行时接入见 Pipeline 模式（ai/pipeline_enabled）；适配器见 visioncore.core.adapters。
    from visioncore.core.detection import BBox as CoreBBox
    from visioncore.core.detection import Detection as CoreDetection


def _polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace 公式计算多边形有向面积的绝对值。"""
    if not polygon or len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def _fallback_bbox_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """无法计算精确多边形 IoU 时退回到 bbox 外接矩形 IoU。"""
    ax1, ay1 = min(p[0] for p in a), min(p[1] for p in a)
    ax2, ay2 = max(p[0] for p in a), max(p[1] for p in a)
    bx1, by1 = min(p[0] for p in b), min(p[1] for p in b)
    bx2, by2 = max(p[0] for p in b), max(p[1] for p in b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def _signed_area2(polygon: list[tuple[float, float]]) -> float:
    """Shoelace 有向面积的两倍；正值 = 逆时针（CCW），负值 = 顺时针。"""
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return area


def _ensure_ccw(polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """统一顶点环绕方向为逆时针（CCW）；已 CCW 则原样返回。"""
    if _signed_area2(polygon) < 0.0:
        return list(reversed(polygon))
    return list(polygon)


def _line_intersection(
    p1: tuple[float, float],
    p2: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
) -> tuple[float, float]:
    """线段 p1->p2 与直线 a->b 的交点（Sutherland-Hodgman 裁剪用）。"""
    d1x = p2[0] - p1[0]
    d1y = p2[1] - p1[1]
    d2x = b[0] - a[0]
    d2y = b[1] - a[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) < 1e-12:
        return p2
    t = ((a[0] - p1[0]) * d2y - (a[1] - p1[1]) * d2x) / denom
    return (p1[0] + t * d1x, p1[1] + t * d1y)


def _clip_convex_ccw(
    subject: list[tuple[float, float]],
    clip: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman：凸多边形 subject 被凸 CCW 多边形 clip 逐边裁剪。

    两多边形均须为凸且顶点 CCW 环绕；输出为交集多边形顶点（可能为空）。
    """
    output = list(subject)
    n = len(clip)
    for i in range(n):
        if not output:
            break
        ax, ay = clip[i]
        bx, by = clip[(i + 1) % n]
        ex = bx - ax
        ey = by - ay
        input_list = output
        output = []
        prev = input_list[-1]
        prev_in = ex * (prev[1] - ay) - ey * (prev[0] - ax) >= 0.0
        for cur in input_list:
            cur_in = ex * (cur[1] - ay) - ey * (cur[0] - ax) >= 0.0
            if cur_in:
                if not prev_in:
                    output.append(_line_intersection(prev, cur, (ax, ay), (bx, by)))
                output.append(cur)
            elif prev_in:
                output.append(_line_intersection(prev, cur, (ax, ay), (bx, by)))
            prev = cur
            prev_in = cur_in
    return output


def _precise_polygon_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """精确凸多边形 IoU：Sutherland-Hodgman 凸裁剪求交集 + shoelace 面积。

    假定两多边形均为凸（OBB 四边形场景恒为凸）。双实现契约：
    与 rust/omintrack/src/geometry.rs::precise_polygon_iou 逐行同算法，
    union = area_a + area_b - inter；union <= 1e-8 返回 0.0。
    不依赖 shapely。
    """
    pa = _ensure_ccw(a)
    pb = _ensure_ccw(b)
    area_a = _polygon_area(pa)
    area_b = _polygon_area(pb)
    inter_poly = _clip_convex_ccw(pa, pb)
    inter = _polygon_area(inter_poly)
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def polygon_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """计算两个凸多边形的精确 IoU（Sutherland-Hodgman + shoelace）。

    假定凸多边形（OBB 四边形场景恒为凸）；环绕方向自动归一。
    自实现精确算法，不依赖 shapely；异常时兜底退回外接 bbox IoU。
    """
    if not a or not b or len(a) < 3 or len(b) < 3:
        return 0.0
    try:
        return _precise_polygon_iou(a, b)
    except Exception:
        return _fallback_bbox_iou(a, b)


def polygon_center(polygon: list[tuple[float, float]]) -> tuple[float, float]:
    """多边形顶点平均值作为中心（与 bbox 中心不同）。"""
    if not polygon:
        return 0.0, 0.0
    n = len(polygon)
    cx = sum(p[0] for p in polygon) / n
    cy = sum(p[1] for p in polygon) / n
    return float(cx), float(cy)


def polygon_to_bbox(polygon: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """从多边形顶点得到轴对齐外接矩形 [x1, y1, x2, y2]。"""
    if not polygon:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))

@dataclass
class Keypoint:
    """关键点（用于姿态检测等任务）。"""

    x: float
    y: float
    conf: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {"x": float(self.x), "y": float(self.y), "conf": float(self.conf)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Keypoint":
        return cls(
            x=float(d.get("x", 0.0)),
            y=float(d.get("y", 0.0)),
            conf=float(d.get("conf", 1.0)),
        )


@dataclass
class Detection:
    """统一检测结果类型，向后兼容现有 dict schema。

    字段约定：
    - bbox: [x1, y1, x2, y2] 轴对齐包围盒，所有任务必填。
    - polygon: 4 个归一化角点 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]，OBB 任务必填。
    - keypoints: 关键点列表，Pose 任务必填。
    - track_id / track_state: 由 Tracker 后续附加。

    VisionCore 适配（现可用）:
        此类型可通过 visioncore.core.adapters.to_core_detection() 转换为
        visioncore.core.detection.Detection (不可变, frozen+slots)。
        反向转换使用 from_core_detection() / from_core_with_extras()。
    """

    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    polygon: list[tuple[float, float]] | None = None
    keypoints: list[Keypoint] | None = None
    track_id: int | None = None
    track_state: str = "normal"

    def to_dict(self) -> dict[str, Any]:
        """转换为现有系统已识别的 dict schema，保证向后兼容。"""
        d: dict[str, Any] = {
            "x1": float(self.bbox[0]),
            "y1": float(self.bbox[1]),
            "x2": float(self.bbox[2]),
            "y2": float(self.bbox[3]),
            "confidence": float(self.confidence),
            "label": str(self.label),
            "class_id": int(self.class_id),
            "track_state": str(self.track_state),
        }
        if self.track_id is not None:
            d["track_id"] = int(self.track_id)
        if self.polygon is not None:
            d["polygon"] = [[float(x), float(y)] for x, y in self.polygon]
        if self.keypoints is not None:
            d["keypoints"] = [kp.to_dict() for kp in self.keypoints]
        return d

    def has_polygon(self) -> bool:
        """是否为 OBB 检测（含四边形）。"""
        return self.polygon is not None and len(self.polygon) == 4

    def polygon_iou_with(self, other: "Detection") -> float:
        """与另一个 Detection 的 polygon 计算 IoU；任一不含 polygon 则返回 0。"""
        if not self.has_polygon() or not other.has_polygon():
            return 0.0
        return polygon_iou(self.polygon, other.polygon)

    def polygon_center_point(self) -> tuple[float, float]:
        """返回当前 polygon 的中心点（无 polygon 时退回到 bbox 中心）。"""
        if self.has_polygon():
            return polygon_center(self.polygon)
        return (self.bbox[0] + self.bbox[2]) * 0.5, (self.bbox[1] + self.bbox[3]) * 0.5

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Detection":
        """从现有 dict schema 反序列化。"""
        x1 = float(d.get("x1", d.get("x", 0.0)))
        y1 = float(d.get("y1", d.get("y", 0.0)))
        x2 = float(d.get("x2", 0.0))
        y2 = float(d.get("y2", 0.0))
        polygon = d.get("polygon")
        keypoints = d.get("keypoints")
        return cls(
            class_id=int(d.get("class_id", 0)),
            label=str(d.get("label", "")),
            confidence=float(d.get("confidence", 0.0)),
            bbox=(x1, y1, x2, y2),
            polygon=[(float(p[0]), float(p[1])) for p in polygon] if polygon else None,
            keypoints=[Keypoint.from_dict(kp) for kp in keypoints] if keypoints else None,
            track_id=int(d["track_id"]) if d.get("track_id") is not None else None,
            track_state=str(d.get("track_state", "normal")),
        )
