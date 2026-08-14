from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    # VisionCore 未来统一模型入口 —— 当前仅用于类型检查，不影响运行时。
    # 适配器见 visioncore.core.adapters (to_core_detection / from_core_detection)。
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


def _shapely_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """使用 shapely 计算两个多边形的精确 IoU；失败时返回 None。"""
    try:
        from shapely.geometry import Polygon
        from shapely.errors import TopologicalError
        pa = Polygon(a)
        pb = Polygon(b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        try:
            inter = pa.intersection(pb).area
        except TopologicalError:
            inter = 0.0
        union = pa.area + pb.area - inter
        return inter / union if union > 1e-8 else 0.0
    except Exception:
        return None


def polygon_iou(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    """计算两个多边形的 IoU，优先使用 shapely，不可用时退回到 bbox IoU。"""
    if not a or not b or len(a) < 3 or len(b) < 3:
        return 0.0
    iou = _shapely_iou(a, b)
    if iou is not None:
        return iou
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

    VisionCore 未来入口:
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
