//! 几何工具：box IoU / 多边形外接 bbox IoU / 多边形中心。
//!
//! 等价性说明：Python 侧 `ai.detection.polygon_iou` 优先尝试 shapely，
//! 运行环境无 shapely 时 `_shapely_iou` 返回 None，实际走 `_fallback_bbox_iou`
//! 分支 —— 即"两个多边形顶点集的轴对齐外接矩形的 IoU"。
//! 因此 Rust 直接实现外接 bbox IoU 与 Python 运行行为完全等价
//! （若未来环境装了 shapely，两条路径才会出现语义分歧，届时需重评估）。

pub type BBox = [f64; 4];
pub type Point = [f64; 2];

#[inline]
pub fn clamp01(v: f64) -> f64 {
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

/// 两个 [x1,y1,x2,y2] 归一化 box 的 IoU（对应 Python `_box_iou`）。
pub fn box_iou(a: &BBox, b: &BBox) -> f64 {
    let ix1 = a[0].max(b[0]);
    let iy1 = a[1].max(b[1]);
    let ix2 = a[2].min(b[2]);
    let iy2 = a[3].min(b[3]);
    let inter_w = (ix2 - ix1).max(0.0);
    let inter_h = (iy2 - iy1).max(0.0);
    let inter = inter_w * inter_h;
    let area_a = (a[2] - a[0]) * (a[3] - a[1]);
    let area_b = (b[2] - b[0]) * (b[3] - b[1]);
    let union = area_a + area_b - inter;
    if union > 1e-8 {
        inter / union
    } else {
        0.0
    }
}

/// 多边形顶点集的轴对齐外接 bbox IoU（对应 Python `_fallback_bbox_iou`）。
pub fn polygon_bbox_iou(a: &[Point], b: &[Point]) -> f64 {
    let (ax1, ay1, ax2, ay2) = poly_bounds(a);
    let (bx1, by1, bx2, by2) = poly_bounds(b);
    box_iou(&[ax1, ay1, ax2, ay2], &[bx1, by1, bx2, by2])
}

fn poly_bounds(p: &[Point]) -> (f64, f64, f64, f64) {
    let mut x1 = f64::INFINITY;
    let mut y1 = f64::INFINITY;
    let mut x2 = f64::NEG_INFINITY;
    let mut y2 = f64::NEG_INFINITY;
    for pt in p {
        x1 = x1.min(pt[0]);
        y1 = y1.min(pt[1]);
        x2 = x2.max(pt[0]);
        y2 = y2.max(pt[1]);
    }
    (x1, y1, x2, y2)
}

/// 多边形中心 = 顶点平均值（对应 Python `polygon_center`）。
pub fn polygon_center(p: &[Point]) -> (f64, f64) {
    if p.is_empty() {
        return (0.0, 0.0);
    }
    let n = p.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for pt in p {
        cx += pt[0];
        cy += pt[1];
    }
    (cx / n, cy / n)
}

/// 有 polygon（均为 4 顶点）则走外接 bbox IoU，否则 bbox IoU（对应 `_shape_iou`）。
pub fn shape_iou(
    trk_box: &BBox,
    trk_polygon: Option<&Vec<Point>>,
    det_box: &BBox,
    det_polygon: Option<&Vec<Point>>,
) -> f64 {
    if let (Some(tp), Some(dp)) = (trk_polygon, det_polygon) {
        if tp.len() == 4 && dp.len() == 4 {
            return polygon_bbox_iou(tp, dp);
        }
    }
    box_iou(trk_box, det_box)
}

/// 优先 polygon 中心（>=3 顶点），否则 bbox 中心（对应 `_shape_center`）。
pub fn shape_center(box_: &BBox, polygon: Option<&Vec<Point>>) -> (f64, f64) {
    if let Some(p) = polygon {
        if p.len() >= 3 {
            return polygon_center(p);
        }
    }
    (
        (box_[0] + box_[2]) * 0.5,
        (box_[1] + box_[3]) * 0.5,
    )
}

/// [x1,y1,x2,y2] -> (cx, cy, w, h)，w/h 下限 1e-5（对应 `_box_to_cxcywh`）。
pub fn box_to_cxcywh(box_: &BBox) -> (f64, f64, f64, f64) {
    let w = (box_[2] - box_[0]).max(1e-5);
    let h = (box_[3] - box_[1]).max(1e-5);
    let cx = box_[0] + w * 0.5;
    let cy = box_[1] + h * 0.5;
    (cx, cy, w, h)
}
