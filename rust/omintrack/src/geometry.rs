//! 几何工具：box IoU / 精确凸多边形 IoU（Sutherland-Hodgman + shoelace）/ 中心。
//!
//! 等价性说明：Python 侧 `ai.detection.polygon_iou` 自实现精确凸多边形
//! IoU（Sutherland-Hodgman 裁剪 + shoelace 面积，不再依赖 shapely）。
//! 本模块 `precise_polygon_iou` 与 Python `_precise_polygon_iou` 逐行同算法
//! （同样的环绕方向归一、裁剪顺序与浮点运算次序），供双后端行为锁定。
//! Python 侧精确实现的异常兜底 `_fallback_bbox_iou`（外接 bbox IoU）
//! 仅存在于 Python，Rust 主路径不实现、不参与匹配语义。

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

/// 高度 1 维 IoU（HMIoU 调制项，对应 Python `_height_iou`，
/// docs/tracker-nsa-ocm-hmiou.md §2.3）；交集长度 max(0,·) 防负为保护性改动。
pub fn height_iou(a: &BBox, b: &BBox) -> f64 {
    let inter = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
    let span = a[3].max(b[3]) - a[1].min(b[1]);
    if span > 1e-8 {
        inter / span
    } else {
        0.0
    }
}

/// Shoelace 有向面积的两倍；正值 = CCW，负值 = CW。
fn signed_area2(p: &[Point]) -> f64 {
    let n = p.len();
    let mut area = 0.0;
    for i in 0..n {
        let (x1, y1) = (p[i][0], p[i][1]);
        let (x2, y2) = (p[(i + 1) % n][0], p[(i + 1) % n][1]);
        area += x1 * y2 - x2 * y1;
    }
    area
}

/// 统一顶点环绕方向为 CCW；已 CCW 则原样返回。
fn ensure_ccw(p: &[Point]) -> Vec<Point> {
    if signed_area2(p) < 0.0 {
        p.iter().rev().copied().collect()
    } else {
        p.to_vec()
    }
}

/// Shoelace 多边形面积绝对值（少于 3 顶点为 0，对应 Python `_polygon_area`）。
fn polygon_area(p: &[Point]) -> f64 {
    if p.len() < 3 {
        return 0.0;
    }
    signed_area2(p).abs() * 0.5
}

/// 线段 p1->p2 与直线 a->b 的交点（Sutherland-Hodgman 裁剪用）。
fn line_intersection(p1: Point, p2: Point, a: Point, b: Point) -> Point {
    let d1x = p2[0] - p1[0];
    let d1y = p2[1] - p1[1];
    let d2x = b[0] - a[0];
    let d2y = b[1] - a[1];
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 {
        return p2;
    }
    let t = ((a[0] - p1[0]) * d2y - (a[1] - p1[1]) * d2x) / denom;
    [p1[0] + t * d1x, p1[1] + t * d1y]
}

/// Sutherland-Hodgman：凸多边形 subject 被凸 CCW 多边形 clip 逐边裁剪。
/// 与 Python `_clip_convex_ccw` 逐行同算法（同样的边序与浮点运算次序）。
fn clip_convex_ccw(subject: &[Point], clip: &[Point]) -> Vec<Point> {
    let mut output = subject.to_vec();
    let n = clip.len();
    for i in 0..n {
        if output.is_empty() {
            break;
        }
        let a = clip[i];
        let b = clip[(i + 1) % n];
        let ex = b[0] - a[0];
        let ey = b[1] - a[1];
        let input_list = output;
        output = Vec::new();
        let mut prev = *input_list.last().unwrap();
        let mut prev_in = ex * (prev[1] - a[1]) - ey * (prev[0] - a[0]) >= 0.0;
        for &cur in &input_list {
            let cur_in = ex * (cur[1] - a[1]) - ey * (cur[0] - a[0]) >= 0.0;
            if cur_in {
                if !prev_in {
                    output.push(line_intersection(prev, cur, a, b));
                }
                output.push(cur);
            } else if prev_in {
                output.push(line_intersection(prev, cur, a, b));
            }
            prev = cur;
            prev_in = cur_in;
        }
    }
    output
}

/// 精确凸多边形 IoU（对应 Python `_precise_polygon_iou`，逐行同算法）。
/// union = area_a + area_b - inter；union <= 1e-8 返回 0.0。
pub fn precise_polygon_iou(a: &[Point], b: &[Point]) -> f64 {
    let pa = ensure_ccw(a);
    let pb = ensure_ccw(b);
    let area_a = polygon_area(&pa);
    let area_b = polygon_area(&pb);
    let inter_poly = clip_convex_ccw(&pa, &pb);
    let inter = polygon_area(&inter_poly);
    let union = area_a + area_b - inter;
    if union > 1e-8 {
        inter / union
    } else {
        0.0
    }
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

/// 有 polygon（均为 4 顶点）则走精确凸多边形 IoU，否则 bbox IoU（对应 `_shape_iou`）。
pub fn shape_iou(
    trk_box: &BBox,
    trk_polygon: Option<&Vec<Point>>,
    det_box: &BBox,
    det_polygon: Option<&Vec<Point>>,
) -> f64 {
    if let (Some(tp), Some(dp)) = (trk_polygon, det_polygon) {
        if tp.len() == 4 && dp.len() == 4 {
            return precise_polygon_iou(tp, dp);
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
