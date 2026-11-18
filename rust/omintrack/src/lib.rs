//! omintrack：OminVisionFlowSight 多目标跟踪器 Rust 核心（PyO3 扩展）。
//!
//! 模块划分：
//! - geometry.rs   box/多边形 IoU、中心（含"多边形外接 bbox IoU"等价说明）
//! - hungarian.rs  O(n^3) 匈牙利（最小化，稠密方阵，确定性行优先 tie-break）
//! - filters.rs    UKF / MCUKF / ManifoldUKF（nalgebra 静态矩阵）
//! - adaptor.rs    AUTO 自适应滤波器选择器（阈值 + hysteresis）
//! - tracker.rs    _Track + DetectionTracker 主逻辑（匹配三策略/重检测/清理/AUTO）
//! - py_api.rs     PyO3 class RustTracker 绑定
//!
//! frame 相关部分（`_compute_color_hist_for_box`、ReID 提取）不在 Rust：
//! 由 Python 侧预处理成 det_hist / det_emb 传入；`update_color_hists` /
//! `appearance_update_rois` 供 facade 完成外观 EMA 写回闭环。

mod adaptor;
mod filters;
mod geometry;
mod hungarian;
mod py_api;
mod tracker;

use pyo3::prelude::*;

/// 测试辅助：Bhattacharyya 相似度（对拍 cv2.compareHist 用，语义与
/// Track::appearance_similarity 完全一致）。
#[pyfunction]
fn _bhattacharyya_similarity(h1: Vec<f64>, h2: Vec<f64>) -> f64 {
    tracker::bhattacharyya_similarity(&h1, &h2)
}

/// 测试辅助：匈牙利最小化分配（对拍 scipy.linear_sum_assignment 用）。
/// 输入矩形代价矩阵，返回 (row, col) 配对（剔除补零虚拟行列）。
#[pyfunction]
fn _hungarian_pairs(cost: Vec<Vec<f64>>) -> Vec<(usize, usize)> {
    hungarian::hungarian(&cost)
}

#[pymodule]
fn omintrack(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_api::RustTracker>()?;
    m.add_function(wrap_pyfunction!(_bhattacharyya_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(_hungarian_pairs, m)?)?;
    Ok(())
}
