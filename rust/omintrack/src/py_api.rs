//! PyO3 绑定层：RustTracker class + 测试辅助 pyfunction。

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyTuple;

use numpy::{PyArray1, PyArrayMethods};

use crate::tracker::{
    DetInput, Keypoint, RedetectCtx, TrackSnapshot, TrackerCore,
    TrackerParams,
};

fn parse_keypoints(obj: &PyObject, py: Python<'_>) -> PyResult<Vec<Keypoint>> {
    let list: Vec<HashMap<String, PyObject>> = obj.extract(py)?;
    let mut out = Vec::with_capacity(list.len());
    for kp in &list {
        let x: f64 = kp
            .get("x")
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("keypoint missing 'x'"))?
            .extract(py)?;
        let y: f64 = kp
            .get("y")
            .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("keypoint missing 'y'"))?
            .extract(py)?;
        let conf: f64 = match kp.get("conf") {
            Some(c) => c.extract(py)?,
            None => 0.5,
        };
        out.push(Keypoint { x, y, conf });
    }
    Ok(out)
}

fn parse_polygon(obj: &PyObject, py: Python<'_>) -> PyResult<Option<Vec<[f64; 2]>>> {
    let list: Vec<Vec<f64>> = obj.extract(py)?;
    if list.len() != 4 {
        return Ok(None); // 与 `_extract_polygon` 一致：仅接受 4 顶点
    }
    let mut poly = Vec::with_capacity(4);
    for p in &list {
        if p.len() != 2 {
            return Ok(None);
        }
        poly.push([p[0], p[1]]);
    }
    Ok(Some(poly))
}

fn get_f64(hm: &HashMap<String, PyObject>, key: &str, py: Python<'_>) -> PyResult<f64> {
    hm.get(key)
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(format!("detection missing '{key}'")))?
        .extract(py)
}

/// 从 Python 对象提取 f64 向量：ndarray(f64/f32) 零拷贝快路径，list 回退兼容。
///
/// 生产路径传入 NumPy ndarray（hist float32 520 维 / emb 256 维），
/// 逐元素构造 PyFloat 的 list 慢路径每帧 ~159µs，快路径直接读底层 buffer。
/// f32 -> f64 用 `f64::from` 无损 widening，与旧行为逐位一致。
fn extract_f64_vector(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if let Ok(arr) = obj.downcast::<PyArray1<f64>>() {
        let arr = arr.readonly();
        if let Ok(slice) = arr.as_slice() {
            return Ok(slice.to_vec());
        }
        return Ok(arr.as_array().iter().copied().collect());
    }
    if let Ok(arr) = obj.downcast::<PyArray1<f32>>() {
        let arr = arr.readonly();
        if let Ok(slice) = arr.as_slice() {
            return Ok(slice.iter().map(|&v| f64::from(v)).collect());
        }
        return Ok(arr.as_array().iter().map(|&v| f64::from(v)).collect());
    }
    obj.extract::<Vec<f64>>()
}

fn parse_det(hm: &HashMap<String, PyObject>, py: Python<'_>) -> PyResult<DetInput> {
    let x1 = get_f64(hm, "x1", py)?;
    let y1 = get_f64(hm, "y1", py)?;
    let x2 = get_f64(hm, "x2", py)?;
    let y2 = get_f64(hm, "y2", py)?;
    let confidence = get_f64(hm, "confidence", py)?;
    // 对齐 Python `det.get("label", "")`：缺失时默认为空字符串
    let label: String = match hm.get("label") {
        Some(v) => v.extract(py)?,
        None => String::new(),
    };
    let keypoints = match hm.get("keypoints") {
        Some(o) => parse_keypoints(o, py)?,
        None => Vec::new(),
    };
    let polygon = match hm.get("polygon") {
        Some(o) => parse_polygon(o, py)?,
        None => None,
    };
    Ok(DetInput {
        box_: [x1, y1, x2, y2],
        confidence,
        label,
        keypoints,
        polygon,
    })
}

fn snapshot_to_dict(py: Python<'_>, s: &TrackSnapshot) -> HashMap<String, PyObject> {
    let mut d = HashMap::with_capacity(9);
    d.insert("track_id".to_string(), s.track_id.to_object(py));
    d.insert("x1".to_string(), s.box_[0].to_object(py));
    d.insert("y1".to_string(), s.box_[1].to_object(py));
    d.insert("x2".to_string(), s.box_[2].to_object(py));
    d.insert("y2".to_string(), s.box_[3].to_object(py));
    d.insert("confidence".to_string(), s.confidence.to_object(py));
    d.insert("label".to_string(), s.label.to_object(py));
    let kpts: Vec<HashMap<String, f64>> = s
        .keypoints
        .iter()
        .map(|k| {
            let mut m = HashMap::with_capacity(3);
            m.insert("x".to_string(), k.x);
            m.insert("y".to_string(), k.y);
            m.insert("conf".to_string(), k.conf);
            m
        })
        .collect();
    d.insert("keypoints".to_string(), kpts.to_object(py));
    d.insert("predicted_only".to_string(), s.predicted_only.to_object(py));
    d.insert("track_state".to_string(), s.track_state.to_object(py));
    if let Some(p) = &s.polygon {
        d.insert("polygon".to_string(), p.to_object(py));
    }
    d
}

/// PyO3 redetect 回调封装：保持 GIL 调用，异常吞掉返回 None（对齐 Python try/except）。
struct PyRedetectCtx<'py> {
    py: Python<'py>,
    obj: &'py PyObject,
}

impl<'py> RedetectCtx for PyRedetectCtx<'py> {
    fn try_redetect(&mut self, box_: &[f64; 4], confidence: f64) -> Option<DetInput> {
        // Python 调用约定：redetect_callback(box_list[4], confidence) —— 两个位置参数
        let box_list = box_.to_object(self.py);
        let args = PyTuple::new_bound(self.py, [box_list, confidence.to_object(self.py)]);
        let res = match self.obj.call1(self.py, args) {
            Ok(r) => r,
            Err(_err) => {
                // Python: try/except Exception → refined=None（异常不向外传播）
                // PyErr 被 drop 即丢弃（pyo3 从解释器取走错误指示器时不向外传播）
                return None;
            }
        };
        // None → None
        if res.is_none(self.py) {
            return None;
        }
        // dict → DetInput；缺键/格式错误防御性返回 None
        let hm: HashMap<String, PyObject> = match res.extract(self.py) {
            Ok(h) => h,
            Err(_) => return None,
        };
        match parse_det(&hm, self.py) {
            Ok(d) => Some(d),
            Err(_) => None,
        }
    }
}

/// PyO3 class：RustTracker
#[pyclass]
pub struct RustTracker {
    core: TrackerCore,
}

#[pymethods]
impl RustTracker {
    #[new]
    #[pyo3(signature = (iou_threshold=0.25, max_misses=4, smooth_alpha=0.85, min_hits_to_show=1,
                        redetect_enabled=false, redetect_max_retries=2, redetect_iou_threshold=0.20,
                        filter_type="ukf", mcukf_kernel_sigma=1.0, matching_strategy="hungarian",
                        high_conf_thresh=0.5, velocity_clip=0.30, second_stage_iou_min=0.1,
                        redetect_budget=1, max_output_misses=0,
                        nsa_enabled=false, nsa_interval=1.0, nsa_min=0.1,
                        ocm_enabled=false, ocm_delta_t=3, ocm_weight=0.2,
                        hmiou_enabled=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        iou_threshold: f64,
        max_misses: i32,
        smooth_alpha: f64,
        min_hits_to_show: i32,
        redetect_enabled: bool,
        redetect_max_retries: i32,
        redetect_iou_threshold: f64,
        filter_type: &str,
        mcukf_kernel_sigma: f64,
        matching_strategy: &str,
        high_conf_thresh: f64,
        velocity_clip: f64,
        second_stage_iou_min: f64,
        redetect_budget: i32,
        max_output_misses: i32,
        nsa_enabled: bool,
        nsa_interval: f64,
        nsa_min: f64,
        ocm_enabled: bool,
        ocm_delta_t: i32,
        ocm_weight: f64,
        hmiou_enabled: bool,
    ) -> Self {
        let core = TrackerCore::new(TrackerParams {
            iou_threshold,
            max_misses,
            smooth_alpha,
            min_hits_to_show,
            redetect_enabled,
            redetect_max_retries,
            redetect_iou_threshold,
            filter_type: filter_type.to_string(),
            mcukf_kernel_sigma,
            matching_strategy: matching_strategy.to_string(),
            high_conf_thresh,
            velocity_clip,
            second_stage_iou_min,
            redetect_budget,
            max_output_misses,
            nsa_enabled,
            nsa_interval,
            nsa_min,
            ocm_enabled,
            ocm_delta_t,
            ocm_weight,
            hmiou_enabled,
        });
        RustTracker { core }
    }

    #[pyo3(signature = (detections, det_hist=None, det_emb=None, reid_active=false, redetect_callback=None))]
    fn update(
        &mut self,
        py: Python<'_>,
        detections: Vec<HashMap<String, PyObject>>,
        det_hist: Option<Vec<Option<PyObject>>>,
        det_emb: Option<Vec<Option<PyObject>>>,
        reid_active: bool,
        redetect_callback: Option<PyObject>,
    ) -> PyResult<Vec<HashMap<String, PyObject>>> {
        let n = detections.len();
        let dets: Vec<DetInput> = detections
            .iter()
            .map(|d| parse_det(d, py))
            .collect::<PyResult<_>>()?;
        // 外观数组在 FFI 边界用零拷贝/兼容多路径提取，core 入参仍是 Vec<Option<Vec<f64>>>
        let empty_hist: Vec<Option<Vec<f64>>> = vec![None; n];
        let empty_emb: Vec<Option<Vec<f64>>> = vec![None; n];
        let extract_vecs =
            |v: Vec<Option<PyObject>>| -> PyResult<Vec<Option<Vec<f64>>>> {
                v.into_iter()
                    .map(|o| o.map(|o| extract_f64_vector(o.bind(py))).transpose())
                    .collect()
            };
        let det_hist_owned: Option<Vec<Option<Vec<f64>>>> =
            det_hist.map(extract_vecs).transpose()?;
        let det_emb_owned: Option<Vec<Option<Vec<f64>>>> =
            det_emb.map(extract_vecs).transpose()?;
        let det_hist = det_hist_owned.as_deref().unwrap_or(&empty_hist);
        let det_emb = det_emb_owned.as_deref().unwrap_or(&empty_emb);

        let mut cb_holder;
        let mut null_ctx = NullCtx;
        let cb: &mut dyn RedetectCtx = match &redetect_callback {
            Some(obj) => {
                cb_holder = PyRedetectCtx { py, obj };
                &mut cb_holder
            }
            None => &mut null_ctx,
        };

        let snapshots = self
            .core
            .update(&dets, det_hist, det_emb, reid_active, redetect_callback.is_some(), cb)
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        Ok(snapshots.iter().map(|s| snapshot_to_dict(py, s)).collect())
    }

    fn predict_step(&mut self, py: Python<'_>) -> Vec<HashMap<String, PyObject>> {
        self.core
            .predict_step()
            .iter()
            .map(|s| snapshot_to_dict(py, s))
            .collect()
    }

    fn reset(&mut self) {
        self.core.reset();
    }

    fn set_smooth_alpha(&mut self, alpha: f64) {
        self.core.set_smooth_alpha(alpha);
    }

    #[pyo3(signature = (enabled=None, max_retries=None, iou_threshold=None))]
    fn set_redetect_config(
        &mut self,
        enabled: Option<bool>,
        max_retries: Option<i32>,
        iou_threshold: Option<f64>,
    ) {
        self.core
            .set_redetect_config(enabled, max_retries, iou_threshold);
    }

    #[pyo3(signature = (filter_type, kernel_sigma=None))]
    fn set_filter_type(&mut self, filter_type: &str, kernel_sigma: Option<f64>) {
        self.core.set_filter_type(filter_type, kernel_sigma);
    }

    #[pyo3(signature = (iou_threshold=None, max_misses=None, matching_strategy=None,
                        high_conf_thresh=None, velocity_clip=None, second_stage_iou_min=None,
                        max_output_misses=None,
                        nsa_enabled=None, nsa_interval=None, nsa_min=None,
                        ocm_enabled=None, ocm_delta_t=None, ocm_weight=None,
                        hmiou_enabled=None))]
    #[allow(clippy::too_many_arguments)]
    fn set_tracker_params(
        &mut self,
        iou_threshold: Option<f64>,
        max_misses: Option<i32>,
        matching_strategy: Option<&str>,
        high_conf_thresh: Option<f64>,
        velocity_clip: Option<f64>,
        second_stage_iou_min: Option<f64>,
        max_output_misses: Option<i32>,
        nsa_enabled: Option<bool>,
        nsa_interval: Option<f64>,
        nsa_min: Option<f64>,
        ocm_enabled: Option<bool>,
        ocm_delta_t: Option<i32>,
        ocm_weight: Option<f64>,
        hmiou_enabled: Option<bool>,
    ) {
        self.core.set_tracker_params(
            iou_threshold,
            max_misses,
            matching_strategy,
            high_conf_thresh,
            velocity_clip,
            second_stage_iou_min,
            max_output_misses,
            nsa_enabled,
            nsa_interval,
            nsa_min,
            ocm_enabled,
            ocm_delta_t,
            ocm_weight,
            hmiou_enabled,
        );
    }

    /// facade 外观 ROI：所有 label=="person" 且 misses==0 的 track 的 {tid: [x1,y1,x2,y2]}
    fn appearance_update_rois(&self, py: Python<'_>) -> HashMap<i64, PyObject> {
        self.core
            .appearance_update_rois()
            .into_iter()
            .map(|(tid, b)| (tid, b.to_object(py)))
            .collect()
    }

    /// 写回外部预计算的 HSV 颜色直方图（facade 用；misses>0 的 track 防御性跳过）。
    fn update_color_hists(
        &mut self,
        py: Python<'_>,
        hists: HashMap<i64, PyObject>,
    ) -> PyResult<()> {
        let pairs: Vec<(i64, Vec<f64>)> = hists
            .into_iter()
            .map(|(tid, obj)| extract_f64_vector(obj.bind(py)).map(|v| (tid, v)))
            .collect::<PyResult<_>>()?;
        self.core.update_color_hists(&pairs);
        Ok(())
    }

    fn track_count(&self) -> usize {
        self.core.track_count()
    }

    /// facade ReID 激活门控：当前 label=="person" 的 track 数。
    fn person_track_count(&self) -> i64 {
        self.core.person_track_count()
    }

    /// 测试内省用：AUTO 模式返回当前适配器类型，否则 None。
    fn adaptor_current_type(&self) -> Option<String> {
        self.core.adaptor.as_ref().map(|a| a.current_type().to_string())
    }
}

/// 无回调时的空上下文（redetect_enabled=false 时走此分支，与 Python 行为一致）。
struct NullCtx;

impl RedetectCtx for NullCtx {
    fn try_redetect(&mut self, _box_: &[f64; 4], _confidence: f64) -> Option<DetInput> {
        None
    }
}





