//! 跟踪器核心：_Track + DetectionTracker 的 update/predict_step/匹配/重检测逻辑

use std::collections::{BTreeMap, BTreeSet};

use crate::adaptor::FilterAdaptor;
use crate::filters::{p_condition_number, FilterKind, KalmanFilter, Mat4, Mat8, Vec8};
use crate::geometry::*;

const COLOR_EMA: f64 = 0.9; // 直方图 EMA 系数
const EMBEDDING_EMA: f64 = 0.8; // embedding EMA 系数

#[derive(Clone)]
pub struct Keypoint {
    pub x: f64,
    pub y: f64,
    pub conf: f64,
}

/// 输入检测（dict schema：x1,y1,x2,y2,confidence,label,可选 polygon/keypoints）
#[derive(Clone)]
pub struct DetInput {
    pub box_: BBox,
    pub confidence: f64,
    pub label: String,
    pub keypoints: Vec<Keypoint>,
    pub polygon: Option<Vec<Point>>,
}

/// 输出快照（与 `_Track.to_dict()` 键完全一致）
#[derive(Clone)]
pub struct TrackSnapshot {
    pub track_id: i64,
    pub box_: BBox,
    pub confidence: f64,
    pub label: String,
    pub keypoints: Vec<Keypoint>,
    pub predicted_only: bool,
    pub track_state: String,
    pub polygon: Option<Vec<Point>>,
}

/// ORU 丢失前 KF 全量快照（x, P, Q, R，docs/tracker-oru-ocr-tcm.md §2.1）。
#[derive(Clone, Copy)]
pub struct KfSnapshot {
    pub x: Vec8,
    pub p: Mat8,
    pub q: Mat8,
    pub r: Mat4,
}

pub struct Track {
    pub track_id: i64,
    pub label: String,
    pub box_: BBox,
    pub confidence: f64,
    pub keypoints: Vec<Keypoint>,
    pub polygon: Option<Vec<Point>>,
    pub prev_polygon: Option<Vec<Point>>,
    pub hits: i32,
    pub misses: i32,
    pub age: i32,
    pub kalman: KalmanFilter,
    pub predicted_only: bool,
    pub track_state: String,
    pub redetect_attempts: i32,
    pub color_hist: Option<Vec<f64>>,
    pub embedding: Option<Vec<f64>>,
    pub prev_box: BBox,
    // OCM 观测历史（docs/tracker-nsa-ocm-hmiou.md §2.2）：
    // observations 仅在匹配成功时写入；velocity 为观测动量方向 (vy, vx)
    pub observations: std::collections::BTreeMap<i32, BBox>,
    pub last_observation: Option<BBox>,
    pub velocity: Option<(f64, f64)>,
    // ORU：最近一次 correct 后的 KF 全量快照（§2.1，仅 oru_enabled 时维护）
    pub kf_snapshot: Option<KfSnapshot>,
    // TCM 置信度状态（§2.3）：最近观测 conf、EMA 速度、上一观测 conf
    pub tcm_conf_est: Option<f64>,
    pub tcm_conf_vel: f64,
    pub tcm_conf_prev: Option<f64>,
}

impl Track {
    pub fn new(
        track_id: i64,
        label: String,
        box_: BBox,
        confidence: f64,
        keypoints: Vec<Keypoint>,
        kind: FilterKind,
        kernel_sigma: f64,
        polygon: Option<Vec<Point>>,
        velocity_clip: f64,
        nsa_enabled: bool,
        nsa_interval: f64,
        nsa_min: f64,
        nsa_conf_thresh: f64,
        nsa_interval_sec: f64,
    ) -> Track {
        let mut kalman = KalmanFilter::new(kind, &box_, velocity_clip, kernel_sigma);
        // NSA-KF 配置透传到滤波器（默认关闭）
        kalman.nsa_enabled = nsa_enabled;
        kalman.nsa_interval = nsa_interval;
        kalman.nsa_min = nsa_min;
        kalman.nsa_conf_thresh = nsa_conf_thresh;
        kalman.nsa_interval_sec = nsa_interval_sec;
        let prev_polygon = polygon.clone();
        Track {
            track_id,
            label,
            box_,
            confidence,
            keypoints,
            polygon,
            prev_polygon,
            hits: 1,
            misses: 0,
            age: 0,
            kalman,
            predicted_only: false,
            track_state: "normal".to_string(),
            redetect_attempts: 0,
            color_hist: None,
            embedding: None,
            prev_box: box_,
            observations: std::collections::BTreeMap::new(),
            last_observation: Some(box_), // 以创建检测初始化（对应 Python _Track.__init__）
            velocity: None,
            kf_snapshot: None,
            tcm_conf_est: None,
            tcm_conf_vel: 0.0,
            tcm_conf_prev: None,
        }
    }

    /// TCM 置信度状态更新（匹配成功时调用，§2.3）：conf + EMA 速度。
    pub fn tcm_update(&mut self, conf: f64) {
        if let Some(est) = self.tcm_conf_est {
            let delta = conf - est;
            self.tcm_conf_vel = 0.7 * self.tcm_conf_vel + 0.3 * delta;
            self.tcm_conf_prev = Some(est);
        }
        self.tcm_conf_est = Some(conf);
    }

    /// 高置信度阶段 conf_pred：滤波状态一步预测（est + vel）。
    pub fn tcm_conf_pred_high(&self) -> Option<f64> {
        self.tcm_conf_est.map(|est| est + self.tcm_conf_vel)
    }

    /// 低置信度阶段 conf_pred：最近两个观测线性外推（HybridSORT 二阶段语义；
    /// 当前默认不在第二阶段应用 TCM 项，此方法为契约完整性保留）。
    pub fn tcm_conf_pred_low(&self) -> Option<f64> {
        let est = self.tcm_conf_est?;
        match self.tcm_conf_prev {
            Some(prev) => Some(2.0 * est - prev),
            None => Some(est),
        }
    }

    /// OCM 参考观测：age-delta_t → 逐帧向后回退到最近可用观测 → last_observation
    /// （对应 Python `_Track.previous_observation`）。
    pub fn previous_observation(&self, delta_t: i32) -> Option<BBox> {
        for a in (self.age - delta_t)..self.age {
            if let Some(b) = self.observations.get(&a) {
                return Some(*b);
            }
        }
        self.last_observation
    }

    /// 匹配成功时记录观测并更新 OCM 速度方向（仅保留最近 delta_t+1 个 age；
    /// 对应 Python `_Track.record_observation`）。
    pub fn record_observation(&mut self, box_: &BBox, delta_t: i32) {
        let prev = self.previous_observation(delta_t);
        self.velocity = prev.and_then(|p| speed_direction(&p, box_));
        self.observations.insert(self.age, *box_);
        let min_age = self.age - delta_t;
        self.observations.retain(|&a, _| a >= min_age);
        self.last_observation = Some(*box_);
    }

    pub fn to_snapshot(&self) -> TrackSnapshot {
        TrackSnapshot {
            track_id: self.track_id,
            box_: self.box_,
            confidence: self.confidence,
            label: self.label.clone(),
            keypoints: self.keypoints.clone(),
            predicted_only: self.predicted_only,
            track_state: self.track_state.clone(),
            polygon: self.polygon.clone(),
        }
    }

    /// 热切换滤波器类型，保留状态 (x, P, Q, R)（对应 `swap_filter`）。
    pub fn swap_filter(&mut self, kind: FilterKind, kernel_sigma: f64) {
        let mut new_kf = KalmanFilter::new(kind, &self.box_, self.kalman.velocity_clip, kernel_sigma);
        new_kf.x = self.kalman.x;
        new_kf.p = self.kalman.p;
        new_kf.q = self.kalman.q;
        new_kf.r = self.kalman.r;
        // NSA-KF 配置随滤波器热切换保留
        new_kf.nsa_enabled = self.kalman.nsa_enabled;
        new_kf.nsa_interval = self.kalman.nsa_interval;
        new_kf.nsa_min = self.kalman.nsa_min;
        new_kf.nsa_conf_thresh = self.kalman.nsa_conf_thresh;
        new_kf.nsa_interval_sec = self.kalman.nsa_interval_sec;
        self.kalman = new_kf;
    }

    /// 按预测 box 的平移/缩放外推 polygon（对应 `predict_polygon`）。
    pub fn predict_polygon(&mut self, old_box: &BBox, new_box: &BBox) {
        let poly = match &self.polygon {
            Some(p) if p.len() == 4 => p.clone(),
            _ => return,
        };
        self.prev_polygon = Some(poly.clone());
        let old_cx = (old_box[0] + old_box[2]) * 0.5;
        let old_cy = (old_box[1] + old_box[3]) * 0.5;
        let new_cx = (new_box[0] + new_box[2]) * 0.5;
        let new_cy = (new_box[1] + new_box[3]) * 0.5;
        let old_w = (old_box[2] - old_box[0]).max(1e-5);
        let old_h = (old_box[3] - old_box[1]).max(1e-5);
        let new_w = (new_box[2] - new_box[0]).max(1e-5);
        let new_h = (new_box[3] - new_box[1]).max(1e-5);
        let scale_x = new_w / old_w;
        let scale_y = new_h / old_h;
        let mut predicted = Vec::with_capacity(4);
        for pt in &poly {
            let px = new_cx + (pt[0] - old_cx) * scale_x;
            let py = new_cy + (pt[1] - old_cy) * scale_y;
            predicted.push([clamp01(px), clamp01(py)]);
        }
        self.polygon = Some(predicted);
    }

    /// 写回外部预计算的颜色直方图（对应 `_Track.update_color_hist` 的 EMA 部分；
    /// ROI 提取与直方图计算由 Python facade 完成，经 `update_color_hists` 传入）。
    pub fn apply_color_hist(&mut self, hist: &[f64]) {
        if self.misses > 0 {
            return; // 遮挡期不更新（防御性跳过，与 Python 语义一致）
        }
        match &self.color_hist {
            None => self.color_hist = Some(hist.to_vec()),
            Some(old) => {
                let merged: Vec<f64> = old
                    .iter()
                    .zip(hist.iter())
                    .map(|(o, n)| COLOR_EMA * o + (1.0 - COLOR_EMA) * n)
                    .collect();
                self.color_hist = Some(merged);
            }
        }
    }

    /// 更新 ReID embedding（EMA 0.8/0.2 + L2 归一化，对应 `update_embedding`）。
    pub fn update_embedding(&mut self, emb: Option<&[f64]>) {
        let emb = match emb {
            Some(e) => e,
            None => return,
        };
        if self.misses > 0 {
            return; // 遮挡期不更新
        }
        match &self.embedding {
            None => self.embedding = Some(emb.to_vec()),
            Some(old) => {
                let merged: Vec<f64> = old
                    .iter()
                    .zip(emb.iter())
                    .map(|(o, n)| EMBEDDING_EMA * o + (1.0 - EMBEDDING_EMA) * n)
                    .collect();
                self.embedding = Some(merged);
            }
        }
        // L2 归一化
        if let Some(e) = &mut self.embedding {
            let norm: f64 = e.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 1e-6 {
                for v in e.iter_mut() {
                    *v /= norm;
                }
            }
        }
    }

    /// Bhattacharyya 相似度（对应 `appearance_similarity`，复刻
    /// cv2.compareHist(HISTCMP_BHATTACHARYYA)：hist 已 L2 归一化，
    /// bc = sum(sqrt(h1_i*h2_i))；mean = sum(h)/L；
    /// dist = sqrt(max(1 - bc/sqrt(mean1*mean2*L^2), 0))；similarity = clamp(1-dist, 0, 1)）。
    pub fn appearance_similarity(&self, other_hist: Option<&[f64]>) -> f64 {
        bhattacharyya_similarity_opt(self.color_hist.as_deref(), other_hist)
    }
}

/// Bhattacharyya 相似度（None 输入返回 0.5，与 Python 行为一致）。
pub fn bhattacharyya_similarity_opt(h1: Option<&[f64]>, h2: Option<&[f64]>) -> f64 {
    let (h1, h2) = match (h1, h2) {
        (Some(a), Some(b)) => (a, b),
        _ => return 0.5,
    };
    bhattacharyya_similarity(h1, h2)
}

/// Bhattacharyya 相似度（独立函数，供测试对拍 cv2）。
pub fn bhattacharyya_similarity(h1: &[f64], h2: &[f64]) -> f64 {
    if h1.len() != h2.len() || h1.is_empty() {
        return 0.5;
    }
    let l = h1.len() as f64;
    let mut bc = 0.0f64;
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for i in 0..h1.len() {
        bc += (h1[i] * h2[i]).sqrt();
        s1 += h1[i];
        s2 += h2[i];
    }
    let mean1 = s1 / l;
    let mean2 = s2 / l;
    let denom = (mean1 * mean2 * l * l).sqrt();
    let dist = (1.0 - bc / denom).max(0.0).sqrt();
    (1.0 - dist).max(0.0).min(1.0)
}

/// 余弦相似度（对应 ReIDBackend.cosine_similarity）。
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let na: f64 = a.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb: f64 = b.iter().map(|v| v * v).sum::<f64>().sqrt();
    if na < 1e-6 || nb < 1e-6 {
        return 0.0;
    }
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    dot / (na * nb)
}

/// 重检测回调抽象（py_api 用 PyO3 callable 实现）。
pub trait RedetectCtx {
    /// 输入归一化预测框 [x1,y1,x2,y2] 与原置信度，返回 detection 或 None。
    /// 回调抛异常时由实现方吞掉并返回 None（对齐 Python try/except）。
    fn try_redetect(&mut self, box_: &BBox, confidence: f64) -> Option<DetInput>;
}

pub struct TrackerParams {
    pub iou_threshold: f64,
    pub max_misses: i32,
    pub smooth_alpha: f64,
    pub min_hits_to_show: i32,
    pub redetect_enabled: bool,
    pub redetect_max_retries: i32,
    pub redetect_iou_threshold: f64,
    pub filter_type: String,
    pub mcukf_kernel_sigma: f64,
    pub matching_strategy: String,
    pub high_conf_thresh: f64,
    pub velocity_clip: f64,
    pub second_stage_iou_min: f64,
    pub redetect_budget: i32,
    pub max_output_misses: i32,
    // 跟踪增强（docs/tracker-nsa-ocm-hmiou.md，默认全部关闭 opt-in）
    pub nsa_enabled: bool,
    pub nsa_interval: f64,
    pub nsa_min: f64,
    pub nsa_conf_thresh: f64,
    pub nsa_interval_sec: f64,
    pub ocm_enabled: bool,
    pub ocm_delta_t: i32,
    pub ocm_weight: f64,
    pub hmiou_enabled: bool,
    // 批次 3（docs/tracker-oru-ocr-tcm.md，默认全部关闭 opt-in）
    pub oru_enabled: bool,
    pub ocr_enabled: bool,
    pub ocr_threshold: f64,
    pub tcm_enabled: bool,
    pub tcm_weight: f64,
}

impl Default for TrackerParams {
    fn default() -> TrackerParams {
        TrackerParams {
            iou_threshold: 0.25,
            max_misses: 4,
            smooth_alpha: 0.85,
            min_hits_to_show: 1,
            redetect_enabled: false,
            redetect_max_retries: 2,
            redetect_iou_threshold: 0.20,
            filter_type: "ukf".to_string(),
            mcukf_kernel_sigma: 1.0, // 相关熵核带宽 σ 默认值（无尺寸缩放）
            matching_strategy: "hungarian".to_string(),
            high_conf_thresh: 0.5,
            velocity_clip: 0.30,
            second_stage_iou_min: 0.1,
            redetect_budget: 1,
            max_output_misses: 0, // 输出门控：misses 超过此值的 track 不输出（内部生命周期不受影响）
            nsa_enabled: false,
            nsa_interval: 1.0,
            nsa_min: 0.1,
            nsa_conf_thresh: 0.6,
            nsa_interval_sec: 2.0,
            ocm_enabled: false,
            ocm_delta_t: 3,
            ocm_weight: 0.2,
            hmiou_enabled: false,
            oru_enabled: false,
            ocr_enabled: false,
            ocr_threshold: 0.3,
            tcm_enabled: false,
            tcm_weight: 0.2,
        }
    }
}

pub struct TrackerCore {
    pub iou_threshold: f64,
    pub max_misses: i32,
    pub max_output_misses: i32,
    pub smooth_alpha: f64,
    pub min_hits: i32,
    pub next_id: i64,
    pub tracks: BTreeMap<i64, Track>,
    pub redetect_enabled: bool,
    pub redetect_max_retries: i32,
    pub redetect_iou_threshold: f64,
    pub redetect_budget: i32,
    pub filter_type: String,
    pub mcukf_kernel_sigma: f64,
    pub matching_strategy: String,
    pub high_conf_thresh: f64,
    pub velocity_clip: f64,
    pub second_stage_iou_min: f64,
    pub adaptor: Option<FilterAdaptor>,
    // 跟踪增强配置（默认全关 ⇒ 默认路径行为逐位不变）
    pub nsa_enabled: bool,
    pub nsa_interval: f64,
    pub nsa_min: f64,
    pub nsa_conf_thresh: f64,
    pub nsa_interval_sec: f64,
    pub ocm_enabled: bool,
    pub ocm_delta_t: i32,
    pub ocm_weight: f64,
    pub hmiou_enabled: bool,
    // 批次 3 配置（默认全关 ⇒ 默认路径行为逐位不变）
    pub oru_enabled: bool,
    pub ocr_enabled: bool,
    pub ocr_threshold: f64,
    pub tcm_enabled: bool,
    pub tcm_weight: f64,
}

impl TrackerCore {
    pub fn new(p: TrackerParams) -> TrackerCore {
        let adaptor = if p.filter_type == "auto" {
            Some(FilterAdaptor::new(p.mcukf_kernel_sigma))
        } else {
            None
        };
        TrackerCore {
            iou_threshold: p.iou_threshold,
            max_misses: p.max_misses,
            max_output_misses: p.max_output_misses.max(0),
            smooth_alpha: p.smooth_alpha,
            min_hits: p.min_hits_to_show,
            next_id: 1,
            tracks: BTreeMap::new(),
            redetect_enabled: p.redetect_enabled,
            redetect_max_retries: p.redetect_max_retries.max(0),
            redetect_iou_threshold: p.redetect_iou_threshold,
            redetect_budget: p.redetect_budget.max(0),
            filter_type: p.filter_type,
            mcukf_kernel_sigma: p.mcukf_kernel_sigma,
            matching_strategy: p.matching_strategy,
            high_conf_thresh: p.high_conf_thresh,
            velocity_clip: p.velocity_clip,
            second_stage_iou_min: p.second_stage_iou_min,
            adaptor,
            nsa_enabled: p.nsa_enabled,
            nsa_interval: p.nsa_interval,
            nsa_min: p.nsa_min,
            nsa_conf_thresh: p.nsa_conf_thresh,
            nsa_interval_sec: p.nsa_interval_sec,
            ocm_enabled: p.ocm_enabled,
            ocm_delta_t: p.ocm_delta_t.max(1),
            ocm_weight: p.ocm_weight,
            hmiou_enabled: p.hmiou_enabled,
            oru_enabled: p.oru_enabled,
            ocr_enabled: p.ocr_enabled,
            ocr_threshold: p.ocr_threshold,
            tcm_enabled: p.tcm_enabled,
            tcm_weight: p.tcm_weight,
        }
    }

    fn effective_kind(&self) -> FilterKind {
        let name = match &self.adaptor {
            Some(a) => a.current_type(),
            None => &self.filter_type,
        };
        FilterKind::from_name(name)
    }

    pub fn reset(&mut self) {
        self.tracks.clear();
        self.next_id = 1;
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }

    pub fn person_track_count(&self) -> i64 {
        self.tracks.values().filter(|t| t.label == "person").count() as i64
    }

    /// facade 外观 ROI：所有 label=="person" 且 misses==0 的 track 的 {tid: [x1,y1,x2,y2]}
    /// （对应 `_Track.update_color_hist` 的 misses>0 跳过语义；facade 从帧算 HSV 后
    /// 经 `update_color_hists` 写回）。
    pub fn appearance_update_rois(&self) -> Vec<(i64, BBox)> {
        self.tracks
            .values()
            .filter(|t| t.label == "person" && t.misses == 0)
            .map(|t| (t.track_id, t.box_))
            .collect()
    }

    pub fn set_smooth_alpha(&mut self, alpha: f64) {
        self.smooth_alpha = alpha.clamp(0.0, 1.0);
    }

    pub fn set_redetect_config(&mut self, enabled: Option<bool>, max_retries: Option<i32>, iou_threshold: Option<f64>) {
        if let Some(e) = enabled {
            self.redetect_enabled = e;
        }
        if let Some(m) = max_retries {
            self.redetect_max_retries = m.max(0);
        }
        if let Some(i) = iou_threshold {
            self.redetect_iou_threshold = i;
        }
    }

    pub fn set_filter_type(&mut self, filter_type: &str, kernel_sigma: Option<f64>) {
        self.filter_type = filter_type.to_string();
        if let Some(s) = kernel_sigma {
            self.mcukf_kernel_sigma = s;
        }
        if filter_type == "auto" {
            match &mut self.adaptor {
                None => self.adaptor = Some(FilterAdaptor::new(self.mcukf_kernel_sigma)),
                Some(a) => {
                    if kernel_sigma.is_some() {
                        a.set_kernel_sigma(self.mcukf_kernel_sigma);
                    }
                }
            }
        } else {
            self.adaptor = None;
        }
    }

    pub fn set_tracker_params(
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
        nsa_conf_thresh: Option<f64>,
        nsa_interval_sec: Option<f64>,
        ocm_enabled: Option<bool>,
        ocm_delta_t: Option<i32>,
        ocm_weight: Option<f64>,
        hmiou_enabled: Option<bool>,
        oru_enabled: Option<bool>,
        ocr_enabled: Option<bool>,
        ocr_threshold: Option<f64>,
        tcm_enabled: Option<bool>,
        tcm_weight: Option<f64>,
    ) {
        if let Some(v) = iou_threshold {
            self.iou_threshold = v;
        }
        if let Some(v) = max_misses {
            self.max_misses = v;
        }
        if let Some(v) = matching_strategy {
            self.matching_strategy = v.to_string();
        }
        if let Some(v) = high_conf_thresh {
            self.high_conf_thresh = v;
        }
        if let Some(v) = velocity_clip {
            self.velocity_clip = v;
            for trk in self.tracks.values_mut() {
                trk.kalman.velocity_clip = v;
            }
        }
        if let Some(v) = second_stage_iou_min {
            self.second_stage_iou_min = v;
        }
        if let Some(v) = max_output_misses {
            self.max_output_misses = v.max(0);
        }
        // 跟踪增强参数（NSA 需同步到已有 track 的滤波器）
        if let Some(v) = nsa_enabled {
            self.nsa_enabled = v;
        }
        if let Some(v) = nsa_interval {
            self.nsa_interval = v;
        }
        if let Some(v) = nsa_min {
            self.nsa_min = v;
        }
        if let Some(v) = nsa_conf_thresh {
            self.nsa_conf_thresh = v;
        }
        if let Some(v) = nsa_interval_sec {
            self.nsa_interval_sec = v;
        }
        if nsa_enabled.is_some()
            || nsa_interval.is_some()
            || nsa_min.is_some()
            || nsa_conf_thresh.is_some()
            || nsa_interval_sec.is_some()
        {
            for trk in self.tracks.values_mut() {
                trk.kalman.nsa_enabled = self.nsa_enabled;
                trk.kalman.nsa_interval = self.nsa_interval;
                trk.kalman.nsa_min = self.nsa_min;
                trk.kalman.nsa_conf_thresh = self.nsa_conf_thresh;
                trk.kalman.nsa_interval_sec = self.nsa_interval_sec;
            }
        }
        if let Some(v) = ocm_enabled {
            self.ocm_enabled = v;
        }
        if let Some(v) = ocm_delta_t {
            self.ocm_delta_t = v.max(1);
        }
        if let Some(v) = ocm_weight {
            self.ocm_weight = v;
        }
        if let Some(v) = hmiou_enabled {
            self.hmiou_enabled = v;
        }
        if let Some(v) = oru_enabled {
            self.oru_enabled = v;
        }
        if let Some(v) = ocr_enabled {
            self.ocr_enabled = v;
        }
        if let Some(v) = ocr_threshold {
            self.ocr_threshold = v;
        }
        if let Some(v) = tcm_enabled {
            self.tcm_enabled = v;
        }
        if let Some(v) = tcm_weight {
            self.tcm_weight = v;
        }
    }

    /// 跳帧路径专用：仅推进卡尔曼（对应 `predict_step`）。
    pub fn predict_step(&mut self) -> Vec<TrackSnapshot> {
        for trk in self.tracks.values_mut() {
            let old_box = trk.box_;
            let predicted = trk.kalman.predict();
            trk.box_ = predicted;
            trk.predict_polygon(&old_box, &predicted);
            trk.predicted_only = true;
            trk.track_state = "predicted".to_string();
        }
        self.active_tracks()
    }

    fn active_tracks(&self) -> Vec<TrackSnapshot> {
        // 输出门控：除 hits 达标外，还要求 misses 未超 max_output_misses；
        // 仅影响可见性，内部 track 仍按 max_misses 保留、可被后续 correct 恢复显示
        self.tracks
            .values()
            .filter(|t| t.hits >= self.min_hits && t.misses <= self.max_output_misses)
            .map(|t| t.to_snapshot())
            .collect()
    }

    fn cleanup_expired(&mut self) {
        let max_misses = self.max_misses;
        self.tracks.retain(|_, t| t.misses <= max_misses);
    }

    /// AUTO 模式：评估是否需要热切换滤波器类型（对应 `_evaluate_auto_switch`）。
    fn evaluate_auto_switch(&mut self) {
        if self.adaptor.is_none() {
            return;
        }
        let min_hits = self.min_hits;
        let mut conds: Vec<f64> = self
            .tracks
            .values()
            .filter(|t| t.hits >= min_hits)
            .map(|t| p_condition_number(&t.kalman.p))
            .collect();
        conds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = conds.len();
        let median_cond = if n == 0 {
            1.0
        } else if n % 2 == 1 {
            conds[n / 2]
        } else {
            (conds[n / 2 - 1] + conds[n / 2]) * 0.5
        };
        let new_type = self.adaptor.as_mut().unwrap().evaluate(median_cond);
        if let Some(type_name) = new_type {
            let kind = FilterKind::from_name(&type_name);
            let sigma = self.mcukf_kernel_sigma;
            for trk in self.tracks.values_mut() {
                trk.swap_filter(kind, sigma);
            }
        }
    }

    /// 写回外部预计算的 HSV 颜色直方图（facade 用；misses>0 的 track 跳过）。
    pub fn update_color_hists(&mut self, hists: &[(i64, Vec<f64>)]) {
        for (tid, hist) in hists {
            if let Some(trk) = self.tracks.get_mut(tid) {
                trk.apply_color_hist(hist);
            }
        }
    }

    /// 主更新流程（对应 `update`；frame 相关外观预处理由 Python 侧完成，
    /// 以 det_hist / det_emb / reid_active 形式传入）。
    pub fn update(
        &mut self,
        dets: &[DetInput],
        det_hist: &[Option<Vec<f64>>],
        det_emb: &[Option<Vec<f64>>],
        reid_active: bool,
        redetect_callback_present: bool,
        cb: &mut dyn RedetectCtx,
    ) -> Result<Vec<TrackSnapshot>, String> {
        // --- 0. 先 predict 再匹配 ---
        for trk in self.tracks.values_mut() {
            trk.age += 1;
            let old_box = trk.box_;
            let predicted = trk.kalman.predict();
            trk.box_ = predicted;
            trk.predict_polygon(&old_box, &predicted);
        }

        if dets.is_empty() {
            let matched = BTreeSet::new();
            self.try_redetect_for_unmatched(&matched, redetect_callback_present, cb)?;
            self.cleanup_expired();
            self.evaluate_auto_switch();
            return Ok(self.active_tracks());
        }

        // --- 2. 匹配策略分发 ---
        let det_conf: Vec<f64> = dets.iter().map(|d| d.confidence).collect();
        let (mut matched_tids, mut matched_dis, skip_new) =
            self.run_matching(dets, &det_conf, det_hist, det_emb, reid_active)?;

        // --- 2.5 OCR：丢失态 track 用 last_observation 与剩余检测二次关联（§2.2；
        // 在新建 track 与 ROI 重检测之前——用已有剩余检测，零额外推理）---
        if self.ocr_enabled {
            self.match_ocr(dets, &mut matched_tids, &mut matched_dis)?;
        }

        // --- 3. 未匹配的检测 → 新建 track ---
        for (i, det) in dets.iter().enumerate() {
            if !matched_dis.contains(&i) && !skip_new.contains(&i) {
                let tid = self.next_id;
                self.next_id += 1;
                let kind = self.effective_kind();
                let mut trk = Track::new(
                    tid,
                    det.label.clone(),
                    det.box_,
                    det.confidence,
                    det.keypoints.clone(),
                    kind,
                    self.mcukf_kernel_sigma,
                    det.polygon.clone(),
                    self.velocity_clip,
                    self.nsa_enabled,
                    self.nsa_interval,
                    self.nsa_min,
                    self.nsa_conf_thresh,
                    self.nsa_interval_sec,
                );
                if let Some(Some(h)) = det_hist.get(i) {
                    trk.color_hist = Some(h.clone());
                }
                if let Some(Some(e)) = det_emb.get(i) {
                    trk.embedding = Some(e.clone());
                }
                // TCM：以创建观测初始化置信度状态
                if self.tcm_enabled {
                    trk.tcm_update(det.confidence);
                }
                self.tracks.insert(tid, trk);
                matched_tids.insert(tid); // 新建 track 已 accounted
            }
        }

        // --- 4. 未匹配的 track → 重检测或 miss+1 ---
        self.try_redetect_for_unmatched(&matched_tids, redetect_callback_present, cb)?;

        // --- 5. 清理超期 track ---
        self.cleanup_expired();

        // --- 6. AUTO 评估 ---
        self.evaluate_auto_switch();

        Ok(self.active_tracks())
    }

    // ===================================================================
    // 匹配策略：greedy / hungarian / cascade
    // ===================================================================

    fn run_matching(
        &mut self,
        dets: &[DetInput],
        det_conf: &[f64],
        det_hist: &[Option<Vec<f64>>],
        det_emb: &[Option<Vec<f64>>],
        reid_active: bool,
    ) -> Result<(BTreeSet<i64>, BTreeSet<usize>, BTreeSet<usize>), String> {
        match self.matching_strategy.as_str() {
            "cascade" => self.match_cascade(dets, det_conf, det_hist, det_emb, reid_active),
            "hungarian" => {
                let all: Vec<usize> = (0..dets.len()).collect();
                let (tids, dis) = self.match_hungarian(
                    dets, &all, det_hist, det_emb, reid_active,
                    self.iou_threshold, 0.05, None, true,
                )?;
                Ok((tids, dis, BTreeSet::new()))
            }
            _ => self.match_greedy(dets, det_conf, det_hist),
        }
    }

    fn match_greedy(
        &mut self,
        dets: &[DetInput],
        det_conf: &[f64],
        det_hist: &[Option<Vec<f64>>],
    ) -> Result<(BTreeSet<i64>, BTreeSet<usize>, BTreeSet<usize>), String> {
        let mut matched_tids: BTreeSet<i64> = BTreeSet::new();
        let mut matched_dis: BTreeSet<usize> = BTreeSet::new();
        // 按置信度降序（稳定排序；测试数据置信度无并列）
        let mut det_order: Vec<usize> = (0..dets.len()).collect();
        det_order.sort_by(|&a, &b| {
            det_conf[b].partial_cmp(&det_conf[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let iou_threshold = self.iou_threshold;
        let alpha = self.smooth_alpha;

        for &di in &det_order {
            if matched_dis.contains(&di) {
                continue;
            }
            let det = &dets[di];
            let (det_cx, det_cy) = shape_center(&det.box_, det.polygon.as_ref());
            let mut best_score = -1.0f64;
            let mut best_tid: Option<i64> = None;

            for (&tid, trk) in self.tracks.iter() {
                if matched_tids.contains(&tid) || trk.label != det.label {
                    continue;
                }
                let iou = shape_iou(&trk.box_, trk.polygon.as_ref(), &det.box_, det.polygon.as_ref());
                let (trk_cx, trk_cy) = shape_center(&trk.box_, trk.polygon.as_ref());
                let dist = ((trk_cx - det_cx).powi(2) + (trk_cy - det_cy).powi(2)).sqrt();
                let avg_size = ((trk.box_[2] - trk.box_[0])
                    + (det.box_[2] - det.box_[0])
                    + (trk.box_[3] - trk.box_[1])
                    + (det.box_[3] - det.box_[1]))
                    * 0.25;
                let center_close = avg_size > 1e-3 && dist < 0.5 * avg_size;
                let accept = iou > iou_threshold || (iou > 0.05 && center_close);
                if accept {
                    let appearance_bonus =
                        trk.appearance_similarity(det_hist.get(di).and_then(|h| h.as_deref())) * 0.15;
                    let score = iou
                        + (1.0 - (dist / avg_size.max(1e-3)).min(1.0)) * 0.1
                        + appearance_bonus;
                    if score > best_score {
                        best_score = score;
                        best_tid = Some(tid);
                    }
                }
            }

            if let Some(tid) = best_tid {
                matched_tids.insert(tid);
                matched_dis.insert(di);
                let new_box = det.box_;
                let det_clone = det.clone();
                let ocm_dt = if self.ocm_enabled { Some(self.ocm_delta_t) } else { None };
                let (oru, tcm) = (self.oru_enabled, self.tcm_enabled);
                let trk = self.tracks.get_mut(&tid).unwrap();
                update_track(trk, &det_clone, &new_box, alpha, &mut self.adaptor, ocm_dt, oru, tcm)?;
            }
        }
        Ok((matched_tids, matched_dis, BTreeSet::new()))
    }

    #[allow(clippy::too_many_arguments)]
    fn match_hungarian(
        &mut self,
        dets: &[DetInput],
        det_indices: &[usize],
        det_hist: &[Option<Vec<f64>>],
        det_emb: &[Option<Vec<f64>>],
        reid_active: bool,
        iou_gate: f64,
        appearance_weight: f64,
        track_filter: Option<&BTreeSet<i64>>,
        tcm_apply: bool,
    ) -> Result<(BTreeSet<i64>, BTreeSet<usize>), String> {
        let mut matched_tids: BTreeSet<i64> = BTreeSet::new();
        let mut matched_dis: BTreeSet<usize> = BTreeSet::new();

        // 按 label 分组匹配（labels 排序保证确定性；Python set 迭代序不影响结果
        // 因为各 label 的 track/det 子集不相交）
        let mut labels: BTreeSet<&str> = BTreeSet::new();
        for &di in det_indices {
            if !matched_dis.contains(&di) {
                labels.insert(dets[di].label.as_str());
            }
        }
        let track_labels: BTreeSet<String> =
            self.tracks.values().map(|t| t.label.clone()).collect();
        let alpha = self.smooth_alpha;
        let hmiou_enabled = self.hmiou_enabled;
        let ocm_enabled = self.ocm_enabled;
        let ocm_delta_t = self.ocm_delta_t;
        let ocm_weight = self.ocm_weight;
        let tcm_enabled = self.tcm_enabled;
        let tcm_weight = self.tcm_weight;

        for label in labels {
            if !track_labels.contains(label) {
                continue;
            }
            let local_dets: Vec<usize> = det_indices
                .iter()
                .copied()
                .filter(|&i| dets[i].label == label && !matched_dis.contains(&i))
                .collect();
            let track_items: Vec<i64> = self
                .tracks
                .keys()
                .copied()
                .filter(|tid| {
                    !matched_tids.contains(tid)
                        && self.tracks[tid].label == label
                        && track_filter.map_or(true, |f| f.contains(tid))
                })
                .collect();
            if local_dets.is_empty() || track_items.is_empty() {
                continue;
            }

            let n_dets = local_dets.len();
            let n_tracks = track_items.len();
            let mut cost = vec![vec![1e6f64; n_tracks]; n_dets];
            // 门控通过掩码：匹配有效性以此为准（P0-1 修复：OCM 负代价可使合法
            // pair 的 cost > 0.999，旧的 cost<0.999 判据会误丢合法匹配）
            let mut accepted = vec![vec![false; n_tracks]; n_dets];

            for (di_idx, &di) in local_dets.iter().enumerate() {
                let det = &dets[di];
                let (det_cx, det_cy) = shape_center(&det.box_, det.polygon.as_ref());
                for (ti_idx, &tid) in track_items.iter().enumerate() {
                    let trk = &self.tracks[&tid];
                    let mut iou =
                        shape_iou(&trk.box_, trk.polygon.as_ref(), &det.box_, det.polygon.as_ref());
                    // HMIoU：IoU 项与门控统一以 hmiou 替换 2D IoU（§2.3）；
                    // OBB polygon 对不适用（保持 2D polygon IoU）
                    let polygon_pair = matches!(
                        (&trk.polygon, &det.polygon),
                        (Some(tp), Some(dp)) if tp.len() == 4 && dp.len() == 4
                    );
                    if hmiou_enabled && !polygon_pair {
                        iou *= height_iou(&trk.box_, &det.box_);
                    }
                    let (trk_cx, trk_cy) = shape_center(&trk.box_, trk.polygon.as_ref());
                    let dist = ((trk_cx - det_cx).powi(2) + (trk_cy - det_cy).powi(2)).sqrt();
                    let avg_size = ((trk.box_[2] - trk.box_[0])
                        + (det.box_[2] - det.box_[0])
                        + (trk.box_[3] - trk.box_[1])
                        + (det.box_[3] - det.box_[1]))
                        * 0.25;
                    let center_close = avg_size > 1e-3 && dist < 0.5 * avg_size;
                    let accept = iou > iou_gate || (iou > 0.05 && center_close);
                    if accept {
                        let center_proximity = 1.0 - (dist / avg_size.max(1e-3)).min(1.0);
                        let det_emb_i = det_emb.get(di).and_then(|e| e.as_deref());
                        let mut score = if reid_active && det_emb_i.is_some() && trk.embedding.is_some() {
                            let reid_sim =
                                cosine_similarity(trk.embedding.as_ref().unwrap(), det_emb_i.unwrap())
                                    .max(0.0);
                            0.4 * iou + 0.4 * reid_sim + 0.2 * center_proximity
                        } else {
                            let appearance_sim =
                                trk.appearance_similarity(det_hist.get(di).and_then(|h| h.as_deref()));
                            0.85 * iou + 0.10 * center_proximity + appearance_weight * appearance_sim
                        };
                        // OCM：观测中心动量方向代价叠加到 score（§2.2；ReID 分支同样叠加）
                        if ocm_enabled {
                            if let Some(vel) = trk.velocity {
                                if let Some(prev_obs) = trk.previous_observation(ocm_delta_t) {
                                    if let Some(dir) = speed_direction(&prev_obs, &det.box_) {
                                        let cos_sim =
                                            (vel.0 * dir.0 + vel.1 * dir.1).clamp(-1.0, 1.0);
                                        let theta = cos_sim.acos();
                                        let angle_term = (std::f64::consts::PI * 0.5
                                            - theta.abs())
                                            / std::f64::consts::PI;
                                        score += angle_term * ocm_weight * det.confidence;
                                    }
                                }
                            }
                        }
                        // TCM：置信度一致性奖励（§2.3；仅第一阶段/单阶段，
                        // 第二阶段不加以免低分找回被置信度差压制）
                        if tcm_enabled && tcm_apply {
                            if let Some(conf_pred) = trk.tcm_conf_pred_high() {
                                score -= (det.confidence - conf_pred).abs() * tcm_weight;
                            }
                        }
                        cost[di_idx][ti_idx] = 1.0 - score;
                        accepted[di_idx][ti_idx] = true;
                    }
                }
            }

            let pairs = crate::hungarian::hungarian(&cost);
            for (r, c) in pairs {
                if accepted[r][c] { // 有效匹配（门控布尔掩码；不再以 cost 哨兵判定）
                    let di = local_dets[r];
                    let tid = track_items[c];
                    matched_tids.insert(tid);
                    matched_dis.insert(di);
                    let new_box = dets[di].box_;
                    let det_clone = dets[di].clone();
                    let update_emb =
                        reid_active && det_emb.get(di).and_then(|e| e.as_ref()).is_some();
                    let ocm_dt = if self.ocm_enabled { Some(self.ocm_delta_t) } else { None };
                    let (oru, tcm) = (self.oru_enabled, self.tcm_enabled);
                    let trk = self.tracks.get_mut(&tid).unwrap();
                    update_track(trk, &det_clone, &new_box, alpha, &mut self.adaptor, ocm_dt, oru, tcm)?;
                    if update_emb {
                        let emb = det_emb[di].as_ref().unwrap().clone();
                        let trk = self.tracks.get_mut(&tid).unwrap();
                        trk.update_embedding(Some(&emb));
                    }
                }
            }
        }
        Ok((matched_tids, matched_dis))
    }

    /// OCR 观测中心恢复（§2.2）：丢失态 track 用 last_observation（非预测框）
    /// 与剩余未匹配检测做 IoU 二次关联（匈牙利分配，门控 ocr_threshold，同类）。
    /// 匹配成功走正常 update_track（OCM 观测照常记录；ORU 开启且丢失态自动重放）。
    fn match_ocr(
        &mut self,
        dets: &[DetInput],
        matched_tids: &mut BTreeSet<i64>,
        matched_dis: &mut BTreeSet<usize>,
    ) -> Result<(), String> {
        let lost: Vec<i64> = self
            .tracks
            .iter()
            .filter(|(tid, trk)| {
                !matched_tids.contains(tid) && trk.misses > 0 && trk.last_observation.is_some()
            })
            .map(|(tid, _)| *tid)
            .collect();
        let det_indices: Vec<usize> = (0..dets.len())
            .filter(|i| !matched_dis.contains(i))
            .collect();
        if lost.is_empty() || det_indices.is_empty() {
            return Ok(());
        }
        let n_dets = det_indices.len();
        let n_tracks = lost.len();
        let mut cost = vec![vec![1e6f64; n_tracks]; n_dets];
        let mut accepted = vec![vec![false; n_tracks]; n_dets];
        for (di_idx, &di) in det_indices.iter().enumerate() {
            let det = &dets[di];
            for (ti_idx, &tid) in lost.iter().enumerate() {
                let trk = &self.tracks[&tid];
                if trk.label != det.label {
                    continue;
                }
                let iou = box_iou(&trk.last_observation.unwrap(), &det.box_);
                if iou > self.ocr_threshold {
                    cost[di_idx][ti_idx] = 1.0 - iou;
                    accepted[di_idx][ti_idx] = true;
                }
            }
        }
        let alpha = self.smooth_alpha;
        let ocm_dt = if self.ocm_enabled { Some(self.ocm_delta_t) } else { None };
        let oru = self.oru_enabled;
        let tcm = self.tcm_enabled;
        let pairs = crate::hungarian::hungarian(&cost);
        for (r, c) in pairs {
            if accepted[r][c] {
                let di = det_indices[r];
                let tid = lost[c];
                matched_tids.insert(tid);
                matched_dis.insert(di);
                let new_box = dets[di].box_;
                let det_clone = dets[di].clone();
                let trk = self.tracks.get_mut(&tid).unwrap();
                update_track(trk, &det_clone, &new_box, alpha, &mut self.adaptor, ocm_dt, oru, tcm)?;
            }
        }
        Ok(())
    }

    fn match_cascade(
        &mut self,
        dets: &[DetInput],
        det_conf: &[f64],
        det_hist: &[Option<Vec<f64>>],
        det_emb: &[Option<Vec<f64>>],
        reid_active: bool,
    ) -> Result<(BTreeSet<i64>, BTreeSet<usize>, BTreeSet<usize>), String> {
        let high_conf_thresh = self.high_conf_thresh;
        // 高置信度索引：Python 版同样计算出但从未使用（quirk 照搬），保留语义占位
        let _high: Vec<usize> = (0..dets.len())
            .filter(|&i| det_conf[i] >= high_conf_thresh)
            .collect();
        let low: BTreeSet<usize> = (0..dets.len())
            .filter(|&i| det_conf[i] < high_conf_thresh)
            .collect();

        // Stage 1: Python 传入全部 detections（含低置信度，注释与实现不一致的 quirk，照搬）
        let all: Vec<usize> = (0..dets.len()).collect();
        let (mut matched_tids, mut matched_dis) = self.match_hungarian(
            dets, &all, det_hist, det_emb, reid_active,
            self.iou_threshold, 0.05, None, true,
        )?;

        // Stage 2: 低置信度检测 vs Stage1 未匹配 track
        let unmatched_tids: BTreeSet<i64> = self
            .tracks
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            .difference(&matched_tids)
            .copied()
            .collect();
        if !low.is_empty() && !unmatched_tids.is_empty() {
            let low_sorted: Vec<usize> = low.iter().copied().collect();
            let (stage2_tids, stage2_dis) = self.match_hungarian(
                dets, &low_sorted, det_hist, det_emb, reid_active,
                self.second_stage_iou_min, 0.30, Some(&unmatched_tids), false,
            )?;
            matched_dis.extend(stage2_dis);
            matched_tids.extend(stage2_tids);
        }

        let skip_new: BTreeSet<usize> = low.difference(&matched_dis).copied().collect();
        Ok((matched_tids, matched_dis, skip_new))
    }

    fn try_redetect_for_unmatched(
        &mut self,
        matched_track_ids: &BTreeSet<i64>,
        callback_present: bool,
        cb: &mut dyn RedetectCtx,
    ) -> Result<(), String> {
        let redetect_available =
            self.redetect_enabled && callback_present && self.redetect_max_retries > 0;
        let budget = self.redetect_budget.max(0) as usize;

        // 收集重检测候选；非候选直接 miss+1
        let mut candidates: Vec<i64> = Vec::new();
        for (&tid, trk) in self.tracks.iter_mut() {
            if matched_track_ids.contains(&tid) {
                continue;
            }
            if trk.label == "person"
                && redetect_available
                && trk.redetect_attempts < self.redetect_max_retries
            {
                candidates.push(tid);
            } else {
                trk.misses += 1;
                trk.predicted_only = true;
                trk.track_state = "predicted".to_string();
                if let Some(a) = &mut self.adaptor {
                    a.record_miss();
                }
            }
        }
        // 按 misses 降序（稳定排序 → 并列时保持 tid 升序，与 Python dict 序一致）
        let tracks = &self.tracks;
        candidates.sort_by_key(|&tid| -tracks[&tid].misses);

        for (idx, &tid) in candidates.iter().enumerate() {
            let (trk_box, trk_conf) = {
                let trk = &self.tracks[&tid];
                (trk.box_, trk.confidence)
            };
            if idx < budget {
                let refined = cb.try_redetect(&trk_box, trk_conf);
                let mut success = false;
                if let Some(refined) = refined {
                    let refined_box = refined.box_;
                    let iou_with_pred = box_iou(&trk_box, &refined_box);
                    if iou_with_pred >= self.redetect_iou_threshold {
                        let alpha = self.smooth_alpha;
                        let ocm_dt = if self.ocm_enabled { Some(self.ocm_delta_t) } else { None };
                        let (oru, tcm) = (self.oru_enabled, self.tcm_enabled);
                        let trk = self.tracks.get_mut(&tid).unwrap();
                        update_track(trk, &refined, &refined_box, alpha, &mut self.adaptor, ocm_dt, oru, tcm)?;
                        trk.track_state = "redetected".to_string();
                        trk.predicted_only = false;
                        trk.redetect_attempts = 0;
                        success = true;
                    }
                }
                if !success {
                    // 重检测失败（预算内）：attempts+1 + miss+1
                    if let Some(trk) = self.tracks.get_mut(&tid) {
                        trk.redetect_attempts += 1;
                    }
                } else {
                    continue;
                }
            }
            // 超预算或重检测失败：miss+1，attempts 不变（超预算时）
            let trk = self.tracks.get_mut(&tid).unwrap();
            trk.misses += 1;
            trk.predicted_only = true;
            trk.track_state = "predicted".to_string();
            if let Some(a) = &mut self.adaptor {
                a.record_miss();
            }
        }
        Ok(())
    }
}

/// 对 [cx,cy,w,h] 四分量做线性插值并转回 [x1,y1,x2,y2]
/// （ORU 虚拟观测，对应 Python `_lerp_box_cxcywh`）。
pub fn lerp_box_cxcywh(a: &BBox, b: &BBox, t: f64) -> BBox {
    let (acx, acy, aw, ah) = box_to_cxcywh(a);
    let (bcx, bcy, bw, bh) = box_to_cxcywh(b);
    let cx = acx + (bcx - acx) * t;
    let cy = acy + (bcy - acy) * t;
    let w = aw + (bw - aw) * t;
    let h = ah + (bh - ah) * t;
    [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
}

/// 观测中心动量方向 (Δcy, Δcx)/norm（对应 Python `_speed_direction`；
/// 零位移返回 None，保护性处理双端一致）。
pub fn speed_direction(prev_box: &BBox, cur_box: &BBox) -> Option<(f64, f64)> {
    let dy = (cur_box[1] + cur_box[3]) * 0.5 - (prev_box[1] + prev_box[3]) * 0.5;
    let dx = (cur_box[0] + cur_box[2]) * 0.5 - (prev_box[0] + prev_box[2]) * 0.5;
    let norm = (dy * dy + dx * dx).sqrt();
    if norm < 1e-12 {
        return None;
    }
    Some((dy / norm, dx / norm))
}

/// 匹配成功后的 track 更新（对应 `_update_track`；tracker 级参数
/// smooth_alpha / adaptor 由调用方传入以解决借用冲突）。
fn update_track(
    trk: &mut Track,
    det: &DetInput,
    new_box: &BBox,
    alpha: f64,
    adaptor: &mut Option<FilterAdaptor>,
    ocm_delta_t: Option<i32>,
    oru_enabled: bool,
    tcm_enabled: bool,
) -> Result<(), String> {
    // ORU（§2.1）：丢失态（本帧关联前 misses>0）恢复时，回滚到最近 correct
    // 后的 KF 快照，用 last_observation→当前观测的线性插值虚拟观测逐帧
    // predict+correct 重放丢失的 N 帧，最后 predict+correct 真实观测。
    let corrected_box = if oru_enabled
        && trk.misses > 0
        && trk.kf_snapshot.is_some()
        && trk.last_observation.is_some()
    {
        let n_lost = trk.misses;
        let snap = trk.kf_snapshot.unwrap();
        trk.kalman.x = snap.x;
        trk.kalman.p = snap.p;
        trk.kalman.q = snap.q;
        trk.kalman.r = snap.r;
        let last_obs = trk.last_observation.unwrap();
        for k in 1..=n_lost {
            let vbox = lerp_box_cxcywh(&last_obs, new_box, k as f64 / (n_lost + 1) as f64);
            trk.kalman.predict();
            trk.kalman.correct(&vbox, Some(det.confidence))?; // 虚拟观测 conf=当前观测 conf
        }
        trk.kalman.predict();
        trk.kalman.correct(new_box, Some(det.confidence))?
    } else {
        // 卡尔曼校正（conf 供 NSA-KF 缩放观测噪声）
        trk.kalman.correct(new_box, Some(det.confidence))?
    };

    // AUTO 模式：记录 NIS 和 Cholesky 失败
    if let Some(a) = adaptor {
        let nis = trk.kalman.last_innovation_sq / trk.kalman.last_s_trace.max(1e-10);
        a.record_correct(nis);
        if trk.kalman.cholesky_failed {
            a.record_cholesky_fail();
            trk.kalman.cholesky_failed = false;
        }
    }

    // ORU：保存 correct 后的 KF 全量快照（丢失时以此回滚）
    if oru_enabled {
        trk.kf_snapshot = Some(KfSnapshot {
            x: trk.kalman.x,
            p: trk.kalman.p,
            q: trk.kalman.q,
            r: trk.kalman.r,
        });
    }

    // OCM：仅匹配成功时记录观测并更新速度方向（§2.2）；
    // ORU/OCR 复用 last_observation（OCM 关闭时也要维护）
    if let Some(dt) = ocm_delta_t {
        trk.record_observation(new_box, dt);
    } else if oru_enabled {
        trk.last_observation = Some(*new_box);
    }

    // TCM：置信度状态更新（§2.3）
    if tcm_enabled {
        trk.tcm_update(det.confidence);
    }

    // EMA 平滑 box（用 corrected_box 而非裸 det box）
    for i in 0..4 {
        trk.box_[i] = alpha * corrected_box[i] + (1.0 - alpha) * trk.box_[i];
    }

    // 置信度 EMA
    trk.confidence = 0.7 * det.confidence + 0.3 * trk.confidence;

    // 关键点独立 EMA 平滑或平移缩放
    if det.keypoints.len() == 17 && trk.keypoints.len() == 17 {
        for j in 0..17 {
            trk.keypoints[j].x = alpha * det.keypoints[j].x + (1.0 - alpha) * trk.keypoints[j].x;
            trk.keypoints[j].y = alpha * det.keypoints[j].y + (1.0 - alpha) * trk.keypoints[j].y;
            trk.keypoints[j].conf = 0.7 * det.keypoints[j].conf + 0.3 * trk.keypoints[j].conf;
        }
    } else if det.keypoints.is_empty() && trk.keypoints.len() == 17 {
        // ROI 重检测无关键点：按新 bbox 中心与尺度平移旧关键点
        warp_keypoints_in_place(&mut trk.keypoints, &trk.prev_box, &trk.box_);
    } else {
        trk.keypoints = det.keypoints.clone();
    }

    trk.prev_box = trk.box_;

    // OBB polygon 平滑：新观测有 polygon 则 EMA，否则保持上一帧
    if let Some(new_polygon) = &det.polygon {
        trk.prev_polygon = trk.polygon.clone();
        trk.polygon = Some(smooth_polygon(trk.polygon.as_ref(), new_polygon, alpha));
    }

    trk.hits += 1;
    trk.misses = 0;
    trk.predicted_only = false;
    trk.track_state = "normal".to_string();
    trk.redetect_attempts = 0;
    Ok(())
}

/// 按新 bbox 的中心与尺度平移缩放旧关键点（对应 `_warp_keypoints`，原地版）。
fn warp_keypoints_in_place(kpts: &mut [Keypoint], old_box: &BBox, new_box: &BBox) {
    let old_cx = (old_box[0] + old_box[2]) * 0.5;
    let old_cy = (old_box[1] + old_box[3]) * 0.5;
    let new_cx = (new_box[0] + new_box[2]) * 0.5;
    let new_cy = (new_box[1] + new_box[3]) * 0.5;
    let scale_x = (new_box[2] - new_box[0]) / (old_box[2] - old_box[0]).max(1e-5);
    let scale_y = (new_box[3] - new_box[1]) / (old_box[3] - old_box[1]).max(1e-5);
    for kp in kpts.iter_mut() {
        kp.x = clamp01(new_cx + (kp.x - old_cx) * scale_x);
        kp.y = clamp01(new_cy + (kp.y - old_cy) * scale_y);
    }
}

/// polygon EMA 平滑（对应 `_smooth_polygon`）。
fn smooth_polygon(old: Option<&Vec<Point>>, new: &[Point], alpha: f64) -> Vec<Point> {
    match old {
        Some(o) if o.len() == new.len() => o
            .iter()
            .zip(new.iter())
            .map(|(a, b)| {
                [
                    alpha * b[0] + (1.0 - alpha) * a[0],
                    alpha * b[1] + (1.0 - alpha) * a[1],
                ]
            })
            .collect(),
        _ => new.to_vec(),
    }
}
