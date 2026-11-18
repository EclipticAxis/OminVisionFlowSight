//! AUTO 自适应滤波器选择器（对应 `_FilterAdaptor`）。

use std::collections::HashMap;

const EVAL_INTERVAL: usize = 30; // 每 30 次更新评估一次（约 1 秒 @ 30fps）
const NIS_THRESHOLD: f64 = 8.0; // 平均 NIS 阈值：放宽，减少频繁切换
const COND_THRESHOLD: f64 = 1000.0; // P 条件数阈值：Manifold 作为安全网
const MISS_RATE_THRESHOLD: f64 = 0.5; // 缺失率阈值：放宽
const CHOLESKY_FAIL_THRESHOLD: i32 = 2; // Cholesky 失败次数阈值
const HYSTERESIS: i32 = 2; // 连续 N 次评估一致才切换（防抖）

pub struct FilterAdaptor {
    kernel_sigma: f64,
    current_type: String,
    nis_window: Vec<f64>,
    miss_count: i32,
    total_count: i32,
    cholesky_fails: i32,
    update_count: usize,
    switch_votes: HashMap<String, i32>,
}

impl FilterAdaptor {
    pub fn new(kernel_sigma: f64) -> FilterAdaptor {
        FilterAdaptor {
            kernel_sigma,
            current_type: "ukf".to_string(),
            nis_window: Vec::new(),
            miss_count: 0,
            total_count: 0,
            cholesky_fails: 0,
            update_count: 0,
            switch_votes: HashMap::new(),
        }
    }

    pub fn set_kernel_sigma(&mut self, sigma: f64) {
        self.kernel_sigma = sigma;
    }

    pub fn current_type(&self) -> &str {
        &self.current_type
    }

    /// 记录一次 correct() 的 NIS 值。
    pub fn record_correct(&mut self, nis: f64) {
        self.nis_window.push(nis);
        self.total_count += 1;
        self.update_count += 1;
    }

    /// 记录一次 track miss。
    pub fn record_miss(&mut self) {
        self.miss_count += 1;
        self.total_count += 1;
    }

    /// 记录一次 Cholesky 分解失败。
    pub fn record_cholesky_fail(&mut self) {
        self.cholesky_fails += 1;
    }

    /// 评估是否需要切换滤波器类型（对应 `evaluate`）。
    /// p_cond: 当前活跃 track 的 P 矩阵条件数中位数
    pub fn evaluate(&mut self, p_cond: f64) -> Option<String> {
        if self.update_count < EVAL_INTERVAL {
            return None;
        }
        let start = self.nis_window.len().saturating_sub(EVAL_INTERVAL);
        let recent = &self.nis_window[start..];
        let mean_nis = if recent.is_empty() {
            0.0
        } else {
            recent.iter().sum::<f64>() / recent.len() as f64
        };
        let miss_rate = self.miss_count as f64 / (self.total_count.max(1) as f64);

        // 决策树（优先级从高到低）
        let target = if p_cond > COND_THRESHOLD || self.cholesky_fails > CHOLESKY_FAIL_THRESHOLD {
            "manifold_ukf"
        } else if mean_nis > NIS_THRESHOLD || miss_rate > MISS_RATE_THRESHOLD {
            "mcukf"
        } else {
            "ukf"
        };

        if target != self.current_type {
            let votes = self.switch_votes.entry(target.to_string()).or_insert(0);
            *votes += 1;
            if *votes >= HYSTERESIS {
                self.current_type = target.to_string();
                self.switch_votes.clear();
                self.reset_window();
                return Some(target.to_string());
            }
        } else {
            self.switch_votes.clear();
        }
        self.update_count = 0;
        None
    }

    fn reset_window(&mut self) {
        self.nis_window.clear();
        self.miss_count = 0;
        self.total_count = 0;
        self.cholesky_fails = 0;
        self.update_count = 0;
    }
}
