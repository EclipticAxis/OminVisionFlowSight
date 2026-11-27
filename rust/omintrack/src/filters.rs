//! UKF / MCUKF / ManifoldUKF 移植（对应 ai/tracker.py 的三个滤波器类）。
//!
//! 状态 x = [cx, cy, w, h, vx, vy, vw, vh]，sigma 点 2n+1=17 个。
//! 所有常量、矩阵运算顺序、裁剪/自适应逻辑逐行对齐 Python 实现。
//! NumPy(LAPACK) 与 nalgebra 的浮点结果存在微小差异（LU/Cholesky/EIGH
//! 内部实现不同），属预期，等价性测试用容差吸收。

use nalgebra::{Cholesky, SMatrix, SVector};

use crate::geometry::{box_to_cxcywh, clamp01, BBox};

pub type Vec8 = SVector<f64, 8>;
pub type Mat8 = SMatrix<f64, 8, 8>;
pub type Mat4 = SMatrix<f64, 4, 4>;
pub type Mat84 = SMatrix<f64, 8, 4>;
pub type Sigmas = SMatrix<f64, 8, 17>; // 2n+1 = 17 列
pub const N: usize = 8;
pub const N_SIGMA: usize = 17;

const ALPHA: f64 = 1e-3;
const BETA: f64 = 2.0;
const KAPPA: f64 = 0.0;
const LAMBDA: f64 = ALPHA * ALPHA * (N as f64 + KAPPA) - N as f64; // = 8e-6 - 8
const GAMMA: f64 = 0.002_828_427_124_746_190_3; // (N + LAMBDA).sqrt() = sqrt(8e-6)
const P_MAX_VEC: [f64; N] = [0.05, 0.05, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FilterKind {
    Ukf,
    McUkf,
    ManifoldUkf,
}

impl FilterKind {
    pub fn from_name(name: &str) -> FilterKind {
        match name {
            "mcukf" => FilterKind::McUkf,
            "manifold_ukf" => FilterKind::ManifoldUkf,
            _ => FilterKind::Ukf,
        }
    }
}

pub struct KalmanFilter {
    pub kind: FilterKind,
    pub x: Vec8,
    pub p: Mat8,
    pub q: Mat8,
    pub r: Mat4,
    pub velocity_clip: f64,
    // NSA-KF 观测噪声自适应（docs/tracker-nsa-ocm-hmiou.md §2.1，默认关闭）
    pub nsa_enabled: bool,
    pub nsa_interval: f64,
    pub nsa_min: f64,
    kernel_sigma: f64, // 相关熵核带宽 σ（固定值，无尺寸缩放）
    max_iter: usize,   // 不动点迭代上限（默认 3）
    wm: [f64; N_SIGMA],
    wc: [f64; N_SIGMA],
    // AUTO 自适应监控属性
    pub last_innovation_sq: f64,
    pub last_s_trace: f64,
    pub cholesky_failed: bool,
    miss_streak: i32,
}

impl KalmanFilter {
    pub fn new(kind: FilterKind, box_: &BBox, velocity_clip: f64, kernel_sigma: f64) -> KalmanFilter {
        let (cx, cy, w, h) = box_to_cxcywh(box_);
        let n_lambda = N as f64 + LAMBDA;
        let mut wm = [0.0f64; N_SIGMA];
        let mut wc: [f64; N_SIGMA];
        wm[0] = LAMBDA / n_lambda;
        for item in wm.iter_mut().take(N_SIGMA).skip(1) {
            *item = 1.0 / (2.0 * n_lambda);
        }
        wc = wm;
        wc[0] += 1.0 - ALPHA * ALPHA + BETA;
        let mut x = Vec8::zeros();
        x[0] = cx;
        x[1] = cy;
        x[2] = w;
        x[3] = h;
        KalmanFilter {
            kind,
            x,
            p: Mat8::identity() * 0.02,
            q: Mat8::identity() * 0.0010,
            r: Mat4::identity() * 0.012,
            velocity_clip,
            nsa_enabled: false,
            nsa_interval: 1.0,
            nsa_min: 0.1,
            kernel_sigma: kernel_sigma.max(0.01),
            max_iter: 3,
            wm,
            wc,
            last_innovation_sq: 0.0,
            last_s_trace: 1.0,
            cholesky_failed: false,
            miss_streak: 0,
        }
    }

    /// 按目标尺度构造观测噪声 R（对应 `_make_r`）。
    fn make_r(&mut self, w: f64, h: f64) {
        let base_pos_var = 0.010;
        let base_size_var = 0.015;
        let scale_factor = (0.05 / w.max(h).max(1e-5)).max(1.0);
        let mut r = Mat4::zeros();
        r[(0, 0)] = base_pos_var * scale_factor;
        r[(1, 1)] = base_pos_var * scale_factor;
        r[(2, 2)] = base_size_var;
        r[(3, 3)] = base_size_var;
        self.r = r;
    }

    /// NSA-KF 乘子（对应 `_nsa_multiplier`）：关闭时恒 1.0；
    /// nsa_min 截断为保护性改动（防 conf→1 时 R 退化），非原文行为。
    fn nsa_multiplier(&self, conf: f64) -> f64 {
        if !self.nsa_enabled {
            return 1.0;
        }
        ((1.0 - conf) * self.nsa_interval).clamp(self.nsa_min, 1.0)
    }

    /// NIS 自适应 Q（对应 `_adapt_q`）。
    fn adapt_q(&mut self, nis: f64) {
        if nis > 2.0 {
            for i in 0..4 {
                for j in 0..4 {
                    self.q[(i, j)] = (self.q[(i, j)] * 1.05).min(0.05);
                }
            }
            for i in 4..8 {
                for j in 4..8 {
                    self.q[(i, j)] = (self.q[(i, j)] * 1.10).min(0.02);
                }
            }
            self.miss_streak = (self.miss_streak + 1).min(10);
        } else if nis < 0.3 {
            for i in 0..4 {
                for j in 0..4 {
                    self.q[(i, j)] = (self.q[(i, j)] * 0.98).max(1e-5);
                }
            }
            for i in 4..8 {
                for j in 4..8 {
                    self.q[(i, j)] = (self.q[(i, j)] * 0.95).max(1e-6);
                }
            }
            self.miss_streak = (self.miss_streak - 1).max(0);
        } else {
            self.miss_streak = (self.miss_streak - 1).max(0);
        }
    }

    /// 限制速度/尺度状态（对应 `_clip_state`）。
    fn clip_state(&mut self) {
        for i in 4..8 {
            self.x[i] = self.x[i].clamp(-self.velocity_clip, self.velocity_clip);
        }
        self.x[2] = self.x[2].max(1e-5).min(1.0);
        self.x[3] = self.x[3].max(1e-5).min(1.0);
    }

    /// 按分量上限裁剪协方差（对应 `_clip_p_variance`）。
    fn clip_p_variance(&mut self) {
        for i in 0..N {
            let max_var = P_MAX_VEC[i];
            if self.p[(i, i)] > max_var {
                let scale = (max_var / self.p[(i, i)]).sqrt();
                for j in 0..N {
                    self.p[(i, j)] *= scale;
                    self.p[(j, i)] *= scale;
                }
            }
        }
        for i in 0..N {
            if self.p[(i, i)] < 1e-8 {
                self.p[(i, i)] = 1e-8;
            }
        }
    }

    /// 对称化
    fn symmetrize(p: &mut Mat8) {
        *p = (*p + p.transpose()) * 0.5;
    }

    /// NumPy np.linalg.cholesky 等价：P = L L^T，非正定返回 None。
    fn cholesky(p: &Mat8) -> Option<Mat8> {
        Cholesky::new(*p).map(|c| c.l())
    }

    /// 生成 2n+1=17 个 sigma 点（UKF：Cholesky；Manifold：EIGH 矩阵平方根）。
    fn sigma_points(&mut self) -> Sigmas {
        match self.kind {
            FilterKind::ManifoldUkf => self.sigma_points_manifold(),
            _ => self.sigma_points_cholesky(),
        }
    }

    fn sigma_points_cholesky(&mut self) -> Sigmas {
        self.clip_p_variance();
        let l = match Self::cholesky(&self.p) {
            Some(l) => l,
            None => {
                // P 非正定：对角加载后重试，仍失败则重置
                self.cholesky_failed = true;
                Self::symmetrize(&mut self.p);
                let min_diag = (0..N).map(|i| self.p[(i, i)]).fold(f64::INFINITY, f64::min);
                let diag_load = 1e-6f64.max(min_diag.abs() * 1.1);
                for i in 0..N {
                    self.p[(i, i)] += diag_load;
                }
                match Self::cholesky(&self.p) {
                    Some(l) => l,
                    None => {
                        self.p = Mat8::identity() * 0.02;
                        Self::cholesky(&self.p).expect("identity is positive definite")
                    }
                }
            }
        };
        let mut sigmas = Sigmas::zeros();
        sigmas.set_column(0, &self.x);
        for i in 0..N {
            let offset = l.column(i) * GAMMA;
            sigmas.set_column(i + 1, &(self.x + offset));
            sigmas.set_column(N + i + 1, &(self.x - offset));
        }
        sigmas
    }

    fn sigma_points_manifold(&mut self) -> Sigmas {
        // P^{1/2} = V diag(sqrt(lambda)) V^T，特征值裁剪到 >= 1e-8
        let mut p_sym = self.p;
        Self::symmetrize(&mut p_sym);
        let eig = p_sym.symmetric_eigen();
        let mut vals = eig.eigenvalues.map(|e| e.max(1e-8));
        vals = vals.map(|e| e.sqrt());
        let p_sqrt = eig.eigenvectors * Mat8::from_diagonal(&vals) * eig.eigenvectors.transpose();
        let mut sigmas = Sigmas::zeros();
        sigmas.set_column(0, &self.x);
        for i in 0..N {
            let offset = p_sqrt.column(i) * GAMMA;
            sigmas.set_column(i + 1, &(self.x + offset));
            sigmas.set_column(N + i + 1, &(self.x - offset));
        }
        sigmas
    }

    /// SPD 投影：特征值裁剪到 [1e-8, 0.05]（对应 Manifold `_project_spd`）。
    fn project_spd(&mut self) {
        let mut p_sym = self.p;
        Self::symmetrize(&mut p_sym);
        let eig = p_sym.symmetric_eigen();
        let vals = eig.eigenvalues.map(|e| e.max(1e-8).min(0.05));
        self.p = eig.eigenvectors * Mat8::from_diagonal(&vals) * eig.eigenvectors.transpose();
        self.clip_p_variance();
    }

    /// 无迹变换预测（对应 `predict`）。
    pub fn predict(&mut self) -> BBox {
        let sigmas = self.sigma_points();
        let mut sigmas_pred = sigmas;
        for j in 0..N_SIGMA {
            for i in 0..4 {
                sigmas_pred[(i, j)] += sigmas[(4 + i, j)];
            }
        }
        // 预测均值
        let mut x = Vec8::zeros();
        for j in 0..N_SIGMA {
            x += sigmas_pred.column(j) * self.wm[j];
        }
        self.x = x;
        self.clip_state();
        // 预测协方差
        let mut p = Mat8::zeros();
        for j in 0..N_SIGMA {
            let d = sigmas_pred.column(j) - self.x;
            p += self.wc[j] * (d * d.transpose());
        }
        self.p = p + self.q;
        match self.kind {
            FilterKind::ManifoldUkf => self.project_spd(),
            _ => {
                Self::symmetrize(&mut self.p);
                self.clip_p_variance();
            }
        }
        self.to_box()
    }

    /// UKF / ManifoldUKF 更新（对应两者的 `correct`）。
    fn correct_base(&mut self, box_: &BBox, conf: f64) -> Result<BBox, String> {
        let (cx, cy, w, h) = box_to_cxcywh(box_);
        let z = Vec8::from_row_slice(&[cx, cy, w, h, 0.0, 0.0, 0.0, 0.0]); // 仅用前 4 维
        self.make_r(w, h);

        let sigmas = self.sigma_points();
        let mut z_pred = SVector::<f64, 4>::zeros();
        for j in 0..N_SIGMA {
            for i in 0..4 {
                z_pred[i] += self.wm[j] * sigmas[(i, j)];
            }
        }
        // S = (dz * wc) @ dz^T + r
        let mut s = Mat4::zeros();
        for j in 0..N_SIGMA {
            let mut dz = SVector::<f64, 4>::zeros();
            for i in 0..4 {
                dz[i] = sigmas[(i, j)] - z_pred[i];
            }
            s += self.wc[j] * (dz * dz.transpose());
        }
        s += self.r;

        // 创新息统计
        let mut innov_sq = 0.0f64;
        for i in 0..4 {
            let d = z[i] - z_pred[i];
            innov_sq += d * d;
        }
        self.last_innovation_sq = innov_sq;
        self.last_s_trace = s.trace();
        let nis = self.last_innovation_sq / self.last_s_trace.max(1e-10);

        if self.kind == FilterKind::Ukf {
            self.adapt_q(nis);
        }

        // NSA-KF：仅影响 K 与状态/协方差更新的有效 S（NIS/AUTO 监控已用未缩放 R）；
        // R 为对角，仅对角元需要调整
        let s_trace_nominal = s.trace();
        let nsa_mult = self.nsa_multiplier(conf);
        if nsa_mult != 1.0 {
            for i in 0..4 {
                s[(i, i)] += (nsa_mult - 1.0) * self.r[(i, i)];
            }
        }

        // Pxz = (dx * wc) @ dz^T
        let mut pxz = Mat84::zeros();
        for j in 0..N_SIGMA {
            let mut dz = SVector::<f64, 4>::zeros();
            for i in 0..4 {
                dz[i] = sigmas[(i, j)] - z_pred[i];
            }
            let dx = sigmas.column(j) - self.x;
            pxz += self.wc[j] * (dx * dz.transpose());
        }
        let s_inv = s
            .try_inverse()
            .ok_or_else(|| "UKF correct: innovation covariance S is singular".to_string())?;
        let k = pxz * s_inv;
        let mut x = self.x;
        for i in 0..8 {
            let mut upd = 0.0;
            for jj in 0..4 {
                upd += k[(i, jj)] * (z[jj] - z_pred[jj]);
            }
            x[i] += upd;
        }
        self.x = x;
        self.clip_state();
        // p = p - K S K^T
        self.p -= k * s * k.transpose();
        match self.kind {
            FilterKind::ManifoldUkf => self.project_spd(),
            _ => {
                Self::symmetrize(&mut self.p);
                self.clip_p_variance();
            }
        }
        if self.kind == FilterKind::ManifoldUkf {
            // ManifoldUKF 在末尾记录创新息统计并自适应 Q（用未 NSA 缩放的 S）
            self.last_innovation_sq = innov_sq;
            self.last_s_trace = s_trace_nominal;
            let nis2 = self.last_innovation_sq / self.last_s_trace.max(1e-10);
            self.adapt_q(nis2);
        }
        Ok(self.to_box())
    }

    /// MCUKF 更新：Wm/Wc 标准无迹变换 + 相关熵 R_eff 不动点迭代 + Joseph 协方差。
    ///
    /// 冻结契约（逐行对应 Python `_DetectionMCUKF.correct`）：
    /// 1. 标准 Wm/Wc 无迹变换得 z_pred / Pzz / Pxz，S0 = Pzz + R 仅用于原始 NIS；
    /// 2. 最多 max_iter 轮不动点迭代：residual = z - (z_pred + H (x_iter - x_pred))，
    ///    y 按名义 R 对角 std 白化，c = clip(exp(-y²/(2σ²)), 1e-4, 1)，
    ///    R_eff = R C⁻¹（对角），K = Pxz (Pzz + R_eff)⁻¹，x_new = x_pred + K (z - z_pred)；
    /// 3. 位移 < 1e-6 接受并停止；达到上限使用最后鲁棒迭代的 K / R_eff / x；
    /// 4. 最终 Joseph 用同一 K / R_eff，NIS 用原始 r0 / S0；核带宽 σ 无尺寸缩放。
    fn correct_mcukf(&mut self, box_: &BBox, conf: f64) -> Result<BBox, String> {
        let (cx, cy, w, h) = box_to_cxcywh(box_);
        let z = Vec8::from_row_slice(&[cx, cy, w, h, 0.0, 0.0, 0.0, 0.0]);
        self.make_r(w, h);
        // NSA-KF：先得到 NSA 缩放的名义 R，IRLS 白化在其基础上进行（叠加顺序 §2.1）；
        // NIS 统计仍用未缩放的 self.r（S0 = Pzz + R）
        let nsa_mult = self.nsa_multiplier(conf);
        let sigma = self.kernel_sigma;
        let two_sigma_sq = 2.0 * sigma * sigma;

        let sigmas = self.sigma_points();
        let x_pred = self.x;
        let p_pred = self.p;

        // 标准 Wm/Wc 无迹变换：观测均值 z_pred、观测协方差 Pzz、交叉协方差 Pxz
        let mut z_pred = SVector::<f64, 4>::zeros();
        for j in 0..N_SIGMA {
            for i in 0..4 {
                z_pred[i] += self.wm[j] * sigmas[(i, j)];
            }
        }
        let mut pzz = Mat4::zeros();
        let mut pxz = Mat84::zeros();
        for j in 0..N_SIGMA {
            let mut dz = SVector::<f64, 4>::zeros();
            for i in 0..4 {
                dz[i] = sigmas[(i, j)] - z_pred[i];
            }
            pzz += self.wc[j] * (dz * dz.transpose());
            let dx = sigmas.column(j) - x_pred;
            pxz += self.wc[j] * (dx * dz.transpose());
        }
        // 原始创新协方差 S0 = Pzz + R，仅用于 NIS 统计
        let s0 = pzz + self.r;

        // 相关熵 R_eff 不动点迭代（最多 max_iter 轮）
        let mut k = Mat84::zeros();
        let mut r_eff = Mat4::zeros();
        let mut x_iter = x_pred;
        for _ in 0..self.max_iter {
            // residual = z - (z_pred + H (x_iter - x_pred))，H 取状态前 4 维
            let mut residual = SVector::<f64, 4>::zeros();
            for i in 0..4 {
                residual[i] = z[i] - (z_pred[i] + x_iter[i] - x_pred[i]);
            }
            // y 按名义 R 对角 std 白化，C = diag(c)，R_eff = R C⁻¹（对角）
            r_eff = Mat4::zeros();
            for i in 0..4 {
                let r_nsa_ii = self.r[(i, i)] * nsa_mult; // NSA 缩放后的名义 R 对角元
                let y = residual[i] / r_nsa_ii.sqrt();
                let c = (-y * y / two_sigma_sq).exp().clamp(1e-4, 1.0);
                r_eff[(i, i)] = r_nsa_ii / c;
            }
            let s_r = pzz + r_eff;
            let s_inv = s_r
                .try_inverse()
                .ok_or_else(|| "UKF correct: innovation covariance S is singular".to_string())?;
            k = pxz * s_inv;
            let x_old = x_iter;
            let mut x_new = x_pred;
            for i in 0..8 {
                for j in 0..4 {
                    x_new[i] += k[(i, j)] * (z[j] - z_pred[j]);
                }
            }
            x_iter = x_new;
            // 位移 < 1e-6 接受并停止；达到上限自然使用最后鲁棒迭代
            let mut norm_sq = 0.0;
            for i in 0..8 {
                let d = x_new[i] - x_old[i];
                norm_sq += d * d;
            }
            if norm_sq.sqrt() < 1e-6 {
                break;
            }
        }
        self.x = x_iter;

        // 创新息统计：原始 r0 = z - z_pred，S0 = Pzz + R
        let mut innov_sq = 0.0;
        for i in 0..4 {
            let d = z[i] - z_pred[i];
            innov_sq += d * d;
        }
        self.last_innovation_sq = innov_sq;
        self.last_s_trace = s0.trace();
        let nis = self.last_innovation_sq / self.last_s_trace.max(1e-10);
        self.adapt_q(nis);

        // Joseph 形式协方差更新（同一 K / R_eff）
        let mut h = Mat84::zeros();
        for i in 0..4 {
            h[(i, i)] = 1.0;
        }
        let i_kh = Mat8::identity() - k * h.transpose();
        self.p = i_kh * p_pred * i_kh.transpose() + k * r_eff * k.transpose();
        Self::symmetrize(&mut self.p);
        self.clip_state();
        self.clip_p_variance();
        Ok(self.to_box())
    }

    pub fn correct(&mut self, box_: &BBox, conf: f64) -> Result<BBox, String> {
        if self.kind == FilterKind::McUkf {
            self.correct_mcukf(box_, conf)
        } else {
            self.correct_base(box_, conf)
        }
    }

    /// 状态 -> 归一化 box（对应 `to_box`）。
    pub fn to_box(&self) -> BBox {
        let cx = self.x[0];
        let cy = self.x[1];
        let w = self.x[2].max(1e-5);
        let h = self.x[3].max(1e-5);
        [
            clamp01(cx - w * 0.5),
            clamp01(cy - h * 0.5),
            clamp01(cx + w * 0.5),
            clamp01(cy + h * 0.5),
        ]
    }
}

/// P 矩阵条件数 = s_max / s_min（对应 np.linalg.cond，SVD）。
pub fn p_condition_number(p: &Mat8) -> f64 {
    let svd = nalgebra::SVD::new(*p, false, false);
    let s = &svd.singular_values;
    let smax = s.max();
    let smin = s.min();
    if smin <= 0.0 {
        f64::INFINITY
    } else {
        smax / smin
    }
}



