# =============================================================================
# VisionDataPlatform - 轻量 IoU-Greedy 跟踪器 + EMA 平滑
# =============================================================================
# 不依赖外部跟踪包，纯 NumPy，供 InferWorker 内部使用。
# 设计要点：
#   - 同标签（person）间用 IoU 做贪婪匹配；
#   - 对 bbox 和 17 个关键点分别做 EMA 平滑；
#   - 未命中 N 帧后删除，防止鬼影；
#   - 输出保持与现有 detection schema 一致，仅多一个 track_id。
# =============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import cv2

from ai.detection import Detection, polygon_iou, polygon_center

if TYPE_CHECKING:
    # VisionCore 未来统一模型入口 —— 当前仅用于类型检查，不影响运行时。
    # DetectionTracker 的运行逻辑保持不变；这些类型为未来迁移做准备。
    from visioncore.core import Detection as CoreDetection
    from visioncore.core import Track as CoreTrack

def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    """计算两个归一化 box 的 IoU。输入均为 [x1, y1, x2, y2]。"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def _shape_iou(
    trk_box: np.ndarray,
    trk_polygon: list[tuple[float, float]] | None,
    det_box: np.ndarray,
    det_polygon: list[tuple[float, float]] | None,
) -> float:
    """计算 track 与 detection 的 IoU：有 polygon 则使用 OBB IoU，否则回退到 bbox IoU。"""
    if trk_polygon is not None and det_polygon is not None and len(trk_polygon) == 4 and len(det_polygon) == 4:
        return polygon_iou(trk_polygon, det_polygon)
    return _box_iou(trk_box, det_box)


def _shape_center(
    box: np.ndarray,
    polygon: list[tuple[float, float]] | None,
) -> tuple[float, float]:
    """优先使用 polygon 中心，否则使用 bbox 中心。"""
    if polygon is not None and len(polygon) >= 3:
        return polygon_center(polygon)
    return (float(box[0]) + float(box[2])) * 0.5, (float(box[1]) + float(box[3])) * 0.5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _box_to_cxcywh(box: np.ndarray) -> tuple[float, float, float, float]:
    """[x1, y1, x2, y2] → (cx, cy, w, h)。"""
    w = max(1e-5, float(box[2] - box[0]))
    h = max(1e-5, float(box[3] - box[1]))
    cx = float(box[0]) + w * 0.5
    cy = float(box[1]) + h * 0.5
    return cx, cy, w, h


def _compute_hsv_hist(roi: np.ndarray) -> np.ndarray | None:
    """计算增强 HSV 直方图：H[32]×S[16] + V[8] = 520 维，L2 归一化。"""
    if roi is None or roi.size == 0:
        return None
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    except cv2.error:
        return None
    # H-S 联合直方图
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [32, 16], [0, 180, 0, 256])
    # V 通道直方图
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    # 拼接 + L2 归一化
    hist = np.concatenate([hist_hs.flatten(), hist_v.flatten()])
    cv2.normalize(hist, hist, norm_type=cv2.NORM_L2)
    return hist.flatten()


class _DetectionUKF:
    """无迹卡尔曼滤波（UKF），8 维状态 [cx, cy, w, h, vx, vy, vw, vh]。

    用 sigma 点采样 + 无迹变换替代线性矩阵乘法传播协方差。
    纯 NumPy 实现，不依赖 filterpy/scipy。
    对线性运动模型，结果与线性 KF 等价但协方差传播更鲁棒，
    且未来可扩展为非线性运动模型（如 CTRV）无需重新设计滤波器。
    """

    # === UKF 参数（类常量，预计算） ===
    _N = 8                                          # 状态维度
    _ALPHA = 1e-3                                   # sigma 点扩散系数：保守值，降低尾部权重
    _BETA = 2.0                                     # 高斯先验最优
    _KAPPA = 0.0                                    # 次要缩放参数
    _LAMBDA = _ALPHA ** 2 * (_N + _KAPPA) - _N      # 复合缩放参数
    _GAMMA = (_N + _LAMBDA) ** 0.5                  # sigma 点半径
    # 协方差上界：按状态分量分别限制，避免位置/尺度/速度相互污染
    _P_MAX_POSITION = 0.05
    _P_MAX_SIZE = 0.02
    _P_MAX_VELOCITY = 0.01
    _P_MAX_VEC = np.array(
        [_P_MAX_POSITION, _P_MAX_POSITION, _P_MAX_SIZE, _P_MAX_SIZE,
         _P_MAX_VELOCITY, _P_MAX_VELOCITY, _P_MAX_VELOCITY, _P_MAX_VELOCITY],
        dtype=np.float64,
    )

    def __init__(self, box: np.ndarray, velocity_clip: float = 0.30):
        cx, cy, w, h = _box_to_cxcywh(box)
        self.x = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float64)
        self.p = np.eye(8, dtype=np.float64) * 0.02
        self.q = np.eye(8, dtype=np.float64) * 0.0010
        self.r = np.eye(4, dtype=np.float64) * 0.012
        self._velocity_clip = float(velocity_clip)             # 速度限幅，可配置
        # 预计算 sigma 点权重（2n+1 = 17 个点）
        n_lambda = self._N + self._LAMBDA
        self._wm = np.full(2 * self._N + 1, 1.0 / (2.0 * n_lambda), dtype=np.float64)
        self._wm[0] = self._LAMBDA / n_lambda
        self._wc = self._wm.copy()
        self._wc[0] += (1.0 - self._ALPHA ** 2 + self._BETA)
        # AUTO 自适应监控属性
        self.last_innovation_sq = 0.0                # ||z - z_pred||²
        self.last_s_trace = 1.0                      # trace(S)
        self.cholesky_failed = False                 # Cholesky 分解失败标志
        self._miss_streak = 0                        # 连续 miss 次数，用于自适应 Q

    def _make_r(self, w: float, h: float) -> np.ndarray:
        """按目标尺度构造观测噪声 R：小目标位置噪声更大。"""
        base_pos_var = 0.010
        base_size_var = 0.015
        scale_factor = max(1.0, 0.05 / max(w, h, 1e-5))
        return np.diag(np.array([
            base_pos_var * scale_factor,
            base_pos_var * scale_factor,
            base_size_var,
            base_size_var,
        ], dtype=np.float64))

    def _adapt_q(self, nis: float) -> None:
        """根据 NIS 自适应调整过程噪声 Q。"""
        if nis > 2.0:
            # 运动模型不匹配或离群观测，增大过程噪声
            self.q[:4, :4] = np.minimum(self.q[:4, :4] * 1.05, 0.05)
            self.q[4:, 4:] = np.minimum(self.q[4:, 4:] * 1.10, 0.02)
            self._miss_streak = min(self._miss_streak + 1, 10)
        elif nis < 0.3:
            # 运动稳定，降低过程噪声
            self.q[:4, :4] = np.maximum(self.q[:4, :4] * 0.98, 1e-5)
            self.q[4:, 4:] = np.maximum(self.q[4:, 4:] * 0.95, 1e-6)
            self._miss_streak = max(self._miss_streak - 1, 0)
        else:
            self._miss_streak = max(self._miss_streak - 1, 0)

    def _clip_state(self) -> None:
        """限制速度状态，防止初始速度估计爆炸。"""
        self.x[4:8] = np.clip(self.x[4:8], -self._velocity_clip, self._velocity_clip)
        # 保证尺度为正且不过大
        self.x[2, 0] = max(1e-5, min(1.0, self.x[2, 0]))
        self.x[3, 0] = max(1e-5, min(1.0, self.x[3, 0]))

    def _clip_p_variance(self) -> None:
        """按状态分量裁剪协方差对角线，保持正定结构。"""
        for i in range(self._N):
            max_var = self._P_MAX_VEC[i]
            if self.p[i, i] > max_var:
                scale = (max_var / self.p[i, i]) ** 0.5
                self.p[i, :] *= scale
                self.p[:, i] *= scale
        # 保证非负对角线
        diag = np.maximum(np.diag(self.p), 1e-8)
        np.fill_diagonal(self.p, diag)

    def _sigma_points(self) -> np.ndarray:
        """生成 2n+1=17 个 sigma 点，返回 (8, 17) 矩阵。

        X⁰ = x（均值）
        Xⁱ = x + γ·L[:, i]   (i=1..n，正方向)
        Xⁱ = x - γ·L[:, i-n] (i=n+1..2n，负方向)
        其中 L = Cholesky(P)。
        """
        self._clip_p_variance()
        try:
            L = np.linalg.cholesky(self.p)
        except np.linalg.LinAlgError:
            # P 非正定：施加对角加载后重试，仍失败则重置
            self.cholesky_failed = True
            self.p = (self.p + self.p.T) * 0.5
            diag_load = max(1e-6, abs(np.min(np.diag(self.p))) * 1.1)
            np.fill_diagonal(self.p, np.diag(self.p) + diag_load)
            try:
                L = np.linalg.cholesky(self.p)
            except np.linalg.LinAlgError:
                self.p = np.eye(self._N, dtype=np.float64) * 0.02
                L = np.linalg.cholesky(self.p)
        sigmas = np.zeros((self._N, 2 * self._N + 1), dtype=np.float64)
        sigmas[:, 0:1] = self.x
        for i in range(self._N):
            offset = self._GAMMA * L[:, i:i + 1]
            sigmas[:, i + 1:i + 2] = self.x + offset
            sigmas[:, self._N + i + 1:self._N + i + 2] = self.x - offset
        return sigmas

    def predict(self) -> np.ndarray:
        """UKF 预测：sigma 点通过运动模型传播 + 无迹变换。

        运动模型 f(x) = F·x（匀速：位置 += 速度 * dt, dt=1）
        用索引操作替代矩阵乘法：sigmas_pred[0:4] += sigmas[4:8]
        """
        sigmas = self._sigma_points()
        # 运动模型传播：cx+=vx, cy+=vy, w+=vw, h+=vh
        sigmas_pred = sigmas.copy()
        sigmas_pred[0:4, :] += sigmas[4:8, :]
        # 无迹变换：预测均值
        self.x = (self._wm * sigmas_pred).sum(axis=1, keepdims=True)
        # 限制状态，防止速度/尺度爆炸
        self._clip_state()
        # 无迹变换：预测协方差
        d = sigmas_pred - self.x                       # (8, 17) 偏差
        self.p = (d * self._wc) @ d.T + self.q
        # 对称化 + 上限裁剪，防止数值漂移
        self.p = (self.p + self.p.T) * 0.5
        self._clip_p_variance()
        return self.to_box()

    def correct(self, box: np.ndarray) -> np.ndarray:
        """UKF 更新：sigma 点通过观测模型 + 无迹变换计算卡尔曼增益。

        观测模型 h(x) = H·x（直接取前 4 维 [cx, cy, w, h]）
        """
        cx, cy, w, h = _box_to_cxcywh(box)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)

        # 按目标尺度自适应观测噪声 R
        self.r = self._make_r(w, h)

        sigmas = self._sigma_points()
        # 观测模型传播：取前 4 维
        z_sigmas = sigmas[0:4, :]                      # (4, 17)
        # 预测观测均值
        z_pred = (self._wm * z_sigmas).sum(axis=1, keepdims=True)  # (4, 1)
        # 观测协方差 S
        dz = z_sigmas - z_pred                         # (4, 17)
        S = (dz * self._wc) @ dz.T + self.r            # (4, 4)
        # 记录创新息统计（供 AUTO 适配器使用）
        innovation = z - z_pred
        self.last_innovation_sq = float(np.sum(innovation ** 2))
        self.last_s_trace = float(np.trace(S))
        nis = self.last_innovation_sq / max(self.last_s_trace, 1e-10)
        # 自适应 Q
        self._adapt_q(nis)
        # 状态-观测交叉协方差 Pxz
        dx = sigmas - self.x                           # (8, 17)
        Pxz = (dx * self._wc) @ dz.T                   # (8, 4)
        # 卡尔曼增益
        K = Pxz @ np.linalg.inv(S)                     # (8, 4)
        # 状态更新
        self.x = self.x + K @ (z - z_pred)
        # 限制状态
        self._clip_state()
        # 协方差更新
        self.p = self.p - K @ S @ K.T
        # 对称化 + 上限裁剪
        self.p = (self.p + self.p.T) * 0.5
        self._clip_p_variance()
        return self.to_box()

    def to_box(self) -> np.ndarray:
        cx = float(self.x[0, 0])
        cy = float(self.x[1, 0])
        w = max(1e-5, float(self.x[2, 0]))
        h = max(1e-5, float(self.x[3, 0]))
        return np.array([
            _clamp01(cx - w * 0.5),
            _clamp01(cy - h * 0.5),
            _clamp01(cx + w * 0.5),
            _clamp01(cy + h * 0.5),
        ], dtype=np.float64)


class _DetectionMCUKF(_DetectionUKF):
    """最大相关熵 UKF（Maximum Correntropy UKF）。

    在标准 UKF 基础上，用高斯核函数 exp(-e²/2σ²) 加权每个 sigma 点的观测残差，
    自动降低大误差（离群检测框）的影响。对脉冲噪声、遮挡跳变、YOLO 误检鲁棒。

    与标准 UKF 的区别仅在 correct()：
    - 残差通过高斯核加权 → 离群观测自动降权
    - 不动点迭代（默认 3 次）收敛状态估计
    - Joseph 形式协方差更新保证数值稳定性
    """

    def __init__(self, box: np.ndarray, kernel_sigma: float = 0.4, max_iter: int = 3,
                 velocity_clip: float = 0.30):
        super().__init__(box, velocity_clip=velocity_clip)
        self._kernel_sigma = float(max(0.01, kernel_sigma))
        self._max_iter = int(max(1, max_iter))

    def correct(self, box: np.ndarray) -> np.ndarray:
        """MCUKF 更新：相关熵加权 + 不动点迭代 + Joseph 形式协方差。"""
        cx, cy, w, h = _box_to_cxcywh(box)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)

        # 按目标尺度自适应观测噪声 R 与核带宽 σ
        self.r = self._make_r(w, h)
        sigma = max(0.05, min(self._kernel_sigma, min(w, h) * 0.5))
        two_sigma_sq = 2.0 * sigma ** 2

        sigmas = self._sigma_points()
        z_sigmas = sigmas[0:4, :]                        # (4, 17) 观测传播

        x_pred = self.x.copy()                           # 保存预测状态
        p_pred = self.p.copy()                           # 保存预测协方差

        K = None
        z_pred = None
        for iteration in range(self._max_iter):
            # 每个 sigma 点的观测残差
            residuals = z - z_sigmas                     # (4, 17)
            residual_sq = np.sum(residuals ** 2, axis=0) # (17,) ||e_i||²

            # 高斯核加权
            kernel_w = np.exp(-residual_sq / two_sigma_sq)
            kernel_w[0] = max(kernel_w[0], 1e-10)        # 中心点保底
            total = kernel_w.sum()
            if total < 1e-10:
                kernel_w = np.full_like(kernel_w, 1.0 / len(kernel_w))
            else:
                kernel_w = kernel_w / total

            # 加权观测均值
            z_pred = (kernel_w * z_sigmas).sum(axis=1, keepdims=True)  # (4, 1)

            # 加权观测协方差 S
            dz = z_sigmas - z_pred                       # (4, 17)
            S = (dz * kernel_w) @ dz.T + self.r          # (4, 4)

            # 加权交叉协方差 Pxz
            dx = sigmas - self.x                         # (8, 17)
            Pxz = (dx * kernel_w) @ dz.T                 # (8, 4)

            # Kalman 增益
            K = Pxz @ np.linalg.inv(S)                   # (8, 4)

            # 状态更新
            self.x = x_pred + K @ (z - z_pred)

            # 收敛检查
            if iteration > 0 and np.linalg.norm(self.x - x_prev) < 1e-6:
                break
            x_prev = self.x.copy()

        if K is None or z_pred is None:
            K = np.zeros((8, 4), dtype=np.float64)
            z_pred = (self._wm * z_sigmas).sum(axis=1, keepdims=True)

        # 记录创新息统计（供 AUTO 适配器使用）
        innovation = z - z_pred
        self.last_innovation_sq = float(np.sum(innovation ** 2))
        self.last_s_trace = float(np.trace(S)) if 'S' in locals() else 1.0
        nis = self.last_innovation_sq / max(self.last_s_trace, 1e-10)
        self._adapt_q(nis)

        # Joseph 形式协方差更新（数值稳定）
        H = np.zeros((4, 8), dtype=np.float64)
        H[:4, :4] = np.eye(4)
        I_KH = np.eye(8, dtype=np.float64) - K @ H
        self.p = I_KH @ p_pred @ I_KH.T + K @ self.r @ K.T

        # 对称化 + 按分量上限裁剪
        self.p = (self.p + self.p.T) * 0.5
        self._clip_state()
        self._clip_p_variance()
        return self.to_box()


class _DetectionManifoldUKF(_DetectionUKF):
    """流形 UKF（Manifold UKF）。

    协方差矩阵 P 始终保持在对称正定（SPD）流形上：
    - sigma 点用特征分解 P^{1/2} = V·diag(√λ)·Vᵀ 替代 Cholesky 分解
    - 每次 predict/correct 后做 SPD 投影（特征值裁剪到 [1e-8, P_MAX]）
    - 更鲁棒的数值稳定性，P 永远不会失去正定性

    适合长时间跳帧预测、高维状态等容易导致 P 退化的场景。
    """

    def _sigma_points(self) -> np.ndarray:
        """用特征分解替代 Cholesky 生成 sigma 点，保证 P 在 SPD 流形上。"""
        # 对称化
        p_sym = (self.p + self.p.T) * 0.5
        # 特征分解（eigh 对对称矩阵更稳定）
        eigenvalues, eigenvectors = np.linalg.eigh(p_sym)
        # 裁剪负特征值
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        # 矩阵平方根 P^{1/2} = V · diag(√λ) · Vᵀ
        p_sqrt = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T

        sigmas = np.zeros((self._N, 2 * self._N + 1), dtype=np.float64)
        sigmas[:, 0:1] = self.x
        for i in range(self._N):
            offset = self._GAMMA * p_sqrt[:, i:i + 1]
            sigmas[:, i + 1:i + 2] = self.x + offset
            sigmas[:, self._N + i + 1:self._N + i + 2] = self.x - offset
        return sigmas

    def _project_spd(self) -> None:
        """SPD 投影：特征值裁剪到 [1e-8, 0.05]，再按分量限制方差。"""
        p_sym = (self.p + self.p.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(p_sym)
        eigenvalues = np.clip(eigenvalues, 1e-8, 0.05)
        self.p = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        self._clip_p_variance()

    def predict(self) -> np.ndarray:
        """与 UKF 相同的 sigma 传播，但用 SPD 投影替代简单裁剪。"""
        sigmas = self._sigma_points()
        sigmas_pred = sigmas.copy()
        sigmas_pred[0:4, :] += sigmas[4:8, :]
        self.x = (self._wm * sigmas_pred).sum(axis=1, keepdims=True)
        self._clip_state()
        d = sigmas_pred - self.x
        self.p = (d * self._wc) @ d.T + self.q
        self._project_spd()
        return self.to_box()

    def correct(self, box: np.ndarray) -> np.ndarray:
        """与 UKF 相同的更新，但用 SPD 投影替代简单对称化。"""
        cx, cy, w, h = _box_to_cxcywh(box)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)
        self.r = self._make_r(w, h)
        sigmas = self._sigma_points()
        z_sigmas = sigmas[0:4, :]
        z_pred = (self._wm * z_sigmas).sum(axis=1, keepdims=True)
        dz = z_sigmas - z_pred
        S = (dz * self._wc) @ dz.T + self.r
        dx = sigmas - self.x
        Pxz = (dx * self._wc) @ dz.T
        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ (z - z_pred)
        self._clip_state()
        self.p = self.p - K @ S @ K.T
        self._project_spd()
        # 记录创新息统计
        innovation = z - z_pred
        self.last_innovation_sq = float(np.sum(innovation ** 2))
        self.last_s_trace = float(np.trace(S))
        nis = self.last_innovation_sq / max(self.last_s_trace, 1e-10)
        self._adapt_q(nis)
        return self.to_box()


class _FilterAdaptor:
    """自适应滤波器选择器：基于运行时统计自动切换 UKF/MCUKF/Manifold UKF。

    监控信号：
    1. 归一化新息平方 (NIS) = ||z - z_pred||² / trace(S)
    2. 缺失率 = miss_count / total_count
    3. P 条件数 = max(λ) / min(λ)
    4. Cholesky 分解失败次数

    切换策略（优先级从高到低）：
    - P 条件数 > 500 或 Cholesky 失败 > 2 → manifold_ukf（SPD 保证）
    - 平均 NIS > 5.0 或 缺失率 > 40% → mcukf（抗离群）
    - 正常 → ukf（最快）

    防抖：需连续 HYSTERESIS 次评估都指向同一目标才切换。
    """

    _EVAL_INTERVAL = 30           # 每 30 次更新评估一次（约 1 秒 @ 30fps）
    _NIS_THRESHOLD = 8.0          # 平均 NIS 阈值：放宽，减少频繁切换
    _COND_THRESHOLD = 1000.0      # P 条件数阈值：Manifold 作为安全网
    _MISS_RATE_THRESHOLD = 0.5    # 缺失率阈值：放宽
    _CHOLESKY_FAIL_THRESHOLD = 2  # Cholesky 失败次数阈值
    _HYSTERESIS = 2               # 连续 N 次评估一致才切换（防抖）

    def __init__(self, kernel_sigma: float = 0.4):
        self._kernel_sigma = kernel_sigma
        self._current_type = "ukf"
        self._nis_window: list[float] = []
        self._miss_count = 0
        self._total_count = 0
        self._cholesky_fails = 0
        self._update_count = 0
        self._switch_votes: dict[str, int] = {}

    @property
    def current_type(self) -> str:
        return self._current_type

    def record_correct(self, nis: float) -> None:
        """记录一次 correct() 的 NIS 值。"""
        self._nis_window.append(nis)
        self._total_count += 1
        self._update_count += 1

    def record_miss(self) -> None:
        """记录一次 track miss。"""
        self._miss_count += 1
        self._total_count += 1

    def record_cholesky_fail(self) -> None:
        """记录一次 Cholesky 分解失败。"""
        self._cholesky_fails += 1

    def evaluate(self, p_cond: float) -> str | None:
        """评估是否需要切换滤波器类型。

        p_cond: 当前活跃 track 的 P 矩阵条件数中位数
        返回: 新类型字符串（需要切换）或 None（保持当前）
        """
        if self._update_count < self._EVAL_INTERVAL:
            return None

        # 计算窗口统计
        recent_nis = self._nis_window[-self._EVAL_INTERVAL:]
        mean_nis = float(np.mean(recent_nis)) if recent_nis else 0.0
        miss_rate = self._miss_count / max(1, self._total_count)

        # 决策树（优先级从高到低）
        if p_cond > self._COND_THRESHOLD or self._cholesky_fails > self._CHOLESKY_FAIL_THRESHOLD:
            target = "manifold_ukf"
        elif mean_nis > self._NIS_THRESHOLD or miss_rate > self._MISS_RATE_THRESHOLD:
            target = "mcukf"
        else:
            target = "ukf"

        # 防抖：需连续 HYSTERESIS 次评估都指向同一目标才切换
        if target != self._current_type:
            self._switch_votes[target] = self._switch_votes.get(target, 0) + 1
            if self._switch_votes[target] >= self._HYSTERESIS:
                self._current_type = target
                self._switch_votes.clear()
                self._reset_window()
                return target
        else:
            self._switch_votes.clear()

        self._update_count = 0
        return None

    def _reset_window(self) -> None:
        self._nis_window.clear()
        self._miss_count = 0
        self._total_count = 0
        self._cholesky_fails = 0
        self._update_count = 0


class _Track:
    __slots__ = (
        "track_id", "label", "box", "confidence", "keypoints", "polygon", "prev_polygon",
        "hits", "misses", "age",
        "kalman", "predicted_only", "track_state", "redetect_attempts",
        "color_hist", "prev_box", "_color_ema",
        "embedding", "_embedding_ema",
    )

    def __init__(self, track_id: int, label: str, box: np.ndarray, confidence: float, keypoints: list[dict],
                 filter_type: str = "ukf", mcukf_kernel_sigma: float = 0.4,
                 polygon: list[tuple[float, float]] | None = None,
                 velocity_clip: float = 0.30):
        self.track_id = track_id
        self.label = label
        self.box = box.astype(np.float64)                     # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.keypoints = keypoints                             # list[dict]
        self.polygon = polygon                                 # OBB 四边形角点 [(x,y), ...]
        self.hits = 1
        self.misses = 0
        self.age = 0
        # 根据类型创建滤波器
        if filter_type == "mcukf":
            self.kalman = _DetectionMCUKF(box, kernel_sigma=mcukf_kernel_sigma, velocity_clip=velocity_clip)
        elif filter_type == "manifold_ukf":
            self.kalman = _DetectionManifoldUKF(box, velocity_clip=velocity_clip)
        else:
            self.kalman = _DetectionUKF(box, velocity_clip=velocity_clip)
        self.predicted_only = False                          # 是否为纯预测帧
        self.track_state = "normal"                          # normal / predicted / redetected
        self.redetect_attempts = 0                           # 已重检测尝试次数
        self.color_hist = None                               # HSV 颜色直方图（用于 ID 恢复）
        self._color_ema = 0.9                                # 直方图 EMA 系数
        self.embedding = None                                # ReID embedding（256 维）
        self._embedding_ema = 0.8                            # embedding EMA 系数
        self.prev_box = self.box.copy()                      # 上一帧 box，用于关键点平移
        self.prev_polygon = [(float(x), float(y)) for x, y in polygon] if polygon else None  # 上一帧 polygon

    def to_dict(self) -> dict:
        d = {
            "track_id": self.track_id,
            "x1": float(self.box[0]),
            "y1": float(self.box[1]),
            "x2": float(self.box[2]),
            "y2": float(self.box[3]),
            "confidence": float(self.confidence),
            "label": self.label,
            "keypoints": self.keypoints,
            "predicted_only": self.predicted_only,
            "track_state": self.track_state,
        }
        if self.polygon is not None:
            d["polygon"] = [[float(x), float(y)] for x, y in self.polygon]
        return d

    def swap_filter(self, filter_type: str, kernel_sigma: float = 0.4) -> None:
        """热切换滤波器类型，保留状态 (x, P, Q, R)。

        用于 AUTO 模式下自适应切换：创建新类型滤波器实例，
        将旧滤波器的状态复制过去，实现无缝切换。
        """
        old = self.kalman
        if filter_type == "mcukf":
            new = _DetectionMCUKF(self.box, kernel_sigma=kernel_sigma, velocity_clip=old._velocity_clip)
        elif filter_type == "manifold_ukf":
            new = _DetectionManifoldUKF(self.box, velocity_clip=old._velocity_clip)
        else:
            new = _DetectionUKF(self.box, velocity_clip=old._velocity_clip)
        # 保留状态
        new.x = old.x.copy()
        new.p = old.p.copy()
        new.q = old.q.copy()
        new.r = old.r.copy()
        self.kalman = new

    def predict_polygon(self, old_box: np.ndarray, new_box: np.ndarray) -> None:
        """按卡尔曼预测 box 的平移/缩放外推 polygon，跳帧时保持旋转姿态。

        保持 polygon 相对形状不变，仅跟随 bbox 中心平移和尺度变化。
        """
        if self.polygon is None or len(self.polygon) != 4:
            return
        self.prev_polygon = [(float(x), float(y)) for x, y in self.polygon]
        old_cx = (float(old_box[0]) + float(old_box[2])) * 0.5
        old_cy = (float(old_box[1]) + float(old_box[3])) * 0.5
        new_cx = (float(new_box[0]) + float(new_box[2])) * 0.5
        new_cy = (float(new_box[1]) + float(new_box[3])) * 0.5
        old_w = max(1e-5, float(old_box[2]) - float(old_box[0]))
        old_h = max(1e-5, float(old_box[3]) - float(old_box[1]))
        new_w = max(1e-5, float(new_box[2]) - float(new_box[0]))
        new_h = max(1e-5, float(new_box[3]) - float(new_box[1]))
        scale_x = new_w / old_w
        scale_y = new_h / old_h
        predicted = []
        for x, y in self.polygon:
            px = new_cx + (float(x) - old_cx) * scale_x
            py = new_cy + (float(y) - old_cy) * scale_y
            predicted.append((_clamp01(px), _clamp01(py)))
        self.polygon = predicted

    def update_color_hist(self, frame: np.ndarray) -> None:
        """从帧中更新该 track 的 HSV 颜色直方图（中心 50% 区域）。

        增强：H[32]S[16]+V[8] = 520维，Bhattacharyya 距离更稳定。
        遮挡期（misses>0）停止更新避免污染。
        """
        if frame is None or frame.size == 0:
            return
        if self.misses > 0:
            return  # 遮挡期不更新
        frame_h, frame_w = frame.shape[:2]
        x1 = int(_clamp01(self.box[0]) * frame_w)
        y1 = int(_clamp01(self.box[1]) * frame_h)
        x2 = int(_clamp01(self.box[2]) * frame_w)
        y2 = int(_clamp01(self.box[3]) * frame_h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        # 取中心 50% 区域，减少背景干扰
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w2, h2 = max(2, (x2 - x1) // 4), max(2, (y2 - y1) // 4)
        cx1, cy1 = max(0, cx - w2), max(0, cy - h2)
        cx2, cy2 = min(frame_w, cx + w2), min(frame_h, cy + h2)
        roi = frame[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return
        hist = _compute_hsv_hist(roi)
        if hist is None:
            return
        if self.color_hist is None:
            self.color_hist = hist
        else:
            self.color_hist = self._color_ema * self.color_hist + (1.0 - self._color_ema) * hist

    def update_embedding(self, emb: np.ndarray | None) -> None:
        """更新 ReID embedding（EMA 0.8/0.2），遮挡期不更新。"""
        if emb is None:
            return
        if self.misses > 0:
            return  # 遮挡期不更新
        if self.embedding is None:
            self.embedding = emb.copy()
        else:
            self.embedding = self._embedding_ema * self.embedding + (1.0 - self._embedding_ema) * emb
        # L2 归一化
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding = self.embedding / norm

    def appearance_similarity(self, other_hist: np.ndarray | None) -> float:
        """计算与另一颜色直方图的 Bhattacharyya 相似度，返回 [0, 1]。

        Bhattacharyya 距离越小越相似，1-dist 归一化到 [0, 1]。
        """
        if self.color_hist is None or other_hist is None:
            return 0.5
        dist = float(cv2.compareHist(self.color_hist, other_hist, cv2.HISTCMP_BHATTACHARYYA))
        return float(max(0.0, min(1.0, 1.0 - dist)))


class DetectionTracker:
    """轻量跟踪器，每帧调用 update() 即可。"""

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_misses: int = 4,
        smooth_alpha: float = 0.85,
        min_hits_to_show: int = 1,
        redetect_enabled: bool = False,
        redetect_max_retries: int = 2,
        redetect_iou_threshold: float = 0.20,
        filter_type: str = "ukf",
        mcukf_kernel_sigma: float = 0.25,
        matching_strategy: str = "hungarian",
        high_conf_thresh: float = 0.5,
        velocity_clip: float = 0.30,
        second_stage_iou_min: float = 0.1,
        redetect_budget: int = 1,
        reid_backend=None,
        reid_min_tracks: int = 3,
        reid_stride: int = 3,
    ):
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._smooth_alpha = smooth_alpha
        self._min_hits = min_hits_to_show
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}
        # ROI 重检测配置
        self._redetect_enabled = bool(redetect_enabled)
        self._redetect_max_retries = max(0, int(redetect_max_retries))
        self._redetect_iou_threshold = float(redetect_iou_threshold)
        self._redetect_budget = max(0, int(redetect_budget))
        # 滤波器类型配置
        self._filter_type = filter_type
        self._mcukf_kernel_sigma = float(mcukf_kernel_sigma)
        # 匹配策略配置
        self._matching_strategy = matching_strategy            # greedy / hungarian / cascade
        self._high_conf_thresh = float(high_conf_thresh)
        self._velocity_clip = float(velocity_clip)
        self._second_stage_iou_min = float(second_stage_iou_min)
        # ReID 配置
        self._reid_backend = reid_backend
        self._reid_min_tracks = int(reid_min_tracks)
        self._reid_stride = max(1, int(reid_stride))
        self._reid_frame_counter = 0
        # AUTO 自适应适配器
        self._adaptor: _FilterAdaptor | None = None
        if filter_type == "auto":
            self._adaptor = _FilterAdaptor(kernel_sigma=self._mcukf_kernel_sigma)

    def _get_effective_filter_type(self) -> str:
        """获取当前应使用的滤波器类型（AUTO 模式下由适配器决定）。"""
        if self._adaptor is not None:
            return self._adaptor.current_type
        return self._filter_type

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def set_smooth_alpha(self, alpha: float) -> None:
        self._smooth_alpha = float(np.clip(alpha, 0.0, 1.0))

    def set_redetect_config(
        self,
        *,
        enabled: bool | None = None,
        max_retries: int | None = None,
        iou_threshold: float | None = None,
    ) -> None:
        """运行时更新重检测配置，对所有现有 track 生效。"""
        if enabled is not None:
            self._redetect_enabled = bool(enabled)
        if max_retries is not None:
            self._redetect_max_retries = max(0, int(max_retries))
        if iou_threshold is not None:
            self._redetect_iou_threshold = float(iou_threshold)

    def set_filter_type(self, filter_type: str, kernel_sigma: float | None = None) -> None:
        """运行时更新滤波器类型。已有 track 保持旧滤波器，新 track 使用新类型。"""
        self._filter_type = filter_type
        if kernel_sigma is not None:
            self._mcukf_kernel_sigma = float(kernel_sigma)
        # AUTO 模式管理适配器
        if filter_type == "auto":
            if self._adaptor is None:
                self._adaptor = _FilterAdaptor(kernel_sigma=self._mcukf_kernel_sigma)
            elif kernel_sigma is not None:
                self._adaptor._kernel_sigma = self._mcukf_kernel_sigma
        else:
            self._adaptor = None

    def set_tracker_params(
        self,
        *,
        iou_threshold: float | None = None,
        max_misses: int | None = None,
        matching_strategy: str | None = None,
        high_conf_thresh: float | None = None,
        velocity_clip: float | None = None,
        second_stage_iou_min: float | None = None,
    ) -> None:
        """批量更新跟踪器参数，运行时生效。

        已有 track 的 velocity_clip 也会同步更新（影响下一帧 _clip_state）。
        """
        if iou_threshold is not None:
            self._iou_threshold = float(iou_threshold)
        if max_misses is not None:
            self._max_misses = int(max_misses)
        if matching_strategy is not None:
            self._matching_strategy = str(matching_strategy)
        if high_conf_thresh is not None:
            self._high_conf_thresh = float(high_conf_thresh)
        if velocity_clip is not None:
            self._velocity_clip = float(velocity_clip)
            # 同步到已有 track 的滤波器
            for trk in self._tracks.values():
                trk.kalman._velocity_clip = float(velocity_clip)
        if second_stage_iou_min is not None:
            self._second_stage_iou_min = float(second_stage_iou_min)

    def update_appearance(self, frame: np.ndarray) -> None:
        """为所有活跃 track 更新外观颜色直方图。"""
        for trk in self._tracks.values():
            if trk.label == "person":
                trk.update_color_hist(frame)

    def predict_step(self) -> list[dict]:
        """跳帧路径专用：仅推进卡尔曼，不做匹配/重检测/删除。

        - 推进所有 track 的 kalman.predict()，用预测框替换 box
        - 标记 predicted_only=True / track_state="predicted"
        - misses/age 不变（跳帧不视为 miss，避免 stride=2 时 track 被过快清除）
        - 不删除 track（生命周期仍由后续推理帧的 update 控制）
        """
        for trk in self._tracks.values():
            old_box = trk.box.copy()
            trk.box = trk.kalman.predict()
            trk.predict_polygon(old_box, trk.box)
            trk.predicted_only = True
            trk.track_state = "predicted"
        return self._active_tracks()

    def update(
        self,
        detections: list[dict] | list[Detection],
        redetect_callback=None,
        frame: np.ndarray | None = None,
    ) -> list[dict]:
        """
        输入：原始 YOLO 输出（归一化坐标），可选 redetect_callback 与当前帧。
        输出：平滑后的 detection list，每个 dict 含 track_id。

        兼容 Detection 类型和既有 dict schema；内部统一转换为 dict 处理。

        redetect_callback: Callable[[np.ndarray, float], dict | None]
            输入归一化预测框 [x1,y1,x2,y2] 与原置信度，返回 detection dict 或 None。
            由 InferWorker 注入，tracker 本身不依赖 frame/model。
        frame: 当前帧（BGR），用于计算检测外观直方图辅助匹配。
        """
        # 统一转换为 dict，保留 polygon 等扩展字段
        detections = [d.to_dict() if isinstance(d, Detection) else d for d in detections]

        # --- 0. 推进所有 track 的卡尔曼（与 RectangleTracker 一致：先 predict 再匹配）---
        for trk in self._tracks.values():
            trk.age += 1
            old_box = trk.box.copy()
            predicted_box = trk.kalman.predict()
            trk.box = predicted_box
            trk.predict_polygon(old_box, predicted_box)

        # --- 1. 准备本次输入的 box 矩阵与外观直方图 ---
        n = len(detections)
        det_hist = [None] * n
        det_emb = [None] * n
        if frame is not None and frame.size > 0:
            for i, det in enumerate(detections):
                if det.get("label") == "person":
                    det_hist[i] = self._compute_color_hist_for_box(frame, det)

        # ReID embedding 准备：仅当条件满足时计算
        reid_active = False
        if self._reid_backend is not None and frame is not None and frame.size > 0:
            person_count = sum(1 for d in detections if d.get("label") == "person")
            track_person_count = sum(1 for t in self._tracks.values() if t.label == "person")
            if person_count >= 1 and track_person_count >= self._reid_min_tracks \
               and self._reid_frame_counter % self._reid_stride == 0:
                reid_active = True
                frame_h, frame_w = frame.shape[:2]
                for i, det in enumerate(detections):
                    if det.get("label") == "person":
                        x1 = int(_clamp01(float(det["x1"])) * frame_w)
                        y1 = int(_clamp01(float(det["y1"])) * frame_h)
                        x2 = int(_clamp01(float(det["x2"])) * frame_w)
                        y2 = int(_clamp01(float(det["y2"])) * frame_h)
                        if x2 > x1 and y2 > y1:
                            det_emb[i] = self._reid_backend.extract(frame[y1:y2, x1:x2])

        if n == 0:
            # 没有新检测：尝试对未匹配 track 做重检测，否则 miss+1
            self._try_redetect_for_unmatched(set(), redetect_callback)
            self._cleanup_expired()
            self._evaluate_auto_switch()
            self._reid_frame_counter += 1
            return self._active_tracks()

        boxes = np.zeros((n, 4), dtype=np.float64)
        for i, det in enumerate(detections):
            boxes[i] = [det["x1"], det["y1"], det["x2"], det["y2"]]

        # --- 2. 匹配策略分发 ---
        det_conf = np.array([det["confidence"] for det in detections], dtype=np.float64)
        matched_track_ids, matched_det_indices, skip_new = self._run_matching(
            detections, boxes, det_hist, det_conf, det_emb, reid_active
        )

        # --- 3. 未匹配的检测 → 新建 track（级联模式下低置信度检测不新建）---
        for i, det in enumerate(detections):
            if i not in matched_det_indices and i not in skip_new:
                tid = self._next_id
                self._next_id += 1
                box = np.array([det["x1"], det["y1"], det["x2"], det["y2"]], dtype=np.float64)
                kpts = self._clone_keypoints(det.get("keypoints", []))
                polygon = self._extract_polygon(det)
                trk = _Track(tid, det.get("label", ""), box, det["confidence"], kpts,
                             filter_type=self._get_effective_filter_type(),
                             mcukf_kernel_sigma=self._mcukf_kernel_sigma,
                             polygon=polygon,
                             velocity_clip=self._velocity_clip)
                # 初始化外观直方图
                if det_hist[i] is not None:
                    trk.color_hist = det_hist[i]
                # 初始化 ReID embedding
                if det_emb[i] is not None:
                    trk.embedding = det_emb[i].copy()
                self._tracks[tid] = trk
                matched_track_ids.add(tid)  # 新建 track 已被 accounted，不参与后续 miss 逻辑

        # --- 4. 未匹配的 track → 尝试重检测或 miss+1 ---
        self._try_redetect_for_unmatched(matched_track_ids, redetect_callback)

        # --- 5. 清理超期 track ---
        self._cleanup_expired()

        # --- 6. AUTO 模式：评估切换 ---
        self._evaluate_auto_switch()

        self._reid_frame_counter += 1

        return self._active_tracks()

    def _evaluate_auto_switch(self) -> None:
        """AUTO 模式：评估是否需要热切换滤波器类型。"""
        if self._adaptor is None:
            return
        conds = [float(np.linalg.cond(t.kalman.p)) for t in self._tracks.values() if t.hits >= self._min_hits]
        median_cond = float(np.median(conds)) if conds else 1.0
        new_type = self._adaptor.evaluate(median_cond)
        if new_type is not None:
            for trk in self._tracks.values():
                trk.swap_filter(new_type, self._mcukf_kernel_sigma)

    # ===================================================================
    # 匹配策略：greedy / hungarian / cascade
    # ===================================================================

    def _run_matching(
        self,
        detections: list[dict],
        boxes: np.ndarray,
        det_hist: list,
        det_conf: np.ndarray,
        det_emb: list | None = None,
        reid_active: bool = False,
    ) -> tuple[set[int], set[int], set[int]]:
        """分发到具体匹配策略。

        Returns:
            matched_track_ids: 已匹配的 track ID 集合
            matched_det_indices: 已匹配的检测索引集合
            skip_new: 不应新建 track 的检测索引集合（级联模式下的低置信度检测）
        """
        strategy = self._matching_strategy
        if strategy == "cascade":
            return self._match_cascade(detections, boxes, det_hist, det_conf, det_emb, reid_active)
        elif strategy == "hungarian":
            matched_tids, matched_dis = self._match_hungarian(
                detections, boxes, det_hist, det_conf,
                iou_gate=self._iou_threshold,
                det_emb=det_emb, reid_active=reid_active,
            )
            return matched_tids, matched_dis, set()
        else:
            return self._match_greedy(detections, boxes, det_hist, det_conf)

    def _match_greedy(
        self,
        detections: list[dict],
        boxes: np.ndarray,
        det_hist: list,
        det_conf: np.ndarray,
    ) -> tuple[set[int], set[int], set[int]]:
        """原始贪婪匹配（按置信度降序逐检测找最佳 track），保留为回退。"""
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()
        det_order = np.argsort(-det_conf)

        for di in det_order:
            if di in matched_det_indices:
                continue
            best_score = -1.0
            best_tid = None
            det_box = boxes[di]
            det_label = detections[di].get("label", "")
            det_polygon = self._extract_polygon(detections[di])
            det_cx, det_cy = _shape_center(det_box, det_polygon)

            for tid, trk in self._tracks.items():
                if tid in matched_track_ids or trk.label != det_label:
                    continue
                iou = _shape_iou(trk.box, trk.polygon, det_box, det_polygon)
                trk_cx, trk_cy = _shape_center(trk.box, trk.polygon)
                dist = ((trk_cx - det_cx) ** 2 + (trk_cy - det_cy) ** 2) ** 0.5
                avg_size = (
                    (trk.box[2] - trk.box[0]) + (det_box[2] - det_box[0])
                    + (trk.box[3] - trk.box[1]) + (det_box[3] - det_box[1])
                ) * 0.25
                center_close = avg_size > 1e-3 and dist < 0.5 * avg_size
                accept = iou > self._iou_threshold or (iou > 0.05 and center_close)
                if accept:
                    appearance_bonus = trk.appearance_similarity(det_hist[di]) * 0.15
                    score = iou + (1.0 - min(1.0, dist / max(avg_size, 1e-3))) * 0.1 + appearance_bonus
                    if score > best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                matched_track_ids.add(best_tid)
                matched_det_indices.add(di)
                self._update_track(best_tid, detections[di], boxes[di])

        return matched_track_ids, matched_det_indices, set()

    def _match_hungarian(
        self,
        detections: list[dict],
        boxes: np.ndarray,
        det_hist: list,
        det_conf: np.ndarray,
        iou_gate: float | None = None,
        appearance_weight: float = 0.05,
        track_filter: set[int] | None = None,
        det_emb: list | None = None,
        reid_active: bool = False,
    ) -> tuple[set[int], set[int]]:
        """匈牙利全局最优匹配。

        cost = 1 - (0.85*IoU + 0.10*center_proximity + appearance_weight*appearance)
        ReID 启用时：cost = 1 - (0.4*IoU + 0.4*reid_sim + 0.2*center_proximity)
        门控：IoU < iou_gate 且不满足中心距回退 → cost=1e6（匈牙利自动跳过）

        Args:
            track_filter: 仅匹配这些 track ID（None=全部）。用于级联 Stage 2。
            det_emb: 检测的 ReID embedding 列表（None=未启用）
            reid_active: ReID 是否在本帧激活
        Returns:
            (matched_track_ids, matched_det_indices)
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # scipy 不可用 → 回退贪婪
            tids, dis, _ = self._match_greedy(detections, boxes, det_hist, det_conf)
            return tids, dis

        gate = self._iou_threshold if iou_gate is None else iou_gate
        matched_track_ids: set[int] = set()
        matched_det_indices: set[int] = set()

        # 按 label 分组匹配
        all_labels = set(d.get("label", "") for d in detections)
        all_labels &= set(trk.label for trk in self._tracks.values())

        for label in all_labels:
            det_indices = [i for i, d in enumerate(detections)
                           if d.get("label", "") == label and i not in matched_det_indices]
            track_items = [(tid, trk) for tid, trk in self._tracks.items()
                           if tid not in matched_track_ids and trk.label == label
                           and (track_filter is None or tid in track_filter)]
            if not det_indices or not track_items:
                continue

            n_dets = len(det_indices)
            n_tracks = len(track_items)
            cost = np.full((n_dets, n_tracks), 1e6, dtype=np.float64)

            for di_idx, di in enumerate(det_indices):
                det_box = boxes[di]
                det_polygon = self._extract_polygon(detections[di])
                det_cx, det_cy = _shape_center(det_box, det_polygon)

                for ti_idx, (tid, trk) in enumerate(track_items):
                    iou = _shape_iou(trk.box, trk.polygon, det_box, det_polygon)
                    trk_cx, trk_cy = _shape_center(trk.box, trk.polygon)
                    dist = ((trk_cx - det_cx) ** 2 + (trk_cy - det_cy) ** 2) ** 0.5
                    avg_size = (
                        (trk.box[2] - trk.box[0]) + (det_box[2] - det_box[0])
                        + (trk.box[3] - trk.box[1]) + (det_box[3] - det_box[1])
                    ) * 0.25
                    center_close = avg_size > 1e-3 and dist < 0.5 * avg_size
                    accept = iou > gate or (iou > 0.05 and center_close)
                    if accept:
                        center_proximity = 1.0 - min(1.0, dist / max(avg_size, 1e-3))
                        # ReID 分支：有 embedding 时用 ReID+IoU+center
                        if reid_active and det_emb is not None and det_emb[di] is not None and trk.embedding is not None:
                            from ai.reid_backend import ReIDBackend
                            reid_sim = max(0.0, ReIDBackend.cosine_similarity(trk.embedding, det_emb[di]))
                            score = 0.4 * iou + 0.4 * reid_sim + 0.2 * center_proximity
                        else:
                            appearance_sim = trk.appearance_similarity(det_hist[di])
                            score = 0.85 * iou + 0.10 * center_proximity + appearance_weight * appearance_sim
                        cost[di_idx, ti_idx] = 1.0 - score

            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < 0.999:  # 有效匹配（非门控）
                    di = det_indices[r]
                    tid = track_items[c][0]
                    matched_track_ids.add(tid)
                    matched_det_indices.add(di)
                    self._update_track(tid, detections[di], boxes[di])
                    # 匹配成功后更新 ReID embedding
                    if reid_active and det_emb is not None and det_emb[di] is not None:
                        self._tracks[tid].update_embedding(det_emb[di])

        return matched_track_ids, matched_det_indices

    def _match_cascade(
        self,
        detections: list[dict],
        boxes: np.ndarray,
        det_hist: list,
        det_conf: np.ndarray,
        det_emb: list | None = None,
        reid_active: bool = False,
    ) -> tuple[set[int], set[int], set[int]]:
        """ByteTrack 风格两阶段级联匹配。

        Stage 1: 高置信度检测 vs 全部 track，匈牙利 + 严格门控
        Stage 2: 低置信度检测 vs Stage1 未匹配 track，匈牙利 + 放宽门控 + 外观加权
                 低置信度检测仅关联已有 track，不新建（返回 skip_new 集合）
        """
        # 分离高/低置信度检测
        high_mask = det_conf >= self._high_conf_thresh
        high_indices = set(int(i) for i in np.where(high_mask)[0])
        low_indices = set(int(i) for i in np.where(~high_mask)[0])

        # Stage 1: 高置信度检测 vs 全部 track
        matched_tids, matched_dis = self._match_hungarian(
            detections, boxes, det_hist, det_conf,
            iou_gate=self._iou_threshold,
            appearance_weight=0.05,
            det_emb=det_emb, reid_active=reid_active,
        )

        # Stage 2: 低置信度检测 vs Stage1 未匹配 track
        # 构造仅含低置信度检测的子集进行匹配
        unmatched_tids = set(self._tracks.keys()) - matched_tids
        if low_indices and unmatched_tids:
            # 临时过滤 detections/boxes/det_hist 仅保留低置信度
            low_det_list = [detections[i] for i in sorted(low_indices)]
            low_boxes = np.array([boxes[i] for i in sorted(low_indices)], dtype=np.float64)
            low_hist = [det_hist[i] for i in sorted(low_indices)]
            low_conf = np.array([det_conf[i] for i in sorted(low_indices)], dtype=np.float64)
            low_emb = [det_emb[i] if det_emb else None for i in sorted(low_indices)] if det_emb else None

            # 映射回原始索引
            low_idx_map = sorted(low_indices)

            stage2_tids, stage2_dis_local = self._match_hungarian(
                low_det_list, low_boxes, low_hist, low_conf,
                iou_gate=self._second_stage_iou_min,
                appearance_weight=0.30,  # Stage 2 外观权重提升
                track_filter=unmatched_tids,
                det_emb=low_emb, reid_active=reid_active,
            )

            # 将 local 索引映射回原始
            for local_i in stage2_dis_local:
                original_i = low_idx_map[local_i]
                matched_dis.add(original_i)
            matched_tids |= stage2_tids

        # 低置信度未匹配检测不新建 track
        skip_new = low_indices - matched_dis

        return matched_tids, matched_dis, skip_new

    def _try_redetect_for_unmatched(
        self,
        matched_track_ids: set[int],
        redetect_callback,
    ) -> None:
        """对未匹配的 track 尝试 ROI 重检测，失败则 miss+1。

        预算限流：每帧最多对 budget 个 track 执行重检测（按 misses 降序优先）。
        超预算的候选仅 miss+1，不累加 redetect_attempts（延后到下一帧重试）。
        """
        redetect_available = (
            self._redetect_enabled
            and redetect_callback is not None
            and self._redetect_max_retries > 0
        )
        budget = max(0, self._redetect_budget)

        # 收集重检测候选（person + 条件满足）
        candidates: list[tuple[int, _Track]] = []
        for tid, trk in list(self._tracks.items()):
            if tid in matched_track_ids:
                continue
            if trk.label == "person" and redetect_available and trk.redetect_attempts < self._redetect_max_retries:
                candidates.append((tid, trk))
            else:
                # 非候选：直接 miss+1
                trk.misses += 1
                trk.predicted_only = True
                trk.track_state = "predicted"
                if self._adaptor is not None:
                    self._adaptor.record_miss()

        # 按 misses 降序排序（丢失最久优先）
        candidates.sort(key=lambda x: x[1].misses, reverse=True)

        # 预算内的候选执行重检测
        for idx, (tid, trk) in enumerate(candidates):
            if idx < budget:
                # 执行重检测
                try:
                    refined = redetect_callback(trk.box, trk.confidence)
                except Exception:
                    refined = None

                if refined is not None:
                    refined_box = np.array(
                        [refined["x1"], refined["y1"], refined["x2"], refined["y2"]],
                        dtype=np.float64,
                    )
                    iou_with_pred = _box_iou(trk.box, refined_box)
                    if iou_with_pred >= self._redetect_iou_threshold:
                        self._update_track(tid, refined, refined_box)
                        trk.track_state = "redetected"
                        trk.predicted_only = False
                        trk.redetect_attempts = 0
                        continue
                # 重检测失败（预算内）：attempts+1 + miss+1
                trk.redetect_attempts += 1
            # 超预算或重检测失败：miss+1，attempts 不变（超预算时）
            trk.misses += 1
            trk.predicted_only = True
            trk.track_state = "predicted"
            if self._adaptor is not None:
                self._adaptor.record_miss()

    def _cleanup_expired(self) -> None:
        """清理 misses 超期的 track。"""
        for tid in list(self._tracks.keys()):
            if self._tracks[tid].misses > self._max_misses:
                del self._tracks[tid]

    def _compute_color_hist_for_box(self, frame: np.ndarray, det: dict) -> np.ndarray | None:
        """为检测框计算增强 HSV 颜色直方图（中心 50% 区域）。"""
        if frame is None or frame.size == 0:
            return None
        frame_h, frame_w = frame.shape[:2]
        x1 = int(_clamp01(float(det["x1"])) * frame_w)
        y1 = int(_clamp01(float(det["y1"])) * frame_h)
        x2 = int(_clamp01(float(det["x2"])) * frame_w)
        y2 = int(_clamp01(float(det["y2"])) * frame_h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        w2, h2 = max(2, (x2 - x1) // 4), max(2, (y2 - y1) // 4)
        cx1, cy1 = max(0, cx - w2), max(0, cy - h2)
        cx2, cy2 = min(frame_w, cx + w2), min(frame_h, cy + h2)
        roi = frame[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return None
        return _compute_hsv_hist(roi)
        return hist

    def _warp_keypoints(self, old_kpts: list[dict], old_box: np.ndarray, new_box: np.ndarray) -> list[dict]:
        """按新 bbox 的中心与尺度平移缩放旧关键点。"""
        old_cx = (old_box[0] + old_box[2]) * 0.5
        old_cy = (old_box[1] + old_box[3]) * 0.5
        new_cx = (new_box[0] + new_box[2]) * 0.5
        new_cy = (new_box[1] + new_box[3]) * 0.5
        scale_x = (new_box[2] - new_box[0]) / max(old_box[2] - old_box[0], 1e-5)
        scale_y = (new_box[3] - new_box[1]) / max(old_box[3] - old_box[1], 1e-5)
        warped = []
        for kp in old_kpts:
            warped.append({
                "x": float(_clamp01(new_cx + (kp["x"] - old_cx) * scale_x)),
                "y": float(_clamp01(new_cy + (kp["y"] - old_cy) * scale_y)),
                "conf": float(kp.get("conf", 0.5)),
            })
        return warped

    def _update_track(self, tid: int, det: dict, new_box: np.ndarray) -> None:
        trk = self._tracks[tid]
        alpha = self._smooth_alpha

        # 卡尔曼校正：让速度估计吸收本次观测
        corrected_box = trk.kalman.correct(new_box)

        # AUTO 模式：记录 NIS 和 Cholesky 失败
        if self._adaptor is not None:
            nis = trk.kalman.last_innovation_sq / max(trk.kalman.last_s_trace, 1e-10)
            self._adaptor.record_correct(nis)
            if getattr(trk.kalman, 'cholesky_failed', False):
                self._adaptor.record_cholesky_fail()
                trk.kalman.cholesky_failed = False

        # EMA 平滑 box（用 corrected_box 而非裸 det box，类似 RectangleTracker）
        trk.box = alpha * corrected_box + (1.0 - alpha) * trk.box

        # 置信度 EMA（略偏向新值，让缓慢下降的目标自然淡出）
        trk.confidence = 0.7 * det["confidence"] + 0.3 * trk.confidence

        # 关键点独立 EMA 平滑或平移缩放
        old_box = trk.prev_box
        new_kpts = det.get("keypoints", [])
        if len(new_kpts) == 17 and len(trk.keypoints) == 17:
            for j in range(17):
                nk = new_kpts[j]
                tk = trk.keypoints[j]
                tk["x"] = float(alpha * nk["x"] + (1.0 - alpha) * tk["x"])
                tk["y"] = float(alpha * nk["y"] + (1.0 - alpha) * tk["y"])
                tk["conf"] = float(0.7 * nk["conf"] + 0.3 * tk["conf"])
        elif len(new_kpts) == 0 and len(trk.keypoints) == 17:
            # ROI 重检测无关键点：按新 bbox 中心与尺度平移旧关键点，避免骨架闪烁
            trk.keypoints = self._warp_keypoints(trk.keypoints, old_box, trk.box)
        else:
            trk.keypoints = self._clone_keypoints(new_kpts)

        # 保存上一帧 box 供下次关键点平移使用
        trk.prev_box = trk.box.copy()

        # OBB polygon 平滑：新观测有 polygon 则 EMA，否则保持上一帧
        new_polygon = self._extract_polygon(det)
        if new_polygon is not None:
            trk.prev_polygon = self._clone_polygon(trk.polygon)
            trk.polygon = self._smooth_polygon(trk.polygon, new_polygon, alpha)

        trk.hits += 1
        trk.misses = 0
        # 状态复位
        trk.predicted_only = False
        trk.track_state = "normal"
        trk.redetect_attempts = 0

    def _active_tracks(self) -> list[dict]:
        results = []
        for trk in self._tracks.values():
            if trk.hits >= self._min_hits:
                results.append(trk.to_dict())
        return results

    @staticmethod
    def _clone_keypoints(kpts: list[dict]) -> list[dict]:
        return [{"x": float(kp["x"]), "y": float(kp["y"]), "conf": float(kp.get("conf", 0.5))} for kp in kpts]

    @staticmethod
    def _extract_polygon(det: dict) -> list[tuple[float, float]] | None:
        polygon = det.get("polygon")
        if not polygon or len(polygon) != 4:
            return None
        try:
            return [(float(p[0]), float(p[1])) for p in polygon]
        except Exception:
            return None

    @staticmethod
    def _clone_polygon(polygon: list[tuple[float, float]] | None) -> list[tuple[float, float]] | None:
        if polygon is None:
            return None
        return [(float(x), float(y)) for x, y in polygon]

    @staticmethod
    def _smooth_polygon(
        old: list[tuple[float, float]] | None,
        new: list[tuple[float, float]],
        alpha: float,
    ) -> list[tuple[float, float]]:
        if old is None or len(old) != len(new):
            return new
        return [
            (alpha * nx + (1.0 - alpha) * ox, alpha * ny + (1.0 - alpha) * oy)
            for (ox, oy), (nx, ny) in zip(old, new)
        ]
