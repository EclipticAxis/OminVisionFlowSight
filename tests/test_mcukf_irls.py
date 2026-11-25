r"""IRLS-MCC UKF（``ai.tracker._DetectionMCUKF``）契约正确性测试。

契约（ai.tracker._DetectionMCUKF 的正确性行为规格）：
  1. 观测统计使用标准 UT 矩（``_wm`` / ``_wc`` 加权），不做核重加权；
  2. 残差经 R 白化后逐通道计算相关熵权
     ``c_i = clip(exp(-e_i^2 / (2 sigma^2)), 1e-4, 1)``（c_min=1e-4
     控制 S_eff 条件数）；
  3. 等效观测噪声 ``R_eff = R C^{-1}``（``C = diag(c)``），增益与 Joseph
     协方差更新使用同一个 ``R_eff``；
  4. 最多 3 次真实 IRLS 不动点迭代（``max_iter`` 默认 3）；
  5. 默认核带宽 ``sigma = 1.0``。

期望来源是契约数学 + 本文件内的独立参考实现 ``_reference_irls``
（按契约文字重新推导编写），不复制 ``ai.tracker`` 中 MCUKF 的生产代码。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.tracker import _DetectionMCUKF, _DetectionUKF, DetectionTracker

# 初始框：cx=0.30 cy=0.30 w=0.10 h=0.14（归一化坐标）
INIT_BOX = np.array([0.25, 0.23, 0.35, 0.37], dtype=np.float64)
INIT_W, INIT_H = 0.10, 0.14


def _xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0],
                    dtype=np.float64)


def _diag_r(filter_) -> np.ndarray:
    """取该滤波器对初始尺度的观测噪声对角标准差（R 本身不是被测对象）。"""
    r = filter_._make_r(INIT_W, INIT_H)
    return np.sqrt(np.diag(r))


def _paired_predict():
    """构造同一预测点上的一对 UKF / MCUKF，返回 (ukf, mcukf, x_pred, p_pred)。"""
    ukf = _DetectionUKF(INIT_BOX.copy())
    mc = _DetectionMCUKF(INIT_BOX.copy())
    ukf.predict()
    mc.predict()
    x_pred, p_pred = mc.x.copy(), mc.p.copy()
    assert np.allclose(ukf.x, x_pred) and np.allclose(ukf.p, p_pred)
    return ukf, mc, x_pred, p_pred


def _assert_psd(p: np.ndarray, eps: float = 1e-9) -> None:
    """协方差健康检查：有限、对称、无显著负特征值。"""
    assert np.all(np.isfinite(p))
    assert np.allclose(p, p.T, atol=1e-12)
    eig = np.linalg.eigvalsh((p + p.T) * 0.5)
    assert eig.min() > -eps, f"P 出现显著负特征值: {eig.min()}"


# ======================================================================
# 独立参考实现（契约数学，非生产代码复制）
# ======================================================================

def _reference_irls(x_pred, p_pred, z, r, sigma, iters):
    """按契约重推导的 MCC-IRLS UKF 单次更新（用于交叉验证迭代语义）。

    - sigma 点 / UT 矩：标准 unscented 变换（Cholesky 散布 + wm/wc 加权）；
    - 相关熵权：白化残差 ``e_w = R^{-1/2} (z - H x_k)`` 逐通道
      ``c_i = clip(exp(-e_wi^2 / (2 sigma^2)), 1e-4, 1)``（c_min=1e-4
      控制 S_eff 条件数）；
    - ``R_eff = R C^{-1}``，``K = Pxz (Pzz + R_eff)^{-1}``；
    - IRLS：不动点迭代围绕当前估计重算残差（真实迭代，非重复同权）；
    - Joseph：``P = (I - KH) P_pred (I - KH)^T + K R_eff K^T``。
    """
    n = 8
    alpha, beta, kappa = 1e-3, 2.0, 0.0
    lam = alpha ** 2 * (n + kappa) - n
    gamma = (n + lam) ** 0.5
    wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
    wm[0] = lam / (n + lam)
    wc = wm.copy()
    wc[0] += 1.0 - alpha ** 2 + beta

    p_sym = (p_pred + p_pred.T) * 0.5
    chol = np.linalg.cholesky(p_sym + 1e-12 * np.eye(n))
    sigmas = np.zeros((n, 2 * n + 1))
    sigmas[:, 0] = x_pred[:, 0]
    for i in range(n):
        off = gamma * chol[:, i]
        sigmas[:, i + 1] = x_pred[:, 0] + off
        sigmas[:, n + i + 1] = x_pred[:, 0] - off

    z_sigmas = sigmas[0:4, :]
    z_pred = (wm * z_sigmas).sum(axis=1, keepdims=True)
    dz = z_sigmas - z_pred
    pzz = (dz * wc) @ dz.T
    dx = sigmas - x_pred
    pxz = (dx * wc) @ dz.T

    h = np.zeros((4, n))
    h[:4, :4] = np.eye(4)
    r_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(r)))

    x = x_pred.copy()
    k_gain = np.zeros((n, 4))
    r_eff = r.copy()
    for _ in range(iters):
        e_w = r_inv_sqrt @ (z - h @ x)
        c = np.exp(-(e_w[:, 0] ** 2) / (2.0 * sigma ** 2))
        c = np.clip(c, 1e-4, 1.0)
        r_eff = r @ np.diag(1.0 / c)
        s_eff = pzz + r_eff
        k_gain = pxz @ np.linalg.inv(s_eff)
        x = x_pred + k_gain @ (z - z_pred)

    i_kh = np.eye(n) - k_gain @ h
    p_post = i_kh @ p_sym @ i_kh.T + k_gain @ r_eff @ k_gain.T
    return x, p_post


# ======================================================================
# 契约 5：默认 sigma = 1.0（滤波器级 + 跟踪器级）
# ======================================================================

def test_default_kernel_sigma_is_one():
    mc = _DetectionMCUKF(INIT_BOX.copy())
    assert mc._kernel_sigma == pytest.approx(1.0), (
        f"契约要求默认核带宽 sigma=1.0，实际 {mc._kernel_sigma}"
    )
    tracker = DetectionTracker(backend="python", filter_type="mcukf")
    assert tracker._mcukf_kernel_sigma == pytest.approx(1.0), (
        f"跟踪器级默认 mcukf_kernel_sigma 应为 1.0，实际 {tracker._mcukf_kernel_sigma}"
    )


def test_default_max_iter_is_three():
    mc = _DetectionMCUKF(INIT_BOX.copy())
    assert mc._max_iter == 3


# ======================================================================
# 契约 1+2+3：sigma -> oo 时必须退化回标准 UKF（数值等价）
# ======================================================================

def test_huge_sigma_matches_ukf():
    """kernel_sigma=1e6 时相关熵权全≈1，MCUKF 的 x/P 应与 UKF 数值等价。"""
    rng = np.random.default_rng(7)
    ukf = _DetectionUKF(INIT_BOX.copy())
    mc = _DetectionMCUKF(INIT_BOX.copy(), kernel_sigma=1e6)
    # 序列：缓慢平移 + 小幅噪声 + 尺寸微扰
    for f in range(1, 15):
        box = _xyxy(0.30 + 0.006 * f + rng.normal(0, 0.005),
                    0.30 + 0.003 * f + rng.normal(0, 0.005),
                    INIT_W + rng.normal(0, 0.002),
                    INIT_H + rng.normal(0, 0.002))
        ukf.predict()
        mc.predict()
        ukf.correct(box.copy())
        mc.correct(box.copy())
        assert np.allclose(mc.x, ukf.x, rtol=1e-6, atol=1e-8), (
            f"第 {f} 帧 x 偏离 UKF: max|dx|={np.abs(mc.x - ukf.x).max()}"
        )
        assert np.allclose(mc.p, ukf.p, rtol=1e-6, atol=1e-8), (
            f"第 {f} 帧 P 偏离 UKF: max|dP|={np.abs(mc.p - ukf.p).max()}"
        )
    # 反退化检查：状态确实跟随了量测（防止两边同样坏掉导致假相等）
    assert abs(mc.x[0, 0] - (0.30 + 0.006 * 14)) < 0.05
    assert mc.x[4, 0] > 0.001  # 学到了 +x 速度


# ======================================================================
# 正常量测：可更新、收敛到匀速、Joseph 协方差健康
# ======================================================================

def test_gaussian_measurements_track_constant_velocity():
    mc = _DetectionMCUKF(INIT_BOX.copy())
    vx, vy = 0.004, 0.002
    rng = np.random.default_rng(11)
    for f in range(1, 26):
        box = _xyxy(0.30 + vx * f + rng.normal(0, 0.01),
                    0.30 + vy * f + rng.normal(0, 0.01),
                    INIT_W + rng.normal(0, 0.004),
                    INIT_H + rng.normal(0, 0.004))
        mc.predict()
        p_pre = mc.p.copy()
        mc.correct(box)
        _assert_psd(mc.p)
        # 正常量测下 Joseph 更新应使协方差收缩（trace 减小）
        assert np.trace(mc.p) < np.trace(p_pre) + 1e-12
    # 位置跟上量测、速度学到接近真实匀速
    z_last_cx, z_last_cy = 0.30 + vx * 25, 0.30 + vy * 25
    assert abs(mc.x[0, 0] - z_last_cx) < 0.05
    assert abs(mc.x[1, 0] - z_last_cy) < 0.05
    assert abs(mc.x[2, 0] - INIT_W) < 0.02 and abs(mc.x[3, 0] - INIT_H) < 0.02
    assert abs(mc.x[4, 0] - vx) < 0.003, f"vx={mc.x[4, 0]} 未收敛到 {vx}"
    assert abs(mc.x[5, 0] - vy) < 0.003, f"vy={mc.x[5, 0]} 未收敛到 {vy}"


def test_consistent_measurement_barely_moves_state():
    """量测恰在预测位置上时，更新应近似恒等（ innovation ≈ 0 ）。"""
    mc = _DetectionMCUKF(INIT_BOX.copy())
    mc.predict()
    x_pre = mc.x.copy()
    z_cx, z_cy = float(x_pre[0, 0]), float(x_pre[1, 0])
    z_w, z_h = float(x_pre[2, 0]), float(x_pre[3, 0])
    mc.correct(_xyxy(z_cx, z_cy, z_w, z_h))
    assert np.allclose(mc.x, x_pre, atol=1e-9), (
        f"零创新息更新应恒等，实际 max|dx|={np.abs(mc.x - x_pre).max()}"
    )


# ======================================================================
# 极端但有限的归一化离群：x/P 基本保持预测值
# ======================================================================

def test_extreme_outlier_preserves_prediction():
    mc = _DetectionMCUKF(INIT_BOX.copy())  # 默认 sigma=1.0
    mc.predict()
    x_pre, p_pre = mc.x.copy(), mc.p.copy()
    std = _diag_r(mc)
    # 每个通道偏移 30 倍自身标准差：c 被 clip 到下界 1e-4
    # -> R_eff = R C^-1 = 1e4 * R -> K ≈ 0（极端但有限的归一化离群）
    k = 30.0
    out_box = _xyxy(0.30 + k * std[0], 0.30 + k * std[1],
                    INIT_W + k * std[2], INIT_H + k * std[3])
    mc.correct(out_box)
    dx = np.abs(mc.x - x_pre)
    # 位置/尺度分量偏离预测极小（< 0.2 倍对应标准差，离群本身为 30 倍）
    assert dx[:4].max() < 0.2 * std.min(), (
        f"极端离群下状态仍被拉动: max|dx|={dx[:4].max()}"
    )
    # 速度分量也不应被离群量测拉动
    assert dx[4:].max() < 0.2 * std.min()
    # Joseph + R_eff≈1e4 R、K≈0 => P 应基本等于预测协方差
    rel_dp = abs(np.trace(mc.p) - np.trace(p_pre)) / np.trace(p_pre)
    assert rel_dp < 0.05, f"极端离群下 P 偏离预测 {rel_dp:.1%}"
    _assert_psd(mc.p)


# ======================================================================
# 单通道离群：相对 UKF 显著抑制该通道的更新
# ======================================================================

def test_single_channel_outlier_suppressed_vs_ukf():
    ukf, mc, x_pre, _ = _paired_predict()
    # 仅 cx 通道 +0.5（= 5 倍位置标准差），其余通道与预测一致
    out_box = _xyxy(0.30 + 0.5, 0.30, INIT_W, INIT_H)
    ukf.correct(out_box.copy())
    mc.correct(out_box.copy())
    d_ukf = abs(ukf.x[0, 0] - x_pre[0, 0])
    d_mc = abs(mc.x[0, 0] - x_pre[0, 0])
    assert d_ukf > 1e-3, "UKF 参考应被离群明显拉动（场景有效性自检）"
    assert d_mc < 0.2 * d_ukf, (
        f"MCUKF 未显著抑制离群通道: d_mc={d_mc}, d_ukf={d_ukf}"
    )
    assert d_mc < 0.02, f"离群通道更新应接近零，实际 {d_mc}"
    # 未离群的 cy 通道：两者都应几乎不动（抑制只应作用于离群通道）
    assert abs(mc.x[1, 0] - x_pre[1, 0]) < 0.02
    assert abs(ukf.x[1, 0] - x_pre[1, 0]) < 0.02


def test_whitening_makes_suppression_scale_aware():
    """相同原始偏移下，R 白化使窄方差通道（位置）比重缩放通道（尺寸）更受抑制。"""
    std = _diag_r(_DetectionMCUKF(INIT_BOX.copy()))
    raw = 0.35  # 位置通道 = 3.5σ_pos；尺寸通道 = 0.35/0.1225 ≈ 2.86σ_size
    sup = {}
    for ch, idx in (("cx", 0), ("w", 2)):
        ukf, mc, x_pre, _ = _paired_predict()
        cx, cy, w, h = 0.30, 0.30, INIT_W, INIT_H
        if ch == "cx":
            cx += raw
        else:
            w += raw
        out_box = _xyxy(cx, cy, w, h)
        ukf.correct(out_box.copy())
        mc.correct(out_box.copy())
        d_ukf = abs(ukf.x[idx, 0] - x_pre[idx, 0])
        d_mc = abs(mc.x[idx, 0] - x_pre[idx, 0])
        assert d_ukf > 1e-3
        sup[ch] = d_mc / d_ukf
    # 白化契约：3.5σ 的 cx 应被强烈抑制，2.86σ 的 w 抑制明显更弱；
    # 若实现按原始残差（未白化）加权，两通道抑制率应几乎相等。
    # （c_cx = exp(-3.5^2/2) ≈ 2.2e-3 > c_min，未被 clip，判别有效。）
    assert sup["cx"] < 0.1, f"cx 抑制率应很小，实际 {sup['cx']}"
    assert sup["w"] > 3.0 * sup["cx"], (
        f"白化失效：相同原始偏移下 cx 抑制率 {sup['cx']:.4f} 未显著小于 w {sup['w']:.4f}"
    )


# ======================================================================
# 契约 4：真实 IRLS —— max_iter=1 与 max_iter=3 在中等离群下结果不同
# ======================================================================

def test_max_iter_one_vs_three_differs_on_moderate_outlier():
    std = _diag_r(_DetectionMCUKF(INIT_BOX.copy()))
    # 归一化中等离群 k=2：c = exp(-2) ≈ 0.135（clip 区间内）。
    # 参考实现自检：gap(1,3) ≈ 7.8e-2 且 gap(3,6) ≈ 1.2e-2 未收敛，
    # 说明该场景下迭代次数真实影响不动点结果。
    k = 2.0
    out_box = _xyxy(0.30 + k * std[0], 0.30, INIT_W, INIT_H)

    # 场景自检：契约参考实现下 1 次与 3 次迭代确实给出不同结果
    probe = _DetectionMCUKF(INIT_BOX.copy())
    probe.predict()
    r = probe._make_r(INIT_W, INIT_H)
    z = np.array([[0.30 + k * std[0]], [0.30], [INIT_W], [INIT_H]])
    x_ref1, _ = _reference_irls(probe.x.copy(), probe.p.copy(), z, r, sigma=1.0, iters=1)
    x_ref3, _ = _reference_irls(probe.x.copy(), probe.p.copy(), z, r, sigma=1.0, iters=3)
    ref_gap = np.linalg.norm(x_ref3 - x_ref1)
    assert ref_gap > 1e-3, f"场景自检失败：参考实现 1/3 迭代无差异（{ref_gap}）"

    mc1 = _DetectionMCUKF(INIT_BOX.copy(), max_iter=1)
    mc3 = _DetectionMCUKF(INIT_BOX.copy(), max_iter=3)
    mc1.predict()
    mc3.predict()
    assert np.allclose(mc1.x, mc3.x) and np.allclose(mc1.p, mc3.p)
    mc1.correct(out_box.copy())
    mc3.correct(out_box.copy())
    gap = np.linalg.norm(mc3.x - mc1.x)
    assert gap > 1e-3, (
        f"max_iter=1 与 max_iter=3 在中等离群下结果相同（gap={gap}），"
        f"疑似未做真实 IRLS 迭代"
    )


def test_max_iter_three_matches_reference_irls():
    """max_iter=3 的输出应与契约参考实现（3 次真实不动点迭代）一致。"""
    std = _diag_r(_DetectionMCUKF(INIT_BOX.copy()))
    k = 2.0  # 同 test_max_iter_one_vs_three，取迭代敏感区
    z_cx = 0.30 + k * std[0]
    mc = _DetectionMCUKF(INIT_BOX.copy(), max_iter=3)
    mc.predict()
    x_pre, p_pre = mc.x.copy(), mc.p.copy()
    r = mc._make_r(INIT_W, INIT_H)
    z = np.array([[z_cx], [0.30], [INIT_W], [INIT_H]])
    x_ref, p_ref = _reference_irls(x_pre, p_pre, z, r, sigma=1.0, iters=3)
    mc.correct(_xyxy(z_cx, 0.30, INIT_W, INIT_H))
    assert np.allclose(mc.x, x_ref, rtol=1e-4, atol=1e-6), (
        f"max_iter=3 状态与契约参考实现不符: max|dx|={np.abs(mc.x - x_ref).max()}"
    )
    assert np.allclose(mc.p, p_ref, rtol=1e-3, atol=1e-8), (
        f"max_iter=3 协方差与契约 Joseph/R_eff 参考不符: "
        f"max|dP|={np.abs(mc.p - p_ref).max()}"
    )


# ======================================================================
# 长序列鲁棒性：噪声 + 周期性离群后 P 有限、对称、无负特征值
# ======================================================================

def test_long_sequence_with_noise_and_outliers_p_stays_valid():
    mc = _DetectionMCUKF(INIT_BOX.copy())
    rng = np.random.default_rng(23)
    p_max_vec = _DetectionUKF._P_MAX_VEC
    for f in range(1, 201):
        cx = 0.30 + 0.004 * f + rng.normal(0, 0.01)
        cy = 0.30 + 0.002 * f + rng.normal(0, 0.01)
        w = INIT_W + rng.normal(0, 0.004)
        h = INIT_H + rng.normal(0, 0.004)
        if f % 13 == 0:
            # 周期性单通道离群（8 倍位置标准差）
            cx += 8.0 * 0.1
        mc.predict()
        mc.correct(_xyxy(cx, cy, w, h))
        assert np.all(np.isfinite(mc.x)), f"第 {f} 帧状态出现非有限值"
        _assert_psd(mc.p, eps=1e-8)
        assert np.all(np.diag(mc.p) <= p_max_vec + 1e-9), (
            f"第 {f} 帧 P 对角线超出分量上界"
        )
    # 滤波器仍正常跟踪（未被离群带崩）
    assert abs(mc.x[0, 0] - (0.30 + 0.004 * 200)) < 0.1
    assert 1e-5 < mc.x[4, 0] < 0.05
