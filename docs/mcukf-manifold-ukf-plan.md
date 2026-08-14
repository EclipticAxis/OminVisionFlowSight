# MCUKF + Manifold UKF 实现方案

## 一、需求

在现有 `_DetectionUKF`（标准无迹卡尔曼滤波）基础上，新增两种高级滤波器：

| 滤波器 | 全称 | 核心优势 | 适用场景 |
|---|---|---|---|
| **MCUKF** | Maximum Correntropy UKF（最大相关熵 UKF） | 对非高斯脉冲噪声、离群检测框鲁棒 | YOLO 误检/遮挡跳变 |
| **Manifold UKF** | Manifold UKF（流形 UKF） | 协方差矩阵 P 始终保持正定，数值稳定性更强 | 长时间跳帧/高维状态 |

## 二、算法原理

### 2.1 MCUKF — 最大相关熵 UKF

**核心思想**：标准 UKF 使用最小均方误差（MMSE）准则，对高斯噪声最优但对离群值敏感。MCUKF 用最大相关熵准则（MCC）替代 MMSE，通过高斯核函数自动降低大误差观测的权重。

**相关熵定义**：
```
V(X, Y) = E[κ(X - Y)]，其中 κ(e) = exp(-e² / (2σ²))
```

**与标准 UKF 的关键区别**（仅在 update 步骤）：

| 步骤 | 标准 UKF | MCUKF |
|---|---|---|
| 优化准则 | MMSE | MCC（高斯核加权） |
| 观测权重 | 均匀 Wmⁱ | 相关熵加权 G_σ(eⁱ) |
| Kalman 增益 | 固定协方差驱动 | 相关熵调制协方差 |
| 求解方式 | 闭式解 | **不动点迭代**（2-3 次） |
| 对离群值 | 敏感 | 鲁棒（自动降权） |

**MCUKF update 算法**：

```
1. 生成 sigma 点（与 UKF 相同）
2. 观测传播：Zⁱ = h(Xⁱ)，i = 0..2n
3. 计算每个 sigma 点的残差：eⁱ = z - Zⁱ
4. 计算相关熵权重：wⁱ = exp(-||eⁱ||² / (2σ²))
5. 归一化权重：wⁱ = wⁱ / Σwʲ
6. 加权观测均值：z_pred = Σ wⁱ · Zⁱ
7. 加权观测协方差：S = Σ wⁱ · (Zⁱ - z_pred)(Zⁱ - z_pred)ᵀ + R
8. 加权交叉协方差：Pxz = Σ wⁱ · (Xⁱ - x_pred)(Zⁱ - z_pred)ᵀ
9. Kalman 增益：K = Pxz · S⁻¹
10. 状态更新：x = x_pred + K · (z - z_pred)
11. [迭代步骤 3-10，直到收敛或达到最大迭代次数]
12. 协方差更新（Joseph 形式）：P = (I - KH)P(I - KH)ᵀ + KRKᵀ
```

**关键参数 — 核带宽 σ**：
- σ 大 → 退化为标准 UKF（鲁棒性低，收敛快）
- σ 小 → 强鲁棒（可能收敛慢）
- **推荐值**：σ = 2~5 × 观测噪声标准差。当前 R = 0.012·I，std ≈ 0.11，故 σ ≈ 0.3~0.5

### 2.2 Manifold UKF — 流形 UKF

**核心思想**：标准 UKF 在欧氏空间中操作协方差矩阵 P，但 P 本质上生活在 SPD（对称正定）流形 S++^n 上。标准算术运算（加减、算术平均）不尊重流形结构，可能导致 P 失去正定性。

**关键流形运算**（通过特征分解实现，纯 NumPy）：

```
矩阵平方根：P^{1/2} = V · diag(√λ) · Vᵀ    （特征分解 P = VΛVᵀ）
矩阵对数：  logm(P) = V · diag(log(λ)) · Vᵀ
矩阵指数：  expm(S) = V · diag(exp(λ)) · Vᵀ
```

**与标准 UKF 的关键区别**：

| 方面 | 标准 UKF | Manifold UKF |
|---|---|---|
| sigma 点分解 | Cholesky(P) → L | **特征分解** P = VΛVᵀ → P^{1/2} |
| 均值计算 | 算术平均 | 算术平均（状态在欧氏空间） |
| 协方差传播 | F·P·Fᵀ + Q | **P^{1/2} · Δ · P^{1/2}** （SPD 保持） |
| P 更新后 | 仅对称化 + 裁剪 | **特征值投影** → 强制 SPD |
| 数值稳定性 | Cholesky 可能失败 | 特征分解更鲁棒 |

**Manifold UKF 算法**：

```
sigma 点生成：
  P = VΛVᵀ  （特征分解）
  P_sqrt = V · diag(√λ) · Vᵀ
  X⁰ = x
  Xⁱ = x + γ · P_sqrt[:, i]      (i=1..n)
  Xⁱ = x - γ · P_sqrt[:, i-n]    (i=n+1..2n)

predict（与 UKF 相同的 sigma 传播 + 无迹变换）

correct 后的 SPD 投影：
  P_new = (P + Pᵀ) / 2                     # 对称化
  λ, V = eig(P_new)                         # 特征分解
  λ = clip(λ, min=1e-8, max=_P_MAX)        # 裁剪特征值到正定区间
  P_new = V · diag(λ) · Vᵀ                  # 重建 SPD 矩阵
```

**关键设计决策**：状态向量 [cx,cy,w,h,vx,vy,vw,vh] 本身是欧氏的（bbox 坐标），不需要流形运算。流形运算仅应用于**协方差矩阵 P**，保证其始终在 SPD 流形上。这避免了昂贵的 Fréchet 均值迭代，同时获得 SPD 保证的核心收益。

## 三、架构设计

### 3.1 类继承结构

```
_DetectionUKF (现有，标准 UKF)
    │
    ├── _DetectionMCUKF (新增)
    │   └── 重写 correct()：相关熵加权 + 不动点迭代
    │
    └── _DetectionManifoldUKF (新增)
        └── 重写 _sigma_points()：特征分解替代 Cholesky
        └── 重写 correct()：SPD 投影
        └── 重写 predict()：SPD 投影
```

所有三个类共享相同接口：`predict()` → `[x1,y1,x2,y2]`，`correct(box)` → `[x1,y1,x2,y2]`，`to_box()` → `[x1,y1,x2,y2]`

### 3.2 滤波器选择

```python
# DetectionTracker 新增参数
def __init__(self, ..., filter_type: str = "ukf"):
    # filter_type: "ukf" | "mcukf" | "manifold_ukf"
```

```python
# _Track.__init__ 根据类型创建滤波器
if filter_type == "mcukf":
    self.kalman = _DetectionMCUKF(box, kernel_sigma=0.4)
elif filter_type == "manifold_ukf":
    self.kalman = _DetectionManifoldUKF(box)
else:
    self.kalman = _DetectionUKF(box)
```

### 3.3 配置透传

```
SettingsDialog → MainWindow._apply_ai_settings → InferWorker.set_filter_type → 
    遍历 _trackers → tracker.set_filter_type() → 下一帧 _Track 创建时使用新类型
```

## 四、文件改动清单

| 文件 | 改动 |
|---|---|
| `ai/tracker.py` | 新增 `_DetectionMCUKF` 类、`_DetectionManifoldUKF` 类；`DetectionTracker` 新增 `filter_type` 参数 + `set_filter_type()` 方法；`_Track` 接受 `filter_type` |
| `ai/inference.py` | `InferWorker` 新增 `_filter_type` 字段 + `set_filter_type()` setter；`register_slot()` 传入类型 |
| `gui/settings_dialog.py` | 新增"滤波器类型"下拉选择（UKF / MCUKF / Manifold UKF） |
| `gui/main_window.py` | `_apply_ai_settings()` 透传 filter_type |

## 五、`ai/tracker.py` 详细设计

### 5.1 `_DetectionMCUKF`

```python
class _DetectionMCUKF(_DetectionUKF):
    """最大相关熵 UKF：用高斯核加权降低离群观测的影响。

    与标准 UKF 的区别仅在 correct()：
    - 每个 sigma 点的残差通过高斯核 exp(-e²/2σ²) 加权
    - 不动点迭代（默认 3 次）收敛状态估计
    - 协方差更新用 Joseph 形式保证数值稳定性
    """

    def __init__(self, box: np.ndarray, kernel_sigma: float = 0.4, max_iter: int = 3):
        super().__init__(box)
        self._kernel_sigma = float(kernel_sigma)
        self._max_iter = int(max_iter)

    def correct(self, box: np.ndarray) -> np.ndarray:
        cx, cy, w, h = _box_to_cxcywh(box)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)

        sigmas = self._sigma_points()
        z_sigmas = sigmas[0:4, :]                    # (4, 17) 观测传播

        x_orig = self.x.copy()                       # 保存预测状态
        p_pred = self.p.copy()                       # 保存预测协方差

        for iteration in range(self._max_iter):
            # 计算每个 sigma 点的残差
            residuals = z - z_sigmas                 # (4, 17)
            residual_norms = np.sum(residuals ** 2, axis=0)  # (17,) 每点的 ||e_i||²

            # 高斯核加权
            kernel_weights = np.exp(-residual_norms / (2.0 * self._kernel_sigma ** 2))
            kernel_weights[0] = max(kernel_weights[0], 1e-10)  # 中心点保底
            total = kernel_weights.sum()
            if total < 1e-10:
                kernel_weights = np.ones_like(kernel_weights) / len(kernel_weights)
            else:
                kernel_weights = kernel_weights / total

            # 加权观测均值
            z_pred = (kernel_weights * z_sigmas).sum(axis=1, keepdims=True)  # (4, 1)

            # 加权观测协方差 S
            dz = z_sigmas - z_pred                   # (4, 17)
            S = (dz * kernel_weights) @ dz.T + self.r

            # 加权交叉协方差 Pxz
            dx = sigmas - self.x                     # (8, 17)
            Pxz = (dx * kernel_weights) @ dz.T       # (8, 4)

            # Kalman 增益
            K = Pxz @ np.linalg.inv(S)               # (8, 4)

            # 状态更新
            self.x = x_orig + K @ (z - z_pred)

            # 收敛检查
            if iteration > 0 and np.linalg.norm(self.x - x_prev) < 1e-6:
                break
            x_prev = self.x.copy()

        # Joseph 形式协方差更新（数值稳定）
        H = np.zeros((4, 8), dtype=np.float64)
        H[:4, :4] = np.eye(4)
        I_KH = np.eye(8) - K @ H
        self.p = I_KH @ p_pred @ I_KH.T + K @ self.r @ K.T

        # SPD 投影
        self.p = (self.p + self.p.T) * 0.5
        np.minimum(self.p, self._P_MAX, out=self.p)
        return self.to_box()
```

### 5.2 `_DetectionManifoldUKF`

```python
class _DetectionManifoldUKF(_DetectionUKF):
    """流形 UKF：协方差 P 始终保持在对称正定 (SPD) 流形上。

    与标准 UKF 的区别：
    - sigma 点用特征分解 P^{1/2} 替代 Cholesky 分解
    - 每次 predict/correct 后做 SPD 投影（特征值裁剪）
    - 更鲁棒的数值稳定性，P 永远不会失去正定性
    """

    def _sigma_points(self) -> np.ndarray:
        """用特征分解替代 Cholesky 生成 sigma 点。"""
        # 对称化
        p_sym = (self.p + self.p.T) * 0.5
        # 特征分解
        eigenvalues, eigenvectors = np.linalg.eigh(p_sym)
        # 裁剪负特征值（保证 SPD）
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
        """SPD 投影：特征值裁剪到 [1e-8, _P_MAX]。"""
        p_sym = (self.p + self.p.T) * 0.5
        eigenvalues, eigenvectors = np.linalg.eigh(p_sym)
        eigenvalues = np.clip(eigenvalues, 1e-8, self._P_MAX)
        self.p = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def predict(self) -> np.ndarray:
        """与 UKF 相同的 sigma 传播，但用 SPD 投影替代简单裁剪。"""
        sigmas = self._sigma_points()
        sigmas_pred = sigmas.copy()
        sigmas_pred[0:4, :] += sigmas[4:8, :]
        self.x = (self._wm * sigmas_pred).sum(axis=1, keepdims=True)
        d = sigmas_pred - self.x
        self.p = (d * self._wc) @ d.T + self.q
        self._project_spd()                          # SPD 投影替代简单裁剪
        return self.to_box()

    def correct(self, box: np.ndarray) -> np.ndarray:
        """与 UKF 相同的更新，但用 SPD 投影替代简单对称化。"""
        cx, cy, w, h = _box_to_cxcywh(box)
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float64)
        sigmas = self._sigma_points()
        z_sigmas = sigmas[0:4, :]
        z_pred = (self._wm * z_sigmas).sum(axis=1, keepdims=True)
        dz = z_sigmas - z_pred
        S = (dz * self._wc) @ dz.T + self.r
        dx = sigmas - self.x
        Pxz = (dx * self._wc) @ dz.T
        K = Pxz @ np.linalg.inv(S)
        self.x = self.x + K @ (z - z_pred)
        self.p = self.p - K @ S @ K.T
        self._project_spd()                          # SPD 投影
        return self.to_box()
```

### 5.3 `DetectionTracker` 改造

```python
class DetectionTracker:
    def __init__(
        self,
        ...,
        filter_type: str = "ukf",          # 新增
        mcukf_kernel_sigma: float = 0.4,   # 新增
    ):
        ...
        self._filter_type = filter_type
        self._mcukf_kernel_sigma = mcukf_kernel_sigma

    def set_filter_type(self, filter_type: str, kernel_sigma: float | None = None) -> None:
        self._filter_type = filter_type
        if kernel_sigma is not None:
            self._mcukf_kernel_sigma = kernel_sigma
        # 已有 track 的滤波器不热切换，下一帧新建 track 时生效
```

`_Track.__init__` 根据 `filter_type` 创建对应滤波器实例。

## 六、配置项设计

| 配置项 | QSettings Key | 类型 | 默认 | 选项 |
|---|---|---|---|---|
| 滤波器类型 | `ai/filter_type` | str | "ukf" | ukf / mcukf / manifold_ukf |
| MCUKF 核带宽 | `ai/mcukf_kernel_sigma` | float | 0.4 | 0.1 ~ 2.0 |

GUI 下拉框：`UKF（标准）` / `MCUKF（鲁棒抗离群）` / `Manifold UKF（SPD 稳定）`

## 七、性能对比预估

| 指标 | UKF | MCUKF | Manifold UKF |
|---|---|---|---|
| sigma 点 | 17 | 17 | 17 |
| predict 开销 | ~0.1μs | ~0.1μs（相同） | ~0.3μs（eigh 分解） |
| correct 开销 | ~0.1μs | ~0.5μs（3 次迭代） | ~0.3μs（eigh 投影） |
| 总单帧开销 | ~0.2μs | ~0.6μs | ~0.6μs |
| 对离群值鲁棒性 | 低 | **高** | 中 |
| P 正定性保证 | Cholesky 回退 | Cholesky 回退 | **特征值投影** |
| 适合场景 | 通用 | 噪声大/遮挡多 | 长时间跳帧 |

> 所有滤波器开销相对 YOLO 推理（~10ms）均可忽略（< 0.01%）。

## 八、风险与缓解

| 风险 | 滤波器 | 缓解 |
|---|---|---|
| 不动点迭代不收敛 | MCUKF | 限制 max_iter=3；收敛阈值 1e-6；退化时回退到标准 UKF 更新 |
| 核带宽 σ 过小导致拒绝有效观测 | MCUKF | 默认 σ=0.4（约 3.6× 噪声 std）；设置下限 0.1 |
| 特征分解开销 | Manifold UKF | n=8 矩阵 eigh ~0.3μs，可接受 |
| 滤波器热切换不生效 | 全部 | 已有 track 保持旧滤波器，新 track 使用新类型；调 set_filter_type 后 reset() 可强制全部重建 |
| Joseph 形式计算量 | MCUKF | 仅多一次矩阵乘法，n=8 可忽略 |

## 九、实现顺序

1. `ai/tracker.py` — 新增 `_DetectionMCUKF` + `_DetectionManifoldUKF`；`DetectionTracker` 加 `filter_type` 参数
2. `ai/inference.py` — `InferWorker` 加 `set_filter_type()` + `register_slot` 透传
3. `gui/settings_dialog.py` — 滤波器类型下拉框 + MCUKF σ 配置
4. `gui/main_window.py` — 透传配置
5. 单元测试 — 三种滤波器对比验证（离群值鲁棒性、SPD 保持、性能）
