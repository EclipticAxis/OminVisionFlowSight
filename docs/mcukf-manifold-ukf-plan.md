# MCUKF + Manifold UKF 实现方案

> **勘误（2026-11-25，修复分支待 PR 合并）**：本文 §2.1 / §5.1 / §6 中 MCUKF 的伪码与 σ 语义**已被新实现取代**。旧写法把归一化凸核权误当 UT 的 Wm/Wc 使用，导致增益坍缩与假迭代；σ 按量测噪声 std 倍数取值的推荐（默认 0.4）随之失效。权威行为契约见 [mcukf-irls-correction](mcukf-irls-correction.md)。Manifold UKF（§2.2 / §5.2）部分不受影响。

## 一、需求

在现有 `_DetectionUKF`（标准无迹卡尔曼滤波）基础上，新增两种高级滤波器：

| 滤波器 | 全称 | 核心优势 | 适用场景 |
|---|---|---|---|
| **MCUKF** | Maximum Correntropy UKF（最大相关熵 UKF） | 对非高斯脉冲噪声、离群检测框鲁棒 | YOLO 误检/遮挡跳变 |
| **Manifold UKF** | Manifold UKF（流形 UKF） | 协方差矩阵 P 始终保持正定，数值稳定性更强 | 长时间跳帧/高维状态 |

## 二、算法原理

### 2.1 MCUKF — 最大相关熵 UKF

> **（2026-11-25 勘误）本节原始伪码已被取代**：旧伪码把归一化凸核权当作 UT 的 Wm/Wc 重建 z_pred / S / Pxz，导致增益坍缩且迭代为"假迭代"。以下为修正后的行为摘要，完整契约（含退化路径、边界声明）见 [mcukf-irls-correction](mcukf-irls-correction.md)。

**核心思想**：标准 UKF 使用最小均方误差（MMSE）准则，对高斯噪声最优但对离群值敏感。MCUKF 用最大相关熵准则（MCC）替代 MMSE：保留标准 UT 结构不变，把相关熵作用于**量测噪声协方差**（IRLS 视角），等效地对离群量测"涨 R"、缩小其增益。

**相关熵定义**：
```
V(X, Y) = E[κ(X - Y)]，其中 κ(e) = exp(-e² / (2σ²))
```

**与标准 UKF 的关键区别**（仅在一次量测的 update 内部）：

| 步骤 | 标准 UKF | MCUKF（IRLS 修正版） |
|---|---|---|
| 优化准则 | MMSE | MCC（IRLS 重加权最小二乘） |
| z_pred / Pzz / Pxz | UT 权 Wm/Wc 加权 | **与标准 UKF 完全相同**（不被核权污染） |
| 量测噪声 | 恒为 R | R_eff = R·C⁻¹，逐通道相关熵降权 |
| 残差空间 | 原始量测空间 | 白化残差 y = R^{-1/2}·residual_t；residual_t = z − (z_pred + H(x_iter − x_pred)) 每轮重算（σ 无量纲） |
| Kalman 增益 | K = Pxz·(Pzz+R)⁻¹ | K = Pxz·(Pzz+R_eff)⁻¹ |
| 求解方式 | 闭式解 | IRLS，最多 3 次迭代，位移 1e-6 收敛 |
| 协方差更新 | Joseph（R） | Joseph（**同一 R_eff**） |
| 对离群值 | 敏感 | 鲁棒（K → 0，P → 预测协方差） |

**修正后 update 算法**：

```
1. 生成 sigma 点 + 观测传播（与 UKF 相同）
2. z_pred / Pzz / Pxz ← 标准 UT（Wm/Wc 不变）
3. S0 = Pzz + R              ← AUTO / NIS 永远使用 S0
4. IRLS 迭代（最多 3 次；x_iter 首轮 = x_pred）：
   residual_t = z - (z_pred + H·(x_iter - x_pred))  # 当前迭代点残差，每轮重算
   y = R^{-1/2}·residual_t                          # 对角 R 白化
   c_j = clip(exp(-y_j²/(2σ²)), 1e-4, 1)            # 逐通道核权（c_min=1e-4 限条件数）
   R_eff = R·C⁻¹,  C = diag(c)
   K = Pxz·(Pzz + R_eff)⁻¹
   x_new = x_pred + K·(z - z_pred)   # 自 x_pred 重算，不累加
   若 ‖x_new - x_iter‖ < 1e-6 则收敛退出
   达上限 → 采用最后一轮的 x_new / K / R_eff
   x_iter ← x_new
5. 定稿 x = x_new；Joseph：P = (I-KH)·P_pred·(I-KH)ᵀ + K·R_eff·Kᵀ
   （K / R_eff 取最后一轮，与 x_new 同轮一致）
```

**关键参数 — 核带宽 σ（修正后语义）**：
- σ 是**白化残差空间的无量纲**参数，默认 **1.0**（不再是量测噪声 std 的倍数，旧推荐 0.4 作废）
- σ 大 → c_j → 1，退化为标准 UKF；σ 小 → 强鲁棒
- R 白化使正常帧也按 c_j<1 降权（幅度随信噪比变化），属**有意的鲁棒保守性**而非恒等 UKF；对比 S0 白化（需 σ≈2 才相当），R 白化 + σ=1.0 离群抑制更好，正常/真实突变恢复无显著劣势
- c_min = **1e-4**（非 1e-8）：限制 R_eff / S 条件数不过度极端
- GUI 范围 **0.5–3.0**；旧配置文件超出该范围时由 loader 回退到 1.0
- 普通噪声下 MCUKF **不恒等**于 UKF；σ 需按场景（噪声尺度 / 框抖动 / 误检率）标定，不存在全场景最优默认值

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
    self.kalman = _DetectionMCUKF(box, kernel_sigma=1.0)  # 2026-11-25 新无量纲语义
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

> **（2026-11-25 勘误）以下类伪码为旧实现，已被取代，仅留档**：其中把归一化凸核权当作 UT 权重建 z_pred / S / Pxz（增益坍缩、假迭代），且 σ 语义（0.4，按噪声 std 倍数）已作废。当前实现的行为契约（标准 UT + R 对角白化 IRLS，σ 无量纲默认 1.0）见 [mcukf-irls-correction](mcukf-irls-correction.md)。

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
        mcukf_kernel_sigma: float = 1.0,   # 新增（无量纲；勘误 2026-11-25，旧值 0.4 已作废）
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
| MCUKF 核带宽 | `ai/mcukf_kernel_sigma` | float | 1.0 | 0.5 ~ 3.0（无量纲；旧配置超出范围由 loader 回退 1.0） |

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
| IRLS 迭代达上限 | MCUKF | 最多 3 次；位移阈值 1e-6；达上限采用最后一次鲁棒迭代；大 σ / 零残差自然退化回标准 UKF |
| 核带宽 σ 设置不当（过小拒收正常观测 / 过大失去鲁棒性） | MCUKF | σ 无量纲默认 1.0，GUI 0.5–3.0；旧配置超范围由 loader 回退 1.0；按场景标定，不宣称全场景最优 |
| 特征分解开销 | Manifold UKF | n=8 矩阵 eigh ~0.3μs，可接受 |
| 滤波器热切换不生效 | 全部 | 已有 track 保持旧滤波器，新 track 使用新类型；调 set_filter_type 后 reset() 可强制全部重建 |
| Joseph 形式计算量 | MCUKF | 仅多一次矩阵乘法，n=8 可忽略 |

## 九、实现顺序

1. `ai/tracker.py` — 新增 `_DetectionMCUKF` + `_DetectionManifoldUKF`；`DetectionTracker` 加 `filter_type` 参数
2. `ai/inference.py` — `InferWorker` 加 `set_filter_type()` + `register_slot` 透传
3. `gui/settings_dialog.py` — 滤波器类型下拉框 + MCUKF σ 配置
4. `gui/main_window.py` — 透传配置
5. 单元测试 — 三种滤波器对比验证（离群值鲁棒性、SPD 保持、性能）
