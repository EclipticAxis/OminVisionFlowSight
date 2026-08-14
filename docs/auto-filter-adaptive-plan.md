# AUTO 自适应滤波器切换 — 实现方案

## 一、需求

在 GUI 设置的滤波器类型下拉框中新增 **AUTO（自适应切换）** 选项。选中后，系统自动根据运行时环境在 UKF / MCUKF / Manifold UKF 三种滤波器间智能切换。

## 二、自适应切换原理

### 2.1 监控信号

| 信号 | 含义 | 来源 | 切换触发 |
|---|---|---|---|
| **归一化新息平方 (NIS)** | `‖z - z_pred‖² / trace(S)`，衡量观测与预测的偏差 | UKF `correct()` 内部 | NIS 高 → 存在离群值 → **MCUKF** |
| **缺失率** | `miss_count / total_updates`，衡量跟踪稳定性 | `DetectionTracker.update()` | 缺失率高 → 遮挡/噪声 → **MCUKF** |
| **P 条件数** | `max(λ) / min(λ)`，衡量协方差数值稳定性 | `np.linalg.cond(P)` | 条件数高 → P 接近奇异 → **Manifold UKF** |
| **Cholesky 失败计数** | `_sigma_points()` 中 Cholesky 分解失败次数 | UKF `_sigma_points()` | 频繁失败 → **Manifold UKF** |

### 2.2 切换决策树

```
每 EVAL_INTERVAL=30 次更新评估一次（约 1 秒 @ 30fps）：

  if P 条件数 > 500  OR  Cholesky 失败次数 > 2:
      → target = manifold_ukf    # P 不稳定，需要 SPD 保证
  elif 平均 NIS > 5.0  OR  缺失率 > 40%:
      → target = mcukf           # 离群值多/遮挡严重，需要鲁棒性
  else:
      → target = ukf             # 环境稳定，用最快的基础 UKF

防抖：需连续 HYSTERESIS=2 次评估都指向同一非当前类型才切换
```

### 2.3 NIS 阈值依据

NIS（Normalized Innovation Squared）在卡方分布下：
- 4 维观测空间，95% 置信区间上界 = 9.49
- 取 5.0 作为保守阈值（略低于 95% 置信区间），让系统对中等程度的离群值就切换到 MCUKF

## 三、架构设计

### 3.1 新增 `_FilterAdaptor` 类

```python
class _FilterAdaptor:
    """自适应滤波器选择器：基于运行时统计自动切换 UKF/MCUKF/Manifold UKF。"""

    _EVAL_INTERVAL = 30       # 每 30 次更新评估一次
    _NIS_THRESHOLD = 5.0      # 平均 NIS 阈值
    _COND_THRESHOLD = 500.0   # P 条件数阈值
    _MISS_RATE_THRESHOLD = 0.4  # 缺失率阈值
    _CHOLESKY_FAIL_THRESHOLD = 2  # Cholesky 失败次数阈值
    _HYSTERESIS = 2           # 连续 N 次评估一致才切换（防抖）

    def __init__(self, kernel_sigma: float = 0.4):
        self._kernel_sigma = kernel_sigma
        self._current_type = "ukf"
        self._nis_window: list[float] = []
        self._miss_count = 0
        self._total_count = 0
        self._cholesky_fails = 0
        self._update_count = 0
        self._switch_votes: dict[str, int] = {}

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

    @property
    def current_type(self) -> str:
        return self._current_type

    def evaluate(self, p_cond: float) -> str | None:
        """评估是否需要切换。返回新类型或 None（保持当前）。

        p_cond: 当前活跃 track 的 P 矩阵条件数中位数
        """
        if self._update_count < self._EVAL_INTERVAL:
            return None

        # 计算窗口统计
        recent_nis = self._nis_window[-self._EVAL_INTERVAL:]
        mean_nis = float(np.mean(recent_nis)) if recent_nis else 0.0
        miss_rate = self._miss_count / max(1, self._total_count)

        # 决策树
        if p_cond > self._COND_THRESHOLD or self._cholesky_fails > self._CHOLESKY_FAIL_THRESHOLD:
            target = "manifold_ukf"
        elif mean_nis > self._NIS_THRESHOLD or miss_rate > self._MISS_RATE_THRESHOLD:
            target = "mcukf"
        else:
            target = "ukf"

        # 防抖
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
```

### 3.2 UKF 基类新增创新息记录

在 `_DetectionUKF.correct()` 中，计算完 `z_pred` 和 `S` 后，记录 NIS：

```python
# 在 correct() 中新增（z_pred 和 S 计算完成后）
innovation = z - z_pred                          # (4, 1)
self.last_innovation_sq = float(innovation.T @ innovation)
self.last_s_trace = float(np.trace(S))
```

在 `_DetectionUKF._sigma_points()` 的 Cholesky 失败分支中，设置标志：
```python
except np.linalg.LinAlgError:
    self.cholesky_failed = True  # 新增标志
    self.p = np.eye(self._N) * 0.02
    L = np.linalg.cholesky(self.p)
```

### 3.3 `_Track` 新增 `swap_filter()` 热切换方法

```python
def swap_filter(self, filter_type: str, kernel_sigma: float = 0.4) -> None:
    """热切换滤波器类型，保留状态 (x, P, Q, R)。

    用于 AUTO 模式下自适应切换：创建新类型滤波器实例，
    将旧滤波器的状态复制过去，实现无缝切换。
    """
    old = self.kalman
    if filter_type == "mcukf":
        new = _DetectionMCUKF(self.box, kernel_sigma=kernel_sigma)
    elif filter_type == "manifold_ukf":
        new = _DetectionManifoldUKF(self.box)
    else:
        new = _DetectionUKF(self.box)
    # 保留状态
    new.x = old.x.copy()
    new.p = old.p.copy()
    new.q = old.q.copy()
    new.r = old.r.copy()
    self.kalman = new
```

### 3.4 `DetectionTracker` 集成

```python
class DetectionTracker:
    def __init__(self, ..., filter_type: str = "ukf", ...):
        ...
        self._filter_type = filter_type
        self._adaptor: _FilterAdaptor | None = None
        if filter_type == "auto":
            self._adaptor = _FilterAdaptor(kernel_sigma=self._mcukf_kernel_sigma)

    def _get_effective_filter_type(self) -> str:
        """获取当前应使用的滤波器类型（AUTO 模式下由适配器决定）。"""
        if self._adaptor is not None:
            return self._adaptor.current_type
        return self._filter_type

    def _update_track(self, tid, det, new_box):
        trk = self._tracks[tid]
        corrected_box = trk.kalman.correct(new_box)
        # AUTO 模式：记录 NIS
        if self._adaptor is not None:
            nis = trk.kalman.last_innovation_sq / max(trk.kalman.last_s_trace, 1e-10)
            self._adaptor.record_correct(nis)
            if getattr(trk.kalman, 'cholesky_failed', False):
                self._adaptor.record_cholesky_fail()
                trk.kalman.cholesky_failed = False
        ...

    def update(self, detections, redetect_callback=None):
        ...
        # AUTO 模式：记录 miss 统计
        if self._adaptor is not None:
            for tid, trk in self._tracks.items():
                if tid not in matched_track_ids:
                    self._adaptor.record_miss()
        ...
        # AUTO 模式：评估切换
        if self._adaptor is not None:
            conds = [np.linalg.cond(t.kalman.p) for t in self._tracks.values()]
            median_cond = float(np.median(conds)) if conds else 1.0
            new_type = self._adaptor.evaluate(median_cond)
            if new_type is not None:
                # 热切换所有 track 的滤波器
                for trk in self._tracks.values():
                    trk.swap_filter(new_type, self._mcukf_kernel_sigma)
        return self._active_tracks()
```

### 3.5 `_Track.__init__` 使用 effective filter type

新建 track 时，使用 `_get_effective_filter_type()` 而非 `self._filter_type`：
```python
self._tracks[tid] = _Track(tid, ..., 
                           filter_type=self._get_effective_filter_type(),
                           mcukf_kernel_sigma=self._mcukf_kernel_sigma)
```

## 四、文件改动清单

| 文件 | 改动 |
|---|---|
| `ai/tracker.py` | 新增 `_FilterAdaptor` 类；`_DetectionUKF` 新增 `last_innovation_sq`/`last_s_trace`/`cholesky_failed` 属性；`_Track` 新增 `swap_filter()` 方法；`DetectionTracker` 集成适配器 |
| `ai/inference.py` | 无改动（`set_filter_type("auto")` 已支持字符串透传） |
| `gui/settings_dialog.py` | 下拉框新增 "AUTO（自适应切换）" 选项；`load_ai_filter_type()` 新增 "auto" 合法值 |
| `gui/main_window.py` | 无改动（`set_filter_type` 透传 "auto"） |

## 五、GUI 下拉框选项

```
UKF（标准无迹卡尔曼）           → "ukf"
MCUKF（最大相关熵，抗离群）      → "mcukf"
Manifold UKF（流形，SPD 稳定）   → "manifold_ukf"
AUTO（自适应切换）              → "auto"        ← 新增
```

## 六、切换流程图

```
AUTO 模式运行时：

  每次推理帧 update():
    ├─ track 匹配成功 → kalman.correct() → 记录 NIS
    ├─ track 未匹配   → 记录 miss
    └─ update 结束后:
        if update_count >= 30:
            计算 median(P 条件数)
            evaluate(median_cond):
                P 不稳定 → target = manifold_ukf
                NIS 高/miss 多 → target = mcukf
                正常 → target = ukf
            if target != current AND 连续 2 次一致:
                hot-swap 所有 track 的滤波器（保留 x, P, Q, R）
                重置统计窗口
```

## 七、关键设计决策

1. **热切换保留状态**：`swap_filter()` 创建新滤波器实例后，将旧实例的 `x/P/Q/R` 复制过去，确保切换时无状态丢失
2. **防抖机制**：连续 2 次评估（约 2 秒）都指向同一非当前类型才切换，避免在边界条件下来回震荡
3. **评估频率**：每 30 次更新评估一次（约 1 秒 @ 30fps），平衡响应速度和计算开销
4. **P 条件数**：使用 `np.linalg.cond(P)` 计算所有活跃 track 的中位数，避免单个异常 track 触发误切换
5. **NIS 归一化**：用 `‖innovation‖² / trace(S)` 而非裸残差，自适应不同观测噪声水平
6. **Cholesky 失败计数**：Cholesky 分解失败是 P 退化的直接信号，计入 Manifold UKF 切换依据
