# VisionBata 多模块集成后识别性能下降分析与恢复方案

> 背景：在集成双边滤波/NL-Means 去噪、MCUKF、Manifold UKF 及 AUTO 自适应切换后，人物识别、人体识别、手势识别及矩形识别的准确率与稳定性均出现明显下降。本文针对四大可能原因进行根因分析，并给出可落地的参数调优策略、特征融合方案与模块化隔离测试方法。

---

## 一、根因分析

### 1.1 滤波算法过度平滑导致特征边缘信息丢失

#### 现象
- 人物/手势检测框漂移、置信度下降；矩形检测漏检增加；骨架关键点抖动。
- 在**低对比度、小目标、快速运动**场景下尤为明显。

#### 机制
| 去噪方法 | 作用域 | 风险点 |
|---|---|---|
| 双边滤波 | 空间+颜色域 | `sigmaColor/sigmaSpace` 过大时，会把弱边缘（如手指、矩形细边框）与背景融合 |
| NL-Means | 全图 patch 相似性 | `h` 过高或降采样 2x 后上采样，会抹平高频细节，导致 Canny/边缘检测失效 |

对下游任务的影响链：

```
过度去噪 → 边缘/纹理弱化 → YOLO 特征响应降低 → 置信度下降/漏检
                    ↓
            矩形 Canny 边缘断裂 → 轮廓提取失败 → 矩形漏检
                    ↓
            关键点热力图峰值扩散 → 骨架关键点偏移 → 手势识别错误
```

#### 关键观察
当前 `_denoise_frame()` 在**所有推理帧**上全局应用去噪，且：
- `bilateral` 的 `sigma = 25 + (strength-1)*5`，strength=10 时 `sigma=70`，已远超常规推荐（5~50）。
- `nl_means` 的 `h = 3 + (strength-1)*1.2`，strength=10 时 `h=13.8`，对 720p 以上图像还会降采样 2x 处理，细节损失显著。
- **去噪作用于 YOLO 推理帧，但去噪后的帧同时被用于矩形检测（Canny）和手势 ROI 提取**，形成全局影响。

---

### 1.2 UKF 状态估计与目标检测特征空间不匹配

#### 现象
- 跟踪框滞后于真实目标；快速运动时目标丢失；ID 切换频繁。
- AUTO 模式在高动态场景下反复切换滤波器。

#### 机制
当前 `_DetectionUKF` 的状态为 `[cx, cy, w, h, vx, vy, vw, vh]`，运动模型为**匀速线性模型**（`dt=1`）。问题在于：

1. **运动模型过简**：真实人体/手势运动多为加速、转身、遮挡等非线性运动，匀速假设在快速动作下误差大。
2. **观测与状态空间维度不一致**：观测只有 `cx, cy, w, h`，但状态包含速度。速度估计在初始几帧不稳定，导致预测框外推过度。
3. **UKF sigma 点参数激进**：
   - `α=1.0, β=2, κ=0, λ=0`，中心 sigma 点权重 `Wm[0]=0`，意味着均值点被完全忽略。
   - 这导致协方差传播过度依赖分布尾部，对小样本（单帧）观测敏感。
4. **过程噪声 Q 与观测噪声 R 未按场景自适应**：
   - 固定 `Q=0.0015*I`, `R=0.012*I`。
   - 高分辨率/远距小目标需要更小的 R；遮挡/运动模糊时需要更大的 Q。
5. **MCUKF 核带宽 σ 可能过强**：
   - 默认 `σ=0.4`，归一化坐标下对应约 0.4 的 bbox 尺度。
   - 当目标真实变化较大（如坐下、挥手）时，会被误判为离群值而降权，造成响应滞后。

#### 关键观察
- `predict()` 在 `update()` 开始时无条件执行，若上一步 `correct()` 结果偏差大，下一步预测会进一步放大误差。
- `last_innovation_sq / trace(S)` 作为 NIS，在归一化坐标下阈值 `5.0` 可能过低，导致 AUTO 频繁切换到 MCUKF。

---

### 1.3 流形约束与识别模型假设冲突

#### 现象
- Manifold UKF 下跟踪框收缩或膨胀异常；长时间静止后突然运动时跟踪丢失。
- 矩形跟踪（线性卡尔曼）与人物跟踪（UKF）行为不一致。

#### 机制
`_DetectionManifoldUKF._project_spd()` 将协方差特征值裁剪到 `[1e-8, P_MAX=1.0]`：

1. **上界 P_MAX=1.0 过大**：协方差允许膨胀到接近满图尺度，导致不确定性被过度表达，sigma 点扩散极广。
2. **SPD 投影后未对位置/尺度分量分别约束**：`cx, cy` 与 `w, h` 共用同一特征值裁剪范围，尺度分量的不确定性容易“污染”位置估计。
3. **与 YOLO 输出分布不一致**：YOLO 的 bbox 误差通常与目标尺度相关（小目标位置噪声大，大目标相对小）。Manifold UKF 的各向同性约束与之冲突。
4. **与矩形检测器假设冲突**：矩形检测依赖边缘几何，`P` 的过度膨胀导致预测框覆盖背景区域，在 ROI 重检测时引入大量背景噪声。

#### 关键观察
- `np.minimum(self.p, self._P_MAX, out=self.p)` 是对**整个矩阵**逐元素裁剪，而非仅对角线方差，会截断协方差结构。
- `Manifold UKF` 的 `eigh` 分解每次 `predict/correct` 执行两次，计算开销大，在 AUTO 切换时可能引入延迟抖动。

---

### 1.4 多模块耦合引入的累积误差

#### 现象
- 单独关闭去噪或切换 UKF 时性能部分恢复，但多模块同时开启时性能急剧恶化。
- 错误会在“去噪 → 检测 → 跟踪 → 手势/矩形 ROI → 重检测”链路中逐级放大。

#### 耦合链路

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  全局去噪    │───▶│  YOLO 检测   │───▶│ Detection   │
│ (bilateral/ │    │ (person/pose)│    │ Tracker     │
│  nl_means)  │    └──────┬──────┘    │ (UKF/MCUKF/  │
└─────────────┘           │           │  Manifold)   │
                          │           └──────┬──────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐
                   │ 手势识别     │    │ ROI 重检测   │
                   │ (hand/body) │◀───│ (predict框) │
                   └─────────────┘    └─────────────┘
                          ▲
                          │
                   ┌──────┴──────┐
                   │ 矩形检测     │
                   │ (Canny/轮廓) │
                   └─────────────┘
```

#### 具体问题
1. **去噪帧作为唯一输入**：矩形检测与手势识别共享同一张去噪后的帧，而它们对噪声/边缘的需求不同。
2. **跟踪器预测框主导 ROI 重检测**：预测误差大时，ROI 裁剪偏离目标，重检测返回错误框，进一步污染卡尔曼状态。
3. **手势 ROI 基于跟踪后的人体关键点**：若跟踪框滞后或关键点偏移，手部 ROI 定位错误，手势识别失败。
4. **RectangleTracker 仍用线性卡尔曼**：其 `smooth_alpha` 与 `max_misses` 和 `DetectionTracker` 不同步，造成显示时人与矩形相对位置漂移。
5. **AUTO 切换未考虑业务场景**：在人物静止但摄像头轻微抖动时，NIS 可能被误判为异常，切换到 MCUKF 反而抑制正常响应。

---

## 二、针对性解决方案

### 2.1 去噪策略：分任务、自适应、可回退

#### 2.1.1 按任务选择输入
不要对全图统一去噪，应提供两条输入：

| 任务 | 推荐输入 | 原因 |
|---|---|---|
| YOLO 人物/姿态检测 | 轻去噪或无去噪 | 保留边缘与纹理，避免漏检 |
| 矩形检测 | 原始帧或极低强度双边 | Canny 需要清晰边缘 |
| 手势识别（hand） | 原始帧或轻 bilateral | 手部细节关键 |
| 远程/夜景小目标 | 低强度 NL-Means | 抑制 sensor 噪声同时保留结构 |

#### 2.1.2 参数调优建议
将当前 `strength 1~10` 线性映射改为**非线性保守映射**，并增加按分辨率自适应：

**bilateral 推荐参数**
```python
def _denoise_bilateral(frame, strength):
    # strength 1~10 → 保守映射
    d = 5 if strength <= 4 else 7  # 不再到 9
    sigma_color = 10.0 + strength * 3.0   # 13~40，而非 25~70
    sigma_space = 5.0 + strength * 2.0    # 7~25
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
```

**nl_means 推荐参数**
```python
def _denoise_nlmeans(frame, strength):
    h = 2.0 + strength * 0.8            # 2.8~10，而非 3~14
    template_size = 5 if frame.shape[0] > 720 else 7
    search_size = 15 if frame.shape[0] > 720 else 21
    # 720p 以上：仅在 strength>=7 时才降采样，且用 1.5x 而非 2x
    if frame.shape[0] > 720 and strength >= 7:
        scale = 1.5
        small = cv2.resize(frame, (int(frame.shape[1]/scale), int(frame.shape[0]/scale)))
        denoised = cv2.fastNlMeansDenoisingColored(small, None, h, h, template_size, search_size)
        return cv2.resize(denoised, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.fastNlMeansDenoisingColored(frame, None, h, h, template_size, search_size)
```

#### 2.1.3 增加去噪质量门控
在去噪前后分别对 YOLO 跑一次推理（仅在调试模式），计算去噪收益：

```python
def _denoise_gain_metric(frame_before, frame_after, model, conf):
    dets_before = self._predict_model(model, frame_before, conf)
    dets_after = self._predict_model(model, frame_after, conf)
    # 用 person 检测的平均置信度增益衡量
    conf_before = mean([d['confidence'] for d in dets_before if d['label']=='person']) or 0
    conf_after = mean([d['confidence'] for d in dets_after if d['label']=='person']) or 0
    return conf_after - conf_before
```

若连续 N 帧去噪收益 `< 0`，自动回退到 `method="none"` 或降低 strength。

#### 2.1.4 矩形检测去噪隔离
矩形检测器 `RectangleDetector.detect()` 内部已经做了 `CLAHE + GaussianBlur(5,5)`，不应再接受全局去噪后的帧。建议：

```python
# inference.py
rect_dets = self._rectangle_detector.detect(original_frame)  # 用原图
```

---

### 2.2 UKF 状态估计与检测特征空间对齐

#### 2.2.1 调整 UKF sigma 点参数
当前参数 `α=1.0, κ=0` 导致中心点权重为 0，应调整为更保守的缩放：

```python
_ALPHA = 0.4     # 减小 spread，降低尾部 sigma 点权重
_KAPPA = 3.0     # 提高中心点权重，避免 Wm[0]=0
# λ = 0.16 * (8+3) - 8 = -6.24，但需保证 n+λ > 0
```

更安全的做法是让 `n + λ > 0`：

```python
_ALPHA = 0.5
_KAPPA = 5.0
_LAMBDA = _ALPHA**2 * (_N + _KAPPA) - _N  # 0.25*13 - 8 = -4.75 仍不够
```

推荐直接使用标准 Van der Merwe 推荐：

```python
_ALPHA = 0.001   # 极小 spread，接近 EKF 行为
_BETA = 2.0
_KAPPA = 0.0
_LAMBDA = _ALPHA**2 * (_N + _KAPPA) - _N  # ≈ -8，但 n+λ 极小正
```

**更稳妥的做法**：对状态维数 `n=8`，取 `α=1e-3, β=2, κ=0`，则：
- `λ ≈ -7.999`
- `γ = sqrt(n+λ) ≈ 0.0316`
- `Wm[0] = λ/(n+λ) ≈ -0.99997`
- `Wc[0] = λ/(n+λ) + (1-α²+β) ≈ 2.99997`

这是标准 UKF 参数，能稳定工作。当前 `α=1.0` 确实过激进。

#### 2.2.2 引入自适应过程噪声
根据跟踪质量动态调整 `Q`：

```python
def _adapt_q(self, innovation_sq, s_trace):
    nis = innovation_sq / max(s_trace, 1e-10)
    if nis > 2.0:
        # 运动模型不匹配，增大过程噪声
        self.q[:4, :4] *= 1.05
        self.q[4:, 4:] *= 1.10
    elif nis < 0.3:
        # 运动稳定，降低过程噪声
        self.q[:4, :4] *= 0.98
        self.q[4:, 4:] *= 0.95
    # 限制范围
    np.clip(self.q, 1e-6, 0.1, out=self.q)
```

#### 2.2.3 观测噪声按目标尺度自适应
YOLO 的 bbox 误差与目标尺度相关，应在 `correct()` 中根据当前 `w, h` 调整 `R`：

```python
def _scale_r(self, w, h):
    # 小目标（<0.05 归一化尺度）位置噪声更大
    base_pos_var = 0.01
    base_size_var = 0.015
    scale_factor = max(1.0, 0.05 / max(w, h))
    r = np.diag([
        base_pos_var * scale_factor,
        base_pos_var * scale_factor,
        base_size_var,
        base_size_var,
    ])
    return r
```

#### 2.2.4 限制速度状态更新量
防止初始速度估计爆炸：

```python
# 在 correct() 状态更新后
self.x[4:8] = np.clip(self.x[4:8], -0.3, 0.3)  # 归一化坐标下每帧最大移动 30%
```

#### 2.2.5 MCUKF 核带宽按目标尺度自适应
```python
sigma = max(0.05, min(self._kernel_sigma, min(w, h) * 0.5))
```
避免小目标正常位移被当作离群值。

---

### 2.3 流形约束与识别模型解耦

#### 2.3.1 收紧 P 上界并分量裁剪
将 `P_MAX` 从 `1.0` 改为按状态分量分别限制：

```python
_P_MAX_POSITION = 0.05   # cx/cy 最大方差：约 22% 图像宽度
_P_MAX_SIZE = 0.02       # w/h 最大方差
_P_MAX_VELOCITY = 0.01   # vx/vy/vw/vh 最大方差
```

在 `_project_spd()` 后追加对角线裁剪：

```python
def _clip_p_variance(self):
    max_vars = [self._P_MAX_POSITION, self._P_MAX_POSITION,
                self._P_MAX_SIZE, self._P_MAX_SIZE,
                self._P_MAX_VELOCITY, self._P_MAX_VELOCITY,
                self._P_MAX_VELOCITY, self._P_MAX_VELOCITY]
    for i, max_var in enumerate(max_vars):
        if self.p[i, i] > max_var:
            # 仅收缩该维度，不改变协方差结构
            scale = (max_var / self.p[i, i]) ** 0.5
            self.p[i, :] *= scale
            self.p[:, i] *= scale
```

#### 2.3.2 只在数值不稳定时启用 Manifold UKF
Manifold UKF 计算开销大，应作为“安全网”而非常驻：

```python
# AUTO 适配器调整优先级
if cholesky_fails > 0 or p_cond > 1000:
    target = "manifold_ukf"
elif mean_nis > 8.0 or miss_rate > 0.5:
    target = "mcukf"
else:
    target = "ukf"
```

#### 2.3.3 为矩形跟踪引入 UKF 选项或统一滤波器接口
`RectangleTracker` 使用线性卡尔曼，与 `DetectionTracker` 行为不一致。建议：
- 方案 A：让 `RectangleTracker` 也能选择 `ukf/mcukf/manifold_ukf`（共用 `_DetectionUKF` 族）。
- 方案 B：至少统一 `max_misses` 和 `smooth_alpha` 的默认值，避免人与矩形生命周期不同步。

---

### 2.4 多模块解耦与误差隔离

#### 2.4.1 输入解耦：多版本帧并行
将 `_run_inference()` 重构为接收可选的预处理帧：

```python
# inference.py
original_frame = frame.copy()
denoised_frame = self._denoise_frame(original_frame) if self._denoise_method != "none" else original_frame

# 人物/姿态检测用去噪帧（若开启）
person_dets = self._run_inference(denoised_frame)

# 矩形检测用原图
rect_dets = self._rectangle_detector.detect(original_frame)

# 手势识别在人体关键点确定后，从原图裁剪 ROI
hand_dets = self._run_hand_gesture_detection(original_frame, person_dets)
```

#### 2.4.2 重检测质量门控
当前重检测成功后直接 `_update_track()`，未做一致性校验。建议增加：

```python
def _redetect_quality_gate(self, predicted_box, refined_box, original_confidence):
    iou = _box_iou(predicted_box, refined_box)
    conf = refined_box['confidence']
    # 与历史置信度比较，避免召回到低置信度误检
    if conf < original_confidence * 0.6 and iou < 0.3:
        return False
    # 重检测框不应突然放大/缩小
    pred_w = predicted_box[2] - predicted_box[0]
    pred_h = predicted_box[3] - predicted_box[1]
    ref_w = refined_box['x2'] - refined_box['x1']
    ref_h = refined_box['y2'] - refined_box['y1']
    if ref_w < pred_w * 0.5 or ref_w > pred_w * 1.5:
        return False
    if ref_h < pred_h * 0.5 or ref_h > pred_h * 1.5:
        return False
    return True
```

#### 2.4.3 关键点驱动的手势 ROI 校正
手势识别不应仅依赖跟踪后的关键点，还应做局部二次验证：

```python
def _build_hand_rois(self, detections, frame_w, frame_h):
    rois = []
    for det in detections:
        kpts = det.get('keypoints', [])
        if len(kpts) != 17:
            continue
        for wrist_idx in (9, 10):
            wrist = kpts[wrist_idx]
            if wrist.get('conf', 0) < 0.4:  # 提高阈值
                continue
            # 增加 elbow-shoulder 方向向量约束，避免孤立手腕点
            elbow = kpts[wrist_idx - 2]
            if elbow.get('conf', 0) < 0.3:
                continue
            cx, cy = wrist['x'], wrist['y']
            side = max(0.15, min(0.4, ...))  # 限制 ROI 大小
            rois.append((cx-side/2, cy-side/2, cx+side/2, cy+side/2))
    return rois
```

#### 2.4.4 跟踪与检测解耦：独立评估各模块
为每个识别任务维护独立的评估指标，避免“人物好了但矩形坏了”被掩盖：

| 任务 | 关键指标 | 健康阈值（建议） |
|---|---|---|
| 人物检测 | 每帧 person 检出率、平均置信度、ID 切换率 | 检出率 > 0.9，ID 切换 < 0.1/s |
| 人体姿态 | 关键点有效比例、 wrist/ankle 置信度 | 17 点中 conf>0.3 的点 > 12 |
| 手势识别 | 每帧手势检出数、与 wrist 点距离 | 手势框中心距 wrist < 0.15 |
| 矩形检测 | 检出率、Canny 边缘响应强度 | 边缘响应 > 阈值 |

---

## 三、特征融合方案

### 3.1 多线索融合跟踪
当前匹配仅使用 IoU + 中心距。建议增加**外观/颜色直方图**作为辅助线索，在遮挡后恢复 ID：

```python
class _Track:
    def __init__(...):
        self.color_hist = None  # HSV 颜色直方图

    def update_appearance(self, frame, box):
        # 提取 HSV 直方图并做指数平均
        roi = crop(frame, box)
        hist = cv2.calcHist([roi], [0,1], None, [16,16], [0,180,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        if self.color_hist is None:
            self.color_hist = hist
        else:
            self.color_hist = 0.9 * self.color_hist + 0.1 * hist
```

匹配代价函数：

```python
cost = -iou + 0.2 * center_distance + 0.3 * (1 - hist_correlation)
```

### 3.2 多模型输出融合
YOLO 检测与 ROI 重检测结果融合，而非二选一：

```python
def _fuse_detection_and_redetect(self, det, redet):
    if redet is None:
        return det
    # 如果两者 IoU 高，取加权平均
    if _box_iou(det, redet) > 0.5:
        alpha = redet['confidence'] / (det['confidence'] + redet['confidence'])
        return weighted_box(det, redet, alpha)
    # 如果 IoU 低，保留置信度高的
    return det if det['confidence'] > redet['confidence'] else redet
```

### 3.3 关键点与 bbox 联合平滑
当前仅平滑 bbox，关键点独立 EMA。建议将关键点纳入 UKF 观测或至少用 bbox 尺度归一化关键点：

```python
# 当重检测无关键点时，用旧关键点按新 bbox 中心平移并缩放
def _warp_keypoints(self, old_kpts, old_box, new_box):
    old_cx = (old_box[0] + old_box[2]) / 2
    old_cy = (old_box[1] + old_box[3]) / 2
    new_cx = (new_box[0] + new_box[2]) / 2
    new_cy = (new_box[1] + new_box[3]) / 2
    scale_x = (new_box[2] - new_box[0]) / max(old_box[2] - old_box[0], 1e-5)
    scale_y = (new_box[3] - new_box[1]) / max(old_box[3] - old_box[1], 1e-5)
    for kp in old_kpts:
        kp['x'] = new_cx + (kp['x'] - old_cx) * scale_x
        kp['y'] = new_cy + (kp['y'] - old_cy) * scale_y
    return old_kpts
```

---

## 四、模块化隔离测试方法

### 4.1 单元测试矩阵

| 模块 | 测试项 | 通过标准 |
|---|---|---|
| `_denoise_frame` | 对含高斯噪声、椒盐噪声、真实夜景视频测试 | PSNR 提升且 SSIM 不下降 > 3% |
| `_DetectionUKF` | 匀速/加速/静止/遮挡场景 | 位置 RMSE < 0.05，无发散 |
| `_DetectionMCUKF` | 突变量测注入 | 异常帧后 3 帧内恢复 |
| `_DetectionManifoldUKF` | 长时间跳帧（stride=5） | P 条件数 < 500 |
| `_FilterAdaptor` | 模拟高 NIS/高条件数场景 | 切换符合预期，无震荡 |
| `DetectionTracker.update` | IoU 匹配、新建 track、miss 清理 | ID 切换率可重复 |
| `RectangleDetector` | 原图 vs 去噪图 | 原图检出率更高或持平 |

### 4.2 A/B 测试协议

**基准组（Baseline）**：关闭去噪、filter_type="ukf"、关闭 ROI 重检测。

**实验组**：逐项开启功能，每次只改变一个变量：
1. 仅开启 bilateral，strength=1/3/5/7/10
2. 仅开启 nl_means，strength=1/3/5/7/10
3. 仅切换 filter_type="mcukf"，σ=0.1/0.4/1.0
4. 仅切换 filter_type="manifold_ukf"
5. 仅切换 filter_type="auto"
6. 仅开启 ROI 重检测，retries=1/2/3
7. 组合：去噪 + UKF + 重检测

每个组合在**固定测试集**（建议 5 段 30 秒视频）上跑 3 次，记录：
- person 检出率、平均 conf、ID 切换次数
- 关键点有效比例
- 手势识别准确率（人工标注 100 帧）
- 矩形检出率、假阳性数
- 每帧端到端耗时

### 4.3 在线监控与回退

在 `InferWorker` 中增加运行时健康度监控：

```python
class _HealthMonitor:
    def __init__(self):
        self.person_conf_history = deque(maxlen=90)  # 3s @ 30fps
        self.id_switch_count = 0

    def update(self, detections):
        person_confs = [d['confidence'] for d in detections if d['label']=='person']
        self.person_conf_history.append(mean(person_confs) if person_confs else 0)

    def is_healthy(self):
        if len(self.person_conf_history) < 30:
            return True
        recent_mean = mean(list(self.person_conf_history)[-30:])
        # 如果最近 1 秒人物平均置信度持续低于阈值，建议回退
        return recent_mean > 0.35
```

当健康度连续 3 秒不达标时，自动回退到：
- `denoise_method = "none"`
- `filter_type = "ukf"`
- `person_redetect_max_retries = 0`

---

## 五、推荐实施顺序

1. **立即止损（1 天）**
   - 将 `denoise_method` 默认改为 `"none"`。
   - 将 `filter_type` 默认改为 `"ukf"`。
   - 为矩形检测传入原图而非去噪帧。

2. **UKF 稳定性修复（2 天）**
   - 调整 sigma 点参数（α=1e-3）。
   - 增加自适应 Q/R 与速度限幅。
   - 修复 AUTO 适配器 NIS 阈值过低问题。

3. **去噪精细化（2 天）**
   - 实现分任务输入与保守参数映射。
   - 增加去噪收益监控。

4. **耦合解耦（3 天）**
   - 重检测质量门控。
   - 手势 ROI 二次校验。
   - 跟踪与检测独立评估指标。

5. **性能回归验证（2 天）**
   - 跑完 A/B 测试矩阵。
   - 固化推荐参数组合到 `SettingsDialog` 默认值。

---

## 六、关键参数速查表

| 参数 | 当前值 | 推荐值 | 说明 |
|---|---|---|---|
| `_ALPHA` | 1.0 | 1e-3 | 降低 sigma 点 spread |
| `_KAPPA` | 0.0 | 0.0 | 可保持 |
| `_BETA` | 2.0 | 2.0 | 保持 |
| `_P_MAX` | 1.0 | 按分量 0.02~0.05 | 避免协方差膨胀 |
| `q` 初始 | 0.0015 | 0.001 | 更保守 |
| `r` 初始 | 0.012 | 自适应 | 按目标尺度调整 |
| `mcukf σ` | 0.4 | 0.2~0.3 | 避免过度抑制正常变化 |
| `bilateral d` | 5/9 | 5/7 | 减少空间模糊 |
| `bilateral σ` | 25~70 | 13~40 | 保留边缘 |
| `nl_means h` | 3~14 | 2.8~10 | 降低纹理损失 |
| `AUTO NIS threshold` | 5.0 | 8.0 | 减少频繁切换 |
| `AUTO cond threshold` | 500 | 1000 | Manifold 作为安全网 |
| `redetect roi_pad` | 0.15 | 0.10~0.20 | 按场景微调 |
| `redetect max_retries` | 2 | 1~2 | 过多会引入误检 |
| `smooth_alpha` | 0.85 | 0.80~0.90 | 人物/矩形保持一致 |

---

## 七、结论

性能下降并非单一模块导致，而是**去噪过度平滑、UKF 参数激进、Manifold 约束过宽、多模块输入耦合**共同作用的结果。恢复性能的核心策略是：

1. **保守化默认参数**：先让系统回到稳定基线，再逐步开启增强功能。
2. **输入解耦**：不同识别任务使用最适合的图像输入。
3. **质量门控**：任何增强模块都需有收益评估和自动回退机制。
4. **独立评估**：人物、姿态、手势、矩形分别度量，避免单指标掩盖问题。

建议先按“推荐实施顺序”完成前 3 步，即可显著恢复识别性能；再逐步引入更高级的特征融合与自适应策略。

---

## 八、实现记录（2026-07-08）

以下方案已落地到代码，并通过 `py_compile` 语法检查。

### 8.1 已修改文件

| 文件 | 主要改动 |
|---|---|
| `ai/tracker.py` | UKF α=1e-3；按分量 P_MAX 限制；自适应 Q/R；速度/尺度限幅；MCUKF 核带宽按目标尺度自适应；Manifold SPD 投影收紧；AUTO 阈值放宽（NIS 8.0, cond 1000）；新增 HSV 颜色直方图外观特征；关键点按 bbox 中心/尺度平移。 |
| `ai/inference.py` | 去噪参数保守化（bilateral σ=13~40, nl_means h=2.8~10）；人物/姿态用去噪帧，矩形/手势用原始帧；重检测增加 IoU/置信度/尺度质量门控；手势 ROI 增加肘部约束与 wrist 阈值 0.40；新增运行时健康监控自动回退。 |
| `gui/settings_dialog.py` | MCUKF σ 默认值 0.25，filter 与去噪默认保持保守。 |

### 8.2 关键代码映射

| 推荐策略 | 代码位置 |
|---|---|
| UKF α=1e-3 | `ai/tracker.py: _DetectionUKF._ALPHA` |
| 按分量 P_MAX | `_DetectionUKF._P_MAX_VEC`, `_clip_p_variance()` |
| 自适应 Q/R | `_DetectionUKF._make_r()`, `_adapt_q()` |
| 速度/尺度限幅 | `_clip_state()` |
| 输入解耦 | `ai/inference.py: run() 中 original_frame / infer_frame` |
| 重检测质量门控 | `ai/inference.py: _redetect_quality_gate()` |
| 手势 ROI 约束 | `ai/inference.py: _attach_gestures()` 中 wrist 0.40 + elbow 0.30 |
| 颜色直方图匹配 | `ai/tracker.py: _Track.color_hist`, `appearance_similarity()` |
| 关键点平移缩放 | `ai/tracker.py: _warp_keypoints()` |
| 健康监控回退 | `ai/inference.py: _HealthMonitor`, `run()` 中触发逻辑 |

### 8.3 验证命令

```bash
python -m py_compile ai/tracker.py ai/inference.py gui/settings_dialog.py gui/main_window.py
```

结果：通过，无语法错误。

### 8.4 下一步建议

1. 运行单元测试，确保 `_DetectionUKF`, `_DetectionMCUKF`, `_DetectionManifoldUKF`, `DetectionTracker` 基础功能正常。
2. 在固定测试集上执行 A/B 测试，按 4.2 节协议逐项验证。
3. 根据测试结果微调 `bilateral`/`nl_means` 参数与 `AUTO` 阈值，最终固化推荐参数组合。
