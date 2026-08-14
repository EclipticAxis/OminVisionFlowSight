# 双边滤波 / NL-Means 去噪 — 实现方案

## 一、需求

在 YOLO 推理前增加图像去噪预处理，支持双边滤波（Bilateral Filter）和非局部均值（NL-Means）两种算法，减少摄像头噪声对检测精度的影响。

## 二、算法对比

| | 双边滤波 (Bilateral) | NL-Means (Non-Local Means) |
|---|---|---|
| **原理** | 空间距离 + 灰度差异双核加权，边缘保持去噪 | 全图搜索相似 patch，加权平均去噪 |
| **去噪质量** | 中等（适合轻度噪声） | 高（适合强噪声，细节保持好） |
| **速度 (1080p)** | **~3-8ms**（实时可用） | ~30-80ms（需降采样或小窗口） |
| **OpenCV 接口** | `cv2.bilateralFilter` | `cv2.fastNlMeansDenoisingColored` |
| **关键参数** | d（邻域直径）, sigmaColor, sigmaSpace | h（滤波强度）, templateWindowSize, searchWindowSize |
| **适用场景** | 实时视频，低-中噪声 | 离线/高精度模式，中-高噪声 |

## 三、集成方案

### 3.1 集成点

在 `InferWorker.run()` 主循环中，推理帧分支内、`_run_inference(frame)` 调用前插入去噪步骤：

```python
# run() 主循环中
if should_infer:
    if self._should_run_yolo_inference():
        # ★ 新增：推理前去噪
        if self._denoise_method != "none":
            frame = self._denoise_frame(frame)
        raw_detections = self._run_inference(frame)
        ...
```

**设计理由**：
- 只对推理帧去噪（跳帧不需要，跳帧复用上次结果）
- 只在 YOLO 推理开启时去噪（不需要时零开销）
- 去噪后的 frame 同时可用于后续的 ROI 重检测（`_redetect_person_in_roi` 也会受益）

### 3.2 去噪方法实现

```python
def _denoise_frame(self, frame: np.ndarray) -> np.ndarray:
    """对帧做去噪预处理，返回去噪后的帧。"""
    method = self._denoise_method
    strength = self._denoise_strength  # 1-10

    if method == "bilateral":
        # 双边滤波：strength 映射到 d 和 sigma
        # d=5~9, sigmaColor=25~75, sigmaSpace=25~75
        d = 5 if strength <= 5 else 9
        sigma = 25.0 + (strength - 1) * 5.0  # 25~70
        return cv2.bilateralFilter(frame, d, sigma, sigma)

    elif method == "nl_means":
        # NL-Means：strength 映射到 h
        # h=3~15, templateWindowSize=7, searchWindowSize=21
        h = 3.0 + (strength - 1) * 1.2  # 3~14
        return cv2.fastNlMeansDenoisingColored(
            frame, None, h, h, 7, 21
        )

    return frame
```

### 3.3 强度参数映射

`denoise_strength` (1-10) 统一控制两种算法的强度：

| Strength | Bilateral | NL-Means | 效果 |
|---|---|---|---|
| 1-3 | d=5, σ=25-35 | h=3-5 | 轻度去噪，保留最多细节 |
| 4-5 | d=5, σ=40-50 | h=6-8 | 中度去噪（推荐默认） |
| 6-10 | d=9, σ=55-70 | h=9-14 | 强力去噪，可能模糊细节 |

### 3.4 NL-Means 性能优化

NL-Means 在 1080p 上较慢（30-80ms）。优化策略：
1. **降采样去噪**：对帧先缩小 2x → 去噪 → 放大回原尺寸，速度提升 ~4x
2. **小窗口**：templateWindowSize=7, searchWindowSize=21（OpenCV 默认，已有优化）
3. **仅推理帧去噪**：stride=2 时，50% 的帧不需要去噪

```python
# NL-Means 降采样优化（当帧大于 720p 时）
if method == "nl_means" and frame.shape[0] > 720:
    small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
    denoised = cv2.fastNlMeansDenoisingColored(small, None, h, h, 7, 21)
    return cv2.resize(denoised, (frame.shape[1], frame.shape[0]))
```

## 四、文件改动清单

| 文件 | 改动 |
|---|---|
| `ai/inference.py` | `InferWorker.__init__` 新增 `_denoise_method`/`_denoise_strength` 字段；新增 `set_denoise()` setter；新增 `_denoise_frame()` 方法；`run()` 推理分支调用去噪 |
| `gui/settings_dialog.py` | 新增去噪方法下拉框（关闭/双边滤波/NL-Means）+ 去噪强度 SpinBox(1-10) |
| `gui/main_window.py` | `_apply_ai_settings()` 透传去噪配置 |

## 五、配置项设计

| 配置项 | QSettings Key | 类型 | 默认 | 选项 |
|---|---|---|---|---|
| 去噪方法 | `ai/denoise_method` | str | "none" | none / bilateral / nl_means |
| 去噪强度 | `ai/denoise_strength` | int | 5 | 1-10 |

GUI 下拉框：
```
关闭
双边滤波（快，保边去噪）
NL-Means（慢，高质量去噪）
```

## 六、性能预估

| 方法 | 1080p 耗时 | 720p 耗时 | 降采样优化后 | 对 YOLO 推理影响 |
|---|---|---|---|---|
| 关闭 | 0ms | 0ms | — | 无 |
| 双边滤波 (strength=5) | ~5ms | ~2ms | — | +50%（10ms→15ms） |
| NL-Means (strength=5) | ~50ms | ~20ms | ~12ms | +120%（10ms→22ms） |
| NL-Means 降采样 | ~12ms | — | ~12ms | +20%（10ms→12ms） |

> 双边滤波是实时应用的首选；NL-Means 配合降采样优化后也可实时。

## 七、风险与缓解

| 风险 | 缓解 |
|---|---|
| NL-Means 太慢导致掉帧 | 自动降采样优化（>720p 时缩小 2x）；GUI 提示"较慢" |
| 去噪过度导致检测精度下降 | 强度参数可调；默认 strength=5（中等）；关闭选项始终可用 |
| 去噪后 frame 与原图不一致影响绘制 | 去噪仅作用于推理路径，`detection_ready` 信号仍发出归一化坐标；绘制基于原图（`update_frame` 收到的是未去噪帧） |
| cv2 接口在 ONNX 后端下行为不一致 | `_denoise_frame` 在 `_run_inference` 之前调用，对 ONNX 和 ultralytics 后端效果一致 |

## 八、实现顺序

1. `ai/inference.py` — 新增 `_denoise_method`/`_denoise_strength` + `set_denoise()` + `_denoise_frame()` + `run()` 集成
2. `gui/settings_dialog.py` — 去噪方法下拉框 + 强度 SpinBox + save/load
3. `gui/main_window.py` — 透传配置
4. 测试验证 — 语法检查 + 去噪功能验证
