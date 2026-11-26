# 预测输出门控与轨迹生命周期解耦

> **版本**：1.1 — 2026-11-26
> **范围**：`ai/tracker.py`、`ai/tracker_backend.py`、`ai/inference.py`、`rust/omintrack/src/tracker.rs`、`rust/omintrack/src/py_api.rs`、`gui/settings_dialog.py`、`gui/main_window.py`、跟踪快照及新增门控测试。
> **状态**：已合入 main（PR #19 → `0e9d959`）；Rust release 重建通过，双端专项 36 项、GUI 9 项、InferWorker 16 项、快照 14 项、全套件 1429 项通过；MOT17 FRCNN 七序列复验完成。

## 1. 目标与背景

旧实现把 `max_misses` 同时用于两个不同目的：

1. 内部轨迹保留多久，以便遮挡后重关联并保留 `track_id`；
2. 未匹配轨迹的预测框继续对外输出多久。

`misses` 只在推理帧的 `update()` 中增长，跳帧的 `predict_step()` 不增长 miss。于是 `max_misses=4` 在 stride 5 时，最后成功匹配后仍可输出约 24 个视频帧的预测框。公共检测中的误检、目标离开和匹配失败因此被放大为持续幽灵输出。

MOT17 FRCNN 七序列基线显示，UKF 从 stride 1 到 stride 5 的新增误差中 93.4% 来自 FP 增加。简单把 `max_misses` 改为 0 虽能降低 FP，却会提前删除内部轨迹，增加 FN 和部分场景的 IDSW；简单关闭全部跳帧预测也会损失大量有效 TP。因此需要解耦**输出可见性**与**内部生命周期**，而不是继续调整卡尔曼公式。

## 2. 架构

新增独立参数 `max_output_misses`，默认 0；原 `max_misses` 保持默认 4。

```text
每个推理帧 update()
  ├─ 所有内部 track 照常 predict / 匹配 / correct / miss 计数
  ├─ misses > max_misses → 删除内部 track（原逻辑不变）
  └─ 输出前 active_tracks()
       ├─ hits >= min_hits_to_show
       └─ misses <= max_output_misses → 才允许输出

跳帧 predict_step()
  ├─ 不增加 misses（原逻辑不变）
  └─ 复用同一 active_tracks() 输出门控

后续 correct()
  └─ misses 复位为 0 → 原 track_id 恢复可见

SettingsDialog「预测输出保留帧数」
  └─ QSettings ai/max_output_misses → MainWindow → InferWorker → DetectionTracker
```

默认配置下，最后成功匹配后的当前推理间隔内，`misses=0` 的跳帧预测仍正常输出；第一次 unmatched 推理帧使 `misses=1`，预测框停止输出，但内部轨迹仍保留到原来的过期时间。后续匹配成功可恢复原 `track_id`，不会因输出门控强制新建轨迹。

## 3. 契约表

| 契约项 | 行为定义 |
|---|---|
| `max_misses` | 仅控制内部轨迹过期；`misses > max_misses` 时删除，默认 4 |
| `max_output_misses` | 仅控制输出可见性；`misses > max_output_misses` 时不输出，默认 0，负数钳制为 0 |
| 跳帧预测 | `predict_step()` 不增加 miss；`misses=0` 的预测轨迹继续输出 |
| 首次推理 miss | 默认隐藏预测输出，但不删除内部轨迹 |
| 重关联 | `correct()` 复位 misses，原 `track_id` 恢复输出；门控不改变匹配、ROI、ReID、AUTO 或滤波器状态 |
| 运行时设置 | `DetectionTracker` / `RustDetectionTracker` / `RustTracker` / `InferWorker` 的 `set_tracker_params` 均支持同名可选参数；`None` 不改变现值 |
| GUI 配置 | 「预测输出保留帧数」范围 0–10，持久化键 `ai/max_output_misses`；旧配置无键时默认 0。「内部轨迹保留帧数」对应原 `max_misses`，不再暗示它控制预测可见性 |
| 兼容旧可见性 | 同时设置 `max_output_misses=max_misses`（默认两者为 0/4 时取 `max_output_misses=4`）恢复旧的持续预测输出 |
| 输出 schema | 不新增或删除字段；只改变各帧可见的 track 集合 |
| 适用双端 | 仅 `DetectionTracker` 的 Python 与 Rust 实现，使用相同条件 `hits >= min_hits && misses <= max_output_misses`，数值与状态机契约不变 |
| 范围边界 | `ai/rectangle_detector.py::RectangleTracker` 是独立功能线，本次不改变其 misses、预测输出或置信度衰减语义 |

该改动不承诺消除所有 FP：新检测仍立即建轨，`min_hits_to_show=1` 不变；当前推理间隔内的预测也保留。它只阻止 unmatched 推理帧之后的陈旧预测继续输出。

## 4. 验证结果

### 4.1 自动化验证

- `tests/test_prediction_output_gating.py`：Python/Rust 各 5 项，覆盖默认隐藏、内部保留后原 ID 恢复、旧语义兼容、运行时 setter、最终过期删除、负数钳制及 `min_hits_to_show` 交互。
- 双端专项：门控 10 项 + matching 8 项 + Rust 等价 18 项，共 **36 项通过**。
- `tests/test_inferworker_resilience.py`：**16 项通过**，含 InferWorker 广播及后注册槽位继承。
- `tests/test_settings_dialog_max_output_misses.py`：**9 项通过**，覆盖旧配置缺键、loader 钳制、确认保存、推荐恢复、spin tooltip 及 MainWindow 透传。
- 快照：**14 项通过**；仅 AUTO 异常、混合置信度两场景、消失删除、重检测失败这 5 个含 miss 输出的 golden 按新契约重生成，meta 继续锁定内部 misses/state/attempts。
- Rust：`maturin develop --release` 重建成功。
- 全套件：**1429 项通过**。

### 4.2 MOT17 真实数据验证

口径：MOT17 FRCNN 公共检测、7 个 train 序列、Hungarian、Rust、ReID/ROI 关闭、完整序列、每配置单次确定性运行，不作统计显著性声称。基线 `4a1eb95` 与当前工作树使用同一实验脚本和 TrackEval。

`max_output_misses=4` 的兼容臂在 stride 1–5 的全部 35 个序列输出文件与修复前参考结果**逐字节一致**。

UKF 联合指标（`COMBINED_SEQ`）：

| stride | 输出门控 | HOTA | MOTA | IDF1 | FP | FN | IDSW |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | 旧 4 | 39.303 | 43.145 | 45.489 | 10584 | 52061 | 1201 |
| 1 | 新 0 | 39.613 | 45.234 | 46.386 | 5909 | 54461 | 1131 |
| 2 | 旧 4 | 40.590 | 42.315 | 46.400 | 12850 | 50740 | 1188 |
| 2 | 新 0 | 41.242 | 46.635 | 47.903 | 5106 | 53717 | 1104 |
| 3 | 旧 4 | 40.512 | 38.482 | 46.223 | 16975 | 50847 | 1261 |
| 3 | 新 0 | 41.699 | 46.067 | 48.502 | 5401 | 53987 | 1177 |
| 4 | 旧 4 | 39.309 | 32.374 | 44.256 | 22885 | 51601 | 1456 |
| 4 | 新 0 | 41.229 | 44.713 | 47.655 | 6102 | 54612 | 1372 |
| 5 | 旧 4 | 38.464 | 24.538 | 43.127 | 30106 | 53026 | 1610 |
| 5 | 新 0 | 41.188 | 42.242 | 47.769 | 7430 | 55914 | 1516 |

新门控在所有 stride 上提高 HOTA/MOTA/IDF1 并减少 FP 和 IDSW；FN 有所增加，这是停止输出陈旧预测的直接代价，不能隐藏该权衡。

stride 5 的其余滤波器联合复验：

| 滤波器 | HOTA | MOTA | IDF1 | FP | IDSW |
|---|---:|---:|---:|---:|---:|
| MCUKF | 41.150 | 41.595 | 47.661 | 7812 | 1475 |
| Manifold UKF | 41.344 | 42.998 | 48.096 | 7071 | 1376 |
| AUTO | 41.189 | 42.042 | 47.739 | 7547 | 1508 |

真实数据验证目前覆盖 FRCNN + Rust；DPM/SDP 和 Python 后端全序列不在本次性能声明内，Python 后端由双端单元/等价测试覆盖。

## 5. 维护指南

- 不要把输出条件改回只检查 `max_misses`；内部重关联保留时间和预测输出可见性是独立契约。
- 修改门控语义必须同步 Python/Rust、facade、InferWorker 与 GUI 透传；行为回归放入 `tests/test_prediction_output_gating.py`，并运行 `tests/test_tracker_rust_equivalence.py` 等价套件。
- 有意改变默认输出时，运行 `pytest tests/test_tracker_snapshot.py --update-golden` 并逐个审查 meta，确认只改变预期场景的可见性而非内部生命周期。
- 修改 `rust/` 后必须执行 `venv\Scripts\python.exe -m maturin develop --manifest-path rust\omintrack\Cargo.toml --release`，再跑双端等价和全套件。
- 后续优化新轨确认、置信度衰减或按视频帧计算 TTL 时，应作为新的行为契约单独设计，不要隐式改变本参数的 inference-miss 单位。
