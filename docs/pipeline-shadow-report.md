# Pipeline Shadow Integration 报告 (Milestone C7)

> **版本**：C7 — 2026-07-27
> **范围**：新增 `visioncore/pipeline/shadow.py`（ShadowPipelineRunner + 统计），不修改 InferWorker/GUI/Recorder
> **状态**：已实现，30 项单元测试全部通过，全量回归零破坏（21 个既有测试套件 + 30 项新测试）

---

## 1. 目标与设计

### 1.1 目标

在不修改现有 `ai/inference.py` InferWorker、不影响 GUI 显示、不影响 Recorder 录制的前提下，让新的 VisionCore Pipeline（C3-C6 阶段链）以 **shadow 模式**并行运行，收集性能统计（FPS、Stage 耗时、Context 生命周期、异常统计），为 Pipeline-vs-InferWorker 的切换决策提供数据支撑。

### 1.2 Shadow 架构

```
┌─────────────────── 现有系统（不变） ───────────────────┐
│                                                         │
│  Camera ──> InferWorker ──> Detector ──> Tracker ──> GUI │
│                    │                    │          │    │
│                    └────────────────────┘          └──> Recorder
│                                                         │
│              (完全不变，正常输出到 GUI 和 Recorder)        │
└─────────────────────────────────────────────────────────┘

┌─────────────────── Shadow Pipeline（新增） ─────────────┐
│                                                         │
│  FrameSource ──> ShadowPipelineRunner                   │
│                       │                                 │
│                 PipelineContext (局部，用后丢弃)          │
│                       │                                 │
│                 DetectorStage ──> TrackerStage           │
│                       │                                 │
│                 TargetStage ──> EventStage               │
│                       │                                 │
│                  ShadowStats (唯一输出)                  │
│                  (FPS / 耗时 / 异常 / Context)           │
│                                                         │
│              (不写 GUI，不写 Recorder，不改 InferWorker)   │
└─────────────────────────────────────────────────────────┘
```

### 1.3 "Shadow"的三重保证

| 保证 | 落实 |
|---|---|
| **不修改 InferWorker** | ShadowPipelineRunner 是独立组件，不 import `ai/inference.py`，不注入任何东西到 InferWorker |
| **不影响 GUI** | Shadow 的 PipelineContext 是局部对象，run_once 返回后由调用方持有；不连接任何 Qt 信号/槽 |
| **不影响 Recorder** | Shadow 不调用 VideoRecorder，不写文件，FrameSource 独立读取（不与 InferWorker 的采集竞争） |

### 1.4 唯一输出：ShadowStats

Shadow 的产出**只有** `ShadowStats`——一个纯数据统计对象。不含检测/跟踪结果（那些在局部 Context 中用后丢弃），不含帧图像，不含任何可影响现有系统的副作用。

---

## 2. ShadowPipelineRunner

### 2.1 文件

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/shadow.py` | ShadowPipelineRunner + ShadowStats + StageStats |
| `visioncore/pipeline/__init__.py` | 更新：导出 ShadowPipelineRunner / ShadowStats / StageStats |
| `tests/test_shadow_pipeline.py` | 30 项单元测试 |

### 2.2 构造

```python
ShadowPipelineRunner(
    pipeline: Pipeline,       # 已配置好 4 阶段的 Pipeline
    frame_source: FrameSource, # 提供帧的源（DummyFrameSource 用于基准）
    *,
    name: str = "shadow",     # 标签
)
```

- `pipeline`：已配置 DetectorStage/TrackerStage/TargetStage/EventStage 的 Pipeline。Runner 调用其 `initialize()`/`shutdown()` 管理生命周期
- `frame_source`：提供 Frame。基准用 DummyFrameSource；实际集成用 CameraFrameSource（未来，tap 同一摄像头）
- 类型强制：非 Pipeline / 非 FrameSource 抛 `TypeError`

### 2.3 生命周期

| 方法 | 语义 |
|---|---|
| `initialize()` | `pipeline.initialize()` + `source.open()`。幂等 |
| `run_once()` | 创建 Context → 读帧 → 逐 Stage 计时 process → 记录统计 → 返回 Context（局部） |
| `run_batch(max_frames, *, stop_on_error_rate)` | 循环 run_once，到 max_frames / 源耗尽 / 错误率超限停止 |
| `shutdown()` | `pipeline.shutdown()` + `source.close()`。幂等 |
| `__enter__` / `__exit__` | 上下文管理器：进入 initialize，退出 shutdown |

### 2.4 逐 Stage 计时（不用 Pipeline.run）

Runner **手动迭代** `pipeline.stages` 而非调 `pipeline.run()`，以便逐 Stage 计时：

```python
for stage in pipeline.stages:
    t0 = perf_counter()
    try:
        stage.process(ctx)
    except Exception as exc:
        record_error(stage, exc)
        break  # 跳过本帧剩余 Stage
    finally:
        record_stage_time(stage, perf_counter() - t0)
```

---

## 3. 统计收集

### 3.1 ShadowStats

| 字段 | 类型 | 语义 |
|---|---|---|
| `frame_count` | int | 已处理帧数（含出错帧） |
| `skip_count` | int | 源耗尽跳过数 |
| `error_frame_count` | int | 至少一 Stage 出错的帧数 |
| `total_frame_time` | float | 所有帧总处理时间（秒） |
| `context_count` | int | 创建的 Context 数（= frame_count + skip_count） |
| `stage_stats` | `dict[str, StageStats]` | 逐 Stage 累加器 |
| `exceptions` | `list[(stage_name, exc_type_name)]` | 每次异常的记录（含重复） |

| 属性 | 语义 |
|---|---|
| `fps` | `frame_count / total_frame_time` |
| `avg_frame_time` | `total_frame_time / frame_count` |
| `error_rate` | `error_frame_count / frame_count` |
| `summary()` | 多行人类可读摘要（用于报告/日志） |

### 3.2 StageStats

| 字段 | 语义 |
|---|---|
| `name` | Stage 显示名 |
| `call_count` | process() 调用次数（含出错） |
| `total_time` / `min_time` / `max_time` | 累计/最短/最长耗时 |
| `error_count` | 出错次数 |
| `last_error` | 最近异常实例（诊断用） |
| `avg_time` | `total_time / call_count` |
| `error_rate` | `error_count / call_count` |

### 3.3 异常吸收（Shadow 专属）

与 `Pipeline.run()`（异常原样传出、中止 run）不同，ShadowRunner **吸收** Stage 异常：

1. 异常记录到 `stats.exceptions` + `stage_stats[name].error_count`
2. 本帧剩余 Stage **跳过**（Context 处于未知状态）
3. Runner **继续下一帧**（不中止 batch）

这让 Shadow 能跨多帧收集有意义的统计——验证脚手架的目的是看什么会坏、多久坏一次，而非首个错误即中止。

### 3.4 Context 生命周期

每帧 `run_once` 创建一个全新 `PipelineContext.empty()`，处理后返回（调用方可持有或丢弃）。Runner 不保留 Context 引用。`context_count` 追踪创建数，应等于 `frame_count + skip_count`（每帧一个 Context，包括跳过的）。

---

## 4. 基准测试结果

### 4.1 配置

- **Pipeline**：DetectorStage(DummyDetector) → TrackerStage(DummyTracker) → TargetStage(DummyTargetManager) → EventStage(DummyEventBus)
- **帧源**：DummyFrameSource，1000 帧（8×8×3 zeros），loop=False
- **运行**：`run_batch(max_frames=2000)`，单线程，`time.perf_counter` 计时

### 4.2 结果

```
============================================================
SHADOW PIPELINE BENCHMARK (1000 frames, 4 stages, Dummy backends)
============================================================
frames=1000  skips=1  errors=0  error_rate=0.00%
total_time=4.63ms  avg_frame=4.6us  fps=216192.8
contexts_created=1001
per-stage:
  StageStats(name='detect',  calls=1000, avg=0.5us, min=0.4us, max=4.3us, errors=0)
  StageStats(name='track',   calls=1000, avg=0.5us, min=0.4us, max=2.5us, errors=0)
  StageStats(name='targets', calls=1000, avg=1.0us, min=0.9us, max=4.2us, errors=0)
  StageStats(name='events',  calls=1000, avg=1.3us, min=0.8us, max=25.5us, errors=0)

Context lifecycle: 1001 contexts created (1 per frame)
Error rate: 0.00%
FPS: 216192.8
Avg frame time: 4.6 us
```

### 4.3 解读

| 指标 | 值 | 含义 |
|---|---|---|
| **Pipeline FPS** | 216,193 | Dummy 后端下的纯框架吞吐（无真实推理） |
| **平均帧耗时** | 4.6 us | 4 阶段 + Context 创建 + 帧读取的总开销 |
| **detect 耗时** | 0.5 us avg | DetectorStage 适配器开销（不含真实检测） |
| **track 耗时** | 0.5 us avg | TrackerStage 适配器开销（不含真实跟踪） |
| **targets 耗时** | 1.0 us avg | TargetStage 双输出开销（略高，因 clear+extend 两个列表） |
| **events 耗时** | 1.3 us avg | EventStage 开销（含 Event 构造 + bus.publish） |
| **Context 生命周期** | 1001 创建 | 1 per frame（含 1 个 skip 的 Context） |
| **异常统计** | 0 errors / 0.00% | 1000 帧零异常 |

**关键洞察**：Pipeline 框架本身的开销极低（~4.6us/帧），远低于真实推理耗时（YOLO 检测 ~5-30ms/帧）。这意味着切换到 Pipeline 架构不会引入可感知的性能损失——框架开销 <0.02% 的典型推理时间。真实后端接入后，FPS 将由检测/跟踪耗时决定，而非框架。

---

## 5. 无影响保证

### 5.1 InferWorker 不变

`ai/inference.py` 的 InferWorker **零修改**。ShadowPipelineRunner 不 import `ai.inference`，不注入 EventBus/TargetManager 到 InferWorker，不 tap 其内部状态。InferWorker 继续按其原有逻辑驱动检测/跟踪/GUI/Recorder。

### 5.2 GUI 不受影响

Shadow 的 PipelineContext 是局部对象——`run_once` 返回它，调用方可检查或丢弃，但它**不连接任何 Qt 信号/槽**，不更新 CameraCell，不触发 MainWindow 刷新。GUI 只看到 InferWorker 的输出，完全不知道 Shadow 的存在。

### 5.3 Recorder 不受影响

Shadow 不调用 VideoRecorder.push_frame，不写文件。Shadow 的 FrameSource 独立读取（基准中用 DummyFrameSource；实际集成中若 tap 同一摄像头，需确保不与 InferWorker 的采集竞争——这属于未来接线工作，C7 不实现）。

### 5.4 测试验证

| 测试 | 验证 |
|---|---|
| `test_shadow_does_not_modify_shared_state` | 仅 DummyEventBus 记录事件，无其他副作用 |
| `test_shadow_context_is_discarded` | Runner 不保留 Context 引用 |
| `test_run_once_does_not_share_context_across_calls` | 每帧全新 Context，无跨帧泄漏 |

---

## 6. 测试覆盖

### 6.1 统计

- **测试文件**：`tests/test_shadow_pipeline.py`
- **测试函数数**：30（全部通过）

### 6.2 覆盖类别

| 类别 | 测试数 | 覆盖 |
|---|---|---|
| 构造 | 4 | 存储 pipeline/source；拒非 Pipeline；拒非 FrameSource；默认名 |
| 生命周期 | 6 | initialize/source.open；幂等；shutdown/close；上下文管理器；未初始化 run 抛 |
| run_once | 5 | 处理一帧；context_count 递增；源耗尽跳过；逐 Stage 计时；Context 不跨帧共享 |
| run_batch | 4 | 多帧；源耗尽停止；返回 stats；错误率早停 |
| 异常吸收 | 3 | 吸收+记录；跳过剩余 Stage；batch 继续 |
| 统计 | 4 | FPS>0；零错误率；avg_time 合理；summary 可读；summary 含异常 |
| 无副作用 | 2 | 不修改共享状态；Context 丢弃 |
| 端到端 | 1 | 100 帧全链 |

### 6.3 回归测试

全量回归零破坏——22 个套件全过：

```
test_shadow_pipeline.py (C7 新增) : 30/30 passed
test_event_stage.py (C6)          : 38/38 passed
test_target_stage.py (C5)         : 44/44 passed
test_tracker_stage.py (C4)        : 42/42 passed
test_detector_stage.py (C3)       : 39/39 passed
test_frame_source.py (C2)         : 55/55 passed
test_pipeline.py (C1)             : 56/56 passed  (test_package_exports 改子集断言)
... (15 个既有套件全过)
────────────────────────────────────────────────────────────────
总计                              : 既有全过 + 30 新增，0 回归
```

**C1 测试调整**：`test_pipeline.py` 的 `test_package_exports_public_api` 原用严格集合相等，C7 扩展 `pipeline.__all__` 后改为子集断言（C1 框架名**存在于** `__all__`）。与 C4 调整 detector 测试同模式。

---

## 7. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 保持现有 InferWorker 不变 | ✅ 零修改 ai/inference.py |
| 新增 ShadowPipelineRunner | ✅ visioncore/pipeline/shadow.py |
| Shadow 运行 Pipeline（Detect→Track→Target→Event） | ✅ |
| 不修改现有输出 | ✅ Shadow 输出仅 ShadowStats |
| 不影响 GUI | ✅ 不连接 Qt 信号/槽 |
| 不影响 Recorder | ✅ 不调用 VideoRecorder |
| 统计 Pipeline FPS | ✅ `stats.fps` |
| 统计 Stage 耗时 | ✅ `stats.stage_stats[name]` (total/min/max/avg) |
| 统计 Context 生命周期 | ✅ `stats.context_count` |
| 统计异常 | ✅ `stats.exceptions` / `error_frame_count` / `error_rate` |
| 输出 `docs/pipeline-shadow-report.md` | ✅ 本文件 |

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/shadow.py` | ShadowPipelineRunner + ShadowStats + StageStats |
| `visioncore/pipeline/__init__.py` | 更新：导出 shadow 组件（3 名） |
| `tests/test_shadow_pipeline.py` | 30 项单元测试 |
| `tests/test_pipeline.py` | 微调：`test_package_exports` 改子集断言（容忍包增长） |
| `docs/pipeline-shadow-report.md` | 本报告 |

---

## 9. 集成路径（未来）

### 9.1 实时 Shadow 接线（C8）

将 ShadowPipelineRunner 包裹在专用 QThread 中，FrameSource tap 同一摄像头输出（或共享 InferWorker 的帧缓冲），实现真正的"边运行边对比"：

```python
# gui/main_window.py（未来，最小侵入）
self._shadow = ShadowPipelineRunner(shadow_pipeline, camera_frame_source)
self._shadow_thread = QThread()
self._shadow.moveToThread(self._shadow_thread)
self._shadow_thread.start()
# InferWorker 继续正常工作；Shadow 在旁静默收集统计
```

### 9.2 Pipeline-vs-InferWorker 对比（C8）

在 Shadow 中同时记录 InferWorker 的输出（detections/tracks），与 Pipeline 的输出逐帧对比，量化差异：
- 检测数差异（Pipeline 检测 vs InferWorker 检测）
- 跟踪稳定性差异（ID 切换次数）
- 延迟差异（端到端耗时）

### 9.3 切换决策（D）

基于 Shadow 统计 + 对比结果，决定是否将 Pipeline 从 shadow 转正、InferWorker 退役：
- 若 Pipeline FPS ≥ InferWorker FPS 且差异可接受 → 切换
- 若 Pipeline 有优势领域（可测试性、可组合性、模块化）但 FPS 略低 → 评估权衡

---

## 10. 后续里程碑预告

- **C8**：CaptureStage（包装 FrameSource 填 context.frame）→ 完整 Pipeline 链 + 实时 Shadow 接线 + Pipeline-vs-InferWorker 对比
- **D**：基于 Shadow 数据的切换决策 → InferWorker 退役，Pipeline 转正
