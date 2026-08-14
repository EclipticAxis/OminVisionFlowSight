# InferWorker Decomposition 设计文档 (Milestone D9)

> **版本**：D9 — 2026-08-04
> **范围**：新增 `visioncore/pipeline/migration.py`，封装 Legacy → Pipeline 双模式迁移模式
> **状态**：已实现，18 项单元测试全部通过 + 1 个模块 doctest 通过，全量回归零破坏（31 个测试套件）

---

## 1. 目标与背景

### 1.1 迁移策略

`ai/inference.py` 中的 `InferWorker` 已实现 `pipeline_enabled` 开关（D9 前置工作，C8 里程碑）。当 `pipeline_enabled=True` 时，推理帧经 `Pipeline(DetectorStage → TrackerStage → TargetStage → EventStage)` 处理，使用 `visioncore.pipeline.adapters` 中的适配器包装现有 YOLO/Tracker/TargetManager 后端。Legacy 代码路径完全不变。

D9 交付**迁移基础设施**：

- **MigrationPipeline**：可复用的双模式执行封装器，提取 InferWorker 的迁移模式为独立可测试模块。
- **适配器模块测试**：验证 `visioncore.pipeline.adapters` 中四个适配器（Detector / Tracker / TargetManager / EventBus）正确委托到同一后端。
- **双模式一致性测试**：验证两种模式使用同一后端引用、返回相同 `list[dict]` 格式。

### 1.2 迁移架构

```
InferWorker.run()
  │
  ├─ pipeline_enabled=False (Legacy 模式)
  │    └─ 内联代码路径：
  │         _denoise_frame → _run_inference → tracker.update →
  │         _attach_head_attributes → _attach_gestures →
  │         _run_rectangle_detection → _filter_display_detections →
  │         detection_ready.emit → _bypass_update_targets
  │
  └─ pipeline_enabled=True (Pipeline 模式)
       └─ _run_pipeline_mode():
            Pipeline(
              DetectorStage(InferWorkerDetectorAdapter) →
              TrackerStage(InferWorkerTrackerAdapter) →
              TargetStage(InferWorkerTargetManagerAdapter) →
              EventStage(InferWorkerEventBusAdapter)
            ) + 后处理（head attributes, gestures, rectangle, health monitor）
```

### 1.3 设计原则

| 原则 | 落实 |
|---|---|
| **同一后端** | 两种模式调用**相同**的 YOLO 模型、DetectionTracker、TargetManager、EventBus |
| **零行为变更** | `ai/inference.py` 未修改（D9 不改生产代码） |
| **适配器桥接** | `visioncore.pipeline.adapters` 包装现有后端为 Pipeline 阶段接口 |
| **可回滚** | `pipeline_enabled` 默认 `False`，一行切换 |
| **可测试** | `MigrationPipeline` 提供可注入的双模式接口，mock 后端即可测试 |
| **禁止 camera/gui** | 不触碰 `camera/` `gui/` |

### 1.4 关键设计决策

**为什么不在 D9 修改 `ai/inference.py`？**

D9 的目标是**验证迁移基础设施**，而非一步到位的生产切换。`ai/inference.py` 是 2169 行的生产代码（含 PyQt6、torch、ultralytics），修改风险高。D9 通过：

1. 独立的 `MigrationPipeline` 封装器（可测试）
2. 适配器模块的独立测试（验证委托正确性）
3. mock 后端的双模式一致性测试

来证明迁移路径可行，为后续生产切换提供信心。

---

## 2. 文件结构

```
visioncore/pipeline/
├── adapters.py          # (C8) 适配器模块：InferWorker 后端 → Pipeline 阶段接口
├── migration.py         # (D9) 可复用的双模式迁移封装器 ← 新增
└── ...

tests/test_inferworker_migration.py  # 18 项单元测试 ← 新增
docs/inferworker-migration.md        # 本文档 ← 新增
```

---

## 3. 接口

### 3.1 MigrationPipeline

```python
class MigrationPipeline:
    def __init__(
        self,
        detector=None, tracker=None,
        target_manager=None, event_bus=None,
        legacy_process=None,
    ): ...

    @property
    def pipeline_enabled(self) -> bool: ...
    def set_pipeline_enabled(self, enabled: bool) -> None: ...
    def run_frame(self, frame, slot_id=0, frame_id=0, submitted_ts=0.0) -> list[dict]: ...
```

| 方法 | 行为 |
|---|---|
| `set_pipeline_enabled(True/False)` | 切换模式（默认 False = Legacy） |
| `run_frame(frame, ...)` | 处理一帧：Legacy 模式调用 `legacy_process`；Pipeline 模式构建适配器→Pipeline→运行 |
| `detector/tracker/target_manager/event_bus` | 只读属性，暴露后端引用供测试 |

### 3.2 适配器模块（`visioncore.pipeline.adapters`）

| 适配器 | 包装 | 委托到 |
|---|---|---|
| `InferWorkerDetectorAdapter` | `Detector` 接口 | `worker._run_inference(frame)` |
| `InferWorkerTrackerAdapter` | `Tracker` 接口 | `worker._trackers[slot_id].update(dicts)` |
| `InferWorkerTargetManagerAdapter` | `TargetManager` 接口 | `worker._target_mgr.update_targets(tracks)` |
| `InferWorkerEventBusAdapter` | `EventBus` 接口 | `worker._event_bus.publish(event)` |

所有适配器使用 `worker: Any` 类型（无 InferWorker 运行时依赖），仅在 `TYPE_CHECKING` 下引用类型。

### 3.3 结果格式

两种模式均返回 `list[dict]`（legacy detection dict 格式）：

```python
{
    "bbox": (x1, y1, x2, y2),   # 角点格式
    "confidence": float,          # 置信度
    "class_id": int,              # 类别 ID
    "label": str,                 # 类别标签
    "track_id": int,              # 跟踪 ID（tracker 添加）
}
```

---

## 4. 与既有里程碑的关系

```
C8 InferWorker._run_pipeline_mode()
   │   (ai/inference.py 内联实现，已存在)
   │   adapters: InferWorkerDetectorAdapter / TrackerAdapter /
   │             TargetManagerAdapter / EventBusAdapter
   │
   ▼
D9 MigrationPipeline ──── 提取为独立可测试模块
       │
       ├─ run_frame() → Legacy 模式 (legacy_process callable)
       └─ run_frame() → Pipeline 模式 (adapters + Pipeline)
       
D9 测试 ──── mock 后端验证适配器委托 + 双模式一致性
```

---

## 5. 测试覆盖（tests/test_inferworker_migration.py，18 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 模块表面 | 2 | MigrationPipelineError 继承、适配器可导入 |
| 构造 | 2 | 默认构造（enabled=False）、带后端构造 |
| 模式切换 | 2 | set_pipeline_enabled(True/False)、幂等 |
| Legacy 模式 | 2 | 调用 legacy_process、无 process 返回 [] |
| Pipeline 模式 | 3 | DetectorAdapter 调用 worker._run_inference、TrackerAdapter 委托、构造 |
| 双模式一致性 | 2 | 同一后端引用、返回 list[dict] 格式 |
| Pipeline 集成 | 2 | 完整 Pipeline(mock 后端) 运行 + 检测器/跟踪器调用计数、错误返回 [] |
| 零依赖 | 2 | AST 无网络/模型/gui/ai、运行时无 torch |

---

## 6. 零侵入声明

新增 `visioncore/pipeline/migration.py` 和 `tests/test_inferworker_migration.py`。**未修改任何既有文件**（`ai/inference.py` 未改、`camera/` 未改、`gui/` 未改）。全量回归：31 个测试套件全部通过。
