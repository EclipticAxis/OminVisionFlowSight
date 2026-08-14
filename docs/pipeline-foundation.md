# Pipeline Foundation 设计文档 (Milestone C1)

> **版本**：C1 — 2026-07-25
> **范围**：新增 `visioncore/pipeline/` 包，定义新一代 Pipeline 架构的框架基础
> **状态**：已实现，56 项单元测试全部通过，全量回归零破坏（15 个既有测试套件 + 56 项新测试）

---

## 1. 目标与背景

### 1.1 为什么需要 Pipeline Layer

当前 VisionDataPlatform 的核心处理逻辑集中在 `ai/inference.py`（约 1792 行）的单体 `InferWorker` 中——检测、跟踪、目标管理、健康监控、去噪、属性增强全部耦合在一个 QThread 里。战略路线图（`docs/strategic-roadmap.md` M3）明确指出：

> InferWorker 拆分为 3 个 Pipeline：DetectionPipeline / TrackingPipeline / AttributePipeline 分离

如果继续在单体 `InferWorker` 上叠加功能，会产生：

- **耦合扩散**：每加一个增强模块（去噪、ReID、UHD、CHC）都要修改 `inference.py` 多处
- **测试困难**：无法独立测试单个处理阶段（检测/跟踪/投影），必须启动整个推理线程
- **复用受阻**：跟踪引擎、推理调度无法独立成包（M2 路线图目标）
- **替换成本高**：换检测后端（YOLOv8 → YOLO26 → ONNX）需要触碰推理核心

Pipeline Layer 通过引入**抽象基类 `PipelineStage` + 顺序执行器 `Pipeline` + 共享黑板 `PipelineContext`** 解决这些问题。每个处理阶段是一个独立的、可测试的 Stage；Pipeline 只负责按顺序驱动它们。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **纯调度器** | Pipeline 只知道 Stage 列表和顺序，不包含任何 Detector / Tracker / TargetManager |
| **共享黑板** | 单一 `PipelineContext` 流经所有 Stage，Stage 读上游字段、写下游字段 |
| **严格生命周期** | 每个 Stage 遵守 `initialize → process → shutdown`；Pipeline 正向初始化、逆向关闭 |
| **诚实地传播异常** | Stage 的 `process` 异常原样传出 `run()`，框架从不静默吞掉；`shutdown` 是唯一的 best-effort 路径 |
| **框架优先** | C1 只建框架，真实 Stage（检测/跟踪/目标/投影）留给 C2+ |
| **零侵入** | 不修改 `ai/` `gui/` `camera/`，现有 VDP 应用继续运行不变 |

### 1.3 与既有里程碑的关系

```
B1 TargetState (数据快照)  ──┐
B2 ProtocolAdapter (投递 sink) ─┤
                              ├──> C1 Pipeline (调度框架) ──> C2+ 真实 Stage
                              │         │
eventbus (事件总线) ──────────┘         │
                                        ▼
                            未来 InferWorker 重构为 Stage 链
```

C1 是**正交补充**：B1/B2 定义了数据如何流动和投递，eventbus 定义了事件如何分发，C1 定义了**处理阶段如何编排**。四者组合后，未来的 InferWorker 将变成一条 `Pipeline`，其 Stage 产出 `TargetState` 快照并通过 `ProtocolAdapter` 投递，状态变更通过 `EventBus` 通知 GUI。

---

## 2. 文件结构

```
visioncore/pipeline/
├── __init__.py            # 包导出：Pipeline, PipelineStage, PipelineContext, DummyStage, StageError
├── base.py                # PipelineStage(ABC) + StageError + 四方法生命周期契约
├── context.py             # PipelineContext — 可变数据黑板
├── pipeline.py            # Pipeline — 顺序执行器 + 上下文管理器
└── stage.py               # DummyStage — 测试用可配置 no-op Stage
```

与 `visioncore/eventbus/`、`visioncore/protocol/` 的多文件模式一致。未来真实 Stage（CaptureStage / DetectionStage / TrackingStage / TargetManagementStage / ProjectionStage）将作为兄弟模块或子包在 C2+ 加入，均继承 `PipelineStage`，`Pipeline` 与 `PipelineContext` 无需改动。

---

## 3. PipelineContext 字段说明

`PipelineContext` 是流经 Pipeline 的**可变数据信封**。每个 Stage 读取上游字段、写入下游字段，在同一实例上累积 `Frame → Detections → Tracks → Targets → TargetStates` 的处理链。

### 3.1 字段一览

| 字段 | 类型 | 默认 | 语义 | 产出者 → 消费者 |
|---|---|---|---|---|
| `frame` | `Frame \| None` | `None` | 当前捕获帧；`None` 直到采集 Stage 填充 | CaptureStage → DetectionStage |
| `detections` | `list[Detection]` | `[]` | 当前帧检测器输出 | DetectionStage → TrackingStage |
| `tracks` | `list[Track]` | `[]` | 跨帧关联的跟踪对象 | TrackingStage → TargetManagementStage |
| `targets` | `list[Target]` | `[]` | 高层观测目标实体 | TargetManagementStage → ProjectionStage |
| `target_states` | `list[TargetState]` | `[]` | 目标的可观测快照（**snapshot dataclass**，非生命周期枚举） | ProjectionStage → ProtocolAdapter (B2) |
| `timestamp` | `float` | `0.0` | 本次 Pipeline 运行的处理时间戳（秒） | 调用方设置，全体 Stage 读取 |
| `metadata` | `dict[str, Any]` | `{}` | Stage 间通信的扩展键值存储（如 `metadata["order"]` 追踪调度顺序） | 任意 Stage 读写 |

### 3.2 设计要点

- **可变、非 frozen**：与 `Frame`/`Detection`（不可变值对象）不同，Context 是**工作面**，Stage 在其上累积数据。可变性是刻意的，且限定在单次 Pipeline 运行的单线程内。
- **`slots=True`**：每帧都会分配一个 Context，slot 化避免 `__dict__` 在分配剖析中冒头。
- **纯容器**：所有流动字段是 `list`（或单个 Frame / None），无 numpy、无 Qt、无框架耦合——Context 可直接 `dataclasses.asdict` 序列化。

### 3.3 ⚠️ TargetState 命名陷阱（强制）

`target_states` 字段持有的是 `visioncore.state.target_state.TargetState`——**快照 dataclass**，**不是**顶层 `visioncore` 包再导出的生命周期枚举。两者同名但语义完全不同（见 `docs/targetstate-design.md` 与 `visioncore/protocol/base.py` 的警告）：

| 类型 | 位置 | 性质 | 用途 |
|---|---|---|---|
| `visioncore.TargetState`（顶层导出） | `core/target.py` | `Enum` | 标记生命周期阶段（ACTIVE/LOST/LOCKED/RECOVERED/REMOVED） |
| `visioncore.state.target_state.TargetState` | `state/target_state.py` | `@dataclass(frozen=True, slots=True)` | 瞬时观测快照 |

Pipeline 选择**快照**而非枚举的原因：快照是"流动的数据"——正是 Context 应当持有的；枚举是 Target 上的元数据，已被 `targets` 字段间接包含。一个把 Target 投影为快照的 Stage 是自然的处理步骤，且其产出直接喂给协议层（B2）的 `ProtocolAdapter`。

### 3.4 工厂方法

```python
ctx = PipelineContext.empty(timestamp=0.0)   # 全空，仅设时间戳
```

`empty()` 保证所有集合字段独立（每个 Context 有自己的 `metadata` dict，不共享）。

### 3.5 内省

- `summary()`：单行紧凑摘要，如 `"frame=yes det=3 trk=2 tgt=2 ts=2 len=5"`，用于日志。
- `__repr__`：调用 `summary()`，即使集合含上千检测也保持紧凑。

---

## 4. Stage 生命周期

每个 `PipelineStage` 遵守严格的四方法生命周期，由外层 `Pipeline` 驱动。

### 4.1 四个方法

| 方法 | 抽象? | 调用时机 | 语义 |
|---|---|---|---|
| `initialize()` | ✅ 抽象 | 运行前**一次** | 获取资源（加载模型、预热分配器、读配置）。幂等——二次调用为 no-op。失败抛 `StageError`，且必须保证 `shutdown` 仍可安全调用。 |
| `process(context)` | ✅ 抽象 | **每次 run()** | Stage 的真正工作。读上游字段、做处理、写下游字段。异常原样传出 `run()` 并中止本次运行。 |
| `shutdown()` | ✅ 抽象 | 拆除时**一次** | 释放 `initialize` 获取的资源。幂等且**绝不抛异常**——抛了会掩盖触发拆除的真正错误。 |
| `health_check()` | ✅ 抽象 | 任意时刻 | 无副作用探针，返回 `True` 当且仅当 Stage 已初始化且健康。**绝不抛异常**——内部错误时返回 `False`。 |

### 4.2 调用顺序

```
Pipeline.initialize():   Stage1.init → Stage2.init → Stage3.init   (正向)
Pipeline.run(ctx):       Stage1.process(ctx) → Stage2.process(ctx) → Stage3.process(ctx)  (正向，可多次)
Pipeline.shutdown():     Stage3.shutdown → Stage2.shutdown → Stage1.shutdown   (逆向)
```

**逆向关闭**是刻意的：依赖型 Stage 先拆除，被依赖的 Stage 后拆除（如 TrackingStage 依赖 DetectionStage 的模型句柄，应先关 Tracking 再关 Detection）。

### 4.3 命名

每个 Stage 携带一个 `name`（默认为类名）。两个同类 Stage 可有不同 name——这对链式投影 Stage（如先投影位置再投影属性）至关重要，便于日志和诊断区分。

### 4.4 异常契约

| 抛出位置 | 行为 |
|---|---|
| `initialize()` 抛 | 该 Stage 及之后 Stage 不初始化；调用方需 `shutdown()` 清理已初始化的（或用上下文管理器）。异常传出 `initialize()`。 |
| `process()` 抛 | **立即中止本次 run()**，后续 Stage 不调用，异常原样传出 `run()`。`shutdown` 仍会被调用（用上下文管理器保证）。 |
| `shutdown()` 抛 | **被吞**：日志 WARNING 级别记录，继续拆除其余 Stage。终态标志仍置位。 |
| `health_check()` 抛 | **被吞**：该 Stage 视为不健康，`Pipeline.health_check()` 返回 `False`。 |

### 4.5 StageError

`base.py` 定义 `StageError(RuntimeError)`——结构化异常，Stage *可选*地用它标记"可恢复的阶段特定失败"（如模型加载失败、跟踪发散）。Pipeline **不**特殊捕获 `StageError`——它和普通异常一样传出 `run()`。调用方若想要重试/回退语义，自行 `try/except` 包裹 `run()`。

---

## 5. Pipeline 生命周期

### 5.1 三态状态机

Pipeline 通过内部标志 `_initialized` / `_shutdown` 维护三态：

```
                initialize()                    run(ctx)
   NEW  ──────────────────────>  INITIALIZED  ──────────────>  (可多次 run)
                                    │
                                    │ shutdown()
                                    ▼
                                SHUT_DOWN  (终态)
```

| 状态 | `_initialized` | `_shutdown` | 允许的操作 |
|---|---|---|---|
| **NEW** | False | False | `add_stage` / `remove_stage` / `initialize` |
| **INITIALIZED** | True | False | `run`（可多次）/ `shutdown` |
| **SHUT_DOWN** | True→False | True | （终态）`initialize` 和 `run` 均抛 `RuntimeError` |

- `initialize()` 幂等：二次调用为 no-op。
- `shutdown()` 幂等：二次调用为 no-op。
- **shutdown 是终态**：之后不能重新 `initialize`。这匹配"关闭文件"心智模型——重新处理另一个 Context 应创建新 `Pipeline` 实例（开销很小），避免半重建的资源状态。多次 `run` 复用单次 `initialize` 是支持的常见模式（见 §5.3）。

### 5.2 接口

| 方法 | 语义 |
|---|---|
| `add_stage(stage)` | 追加 Stage 到末尾，返回 `self`（支持链式）。仅在 NEW 态允许。拒绝 `None` 和非 `PipelineStage`。 |
| `remove_stage(stage_or_name)` | 按实例身份（`is`）或名字移除首个匹配，返回 `bool`。仅在 NEW 态允许。 |
| `initialize()` | 正向调用每个 Stage 的 `initialize()`。幂等。 |
| `run(context)` | 正向调用每个 Stage 的 `process(context)`，返回同一 `context`（便于链式）。 |
| `shutdown()` | **逆向**调用每个 Stage 的 `shutdown()`，best-effort（吞异常+日志）。幂等。 |
| `health_check()` | 聚合：所有 Stage 健康才返回 `True`，短路于首个不健康 Stage。空 Pipeline 返回 `True`（空真）。 |
| `stages` (property) | 只读 tuple 视图，插入顺序。 |
| `__enter__` / `__exit__` | 上下文管理器：进入时 `initialize`，退出时 `shutdown`（即使 `run` 抛异常也保证拆除）。 |

### 5.3 推荐用法

**模式 A：单次运行（上下文管理器）**
```python
p = Pipeline()
p.add_stage(CaptureStage()).add_stage(DetectionStage()).add_stage(TrackingStage())
with p:                              # initialize
    ctx = p.run(PipelineContext.empty(timestamp=t))
# shutdown() 自动调用，即使 run() 抛异常
```

**模式 B：多次运行复用单次初始化**
```python
with p:
    for frame in stream:
        ctx = PipelineContext.empty(timestamp=frame.timestamp)
        ctx.frame = frame
        p.run(ctx)                   # 复用 initialize，多次 process
# shutdown() 在 with 退出时调用一次
```

**模式 C：异常处理与回退**
```python
try:
    with p:
        p.run(ctx)
except StageError as e:
    # Stage 主动报告失败，可重试或回退到保守配置
    logger.warning("stage failed: %s", e)
    fallback_pipeline.run(ctx)
```

### 5.4 执行模型

Stage **同步、按插入顺序、在调用线程**运行。一次 run 内无并行——Stage N+1 在 Stage N 返回后才开始。这使得 C1 契约平凡正确、易推理。未来里程碑可能增加并行分支执行器，但串行执行器始终是参考实现。

### 5.5 线程安全

`Pipeline` 实例**非线程安全**。设计为单线程（Pipeline 拥有者）驱动。内部状态标志无锁——多线程并发 `run()` 会破坏生命周期状态。需要跨线程共享 Pipeline 的拥有者必须外部串行化。

---

## 6. DummyStage

### 6.1 用途

**可配置 no-op Stage**，用于测试 Pipeline 机制本身，不依赖任何真实视觉逻辑。让测试可验证：

- **Stage 顺序**：每次 `process()` 往 `context.metadata["order"]` 追加 marker，断言列表匹配插入顺序
- **Context 传递**：可变共享 Context，下游 Stage 观察到上游改动
- **异常传播**：`raise_on_process` 配置后，`process()` 抛指定异常
- **生命周期计数**：`initialize_count` / `process_count` / `shutdown_count` 精确断言

### 6.2 配置参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `name` | `"DummyStage"` | Stage 名称 |
| `marker` | = `name` | 追加到 `metadata["order"]` 的值 |
| `raise_on_process` | `None` | 异常实例，`process()` 时抛出（计数+1 后、stamp 前） |
| `healthy` | `False` | 初始健康标志；`initialize()` 置 True，`shutdown()` 置 False |

### 6.3 行为表

| 方法 | 行为 |
|---|---|
| `initialize()` | `initialize_count += 1`；`_healthy = True` |
| `process(ctx)` | `process_count += 1`；若 `raise_on_process` 非空则抛（stamp 前）；否则 `ctx.metadata["order"].append(marker)` |
| `shutdown()` | `shutdown_count += 1`；`_healthy = False`；绝不抛 |
| `health_check()` | 返回 `_healthy` |
| `reset()` | 清零计数与健康，保留 `marker`（便于测试复用） |

---

## 7. 测试覆盖

### 7.1 测试统计

- **测试文件**：`tests/test_pipeline.py`
- **测试函数数**：56（全部通过）
- **运行方式**：`PYTHONPATH=F:/VisionBata F:/VisionBata/venv/Scripts/python.exe tests/test_pipeline.py`（项目无 pytest，采用纯 assert + 手动 runner + 自定义 `raises` 上下文管理器，遵循 `test_obb_stage2.py` 约定）

### 7.2 四大必覆盖领域

| 领域 | 测试 | 验证 |
|---|---|---|
| **Stage 顺序** | `test_stage_order_preserved_in_run` | 三 Stage 的 `metadata["order"]` 匹配插入顺序 |
| | `test_stage_order_uses_marker_not_name` | 顺序列表记录 marker（可与 name 不同） |
| | `test_shutdown_runs_in_reverse_order` | `shutdown()` 逆向驱动（自定义 Stage 记录顺序） |
| **Context 传递** | `test_context_is_same_instance_across_stages` | 所有 Stage 收到同一 `id(context)` |
| | `test_context_downstream_reads_upstream_writes` | 下游 Stage 观察到上游对 metadata/detections 的改动 |
| | `test_run_returns_the_same_context_instance` | `run()` 返回传入的同一实例（非副本） |
| **异常传播** | `test_stage_exception_propagates_out_of_run` | Stage 抛 `ValueError` 原样传出 `run()` |
| | `test_stage_exception_aborts_subsequent_stages` | 抛异常后后续 Stage 的 `process_count == 0` |
| | `test_stage_error_is_a_stage_exception` | `StageError` 不被吞，原样传出 |
| | `test_exception_does_not_prevent_context_manager_teardown` | `with` 内 `run` 抛异常，`shutdown` 仍对每个 Stage 调用 |
| | `test_failed_stage_does_not_stamp_context_captured` | 抛异常的 Stage 不 stamp（先抛后 stamp） |
| **Stage 生命周期** | `test_lifecycle_calls_each_method_exactly_once` | 干净 run 后 init/process/shutdown 各 1 次 |
| | `test_lifecycle_initialize_before_process_before_shutdown` | 追踪顺序：init(正) → process(正) → shutdown(逆) |
| | `test_lifecycle_multiple_runs_reuse_initialization` | 一次 init 支持多次 run，shutdown 一次 |
| | `test_context_manager_is_single_use_due_to_terminal_shutdown` | shutdown 终态：重新 `with` 抛 RuntimeError |

### 7.3 框架契约补充测试

| 类别 | 测试数 | 覆盖 |
|---|---|---|
| 包导出 + 导入一致性 | 4 | `__all__` 精确匹配；ABC 不可实例化；`StageError` 是 `RuntimeError` 子类；缺方法子类不可实例化 |
| PipelineContext | 6 | `empty()` 默认值；时间戳默认；metadata 独立；`summary()` 字段；frame 存在性；`__repr__` 紧凑 |
| PipelineStage ABC | 4 | name 默认类名；name 显式覆盖；`__repr__` 报健康；`health_check` 抛异常时 `__repr__` 不崩 |
| DummyStage | 6 | 默认值；marker 覆盖 name；init 置健康；shutdown 置不健康；process stamp；reset 清零保留 marker |
| Pipeline 注册 | 8 | 顺序追加；链式返回 self；拒绝 None/非 Stage；init 后禁 add；按实例移除；按名移除；init 后禁 remove；len/repr 状态 |
| Pipeline 状态机 | 6 | run 前 init 抛；run 后 shutdown 抛；init 幂等；shutdown 幂等；shutdown 后 init 抛；未 init 直接 shutdown 安全 |
| health_check 聚合 | 4 | 空 Pipeline 空真健康；全健康 True；一不健康 False（短路）；抛异常 Stage 视为不健康 |
| 空边界 | 2 | 空 Pipeline run 是 no-op；空 Pipeline init 标记已初始化 |
| 端到端 | 1 | 三 Stage 链：stamp 顺序、返回 Context、全部 shutdown、shutdown 后不健康 |

### 7.4 回归测试

全量回归零破坏——所有 15 个既有测试套件通过：

```
test_core_models.py            : 31/31 passed
test_target_state.py           : 36/36 passed
test_event_bus.py              : 77/77 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_shadow_integration.py     : 12/12 passed
test_protocol_base.py          : 36/36 passed
test_debug_logger.py           : 20/20 passed
test_json_serializer.py        : 31/31 passed
test_cbor_serializer.py        : 31/31 passed
test_udp_adapter.py            : 23/23 passed
test_state_publisher.py        : 28/28 passed
test_shadow_publisher.py       : 36/36 passed
test_obb_stage2.py             : passed
test_matching.py               : passed
────────────────────────────────────────────────────────────────
test_pipeline.py (新增)        : 56/56 passed
总计                           : 既有全过 + 56 新增，0 回归
```

主应用导入冒烟测试通过：`visioncore.pipeline` 与顶层 `visioncore` 均正常导入，未触碰 `ai/` `gui/` `camera/`。

---

## 8. 未来扩展路线

C1 只建框架。真实处理能力在 C2+ 逐步接入，每个新 Stage 都是 `PipelineStage` 子类，`Pipeline` 与 `PipelineContext` 无需改动。

### 8.1 真实 Stage（C2 — 处理链接入）

| Stage | 文件 | 职责 | Context 字段 |
|---|---|---|---|
| `CaptureStage` | `pipeline/stages/capture.py` | 从摄像头/文件取帧，填充 `context.frame` | 写 `frame` |
| `DetectionStage` | `pipeline/stages/detection.py` | 调用检测后端（YOLOv8/YOLO26/ONNX/UHD），填充 `detections` | 读 `frame`，写 `detections` |
| `TrackingStage` | `pipeline/stages/tracking.py` | 关联检测跨帧成 Track，填充 `tracks` | 读 `detections`，写 `tracks` |
| `TargetManagementStage` | `pipeline/stages/target_management.py` | 包装 `TargetManager.update_targets()`，提升 Track 为 Target | 读 `tracks`，写 `targets` |
| `ProjectionStage` | `pipeline/stages/projection.py` | 把 Target 投影为 `TargetState` 快照，桥接 B2 协议层 | 读 `targets`，写 `target_states` |

每个 Stage 独立可测试——无需启动整个推理线程，只需构造 `PipelineContext` 喂入。

### 8.2 与既有系统集成（C3 — 影子旁路）

参考 eventbus 的 Shadow Integration 模式（`docs/shadow-integration-report.md`）：

- **阶段一（影子）**：在 `ai/inference.py` 现有逻辑旁挂一条 Pipeline，read-only 地复制输入帧跑一遍新 Stage 链，结果仅用于对比验证，不影响 GUI 输出
- **阶段二（双写）**：新旧两路并行产出，对比指标（检测数、跟踪稳定性、延迟）
- **阶段三（切换）**：验证通过后，Pipeline 成为主路径，`InferWorker` 退化为 Pipeline 的宿主线程

### 8.3 高级调度（C4+）

| 能力 | 说明 |
|---|---|
| **并行分支** | 多个独立 Stage 并行执行（如检测 + 属性增强同时跑），汇合后继续串行 |
| **条件跳过** | Stage 声明谓词，`process` 前评估是否跳过（如帧间隔 stride、低置信度跳过跟踪） |
| **异步 Stage** | 支持 `async def process`，Pipeline 用事件循环调度 |
| **Stage 图（DAG）** | 从线性列表升级为有向无环图，支持多输入多输出 Stage |
| **背压 / 限流** | Stage 声明产能，Pipeline 在过载时丢帧或排队 |

### 8.4 可观测性（C5）

- **Pipeline 仪表盘**：每个 Stage 的 `process` 耗时、异常率、健康状态聚合到 GUI
- **Stage 追踪**：`context.metadata["order"]` 扩展为带时间戳的 trace，可视化 Stage 链执行
- **健康监控集成**：`Pipeline.health_check()` 接入现有 `_HealthMonitor` 模式，连续不健康自动回退到保守 Stage

### 8.5 与 Protocol / EventBus 协同

```
Pipeline.run(ctx)
   ├── DetectionStage   ──写 detections──> ctx
   ├── TrackingStage    ──写 tracks──────> ctx
   ├── TargetMgmtStage  ──写 targets─────> ctx ──(状态变更)──> EventBus.publish(TargetLostEvent)
   └── ProjectionStage  ──写 target_states> ctx ──(快照流)──> ProtocolAdapter.publish(state)
```

- **EventBus**（进程内）：TargetManagementStage 在状态转换时发布生命周期事件，GUI 订阅更新
- **ProtocolAdapter**（跨进程）：ProjectionStage 产出快照流，经协议层投递到网络/文件/控制台

三者职责清晰：Pipeline 编排处理，EventBus 分发事件，ProtocolAdapter 投递快照。

---

## 9. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `visioncore/pipeline/` 目录 | ✅ |
| `pipeline/base.py` 定义 `PipelineStage(ABC)` | ✅ 四方法：initialize/process/shutdown/health_check |
| `pipeline/context.py` 定义 `PipelineContext` | ✅ 七字段：frame/detections/tracks/targets/target_states/timestamp/metadata |
| `pipeline/pipeline.py` 定义 `Pipeline` | ✅ add_stage/remove_stage/run + initialize/shutdown/health_check + 上下文管理器 |
| `pipeline/stage.py` 提供 `DummyStage` | ✅ 可配置 marker/raise_on_process/health |
| Pipeline 不含 Detector / Tracker / TargetManager | ✅ 纯调度器，零视觉逻辑 |
| 禁止修改 `ai/` `gui/` `camera/` | ✅ 仅新增文件，零侵入 |
| 不影响现有运行逻辑 | ✅ 全量回归零破坏 |
| 新增 `tests/test_pipeline.py` | ✅ 56 项测试，覆盖四大领域 + 框架契约 |
| 输出 `docs/pipeline-foundation.md` | ✅ 本文件 |

---

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/__init__.py` | 包导出 5 个公共名 + 范围/设计/快速开始说明 |
| `visioncore/pipeline/base.py` | `PipelineStage(ABC)` + `StageError` + 四方法生命周期契约 + `__repr__` |
| `visioncore/pipeline/context.py` | `PipelineContext` 可变数据黑板 + `empty()` 工厂 + `summary()` |
| `visioncore/pipeline/pipeline.py` | `Pipeline` 顺序执行器 + 三态状态机 + 上下文管理器 + 聚合 health_check |
| `visioncore/pipeline/stage.py` | `DummyStage` 可配置测试 Stage |
| `tests/test_pipeline.py` | 56 项单元测试 + 自定义 `raises` + 手动 runner（无 pytest 依赖） |
| `docs/pipeline-foundation.md` | 本设计文档 |

---

## 11. 后续里程碑预告

- **C2**：真实 Stage 接入——CaptureStage / DetectionStage / TrackingStage / TargetManagementStage / ProjectionStage 作为 `PipelineStage` 子类加入
- **C3**：InferWorker 影子旁路——在 `ai/inference.py` 旁挂 Pipeline，read-only 对比验证
- **C4**：并行分支 + 条件跳过 + 异步 Stage
- **C5**：Pipeline 仪表盘 + 健康监控集成 + Stage trace 可视化
- **D**：InferWorker 正式退役，Pipeline 成为主路径
