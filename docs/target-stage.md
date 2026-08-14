# TargetStage 设计文档 (Milestone C5)

> **版本**：C5 — 2026-07-25
> **范围**：新增 `visioncore/pipeline/stages/target_stage.py`，定义目标管理阶段 + TargetManager 接口 + DummyTargetManager
> **状态**：已实现，44 项单元测试全部通过，全量回归零破坏（19 个既有测试套件 + 44 项新测试）

---

## 1. 目标与背景

### 1.1 为什么需要 TargetStage

C3 建立检测阶段（frame → detections），C4 建立跟踪阶段（detections → tracks），但 Pipeline 的**目标管理**与**快照投影**仍未接入——`PipelineContext.targets` 和 `context.target_states` 两个字段没有生产者。

当前目标管理逻辑耦合在 `ai/inference.py` 的 InferWorker 中，直接调用 `visioncore.target_manager.TargetManager` 并耦合 GUI 更新。若 Pipeline 直接依赖具体 TargetManager 或 InferWorker，会产生：

- **UI 耦合**：目标管理与 GUI 更新绑死
- **推理线程耦合**：目标管理逻辑嵌入 InferWorker，无法独立测试
- **测试困难**：无法在不启动推理线程的情况下测试目标生命周期

TargetStage 通过**依赖反转**解决：定义抽象 `TargetManager` 接口（pipeline 阶段契约），TargetStage 依赖接口，DummyTargetManager 是测试双，具体目标管理器在构造时注入。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **接口优先** | `TargetStage(manager)` 依赖 `TargetManager` ABC，非具体类 |
| **双输出** | `update()` 返回 `TargetUpdate(targets, target_states)`，一次产出两者 |
| **禁止 GUI/InferWorker** | AST + 运行时双重守护 |
| **薄适配器** | TargetStage 自身无管理逻辑，全权委托给包装的 TargetManager |
| **REPLACE 语义** | 两个输出均 clear+extend（目标管理器持完整当前集合） |
| **空 tracks 不跳过** | `tracks=[]` 是"推进/老化"信号，总是调用 update |
| **异常保留先验** | update 抛异常时不清空 targets/target_states（clear 在 update 后） |
| **零侵入** | 不修改 `ai/` `camera/` `gui/` |

### 1.3 与 C1-C4 的关系

```
C2 FrameSource → C3 DetectorStage → detections → C4 TrackerStage → tracks
                                                                       │
                                                                       ▼
                                              C5 TargetStage.process(tracks)
                                                       │
                                            manager.update(tracks, ts)
                                                       │
                                              TargetUpdate
                                              ┌───────┴────────┐
                                              ▼                ▼
                                    context.targets    context.target_states
                                    (REPLACE)          (REPLACE)
                                              │                │
                                              ▼                ▼
                                  (C6 目标管理)      (B2 ProtocolAdapter 投递快照)
```

C5 是 Pipeline 的**末段**：读 C4 产出的 tracks，产出 targets（高层目标实体）和 target_states（可投递快照）。target_states 直接桥接 B2 协议层——ProjectionStage 的职责被合并进 TargetManager 接口（一次 update 产出两者）。

### 1.4 与 C3/C4 的对比

| 维度 | DetectorStage (C3) | TrackerStage (C4) | TargetStage (C5) |
|---|---|---|---|
| **输入** | `context.frame` | `context.detections` | `context.tracks` |
| **输出** | `context.detections` (extend) | `context.tracks` (replace) | `context.targets` + `context.target_states` (双 replace) |
| **无输入时** | frame=None → 跳过 | detections=[] → 仍调 update | tracks=[] → 仍调 update |
| **返回类型** | `list[Detection]` | `list[Track]` | `TargetUpdate(targets, target_states)` |
| **状态** | 无状态 | 有状态（跨帧 track） | 有状态（跨帧 target 生命周期） |

C5 的独特之处：**双输出**。一次 `update` 同时产出 targets（内部目标实体）和 target_states（外部可投递快照），封装了"目标管理 + 投影"两步。

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── __init__.py            # 包导出：Detector(C3) + Tracker(C4) + Target(C5)
├── detector_stage.py      # (C3) Detector ABC + DummyDetector + DetectorStage
├── tracker_stage.py       # (C4) Tracker ABC + DummyTracker + TrackerStage
└── target_stage.py        # (C5) TargetManager ABC + TargetUpdate + DummyTargetManager + TargetStage
```

---

## 3. 命名说明（强制阅读）

### 3.1 两个 TargetManager

| 类型 | 位置 | 性质 | 用途 |
|---|---|---|---|
| `visioncore.pipeline.stages.target_stage.TargetManager` | target_stage.py | **ABC 接口** | Pipeline 阶段依赖的契约（依赖反转） |
| `visioncore.target_manager.TargetManager` | target_manager/manager.py | **具体类** | 完整实现（store + lifecycle + event bus） |

两者同名但不同：本 ABC 是**最小契约**（4 方法），具体类是**完整实现**。未来里程碑用适配器让具体类符合本接口。用模块限定导入消歧：

```python
from visioncore.pipeline.stages.target_stage import TargetManager  # ABC
from visioncore.target_manager import TargetManager as ConcreteTargetManager
```

### 3.2 两个 TargetState

| 类型 | 位置 | 性质 | 在本模块的角色 |
|---|---|---|---|
| `visioncore.core.target.TargetState` | core/target.py | **生命周期枚举** (ACTIVE/LOST/...) | `Target.state` 字段 |
| `visioncore.state.target_state.TargetState` | state/target_state.py | **快照 dataclass** (frozen) | `TargetUpdate.target_states` 元素类型 |

本模块用别名消歧：`TargetLifecycleState`（枚举）、`TargetSnapshot`（快照）。这与 PipelineContext (C1) 和 ProtocolAdapter (B2) 的处理方式一致。

---

## 4. TargetManager 接口

### 4.1 类定义

```python
class TargetManager(ABC):
    @abstractmethod
    def initialize(self) -> None: ...
    @abstractmethod
    def update(self, tracks: list[Track], *, timestamp: float = 0.0) -> TargetUpdate: ...
    @abstractmethod
    def shutdown(self) -> None: ...
    @abstractmethod
    def health_check(self) -> bool: ...
```

### 4.2 四个方法

| 方法 | 语义 |
|---|---|
| `initialize()` | 获取/重置资源（清空目标存储、重置 ID 计数器）。幂等。失败抛 `TargetError`。 |
| `update(tracks, *, timestamp)` | 关联 tracks 与现有 targets，推进生命周期，投影快照，返回 `TargetUpdate`。`tracks` 可为空（"推进/老化"信号）。 |
| `shutdown()` | 释放资源。幂等，绝不抛异常。 |
| `health_check()` | 无副作用探针，返回 `True` 当且仅当已初始化且就绪。绝不抛异常。 |

### 4.3 update([]) 的"推进/老化"语义

调用 `update([])`（空 tracks）是合法的"本帧无新 track，推进/老化现有目标"信号。管理器应：
- 标记未匹配的 ACTIVE 目标为 LOST（超过 stale_threshold）
- 移除超过 removal_threshold 的 LOST 目标
- 投影剩余目标的快照
- 返回减少后的 target 集 + 快照

`update([])` **不是** no-op，**不抛异常**。TargetStage 据此在无 tracks 的帧仍调用 update。

### 4.4 TargetError

`TargetError(RuntimeError)` 是结构化异常，管理器 *可选* 地用它标记可恢复故障（存储不一致、生命周期转换被拒、投影产生无效快照）。Pipeline **不**特殊捕获——和普通异常一样传出 `run()`。

---

## 5. TargetUpdate 返回类型

### 5.1 定义

```python
@dataclass(frozen=True, slots=True)
class TargetUpdate:
    targets: list[Target]
    target_states: list[TargetSnapshot]
```

### 5.2 设计要点

- **frozen + slots**：与代码库数据模型风格一致（Event/Detection 等）
- **双列表**：封装"目标管理 + 投影"两步的产出
- **长度可不等**：`len(targets)` 不必等于 `len(target_states)`——管理器可只投影 ACTIVE 目标的快照，而 targets 含 LOST 目标
- **列表归调用方**：管理器返回后不得保留后改的引用

### 5.3 为何合并目标管理与投影

C1 路线图原计划分离 TargetMgmtStage 与 ProjectionStage。C5 合并它们，因为：
1. **投影依赖目标状态**：快照的 confidence/cx/cy 来自 target.track.detection，投影紧耦合目标管理
2. **减少阶段数**：一次 update 产出两者，避免两个阶段间的重复 target 遍历
3. **原子性**：targets 与 target_states 保持一致（同一 update 调用产出），不会出现 targets 更新了但 snapshots 是旧的

---

## 6. TargetStage

### 6.1 数据流

```
process(context):
    tracks = context.tracks
    result = manager.update(tracks, timestamp=context.timestamp)
    # update 成功后才 clear+extend（异常时保留先验）
    context.targets.clear()
    context.targets.extend(result.targets)
    context.target_states.clear()
    context.target_states.extend(result.target_states)
```

### 6.2 设计决策

| 决策 | 理由 |
|---|---|
| **双 REPLACE（clear+extend）** | 管理器持完整当前集合；保留 list 对象身份 |
| **总是调用 update** | `tracks=[]` 是"推进/老化"信号；管理器需推进目标生命周期 |
| **clear 在 update 后** | 若 update 抛异常，targets/target_states 保持原状（不丢失先验） |
| **转发 context.timestamp** | 管理器需时间戳做 stale/removal 决策，快照需时间戳 |
| **生命周期全委托** | stage 是薄适配器，无自身状态 |

### 6.3 异常时保留先验状态

关键实现：`process` 先调 `manager.update()`，**成功后**才 `clear()`+`extend()`。若 `update` 抛异常，两个 clear 都不执行，`context.targets` 和 `context.target_states` 保持原状。测试 `test_process_exception_preserves_prior_targets` 守护此语义——异常时不丢失既有 targets 和 snapshots。

---

## 7. DummyTargetManager

### 7.1 用途

**可配置测试管理器**——无存储、无生命周期、无匹配，返回预定的 `TargetUpdate`。

### 7.2 配置

| 参数 | 默认 | 作用 |
|---|---|---|
| `targets` | `None` | 返回的 Target 列表；`None` 时生成 1 个（S0-T0001, ACTIVE person） |
| `target_states` | `None` | 返回的快照列表；`None` 时生成 1 个（匹配默认 target） |
| `raise_on_update` | `None` | 异常实例，下次 `update()` 抛出；持续到清除 |

### 7.3 行为

- `update()` 返回**新列表**（TargetUpdate 持 fresh lists），记录 `last_tracks`/`last_timestamp`
- 计数器：`initialize_count`/`update_count`/`shutdown_count`
- `reset()` 清零计数与状态，保留 buffers

---

## 8. 测试覆盖

### 8.1 统计

- **测试文件**：`tests/test_target_stage.py`
- **测试函数数**：44（全部通过）

### 8.2 覆盖类别

| 类别 | 测试数 | 覆盖 |
|---|---|---|
| 包导出 + ABC 契约 | 4 | target 名在 __all__；TargetError 是 RuntimeError 子类；TargetManager 不可实例化；缺方法子类失败 |
| TargetUpdate | 3 | frozen+字段；持双列表；repr 紧凑 |
| DummyTargetManager | 8 | 默认双输出；自定义；fresh lists；记录 last_tracks/timestamp；生命周期+计数器；raise_on_update；持续到清除；reset |
| TargetStage 构造 | 6 | 包装 manager；默认名；自定义名；拒 None；拒非 TargetManager；是 PipelineStage 子类 |
| 生命周期委托 | 4 | initialize/shutdown/health_check 委托；repr |
| process 双输出数据流 | 8 | 读 tracks 写双输出；自定义；**REPLACE targets**；**REPLACE target_states**；保留 list 身份；**空 tracks 仍调 update**；转发 timestamp；update_count 递增；转发精确列表 |
| 异常传播 | 3 | TargetError 传播；任意异常传播；**异常保留先验 targets+snapshots** |
| Pipeline 集成 | 3 | 端到端；生命周期一次/多次 run |
| 端到端 Detect→Track→Target | 1 | C3→C4→C5 全链 |
| 无 GUI/InferWorker | 3 | AST 无 gui/PyQt/inference；运行时不加载 PyQt/ai.inference；接口仅 4 方法 |

### 8.3 关键测试

| 测试 | 验证 |
|---|---|
| `test_process_reads_tracks_writes_both_outputs` | 双输出：tracks → targets + target_states |
| `test_process_replaces_not_extends_targets` | REPLACE targets（旧 target 被清空） |
| `test_process_replaces_not_extends_target_states` | REPLACE target_states（旧快照被清空） |
| `test_process_with_empty_tracks_still_calls_update` | 空 tracks 不跳过 |
| `test_process_forwards_timestamp` | context.timestamp 转发给 manager |
| `test_process_exception_preserves_prior_targets` | 异常时 targets+snapshots 都保留 |
| `test_end_to_end_detect_track_target` | C3→C4→C5 全链一致 |
| `test_no_gui_inferworker_in_source` | AST 无 GUI/InferWorker |

### 8.4 回归测试

全量回归零破坏——20 个套件全过：

```
test_target_stage.py (C5 新增)   : 44/44 passed
test_tracker_stage.py (C4)       : 42/42 passed
test_detector_stage.py (C3)      : 39/39 passed
test_frame_source.py (C2)        : 55/55 passed
test_pipeline.py (C1)            : 56/56 passed
... (15 个既有套件全过)
────────────────────────────────────────────────────────────────
总计                             : 既有全过 + 44 新增，0 回归
```

---

## 9. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `target_stage.py` | ✅ |
| TargetStage 继承 PipelineStage | ✅ |
| 输入 tracks | ✅ |
| 调用 TargetManager | ✅ TargetManager ABC 接口 |
| 输出 targets + target_states | ✅ TargetUpdate 双输出 |
| 禁止 GUI / InferWorker | ✅ AST + 运行时双重守护 |
| 新增 DummyTargetManager | ✅ 返回固定 targets + snapshots |
| 新增 `tests/test_target_stage.py` | ✅ 44 项测试 |
| 输出 `docs/target-stage.md` | ✅ 本文件 |
| 不修改 ai/camera/gui | ✅ 仅新增文件 + 更新 stages/__init__.py |

---

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/stages/target_stage.py` | TargetManager ABC + TargetError + TargetUpdate + DummyTargetManager + TargetStage |
| `visioncore/pipeline/stages/__init__.py` | 更新：追加 target 导出 |
| `tests/test_target_stage.py` | 44 项单元测试 |
| `docs/target-stage.md` | 本设计文档 |

---

## 11. 未来扩展路线

### 11.1 真实目标管理器适配（C6）

将现有 `visioncore.target_manager.TargetManager`（具体类）适配为本 ABC 接口：

```python
class ConcreteTargetManagerAdapter(TargetManager):
    def __init__(self, concrete: visioncore.target_manager.TargetManager):
        self._inner = concrete
    def update(self, tracks, *, timestamp=0.0):
        self._inner.update_targets(tracks, slot_id=0, timestamp=timestamp)
        targets = self._inner.get_all_targets()
        snapshots = [project(t) for t in targets if t.state == TargetState.ACTIVE]
        return TargetUpdate(targets, snapshots)
```

### 11.2 完整 Pipeline 链（C6+）

```
CaptureStage(C2 FrameSource) → DetectorStage(C3) → TrackerStage(C4) → TargetStage(C5)
                                                                        │
                                                                        ▼
                                                    target_states → B2 ProtocolAdapter
                                                    targets → EventBus 生命周期事件
```

C5 完成了链的末段。仅剩 CaptureStage（包装 FrameSource 填 context.frame）未实现。

### 11.3 多摄像头与融合（C7+）

- 每个摄像头一条 Pipeline 链，独立 TargetStage
- 跨摄像头融合：ReID + 全局 ID 分配，融合产物写入 global target_states

---

## 12. 后续里程碑预告

- **C6**：CaptureStage（包装 FrameSource）+ 真实后端适配器（Detector/Tracker/TargetManager 包装现有 ai/ 实现）→ 完整 Pipeline 链
- **C7**：InferWorker 影子旁路——Pipeline 与现有 InferWorker 并行对比验证
- **D**：InferWorker 退役，Pipeline 转正
