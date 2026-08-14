# EventStage 设计文档 (Milestone C6)

> **版本**：C6 — 2026-07-25
> **范围**：新增 `visioncore/pipeline/stages/event_stage.py`，定义事件发射 sink 阶段 + EventBus 接口 + DummyEventBus
> **状态**：已实现，38 项单元测试全部通过，全量回归零破坏（20 个既有测试套件 + 38 项新测试）

---

## 1. 目标与背景

### 1.1 为什么需要 EventStage

C3-C5 建立了检测→跟踪→目标管理链，产出 `context.targets` 和 `context.target_states`。但 Pipeline 的**事件输出**仍未接入——target_states 快照无法到达 EventBus 订阅者（GUI 更新、日志、告警）。

当前事件发布逻辑耦合在 `ai/inference.py` 的 InferWorker 中，直接调用 EventBus 并混入网络/协议逻辑。若 Pipeline 直接依赖网络/协议层，会产生：

- **传输耦合**：事件发射绑死到 UDP/协议适配器
- **测试困难**：无法在不启动网络栈的情况下测试事件流
- **职责混淆**：进程内事件分发与跨进程传输混在一起

EventStage 通过**纯进程内 sink** 解决：定义抽象 `EventBus` 接口（仅 `publish`），EventStage 读 target_states 生成 Event 发布到 bus，**禁止**网络/UDP/Protocol。跨进程传输是 ProtocolAdapter（B2）的职责，不是本阶段。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **纯进程内** | **禁止**网络/UDP/Protocol；EventStage 只发布到 in-process EventBus |
| **接口优先** | `EventStage(bus)` 依赖 `EventBus` ABC（仅 `publish`），非具体类 |
| **Sink 阶段** | 只读 `context.target_states`，**不写** context（终端输出阶段） |
| **一快照一事件** | 每个 target_state 生成一个 Event，event_type 可配置 |
| **时间戳来自快照** | Event.timestamp = snapshot.timestamp（数据时间一致性，非墙钟） |
| **零侵入** | 不修改 `ai/` `camera/` `gui/` |

### 1.3 与 C3-C5 的关键差异：Sink 阶段

| 维度 | Detector/Tracker/Target (C3-C5) | EventStage (C6) |
|---|---|---|
| **context 交互** | 读一字段，写另一字段 | **只读** target_states，不写 |
| **阶段角色** | 处理器（数据变换） | **Sink**（终端输出） |
| **生命周期** | initialize/shutdown 委托给包装对象 | initialize/shutdown **no-op**（bus 构造即就绪） |
| **包装接口** | Detector/Tracker/TargetManager（处理器） | EventBus（sink，仅 publish） |

EventStage 是 Pipeline 事件侧的**终端**——它消费 target_states 并发射事件，不产生供下游阶段消费的 context 字段。

### 1.4 与 EventBus / Protocol 的职责划分

```
EventStage (C6)          ──publish(Event)──>  EventBus (进程内, 订阅/分发)
  读 target_states                                │
  生成 Event                                       ├── GUI 订阅者 (更新显示)
                                                  ├── 日志订阅者 (审计)
                                                  └── 告警订阅者 (阈值告警)

ProtocolAdapter (B2)     <──publish(TargetState)──  (未来 CaptureStage 或直接接线)
  跨进程投递快照                                   └── UDP/文件/网络 sink
```

- **EventBus**（进程内）：EventStage 用它做进程内事件分发（GUI/日志/告警订阅）
- **ProtocolAdapter**（跨进程）：投递 TargetState 快照到外部系统（UDP/文件）

两者正交：EventStage 只用 EventBus（进程内），**禁止** ProtocolAdapter（跨进程）。跨进程投递由未来阶段或直接接线 ProtocolAdapter 完成，不属本阶段。

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── __init__.py            # 包导出：Detector(C3) + Tracker(C4) + Target(C5) + Event(C6)
├── detector_stage.py      # (C3)
├── tracker_stage.py       # (C4)
├── target_stage.py        # (C5)
└── event_stage.py         # (C6) EventBus ABC + DummyEventBus + EventStage
```

---

## 3. EventBus 接口

### 3.1 类定义

```python
class EventBus(ABC):
    @abstractmethod
    def publish(self, event: Event) -> int: ...

    def health_check(self) -> bool:
        """Concrete default: always True. Override for custom logic."""
        return True
```

### 3.2 设计要点

- **极简接口**：仅 `publish`（抽象）+ `health_check`（具体默认 True）。订阅/退订/类型化事件机制在具体 `visioncore.eventbus.EventBus` 上，不在本契约
- **publish 返回 int**：成功通知的订阅者数（匹配具体 EventBus 语义）；无订阅者返回 0
- **health_check 具体默认**：in-process bus 构造即就绪，默认健康；子类可覆盖

### 3.3 命名说明

| 类型 | 位置 | 性质 |
|---|---|---|
| `visioncore.pipeline.stages.event_stage.EventBus` | event_stage.py | **ABC 接口**（pipeline 阶段契约） |
| `visioncore.eventbus.EventBus` | eventbus/bus.py | **具体类**（subscribe/unsubscribe/dispatcher/typed events） |

两者同名：本 ABC 是最小 `publish` 契约（依赖反转），具体类是完整实现（已有 `publish`，符合本接口）。用模块限定导入消歧。

---

## 4. EventStage

### 4.1 数据流

```
process(context):
    for snapshot in context.target_states:
        event = Event(
            event_type=self._event_type,        # 默认 "target.snapshot"
            timestamp=snapshot.timestamp,        # 来自快照，非墙钟
            payload={"snapshot": snapshot},      # 快照对象直接携带（frozen, 安全共享）
        )
        bus.publish(event)
```

### 4.2 设计决策

| 决策 | 理由 |
|---|---|
| **Sink（不写 context）** | 终端输出阶段；target_states 已由 TargetStage 产出，EventStage 只消费 |
| **一快照一事件** | 简单正确；变更检测（仅状态变化时发射）是未来优化 |
| **timestamp 来自快照** | 事件时间与数据时间一致，不因 publish 延迟而漂移 |
| **payload 携带快照对象** | 快照是 frozen（immutable），共享引用安全；避免 to_dict() 拷贝开销 |
| **event_type 可配置** | 不同实例可发射不同事件类型（target.active / target.lost / target.snapshot） |
| **initialize/shutdown no-op** | bus 构造即就绪；stage 不拥有 bus（调用方管理 bus 生命周期） |
| **空 target_states 不报错** | 发布零事件是合法的"本帧无快照"，非错误 |

### 4.3 构造

```python
EventStage(event_bus: EventBus, *, name="EventStage", event_type="target.snapshot")
```

- `event_bus` 必须是 `EventBus` 实例（接口强制）。`TypeError` 守护。
- `event_type` 非空字符串。`ValueError` 守护空串。

### 4.4 异常传播

若 `bus.publish(event)` 抛异常（如 bus 内部分发错误），异常原样传出 `process()`，**后续快照不再发布**（循环中断）。这遵循 C1"诚实异常"契约——不静默吞掉 bus 故障。测试 `test_process_exception_stops_at_failing_snapshot` 守护：第 N 个快照 publish 失败时，N+1 及之后不发布。

---

## 5. DummyEventBus

### 5.1 用途

**测试事件总线**——无真实订阅者分发，将每个 published Event 追加到列表，使事件流可观测可断言。

### 5.2 配置与行为

| 属性 | 作用 |
|---|---|
| `published` | 已发布事件列表（publish 顺序），每个元素是原 Event 对象（非拷贝） |
| `raise_on_publish` | 异常实例，下次 `publish()` 抛出（异常传播测试）；持续到清除 |
| `publish_count` | publish 调用次数（含抛异常的调用） |
| `event_types` | 便捷属性：已发布事件的 event_type 字符串列表 |
| `reset()` | 清空 published 与 publish_count |

---

## 6. 测试覆盖

### 6.1 统计

- **测试文件**：`tests/test_event_stage.py`
- **测试函数数**：38（全部通过）

### 6.2 覆盖类别

| 类别 | 测试数 | 覆盖 |
|---|---|---|
| 包导出 + ABC 契约 | 4 | event 名在 __all__；EventBus 不可实例化；缺 publish 子类失败；health_check 默认 True |
| DummyEventBus | 6 | 记录事件；返回 1；publish_count 含异常；raise 持续到清除；event_types 属性；reset |
| EventStage 构造 | 8 | 包装 bus；默认名；自定义名；默认 event_type；自定义 event_type；拒 None；拒非 bus；拒空 event_type；是 PipelineStage |
| 生命周期 | 4 | initialize no-op；shutdown no-op；health_check 委托；repr |
| process 数据流 | 7 | 一快照一事件；空快照零事件；event_type 匹配配置；timestamp 来自快照；payload 携带快照；published_count 递增；**不修改 context（sink）** |
| 异常传播 | 2 | bus 异常传播；异常停止于失败快照（后续不发布） |
| Pipeline 集成 | 2 | 端到端；生命周期 |
| 端到端全链 | 1 | C3→C4→C5→C6：Detect→Track→Target→Event |
| 无网络/UDP/Protocol | 3 | AST 无 socket/udp/protocol；运行时不加载 visioncore.protocol/socket；接口极简（仅 publish 抽象） |

### 6.3 关键测试

| 测试 | 验证 |
|---|---|
| `test_process_does_not_modify_context` | **Sink 语义**：process 不写 context |
| `test_process_event_timestamp_from_snapshot` | timestamp 来自快照非墙钟 |
| `test_process_event_payload_carries_snapshot` | payload 携带快照对象（is 检查） |
| `test_process_with_empty_target_states_publishes_zero` | 空快照不报错，发零事件 |
| `test_process_exception_stops_at_failing_snapshot` | 异常中断循环，后续不发布 |
| `test_end_to_end_full_chain` | C3→C4→C5→C6 全链一致 |
| `test_no_network_udp_protocol_in_source` | AST 无 socket/udp/protocol |
| `test_no_protocol_loaded_at_runtime` | 运行时不加载 visioncore.protocol |

### 6.4 回归测试

全量回归零破坏——21 个套件全过：

```
test_event_stage.py (C6 新增)   : 38/38 passed
test_target_stage.py (C5)       : 44/44 passed
test_tracker_stage.py (C4)      : 42/42 passed
test_detector_stage.py (C3)     : 39/39 passed
test_frame_source.py (C2)       : 55/55 passed
test_pipeline.py (C1)           : 56/56 passed
... (15 个既有套件全过)
────────────────────────────────────────────────────────────────
总计                            : 既有全过 + 38 新增，0 回归
```

---

## 7. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `event_stage.py` | ✅ |
| EventStage 继承 PipelineStage | ✅ |
| 读取 TargetStates | ✅ `context.target_states` |
| 生成 Vision Events | ✅ 每个 snapshot 一个 `visioncore.core.event.Event` |
| 发送 EventBus | ✅ EventBus ABC 接口，`publish(event)` |
| 禁止网络/UDP/Protocol | ✅ AST + 运行时双重守护 |
| 新增 DummyEventBus | ✅ 记录已发布事件 |
| 新增 `tests/test_event_stage.py` | ✅ 38 项测试 |
| 输出 `docs/event-stage.md` | ✅ 本文件 |
| 不修改 ai/camera/gui | ✅ 仅新增文件 + 更新 stages/__init__.py |

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/stages/event_stage.py` | EventBus ABC + DummyEventBus + EventStage |
| `visioncore/pipeline/stages/__init__.py` | 更新：追加 event 导出（3 名） |
| `tests/test_event_stage.py` | 38 项单元测试 |
| `docs/event-stage.md` | 本设计文档 |

---

## 9. 未来扩展路线

### 9.1 真实 EventBus 接入（C7）

将现有 `visioncore.eventbus.EventBus`（具体类）直接用作 EventStage 的 bus——它已有 `publish` 方法，符合本 ABC 接口。订阅者（GUI、日志、告警）注册到具体 bus，EventStage 发布的事件自动路由到它们。

### 9.2 变更检测事件（C7+）

当前 EventStage 每帧每个快照发一个事件。未来可加"变更检测"模式：
- 仅当 target 状态变化（ACTIVE→LOST、新 target、位置跳变）时发射事件
- 使用 typed events（TargetCreatedEvent / TargetLostEvent / TargetRecoveredEvent）替代通用 Event
- 减少事件量，聚焦状态转换

### 9.3 多事件源 Pipeline（C8）

一个 Pipeline 可链多个 EventStage，每个用不同 event_type：
- `EventStage(bus1, event_type="target.snapshot")` — 全量快照流
- `EventStage(bus2, event_type="target.alert")` — 仅告警级事件
- 各自发布到不同 bus，订阅者按 bus 隔离

### 9.4 与 Protocol 层协同（C8+）

```
Pipeline 末段:
  TargetStage ──> context.target_states ──┬──> EventStage ──> EventBus (进程内)
                                          └──> (未来) ProtocolStage ──> ProtocolAdapter (跨进程)
```

EventStage 用 EventBus 做进程内分发；未来 ProtocolStage 用 ProtocolAdapter 做跨进程投递。两者从同一 target_states 读取，互不干扰。

---

## 10. 后续里程碑预告

- **C7**：CaptureStage（包装 FrameSource 填 context.frame）+ 真实 EventBus 接入 → 完整 Pipeline 链 + 事件分发
- **C8**：变更检测事件 + typed events + 多事件源 Pipeline
- **D**：InferWorker 影子旁路 → 退役，Pipeline 转正
