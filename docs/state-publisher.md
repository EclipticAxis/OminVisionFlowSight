# StatePublisher 设计文档 (Milestone B6)

> **版本**：B6 — 2026-07-20
> **范围**：新增 `visioncore/protocol/publisher/state_publisher.py`，实现 StatePublisher
> **状态**：已实现，28 项单元测试全部通过，410 项回归零破坏

---

## 1. 目标与背景

### 1.1 为什么需要 StatePublisher

VisionCore 现有两个独立的管道：

1. **EventBus 管道**（B2 阶段 3）：TargetManager 在目标生命周期转换时发布事件（TargetCreatedEvent / TargetLostEvent / ...）。这些事件是**控制平面**——通知系统"某个目标的状态变了"。

2. **ProtocolAdapter 管道**（B2-B5）：ProtocolAdapter 接收 TargetState 快照并发送到下游（UDP / 控制台 / 文件）。这些快照是**数据平面**——传递"某个目标此刻的观测状态"。

两条管道之间没有桥梁——EventBus 的事件不会自动流到 ProtocolAdapter，ProtocolAdapter 也不知道 EventBus 的存在。

**StatePublisher 就是这座桥**：

```
TargetManager ──(transition)──> EventBus ──(event)──> StatePublisher
                                                          │
                                                   _event_to_state()
                                                          │
                                                          ▼
                                                    TargetState
                                                          │
                                              ProtocolAdapter.publish()
                                                          │
                                                          ▼
                                                  UDP / Console / File
```

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **纯观察者** | 只读 EventBus，只写 ProtocolAdapter。不修改 TargetManager / Target / EventBus 状态 |
| **不修改 GUI** | gui/ 目录零改动（测试 `test_no_gui_files_modified` 守护） |
| **不修改 InferWorker** | ai/inference.py 零改动（测试 `test_no_inferworker_modification` 守护） |
| **错误隔离** | adapter.publish() 失败不传播回 EventBus——捕获+日志，总线继续运行 |
| **类型安全订阅** | 使用类订阅（`TargetCreatedEvent` 而非字符串 `"target.created"`），handler 收到精确类型 |

---

## 2. 文件结构

```
visioncore/protocol/
├── __init__.py                    # B2: ProtocolAdapter, NullAdapter, ConsoleAdapter
├── base.py                        # B2: ProtocolAdapter ABC
├── null_adapter.py                # B2: NullAdapter
├── console_adapter.py             # B2: ConsoleAdapter
├── serialization/                 # B3/B4
│   ├── json_serializer.py         # B3: JSON
│   └── cbor_serializer.py         # B4: CBOR
├── adapters/                      # B5
│   └── udp_adapter.py             # B5: UDPAdapter
└── publisher/                     # B6（新增）
    ├── __init__.py
    └── state_publisher.py         # B6: StatePublisher
```

`publisher/` 子包与 `serialization/` / `adapters/` 平行——序列化负责"快照→字节"，适配器负责"字节→传输"，发布者负责"事件→快照"。

**未修改** `visioncore/protocol/__init__.py`——StatePublisher 通过 `from visioncore.protocol.publisher import StatePublisher` 访问。

---

## 3. 构造与接口

### 3.1 构造器

```python
StatePublisher(
    event_bus,    # EventBus — 订阅事件的目标总线
    adapter,      # ProtocolAdapter — 发布快照的目标适配器
)
```

StatePublisher 不是 ProtocolAdapter 的子类——它是一个**驱动者**（driver），持有适配器引用并在事件到达时调用 `adapter.publish()`。

### 3.2 生命周期方法

| 方法 | 行为 |
|---|---|
| `start()` | 调用 `adapter.connect()`，然后用类订阅注册 5 个事件处理器。幂等。 |
| `stop()` | 取消全部 5 个订阅，调用 `adapter.disconnect()`。幂等，绝不抛。 |
| `__enter__` / `__exit__` | 上下文管理器：enter 调 start，exit 调 stop（即使异常也 stop）。 |
| `is_started` (property) | 返回当前是否已启动。 |
| `adapter` (property) | 返回注入的 ProtocolAdapter。 |

### 3.3 事件订阅

使用 EventBus 的**类订阅**模式（非字符串订阅）：

```python
self._bus.subscribe(TargetCreatedEvent, self._on_event)
self._bus.subscribe(TargetLostEvent, self._on_event)
self._bus.subscribe(TargetRecoveredEvent, self._on_event)
self._bus.subscribe(TargetLockedEvent, self._on_event)
self._bus.subscribe(TargetRemovedEvent, self._on_event)
```

5 个事件类型共享同一个 handler `_on_event`——因为转换逻辑对所有事件类型一致（提取数据 → 生成 TargetState → 发布）。不同事件的差异通过 `event.event_type` 在 metadata 中区分。

类订阅的优势：handler 收到的是精确的事件子类实例（`TargetCreatedEvent`），不是任何 `event_type` 匹配的泛型 Event。类型安全。

### 3.4 事件 → TargetState 转换

`_event_to_state(event)` 方法将生命周期事件转换为 TargetState 快照：

| TargetState 字段 | 数据来源 |
|---|---|
| `target_id` | `payload.get("track_id")` 或从 `event.target_id` 字符串解析的 seq |
| `local_id` | 同 `target_id` |
| `global_id` | `None`（B6 无跨摄像头融合） |
| `label` | `payload["detection"]["class_name"]` 或 `"unknown"` |
| `confidence` | `payload["detection"]["score"]` 或 `0.0` |
| `cx`, `cy` | `payload["detection"]["bbox"]["x"/"y"]` 或 `0.0` |
| `vx`, `vy` | `0.0`（事件不携带速度数据） |
| `width`, `height` | `payload["detection"]["bbox"]["w"/"h"]` 或 `0.0` |
| `timestamp` | `event.timestamp` |
| `camera_id` | `event.slot_id` |
| `metadata` | `{"event_type": ..., "event_id": ..., **payload}`（排除 `detection` 键） |

**target_id 解析**：事件中的 `target_id` 是字符串格式 `"S{slot}-T{seq:04d}"`（如 `"S0-T0001"`），由 `target_id_factory` 生成。StatePublisher 用正则 `_TARGET_ID_RE` 解析出 `(slot, seq)` 元组，`seq` 作为 `TargetState.target_id`（int）。

**detection 提取**：如果 `payload` 包含 `detection` 字典（典型为 TargetCreatedEvent 的 payload），从中提取 bbox/score/class_name 填充位置字段。detection 键本身从 metadata 中排除（已提取到位置字段，不重复存储）。

**metadata 构建**：始终包含 `event_type` + `event_id`（让接收端知道生命周期上下文），合并 payload 中的其余字段（如 `track_id` / `priority` / `locked_by` / `reason` 等）。

---

## 4. 错误隔离

### 4.1 设计

StatePublisher 的 `_on_event` handler 有两层 try/except：

```python
def _on_event(self, event):
    try:
        state = self._event_to_state(event)
    except Exception:
        logger.exception("failed to convert event to TargetState")
        return  # 不发布，但不崩溃

    try:
        self._adapter.publish(state)
    except Exception:
        logger.warning("adapter.publish() failed, ignoring", exc_info=True)
```

### 4.2 为什么重要

EventBus 的回调在**发布线程**上同步执行。如果 StatePublisher 的 handler 抛出未捕获异常：

1. EventBus 的异常隔离会捕获它（`logger.exception` + 吞掉）
2. 但该 subscriber 在后续发布中可能被跳过（取决于 EventBus 实现）
3. 更重要的是：**单次传输失败不应影响事件分发**

StatePublisher 显式捕获所有异常，确保：
- 网络故障（UDP 不可达）不崩溃总线
- 序列化错误（metadata 含不可序列化值）不崩溃总线
- 适配器未连接不崩溃总线

测试 `test_adapter_failure_does_not_crash_bus` 验证：FailingAdapter 的 `publish()` 抛 `OSError` 后，后续事件仍正常分发。

---

## 5. 不修改约束验证

### 5.1 不修改 GUI

测试 `test_no_gui_files_modified` 扫描 `gui/` 目录的全部 `.py` 文件，验证不含 `"StatePublisher"` 或 `"state_publisher"` 引用。

### 5.2 不修改 InferWorker

测试 `test_no_inferworker_modification` 检查 `ai/inference.py` 不含 `"StatePublisher"` 或 `"state_publisher"` 引用。

### 5.3 集成方式

StatePublisher 在**调用方**（如 main.py 或未来的启动脚本）中创建和启动，不在 InferWorker 内部：

```python
# 未来的 main.py 集成（不在 B6 范围内）
bus = EventBus()
adapter = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
publisher = StatePublisher(bus, adapter)
publisher.start()

# InferWorker 使用同一个 bus（B2 影子集成已建立）
# TargetManager 发事件 → bus → publisher → adapter → UDP
```

---

## 6. 测试覆盖

### 6.1 测试统计

- **测试文件**：`tests/test_state_publisher.py`
- **测试函数数**：28（全部通过）

### 6.2 覆盖类别

| 类别 | 测试数 | 覆盖点 |
|---|---|---|
| 类型注解约定 | 1 | AST 检查无 forbidden import |
| 不修改约束 | 2 | GUI 文件扫描 / InferWorker 文件扫描 |
| 包导出 + 导入一致性 | 2 | __all__ / 直接导入一致 |
| 生命周期 | 6 | start 订阅5事件 / start 连接适配器 / start 幂等 / stop 幂等 / stop 取消订阅 / 上下文管理器 |
| 上下文异常 | 1 | 异常时仍 stop |
| 事件→快照转换 | 8 | 5 种事件各一 + detection 提取 + target_id 解析 + 无 track_id 时用 seq |
| 全事件流 | 1 | 5 种事件连续发布，顺序保持 |
| 错误隔离 | 1 | FailingAdapter 不崩溃总线 |
| 不修改总线 | 1 | publisher 只订阅不发布事件 |
| metadata | 2 | 含 event_type+event_id / 排除 detection 键 |
| 字段默认值 | 2 | vx/vy=0.0 / global_id=None |
| repr | 1 | 含 started+adapter+subscriptions |

### 6.3 回归测试

```
test_target_state.py           : 36/36 passed
test_core_models.py            : 31/31 passed
test_event_bus.py              : 77/77 passed
test_shadow_integration.py     : 12/12 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_protocol_base.py          : 36/36 passed
test_json_serializer.py        : 31/31 passed
test_cbor_serializer.py        : 31/31 passed
test_udp_adapter.py            : 23/23 passed
test_state_publisher.py        : 28/28 passed  (新增)
────────────────────────────────────────────────────────────────
总计                           : 410/410 passed, 0 回归
```

---

## 7. 使用示例

### 7.1 基本使用

```python
from visioncore.eventbus import EventBus
from visioncore.protocol import NullAdapter
from visioncore.protocol.publisher import StatePublisher

bus = EventBus()
adapter = NullAdapter()  # 或 UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)

with StatePublisher(bus, adapter) as publisher:
    # 在此期间，TargetManager 在 bus 上发布的事件会自动
    # 转换为 TargetState 并通过 adapter 发布
    ...
# stop() 自动调用
```

### 7.2 与 UDP 传输集成

```python
from visioncore.protocol.adapters import UDPAdapter
from visioncore.protocol.serialization import CBORSerializer

bus = EventBus()
adapter = UDPAdapter(
    serializer=CBORSerializer(),
    host="127.0.0.1",
    port=9999,
)

publisher = StatePublisher(bus, adapter)
publisher.start()

# 现在 TargetManager 的生命周期事件通过 CBOR 编码
# 经 UDP 发送到 127.0.0.1:9999
```

### 7.3 与 Console 调试集成

```python
from visioncore.protocol import ConsoleAdapter

adapter = ConsoleAdapter()
publisher = StatePublisher(bus, adapter)
publisher.start()

# 每个生命周期事件在控制台输出一行日志
# INFO: [PUBLISH] id=42 label='person' conf=0.920 pos=(0.5000,0.4000) ...
```

---

## 8. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `protocol/publisher/state_publisher.py` | ✅ |
| `StatePublisher` 类 | ✅ |
| 订阅 5 种生命周期事件 | ✅ 类订阅 TargetCreated/Lost/Recovered/Locked/Removed |
| 生成 TargetState | ✅ `_event_to_state` 转换 |
| 调用 `ProtocolAdapter.publish()` | ✅ `_on_event` 中调用 |
| 禁止修改 GUI | ✅ 测试 `test_no_gui_files_modified` 守护 |
| 禁止修改 InferWorker | ✅ 测试 `test_no_inferworker_modification` 守护 |
| 输出 `docs/state-publisher.md` | ✅ 本文件 |

---

## 9. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/protocol/publisher/__init__.py` | 子包导出 StatePublisher |
| `visioncore/protocol/publisher/state_publisher.py` | StatePublisher 类 + `_parse_target_id` 辅助 |
| `tests/test_state_publisher.py` | 28 项单元测试 + `__main__` runner |
| `docs/state-publisher.md` | 本文件 |

---

## 10. 后续里程碑

- **B7**：`main.py` 集成 StatePublisher（将 bus + adapter + publisher 接入启动流程）
- **B8**：`FileAdapter` — 录制元数据文件 sink
- **C**：健康监控集成 + 自动回退 NullAdapter
- **D**：`UDPReceiver` — 接收端反序列化快照流
- **E**：`TargetStore` 查询增强 — StatePublisher 可选查询 TargetStore 获取完整 Track 数据（位置/速度）
