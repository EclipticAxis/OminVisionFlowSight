# Protocol Foundation 设计文档 (Milestone B2)

> **版本**：B2 — 2026-07-19
> **范围**：新增 `visioncore/protocol/` 包，定义统一的协议适配器抽象层
> **状态**：已实现，36 项单元测试全部通过，297 项回归零破坏

---

## 1. 目标与背景

### 1.1 为什么需要 Protocol Layer

B1 定义了 `TargetState` 快照——一个不可变的目标观测数据结构。随着 VisionCore 向跨阶段、跨线程、跨摄像头方向演进，这些快照需要被**投递**到各种下游消费者：

- 调试控制台（开发期人类可读输出）
- 网络传输（跨进程/跨机器，未来 UDP adapter）
- 文件存储（录制元数据，未来 FileAdapter）
- 消息总线（EventBus 已有，但协议层提供更通用的 sink 抽象）

如果每个生产者都直接硬编码具体传输方式（`print` / `socket.send` / `file.write`），会产生：

- **传输耦合**：生产者代码绑死到特定 I/O
- **测试困难**：无法用无副价的 sink 替换真实传输
- **切换成本**：换传输需要改多处生产者代码

Protocol Layer 通过引入一个**抽象基类 `ProtocolAdapter`** + **多个具体适配器**解决这些问题。生产者只依赖抽象，具体适配器在运行时注入。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **传输不可知** | base.py 零网络代码（无 UDP/ROS2/MAVLink/socket） |
| **契约优先** | ABC 定义 5 方法契约，具体适配器只实现不扩展 |
| **安全默认** | NullAdapter 作为零副作用默认 sink |
| **可观测** | ConsoleAdapter 提供开发期人类可读输出 |
| **类型严格** | 强制 `from visioncore.state.target_state import TargetState` |

### 1.3 与 B1 的关系

B1 提供了**数据**（TargetState 快照），B2 提供了**管道**（ProtocolAdapter sink）。两者正交：

```
B1 TargetState (data)  ──publish()──>  B2 ProtocolAdapter (sink)
                                            │
                                            ├── NullAdapter (discard)
                                            ├── ConsoleAdapter (log)
                                            └── [未来] UdpAdapter (network)
```

---

## 2. 文件结构

```
visioncore/protocol/
├── __init__.py            # 包导出：ProtocolAdapter, NullAdapter, ConsoleAdapter
├── base.py                # ProtocolAdapter(ABC) + 上下文管理器
├── null_adapter.py        # NullAdapter — 无操作 sink
└── console_adapter.py     # ConsoleAdapter — 日志 sink
```

与 `visioncore/eventbus/` 的多文件模式一致（subscriber.py / dispatcher.py / bus.py / debug_logger.py / events.py）。未来网络适配器将作为兄弟模块加入（如 `udp_adapter.py`）。

---

## 3. ProtocolAdapter 契约

### 3.1 类定义

```python
class ProtocolAdapter(ABC):
    @abstractmethod
    def connect(self) -> None: ...
    @abstractmethod
    def disconnect(self) -> None: ...
    @abstractmethod
    def publish(self, state: TargetState) -> None: ...
    @abstractmethod
    def health_check(self) -> bool: ...
    # 具体方法（可覆盖）
    def publish_many(self, states: Sequence[TargetState]) -> None: ...
    # 上下文管理器（具体，不可覆盖）
    def __enter__(self) -> "ProtocolAdapter": ...
    def __exit__(self, ...) -> None: ...
```

### 3.2 五个方法

| 方法 | 抽象? | 语义 |
|---|---|---|
| `connect()` | ✅ 抽象 | 建立到下游消费者的连接。幂等——已连接时再调用为 no-op。失败抛 `ConnectionError`。 |
| `disconnect()` | ✅ 抽象 | 拆除连接。幂等——已断开时再调用为 no-op。**绝不能抛异常**。 |
| `publish(state)` | ✅ 抽象 | 发布单个 TargetState。未连接抛 `RuntimeError`，非 TargetState 抛 `TypeError`。 |
| `publish_many(states)` | ❌ 具体 | 批量发布。默认实现：先急切类型检查全部元素，再循环 `publish()`。子类可覆盖以实现批量优化（如单 UDP 数据报多快照）。 |
| `health_check()` | ✅ 抽象 | 返回 `True` 当且仅当适配器已连接且可接受发布。**无副作用，绝不抛异常**——内部错误时返回 `False`。 |

### 3.3 上下文管理器

`__enter__` 调用 `connect()` 并返回 `self`；`__exit__` 调用 `disconnect()` 且**不抑制异常**（返回 `None`）。即使在 `with` 块内抛出异常，`disconnect()` 也保证被调用：

```python
with adapter:
    adapter.publish(state)
    raise ValueError("oops")
# disconnect() 已被调用，ValueError 继续传播
```

### 3.4 publish_many 的急切类型检查

默认实现先扫描整个序列做类型检查，**再**开始发布：

```python
for i, s in enumerate(states):
    if not isinstance(s, TargetState):
        raise TypeError(f"states[{i}] must be TargetState, got {type(s).__name__}")
for s in states:
    self.publish(s)
```

这保证**要么全批发布，要么全批失败**——不会出现"发布了一半然后类型错误"的半批量交付。空序列是 no-op，且不需要连接。

---

## 4. NullAdapter

### 4.1 用途

**无操作 sink**——所有 `publish` 调用被静默丢弃。应用场景：

1. **默认 sink**：当未配置真实传输时，注入 NullAdapter 避免空指针
2. **测试夹具**：在单元测试中填充 ProtocolAdapter 槽位但不产生副作用
3. **性能基准**：测量纯调用开销（无 I/O 干扰）

### 4.2 行为

| 操作 | 行为 |
|---|---|
| `connect()` | 设置 `_connected = True` |
| `disconnect()` | 设置 `_connected = False`，绝不抛 |
| `publish(state)` | 检查连接 + 类型，然后**丢弃** |
| `publish_many(states)` | 继承默认实现 |
| `health_check()` | 返回 `_connected` |

### 4.3 契约严格执行

NullAdapter 即使什么都不做，仍然严格执行完整契约：

- 未连接时 `publish` 抛 `RuntimeError`
- 非 TargetState 抛 `TypeError`（错误信息含实际类型名）

这使得 NullAdapter 可用于**测试契约执行本身**——如果代码意外在未连接时发布或传入错误类型，NullAdapter 会和真实适配器一样报错。

---

## 5. ConsoleAdapter

### 5.1 用途

**日志 sink**——每次 `publish` 将 TargetState 格式化为单行可解析记录，通过标准 `logging` 框架以 `INFO` 级别输出。应用场景：

1. **开发期调试**：在控制台实时观察快照流
2. **日志归档**：通过 logging 配置将快照流写入文件
3. **生产可观测**：配合 logging 过滤器做特定条件的快照审计

### 5.2 为什么用 logging 而非 print

| 考量 | logging | print |
|---|---|---|
| 项目约定 | ✅ DebugEventLogger 已用 logging | ❌ 项目约定不用 print |
| 可重定向 | ✅ logging 配置可路由到文件/网络/任何 handler | ❌ 硬编码 stdout |
| 可过滤 | ✅ 按 level/logger name 过滤 | ❌ 无过滤 |
| 惰性格式化 | ✅ `%`-args 仅在 level 达到时才格式化 | ❌ 总是格式化 |
| 测试友好 | ✅ 注入 capture handler 即可断言 | ❌ 需 capture stdout |

### 5.3 输出格式

每次 `publish` 输出一行：

```
INFO visioncore.protocol.console_adapter: [PUBLISH] id=42 label='person'
conf=0.920 pos=(0.5000,0.4000) vel=(0.0100,-0.0020) size=0.1200x0.3000
ts=12.5000 cam=0 metadata_keys=[]
```

字段选择原则：**最诊断性的字段在前**（id/label/conf/pos），metadata 仅显示键名（值可能很大，不 dump 到日志）。

### 5.4 自定义 logger 注入

构造器接受可选 `logger` 参数：

```python
adapter = ConsoleAdapter(logger=my_custom_logger)
```

测试中可注入带 capture handler 的 logger 断言输出，无需 capture stdout。

---

## 6. 类型注解约定（强制）

### 6.1 规则

**所有** `visioncore/protocol/` 模块必须使用：

```python
from visioncore.state.target_state import TargetState
```

**禁止**使用：

```python
from visioncore import TargetState  # ❌ 这是生命周期枚举，不是快照！
```

### 6.2 原因

`visioncore/__init__.py` 顶层导出的 `TargetState` 是 `visioncore.core.target.TargetState`——一个**生命周期枚举**（ACTIVE/LOST/LOCKED/RECOVERED/REMOVED）。而 protocol 层需要的是 `visioncore.state.target_state.TargetState`——B1 定义的**快照 dataclass**。两者同名但语义完全不同：

| 类型 | 位置 | 性质 | 用途 |
|---|---|---|---|
| `visioncore.TargetState`（顶层导出） | `core/target.py` | `Enum` | 标记生命周期阶段 |
| `visioncore.state.target_state.TargetState` | `state/target_state.py` | `@dataclass(frozen=True, slots=True)` | 瞬时观测快照 |

如果误用顶层导入，`publish(state)` 的类型检查 `isinstance(state, TargetState)` 会检查枚举类型，导致所有快照都被拒绝。

### 6.3 自动验证

`tests/test_protocol_base.py` 中的 `test_type_annotation_convention_no_top_level_import` 用 AST 解析每个 protocol 模块的源码，检查 `ImportFrom` 节点，确保没有 `from visioncore import TargetState`。docstring 中提及该模式（用于解释为何禁用）不会被误报。

### 6.4 验证结果

```
visioncore.protocol.base:
  line 53: from visioncore.state.target_state import TargetState
visioncore.protocol.null_adapter:
  line 21: from visioncore.state.target_state import TargetState
visioncore.protocol.console_adapter:
  line 28: from visioncore.state.target_state import TargetState

All imports verified: 0 forbidden, all use visioncore.state.target_state
```

---

## 7. 线程安全

### 7.1 TargetState（B1）——线程安全

TargetState 是 `frozen=True` + `slots=True` dataclass。实例创建后不可变，可安全跨线程共享无需锁。`metadata` dict 物理上可变，但按约定视为只读（`copy_with` / `from_dict` 做防御性拷贝）。

### 7.2 ProtocolAdapter——非线程安全

ProtocolAdapter **不要求**线程安全。适配器可能有可变状态（连接标志、缓冲区、socket），生产者跨线程共享同一适配器时必须自行串行化（如用 `threading.Lock`）。

这是有意的设计选择：

- **简单性**：不在基类强加锁开销
- **灵活性**：单线程场景无需锁成本
- **真实场景**：InferWorker 是单推理线程，不需要锁
- **未来扩展**：如需线程安全适配器，子类可自行加锁

### 7.3 推荐模式

跨线程发布快照的推荐模式：

```python
# 生产者侧：每个 InferWorker 持有自己的 adapter 引用
# 共享 adapter 时用 lock
class ThreadSafeAdapter(ProtocolAdapter):
    def __init__(self, inner: ProtocolAdapter):
        self._inner = inner
        self._lock = threading.Lock()
    def publish(self, state):
        with self._lock:
            self._inner.publish(state)
    # ... 其他方法委托
```

---

## 8. 未来扩展

### 8.1 网络适配器（B3+）

未来里程碑将添加网络适配器作为 `visioncore/protocol/` 的兄弟模块：

| 适配器 | 文件 | 传输 |
|---|---|---|
| `UdpAdapter` | `udp_adapter.py` | UDP 单播/多播 |
| `TcpAdapter` | `tcp_adapter.py` | TCP 长连接 |
| `FileAdapter` | `file_adapter.py` | 文件追加（录制元数据） |
| `ZmqAdapter` | `zmq_adapter.py` | ZeroMQ pub/sub |

每个适配器**独立实现传输**，base.py 不引入任何传输库。网络适配器将是 `visioncore.protocol` 包中**唯一** import 传输库（socket/zmq 等）的模块。

### 8.2 批量优化

真实网络适配器应覆盖 `publish_many` 以利用批量传输：

```python
class UdpAdapter(ProtocolAdapter):
    def publish_many(self, states):
        # 单 UDP 数据报携带多个序列化快照
        payload = b"".join(self._serialize(s) for s in states)
        self._sock.send(payload)
```

默认实现（循环 `publish`）对网络适配器是反模式——每个快照一个数据报会导致 N 倍网络开销。

### 8.3 健康监控集成

`health_check()` 设计为可与未来的 `_HealthMonitor`（类似 `ai/inference.py` 中的 tracker 健康监控）集成：

```python
# 未来 InferWorker 中的健康监控
if not self._adapter.health_check():
    logger.warning("Protocol adapter unhealthy, falling back to NullAdapter")
    self._adapter = NullAdapter()
    self._adapter.connect()
```

### 8.4 与 EventBus 的关系

EventBus 是**进程内**事件分发（订阅/发布模式，同步调用），ProtocolAdapter 是**单向 sink**（只写不读，可跨进程）。两者互补：

- EventBus：内部模块间通信（TargetManager → GUI 更新）
- ProtocolAdapter：外部系统投递（快照流 → 网络/文件/控制台）

---

## 9. 测试覆盖

### 9.1 测试统计

- **测试文件**：`tests/test_protocol_base.py`
- **测试函数数**：36（全部通过）
- **覆盖类别**：

| 类别 | 测试数 |
|---|---|
| 类型注解约定 | 2 |
| 包导出 + 导入一致性 | 2 |
| ProtocolAdapter ABC 契约 | 4 |
| 子类契约执行 | 2 |
| NullAdapter | 12 |
| ConsoleAdapter | 11 |
| 基类 __repr__ | 2 |
| 上下文管理器 | (含在 NullAdapter/ConsoleAdapter 测试中) |

### 9.2 关键测试点

| 测试 | 验证 |
|---|---|
| `test_type_annotation_convention_no_top_level_import` | AST 解析确保无 `from visioncore import TargetState` |
| `test_protocol_uses_snapshot_not_enum` | publish 接受快照 dataclass，拒绝枚举 |
| `test_protocol_adapter_is_abc` | ABC 不可直接实例化 |
| `test_abstract_methods_defined` | 4 个抽象方法正确声明 |
| `test_publish_many_is_concrete_default` | publish_many 不是抽象，有默认实现 |
| `test_subclass_missing_method_fails` | 缺方法的子类不可实例化 |
| `test_null_adapter_publish_many_rejects_non_target_state` | 急切类型检查在发布前 |
| `test_null_adapter_context_manager_disconnects_on_exception` | 异常时仍 disconnect |
| `test_console_adapter_publish_logs_state` | 日志含 target_id/label/conf |
| `test_base_repr_survives_health_check_exception` | 健康检查抛异常时 repr 不崩溃 |

### 9.3 回归测试

```
test_target_state.py           : 36/36 passed
test_core_models.py            : 31/31 passed
test_event_bus.py              : 77/77 passed
test_shadow_integration.py     : 12/12 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_protocol_base.py          : 36/36 passed  (新增)
────────────────────────────────────────────────────────────────
总计                           : 297/297 passed, 0 回归
```

---

## 10. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `visioncore/protocol/` 目录 | ✅ |
| `protocol/base.py` 定义 `ProtocolAdapter(ABC)` | ✅ |
| 5 方法：connect/disconnect/publish/publish_many/health_check | ✅ |
| 新增 NullAdapter | ✅ `null_adapter.py` |
| 新增 ConsoleAdapter | ✅ `console_adapter.py` |
| 类型注解必须用 `visioncore.state.target_state.TargetState` | ✅ AST 验证 |
| 禁止 `from visioncore import TargetState` | ✅ AST 测试守护 |
| 新增 `tests/test_protocol_base.py` | ✅ 36 项测试 |
| 输出 `docs/protocol-foundation.md` | ✅ 本文件 |

---

## 11. 文件清单

| 文件 | 行数 | 说明 |
|---|---|---|
| `visioncore/protocol/__init__.py` | ~55 | 包导出，类型注解约定说明 |
| `visioncore/protocol/base.py` | ~190 | ProtocolAdapter ABC + 上下文管理器 + 默认 publish_many |
| `visioncore/protocol/null_adapter.py` | ~105 | NullAdapter — 无操作 sink |
| `visioncore/protocol/console_adapter.py` | ~140 | ConsoleAdapter — 日志 sink |
| `tests/test_protocol_base.py` | ~470 | 36 项单元测试 + `__main__` runner |
| `docs/protocol-foundation.md` | 本文件 | 设计文档 |

---

## 12. 后续里程碑预告

- **B3**：`UdpAdapter` — 第一个网络适配器，实现 UDP 单播/多播
- **B4**：InferWorker 接入 ProtocolAdapter（影子旁路，read-only 发布快照流）
- **B5**：`FileAdapter` — 录制元数据文件 sink
- **C**：健康监控集成 + 自动回退到 NullAdapter
- **D**：跨摄像头融合 + 融合产物的全局快照流
