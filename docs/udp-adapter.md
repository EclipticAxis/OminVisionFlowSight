# UDP Adapter 设计文档 (Milestone B5)

> **版本**：B5 — 2026-07-20
> **范围**：新增 `visioncore/protocol/adapters/udp_adapter.py`，实现 UDPAdapter
> **状态**：已实现，23 项单元测试全部通过，382 项回归零破坏

---

## 1. 目标与背景

### 1.1 为什么需要 UDPAdapter

B2 定义了 `ProtocolAdapter` 抽象（connect/disconnect/publish/...），B3/B4 提供了 JSON/CBOR 序列化器。但至今没有一个**真实传输适配器**——只有 NullAdapter（丢弃）和 ConsoleAdapter（日志）。

B5 填补这一空白：`UDPAdapter` 是第一个真正通过网络发送 TargetState 快照的适配器。它将快照序列化为字节，通过 UDP 协议发送到指定目标。

### 1.2 为什么选 UDP

| 传输 | 适合 VisionCore? | 理由 |
|---|---|---|
| **UDP** | ✅ | 无连接、低延迟、支持广播/多播、适合实时快照流（偶尔丢包可接受） |
| TCP | ⚠️ 未来 | 可靠传输但握手开销大，适合需要保证交付的场景（录像元数据） |
| 文件 | ⚠️ 未来 | 离线存储，适合长时间录制 |
| 共享内存 | ⚠️ 未来 | 进程间通信，零拷贝但限于同机 |

VisionCore 的快照流是**实时观测数据**——偶尔丢一个快照不影响系统运行，但延迟积累会影响决策。UDP 的无连接特性正好满足这一需求。

### 1.3 设计原则

| 原则 | 落实 |
|---|---|
| **序列化器注入** | 构造器接受 `serializer` 参数，调用 `serializer.serialize()` 发送 |
| **禁止硬编码格式** | 模块不 import `json`，不调用 `json.dumps/loads`，格式由注入的序列化器决定 |
| **str/bytes 兼容** | 序列化器返回 `str`（JSON）时自动 UTF-8 编码；返回 `bytes`（CBOR）时直接发送 |
| **传输不可知** | 序列化器不知道目标传输；传输适配器不知道序列化格式 |

---

## 2. 文件结构

```
visioncore/protocol/
├── __init__.py                    # B2: ProtocolAdapter, NullAdapter, ConsoleAdapter
├── base.py                        # B2: ProtocolAdapter ABC
├── null_adapter.py                # B2: NullAdapter
├── console_adapter.py             # B2: ConsoleAdapter
├── serialization/                 # B3/B4: 序列化子包
│   ├── __init__.py
│   ├── json_serializer.py         # B3: JSON
│   └── cbor_serializer.py         # B4: CBOR
└── adapters/                      # B5: 传输适配器子包（新增）
    ├── __init__.py
    └── udp_adapter.py             # B5: UDPAdapter
```

`adapters/` 子包与 `serialization/` 子包平行——序列化负责"快照→字节"，适配器负责"字节→传输"。未来 `tcp_adapter.py` / `file_adapter.py` 将作为兄弟模块加入。

**未修改** `visioncore/protocol/__init__.py`——UDPAdapter 通过 `from visioncore.protocol.adapters import UDPAdapter` 访问。

---

## 3. 构造与接口

### 3.1 构造器

```python
UDPAdapter(
    serializer,   # 任何有 serialize(state)->str|bytes 方法的对象
    host,         # 目标主机/IP（单播或多播组地址）
    port,         # 目标 UDP 端口
)
```

`serializer` 参数接受任何符合 `_Serializer` 结构化协议（`typing.Protocol`）的对象——只要有 `serialize(state: TargetState) -> str | bytes` 方法即可。目前包括：

- `TargetStateSerializer`（JSON，返回 `str`）
- `CBORSerializer`（CBOR，返回 `bytes`）

### 3.2 五个方法

| 方法 | 实现 | 行为 |
|---|---|---|
| `connect()` | 重写 | 创建 `SOCK_DGRAM` socket，调用 `socket.connect((host, port))` 设置默认目标。UDP 的 connect 不做握手，只设置默认地址。幂等。 |
| `disconnect()` | 重写 | 关闭 socket，置 `None`。幂等，绝不抛。 |
| `publish(state)` | 重写 | 调用 `serializer.serialize(state)` 获取载荷；若为 `str` 则 UTF-8 编码；`socket.send(payload)` 发送单个 UDP 数据报。 |
| `publish_many(states)` | 继承 | 默认实现：急切类型检查 + 循环 `publish()`。每个快照作为独立数据报发送。 |
| `health_check()` | 重写 | 返回 `self._sock is not None`。 |

### 3.3 序列化器注入（核心设计）

**关键约束**：适配器**不硬编码 JSON**。模块源码：

- ❌ 不 `import json`
- ❌ 不调用 `json.dumps()` / `json.loads()`
- ❌ 不引用 `TargetStateSerializer` 或 `CBORSerializer`
- ✅ 只调用 `self._serializer.serialize(state)`

测试 `test_no_hardcoded_json` 用 AST 解析验证以上约束。

**str/bytes 兼容处理**：

```python
payload = self._serializer.serialize(state)
if isinstance(payload, str):
    payload = payload.encode("utf-8")   # JSON → UTF-8 bytes
# CBOR 已经是 bytes，直接发送
self._sock.send(payload)
```

这使得同一适配器实例可以用 JSON（调试时人类可读）或 CBOR（生产时紧凑高效），只需在构造时注入不同序列化器。

### 3.4 上下文管理器

继承自 `ProtocolAdapter` 基类：

```python
with UDPAdapter(CBORSerializer(), "127.0.0.1", 9999) as adapter:
    adapter.publish(state)
# disconnect() 自动调用，即使发生异常
```

---

## 4. MTU 与分片

### 4.1 UDP 数据报大小限制

| 层 | 最大载荷 |
|---|---|
| IPv4 理论上限 | 65,507 bytes（65535 - 20 IP头 - 8 UDP头） |
| 典型以太网 MTU | 1,500 bytes → UDP 载荷 1,472 bytes |
| Windows loopback MTU | 65,535 bytes（无分片限制） |

### 4.2 适配器策略

UDPAdapter **不强制**载荷大小限制——依赖 OS 处理分片：

- 载荷 ≤ 1,472 bytes：单包传输，无分片，最适合生产网络
- 1,472 < 载荷 ≤ 65,507：OS 自动分片，在支持分片的网络上工作；某些防火墙/路由器可能丢弃分片
- 载荷 > 65,507：`socket.send()` 抛 `OSError`

### 4.3 推荐

| 场景 | 推荐序列化器 | 典型载荷大小 | 分片风险 |
|---|---|---|---|
| 单目标快照（minimal/rich） | JSON 或 CBOR | 200-450 bytes | 无（< 1472） |
| 多目标批量 | CBOR | 取决于目标数 | 可能分片 |
| 大 metadata（100+ keys） | CBOR | ~6.8KB | loopback 无风险；生产网络需分片或拆包 |

**生产建议**：保持单数据报载荷 < 1,472 bytes。如需传输更大数据，考虑：
1. 使用 CBOR（比 JSON 小 11-32%）
2. 将大 metadata 拆分为多个快照
3. 切换到 TCP 适配器（未来 B6+）

---

## 5. 多播支持

UDPAdapter 透明支持多播——将 `host` 设为多播组地址即可：

```python
# 发送到多播组 239.1.1.1
adapter = UDPAdapter(CBORSerializer(), "239.1.1.1", 5000)
adapter.connect()
adapter.publish(state)  # 所有订阅了 239.1.1.1:5000 的接收者都会收到
```

发送侧无需特殊配置——OS 自动将多播地址路由到正确的网络接口。接收侧需要 `socket.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, ...)` 加入多播组。

---

## 6. 错误处理

| 操作 | 异常 | 处理 |
|---|---|---|
| `connect()` 失败 | `OSError` | 传播给调用方（地址无效/端口被占/权限不足） |
| `publish()` 未连接 | `RuntimeError` | 提示 "Call connect() first" |
| `publish()` 非 TargetState | `TypeError` | 来自类型检查（先于序列化器调用） |
| `publish()` 序列化失败 | `TypeError` | 来自序列化器（metadata 含不可序列化值） |
| `publish()` 发送失败 | `OSError` | 传播（目标不可达/网络错误/载荷过大） |
| `disconnect()` 关闭失败 | 吞掉 + 日志 | 契约要求 disconnect 绝不抛异常 |

---

## 7. 测试覆盖

### 7.1 B5 规范要求的 3 类测试

| 类别 | 测试 | 验证 |
|---|---|---|
| **loopback** | `test_loopback_cbor` / `test_loopback_json` | 发送→接收→反序列化→比较相等。验证 CBOR 和 JSON 两种序列化器。 |
| **multiple states** | `test_multiple_states_cbor` / `test_multiple_states_json` | 连续发送 3-5 个快照，逐个接收，验证顺序保持。 |
| **large payload** | `test_large_payload_cbor` / `test_large_payload_json` | 100-key metadata（CBOR ~6.8KB / JSON ~9.9KB），超过 MTU 1472，loopback 验证。 |

### 7.2 额外覆盖

| 类别 | 测试数 | 覆盖点 |
|---|---|---|
| 类型注解约定 + 禁止 JSON | 2 | AST 检查无 forbidden import / 无 json import + 无 json.dumps/loads 调用 |
| 包导出 + 继承 | 3 | __all__ / 直接导入一致 / issubclass ProtocolAdapter |
| 构造 + repr | 2 | 参数存储 / repr 含序列化器类型+host+port+连接状态 |
| 生命周期契约 | 4 | 初始未连接 / connect+disconnect 幂等 / 未连接 publish 抛错 / 上下文管理器 |
| 上下文管理器异常 | 1 | 异常时仍 disconnect |
| 类型强制 | 1 | 非 TargetState 拒绝 |
| 重连 | 1 | disconnect 后 reconnect 创建新 socket |
| publish_many 继承 | 2 | 循环 publish + 空序列列 no-op |
| 序列化器注入 | 1 | 同一适配器代码路径支持 JSON 和 CBOR |

### 7.3 测试方法

所有 loopback 测试使用**真实 UDP socket** 在 `127.0.0.1` 上：
1. 用 `socket.bind(("127.0.0.1", 0))` 绑定到临时端口
2. 创建 UDPAdapter 指向该端口
3. 发布快照
4. 用 `recvfrom()` 接收
5. 反序列化并比较

这比 mock 更可靠——验证了真实的 socket 行为（编码、发送、接收、缓冲）。

### 7.4 回归测试

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
test_udp_adapter.py            : 23/23 passed  (新增)
────────────────────────────────────────────────────────────────
总计                           : 382/382 passed, 0 回归
```

---

## 8. 使用示例

### 8.1 调试：JSON + 控制台 + UDP 同时

```python
from visioncore.protocol import ConsoleAdapter, NullAdapter
from visioncore.protocol.adapters import UDPAdapter
from visioncore.protocol.serialization import TargetStateSerializer, CBORSerializer

# 调试模式：JSON 发送到 UDP（可用 tcpdump/wireshark 查看）
debug_adapter = UDPAdapter(TargetStateSerializer(), "127.0.0.1", 9999)
with debug_adapter:
    debug_adapter.publish(state)
```

### 8.2 生产：CBOR + UDP 多播

```python
# 生产模式：CBOR 紧凑传输到多播组
prod_adapter = UDPAdapter(CBORSerializer(), "239.1.1.1", 5000)
with prod_adapter:
    for state in state_stream:
        prod_adapter.publish(state)
```

### 8.3 与 InferWorker 集成（未来 B6）

```python
# InferWorker 注入 UDPAdapter 影子旁路
class InferWorker(QThread):
    def __init__(self):
        self._protocol = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
        self._protocol.connect()

    def _on_track_updated(self, track):
        state = TargetState(
            target_id=track.track_id, ...
        )
        self._protocol.publish(state)  # 影子旁路，不影响主推理
```

---

## 9. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `protocol/adapters/udp_adapter.py` | ✅ |
| `UDPAdapter` 继承 `ProtocolAdapter` | ✅ `issubclass(UDPAdapter, ProtocolAdapter)` |
| 构造 `UDPAdapter(serializer, host, port)` | ✅ |
| 禁止内部写死 JSON | ✅ AST 测试验证：不 import json，不调用 json.dumps/loads |
| 通过 `serializer.serialize()` 发送 | ✅ publish 调用 `self._serializer.serialize(state)` |
| 测试：loopback | ✅ 2 项（CBOR + JSON） |
| 测试：multiple states | ✅ 2 项（CBOR + JSON） |
| 测试：large payload | ✅ 2 项（CBOR + JSON） |
| 输出 `docs/udp-adapter.md` | ✅ 本文件 |

---

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/protocol/adapters/__init__.py` | 子包导出 UDPAdapter |
| `visioncore/protocol/adapters/udp_adapter.py` | UDPAdapter 类 + `_Serializer` Protocol |
| `tests/test_udp_adapter.py` | 23 项单元测试 + `__main__` runner |
| `docs/udp-adapter.md` | 本文件 |

---

## 11. 后续里程碑

- **B6**：InferWorker 接入 ProtocolAdapter（影子旁路发布快照流）
- **B7**：`FileAdapter` — 录制元数据文件 sink
- **B8**：`TcpAdapter` — 可靠传输（可选）
- **C**：健康监控集成 + 自动回退 NullAdapter
- **D**：接收端 `UDPReceiver` — 接收并反序列化快照流
