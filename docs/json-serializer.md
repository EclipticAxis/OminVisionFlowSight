# JSON Serializer 设计文档 (Milestone B3)

> **版本**：B3 — 2026-07-20
> **范围**：新增 `visioncore/protocol/serialization/` 子包，定义 TargetState 的 JSON 序列化器
> **状态**：已实现，31 项单元测试全部通过，328 项回归零破坏

---

## 1. 目标与背景

### 1.1 为什么需要独立的序列化器

B2 定义了 `ProtocolAdapter` 抽象——一个接收 TargetState 快照的 sink。但适配器只定义了**投递契约**（publish/disconnect/...），没有定义**线格式**（wire format）——即快照如何被编码为可传输的字节序列。

如果让每个适配器自行选择编码方式：

- **格式不一致**：ConsoleAdapter 用 `repr()`，未来 UdpAdapter 可能用 JSON，FileAdapter 可能用 CSV
- **测试困难**：序列化逻辑散布在各适配器中，无法独立测试
- **切换成本**：换格式需要改每个适配器

B3 引入独立的 `TargetStateSerializer`，将序列化与传输**解耦**：

```
TargetState (data)
    │
    ▼
TargetStateSerializer.serialize()  ──>  JSON string  ──>  ProtocolAdapter.publish_bytes()
                                                              │
                                                              ├── NullAdapter (discard)
                                                              ├── ConsoleAdapter (log)
                                                              └── [未来] UdpAdapter (send)
```

适配器只负责"把字节送到目的地"，序列化器负责"把快照变成字节"。两者可独立演进。

### 1.2 为什么选 JSON 作为首个格式

| 考量 | JSON | MessagePack | Protobuf |
|---|---|---|---|
| 标准库支持 | ✅ `json` 内置 | ❌ 需第三方 | ❌ 需第三方 + schema 编译 |
| 人类可读 | ✅ 文本格式 | ❌ 二进制 | ❌ 二进制 |
| 调试友好 | ✅ 可直接打印/日志 | ❌ 需解码 | ❌ 需解码 |
| 元数据灵活 | ✅ 任意 JSON 值 | ✅ 任意 msgpack 值 | ❌ 需 schema 定义 |
| 体积 | 中等 | 小 | 小 |
| 速度 | 中等 | 快 | 最快 |

B3 选 JSON 因为：**零依赖、人类可读、调试友好、元数据天然兼容**。未来如需更高性能或更小体积，可平行添加 `MessagePackSerializer` 而不替换 JSON。

---

## 2. 文件结构

```
visioncore/protocol/serialization/
├── __init__.py            # 包导出 TargetStateSerializer
└── json_serializer.py     # TargetStateSerializer 类
```

作为 `visioncore/protocol/` 的子包，与 `base.py` / `null_adapter.py` / `console_adapter.py` 平级。未来二进制序列化器（如 `msgpack_serializer.py`）将作为兄弟模块加入。

**未修改** `visioncore/protocol/__init__.py`——序列化器通过 `from visioncore.protocol.serialization import TargetStateSerializer` 访问，保持 B2 包导出不变。

---

## 3. 接口

### 3.1 类定义

```python
class TargetStateSerializer:
    def __init__(self, *, indent: int | None = None, sort_keys: bool = False) -> None: ...

    def serialize(self, state: TargetState) -> str: ...

    def deserialize(self, data: str | bytes) -> TargetState: ...
```

### 3.2 `serialize(state: TargetState) -> str`

将 TargetState 序列化为 JSON 字符串。

**实现**：委托给 `state.to_dict()` + `json.dumps()`。TargetState 的 14 个字段全部是 JSON 原生类型（int/float/str/dict），无需自定义 encoder。

**输出格式**（默认 compact）：

```json
{"target_id": 42, "local_id": 1, "global_id": null, "label": "person",
 "confidence": 0.92, "cx": 0.5, "cy": 0.4, "vx": 0.01, "vy": -0.002,
 "width": 0.12, "height": 0.3, "timestamp": 12.5, "camera_id": 0,
 "metadata": {"src": "yolo26"}}
```

**错误处理**：

| 输入 | 异常 | 说明 |
|---|---|---|
| 非 TargetState（None/str/int/...） | `TypeError` | 错误信息含实际类型名 |
| metadata 含非 JSON 可序列化值（set/自定义对象） | `TypeError` | 错误信息含 "non-JSON-serializable" |

### 3.3 `deserialize(data: str | bytes) -> TargetState`

将 JSON 字符串（或 UTF-8 bytes）反序列化为 TargetState。

**实现**：`json.loads()` + 类型校验 + 委托给 `TargetState.from_dict()`。接受 `bytes` 是网络代码的便利——网络缓冲区天然是字节。

**错误处理**（三层防御）：

| 层 | 输入 | 异常 | 说明 |
|---|---|---|---|
| 1. 类型 | 非 str/bytes（None/int/...） | `TypeError` | 错误信息含实际类型名 |
| 2. JSON 语法 | 畸形 JSON（`"{not valid"`） | `ValueError` | 包装 `json.JSONDecodeError`，信息含 "not valid JSON" |
| 2. JSON 语义 | 合法 JSON 但非 dict（`"[1,2,3]"` / `"42"`） | `ValueError` | 信息含 "must decode to a dict" |
| 3. TargetState 契约 | dict 缺必填字段 | `TypeError` | 来自 `TargetState.from_dict` → dataclass 构造器 |
| 3. TargetState 契约 | `metadata=None` 或非 dict | `TypeError` | 来自 B1.1 收紧的 metadata 类型校验 |

### 3.4 格式化选项

构造器接受两个可选参数（仅影响 `serialize`，不影响 `deserialize`）：

| 参数 | 默认 | 用途 |
|---|---|---|
| `indent` | `None` | 非 None 时 pretty-print（每级缩进 N 空格）。用于日志/调试/文件转储。 |
| `sort_keys` | `False` | True 时按键名字典序排序。用于测试确定性输出 + 内容寻址缓存。 |

---

## 4. 与 B1/B2 的关系

### 4.1 委托链

序列化器**不重新实现** TargetState 的 dict 转换——它委托给 B1 已验证的 `to_dict()` / `from_dict()`：

```
serialize(state):
    state.to_dict()  ──>  json.dumps()  ──>  str

deserialize(data):
    json.loads()  ──>  TargetState.from_dict()  ──>  TargetState
```

这意味着：
- B1.1 的 metadata 类型校验（拒绝 None / 非 dict）在反序列化时自动生效
- `to_dict()` 的防御性深拷贝在序列化时自动生效
- `from_dict()` 的未知键忽略（前向兼容）在反序列化时自动生效

序列化器只增加：JSON 编解码 + 一致的错误处理 + 格式化选项。

### 4.2 类型注解约定

遵循 B2 建立的强制约定：

```python
from visioncore.state.target_state import TargetState  # ✅ 正确
```

禁止：

```python
from visioncore import TargetState  # ❌ 这是生命周期枚举
```

由 AST 测试 `test_type_annotation_convention_no_top_level_import` 守护。

### 4.3 传输不可知

与 B2 一致，序列化器**零网络代码**：

| 禁止 | 验证 |
|---|---|
| `socket` | AST 检查通过 |
| `udp` | AST 检查通过 |
| `ros2` / `ros` | AST 检查通过 |
| `mavlink` / `mav` | AST 检查通过 |

序列化器只产生/消费 `str` / `bytes`，如何送到网络是适配器的事。

---

## 5. 错误处理设计

### 5.1 三层防御

反序列化是外部输入的入口，必须对任意输入安全。三层防御：

```
输入 bytes/str
    │
    ▼
[层 1: 类型] 是 str 或 bytes? ──no──> TypeError
    │ yes
    ▼
[层 2: JSON] 合法 JSON? ──no──> ValueError (语法)
    │ yes                                │
    │                                    ▼
    │                              解码为 dict? ──no──> ValueError (语义)
    │ yes
    ▼
[层 3: TargetState 契约] dict 满足 TargetState 字段要求?
    │                                    │
    │ yes                                ├── 缺必填字段 ──> TypeError
    │                                    ├── metadata=None ──> TypeError (B1.1)
    │                                    └── metadata 非 dict ──> TypeError (B1.1)
    ▼
返回 TargetState
```

### 5.2 异常类型选择

| 异常 | 语义 | 调用方处理方式 |
|---|---|---|
| `TypeError` | 输入类型错误（编程错误） | 不应捕获——修复调用代码 |
| `ValueError` | 输入值错误（数据错误） | 可捕获——记录并跳过坏数据 |

**关键区分**：畸形 JSON 是数据错误（`ValueError`），因为输入来自不可信源（网络/文件）；缺字段是数据错误但表现为 `TypeError`（因为来自 dataclass 构造器）——这是 Python dataclass 的固有行为，序列化器不额外包装。

### 5.3 前向兼容

`from_dict` 静默忽略未知键。这意味着：

```json
{"target_id": 1, ..., "future_field": "unknown"}
```

能正常反序列化，`future_field` 被丢弃。这允许：

- 新版生产者添加字段，旧版消费者仍能解析
- 格式演进无需版本协商（代价是未知字段被静默丢弃）

未来如需严格模式（拒绝未知键），可添加 `strict=True` 构造参数。

---

## 6. 测试覆盖

### 6.1 测试统计

- **测试文件**：`tests/test_json_serializer.py`
- **测试函数数**：31（全部通过）

### 6.2 B3 规范要求的 3 类测试

| 类别 | 测试数 | 覆盖点 |
|---|---|---|
| **roundtrip** | 9 | 基本/metadata/嵌套metadata/None可选/空metadata/极端浮点/零负ID/产出合法JSON/未知键忽略 |
| **invalid json** | 4 | 畸形语法/非dict值/bytes输入/错误输入类型 |
| **missing field** | 4 | 缺必填字段/缺metadata(允许)/缺可选字段键(拒绝)/metadata=None/非dict metadata |

### 6.3 额外覆盖

| 类别 | 测试数 | 覆盖点 |
|---|---|---|
| 类型注解约定 | 2 | AST 检查无 forbidden import / serialize 拒绝枚举 |
| 包导出 + 导入一致性 | 2 | __all__ 包含 / 直接导入一致 |
| serialize 类型强制 | 2 | 非 TargetState 拒绝 / 非JSON可序列化metadata拒绝 |
| 格式化选项 | 4 | indent pretty-print / compact / sort_keys 确定性 / repr |
| 幂等与独立 | 3 | 不修改输入 / 返回独立字符串 / 可复用 |

### 6.4 回归测试

```
test_target_state.py           : 36/36 passed
test_core_models.py            : 31/31 passed
test_event_bus.py              : 77/77 passed
test_shadow_integration.py     : 12/12 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_protocol_base.py          : 36/36 passed
test_json_serializer.py        : 31/31 passed  (新增)
────────────────────────────────────────────────────────────────
总计                           : 328/328 passed, 0 回归
```

---

## 7. 未来扩展

### 7.1 二进制序列化器

如需更高性能或更小体积，可平行添加 `MessagePackSerializer`：

```python
# visioncore/protocol/serialization/msgpack_serializer.py (未来)
class MessagePackSerializer:
    def serialize(self, state: TargetState) -> bytes: ...
    def deserialize(self, data: bytes) -> TargetState: ...
```

接口与 JSON 序列化器一致（`serialize` / `deserialize`），只是输出 `bytes` 而非 `str`。调用方可根据传输需求选择。

### 7.2 批量序列化

如需高效传输多快照，可添加 `serialize_many` / `deserialize_many`：

```python
def serialize_many(self, states: Sequence[TargetState]) -> str:
    return json.dumps([s.to_dict() for s in states])
```

这对网络适配器的批量优化（单 UDP 数据报多快照）有用。B3 暂不实现——YAGNI。

### 7.3 格式版本化

当前输出无版本包装。未来如需严格版本控制：

```json
{"version": 1, "state": {"target_id": 42, ...}}
```

`deserialize` 可检测 `version` 键并路由到对应解析器。`from_dict` 的未知键忽略特性使这一演进无需破坏旧消费者。

### 7.4 适配器集成

未来 `UdpAdapter` 将使用序列化器内部：

```python
class UdpAdapter(ProtocolAdapter):
    def __init__(self, serializer: TargetStateSerializer):
        self._serializer = serializer
        ...
    def publish(self, state: TargetState):
        payload = self._serializer.serialize(state).encode("utf-8")
        self._sock.send(payload)
```

适配器与序列化器解耦——同一适配器可配 JSON 或 MessagePack，只需换序列化器实例。

---

## 8. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `protocol/serialization/` 目录 | ✅ |
| `json_serializer.py` 实现 `TargetStateSerializer` | ✅ |
| 接口 `serialize()` / `deserialize()` | ✅ |
| 禁止 socket | ✅ AST 验证 |
| 禁止 udp | ✅ AST 验证 |
| 禁止 ros2 | ✅ AST 验证 |
| 禁止 mavlink | ✅ AST 验证 |
| 测试：roundtrip | ✅ 9 项 |
| 测试：invalid json | ✅ 4 项 |
| 测试：missing field | ✅ 4 项 |
| 输出 `docs/json-serializer.md` | ✅ 本文件 |

---

## 9. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/protocol/serialization/__init__.py` | 包导出，类型注解约定说明 |
| `visioncore/protocol/serialization/json_serializer.py` | TargetStateSerializer 类 |
| `tests/test_json_serializer.py` | 31 项单元测试 + `__main__` runner |
| `docs/json-serializer.md` | 本文件 |

---

## 10. 后续里程碑预告

- **B4**：`UdpAdapter` — 第一个网络适配器，内部使用 TargetStateSerializer
- **B5**：InferWorker 接入 ProtocolAdapter（影子旁路发布快照流）
- **B6**：`FileAdapter` — 录制元数据文件 sink
- **C**：`MessagePackSerializer` — 二进制高性能序列化（可选）
- **D**：健康监控集成 + 自动回退 NullAdapter
