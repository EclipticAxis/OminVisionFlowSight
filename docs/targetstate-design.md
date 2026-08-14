# TargetState 设计文档 (Milestone B1)

> **版本**：B1 — 2026-07-19
> **范围**：新增 `visioncore/state/` 包，定义统一的"目标状态快照"数据结构
> **状态**：已实现，32 项单元测试全部通过，120 项回归零破坏

---

## 1. 目标与背景

### 1.1 为什么需要新的 TargetState

VisionCore 现有数据模型已经定义了：

- `visioncore.core.Track` — 可变容器，绑定 `track_id` + 最新 `Detection` + 速度
- `visioncore.core.Target` — 可变高层实体，聚合 `Track` + 生命周期状态 + 属性
- `visioncore.core.TargetState`（**枚举**）— 标记 *生命周期阶段*：`ACTIVE / LOST / LOCKED / RECOVERED / REMOVED`

这三个类型回答的是 **"这个目标现在处于什么阶段、绑了哪条 track"**，但它们都是**可变**的、被 Tracker / TargetManager 在原地更新。

随着 VisionCore 向跨阶段、跨线程、跨摄像头方向演进，需要一个**不可变**的、可安全共享与序列化的"**瞬时观测快照**"——记录某个目标在某一时刻的：

- 身份（是谁）
- 语义（是什么）
- 运动学（在哪里、多快）
- 尺寸（多大）
- 来源（哪个摄像头、什么时间）

这就是 `visioncore.state.TargetState` 的职责。它是一个**纯数据快照**，不携带任何行为、不引用任何传输协议、不依赖 Qt/numpy/外部框架。

### 1.2 与现有 `visioncore.core.TargetState`（枚举）的关系

两者**同名但语义完全不同**，刻意共存：

| 类型 | 位置 | 性质 | 回答的问题 |
|---|---|---|---|
| `visioncore.core.TargetState` | `core/target.py` | `Enum` | "目标处于生命周期的哪个阶段？" |
| `visioncore.state.TargetState` | `state/target_state.py` | `@dataclass(frozen=True, slots=True)` | "目标此刻的可观测状态是什么？" |

为避免命名冲突，**新的快照类型不会从 `visioncore/__init__.py` 顶层导出**。消费者必须显式导入：

```python
from visioncore.core  import TargetState as TargetLifecycle   # 枚举
from visioncore.state import TargetState                       # 快照
```

未来如果出现混淆，可考虑将快照类型重命名为 `TargetSnapshot`——但 B1 阶段保持用户指定的 `TargetState` 名称。

---

## 2. 字段含义

`TargetState` 共 14 个字段，全部为基础类型（`int / float / str / dict`），无 numpy / Qt / 框架耦合。

### 2.1 身份三元组

| 字段 | 类型 | 含义 |
|---|---|---|
| `target_id` | `int` | 当前流水线范围内的稳定目标 ID。通常由 `TargetManager` 分配，等价于单摄像头 track_id。**必填**——没有身份的快照无意义。 |
| `local_id` | `int \| None` | 摄像头/槽位内的局部 ID。多摄像头融合时用于消歧（不同摄像头的 `target_id` 可能重叠）。单摄像头系统下可与 `target_id` 相同或留 `None`。 |
| `global_id` | `int \| None` | 跨摄像头全局 ID，由全局融合 / ReID 阶段分配。**留 `None` 表示尚未建立全局身份**——这是单摄像头部署的常态。 |

**设计意图**：三元组支持从单摄像头（只用 `target_id`）平滑扩展到多摄像头融合（追加 `global_id`），无需修改字段。

### 2.2 语义与置信度

| 字段 | 类型 | 含义 |
|---|---|---|
| `label` | `str` | 语义类别标签（如 `"person"` / `"vehicle"` / `"head"`）。**权威身份是字符串本身**，消费者不应假定固定词表。 |
| `confidence` | `float` | 生产者置信度，范围 `[0.0, 1.0]`。可由检测器分数单独构成，也可融合跟踪器稳定性。**构造时不强校验**——边界裁剪由流水线负责。 |

### 2.3 运动学与尺寸（归一化坐标）

| 字段 | 类型 | 含义 |
|---|---|---|
| `cx`, `cy` | `float` | 包围盒中心，归一化到 `[0.0, 1.0]`，与显示分辨率解耦。 |
| `vx`, `vy` | `float` | 估计瞬时速度（归一化单位/秒）。无估计时为 `0.0`。 |
| `width`, `height` | `float` | 包围盒宽高，归一化 `[0.0, 1.0]`。 |

**与 `visioncore.core.BBox` 的关系**：`BBox` 将 `(x, y, w, h)` 打包成独立值对象。TargetState **不嵌套 BBox**，而是把四个分量平铺为顶层字段——目的是让序列化产物是扁平 dict，便于 JSON / 日志 / 跨语言传输。

### 2.4 时间与来源

| 字段 | 类型 | 含义 |
|---|---|---|
| `timestamp` | `float` | 快照生成时刻（秒）。**时间基准由生产者定义**（墙钟或单调时钟），消费者不应假定特定纪元。用于时序排序与延迟测量。 |
| `camera_id` | `int \| None` | 源摄像头/槽位 ID。`None` 表示快照来自非摄像头源（如跨摄像头 ReID 融合产出的"全局目标"）。 |

### 2.5 扩展元数据

| 字段 | 类型 | 含义 |
|---|---|---|
| `metadata` | `dict[str, Any]` | 生产者专有的扩展键值对。常见键：`"source"`（检测后端名）、`"reid_hash"`（ReID 嵌入哈希）、`"roi_tag"`（感兴趣区域标签）。默认空 dict。 |

**注意**：Python 没有真正的 frozen dict。`metadata` 在物理上是可变的，但**按约定构造后应视为只读**。需要修改时使用 `copy_with(metadata={...})`，不要原地改。

---

## 3. 方法

### 3.1 `to_dict() -> dict[str, Any]`

序列化为扁平 dict。使用 `dataclasses.asdict` 做递归拷贝——`metadata` 内的嵌套 dict/list 也会被深拷贝，调用者对返回 dict 的任何修改都不会泄漏回快照。

### 3.2 `from_dict(data: dict) -> TargetState` (classmethod)

`to_dict` 的逆操作。两个语义保证：

- **未知键静默忽略**——前向兼容：新版生产者加了字段，旧版消费者能安全反序列化。
- **`metadata` 防御性浅拷贝**——调用者传入的 dict 在构造后被修改不会影响快照。
- **缺失必填字段**抛 `TypeError`（来自 dataclass 构造器）。

### 3.3 `copy_with(**overrides) -> TargetState`

派生新快照。由于 TargetState 是 frozen 的，原地修改不可能；`copy_with` 是唯一支持的"演化"方式：

```python
next_state = state.copy_with(
    cx=state.cx + state.vx * dt,
    cy=state.cy + state.vy * dt,
    timestamp=state.timestamp + dt,
)
```

两个保证：

- **未知字段名立即抛 `TypeError`**——拼写错误（如 `targe_id`）不会静默失效。
- **未覆盖 `metadata` 时自动浅拷贝**——派生快照的 metadata 是独立 dict，与原快照互不影响。

---

## 4. 兼容策略

### 4.1 零破坏接入

B1 的硬约束是**不动现有业务逻辑**。具体落实：

- **不修改 `visioncore/__init__.py`**——保留原有 7 个顶层导出（`Frame / Detection / BBox / Track / Target / TargetState(枚举) / Event`）。
- **不修改 `visioncore/core/`**——`core.TargetState` 枚举与 `Target` 类保持原样。
- **不修改 `ai/ gui/ camera/ common/`**——B1 是纯新增。
- **新增独立包 `visioncore/state/`**，自包含 `__init__.py` 与 `target_state.py`。
- **新增独立测试 `tests/test_target_state.py`**——32 项测试覆盖构造 / 冻结 / 序列化 / 反序列化 / copy_with / 与枚举共存。

### 4.2 共存验证

```
visioncore.TargetState            : enum      ← 顶层导出未变
visioncore.core.TargetState       : enum      ← 原枚举保持
visioncore.state.TargetState      : dataclass ← 新快照
TopLevel is CoreEnum              : True
TopLevel is StateSnapshot         : False
StateSnapshot fields              : 14
```

152 项测试全过（32 新增 + 31 core_models + 77 event_bus + 12 shadow_integration），零回归。

### 4.3 与 legacy `Track` / `Target` 的桥接

B1 **不实现**桥接逻辑——只定义快照类型。后续里程碑可在 `visioncore/state/adapters.py` 中加入：

```python
def track_to_state(track: Track, camera_id: int, timestamp: float) -> TargetState: ...
def target_to_state(target: Target, timestamp: float) -> TargetState: ...
```

这些适配器将 lazy-import `visioncore.core`，避免循环依赖（沿用 `core/adapters.py` 的既有模式）。

---

## 5. 未来 GlobalTarget 扩展方式

`global_id` 字段是为多摄像头融合预留的钩子。扩展路径如下：

### 5.1 阶段 C — 单摄像头快照流（已完成 B1 后）

`InferWorker` 在每帧推理后，将每个 active track 转换为 `TargetState` 快照，推送到 `EventBus` 或共享缓冲区。下游消费者（GUI 显示、决策模块、录像元数据）订阅快照流，不再直接读 `Track`。

```
Track ──[per-frame adapter]──> TargetState ──> EventBus ──> consumers
```

此阶段 `global_id` 始终为 `None`。

### 5.2 阶段 D — 跨摄像头融合

新增 `visioncore/fusion/` 包，实现 `GlobalFusionStage`：

- 订阅多个摄像头的 `TargetState` 快照流
- 对时间窗内的快照做 ReID 特征匹配（复用 `ai/reid_backend.py` 的嵌入）
- 为匹配的跨摄像头目标分配 `global_id`
- 输出**新的** `TargetState` 快照，其中 `global_id` 非 `None`、`camera_id` 设为 `None`（表示融合产物）

```
[Cam0.TargetState] ─┐
[Cam1.TargetState] ─┼─> GlobalFusionStage ─> TargetState(global_id=42, camera_id=None)
[Cam2.TargetState] ─┘
```

**关键设计点**：融合产物**复用同一个 `TargetState` 类型**——不需要新 class。`global_id` 是否非 `None` + `camera_id` 是否为 `None` 这两个字段组合就是"这是融合产物"的信号。这样下游消费者无需类型分支。

### 5.3 阶段 E — GlobalTarget 聚合（可选）

如果未来需要更丰富的全局目标语义（跨摄像头的轨迹历史、出现/消失时间戳、关联的局部 track 列表），可在 `visioncore/state/global_target.py` 中新增 `GlobalTarget` 类型：

```python
@dataclass(slots=True, frozen=True)
class GlobalTarget:
    global_id: int
    member_states: tuple[TargetState, ...]   # 各摄像头的最新快照
    first_seen: float
    last_seen: float
    attributes: dict[str, Any]
```

`GlobalTarget` **聚合** `TargetState`，不替换它——快照仍然是数据流的最小单元，`GlobalTarget` 是更高层的视图。这种"快照 + 聚合"分层与现有 `Detection → Track → Target` 的演进模式一致。

### 5.4 传输层解耦

B1 明确**禁止**引入任何协议代码（无 UDP / ROS2 / MAVLink）。`TargetState` 是纯数据结构，传输由未来的适配器层负责：

```python
# 未来 visioncore/transport/udp_adapter.py（B1 之外）
def serialize_for_udp(state: TargetState) -> bytes: ...
def deserialize_from_udp(data: bytes) -> TargetState: ...
```

适配器可基于 `to_dict()` + JSON / MessagePack / Protobuf 实现，无需触碰核心类型。

---

## 6. 设计约束回顾

| 约束 | 落实方式 |
|---|---|
| `@dataclass(slots=True, frozen=True)` | ✅ `target_state.py` 第 90 行 |
| 14 个字段全部基础类型 | ✅ int/float/str/dict，无 numpy/Qt |
| `to_dict / from_dict / copy_with` | ✅ 三个方法均实现并测试 |
| 禁止协议代码 / UDP / ROS2 / MAVLink | ✅ 模块零网络依赖，仅 `dataclasses` + `typing` |
| 禁止修改 `ai/ gui/ camera/ common/` | ✅ 全部新增，零修改 |
| 禁止修改现有业务逻辑 | ✅ `visioncore/__init__.py` 未动 |
| 新增测试覆盖 5 类场景 | ✅ 32 项测试：构造(7) + 冻结(4) + 序列化(4) + 反序列化(6) + copy_with(8) + 共存(1) + repr(2) |
| 输出设计文档 | ✅ 本文件 |

---

## 7. 文件清单

| 文件 | 行数 | 说明 |
|---|---|---|
| `visioncore/state/__init__.py` | ~40 | 包导出，明确命名共存策略 |
| `visioncore/state/target_state.py` | ~280 | TargetState dataclass + 3 方法 + repr |
| `tests/test_target_state.py` | ~370 | 32 项单元测试 + `__main__` runner |
| `docs/targetstate-design.md` | 本文件 | 设计文档 |

---

## 8. 后续里程碑预告

- **B2**：`visioncore/state/adapters.py` — `Track ↔ TargetState` / `Target ↔ TargetState` 桥接
- **B3**：`InferWorker` 接入快照流（影子旁路，read-only）
- **C**：GUI 消费快照流替换直接读 Track
- **D**：`visioncore/fusion/` 跨摄像头融合
- **E**：`GlobalTarget` 聚合类型（可选）
