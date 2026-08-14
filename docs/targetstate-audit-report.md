# Milestone B1 审计报告 — `visioncore/state/`

> **审计日期**：2026-07-19
> **审计对象**：`visioncore/state/__init__.py` + `visioncore/state/target_state.py` + `tests/test_target_state.py`
> **审计方法**：源码静态审查 + runtime 验证 + 边界用例实测（非仅视觉检查）
> **审计员**：WorkBuddy（自动化）
> **总体评分**：**PASS WITH WARNING**

---

## 0. 评分汇总

| # | 审计项 | 评分 | 关键证据 |
|---|---|---|---|
| 1 | TargetState 是否 immutable | **PASS** | `frozen=True` + `slots=True` + 14 字段全部拒赋值 + 无 `__dict__` |
| 2 | 是否存在业务逻辑 | **PASS** | 仅数据转换 + 防御性拷贝 + 输入校验，零状态机/I/O/副作用 |
| 3 | 是否引入协议耦合 | **PASS** | 零网络/传输 import；协议关键字全部出现在 docstring 中明确声明"无协议" |
| 4 | 是否依赖 `ai/` | **PASS** | `target_state.py` 独立加载仅依赖 stdlib（独立 load 测试验证） |
| 5 | 是否依赖 `gui/` | **PASS** | 同上 |
| 6 | 是否依赖 `target_manager` | **PASS** | 同上；docstring 中的 `TargetManager` 引用仅为说明文字 |
| 7 | `to_dict` / `from_dict` 是否对称 | **PASS WITH WARNING** | 12 个边界用例 10 项通过；2 项发现 metadata 类型未校验 |
| 8 | `copy_with` 是否正确 | **PASS** | 13 个边界用例全部通过（隔离/拒绝/链式/类型保持） |
| 9 | 测试覆盖率 | **PASS WITH WARNING** | 32 项测试覆盖 7 大类；发现 9 处次要 gap，无核心路径缺失 |

**总体评分**：**PASS WITH WARNING**

理由：核心契约（不可变、无协议耦合、依赖隔离、对称性、copy_with 正确性）全部满足。3 项 WARNING 均为次要问题（metadata 类型校验缺失、测试边界覆盖有 gap、父包预先存在的 numpy eager-load），不影响 B1 的功能正确性，但建议在 B2 之前修复。

---

## 1. 审计项 1 — TargetState 是否 immutable

**评分**：**PASS**

### 验证方法
Runtime 实测，非视觉检查。

### 证据

```
=== AUDIT POINT 1: Immutability ===
  frozen=True  : True
  slots=True   : True
  __slots__    : ('target_id', 'local_id', 'global_id', 'label', 'confidence',
                  'cx', 'cy', 'vx', 'vy', 'width', 'height', 'timestamp',
                  'camera_id', 'metadata')
  All 14 fields reject assignment: True
  Instance has __dict__: False  (must be False)
  Public methods (excluding to_dict/from_dict/copy_with): []  (must be [])
  delattr rejected: True
  default metadata distinct per instance: True
```

### 检查清单

| 检查项 | 期望 | 实测 |
|---|---|---|
| `@dataclass(slots=True, frozen=True)` 装饰器 | 存在 | ✅ `target_state.py:75` |
| `__dataclass_params__.frozen` | `True` | ✅ |
| `__slots__` 包含全部 14 字段 | 是 | ✅ |
| 实例无 `__dict__` | 是 | ✅ |
| 全部 14 字段 `setattr` 抛 `FrozenInstanceError` | 是 | ✅ |
| `delattr` 抛 `FrozenInstanceError` | 是 | ✅ |
| 无公共 setter 方法 | 是 | ✅ 仅有 `to_dict / from_dict / copy_with` |
| `default_factory=dict` 每实例独立 | 是 | ✅ |

### 注意点（非缺陷）
`metadata: dict[str, Any]` 字段在物理上是可变的（Python 无 frozen dict）。这是标准 Python 模式，与既有 `visioncore.core.Event.payload` 一致。docstring 已明确"按约定构造后视为只读"，且 `copy_with` / `from_dict` 均做防御性拷贝。**不构成 immutability 违反**。

---

## 2. 审计项 2 — 是否存在业务逻辑

**评分**：**PASS**

### 验证方法
源码静态扫描全部控制流语句 + 功能语义审查。

### 证据

`target_state.py` 中的全部控制流语句（13 处）：

| 行 | 语句 | 语义类别 |
|---|---|---|
| 169 | `return dataclasses.asdict(self)` | 数据转换（to_dict） |
| 208-209 | `for key, value in data.items(): if key in field_names:` | 数据过滤（from_dict 丢弃未知键） |
| 215 | `if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):` | 防御性拷贝守卫 |
| 221 | `return cls(**kwargs)` | 数据构造 |
| 262-263 | `if bad_keys: raise TypeError(...)` | 输入校验（copy_with 拒绝未知字段） |
| 279 | `if "metadata" not in overrides:` | 防御性拷贝守卫 |
| 282 | `return type(self)(**new_values)` | 数据构造 |
| 294 | `if self.metadata:` | repr 显示逻辑 |
| 301 | `return (...)` | repr 构造 |

### 检查清单

| 业务逻辑类别 | 是否存在 |
|---|---|
| 状态机 / 状态转换 | ❌ 无 |
| I/O（文件/网络/数据库） | ❌ 无 |
| 副作用（日志/全局变量/线程交互） | ❌ 无 |
| 领域规则（运动学计算/置信度裁剪/ID 生成） | ❌ 无 |
| 外部服务调用 | ❌ 无 |
| 定时 / 调度 | ❌ 无 |
| 事件发布 | ❌ 无 |
| 业务校验（除"是否已知字段名"外的输入校验） | ❌ 无 |

模块仅包含 3 类**值对象基础设施**：
1. 数据转换（dict ↔ dataclass）
2. 防御性拷贝（metadata 隔离）
3. 输入校验（拒绝未知字段名 —— 编程错误防护，非业务规则）

这 3 类都不是业务逻辑。✅

---

## 3. 审计项 3 — 是否引入协议耦合

**评分**：**PASS**

### 验证方法
- Grep 全部网络/传输/序列化关键字
- runtime `sys.modules` 检查
- 独立加载测试（绕过父包 init）

### 证据 1 — Grep 关键字

```
\b(udp|ros2?|mavlink|socket|asyncio|threading|multiprocessing|protocol|
   serialize|pickle|json|network|transport|zmq|websocket|http|grpc|
   protobuf|messagepack)\b
```

全部 6 处匹配均在 **docstring** 中：

| 行 | 内容 | 性质 |
|---|---|---|
| 31 | `* **No protocol code**: this module contains *no* transport, *no* ROS2,` | docstring 明确声明无协议 |
| 32 | `*no* MAVLink, *no* UDP. It is a pure data structure.` | 同上 |
| 33 | `transport is the responsibility of a future adapter layer.` | 同上 |
| 83 | `modules, transport adapters) consume.` | docstring 描述下游消费者 |
| 156 | `A new dict with one entry per field, suitable for JSON` | docstring 说明 to_dict 输出适合 JSON 序列化 |
| 158 | `JSON-serialisable -- the core schema does not enforce this).` | 同上 |

**零实际协议代码 import 或调用。**

### 证据 2 — 独立加载测试

绕过 `visioncore/__init__.py`，直接用 `importlib.util.spec_from_file_location` 加载 `target_state.py`，检查 transitively 加载的模块：

```
=== Modules loaded by target_state.py ALONE (parent init bypassed) ===
  ast, collections.abc, copy, copyreg, dataclasses, dis, enum,
  importlib.machinery, inspect, linecache, opcode, re, re._casefix,
  re._compiler, re._constants, re._parser, token, tokenize, typing,
  typing.io, typing.re, visioncore.state.target_state, weakref

=== Forbidden deps actually loaded by target_state.py ===
  NONE -- target_state.py is fully self-contained (stdlib only).
```

✅ 仅 stdlib（`dataclasses / typing / re / copy` 等），零协议库。

---

## 4. 审计项 4 — 是否依赖 `ai/`

**评分**：**PASS**

### 验证方法
- 静态 grep `import ai` / `from ai`
- runtime `sys.modules` 检查
- 独立加载测试

### 证据

```
=== Forbidden deps actually loaded by target_state.py ===
  NONE -- target_state.py is fully self-contained (stdlib only).
```

`target_state.py` 的全部 import 语句（4 行）：

```python
from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Any
```

零 `ai` 依赖。✅

---

## 5. 审计项 5 — 是否依赖 `gui/`

**评分**：**PASS**

### 验证方法
同审计项 4。

### 证据
同上 —— 独立加载测试中 `gui` / `PyQt` / `cv2` 均未出现在 sys.modules。零 `gui` 依赖。✅

---

## 6. 审计项 6 — 是否依赖 `target_manager`

**评分**：**PASS**

### 验证方法
- 静态 grep `target_manager`
- runtime `sys.modules` 检查
- 独立加载测试

### 证据 1 — Grep 命中

```
line 88: token='target_manager': :class:`visioncore.target_manager.TargetManager`. Must be
```

**此命中在 `target_id` 字段的 docstring 中**（仅说明 target_id 通常由 TargetManager 分配），不是 import 语句。

### 证据 2 — 独立加载测试

```
=== Forbidden deps actually loaded by target_state.py ===
  NONE
```

`visioncore.target_manager` 未出现在 sys.modules。零依赖。✅

---

## 7. 审计项 7 — `to_dict` / `from_dict` 是否对称

**评分**：**PASS WITH WARNING**

### 验证方法
12 个边界用例实测。

### 证据

| # | 用例 | 结果 |
|---|---|---|
| 1 | 全 None 可选字段 + 空 metadata 往返 | ✅ 相等 |
| 2 | 嵌套 metadata（dict in dict, list）往返 | ✅ 相等 |
| 3 | 极端浮点数（0.999999, -123.456, 1e10, 0.0）往返 | ✅ 相等 |
| 4 | 零/负 ID（target_id=0, local_id=-1, camera_id=-99）往返 | ✅ 相等 |
| 5 | from_dict 容忍未知键（前向兼容） | ✅ 相等 |
| 6 | to_dict metadata 深拷贝隔离（嵌套 mutation 不泄漏） | ✅ 原快照不变 |
| 7 | from_dict metadata 浅拷贝隔离（caller dict mutation 不泄漏） | ✅ 快照不变 |
| 8 | from_dict 缺失必填字段抛 TypeError | ✅ 抛出 |
| 9 | **from_dict 接受 `metadata=None`** | ⚠️ 接受，metadata 变为 None |
| 10 | **from_dict 接受 `metadata='not a dict'`** | ⚠️ 接受，metadata 变为字符串 |
| 11 | 类型身份保持（isinstance + 同类） | ✅ |
| 12 | to_dict 每次返回新 dict（不缓存） | ✅ |

### WARNING 详情

**Case 9 & 10**：`from_dict` 的 metadata 防御性拷贝守卫为：

```python
if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
    kwargs["metadata"] = dict(kwargs["metadata"])
```

当传入 `metadata=None` 或 `metadata="string"` 时，`isinstance(..., dict)` 为 `False`，跳过拷贝，原值直接传给构造器。由于 dataclass 构造器不做类型校验，最终 `ts.metadata` 是 `None` 或字符串——**违反 `dict[str, Any]` 类型注解**。

**影响**：
- 不影响正常使用路径（生产者总是传 dict）
- 但若消费者依赖 `ts.metadata.keys()` 等方法，非 dict 值会抛 `AttributeError`
- 与 `visioncore.core.Event.payload` 行为一致（也不校验），但仍是契约 gap

**建议修复**（B2 之前）：

```python
if "metadata" in kwargs:
    if kwargs["metadata"] is None:
        kwargs["metadata"] = {}  # 或抛 TypeError
    elif isinstance(kwargs["metadata"], dict):
        kwargs["metadata"] = dict(kwargs["metadata"])
    else:
        raise TypeError(f"metadata must be dict, got {type(kwargs['metadata']).__name__}")
```

---

## 8. 审计项 8 — `copy_with` 是否正确

**评分**：**PASS**

### 验证方法
13 个边界用例实测。

### 证据

| # | 用例 | 结果 |
|---|---|---|
| 1 | 单字段覆盖 + 其他字段保持 | ✅ |
| 2 | 多字段覆盖 | ✅ |
| 3 | 原 snapshot 不被修改（无 mutation leak） | ✅ |
| 4 | 未知字段名抛 TypeError + 错误信息含字段名 | ✅ |
| 5 | metadata 未覆盖时浅拷贝（独立 dict） | ✅ |
| 6 | 派生 snapshot 的 metadata mutation 不影响原 | ✅ |
| 7 | metadata 显式覆盖时完全替换 | ✅ |
| 8 | metadata 覆盖时存引用（caller 知情） | ✅ 设计如此 |
| 9 | 无覆盖时返回相等但独立的 twin + metadata 独立 | ✅ |
| 10 | 链式调用（3+ hops） | ✅ |
| 11 | None 覆盖可选字段 | ✅ |
| 12 | 返回类型保持（`type(self)` 子类安全） | ✅ |
| 13 | 全部 14 字段名都被接受（无假拒绝） | ✅ |

### 设计观察（非缺陷）

Case 8：当 caller 显式传 `metadata=some_dict` 给 `copy_with` 时，`some_dict` 引用被直接存储（不拷贝）。这是**设计意图**——caller 主动传入自己的 dict，若需隔离可显式 `.copy()`。docstring 已说明"shallow-copied so the new snapshot's metadata is independent of the original's"——指的是**原 snapshot 的 metadata**被拷贝，而 caller 新传入的 dict 不被拷贝。行为一致且可预测。✅

---

## 9. 审计项 9 — 测试覆盖率

**评分**：**PASS WITH WARNING**

### 测试统计

- **测试文件**：`tests/test_target_state.py`
- **测试函数数**：32（全部通过，0 失败）
- **运行方式**：`PYTHONPATH=F:/VisionBata python tests/test_target_state.py`（项目无 pytest）
- **行覆盖率工具**：❌ venv 未安装 `coverage.py`，无法做自动行覆盖率测量（这是项目级缺失，非 B1 特有）

### 覆盖率分类

| 类别 | 测试数 |
|---|---|
| Construction（构造） | 7 |
| Frozen（不可变性） | 4 |
| to_dict（序列化） | 4 |
| from_dict（反序列化） | 6 |
| copy_with（派生） | 8 |
| Coexistence（与枚举共存） | 1 |
| repr（表示） | 2 |

### 测试类型分布

| 类型 | 数量 |
|---|---|
| Positive（happy path） | 19 |
| Negative（拒绝路径） | 5 |
| Edge case（边界） | 6 |
| Symmetry（往返对称） | 1 |
| Architecture（架构共存） | 1 |

### 覆盖 Gap 清单

| Gap | 严重度 | 说明 |
|---|---|---|
| from_dict 接受 `metadata=None` | WARNING | 审计项 7 发现，无测试覆盖 |
| from_dict 接受 `metadata=str` | WARNING | 同上 |
| to_dict 的 metadata 值 JSON 可序列化性 | INFO | docstring 提及但无测试 |
| confidence 范围 `[0,1]` 不校验 | INFO | docstring 明示"not validated"，无测试文档化此行为 |
| copy_with 链式调用（3+ hops） | INFO | 审计项 8 验证通过，无测试 |
| copy_with None 覆盖可选字段 | INFO | 同上 |
| copy_with 全 14 字段被接受 | INFO | 同上 |
| 线程安全 / 并发访问 | INFO | frozen 设计上安全，无显式测试 |
| 相等性语义（`==` 两实例） | INFO | 依赖 dataclass `__eq__`，无显式测试 |

### 评价

**核心路径覆盖完整**：B1 规范要求的 5 类场景（构造 / 冻结 / 序列化 / 反序列化 / copy_with）均有覆盖，且每类有 4-8 个用例（含正/负/边界）。

**次要 gap 集中在边界**：9 个 gap 中 7 个是 INFO 级（设计行为文档化缺失测试），2 个是 WARNING 级（metadata 类型校验缺失，已在审计项 7 标记）。

**建议**：B2 之前补 3 个测试：(1) from_dict 非 dict metadata 抛 TypeError，(2) copy_with 链式，(3) 相等性语义。其余 INFO 级 gap 可不补。

---

## 10. 跨切关注点 — 父包 eager-load 副作用

**评分**：**WARNING（pre-existing，非 B1 引入）**

### 发现

正常 `from visioncore.state import TargetState` 会触发 `visioncore/__init__.py` 执行，而后者 eagerly 导入 `visioncore.core.*`（含 `visioncore.core.frame`，后者 `import numpy`）。结果：

```
=== Modules loaded transitively by importing visioncore.state ===
  numpy (大量子模块)
  visioncore.core
  visioncore.core.adapters / detection / event / frame / target / track
```

### 责任归属

- **`visioncore/state/target_state.py` 本身**：✅ stdlib-only（独立加载测试证明）
- **`visioncore/state/__init__.py` 本身**：✅ 仅 `from visioncore.state.target_state import TargetState`
- **`visioncore/__init__.py`（pre-existing）**：⚠️ eagerly 导入 core，导致 numpy 被加载

**这不是 B1 引入的问题**——B1 未修改 `visioncore/__init__.py`。但意味着 `visioncore.state` 在 numpy-free 环境中无法独立使用，除非修改父包 init（属于 B2+ 范围）。

### 建议
- B1 阶段：保持现状（不动父包 init，零破坏）
- B2 阶段：考虑将 `visioncore/__init__.py` 的 core 导入改为 lazy（`__getattr__` 模式），让 `visioncore.state` 真正独立

---

## 11. 回归测试

### 证据

```
=== test_target_state.py ===
  32/32 passed, 0 failed

=== test_core_models.py（visioncore.core 回归） ===
  31/31 passed, 0 failed

=== test_event_bus.py（EventBus 回归） ===
  77/77 tests passed

=== test_shadow_integration.py（InferWorker 影子集成回归） ===
  12/12 tests passed
```

**总计 152 项测试全过，零回归。**

### 命名共存验证

```
visioncore.TargetState            : enum      ← 顶层导出未变
visioncore.core.TargetState       : enum      ← 原枚举保持
visioncore.state.TargetState      : dataclass ← 新快照
TopLevel is CoreEnum              : True
TopLevel is StateSnapshot         : False
StateSnapshot fields              : 14
```

✅ 新快照类型未破坏现有 `visioncore.TargetState`（枚举）的顶层导出。

---

## 12. 总体结论

### 评分：**PASS WITH WARNING**

### 评分理由

**PASS 部分（核心契约全部满足）**：
- ✅ TargetState 真正 immutable（frozen + slots + 无 setter）
- ✅ 零业务逻辑（仅值对象基础设施）
- ✅ 零协议耦合（无 UDP/ROS2/MAVLink/socket/...）
- ✅ 零依赖 ai/ gui/ camera/ common/ target_manager eventbus
- ✅ to_dict/from_dict 对称（12 用例 10 通过）
- ✅ copy_with 正确（13 用例全通过）
- ✅ 测试覆盖核心路径（32 项 / 7 类）
- ✅ 零回归（152 项测试全过）

**WARNING 部分（3 项次要问题，建议 B2 前修复）**：
- ⚠️ `from_dict` 不校验 metadata 类型（接受 None / 非 dict 值）—— 审计项 7 Case 9/10
- ⚠️ 测试覆盖有 9 处次要 gap（2 WARNING + 7 INFO）—— 审计项 9
- ⚠️ 父包 `visioncore/__init__.py` eager-loads numpy（pre-existing，非 B1 引入）—— 跨切关注点

### 建议行动项

| 优先级 | 行动 | 里程碑 |
|---|---|---|
| P1 | 修复 `from_dict` metadata 类型校验（抛 TypeError 或归一化为 `{}`） | B2 之前 |
| P2 | 补 3 个测试：非 dict metadata 抛错 / copy_with 链式 / 相等性语义 | B2 之前 |
| P3 | 将 `visioncore/__init__.py` core 导入改为 lazy | B2+ |
| P4 | 安装 `coverage.py` 做自动行覆盖率测量 | 项目级 |

### 最终判定

**Milestone B1 满足验收标准，准予进入 B2。** 3 项 WARNING 不阻塞当前里程碑的功能正确性，但 P1 项应在 B2 开始前修复以收紧契约。

---

## 附录 A — 审计执行命令清单

| 审计项 | 命令 |
|---|---|
| 1 | `python -c "..."` 检查 `__dataclass_params__` / `__slots__` / setattr / delattr |
| 2 | Grep `^\s*(if\|for\|while\|try\|with\|class\|def\|return\|raise\|yield\|assert\|global\|nonlocal\|import\|from)\s` |
| 3 | Grep `\b(udp\|ros2?\|mavlink\|socket\|asyncio\|threading\|...)\b` + 独立加载测试 |
| 4-6 | `importlib.util.spec_from_file_location` 绕过父包 init + sys.modules diff |
| 7 | 12 个边界用例脚本 |
| 8 | 13 个边界用例脚本 |
| 9 | 测试函数清单 + 分类映射 + gap 分析 |

## 附录 B — 文件清单

| 文件 | 行数 | 状态 |
|---|---|---|
| `visioncore/state/__init__.py` | 41 | 新增，审计通过 |
| `visioncore/state/target_state.py` | 314 | 新增，审计通过（含 1 项 P1 修复建议） |
| `tests/test_target_state.py` | ~470 | 新增，32 项测试全过（含 3 项 P2 补测建议） |
| `docs/targetstate-design.md` | 已存在 | 设计文档，本次审计未修改 |
| `docs/targetstate-audit-report.md` | 本文件 | 新增 |
