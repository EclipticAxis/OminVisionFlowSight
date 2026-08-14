# Plugin Interface Layer 设计文档 (Milestone D6)

> **版本**：D6 — 2026-08-04
> **范围**：新增 `visioncore/plugin/` 包，定义插件系统核心接口与基础类
> **状态**：已实现，39 项单元测试全部通过 + 2 个模块 doctest 通过，全量回归零破坏（28 个测试套件）

---

## 1. 目标与背景

### 1.1 为什么需要插件层

VisionCore 的未来插件（检测器、跟踪器、算法）需要统一的集成契约：实现接口即可被加载、查询和管理，管线与插件解耦。D6 交付**接口层**：

- **PluginInterface**：所有插件的基类，定义 `load() -> run() -> shutdown()` 生命周期契约。
- **PluginManager**：插件注册表抽象基类（注册、注销、查询），管理插件的加载与生命周期。
- **MemoryPluginManager**：内存注册表参考实现。
- **NullPlugin / ConsolePlugin**：示例插件，验证契约可用性。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **最小契约** | 生命周期仅 `load()` / `run()` / `shutdown()`，基类提供空实现（默认 no-op load、身份透传 run、no-op shutdown），子类按需覆写 |
| **ABC 强制** | `@abstractmethod` 保证不完整子类无法实例化（TypeError） |
| **全路径类型标注** | 接口签名涉及类型时显式使用完整路径（如 `"visioncore.state.target_state.TargetState"`），不依赖导入别名 |
| **禁止网络/协议** | 无 socket/requests/ROS2/MAVLink/ZMQ 代码 |
| **禁止 GUI/ai/camera** | 不触碰 `ai/` `gui/` `camera/` |
| **注册校验** | 非 `PluginInterface` 拒绝、重名拒绝（`PluginError`），注销幂等 |
| **快照不可变** | `run()` 操作 `TargetState` 冻结快照，返回派生副本或原实例 |

### 1.3 全路径类型标注约定

D6 规格要求所有接口涉及类型时使用完整路径。实现方式：在 `from __future__ import annotations` 下把标注写成完整路径字符串，例如：

```python
# visioncore/plugin/base.py
def run(
    self,
    target_state: "visioncore.state.target_state.TargetState",
) -> "visioncore.state.target_state.TargetState":
    ...

def register(
    self,
    plugin: "visioncore.plugin.base.PluginInterface",
) -> None:
    ...
```

`import visioncore.state.target_state`（F401 注释说明）在模块命名空间绑定顶层 `visioncore` 名字，使完整路径字符串可被类型检查器/内省工具解析。

---

## 2. 文件结构

```
visioncore/
├── pipeline/        # (C1-C5, D1-D5) 管线与高级阶段
├── state/           # TargetState 冻结快照
└── plugin/          # (D6) 插件接口层 ← 新增
    ├── __init__.py  # 包导出（PluginError/PluginInterface/PluginManager/
    │                #   MemoryPluginManager/NullPlugin/ConsolePlugin）
    └── base.py      # 全部接口与示例实现 ← 核心

tests/test_plugin_base.py  # 39 项单元测试 ← 新增
docs/plugin-interface.md   # 本文档 ← 新增
```

---

## 3. 接口

### 3.1 PluginInterface（抽象基类）

```python
class PluginInterface(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...          # 注册键
    @property
    @abstractmethod
    def version(self) -> str: ...
    @abstractmethod
    def load(self) -> None: ...          # 空实现（no-op），可覆写
    @abstractmethod
    def run(self, target_state) -> TargetState: ...  # 空实现（身份透传）
    @abstractmethod
    def shutdown(self) -> None: ...      # 空实现（no-op），可覆写
```

- 生命周期契约：`load()`（一次性获取资源）→ `run()`（处理一个目标快照）→ `shutdown()`（释放资源，不得抛异常）
- 基类为空实现提供**默认行为**，但 ABC 仍强制子类声明完整契约（含 `name`/`version`）
- `run()` 的参数/返回值类型为完整路径 `"visioncore.state.target_state.TargetState"`

### 3.2 PluginManager（抽象基类）+ MemoryPluginManager

```python
class PluginManager(ABC):
    @abstractmethod
    def register(self, plugin: PluginInterface) -> None: ...
    @abstractmethod
    def unregister(self, name: str) -> None: ...
    @abstractmethod
    def get(self, name: str) -> PluginInterface | None: ...
    @abstractmethod
    def list(self) -> list[PluginInterface]: ...
    # 基于抽象契约实现的具体便捷方法：
    def names(self) -> list[str]: ...
    def __contains__(self, name) -> bool: ...
    def load_all(self) -> None: ...
    def shutdown_all(self) -> None: ...
```

`MemoryPluginManager(PluginManager)` 是内存参考实现（按注册顺序存储的 dict）：

- `register`：校验 `isinstance(plugin, PluginInterface)`（否则 `PluginError`）；重名拒绝（`PluginError`）
- `unregister`：未知名字静默忽略（幂等）
- `get`：未注册返回 `None`，不抛异常
- `load_all()` / `shutdown_all()`：驱动全部插件生命周期

### 3.3 示例插件

| 插件 | 行为 |
|---|---|
| `NullPlugin(name="null")` | 纯 no-op：`load`/`shutdown` 切换 `_loaded` 标志（`health_check()` 反映）；`run()` 原样返回输入快照 |
| `ConsolePlugin(name="console")` | 通过标准 `logging` 打印 load/run/shutdown 事件，`run()` 透传快照；**无任何网络通信** |

### 3.4 使用示例

```python
from visioncore.plugin import ConsolePlugin, MemoryPluginManager, NullPlugin
from visioncore.state.target_state import TargetState

manager = MemoryPluginManager()
manager.register(NullPlugin())
manager.register(ConsolePlugin("console"))
manager.load_all()                      # 调用每个插件的 load()

ts = TargetState(1, 1, None, "person", 0.9, 0.5, 0.5,
                 0.0, 0.0, 0.1, 0.2, 0.0, 0, {})
out = manager.get("console").run(ts)    # 处理一个快照

manager.shutdown_all()
```

---

## 4. 与既有里程碑的关系

```
C1-D5 管线阶段（PipelineStage / AdvancedStage）
                  │
                  ▼
D6 插件层（PluginInterface）── 未来的检测器/跟踪器/算法插件
                  │      实现接口即可被集成：
                  │         load() → run(TargetState) → shutdown()
                  ▼
D6 PluginManager（注册/查询/生命周期管理）
                  │
                  ▼
D6 示例插件：NullPlugin（no-op）/ ConsolePlugin（仅日志，无网络）
```

`run()` 直接操作 `visioncore.state.target_state.TargetState`（D4/D5 扩展后的 18 字段快照），使插件输出可无缝汇入管线 `context.target_states`。

---

## 5. 测试覆盖（tests/test_plugin_base.py，39 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 包表面 | 6 | ABC 子类关系、PluginError 继承 |
| 抽象性 | 3 | 接口/管理器不可实例化、不完整子类不可实例化 |
| NullPlugin | 7 | 基本方法可调用、run() 身份透传（`is` 断言）、快照保留、完整生命周期 |
| ConsolePlugin | 6 | 基本方法可调用、run() 返回输入、生命周期 |
| PluginManager 接口 | 10 | register/get/contains/unregister/list/names 不抛异常、未知查询返回 None、load_all/shutdown_all、空注册表 no-op |
| 注册校验 | 3 | 非插件拒绝、重名拒绝、注销后重注册 |
| 全路径标注 | 3 | `run()` 使用 `visioncore.state.target_state.TargetState`、管理器使用 `visioncore.plugin.base.PluginInterface`、类型往返 |
| 零依赖 | 2 | AST 无网络/模型/gui/ai 导入、运行时无 torch |

---

## 6. 零侵入声明

新增 `visioncore/plugin/` 包与 `tests/test_plugin_base.py`，**未修改任何既有文件**（沿用 `visioncore.state` 的惯例，插件层不从 `visioncore/__init__.py` 顶层再导出）。全量回归：28 个测试套件全部通过。
