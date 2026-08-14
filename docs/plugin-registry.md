# Plugin Registry 设计文档 (Milestone D7)

> **版本**：D7 — 2026-08-04
> **范围**：新增 `visioncore/plugin/registry.py`，实现插件动态注册与发现机制
> **状态**：已实现，24 项单元测试全部通过 + 1 个模块 doctest 通过，全量回归零破坏（29 个测试套件）

---

## 1. 目标与背景

### 1.1 为什么需要插件注册表

D6 定义了插件接口层（`PluginInterface` / `PluginManager`），但只有静态的内存注册。D7 引入 **`PluginRegistry`**，扩展发现能力：

- **`get_plugin(name)`**：外部查询名（D7 规格指定）。
- **`load_entrypoints(group_name)`**：通过 `importlib.metadata` 扫描 setuptools entry points，支持通过包配置（`pyproject.toml` / `setup.cfg`）发布插件。
- **`load_package(dotted_path)`**：按 dotted path 导入模块，扫描其中的 `PluginInterface` 实例——用于进程内配置驱动加载和无法依赖已安装 entry points 的测试场景。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **继承 PluginManager** | 复用 `names()` / `__contains__()` / `load_all()` / `shutdown_all()` 便捷方法 |
| **标准库发现** | `importlib.metadata`（Python >= 3.9），无第三方依赖 |
| **容错加载** | entry points 加载失败时记录 warning 并跳过，注册表保持可用 |
| **重复/非法拒绝** | 与 D6 `MemoryPluginManager` 一致的校验规则 |
| **零重量** | 无 pluggy / setuptools 运行时依赖，仅使用标准库 |
| **禁止网络/协议** | 无 socket/requests/ROS2/MAVLink |

---

## 2. 文件结构

```
visioncore/plugin/
├── __init__.py       # D6 包导出 + D7 新增 PluginRegistry
├── base.py           # D6 接口层（PluginInterface / PluginManager / 示例插件）
└── registry.py       # D7 动态注册表 ← 新增

tests/test_plugin_registry.py  # 24 项单元测试 ← 新增
tests/__init__.py              # 使 tests/ 可作为包导入（load_package 测试需要）
tests/_plugin_fixtures.py      # 测试用插件实例模块（load_package 加载目标）
docs/plugin-registry.md        # 本文档 ← 新增
```

---

## 3. 接口

### 3.1 PluginRegistry

```python
class PluginRegistry(PluginManager):
    def __init__(self) -> None: ...
    # --- PluginManager 抽象契约的具体实现 ---
    def register(self, plugin: PluginInterface) -> None: ...
    def unregister(self, name: str) -> None: ...
    def get(self, name: str) -> PluginInterface | None: ...
    def list(self) -> list[PluginInterface]: ...
    # --- D7 新增 ---
    def get_plugin(self, name: str) -> PluginInterface | None: ...
    def load_entrypoints(self, group_name: str) -> list[PluginInterface]: ...
    def load_package(self, dotted_path: str) -> list[PluginInterface]: ...
    # --- 便捷 ---
    def __len__(self) -> int: ...
```

| 方法 | 行为 |
|---|---|
| `register(plugin)` | 注册插件，非 PluginInterface 或重名抛 `PluginError` |
| `unregister(name)` | 注销（幂等，未知名静默忽略） |
| `get(name)` / `get_plugin(name)` | 按名查询（`get_plugin` 是 D7 规格外部查询名，委托给 `get`） |
| `list()` | 注册顺序返回全部插件 |
| `names()` / `__contains__()` / `load_all()` / `shutdown_all()` | 继承自 `PluginManager` |
| `load_entrypoints(group)` | 扫描 entry points，实例/子类自动注册，失败跳过 |
| `load_package(dotted_path)` | 导入模块，扫描 `PluginInterface` 实例属性 |

### 3.2 entry-point 发现流程

```
load_entrypoints("visioncore.plugin")
  │
  ├─ importlib.metadata.entry_points(group="visioncore.plugin")
  │    返回 [EntryPoint("my_plugin", "mypkg:MyPlugin"), ...]
  │
  ├─ for ep in entry_points:
  │    loaded = ep.load()
  │    if isinstance(loaded, PluginInterface):   ← 实例，直接注册
  │        registry.register(loaded)
  │    elif issubclass(loaded, PluginInterface): ← 类，先实例化
  │        registry.register(loaded())
  │    else: skip + warning
  │
  └─ return [registered plugins]
```

### 3.3 dotted-path 模块加载流程

```
load_package("mypackage.plugins.my_plugin")
  │
  ├─ module = importlib.import_module(dotted_path)
  │
  ├─ for attr in dir(module):
  │    obj = getattr(module, attr)
  │    if isinstance(obj, PluginInterface):
  │        registry.register(obj)
  │
  └─ return [registered plugin instances]
```

---

## 4. 与既有里程碑的关系

```
D6 PluginInterface / PluginManager / MemoryPluginManager
                  │
                  ▼
D7 PluginRegistry ──── 继承 PluginManager，新增发现机制
       │
       ├─ load_entrypoints() ── setuptools entry points → 自动注册
       ├─ load_package()     ── dotted-path 模块扫描 → 自动注册
       └─ get_plugin()       ── D7 规格外部查询名
```

---

## 5. 测试覆盖（tests/test_plugin_registry.py，24 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 包表面 | 2 | PluginManager 子类、__init__.py 可导入 |
| 构造 | 2 | 默认构造、初始为空（len/length/names） |
| 注册/查询 | 4 | register→get_plugin、未知返回 None、多插件注册、列表顺序 |
| 注销 | 3 | 删除、未知名幂等、注销后重注册 |
| 管理器便捷 | 3 | names / contains / load_all + shutdown_all |
| 校验 | 2 | 非插件拒绝、重名拒绝 |
| load_package | 4 | 真实模块发现（alpha/beta）、无插件模块返回空、不存在模块抛 PluginError、重复加载跳过 |
| load_entrypoints | 2 | 不存在组返回空（不抛异常）、importlib.metadata 可用 |
| 零依赖 | 2 | AST 无网络/模型/gui/ai、运行时无 torch |

---

## 6. 零侵入声明

新增 `visioncore/plugin/registry.py`，更新 `visioncore/plugin/__init__.py`（新增导出 `PluginRegistry`）。未修改任何其他既有文件。全量回归：29 个测试套件全部通过。
