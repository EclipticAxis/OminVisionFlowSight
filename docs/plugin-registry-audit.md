# Milestone D7 审计报告

> **审计日期**：2026-08-04
> **审计范围**：`visioncore/plugin/registry.py`（1 个文件，332 行）+ `visioncore/plugin/__init__.py` 更新 + `tests/test_plugin_registry.py`
> **评分**：**PASS**

---

## 0. 审计总结

D7 实现严格遵循规格。`PluginRegistry` 继承 `PluginManager`（D6），正确实现了注册（`register`）、查找（`get` / `get_plugin`）、注销（`unregister`）、枚举（`list`）全部方法，内部以 `dict[str, PluginInterface]` 维护唯一标识映射。`load_entrypoints()` 通过 `importlib.metadata` 扫描 setuptools entry points（带版本兼容 fallback + 容错日志），`load_package()` 按 dotted-path 导入模块并扫描实例——手动注册与自动发现场景均有测试覆盖。24 项测试验证了多插件注册（3 个）、列表顺序、重名拒绝（`PluginError`）、未知查询返回 `None`、注销幂等等。全量回归 29 套件 + 1 个 doctest 全部通过。

未发现任何问题。

---

## 1. PluginRegistry 正确实现注册、查找插件方法

**结论：PASS**

`PluginRegistry(PluginManager)`（`registry.py:63`）实现 `PluginManager` 四项抽象方法，内部存储为 `_plugins: dict[str, PluginInterface]`（`registry.py:97`）。

| 方法 | 行为 | 证据 |
|---|---|---|
| `register(plugin)` | 校验 `isinstance(plugin, PluginInterface)`（否则 `PluginError`）；校验 `name` 未重复（否则 `PluginError`）；写入 `self._plugins[name]` | `registry.py:104-124` |
| `unregister(name)` | `self._plugins.pop(name, None)` — 未知名静默忽略（幂等） | `registry.py:127-132` |
| `get(name)` | `self._plugins.get(name)` — 未注册返回 `None` | `registry.py:135-140` |
| `get_plugin(name)` | D7 规格外部查询名，委托给 `get()` | `registry.py:150-164` |
| `list()` | `list(self._plugins.values())` — 注册顺序 | `registry.py:142-147` |
| `names()` / `__contains__()` / `load_all()` / `shutdown_all()` | 继承自 `PluginManager`（组合实现） | D6 已审计 |
| `__len__()` | `len(self._plugins)` | `registry.py:323` |

内部字典键为 `plugin.name`（`PluginInterface` 的 `name` 属性），即插件的唯一标识。

---

## 2. entry-point 逻辑检测（及手动注册场景）

**结论：PASS**

### 2.1 entry-point 发现（`load_entrypoints`）

`load_entrypoints(group_name)`（`registry.py:168`）实现流程：

| 步骤 | 实现 | 证据 |
|---|---|---|
| 导入 `importlib.metadata` | `try: from importlib.metadata import entry_points`；失败时 `logger.warning` + 返回空 | `registry.py:191-197` |
| 扫描 entry points | `entry_points(group=group_name)`；Python 3.12+ 与 3.9-3.11 API 兼容 fallback | `registry.py:203-213` |
| 逐个加载 | `ep.load()`，失败时 `logger.warning` 并跳过 | `registry.py:218-220` |
| 类型判断 | `isinstance(loaded, PluginInterface)` → 直接注册；`issubclass` → 先实例化再注册 | `registry.py:227-239` |
| 注册 | `self.register(plugin)`；重复名跳过（`logger.warning`） | `registry.py:247-257` |

### 2.2 手动注册场景

| 测试 | 场景 | 证据 |
|---|---|---|
| `test_registry_register_then_get_plugin` | 单插件注册→查询 | `test_plugin_registry.py:114` |
| `test_registry_register_multiple` | 3 个插件注册→逐一查询 | `test_plugin_registry.py:126` |
| `test_registry_list_order` | 3 个插件注册→列表顺序 | `test_plugin_registry.py:140` |
| `test_registry_register_after_unregister_ok` | 注销→重注册→查询 | `test_plugin_registry.py:168` |
| `test_registry_load_entrypoints_no_entries` | 不存在的组→返回空不抛异常 | `test_plugin_registry.py:269` |
| `test_registry_load_entrypoints_importlib_available` | `importlib.metadata.entry_points` 可用 | `test_plugin_registry.py:277` |

---

## 3. 注册插件有唯一标识，查询功能正常

**结论：PASS**

插件以 `PluginInterface.name` 属性作为唯一标识（注册键），内部字典 `self._plugins` 以 `name: str → PluginInterface` 映射存储。

| 检查项 | 结果 | 证据 |
|---|---|---|
| 唯一标识 | ✅ | `plugin.name` 作为键；`isinstance` 校验确保 name 属性存在 | `registry.py:106-118` |
| 按名查询 | ✅ | `get_plugin("null") is plugin` | `test_plugin_registry.py:118` |
| 未知查询返回 None | ✅ | `get_plugin("missing") is None` | `test_plugin_registry.py:123` |
| 查询不抛异常 | ✅ | 未知查询不抛异常，`None` 返回 | 同上 |
| 注册后字典长度正确 | ✅ | `len(registry) == 3` | `test_plugin_registry.py:132` |
| 列表保持注册顺序 | ✅ | `[a, b, c]` 顺序 | `test_plugin_registry.py:150-151` |

---

## 4. 多插件注册与冲突处理

**结论：PASS**

| 场景 | 行为 | 证据 |
|---|---|---|
| 多插件注册（3 个） | 全部成功，`len==3`，`list()` 顺序正确 | `test_plugin_registry.py:126-138` |
| 重名注册 | 抛 `PluginError("...already registered")` | `test_plugin_registry.py:219-222` |
| 非 PluginInterface 注册 | 抛 `PluginError("...expected a PluginInterface")` | `test_plugin_registry.py:213-216` |
| 注销后重注册 | 同名重新注册成功 | `test_plugin_registry.py:168-173` |
| 未知名注销 | 静默忽略（幂等，不抛异常） | `test_plugin_registry.py:163-166` |
| load_package 重复加载 | 第二次跳过已注册名 | `test_plugin_registry.py:256-263` |
| load_package 不存在模块 | 抛 `PluginError("failed to import")` | `test_plugin_registry.py:249-252` |

---

## 5. 最终评分

# **PASS**

所有四项检查全部通过。D7 实现严格遵循规格：PluginRegistry 正确实现注册/查找/注销/枚举全部方法、entry-point 发现逻辑完整（带 Python 3.9-3.12 兼容 + 容错）、手动注册与自动发现场景均有测试覆盖、多插件注册（3 个）和冲突处理（重名/非插件/注销后重注册/重复加载）验证充分。未发现任何问题。
