# Milestone D8 审计报告

> **审计日期**：2026-08-04
> **审计范围**：`visioncore/pipeline/config/pipeline_config.py`（1 个文件，420 行）+ `visioncore/config/__init__.py` + `config/visioncore_pipeline.json` + `visioncore/pipeline/pipeline.py` 更新 + `tests/test_pipeline_config.py`
> **评分**：**PASS**

---

## 0. 审计总结

D8 实现严格遵循规格。`PipelineConfig` 正确读取 JSON（`json.loads`）/ YAML（`yaml.safe_load`）配置文件，通过 `resolve_stage()` 解析阶段类型名→类（BUILTIN_STAGES 内置注册表 + `extra_stages` + `PluginRegistry`），逐个实例化并通过 `Pipeline.add_stage()` 按配置顺序组装流水线；`Pipeline.load_config()` 提供便捷入口；`pipeline_enabled=False` 时返回空流水线。配置中**仅指定阶段顺序和插件名称/参数**，零硬编码业务逻辑。27 项测试验证了多种阶段组合（单阶段、双阶段、空列表、额外阶段覆盖）的构建结果。全量回归 30 套件 + 3 个 doctest 全部通过。

未发现任何问题。

---

## 1. 配置文件读取逻辑正确，支持 YAML/JSON

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 格式分发 | ✅ | `load_file()` 按后缀 `.json` / `.yaml` / `.yml` 分发 | `pipeline_config.py:218-243` |
| JSON 解析 | ✅ | `json.loads(text)` | `pipeline_config.py:235` |
| YAML 解析 | ✅ | `yaml.safe_load(text)`（安全解析，不执行代码） | `pipeline_config.py:228` |
| YAML 缺失容错 | ✅ | `ImportError` 时抛 `PipelineConfigError("YAML config requires 'pyyaml'")` | `pipeline_config.py:222-226` |
| 未知格式拒绝 | ✅ | 非 `.json`/`.yaml`/`.yml` 抛 `PipelineConfigError("unsupported config format")` | `pipeline_config.py:242` |
| 根类型校验 | ✅ | 解析结果非 dict 抛 `PipelineConfigError` | `pipeline_config.py:248` |
| 测试覆盖 | ✅ | `test_build_from_file_simple`（JSON 通过）、`test_bad_json_raises`（非法 JSON）、`test_unsupported_format_raises`（未知格式） | `test_pipeline_config.py:109/256/264` |

---

## 2. Pipeline 的 add_stage 和 run 根据配置动态加载阶段

**结论：PASS**

`build_from_config()`（`pipeline_config.py:302-393`）的核心流程：

```python
for i, entry in enumerate(stages_cfg):          # 遍历配置中的阶段列表
    stage_type = entry.get("type")               # 读取阶段类型名
    params = entry.get("params", {})             # 读取构造参数
    stage_cls = cls.resolve_stage(stage_type, ...) # 解析类型名→类
    stage_instance = stage_cls(**params)           # 实例化
    pipeline.add_stage(stage_instance)             # 按序加入流水线
```

| 检查项 | 结果 | 证据 |
|---|---|---|
| 按配置顺序组装 | ✅ | `for i, entry in enumerate(stages_cfg)` + `pipeline.add_stage(stage_instance)` | `pipeline_config.py:362/387` |
| 阶段类型名→类解析 | ✅ | `resolve_stage()` 三级查找：`extra_stages → BUILTIN_STAGES → plugin_registry` | `pipeline_config.py:285-290` |
| 参数传递到构造器 | ✅ | `stage_cls(**params)` | `pipeline_config.py:382` |
| 返回 NEW 状态流水线 | ✅ | `pipeline = Pipeline()` 后不调用 `initialize()` | `pipeline_config.py:352` |
| 测试验证运行 | ✅ | `test_pipeline_is_runnable_after_config_load`：配置 `["capture","detect"]` → `ctx.metadata["order"] == ["capture","detect"]` | `test_pipeline_config.py:150-162` |
| Pipeline.load_config | ✅ | 委托给 `PipelineConfig.build_from_file()` | `pipeline.py:515-536` |

---

## 3. 测试验证配置生效：指定不同阶段组合时结果符合预期

**结论：PASS**

| 测试 | 阶段组合 | 预期 | 证据 |
|---|---|---|---|
| `test_build_from_file_simple` | 2× DummyStage (a,b) | `len==2, names==["a","b"]` | `test_pipeline_config.py:109` |
| `test_build_from_file_no_params` | 1× DummyStage (无 params) | `len==1, name=="DummyStage"` | `test_pipeline_config.py:123` |
| `test_build_from_file_empty_stages` | 空列表 | `len==0` | `test_pipeline_config.py:133` |
| `test_build_from_file_pipeline_is_new_state` | 1× DummyStage | `initialized==False, shutdown_done==False` | `test_pipeline_config.py:140` |
| `test_pipeline_is_runnable_after_config_load` | 2× DummyStage (capture,detect) | `metadata["order"]==["capture","detect"]` | `test_pipeline_config.py:150` |
| `test_resolve_stage_extra` | 自定义 MyStage | `cls is MyStage` | `test_pipeline_config.py:175` |
| `test_resolve_stage_extra_overrides_builtin` | 自定义 DummyStage 覆盖内置 | `cls is shadow_class` | `test_pipeline_config.py:187` |
| `test_pipeline_load_config` | 1× DummyStage (x) | `len==1, name=="x"` | `test_pipeline_config.py:211` |
| `test_pipeline_enabled_false_returns_empty` | 1× DummyStage + enabled=False | `len==0` | `test_pipeline_config.py:232` |
| 配置验证（8 项） | 各类非法配置 | 各自抛 `PipelineConfigError` 匹配消息 | `test_pipeline_config.py:251-317` |

---

## 4. 禁止在配置中硬编码业务逻辑（仅指定阶段顺序和插件名称）

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 配置结构为纯数据 | ✅ | `{"stages": [{"type": str, "params": dict}]}` — 无代码块、无回调、无表达式 | `config/visioncore_pipeline.json` |
| 阶段类型名仅为标识符 | ✅ | `entry.get("type")` → `str` → `resolve_stage()` 查表 | `pipeline_config.py:370/374` |
| 参数为纯字典 | ✅ | `entry.get("params", {})` → `stage_cls(**params)` — 直接传递到构造器，无中间逻辑 | `pipeline_config.py:376/382` |
| 零条件/分支逻辑 | ✅ | `build_from_config()` 中无 `if`/`for` 基于 config 值的业务分支（除 `pipeline_enabled` 全局标志外） |
| 阶段注册表惰性加载 | ✅ | `_get_builtin_stages()` 仅导入存在的模块（`try/except ImportError`），无业务逻辑注入 | `pipeline_config.py:78-145` |
| `pipeline_enabled` 为全局开关 | ✅ | `settings.pipeline_enabled` 检查在构建前（`build_from_config:325-330`），非配置内嵌逻辑 | `visioncore/config/__init__.py` |

---

## 5. 最终评分

# **PASS**

所有四项检查全部通过。D8 实现严格遵循规格：配置文件读取逻辑正确支持 JSON/YAML、Pipeline 的 add_stage/run 根据配置动态加载阶段并按序执行、测试验证了多种阶段组合的构建结果与运行行为、配置中仅含阶段顺序和插件名称（零硬编码业务逻辑）。未发现任何问题。
