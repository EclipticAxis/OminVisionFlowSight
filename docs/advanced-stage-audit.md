# Milestone D1 审计报告

> **审计日期**：2026-08-03
> **审计范围**：`visioncore/pipeline/stages/advanced/`（3 个 Python 模块）+ `tests/test_advanced_stage.py`
> **评分**：**PASS**

---

## 0. 审计总结

D1 实现严格遵循规格。`AdvancedStage` 正确继承 `PipelineStage`，四方法生命周期全部为 `@abstractmethod`；`StageCapability` 为 `@dataclass(frozen=True, kw_only=True)`，五字段齐全；`DummyAdvancedStage` 实现全部接口，无外部副作用。三个源文件仅依赖 `visioncore.pipeline.base`、`visioncore.pipeline.context`（TYPE_CHECKING）和 `visioncore.pipeline.stages.advanced.stage_capability`，**不依赖 ai/gui/camera/target_manager**。47 项单元测试 + 2 个模块 doctest 全部通过。

发现 1 个 **WARNING 级别**代码质量问题（不影响功能）。

---

## 1. AdvancedStage 继承与方法定义

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 继承 `PipelineStage` | ✅ | `advanced_stage.py:152` — `class AdvancedStage(PipelineStage):` |
| `initialize` 抽象 | ✅ | `advanced_stage.py:235` — `@abstractmethod`，仅 docstring |
| `process` 抽象 | ✅ | `advanced_stage.py:245` — `@abstractmethod`，仅 docstring |
| `shutdown` 抽象 | ✅ | `advanced_stage.py:257` — `@abstractmethod`，仅 docstring |
| `health_check` 抽象 | ✅ | `advanced_stage.py:266` — `@abstractmethod`，仅 docstring |
| `capability` 抽象属性 | ✅ | `advanced_stage.py:214-216` — `@property @abstractmethod`，仅 docstring |
| 具体方法无业务逻辑 | ✅ | `check_context` — 纯契约校验（字段存在性 + list 类型检查）；`__repr__` — 纯显示 |
| `__slots__ = ()` | ✅ | `advanced_stage.py:208` — 无额外实例字段 |

AdvancedStage 增加了 `capability` 抽象属性和 `check_context` 具体校验方法，两者均为抽象层基础设施，不含任何业务处理逻辑。四方法生命周期与 `PipelineStage` 完全一致，Pipeline 无需任何改动即可调度 AdvancedStage。

---

## 2. StageCapability 数据类

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| `@dataclass(frozen=True)` | ✅ | `stage_capability.py:81` — `@dataclass(frozen=True, kw_only=True)` |
| `kw_only=True` | ✅ | 同上 |
| 字段 `name: str` | ✅ | `stage_capability.py:132` |
| 字段 `version: str` | ✅ | `stage_capability.py:133` |
| 字段 `description: str` | ✅ | `stage_capability.py:134` |
| 字段 `required_context: list[str]` | ✅ | `stage_capability.py:135` |
| 字段 `provided_context: list[str]` | ✅ | `stage_capability.py:136` |
| 不可变 | ✅ | `test_capability_is_frozen` — 赋值抛 `FrozenInstanceError` |
| 关键字构造可读性 | ✅ | `test_capability_keyword_construction_reads_clearly` — 五字段全部关键字构造，值正确 |
| `"target_states"` 词汇 | ✅ | 文档化 `List[TargetState]` 契约；`test_capability_provided_context_may_describe_target_states` |

五个字段名与规格完全一致。`kw_only=True` 强制所有参数使用关键字传递，避免位置混淆。

---

## 3. 关键字构造强制检查

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 位置参数被拒绝 | ✅ | `test_capability_kw_only_rejects_positional` — `StageCapability("a", "1.0.0", "d", [], [])` 抛 `TypeError` |
| 签名验证所有参数 KEYWORD_ONLY | ✅ | `test_capability_is_frozen_kw_only_dataclass` — `inspect.signature` 检查每个参数的 `kind` |
| `frozen` 参数验证 | ✅ | `StageCapability.__dataclass_params__.frozen is True` |

---

## 4. DummyAdvancedStage 接口实现与副作用

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 继承 `AdvancedStage` | ✅ | `advanced_stage.py:389` — `class DummyAdvancedStage(AdvancedStage):` |
| 实现 `capability` 属性 | ✅ | `advanced_stage.py:511-514` — 返回构造时注入的 `_capability` |
| 实现 `initialize` | ✅ | `advanced_stage.py:520-525` — 仅计数器 + healthy 标志 |
| 实现 `process` | ✅ | `advanced_stage.py:527-586` — check_context + trace + read + write |
| 实现 `shutdown` | ✅ | `advanced_stage.py:588-593` — 仅计数器 + healthy 标志 |
| 实现 `health_check` | ✅ | `advanced_stage.py:595-597` — 返回 `_healthy` |
| 无文件 I/O | ✅ | 无 `open()`、`pathlib`、`os` 等文件操作 |
| 无网络调用 | ✅ | 无 `socket`、`urllib`、`requests` 等网络操作 |
| 无外部系统调用 | ✅ | 仅操作 `context.metadata` 与声明的 context 列表 |
| `raise_on_process` 不触碰 context | ✅ | `test_dummy_raise_on_process` — `ctx.metadata == {}` |
| `reset()` 保留 capability | ✅ | `test_dummy_capability_unchanged_by_reset` — capability 引用不变 |

---

## 5. 禁止依赖检查

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 无 `ai/` 导入 | ✅ | AST 分析：imports 仅为 `__future__`、`abc`、`dataclasses`、`typing`、`visioncore.pipeline.*` |
| 无 `gui/` / `PyQt` 导入 | ✅ | `test_no_gui_inferworker_in_source` — AST 扫描无违规 |
| 无 `camera/` 导入 | ✅ | 同上 |
| 无 `target_manager` 引用 | ✅ | 无直接或间接引用 `visioncore.target_manager` |
| 运行时无 GUI 加载 | ✅ | `test_no_gui_inferworker_loaded_at_runtime` — `sys.modules` 无 PyQt |
| 既有包不受影响 | ✅ | `test_imports_do_not_break_existing_package` — stages 包导出完整 |

依赖关系图：
```
advanced/__init__.py
  ├─ advanced_stage.py
  │    ├─ visioncore.pipeline.base         (PipelineStage, StageError)
  │    ├─ visioncore.pipeline.context      (TYPE_CHECKING only — 仅类型提示，无运行时依赖)
  │    └─ stage_capability.py              (StageCapability)
  └─ stage_capability.py
       └─ dataclasses                      (标准库)
```

源码中出现的 "gui"、"camera"、"ai" 字样均为 **docstring 中的文档说明**（"不修改 ai/gui/camera"），非实际导入。

---

## 6. 测试覆盖

**结论：PASS**

共 47 项单元测试 + 2 个模块 doctest，全部通过。

| 覆盖领域 | 测试数 | 关键测试 |
|---|---|---|
| 包表面 / 导出 | 3 | `test_package_exports_advanced_names`, `test_imports_do_not_break_existing_package` |
| StageCapability | 7 | frozen/kw_only/fields/keyword-readable/"target_states"/list identity/docstring contract |
| AdvancedStage ABC | 5 | isabstract/issubclass/4-method consistency/subclass failure modes/concrete subclass |
| check_context | 6 | passes/missing required/missing provided/not list/error message/first violation |
| DummyAdvancedStage | 13 | lifecycle callable/counters/idempotent init/trace/read/write/raise/reset |
| 上下文约束 | 4 | tracing context/undeclared read/undeclared write/real context |
| 管线集成 | 2 | pipeline add_stage→run→shutdown/repr diagnostics |
| 零依赖守护 | 2 | AST scan + runtime sys.modules check |
| repr | 1 | `test_advanced_stage_repr_reports_capability` — name + capability + healthy |

### 生命周期方法覆盖

| 生命周期方法 | 覆盖测试 |
|---|---|
| `initialize()` | `test_dummy_lifecycle_methods_are_callable`, `test_dummy_initialize_idempotent_health` |
| `process()` | `test_dummy_lifecycle_methods_are_callable`, `test_dummy_process_traces_marker`, `test_dummy_process_reads_required_field`, `test_dummy_process_writes_provided_field`, `test_dummy_process_uses_write_value`, `test_dummy_process_preserves_list_identity`, `test_dummy_process_no_required_field_skips_read_record`, `test_dummy_raise_on_process` |
| `shutdown()` | `test_dummy_lifecycle_methods_are_callable`, `test_dummy_shutdown_never_raises` |
| `health_check()` | `test_dummy_lifecycle_methods_are_callable` (False→True→False 转换) |

### 能力描述覆盖

| 能力描述检查 | 覆盖测试 |
|---|---|
| `frozen` 属性 | `test_capability_is_frozen` |
| `kw_only` 强制 | `test_capability_is_frozen_kw_only_dataclass`, `test_capability_kw_only_rejects_positional` |
| 字段正确初始化 | `test_capability_keyword_construction_reads_clearly` |
| 关键字参数可读性 | `test_capability_keyword_construction_reads_clearly` |
| `"target_states"` 词汇 | `test_capability_provided_context_may_describe_target_states`, `test_capability_may_contain_target_states_with_type_annotation_contract` |

### 上下文约束覆盖

| 约束检查 | 覆盖测试 |
|---|---|
| `process` 只读 declared fields | `test_process_touches_only_declared_fields` — `_TracingContext` 验证访问集 ⊆ 声明字段 |
| `process` 只写 declared fields | `test_process_touches_only_declared_fields` — 同上 |
| 缺声明字段报错 | `test_process_blocks_undeclared_read`, `test_process_blocks_undeclared_write` |
| 真实 context 端到端 | `test_process_contract_honest_on_real_context` |

---

## 7. WARNING

### WARNING-1：`advanced_reads` 类型注解与实际赋值不一致

`advanced_stage.py:571`：
```python
reads: list[Any] = context.metadata.setdefault("advanced_reads", {})
```

类型注解为 `list[Any]`，但实际赋值为 `dict`（`{}`）。`test_dummy_process_reads_required_field` 正确断言 `== {"detections": 2}`（是一个 dict），说明功能正确但类型注解误导。

**严重程度**：低 — 纯代码质量问题，Python 不在运行时检查变量注解，无功能影响。

**修复建议**：将 `reads: list[Any]` 改为 `reads: dict[str, int]`。

---

## 8. 最终评分

# **PASS**

所有六项检查全部通过。1 个 WARNING 为代码质量级别（类型注解不一致），不影响功能、安全性或里程碑交付。D1 实现严格遵循规格，零侵入既有代码，测试覆盖完整。
