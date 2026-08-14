# Milestone D4 审计报告

> **审计日期**：2026-08-03
> **审计范围**：`visioncore/pipeline/stages/attribute_gesture_stage.py`（1 个文件，297 行）+ `tests/test_attribute_gesture_stage.py` + `visioncore/state/target_state.py` 修改
> **评分**：**PASS**

---

## 0. 审计总结

D4 实现严格遵循规格。`AttributeStage` 和 `GestureStage` 均继承自 `AdvancedStage`（D1），`process()` 使用 `TargetState.copy_with()` 为每个快照设置占位字段（`attributes={}` / `gesture=None`），以 REPLACE 语义回写 `context.target_states`。源文件仅依赖 `visioncore.pipeline.stages.advanced`（D1）、`visioncore.state.target_state` 和标准库，零依赖 ai/gui/camera/torch/onnx 等。36 项测试 + 2 个 doctest 全部通过。

未发现任何问题。

---

## 1. AttributeStage 和 GestureStage 继承与定义

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| `AttributeStage` 继承 `AdvancedStage` | ✅ | `attribute_gesture_stage.py:124` — `class AttributeStage(AdvancedStage):` |
| `GestureStage` 继承 `AdvancedStage` | ✅ | `attribute_gesture_stage.py:214` — `class GestureStage(AdvancedStage):` |
| `AttributeStage` 实现全部抽象成员 | ✅ | `capability`(166) + `initialize`(171) + `process`(176) + `shutdown`(195) + `health_check`(200) |
| `GestureStage` 实现全部抽象成员 | ✅ | `capability`(256) + `initialize`(261) + `process`(266) + `shutdown`(285) + `health_check`(290) |
| `__slots__` | ✅ | `AttributeStage: ("_estimator", "_ready")`; `GestureStage: ("_recogniser", "_ready")` |
| 构造接受可选后端 | ✅ | `AttributeStage(estimator=None, *, name=None)`; `GestureStage(recogniser=None, *, name=None)` |
| 后端不被调用 | ✅ | `process()` 中无 `self._estimator` / `self._recogniser` 引用 |

---

## 2. process() 为每个 TargetState 添加字段

**结论：PASS**

`AttributeStage.process()`（`attribute_gesture_stage.py:185-193`）：
```python
self.check_context(context)
originals: list[TargetState] = list(context.target_states)
enriched: list[TargetState] = [ts.copy_with(attributes={}) for ts in originals]
context.target_states.clear()
context.target_states.extend(enriched)
```

`GestureStage.process()`（`attribute_gesture_stage.py:275-283`）：
```python
self.check_context(context)
originals: list[TargetState] = list(context.target_states)
enriched: list[TargetState] = [ts.copy_with(gesture=None) for ts in originals]
context.target_states.clear()
context.target_states.extend(enriched)
```

| 检查项 | 结果 | 证据 |
|---|---|---|
| 使用 `copy_with()` 派生新快照 | ✅ | `ts.copy_with(attributes={})` / `ts.copy_with(gesture=None)` |
| REPLACE 语义 (clear+extend) | ✅ | `context.target_states.clear()` + `.extend(enriched)` |
| 空列表安全 | ✅ | `for ts in []` 产生空 `enriched`，clear+extend 无副作用 |
| 原始快照字段保留 | ✅ | `copy_with` 仅覆盖指定字段，其余携带原值 |

**测试验证**：

| 测试 | 验证 |
|---|---|
| `test_attribute_sets_attributes_on_each_snapshot` | 3 个快照均有 `ts.attributes == {}` |
| `test_gesture_sets_gesture_on_each_snapshot` | 2 个快照均有 `ts.gesture is None` |
| `test_attribute_preserves_snapshot_fields` | `target_id == 42`, `label == "car"` 不变 |
| `test_gesture_preserves_snapshot_fields` | `target_id == 7`, `label == "head"` 不变 |
| `test_attribute_then_gesture_chain` | 链式后两者字段均存在 |
| `test_attribute_replaces_stale_snapshots` | `ctx.target_states[0] is not old` |

---

## 3. 使用 TargetState 类型，不创建新类型

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 使用 `visioncore.state.target_state.TargetState` | ✅ | `attribute_gesture_stage.py:79` — `from visioncore.state.target_state import TargetState` |
| 不创建新类型 | ✅ | 无 `class`、`@dataclass`、`NamedTuple`、`TypedDict` 定义 |
| `copy_with()` 返回 `TargetState` | ✅ | `TargetState.copy_with()` 返回同类型实例（`type(self)(**new_values)`） |
| 类型注解使用 `TargetState` | ✅ | `originals: list[TargetState]`, `enriched: list[TargetState]` |
| 测试使用 `TargetState` | ✅ | `test_attribute_gesture_stage.py:36` — `from visioncore.state.target_state import TargetState` |

---

## 4. 测试验证属性字段存在与类型

**结论：PASS**

共 36 项测试 + 2 个 doctest，全部通过。

| 验证项 | 测试 | 断言 |
|---|---|---|
| `attributes` 字段存在 | `test_attribute_sets_attributes_on_each_snapshot` | `hasattr(ts, "attributes")` |
| `attributes` 类型为 `dict` | `test_attribute_sets_attributes_on_each_snapshot` | `ts.attributes == {}`（`{}` 是 `dict`） |
| `gesture` 字段存在 | `test_gesture_sets_gesture_on_each_snapshot` | `hasattr(ts, "gesture")` |
| `gesture` 类型为 `None`（`NoneType`） | `test_gesture_sets_gesture_on_each_snapshot` | `ts.gesture is None` |
| 链式两者共存 | `test_attribute_then_gesture_chain` | `ts.attributes == {} and ts.gesture is None` |
| 管线端到端 | `test_pipeline_with_attribute_stage` | `ctx.target_states[0].attributes == {}` |
| 管线端到端 | `test_pipeline_with_gesture_stage` | `ctx.target_states[0].gesture is None` |

`TargetState` 的 frozen dataclass 类型注解 `attributes: dict[str, Any]` 和 `gesture: str | None` 已在 `target_state.py` 中定义，`copy_with()` 在构造时执行类型约束。

---

## 5. 外部依赖与硬编码检查

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 无 torch/onnx/cv2 导入 | ✅ | `test_no_model_imports_in_source` — AST 扫描无违规 |
| 无 gui/ai 导入 | ✅ | `test_no_gui_ai_in_source` — AST 扫描无违规 |
| 运行时无 torch 加载 | ✅ | `test_no_torch_loaded_at_runtime` — `sys.modules` 无 torch |
| 无硬编码模型调用 | ✅ | `process()` 中仅 `copy_with(attributes={})` / `copy_with(gesture=None)`，无函数/方法调用 |
| estimator/recogniser 不被调用 | ✅ | `process()` 中无 `self._estimator` / `self._recogniser` 引用 |

源文件全部 import：
```
from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING
from visioncore.pipeline.stages.advanced import (AdvancedStage, StageCapability)
from visioncore.state.target_state import TargetState
if TYPE_CHECKING: from visioncore.pipeline.context import PipelineContext
```

---

## 6. 最终评分

# **PASS**

所有五项检查全部通过。D4 实现严格遵循规格：两个阶段均继承 AdvancedStage、使用 TargetState.copy_with() 添加占位字段、零外部依赖、36 项测试覆盖充分。未发现任何问题。
