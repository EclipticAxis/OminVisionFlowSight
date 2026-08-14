# Milestone D5 审计报告

> **审计日期**：2026-08-04
> **审计范围**：`visioncore/pipeline/stages/rectangle_filter_stage.py`（1 个文件，443 行）+ `tests/test_rectangle_filter_stage.py` + `visioncore/state/target_state.py` 修改
> **评分**：**PASS**

---

## 0. 审计总结

D5 实现严格遵循规格。`RectangleStage`、`FilterStage`、`HealthStage` 均正确继承 `AdvancedStage`（D1）：`RectangleStage` 用纯算术把 `cx/cy/width/height` 换算为 `rect=(x1,y1,x2,y2)` 并 `copy_with` 回写；`FilterStage` 只评估占位谓词 `confidence < 0`，**不修改列表、不删除任何目标**；`HealthStage` 为每个快照写入 `health=1.0`。源文件仅依赖标准库 + `visioncore.pipeline.stages.advanced`（D1）+ `visioncore.state.target_state`，零 ai/gui/camera/torch/onnx 依赖。51 项测试 + 1 个模块 doctest 全部通过，全量回归 27 个套件零破坏。

未发现任何问题。

---

## 1. 三个阶段的继承与定义

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| `RectangleStage` 继承 `AdvancedStage` | ✅ | `rectangle_filter_stage.py:148` — `class RectangleStage(AdvancedStage):` |
| `FilterStage` 继承 `AdvancedStage` | ✅ | `rectangle_filter_stage.py:266` — `class FilterStage(AdvancedStage):` |
| `HealthStage` 继承 `AdvancedStage` | ✅ | `rectangle_filter_stage.py:361` — `class HealthStage(AdvancedStage):` |
| 实现全部抽象成员 | ✅ | 三阶段均实现 `capability`(108/120/132) + `initialize` + `process`(208/323/413) + `shutdown` + `health_check` |
| `__slots__` | ✅ | `RectangleStage: ("_rect_builder", "_ready")`(175)；`FilterStage: ("_filterer", "_ready")`(290)；`HealthStage: ("_health_monitor", "_ready")`(380) |
| 构造接受算法实例但暂不调用 | ✅ | 三阶段构造参数 `rect_builder`/`filterer`/`health_monitor` 默认 `None`；`process()` 内无任何对后端的引用 |
| capability 声明正确 | ✅ | `name="rectangle"/"filter"/"health"`，`version="0.1.0"`，`required=["target_states"]`，`provided=["target_states"]` |

---

## 2. process() 为每个 TargetState 添加示例字段

**结论：PASS**

`RectangleStage.process()`（`rectangle_filter_stage.py:217-223`）：
```python
self.check_context(context)
originals: list[TargetState] = list(context.target_states)
enriched: list[TargetState] = [
    ts.copy_with(rect=self._derive_rect(ts)) for ts in originals
]
context.target_states.clear()
context.target_states.extend(enriched)
```

`_derive_rect()`（`rectangle_filter_stage.py:241-245`）为纯算术：
```python
x1 = ts.cx - ts.width / 2.0;  y1 = ts.cy - ts.height / 2.0
x2 = ts.cx + ts.width / 2.0;  y2 = ts.cy + ts.height / 2.0
return (x1, y1, x2, y2)
```

`HealthStage.process()`（`rectangle_filter_stage.py:422-424`）：
```python
enriched = [ts.copy_with(health=1.0) for ts in originals]
```

| 检查项 | 结果 | 证据 |
|---|---|---|
| 使用 `copy_with()` 派生新快照 | ✅ | `copy_with(rect=self._derive_rect(ts))` / `copy_with(health=1.0)` |
| REPLACE 语义 (clear+extend) | ✅ | `context.target_states.clear()` + `.extend(enriched)` |
| `rect` 类型为 `tuple[float,float,float,float]` | ✅ | 字段声明 `target_state.py:154` — `rect: tuple[float, float, float, float] \| None = None`；`_derive_rect` 返回 4 元组 |
| `health` 类型为 `float` | ✅ | 字段声明 `target_state.py:155` — `health: float \| None = None`；写入字面量 `1.0` |
| 空列表安全 | ✅ | `for ts in []` 产生空 `enriched`，clear+extend 无副作用 |
| 原始快照字段保留 | ✅ | `copy_with` 仅覆盖指定字段，其余携带原值 |
| `rect` 换算与字段定义联动 | ✅ | 新增字段均有默认值，既有构造调用不受影响（18 字段计数测试通过） |

---

## 3. 不实际丢弃任何目标（可扩展）

**结论：PASS**

`FilterStage.process()`（`rectangle_filter_stage.py:332-340`）：
```python
self.check_context(context)
pending_drop: list[TargetState] = [
    ts for ts in context.target_states if ts.confidence < 0.0
]
logger.debug(
    "FilterStage.process: name=%s evaluated=%d pending_drop=%d "
    "(D5 placeholder: deletes nothing)",
    self._name, len(context.target_states), len(pending_drop),
)
```

| 检查项 | 结果 | 证据 |
|---|---|---|
| 谓词与规格示例一致 | ✅ | 丢弃条件 `if target.confidence < 0`（`rectangle_filter_stage.py:334`） |
| 不删除目标 | ✅ | 只读列表构建 `pending_drop` 并仅 `logger.debug`，**从不 clear/extend/remove** |
| 不改动列表内容与元素身份 | ✅ | `context.target_states` 完全未触达（无任何写操作） |
| 可扩展性 | ✅ | 未来注入 `filterer` 后在同一评估点（`ts.confidence < 0.0` 处）执行真正抑制，外壳结构不变 |
| 对异常数据仍不删除 | ✅ | `test_filter_keeps_anomalous_negative_confidence`（`test_rectangle_filter_stage.py:412`）用 `confidence=-0.5` 的快照验证仍被保留 |

---

## 4. 测试验证字段存在、类型与无误删除

**结论：PASS**

共 51 项测试 + 1 个 doctest，全部通过（`Result: 51 passed, 0 failed, 51 total`）。

| 验证项 | 测试（行号） | 断言 |
|---|---|---|
| `rect` 字段存在且类型正确 | `test_rectangle_adds_rect_field_with_correct_type`(274) | `hasattr(ts,"rect")`、`isinstance(ts.rect, tuple)`、`len(ts.rect)==4`、每个元素 `isinstance(v, float)` |
| `rect` 换算值正确 | `test_rectangle_rect_values_match_corner_derivation`(290) | `isclose(rect, (0.4, 0.3, 0.6, 0.7))`（cx=0.5,cy=0.5,w=0.2,h=0.4） |
| 逐快照独立换算 | `test_rectangle_rect_respects_per_snapshot_box`(305) | 不同框→不同 rect |
| Filter 可调用且不删除 | `test_filter_deletes_nothing_but_is_callable`(396) | `len` 不变、列表身份不变、**元素身份不变**（`ts is originals[i]`） |
| 异常数据仍不删除 | `test_filter_keeps_anomalous_negative_confidence`(412) | `confidence=-0.5` 快照保留 |
| `health` 字段数值类型 | `test_health_adds_health_numeric_field_on_each`(460) | `hasattr(ts,"health")`、`isinstance(ts.health, float)`、`ts.health == 1.0` |
| 后端从不被调用 | `test_rectangle/filter/health_backend_not_called`(545/554/563) | `_SpyBackend.called is False` |
| 链式 | `test_rectangle_filter_health_chain`(576) | rect+health 共存，无删除 |
| 管线集成 | `test_pipeline_with_d5_stages`(615) | Pipeline 运行后 rect tuple×4 + health==1.0 |
| `TargetState` 新增字段 | `test_construction_default_rect_health_none` / `test_rect_and_health_round_trip`（`test_target_state.py`） | 默认 None；tuple/float 经 `to_dict`/`from_dict`/`copy_with` 往返不丢 |

---

## 5. 依赖检查（仅标准库 + 既有管线组件）

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 无模型库导入 | ✅ | `test_no_model_imports_in_source`(638) — AST 扫描 yolo/rt_detr/groundingdino/sam/ultralytics/torch/onnx/cv2 零违规 |
| 无 gui/ai 导入 | ✅ | `test_no_gui_ai_in_source`(660) — AST 扫描 gui/PyQt5/PyQt6/inference/inferworker 零违规 |
| 运行时无 torch 加载 | ✅ | `test_no_torch_loaded_at_runtime`(681) — `sys.modules` 无 torch |
| 不调用任何具体图像处理函数 | ✅ | `process()` 中仅算术运算（`width/2.0` 等）+ `copy_with` + 列表操作 |

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

所有五项检查全部通过。D5 实现严格遵循规格：三个阶段均正确继承 AdvancedStage、process() 使用 TargetState.copy_with() 添加类型正确的示例字段（rect 4 元组 / health 浮点）、FilterStage 只评估谓词绝不删除目标、51 项测试覆盖字段存在/类型/无误删除/后端不被调用、零外部依赖。未发现任何问题。
