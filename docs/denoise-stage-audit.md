# Milestone D2 审计报告

> **审计日期**：2026-08-03
> **审计范围**：`visioncore/pipeline/stages/denoise_stage.py`（1 个文件，219 行）+ `tests/test_denoise_stage.py`
> **评分**：**PASS**

---

## 0. 审计总结

D2 实现严格遵循规格。`DenoiseStage` 继承自 `AdvancedStage`（D1），构造时接受可选 `denoiser` 参数并持有引用但不调用；`process()` 仅执行 `check_context` 校验 + 日志，**不修改 `context.frame`**。源文件零依赖 ai/gui/camera/cv2/torch 等外部库，仅依赖 `visioncore.pipeline.stages.advanced`（D1 抽象层）和标准库。28 项测试 + 1 个 doctest 全部通过。

未发现任何问题。

---

## 1. DenoiseStage 继承与依赖

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 继承 `AdvancedStage` | ✅ | `denoise_stage.py:108` — `class DenoiseStage(AdvancedStage):` |
| `__slots__` | ✅ | `denoise_stage.py:132` — `__slots__ = ("_denoiser", "_ready")`，仅两个实例字段 |
| 无多余依赖 | ✅ | 仅 `from visioncore.pipeline.stages.advanced import (AdvancedStage, StageCapability)` + 标准库 `logging` / `typing` |
| `PipelineContext` 仅 TYPE_CHECKING | ✅ | `denoise_stage.py:78-79` — `if TYPE_CHECKING: from visioncore.pipeline.context import PipelineContext` |
| 不修改 context 字段类型 | ✅ | `provided_context=[]`（D2 passthrough 不写回任何字段） |

---

## 2. 禁止 AI 模型 / cv2 检查

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 无 `cv2` 导入 | ✅ | AST 分析：imports 仅为 `__future__`、`logging`、`typing`、`visioncore.pipeline.stages.advanced`、`visioncore.pipeline.context` |
| 无 `torch` / `onnx` / `ultralytics` | ✅ | 同上 |
| 无 `ai/` 引用 | ✅ | `test_no_gui_ai_in_source` — AST 扫描无违规 |
| 无 `gui/` / `PyQt` 引用 | ✅ | 同上 |
| 运行时无 cv2 加载 | ✅ | `test_no_cv2_loaded_at_runtime` — `sys.modules` 无 cv2 |
| denoiser 参数不被调用 | ✅ | `process()` 方法中无 `self._denoiser` 引用（仅 `__init__` 存储 + `denoiser` property 暴露） |

源文件全部 import：
```
from __future__ import annotations
import logging
from typing import Any, TYPE_CHECKING
from visioncore.pipeline.stages.advanced import (AdvancedStage, StageCapability)
if TYPE_CHECKING: from visioncore.pipeline.context import PipelineContext
```

---

## 3. process() 实现

**结论：PASS**

`denoise_stage.py:198-201`：
```python
def process(self, context: "PipelineContext") -> None:
    self.check_context(context)
    logger.debug("DenoiseStage.process: name=%s frame=%s (passthrough)",
                 self._name,
                 context.frame.frame_id if context.frame is not None else None)
```

| 检查项 | 结果 | 证据 |
|---|---|---|
| 调用 `check_context` | ✅ | `denoise_stage.py:198` |
| 不修改 `context.frame` | ✅ | 无赋值语句；仅读取 `context.frame.frame_id` 用于日志 |
| 处理 `frame=None` | ✅ | `if context.frame is not None else None` 三元表达式 |
| 无隐藏副作用 | ✅ | 仅 `logger.debug`（标准日志，无状态变更） |

---

## 4. 测试覆盖

**结论：PASS**

共 28 项测试 + 1 个 doctest，全部通过。

| 覆盖领域 | 测试数 | 关键测试 |
|---|---|---|
| 包表面 | 3 | 可导入、AdvancedStage 子类、PipelineStage 子类 |
| 构造 | 5 | 默认、with denoiser、None denoiser、自定义 name、name keyword-only |
| 能力描述 | 5 | name/version/required/provided/description |
| 生命周期 | 5 | 可调用、状态转换、幂等 init、safe shutdown、repr |
| **Passthrough** | **5** | **frame identity (`is`)、None frame、字段不变、check_context 强制、多次调用** |
| 管线集成 | 2 | Pipeline 运行、DenoiseStage→DetectorStage 全链 |
| 零依赖 | 3 | AST 无 cv2、AST 无 gui/ai、运行时无 cv2 |

### Passthrough 验证深度

| 验证方式 | 测试 | 证据 |
|---|---|---|
| 对象身份（`is`） | `test_process_preserves_frame_identity` | `ctx.frame is original` — 同一对象引用 |
| 字段值不变 | `test_process_does_not_mutate_frame_fields` | frame_id/timestamp/source_id/image.shape 全部断言 |
| None 通过 | `test_process_with_none_frame` | `ctx.frame is None` 处理前后一致 |
| 多次调用无副作用 | `test_process_multiple_calls` | 三次 process 后 `ctx.frame is f` |
| 契约强制 | `test_process_validates_context_contract` | `del ctx.frame` → `AdvancedStageError` |

---

## 5. 模块依赖

**结论：PASS**

`denoise_stage.py` 的完整依赖链：

```
denoise_stage.py
  ├─ visioncore.pipeline.stages.advanced   (D1 抽象层: AdvancedStage + StageCapability)
  │    ├─ visioncore.pipeline.base         (C1 框架: PipelineStage)
  │    └─ visioncore.pipeline.context      (TYPE_CHECKING only)
  ├─ logging                               (标准库)
  └─ typing                                (标准库)
```

| 检查项 | 结果 |
|---|---|
| 仅依赖 pipeline 基类 | ✅ — 最深依赖为 `visioncore.pipeline.base`（C1） |
| 不依赖具体业务模块 | ✅ — 无 `ai/`、`camera/`、`gui/`、`target_manager`、`eventbus` |
| 不依赖外部库 | ✅ — 无 `cv2`、`torch`、`numpy`、`onnx` 等 |

---

## 6. 最终评分

# **PASS**

所有五项检查全部通过。D2 实现严格遵循规格：继承 AdvancedStage、零外部依赖、process() 纯 passthrough、28 项测试覆盖充分。未发现任何问题。
