# Pipeline 审计报告 (Milestone C1)

> **审计目标**：`visioncore/pipeline/`
> **审计日期**：2026-07-25
> **审计范围**：C1 交付的 5 个实现文件 + 1 个测试文件
> **审计方法**：静态导入分析（grep + AST 解析）+ 运行时导入追踪（sys.modules）+ 回归测试
> **最终评分**：**PASS**（9 项审计点全部通过）

---

## 0. 执行摘要

Milestone C1 的 `visioncore/pipeline/` 包是一个**纯调度框架**：运行时仅依赖 Python 标准库（logging / abc / dataclasses / typing）与自身，对 `ai/`、`gui/`、`camera/`、Detector、Tracker、TargetManager **零依赖**。所有对 `visioncore.core.*`（纯数据模型）和 `visioncore.state.target_state.TargetState`（快照）的引用都在 `if TYPE_CHECKING:` 守卫下，仅供类型检查器，运行时不加载。无循环依赖，无生产代码引用该包，现有 VDP 应用运行逻辑完全不受影响。

9 项审计点全部 **PASS**。唯一的非阻塞观察：导入 `visioncore.pipeline` 会经 `visioncore/__init__.py → visioncore.core.frame` 间接加载 numpy——这是 visioncore 顶层包预有的数据模型基础属性（`Frame` 持有 `np.ndarray`），非 C1 引入，亦非视觉/业务耦合，不影响任何审计准则。

---

## 1. 审计范围

### 1.1 受审文件

| 文件 | 行数 | 职责 |
|---|---|---|
| `visioncore/pipeline/__init__.py` | ~85 | 包导出 + 范围/设计说明 |
| `visioncore/pipeline/base.py` | ~260 | `PipelineStage(ABC)` + `StageError` + 四方法生命周期 |
| `visioncore/pipeline/context.py` | ~210 | `PipelineContext` 可变数据黑板 |
| `visioncore/pipeline/pipeline.py` | ~480 | `Pipeline` 顺序执行器 + 状态机 + 上下文管理器 |
| `visioncore/pipeline/stage.py` | ~230 | `DummyStage` 测试用 Stage |
| `tests/test_pipeline.py` | ~870 | 56 项单元测试 |

### 1.2 审计准则（9 项）

1. Pipeline 是否独立
2. 是否依赖 `ai/`
3. 是否依赖 GUI
4. 是否依赖 Detector
5. 是否依赖 Tracker
6. PipelineContext 是否成为统一上下文
7. Stage 是否完全抽象
8. DummyStage 是否覆盖测试
9. 是否产生循环依赖

---

## 2. 审计方法

| 方法 | 工具 | 验证内容 |
|---|---|---|
| 静态导入枚举 | `grep "^\s*(import\|from)"` | 列出包内全部 import 语句 |
| 业务耦合关键词扫描 | `grep -i "Detector\|Tracker\|TargetManager\|InferWorker\|ai\.\|gui\.\|camera\.\|PyQt\|numpy\|cv2\|onnx\|ultralytics\|mediapipe"` | 确认无代码级业务依赖（区分 docstring 散文与真实代码） |
| 反向引用扫描 | `grep "import\s+visioncore\.pipeline\|from\s+visioncore\.pipeline"` | 确认无生产代码引用该包 |
| AST 解析 | `ast.parse` + `if TYPE_CHECKING` 守卫识别 | 权威区分运行时导入与类型检查专用导入 |
| 运行时 `sys.modules` 追踪 | `__import__` + `sys.modules` 快照 | 确认副作用加载范围、无循环导入 |
| 抽象性验证 | 尝试实例化 `PipelineStage()` | 确认 ABC 不可实例化 |
| 数据模型内省 | `dataclasses.fields(PipelineContext)` | 确认 7 字段完整 |
| 回归测试 | 15 既有套件 + 56 新测试 | 确认零破坏 |

---

## 3. 逐项审计发现

### 3.1 审计点 1 — Pipeline 是否独立 ✅ PASS

**准则**：Pipeline 包应是一个自包含的框架，不与具体视觉实现耦合。

**证据**：

(a) 包内全部 import 语句（grep 枚举）按运行时/类型检查分类：

| 文件 | 运行时导入 | TYPE_CHECKING 专用导入 |
|---|---|---|
| `base.py` | `logging`, `abc.ABC/abstractmethod`, `typing.TYPE_CHECKING` | `visioncore.pipeline.context.PipelineContext` |
| `context.py` | `dataclasses`, `typing.TYPE_CHECKING/Any` | `visioncore.core.{detection,frame,target,track}.*`, `visioncore.state.target_state.TargetState` |
| `stage.py` | `logging`, `typing.TYPE_CHECKING/Any` | `visioncore.pipeline.context.PipelineContext` |
| `pipeline.py` | `logging`, `typing.TYPE_CHECKING` | `visioncore.pipeline.context.PipelineContext` |
| `__init__.py` | `visioncore.pipeline.{base,context,pipeline,stage}` | — |

(b) 反向引用扫描：`visioncore.pipeline` 仅被以下文件引用：
- `tests/test_pipeline.py`（测试）
- `visioncore/pipeline/*`（包自身）

**无任何生产代码**（`ai/`、`gui/`、`camera/`、`common/`、`main.py`、`demo.py`、其他 `visioncore/*` 子包）导入 `visioncore.pipeline`。

**结论**：Pipeline 是一个自包含框架，运行时仅依赖标准库与自身。完全独立。

---

### 3.2 审计点 2 — 是否依赖 `ai/` ✅ PASS

**准则**：包内不得 `import ai.*` 或在运行时引用 `ai` 模块。

**证据**：
- grep `ai\.` 在 `visioncore/pipeline/` 内**零匹配**。
- 运行时 `sys.modules` 快照：导入 `visioncore.pipeline` 后，`ai` **不在** `sys.modules` 中。

**结论**：零 `ai/` 依赖。

---

### 3.3 审计点 3 — 是否依赖 GUI ✅ PASS

**准则**：包内不得 `import gui.*`、`PyQt6`、`PyQt5` 或任何 GUI 框架。

**证据**：
- grep `gui\.` / `PyQt` 在 `visioncore/pipeline/` 内**零匹配**。
- 运行时 `sys.modules` 快照：`PyQt6`、`PyQt5` **均不在** `sys.modules` 中。

**结论**：零 GUI 依赖。

---

### 3.4 审计点 4 — 是否依赖 Detector ✅ PASS

**准则**：包内不得引用任何检测器类（`OnnxYoloBackend`、`Yolo26Backend`、`UHDBackend`、`HeadClassifier` 等）或 `ai.detection`。

**证据**：
- grep `Detector` 在 `visioncore/pipeline/` 内的匹配**全部位于 docstring 散文中**（描述"包不包含 detector"），无任何代码级引用。
- 无 `from ai.detection import`、无 `OnnxYoloBackend`、无 `Yolo26Backend` 字样。
- `visioncore.core.detection` 仅在 `context.py` 的 `if TYPE_CHECKING:` 下作为类型注解出现（AST 验证，line 74），运行时不导入。

**结论**：零 Detector 依赖。

---

### 3.5 审计点 5 — 是否依赖 Tracker ✅ PASS

**准则**：包内不得引用 `DetectionTracker`、`RectangleTracker`、`ai.tracker` 或任何跟踪器。

**证据**：
- grep `Tracker` 在 `visioncore/pipeline/` 内的匹配**全部位于 docstring 散文中**（如 pipeline.py:18 "any Track-producing tracker"），无代码级引用。
- 无 `from ai.tracker import`、无 `DetectionTracker`、无 `RectangleTracker` 字样。
- `visioncore.core.track` 仅在 `context.py` 的 `if TYPE_CHECKING:` 下作为类型注解出现（AST 验证，line 77），运行时不导入。

**附加**：原 C1 规约还要求"Pipeline 不包含任何 TargetManager"。grep `TargetManager` 匹配**全部为 docstring 散文**（pipeline.py:19 明确声明"any TargetManager"被排除），无代码引用。`visioncore.target_manager` **不在**运行时 `sys.modules` 中。

**结论**：零 Tracker 依赖（且零 TargetManager 依赖，超出准则要求）。

---

### 3.6 审计点 6 — PipelineContext 是否成为统一上下文 ✅ PASS

**准则**：单一 Context 实例流经所有 Stage；包含全部规定字段；上下游 Stage 共享同一实例。

**证据**：

(a) **字段完整性**（`dataclasses.fields` 内省）——7 字段全部就位：

| 字段 | 类型 | 规约要求 |
|---|---|---|
| `frame` | `Frame \| None` | ✅ Frame |
| `detections` | `list[Detection]` | ✅ Detections |
| `tracks` | `list[Track]` | ✅ Tracks |
| `targets` | `list[Target]` | ✅ Targets |
| `target_states` | `list[TargetState]` | ✅ TargetState |
| `timestamp` | `float` | ✅ Timestamp |
| `metadata` | `dict[str, Any]` | ✅ Metadata |

(b) **统一实例传递**（测试 `test_context_is_same_instance_across_stages`）：3 个 Stage 各自记录 `id(context)`，断言 `len(set(seen)) == 1`——所有 Stage 收到同一对象实例。

(c) **共享黑板语义**（测试 `test_context_downstream_reads_upstream_writes`）：上游 Stage 写入 `context.metadata["produced"]` 与 `context.detections`，下游 Stage 观察到这些改动——证明读写发生在同一实例上。

(d) **run() 返回同一实例**（测试 `test_run_returns_the_same_context_instance`）：`result is ctx` 断言通过。

**结论**：PipelineContext 是真正的统一上下文，7 字段完整，共享黑板语义经测试验证。

---

### 3.7 审计点 7 — Stage 是否完全抽象 ✅ PASS

**准则**：`PipelineStage` 应为 ABC，全部接口方法 `@abstractmethod`，不可直接实例化，子类必须实现全部方法。

**证据**：

(a) **抽象方法数量**（grep `@abstractmethod`）：4 个，覆盖规约要求的全部接口：

| 行号 | 方法 |
|---|---|
| base.py:182 | `initialize(self) -> None` |
| base.py:196 | `process(self, context: PipelineContext) -> None` |
| base.py:220 | `shutdown(self) -> None` |
| base.py:236 | `health_check(self) -> bool` |

(b) **不可实例化**（运行时验证）：
```
PipelineStage()
→ TypeError: Can't instantiate abstract class PipelineStage
  with abstract methods health_check, initialize, process, shutdown
```

(c) **子类必须实现全部方法**（测试 `test_subclass_missing_methods_fails`）：故意省略 3 个方法的 `Incomplete` 子类触发 `TypeError`，无法实例化。

(d) **具体方法不破坏抽象性**：`name` property、`__repr__`、`__init__(name)` 是具体方法（非 abstract），仅为子类提供便利，不影响 ABC 契约。

**结论**：Stage 完全抽象，4 方法全 `@abstractmethod`，ABC 机制经运行时验证生效。

---

### 3.8 审计点 8 — DummyStage 是否覆盖测试 ✅ PASS

**准则**：DummyStage 应支撑四大必测领域（Stage 顺序、Context 传递、异常传播、Stage 生命周期）的测试。

**证据**：

(a) **使用频度**：`DummyStage` 在 `tests/test_pipeline.py` 中出现 **54 次**，是测试的主力 Stage。

(b) **四大领域覆盖映射**：

| 领域 | 测试函数 | DummyStage 角色 |
|---|---|---|
| **Stage 顺序** | `test_stage_order_preserved_in_run` | 3 个 DummyStage，断言 `metadata["order"]` 匹配插入顺序 |
| | `test_stage_order_uses_marker_not_name` | marker 与 name 解耦验证 |
| | `test_shutdown_runs_in_reverse_order` | `OrderTrackingStage(DummyStage)` 子类记录 shutdown 顺序 |
| **Context 传递** | `test_context_is_same_instance_across_stages` | 多 Stage 共享 id(context) |
| | `test_context_downstream_reads_upstream_writes` | 自定义 Stage 验证上下游共享（DummyStage 旁证模式） |
| | `test_run_returns_the_same_context_instance` | run 返回同一实例 |
| **异常传播** | `test_stage_exception_propagates_out_of_run` | `raise_on_process=ValueError` |
| | `test_stage_exception_aborts_subsequent_stages` | 断言后续 Stage `process_count==0` |
| | `test_stage_error_is_a_stage_exception` | `raise_on_process=StageError` |
| | `test_exception_does_not_prevent_context_manager_teardown` | 异常后 shutdown 仍执行 |
| | `test_failed_stage_does_not_stamp_context_captured` | 抛异常 Stage 不 stamp |
| **Stage 生命周期** | `test_lifecycle_calls_each_method_exactly_once` | init/process/shutdown 计数各 1 |
| | `test_lifecycle_initialize_before_process_before_shutdown` | 顺序追踪 |
| | `test_lifecycle_multiple_runs_reuse_initialization` | 一次 init 支持多次 run |
| | `test_context_manager_is_single_use_due_to_terminal_shutdown` | 终态语义 |

(c) **DummyStage 专属测试**（6 项）：defaults、marker 覆盖 name、initialize 置健康、shutdown 置不健康、process stamp、reset 清零。

(d) **配置维度覆盖**：`name`、`marker`、`raise_on_process`、`healthy`、计数器、`reset()` 全部有对应测试。

**结论**：DummyStage 全面支撑四大领域测试，配置维度完整覆盖。

---

### 3.9 审计点 9 — 是否产生循环依赖 ✅ PASS

**准则**：包内无循环导入；包不与外部模块形成导入环。

**证据**：

(a) **包内 DAG**（运行时导入边）：

```
__init__.py ──> base.py
           ──> context.py        (context 无运行时 visioncore 依赖)
           ──> pipeline.py ──> base.py
           ──> stage.py    ──> base.py
```

`base.py`、`context.py` 是叶子（无运行时 visioncore 依赖）；`stage.py`、`pipeline.py` 仅依赖 `base.py`；`__init__.py` 聚合全部。**无环**。

(b) **独立导入测试**：逐个 `__import__` 五个子模块，全部成功无 `ImportError`/`CircularImport`：
```
OK import visioncore.pipeline.base
OK import visioncore.pipeline.context
OK import visioncore.pipeline.stage
OK import visioncore.pipeline.pipeline
OK import visioncore.pipeline
```

(c) **AST 权威验证 — context.py 运行时导入**（区分 TYPE_CHECKING 守卫）：

```
Real runtime imports in context.py (NOT under TYPE_CHECKING):
  line 68: from __future__ import ...
  line 70: from dataclasses import ...
  line 71: from typing import ...

TYPE_CHECKING-only imports (NOT loaded at runtime):
  line 74: from visioncore.core.detection import ...
  line 75: from visioncore.core.frame import ...
  line 76: from visioncore.core.target import ...
  line 77: from visioncore.core.track import ...
  line 78: from visioncore.state.target_state import ...
```

context.py 运行时**零** visioncore 依赖——所有 `visioncore.core.*` 与 `visioncore.state.*` 引用都在 `if TYPE_CHECKING:` 下，配合 `from __future__ import annotations`（PEP 563，注解为字符串），类型检查器专用，运行时不执行。

(d) **外部反向引用**：如审计点 1 所述，无生产代码导入 `visioncore.pipeline`，故不可能形成外部环。

(e) **副作用加载范围**：导入 `visioncore.pipeline` 后 `sys.modules` 中的 `visioncore.*`：
```
visioncore.core / .adapters / .detection / .event / .frame / .target / .track
visioncore.pipeline / .base / .context / .pipeline / .stage
```

`visioncore.state`、`visioncore.target_manager`、`visioncore.eventbus`、`visioncore.protocol` **均未加载**——pipeline 不拉起这些较重子系统。`visioncore.core`（纯数据模型）被加载仅因 `visioncore/__init__.py` 顶层包初始化的预有设计，且 `visioncore.core` 不反向导入 pipeline，无环。

**结论**：无循环依赖。包内 DAG 无环，独立导入全成功，context.py 运行时零 visioncore 依赖（AST 验证），无外部反向引用。

---

## 4. 非阻塞观察（信息性，不影响评分）

### 4.1 numpy 间接加载

**现象**：导入 `visioncore.pipeline` 后，`numpy` 出现在 `sys.modules` 中。

**根因**：`visioncore/__init__.py` 顶层包初始化时执行 `from visioncore.core import Frame, ...`，而 `visioncore/core/frame.py` 定义 `Frame` 时 `image: np.ndarray` 字段需要 `import numpy`。这是 **visioncore 顶层包预有的数据模型基础属性**（B1 之前就存在），**非 C1 引入**。

**影响评估**：
- pipeline 代码本身（base/context/stage/pipeline）**从不** `import numpy`（grep 零匹配于代码行，仅 docstring 提及"no numpy in the schema"）。
- numpy 是数值计算基础库，不属于审计准则所针对的视觉/业务耦合（ai/gui/Detector/Tracker）。
- 不构成循环依赖（`visioncore.core` 不反向导入 pipeline）。
- 不影响现有运行逻辑（VDP 本就依赖 numpy）。

**结论**：信息性观察，不触发任何审计准则的失败或警告。若未来希望 pipeline 包可独立于 numpy 加载，需重构 `visioncore/__init__.py` 使其不 eager-export `Frame`（属于顶层包设计范畴，超出 C1 范围）。

---

## 5. 回归验证

审计期间重跑全量测试套件，确认 C1 零破坏：

```
test_core_models.py            : 31/31 passed
test_target_state.py           : 36/36 passed
test_event_bus.py              : 77/77 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_shadow_integration.py     : 12/12 passed
test_protocol_base.py          : 36/36 passed
test_debug_logger.py           : 20/20 passed
test_json_serializer.py        : 31/31 passed
test_cbor_serializer.py        : 31/31 passed
test_udp_adapter.py            : 23/23 passed
test_state_publisher.py        : 28/28 passed
test_shadow_publisher.py       : 36/36 passed
test_obb_stage2.py             : passed
test_matching.py               : passed
────────────────────────────────────────────────────────────────
test_pipeline.py (C1 新增)     : 56/56 passed
总计                           : 既有全过 + 56 新增，0 回归
```

主应用导入冒烟通过：`visioncore` 顶层包与 `visioncore.pipeline` 均正常导入，`ai/`、`gui/`、`camera/` 未被触碰。

---

## 6. 评分汇总

| # | 审计点 | 评分 | 关键证据 |
|---|---|---|---|
| 1 | Pipeline 是否独立 | **PASS** | 运行时仅依赖 stdlib + 自身；无生产代码引用 |
| 2 | 是否依赖 `ai/` | **PASS** | grep 零匹配；`ai` 不在 sys.modules |
| 3 | 是否依赖 GUI | **PASS** | 无 gui/PyQt 引用；PyQt 不在 sys.modules |
| 4 | 是否依赖 Detector | **PASS** | 无检测器类引用；core.detection 仅 TYPE_CHECKING |
| 5 | 是否依赖 Tracker | **PASS** | 无跟踪器引用；core.track 仅 TYPE_CHECKING；且无 TargetManager |
| 6 | PipelineContext 统一上下文 | **PASS** | 7 字段完整；id() 共享测试通过；上下游读写共享验证 |
| 7 | Stage 完全抽象 | **PASS** | 4 @abstractmethod；实例化触发 TypeError；子类缺方法失败 |
| 8 | DummyStage 覆盖测试 | **PASS** | 54 次使用；四大领域全覆盖；6 项专属测试 |
| 9 | 循环依赖 | **PASS** | 包内 DAG 无环；独立导入全成功；AST 验证 context 零运行时依赖 |

### 最终评分：**PASS**

9 项审计点全部通过。`visioncore/pipeline/` 是一个纯净、独立、完全抽象的调度框架，满足 C1 规约的全部要求，对现有系统零侵入、零回归。

---

## 7. 建议（信息性，非阻塞）

以下建议供后续里程碑参考，不影响 C1 评分：

1. **C2 真实 Stage 接入时复用本审计方法**：当 CaptureStage/DetectionStage/TrackingStage 引入 `ai.*` 依赖时，应对每个 Stage 单独做依赖审计，确保业务耦合 confined 在 Stage 内、不泄漏到 `pipeline.py`/`context.py`/`base.py`。
2. **考虑 `visioncore/__init__.py` 懒加载**：当前顶层包 eager-export `Frame` 导致 numpy 成为任何 `visioncore.*` 导入的副作用。若希望 pipeline 包可纯独立加载，可让 `visioncore/__init__.py` 改为懒加载或移除 eager export。此为顶层包设计议题，超出 C1 范围。
3. **保留 AST 守护测试**：建议在 C2 中新增一个 AST 测试，断言 `pipeline.py`/`context.py`/`base.py` 运行时导入集中不含 `ai.*`/`gui.*`/`camera.*`，防止未来回归（类似 protocol 层的 `test_type_annotation_convention_no_top_level_import`）。

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/__init__.py` | 包导出（5 名）+ 范围说明 |
| `visioncore/pipeline/base.py` | `PipelineStage(ABC)` + `StageError` |
| `visioncore/pipeline/context.py` | `PipelineContext`（7 字段，TYPE_CHECKING-only 依赖） |
| `visioncore/pipeline/pipeline.py` | `Pipeline` 执行器 + 状态机 |
| `visioncore/pipeline/stage.py` | `DummyStage` |
| `tests/test_pipeline.py` | 56 项测试（含四大领域 + 框架契约） |
| `docs/pipeline-foundation.md` | 设计文档 |
| `docs/pipeline-audit-report.md` | 本审计报告 |
