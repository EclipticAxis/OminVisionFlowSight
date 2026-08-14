# Processor Migration Report (Milestone D9.3)

> **日期**：2026-08-04
> **范围**：从 `ai/inference.py` 迁移后处理逻辑到 `visioncore/plugin/processors/`
> **状态**：已实现，29 项测试全通过 + 全量回归零破坏（34 个套件）

---

## 1. 分析：源代码定位

### 1.1 Legacy 代码（`ai/inference.py`）

| 方法 | 行号 | 功能 |
|---|---|---|
| `final_dedupe` | L131-141 | IoU 去重（per-label 阈值） |
| `_dedupe_priority` | L144-147 | 排序优先级（track_hits → confidence） |
| `_dict_iou` | L159-160 | 检测框 IoU 计算 |
| `_filter_display_detections` | L1624-1627 | 按 person_enabled 过滤 |
| `_filter_inference_detections` | L1613-1622 | 按 person+features 过滤 + keypoints strip |
| `_attach_gestures` | L1546-1552 | 附加手势标签（模型相关，未迁移） |
| `_attach_head_attributes` | L1554-1595 | 附加头部属性（模型相关，未迁移） |

### 1.2 迁移的逻辑

```
FilterProcessor (from _filter_display_detections + _filter_inference_detections)
  ├─ person_enabled=False → 移除 person 检测
  ├─ skeleton_enabled=False → strip keypoints
  └─ conf_floor → 移除低置信度检测

BBoxProcessor (from final_dedupe + _dict_iou + _dedupe_priority)
  ├─ 按优先级排序（track_hits, confidence）
  ├─ IoU 去重（per-label 阈值）
  └─ 可选 bbox 坐标转换（corner → centre）
```

---

## 2. 文件结构

```
visioncore/plugin/processors/
├── __init__.py          # 包导出
└── processor.py         # ProcessorPlugin ABC, FilterProcessor, BBoxProcessor

tests/test_processor_migration.py  # 29 项测试
docs/processor-migration.md        # 本文档
```

---

## 3. 接口

### 3.1 ProcessorPlugin（ABC）

```python
class ProcessorPlugin(PluginInterface):
    @abstractmethod
    def process(
        self,
        detections: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...
```

继承 `PluginInterface`（name/version/load/run/shutdown），新增 `process()` 方法。

### 3.2 FilterProcessor

```python
class FilterProcessor(ProcessorPlugin):
    def __init__(self, conf_floor=0.0, *, name=None): ...
    def process(detections, context=None) -> list[dict]: ...
```

| context 键 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `person_enabled` | bool | True | False 时移除 person 检测 |
| `skeleton_enabled` | bool | False | False 时 strip person keypoints |

| 参数 | 说明 |
|---|---|
| `conf_floor` | 置信度下限（低于此值的检测被丢弃） |

### 3.3 BBoxProcessor

```python
class BBoxProcessor(ProcessorPlugin):
    def __init__(self, iou_thresholds=None, normalise=False, *, name=None): ...
    def process(detections, context=None) -> list[dict]: ...
```

| 参数 | 说明 |
|---|---|
| `iou_thresholds` | per-label IoU 阈值（默认 person=0.55, rectangle=0.30, hand_gesture=0.35） |
| `normalise` | True 时将 bbox 从 corner form 转为 centre form |

### 3.4 Legacy 代码对应关系

| Legacy 代码 | Processor 等价物 |
|---|---|
| `_filter_display_detections` | `FilterProcessor.process(person_enabled=...)` |
| `_filter_inference_detections` | `FilterProcessor.process(skeleton_enabled=...)` |
| `final_dedupe` | `BBoxProcessor.process()` |
| `_dict_iou` / `_tuple_iou` | `BBoxProcessor._iou()` |
| `_dedupe_priority` | `BBoxProcessor._priority()` |

### 3.5 使用示例

```python
from visioncore.plugin.processors import FilterProcessor, BBoxProcessor

# 组合使用
fp = FilterProcessor(conf_floor=0.3)
bp = BBoxProcessor(iou_thresholds={"person": 0.5})

detections = [...]  # 来自检测器
filtered = fp.process(detections, {"person_enabled": True, "skeleton_enabled": False})
deduped = bp.process(filtered)
```

---

## 4. 设计决策

### 4.1 为什么不迁移 _attach_gestures / _attach_head_attributes？

这两个方法依赖模型（`GestureRecognizer`、`HeadClassificationBackend`），属于模型相关的后处理，不是纯数据变换。D9.3 迁移的是**纯数据后处理**（过滤、去重、坐标转换），模型相关的附加逻辑将在后续里程碑迁移。

### 4.2 context 为什么是 dict 而非 PipelineContext？

ProcessorPlugin 的 `process()` 使用 `dict[str, Any]` 作为 context，而非 `PipelineContext`。这使得处理器可以在 Pipeline 内外自由组合（例如在 InferWorker 的 Legacy 路径中直接调用），最大可组合性。

### 4.3 与 PluginInterface 的关系

ProcessorPlugin 继承 PluginInterface，所以每个处理器也是一个插件：可以注册到 PluginRegistry、通过配置发现、有 name/version/lifecycle。`process()` 是处理器特定的扩展方法。

---

## 5. 测试覆盖（29 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 模块表面 | 4 | ABC 继承关系、process 方法存在 |
| FilterProcessor 合约 | 3 | 默认值、自定义 name、lifecycle |
| FilterProcessor 过滤逻辑 | 7 | 全部通过、person 移除、conf_floor、keypoints strip/keep、空输入、无 context |
| BBoxProcessor 合约 | 2 | 默认值、lifecycle |
| BBoxProcessor 去重 | 7 | 重叠去重、非重叠保留、不同标签、优先级排序、normalise、自定义阈值、空输入 |
| 输入/输出一致性 | 3 | Filter 不变异输入、BBox 不变异输入、链式组合 |
| 零依赖 | 2 | AST 无网络/模型/gui/ai、运行时无 torch |

---

## 6. 零侵入声明

新增 `visioncore/plugin/processors/` 包和 `tests/test_processor_migration.py`。**未修改任何既有文件**。全量回归：34 个测试套件全部通过。
