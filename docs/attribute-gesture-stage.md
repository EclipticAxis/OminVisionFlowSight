# AttributeStage / GestureStage 设计文档 (Milestone D4)

> **版本**：D4 — 2026-08-03
> **范围**：新增 `visioncore/pipeline/stages/attribute_gesture_stage.py`，定义目标属性估计和手势识别阶段接口占位
> **状态**：已实现，36 项单元测试全部通过 + 2 个模块 doctest 通过，全量回归零破坏（26 个测试套件）

---

## 1. 目标与背景

### 1.1 为什么需要 AttributeStage 和 GestureStage

C5 的 TargetStage 产出 `target_states`（目标快照），但快照仅包含运动学信息（位置、速度、尺寸）和身份信息。要实现更丰富的目标描述，需要两个后处理阶段：

- **AttributeStage**：为每个快照附加属性标签（如年龄、颜色、服装等），写入 `TargetState.attributes` 字段。
- **GestureStage**：为每个快照附加手势标签（如挥手、指向等），写入 `TargetState.gesture` 字段。

D4 交付的是**架构占位**：定义两个阶段外壳，扩展 TargetState 数据模型，确保 Pipeline 可调度属性/手势钩子，但不引入任何实际算法。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **依赖反转** | 构造时注入 `estimator` / `recogniser`（默认 `None`），阶段不关心具体实现 |
| **零算法逻辑** | D4 写入占位值（`attributes={}` / `gesture=None`） |
| **禁止模型导入** | 无 torch/onnx/cv2/YOLO/DETR/SAM |
| **禁止 GUI/ai** | 不触碰 `ai/` `camera/` `gui/` |
| **TargetState.copy_with** | 利用 frozen dataclass 的 `copy_with()` 派生快照，保持不可变性 |
| **REPLACE 语义** | clear+extend（与 TargetStage 一致），旧快照被替换 |

### 1.3 TargetState 数据模型扩展

D4 向 `TargetState` 冻结数据类新增两个字段：

```python
# visioncore/state/target_state.py — 新增字段
attributes: dict[str, Any] = field(default_factory=dict)  # 属性标签
gesture: str | None = None                                 # 手势标签
```

两个字段均有默认值，**不影响任何现有代码**——所有不传递这两个参数的构造调用自动获得默认值。

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── advanced/                    # (D1) 高级阶段抽象层
├── denoise_stage.py             # (D2) 去噪阶段占位
├── redetect_stage.py            # (D3) 重检测阶段占位
└── attribute_gesture_stage.py   # (D4) 属性+手势阶段占位 ← 新增

visioncore/state/
└── target_state.py              # 新增 attributes + gesture 字段

tests/test_attribute_gesture_stage.py  # 36 项单元测试 ← 新增
docs/attribute-gesture-stage.md        # 本文档 ← 新增
```

---

## 3. 接口

### 3.1 AttributeStage

```python
class AttributeStage(AdvancedStage):
    def __init__(self, estimator=None, *, name=None): ...
    def process(self, context): ...  # copy_with(attributes={}) for each snapshot
```

- **capability**: `name="attribute"`, `required=["target_states"]`, `provided=["target_states"]`
- **process**: 遍历 `context.target_states`，用 `copy_with(attributes={})` 派生新快照，REPLACE 回列表

### 3.2 GestureStage

```python
class GestureStage(AdvancedStage):
    def __init__(self, recogniser=None, *, name=None): ...
    def process(self, context): ...  # copy_with(gesture=None) for each snapshot
```

- **capability**: `name="gesture"`, `required=["target_states"]`, `provided=["target_states"]`
- **process**: 遍历 `context.target_states`，用 `copy_with(gesture=None)` 派生新快照，REPLACE 回列表

### 3.3 数据流

```
context.target_states ──> AttributeStage.process
                                │
                      for each snapshot: snapshot.copy_with(attributes={})
                                │
                                ▼
                      context.target_states (REPLACE)

context.target_states ──> GestureStage.process
                                │
                      for each snapshot: snapshot.copy_with(gesture=None)
                                │
                                ▼
                      context.target_states (REPLACE)
```

### 3.4 链式使用

两个阶段可链式调用（顺序无关）：

```python
attr = AttributeStage(name="attr")
gest = GestureStage(name="gest")
p = Pipeline()
p.add_stage(attr)
p.add_stage(gest)
# 结果: 每个快照同时具有 attributes={} 和 gesture=None
```

---

## 4. 与既有里程碑的关系

```
C5 TargetStage → targets + target_states
                    │
                    ▼
D4 AttributeStage → target_states (每个 snapshot 带 attributes={})
         │
         ▼
D4 GestureStage → target_states (每个 snapshot 带 gesture=None)
         │
         ▼
B2 ProtocolAdapter → 投递富化快照
```

---

## 5. 测试覆盖（tests/test_attribute_gesture_stage.py，36 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 包表面 | 4 | AdvancedStage 子类、PipelineStage 子类 |
| 构造 | 6 | 默认/with estimator/with recogniser/自定义 name |
| 能力描述 | 2 | name/version/required/provided/description |
| 生命周期 | 4 | 可调用、状态转换、幂等 init、safe shutdown |
| AttributeStage.process | 7 | attributes 设置、list 身份、字段保留、空列表、REPLACE、check_context、不碰其他字段 |
| GestureStage.process | 6 | gesture 设置、list 身份、字段保留、空列表、REPLACE、check_context |
| 链式 | 2 | Attribute→Gesture、Gesture→Attribute |
| 管线集成 | 2 | Pipeline 运行 |
| 零依赖 | 3 | AST 无模型、AST 无 gui/ai、运行时无 torch |

---

## 6. 零侵入声明

修改了 `visioncore/state/target_state.py`（新增两个带默认值的字段 + repr 更新），未修改任何其他既有文件。全量回归：26 个测试套件全部通过。
