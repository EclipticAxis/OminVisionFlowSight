# RectangleStage / FilterStage / HealthStage 设计文档 (Milestone D5)

> **版本**：D5 — 2026-08-04
> **范围**：新增 `visioncore/pipeline/stages/rectangle_filter_stage.py`，定义目标框归一化、目标滤波、健康检测阶段接口占位
> **状态**：已实现，51 项单元测试全部通过 + 1 个模块 doctest 通过，全量回归零破坏（27 个测试套件）

---

## 1. 目标与背景

### 1.1 为什么需要 RectangleStage / FilterStage / HealthStage

C5 的 TargetStage 产出 `target_states`（目标快照），快照中的框以中心+尺寸表示（`cx`/`cy`/`width`/`height`），且没有任何质量筛选或健康状态概念。D5 引入三个后处理扩展点：

- **RectangleStage**：把中心+尺寸框转换为角点表示 `rect=(x1,y1,x2,y2)`，写入 `TargetState.rect` 字段——下游算法（如 IoU 计算、绘制、ROI 判断）通常需要角点表示。
- **FilterStage**：目标滤波扩展点——通过条件（如占位 `if target.confidence<0`）评估 `context.target_states`，为后续滤除噪声目标提供挂载位置。D5 的占位谓词**不实际删除任何目标**。
- **HealthStage**：健康检测扩展点——为每个快照写入示例健康值 `health=1.0`，为后续检测目标健康状态（如跟踪质量退化）提供挂载位置。

D5 交付的是**架构占位**：定义三个阶段外壳，扩展 TargetState 数据模型，确保 Pipeline 可调度框归一化/滤波/健康检查钩子，但不引入任何实际算法。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **依赖反转** | 构造时注入 `rect_builder` / `filterer` / `health_monitor`（默认 `None`），阶段不关心具体实现 |
| **零算法逻辑** | RectangleStage 仅做纯算术换算（`x1=cx-width/2` 等）；FilterStage 仅评估谓词不删除；HealthStage 写入占位值 `health=1.0` |
| **禁止图像处理调用** | 不调用任何具体图像处理函数 |
| **禁止模型导入** | 无 torch/onnx/cv2/YOLO/DETR/SAM |
| **禁止 GUI/ai** | 不触碰 `ai/` `camera/` `gui/` |
| **TargetState.copy_with** | 利用 frozen dataclass 的 `copy_with()` 派生快照，保持不可变性 |
| **REPLACE 语义** | clear+extend（与 TargetStage 一致），旧快照被替换（FilterStage 除外——它不改动列表） |

### 1.3 TargetState 数据模型扩展

D5 向 `TargetState` 冻结数据类新增两个字段：

```python
# visioncore/state/target_state.py — 新增字段
rect: tuple[float, float, float, float] | None = None  # 角点框 (x1,y1,x2,y2)
health: float | None = None                            # 健康值 [0.0, 1.0]
```

两个字段均有默认值，**不影响任何现有代码**——所有不传递这两个参数的构造调用自动获得默认值。`to_dict`/`from_dict` 往返完整保留 tuple 与 float（`dataclasses.asdict` 递归处理 tuple）。

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── advanced/                    # (D1) 高级阶段抽象层
├── denoise_stage.py             # (D2) 去噪阶段占位
├── redetect_stage.py            # (D3) 重检测阶段占位
├── attribute_gesture_stage.py   # (D4) 属性+手势阶段占位
└── rectangle_filter_stage.py    # (D5) 矩形/滤波/健康阶段占位 ← 新增

visioncore/state/
└── target_state.py              # 新增 rect + health 字段（含 repr 更新）

tests/test_rectangle_filter_stage.py  # 51 项单元测试 ← 新增
docs/rectangle-filter-stage.md        # 本文档 ← 新增
```

---

## 3. 接口

### 3.1 RectangleStage

```python
class RectangleStage(AdvancedStage):
    def __init__(self, rect_builder=None, *, name=None): ...
    def process(self, context): ...  # copy_with(rect=(x1,y1,x2,y2)) for each snapshot
```

- **capability**: `name="rectangle"`, `required=["target_states"]`, `provided=["target_states"]`
- **process**: 遍历 `context.target_states`，用 `_derive_rect()` 把 `cx/cy/width/height` 换算为角点框，`copy_with(rect=...)` 派生新快照，REPLACE 回列表
- **换算公式**（纯算术，非图像处理）：
  ```
  x1 = cx - width/2    y1 = cy - height/2
  x2 = cx + width/2    y2 = cy + height/2
  ```

### 3.2 FilterStage

```python
class FilterStage(AdvancedStage):
    def __init__(self, filterer=None, *, name=None): ...
    def process(self, context): ...  # 评估谓词，但不删除任何目标
```

- **capability**: `name="filter"`, `required=["target_states"]`, `provided=["target_states"]`
- **process**: 评估 D5 占位谓词（丢弃条件 `target.confidence < 0`）并记录日志；由于 `confidence` 定义域为 `[0.0, 1.0]`，该条件对合法数据永不触发，且 D5 外壳**故意不删除任何目标**——列表内容与元素身份均保持不变（未来注入 filterer 后在同一评估点执行真正的抑制）

### 3.3 HealthStage

```python
class HealthStage(AdvancedStage):
    def __init__(self, health_monitor=None, *, name=None): ...
    def process(self, context): ...  # copy_with(health=1.0) for each snapshot
```

- **capability**: `name="health"`, `required=["target_states"]`, `provided=["target_states"]`
- **process**: 遍历 `context.target_states`，用 `copy_with(health=1.0)` 派生新快照，REPLACE 回列表

### 3.4 数据流

```
context.target_states ──> RectangleStage.process
                                │
                      for each snapshot:
                          x1,y1 = cx∓w/2, cy∓h/2 → rect=(x1,y1,x2,y2)
                          snapshot.copy_with(rect=...)
                                │
                                ▼
                      context.target_states (REPLACE)

context.target_states ──> FilterStage.process
                                │
                      谓词评估 (confidence < 0)
                      ── D5 占位：不删除任何目标 ──

context.target_states ──> HealthStage.process
                                │
                      for each snapshot: snapshot.copy_with(health=1.0)
                                │
                                ▼
                      context.target_states (REPLACE)
```

### 3.5 链式使用

三个阶段可链式调用：

```python
rect = RectangleStage(name="rect")
filt = FilterStage(name="filter")
health = HealthStage(name="health")
p = Pipeline()
p.add_stage(rect)
p.add_stage(filt)
p.add_stage(health)
# 结果: 每个快照同时具有 rect=(x1,y1,x2,y2) 和 health=1.0，且无目标被删除
```

---

## 4. 与既有里程碑的关系

```
C5 TargetStage → targets + target_states (中心+尺寸框)
                    │
                    ▼
D5 RectangleStage → target_states (每个 snapshot 带 rect 角点框)
         │
         ▼
D5 FilterStage → target_states (谓词评估点，暂不删除)
         │
         ▼
D5 HealthStage → target_states (每个 snapshot 带 health=1.0)
         │
         ▼
B2 ProtocolAdapter → 投递富化快照
```

---

## 5. 测试覆盖（tests/test_rectangle_filter_stage.py，51 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 包表面 | 6 | AdvancedStage 子类、PipelineStage 子类 |
| 构造 | 9 | 默认/with 后端/自定义 name（3 阶段 × 3） |
| 能力描述 | 3 | name/version/required/provided/description |
| 生命周期 | 3 | 可调用、状态转换 |
| RectangleStage.process | 9 | rect 字段存在且类型正确（tuple×4、float）、换算值正确（isclose）、逐快照差异、list 身份、字段保留、空列表、REPLACE、check_context、不碰其他字段 |
| FilterStage.process | 5 | 可调用且不删除（列表+元素身份不变）、负 confidence 快照仍保留、空列表、check_context、不碰其他字段 |
| HealthStage.process | 7 | health 数值字段（float==1.0）、list 身份、字段保留、空列表、REPLACE、check_context、不碰其他字段 |
| 后端不被调用 | 3 | _SpyBackend 断言过程期从未调用 |
| 链式 | 2 | Rectangle→Filter→Health、Health→Rectangle |
| 管线集成 | 1 | Pipeline 运行后 rect+health 就位 |
| 零依赖 | 3 | AST 无模型、AST 无 gui/ai、运行时无 torch |

另外 `tests/test_target_state.py` 从 36 项增至 38 项：新增 `rect`/`health` 默认值为 None 与 tuple/float 往返测试；字段计数 16→18、`to_dict` 键集合更新。

---

## 6. 零侵入声明

修改了 `visioncore/state/target_state.py`（新增两个带默认值的字段 + repr 更新）和 `tests/test_target_state.py`（字段计数/键集合同步），未修改任何其他既有文件。全量回归：27 个测试套件全部通过。
