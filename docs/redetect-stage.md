# RedetectStage 设计文档 (Milestone D3)

> **版本**：D3 — 2026-08-03
> **范围**：新增 `visioncore/pipeline/stages/redetect_stage.py`，定义目标再检测阶段接口占位
> **状态**：已实现，29 项单元测试全部通过 + 1 个模块 doctest 通过，全量回归零破坏（25 个测试套件）

---

## 1. 目标与背景

### 1.1 为什么需要 RedetectStage

C3 的 DetectorStage 在帧进入时执行一次性检测，C4 的 TrackerStage 将检测关联为轨迹。但对于**低置信度轨迹**、**遮挡后重现的目标**、或**跟踪漂移**的情况，一次性检测往往不够。RedetectStage 是一个**后跟踪阶段**，排在 TrackerStage 之后，用于对已跟踪目标进行重检测或精炼——例如裁剪跟踪区域后重新运行检测器、使用注意力机制聚焦低置信度区域等。

D3 交付的是**架构占位**：定义 RedetectStage 外壳、声明依赖反转接口、确保 Pipeline 可在跟踪后挂载重检测钩子，但不引入任何实际重检测逻辑。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **依赖反转** | 构造时注入 `redetector` 对象（默认 `None`），阶段不关心具体实现 |
| **零重检测逻辑** | D3 是纯 passthrough——`process()` 不修改 `context.tracks` |
| **禁止特定模型** | 无 YOLO/RT-DETR/GroundingDINO/SAM/torch/onnx 导入 |
| **禁止 GUI/ai** | 不触碰 `ai/` `camera/` `gui/` |
| **AdvancedStage 子类** | 遵循 D1 抽象层契约，与 Pipeline 无缝集成 |
| **列表不变性尊重** | passthrough 保持 list 对象身份和元素引用 |

### 1.3 与 D1/D2 的关系

```
C3 DetectorStage → detections → C4 TrackerStage → tracks
                                                     │
                                                     ▼
                                        D3 RedetectStage.process(tracks)
                                                     │
                                           tracks (unchanged / refined)
                                                     │
                                                     ▼
                                        C5 TargetStage → targets + target_states
```

D3 是 D1 `AdvancedStage` 的第二个具体插件，与 D2 `DenoiseStage` 并列。DenoiseStage 是**预处理**（检测前），RedetectStage 是**后处理**（跟踪后）。

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── advanced/              # (D1) 高级阶段抽象层
├── denoise_stage.py       # (D2) 去噪阶段占位
└── redetect_stage.py      # (D3) 重检测阶段占位 ← 新增

tests/test_redetect_stage.py   # 29 项单元测试 ← 新增
docs/redetect-stage.md         # 本文档 ← 新增
```

---

## 3. RedetectStage 接口

### 3.1 类定义

```python
class RedetectStage(AdvancedStage):
    def __init__(self, redetector=None, *, name=None): ...
    @property
    def capability(self) -> StageCapability: ...
    def initialize(self) -> None: ...
    def process(self, context: PipelineContext) -> None: ...
    def shutdown(self) -> None: ...
    def health_check(self) -> bool: ...
    @property
    def redetector(self) -> Any | None: ...
```

### 3.2 构造参数

| 参数 | 类型 | 默认值 | 语义 |
|---|---|---|---|
| `redetector` | `Any \| None` | `None` | 重检测后端实例。`None` = 纯 passthrough（D3 默认） |
| `name` | `str \| None` | `None`（→ `"RedetectStage"`） | 阶段名称，keyword-only |

### 3.3 Capability 声明

```python
StageCapability(
    name="redetect",
    version="0.1.0",
    description="Re-detection stage placeholder: passes context.tracks through unchanged...",
    required_context=["tracks"],
    provided_context=[],          # D3 passthrough 不写回任何字段
)
```

### 3.4 数据流

D3（当前 passthrough）：
```
context.tracks ──> RedetectStage.process ──> context.tracks (unchanged)
```

未来（注入真实 redetector）：
```
context.tracks ──> redetector.redetect(tracks, ...) ──> context.tracks (replaced)
```

---

## 4. 与 D2 的对比

| 维度 | DenoiseStage (D2) | RedetectStage (D3) |
|---|---|---|
| **定位** | 预处理（检测前） | 后处理（跟踪后） |
| **读取** | `context.frame` | `context.tracks` |
| **写入** | 无（passthrough） | 无（passthrough） |
| **包装对象** | `denoiser` | `redetector` |
| **未来写入** | `context.frame`（替换帧） | `context.tracks`（替换轨迹） |

两者遵循相同的架构模式：`AdvancedStage` 子类 + 可选注入后端 + passthrough 占位。

---

## 5. 测试覆盖（tests/test_redetect_stage.py，29 项）

| 分组 | 覆盖点 |
|---|---|
| **包表面** | 可导入、AdvancedStage 子类、PipelineStage 子类 |
| **构造** | 默认、with redetector、None、自定义 name、name keyword-only |
| **能力描述** | name="redetect"、version="0.1.0"、required=["tracks"]、provided=[]、description 含 "placeholder" |
| **生命周期** | 四方法可调用、healthy 转换、幂等 init、safe shutdown、repr |
| **Passthrough** | list 对象身份保持（`is`）、内容不变、空 tracks、Track 字段不变、check_context 强制、多次调用 |
| **管线集成** | Pipeline 运行、Detector→Tracker→Redetect 全链 |
| **零依赖** | AST 无 yolo/detr/sam/torch/onnx、AST 无 gui/ai、运行时无 torch |

---

## 6. 零侵入声明

D3 未修改任何既有文件。`visioncore/pipeline/stages/__init__.py` 未修改。全量回归：25 个测试套件全部通过。
