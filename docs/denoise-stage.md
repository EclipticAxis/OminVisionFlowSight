# DenoiseStage 设计文档 (Milestone D2)

> **版本**：D2 — 2026-08-03
> **范围**：新增 `visioncore/pipeline/stages/denoise_stage.py`，定义去噪阶段接口占位
> **状态**：已实现，28 项单元测试全部通过 + 1 个模块 doctest 通过，全量回归零破坏（24 个测试套件）

---

## 1. 目标与背景

### 1.1 为什么需要 DenoiseStage

C3–C5 建立了核心链：检测 → 跟踪 → 目标管理。但**原始帧质量**直接影响检测准确率——噪声、低光照、压缩伪影都会降低检测置信度。DenoiseStage 是 Pipeline 的**第一个可插拔处理步骤**，排在检测之前，负责在帧进入检测器前进行预处理。

D2 交付的是**架构占位**：定义 DenoiseStage 外壳、声明依赖反转接口、确保 Pipeline 可调度高级阶段，但不引入任何实际去噪逻辑。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **依赖反转** | 构造时注入 `denoiser` 对象（默认 `None`），阶段不关心具体实现 |
| **零去噪逻辑** | D2 是纯 passthrough——`process()` 不修改 `context.frame` |
| **禁止 OpenCV** | 无 `cv2` 导入；AST + 运行时双重守护 |
| **禁止 GUI/ai** | 不触碰 `ai/` `camera/` `gui/` |
| **AdvancedStage 子类** | 遵循 D1 抽象层契约，与 Pipeline 无缝集成 |
| **帧不变性尊重** | `Frame` 是 frozen dataclass，passthrough 保持对象身份 |

### 1.3 与 D1 的关系

DenoiseStage 是 D1 `AdvancedStage` 的**第一个具体插件**，验证了高级阶段抽象层的可用性：

```
D1 抽象层: AdvancedStage + StageCapability + check_context
                │
                ▼
D2 首个插件: DenoiseStage(AdvancedStage)
                │
                ▼
未来: DenoiseStage + 真实 Denoiser 后端 (bilateral/NLM/learned)
```

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── __init__.py            # 未修改（DenoiseStage 不加入 stages 包导出）
├── detector_stage.py      # (C3)
├── tracker_stage.py       # (C4)
├── target_stage.py        # (C5)
├── event_stage.py         # (C6)
├── advanced/              # (D1) 高级阶段抽象层
│   ├── __init__.py
│   ├── advanced_stage.py
│   └── stage_capability.py
└── denoise_stage.py       # (D2) 去噪阶段占位 ← 新增

tests/test_denoise_stage.py    # 28 项单元测试 ← 新增
docs/denoise-stage.md          # 本文档 ← 新增
```

---

## 3. DenoiseStage 接口

### 3.1 类定义

```python
class DenoiseStage(AdvancedStage):
    def __init__(self, denoiser=None, *, name=None): ...
    @property
    def capability(self) -> StageCapability: ...
    def initialize(self) -> None: ...
    def process(self, context: PipelineContext) -> None: ...
    def shutdown(self) -> None: ...
    def health_check(self) -> bool: ...
    @property
    def denoiser(self) -> Any | None: ...
```

### 3.2 构造参数

| 参数 | 类型 | 默认值 | 语义 |
|---|---|---|---|
| `denoiser` | `Any \| None` | `None` | 去噪后端实例。`None` = 纯 passthrough（D2 默认） |
| `name` | `str \| None` | `None`（→ `"DenoiseStage"`） | 阶段名称，keyword-only |

### 3.3 Capability 声明

```python
StageCapability(
    name="denoise",
    version="0.1.0",
    description="Denoising stage placeholder: passes context.frame through unchanged...",
    required_context=["frame"],
    provided_context=[],          # D2 passthrough 不写回任何字段
)
```

- **required_context**: `["frame"]`——必须有帧才能去噪
- **provided_context**: `[]`——D2 passthrough 不修改 context（未来注入真实 denoiser 后改为 `["frame"]`）

### 3.4 数据流

D2（当前 passthrough）：
```
context.frame ──> DenoiseStage.process ──> context.frame (unchanged)
```

未来（注入真实 denoiser）：
```
context.frame ──> denoiser.denoise(frame) ──> context.frame (replaced)
```

---

## 4. 依赖反转

DenoiseStage 遵循与 DetectorStage / TrackerStage / TargetStage 相同的依赖反转模式：

| 阶段 | 包装的接口 | D2 测试双 | 真实后端（未来） |
|---|---|---|---|
| DetectorStage (C3) | `Detector` ABC | `DummyDetector` | YOLO/RT-DETR/... |
| TrackerStage (C4) | `Tracker` ABC | `DummyTracker` | ByteTrack/BoTSORT/... |
| TargetStage (C5) | `TargetManager` ABC | `DummyTargetManager` | 具体 TargetManager |
| **DenoiseStage (D2)** | `denoiser` (Any) | `None` | Bilateral/NLM/learned... |

D2 的 denoiser 参数类型为 `Any`（未来里程碑将定义 `Denoiser` ABC 并收紧类型注解）。当前 `None` 是合法的"无 denoiser"值。

---

## 5. 测试覆盖（tests/test_denoise_stage.py，28 项）

| 分组 | 覆盖点 |
|---|---|
| **包表面** | `DenoiseStage` 可从 `denoise_stage` 导入；是 `AdvancedStage` / `PipelineStage` 子类 |
| **构造** | 默认（denoiser=None, name="DenoiseStage"）、with denoiser、explicit None、自定义 name、name keyword-only |
| **能力描述** | name="denoise"、version="0.1.0"、required=["frame"]、provided=[]、description 含 "placeholder" |
| **生命周期** | 四方法可调用、healthy False→True→False、initialize 幂等、shutdown 不抛异常、repr 含 health |
| **Passthrough** | Frame 对象身份保持（`is`）、None frame 正常通过、帧字段不变、check_context 强制、多次调用无副作用 |
| **管线集成** | Pipeline 内运行、DenoiseStage→DetectorStage 全链 |
| **零依赖** | AST 无 cv2/gui/ai 导入、运行时 cv2 未加载 |

---

## 6. 零侵入声明

D2 未修改任何既有文件：

- `ai/` `camera/` `gui/`：完全未触碰
- 既有阶段与 D1 抽象层：未修改
- `visioncore/pipeline/stages/__init__.py`：未修改（DenoiseStage 不加入包导出，保持 D1 的 advanced 子包独立模式）

全量回归：24 个测试套件全部通过（含 D1 的 47 项高级阶段测试）。
