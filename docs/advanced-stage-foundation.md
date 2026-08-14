# AdvancedStage 抽象层设计文档 (Milestone D1)

> **版本**：D1 — 2026-08-02
> **范围**：新增 `visioncore/pipeline/stages/advanced/` 子包，建立高级视觉处理阶段的抽象层（去噪、重检测、属性估计等插件预留接口）
> **状态**：已实现，47 项单元测试全部通过 + 2 个模块 doctest 通过，全量回归零破坏（23 个既有测试套件 + 47 项新测试）

---

## 1. 目标与背景

### 1.1 为什么需要 AdvancedStage

C1–C5 里程碑建立了核心链：`Pipeline` 依次驱动 **检测（C3）→ 跟踪（C4）→ 目标管理（C5）** 阶段，产出 `targets` 与 `target_states`。但"核心链之后"的高级处理——**去噪（denoise）、重检测（re-detection）、属性估计（attribute estimation）、融合（fusion）**——没有任何抽象层：要么写死进具体阶段，要么散落在应用层。

D1 的目标是**预留接口**：在核心链之上定义高级处理阶段的统一契约，让未来的高级插件以与普通阶段完全相同的方式挂进 Pipeline，同时声明自己的**数据契约**（读什么、写什么），使管线、配置界面、自动调度器无需运行阶段即可了解其输入输出。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **接口优先** | `AdvancedStage(PipelineStage)` 是 ABC，四方法生命周期与 PipelineStage 完全一致 |
| **能力声明** | `StageCapability`（frozen + kw_only）声明 name/version/description/required_context/provided_context |
| **契约校验** | `check_context(context)` 在 process 前校验 required/provided 字段，违规抛 `AdvancedStageError` |
| **关键字构造可读** | `StageCapability` 所有字段 `kw_only=True`，构造即文档 |
| **目标状态词汇** | `provided_context=["target_states"]` 描述输出 `List[visioncore.state.target_state.TargetState]` 快照列表 |
| **零侵入** | 不修改 `ai/` `camera/` `gui/`，不修改任何既有阶段与业务逻辑 |
| **插件即子类** | 具体插件 = `AdvancedStage` 子类 + capability 声明，与 C3/C4/C5 的接口模式一致 |

### 1.3 与既有里程碑的关系

```
              ┌────────────────────────────────────────────┐
              │        核心链 (C1–C5, 不修改)               │
              │                                            │
  C2 Frame → C3 DetectorStage → detections → C4 TrackerStage
              → tracks → C5 TargetStage → targets + target_states
              └──────────────────────┬─────────────────────┘
                                     ▼
              ┌────────────────────────────────────────────┐
              │   高级处理阶段 (D1+ 插件层, AdvancedStage)   │
              │                                            │
              │  DenoiseStage    → 写 frame / metadata      │
              │  ReDetectStage   → 写 detections            │
              │  AttributeStage  → 写 target_states         │
              │  ...(D2+ 逐个落地)                           │
              └────────────────────────────────────────────┘
```

D1 **只交付抽象层**（基类 + 能力声明 + 测试占位），不交付任何具体插件。具体插件在 D2+ 作为 `AdvancedStage` 子类逐个落地，与 C3/C4/C5"接口先行、实现后置"的模式完全一致。

---

## 2. 文件结构

```
visioncore/pipeline/stages/advanced/
├── __init__.py            # 包导出：AdvancedStage / StageCapability / DummyAdvancedStage / AdvancedStageError
├── stage_capability.py    # (D1) StageCapability frozen kw-only dataclass
└── advanced_stage.py      # (D1) AdvancedStage ABC + AdvancedStageError + DummyAdvancedStage

tests/test_advanced_stage.py      # 47 项单元测试
docs/advanced-stage-foundation.md # 本文档
```

---

## 3. AdvancedStage 的定位与 PipelineStage 的关系

### 3.1 继承关系

`AdvancedStage` **是** `PipelineStage`（`class AdvancedStage(PipelineStage)`）。它继承：

- 名称身份（`name` 属性，默认类名，可自定义）；
- 完整生命周期契约：`initialize / process / shutdown / health_check` **四个抽象方法原样保留**；
- `__repr__` 诊断能力（扩展为附带 capability 信息）。

因此 **Pipeline 无需任何改动**即可调度 AdvancedStage —— 它对管线而言就是一个普通阶段。

### 3.2 差异：两处新增

| 新增 | 类型 | 语义 |
|---|---|---|
| `capability` | 抽象属性 → `StageCapability` | 阶段声明"我是谁、读什么、写什么" |
| `check_context(context)` | 具体方法 → `None` 或抛 `AdvancedStageError` | process 前的数据契约运行时校验 |

### 3.3 接口对比

| 维度 | PipelineStage (C1) | AdvancedStage (D1) |
|---|---|---|
| 生命周期 | initialize / process / shutdown / health_check（抽象） | **完全相同**（抽象） |
| 额外抽象成员 | 无 | `capability` 属性 |
| 具体辅助 | `name` / `__repr__` | + `check_context(context)` |
| 子类必须实现 | 4 个生命周期方法 | 4 个生命周期方法 + `capability` |

---

## 4. StageCapability 字段含义

```python
@dataclass(frozen=True, kw_only=True)
class StageCapability:
    name: str              # 插件标识，如 "denoise"，跨版本稳定
    version: str           # 插件语义化版本，如 "1.2.0"
    description: str       # 人类可读的一行描述
    required_context: list[str]   # 读：PipelineContext 字段名列表
    provided_context: list[str]   # 写：PipelineContext 字段名列表
```

| 字段 | 含义 | 示例 |
|---|---|---|
| `name` | 稳定插件标识；用于日志、诊断、能力查找 | `"denoise"` |
| `version` | 语义化版本；管线可据此发现不兼容升级 | `"1.2.0"` |
| `description` | 一句话说明插件做什么 | `"Bilateral denoising of the frame before detection."` |
| `required_context` | 阶段**必须读取**的 context 字段名；缺失则 `check_context` 抛错 | `["frame"]` |
| `provided_context` | 阶段**写入**的 context 字段名；缺失或非 list 则 `check_context` 抛错 | `["target_states"]` |

### 4.1 为什么 `frozen=True, kw_only=True`

- **frozen**：能力声明是不可变元数据，构造后不可篡改（列表内容按约定只读）；
- **kw_only**：强制按关键字构造，五字段参数自述其意：

```python
# 可读性强：每一行都是一个事实
cap = StageCapability(
    name="re-detect",
    version="0.3.1",
    description="Re-runs detection on low-confidence regions.",
    required_context=["frame", "detections"],
    provided_context=["target_states"],
)
```

### 4.2 `provided_context` 的 "target_states" 词汇

`provided_context` 是字符串列表，可包含 `"target_states"` 以声明"本插件输出目标状态快照列表"。该字段的类型契约：

```
provided_context=["target_states"]  ⇒  输出为 List[visioncore.state.target_state.TargetState]
```

（快照 dataclass，**不是** `visioncore.core.target.TargetState` 生命周期枚举——与 C1 上下文、B2 协议层相同的双 TargetState 注意事项，见 `docs/targetstate-design.md`。）

当前 PipelineContext 全量字段词汇（C1）：`"frame" / "detections" / "tracks" / "targets" / "target_states"`。D2+ 新增 context 字段时可扩充词汇。

---

## 5. 上下文约束（context contract）

### 5.1 契约定义

每个高级阶段通过 `capability` 声明契约：

- **读侧（required_context）**：process 内只允许读取这些字段（外加自由共享的 `context.metadata` 与 `context.timestamp`）；
- **写侧（provided_context）**：process 内只允许写入这些字段，且约定为 **list 型字段**（append / clear+extend）。

### 5.2 `check_context` 运行时校验

```python
def check_context(self, context: PipelineContext) -> None:
    for name in self.capability.required_context:
        assert hasattr(context, name)   # 否则抛 AdvancedStageError
    for name in self.capability.provided_context:
        assert hasattr(context, name)   # 否则抛 AdvancedStageError
        assert isinstance(getattr(context, name), list)  # 否则抛 AdvancedStageError
```

违规时抛出 `AdvancedStageError`，消息包含阶段名与字段名，如：

```
denoise: required_context field 'frame' is missing from context
```

### 5.3 "只读/写声明字段"的验证方式

`check_context` 是**声明式**校验（验证字段存在性），不插桩读写。子类必须自觉只触碰声明字段——该纪律由测试用**限制性 context 模拟**（`_TracingContext`）验证：模拟 context 只暴露声明字段，任何未声明字段的访问抛 `AttributeError`；`process` 若能在该模拟上完整跑通，则证明它从未触碰契约之外的数据。

---

## 6. 插件扩展方式

### 6.1 三步写一个高级插件

```python
from visioncore.pipeline.stages.advanced import (
    AdvancedStage, StageCapability,
)
from visioncore.pipeline.context import PipelineContext

class DenoiseStage(AdvancedStage):

    @property
    def capability(self) -> StageCapability:
        return StageCapability(
            name="denoise",
            version="1.0.0",
            description="Bilateral denoising of the frame before detection.",
            required_context=["frame"],
            provided_context=["frame"],
        )

    def initialize(self) -> None:
        self._ready = True          # 加载模型、预热分配器等

    def process(self, context: PipelineContext) -> None:
        self.check_context(context)                 # 1. 契约校验
        frame = context.frame                       # 2. 只读 required_context
        if frame is not None:
            frame = self._denoise(frame)            # 3. 干活
        context.frame = frame                       # 4. 只写 provided_context

    def shutdown(self) -> None:
        self._ready = False

    def health_check(self) -> bool:
        return getattr(self, "_ready", False)
```

### 6.2 挂进 Pipeline

```python
p = Pipeline()
p.add_stage(DetectorStage(...))       # C3
p.add_stage(TrackerStage(...))        # C4
p.add_stage(TargetStage(...))         # C5
p.add_stage(DenoiseStage())           # D2+ 插件：与普通阶段无异
```

高级插件常作为**后处理**（post-target）阶段排在核心链之后：重检测 / 属性估计读取 `targets` / `target_states`，把富化后的快照写回 `target_states`。

### 6.3 测试占位：DummyAdvancedStage

`DummyAdvancedStage` 是契约的**可观察测试双**（对齐 C1 `DummyStage` 风格）：

- 默认 capability：`required_context=["detections"]`、`provided_context=["target_states"]`；
- `process` 依次执行：`check_context` → 写 `metadata["advanced_trace"]`（marker）→ 读首个 required 字段长度进 `metadata["advanced_reads"]` → 向每个 provided list 追加 `write_value`（默认 marker）；
- 计数器（initialize/process/shutdown）+ `raise_on_process` + `healthy` + `reset()`，便于测试生命周期与异常路径。

---

## 7. 测试覆盖（tests/test_advanced_stage.py，47 项）

| 分组 | 覆盖点 |
|---|---|
| **包表面** | `advanced.__all__` 四名称；`AdvancedStageError ⊂ RuntimeError`；既有 stages 包不受影响 |
| **能力描述** | frozen/kw_only dataclass；关键字构造保留字段值；位置参数构造抛 `TypeError`；`FrozenInstanceError`；`provided_context` 可含 `"target_states"`；列表身份保持 |
| **生命周期调用** | `initialize/process/shutdown/health_check` 可调用性；计数器 (1,1,1)；healthy 转换；`AdvancedStage` 不可实例化；缺 `capability`/缺生命周期方法均实例化失败；四方法接口与 PipelineStage 一致 |
| **能力校验** | `check_context` 通过/缺 required/缺 provided/provided 非 list/错误消息含字段名/首违例优先 |
| **Dummy 行为** | 默认与自定义 capability；trace/read/write 三路可观察行为；`raise_on_process` 后 context 未被触碰；`reset()` 保留 capability |
| **上下文约束** | `_TracingContext` 模拟：process 成功 ⇒ 访问集合 ⊆ 声明字段；缺声明读/写字段 ⇒ `AdvancedStageError` 且无副作用写入 |
| **管线集成** | AdvancedStage 像普通阶段一样在 `Pipeline` 中运行（add_stage → run → shutdown） |
| **零依赖守护** | AST + 运行时双重校验：无 `gui` / `PyQt` / `inference` 引用 |

---

## 8. 零侵入声明

D1 未修改任何既有文件：

- `ai/` `camera/` `gui/`：完全未触碰；
- 既有阶段（detector/tracker/target/event）与 C1 框架：未修改；
- `visioncore/pipeline/stages/__init__.py`：未修改（高级包自带独立 `__init__.py`）。

全量回归：23 个既有测试套件全部通过（其中 4 个 AST 源文件检查套件在 GBK 编码环境下读取 UTF-8 源文件需 `PYTHONUTF8=1`，属既有环境问题，与本次改动无关），新增 47 项测试 + 2 模块 doctest 全部通过。
