# Pipeline Configuration 设计文档 (Milestone D8)

> **版本**：D8 — 2026-08-04
> **范围**：新增 `visioncore/pipeline/config/pipeline_config.py`，支持通过配置文件动态选择和顺序组合流水线阶段与插件
> **状态**：已实现，27 项单元测试全部通过 + 3 个模块 doctest 通过，全量回归零破坏（30 个测试套件）

---

## 1. 目标与背景

### 1.1 为什么需要流水线配置

C1-D7 的流水线完全通过代码组装（`add_stage()` 硬编码调用）。D8 引入**配置驱动的流水线组装**：

- **PipelineConfig**：读取 JSON / YAML 配置文件，按配置中的阶段列表顺序调用 `add_stage()` 构建流水线。
- **BUILTIN_STAGES 注册表**：惰性加载的阶段类型名→类映射，涵盖 DummyStage 和全部 D2-D5 高级阶段。
- **Pipeline.load_config()**：便捷类方法，一行代码从配置文件构建流水线。
- **visioncore.config**：项目级配置接口（`pipeline_enabled` 标志 + `pipeline_config_path` 路径参数）。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **不修改既有阶段代码** | 全部阶段类型通过惰性 import 注册，不改变任何阶段模块 |
| **渐进式注册** | BUILTIN_STAGES 仅导入存在的模块（try/except），缺失的高级阶段静默跳过 |
| **PluginRegistry 兼容** | `extra_stages` 和 `plugin_registry` 参数支持插件驱动的阶段解析 |
| **pipeline_enabled 标志** | `settings.pipeline_enabled = False` 时 `build_from_config()` 返回空流水线 |
| **JSON 内置 / YAML 可选** | JSON 无外部依赖；YAML 需要 `pyyaml`（已安装），失败时报明确错误 |
| **配置验证** | 缺少 type、参数类型错误、未知阶段类型均抛 `PipelineConfigError` |
| **禁止网络/协议** | 无 socket/requests/ROS2/MAVLink |

### 1.3 配置文件格式

JSON 示例（`config/visioncore_pipeline.json`）：

```json
{
  "stages": [
    {"type": "DummyStage",  "params": {"name": "capture"}},
    {"type": "DenoiseStage","params": {"name": "denoise"}},
    {"type": "DummyStage",  "params": {"name": "detect"}}
  ]
}
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `stages` | `list[dict]` | 是 | 按顺序执行的阶段列表 |
| `stages[i].type` | `str` | 是 | 阶段类型名（对应 BUILTIN_STAGES 键或 PluginRegistry 插件名） |
| `stages[i].params` | `dict` | 否 | 阶段构造参数（默认 `{}`） |

---

## 2. 文件结构

```
visioncore/
├── config/                      # (D8) 项目配置
│   └── __init__.py              # PipelineSettings + settings 单例
├── pipeline/
│   ├── config/                  # (D8) 流水线配置
│   │   ├── __init__.py          # 包导出
│   │   └── pipeline_config.py   # PipelineConfig + BUILTIN_STAGES
│   └── pipeline.py              # Pipeline.load_config() 类方法 ← 更新
└── ...

config/
└── visioncore_pipeline.json     # (D8) 示例配置文件

tests/test_pipeline_config.py    # 27 项单元测试 ← 新增
docs/pipeline-config.md          # 本文档 ← 新增
```

---

## 3. 接口

### 3.1 PipelineConfig

```python
class PipelineConfig:
    @staticmethod
    def load_file(path: str) -> dict[str, Any]: ...
    @staticmethod
    def resolve_stage(
        stage_type: str,
        extra_stages: dict[str, type[PipelineStage]] | None = None,
        plugin_registry: PluginRegistry | None = None,
    ) -> type[PipelineStage]: ...
    @classmethod
    def build_from_config(cls, config, extra_stages=None, plugin_registry=None) -> Pipeline: ...
    @classmethod
    def build_from_file(cls, path, extra_stages=None, plugin_registry=None) -> Pipeline: ...
```

| 方法 | 行为 |
|---|---|
| `load_file(path)` | 读取 JSON / YAML 文件并返回解析后的 dict；未知格式抛 `PipelineConfigError` |
| `resolve_stage(name)` | 按 `extra_stages → BUILTIN_STAGES → plugin_registry` 顺序查找阶段类；未找到抛 `PipelineConfigError` |
| `build_from_config(cfg)` | 解析 config dict，逐个实例化阶段并 `add_stage()` 到新 Pipeline；`pipeline_enabled=False` 时返回空流水线 |
| `build_from_file(path)` | 便捷组合：`load_file` + `build_from_config` |

### 3.2 BUILTIN_STAGES（惰性注册表）

```python
BUILTIN_STAGES = {
    "DummyStage": DummyStage,
    "DenoiseStage": DenoiseStage,      # D2 — optional
    "RedetectStage": RedetectStage,    # D3 — optional
    "AttributeStage": AttributeStage,  # D4 — optional
    "GestureStage": GestureStage,      # D4 — optional
    "RectangleStage": RectangleStage,  # D5 — optional
    "FilterStage": FilterStage,        # D5 — optional
    "HealthStage": HealthStage,        # D5 — optional
    "DetectorStage": DetectorStage,    # C3 — optional
    "TrackerStage": TrackerStage,      # C4 — optional
    "TargetStage": TargetStage,        # C5 — optional
    "EventStage": EventStage,          # C6 — optional
}
```

可选阶段通过 `try/except ImportError` 惰性加载，不影响缺失模块。

### 3.3 Pipeline.load_config()

```python
class Pipeline:
    @classmethod
    def load_config(
        cls, path, extra_stages=None, plugin_registry=None,
    ) -> "Pipeline": ...
```

便捷类方法，委托给 `PipelineConfig.build_from_file()`。返回 NEW 状态的流水线（未初始化）。

### 3.4 visioncore.config

```python
@dataclass
class PipelineSettings:
    pipeline_enabled: bool = True
    pipeline_config_path: str | None = None

settings: PipelineSettings = PipelineSettings()
```

`build_from_config()` 在构建前检查 `settings.pipeline_enabled`；为 `False` 时返回空流水线。

### 3.5 使用示例

```python
from visioncore.pipeline.config import PipelineConfig
from visioncore.pipeline import Pipeline
from visioncore.pipeline.context import PipelineContext

# 方式一：PipelineConfig 直接使用
pipeline = PipelineConfig.build_from_file("config/visioncore_pipeline.json")

# 方式二：Pipeline.load_config 便捷方法
pipeline = Pipeline.load_config("config/visioncore_pipeline.json")

with pipeline:
    ctx = pipeline.run(PipelineContext.empty())
print(ctx.metadata["order"])  # ['capture', 'denoise', 'detect']
```

---

## 4. 与既有里程碑的关系

```
C1 Pipeline + PipelineStage (流水线基础)
                  │
C2-C6 各阶段实现 (DummyStage, DenoiseStage, ...)
                  │
D7 PluginRegistry (插件发现)
                  │
                  ▼
D8 PipelineConfig ──── 配置驱动组装
       │
       ├─ load_file()      ── 读取 JSON / YAML
       ├─ resolve_stage()  ── 解析阶段类型（内置 → 额外 → 插件）
       ├─ build_from_config() → Pipeline
       └─ Pipeline.load_config() ── 便捷入口
```

---

## 5. 测试覆盖（tests/test_pipeline_config.py，27 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 模块表面 | 3 | PipelineConfigError 继承、BUILTIN_STAGES 包含 DummyStage、含高级阶段 |
| JSON 构建 | 5 | 两阶段构建、无参数默认、空列表、NEW 状态、可初始化运行 |
| 阶段解析 | 4 | 解析内置、额外阶段、额外覆盖内置、未知抛异常 |
| Pipeline.load_config | 2 | 便捷方法等效、每次返回新实例 |
| pipeline_enabled | 1 | False 返回空流水线 |
| 配置验证 | 8 | 文件缺失、JSON 非法、格式不支持、缺少 stages 键、stages 非列表、entry 非 dict、缺少 type、参数非 dict、未知类型、参数无效 |
| 零依赖 | 2 | AST 无网络/模型/gui/ai、运行时无 torch |

---

## 6. 零侵入声明

新增 `visioncore/pipeline/config/` 包、`visioncore/config/` 包、`config/visioncore_pipeline.json` 示例配置、`tests/test_pipeline_config.py`。更新了 `visioncore/pipeline/pipeline.py`（新增 `Pipeline.load_config()` 类方法 + `TYPE_CHECKING` 导入）。未修改任何其他既有文件。全量回归：30 个测试套件全部通过。
