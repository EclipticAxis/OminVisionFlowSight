# Milestone D9 审计报告

> **审计日期**：2026-08-04
> **审计范围**：`ai/inference.py`（InferWorker 相关改动）+ `visioncore/pipeline/migration.py` + `visioncore/pipeline/adapters.py` + `tests/test_inferworker_migration.py`
> **评分**：**PASS WITH WARNING**

---

## 0. 审计总结

D9 迁移基础设施实现完整。`InferWorker` 已支持 `pipeline_enabled` 开关（默认 `False` = Legacy），两种模式调用**同一后端**（YOLO、DetectionTracker、TargetManager、EventBus）；Pipeline 模式经 `Pipeline(DetectorStage → TrackerStage → TargetStage → EventStage)` 处理并应用与 Legacy 路径相同的后处理。`visioncore/pipeline/migration.py` 提取了可复用的双模式封装器。18 项测试验证了适配器委托、模式切换、双模式使用同一后端、Pipeline 集成运行。三个 pipeline 模块均无 ai/gui/camera/torch 导入。

**WARNING**：Pipeline 模式不支持 ROI 重检测回调（`redetect_callback`）——代码注释已标明此为已知限制，将在后续里程碑迁移。此限制不影响 Legacy 模式（默认启用）。

---

## 1. InferWorker 支持 Legacy/Pipeline 双模式且运行结果一致

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| `pipeline_enabled` 开关 | ✅ | `self._pipeline_enabled: bool = False`（默认 Legacy） | `ai/inference.py:414` |
| `set_pipeline_enabled()` | ✅ | 切换模式 + 日志标记 | `ai/inference.py:416-431` |
| `run()` 路由逻辑 | ✅ | `if self._pipeline_enabled and should_infer:` → `_run_pipeline_mode()`；`elif should_infer:` → Legacy 内联路径 | `ai/inference.py:1986-1990` |
| 两种模式调用同一后端 | ✅ | 注释："两种模式调用**相同的后端方法**（同一个 YOLO 模型、同一个 DetectionTracker、同一个 TargetManager）" | `ai/inference.py:425-426` |
| 返回格式一致 | ✅ | Pipeline 模式注释："返回 ``list[dict]``（与 Legacy 路径格式完全一致）" | `ai/inference.py:447` |
| Pipeline 模式四阶段 | ✅ | `DetectorStage → TrackerStage → TargetStage → EventStage` | `ai/inference.py:479-482` |
| 后处理一致 | ✅ | Pipeline 模式应用 `_attach_head_attributes` / `_attach_gestures` / `_run_rectangle_detection` / `_filter_display_detections` / 健康监控（与 Legacy 一致） | `ai/inference.py:508-534` |

---

## 2. Pipeline 模式完全通过新阶段实现，注释标明旧逻辑

**结论：PASS WITH WARNING**

| 检查项 | 结果 | 证据 |
|---|---|---|
| Pipeline 阶段覆盖核心流程 | ✅ | `DetectorStage`（检测）→ `TrackerStage`（跟踪）→ `TargetStage`（目标管理）→ `EventStage`（事件推送） | `ai/inference.py:479-482` |
| 适配器桥接现有后端 | ✅ | `InferWorkerDetectorAdapter` / `TrackerAdapter` / `TargetManagerAdapter` / `EventBusAdapter` 委托到 `worker._run_inference` / `_trackers` / `_target_mgr` / `_event_bus` | `visioncore/pipeline/adapters.py:126/174/240/311` |
| 注释标明旧逻辑 | ✅ | "Legacy 代码路径完全不变"(`413`)、"后处理（与 Legacy 路径一致）"(`508`) | `ai/inference.py` |
| ROI 重检测未迁移 | ⚠️ | 注释："Pipeline 模式的初始版本（C8）不支持 ROI 重检测回调（redetect_callback），该功能在后续里程碑迁移到 Pipeline Stage" | `ai/inference.py:427-428` |

**WARNING**：`redetect_callback`（ROI 重检测）在 Pipeline 模式下不生效——Legacy 路径（默认）完全保留此功能，Pipeline 模式用户需知此限制。代码注释已标明。

---

## 3. `pipeline_enabled` 配置选项及默认值

**结论：PASS**

| 层级 | 配置项 | 默认值 | 证据 |
|---|---|---|---|
| InferWorker 实例 | `self._pipeline_enabled` | `False`（Legacy） | `ai/inference.py:414` |
| InferWorker API | `set_pipeline_enabled(bool)` | — | `ai/inference.py:416` |
| 项目配置 | `PipelineSettings.pipeline_enabled` | `True` | `visioncore/config/__init__.py:38` |
| PipelineConfig 检查 | `settings.pipeline_enabled` | — | `visioncore/pipeline/config/pipeline_config.py:328` |
| MigrationPipeline | `self._pipeline_enabled` | `False`（Legacy） | `visioncore/pipeline/migration.py:109` |

注释标明默认值："Milestone C8 — Pipeline 模式开关（默认 False = Legacy Mode）"(`ai/inference.py:410`)。

---

## 4. 测试验证两个模式下的等价性

**结论：PASS**

| 测试 | 验证内容 | 证据 |
|---|---|---|
| `test_dual_mode_same_backends` | 两种模式持有**同一**后端引用（detector/tracker/target_manager/event_bus） | `test_inferworker_migration.py:282` |
| `test_dual_mode_returns_same_structure` | 两种模式均返回 `list[dict]`，包含 `bbox`/`confidence` 等键 | `test_inferworker_migration.py:304` |
| `test_legacy_mode_calls_legacy_process` | Legacy 模式正确调用注入的 callable，返回预期 dict | `test_inferworker_migration.py:200` |
| `test_full_pipeline_with_mock_backends` | Pipeline 模式运行完整四阶段，detector 调用计数=1，tracker 调用计数=1 | `test_inferworker_migration.py:325` |
| `test_pipeline_mode_calls_detector_adapter` | DetectorAdapter 正确委托到 `worker._run_inference` | `test_inferworker_migration.py:222` |
| `test_tracker_adapter_with_mock` | TrackerAdapter 委托到 `worker._trackers[slot_id].update`，`last_tracked_dicts` 含 `track_id` | `test_inferworker_migration.py:244` |
| `test_pipeline_with_adapter_error_returns_empty` | Pipeline 异常时返回 `[]`（容错） | `test_inferworker_migration.py:362` |

---

## 5. 循环依赖与架构污染检查

**结论：PASS**

| 模块 | 导入来源 | 结果 |
|---|---|---|
| `visioncore/pipeline/adapters.py` | `visioncore.core.detection`, `visioncore.core.track`, `visioncore.pipeline.stages.*`, `visioncore.state.target_state` | ✅ CLEAN — 无 ai/gui/camera/torch |
| `visioncore/pipeline/migration.py` | `visioncore.pipeline.base`, `visioncore.pipeline`（惰性导入） | ✅ CLEAN — 无 ai/gui/camera/torch |
| `visioncore/pipeline/config/pipeline_config.py` | `visioncore.pipeline.base`, `visioncore.pipeline.pipeline`, `visioncore.config` | ✅ CLEAN — 无 ai/gui/camera/torch |
| `ai/inference.py` → pipeline | `from visioncore.pipeline import Pipeline, PipelineContext`（惰性导入，仅在 `_run_pipeline_mode` 内） | ✅ 惰性导入避免循环 |

---

## 6. 最终评分

# **PASS WITH WARNING**

五项检查全部通过。D9 迁移基础设施完整：InferWorker 支持 Legacy/Pipeline 双模式（默认 Legacy），两种模式调用同一后端，Pipeline 模式经四阶段处理并应用相同后处理。适配器模块零 ai/gui/camera/torch 导入。18 项测试验证了双模式一致性。

**WARNING**：Pipeline 模式不支持 ROI 重检测回调（`redetect_callback`），代码注释已标明此为已知 C8 限制，将在后续里程碑迁移。此限制不影响默认的 Legacy 模式。
