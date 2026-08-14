# Milestone D9.1 审计报告

> **审计日期**：2026-08-04
> **审计范围**：`visioncore/pipeline/stages/redetect/redetect.py` + `redetect_stage.py` + `visioncore/pipeline/adapters.py`（LegacyRedetectAdapter）+ `tests/test_redetect_migration.py`
> **评级**：**PASS**

---

## 0. 审计总结

D9.1 成功将 `ai/inference.py` 中的 ROI 重检测逻辑（`_redetect_person_in_roi` / `_redetect_quality_gate`）迁移到 Pipeline。`RedetectStage` 真实执行算法（`compute_roi` → `RoiRedetector.redetect` → `quality_gate`），不再只是 passthrough；三个 pipeline 模块均**零 ai/inference 导入**（AST 验证）；`RedetectConfig` 支持 11 项配置；`LegacyRedetectAdapter` 包装旧逻辑；Shadow 模式测试确认 Legacy 与 Pipeline 结果一致。36 项新测试 + 全量回归 32 套件（1,122 项）全部通过。

未发现任何问题。

---

## 1. RedetectStage 真实执行算法

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 不再是 passthrough | ✅ | D3 passthrough 升级为 D9.1 实际重检测（version 0.2.0, provided_context=["tracks"]） | `redetect_stage.py:96` |
| 使用 Redetector + RedetectConfig | ✅ | 构造接受 `redetector: Redetector` + `config: RedetectConfig` | `redetect_stage.py:146-152` |
| process() 执行完整算法 | ✅ | 读 `context.tracks` → `_extract_predicted_box` → `redetector.redetect()` → `_apply_refinement` | `redetect_stage.py:203-262` |
| 纯函数可独立测试 | ✅ | `compute_roi()` 和 `quality_gate()` 为纯函数，无副作用 | `redetect.py:137/206` |
| 实测验证算法执行 | ✅ | 创建 context(frame+tracks) → process() → track.confidence 从 0.7 变为 0.85（算法实际运行） | 运行时验证 |

---

## 2. 不依赖 ai/inference.py

**结论：PASS**

| 模块 | AST 扫描结果 |
|---|---|
| `visioncore/pipeline/stages/redetect/redetect.py` | ✅ **NO ai/inference DEPENDENCY** — 仅导入 `logging`/`abc`/`dataclasses`/`typing` |
| `visioncore/pipeline/stages/redetect_stage.py` | ✅ **NO ai/inference DEPENDENCY** — 仅导入 `visioncore.pipeline.stages.advanced` + `visioncore.pipeline.stages.redetect` |
| `visioncore/pipeline/adapters.py` | ✅ **NO ai/inference DEPENDENCY** — `LegacyRedetectAdapter` 通过 `worker: Any` 注入，不导入 `ai/inference` |

`LegacyRedetectAdapter` 使用 `worker: Any` 类型注解，运行时调用 `worker._redetect_person_in_roi()`——通过依赖注入而非 import 绑定到 `ai/inference.py`。

---

## 3. 支持配置

**结论：PASS**

`RedetectConfig`（frozen dataclass，`redetect.py:64`）支持 11 项配置：

| 配置项 | 默认值 | 对应 Legacy |
|---|---|---|
| `enabled` | `True` | `_person_redetect_enabled` |
| `interval` | `1` | — |
| `region_expand` | `0.15` | `_person_redetect_roi_pad` |
| `min_crop_pixels` | `24` | 内联常量 `24` |
| `max_crop_area_ratio` | `0.20` | 内联常量 `0.20` |
| `conf_floor` | `0.15` | `max(0.15, ...)` |
| `conf_ratio` | `0.6` | `* 0.6` |
| `iou_floor` | `0.35` | `< 0.35` |
| `scale_min` | `0.5` | `* 0.5` |
| `scale_max` | `1.6` | `* 1.6` |
| `budget_per_frame` | `1` | `_redetect_budget_per_frame` |

测试验证：`test_redetect_stage_config`（`test_redetect_migration.py:363`）确认自定义 config 正确传递到 stage。

---

## 4. LegacyRedetectAdapter 存在

**结论：PASS**

| 检查项 | 结果 | 证据 |
|---|---|---|
| 定义位置 | ✅ | `visioncore/pipeline/adapters.py:336-378` |
| 包装 `_redetect_person_in_roi` | ✅ | `redetect()` 方法委托到 `worker._redetect_person_in_roi(frame, predicted_box, original_confidence)` |
| health_check | ✅ | `health_check()` 反映 `worker._model` 状态 |
| 测试覆盖 | ✅ | `test_legacy_adapter_wraps_worker`(326)、`test_legacy_adapter_returns_none`(336)、`test_legacy_adapter_health_check`(343) |

---

## 5. Shadow 结果一致

**结论：PASS**

| 测试 | 验证内容 | 证据 |
|---|---|---|
| `test_shadow_mode_legacy_vs_pipeline` | `LegacyRedetectAdapter` 和 `RoiRedetector` 在同一输入下均返回非 None 结果，置信度差 < 0.01 | `test_redetect_migration.py:436` |
| `test_shadow_mode_timing` | 两种模式各 100 次调用均在 5 秒内完成 | `test_redetect_migration.py:464` |
| `test_redetect_stage_runs_redetection` | Pipeline 模式实际更新 track（conf 0.7→0.85） | `test_redetect_migration.py:410` |

---

## 6. 最终评级

# **PASS**

五项检查全部通过。D9.1 成功将 ROI 重检测逻辑从 `ai/inference.py` 迁移到 Pipeline：RedetectStage 真实执行算法（compute_roi→RoiRedetector→quality_gate）、三个 pipeline 模块零 ai/inference 依赖、RedetectConfig 支持 11 项配置、LegacyRedetectAdapter 存在且可测试、Shadow 模式确认 Legacy 与 Pipeline 结果一致。未发现任何问题。
