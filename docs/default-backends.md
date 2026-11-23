# 默认行为变更：Rust 跟踪后端 auto 默认 + ReID 默认启用

> **版本**：1.0 — 2026-11-22
> **范围**：`ai/tracker.py`（`DetectionTracker.__new__` auto 语义）、`ai/inference.py`（`InferWorker` 后端模式透传）、`gui/settings_dialog.py`（跟踪后端配置行 + ReID 三态默认加载）、`gui/main_window.py`（配置链接通）、`tests/test_default_backends.py`（新增 16 项）
> **状态**：已合入 main（PR #9 → `7f78651`）；新增 16 项测试，全套件 1366 项全绿。1.1 已合入 main（PR #15 → `e9c01b2`，补 ReID 配置排查指引，见 §3）

---

## 1. 目标与背景

两项"做了但没默认开"的能力在发布场景中等于不存在：

1. **Rust 跟踪内核默认不上场**：omintrack 实测平均 ~37x 提速（17x~60x，见
   [rust-tracker-core](rust-tracker-core.md) §5.3），且 1.1 已落地外观数组 FFI
   零拷贝优化；但缺省后端仍是 "python"，未构建 Rust 的用户拿不到性能，
   构建了的用户也不知道要切换。FFI 优化后 "Rust 可用但默认 Python"
   的默认行为已无正当理由。
2. **ReID 接入后仍默认关闭**：OSNet 真实模型已随仓库分发
   （`models/osnet_x0_25_256x128_msmt17.onnx`，见
   [reid-osnet-backend](reid-osnet-backend.md)），mock 已有双側止损门控；
   但 `ai/reid_model_path` 键缺省时按 "关闭"（""）处理，生产默认永远走
   无外观加权路径。

本次变更：跟踪后端缺省 "python" → "auto"（omintrack 可导入即 Rust）；
ReID 键缺省时默认启用随仓库模型，**尊重用户主动关闭**。

## 2. 契约表

### 2.1 跟踪后端解析（`ai/tracker.py::DetectionTracker.__new__`）

| 项 | 契约 |
|---|---|
| 解析优先级 | 显式 `backend` 参数 > 环境变量 `OMINTRACK_BACKEND` > 缺省 `"auto"` |
| auto 语义 | 构造时尝试 `import omintrack`：成功 → 按 `"rust"` 走；`except (ImportError, OSError)` → 回退 `"python"`，debug 级日志（含堆栈）。`.pyd` DLL 加载失败属 `OSError`，必须捕获 |
| 显式 `"rust"` | 保持现状：扩展不可用时照常抛错，**不静默回退** |
| 显式 `"python"` | 恒为 Python 实现，忽略 omintrack 可用性 |
| 环境变量 `"auto"` | 与缺省完全一致 |
| 数值差异 | track ID / 状态机 / predicted_only / label 逐帧精确一致；坐标数值 1e-6 容差内（330 帧递推实测最大偏差 ~1e-9，源自 nalgebra 与 NumPy/LAPACK 内部路径差异，见 rust-tracker-core §3.1） |
| 拦截范围 | 仅精确类 `DetectionTracker`（子类不被拦截），与 v1.0 一致 |

### 2.2 GUI 配置链（GUI → main_window → inference → tracker）

| 项 | 契约 |
|---|---|
| QSettings 键 | `ai/tracker_backend`，取值 `"auto"` / `"python"` / `"rust"`，推荐默认 `"auto"` |
| 设置对话框 | "跟踪后端"行位于滤波器类型行之后；项为 `("自动（推荐）", "auto")`、`("Python", "python")`，**仅当 `importlib.util.find_spec("omintrack") is not None`** 时追加 `("Rust", "rust")`（防止用户选到未构建必炸项） |
| tooltip | "自动：Rust 扩展可用即用 Rust，否则 Python；切换后对新接入/重置的相机生效" |
| main_window | 启动加载处 + `_apply_ai_settings` 两处调用 `InferWorker.set_tracker_backend`（仿 `set_pipeline_enabled` 模式）；cfg 键缺省 `"auto"` |
| InferWorker | `_tracker_backend` 属性（默认 `"auto"`）+ `set_tracker_backend(mode)` setter；**运行中不重建既有 tracker**，仅对新接入（register_slot）/ 重置的 slot 生效 |
| register_slot | 构造 `DetectionTracker` 时透传 `backend=self._tracker_backend`；`"auto"` 字符串直接传入由 `__new__` 解析 |

### 2.3 ReID 模型路径三态默认加载（`settings_dialog.load_ai_reid_model_path`）

| QSettings 键 `ai/reid_model_path` 状态 | 返回值 | 语义 |
|---|---|---|
| **不存在**（首次启动 / 旧版升级） | 默认模型发现：① `models/osnet_x0_25_256x128_msmt17.onnx` 存在 → 其字符串；② 否则 `models/osnet*.onnx` 首个；③ 都没有 → `""` | 默认**启用**（随仓库模型） |
| 存在，值为 `""` | `""` | **尊重用户主动关闭** |
| 存在，非空 | 存储值原样返回 | 用户自定义路径 |

- **contains 语义**：区分"无键"与"键为空串"必须用 `QSettings.contains()`，不能用 `value(key, default)` 的默认值合并。
- **推荐恢复**："恢复推荐参数"按钮对 ReID 模型行走同一"默认模型发现"逻辑（推荐 = 启用随仓库模型），不得直接回填 `_RECOMMENDED_REID_MODEL_PATH`（= `""` 会把推荐恢复成关闭）。
- **enabled 推导不变**：`main_window` 仍按 `路径非空 and reid_min_tracks > 0` 推导启用，自动生效；`ai/reid_enabled` 键不动。

## 3. 维护指南

- **ReID 配置排查（1.1）**：不要只看 i/reid_enabled（派生量）；看 i/reid_model_path 三态——键不存在=默认发现模型启用、键存在且为空字符串=用户主动关闭、键存在非空=指定模型。

- **发布 / 无构建环境行为**：未构建 Rust 的机器上 auto 回退 Python，行为与历史版本一致（debug 日志可见回退）；无 `models/osnet*.onnx` 的机器上 ReID 默认关闭且不炸应用。
- **如何强制 Python 对拍**：环境变量 `OMINTRACK_BACKEND=python` 覆盖所有默认构造点；或显式 `DetectionTracker(backend="python")`。等价性 / 快照测试的 Python 侧均已显式 `backend="python"`，不受 auto 默认影响。
- **Rust 项门控**：`omintrack` 未构建的打包环境，GUI 不出现 "Rust" 项；若用户配置里残留 `"rust"`（换机/卸载扩展），下拉 `findData` 失配自然落回 "auto"，不炸。
- **运行中切换后端**：不重建既有 tracker；需要立即生效时重置/重接相机 slot。
- **测试**：`tests/test_default_backends.py`（16 项）覆盖 auto 三态、环境变量优先级、ReID 三态（内存假 QSettings，严禁写真实用户配置）、GUI offscreen 冒烟。

## 4. 验证

- `pytest tests/test_default_backends.py -q`：16 项全绿。
- `pytest tests/test_tracker_rust_equivalence.py tests/test_tracker_snapshot.py tests/test_reid_osnet.py -q`：41 项全绿（Python 对拍侧显式 `backend="python"`，未 `--update-golden`）。
- `pytest tests -q`：全套件全绿。
- 冒烟：`python -c "from ai.tracker import DetectionTracker; t=DetectionTracker(); print(type(t).__name__)"` → 已构建机器输出 `RustDetectionTracker`。
