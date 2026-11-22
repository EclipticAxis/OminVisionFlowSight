# ReID OSNet 后端 设计文档

> **版本**：1.0 — 2026-11-22
> **范围**：`ai/reid_backend.py`（重写）、`ai/tracker.py`（mock 止损门控）、`ai/tracker_backend.py`（mock 止损门控）、`ai/inference.py`（mock 提示）、`gui/settings_dialog.py`（ReID 模型配置行）、`gui/main_window.py`（配置链接通）、`tests/test_reid_osnet.py`（新增 9 项）、`models/osnet_x0_25_256x128_msmt17.onnx`（PINTO0309 模型随仓库分发）
> **状态**：⏳ 待合并（全套件 1350 项全绿：1341 既有 + 9 新增）

---

## 1. 目标与背景

旧 ReID 插件存在三个已实测确认的问题：

1. **mock 向量无语义、注入噪声**：旧 mock 用 `hash(roi_bytes)` 生成随机 256 维向量——
   同一人相邻帧 cosine≈0（与不同人同分布），却以 0.4 权重进入匹配 score
   （`0.4·IoU + 0.4·reid_sim + 0.2·center`），开启 ReID 反而比不开更差；
2. **跨进程不稳定**：`hash(bytes)` 在 Python 进程间随机化（PYTHONHASHSEED），
   跨进程/重启后同一 ROI 得到不同向量，无法做离线对比与回放；
3. **真实模式从未接入**：旧 `_real_extract` 预处理错误（resize 128x64、无 ImageNet
   归一化），推理异常时静默回退 mock 向量（问题 1 的噪声被静默注入）；
   GUI 无 ReID 模型路径配置，生产永远走 mock。

本次改动：接入真实 OSNet-x0.25 ONNX 模型，并为 mock 加"止损门控"——mock 时
ReID 匹配永不激活，mock 仅保留测试/管线验证用途。

## 2. 架构

```
GUI 设置对话框（ReID 模型行：关闭 / models/osnet*.onnx / 浏览）
    │ QSettings 键 ai/reid_model_path
    ▼
gui/main_window._apply_ai_settings
    │ enabled = 路径非空 and reid_min_tracks>0；model_path 透传
    ▼
ai/inference.InferWorker.set_reid_config ──► ReIDBackend(model_path)
    │                                            │  None/"" → 自动发现
    │                                            │  models/osnet_x0_25_256x128_msmt17.onnx
    │                                            │  → 否则 glob osnet*.onnx 首个 → 否则 mock
    ▼                                            ▼
tracker._reid_backend（Python DetectionTracker / RustDetectionTracker facade 双側）
    │
    ▼ reid_active 门控（双側同步，契约）
backend 非 mock AND person_count≥1 AND track_person_count≥reid_min_tracks
AND frame_counter % reid_stride == 0 AND frame 非空
    │
    ▼ reid_active 时
hungarian 匹配代价 = 0.4·IoU + 0.4·cosine(embedding) + 0.2·center
匹配成功 → track embedding EMA(0.8/0.2) + L2 归一化
```

分工不变：跟踪核心（UKF 滤波族、匹配引擎）在 Python/Rust 双实现中；
ReID embedding 提取（帧相关）保留在 Python 侧，以 `det_emb`/`reid_active`
传入 Rust 核心（见 `docs/rust-tracker-core.md` §2.2）。**本次不改 Rust 任何文件。**

## 3. 契约表

| 项 | 契约 |
|----|------|
| 模型文件 | `models/osnet_x0_25_256x128_msmt17.onnx`（PINTO0309，msmt17 权重，输入 NCHW `(1,3,256,128)` 即 H=256 W=128，输出 `(1,512)`） |
| 预处理 | `cv2.resize(roi, (W=128, H=256))` → BGR→RGB → `float32/255` → ImageNet 归一化 `mean=[0.485,0.456,0.406]`、`std=[0.229,0.224,0.225]`（**RGB 通道序**）→ CHW → batch `(1,3,256,128)` float32 |
| 输出 | `flatten → float32 → L2 归一化`；**维度自适应**（本模型 512 维）；`EMBEDDING_DIM=256` 仅 mock 使用 |
| 推理失败 | `_real_extract` 任何异常 → `logger.warning` + **返回 None**（该检测本帧无 embedding，匹配自然回退 HSV 外观路径）；**禁止回退 mock 向量** |
| is_mock | `ReIDBackend.is_mock` property（= `self._mock`，保留 `_mock` 字段兼容 inference.py 既有读取）；显式路径加载失败 → mock + warning；自动发现无模型 → mock + `logger.warning("ReID mock 模式仅供管线验证，匹配不会激活")` |
| 门控条件（双側同步） | `reid_active` 追加 `not getattr(backend, "is_mock", False)`；**getattr 默认 False** 兼容无 `is_mock` 的自定义 backend（如 tests 的 `_FakeReidBackend`，行为不变）；不加其他门控改动 |
| 配置链 | QSettings 键 `ai/reid_model_path`（ComboBox 首项 `("关闭", "")`）；`enabled = bool(路径) and reid_min_tracks > 0`；`model_path=路径 or None`；既有 `_KEY_AI_REID_ENABLED` 加载逻辑不动 |
| 匹配权重 | ReID 激活且双侧 embedding 存在时 `0.4·IoU + 0.4·cosine + 0.2·center`；**不变** |

## 4. 维护指南

- **更换 OSNet 变体**：注意输入尺寸（当前契约 256×128）与归一化约定
  （ImageNet mean/std、RGB 通道序）；其他输入尺寸的变体需同步改
  `_real_extract` 的 resize 与文档契约表，并更新 smoke 测试的维度断言。
- **模型随仓库分发**：`models/` 下已有惯例（UHD/CHC 同 PINTO0309）；
  缺模型时一切自动回退 mock 且不炸应用，smoke 测试按 skip-if-missing 惯例跳过。
- **mock 仅测试用途**：mock 向量无语义，双側门控保证其不会进入生产匹配；
  新增消费 ReID embedding 的代码必须复用同一 `is_mock` 门控约定。
- **双后端同步**：`ai/tracker.py` 与 `ai/tracker_backend.py` 的门控条件必须逐字一致
  （Rust 核心本身不感知 mock，门控在 Python 适配层双側）。

## 5. 验证

- `pytest tests/test_reid_osnet.py -q`：9 项全绿（预处理契约 / 异常回退 None /
  mock 门控 Python+Rust 双 backend / 正对照 / 真实模型 smoke + 自动发现）。
- `pytest tests/test_tracker_rust_equivalence.py tests/test_tracker_snapshot.py -q`：32 项全绿
  （`_FakeReidBackend` 无 `is_mock` → getattr 默认 False → 行为不变）。
- `pytest tests -q`：全套件 1350 项全绿（1341 既有 + 9 新增），未改任何既有测试、未 `--update-golden`。
- 端到端冒烟：`ReIDBackend()` 自动发现模型 → `is_mock=False` → 合成 64×32 BGR ROI
  输出 512 维 float32、有限、L2 范数 1.0、同 ROI 逐位一致、不同 ROI 不同。
- GUI import 冒烟：`import gui.settings_dialog` / `import gui.main_window` 通过（PyQt6）。
