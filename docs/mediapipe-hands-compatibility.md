# MediaPipe Hands 兼容性约束

> **版本**：1.1.0（2026-11-20）
> **范围**：`ai/hand_gesture_recognizer.py`、`tests/test_hand_gesture_recognizer.py`（1.0.0 版另含 `requirements.txt`）
> **状态**：待合并（本地实现，8 项专项测试与全套件 1336 项通过，2026-11-20）；1.0.0 已合入 main（PR #3 → `d2d8349`，2026-11-19）

---

## 1. 目标与背景

`ai/hand_gesture_recognizer.py` 的 `HandGestureRecognizer._ensure_hands()` 使用
mediapipe 0.10.x 旧式 API `mp.solutions.hands.Hands(...)` 初始化手势识别器。

**问题**：当前验证的 mediapipe 0.10.35 与 1.0.1 已移除 `mp.solutions` 模块。由于 `_ensure_hands()` 内部
`try/except` 捕获所有异常并静默返回 `None`，安装 1.x 后手势识别**永不生效且无任何报错**，
属于隐蔽失效。

**目标**：
1. 用 `requirements.txt` 版本约束阻止 pip 再次解析到不兼容版本；
2. 用守护测试真实验证 venv 中已安装 mediapipe 提供 `solutions.hands`，
   且 `HandGestureRecognizer` 能成功初始化底层 `Hands` 实例；
3. 明确迁移路径（若未来要升级到 mediapipe 1.x / Tasks API）。

## 2. 契约

| # | 契约 | 约束类型 | 说明 |
|---|------|----------|------|
| C1 | `requirements.txt` 中 mediapipe 约束为 `>=0.10.14,<0.10.15` | 修改契约 | 0.10.14 提供 Solutions API 且其 protobuf 约束与 ONNX 兼容；上界排除后续变更依赖及移除 Solutions API 的版本 |
| C2 | 已安装 mediapipe 必须暴露 `mp.solutions.hands.Hands` | 行为契约 | 由 `tests/test_hand_gesture_recognizer.py::test_installed_mediapipe_exposes_solutions_hands` 守护，不 mock |
| C3 | `HandGestureRecognizer.detect()` 能将 `_hands` 初始化为非 None | 行为契约 | 由 `test_recognizer_initializes_underlying_hands` 守护，喂空白帧走完 `_ensure_hands` 真实路径（无摄像头/无网络） |
| C4 | `HandGestureRecognizer.close()` 后 `_hands` 归位 None | 行为契约 | 同上测试守护，防止资源泄漏 |
| C5 | `_ensure_hands()` 静默降级行为**保持不变** | 行为契约 | 1.0.0 修复只约束依赖与加守护测试，未改 `_ensure_hands()` 逻辑 |
| C6 | `detect()` 传入 `hands.process()` 的数组必须 C 连续 | 行为契约 | MediaPipe 0.10.14 reference mode 对非连续输入抛 `ValueError`（c_contiguous）；`detect_in_rois()` 的 ROI 切片（`frame[y1:y2, x1:x2]`）天然非连续，曾导致每次 ROI 检测记录一条 ERROR 并返回空 |
| C7 | `detect()` 用 `np.ascontiguousarray()` 归一化布局：C 连续输入**零复制**（返回原对象，fake hands 以对象身份断言），仅非连续输入复制为连续副本 | 行为契约 | 由 `test_detect_contiguous_input_zero_copy_and_readonly_preserved`、`test_detect_non_contiguous_input_converted_and_original_untouched` 守护 |
| C8 | `process()` 前记录数组 writeable 状态，`try/except/finally` 在**成功与异常两条路径都恢复**该状态；只读输入不得被恢复成 `writeable=True` | 行为契约 | 旧实现在成功/异常路径都把输入恢复成 `True`，破坏调用方的只读约定；由 `test_detect_contiguous_input_zero_copy_and_readonly_preserved`、`test_detect_process_exception_restores_writeable_and_returns_empty`、`test_detect_readonly_frame_exception_keeps_readonly` 守护 |
| C9 | `detect()` 的返回结构、日志文案（`MediaPipe hand detection failed: %s`）、手势分类与 ROI 坐标映射行为**保持不变** | 行为契约 | 本修复只改 `process()` 输入的数组布局与 writeable 生命周期 |

## 3. 维护指南

- **升级 mediapipe**：不要直接放宽 `<0.10.15`。升级到已移除 Solutions API 的版本需迁移到 Tasks API
  （`mediapipe.tasks.python.vision.GestureRecognizer` / `HandLandmarker`），
  属于行为变更，须同步改 `_ensure_hands()`、更新本契约表、并按
  PROJECT_BIBLE §17 沉淀迁移文档。
- **守护测试失败时**：先用
  `venv\Scripts\python.exe -c "import mediapipe as mp; print(mp.__version__, hasattr(mp, 'solutions'))"`
  确认 venv 实际版本；版本不在兼容范围时按 `requirements.txt` 安装
  （`pip install "mediapipe>=0.10.14,<0.10.15"`），不要改测试去适配无 Solutions API 的版本。
- **测试可在无摄像头环境运行**：只构造 64x64 空白 numpy 帧，不触发实际模型推理依赖外部资源。
- 运行：`venv\Scripts\python.exe -m pytest tests/test_hand_gesture_recognizer.py -v`

## 4. 数组布局契约（1.1.0 新增）

`detect()` 在进入 MediaPipe 前统一执行：

1. `process_frame = np.ascontiguousarray(frame)` —— 保证 C 连续；连续输入原样返回（零复制），
   非连续 ROI 切片才产生副本，原始切片的布局与 writeable 状态均不被触碰。
2. 记录 `was_writeable = process_frame.flags.writeable`，在 `process_frame` 上设 `writeable=False` 后调用
   `hands.process(process_frame)`；`finally` 中恢复 `was_writeable`，成功/异常一致。

**维护要点**：
- `writeable` 标志只能作用于实际传入 MediaPipe 的 `process_frame`，且必须在 `finally` 中恢复；
  不得无恢复地修改调用方的 `frame`，破坏其只读约定。
- 若未来迁移到 MediaPipe Tasks API，`HandLandmarker.detect_for_video` 接受 `mp.Image`，
  布局约束不同；迁移时本节 C6–C9 需按新 API 重新验证并更新。

## 5. 历史

- **1.0.0（2026-11-19）**：mediapipe 依赖兼容性修复。用 `requirements.txt` 约束
  `>=0.10.14,<0.10.15` 阻止 pip 解析到已移除 `mp.solutions` 的版本，并新增 2 项守护测试
  （C1–C5）。已合入 main（PR #3 → `d2d8349`），全套件 1279 项通过（2026-11-19）。
- **1.1.0（2026-11-20）**：数组布局与 writeable 生命周期修复（C6–C9），专项测试由 2 项增至 8 项。
  状态：待合并（本地实现，8 项专项测试与全套件 1336 项通过）。
