# MediaPipe Hands 兼容性约束

> **版本**：1.0.0（2026-11-19）
> **范围**：`requirements.txt`、`ai/hand_gesture_recognizer.py`、`tests/test_hand_gesture_recognizer.py`
> **状态**：待合并；venv 已安装 mediapipe 0.10.14，2 项专项测试与全套件 1279 项均通过（2026-11-19）

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
| C5 | `_ensure_hands()` 静默降级行为**保持不变** | 行为契约 | 本修复只约束依赖与加守护测试，不改 `ai/hand_gesture_recognizer.py` 任何逻辑 |

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
