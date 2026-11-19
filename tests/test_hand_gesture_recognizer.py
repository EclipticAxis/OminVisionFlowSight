"""MediaPipe Hands API 兼容性守护测试。

锁定 `ai/hand_gesture_recognizer.py` 依赖的 mediapipe 0.10.x 旧式 API
(`mp.solutions.hands`)。mediapipe 1.x 移除了 `mp.solutions`，导致
HandGestureRecognizer 静默降级（`_ensure_hands` 捕获异常后返回 None，
手势识别永不生效且不报错）。本测试确保：

    1. venv 中已安装的 mediapipe 真实提供 `solutions.hands.Hands`；
    2. HandGestureRecognizer 能真实初始化底层 Hands 实例，
       且 `close()` 正确释放。

不访问摄像头、不联网、不 mock 核心兼容性检查。
"""
from __future__ import annotations

import traceback
from typing import Any

import numpy as np

from ai.hand_gesture_recognizer import HandGestureRecognizer


# ======================================================================
# Tests
# ======================================================================

def test_installed_mediapipe_exposes_solutions_hands() -> None:
    import mediapipe as mp

    assert hasattr(mp, "solutions"), (
        f"mediapipe {getattr(mp, '__version__', '?')} 未提供 mp.solutions；"
        "ai/hand_gesture_recognizer.py 需要 mediapipe>=0.10.14,<0.10.15"
    )
    hands_module = mp.solutions.hands
    assert hasattr(hands_module, "Hands"), "mp.solutions.hands 缺少 Hands 类"


def test_recognizer_initializes_underlying_hands() -> None:
    recognizer = HandGestureRecognizer(max_num_hands=2)
    try:
        # 喂一张空白帧走完 _ensure_hands 初始化路径（无摄像头/无推理依赖）。
        detections = recognizer.detect(np.zeros((64, 64, 3), dtype=np.uint8))
        assert isinstance(detections, list)
        assert recognizer._hands is not None, (
            "HandGestureRecognizer 未能初始化 mp.solutions.hands.Hands；"
            "已安装 mediapipe 与旧式 solutions API 不兼容"
        )
    finally:
        recognizer.close()
    assert recognizer._hands is None, "close() 后底层 Hands 应被释放"


# ======================================================================
# Runner
# ======================================================================

def _collect_tests() -> list[tuple[str, Any]]:
    g = globals()
    return sorted(
        (name, g[name]) for name in g
        if name.startswith("test_") and callable(g[name])
    )


if __name__ == "__main__":
    tests = _collect_tests()
    print(f"Running {len(tests)} mediapipe hands compatibility tests...\n")
    passed = 0
    failed = 0
    failures: list[tuple[str, str]] = []
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            tb = traceback.format_exc()
            failures.append((name, tb))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'=' * 60}")
    print(f"Result: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        print(f"\n{'=' * 60}\nFailure details:\n")
        for name, tb in failures:
            print(f"---- {name} ----")
            print(tb)
        raise SystemExit(1)
    print("All mediapipe hands compatibility tests passed.")
