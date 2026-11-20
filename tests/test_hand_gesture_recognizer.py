"""MediaPipe Hands API 兼容性守护测试。

锁定 `ai/hand_gesture_recognizer.py` 依赖的 mediapipe 0.10.x 旧式 API
(`mp.solutions.hands`)。mediapipe 1.x 移除了 `mp.solutions`，导致
HandGestureRecognizer 静默降级（`_ensure_hands` 捕获异常后返回 None，
手势识别永不生效且不报错）。本测试确保：

    1. venv 中已安装的 mediapipe 真实提供 `solutions.hands.Hands`；
    2. HandGestureRecognizer 能真实初始化底层 Hands 实例，
       且 `close()` 正确释放；
    3. `detect()` 的数组布局契约：非连续 ROI 切片不再触发
       MediaPipe 0.10.14 reference mode 的 c_contiguous ValueError；
       process 输入保证 C 连续（连续输入零复制）；成功/异常两条
       路径都恢复 process 前数组的原始 writeable 状态。

不访问摄像头、不联网；核心兼容性检查（1、2 及真实 ROI 路径）
不 mock，数组布局细节用 fake hands 精确断言。
"""
from __future__ import annotations

import logging
import traceback
from typing import Any

import numpy as np

from ai.hand_gesture_recognizer import HandGestureRecognizer


# ======================================================================
# Fakes & helpers
# ======================================================================


class _FakeResults:
    def __init__(self) -> None:
        self.multi_hand_landmarks = None
        self.multi_handedness = None


class _FakeHands:
    """捕获 process 输入，支持注入异常；用于精确断言数组布局契约。"""

    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[np.ndarray] = []
        self.error = error
        self.closed = False

    def process(self, image: np.ndarray):
        self.calls.append(image)
        if self.error is not None:
            raise self.error
        return _FakeResults()

    def close(self) -> None:
        self.closed = True


def _blank_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _error_records(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.ERROR
    ]


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



def test_detect_in_rois_real_non_contiguous_slice_no_mediapipe_error(caplog) -> None:
    """真实非连续 ROI 切片走真实 MediaPipe 路径，不得记录 ERROR。"""
    recognizer = HandGestureRecognizer(max_num_hands=1)
    try:
        frame = _blank_frame(160, 160)
        roi = frame[32:128, 32:128]
        assert not roi.flags.c_contiguous, "前置条件：ROI 切片必须非 C 连续"
        with caplog.at_level(logging.ERROR):
            detections = recognizer.detect_in_rois(frame, [(0.2, 0.2, 0.8, 0.8)])
        assert isinstance(detections, list)
        assert recognizer._hands is not None, "应走完 _ensure_hands 真实初始化路径"
        assert not _error_records(caplog), (
            f"非连续 ROI 不应记录 ERROR: {_error_records(caplog)}"
        )
    finally:
        recognizer.close()


def test_detect_blank_frame_calls_process_without_errors(caplog) -> None:
    """空白帧不得断言检测到手；只断言 process 被调用且无错误。"""
    recognizer = HandGestureRecognizer()
    fake = _FakeHands()
    recognizer._hands = fake
    try:
        with caplog.at_level(logging.ERROR):
            detections = recognizer.detect(_blank_frame())
        assert detections == []
        assert len(fake.calls) == 1, "process 必须被调用一次"
        assert not _error_records(caplog), f"不应记录 ERROR: {_error_records(caplog)}"
    finally:
        recognizer.close()


def test_detect_contiguous_input_zero_copy_and_readonly_preserved() -> None:
    """C 连续输入必须零复制（对象身份不变）；只读输入成功后仍只读。"""
    recognizer = HandGestureRecognizer()
    fake = _FakeHands()
    recognizer._hands = fake
    try:
        frame = _blank_frame()
        frame.flags.writeable = False
        detections = recognizer.detect(frame)
        assert detections == []
        assert len(fake.calls) == 1
        assert fake.calls[0] is frame, "连续输入不得额外复制（对象身份）"
        assert frame.flags.writeable is False, "只读输入成功后不得被恢复成可写"
    finally:
        recognizer.close()


def test_detect_non_contiguous_input_converted_and_original_untouched() -> None:
    """非连续输入必须转为 C 连续副本；原始切片不得被改动 writeable。"""
    recognizer = HandGestureRecognizer()
    fake = _FakeHands()
    recognizer._hands = fake
    try:
        frame = _blank_frame(128, 128)
        roi = frame[16:80, 16:80]
        assert not roi.flags.c_contiguous, "前置条件：切片必须非 C 连续"
        detections = recognizer.detect(roi)
        assert detections == []
        assert len(fake.calls) == 1
        arg = fake.calls[0]
        assert arg is not roi, "非连续输入必须转换为连续副本"
        assert arg.flags.c_contiguous, "process 输入必须 C 连续"
        assert roi.flags.writeable is True, "原始切片的 writeable 状态不得被改动"
        assert not roi.flags.c_contiguous, "原始切片布局不得被改动"
    finally:
        recognizer.close()


def test_detect_process_exception_restores_writeable_and_returns_empty(caplog) -> None:
    """process 抛异常时记录 ERROR、返回 []，并恢复原始 writeable 状态。"""
    recognizer = HandGestureRecognizer()
    fake = _FakeHands(error=RuntimeError("boom"))
    recognizer._hands = fake
    try:
        frame = _blank_frame()
        assert frame.flags.writeable is True
        with caplog.at_level(logging.ERROR):
            detections = recognizer.detect(frame)
        assert detections == []
        assert frame.flags.writeable is True, "异常路径必须恢复原始 writeable=True"
        messages = _error_records(caplog)
        assert messages and any("MediaPipe hand detection failed" in m for m in messages), (
            f"应记录 MediaPipe 错误日志: {messages}"
        )
    finally:
        recognizer.close()


def test_detect_readonly_frame_exception_keeps_readonly(caplog) -> None:
    """只读输入 + process 抛异常：仍保持只读，不得被恢复成可写。"""
    recognizer = HandGestureRecognizer()
    fake = _FakeHands(error=ValueError("c_contiguous"))
    recognizer._hands = fake
    try:
        frame = _blank_frame()
        frame.flags.writeable = False
        with caplog.at_level(logging.ERROR):
            detections = recognizer.detect(frame)
        assert detections == []
        assert frame.flags.writeable is False, "只读输入异常后仍必须只读"
    finally:
        recognizer.close()


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
