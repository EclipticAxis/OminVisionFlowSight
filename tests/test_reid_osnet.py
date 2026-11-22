r"""OSNet ReID 后端契约测试：预处理 / 异常回退 / mock 止损门控 / 真实模型 smoke。

- 不依赖模型的单测用 stub session 替换 ``ReIDBackend._session``，绕过模型加载；
- mock 门控场景构造参考 tests/test_tracker_rust_equivalence.py 的 ReID 场景
  （3+ person tracks、stride=1、非空 frame），双 backend（Python/Rust facade）分别验证；
- 真实模型 smoke 遵循 skip-if-missing 惯例（模型随仓库分发于 models/）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai.reid_backend import ReIDBackend  # noqa: E402
from ai.tracker import DetectionTracker  # noqa: E402
from ai.tracker_backend import RustDetectionTracker  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _REPO_ROOT / "models" / "osnet_x0_25_256x128_msmt17.onnx"


# ======================================================================
# stub 基础设施
# ======================================================================

class _StubSession:
    """记录收到的输入 batch、返回固定输出的 stub ONNX session。"""

    def __init__(self, output: np.ndarray | None = None, exc: Exception | None = None):
        self.calls: list[np.ndarray] = []
        self._output = output if output is not None else np.zeros((1, 512), dtype=np.float32)
        self._exc = exc

    def run(self, output_names, feed):
        self.calls.append(feed["input"])
        if self._exc is not None:
            raise self._exc
        return [self._output]


def _make_real_backend(session: _StubSession) -> ReIDBackend:
    """构造真实模式的 backend 但绕过模型加载（显式路径加载失败 → mock 后替换）。"""
    backend = ReIDBackend("nonexistent_osnet.onnx")
    assert backend.is_mock  # 加载失败回退 mock
    backend._session = session
    backend._input_name = "input"
    backend._mock = False
    return backend


class _CountingBackend:
    """包装 ReIDBackend 统计 extract 调用次数；is_mock 透传（门控读取点）。"""

    def __init__(self, backend):
        self._backend = backend
        self.calls = 0

    @property
    def is_mock(self) -> bool:
        return getattr(self._backend, "is_mock", False)

    def extract(self, roi):
        self.calls += 1
        return self._backend.extract(roi)


# ======================================================================
# a) 预处理契约（stub session，不依赖模型）
# ======================================================================

def test_real_extract_preprocessing_contract():
    """真实模式预处理契约：batch (1,3,256,128) float32、ImageNet 归一化、确定性。"""
    import cv2

    session = _StubSession(output=np.ones((1, 512), dtype=np.float32))
    backend = _make_real_backend(session)

    rng = np.random.default_rng(0)
    roi = rng.integers(0, 256, size=(64, 32, 3), dtype=np.uint8)

    emb1 = backend.extract(roi)
    emb2 = backend.extract(roi)
    assert emb1 is not None and emb2 is not None
    assert emb1.shape == (512,)
    assert emb1.dtype == np.float32
    assert np.isfinite(emb1).all()
    assert np.linalg.norm(emb1) == pytest.approx(1.0, abs=1e-5)
    assert np.array_equal(emb1, emb2)

    # 两次调用的输入 batch 形状 / dtype / 数值逐位一致
    assert len(session.calls) == 2
    for batch in session.calls:
        assert batch.shape == (1, 3, 256, 128)
        assert batch.dtype == np.float32
        assert np.isfinite(batch).all()
    assert np.array_equal(session.calls[0], session.calls[1])

    # 与预处理契约手工复现逐位一致：resize (W=128,H=256) → BGR→RGB → /255
    # → ImageNet mean/std（RGB 通道序）→ CHW → batch
    resized = cv2.resize(roi, (128, 256))[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    expected = np.transpose((resized - mean) / std, (2, 0, 1))[None, ...]
    assert expected.shape == (1, 3, 256, 128)
    assert np.array_equal(session.calls[0], expected)


def test_real_extract_invalid_roi_returns_none():
    session = _StubSession()
    backend = _make_real_backend(session)
    assert backend.extract(None) is None
    assert backend.extract(np.empty((0, 0, 3), dtype=np.uint8)) is None
    assert session.calls == []


# ======================================================================
# b) 推理异常 → None（禁止回退 mock 向量）
# ======================================================================

def test_real_extract_inference_error_returns_none():
    session = _StubSession(exc=RuntimeError("inference boom"))
    backend = _make_real_backend(session)
    roi = np.zeros((64, 32, 3), dtype=np.uint8)
    # 返回 None 而非 mock 向量（mock 向量维度为 EMBEDDING_DIM=256 且非 None）
    assert backend.extract(roi) is None
    assert backend.extract(roi) is None


# ======================================================================
# c) mock 止损门控（双 backend：Python DetectionTracker + Rust facade）
# ======================================================================

def _person_dets(frame_idx: int) -> list[dict]:
    """三个空间分离的 person 检测，逐帧微移（保证 3+ person tracks 稳定存在）。"""
    dets = []
    for x in (0.10, 0.40, 0.70):
        shift = 0.001 * frame_idx
        dets.append({
            "x1": x + shift, "y1": 0.20,
            "x2": x + 0.15 + shift, "y2": 0.60,
            "confidence": 0.9, "label": "person",
        })
    return dets


def _run_person_scenario(tracker, frames: int = 6) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for f in range(frames):
        tracker.update(_person_dets(f), frame=frame)


def _make_py_tracker(backend):
    return DetectionTracker(
        matching_strategy="hungarian",
        reid_backend=backend, reid_min_tracks=3, reid_stride=1,
    )


def _make_rust_tracker(backend):
    return RustDetectionTracker(
        matching_strategy="hungarian",
        reid_backend=backend, reid_min_tracks=3, reid_stride=1,
    )


def test_mock_backend_reid_not_active_python():
    """mock backend：Python tracker 的 reid_active 门控不激活，extract 零调用。"""
    backend = _CountingBackend(ReIDBackend("nonexistent_osnet.onnx"))
    assert backend.is_mock
    tracker = _make_py_tracker(backend)
    _run_person_scenario(tracker)
    assert backend.calls == 0


def test_mock_backend_reid_not_active_rust_facade():
    """mock backend：Rust facade 的门控与 Python 侧等价，extract 零调用。"""
    backend = _CountingBackend(ReIDBackend("nonexistent_osnet.onnx"))
    assert backend.is_mock
    tracker = _make_rust_tracker(backend)
    _run_person_scenario(tracker)
    assert backend.calls == 0


def test_non_mock_backend_reid_active_positive_control():
    """正对照：is_mock=False 时同一场景必须触发 extract（证明场景构造有效）。"""
    backend = _CountingBackend(_make_real_backend(_StubSession()))
    assert not backend.is_mock
    tracker = _make_py_tracker(backend)
    _run_person_scenario(tracker)
    assert backend.calls > 0


def test_no_backend_reid_not_active():
    """无 backend 时自然不激活（回归保护）。"""
    tracker = _make_py_tracker(None)
    _run_person_scenario(tracker)


# ======================================================================
# 真实模型 smoke（skip-if-missing 惯例）
# ======================================================================

@pytest.mark.skipif(not _MODEL_PATH.exists(), reason="OSNet ONNX model not present in models/")
def test_real_model_smoke():
    """真实模型：512 维、有限、L2 归一化、同 ROI 确定、不同 ROI 不同。"""
    backend = ReIDBackend(str(_MODEL_PATH))
    assert not backend.is_mock

    rng = np.random.default_rng(42)
    roi = rng.integers(0, 256, size=(64, 32, 3), dtype=np.uint8)
    emb1 = backend.extract(roi)
    emb2 = backend.extract(roi)
    assert emb1 is not None
    assert emb1.shape == (512,)
    assert np.isfinite(emb1).all()
    assert np.linalg.norm(emb1) == pytest.approx(1.0, abs=1e-4)
    assert np.array_equal(emb1, emb2)

    other = np.zeros((64, 32, 3), dtype=np.uint8)
    emb3 = backend.extract(other)
    assert emb3 is not None
    assert not np.array_equal(emb1, emb3)


@pytest.mark.skipif(not _MODEL_PATH.exists(), reason="OSNet ONNX model not present in models/")
def test_real_model_auto_discovery():
    """自动发现：models/osnet_x0_25_256x128_msmt17.onnx 存在 → 真实模式。"""
    backend = ReIDBackend()  # pytest 从仓库根目录运行，cwd 即仓库根
    assert not backend.is_mock
    assert backend._model_path and "osnet" in backend._model_path.lower()
