"""DepthWorker（YOLO26 深度推理线程）单元测试。

全部使用 fake model / monkeypatch：不启动真实相机、不加载真实模型、
不 import torch / ultralytics（cv2 仅在直接验证 colorize 输出时用）。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ai.depth_worker import DEFAULT_MODEL_PATH, DepthWorker, colorize_depth
from tests._runtime_checks import assert_import_clean


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------


class SignalSpy:
    """极简信号记录器（线程安全；配合 DirectConnection 使用）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[tuple] = []

    def __call__(self, *args) -> None:
        with self._lock:
            self.calls.append(args)

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.calls)


def wait_until(pred, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return False


def make_frame(value: int = 0, shape: tuple[int, int] = (32, 48)) -> np.ndarray:
    return np.full((*shape, 3), value, dtype=np.uint8)


class FakeModel:
    """记录 predict 调用参数、按帧内容生成确定性深度的假模型。"""

    def __init__(self, out_scale: float = 1.0):
        self.calls: list[dict] = []
        self.out_scale = out_scale

    def predict(self, frame, imgsz, device, verbose):
        self.calls.append(
            {"frame": frame.copy(), "imgsz": imgsz, "device": device, "verbose": verbose}
        )
        h, w = frame.shape[:2]
        oh, ow = max(1, int(h * self.out_scale)), max(1, int(w * self.out_scale))
        value = float(frame[0, 0, 0])
        depth = np.full((1, oh, ow), value, dtype=np.float32)  # (1,H,W) 模拟 ultralytics
        return [SimpleNamespace(depth=SimpleNamespace(data=depth))]


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def worker(qapp, monkeypatch):
    """device='cpu' 避免 torch 探测；模型由 _load_model 懒加载（可 monkeypatch）。"""
    w = DepthWorker(model_path=None, imgsz=640, device="cpu")
    spies = SimpleNamespace(depth=SignalSpy(), error=SignalSpy())
    w.depth_ready.connect(spies.depth, Qt.ConnectionType.DirectConnection)
    w.error_occurred.connect(spies.error, Qt.ConnectionType.DirectConnection)
    yield w, spies
    w.stop(3000)


def install_fake_model(worker: DepthWorker, model) -> None:
    worker._load_model = lambda: model  # type: ignore[assignment]


# ----------------------------------------------------------------------
# import 副作用
# ----------------------------------------------------------------------


def test_import_has_no_heavy_side_effects():
    assert_import_clean("ai.depth_worker", ["torch", "cv2", "ultralytics"])


def test_default_model_path():
    assert DEFAULT_MODEL_PATH.name == "yolo26s-depth.pt"
    assert DEFAULT_MODEL_PATH.parent.name == "models"


# ----------------------------------------------------------------------
# colorize_depth 纯函数
# ----------------------------------------------------------------------


def test_colorize_shape_dtype_rgb_contiguous():
    import cv2

    depth = np.linspace(0.0, 20.0, 24 * 32, dtype=np.float32).reshape(24, 32)
    out = colorize_depth(depth, vmin=0.0, vmax=20.0)
    assert out.shape == (24, 32, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]

    # 与手工 BGR→RGB 参考实现逐像素一致（验证确实做了 BGR→RGB）
    gray = np.clip(depth / 20.0, 0, 1)
    gray = (gray * 255.0).round().astype(np.uint8)
    ref = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(out, ref)


def test_colorize_metric_range_clipping():
    def px(v):
        return colorize_depth(np.full((4, 4), v, dtype=np.float32))[0, 0]

    # 超出固定 range 的值被钳到端点：色彩跨帧稳定
    np.testing.assert_array_equal(px(-5.0), px(0.0))
    np.testing.assert_array_equal(px(100.0), px(20.0))
    # 范围内不同值颜色不同
    assert not np.array_equal(px(5.0), px(15.0))


def test_colorize_nan_inf_map_to_low_end():
    depth = np.array([[np.nan, np.inf], [-np.inf, 10.0]], dtype=np.float32)
    out = colorize_depth(depth)
    assert out.shape == (2, 2, 3)
    low = colorize_depth(np.zeros((1, 1), dtype=np.float32))[0, 0]
    np.testing.assert_array_equal(out[0, 0], low)  # NaN → 低端
    np.testing.assert_array_equal(out[0, 1], low)  # +Inf → 低端
    np.testing.assert_array_equal(out[1, 0], low)  # -Inf → 低端


def test_colorize_posinf_not_mapped_to_high_end():
    """+Inf 与 NaN/-Inf 一样映射低值端，而非 vmax 高值端（0-20m Inferno 不变）。"""
    low = colorize_depth(np.zeros((1, 1), dtype=np.float32))[0, 0]
    high = colorize_depth(np.full((1, 1), 20.0, dtype=np.float32))[0, 0]
    out = colorize_depth(np.array([[np.inf, 20.0]], dtype=np.float32))
    np.testing.assert_array_equal(out[0, 0], low)   # +Inf → 低端
    np.testing.assert_array_equal(out[0, 1], high)  # 有限 vmax 仍映射高端
    assert not np.array_equal(out[0, 0], out[0, 1])


def test_colorize_squeezes_channel_dim_and_rejects_bad_shape():
    out = colorize_depth(np.zeros((1, 8, 8), dtype=np.float32))
    assert out.shape == (8, 8, 3)
    with pytest.raises(ValueError):
        colorize_depth(np.zeros((2, 8, 8), dtype=np.float32))
    with pytest.raises(ValueError):
        colorize_depth(np.zeros((4, 4)), vmax=0.0, vmin=0.0)
    with pytest.raises(ValueError):
        colorize_depth(np.zeros((4, 4)), mode="bogus")


# ----------------------------------------------------------------------
# 启停 / 接受帧语义
# ----------------------------------------------------------------------


def test_submit_rejected_when_disabled(worker):
    w, spies = worker
    model = FakeModel()
    load_calls: list[int] = []

    def counting_load():
        load_calls.append(1)
        return model

    w._load_model = counting_load  # type: ignore[assignment]

    w.register_slot(0)
    w.start()
    w.submit_frame(0, make_frame(7))  # 未 enable → 拒帧
    time.sleep(0.2)
    assert spies.depth.count == 0
    assert spies.error.count == 0
    assert load_calls == []  # 模型从未被加载
    assert model.calls == []


def test_submit_rejected_for_unregistered_slot(worker):
    w, spies = worker
    model = FakeModel()
    install_fake_model(w, model)
    w.set_enabled(True)
    w.start()
    w.submit_frame(3, make_frame(7))  # slot 3 未 register → 拒帧
    time.sleep(0.2)
    assert spies.depth.count == 0
    assert model.calls == []


def test_enable_then_depth_ready(worker):
    w, spies = worker
    model = FakeModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))

    assert wait_until(lambda: spies.depth.count >= 1)
    slot_id, img = spies.depth.calls[0]
    assert slot_id == 0
    assert img.shape == (32, 48, 3)
    assert img.dtype == np.uint8 and img.flags["C_CONTIGUOUS"]
    # predict 契约：imgsz / device / verbose=False
    assert model.calls[0]["imgsz"] == 640
    assert model.calls[0]["device"] == "cpu"
    assert model.calls[0]["verbose"] is False


def test_output_resized_to_input_size(worker):
    w, spies = worker
    install_fake_model(w, FakeModel(out_scale=0.5))  # 模型输出半尺寸深度
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10, shape=(40, 64)))
    assert wait_until(lambda: spies.depth.count >= 1)
    assert spies.depth.calls[0][1].shape == (40, 64, 3)


def test_rgb_input_is_converted_to_bgr_for_ultralytics(worker):
    w, _spies = worker
    model = FakeModel()
    install_fake_model(w, model)
    w._model = model
    w._resolved_device = "cpu"
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:] = (1, 2, 3)

    w._infer_depth(frame)

    assert model.calls[0]["frame"][0, 0].tolist() == [3, 2, 1]


def test_disable_clears_pending_frames(worker):
    w, spies = worker
    model = FakeModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.submit_frame(0, make_frame(10))
    w.set_enabled(False)  # 线程未 start，pending 必须被清空
    w.start()
    time.sleep(0.3)
    assert spies.depth.count == 0
    assert model.calls == []
    # 重新 enable 后可正常出图
    w.set_enabled(True)
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.depth.count >= 1)


def test_disable_after_inference_blocks_emit(worker):
    """推理解除（完成）时若已禁用，不得发新深度帧。"""
    w, spies = worker

    class GatedModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def predict(self, frame, imgsz, device, verbose):
            self.started.set()
            assert self.release.wait(5)
            return super().predict(frame, imgsz, device, verbose)

    model = GatedModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert model.started.wait(5)
    w.set_enabled(False)  # 推理进行中禁用
    model.release.set()
    time.sleep(0.3)
    assert spies.depth.count == 0


def test_disable_during_model_load_skips_inference(worker):
    w, spies = worker
    model = FakeModel()
    loaded = threading.Event()
    release = threading.Event()

    def gated_load():
        loaded.set()
        assert release.wait(5)
        return model

    w._load_model = gated_load  # type: ignore[assignment]
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert loaded.wait(5)
    w.set_enabled(False)
    release.set()
    time.sleep(0.3)

    assert model.calls == []
    assert spies.depth.count == 0


def test_latest_frame_overwrites_and_round_robin(worker):
    """worker 忙时同 slot 只保留最新帧；多 slot 公平轮询。"""
    w, spies = worker

    class GatedModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def predict(self, frame, imgsz, device, verbose):
            if not self.calls:
                self.started.set()
                assert self.release.wait(5)
            return super().predict(frame, imgsz, device, verbose)

    model = GatedModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.register_slot(1)
    w.set_enabled(True)
    w.start()

    w.submit_frame(0, make_frame(10))
    assert model.started.wait(5)  # worker 正忙于 slot0 第 1 帧
    w.submit_frame(0, make_frame(20))  # 将被覆盖
    w.submit_frame(1, make_frame(99))
    w.submit_frame(0, make_frame(30))  # 覆盖 20
    model.release.set()

    assert wait_until(lambda: spies.depth.count >= 3)
    values = [c["frame"][0, 0, 0] for c in model.calls]
    assert values[0] == 10
    assert 20 not in values  # 被 30 覆盖
    assert set(values[1:]) == {30, 99}
    # 公平轮询：slot0 被连续提交多帧，slot1 仍各获得一次处理
    slots = [c[0] for c in spies.depth.calls]
    assert set(slots) == {0, 1}


# ----------------------------------------------------------------------
# 错误路径
# ----------------------------------------------------------------------


def test_model_load_failure_emits_minus_one_and_disables(worker):
    w, spies = worker

    def bad_load():
        raise RuntimeError("boom-load")

    w._load_model = bad_load  # type: ignore[assignment]
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))

    assert wait_until(lambda: spies.error.count >= 1)
    slot_id, msg = spies.error.calls[0]
    assert slot_id == -1
    assert "boom-load" in msg
    # 内部禁用：后续 submit 被拒，不 busy 重试（只报一次错）
    w.submit_frame(0, make_frame(11))
    time.sleep(0.3)
    assert spies.error.count == 1
    assert spies.depth.count == 0
    assert w.isRunning()  # 线程保持等待而非退出/忙等


def test_manual_reenable_retries_after_model_load_failure(worker):
    w, spies = worker
    attempts = {"count": 0}
    model = FakeModel()

    def load_then_recover():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("temporary-load-error")
        return model

    w._load_model = load_then_recover  # type: ignore[assignment]
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.error.count == 1)

    w.set_enabled(True)  # 用户手动重开：允许一次新的加载尝试
    w.submit_frame(0, make_frame(11))
    assert wait_until(lambda: spies.depth.count == 1)
    assert attempts["count"] == 2


def test_inference_error_emits_slot_error_then_continues(worker):
    w, spies = worker

    class FlakyModel(FakeModel):
        def predict(self, frame, imgsz, device, verbose):
            if not self.calls:
                self.calls.append({})  # 占住“第一次”
                raise RuntimeError("boom-infer")
            return super().predict(frame, imgsz, device, verbose)

    model = FlakyModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))  # 第 1 帧失败（第一次必报，不节流吞掉）
    assert wait_until(lambda: spies.error.count >= 1)
    slot_id, msg = spies.error.calls[0]
    assert slot_id == 0
    assert "boom-infer" in msg

    w.submit_frame(0, make_frame(10))  # 第 2 帧成功，线程未被拖死
    assert wait_until(lambda: spies.depth.count >= 1)


def test_infer_error_throttled_first_and_every_30(worker):
    """连续失败节流：第 1 次及每第 30 次上报，其余只计数不上报。"""
    w, spies = worker

    class AlwaysFailModel(FakeModel):
        def predict(self, frame, imgsz, device, verbose):
            self.calls.append({})
            raise RuntimeError("boom-throttle")

    model = AlwaysFailModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    for i in range(31):
        w.submit_frame(0, make_frame(10))
        # 等本帧真正进入 predict 后再提交下一帧，避免覆盖式 pending 丢帧
        assert wait_until(lambda n=i + 1: len(model.calls) >= n)
    # 第 1、30 次上报 → 共 2 次；31 帧全部失败
    assert len(model.calls) == 31
    assert spies.error.count == 2
    assert spies.depth.count == 0


def test_infer_error_count_resets_after_success(worker):
    """成功一次后连续失败计数清零：再次失败重新按“首次”上报。"""
    w, spies = worker

    class FlakyModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.fail_next = True

        def predict(self, frame, imgsz, device, verbose):
            if self.fail_next:
                self.fail_next = False
                self.calls.append({})
                raise RuntimeError("boom-reset")
            return super().predict(frame, imgsz, device, verbose)

    model = FlakyModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))  # 失败 → 第 1 次必报
    assert wait_until(lambda: spies.error.count == 1)
    w.submit_frame(0, make_frame(10))  # 成功 → 计数清零
    assert wait_until(lambda: spies.depth.count == 1)
    model.fail_next = True
    w.submit_frame(0, make_frame(10))  # 再次失败 → 重新按首次上报
    assert wait_until(lambda: spies.error.count == 2)


def test_infer_error_count_cleared_on_unregister(worker):
    """unregister_slot 清空该 slot 的失败计数：重新注册后再次失败按首次上报。"""
    w, spies = worker

    class AlwaysFailModel(FakeModel):
        def predict(self, frame, imgsz, device, verbose):
            self.calls.append({})
            raise RuntimeError("boom-unregister")

    model = AlwaysFailModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.error.count == 1)
    assert wait_until(lambda: len(model.calls) == 1)

    w.unregister_slot(0)
    assert w._infer_error_counts == {}  # 计数随 unregister 清零

    w.register_slot(0)
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.error.count == 2)  # 重新按首次上报


def test_stop_cooperative_with_inflight_inference(worker):
    """stop 期间有在飞推理：不 terminate，推理完成后协作退出且不再发帧。"""
    w, spies = worker

    class GatedModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.started = threading.Event()
            self.release = threading.Event()

        def predict(self, frame, imgsz, device, verbose):
            self.started.set()
            assert self.release.wait(5)
            return super().predict(frame, imgsz, device, verbose)

    model = GatedModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert model.started.wait(5)  # 推理在飞

    stopper = threading.Thread(target=w.stop, args=(3000,))
    stopper.start()
    time.sleep(0.1)  # 让 stop 先置 _running=False 并进入 wait
    model.release.set()
    stopper.join(5)
    assert not stopper.is_alive()
    assert not w.isRunning()
    assert spies.depth.count == 0  # 停止后即使推理完成也不发新深度帧


# ----------------------------------------------------------------------
# stop 语义
# ----------------------------------------------------------------------


def test_stop_idempotent_and_safe_before_start(qapp):
    w = DepthWorker(device="cpu")
    w.stop()  # 未 start 时安全
    w.stop()
    w.set_enabled(True)
    w.register_slot(0)
    w.start()
    assert wait_until(w.isRunning)
    w.stop(3000)
    assert not w.isRunning()
    w.stop(3000)  # 重复 stop 安全
    assert not w.isRunning()
