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

from ai.depth_worker import (
    DEFAULT_MODEL_PATH,
    DepthWorker,
    colorize_depth,
    filter_depth_ema,
    update_adaptive_range,
)
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
    spies = SimpleNamespace(depth=SignalSpy(), error=SignalSpy(), range=SignalSpy())
    w.depth_ready.connect(spies.depth, Qt.ConnectionType.DirectConnection)
    w.depth_range_ready.connect(spies.range, Qt.ConnectionType.DirectConnection)
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
    out = colorize_depth(depth, vmin=0.0, vmax=20.0, mode="metric")
    assert out.shape == (24, 32, 3)
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]

    # 默认 Turbo 色带；与手工 BGR→RGB 参考实现逐像素一致（验证确实做了 BGR→RGB）
    gray = np.clip(depth / 20.0, 0, 1)
    gray = (gray * 255.0).round().astype(np.uint8)
    ref = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
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


def test_colorize_posinf_maps_to_far_low_end():
    """无效值映射远处低端；有限近处值才映射近处暖色高端。"""
    low = colorize_depth(np.zeros((1, 1), dtype=np.float32))[0, 0]
    high = colorize_depth(np.full((1, 1), 1.0, dtype=np.float32))[0, 0]
    out = colorize_depth(np.array([[np.inf, 1.0]], dtype=np.float32))
    np.testing.assert_array_equal(out[0, 0], low)   # +Inf → 低端
    np.testing.assert_array_equal(out[0, 1], high)  # 近处 → 高端
    assert not np.array_equal(out[0, 0], out[0, 1])


def test_disparity_default_near_warm_far_cool_with_gamma():
    import cv2

    depth = np.array([[1.0, 4.0, 20.0]], dtype=np.float32)
    out = colorize_depth(depth, vmin=0.0, vmax=20.0, cmap="turbo")

    disparity_far = 1.0 / 20.0
    disparity_near = 1.0  # 固定 0–20 m 的显示近端 = 5% × 20 = 1 m，避免 1/0 奇点
    normalized = ((0.25 - disparity_far) / (disparity_near - disparity_far)) ** 0.5
    gray = np.array([[255, round(float(normalized) * 255), 0]], dtype=np.uint8)
    expected = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    np.testing.assert_array_equal(out, expected)
    with pytest.raises(ValueError):
        colorize_depth(depth, gamma=0.0)


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


def test_range_change_during_inference_suppresses_stale_frame(worker):
    """量程切换时，在飞旧帧既不写状态，也不得发给 GUI 造成瞬时闪帧。"""
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

    w.set_visualization("turbo", "0-3")
    model.release.set()
    time.sleep(0.3)
    assert spies.depth.count == 0
    assert spies.range.count == 0

    w.submit_frame(0, make_frame(11))
    assert wait_until(lambda: spies.depth.count == 1)
    assert spies.range.calls[-1][1:] == (0.0, 3.0)


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

# ----------------------------------------------------------------------
# 运动自适应 EMA 滤波（纯函数）
# ----------------------------------------------------------------------


def test_filter_first_frame_uses_current():
    cur = np.full((4, 4), 5.0, dtype=np.float32)
    out = filter_depth_ema(None, cur)
    np.testing.assert_allclose(out, cur)
    assert out.dtype == np.float32


def test_filter_static_flicker_damped():
    """静态像素 ±0.1m 闪烁（阈值内）被 alpha=0.25 显著衰减。"""
    base = np.full((8, 8), 5.0, dtype=np.float32)
    state = filter_depth_ema(None, base)
    outs = []
    for i in range(8):
        cur = base + (0.1 if i % 2 == 0 else -0.1)
        state = filter_depth_ema(state, cur)
        outs.append(float(state[0, 0]))
    # 输入在 4.9/5.1 间跳变（峰峰 0.2），输出峰峰应远小于输入
    assert max(outs) - min(outs) < 0.05
    assert 4.9 < min(outs) <= max(outs) < 5.1


def test_filter_motion_pixel_fast_follow():
    """5m 跳变超过阈值 → 运动像素 alpha=0.85 快速跟随。"""
    prev = np.full((4, 4), 5.0, dtype=np.float32)
    cur = np.full((4, 4), 10.0, dtype=np.float32)
    out = filter_depth_ema(prev, cur)
    np.testing.assert_allclose(out, 5.0 + 0.85 * 5.0, rtol=1e-6)


def test_filter_motion_threshold_relative():
    """相对阈值：20m 处 1m 跳变（阈值 3m）视为静态，alpha=0.25。"""
    prev = np.full((2, 2), 20.0, dtype=np.float32)
    cur = np.full((2, 2), 21.0, dtype=np.float32)
    out = filter_depth_ema(prev, cur)
    np.testing.assert_allclose(out, 20.0 + 0.25 * 1.0, rtol=1e-6)


def test_filter_invalid_and_new_pixels():
    """当前无效不得沿用旧深度；新出现/上帧无效直接采用当前值。"""
    prev = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    cur = np.array([[np.nan, 2.0], [-1.0, 4.1]], dtype=np.float32)
    out = filter_depth_ema(prev, cur)
    assert np.isnan(out[0, 0])  # 当前无效：输出 NaN，不沿用旧深度 1.0
    assert out[0, 1] == 2.0     # 新出现（上帧无效）→ 直接采用当前值
    assert np.isnan(out[1, 0])  # cur<=0 视为无效
    np.testing.assert_allclose(out[1, 1], 4.0 + 0.25 * 0.1, rtol=1e-5)  # 静态 EMA


def test_filter_shape_change_resets():
    prev = np.full((4, 4), 5.0, dtype=np.float32)
    cur = np.full((2, 8), 7.0, dtype=np.float32)
    out = filter_depth_ema(prev, cur)
    np.testing.assert_allclose(out, cur)


# ----------------------------------------------------------------------
# 稳定自适应量程（纯函数）
# ----------------------------------------------------------------------


def test_auto_range_first_frame_uses_target_and_min_span():
    filtered = np.full((10, 10), 5.0, dtype=np.float32)
    vmin, vmax = update_adaptive_range(None, filtered)
    # 恒定深度：目标跨度不足 0.5m，以中点扩到最小量程
    assert vmin == pytest.approx(5.0 - 0.25)
    assert vmax == pytest.approx(5.0 + 0.25)
    assert vmax - vmin >= 0.5


def test_auto_range_ignores_outliers():
    filtered = np.full((100, 100), 5.0, dtype=np.float32)
    filtered[0, :50] = 100.0  # 0.5% 远距野点，不应支配 98% 百分位
    vmin, vmax = update_adaptive_range(None, filtered)
    assert vmax < 10.0
    assert vmin < vmax


def test_auto_range_slow_ema_toward_target():
    prev = (0.0, 10.0)
    filtered = np.full((10, 10), 20.0, dtype=np.float32)  # 目标 (19.75, 20.25)
    vmin, vmax = update_adaptive_range(prev, filtered, alpha=0.05)
    assert vmin == pytest.approx(0.0 + 0.05 * 19.75)
    assert vmax == pytest.approx(10.0 + 0.05 * (20.25 - 10.0))


def test_auto_range_no_valid_keeps_previous():
    filtered = np.full((4, 4), np.nan, dtype=np.float32)
    assert update_adaptive_range((1.0, 2.0), filtered) == (1.0, 2.0)
    assert update_adaptive_range(None, filtered) == (0.0, 20.0)


# ----------------------------------------------------------------------
# 可视化设置（cmap / range_mode）
# ----------------------------------------------------------------------


def test_colorize_default_cmap_is_turbo():
    depth = np.linspace(0.0, 20.0, 64, dtype=np.float32).reshape(8, 8)
    np.testing.assert_array_equal(
        colorize_depth(depth), colorize_depth(depth, cmap="turbo")
    )
    assert not np.array_equal(
        colorize_depth(depth), colorize_depth(depth, cmap="inferno")
    )


def test_colorize_supported_cmaps_match_cv2_reference():
    import cv2

    depth = np.linspace(0.0, 20.0, 64, dtype=np.float32).reshape(8, 8)
    gray = (np.clip(depth / 20.0, 0, 1) * 255.0).round().astype(np.uint8)
    for name in ("turbo", "inferno", "plasma", "viridis"):
        out = colorize_depth(depth, cmap=name, mode="metric")
        ref = cv2.cvtColor(
            cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{name.upper()}")),
            cv2.COLOR_BGR2RGB,
        )
        np.testing.assert_array_equal(out, ref)


def test_colorize_unknown_cmap_rejected():
    with pytest.raises(ValueError):
        colorize_depth(np.zeros((4, 4), dtype=np.float32), cmap="bogus")


def test_visualization_defaults_and_validation(qapp):
    w = DepthWorker(device="cpu")
    assert w._cmap == "turbo"
    assert w._range_mode == "auto"
    with pytest.raises(ValueError):
        DepthWorker(device="cpu", cmap="bogus")
    with pytest.raises(ValueError):
        DepthWorker(device="cpu", range_mode="0-5")
    with pytest.raises(ValueError):
        w.set_visualization("bogus", "auto")
    with pytest.raises(ValueError):
        w.set_visualization("turbo", "0-5")
    # 非法调用不得改变现有设置
    assert w._cmap == "turbo" and w._range_mode == "auto"
    for cmap in ("turbo", "inferno", "plasma", "viridis"):
        for mode in ("auto", "0-3", "0-8", "0-20"):
            w.set_visualization(cmap, mode)
    assert w._cmap == "viridis" and w._range_mode == "0-20"


# ----------------------------------------------------------------------
# worker 集成：滤波 / 量程 / 信号
# ----------------------------------------------------------------------


def test_depth_range_ready_emitted_before_depth_ready(worker):
    w, spies = worker
    order = SignalSpy()
    w.depth_range_ready.connect(
        lambda s, lo, hi: order("range", s, lo, hi), Qt.ConnectionType.DirectConnection
    )
    w.depth_ready.connect(
        lambda s, img: order("depth", s), Qt.ConnectionType.DirectConnection
    )
    install_fake_model(w, FakeModel())
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: order.count >= 2)
    kinds = [c[0] for c in order.calls]
    assert kinds == ["range", "depth"] * (len(kinds) // 2)
    # auto 模式恒定深度 10 → 首帧直接采用目标 (9.75, 10.25)
    slot_id, vmin, vmax = order.calls[0][1], order.calls[0][2], order.calls[0][3]
    assert slot_id == 0
    assert vmin == pytest.approx(9.75)
    assert vmax == pytest.approx(10.25)


def test_fixed_range_modes_emit_exact_range(worker):
    w, spies = worker
    install_fake_model(w, FakeModel())
    w.set_visualization("turbo", "0-8")
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.depth.count >= 1)
    assert spies.range.calls[0] == (0, 0.0, 8.0)  # 固定量程，无平滑

    w.set_visualization("turbo", "0-3")
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.depth.count >= 2)
    assert spies.range.calls[-1] == (0, 0.0, 3.0)


def test_motion_filter_applied_before_colorize(worker):
    """固定量程下，跳变帧颜色应对应滤波后的深度（非原始值、非彩色混合）。"""
    w, spies = worker
    install_fake_model(w, FakeModel())
    w.set_visualization("turbo", "0-20")
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.depth.count == 1)
    w.submit_frame(0, make_frame(16))  # 跳变 6m > 阈值 → alpha=0.85
    assert wait_until(lambda: spies.depth.count == 2)
    filtered = filter_depth_ema(
        np.full((32, 48), 10.0, dtype=np.float32),
        np.full((32, 48), 16.0, dtype=np.float32),
    )
    expected = colorize_depth(filtered, vmin=0.0, vmax=20.0, cmap="turbo")
    np.testing.assert_array_equal(spies.depth.calls[1][1], expected)
    raw_next = colorize_depth(
        np.full((32, 48), 16.0, dtype=np.float32), vmin=0.0, vmax=20.0, cmap="turbo"
    )
    assert not np.array_equal(spies.depth.calls[1][1], raw_next)


def test_disable_clears_filter_and_range_state(worker):
    w, spies = worker
    install_fake_model(w, FakeModel())
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.depth.count >= 1)
    assert 0 in w._depth_state and 0 in w._range_state
    w.set_enabled(False)
    assert w._depth_state == {} and w._range_state == {}


def test_unregister_clears_only_that_slot_state(worker):
    w, spies = worker
    install_fake_model(w, FakeModel())
    w.register_slot(0)
    w.register_slot(1)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    w.submit_frame(1, make_frame(12))
    assert wait_until(lambda: spies.depth.count >= 2)
    assert 0 in w._depth_state and 1 in w._depth_state
    w.unregister_slot(0)
    assert 0 not in w._depth_state and 0 not in w._range_state
    assert 1 in w._depth_state and 1 in w._range_state  # 其他 slot 状态保留


def test_range_mode_switch_clears_range_but_keeps_depth_state(worker):
    w, spies = worker
    install_fake_model(w, FakeModel())
    w.register_slot(0)
    w.set_enabled(True)
    w.start()
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.depth.count >= 1)
    assert 0 in w._depth_state and 0 in w._range_state

    # cmap 切换：不重置任何状态
    w.set_visualization("plasma", "auto")
    assert 0 in w._depth_state and 0 in w._range_state

    # 量程模式切换：清量程状态，保留深度滤波状态
    w.set_visualization("plasma", "0-3")
    assert w._range_state == {}
    assert 0 in w._depth_state
# ----------------------------------------------------------------------
# per-slot 代际：unregister / cmap 切换的在飞帧失效语义
# ----------------------------------------------------------------------


class SwitchableGatedModel(FakeModel):
    """block=True 时 predict 阻塞直到 release；用于确定性地构造在飞帧。"""

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False

    def predict(self, frame, imgsz, device, verbose):
        if self.block:
            self.started.set()
            assert self.release.wait(5)
        return super().predict(frame, imgsz, device, verbose)


def test_unregister_other_slot_keeps_inflight_frame_valid(worker):
    """slot0 unregister 只使本 slot 在飞帧失效，不得误伤 slot1 的在飞帧。"""
    w, spies = worker
    model = SwitchableGatedModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.register_slot(1)
    w.set_enabled(True)
    w.start()

    model.block = True
    w.submit_frame(1, make_frame(12))
    assert model.started.wait(5)  # slot1 推理在飞

    w.unregister_slot(0)  # 仅影响 slot0 的代际
    model.release.set()

    assert wait_until(lambda: spies.depth.count >= 1)
    assert spies.depth.calls[0][0] == 1  # slot1 在飞帧正常发出
    assert spies.range.calls[0][0] == 1


def test_cmap_switch_invalidates_inflight_frame_and_keeps_filter_state(worker):
    """cmap 在飞切换：旧色带帧不发出、不写状态；已完成滤波状态保留；下一帧用新色带。"""
    w, spies = worker
    model = SwitchableGatedModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()

    # 第 1 帧完成，建立深度滤波与量程状态
    w.submit_frame(0, make_frame(10))
    assert wait_until(lambda: spies.depth.count == 1)
    assert 0 in w._depth_state
    assert w._range_state[0] == pytest.approx((9.75, 10.25))

    # 第 2 帧在飞时切换 cmap
    model.block = True
    w.submit_frame(0, make_frame(12))
    assert model.started.wait(5)
    w.set_visualization("plasma", "auto")
    # cmap 切换保留已完成的深度滤波与量程状态
    assert 0 in w._depth_state
    assert 0 in w._range_state
    model.release.set()
    time.sleep(0.3)
    # 旧色带在飞帧既不发出，其滤波结果也不写入状态
    assert spies.depth.count == 1
    np.testing.assert_allclose(
        w._depth_state[0], np.full((32, 48), 10.0, dtype=np.float32)
    )

    # 下一帧使用新色带，且基于保留的滤波状态（prev=10，跳变 4m → alpha=0.85）
    model.block = False
    w.submit_frame(0, make_frame(14))
    assert wait_until(lambda: spies.depth.count == 2)
    filtered = filter_depth_ema(
        np.full((32, 48), 10.0, dtype=np.float32),
        np.full((32, 48), 14.0, dtype=np.float32),
    )
    vmin, vmax = update_adaptive_range((9.75, 10.25), filtered)
    expected = colorize_depth(filtered, vmin=vmin, vmax=vmax, cmap="plasma")
    np.testing.assert_array_equal(spies.depth.calls[1][1], expected)
    assert spies.range.calls[-1][1] == pytest.approx(vmin)
    assert spies.range.calls[-1][2] == pytest.approx(vmax)
    stale = colorize_depth(filtered, vmin=vmin, vmax=vmax, cmap="turbo")
    assert not np.array_equal(spies.depth.calls[1][1], stale)  # 确为新色带


def test_unregister_reregister_same_slot_still_drops_stale_frame(worker):
    """快速 unregister→register 同 slot：代际 tombstone 保证旧在飞帧仍被丢弃。"""
    w, spies = worker
    model = SwitchableGatedModel()
    install_fake_model(w, model)
    w.register_slot(0)
    w.set_enabled(True)
    w.start()

    model.block = True
    w.submit_frame(0, make_frame(10))
    assert model.started.wait(5)  # slot0 推理在飞

    w.unregister_slot(0)
    w.register_slot(0)  # 立即重新注册（换相机 / 重开场景）
    model.release.set()
    time.sleep(0.3)
    assert spies.depth.count == 0  # 旧帧不得因代际重置而复生
    assert 0 not in w._depth_state

    model.block = False
    w.submit_frame(0, make_frame(11))
    assert wait_until(lambda: spies.depth.count == 1)
    assert spies.depth.calls[0][0] == 0


def test_set_visualization_same_values_is_noop(qapp):
    """同值 set_visualization 为 no-op：不改设置、不动代际。"""
    w = DepthWorker(device="cpu")
    w.register_slot(0)
    gen_before = dict(w._slot_gen)
    w.set_visualization("turbo", "auto")  # 与当前默认完全相同
    assert w._cmap == "turbo" and w._range_mode == "auto"
    assert dict(w._slot_gen) == gen_before


def test_cmap_change_bumps_generations_but_keeps_state(qapp):
    """cmap 变化递增全部已知 slot 代际（含 tombstone），保留深度/量程状态。"""
    w = DepthWorker(device="cpu")
    w.register_slot(0)
    w.register_slot(1)
    w.unregister_slot(0)  # slot0 代际 tombstone 为 1
    w._depth_state[1] = np.full((2, 2), 5.0, dtype=np.float32)
    w._range_state[1] = (1.0, 2.0)

    w.set_visualization("inferno", "auto")
    assert w._slot_gen[0] == 2 and w._slot_gen[1] == 1
    assert 1 in w._depth_state
    assert w._range_state[1] == (1.0, 2.0)

    # range_mode 变化：清量程状态、保留深度滤波状态、代际继续递增
    w.set_visualization("inferno", "0-3")
    assert w._slot_gen[0] == 3 and w._slot_gen[1] == 2
    assert w._range_state == {}
    assert 1 in w._depth_state


# ----------------------------------------------------------------------
# disparity 着色优化：与参考公式逐像素一致
# ----------------------------------------------------------------------


def _reference_disparity_colorize(depth, vmin, vmax, cmap="turbo", gamma=0.5):
    """优化前 disparity 着色参考实现（布尔花式索引版），用于逐像素比对。"""
    import cv2

    arr = np.asarray(depth, dtype=np.float32)
    far = float(vmax)
    near = 0.05 * far if float(vmin) <= 0.0 else float(vmin)
    valid = np.isfinite(arr) & (arr > 0.0)
    disparity_far = 1.0 / far
    disparity_near = 1.0 / near
    norm = np.zeros_like(arr, dtype=np.float32)
    disparity = 1.0 / arr[valid]
    normalized = (disparity - disparity_far) / (disparity_near - disparity_far)
    norm[valid] = np.power(np.clip(normalized, 0.0, 1.0), gamma)
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    gray = (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    bgr = cv2.applyColorMap(gray, getattr(cv2, f"COLORMAP_{cmap.upper()}"))
    return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def test_disparity_colorize_bitexact_vs_reference_formula():
    """就地优化的 disparity 着色必须与参考公式逐像素一致（含 NaN/±Inf/极端值）。"""
    rng = np.random.default_rng(20261128)
    for _ in range(20):
        shape = (int(rng.integers(4, 64)), int(rng.integers(4, 64)))
        depth = rng.uniform(0.01, 30.0, size=shape).astype(np.float32)
        flat = depth.ravel()
        flat[0] = np.nan
        flat[1 % flat.size] = np.inf
        flat[2 % flat.size] = -np.inf
        flat[3 % flat.size] = 0.0
        flat[4 % flat.size] = -2.0
        flat[5 % flat.size] = 1e-38  # 极大 disparity（float32 上溢边界附近）
        for vmin, vmax in ((0.0, 20.0), (0.0, 3.0), (2.0, 8.0)):
            out = colorize_depth(depth, vmin=vmin, vmax=vmax, cmap="turbo")
            ref = _reference_disparity_colorize(depth, vmin, vmax, cmap="turbo")
            np.testing.assert_array_equal(out, ref)
    # 全无效 / 全零边界
    for depth in (
        np.full((8, 8), np.nan, dtype=np.float32),
        np.zeros((8, 8), dtype=np.float32),
        np.full((8, 8), np.inf, dtype=np.float32),
    ):
        np.testing.assert_array_equal(
            colorize_depth(depth), _reference_disparity_colorize(depth, 0.0, 20.0)
        )
