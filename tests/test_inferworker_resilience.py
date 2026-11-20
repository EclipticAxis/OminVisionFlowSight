"""InferWorker 单槽位单帧异常防火带测试（生产 InferWorker.run 级）。

针对 ``ai.inference.InferWorker`` 的真实 QThread run() 循环，验证：
    - 首帧异常不 emit detection_ready、不调用 _bypass_update_targets，线程存活
    - 第二帧成功正常 emit，error_occurred / processing_recovered 各上报一次
    - 失败后保留最后有效结果（_last_detections 不被异常帧覆盖）
    - 单槽位失败不妨碍同批其他槽位
    - 失败上报按 1/30 限频（第 1、30、60 ... 次）
    - unregister/register 同步清理失败计数
    - 线程可正常 stop
    - TargetManager 单写契约（第 4 步接缝对齐）：Pipeline TargetStage 成功时
      总 update 恰好一次（而非两次）；Legacy / Pipeline 前置失败路径仍经
      run 尾部 bypass 恰好写一次，保留过期清理语义

确定性保障：monkeypatch 禁掉 _initialize_backend/_warmup（不加载模型），
patch _process_slot_frame 注入异常；等待全部采用有界轮询（处理事件 +
短 sleep），禁止无界 sleep。不触碰 orphan inference_controller。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# 复用契约测试的 LegacyScenario / build_worker（TargetManager 单写证明）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtWidgets import QApplication

from ai.inference import InferWorker


FRAME = np.zeros((8, 8, 3), dtype=np.uint8)
GOOD_DET = {"label": "person", "conf": 0.9, "bbox": (0.1, 0.1, 0.5, 0.5)}


class SignalSpy:
    """极简信号记录器（等价 QSignalSpy）。"""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, *args) -> None:
        self.calls.append(args)

    @property
    def count(self) -> int:
        return len(self.calls)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def worker(qapp, monkeypatch):
    """构造不加载模型的真实 InferWorker（backend=cpu 跳过设备探测）。"""
    monkeypatch.setattr(
        InferWorker, "_initialize_backend", lambda self, model_path=None: None
    )
    monkeypatch.setattr(InferWorker, "_warmup", lambda self: None)
    w = InferWorker(model_path=None, conf=0.25, infer_stride=1, backend="cpu")
    yield w
    if w.isRunning():
        w.stop()
    w.wait(3000)


def _wait_until(predicate, timeout_s: float = 5.0) -> bool:
    """有界轮询：处理 Qt 事件 + 短 sleep，直到 predicate 为真或超时。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.002)
    return predicate()


def _connect_spies(w: InferWorker) -> dict[str, SignalSpy]:
    spies = {
        "detection": SignalSpy(),
        "error": SignalSpy(),
        "recovered": SignalSpy(),
    }
    w.detection_ready.connect(spies["detection"])
    w.error_occurred.connect(spies["error"])
    w.processing_recovered.connect(spies["recovered"])
    return spies


def _patch_frame_processing(w: InferWorker, monkeypatch, fake) -> None:
    """以实例属性替换 _process_slot_frame（run() 经 self. 调用，天然命中）。"""
    monkeypatch.setattr(w, "_process_slot_frame", fake)


def _patch_bypass(w: InferWorker, monkeypatch) -> SignalSpy:
    spy = SignalSpy()

    def _bypass(slot_id, detections, timestamp):
        spy(slot_id, detections, timestamp)

    monkeypatch.setattr(w, "_bypass_update_targets", _bypass)
    return spy


# ----------------------------------------------------------------------
# 用例
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# TargetManager 单写契约（第 4 步接缝对齐，run 级证明）
# ----------------------------------------------------------------------

def _build_pipeline_worker(monkeypatch, scenario, pipeline_enabled):
    """经契约测试的 build_worker 构造双模式 worker，并禁 warmup。"""
    from test_inferworker_pipeline_contract import build_worker

    monkeypatch.setattr(InferWorker, "_warmup", lambda self: None)
    return build_worker(
        monkeypatch, scenario, pipeline_enabled=pipeline_enabled,
    )


def _spy_target_updates(worker, monkeypatch):
    """包装 TargetManager.update_targets（类级 patch，slotted 类），返回调用记录。"""
    calls: list = []
    mgr_cls = type(worker._target_mgr)
    real_update = mgr_cls.update_targets

    def _spy(self, tracks, **kwargs):
        calls.append((len(tracks), kwargs.get("slot_id")))
        return real_update(self, tracks, **kwargs)

    monkeypatch.setattr(mgr_cls, "update_targets", _spy)
    return calls


def _pipeline_scenario():
    from test_inferworker_pipeline_contract import LegacyScenario, person_det

    return LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det()],
    )


def test_pipeline_target_stage_success_updates_target_manager_once(qapp, monkeypatch):
    """Pipeline TargetStage 成功：TargetManager 总 update 恰好一次（非两次）。"""
    env = _build_pipeline_worker(monkeypatch, _pipeline_scenario(), True)
    worker = env.worker
    calls = _spy_target_updates(worker, monkeypatch)
    spies = _connect_spies(worker)

    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count == 1), "成功帧未 emit"
        # TargetStage 已完成 update -> run 尾部 bypass 必须跳过：总共一次。
        assert len(calls) == 1, (
            f"TargetStage 成功后 TargetManager 应单写一次，实际 {len(calls)} 次"
        )
        assert calls[0][1] == 0
    finally:
        worker.stop()


def test_pipeline_failure_before_target_stage_still_bypasses_once(qapp, monkeypatch):
    """Pipeline 在 TargetStage 前失败返回 []：bypass 仍执行一次（过期清理）。"""
    env = _build_pipeline_worker(monkeypatch, _pipeline_scenario(), True)
    worker = env.worker

    class _BoomTracker:
        def update(self, *args, **kwargs):
            raise RuntimeError("tracker boom")

        def update_appearance(self, frame):
            pass

        def predict_step(self):
            return []

        def set_redetect_config(self, **kwargs):
            pass

        def reset(self):
            pass

    worker._trackers[0] = _BoomTracker()
    calls = _spy_target_updates(worker, monkeypatch)
    spies = _connect_spies(worker)

    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count == 1), "失败帧应 emit []"
        # Pipeline 返回 []（不抛异常出 run），TargetStage 未完成 update
        # -> run 尾部 bypass 执行一次过期清理写。
        assert len(calls) == 1, (
            f"前置失败路径应经 bypass 单写一次，实际 {len(calls)} 次"
        )
        assert spies["detection"].calls[0][1] == []
    finally:
        worker.stop()


def test_pipeline_event_stage_failure_after_target_stage_stays_single_write(
    qapp, monkeypatch,
):
    """TargetStage 后（EventStage）失败：update 已完成，仍保持单写。"""
    env = _build_pipeline_worker(monkeypatch, _pipeline_scenario(), True)
    worker = env.worker

    def _boom(self, event):
        raise RuntimeError("event bus boom")

    monkeypatch.setattr(type(worker._event_bus), "publish", _boom)
    calls = _spy_target_updates(worker, monkeypatch)
    spies = _connect_spies(worker)

    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count == 1), "失败帧应 emit []"
        # TargetStage 的 update 在 EventStage 失败前已完成 -> 跳过 bypass。
        assert len(calls) == 1, (
            f"TargetStage 已完成的帧应保持单写，实际 {len(calls)} 次"
        )
        assert spies["detection"].calls[0][1] == []
    finally:
        worker.stop()


def test_legacy_mode_bypass_updates_target_manager_once(qapp, monkeypatch):
    """Legacy 模式：run 尾部 bypass 恰好写一次 TargetManager。"""
    env = _build_pipeline_worker(monkeypatch, _pipeline_scenario(), False)
    worker = env.worker
    calls = _spy_target_updates(worker, monkeypatch)
    spies = _connect_spies(worker)

    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count == 1), "成功帧未 emit"
        assert len(calls) == 1, (
            f"Legacy bypass 应单写一次，实际 {len(calls)} 次"
        )
        assert calls[0][1] == 0
    finally:
        worker.stop()


def test_pipeline_postprocess_failure_does_not_suppress_next_legacy_bypass(
    qapp, monkeypatch,
):
    """TargetStage 后处理异常留下的标志不得抑制下一 Legacy 帧的 bypass。"""
    env = _build_pipeline_worker(monkeypatch, _pipeline_scenario(), True)
    worker = env.worker
    calls = _spy_target_updates(worker, monkeypatch)
    spies = _connect_spies(worker)

    def _fail_after_target_update(slot_id, frame_id, submitted_ts, frame):
        # 模拟 TargetStage 已写入，但后续头部/手势/健康处理抛出异常。
        worker._pipeline_target_updated = True
        raise RuntimeError("post-target processing boom")

    monkeypatch.setattr(worker, "_run_pipeline_mode", _fail_after_target_update)
    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["error"].count == 1)
        assert calls == []  # 异常帧本身不 emit、不 bypass
        assert worker._pipeline_target_updated is True

        worker.set_pipeline_enabled(False)
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count == 1)
        assert len(calls) == 1, (
            "上一帧遗留的单写标志不得抑制 Legacy bypass"
        )
        assert calls[0][1] == 0
    finally:
        worker.stop()


def test_first_frame_failure_then_success(qapp, worker, monkeypatch):
    """首帧异常跳过 emit/bypass，第二帧成功恢复，error/recovered 各一次。"""
    state = {"calls": 0}
    bypass = _patch_bypass(worker, monkeypatch)
    spies = _connect_spies(worker)

    def fake(slot_id, frame_id, submitted_ts, frame):
        state["calls"] += 1
        if state["calls"] == 1:
            raise RuntimeError("boom")
        return [dict(GOOD_DET)]

    _patch_frame_processing(worker, monkeypatch, fake)
    worker.register_slot(0)
    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["error"].count == 1), "首帧异常未上报"
        # 异常帧：不 emit detection_ready，不调用 bypass
        assert spies["detection"].count == 0
        assert bypass.count == 0
        assert worker.isRunning()

        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count == 1), "成功帧未 emit"
        assert spies["detection"].calls[0][0] == 0
        assert spies["detection"].calls[0][1] == [GOOD_DET]
        assert bypass.count == 1
        assert _wait_until(lambda: spies["recovered"].count == 1), "恢复未上报"
        assert spies["error"].count == 1  # 限频：不重复上报

        slot_id, message = spies["error"].calls[0]
        assert slot_id == 0
        assert "1" in message and "boom" in message  # slot/frame/count/异常
    finally:
        worker.stop()
    assert not worker.isRunning()


def test_failure_keeps_last_good_result(qapp, worker, monkeypatch):
    """连续异常帧不得覆盖 _last_detections 中最后有效结果。"""
    state = {"calls": 0}
    spies = _connect_spies(worker)

    def fake(slot_id, frame_id, submitted_ts, frame):
        # 模拟真实 _process_slot_frame 的契约：成功路径末尾写
        # _last_detections；异常帧在写入前抛出，不得覆盖旧值。
        state["calls"] += 1
        if state["calls"] == 1:
            detections = [dict(GOOD_DET)]
            worker._last_detections[slot_id] = detections
            return detections
        raise RuntimeError("pipeline broken")

    _patch_frame_processing(worker, monkeypatch, fake)
    worker.register_slot(0)
    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: worker._last_detections.get(0) == [GOOD_DET])

        worker.submit_frame(0, FRAME)  # 第 2 帧失败
        assert _wait_until(lambda: worker._slot_failure_counts.get(0) == 1)
        # 失败帧不更新 _last_detections：仍保留最后有效结果
        assert worker._last_detections[0] == [GOOD_DET]

        worker.submit_frame(0, FRAME)  # 第 3 帧失败
        assert _wait_until(lambda: worker._slot_failure_counts.get(0) == 2)
        assert worker._last_detections[0] == [GOOD_DET]
    finally:
        worker.stop()
    assert not worker.isRunning()


def test_slot_failure_does_not_block_other_slots(qapp, worker, monkeypatch):
    """同批一个槽位持续失败，不影响其他槽位的正常检测。"""
    spies = _connect_spies(worker)

    def fake(slot_id, frame_id, submitted_ts, frame):
        if slot_id == 0:
            raise ValueError("slot0 broken")
        return [{"label": "person", "conf": 0.8, "slot": slot_id}]

    _patch_frame_processing(worker, monkeypatch, fake)
    worker.register_slot(0)
    worker.register_slot(1)
    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        worker.submit_frame(1, FRAME)
        assert _wait_until(
            lambda: spies["detection"].count >= 1 and spies["error"].count == 1
        ), "槽位 1 的成功帧被槽位 0 异常阻塞"
        det_slot = spies["detection"].calls[0][0]
        assert det_slot == 1
        assert spies["error"].calls[0][0] == 0

        # 槽位 1 继续正常工作，线程存活
        worker.submit_frame(1, FRAME)
        assert _wait_until(lambda: spies["detection"].count >= 2)
        assert worker.isRunning()
        assert worker._slot_failure_counts.get(1) is None
    finally:
        worker.stop()
    assert not worker.isRunning()


def test_error_report_throttled_at_1_and_30(qapp, worker, monkeypatch):
    """第 1 次及每第 30 次连续失败才上报；成功后恢复上报一次。"""
    spies = _connect_spies(worker)

    def fake(slot_id, frame_id, submitted_ts, frame):
        raise RuntimeError("always failing")

    _patch_frame_processing(worker, monkeypatch, fake)
    worker.register_slot(0)
    worker.start()
    try:
        for expected in range(1, 31):
            worker.submit_frame(0, FRAME)
            assert _wait_until(
                lambda e=expected: worker._slot_failure_counts.get(0, 0) >= e
            ), f"第 {expected} 次失败未被消费"
        assert _wait_until(lambda: spies["error"].count == 2), (
            f"限频失败：error 次数 = {spies['error'].count}，应为 2"
        )
        msg1, msg30 = spies["error"].calls[0][1], spies["error"].calls[1][1]
        assert "连续第 1 次" in msg1
        assert "连续第 30 次" in msg30
        assert spies["recovered"].count == 0

        # 成功一帧 → 清零并上报恢复
        state = {"ok": False}

        def good_fake(slot_id, frame_id, submitted_ts, frame):
            state["ok"] = True
            return [dict(GOOD_DET)]

        monkeypatch.setattr(worker, "_process_slot_frame", good_fake)
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["recovered"].count == 1), "恢复未上报"
        assert worker._slot_failure_counts.get(0) is None
    finally:
        worker.stop()
    assert not worker.isRunning()


def test_unregister_clears_failure_count(qapp, worker, monkeypatch):
    """unregister/register 后失败计数清零，重新从第 1 次计。"""
    spies = _connect_spies(worker)

    def fake(slot_id, frame_id, submitted_ts, frame):
        raise RuntimeError("flaky")

    _patch_frame_processing(worker, monkeypatch, fake)
    worker.register_slot(0)
    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["error"].count == 1)
        assert "连续第 1 次" in spies["error"].calls[0][1]
        assert worker._slot_failure_counts.get(0) == 1

        worker.unregister_slot(0)
        assert worker._slot_failure_counts.get(0) is None

        worker.register_slot(0)
        assert worker._slot_failure_counts.get(0) is None
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["error"].count == 2)
        assert "连续第 1 次" in spies["error"].calls[1][1], "计数未清零"
        assert spies["recovered"].count == 0, "unregister 不应触发恢复上报"
    finally:
        worker.stop()
    assert not worker.isRunning()


def test_thread_starts_and_stops_cleanly(qapp, worker, monkeypatch):
    """无异常路径下线程可正常启动、处理并 stop。"""
    spies = _connect_spies(worker)
    _patch_frame_processing(
        worker, monkeypatch, lambda slot_id, frame_id, ts, frame: [dict(GOOD_DET)]
    )
    worker.register_slot(0)
    worker.start()
    try:
        assert worker.isRunning()
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count == 1)
        assert spies["error"].count == 0
        assert spies["recovered"].count == 0
    finally:
        worker.stop()
    assert not worker.isRunning()
    worker.wait(1000)
    assert not worker.isRunning()


# ----------------------------------------------------------------------
# run 级双槽位信号契约（第 4 步）：detection_ready 首参数为 slot_id
# ----------------------------------------------------------------------


@pytest.mark.parametrize("pipeline_enabled", [False, True])
def test_detection_ready_slot_id_matches_submitting_slot(qapp, monkeypatch, pipeline_enabled):
    """双槽位提交：detection_ready 首参数必须是 slot_id（不是 frame_id），
    且每个结果与提交槽位严格对应（Legacy 同契约不回归）。

    判别设计：每槽位首帧 frame_id 均为 1——若信号误传 frame_id，槽位 0 的
    首轮发射首参数会是 1 而非 0；两轮内容以 track_id 区分，帧内容以
    全 0 / 全 1 ndarray 区分，杜绝槽位串扰。
    """
    from test_inferworker_pipeline_contract import (
        FakeRectangleTracker,
        FakeTracker,
        LegacyScenario,
        build_worker,
        person_det,
    )

    events: list = []
    scen0 = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det(track_id=10)],
    )
    scen1 = LegacyScenario(
        raw_detections=[person_det()],
        tracker_output=[person_det(track_id=20)],
    )
    env = build_worker(
        monkeypatch, scen0, pipeline_enabled=pipeline_enabled, events=events,
    )
    worker = env.worker
    monkeypatch.setattr(InferWorker, "_warmup", lambda self: None)

    worker.register_slot(1)
    tracker1 = FakeTracker(scen1, events)
    worker._trackers[1] = tracker1
    worker._rectangle_trackers[1] = FakeRectangleTracker()

    spies = _connect_spies(worker)
    frame0 = np.zeros((8, 8, 3), dtype=np.uint8)
    frame1 = np.ones((8, 8, 3), dtype=np.uint8)

    worker.start()
    try:
        # 轮 1：只提交槽位 0（首帧 frame_id == 1）——误传 frame_id 会得 1。
        worker.submit_frame(0, frame0)
        assert _wait_until(lambda: spies["detection"].count == 1), "槽位 0 帧未被处理"
        slot, dets = spies["detection"].calls[0]
        assert slot == 0
        assert dets[0]["track_id"] == 10

        # 轮 2：只提交槽位 1。
        worker.submit_frame(1, frame1)
        assert _wait_until(lambda: spies["detection"].count == 2), "槽位 1 帧未被处理"
        slot, dets = spies["detection"].calls[1]
        assert slot == 1
        assert dets[0]["track_id"] == 20

        # 轮 3：双槽位同批提交，结果与槽位严格一一对应。
        worker.submit_frame(0, frame0)
        worker.submit_frame(1, frame1)
        assert _wait_until(lambda: spies["detection"].count == 4), "双槽位帧未被全部处理"
        seen = {}
        for slot, dets in spies["detection"].calls[2:]:
            seen[slot] = dets[0]["track_id"]
        assert seen == {0: 10, 1: 20}

        # 无槽位串扰：各 tracker 只见过本槽位内容的帧；推理帧序列正确。
        assert np.array_equal(env.tracker.last_frame, frame0)
        assert np.array_equal(tracker1.last_frame, frame1)
        assert env.backend.calls == 4
        expected_frames = [frame0, frame1, frame0, frame1]
        for recorded, expected in zip(env.backend.frames, expected_frames):
            assert np.array_equal(recorded, expected)

        assert spies["error"].count == 0
    finally:
        worker.stop()
    assert not worker.isRunning()


# ----------------------------------------------------------------------
# Pipeline runtime 生命周期收口（0.4.0，run 级证明）
# ----------------------------------------------------------------------


def _spy_pipeline_shutdown(monkeypatch):
    """包装 visioncore Pipeline.shutdown，返回调用记录（spy/计数，无 sleep）。"""
    from visioncore.pipeline import Pipeline

    shutdowns: list = []
    real_shutdown = Pipeline.shutdown

    def _counting_shutdown(self):
        shutdowns.append(self)
        return real_shutdown(self)

    monkeypatch.setattr(Pipeline, "shutdown", _counting_shutdown)
    return shutdowns


def test_unregister_while_running_retires_runtime_and_run_loop_cleans(qapp, monkeypatch):
    """运行中 unregister：不并发 shutdown，标记 retirement 由 run 循环清理。"""
    env = _build_pipeline_worker(monkeypatch, _pipeline_scenario(), True)
    worker = env.worker
    shutdowns = _spy_pipeline_shutdown(monkeypatch)
    spies = _connect_spies(worker)

    worker.start()
    try:
        worker.submit_frame(0, FRAME)
        assert _wait_until(lambda: spies["detection"].count >= 1), "首帧未被处理"
        rt = worker._pipeline_runtimes[0]

        # 运行中 unregister：runtime 暂不 shutdown/移除，仅标记 retirement
        worker.unregister_slot(0)
        assert worker._pipeline_runtimes.get(0) is rt
        assert 0 in worker._retired_pipeline_slots
        assert shutdowns == []

        # inference thread 在后续 run 循环中清理（有界轮询，非 sleep 断言）
        assert _wait_until(lambda: 0 not in worker._pipeline_runtimes), "retirement 未被 run 循环清理"
        assert shutdowns == [rt.pipeline]
        assert 0 not in worker._retired_pipeline_slots
    finally:
        worker.stop()


def test_stop_shuts_down_all_runtimes_and_clears_cache_once(qapp, monkeypatch):
    """stop 等待线程结束后 shutdown 全部 runtime 并清空 cache；重复 stop 不重复 shutdown。"""
    env = _build_pipeline_worker(monkeypatch, _pipeline_scenario(), True)
    worker = env.worker
    shutdowns = _spy_pipeline_shutdown(monkeypatch)
    spies = _connect_spies(worker)

    worker.start()
    worker.submit_frame(0, FRAME)
    assert _wait_until(lambda: spies["detection"].count >= 1), "首帧未被处理"
    rt = worker._pipeline_runtimes[0]

    worker.stop()
    assert not worker.isRunning()
    assert worker._pipeline_runtimes == {}  # cache 清空
    assert shutdowns == [rt.pipeline]  # 每个 runtime 恰好 shutdown 一次

    # 重复 stop：不重复 shutdown、cache 保持空
    worker.stop()
    assert worker._pipeline_runtimes == {}
    assert shutdowns == [rt.pipeline]
