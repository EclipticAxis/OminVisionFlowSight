"""YOLO26 深度估计独立推理线程（DepthWorker）。

设计契约：
- 懒加载：import / ``__init__`` 不引入 torch / cv2 / ultralytics；模型仅在
  worker 线程 ``run()`` 内首次需要推理时加载（单模型、禁止跨线程调用）。
- 每 slot 覆盖式保存最新帧（不排队），多 slot 公平轮询；锁内只做状态
  读写，模型推理在锁外执行；``QWaitCondition`` 防止 lost wakeup。
- 只输出 colorize 后的 contiguous uint8 RGB 图（HxWx3），不跨线程传
  float 大数组；固定 metric range 保持跨帧色彩稳定，不做时域插帧 /
  彩色帧 blending。
- 模型加载失败：``error_occurred(-1, msg)`` + 内部禁用并等待，不忙等、
  不无限重试；单帧推理失败：日志 + ``error_occurred(slot_id, msg)``
  （节流，第一次必报），继续后续帧。
- ``set_enabled(False)`` 清空 pending 帧；禁用时即使推理已结束也不得
  再发新深度帧。``stop()`` 幂等，可安全唤醒退出。
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QMutex, QMutexLocker, QThread, QWaitCondition, pyqtSignal

logger = logging.getLogger("ai.depth_worker")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo26s-depth.pt"

# 单 slot 连续推理失败上报节流：第 1 次及每第 N 次 emit error_occurred
_INFER_ERROR_REPORT_INTERVAL = 30


def colorize_depth(
    depth: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 20.0,
    cmap: str = "inferno",
    mode: str = "metric",
) -> np.ndarray:
    """float 深度图 → contiguous uint8 RGB 彩色图（HxWx3）。

    - ``mode="metric"``（默认）：固定 ``[vmin, vmax]`` 线性映射，跨帧色彩稳定；
    - ``mode="auto"``：按当前帧 min/max 自适应（仅调试用途）；
    - NaN / ±Inf 一律映射到低值端（0.0）。
    """
    import cv2  # 懒加载：模块 import 不引入 cv2

    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"depth 必须是 2D（或 squeeze 后为 2D），实际 shape={arr.shape}")

    if mode == "metric":
        span = float(vmax) - float(vmin)
        if span <= 0:
            raise ValueError(f"vmax 必须大于 vmin，实际 vmin={vmin}, vmax={vmax}")
        norm = (arr - float(vmin)) / span
    elif mode == "auto":
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            norm = np.zeros_like(arr)
        else:
            lo, hi = float(finite.min()), float(finite.max())
            span = hi - lo
            norm = (arr - lo) / span if span > 0 else np.zeros_like(arr)
    else:
        raise ValueError(f"未知 mode: {mode!r}（支持 'metric' / 'auto'）")

    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    gray = (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cmap_flag = getattr(cv2, f"COLORMAP_{str(cmap).upper()}", cv2.COLORMAP_INFERNO)
    bgr = cv2.applyColorMap(gray, cmap_flag)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


class DepthWorker(QThread):
    """YOLO26 深度估计 worker：多 slot 共享单模型，输出彩色深度图。"""

    depth_ready = pyqtSignal(int, object)   # (slot_id, contiguous uint8 HxWx3 RGB)
    error_occurred = pyqtSignal(int, str)   # slot_id=-1 表示模型级错误

    def __init__(
        self,
        model_path: str | Path | None = None,
        imgsz: int = 640,
        device: str | int | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._model_path = str(model_path) if model_path is not None else str(DEFAULT_MODEL_PATH)
        self._imgsz = int(imgsz)
        self._device = device
        self._resolved_device: str | int | None = None
        self._model = None
        self._model_failed = False  # 加载失败后粘性置位：不重试、不忙等
        self._enabled = False
        self._running = False
        self._registered: set[int] = set()
        self._pending: dict[int, np.ndarray] = {}   # slot_id -> 最新帧（覆盖式）
        self._rr: deque[int] = deque()              # 轮询顺序（公平）
        self._infer_error_counts: dict[int, int] = {}
        self._mutex = QMutex()
        self._cond = QWaitCondition()

    # ------------------------------------------------------------------
    # 公共接口（任意线程可调用；start() 由主窗口负责）
    # ------------------------------------------------------------------

    def register_slot(self, slot_id: int) -> None:
        with QMutexLocker(self._mutex):
            self._registered.add(int(slot_id))

    def unregister_slot(self, slot_id: int) -> None:
        with QMutexLocker(self._mutex):
            slot_id = int(slot_id)
            self._registered.discard(slot_id)
            self._pending.pop(slot_id, None)
            self._infer_error_counts.pop(slot_id, None)
            try:
                self._rr.remove(slot_id)
            except ValueError:
                pass

    def submit_frame(self, slot_id: int, frame: np.ndarray) -> None:
        """提交 RGB 帧。仅在 enabled 且 slot 已 register 时接受；覆盖旧帧。"""
        if frame is None:
            return
        slot_id = int(slot_id)
        with QMutexLocker(self._mutex):
            if not self._enabled or slot_id not in self._registered:
                return
            self._pending[slot_id] = frame
            if slot_id not in self._rr:
                self._rr.append(slot_id)
            self._cond.wakeOne()

    def set_enabled(self, enabled: bool) -> None:
        """启停深度推理。禁用时清空 pending 帧；不自动 start/stop 线程。"""
        with QMutexLocker(self._mutex):
            enabled = bool(enabled)
            # 加载失败后不做后台自动重试；用户重新打开开关才清除粘性错误。
            if enabled and not self._enabled:
                self._model_failed = False
            self._enabled = enabled
            if not enabled:
                self._pending.clear()
                self._rr.clear()
            self._cond.wakeAll()

    def stop(self, wait_ms: int = 5000) -> None:
        """请求线程退出并等待。幂等，可在未 start 时安全调用。"""
        with QMutexLocker(self._mutex):
            self._running = False
            self._cond.wakeAll()
        if self.isRunning() and not self.wait(wait_ms):
            logger.warning("DepthWorker 未能在 %d ms 内退出", wait_ms)

    # ------------------------------------------------------------------
    # worker 线程主循环
    # ------------------------------------------------------------------

    def start(self, priority: QThread.Priority = QThread.Priority.InheritPriority) -> None:
        if self.isRunning():
            return
        # 在 start() 而非 run() 中置位：stop() 若先于 run() 执行（竞态），
        # 线程进入 run() 时会立即看到 _running=False 并退出，不会 lost wakeup。
        with QMutexLocker(self._mutex):
            self._running = True
        super().start(priority)

    def run(self) -> None:
        logger.info(
            "DepthWorker started (model=%s, imgsz=%d, device=%s)",
            self._model_path, self._imgsz, self._device,
        )
        try:
            while True:
                # --- 取帧（锁内只做状态读写）---
                with QMutexLocker(self._mutex):
                    while self._running and (not self._enabled or not self._rr):
                        self._cond.wait(self._mutex)
                    if not self._running:
                        break
                    slot_id = self._rr.popleft()
                    frame = self._pending.pop(slot_id, None)
                    if frame is not None:
                        frame = frame.copy()  # 与 submit 方的后续写隔离
                if frame is None:
                    continue

                if not self._ensure_model():
                    continue

                # 模型加载可能耗时数秒；加载期间收到关闭/停止请求时不再推理。
                with QMutexLocker(self._mutex):
                    if (
                        not self._running
                        or not self._enabled
                        or slot_id not in self._registered
                    ):
                        continue

                # --- 推理 + 着色（锁外，禁止跨线程调用模型）---
                try:
                    depth = self._infer_depth(frame)
                    color = colorize_depth(depth)
                except Exception as exc:  # noqa: BLE001 —— 单帧防火带
                    self._report_infer_error(slot_id, exc)
                    continue

                # 禁用时即使推理已结束也不得发新深度帧
                with QMutexLocker(self._mutex):
                    still_active = (
                        self._running and self._enabled and slot_id in self._registered
                    )
                    if still_active:
                        self._infer_error_counts.pop(slot_id, None)
                if still_active:
                    self.depth_ready.emit(slot_id, color)
        finally:
            with QMutexLocker(self._mutex):
                self._running = False
            logger.info("DepthWorker stopped")

    # ------------------------------------------------------------------
    # 模型加载 / 推理（仅 worker 线程调用）
    # ------------------------------------------------------------------

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._model_failed:
            return False
        try:
            self._resolved_device = self._resolve_device()
            self._model = self._load_model()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.exception("深度模型加载失败: %s", self._model_path)
            self._model_failed = True
            with QMutexLocker(self._mutex):
                self._enabled = False
                self._pending.clear()
                self._rr.clear()
            self.error_occurred.emit(-1, f"深度模型加载失败: {exc}")
            return False

    def _load_model(self):
        """加载 YOLO26 深度模型（懒加载 ultralytics；测试可 monkeypatch）。"""
        from ultralytics import YOLO  # noqa: PLC0415 —— 刻意懒加载

        logger.info("加载深度模型 %s (device=%s)", self._model_path, self._resolved_device)
        return YOLO(self._model_path)

    def _resolve_device(self) -> str | int:
        if self._device is not None:
            return self._device
        try:
            import torch  # noqa: PLC0415 —— 刻意懒加载

            return 0 if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 —— 无 torch 时回退 CPU
            return "cpu"

    def _infer_depth(self, frame: np.ndarray) -> np.ndarray:
        """RGB 帧 → float32 深度图（尺寸与输入一致）。"""
        import cv2  # noqa: PLC0415 —— 刻意懒加载

        # Ultralytics 对 numpy 图像采用 OpenCV 的 BGR 通道约定。
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = self._model.predict(
            bgr, imgsz=self._imgsz, device=self._resolved_device, verbose=False
        )
        data = results[0].depth.data
        if hasattr(data, "detach"):
            data = data.detach()
        if hasattr(data, "cpu"):
            data = data.cpu()
        depth = np.asarray(data, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        elif depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        if depth.ndim != 2:
            raise ValueError(f"模型输出深度图维度异常: shape={depth.shape}")
        if depth.shape[:2] != frame.shape[:2]:
            depth = cv2.resize(
                depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
            )
        return depth

    def _report_infer_error(self, slot_id: int, exc: Exception) -> None:
        count = self._infer_error_counts.get(slot_id, 0) + 1
        self._infer_error_counts[slot_id] = count
        if count == 1 or count % _INFER_ERROR_REPORT_INTERVAL == 0:
            logger.warning(
                "DepthWorker slot %d 推理失败（连续 %d 次）: %s", slot_id, count, exc
            )
            self.error_occurred.emit(slot_id, f"深度推理失败(slot {slot_id}): {exc}")
