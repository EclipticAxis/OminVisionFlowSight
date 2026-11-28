"""YOLO26 深度估计独立推理线程（DepthWorker）。

设计契约：
- 懒加载：import / ``__init__`` 不引入 torch / cv2 / ultralytics；模型仅在
  worker 线程 ``run()`` 内首次需要推理时加载（单模型、禁止跨线程调用）。
  不做任何后台预热 / 预加载，模型加载后常驻。
- 每 slot 覆盖式保存最新帧（不排队），多 slot 公平轮询；锁内只做状态
  读写，模型推理在锁外执行；``QWaitCondition`` 防止 lost wakeup。
- 时域滤波在原始米制 float32 深度域进行（运动自适应 EMA：静态像素
  alpha=0.25、运动/突变像素 alpha=0.85），**先滤波再 colorize**，禁止
  彩色图 blending；滤波状态按 slot 保存，shape 变化自动重置。
- 默认显示为近暖远冷：在米制量程内计算 disparity=1/depth，归一化后
  使用 gamma=0.5，再应用 Turbo；滤波和量程状态仍保持米制语义。
- 量程：``range_mode="auto"`` 对滤波后有效深度取 2%/98% 百分位并做
  慢速 EMA（alpha=0.05，首帧直接采用目标，最小跨度 0.5m）；
  ``"0-3"/"0-8"/"0-20"`` 为固定量程，不做范围平滑。
- 只输出 colorize 后的 contiguous uint8 RGB 图（HxWx3，默认 Turbo 色带），
  不跨线程传 float 大数组；每次成功输出先 emit
  ``depth_range_ready(slot, vmin, vmax)`` 再 emit ``depth_ready``。
- 模型加载失败：``error_occurred(-1, msg)`` + 内部禁用并等待，不忙等、
  不无限重试；单帧推理失败：日志 + ``error_occurred(slot_id, msg)``
  （节流，第一次必报），继续后续帧。
- ``set_enabled(False)`` 清空 pending 帧与全部滤波/量程状态；禁用时即使
  推理已结束也不得再发新深度帧。``stop()`` 幂等，可安全唤醒退出。
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

# --- 可视化设置契约 ---
SUPPORTED_CMAPS = ("turbo", "inferno", "plasma", "viridis")
FIXED_RANGE_MODES = {"0-3": (0.0, 3.0), "0-8": (0.0, 8.0), "0-20": (0.0, 20.0)}
SUPPORTED_RANGE_MODES = ("auto", *FIXED_RANGE_MODES)

# --- 运动自适应 EMA 契约 ---
_EMA_ALPHA_STATIC = 0.25
_EMA_ALPHA_MOTION = 0.85
_MOTION_ABS_THRESHOLD = 0.15   # m
_MOTION_REL_THRESHOLD = 0.15   # * 上一帧深度

# --- 自适应量程契约 ---
_RANGE_PERCENTILES = (2.0, 98.0)
_RANGE_EMA_ALPHA = 0.05
_RANGE_MIN_SPAN = 0.5          # m
_AUTO_FALLBACK_RANGE = (0.0, 20.0)  # 无任何有效像素且无历史量程时的兜底


def _validate_cmap(cmap: str) -> str:
    name = str(cmap).lower()
    if name not in SUPPORTED_CMAPS:
        raise ValueError(f"非法 cmap: {cmap!r}（支持 {SUPPORTED_CMAPS}）")
    return name


def _validate_range_mode(range_mode: str) -> str:
    mode = str(range_mode).lower()
    if mode not in SUPPORTED_RANGE_MODES:
        raise ValueError(
            f"非法 range_mode: {range_mode!r}（支持 {SUPPORTED_RANGE_MODES}）"
        )
    return mode


def filter_depth_ema(
    prev: np.ndarray | None,
    current: np.ndarray,
    alpha_static: float = _EMA_ALPHA_STATIC,
    alpha_motion: float = _EMA_ALPHA_MOTION,
) -> np.ndarray:
    """运动自适应 EMA 时域滤波（原始米制 float32 深度域，非彩色图）。

    - 仅有效像素（finite 且 > 0）参与滤波；当前帧无效像素输出 NaN，
      绝不沿用旧深度；
    - 新出现 / 上一帧无效的像素直接采用当前值；
    - 运动像素（``|cur - prev| > max(0.15m, 15% * |prev|)``）用
      ``alpha_motion`` 快速跟随，静态像素用 ``alpha_static``；
    - ``prev`` 为 None 或 shape 不一致时视为首帧：直接采用当前值。
    """
    cur = np.asarray(current, dtype=np.float32)
    cur_valid = np.isfinite(cur) & (cur > 0.0)
    out = np.where(cur_valid, cur, np.nan).astype(np.float32)
    if prev is None:
        return out
    prev = np.asarray(prev, dtype=np.float32)
    if prev.shape != cur.shape:
        return out
    prev_valid = np.isfinite(prev) & (prev > 0.0)
    both = cur_valid & prev_valid
    diff = np.abs(cur - prev)
    thresh = np.maximum(_MOTION_ABS_THRESHOLD, _MOTION_REL_THRESHOLD * np.abs(prev))
    alpha = np.where(diff > thresh, alpha_motion, alpha_static).astype(np.float32)
    blended = prev + alpha * (cur - prev)
    return np.where(both, blended, out).astype(np.float32)


def update_adaptive_range(
    prev_range: tuple[float, float] | None,
    filtered: np.ndarray,
    alpha: float = _RANGE_EMA_ALPHA,
    min_span: float = _RANGE_MIN_SPAN,
    percentiles: tuple[float, float] = _RANGE_PERCENTILES,
) -> tuple[float, float]:
    """稳定自适应量程：滤波后有效深度的 2%/98% 百分位为目标的慢速 EMA。

    - 首帧（``prev_range is None``）直接采用目标；
    - 目标跨度不足 ``min_span`` 时以中点为中心扩到 ``min_span``；
    - 当前帧无有效像素时保持上一量程（无历史则用 ``_AUTO_FALLBACK_RANGE``）。
    """
    arr = np.asarray(filtered, dtype=np.float32)
    valid = arr[np.isfinite(arr) & (arr > 0.0)]
    if valid.size == 0:
        return prev_range if prev_range is not None else _AUTO_FALLBACK_RANGE
    lo, hi = (float(v) for v in np.percentile(valid, percentiles))
    if hi - lo < min_span:
        mid = 0.5 * (lo + hi)
        lo, hi = mid - 0.5 * min_span, mid + 0.5 * min_span
    if prev_range is None:
        return (lo, hi)
    plo, phi = prev_range
    return (plo + alpha * (lo - plo), phi + alpha * (hi - phi))


def colorize_depth(
    depth: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 20.0,
    cmap: str = "turbo",
    mode: str = "disparity",
    gamma: float = 0.5,
) -> np.ndarray:
    """float 深度图 → contiguous uint8 RGB 彩色图（HxWx3）。

    - ``mode="disparity"``（默认）：在 ``[vmin, vmax]`` 米制量程内计算
      ``1/depth``，归一化后应用 ``gamma``；近处映射高端暖色、远处低端冷色；
    - ``mode="metric"``：固定 ``[vmin, vmax]`` 线性映射，近冷远暖；
    - ``mode="auto"``：按当前帧 min/max 自适应（仅调试用途）；
    - 默认色带 Turbo；``cmap`` 需为 OpenCV 已知 colormap（至少支持
      turbo/inferno/plasma/viridis），未知值抛 ``ValueError``；
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

    if mode == "disparity":
        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gamma 必须为正的有限值，实际={gamma!r}")
        far = float(vmax)
        # 固定 0–N 米量程不能让 1/0 奇点吞掉全部颜色；仅此时使用 5% 近端。
        near = 0.05 * far if float(vmin) <= 0.0 else float(vmin)
        if not np.isfinite(near) or not np.isfinite(far) or far <= near:
            raise ValueError(f"vmax 必须大于有效 vmin，实际 vmin={vmin}, vmax={vmax}")
        valid = np.isfinite(arr) & (arr > 0.0)
        disparity_far = 1.0 / far
        disparity_near = 1.0 / near
        # 全数组就地计算（where 屏蔽无效像素）：避免布尔花式索引的
        # gather/scatter 与多份临时数组。无效像素保持 0，经减/除/clip/power
        # 后仍为 0，与参考公式（仅对有效像素计算后回填）逐像素一致。
        norm = np.zeros_like(arr, dtype=np.float32)
        np.divide(1.0, arr, out=norm, where=valid)  # disparity = 1 / depth
        norm -= disparity_far
        norm /= disparity_near - disparity_far
        np.clip(norm, 0.0, 1.0, out=norm)
        np.power(norm, gamma, out=norm)
    elif mode == "metric":
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
        raise ValueError(f"未知 mode: {mode!r}（支持 'disparity' / 'metric' / 'auto'）")

    # NaN / +Inf / -Inf 一律映射低值端（0.0），与 docstring 契约一致；
    # 正常有限深度值的线性映射不受影响。
    norm = np.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)
    gray = (np.clip(norm, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    cmap_flag = getattr(cv2, f"COLORMAP_{str(cmap).upper()}", None)
    if cmap_flag is None:
        raise ValueError(f"未知 cmap: {cmap!r}")
    bgr = cv2.applyColorMap(gray, cmap_flag)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


class DepthWorker(QThread):
    """YOLO26 深度估计 worker：多 slot 共享单模型，输出彩色深度图。"""

    depth_ready = pyqtSignal(int, object)        # (slot_id, contiguous uint8 HxWx3 RGB)
    depth_range_ready = pyqtSignal(int, float, float)  # (slot_id, vmin, vmax) 先于 depth_ready
    error_occurred = pyqtSignal(int, str)        # slot_id=-1 表示模型级错误

    def __init__(
        self,
        model_path: str | Path | None = None,
        imgsz: int = 640,
        device: str | int | None = None,
        cmap: str = "turbo",
        range_mode: str = "auto",
        parent=None,
    ):
        super().__init__(parent)
        self._model_path = str(model_path) if model_path is not None else str(DEFAULT_MODEL_PATH)
        self._imgsz = int(imgsz)
        self._device = device
        self._cmap = _validate_cmap(cmap)
        self._range_mode = _validate_range_mode(range_mode)
        self._resolved_device: str | int | None = None
        self._model = None
        self._model_failed = False  # 加载失败后粘性置位：不重试、不忙等
        self._enabled = False
        self._running = False
        self._registered: set[int] = set()
        self._pending: dict[int, np.ndarray] = {}   # slot_id -> 最新帧（覆盖式）
        self._rr: deque[int] = deque()              # 轮询顺序（公平）
        self._infer_error_counts: dict[int, int] = {}
        self._depth_state: dict[int, np.ndarray] = {}         # slot_id -> 滤波后 float32 深度
        self._range_state: dict[int, tuple[float, float]] = {}  # slot_id -> (vmin, vmax)（auto）
        # 滤波/量程状态代际（per-slot，tombstone：只增不删）。
        # unregister/disable/可视化切换时递增对应 slot 代际使在飞旧帧失效；
        # 条目永不删除，确保 unregister→register 后旧帧不会因代际重置而复生。
        self._slot_gen: dict[int, int] = {}
        self._mutex = QMutex()
        self._cond = QWaitCondition()

    # ------------------------------------------------------------------
    # 公共接口（任意线程可调用；start() 由主窗口负责）
    # ------------------------------------------------------------------

    def register_slot(self, slot_id: int) -> None:
        with QMutexLocker(self._mutex):
            slot_id = int(slot_id)
            self._registered.add(slot_id)
            # tombstone：只初始化不重置，保留历史代际防止旧在飞帧复活
            self._slot_gen.setdefault(slot_id, 0)

    def unregister_slot(self, slot_id: int) -> None:
        with QMutexLocker(self._mutex):
            slot_id = int(slot_id)
            self._registered.discard(slot_id)
            self._pending.pop(slot_id, None)
            self._infer_error_counts.pop(slot_id, None)
            self._depth_state.pop(slot_id, None)
            self._range_state.pop(slot_id, None)
            # 仅使本 slot 的在飞帧失效；其他 slot 代际不受影响。
            # 代际条目保留（tombstone），重新 register 不复位。
            self._slot_gen[slot_id] = self._slot_gen.get(slot_id, 0) + 1
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
        """启停深度推理。禁用时清空 pending 帧与全部滤波/量程状态。"""
        with QMutexLocker(self._mutex):
            enabled = bool(enabled)
            # 加载失败后不做后台自动重试；用户重新打开开关才清除粘性错误。
            if enabled and not self._enabled:
                self._model_failed = False
            self._enabled = enabled
            if not enabled:
                self._pending.clear()
                self._rr.clear()
                self._depth_state.clear()
                self._range_state.clear()
                # 使所有当前/已知 slot 的在飞旧帧失效（tombstone 代际只增不删）
                self._bump_all_generations()
            self._cond.wakeAll()

    def set_visualization(self, cmap: str, range_mode: str) -> None:
        """切换色带 / 量程模式。非法值抛 ``ValueError`` 且不改变现有设置。

        - cmap 或 range_mode 实际变化：递增所有已知 slot 代际，使在飞
          旧色带 / 旧量程帧失效（不发出、不写状态）；
        - cmap 变化保留已完成的深度滤波与量程状态；
        - range_mode 变化继续清空自适应量程状态（固定量程本身不做范围平滑）；
        - 同值调用为 no-op：不改设置、不动代际与状态。
        """
        cmap = _validate_cmap(cmap)
        range_mode = _validate_range_mode(range_mode)
        with QMutexLocker(self._mutex):
            cmap_changed = cmap != self._cmap
            range_changed = range_mode != self._range_mode
            if not cmap_changed and not range_changed:
                return  # 同值 no-op
            self._cmap = cmap
            self._range_mode = range_mode
            if range_changed:
                self._range_state.clear()
            self._bump_all_generations()

    def _bump_all_generations(self) -> None:
        """递增所有已知 slot 的代际（须在 _mutex 内调用）。"""
        for slot in self._slot_gen:
            self._slot_gen[slot] += 1

    def stop(self, wait_ms: int = 5000) -> None:
        """请求线程退出并等待。幂等，可在未 start 时安全调用。"""
        with QMutexLocker(self._mutex):
            self._running = False
            self._cond.wakeAll()
        if self.isRunning() and not self.wait(wait_ms):
            # 超时只可能发生在单次推理（锁外）尚未返回时；_running 已置 False，
            # 该次推理结束后主循环会在下一次取帧处协作退出。这里只收口状态
            # 并告警，绝不使用 QThread.terminate() 强杀线程。
            logger.warning(
                "DepthWorker 未能在 %d ms 内退出（等待在飞推理完成，协作式停止）",
                wait_ms,
            )

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
                    # 快照可视化设置与上一帧滤波/量程状态（锁内只读状态）
                    prev_depth = self._depth_state.get(slot_id)
                    prev_range = self._range_state.get(slot_id)
                    seq = self._slot_gen.get(slot_id, 0)
                    cmap = self._cmap
                    range_mode = self._range_mode

                # --- 推理 + 时域滤波 + 量程 + 着色（锁外，禁止跨线程调用模型）---
                try:
                    depth = self._infer_depth(frame)
                    filtered = filter_depth_ema(prev_depth, depth)
                    if range_mode == "auto":
                        vmin, vmax = update_adaptive_range(prev_range, filtered)
                    else:
                        vmin, vmax = FIXED_RANGE_MODES[range_mode]
                    color = colorize_depth(filtered, vmin=vmin, vmax=vmax, cmap=cmap)
                except Exception as exc:  # noqa: BLE001 —— 单帧防火带
                    self._report_infer_error(slot_id, exc)
                    continue

                # 禁用时即使推理已结束也不得发新深度帧
                with QMutexLocker(self._mutex):
                    still_active = (
                        self._running and self._enabled and slot_id in self._registered
                    )
                    current_generation = still_active and seq == self._slot_gen.get(slot_id, 0)
                    # 清理（disable/unregister/cmap/量程切换）后不得复活旧状态
                    if current_generation:
                        self._depth_state[slot_id] = filtered
                        if range_mode == "auto":
                            self._range_state[slot_id] = (float(vmin), float(vmax))
                    if still_active:
                        self._infer_error_counts.pop(slot_id, None)
                if current_generation:
                    # 过期帧连显示也不得发出，避免快速开关/换相机/改量程时闪一帧旧画面。
                    self.depth_range_ready.emit(slot_id, float(vmin), float(vmax))
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
        # 计数读写统一在同一把 QMutex 内（与 unregister_slot / 成功清理一致）；
        # 锁内只做状态读写，日志与信号 emit 放锁外。
        with QMutexLocker(self._mutex):
            count = self._infer_error_counts.get(slot_id, 0) + 1
            self._infer_error_counts[slot_id] = count
        if count == 1 or count % _INFER_ERROR_REPORT_INTERVAL == 0:
            logger.warning(
                "DepthWorker slot %d 推理失败（连续 %d 次）: %s", slot_id, count, exc
            )
            self.error_occurred.emit(slot_id, f"深度推理失败(slot {slot_id}): {exc}")
