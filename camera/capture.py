from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import av
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal, QMutex

if TYPE_CHECKING:
    # VisionCore 未来统一模型入口 —— 当前仅用于类型检查，不影响运行时。
    # SingleCameraWorker 的运行逻辑保持不变；Frame 为未来迁移做准备。
    from visioncore.core import Frame as CoreFrame


class SingleCameraWorker(QThread):
    frame_ready = pyqtSignal(int, object)  # VisionCore 未来入口: 未来可发射 CoreFrame
    error_occurred = pyqtSignal(int, str)

    def __init__(self, slot_id: int, device_name: str, parent=None):
        super().__init__(parent)
        self._slot_id = slot_id
        self._device_name = device_name
        self._running = False

    def run(self) -> None:
        self._running = True
        logging.info("SingleCameraWorker[%d] starting for device '%s'", self._slot_id, self._device_name)

        try:
            container = av.open(
                f"video={self._device_name}",
                format="dshow",
                options={
                    "rtbufsize": "200M",
                    "video_size": "640x480",
                    "framerate": "30",
                },
            )
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"

            logging.info("SingleCameraWorker[%d] stream opened: %dx%d @ %s fps",
                         self._slot_id, stream.width, stream.height, stream.average_rate)

            for frame in container.decode(stream):
                if not self._running:
                    break

                rgb_array = frame.to_ndarray(format="rgb24")
                self.frame_ready.emit(self._slot_id, rgb_array)

        except Exception as av_err:
            # 兼容新版 PyAV：av.AVError 已被移除，统一捕获所有异常
            if hasattr(av, 'AVError') and isinstance(av_err, av.AVError):
                error_msg = f"PyAV error on slot {self._slot_id}: {av_err}"
            else:
                error_msg = f"PyAV error on slot {self._slot_id}: {av_err}"
            logging.error(error_msg)
            self.error_occurred.emit(self._slot_id, error_msg)
        except Exception as err:
            error_msg = f"Unexpected error on slot {self._slot_id}: {err}"
            logging.error(error_msg)
            self.error_occurred.emit(self._slot_id, error_msg)
        finally:
            try:
                container.close()
            except Exception:
                pass
            logging.info("SingleCameraWorker[%d] stopped", self._slot_id)

    def stop(self) -> None:
        self._running = False
        self.wait(3000)
        if self.isRunning():
            logging.warning("SingleCameraWorker[%d] did not terminate within timeout", self._slot_id)


class CameraCapture:
    def __init__(self):
        self._workers: dict[int, SingleCameraWorker] = {}
        self._mutex = QMutex()

    def start_camera(self, slot_id: int, device_name: str) -> None:
        self._mutex.lock()
        try:
            existing = self._workers.get(slot_id)
            if existing is not None:
                existing.stop()
                existing.deleteLater()
                del self._workers[slot_id]
                logging.info("CameraCapture: stopped previous worker for slot %d", slot_id)

            worker = SingleCameraWorker(slot_id, device_name)
            self._workers[slot_id] = worker
        finally:
            self._mutex.unlock()

        worker.start()
        logging.info("CameraCapture: started worker for slot %d on device '%s'", slot_id, device_name)

    def stop_camera(self, slot_id: int) -> None:
        self._mutex.lock()
        worker = self._workers.pop(slot_id, None)
        self._mutex.unlock()

        if worker is not None:
            worker.stop()
            worker.deleteLater()
            logging.info("CameraCapture: stopped worker for slot %d", slot_id)

    def stop_all(self) -> None:
        self._mutex.lock()
        workers = dict(self._workers)
        self._workers.clear()
        self._mutex.unlock()

        for slot_id, worker in workers.items():
            worker.stop()
            worker.deleteLater()
            logging.info("CameraCapture: stopped worker for slot %d", slot_id)

        logging.info("CameraCapture: all workers stopped")

    def get_worker(self, slot_id: int) -> SingleCameraWorker | None:
        self._mutex.lock()
        try:
            return self._workers.get(slot_id)
        finally:
            self._mutex.unlock()

    def get_all_workers(self) -> list[tuple[int, SingleCameraWorker]]:
        self._mutex.lock()
        try:
            return list(self._workers.items())
        finally:
            self._mutex.unlock()
