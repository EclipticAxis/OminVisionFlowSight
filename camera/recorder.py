import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import av
import numpy as np


class _RecorderThread(threading.Thread):

    def __init__(self, slot_id: int, output_path: str, width: int, height: int, fps: int):
        super().__init__(daemon=True)
        self._slot_id = slot_id
        self._output_path = output_path
        self._width = width
        self._height = height
        self._fps = fps
        self._queue: queue.Queue = queue.Queue(maxsize=120)
        self._sentinel = object()

    def run(self) -> None:
        container = None
        encoder_ctx = None

        try:
            container = av.open(self._output_path, mode="w")
            stream = container.add_stream("h264", rate=self._fps)
            stream.width = self._width
            stream.height = self._height
            stream.pix_fmt = "yuv420p"
            encoder_ctx = stream.codec_context
            encoder_ctx.options = {
                "preset": "ultrafast",
                "tune": "zerolatency",
                "crf": "23",
            }

            logging.info(
                "RecorderThread[%d] started: %s (%dx%d @ %d fps)",
                self._slot_id, self._output_path, self._width, self._height, self._fps,
            )

            while True:
                item = self._queue.get()
                if item is self._sentinel:
                    break

                frame_rgb = item
                video_frame = av.VideoFrame.from_ndarray(frame_rgb, format="rgb24")
                video_frame = video_frame.reformat(format="yuv420p")

                for packet in encoder_ctx.encode(video_frame):
                    container.mux(packet)

            for packet in encoder_ctx.encode(None):
                container.mux(packet)

            logging.info("RecorderThread[%d] flushed encoder residual frames", self._slot_id)

        except Exception as err:
            logging.error("RecorderThread[%d] fatal error: %s", self._slot_id, err)

        finally:
            if container is not None:
                try:
                    container.close()
                except Exception as close_err:
                    logging.error("RecorderThread[%d] error closing container: %s", self._slot_id, close_err)
            logging.info("RecorderThread[%d] stopped, file saved: %s", self._slot_id, self._output_path)

    def submit_frame(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            logging.warning("RecorderThread[%d] queue full, dropping frame", self._slot_id)

    def stop(self) -> None:
        self._queue.put(self._sentinel)
        self.join(timeout=5.0)
        if self.is_alive():
            logging.warning("RecorderThread[%d] did not terminate within timeout", self._slot_id)


class VideoRecorder:

    def __init__(self, output_dir: str | Path = "recordings"):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._recorders: dict[int, _RecorderThread] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="RecorderPool")

    def set_output_dir(self, path: str | Path) -> None:
        self._output_dir = Path(path)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logging.info("VideoRecorder: output directory changed to %s", self._output_dir)

    def start_record(self, slot_id: int, path: str | None = None, width: int = 640, height: int = 480, fps: int = 30) -> None:
        with self._lock:
            existing = self._recorders.get(slot_id)
            if existing is not None:
                existing.stop()
                logging.info("VideoRecorder: stopped previous recording for slot %d", slot_id)

            if path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"slot_{slot_id}_{timestamp}.mp4"
                output_path = str(self._output_dir / filename)
            else:
                output_path = str(path)
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            recorder = _RecorderThread(slot_id, output_path, width, height, fps)
            self._recorders[slot_id] = recorder

        self._pool.submit(recorder.run)
        logging.info("VideoRecorder: started recording slot %d -> %s", slot_id, output_path)

    def start_recording(self, slot_id: int, width: int, height: int, fps: int) -> None:
        self.start_record(slot_id, path=None, width=width, height=height, fps=fps)

    def push_frame(self, slot_id: int, frame: np.ndarray) -> None:
        with self._lock:
            recorder = self._recorders.get(slot_id)
        if recorder is not None:
            recorder.submit_frame(frame)

    def write_frame(self, slot_id: int, frame: np.ndarray) -> None:
        self.push_frame(slot_id, frame)

    def stop_record(self, slot_id: int) -> None:
        with self._lock:
            recorder = self._recorders.pop(slot_id, None)
        if recorder is not None:
            recorder.stop()
            logging.info("VideoRecorder: stopped recording slot %d", slot_id)

    def stop_recording(self, slot_id: int) -> None:
        self.stop_record(slot_id)

    def is_recording(self, slot_id: int) -> bool:
        with self._lock:
            return slot_id in self._recorders

    def stop_all(self) -> None:
        with self._lock:
            recorders = dict(self._recorders)
            self._recorders.clear()

        for slot_id, recorder in recorders.items():
            recorder.stop()
            logging.info("VideoRecorder: stopped recording slot %d", slot_id)

        logging.info("VideoRecorder: all recordings stopped")

    def shutdown(self) -> None:
        self.stop_all()
        self._pool.shutdown(wait=True, cancel_futures=False)
        logging.info("VideoRecorder: thread pool shut down")
