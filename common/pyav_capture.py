"""PyAV 摄像头采集后端 - 最简实现。

封装 PyAV DirectShow 采集，实现 common.CaptureBackend 接口。
线程安全：read() 可在工作线程中调用，open()/close() 在主线程管理。
"""
from __future__ import annotations

import logging
import time

import av
import numpy as np

from common import Frame

logger = logging.getLogger(__name__)


class PyAVCapture:
    """PyAV + DirectShow 摄像头采集。

    实现 common.CaptureBackend 接口（结构化子类型）。
    """

    def __init__(self):
        self._container = None
        self._stream = None
        self._frame_id = 0
        self._stream_id = "default"
        self._is_open = False

    def open(self, source: str, **kwargs) -> bool:
        """打开采集源。

        Args:
            source: 设备名（Windows DirectShow），如 "HD Webcam"
                    或 "video=HD Webcam" 格式。
            kwargs: resolution (w,h), fps
        """
        # 标准化 source：确保有 video= 前缀
        if not source.startswith("video="):
            source = f"video={source}"

        resolution = kwargs.get("resolution", (640, 480))
        fps = kwargs.get("fps", 30)

        try:
            self._container = av.open(
                source,
                format="dshow",
                options={
                    "rtbufsize": "200M",
                    "video_size": f"{resolution[0]}x{resolution[1]}",
                    "framerate": str(fps),
                },
            )
            self._stream = self._container.streams.video[0]
            self._stream.thread_type = "AUTO"
            self._frame_id = 0
            self._is_open = True
            self._stream_id = source

            logger.info("Camera opened: %s (%dx%d @ %d fps)",
                         source, self._stream.width, self._stream.height, fps)
            return True

        except Exception as e:
            logger.error("Failed to open camera '%s': %s", source, e)
            self._is_open = False
            return False

    def read(self) -> Frame | None:
        """读取下一帧。无帧时返回 None（不阻塞）。"""
        if not self._is_open or self._container is None:
            return None

        try:
            for frame in self._container.decode(self._stream):
                rgb = frame.to_ndarray(format="rgb24")
                self._frame_id += 1
                return Frame(
                    stream_id=self._stream_id,
                    frame_id=self._frame_id,
                    timestamp=time.time(),
                    data=rgb,
                )
        except Exception as e:
            logger.error("Camera read error: %s", e)
            self._is_open = False
            return None

        return None

    def close(self) -> None:
        """关闭采集源。"""
        self._is_open = False
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None
            self._stream = None
        logger.info("Camera closed: %s", self._stream_id)

    def is_open(self) -> bool:
        return self._is_open
