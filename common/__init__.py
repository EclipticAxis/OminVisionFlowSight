"""SentinelTrack / VDP - 公共抽象层。

定义捕获、检测、跟踪三类后端的统一接口，以及 Frame 数据模型。
现有 ai/detection.py 的 Detection 和 ai/tracker.py 的 DetectionTracker 已复用，
此处只补充它们之间缺失的胶水类型。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

# 复用已有数据模型
from ai.detection import Detection
from ai.tracker import DetectionTracker

__all__ = [
    "Frame",
    "CaptureBackend",
    "DetectorBackend",
    "TrackerBackend",
    "Detection",
    "DetectionTracker",
]


# ---------------------------------------------------------------------------
# Frame - 统一帧数据模型
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    """一帧视频数据的统一容器。"""

    stream_id: str
    frame_id: int
    timestamp: float
    data: np.ndarray            # RGB (H, W, 3) uint8
    width: int = 0
    height: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.width == 0 and self.data is not None:
            self.width = self.data.shape[1]
        if self.height == 0 and self.data is not None:
            self.height = self.data.shape[0]


# ---------------------------------------------------------------------------
# 后端接口 (Protocol - 结构化子类型，无需继承)
# ---------------------------------------------------------------------------

@runtime_checkable
class CaptureBackend(Protocol):
    """视频采集后端接口。"""

    def open(self, source: str, **kwargs) -> bool:
        """打开采集源（设备名/URL/文件路径），返回是否成功。"""
        ...

    def read(self) -> Frame | None:
        """读取下一帧，无帧时返回 None。"""
        ...

    def close(self) -> None:
        """关闭采集源。"""
        ...

    def is_open(self) -> bool:
        """是否处于打开状态。"""
        ...


@runtime_checkable
class DetectorBackend(Protocol):
    """目标检测后端接口。

    实现者只需返回 list[Detection]，Detection 已在 ai/detection.py 中定义。
    """

    def load_model(self, model_path: str, **kwargs) -> None:
        ...

    def predict(self, frame: np.ndarray, conf: float = 0.5) -> list[Detection]:
        """对 RGB ndarray 执行检测，返回归一化坐标的 Detection 列表。"""
        ...


@runtime_checkable
class TrackerBackend(Protocol):
    """目标跟踪后端接口。

    现有 DetectionTracker (ai/tracker.py) 已实现此接口。
    """

    def update(self, detections: list[Detection]) -> list[dict]:
        """用检测结果更新跟踪，返回带 track_id 的 dict 列表。"""
        ...

    def predict_step(self) -> list[dict]:
        """跳帧时用卡尔曼预测填充，返回预测轨迹。"""
        ...

    def reset(self) -> None:
        """重置跟踪器。"""
        ...
