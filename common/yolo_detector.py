"""PyTorch YOLO 检测后端 - 最简实现。

直接使用 ultralytics YOLO .pt 模型，避开 InferWorker 的复杂性。
返回 ai.detection.Detection 列表，坐标归一化到 [0,1]。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ai.detection import Detection

logger = logging.getLogger(__name__)


class YOLODetector:
    """ultralytics YOLO .pt 模型的薄封装。

    实现 common.DetectorBackend 接口（结构化子类型，无需显式继承）。
    """

    def __init__(self, model_path: str = "models/yolov8n.pt", device: str = "cpu"):
        self._model = None
        self._model_path = model_path
        self._device = device
        self._imgsz = 640

    def load_model(self, model_path: str, **kwargs) -> None:
        from ultralytics import YOLO

        self._model_path = str(model_path)
        self._device = kwargs.get("device", self._device)
        self._imgsz = kwargs.get("imgsz", 640)

        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        self._model = YOLO(self._model_path)
        logger.info("YOLO model loaded: %s (device=%s, imgsz=%d)",
                     self._model_path, self._device, self._imgsz)

    def predict(self, frame: np.ndarray, conf: float = 0.5) -> list[Detection]:
        """对 RGB ndarray 执行检测，返回归一化 Detection 列表。

        关键修复：ultralytics 在某些情况下可能返回 None，
        此处做显式守卫，避免 P0 级 TypeError。
        """
        if self._model is None:
            return []

        try:
            results = self._model(
                frame,
                verbose=False,
                conf=conf,
                imgsz=self._imgsz,
                device=self._device,
            )
        except Exception as e:
            logger.error("YOLO inference failed: %s", e)
            return []

        # P0 修复：显式处理 None
        if results is None:
            return []

        detections: list[Detection] = []

        for r in results:
            if r is None or r.boxes is None:
                continue

            boxes = r.boxes
            if len(boxes) == 0:
                continue

            try:
                xyxyn = boxes.xyxyn.cpu().numpy()    # (N, 4) 归一化
                confs = boxes.conf.cpu().numpy()      # (N,)
                clss = boxes.cls.cpu().numpy().astype(int)  # (N,)
                names = r.names
            except Exception as e:
                logger.error("Failed to parse YOLO output: %s", e)
                continue

            for i in range(len(clss)):
                cls_id = int(clss[i])
                label = names.get(cls_id, str(cls_id))
                x1, y1, x2, y2 = [float(v) for v in xyxyn[i]]

                detections.append(Detection(
                    class_id=cls_id,
                    label=label,
                    confidence=float(confs[i]),
                    bbox=(x1, y1, x2, y2),
                    polygon=None,
                    keypoints=None,
                    track_id=None,
                    track_state=None,
                ))

        return detections
