import logging
from pathlib import Path

import numpy as np

from ai.detection import Detection, Keypoint
from ai.model_task import ModelTask


class Yolo26Backend:
    """Ultralytics YOLO26 模型（.pt）后端封装。

    支持任务：
    - DETECT：普通目标检测，输出 bbox。
    - POSE：姿态检测，输出 bbox + 17 个 COCO 关键点。
    - OBB：有向目标检测，输出 bbox（外接矩形）+ polygon（四边形角点）。

    所有坐标均归一化到 [0, 1]，与现有系统保持一致。
    """

    def __init__(self, model_path: str, task: ModelTask = ModelTask.DETECT):
        self.model_path = str(model_path)
        self.task = task
        self._task_str = self._task_to_ultralytics_str(task)
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path, task=self._task_str)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO26 model {self.model_path}: {e}") from e
        self.providers: list[str] = ["pytorch"]
        logging.info("Yolo26Backend loaded path=%s task=%s", self.model_path, self.task.name)

    @staticmethod
    def _task_to_ultralytics_str(task: ModelTask) -> str | None:
        """Ultralytics YOLO 构造函数的 task 参数：obb 可识别，detect/pose 用 None 让模型自动推断。"""
        if task is ModelTask.OBB:
            return "obb"
        if task is ModelTask.POSE:
            return "pose"
        return None  # detect

    def predict(self, frame: np.ndarray, conf: float = 0.25) -> list[Detection]:
        """对单帧进行推理，返回 Detection 列表。"""
        if frame is None or frame.size == 0:
            return []
        try:
            results = self._model(frame, verbose=False, conf=conf)
        except Exception as e:
            logging.error("Yolo26Backend inference failed: %s", e)
            return []
        return self._parse_results(results, frame.shape[1], frame.shape[0])

    def _parse_results(self, results, frame_w: int, frame_h: int) -> list[Detection]:
        detections: list[Detection] = []
        for r in results:
            if r is None:
                continue
            names = r.names if hasattr(r, "names") else {}
            if self.task is ModelTask.OBB and getattr(r, "obb", None) is not None:
                detections.extend(self._parse_obb(r, names, frame_w, frame_h))
            elif self.task is ModelTask.POSE and getattr(r, "boxes", None) is not None:
                detections.extend(self._parse_pose(r, names, frame_w, frame_h))
            elif getattr(r, "boxes", None) is not None:
                detections.extend(self._parse_detect(r, names, frame_w, frame_h))
        return detections

    def _parse_detect(self, result, names: dict[int, str], frame_w: int, frame_h: int) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or boxes.cls is None:
            return []
        detections: list[Detection] = []
        for cls, conf, xyxy in zip(boxes.cls, boxes.conf, boxes.xyxy):
            x1, y1, x2, y2 = xyxy.cpu().numpy().astype(float)
            detections.append(Detection(
                class_id=int(cls),
                label=self._lookup_name(names, int(cls)),
                confidence=float(conf),
                bbox=(x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h),
            ))
        return detections

    def _parse_obb(self, result, names: dict[int, str], frame_w: int, frame_h: int) -> list[Detection]:
        obb = result.obb
        if obb is None or obb.cls is None:
            return []
        detections: list[Detection] = []
        for cls, conf, xyxyxyxy in zip(obb.cls, obb.conf, obb.xyxyxyxy):
            pts = xyxyxyxy.cpu().numpy().astype(float).reshape(4, 2)
            pts_norm = pts / np.array([[frame_w, frame_h]], dtype=float)
            pts_norm = np.clip(pts_norm, 0.0, 1.0)
            xmin, ymin = pts_norm.min(axis=0)
            xmax, ymax = pts_norm.max(axis=0)
            detections.append(Detection(
                class_id=int(cls),
                label=self._lookup_name(names, int(cls)),
                confidence=float(conf),
                bbox=(float(xmin), float(ymin), float(xmax), float(ymax)),
                polygon=[(float(x), float(y)) for x, y in pts_norm],
            ))
        return detections

    def _parse_pose(self, result, names: dict[int, str], frame_w: int, frame_h: int) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or boxes.cls is None:
            return []
        kpts = getattr(result, "keypoints", None)
        if kpts is None or kpts.xy is None:
            return self._parse_detect(result, names, frame_w, frame_h)

        detections: list[Detection] = []
        keypoints_xy = kpts.xy.cpu().numpy().astype(float)
        keypoints_conf = kpts.conf.cpu().numpy().astype(float) if kpts.conf is not None else None

        for i, (cls, conf, xyxy) in enumerate(zip(boxes.cls, boxes.conf, boxes.xyxy)):
            x1, y1, x2, y2 = xyxy.cpu().numpy().astype(float)
            kps: list[Keypoint] = []
            if i < len(keypoints_xy):
                kps_xy = keypoints_xy[i] / np.array([[frame_w, frame_h]], dtype=float)
                kps_conf = keypoints_conf[i] if keypoints_conf is not None else None
                for j, (px, py) in enumerate(kps_xy):
                    kc = float(kps_conf[j]) if kps_conf is not None and j < len(kps_conf) else 1.0
                    kps.append(Keypoint(x=float(px), y=float(py), conf=kc))
            detections.append(Detection(
                class_id=int(cls),
                label=self._lookup_name(names, int(cls)),
                confidence=float(conf),
                bbox=(x1 / frame_w, y1 / frame_h, x2 / frame_w, y2 / frame_h),
                keypoints=kps if kps else None,
            ))
        return detections

    @staticmethod
    def _lookup_name(names: dict[int, str], class_id: int) -> str:
        if class_id in names:
            return str(names[class_id])
        # COCO 80 fallback
        return _COCO_NAMES.get(class_id, f"class_{class_id}")


# COCO 80 类名称（fallback）
_COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane", 5: "bus",
    6: "train", 7: "truck", 8: "boat", 9: "traffic light", 10: "fire hydrant",
    11: "stop sign", 12: "parking meter", 13: "bench", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 20: "elephant", 21: "bear",
    22: "zebra", 23: "giraffe", 24: "backpack", 25: "umbrella", 26: "handbag",
    27: "tie", 28: "suitcase", 29: "frisbee", 30: "skis", 31: "snowboard",
    32: "sports ball", 33: "kite", 34: "baseball bat", 35: "baseball glove",
    36: "skateboard", 37: "surfboard", 38: "tennis racket", 39: "bottle",
    40: "wine glass", 41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed", 60: "dining table",
    61: "toilet", 62: "tv", 63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard",
    67: "cell phone", 68: "microwave", 69: "oven", 70: "toaster", 71: "sink",
    72: "refrigerator", 73: "book", 74: "clock", 75: "vase", 76: "scissors",
    77: "teddy bear", 78: "hair drier", 79: "toothbrush",
}
