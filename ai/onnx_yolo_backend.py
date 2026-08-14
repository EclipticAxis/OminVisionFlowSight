from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ai.detection import Detection
from ai.model_task import ModelTask


_COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


class OnnxYoloBackend:

    def __init__(self, model_path: str | Path, *, task: ModelTask, provider_mode: str = "auto", imgsz: int = 640):
        self.model_path = str(model_path)
        self.task = task
        self.imgsz = int(imgsz)
        if task not in {ModelTask.DETECT, ModelTask.POSE, ModelTask.OBB}:
            raise ValueError(f"Unsupported ONNX YOLO task: {task}")
        if task is ModelTask.OBB:
            raise NotImplementedError("YOLO OBB ONNX support is reserved for Phase 2")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        try:
            import onnxruntime as ort
        except Exception as err:
            raise RuntimeError(f"onnxruntime is unavailable: {err}") from err

        available = ort.get_available_providers()
        providers = self._select_providers(provider_mode, available)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self._session = ort.InferenceSession(self.model_path, sess_options=session_options, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]
        self._output_shapes = [output.shape for output in self._session.get_outputs()]
        logging.info(
            "ONNX YOLO backend loaded: task=%s path=%s available_providers=%s session_providers=%s outputs=%s",
            self.task.name,
            self.model_path,
            available,
            self._session.get_providers(),
            list(zip(self._output_names, self._output_shapes)),
        )

    @staticmethod
    def _task_str(task: ModelTask) -> str:
        if task is ModelTask.POSE:
            return "pose"
        return "detect"

    @staticmethod
    def _select_providers(provider_mode: str, available: list[str]) -> list[str]:
        mode = str(provider_mode or "auto").lower()
        if mode == "directml":
            if "DmlExecutionProvider" not in available:
                raise RuntimeError(f"DmlExecutionProvider unavailable: {available}")
            return ["DmlExecutionProvider"]
        if mode == "cpu":
            if "CPUExecutionProvider" not in available:
                raise RuntimeError(f"CPUExecutionProvider unavailable: {available}")
            return ["CPUExecutionProvider"]
        if "DmlExecutionProvider" not in available:
            raise RuntimeError(f"DmlExecutionProvider unavailable: {available}")
        providers = ["DmlExecutionProvider"]
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def warmup(self) -> None:
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.predict(dummy, conf=0.25)

    def predict(self, frame: np.ndarray, *, conf: float = 0.25) -> list[Detection]:
        input_tensor, meta = self._preprocess(frame)
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        pred = self._normalize_output(outputs[0])
        if self.task is ModelTask.POSE:
            return self._decode_pose(pred, meta, conf)
        return self._decode_detect(pred, meta, conf)

    def __call__(self, frame: np.ndarray, *, conf: float = 0.25, imgsz: int | None = None, verbose: bool = False):
        _ = imgsz, verbose
        return self.predict(frame, conf=conf)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]
        scale = min(self.imgsz / max(w, 1), self.imgsz / max(h, 1))
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_x = (self.imgsz - new_w) // 2
        pad_y = (self.imgsz - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        return tensor, {"frame_w": w, "frame_h": h, "scale": scale, "pad_x": pad_x, "pad_y": pad_y}

    @staticmethod
    def _normalize_output(output: np.ndarray) -> np.ndarray:
        pred = np.asarray(output)
        if pred.ndim == 3:
            pred = pred[0]
        if pred.ndim != 2:
            raise RuntimeError(f"Unsupported ONNX output shape: {output.shape}")
        if pred.shape[0] < pred.shape[1] and pred.shape[0] in {56, 84, 85}:
            pred = pred.T
        elif pred.shape[0] < 128 and pred.shape[1] > 128:
            pred = pred.T
        return pred.astype(np.float32, copy=False)

    def _decode_detect(self, pred: np.ndarray, meta: dict, conf: float) -> list[Detection]:
        if pred.shape[1] < 5:
            return []
        boxes = pred[:, :4]
        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_ids]
        keep = confidences >= conf
        detections = []
        for box, cls_id, score in zip(boxes[keep], class_ids[keep], confidences[keep]):
            mapped = self._xywh_to_norm_xyxy(box, meta)
            if mapped is None:
                continue
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    label=_COCO_NAMES.get(int(cls_id), str(int(cls_id))),
                    confidence=float(score),
                    bbox=mapped,
                )
            )
        return _nms(detections, 0.45)

    def _decode_pose(self, pred: np.ndarray, meta: dict, conf: float) -> list[Detection]:
        channels = pred.shape[1]
        if channels < 56:
            return []
        if channels == 56:
            score_start = 4
            kpt_start = 5
            class_ids = np.zeros(pred.shape[0], dtype=np.int32)
            confidences = pred[:, score_start]
        else:
            kpt_start = channels - 51
            score_block = pred[:, 4:kpt_start]
            if score_block.shape[1] == 0:
                return []
            class_ids = np.argmax(score_block, axis=1)
            confidences = score_block[np.arange(score_block.shape[0]), class_ids]

        from ai.detection import Keypoint

        keep = confidences >= conf
        detections = []
        for row, cls_id, score in zip(pred[keep], class_ids[keep], confidences[keep]):
            mapped = self._xywh_to_norm_xyxy(row[:4], meta)
            if mapped is None:
                continue
            keypoints = self._decode_keypoints(row[kpt_start:kpt_start + 51], meta)
            detections.append(
                Detection(
                    class_id=int(cls_id),
                    label=_COCO_NAMES.get(int(cls_id), "person"),
                    confidence=float(score),
                    bbox=mapped,
                    keypoints=[Keypoint(x=kp["x"], y=kp["y"], conf=kp["conf"]) for kp in keypoints],
                )
            )
        return _nms(detections, 0.45)

    def _xywh_to_norm_xyxy(self, box: np.ndarray, meta: dict) -> tuple[float, float, float, float] | None:
        cx, cy, bw, bh = [float(v) for v in box]
        x1 = cx - bw * 0.5
        y1 = cy - bh * 0.5
        x2 = cx + bw * 0.5
        y2 = cy + bh * 0.5
        x1 = (x1 - meta["pad_x"]) / meta["scale"]
        y1 = (y1 - meta["pad_y"]) / meta["scale"]
        x2 = (x2 - meta["pad_x"]) / meta["scale"]
        y2 = (y2 - meta["pad_y"]) / meta["scale"]
        frame_w = float(meta["frame_w"])
        frame_h = float(meta["frame_h"])
        x1 = _clamp01(x1 / frame_w)
        y1 = _clamp01(y1 / frame_h)
        x2 = _clamp01(x2 / frame_w)
        y2 = _clamp01(y2 / frame_h)
        if x2 - x1 <= 1e-5 or y2 - y1 <= 1e-5:
            return None
        return x1, y1, x2, y2

    def _decode_keypoints(self, values: np.ndarray, meta: dict) -> list[dict]:
        if values.size < 51:
            return []
        kpts = values.reshape(17, 3)
        frame_w = float(meta["frame_w"])
        frame_h = float(meta["frame_h"])
        decoded = []
        for x, y, score in kpts:
            orig_x = (float(x) - meta["pad_x"]) / meta["scale"]
            orig_y = (float(y) - meta["pad_y"]) / meta["scale"]
            decoded.append({"x": _clamp01(orig_x / frame_w), "y": _clamp01(orig_y / frame_h), "conf": float(score)})
        return decoded


def _nms(detections: list[Detection], iou_threshold: float) -> list[Detection]:
    ordered = sorted(detections, key=lambda det: float(det.confidence), reverse=True)
    kept: list[Detection] = []
    for det in ordered:
        if any(det.label == existing.label and _box_iou(det, existing) > iou_threshold for existing in kept):
            continue
        kept.append(det)
    return kept


def _box_iou(a: Detection, b: Detection) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), float(by2))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, float(ax2) - float(ax1)) * max(0.0, float(ay2) - float(ay1))
    area_b = max(0.0, float(bx2) - float(bx1)) * max(0.0, float(by2) - float(by1))
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
