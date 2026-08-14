from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


class UhdBackend:
    """UHD 64x64 轻量人体检测 ONNX 后端。

    输入：RGB [1, 3, 64, 64]，通过 INTER_NEAREST resize + /255.0 归一化。
    输出：与现有检测 schema 对齐的 detection list，仅输出 ``person`` 类别。

    使用仓库中 ``*_static.onnx`` 含后处理版本，输出名 ``score_classid_cxcywh``，
    形状 ``[1, 100, 6]``，其中 ``cx, cy, w, h`` 为 64x64 输入上的归一化坐标。
    由于预处理采用直接拉伸（非 letterbox），归一化坐标可直接映射回原始帧。
    """

    _INPUT_SIZE = 64

    def __init__(self, model_path: str | Path, *, provider_mode: str = "auto"):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"UHD model not found: {self.model_path}")

        try:
            import onnxruntime as ort
        except Exception as err:
            raise RuntimeError(f"onnxruntime is unavailable: {err}") from err

        available = ort.get_available_providers()
        providers = self._select_providers(provider_mode, available)
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            self.model_path, sess_options=session_options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [output.name for output in self._session.get_outputs()]
        logging.info(
            "UHD backend loaded: path=%s providers=%s outputs=%s",
            self.model_path,
            self._session.get_providers(),
            self._output_names,
        )

    @staticmethod
    def _select_providers(provider_mode: str, available: list[str]) -> list[str]:
        mode = str(provider_mode or "auto").lower()
        if mode == "cpu":
            if "CPUExecutionProvider" not in available:
                raise RuntimeError(f"CPUExecutionProvider unavailable: {available}")
            return ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider"]
            if "CPUExecutionProvider" in available:
                providers.append("CPUExecutionProvider")
            return providers
        if "CPUExecutionProvider" in available:
            logging.warning("DmlExecutionProvider unavailable, falling back to CPU for UHD")
            return ["CPUExecutionProvider"]
        raise RuntimeError(f"No usable execution provider available: {available}")

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def warmup(self) -> None:
        dummy = np.zeros((self._INPUT_SIZE, self._INPUT_SIZE, 3), dtype=np.uint8)
        self.predict(dummy, conf=0.25)

    def predict(self, frame: np.ndarray, *, conf: float = 0.25) -> list[dict]:
        input_tensor, meta = self._preprocess(frame)
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})
        return self._decode_postprocessed(outputs[0], meta, conf)

    def __call__(self, frame: np.ndarray, *, conf: float = 0.25, imgsz: int | None = None, verbose: bool = False):
        _ = imgsz, verbose
        return self.predict(frame, conf=conf)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img,
            (self._INPUT_SIZE, self._INPUT_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # [1, 3, 64, 64]
        return img, {"frame_w": w, "frame_h": h}

    def _decode_postprocessed(
        self, output: np.ndarray, meta: dict, conf: float
    ) -> list[dict]:
        """解码 ``[1, 100, 6]`` -> ``[score, class_id, cx, cy, w, h]``。

        ``cx, cy, w, h`` 为 64x64 输入上的归一化坐标；预处理为直接拉伸，
        因此可直接作为原始帧的归一化坐标使用。
        """
        pred = np.asarray(output)
        if pred.ndim == 3:
            pred = pred[0]
        detections = []
        for det in pred:
            score = float(det[0])
            if score < conf:
                continue
            cx = float(det[2])
            cy = float(det[3])
            bw = float(det[4])
            bh = float(det[5])
            x1 = cx - bw * 0.5
            y1 = cy - bh * 0.5
            x2 = cx + bw * 0.5
            y2 = cy + bh * 0.5
            detections.append({
                "x1": _clamp01(x1),
                "y1": _clamp01(y1),
                "x2": _clamp01(x2),
                "y2": _clamp01(y2),
                "confidence": score,
                "label": "person",
                "keypoints": [],
            })
        return detections


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
