from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


class HeadClassificationBackend:
    """Comprehensive-Head-Classification ONNX 后端。

    基于 person 检测框近似裁剪人头/眼/嘴区域，输出人脸属性概率：
    ``bg_plain``、``masked``、``sunglass``、``hat``、``eye_open``、``mouth_open``。

    使用 ``chc_s_wo_fiqa.onnx``（无 FIQA 分支）可降低计算量；
    如需图像质量评估，可使用含 FIQA 的版本并在初始化时自动识别。
    """

    def __init__(self, model_path: str | Path, *, provider_mode: str = "auto"):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Head classification model not found: {self.model_path}")

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
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._has_fiqa = "head_image_352x352" in self._input_names
        logging.info(
            "HeadClassification backend loaded: path=%s providers=%s has_fiqa=%s outputs=%s",
            self.model_path,
            self._session.get_providers(),
            self._has_fiqa,
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
            logging.warning("DmlExecutionProvider unavailable, falling back to CPU for head classification")
            return ["CPUExecutionProvider"]
        raise RuntimeError(f"No usable execution provider available: {available}")

    @property
    def providers(self) -> list[str]:
        return list(self._session.get_providers())

    def warmup(self) -> None:
        h, w = 480, 640
        det = {"x1": 0.3, "y1": 0.2, "x2": 0.7, "y2": 0.9}
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        self.predict(frame, det)

    @staticmethod
    def _safe_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return frame[y1:y2, x1:x2]

    def _norm_hwc(self, roi: np.ndarray, size: tuple[int, int]) -> np.ndarray:
        """将 BGR ROI 转为 CHW 归一化张量。"""
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        return np.transpose(img, (2, 0, 1))

    def predict(self, frame: np.ndarray, det: dict) -> dict | None:
        """对单个 person 检测框进行属性分类。"""
        if frame is None or frame.size == 0:
            return None
        h, w = frame.shape[:2]
        x1 = int(det["x1"] * w)
        y1 = int(det["y1"] * h)
        x2 = int(det["x2"] * w)
        y2 = int(det["y2"] * h)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None

        # 人头框：取 person 框上半部分（头+肩），约占高度 0~45%
        head_top = y1
        head_bottom = y1 + int((y2 - y1) * 0.45)
        head_roi = self._safe_crop(frame, (x1, head_top, x2, head_bottom))
        if head_roi is None:
            return None

        head_48 = self._norm_hwc(head_roi, (48, 48))[None]
        inputs: dict[str, np.ndarray] = {"head_image_48x48": head_48}

        # 眼睛区域：人头框的 30%~60% 高度，水平收缩到 15%~85%
        eye_top = head_top + int((head_bottom - head_top) * 0.30)
        eye_bottom = head_top + int((head_bottom - head_top) * 0.60)
        eye_left = x1 + int((x2 - x1) * 0.15)
        eye_right = x1 + int((x2 - x1) * 0.85)
        eye_roi = self._safe_crop(frame, (eye_left, eye_top, eye_right, eye_bottom))
        if eye_roi is not None:
            eye_h, eye_w = eye_roi.shape[:2]
            left_eye = self._norm_hwc(eye_roi[:, : eye_w // 2], (40, 24))
            right_eye = self._norm_hwc(eye_roi[:, eye_w // 2 :], (40, 24))
            eyes = np.stack([left_eye, right_eye], axis=0)  # [2, 3, 24, 40]
            inputs["eye_images_24x40"] = eyes
        else:
            inputs["eye_images_24x40"] = np.zeros((2, 3, 24, 40), dtype=np.float32)

        # 嘴巴区域：人头框的 60%~95% 高度，水平收缩到 25%~75%
        mouth_top = head_top + int((head_bottom - head_top) * 0.60)
        mouth_bottom = head_top + int((head_bottom - head_top) * 0.95)
        mouth_left = x1 + int((x2 - x1) * 0.25)
        mouth_right = x1 + int((x2 - x1) * 0.75)
        mouth_roi = self._safe_crop(frame, (mouth_left, mouth_top, mouth_right, mouth_bottom))
        if mouth_roi is not None:
            mouth = self._norm_hwc(mouth_roi, (48, 30))[None]
            inputs["mouth_image_30x48"] = mouth
        else:
            inputs["mouth_image_30x48"] = np.zeros((1, 3, 30, 48), dtype=np.float32)

        # FIQA 可选分支（352x352，需 ImageNet 标准化）
        if self._has_fiqa:
            head_352 = self._norm_hwc(head_roi, (352, 352))[None]
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            inputs["head_image_352x352"] = (head_352 - mean) / std

        try:
            outputs = self._session.run(self._output_names, inputs)
        except Exception as e:
            logging.warning("Head classification inference failed: %s", e)
            return None

        return self._decode_outputs(outputs)

    def _decode_outputs(self, outputs: list[np.ndarray]) -> dict:
        """按输出名称映射结果，避免依赖固定顺序。"""
        by_name: dict[str, np.ndarray] = {
            name: np.asarray(out) for name, out in zip(self._output_names, outputs)
        }
        return {
            "bg_plain": float(by_name["prob_bg_plain"].item()),
            "masked": float(by_name["prob_masked"].item()),
            "sunglass": float(by_name["prob_sunglass"].item()),
            "hat": float(by_name["prob_hat"].item()),
            "eye_open": [
                float(by_name["prob_eye_open"][0].item()),
                float(by_name["prob_eye_open"][1].item()),
            ],
            "mouth_open": float(by_name["prob_mouth_open"].item()),
            "quality_score": None,
        }
