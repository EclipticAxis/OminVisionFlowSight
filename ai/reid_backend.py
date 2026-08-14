"""ReID（行人重识别）后端插件 — 轻量级接口 + mock 实现。

提供目标外观的深度特征 embedding，用于级联匹配 Stage 2 的相似目标区分。

mock 模式：不依赖任何模型文件，基于 ROI 像素哈希生成确定性 256 维向量，
验证完整管线后放入 OSNet-x0.25 ONNX 模型即可生效。

真实模式（预留）：加载 ONNX 模型，输入 128x64 RGB，输出 256 维 L2 归一化 embedding。
"""
from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class ReIDBackend:
    """ReID embedding 提取后端。

    用法：
        backend = ReIDBackend()                    # mock 模式
        backend = ReIDBackend("models/osnet.onnx") # 真实模式
        emb = backend.extract(frame_roi)           # → np.ndarray(256,) 或 None
        sim = ReIDBackend.cosine_similarity(a, b)  # → float
    """

    EMBEDDING_DIM = 256

    def __init__(self, model_path: str | None = None):
        self._session = None
        self._mock = (model_path is None or model_path == "")
        self._model_path = model_path

        if not self._mock:
            self._load_model(model_path)
            logger.info("ReIDBackend loaded model: %s", model_path)
        else:
            logger.info("ReIDBackend initialized in mock mode (no model file)")

    def _load_model(self, model_path: str) -> None:
        """加载 ONNX 模型（预留接口）。"""
        try:
            import onnxruntime as ort

            # 尝试 DirectML → CPU 回退
            try:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
                self._session = ort.InferenceSession(model_path, providers=providers)
            except Exception:
                self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

            self._input_name = self._session.get_inputs()[0].name
            logger.info("ReID ONNX session created with providers: %s", self._session.get_providers())
        except Exception as e:
            logger.warning("ReID model load failed, falling back to mock: %s", e)
            self._mock = True
            self._session = None

    def extract(self, frame_roi: np.ndarray) -> np.ndarray | None:
        """提取 ROI 的 256 维 embedding。

        mock 模式：基于 ROI 像素内容哈希生成确定性向量（同 ROI → 同向量）。
        真实模式：resize 128x64 → ONNX 推理 → L2 归一化。

        Args:
            frame_roi: BGR 或 RGB ndarray (H, W, 3)

        Returns:
            256 维 L2 归一化向量，或 None（ROI 无效时）
        """
        if frame_roi is None or frame_roi.size == 0:
            return None

        if self._mock:
            return self._mock_extract(frame_roi)
        else:
            return self._real_extract(frame_roi)

    def _mock_extract(self, roi: np.ndarray) -> np.ndarray:
        """mock 模式：基于像素哈希的确定性伪 embedding。

        保证同一 ROI 产生同一向量，不同 ROI 产生不同向量，
        可验证 ReID 集成管线是否正确工作。
        """
        # 下采样到 8x4 减少哈希计算量
        small = roi[::max(1, roi.shape[0] // 4), ::max(1, roi.shape[1] // 8)]
        # 基于像素内容生成确定性 seed
        seed = hash(small.tobytes()) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        emb = rng.standard_normal(self.EMBEDDING_DIM).astype(np.float32)
        # L2 归一化
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm
        return emb

    def _real_extract(self, roi: np.ndarray) -> np.ndarray | None:
        """真实模式：ONNX 推理提取 embedding。"""
        if self._session is None:
            return self._mock_extract(roi)

        try:
            import cv2

            # resize 到模型输入尺寸 128x64 (HxW)
            resized = cv2.resize(roi, (64, 128))
            # BGR → RGB，归一化 [0,1]，CHW 转置
            if resized.shape[2] == 3:
                resized = resized[:, :, ::-1]  # BGR → RGB
            normalized = resized.astype(np.float32) / 255.0
            chw = np.transpose(normalized, (2, 0, 1))
            batch = np.expand_dims(chw, axis=0)

            outputs = self._session.run(None, {self._input_name: batch})
            emb = outputs[0].flatten().astype(np.float32)

            # L2 归一化
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.warning("ReID extract failed, using mock: %s", e)
            return self._mock_extract(roi)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度，返回 [-1, 1]。"""
        if a is None or b is None:
            return 0.0
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
