"""ReID（行人重识别）后端插件 — OSNet ONNX 真实模型 + mock 双模式。

提供目标外观的深度特征 embedding，用于级联匹配 Stage 2 / 匈牙利匹配的
相似目标区分（0.4·IoU + 0.4·ReID + 0.2·center 代价）。

真实模式：加载 OSNet ONNX 模型（如 PINTO0309 转换的 osnet_x0_25_msmt17），
预处理契约：cv2.resize(roi, (W=128, H=256)) → BGR→RGB → float32/255 →
ImageNet 归一化（mean=[0.485,0.456,0.406]、std=[0.229,0.224,0.225]，RGB 通道序）
→ CHW → batch (1,3,256,128)；输出 flatten → float32 → L2 归一化。
输出维度自适应（本模型 512 维），EMBEDDING_DIM=256 仅 mock 使用。

mock 模式：不依赖任何模型文件，基于 ROI 像素哈希生成确定性 256 维向量，
仅供测试与管线验证；tracker 双側（ai/tracker.py / ai/tracker_backend.py）
通过 is_mock 门控保证 mock 时 ReID 匹配不会激活（不注入噪声）。
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ReIDBackend:
    """ReID embedding 提取后端（OSNet ONNX 真实模型 / mock 双模式）。

    用法：
        backend = ReIDBackend()                    # 自动发现 models/osnet*.onnx，失败则 mock
        backend = ReIDBackend("models/osnet_x0_25_256x128_msmt17.onnx")  # 真实模式
        backend.is_mock                              # → bool（mock 时匹配不激活）
        emb = backend.extract(frame_roi)             # → np.ndarray(512,) 或 None
        sim = ReIDBackend.cosine_similarity(a, b)    # → float
    """

    EMBEDDING_DIM = 256  # 仅 mock 模式使用；真实模式输出维度自适应（OSNet-x0.25 为 512）

    def __init__(self, model_path: str | None = None):
        self._session = None
        self._input_name = None
        self._model_path = model_path

        if model_path is None or model_path == "":
            # 自动发现：优先固定文件名，其次 glob models/osnet*.onnx 取首个
            model_path = self._discover_model_path()
            if model_path is None:
                self._mock = True
                logger.warning("ReID mock 模式仅供管线验证，匹配不会激活")
                return
            self._model_path = model_path

        self._mock = False
        self._load_model(model_path)
        if not self._mock:
            logger.info("ReIDBackend loaded model: %s", model_path)

    @property
    def is_mock(self) -> bool:
        """是否 mock 模式。mock 时 tracker 双側门控保证 ReID 匹配不激活。"""
        return self._mock

    @staticmethod
    def _discover_model_path() -> str | None:
        """自动发现 OSNet 模型：固定文件名优先，其次 models/osnet*.onnx 首个。"""
        models_dir = Path.cwd() / "models"
        preferred = models_dir / "osnet_x0_25_256x128_msmt17.onnx"
        if preferred.exists():
            return str(preferred)
        candidates = sorted(models_dir.glob("osnet*.onnx"), key=lambda p: p.name.lower())
        if candidates:
            return str(candidates[0])
        return None

    def _load_model(self, model_path: str) -> None:
        """加载 ONNX 模型；失败回退 mock（不炸应用，保持韧性）。"""
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
            self._input_name = None

    def extract(self, frame_roi: np.ndarray) -> np.ndarray | None:
        """提取 ROI 的 L2 归一化 embedding（真实模式维度自适应，本模型 512 维）。

        mock 模式：基于 ROI 像素内容哈希生成确定性 256 维向量（同 ROI → 同向量）。
        真实模式：256x128（HxW）resize + ImageNet 归一化 → ONNX 推理 → L2 归一化；
        推理异常时返回 None（该检测本帧无 embedding，匹配自然回退 HSV 外观路径），
        不回退 mock 向量。

        Args:
            frame_roi: BGR ndarray (H, W, 3)

        Returns:
            L2 归一化向量，或 None（ROI 无效 / 真实模式推理失败时）
        """
        if frame_roi is None or frame_roi.size == 0:
            return None

        if self._mock:
            return self._mock_extract(frame_roi)
        else:
            return self._real_extract(frame_roi)

    def _mock_extract(self, roi: np.ndarray) -> np.ndarray:
        """mock 模式：基于像素哈希的确定性伪 embedding（256 维）。

        保证同一 ROI 产生同一向量，不同 ROI 产生不同向量，
        仅用于验证 ReID 集成管线；生产匹配由 is_mock 门控关闭。
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
        """真实模式：OSNet ONNX 推理提取 embedding（512 维，L2 归一化）。

        预处理契约：cv2.resize(roi, (W=128, H=256)) → BGR→RGB → float32/255 →
        ImageNet 归一化（RGB 通道序 mean/std）→ CHW → batch (1,3,256,128)。
        任何异常 → logger.warning + 返回 None（不回退 mock 向量）。
        """
        try:
            import cv2

            resized = cv2.resize(roi, (128, 256))  # (W, H) = (128, 256)
            if resized.ndim != 3 or resized.shape[2] != 3:
                logger.warning("ReID extract failed: unexpected roi shape %s", resized.shape)
                return None
            rgb = resized[:, :, ::-1]  # BGR → RGB
            normalized = rgb.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            normalized = (normalized - mean) / std
            chw = np.transpose(normalized, (2, 0, 1))
            batch = np.expand_dims(chw, axis=0).astype(np.float32)

            outputs = self._session.run(None, {self._input_name: batch})
            emb = outputs[0].flatten().astype(np.float32)

            # L2 归一化
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                emb = emb / norm
            return emb
        except Exception as e:
            logger.warning("ReID extract failed, this detection gets no embedding: %s", e)
            return None

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
