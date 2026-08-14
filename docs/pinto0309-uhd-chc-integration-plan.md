# PINTO0309 UHD / Comprehensive-Head-Classification 集成方案

> 面向 VisionBata（VisionDataPlatform）PyQt6 多路实时视觉检测平台
> 基于仓库最新 README / Release / 示例代码，结合项目现有 `ai/inference.py`、`onnx_yolo_backend.py`、`tracker.py`、`gui/settings_dialog.py`、`gui/camera_cell.py` 架构进行设计。

> **状态：已实现（2026-07-09）**。新增/修改文件：`ai/uhd_backend.py`、`ai/head_classifier.py`、`ai/inference.py`、`gui/camera_cell.py`、`gui/settings_dialog.py`、`gui/main_window.py`；模型已下载至 `models/`。

---

## 1. 两个模型仓库技术摘要

### 1.1 UHD（Ultra-lightweight Human Detection）

| 项目 | 说明 |
|------|------|
| 仓库 | https://github.com/PINTO0309/UHD |
| 许可证 | MIT |
| 任务 | 单一类别人体检测（仅 `person`） |
| 设计哲学 | 64×64 输入已足够完成有限场景人体检测，反对过度使用 YOLO 等通用大模型 |
| 推荐桌面模型 | `ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_static.onnx`（约 8 MB，Corei9 0.60 ms） |
| 输入 | `[1, 3, 64, 64]` RGB，`float32`，`INTER_NEAREST` resize，通常 `/255.0` |
| 输出（含后处理） | `[1, 100, 6]` = `[score, class_id, cx, cy, w, h]`，已按 score 排序过滤 |
| 输出（无后处理） | `txtywh_obj_quality_cls_x8 [1,56,8,8]` + `anchors [8,2]` + `wh_scale [8,2]`，需 sigmoid/softplus 解码 |
| 后端 | ONNX / TFLite / ESPDL / OpenCV DNN / C++ |
| 关键变体 | `opencv_inter_nearest`（RGB 推荐）、`opencv_inter_nearest_yuv422`（摄像头直出 YUV422）、`distill`（远距离小目标）、`large-object-branch`（短中距离）、`ESE`/`IoU-aware`/`ReLU`/`Swish` |

**推荐选型**：直接使用 `*_static.onnx` **含后处理**版本，避免在 Python 端重新实现 anchor/quality 解码，集成成本最低。

### 1.2 Comprehensive-Head-Classification

| 项目 | 说明 |
|------|------|
| 仓库 | https://github.com/PINTO0309/Comprehensive-Head-Classification |
| 许可证 | MIT |
| 任务 | 单张人头/人脸区域的 7 任务二分类 |
| 推荐模型 | `chc_s_wo_fiqa.onnx`（无 FIQA，少一个 352×352 输入，计算量更低） |
| 输入 | `head_image_48x48 [1,3,48,48]`、`eye_images_24x40 [2,3,24,40]`、`mouth_image_30x48 [1,3,30,48]`、可选 `head_image_352x352 [1,3,352,352]`（FIQA） |
| 预处理 | 前三项：RGB /255.0；FIQA 项：RGB /255.0 后 ImageNet 标准化（mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]） |
| 输出 | `prob_bg_plain`、`prob_masked`、`prob_sunglass`、`prob_hat`、`prob_eye_open [2]`、`prob_mouth_open`、可选 `quality_score` |
| 后端 | ONNX / TFLite，支持 CPU / CUDA / TensorRT / WASM / WebGPU |

**关键约束**：CHC 不是检测器，而是**后处理属性分类器**。必须先获得人头/人脸框，再裁剪出眼、嘴区域，最后输入模型。

---

## 2. 与 VisionBata 现有架构的兼容性分析

### 2.1 现有架构回顾

```text
main.py
  └── gui/main_window.py
        ├── camera/capture.py        # 多路摄像头采集
        ├── ai/inference.py          # InferWorker（单一 QThread）
        │     ├── ai/onnx_yolo_backend.py  # ONNX YOLO detect/pose 后端
        │     ├── ai/tracker.py            # DetectionTracker（UKF/MCUKF/Manifold UKF）
        │     └── ai/rectangle_detector.py # 矩形检测
        └── gui/camera_cell.py       # 绘制 bbox / 标签 / 骨架
        └── gui/settings_dialog.py   # QSettings 配置持久化
```

- `InferWorker` 通过 `_runtime_backend` 区分 `pytorch` / `onnx`。
- `OnnxYoloBackend` 统一封装预处理、session、后处理，输出 `list[dict]`（`x1,y1,x2,y2,confidence,label,keypoints`）。
- `DetectionTracker` 只依赖 `label` 和 `x1..y2`，**不依赖具体模型来源**，可平滑接收 UHD 输出。
- 配置持久化通过 `QSettings(org=VisionBata, app=VisionDataPlatform)`，SettingsDialog 已具备模型下拉框、滤波器、去噪、ROI 重检测等配置项。

### 2.2 兼容性矩阵

| 维度 | UHD | Comprehensive-Head-Classification |
|------|-----|-----------------------------------|
| 与 ONNX Runtime 兼容 | 是，可用 `onnxruntime-directml` | 是 |
| 与现有 `OnnxYoloBackend` 接口兼容 | **否**（输入 64×64，输出 6 维，无 keypoints） | **否**（多输入、多输出、需先有人头框） |
| 可作为主检测后端 | **是**（仅 person） | 否（必须先有检测框） |
| 可作为辅助后端 | 是（轻量 person 检测） | **是**（属性融合） |
| 与 `DetectionTracker` 兼容 | 是（bbox 格式一致） | 不适用（不输出 bbox） |
| 与 ROI 重检测结合 | 是（UHD 可替换 YOLO 做 ROI 重检测） | 否（不参与重检测） |
| 与 DirectML 加速 | 是（session 走 DmlExecutionProvider） | 是 |

---

## 3. 集成策略总览

### 3.1 优先级推荐

**第一阶段：UHD（先做）**
- 价值最大：可作为主检测后端替代 YOLO，显著降低 CPU/GPU 占用。
- 改动清晰：单一输入/输出，可直接复用现有 `OnnxYoloBackend` 的设计模式。
- 风险可控：输出格式与现有检测 schema 一致，tracker 无需改动。

**第二阶段：Comprehensive-Head-Classification（后做）**
- 依赖第一阶段：需要先获得 person 检测框（来自 YOLO 或 UHD）。
- 改动更多：涉及多输入裁剪、属性融合、GUI 标签显示、独立配置。
- 收益：在已有检测基础上增加 hat / mask / sunglasses / eye-open / mouth-open 等属性。

### 3.2 模型文件放置

```text
models/
  ├── yolov8n.pt
  ├── yolov8n.onnx
  ├── yolov8n-pose.pt
  ├── yolov8n-pose.onnx
  ├── uhd_n_64x64_static.onnx          # 推荐：UHD N 含后处理
  ├── uhd_z_64x64_static.onnx          # 可选：更小更快
  ├── chc_s_wo_fiqa.onnx               # 推荐：无 FIQA 版本
  └── chc_s.onnx                       # 可选：含 FIQA
```

### 3.3 新增模块

```text
ai/
  ├── uhd_backend.py          # UHD 专用 ONNX 后端
  └── head_classifier.py      # CHC 属性分类器
```

---

## 4. UHD 详细集成方案

### 4.1 新增 `ai/uhd_backend.py`

设计目标：与 `OnnxYoloBackend` 接口保持一致，使 `InferWorker` 可以无差别调用 `predict(frame, conf=...)`。

```python
# ai/uhd_backend.py
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


class UhdBackend:
    """UHD 64x64 人体检测 ONNX 后端，输出与 YOLO 后端对齐的 detection list。"""

    _INPUT_SIZE = 64

    def __init__(self, model_path: str | Path, provider_mode: str = "directml"):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"UHD model not found: {self.model_path}")

        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = self._select_providers(provider_mode, available)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            self.model_path, sess_options=sess_options, providers=providers
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
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
            return ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

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

    def __call__(self, frame: np.ndarray, *, conf: float = 0.25, imgsz=None, verbose=False):
        return self.predict(frame, conf=conf)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, dict]:
        h, w = frame.shape[:2]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self._INPUT_SIZE, self._INPUT_SIZE), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]  # [1, 3, 64, 64]
        return img, {"frame_w": w, "frame_h": h}

    def _decode_postprocessed(
        self, output: np.ndarray, meta: dict, conf: float
    ) -> list[dict]:
        """输出 shape [1, 100, 6] = [score, class_id, cx, cy, w, h]（已排序）"""
        pred = np.asarray(output)
        if pred.ndim == 3:
            pred = pred[0]
        frame_w, frame_h = float(meta["frame_w"]), float(meta["frame_h"])
        detections = []
        for det in pred:
            score, cls_id, cx, cy, w, h = det
            if score < conf:
                continue
            x1 = (cx - w * 0.5) / frame_w
            y1 = (cy - h * 0.5) / frame_h
            x2 = (cx + w * 0.5) / frame_w
            y2 = (cy + h * 0.5) / frame_h
            detections.append({
                "x1": float(max(0.0, min(1.0, x1))),
                "y1": float(max(0.0, min(1.0, y1))),
                "x2": float(max(0.0, min(1.0, x2))),
                "y2": float(max(0.0, min(1.0, y2))),
                "confidence": float(score),
                "label": "person",
                "keypoints": [],
            })
        return detections
```

**注意**：如果下载了 `*_nopost.onnx`，需要额外实现 anchor/quality/softplus 解码，建议优先使用含后处理模型。

### 4.2 修改 `ai/inference.py`

#### 4.2.1 引入 UHD 后端并识别 UHD 模型

```python
# 在 onnx_yolo_backend 导入后增加
from ai.uhd_backend import UhdBackend


def _is_uhd_model(model_path: str | None) -> bool:
    """通过文件名或模型输入尺寸识别 UHD 模型。"""
    if not model_path:
        return False
    name = Path(model_path).name.lower()
    return "uhd" in name or "ultratinyod" in name
```

#### 4.2.2 让 `InferWorker` 支持 UHD 作为 detect 后端

当前 `_load_onnx_detect_model` 固定构造 `OnnxYoloBackend`。增加分支：

```python
def _load_onnx_detect_model(self, model_path: str | None = None) -> OnnxYoloBackend | UhdBackend:
    if model_path is None:
        resolved = _resolve_artifact_path("yolov8n.onnx")
        model_path = str(resolved) if resolved is not None else ""
    elif not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    if self._detect_model is not None and self._detect_model_path == model_path:
        return self._detect_model

    if _is_uhd_model(model_path):
        model = UhdBackend(model_path, provider_mode=self._backend_mode)
    else:
        model = OnnxYoloBackend(model_path, task="detect", provider_mode="directml", imgsz=self._model_imgsz)

    self._detect_model = model
    self._detect_model_path = model_path
    logging.info("ONNX Detect model loaded path=%s providers=%s", model_path, model.providers)
    return model
```

#### 4.2.3 `_select_inference_model` 适配 UHD

UHD 仅支持 detect，不支持 pose。当启用 skeleton / body gesture / hand gesture 时，需要 pose 模型（yolov8n-pose）提供关键点，再叠加 UHD 的 person 检测结果。

**策略 A（UHD 主检测）**：
- `person` 检测用 UHD；
- `pose` 仅在 skeleton/gesture 启用时跑 `yolov8n-pose`，用于提取关键点；
- 将 UHD 的 person bbox 与 pose 关键点通过 `keypoints_inside_bbox` 或 IoU 关联。

**策略 B（UHD 仅用于 ROI 重检测）**：
- 主检测仍用 YOLO；
- ROI 重检测时根据模型路径选择 UHD 或 YOLO。

推荐**策略 A**，充分发挥 UHD 轻量优势。最小改动版本可以先做策略 B（仅替换重检测后端），但收益较小。本方案以策略 A 为主。

```python
def _select_inference_model(self):
    if self._ai_disabled:
        return None
    # 需要姿态/手势时，加载 pose 模型用于关键点
    if self._skeleton_enabled or self._body_gesture_enabled() or self._hand_gesture_enabled():
        return self._load_pose_model()
    # 纯 person 检测：UHD 或 YOLO
    return self._load_detect_model(self._custom_model_path or None)
```

#### 4.2.4 检测结果融合：UHD bbox + YOLO pose keypoints

在 `_run_inference` 中，当 detect 模型是 `UhdBackend` 且 pose 启用时，将 pose 关键点按坐标匹配到 UHD 检测框：

```python
def _merge_uhd_with_pose(
    self,
    uhd_dets: list[dict],
    pose_dets: list[dict],
) -> list[dict]:
    """将 pose 关键点按中心点落入 UHD bbox 的原则进行融合。"""
    if not pose_dets:
        return uhd_dets

    merged = []
    for udet in uhd_dets:
        best_kpts = []
        best_score = -1.0
        for pdet in pose_dets:
            kpts = pdet.get("keypoints") or []
            if len(kpts) != 17:
                continue
            # 用关键点中心或鼻子点判断是否在 UHD 框内
            nose = kpts[0]
            if (
                udet["x1"] <= nose["x"] <= udet["x2"]
                and udet["y1"] <= nose["y"] <= udet["y2"]
            ):
                # 优先选择框内关键点更多、置信度更高的 pose 结果
                score = sum(kp.get("conf", 0.0) for kp in kpts) / 17
                if score > best_score:
                    best_score = score
                    best_kpts = kpts
        udet = dict(udet)
        udet["keypoints"] = best_kpts
        merged.append(udet)
    return merged
```

在 `_run_inference` 的 ONNX 分支：

```python
if isinstance(model, OnnxYoloBackend):
    self._infer_cycle_count += 1
    detections = self._filter_inference_detections(model.predict(frame, conf=self._conf))
    # ... 保持现有逻辑
    return detections

# 以下处理混合后端：UHD + pose
coarse_results = self._predict_model(model, frame, self._conf)  # pose 模型
pose_dets = self._filter_inference_detections(...)

uhd_model = self._load_detect_model(self._custom_model_path or None)
if isinstance(uhd_model, UhdBackend):
    uhd_dets = uhd_model.predict(frame, conf=self._conf)
    detections = self._merge_uhd_with_pose(uhd_dets, pose_dets)
else:
    detections = pose_dets

# 后续处理：超分辨率、过滤、置信度等复用现有逻辑
```

#### 4.2.5 ROI 重检测与 UHD

现有 `_redetect_person_in_roi` 使用 `self._select_inference_model()` 进行 ROI 重检测。如果当前主模型是 UHD，但 ROI 很小（例如 64×64 级别），UHD 原始输入就是 64×64，直接 resize 到 64×64 会丢失信息。建议：

- 当主模型为 UHD 时，ROI 重检测仍使用 UHD（轻量），但 ROI 不应小于 64×64；
- 或者提供一个独立配置 `"redetect_backend"`，允许 ROI 重检测回退到 YOLO。

最小改动：让 `_redetect_person_in_roi` 内部也走 `_select_inference_model()`，UHD 模型天然支持 ROI 图片（会 resize 到 64×64）。

### 4.3 修改 `gui/settings_dialog.py`

- 模型发现函数 `_discover_model_paths` 保持 `*.pt` / `*.onnx`，UHD 模型已包含在 `*.onnx` 中；
- 下拉框标签从 "YOLO 模型" 改为 "AI 检测模型"，更通用；
- 文件浏览过滤器增加 `*.onnx` 已足够，无需额外改动。

```python
def _create_model_row(self) -> QHBoxLayout:
    # ...
    lbl = QLabel("AI 检测模型")
    # ...
    self._model_combo.addItem("自动 / 默认 YOLO 姿态模型", "")
    # _discover_model_paths 已包含 .onnx 文件
```

---

## 5. Comprehensive-Head-Classification 详细集成方案

### 5.1 新增 `ai/head_classifier.py`

```python
# ai/head_classifier.py
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np


class HeadClassificationBackend:
    """Comprehensive-Head-Classification ONNX 后端。

    输入：人头/人脸框 + 原始帧；输出：人脸属性概率。
    """

    def __init__(self, model_path: str | Path, provider_mode: str = "directml"):
        self.model_path = str(model_path)
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"CHC model not found: {self.model_path}")

        import onnxruntime as ort
        available = ort.get_available_providers()
        providers = self._select_providers(provider_mode, available)
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        self._session = ort.InferenceSession(
            self.model_path, sess_options=sess_options, providers=providers
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}
        self._has_fiqa = "head_image_352x352" in self._input_names
        logging.info(
            "HeadClassification backend loaded: path=%s providers=%s has_fiqa=%s",
            self.model_path,
            self._session.get_providers(),
            self._has_fiqa,
        )

    @staticmethod
    def _select_providers(provider_mode: str, available: list[str]) -> list[str]:
        mode = str(provider_mode or "auto").lower()
        if mode == "cpu":
            return ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @staticmethod
    def _safe_crop(frame: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        return frame[y1:y2, x1:x2]

    def _norm_hwc(self, roi: np.ndarray, size: tuple[int, int]) -> np.ndarray:
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
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        # 1) 人头框：取 person 框上半部分（头+肩）
        head_top = y1
        head_bottom = y1 + int((y2 - y1) * 0.45)
        head_roi = self._safe_crop(frame, (x1, head_top, x2, head_bottom))
        if head_roi is None:
            return None

        head_48 = self._norm_hwc(head_roi, (48, 48))[None]
        inputs = {"head_image_48x48": head_48}

        # 2) 眼睛区域：假设眼睛在人头框的 30%~60% 高度
        eye_top = head_top + int((head_bottom - head_top) * 0.30)
        eye_bottom = head_top + int((head_bottom - head_top) * 0.60)
        eye_left = x1 + int((x2 - x1) * 0.15)
        eye_right = x1 + int((x2 - x1) * 0.85)
        eye_roi = self._safe_crop(frame, (eye_left, eye_top, eye_right, eye_bottom))
        if eye_roi is not None:
            eye_h, eye_w = eye_roi.shape[:2]
            # 左半 + 右半
            left_eye = self._norm_hwc(eye_roi[:, : eye_w // 2], (40, 24))
            right_eye = self._norm_hwc(eye_roi[:, eye_w // 2 :], (40, 24))
            eyes = np.stack([left_eye, right_eye], axis=0)  # [2, 3, 24, 40]
            inputs["eye_images_24x40"] = eyes
        else:
            inputs["eye_images_24x40"] = np.zeros((2, 3, 24, 40), dtype=np.float32)

        # 3) 嘴巴区域：人头框下半部分 60%~90%
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

        # 4) FIQA（可选）
        if self._has_fiqa:
            head_352 = self._norm_hwc(head_roi, (352, 352))[None]
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            inputs["head_image_352x352"] = (head_352 - mean) / std

        try:
            outputs = self._session.run(None, inputs)
        except Exception as e:
            logging.warning("Head classification inference failed: %s", e)
            return None

        return self._decode_outputs(outputs)

    def _decode_outputs(self, outputs: list[np.ndarray]) -> dict:
        """输出顺序：bg_plain, masked, sunglass, hat, eye_open, mouth_open, quality_score"""
        return {
            "bg_plain": float(outputs[0].item()),
            "masked": float(outputs[1].item()),
            "sunglass": float(outputs[2].item()),
            "hat": float(outputs[3].item()),
            "eye_open": [float(outputs[4][0].item()), float(outputs[4][1].item())],
            "mouth_open": float(outputs[5].item()),
            "quality_score": float(outputs[6].item()) if self._has_fiqa else None,
        }
```

### 5.2 修改 `ai/inference.py`：属性融合

#### 5.2.1 初始化 CHC 后端

```python
def __init__(self, ...):
    # ... 现有初始化 ...
    self._head_classifier: HeadClassificationBackend | None = None
    self._head_classifier_enabled = False
    self._head_classifier_path = ""


def set_head_classifier(self, enabled: bool, model_path: str | None = None) -> None:
    self._head_classifier_enabled = bool(enabled)
    if model_path is not None:
        self._head_classifier_path = str(model_path)
    # 模型在首次需要时懒加载
```

#### 5.2.2 懒加载 CHC 模型

```python
def _load_head_classifier(self) -> HeadClassificationBackend | None:
    if not self._head_classifier_enabled:
        return None
    if self._head_classifier is not None:
        return self._head_classifier
    if not self._head_classifier_path:
        resolved = _resolve_artifact_path("chc_s_wo_fiqa.onnx")
        path = str(resolved) if resolved is not None else ""
    else:
        path = self._head_classifier_path
    if not path or not os.path.exists(path):
        logging.warning("Head classifier model not found, disabling head classification")
        self._head_classifier_enabled = False
        return None
    self._head_classifier = HeadClassificationBackend(path, provider_mode=self._backend_mode)
    return self._head_classifier
```

#### 5.2.3 在跟踪后附加属性

在 `run()` 的推理路径中，跟踪器 `tracker.update()` 返回平滑后的 detections，随后附加属性：

```python
# 在 tracker.update 之后、_attach_gestures 之前
detections = tracker.update(...)
detections = self._attach_head_attributes(detections, infer_frame)
detections = self._attach_gestures(detections)
```

```python
def _attach_head_attributes(
    self,
    detections: list[dict],
    frame: np.ndarray,
) -> list[dict]:
    classifier = self._load_head_classifier()
    if classifier is None:
        return detections

    out = []
    for det in detections:
        if det.get("label") != "person":
            out.append(det)
            continue
        attrs = classifier.predict(frame, det)
        if attrs is not None:
            det = dict(det)
            det["head_attrs"] = attrs
        out.append(det)
    return out
```

### 5.3 修改 `gui/camera_cell.py`：显示属性标签

在 `_build_detection_label` 中增加属性显示：

```python
@staticmethod
def _build_detection_label(det: dict, track_id: int) -> str:
    base = f"{det['label']} #{track_id}  {det.get('confidence', 0.0):.2f}"
    attrs = det.get("head_attrs")
    if attrs is not None:
        labels = []
        if attrs.get("hat", 0.0) > 0.5:
            labels.append("hat")
        if attrs.get("sunglass", 0.0) > 0.5:
            labels.append("sunglass")
        if attrs.get("masked", 0.0) > 0.5:
            labels.append("mask")
        eye_open = attrs.get("eye_open", [0.0, 0.0])
        if eye_open[0] > 0.5 and eye_open[1] > 0.5:
            labels.append("eye_open")
        elif eye_open[0] <= 0.5 and eye_open[1] <= 0.5:
            labels.append("eye_closed")
        if attrs.get("mouth_open", 0.0) > 0.5:
            labels.append("mouth_open")
        if labels:
            base = f"{base}  [{'|'.join(labels)}]"
    # ... 后续 gestures 处理保持不变
```

### 5.4 修改 `gui/settings_dialog.py`：CHC 配置

新增设置键和 UI：

```python
_KEY_AI_HEAD_CLASSIFIER_ENABLED = "ai/head_classifier_enabled"
_KEY_AI_HEAD_CLASSIFIER_PATH = "ai/head_classifier_path"

# 在 _build_ui 的 AI 检测设置段添加
layout.addLayout(self._create_head_classifier_row())
```

```python
def _create_head_classifier_row(self) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setSpacing(8)
    lbl = QLabel("头部属性模型")
    lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
    lbl.setFixedWidth(100)
    row.addWidget(lbl)

    self._head_classifier_combo = WheelFocusComboBox()
    self._head_classifier_combo.setFixedHeight(34)
    self._head_classifier_combo.addItem("关闭", "")
    for path in self._discover_head_classifier_paths():
        self._head_classifier_combo.addItem(path.name, str(path))
    row.addWidget(self._head_classifier_combo, stretch=1)
    return row

@staticmethod
def _discover_head_classifier_paths() -> list[Path]:
    models_dir = Path.cwd() / "models"
    if not models_dir.exists():
        return []
    return sorted(models_dir.glob("chc_*.onnx"), key=lambda p: p.name.lower())
```

---

## 6. 关键修改点汇总（按文件）

| 文件 | 修改内容 | 影响范围 |
|------|----------|----------|
| `ai/uhd_backend.py` | 新建 UHD ONNX 后端 | 新增文件 |
| `ai/head_classifier.py` | 新建 CHC 属性分类器 | 新增文件 |
| `ai/inference.py` | 1. 导入两个新后端；2. `_is_uhd_model` 识别；3. `_load_onnx_detect_model` 分支；4. `_select_inference_model` / `_run_inference` 支持 UHD+pose 融合；5. `_attach_head_attributes` 属性融合；6. setter 方法暴露配置 | 核心推理逻辑 |
| `ai/tracker.py` | **无需修改**，UHD 输出与 YOLO 检测 schema 一致 | 无 |
| `gui/camera_cell.py` | `_build_detection_label` 显示 head_attrs | 显示层 |
| `gui/settings_dialog.py` | 1. 模型下拉框文案通用化；2. 增加 CHC 模型选择；3. 增加配置键持久化 | 配置层 |
| `gui/main_window.py` | 1. 从 settings 读取 CHC 配置并调用 `set_head_classifier`；2. 模型切换时透传 | 控制层 |
| `requirements.txt` | 已包含 `onnxruntime-directml`，无需新增 | 无 |
| `models/` | 放置 `uhd_*.onnx` / `chc_*.onnx` | 运行时依赖 |

---

## 7. 性能与多路实时考虑

### 7.1 UHD 性能优势

- UHD N 在 Corei9 上约 **0.6 ms**，Z 约 **0.32 ms**，远低于 YOLOv8n（通常 10~30 ms 视后端）。
- 多路（4 路）场景下，UHD 可显著降低 GPU 占用和帧延迟。
- **DirectML 加速**：`UhdBackend` 与 `OnnxYoloBackend` 共用相同的 provider 选择逻辑，自动使用 DmlExecutionProvider。

### 7.2 CHC 性能成本

- 每路每人每帧一次 CHC 推理；4 路 × 多目标时可能成为瓶颈。
- 建议：
  - 仅对 `person` 检测执行；
  - 每 N 帧执行一次（例如与 `infer_stride` 对齐或单独设置 `head_class_stride`）；
  - 使用 `chc_s_wo_fiqa.onnx` 避免 352×352 分支；
  - 未来可加入人脸检测器（如 `yolomit` 系列）替代 person 框做更精确裁剪。

### 7.3 与 ROI 重检测结合

- UHD 作为重检测后端时，ROI 裁剪会 resize 到 64×64，适合快速确认 "预测框附近是否有人"。
- 由于 UHD 本身输入就是 64×64，**不建议对小于 64×64 的 ROI 使用 UHD 重检测**；可在 `_redetect_person_in_roi` 中增加 `min(crop_w, crop_h) >= 64` 限制。

### 7.4 多路线程适配

- 现有 `InferWorker` 是单一 QThread，所有摄像头共享同一个推理队列（通过 `slot_id` 区分）。
- UHD 轻量，不会显著增加单线程负担；CHC 若每目标都跑，可能拖慢主推理线程，建议引入异步或跳帧策略。

---

## 8. 潜在风险与降级处理

| 风险 | 降级处理 |
|------|----------|
| UHD 模型下载失败或文件缺失 | 回退到 `yolov8n.onnx` / `yolov8n.pt`，与现有逻辑一致 |
| UHD 检测精度不足（小目标/密集场景） | 设置里切换回 YOLO；或在 pose 启用时保留 YOLO pose 关键点 |
| CHC 裁剪区域错误（无清晰人脸） | 跳过属性推理，不阻塞检测；输出空 `head_attrs` |
| CHC 模型推理失败 | `try/except` 捕获，记录 warning，跳过该帧属性 |
| 4 路多目标导致 CHC 延迟过高 | 增加 `head_class_stride`（每 N 帧一次），或仅对最大目标执行 |
| DirectML 初始化失败 | 自动回退 CPUExecutionProvider，与现有 ONNX 后端一致 |
| 健康监控触发 | 与现有机制一致：回退到 `ukf`、关闭去噪、关闭重检测 |

---

## 9. 最小改动实现路径（MVP）

### 阶段 1：UHD 作为主检测后端（1~2 天）

1. 下载 `uhd_n_64x64_static.onnx` 到 `models/`。
2. 新建 `ai/uhd_backend.py`（约 100 行）。
3. 修改 `ai/inference.py`：
   - 增加 `_is_uhd_model` 和 `UhdBackend` 导入；
   - 在 `_load_onnx_detect_model` 中分支；
   - 当 pose 启用时，实现 UHD bbox + pose keypoints 融合；
   - 当 pose 不启用时，直接输出 UHD detections。
4. 修改 `gui/settings_dialog.py`：模型下拉框文案改为 "AI 检测模型"，已能自动发现 UHD 的 `.onnx` 文件。
5. 本地测试：切换模型到 UHD，验证 person 检测、跟踪、显示正常。

### 阶段 2：CHC 属性融合（2~3 天）

1. 下载 `chc_s_wo_fiqa.onnx` 到 `models/`。
2. 新建 `ai/head_classifier.py`（约 180 行）。
3. 修改 `ai/inference.py`：
   - 增加 `_head_classifier` 和 `set_head_classifier`；
   - 在 tracker.update 后调用 `_attach_head_attributes`；
   - 增加跳帧控制（可选）。
4. 修改 `gui/camera_cell.py`：在 `_build_detection_label` 中展示属性。
5. 修改 `gui/settings_dialog.py`：增加 CHC 下拉框和持久化键。
6. 修改 `gui/main_window.py`：读取配置并调用 `set_head_classifier`。
7. 本地测试：启用/关闭 CHC、多目标、多路。

---

## 10. 分步实施 Checklist

### UHD 集成

- [x] 下载并校验 `uhd_n_64x64_static.onnx`（来自 `ultratinyod_res_anc8_w64_64x64_opencv_inter_nearest_static.onnx`，约 7.6 MB）。
- [x] 创建 `ai/uhd_backend.py`，实现 `UhdBackend.predict(frame, conf=...)`。
- [x] 在 `ai/inference.py` 中引入 `UhdBackend` 和 `_is_uhd_model`。
- [x] 修改 `_load_onnx_detect_model`：根据文件名/路径选择 `UhdBackend` 或 `OnnxYoloBackend`。
- [x] 修改 `_initialize_backend`：cpu 模式下遇到 UHD 自动切到 ONNX CPU provider。
- [x] 修改 `_select_inference_model` 与 `_run_inference`：当 pose 启用且 detect 模型为 UHD 时，融合 pose keypoints。
- [x] 验证 ROI 重检测在 UHD 模式下不崩溃（ROI <48px 跳过）。
- [x] 修改 `gui/settings_dialog.py` 模型下拉框文案为 "AI 检测模型"。
- [x] 端到端测试：UHD 检测/属性附加/模型切换均通过；四路需结合实际摄像头验证。
- [x] 更新 `docs/` 和 `memory/` 记录 UHD 集成要点。

### CHC 集成

- [x] 下载并校验 `chc_s_wo_fiqa.onnx`（约 2.8 MB）。
- [x] 创建 `ai/head_classifier.py`，实现 `HeadClassificationBackend.predict(frame, det)`。
- [x] 在 `InferWorker.__init__` 中增加 CHC 相关状态。
- [x] 实现 `set_head_classifier(enabled, model_path, stride)` 并暴露给 `MainWindow`。
- [x] 在 `run()` 的推理路径中，跟踪后附加 `_attach_head_attributes`。
- [x] 修改 `gui/camera_cell.py` 的 `_build_detection_label` 显示属性标签。
- [x] 修改 `gui/settings_dialog.py` 增加 CHC 下拉框、跳帧 SpinBox、浏览按钮和配置键。
- [x] 修改 `gui/main_window.py` 读取 CHC 配置并下发。
- [x] 性能测试：CPU 单目标 CHC 约 16–17ms，通过 `head_class_stride` 控制多目标延迟。
- [x] 更新文档和 memory。

---

## 11. 关键代码片段示例（Diff 风格）

### 11.1 `ai/inference.py`：模型加载分支

```diff
--- a/ai/inference.py
+++ b/ai/inference.py
@@ -25,6 +25,7 @@ from PyQt6.QtCore import QThread, pyqtSignal
 from ai.gesture_recognizer import GestureRecognizer
 from ai.hand_gesture_recognizer import HandGestureRecognizer, normalize_gesture_mode
 from ai.onnx_yolo_backend import OnnxYoloBackend
+from ai.uhd_backend import UhdBackend, _is_uhd_model
 from ai.rectangle_detector import RectangleDetector, RectangleTracker, normalize_rectangle_sensitivity
 from ai.tracker import DetectionTracker, _box_iou
 
@@ -683,8 +684,12 @@ class InferWorker(QThread):
             resolved = _resolve_artifact_path("yolov8n.onnx")
             model_path = str(resolved) if resolved is not None else ""
         elif not os.path.exists(model_path):
             raise FileNotFoundError(f"ONNX model not found: {model_path}")
         if self._detect_model is not None and self._detect_model_path == model_path:
             return self._detect_model
-        model = OnnxYoloBackend(model_path, task="detect", provider_mode="directml", imgsz=self._model_imgsz)
+        if _is_uhd_model(model_path):
+            model = UhdBackend(model_path, provider_mode=self._backend_mode)
+        else:
+            model = OnnxYoloBackend(model_path, task="detect", provider_mode="directml", imgsz=self._model_imgsz)
         self._detect_model = model
         self._detect_model_path = model_path
         logging.info("YOLOv8 ONNX Detect model loaded path=%s providers=%s", model_path, model.providers)
```

### 11.2 `ai/inference.py`：UHD + pose 融合

```diff
--- a/ai/inference.py
+++ b/ai/inference.py
@@ -714,6 +715,32 @@ class InferWorker(QThread):
             return model.predict(frame, conf=conf)
         return model(frame, verbose=False, conf=conf, imgsz=self._model_imgsz, device=self._device)
 
+    def _merge_uhd_with_pose(self, uhd_dets, pose_dets):
+        if not pose_dets:
+            return uhd_dets
+        merged = []
+        for udet in uhd_dets:
+            best_kpts = []
+            best_score = -1.0
+            for pdet in pose_dets:
+                kpts = pdet.get("keypoints") or []
+                if len(kpts) != 17:
+                    continue
+                nose = kpts[0]
+                if udet["x1"] <= nose["x"] <= udet["x2"] and udet["y1"] <= nose["y"] <= udet["y2"]:
+                    score = sum(kp.get("conf", 0.0) for kp in kpts) / 17
+                    if score > best_score:
+                        best_score = score
+                        best_kpts = kpts
+            udet = dict(udet)
+            udet["keypoints"] = best_kpts
+            merged.append(udet)
+        return merged
+
     def _run_inference(self, frame: np.ndarray) -> list[dict]:
         try:
             startup_guard_active = self._is_startup_guard_active()
@@ -726,6 +753,18 @@ class InferWorker(QThread):
             if isinstance(model, OnnxYoloBackend):
                 self._infer_cycle_count += 1
                 detections = self._filter_inference_detections(model.predict(frame, conf=self._conf))
+                # ... 保持原有 ONNX YOLO 路径不变
                 return detections
+            if self._runtime_backend == "onnx" and isinstance(self._detect_model, UhdBackend):
+                # pose 模型用于关键点，UHD 用于 person 检测
+                pose_dets = self._filter_inference_detections(model.predict(frame, conf=self._conf))
+                uhd_dets = self._detect_model.predict(frame, conf=self._conf)
+                detections = self._merge_uhd_with_pose(uhd_dets, pose_dets)
+            else:
+                coarse_results = self._predict_model(model, frame, self._conf)
+                detections = []
+                # ... 原有 PyTorch YOLO 解析逻辑
             
             # ... 后续处理保持不变
```

### 11.3 `gui/camera_cell.py`：属性标签

```diff
--- a/gui/camera_cell.py
+++ b/gui/camera_cell.py
@@ -924,6 +924,24 @@ class CameraCell(QWidget):
     @staticmethod
     def _build_detection_label(det: dict, track_id: int) -> str:
         gestures = det.get("gestures") or []
+        attrs = det.get("head_attrs")
+        attr_labels = []
+        if attrs is not None:
+            if attrs.get("hat", 0.0) > 0.5:
+                attr_labels.append("hat")
+            if attrs.get("sunglass", 0.0) > 0.5:
+                attr_labels.append("sunglass")
+            if attrs.get("masked", 0.0) > 0.5:
+                attr_labels.append("mask")
+            eye_open = attrs.get("eye_open", [0.0, 0.0])
+            if eye_open[0] > 0.5 and eye_open[1] > 0.5:
+                attr_labels.append("eye_open")
+            elif eye_open[0] <= 0.5 and eye_open[1] <= 0.5:
+                attr_labels.append("eye_closed")
+            if attrs.get("mouth_open", 0.0) > 0.5:
+                attr_labels.append("mouth_open")
         if det.get("label") == "person" and gestures:
             names = [str(gesture.get("name", "")) for gesture in gestures[:2] if gesture.get("name")]
             if names:
                 base = f"{' / '.join(names)}  {det.get('confidence', 0.0):.2f}"
             else:
                 base = f"{det['label']} #{track_id}  {det.get('confidence', 0.0):.2f}"
+        elif attr_labels:
+            base = f"{det['label']} #{track_id}  [{'|'.join(attr_labels)}]  {det.get('confidence', 0.0):.2f}"
         elif det.get("label") == "hand_gesture":
             name = str(det.get("gesture_name") or "手部")
             base = f"{name}  {det.get('confidence', 0.0):.2f}"
         else:
             base = f"{det['label']} #{track_id}  {det.get('confidence', 0.0):.2f}"
```

---

## 12. 总结

- **UHD** 是 VisionBata 理想的**轻量人体检测后端**替代方案：单一输入、输出与现有检测 schema 一致、可复用 DirectML/ONNX 后端、tracker 无需修改。
- **Comprehensive-Head-Classification** 是**下游属性增强模块**：需要先有 person 框，通过近似裁剪人头/眼/嘴区域，输出 hat/mask/sunglass/eye/mouth 属性，集成到检测标签中显示。
- **推荐实施顺序**：先 UHD（收益大、改动小），再 CHC（改动多、依赖前置检测）。
- **核心风险可控**：通过模型文件识别、fallback 到 YOLO、try/except 隔离 CHC 推理、健康监控自动回退，确保系统稳定性。

---

*文档生成时间：2026-07-09*
*依据仓库：PINTO0309/UHD、PINTO0309/Comprehensive-Head-Classification、VisionBata 当前代码主分支*
