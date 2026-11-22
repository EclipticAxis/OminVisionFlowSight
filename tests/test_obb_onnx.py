from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ai.model_task import ModelTask
from ai.onnx_yolo_backend import OnnxYoloBackend, _xywhr2xyxyxyxy

REPO_ROOT = Path(__file__).resolve().parents[1]
OBB_ONNX = REPO_ROOT / "models" / "yolo26n-obb.onnx"
OBB_PT = REPO_ROOT / "models" / "yolo26n-obb.pt"
TEST_IMAGE = REPO_ROOT / "models" / "test_obb.jpg"

_DOTA_NAMES = {
    0: "plane", 1: "ship", 2: "storage tank", 3: "baseball diamond", 4: "tennis court",
    5: "basketball court", 6: "ground track field", 7: "harbor", 8: "bridge",
    9: "large vehicle", 10: "small vehicle", 11: "helicopter", 12: "roundabout",
    13: "soccer ball field", 14: "swimming pool",
}


def _decode_backend(class_names: dict[int, str] | None = None) -> OnnxYoloBackend:
    """绕过 __init__ 构造仅用于解码测试的后端（不创建 onnxruntime session）。"""
    backend = OnnxYoloBackend.__new__(OnnxYoloBackend)
    backend.task = ModelTask.OBB
    backend._class_names = dict(class_names or {})
    return backend


def _meta(frame_w: int = 640, frame_h: int = 640, scale: float = 1.0, pad_x: int = 0, pad_y: int = 0) -> dict:
    return {"frame_w": frame_w, "frame_h": frame_h, "scale": scale, "pad_x": pad_x, "pad_y": pad_y}


def _row(cx, cy, w, h, angle, class_id: int, conf: float, num_classes: int = 15) -> np.ndarray:
    """合成一行 OBB 输出，布局与 ultralytics 导出一致：cx, cy, w, h | 每类分数 | angle(末通道, 弧度)。"""
    row = np.zeros(4 + num_classes + 1, dtype=np.float32)
    row[:4] = (cx, cy, w, h)
    row[4 + class_id] = conf
    row[-1] = angle
    return row


# ---------------------------------------------------------------------------
# 合成张量单测（不依赖模型文件）
# ---------------------------------------------------------------------------

def test_xywhr2xyxyxyxy_matches_ultralytics_math():
    # 与 ultralytics ops.py::xywhr2xyxyxyxy 逐行对齐的解析式验证（angle=0.3 rad）
    x = np.array([[100.0, 200.0, 80.0, 40.0, 0.3]], dtype=np.float64)
    corners = _xywhr2xyxyxyxy(x)[0]
    cx, cy, w, h, angle = x[0]
    cos_v, sin_v = math.cos(angle), math.sin(angle)
    vec1 = np.array([w / 2 * cos_v, w / 2 * sin_v])
    vec2 = np.array([-h / 2 * sin_v, h / 2 * cos_v])
    ctr = np.array([cx, cy])
    expected = np.stack([ctr + vec1 + vec2, ctr + vec1 - vec2, ctr - vec1 - vec2, ctr - vec1 + vec2])
    np.testing.assert_allclose(corners, expected, atol=1e-6)
    # 顶点顺序性质：pt1/pt3 对角，pt2/pt4 对角
    diag1 = np.linalg.norm(corners[0] - corners[2])
    diag2 = np.linalg.norm(corners[1] - corners[3])
    assert diag1 == pytest.approx(math.hypot(w, h), rel=1e-6)
    assert diag2 == pytest.approx(math.hypot(w, h), rel=1e-6)


def test_decode_obb_channel_parse_and_conf_filter():
    pred = np.stack([
        _row(320, 320, 100, 50, 0.0, class_id=0, conf=0.90),
        _row(100, 100, 20, 20, 0.5, class_id=2, conf=0.30),   # 低于 conf 阈值
        _row(500, 500, 20, 20, 0.0, class_id=1, conf=0.05),   # 低于 conf 阈值
    ])
    dets = _decode_backend(_DOTA_NAMES)._decode_obb(pred, _meta(), conf=0.5)
    assert len(dets) == 1
    det = dets[0]
    assert det.class_id == 0
    assert det.label == "plane"
    assert det.confidence == pytest.approx(0.90, abs=1e-5)
    assert det.polygon is not None and len(det.polygon) == 4
    assert det.has_polygon()


def test_decode_obb_angle_zero_degenerates_to_axis_aligned_rect():
    pred = np.stack([_row(320, 320, 100, 50, 0.0, class_id=0, conf=0.90)])
    det = _decode_backend()._decode_obb(pred, _meta(), conf=0.5)[0]
    # angle=0 → 轴对齐矩形：外接框 = cx±w/2, cy±h/2（归一化坐标 /640）
    assert det.bbox == pytest.approx((270 / 640, 295 / 640, 370 / 640, 345 / 640), abs=1e-5)
    xs = sorted(p[0] for p in det.polygon)
    ys = sorted(p[1] for p in det.polygon)
    assert xs == pytest.approx([270 / 640, 270 / 640, 370 / 640, 370 / 640], abs=1e-5)
    assert ys == pytest.approx([295 / 640, 295 / 640, 345 / 640, 345 / 640], abs=1e-5)


def test_decode_obb_angle_half_pi_swaps_width_height():
    pred = np.stack([_row(320, 320, 100, 50, math.pi / 2, class_id=0, conf=0.90)])
    det = _decode_backend()._decode_obb(pred, _meta(), conf=0.5)[0]
    x1, y1, x2, y2 = det.bbox
    width, height = (x2 - x1) * 640, (y2 - y1) * 640
    assert width == pytest.approx(50, abs=1e-4)   # w/h 互换
    assert height == pytest.approx(100, abs=1e-4)
    # 外接框中心仍在 (320, 320)
    assert (x1 + x2) / 2 == pytest.approx(320 / 640, abs=1e-5)
    assert (y1 + y2) / 2 == pytest.approx(320 / 640, abs=1e-5)


def test_decode_obb_point_remapping_and_clamp():
    # scale=0.5, pad=(70,170)，frame 1000x500：norm = (pt - pad) / 0.5 / size
    pred = np.stack([_row(200, 300, 400, 200, 0.0, class_id=0, conf=0.90)])
    meta = _meta(frame_w=1000, frame_h=500, scale=0.5, pad_x=70, pad_y=170)
    det = _decode_backend()._decode_obb(pred, meta, conf=0.5)[0]
    # letterbox 顶点 x∈[0,400], y∈[200,400]；反映射后 x∈[-140,660]→clamp 0..0.66，y∈[60,460]→0.12..0.92
    xs = sorted(p[0] for p in det.polygon)
    ys = sorted(p[1] for p in det.polygon)
    assert xs[0] == pytest.approx(0.0, abs=1e-6)
    assert xs[-1] == pytest.approx(660 / 1000, abs=1e-5)
    assert ys[0] == pytest.approx(60 / 500, abs=1e-5)
    assert ys[-1] == pytest.approx(460 / 500, abs=1e-5)
    for x, y in det.polygon:
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def test_decode_obb_empty_results():
    backend = _decode_backend()
    # 全部低于阈值
    pred = np.stack([_row(320, 320, 100, 50, 0.0, class_id=0, conf=0.10)])
    assert backend._decode_obb(pred, _meta(), conf=0.5) == []
    # 通道数不足（<6）
    assert backend._decode_obb(np.zeros((4, 4), dtype=np.float32), _meta(), conf=0.5) == []
    # 空 pred
    assert backend._decode_obb(np.zeros((0, 20), dtype=np.float32), _meta(), conf=0.5) == []


def test_normalize_output_transposes_obb_output():
    output = np.random.rand(1, 20, 8400).astype(np.float32)
    pred = OnnxYoloBackend._normalize_output(output)
    assert pred.shape == (8400, 20)


def test_label_falls_back_to_str_class_id_without_metadata():
    pred = np.stack([_row(320, 320, 100, 50, 0.0, class_id=7, conf=0.90)])
    det = _decode_backend()._decode_obb(pred, _meta(), conf=0.5)[0]
    assert det.label == "7"


def test_label_uses_onnx_metadata_names():
    pred = np.stack([_row(320, 320, 100, 50, 0.0, class_id=9, conf=0.90)])
    det = _decode_backend(_DOTA_NAMES)._decode_obb(pred, _meta(), conf=0.5)[0]
    assert det.label == "large vehicle"


# ---------------------------------------------------------------------------
# 同帧对拍（真实模型，缺失则 skip）
# ---------------------------------------------------------------------------

_real_models = pytest.mark.skipif(
    not (OBB_ONNX.exists() and OBB_PT.exists() and TEST_IMAGE.exists()),
    reason="models/yolo26n-obb.onnx / .pt 或 models/test_obb.jpg 缺失",
)


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a.bbox
    bx1, by1, bx2, by2 = b.bbox
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 1e-8 else 0.0


@_real_models
def test_obb_onnx_vs_pt_same_frame_parity():
    import cv2

    from ai.yolo26_backend import Yolo26Backend

    frame = cv2.imread(str(TEST_IMAGE))
    assert frame is not None

    pt_backend = Yolo26Backend(str(OBB_PT), task=ModelTask.OBB)
    # .pt 侧默认 imgsz=1024（rect），而 ONNX 导出为 640x640；统一到 640 才能同帧对拍
    pt_backend._model.overrides["imgsz"] = 640
    pt_dets = pt_backend.predict(frame, conf=0.25)
    if not pt_dets:
        pytest.skip(".pt 侧未检出任何目标，对拍无意义")

    onnx_backend = OnnxYoloBackend(str(OBB_ONNX), task=ModelTask.OBB, provider_mode="cpu")
    onnx_dets = onnx_backend.predict(frame, conf=0.25)

    # 每个 pt 检测取外接框 IoU 最大的 onnx 检测（共享匹配：只验证存在近重复，
    # 不做一对一分配——NMS 近似下两侧集合本就有差异）。
    pairs = []
    for pt_det in pt_dets:
        best_det, best_iou = None, 0.0
        for onnx_det in onnx_dets:
            v = _bbox_iou(pt_det, onnx_det)
            if v > best_iou:
                best_det, best_iou = onnx_det, v
        pairs.append((pt_det, best_det, best_iou))

    matched = [p for p in pairs if p[2] >= 0.5]
    valid = [p for p in matched if p[2] >= 0.8]

    ious = [p[2] for p in valid]
    conf_diffs = [abs(p[0].confidence - p[1].confidence) for p in valid]
    poly_devs = [
        max(abs(px - ox), abs(py - oy))
        for p in valid
        for (px, py), (ox, oy) in zip(p[0].polygon, p[1].polygon)
    ]

    print("\n[OBB parity] pt=%d onnx=%d matched(IoU>=0.5)=%d valid(IoU>=0.8)=%d" %
          (len(pt_dets), len(onnx_dets), len(matched), len(valid)))
    print("[OBB parity] valid bbox IoU: min=%.4f mean=%.4f" % (min(ious), sum(ious) / len(ious)))
    print("[OBB parity] valid conf diff: max=%.4f" % max(conf_diffs))
    print("[OBB parity] valid polygon dev: max=%.5f" % max(poly_devs))

    assert len(valid) >= max(1, math.ceil(0.5 * len(pt_dets)))
    for pt_det, onnx_det, iou in valid:
        assert iou >= 0.8, f"bbox IoU 不足: {iou:.4f}"
        assert abs(pt_det.confidence - onnx_det.confidence) < 0.1
        assert pt_det.class_id == onnx_det.class_id
        for (px, py), (ox, oy) in zip(pt_det.polygon, onnx_det.polygon):
            assert max(abs(px - ox), abs(py - oy)) < 0.02, "polygon 顶点偏差过大"


# ---------------------------------------------------------------------------
# 选择链：任务推断 + _load_onnx_detect_model 以 OBB 构造
# ---------------------------------------------------------------------------

def test_infer_model_task_from_path_detects_obb():
    from ai.inference import _infer_model_task_from_path

    assert _infer_model_task_from_path(str(OBB_ONNX)) is ModelTask.OBB
    assert _infer_model_task_from_path("models/yolov8n.onnx") is ModelTask.DETECT


@pytest.mark.skipif(not OBB_ONNX.exists(), reason="models/yolo26n-obb.onnx 缺失")
def test_load_onnx_detect_model_constructs_obb_task():
    from ai.inference import InferWorker

    worker = InferWorker.__new__(InferWorker)
    worker._detect_model = None
    worker._detect_model_path = None
    worker._backend_mode = "cpu"
    worker._model_imgsz = 640

    model = worker._load_onnx_detect_model(str(OBB_ONNX))
    assert model.task is ModelTask.OBB
    assert model.providers == ["CPUExecutionProvider"]
