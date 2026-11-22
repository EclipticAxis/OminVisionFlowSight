# OnnxYoloBackend OBB 支持 设计文档

> **版本**：1.0 — 2026-11-22
> **范围**：`ai/onnx_yolo_backend.py`（OBB 解码 + BGR→RGB 预处理修正）、`ai/inference.py`（`_load_onnx_detect_model` 任务推断接通）、`tests/test_obb_onnx.py`（新增 12 项）、`models/yolo26n-obb.onnx`（随仓库分发，9.5MB，opset 18，simplify=False）
> **状态**：已合入 main（PR #10 → `95f9ac9`）；`tests/test_obb_onnx.py` 12 项通过；全套件 1378 项全绿 = 1366 既有 + 12 新增

---

## 1. 目标与背景

OnnxYoloBackend 自引入起在 `__init__` 中对 `ModelTask.OBB` 抛 `NotImplementedError`
（"reserved for Phase 2"），而该预留有两处实际危害：

1. **预留异常不可达**：`ai/inference.py::_load_onnx_detect_model` 将任务硬编码为
   `ModelTask.DETECT`，手动选择 `models/yolo26n-obb.onnx` 时不会以 OBB 构造，
   `NotImplementedError` 永远不会触发；
2. **静默误解析**：OBB 输出（20 通道）被按 DETECT 布局（前 4 通道 box + 其余为类分数）
   解析，angle 通道（末通道）被当作第 15 类分数，产出语义完全错误的 Detection 列表，
   无任何报错。

本次改动完整接通 ONNX OBB：解码、反映射、标签、NMS、任务推断与选择链。

## 2. 契约

### 2.1 通道布局（实测校准，勿凭经验假设）

`models/yolo26n-obb.onnx` 输出 `(1, 20, 8400)`，`_normalize_output` 转置为 `(8400, 20)` 后：

| 索引 | 含义 | 说明 |
|------|------|------|
| 0–3 | `cx, cy, w, h` | letterbox 640x640 坐标系 |
| 4–18 | 15 类分数（DOTA，已 sigmoid） | `class_ids = argmax`，`confidences = max` |
| **19** | **angle（弧度）** | **末通道**，直接可用，无需 sigmoid 解码 |

> 关键事实：angle 在**末通道**而非索引 4。依据：
> - `ultralytics/nn/modules/head.py::OBB._inference`：`torch.cat([preds, x["angle"]], dim=1)`，angle 拼接在 box+cls 之后；`OBB26.forward_head` 注释明确 "OBB theta logits (raw output without sigmoid transformation)"——OBB26 头直接回归弧度；
> - `ultralytics/utils/nms.py::non_max_suppression`：`extra = C - nc - 4`、`mi = 4 + nc`，angle 作为 extra 通道位于末位，与之一致；
> - 实测：同一 anchor 末通道值与 `.pt` 侧 `r.obb` 的 angle 字段逐位一致（0.4880 vs 0.4882）。

### 2.2 xywhr → 四顶点（逐行对齐 ultralytics）

`_xywhr2xyxyxyxy`（`ai/onnx_yolo_backend.py` 模块级函数）逐行复刻
`ultralytics/utils/ops.py::xywhr2xyxyxyxy`（L404–L431）：

```
vec1 = (w/2 * cos(angle), w/2 * sin(angle))
vec2 = (-h/2 * sin(angle), h/2 * cos(angle))
pt1 = ctr + vec1 + vec2 ; pt2 = ctr + vec1 - vec2
pt3 = ctr - vec1 - vec2 ; pt4 = ctr - vec1 + vec2
```

angle 弧度直接代入（无 sigmoid、无 `(s-0.25)·π`——那是旧版 `OBB` 头的变换，OBB26 不适用）。

### 2.3 反映射与 Detection 构造

- 逐点反映射 `_points_to_norm`：`x_norm = (x - pad_x) / scale / frame_w`，
  `y_norm = (y - pad_y) / scale / frame_h`，clamp `[0,1]`；
- `polygon`：4 个归一化角点 `[(x,y),...]`，与 `Yolo26Backend._parse_obb`
  （`ai/yolo26_backend.py` L82–L100）逐字段一致（同顶点顺序、同 clamp）；
- `bbox`：polygon 外接矩形（归一化）；
- `confidence`：最大类分数；`class_id`：argmax 类别；
- `label`：优先 ONNX metadata `custom_metadata_map["names"]`
  （ultralytics 导出时写入，本模型为 DOTA 15 类），解析失败回退 `str(class_id)`；
  本仓库模型 metadata 完整，实际与 `.pt` 侧 `model.names` 语义一致。

### 2.4 NMS 近似

复用现有 `_nms`（**轴对齐外接矩形 IoU**，阈值 0.45，class-aware）作为 rotated NMS 的近似。
边界：密集小目标、高度倾斜框的保留/抑制集合可能与 ultralytics 的 probIoU NMS 不同
（同帧对拍实测：pt 178 vs onnx 165 检出，165 个近重复全部匹配）。
如需精确 rotated NMS 需引入 probIoU 实现（当前禁止新增依赖，未做）。

### 2.5 预处理修正（BGR→RGB）

`_preprocess` 增加 `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`。
依据：OpenCV 采集帧为 BGR，而 ultralytics 训练/推理与导出模型均以 RGB 归一化输入；
此前 DETECT/POSE ONNX 路径同样把 BGR 直喂模型（潜在静默偏差，本次一并修正，
使 ONNX 各任务与 `Yolo26Backend` 输入语义一致）。

### 2.6 任务推断与选择链

`ai/inference.py::_load_onnx_detect_model`：`task = ModelTask.DETECT` 硬编码改为
`task = _infer_model_task_from_path(model_path)`（文件名含 `obb`→OBB、`pose`→POSE、默认 DETECT）。
手动选择 `models/yolo26n-obb.onnx` 时即以 OBB 任务构造并走 `_decode_obb` 解码。

## 3. 验证

| 项 | 结果 |
|----|------|
| `pytest tests/test_obb_onnx.py -q` | 12 项通过（9 项合成 + 对拍 + 选择链 2 项） |
| 同帧对拍（`models/test_obb.jpg`，.pt@640 vs ONNX CPU） | pt=178 / onnx=165；matched(IoU≥0.5)=174；valid(IoU≥0.8)=165 ≥ 89；valid IoU min 0.848 / mean 0.989；conf diff max 0.063；polygon dev max 0.0138 |
| DirectML 冒烟 | `OnnxYoloBackend(..., task=OBB)` auto=Dml+CPU 加载成功，predict 返回 165 检出 |
| `pytest tests -q` 全套件 | 1378 项全绿（1366 既有 + 12 新增） |

对拍注意：`.pt` 检查点默认 imgsz=1024（rect 1024x576），ONNX 导出为 640x640，
对拍时须将 `.pt` 侧 `overrides["imgsz"]` 设为 640（测试内已处理）。

## 4. 维护指南

1. **更换其他 OBB 模型**：必须重新核对通道数与类别数
   （`C = 4 + nc + 1`，angle 恒为末通道）；若类别数不是 15，metadata `names` 解析
   自动适配，但合成测试的 `num_classes` 与 `_DOTA_NAMES` 需同步。
2. **通道布局来源**：以 ultralytics 导出头的 `cat` 顺序为准
   （`head.py::OBB._inference`），不要凭"angle 紧跟 box"的经验假设——
   本模型该假设会导致 angle 被当作类分数（静默错误）。
3. **NMS 近似边界**：密集/倾斜场景下两侧检出集合允许差异；
   下游若依赖与 ultralytics 完全一致的 NMS 结果，需实现 rotated NMS。
4. **模型导出命令**：`yolo export model=yolo26n-obb.pt format=onnx imgsz=640 simplify=False`；
   `simplify=False` 为当前实测可用配置，换 simplify=True 需重新对拍。
5. **不做事项**：不引入 shapely/新依赖；不改 `Yolo26Backend`；不动 Rust 跟踪内核
   （跟踪行为不变，`docs/rust-tracker-core.md` 无需更新）。
