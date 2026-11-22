# 精确 Polygon IoU 双側自实现 设计文档

> **版本**：1.0 — 2026-11-22
> **范围**：`ai/detection.py`（`polygon_iou` 及辅助函数）、`rust/omintrack/src/geometry.rs` + `lib.rs`（新增 `_polygon_iou` 测试辅助）、新增 `tests/test_polygon_iou_precise.py`
> **状态**：⏳ 待合并
> **关联**：[rust-tracker-core](rust-tracker-core.md) §3.2 匹配代价契约、§3.3 修改契约

---

## 1. 目标与背景

### 1.1 问题

旧版 `ai/detection.py::polygon_iou` 优先调用 shapely，shapely 不可用时**静默降级**为
`_fallback_bbox_iou`（顶点集轴对齐外接矩形 IoU）。这带来两个缺陷：

1. **静默精度降级**：无 shapely 的环境中 OBB 匹配代价系统性偏乐观（外接矩形 IoU ≥
   精确 IoU），且无任何日志；
2. **双后端环境分歧**（[rust-tracker-core](rust-tracker-core.md) 原 §3.4 警告）：
   Python 侧按"有无 shapely"走两条不同语义路径，而 Rust `geometry.rs` 固定按外接
   矩形语义实现——本机原无 shapely 时双側一致近似，一旦 Python 环境装上 shapely
   （如本机 venv 现已安装 shapely 2.2.0），两后端 OBB 匹配立即出现语义分歧，
   等价性测试的"锁定"被环境悄悄打破。

### 1.2 方案

双側自实现**精确凸多边形 IoU**，彻底移除 shapely 依赖（生产路径）：

- 交集：**Sutherland–Hodgman 凸多边形裁剪**（被裁剪多边形逐边裁剪，O(n·m)）；
- 面积：**shoelace 公式**（有向面积 ×2，取绝对值 ×0.5）；
- `union = area_a + area_b - inter`；`inter / union if union > 1e-8 else 0.0`
  （与现有 `_fallback_bbox_iou` 及 Rust `box_iou` 的零并约定完全一致）；
- Python 与 Rust **逐行同算法**：同样的环绕方向归一（shoelace 符号检测，
  CW 反转成 CCW）、同样的裁剪边序、同样的浮点运算次序，保证双側单点值
  差异 < 1e-12（实测逐位一致）。

shapely 2.2.0 仅作为**测试对拍参考值**（`tests/test_polygon_iou_precise.py`
skip-if-missing），不进入生产路径，不写入 requirements。

### 1.3 凸性假定与 OBB 依据

Sutherland–Hodgman 裁剪要求**两多边形均为凸**。跟踪器 polygon 场景全部为
OBB（有向边界框）四边形——矩形经旋转/透视变换后恒为凸四边形，算法适用。
非凸或自交多边形**不在契约内**（见 §4 维护指南）。

## 2. 契约表

| 项目 | 契约 |
| --- | --- |
| 交集算法 | Sutherland–Hodgman 凸裁剪：subject 被 clip 的每条有向边（CCW，内侧为左半平面 `cross(edge, p-a) ≥ 0`）逐边裁剪，交点用线段-直线参数式求解 |
| 面积算法 | shoelace 有向面积 ×2 求和取绝对值 ×0.5；<3 顶点为 0.0 |
| 环绕方向 | 裁剪前两多边形统一归一为 CCW（`_signed_area2 < 0` 则反转顶点序）；CW/CCW 输入结果等价 |
| union 约定 | `union = area_a + area_b - inter`；`union > 1e-8` 返回 `inter/union`，否则 0.0（零面积退化、完全重合退化均落此分支语义） |
| 无交集 | 裁剪输出为空 → `inter = 0` → IoU = 0.0 |
| 包含关系 | 小框全在大框内 → 裁剪输出 = 小框 → IoU = area_small / area_big |
| 平行边数值保护 | 线段与裁剪线平行（`denom < 1e-12`）时返回线段终点，不产生 NaN |
| 异常兜底 | Python `polygon_iou` 入口校验（空/<3 顶点 → 0.0）保持；`_precise_polygon_iou` 抛异常时兜底 `_fallback_bbox_iou`（外接 bbox IoU）。Rust 主路径无 try/except 概念，不实现兜底 |
| shapely 依赖 | **生产路径零依赖**；`_shapely_iou` 已删除；shapely 仅测试参考（skip-if-missing） |
| 双側一致性 | Python `_precise_polygon_iou` ↔ Rust `geometry::precise_polygon_iou` 逐行同算法；`omintrack._polygon_iou(a, b)` 测试辅助供 Python 直接对拍，容差 < 1e-9（实测 0 ~ <1e-15） |
| golden 影响范围 | 仅 OBB polygon 场景匹配代价数值变化；13 场景 golden 经 `--update-golden` 重生成**逐字节不变**（见 §5） |

## 3. 调用点

- Python：`ai/tracker.py::_shape_iou` → `ai.detection.polygon_iou`；
  `Detection.polygon_iou_with` 同路径。
- Rust：`tracker.rs` 两处 `shape_iou`（匈牙利代价矩阵构建）→
  `geometry::shape_iou` → `precise_polygon_iou`（双侧 4 顶点 polygon 时；
  否则回退 `box_iou`）。旧 `polygon_bbox_iou` 已删除（Rust 侧不再有调用点）。
- 测试辅助：`omintrack._polygon_iou(a, b)`，仿 `_bhattacharyya_similarity` /
  `_hungarian_pairs` 模式注册于 `lib.rs`。

## 4. 维护指南

- **非凸/自交多边形不支持**：Sutherland–Hodgman 仅保证凸-凸场景正确。
  若未来 polygon 来源不再是 OBB 四边形（如分割掩码轮廓），须先凸包化或
  换通用多边形裁剪算法，并同步双侧；
- **修改算法必须双側同步**：Python `_precise_polygon_iou` /
  `_clip_convex_ccw` / `_line_intersection` / `_signed_area2` 与 Rust
  `precise_polygon_iou` / `clip_convex_ccw` / `line_intersection` /
  `signed_area2` 一一对应，运算次序保持一致（浮点结合律敏感）；
- 改任一侧后：`pytest tests/test_tracker_rust_equivalence.py`（18 项）+
  `pytest tests/test_polygon_iou_precise.py`（13 项）必过；
- 若行为属有意变更（改变匹配代价进而改变指派）：先跑
  `pytest tests/test_tracker_snapshot.py -q` 确认挂的范围，再
  `--update-golden` 并逐场景人工核对 diff；
- 禁止在生产路径重新引入 shapely（或其他几何库）依赖——环境分歧是
  本方案要消除的缺陷；
- Rust 侧保持零 unsafe、f64、现有依赖（pyo3 + numpy + nalgebra）不变。

## 5. golden 更新结果摘要（2026-11-22）

| 场景 | golden diff |
| --- | --- |
| obb_polygon（唯一 polygon 场景） | **逐字节不变**：单 track 平滑序列，相邻帧精确 IoU vs 外接矩形 IoU 最大差 0.2408，但匹配指派/输出未变（代价差异未穿越门控、未改变匈牙利指派序） |
| 其余 12 场景（无 polygon） | 逐字节不变 |

## 6. 验证记录（2026-11-22）

- `cargo check` 零警告；`maturin develop --release` 重建成功；
- 新增 `tests/test_polygon_iou_precise.py` 13 项全绿（含 shapely 交叉对拍
  400+400 组随机凸四边形/旋转矩形，最大误差 < 1e-9；双側对拍 800+ 组，
  实测最大差 < 1e-15）；
- `pytest tests/test_tracker_rust_equivalence.py` 18 项全绿；
- `pytest tests/test_tracker_snapshot.py` 14 项全绿（含 --update-golden 重生成）；
- 全套件 1391 项全绿。
