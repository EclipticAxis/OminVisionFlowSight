# omintrack Rust 跟踪核心 设计文档

> **版本**：1.0 — 2026-11-18
> **范围**：新增 `rust/omintrack/`（PyO3 cdylib）、`ai/tracker_backend.py`（适配层）、`ai/tracker.py` 后端开关、32 项专项测试与性能基准
> **状态**：已合入 main（PR #2 → `3243016`）；18 项等价性 + 13 场景快照测试全部通过；全套件 1277 通过；基准平均 ~37x 加速

---

## 1. 目标与背景

### 1.1 为什么用 Rust 重写跟踪核心

跟踪器（DetectionTracker）是每帧热路径：UKF 滤波族 17 个 sigma 点的矩阵运算、按 label 分组的匈牙利匹配、轨迹生命周期管理。纯 Python + NumPy 实现时，20 tracks × 500 帧合成负载下单线程墙钟约 1.3~1.9s；多路相机（最多 4 路）同时跟踪时，跟踪耗时直接挤占共享推理线程预算。

重写目标按优先级：

1. **行为零变化**：任何场景下与 Python 版输出一致（track_id / 状态机精确一致，数值 1e-6 容差内）——硬约束，见 §3；
2. **数量级提速**：实测平均 ~37x（17x~60x，见 §5.3）；
3. **零侵入接入**：默认后端不变（python），一行参数即可切换，调用方零改动。

### 1.2 为什么不删 Python 实现

Python 版（`ai/tracker.py`）保留为**参考实现与默认后端**：

- 无 Rust 工具链（或未执行 §4.1 构建）时项目开箱即用；
- 等价性测试以 Python 版为 ground truth 对拍 Rust 版；
- 算法迭代可先在 Python 侧验证，再移植 Rust（见 §3.3 修改契约）。

## 2. 架构

### 2.1 模块划分

```
rust/omintrack/src/
├── tracker.rs    # Track + TrackerCore：匹配三策略 / 重检测 / 清理 / AUTO 评估
├── filters.rs    # UKF / MCUKF / ManifoldUKF（nalgebra 静态矩阵，8 维状态 17 sigma 点）
├── hungarian.rs  # O(n^3) 匈牙利（矩形补零成方阵，确定性 tie-break）
├── geometry.rs   # box / 多边形外接 bbox IoU、中心
├── adaptor.rs    # AUTO 自适应滤波器选择（阈值 + hysteresis）
├── py_api.rs     # PyO3 class RustTracker 绑定（dict 进 dict 出）
└── lib.rs        # 模块声明 + 测试辅助 pyfunction
```

### 2.2 双后端与适配层

`DetectionTracker.__new__`（ai/tracker.py）实现后端开关：

- `backend="rust"` 或环境变量 `OMINTRACK_BACKEND=rust` → 返回 `RustDetectionTracker`（ai/tracker_backend.py）实例，构造签名与公开 API 完全一致；
- 默认 `"python"`，行为与历史版本完全一致；仅对精确类 `DetectionTracker` 生效（子类不被拦截）。

`RustDetectionTracker`（facade）的职责划分：

- **Rust 核心**：UKF 族 / AUTO / 匹配 / 重检测 / 生命周期；
- **Python 保留（帧相关计算）**：HSV 直方图（复用 `_PyTracker._compute_color_hist_for_box` 保证逐位一致）、ReID embedding 提取与激活门控、`redetect_callback` 包装（list→ndarray）；
- 外观闭环：Rust 经 `appearance_update_rois()` 给出 ROI → Python 取中心 50% ROI 算直方图 → `update_color_hists` 回传（misses>0 的 track 跳过）。

### 2.3 FFI 边界

- 输入：检测 dict（`x1/y1/x2/y2/confidence/label`，可选 `polygon/keypoints`）+ `det_hist` / `det_emb` / `reid_active` / `redetect_callback`；
- 输出：track dict（键与 `_Track.to_dict()` 完全一致）；
- 错误语义：解析错误 → KeyError/TypeError；UKF 创新协方差 S 奇异 → **RuntimeError**（不 panic）；redetect 回调抛出的 Python 异常被吞掉并按"重检测失败"处理（与 Python 版 try/except 一致）。

## 3. 等价性契约（隐性契约显性化）

### 3.1 数值契约

| 项 | 契约 |
|---|---|
| track_id 集合 / track_state / predicted_only / label / keypoints 数量 / polygon 有无 / AUTO 类型 | **逐帧精确一致** |
| box 坐标 / confidence / keypoints 坐标与 conf / polygon 坐标 | 逐帧 **1e-6** 容差（330 帧递推实测最大偏差 ~7.2e-10；差异源自 nalgebra 与 NumPy/LAPACK 的 LU/Cholesky/EIGH 内部路径） |
| 浮点精度 | Rust 侧全 f64（禁止 f32） |

### 3.2 行为契约（含刻意保留的 quirk）

- **级联 Stage 1 quirk**：Python 版 `_match_cascade` 注释写"Stage 1 高置信度检测"，但实现传入**全部**检测（`high_indices` 计算后从未使用）。Rust 版**照搬此 quirk**（`match_cascade` 中 `_high` 同理占位）。修正它属于行为变更，必须双侧同步并更新 golden；
- 生命周期：新检测立即建轨（hits=1，无 tentative 期）；`misses > max_misses`（默认 4，即第 5 次 miss）删除；
- EMA 顺序与系数：correct → box EMA(α=0.85) → confidence 0.7/0.3 → keypoints（x/y 用 α、conf 用 0.7/0.3）→ 空关键点按 bbox warp → polygon EMA；`prev_box` 在 EMA 之后保存；
- 匹配代价：匈牙利 0.85·IoU + 0.10·center + 0.05·appearance（ReID 激活且双侧 embedding 存在时 0.4/0.4/0.2）；门控回退 IoU>0.05 且中心距<0.5·avg_size；无效匹配 cost≥0.999 丢弃；greedy 的 appearance 权重 0.15；
- 重检测：仅 label=="person"、按 misses 降序、每帧最多 `redetect_budget` 个；refined box 与预测框 IoU≥0.20 才算成功；超预算只 miss+1 不累加 attempts；
- AUTO 适配器：EVAL_INTERVAL=30、NIS>8.0、P 条件数>1000、miss 率>0.5、Cholesky 失败>2、hysteresis=2 次投票才切换；
- 匈牙利确定性：矩形代价补零成方阵，与 scipy `linear_sum_assignment` 数学等价；tie-break 行优先、严格小于才更新；
- Bhattacharyya：任一侧直方图缺失返回 0.5；公式按 float64 复刻 cv2（cv2 内部 float32，交叉验证差 ~3e-9）。

### 3.3 修改契约（两侧同步规则）

**任何跟踪行为变更必须**：

1. Python / Rust 两侧同步修改；
2. `pytest tests/test_tracker_rust_equivalence.py` 通过；
3. 如行为属有意变化：`pytest tests/test_tracker_snapshot.py --update-golden` 再生成 13 个 golden 并人工核对 diff；
4. 复跑全套件。

只改一侧会被等价性测试当场发现——这正是该契约的机器保障。

### 3.4 已知边界

- polygon IoU 按**无 shapely** 环境的 `_fallback_bbox_iou`（顶点集外接矩形）语义实现；若 Python 环境安装 shapely，两后端 OBB 匹配会出现语义分歧（届时需重新评估）；
- UKF 创新协方差 S 奇异：Python 侧原行为为 np.linalg 异常，Rust 侧为 RuntimeError（均向上传播）；
- 缺失 `label` 的检测按空串处理（与 Python `det.get("label", "")` 一致）；
- greedy 排序：Python `np.argsort(-conf)` 非稳定、Rust 稳定排序——置信度完全并列时序可能不同（Python 侧本身不确定，Rust 只会更确定）；
- `update` 期间持有 GIL（重检测回调需要）；如需进一步释放可在无回调路径加 `py.allow_threads`。

## 4. 构建与启用

### 4.1 构建（Windows）

前置 Rust 工具链（本机 `x86_64-pc-windows-gnu`）与 venv 内 maturin：

```powershell
venv\Scripts\python.exe -m maturin develop --manifest-path rust\omintrack\Cargo.toml --release
```

国内镜像配置与排错见 `rust/omintrack/README.md`。**修改 rust/ 源码后必须重新执行本命令**，否则 venv 内 .pyd 与源码不一致（可比对 `venv/.../omintrack*.pyd` 与 `rust/omintrack/target/release/omintrack.dll` 的 MD5 验证）。

### 4.2 启用

```python
tracker = DetectionTracker(backend="rust")   # 显式指定
# 或设置环境变量 OMINTRACK_BACKEND=rust，所有默认构造点自动切换
```

## 5. 验证体系

### 5.1 等价性测试（18 例）

同一合成序列双跑两实现逐帧比对：三种滤波器、AUTO 330 帧切换、交叉轨迹 ID 保持、greedy/cascade 混合置信、消失删除、predict_step、OBB polygon、关键点 EMA、重检测成功/失败、真实帧 HSV 外观、ReID embedding、匈牙利对拍 scipy、Bhattacharyya 对拍 cv2。

### 5.2 行为快照（13 场景 golden）

序列化输出 + 内部 meta（misses/hits/attempts/state/滤波器类型）round 到 9 位小数与 JSON 精确比对；`--update-golden` 再生成。

### 5.3 基准

`benchmark/benchmark_tracker_rust.py`：20 tracks × 500 帧（~18.6 检测/帧），3 滤波器 × 3 策略；当前实测均值 ~37x（ukf+hungarian 最快 ~60x，manifold_ukf 最慢 ~18x）。

## 6. 维护指南

- 改算法先读 §3.2 契约表，判断是否触碰 quirk 依赖点；
- Rust 侧保持零 unsafe；新增依赖需评估（当前仅 pyo3 + nalgebra）；
- 不要把帧相关计算（cv2/模型）移入 Rust——FFI 边界是有意划分；
- 提交前：`cargo check` 零警告 + maturin 重建 + 全套件通过。
