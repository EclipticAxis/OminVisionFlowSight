# omintrack Rust 跟踪核心 设计文档

> **版本**：1.6 — 2026-11-26
> **范围**：1.6 新增 `max_output_misses`，将预测输出可见性与内部轨迹生命周期解耦，见 [prediction-output-gating](prediction-output-gating.md)；此前新增 `rust/omintrack/`（PyO3 cdylib）、`ai/tracker_backend.py`（适配层）、`ai/tracker.py` 后端开关、32 项专项测试与性能基准；1.5 MCUKF 改为标准 UT + R 对角白化的创新域 IRLS-MCC（Python/Rust 双侧、默认 σ=1.0、c_min=1e-4），见 [mcukf-irls-correction](mcukf-irls-correction.md)；1.4 polygon IoU 由"外接 bbox 降级语义"升级为双側自实现精确凸多边形 IoU（Sutherland-Hodgman + shoelace，不再依赖 shapely，消除环境分歧源），见 [precise-polygon-iou](precise-polygon-iou.md)；1.1 追加 `rust/omintrack/src/py_api.rs` 外观数组 FFI 零拷贝优化（`Cargo.toml` 增 numpy crate 依赖）；1.2 追加 ReID mock 止损门控约定（仅 Python 侧 tracker.py / tracker_backend.py 门控条件变更，Rust core 不变，见 §3.2）；1.3 跟踪后端缺省 "python" → "auto"（构造时 import omintrack 探测，ImportError/OSError 回退 Python；显式 "rust" 仍抛错不回退），GUI 配置链见 [default-backends](default-backends.md)
> **状态**：1.0 已合入 main（PR #2 → `3243016`）；1.1 外观数组 FFI 零拷贝优化已合入 main（PR #7 → `3ee6a61`）；1.2 ReID mock 止损门控已合入 main（PR #8 → `1e636e8`，未触碰 Rust 文件）。cargo check / maturin 重建 / 32 项专项 + 全套件 1350 项全绿；1.3 auto 默认已合入 main（PR #9 → `7f78651`；新增 tests/test_default_backends.py 16 项，全套件 1366 项全绿）；1.4 精确 polygon IoU 已合入 main（PR #14 → `e43c46c`）（双側逐行同算法 + `_polygon_iou` 测试辅助，13 项新增测试，goldens 经 --update-golden 重生成逐字节不变——13 场景匹配指派未变，详见 precise-polygon-iou.md）；MCUKF IRLS 修正（2026-11-25 已合入 main（PR #18 → a81ce34）：release 重建、等价 18 项、全套件 1554 项通过，行为契约见 §3.2 末条与 [mcukf-irls-correction](mcukf-irls-correction.md)）；1.6 预测输出门控待合入 main（release 重建、双端专项 32 项、快照 14 项、全套件 1415 项通过，MOT17 验证见 [prediction-output-gating](prediction-output-gating.md) §4）

---

## 1. 目标与背景

### 1.1 为什么用 Rust 重写跟踪核心

跟踪器（DetectionTracker）是每帧热路径：UKF 滤波族 17 个 sigma 点的矩阵运算、按 label 分组的匈牙利匹配、轨迹生命周期管理。纯 Python + NumPy 实现时，20 tracks × 500 帧合成负载下单线程墙钟约 1.3~1.9s；多路相机（最多 4 路）同时跟踪时，跟踪耗时直接挤占共享推理线程预算。

重写目标按优先级：

1. **行为零变化**：任何场景下与 Python 版输出一致（track_id / 状态机精确一致，数值 1e-6 容差内）——硬约束，见 §3；
2. **数量级提速**：实测平均 ~37x（17x~60x，见 §5.3）；
3. **零侵入接入**：缺省后端 "auto"（omintrack 可导入即 Rust，否则回退 python），一行参数即可切换，调用方零改动。

### 1.2 为什么不删 Python 实现

Python 版（`ai/tracker.py`）保留为**参考实现与无构建环境回退**（缺省 "auto" 时 omintrack 不可导入即用它）：

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
├── geometry.rs   # box / 精确凸多边形 IoU（Sutherland-Hodgman）、中心
├── adaptor.rs    # AUTO 自适应滤波器选择（阈值 + hysteresis）
├── py_api.rs     # PyO3 class RustTracker 绑定（dict 进 dict 出）
└── lib.rs        # 模块声明 + 测试辅助 pyfunction
```

### 2.2 双后端与适配层

`DetectionTracker.__new__`（ai/tracker.py）实现后端开关：

- `backend="rust"` 或环境变量 `OMINTRACK_BACKEND=rust` → 返回 `RustDetectionTracker`（ai/tracker_backend.py）实例，构造签名与公开 API 完全一致；
- 缺省 `"auto"`：`import omintrack` 成功 → 按 `"rust"` 走，`ImportError` / `OSError`（.pyd DLL 加载失败）→ 回退 `"python"`（debug 日志），行为与历史版本一致；显式 `"python"` 恒为 Python；显式 `"rust"` 不可用时抛错不回退。仅对精确类 `DetectionTracker` 生效（子类不被拦截）。详见 [default-backends](default-backends.md) §2.1。

`RustDetectionTracker`（facade）的职责划分：

- **Rust 核心**：UKF 族 / AUTO / 匹配 / 重检测 / 生命周期；
- **Python 保留（帧相关计算）**：HSV 直方图（复用 `_PyTracker._compute_color_hist_for_box` 保证逐位一致）、ReID embedding 提取与激活门控、`redetect_callback` 包装（list→ndarray）；
- 外观闭环：Rust 经 `appearance_update_rois()` 给出 ROI → Python 取中心 50% ROI 算直方图 → `update_color_hists` 回传（misses>0 的 track 跳过）。

### 2.3 FFI 边界

- 输入：检测 dict（`x1/y1/x2/y2/confidence/label`，可选 `polygon/keypoints`）+ `det_hist` / `det_emb` / `reid_active` / `redetect_callback`；
- 外观数组输入（v1.1 起）：`det_hist` / `det_emb` 元素与 `update_color_hists` 的 value 支持 NumPy ndarray（float64/float32）零拷贝快路径——`numpy::PyArray1` 直接读底层 buffer，float32 经 `f64::from` 无损 widening；非 contiguous 数组走 `as_array()` 逐元素兜底；非 ndarray 对象回退 list 兼容旧路径（`extract::<Vec<f64>>`）。FFI 形参类型为 `PyObject`，转换后仍以 `Vec<Option<Vec<f64>>>` 传入 core，`tracker.rs` / `filters.rs` 签名与实现完全不变；
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
- 内部生命周期：新检测立即建轨（hits=1，无 tentative 期）；`misses > max_misses`（默认 4，即第 5 次 miss）删除；
- 输出可见性：`hits >= min_hits_to_show` 且 `misses <= max_output_misses`（默认 0）才输出；首次推理 miss 后预测框默认隐藏，内部 track 不提前删除，correct 后以原 ID 恢复；`max_output_misses=max_misses` 恢复旧持续预测语义。详见 [prediction-output-gating](prediction-output-gating.md)；
- EMA 顺序与系数：correct → box EMA(α=0.85) → confidence 0.7/0.3 → keypoints（x/y 用 α、conf 用 0.7/0.3）→ 空关键点按 bbox warp → polygon EMA；`prev_box` 在 EMA 之后保存；
- 匹配代价：匈牙利 0.85·IoU + 0.10·center + 0.05·appearance（ReID 激活且双侧 embedding 存在时 0.4/0.4/0.2；ReID 激活另需 backend 非 mock：is_mock=False，门控在 Python 侧双實施行并使用 getattr 默认 False 兼容自定义 backend，见 [reid-osnet-backend](reid-osnet-backend.md) §3）；门控回退 IoU>0.05 且中心距<0.5·avg_size；无效匹配 cost≥0.999 丢弃；greedy 的 appearance 权重 0.15；polygon 双侧存在（均 4 顶点）时 IoU 为**精确凸多边形 IoU**（Sutherland-Hodgman 裁剪 + shoelace 面积，双側逐行同算法，Rust `geometry::precise_polygon_iou` ↔ Python `ai.detection._precise_polygon_iou`），union ≤ 1e-8 返回 0.0；
- 重检测：仅 label=="person"、按 misses 降序、每帧最多 `redetect_budget` 个；refined box 与预测框 IoU≥0.20 才算成功；超预算只 miss+1 不累加 attempts；
- AUTO 适配器：EVAL_INTERVAL=30、NIS>8.0、P 条件数>1000、miss 率>0.5、Cholesky 失败>2、hysteresis=2 次投票才切换；
- **MCUKF IRLS 修正**（2026-11-25 已合入 main（PR #18 → a81ce34）；Python / Rust 双側须同契约，详见 [mcukf-irls-correction](mcukf-irls-correction.md)）：
  - z_pred / Pzz / Pxz 恒为标准 UT（Wm/Wc 不变），核权**不得**替换 UT 权（旧实现因此增益坍缩，已废弃）；
  - 每通道核权 c_j = clip(exp(-y_j²/(2σ²)), 1e-4, 1)（c_min=1e-4 限制 R_eff / S 条件数），y = R^{-1/2}·residual_t；residual_t = z − (z_pred + H(x_iter − x_pred)) 每轮在当前迭代点重算（对角 R 白化，σ 无量纲）；
  - R_eff = R·C⁻¹（C = diag(c)），K = Pxz·(Pzz + R_eff)⁻¹；每轮 x_new = x_pred + K(z − z_pred)（自 x_pred 重算，不累加）；IRLS 最多 3 次，‖x_new − x_iter‖ 阈值 1e-6，达上限采用最后一轮的 x_new / K / R_eff（同轮一致）；
  - Joseph 协方差更新使用同一 R_eff；AUTO / NIS 一律使用原始 S0 = Pzz + R（不用 R_eff）；
  - 退化路径：大 σ 或零残差 → 标准 UKF；强离群 → K → 0 且 P → 预测协方差；S = Pzz + R_eff 奇异语义与 UKF 一致（Python 异常 / Rust RuntimeError 向上传播）；
  - σ 默认 1.0，GUI 0.5–3.0，旧配置超出范围由 loader 回退 1.0；普通噪声下不恒等于 UKF（R 白化使正常帧按信噪比降权，属有意保守性），σ 需场景标定，非有限输入不承诺；对比 S0 白化 σ≈2，R 白化 + σ=1.0 离群抑制更好且正常/真实突变恢复无显著劣势；
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

### 4.2 启用（缺省 auto，无需手动启用）

```python
tracker = DetectionTracker()                 # 缺省 "auto"： omintrack 可导入即 Rust，否则 Python
tracker = DetectionTracker(backend="rust")   # 显式指定（扩展不可用时抛错，不静默回退）
tracker = DetectionTracker(backend="python") # 显式 Python（对拍 / 调试）
# 或设置环境变量 OMINTRACK_BACKEND=python|rust|auto，覆盖所有默认构造点
```

## 5. 验证体系

### 5.1 等价性测试（18 例）

同一合成序列双跑两实现逐帧比对：三种滤波器、AUTO 330 帧切换、交叉轨迹 ID 保持、greedy/cascade 混合置信、消失删除、predict_step、OBB polygon、关键点 EMA、重检测成功/失败、真实帧 HSV 外观、ReID embedding、匈牙利对拍 scipy、Bhattacharyya 对拍 cv2。

### 5.2 行为快照（13 场景 golden）

序列化输出 + 内部 meta（misses/hits/attempts/state/滤波器类型）round 到 9 位小数与 JSON 精确比对；`--update-golden` 再生成。

### 5.3 基准

`benchmark/benchmark_tracker_rust.py`：20 tracks × 500 帧（~18.6 检测/帧），3 滤波器 × 3 策略；当前实测均值 ~37x（ukf+hungarian 最快 ~60x，manifold_ukf 最慢 ~18x）。

### 5.4 外观数组 FFI 零拷贝优化（v1.1，2026-11-22）

**动机**：生产路径 Python 侧传入 NumPy ndarray（hist 为 float32 520 维、emb 256 维），但旧版 pyo3 形参是 `Vec<f64>`，触发迭代器慢路径逐元素构造 PyFloat；实测每帧 20 检测仅外观提取浪费 ~159µs（比核心计算 63µs 还贵），单帧 update 端到端 ~249µs。

**方案**：`py_api.rs` 边界做向后兼容的多路径提取（`extract_f64_vector`）：ndarray f64/f32 零拷贝 → f32 逐元素 `f64::from` 无损转换 → list 兼容回退。core 与 Python facade、测试一律不改。

**实测（本机，20 tracks 稳定态，20 检测 × 520 维 hist，同一微基准脚本前后对照）**：`update` 端到端（ndarray hist）**249µs → 66µs/帧**，外观提取开销由 +186µs 降至 +4µs（f64）/ +10µs（f32，含 widening）；list 回退路径保持 +28µs 不变；`update_color_hists` ndarray 路径 5.9µs vs list 28.3µs（~4.8x）。行为等价性由 §5.1 / §5.2 原有用例回归保证（f32→f64 widening 与旧路径逐位一致），全套件 1341 项通过。

## 6. 维护指南

- 改算法先读 §3.2 契约表，判断是否触碰 quirk 依赖点；
- Rust 侧保持零 unsafe；新增依赖需评估（当前 pyo3 0.22 + numpy 0.22 + nalgebra 0.32；numpy crate 与 pyo3 版本配套，升级 pyo3 时必须同步核对）；
- 不要把帧相关计算（cv2/模型）移入 Rust——FFI 边界是有意划分；
- 提交前：`cargo check` 零警告 + maturin 重建 + 全套件通过。
