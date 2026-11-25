# MCUKF IRLS 修正：从"凸核权当 UT 权"到标准 UT + R 对角白化

> **版本**：1.0 — 2026-11-25
> **范围**：`ai/tracker.py` / `rust/omintrack/` 中 `_DetectionMCUKF` 的 update 行为修正；替代 [mcukf-manifold-ukf-plan](mcukf-manifold-ukf-plan.md) §2.1 / §5.1 / §6 中的旧 MCUKF 伪码与 σ 语义；`docs/rust-tracker-core.md` §3.2 新增对应行为契约。
> **状态**：已合入 main（PR #18 → a81ce34）：Rust release 重建成功；新增 IRLS 11 项、Rust 等价 18 项、快照 14 项、配置 23 项及全套件 1554 项通过；MOT17 公共序列对照见 §4

---

## 1. 目标与背景

### 1.1 旧实现的缺陷

旧 MCUKF update 把 sigma 点残差的高斯核权重 **归一化后直接当作 UT 的 Wm/Wc 使用**（替换 `self._wm` / `self._wc` 参与 z_pred / S / Pxz 加权）。这有两个根本性错误：

1. **增益坍缩**：σ 点权与 UT 权属于不同数学对象。归一化凸核权使加权协方差系统性低估，K = Pxz·S⁻¹ 在离群帧上不是趋于零而是出现非物理行为；鲁棒性名存实亡。
2. **假迭代**：迭代内 z_pred / S / Pxz 全部用被污染的权重建，不动点迭代不收敛到不动点，只是重复同一场错误加权，"迭代到收敛"是表象。

旧文档中 σ 的语义（按量测噪声标准差的倍数、推荐 2~5× std、默认 0.4）也随之失效——新实现中 σ 是**无量纲**参数。

### 1.2 修正思路

新实现遵循 IRLS（Iteratively Reweighted Least Squares）视角：保留标准 UT 的全部加权结构（Wm/Wc 不变，z_pred / Pzz / Pxz 不变），把相关熵的作用点移到**量测噪声协方差**上——对当前量测残差逐通道白化 R，等效为给该次量测"涨 R"，从而等效缩小该次更新的增益。

正确性边界（明确声明，禁止夸大）：

- 普通高斯噪声下 MCUKF **不恒等**于标准 UKF（除非 σ → ∞ 或残差全零的退化情形）；核函数对非离群残差也施加 c<1 的降权；
- **有意的鲁棒保守性**：R 白化以信噪比为残差尺度，正常帧同样按 c_j<1 被降权，且降权幅度随信噪比变化——这是设计上有意的保守性，而非逼近恒等 UKF；
- **白化空间选择**：采用 R 白化、默认 σ=1.0；相对对 S0 = Pzz + R 白化（残差含预测不确定度，需 σ≈2 才相当）的方案，R 白化离群抑制更强，正常帧与真实突变后的恢复无显著劣势（定性设计依据，无量化基准数据承诺）；
- σ 需要按场景标定：噪声尺度、检测框抖动幅度、误检率都会影响合理取值，不存在"全场景最优"默认；
- 非有限输入（NaN / ±Inf 的 box、σ、残差）不在鲁棒性承诺范围内，行为按实现现状为准，调用方负责输入卫生；
- 本修正不宣称"所有场景下完美"，只承诺机制正确（增益随离群幅度单调受抑）与退化路径明确。

## 2. 架构

### 2.1 update 数据流

```
predict 之后进入 correct(z)：
┌─────────────────────────────────────────────────────────┐
│ 1. sigma 点 + 观测传播（与标准 UKF 完全相同）              │
│    z_pred = Σ Wmⁱ·Zⁱ      （UT 权，不变）                 │
│    Pzz    = Σ Wcⁱ·(Zⁱ-z_pred)(Zⁱ-z_pred)ᵀ               │
│    Pxz    = Σ Wcⁱ·(Xⁱ-x_pred)(Zⁱ-z_pred)ᵀ               │
│    S0     = Pzz + R           ← AUTO / NIS 永远用它       │
├─────────────────────────────────────────────────────────┤
│ 2. IRLS 迭代（最多 3 次；x_iter 首轮 = x_pred）            │
│    residual_t = z - (z_pred + H·(x_iter - x_pred))        │
│      └ 当前迭代点的预测量测残差，每轮重算（非固定 z-z_pred） │
│    y = R^{-1/2}·residual_t        ← 对角 R 白化           │
│    C = diag(c_j), c_j = clip(exp(-y_j²/(2σ²)), 1e-4, 1)  │
│    R_eff = R·C⁻¹                                          │
│    K = Pxz·(Pzz + R_eff)⁻¹                                │
│    x_new = x_pred + K·(z - z_pred)  ← 自 x_pred 重算，     │
│                                       不累加到 x_iter      │
│    收敛：‖x_new - x_iter‖ < 1e-6 则退出                    │
│    达上限：采用最后一轮的 x_new / K / R_eff               │
│    x_iter ← x_new                                         │
├─────────────────────────────────────────────────────────┤
│ 3. 状态与协方差定稿（与最后一轮 IRLS 严格同轮）            │
│    x = x_new                                              │
│    Joseph：P = (I-KH)·P_pred·(I-KH)ᵀ + K·R_eff·Kᵀ        │
│    （K / R_eff 取最后一轮值，与定稿 x_new 一致）            │
└─────────────────────────────────────────────────────────┘
```

### 2.2 退化路径（契约化）

| 条件 | 行为 |
|---|---|
| σ 很大 或 残差全零（‖y‖ → 0） | c_j → 1，R_eff → R，K 与 P 更新退化为标准 UKF |
| 强离群（‖y_j‖ ≫ σ） | c_j → 1e-4 下限，R_eff,j 最多放大 ~1e4（条件数受限），K 该通道 → 0，P → 预测协方差（该次更新近似无信息注入） |
| 迭代达 max_iter=3 | 不报错，采用最后一次鲁棒迭代结果（IRLS 惯例） |
| S = Pzz + R_eff 奇异 | 与 UKF 相同语义：Python 侧 np.linalg 异常 / Rust 侧 RuntimeError 向上传播，不 panic、不静默 |

### 2.3 与标准 UKF / AUTO 的关系

- z_pred / Pzz / Pxz 的**计算**永远走标准 UT，不被核权污染；
- AUTO 适配器的 NIS 与 Cholesky 监控使用 **原始 S0 = Pzz + R**（一次量测的鲁棒降权不影响 AUTO 对滤波器健康度的评估）；
- Joseph 形式的 R 与增益同一 R_eff，保持 P 的语义一致。

## 3. 契约表

| 契约项 | 定义 |
|---|---|
| UT 结构不变性 | z_pred / Pzz / Pxz / Wm / Wc 与标准 UKF 逐行相同（MCUKF 只改 R_eff 与迭代外壳） |
| 白化残差 | y = R^{-1/2}·residual_t；residual_t = z − (z_pred + H(x_iter − x_pred)) 逐轮重算，仅对对角 R 定义（非对角 R 不承诺） |
| 核权 | c_j = clip(exp(-y_j²/(2σ²)), 1e-4, 1)，σ 无量纲，默认 1.0；c_min=1e-4 用于限制 R_eff / S 条件数不过度极端 |
| 有效量测噪声 | R_eff = R·C⁻¹，C = diag(c) |
| 增益 | K = Pxz·(Pzz + R_eff)⁻¹；定稿 K 取最后一轮，与 x_new 同轮 |
| 迭代 | 最多 3 次；‖x_new − x_iter‖ 阈值 1e-6；达上限采用最后一次鲁棒迭代的 x_new / K / R_eff（三者同轮一致） |
| 协方差更新 | Joseph：P = (I-KH)P(I-KH)ᵀ + K·R_eff·Kᵀ，R 取同一 R_eff |
| AUTO / NIS | 一律使用原始 S0 = Pzz + R，不使用 R_eff |
| σ 语义 | 无量纲（白化残差空间）；GUI 范围 0.5–3.0；旧配置超出范围由 loader 回退 1.0 |
| 退化 | 大 σ 或零残差 → 标准 UKF；强离群 → K→0、P→预测 |
| 双实现 | Python `ai/tracker.py` 与 Rust `rust/omintrack/`（filters.rs）行为锁定等价，遵循 rust-tracker-core §3 修改契约 |
| 边界（非承诺） | 普通噪声 ≠ UKF；σ 需场景标定；非有限输入不承诺；不宣称全场景完美 |

## 4. 验证结果

- **构建与双实现**：`maturin develop --release` 成功，安装后的 `omintrack.pyd` 与 `target/release/omintrack.dll` MD5 一致；`tests/test_tracker_rust_equivalence.py` 18 项通过。
- **算法契约**：`tests/test_mcukf_irls.py` 11 项通过，覆盖大 σ 退化、正常量测、极端/单通道离群、真实 IRLS 迭代、独立参考实现和长序列协方差健康。
- **快照与配置**：`tests/test_tracker_snapshot.py --update-golden` 14 项通过；仅 `single_uniform_mcukf.json` 改变（40 帧恢复单 ID、无 predicted 幽灵），`auto_outlier.json` 逐字节不变。默认/迁移/GUI 配置测试 23 项通过。
- **全套件**：`pytest tests/ -q` 共 **1554 项通过**。
- **MOT17-02/04（FRCNN，全序列，Python，stride 1/4）**：
  - stride 1：UKF → MCUKF 的 HOTA `40.357 → 40.639`、MOTA `45.673 → 45.688`、IDF1 `47.653 → 47.799`、IDSW `223 → 222`；
  - stride 4：HOTA `45.803 → 45.801`、MOTA `47.204 → 47.210`、IDF1 `52.441 → 52.439`、IDSW `196 → 196`，整体基本持平。
- **MOT17-10/13 高 stride AUTO**：诊断确认 stride 4/5 四个 run 分别发生 2/2/1/3 次 UKF↔MCUKF 切换；与同配置 UKF 相比，stride 4 指标一致，stride 5 仅 HOTA `-0.001`、MOTA `-0.029`、IDF1 `-0.004`、IDSW 不变。该困难基线本身 MOTA 为负，禁止只引用 AUTO 的负 MOTA 而省略 UKF 对照；本对照未发现切入 MCUKF 后的额外退化。

## 5. 维护指南

- 改 IRLS 任一侧（Python / Rust）必须双侧同步并跑 `pytest tests/test_tracker_rust_equivalence.py`；属有意行为变更时另跑 `--update-golden`（rust-tracker-core §3.3）；
- 不要重新引入"核权替换 UT 权"的写法——任何加权位置变更都先对照 §3 契约表逐条核对；
- σ 的默认值 / 范围变更需同步三处：本文件契约表、`mcukf-manifold-ukf-plan.md` 标注、GUI loader 回退逻辑；
- AUTO/NIS 若改用 R_eff，属于行为变更，需显式更新 rust-tracker-core §3.2 与本文件，并重新评估 AUTO 切换阈值；
- 2026-11-25 之前的 MCUKF / AUTO 基准、golden 分析结论和“切换后 MOTA 崩坏”数据均属于旧算法；用于论文或发布说明前必须按同一数据、同一配置重跑，不得与本次 IRLS 结果混作同版本比较；
- 后续算法变更必须重新执行 §4 验证并同步状态；合入后更新本文件与 rust-tracker-core 的合并状态，禁止预写尚未取得的测试或性能结论。
