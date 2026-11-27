# 跟踪增强批次 1+2 合包：NSA-KF + OCM 方向代价 + HMIoU

> **版本**：0.1.0 — 2026-11-27
> **范围**：`ai/tracker.py` 与 `rust/omintrack/` 双实现；离线评测 runner 增加开关。三个机制**默认全部关闭（opt-in）**，默认路径行为不变（既有 golden / 等价测试不需重生成），开启路径由新增测试覆盖。
> **状态**：已实现并通过验收（2026-11-27）。双端同步 + Rust release 重建；新增 21 项增强测试，全套件 1501 passed + 1 skipped；默认路径 golden 未变（`git status tests/golden/` 为空）；MOT 对照无回归，结果见 §4.1。

---

## 1. 目标与背景

基线证明关联准确度（AssA）是主要瓶颈。本包引入三个互补、低成本的机制：

| 机制 | 作用点 | 出处 | 预期 |
|------|--------|------|------|
| NSA-KF | 观测噪声随检测置信度缩放 | GIAOTracker / HybridSORT `kalmanfilter_score.py` | 低置信度观测降权，减少状态污染 |
| OCM | 关联代价加入观测中心速度方向一致性 | OC-SORT `association.py::associate` | 抑制变向/交叉误配，降 IDSW |
| HMIoU | 高度 1 维 IoU 调制 2D IoU | HybridSORT `association.py::hmiou` | 遮挡/高度重叠时近似深度线索 |

## 2. 契约表（双实现必须逐位一致）

### 2.1 NSA-KF（观测噪声自适应）

- 位置：`_DetectionUKF.correct()`（MCUKF / Manifold 继承同一入口，语义保持一致）。
- 公式：`R_eff = R_scale · clip((1 - conf) · nsa_interval, nsa_min, 1.0)`，其中 `R_scale = self._make_r(w, h)`（既有尺度自适应保留），`conf` 为当前检测置信度。
- 参数：`nsa_enabled=False`（关）、`nsa_interval=1.0`、`nsa_min=0.1`。
- 偏离原文声明：原公式 `R ← (1-conf)·interval·R` 在 conf→1 时 R→0；本实现以 `nsa_min=0.1` 为乘子下界防 R 退化（保护性改动，非原文行为）。缺省 conf=0.5（中性），runner/线上应传入真实 conf。
- 与既有 `_adapt_q`（NIS→Q）正交，互不替代。NIS 统计仍用**未 NSA 缩放前**的 R 计算 S（保持 AUTO 适配器口径不变）——即：先按旧逻辑算 NIS/S 用于监控，再用 R_eff 参与 K 与状态/协方差更新。注意：NSA 只改变 K 与 P 更新，不改 AUTO 监控信号语义。

### 2.2 OCM（观测中心动量方向代价）

- `_Track` 新增状态：
  - `observations: dict[int, box]`（age→观测框，仅保留最近 `ocm_delta_t+1` 个 age）与 `last_observation`；
  - `velocity: (vy, vx) | None`（观测动量方向，归一化）。
- 更新时机：仅在实际匹配成功（correct）时记录观测并更新 velocity；纯 predict/miss 帧不写入观测。
- velocity 计算（`speed_direction(prev_box, cur_box)`）：中心差 `[Δcy, Δcx]/norm`；prev_box 取 `age - ocm_delta_t` 的观测，找不到则逐帧回退到最近可用观测，再退到 `last_observation`；无有效历史则 velocity=None。
- 关联代价（对每个 track×det 对）：
  - `prev_obs = observations[age-delta_t]`（回退同上）；`D = speed_direction(prev_obs, det_box)`；
  - `cos = clip(vy·D_y + vx·D_x, -1, 1)`；`θ = arccos(cos)`；
  - `angle_term = (π/2 - |θ|)/π`（∈[-0.5,+0.5]）；
  - `ocm_cost = angle_term · ocm_weight · det_conf`；velocity 无效时 ocm_cost=0。
- 融合：在既有 `score` 上**加** `ocm_cost`（与 OC-SORT 叠加到 IoU 求最大一致）：`score += ocm_cost`；`cost = 1 - score` 不变。ReID 分支同样叠加。
- 参数：`ocm_enabled=False`、`ocm_delta_t=3`、`ocm_weight=0.2`（对应 vdc_weight）。
- 门控（IoU 阈值 / 中心距回退）不变；OCM 只影响 score 排序，不改变门控通过性。

### 2.3 HMIoU（高度调制 IoU）

- 公式：`hmiou = IoU_2d · h_iou`，`h_iou = max(0, min(y2a,y2b)-max(y1a,y1b)) / (max(y2a,y2b)-min(y1a,y1b))`。
- 偏离原文声明：原文 height 项未 clip 负值，本实现对交集长度 `max(0,·)` 防负（保护性改动）。
- 用途：`hmiou_enabled=True` 时，关联的 **IoU 项与 IoU 门控**统一以 hmiou 替换 2D IoU（与 HybridSORT 用同一 iou_matrix 做门控一致）。center/appearance/OCM 项不变。
- 注意：hmiou ≤ IoU，门控在同等阈值下更严格——评测对照时需固定阈值并注明。
- 参数：`hmiou_enabled=False`。
- OBB polygon 路径不适用 HMIoU（高度语义基于轴对称框），polygon 检测对保持 2D polygon IoU 不变（文档化边界）。

### 2.4 配置与接口

- `DetectionTracker.__init__` / `set_tracker_params` 增加上述参数；Rust `py_api` 同步暴露。
- 默认全关 ⇒ 现有 `tests/test_tracker_snapshot.py` golden 与 `test_tracker_rust_equivalence.py` 默认路径**不应**有任何数值变化（这是验收硬条件）。
- 离线 runner 增加 `--nsa/--ocm/--hmiou`（及对应权重参数）透传。

## 3. 测试计划

- 新增 `tests/test_tracker_enhancements.py`：
  - NSA：conf 高/低时 R_eff 单调性；nsa_min 截断；conf 缺省中性；默认关闭时与现状逐位一致。
  - OCM：构造变向轨迹验证 angle_term 符号与量级；velocity=None 时 ocm_cost=0；观测仅在匹配时写入。
  - HMIoU：hmiou ≤ IoU；高度不重叠时 h_iou=0；门控替换生效。
  - 双后端：开启三特性的等价测试（复用 equivalence 框架，坐标 1e-6、状态机精确）。
- 既有全套件 `pytest tests/` 必须全绿（默认路径不变为前提）。

## 4. 评测计划（合入门槛）

对 DanceTrack val + MOT17 FRCNN 跑四组对照（rust 后端，stride 1）：baseline（已落盘）vs `+nsa` vs `+nsa+ocm` vs `+nsa+ocm+hmiou`；ukf 与 auto 各一组。验收按方案 §6：DanceTrack val HOTA 回退 >0.002（绝对值）或 MOT17 IDSW 上升需书面说明；预期 AssA/IDSW 改善。

### 4.1 实测对照（2026-11-27，rust，stride 1）

DanceTrack val（YOLOv8n 私有检测，25 序列）：

| 配置 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|------|------|------|------|------|------|------|
| ukf baseline | 0.2045 | 0.0842 | 0.5035 | 0.1827 | 0.5485 | 8470 |
| +nsa | 0.2126 | 0.0885 | 0.5183 | 0.1854 | 0.5605 | 8360 |
| +nsa+ocm | 0.2110 | 0.0873 | 0.5175 | 0.1836 | 0.5598 | 8456 |
| +nsa+ocm+hmiou | **0.2130** | **0.0888** | 0.5190 | 0.1865 | 0.5608 | **8352** |
| auto baseline | 0.2046 | 0.0842 | 0.5040 | 0.1827 | 0.5485 | 8476 |
| auto +full | **0.2132** | **0.0888** | 0.5193 | 0.1865 | 0.5607 | **8349** |

MOT17 FRCNN（7 序列）：

| 配置 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|------|------|------|------|------|------|------|
| ukf baseline | 0.3961 | 0.3765 | 0.4188 | 0.4639 | 0.4524 | 1131 |
| +nsa | 0.4147 | 0.4009 | 0.4307 | 0.4818 | 0.4652 | 1065 |
| +nsa+ocm | 0.4149 | 0.4008 | 0.4311 | 0.4815 | 0.4657 | 1045 |
| +nsa+ocm+hmiou | 0.4197 | 0.4083 | 0.4333 | 0.4867 | 0.4685 | 1081 |
| auto baseline | 0.3984 | 0.3807 | 0.4188 | 0.4668 | 0.4516 | 1128 |
| auto +full | **0.4298** | **0.4247** | 0.4368 | **0.4984** | **0.4743** | 1077 |

结论：
- **NSA-KF 是最有效的单项**（两数据集 HOTA/AssA/IDSW 全改善）；OCM 单独叠加收益弱（DanceTrack 上 HOTA 微降 0.0016，IDSW 仍优）；HMIoU 叠加后整体最优。auto+full 在 MOT17 提升最大（HOTA +0.031，AssA +0.044）。
- **全部配置无回归**（HOTA 均 ≥ 基线，IDSW 均下降），满足合入门槛。
- 幅度诚实声明：DanceTrack HOTA +0.009、MOT17 +0.024~0.031，属"小改有效"量级；距 Deep OC-SORT（DanceTrack≈0.55）的差距主因是检测器（DetA≈0.50）与尚未实现的 ORU/OCR/TCM，而非本包。
- 报告：`benchmark/results/20261127_enhance_dancetrack_val.{json,md}`、`20261127_enhance_mot17_frcnn.{json,md}`。

## 5. 引用（公式来源，逐字核对）

- OC-SORT：`github.com/noahcao/OC_SORT` → `trackers/ocsort_tracker/association.py`（`associate`/`speed_direction_batch`）、`ocsort.py`（`speed_direction`、`k_previous_obs`、velocity/observations 维护）。
- Hybrid-SORT：`github.com/ymzis69/HybridSORT` → `trackers/hybird_sort_tracker/association.py`（`hmiou`）、`kalmanfilter_score.py`（NSA-KF 行：`R = (1-confidence)·nsa_kalman_interval·R`）。

## 6. 维护指南

- 任何行为变更仍受 `docs/rust-tracker-core.md` §3.3 约束：双侧同步 + 等价测试 +（仅默认行为变更时）golden 更新 + 全套件。
- 三特性相互正交、可独立开关；后续 ROCM（多角点动量）/ TCM（置信度建模）/ ORU（重更新）在此观测历史基础上扩展，不要重复建设 `_Track.observations`。
- 若未来默认开启任一特性：属默认行为变更，须重生成 golden 并在 PROJECT_BIBLE 记录。
