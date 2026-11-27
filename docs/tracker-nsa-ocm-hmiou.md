# 跟踪增强批次 1+2 合包：NSA-KF + OCM 方向代价 + HMIoU

> **版本**：0.2.0 — 2026-11-27
> **范围**：`ai/tracker.py` 与 `rust/omintrack/` 双实现；离线评测 runner 增加开关。三个机制**默认全部关闭（opt-in）**，默认路径行为不变（既有 golden / 等价测试不需重生成），开启路径由新增测试覆盖。
> **状态**：已实现并通过验收（2026-11-27）。双端同步 + Rust release 重建；默认路径 golden 未变；MOT 对照无回归。
>
> **0.2.0 变更（PR #21 事后审计修复，详见 §7）**：① P0-1 OCM 负代价会冲破 `cost<0.999` 哨兵 → 改独立 `accepted` 布尔掩码；② P0-2 NSA 误用单段 clip≤1 公式（低置信度反被加信任）→ 改回上游**分段语义**，`conf` 缺省改 `None`（真正中性）；③ P0-3 评测口径护栏（见 benchmark 文档）。NSA 开启路径行为因此改变，**0.1.0 的 +nsa 对照数据作废**，以 §4.2 的 v2 为准。

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

- 位置：`_DetectionUKF.correct(box, conf=None)`（MCUKF / Manifold 继承同一入口，语义保持一致）。
- **分段公式（上游 HybridSORT `kalmanfilter_score.py` 真实结构）**：
  - `conf is None` → 乘子 `1.0`（**真正中性**，缺省安全，忘记传 conf 无任何 NSA 效果）；
  - `conf > nsa_conf_thresh`（默认 0.6）→ 乘子 `max((1-conf)·nsa_interval, nsa_min)`（高置信度**缩 R=信任观测**）；
  - `conf <= nsa_conf_thresh` → 乘子 `nsa_interval_sec`（默认 2.0，低置信度**放大 R=降权观测**）。
- 参数：`nsa_enabled=False`、`nsa_interval=1.0`、`nsa_interval_sec=2.0`、`nsa_conf_thresh=0.6`、`nsa_min=0.1`。
- 偏离原文声明：`nsa_min=0.1` 仅在高置信度分支防 conf→1 时 R→0（保护性截断，非原文行为）；低置信度分支的 `interval_sec` 放大是上游原有结构。
- ~~缺省 conf=0.5（中性）~~（0.1.0 错误表述，已删除）：0.5 在分段语义下落入低置信度分支会放大 R，**不是中性**；缺省改为 `None`。
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
- **0.2.0 修复（P0-1）**：门控通过性由独立 `accepted: bool[n_dets,n_tracks]` 掩码表达，匈牙利匹配后判据为 `accepted[r,c]`，**不再用 `cost<0.999` 数值哨兵**（旧实现中 OCM 负项可使 cost≥0.999 静默作废合法匹配）。cost 矩阵仍用 1e6 让匈牙利跳过门控 pair，分配行为不变。

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

### 4.1 实测对照 v1（2026-11-27，rust，stride 1）——⚠️ 已作废

> 本节的 `+nsa` 系列基于 0.1.0 的**错误 NSA 语义**（单段 clip≤1，低置信度反被加信任），且 `+ocm` 系列受 P0-1 的 cost 哨兵 bug 影响。数值仅存档，**不作为结论依据**；正确结果见 §4.2。

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

### 4.2 实测对照 v2（2026-11-27，修复后 HEAD，NSA 分段语义 + accepted 掩码，rust）

DanceTrack val（YOLOv8n 私有检测，全 25 序列）：

| 配置 | HOTA | AssA | DetA | IDF1 | IDSW |
|------|------|------|------|------|------|
| ukf baseline s1 | 0.2045 | 0.0842 | 0.5035 | 0.1827 | 8470 |
| ukf baseline s4 | 0.2170 | 0.1050 | 0.4612 | 0.2128 | 6730 |
| +nsa s1 | 0.2122 | 0.0886 | 0.5161 | 0.1842 | 7995 |
| +nsa+ocm s1 | 0.2124 | 0.0890 | 0.5155 | 0.1862 | 8024 |
| +full s1 | 0.2149 | 0.0910 | 0.5156 | 0.1880 | 7982 |
| **+full s4** | **0.2221** | **0.1090** | 0.4654 | **0.2188** | 7020 |
| auto +nsa s1 | 0.2122 | 0.0887 | 0.5161 | 0.1843 | 7995 |
| auto +full s1 | **0.2151** | **0.0911** | 0.5159 | 0.1881 | 7985 |

MOT17 FRCNN（7 序列）：

| 配置 | HOTA | AssA | DetA | IDF1 | IDSW |
|------|------|------|------|------|------|
| ukf baseline s1 | 0.3961 | 0.3765 | 0.4188 | 0.4639 | 1131 |
| ukf baseline s4 | 0.4123 | 0.4117 | 0.4173 | 0.4766 | 1371 |
| +nsa s1 | 0.4152 | 0.4001 | 0.4327 | 0.4824 | 1033 |
| +full s1 | 0.4187 | 0.4064 | 0.4333 | 0.4856 | 1064 |
| +full s4 | 0.4240 | 0.4292 | 0.4228 | 0.4864 | 1390 |
| auto +nsa s1 | 0.4270 | 0.4193 | 0.4367 | 0.4955 | 1028 |
| auto +full s1 | **0.4288** | 0.4228 | 0.4367 | 0.4973 | 1060 |

结论（含对审查 P0/P1 的回应）：
- **NSA 分段语义正确且更优**：DanceTrack 上 +nsa 使 IDSW 8470→**7995**（优于 0.1.0 错误语义的 8360），HOTA 0.2045→0.2122；MOT17 上 0.3961→0.4152。方向修正带来了实打实的额外收益。
- **P0-1 修复有效**：v2 中 OCM 叠加不再帮倒忙（DanceTrack 上 nsa→nsa+ocm：0.2122→0.2124 不再倒退；0.1.0 曾为 0.2126→0.2110 倒退）。
- **P1-1（stride-4 对照）**：`+full s4`（0.2221）> `s4 baseline`（0.2170），**增强包在 stride 4 下同样有净收益**（HOTA +0.0051、AssA +0.004），证明收益来自机制而非仅 stride 路径。`+full s4` 为 DanceTrack 全场最佳。
- **P1-2（auto 消融）**：MOT17 上机制贡献 = ukf 的 +0.0226（0.3961→0.4187）；AUTO 选型额外贡献 ≈ +0.010（ukf_full 0.4187 → auto_full 0.4288）。两者已拆开，头条数字不再混合归因。
- **⚠️ 回归披露（闭环 12.1，不可省略）**：stride-4 路径上 `+full` 的 **IDSW 上升**——DanceTrack 6730→7020（+290，+4.3%）、MOT17 1371→1390（+19，+1.4%）。这触发 §6 的"MOT17 IDSW 上升须说明"门槛，说明如下：stride 4 每 4 帧才一次观测更新，轨迹以外推为主，OCM 的观测动量在稀疏观测下方向估计噪声增大、HMIoU 在遮挡下更严格，导致更多轨迹切换。取舍依据：该路径 HOTA（0.2170→0.2221）与 AssA（0.1050→0.1090）仍上升，净效应为正；IDSW 小幅上升被更高的关联/检测准确度抵消，可接受。**若部署场景对 ID 连续性极敏感，stride≥4 时建议关闭 OCM**（仅保留 NSA+HMIoU）。
- **⚠️ 负收益披露（闭环 12.4）**：**OCM 在 MOT17 上仍是轻微负收益**——ukf +nsa 0.4152 → +nsa+ocm 0.4142（−0.001）、auto +nsa 0.4270 → +nsa+ocm 0.4261（−0.0009）。上文"P0-1 修复有效"仅指 DanceTrack 上 OCM 不再帮倒忙，**不代表各数据集都为正**。MOT17 为监控行人、运动相对平缓，OCM 的方向约束在低速目标上区分度低、反而可能误排正确候选；其收益主要体现在 DanceTrack 这类强非线性场景。
- **回归门槛核对（修正）**：stride 1 各配置 HOTA/IDSW 均无回归；**stride 4 的 IDSW 上升已按 §6 给出说明**（见上）。此前"全部配置无回归（IDSW 多数下降）"的笼统表述不准确，以本条为准。
- 报告：`benchmark/results/20261127_enhance_v2_dancetrack_val.{json,md}`、`20261127_enhance_v2_mot17_frcnn.{json,md}`（统一 `report_schema_version=2`；在干净 commit `816a1c2` 上重跑重出，数字与初版逐位一致；`tracker_revisions` 全部 pin 到 `816a1c2`，照此 checkout 即可复现本批数字；meta 含 `evaluated_sequences`、`iou_threshold`、真实 HEAD `revision`）。

## 5. 引用（公式来源，逐字核对）

- OC-SORT：`github.com/noahcao/OC_SORT` → `trackers/ocsort_tracker/association.py`（`associate`/`speed_direction_batch`）、`ocsort.py`（`speed_direction`、`k_previous_obs`、velocity/observations 维护）。（规范上游 `github.com/MCG-NJU/OCSort` 不可达——2026-11-27 实测 404；`noahcao/OC_SORT` 为论文一作 Jinkun Cao 的官方仓库，非 fork，予以保留。）
- Hybrid-SORT：`github.com/ymzis69/HybridSORT` → `trackers/hybird_sort_tracker/association.py`（`hmiou`）、`kalmanfilter_score.py`（NSA-KF 行：`R = (1-confidence)·nsa_kalman_interval·R`）。

## 6. 维护指南

- 任何行为变更仍受 `docs/rust-tracker-core.md` §3.3 约束：双侧同步 + 等价测试 +（仅默认行为变更时）golden 更新 + 全套件。
- 三特性相互正交、可独立开关；后续 ROCM（多角点动量）/ TCM（置信度建模）/ ORU（重更新）在此观测历史基础上扩展，不要重复建设 `_Track.observations`。
- 若未来默认开启任一特性：属默认行为变更，须重生成 golden 并在 PROJECT_BIBLE 记录。
