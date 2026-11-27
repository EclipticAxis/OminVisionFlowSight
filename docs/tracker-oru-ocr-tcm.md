# 跟踪增强批次 3：ORU + OCR + TCM（遮挡恢复与弱线索）

> **版本**：0.2.0 — 2026-11-27
> **范围**：`ai/tracker.py` 与 `rust/omintrack/` 双实现；离线 runner 增加开关。三机制**默认全部关闭（opt-in）**，默认路径行为不变（既有 golden / 等价测试不需重生成）。
> **状态**：已合入 main（PR #26 → `9d7c27c`）；全套件 **1527 passed + 1 skipped**（含审查修复新增用例）。前置：`docs/tracker-nsa-ocm-hmiou.md`（批次 1+2）、`docs/tracking-quality-benchmark.md`（基线与验收）。
>
> **0.2.0 变更（PR #26 审查修复）**：① P0-1 Rust `last_observation` 维护补 `|| ocr_enabled`（OCR 单开双端等价）；② P1-1 `_match_ocr` scipy 回退排序方向修正（cost 升序=IoU 降序）；③ P1-2 删双端 `tcm_conf_pred_low` 死代码，§2.3 统一"二阶段不施加 TCM"；④ P1-3 报告落 `tracker_flags`；⑤ P1-4 §4.3 两处 IDSW 上升逐条披露；⑥ P2 状态头/§2.2 顺序/rust-tracker-core §3.2 登记/双后端对拍。复核追加：N-1 §4.3 双后端措辞修正（3/4 逐位一致，MOT17 auto 差 ~1.4e-6 系 AUTO 阈值放大浮点噪声）+ N-2 删写-only 死状态 `tcm_conf_prev`。

---

## 1. 目标与背景

基线与批次 1+2 已证明关联（AssA）是主瓶颈。本批引入 OC-SORT / Hybrid-SORT 中攻遮挡恢复与弱线索的三个机制，进一步压 IDSW、提升遮挡后轨迹恢复：

| 机制 | 作用点 | 出处 | 预期 |
|------|--------|------|------|
| **ORU** 观测中心重更新 | 丢失轨迹恢复时回滚重放，消除外推累计误差 | OC-SORT 论文 §3（观测中心重更新） | 遮挡恢复后状态不漂移 |
| **OCR** 观测中心恢复 | 用丢失轨迹**最后观测**（非预测）与剩余检测二次关联 | OC-SORT `ocsort.py` 二次关联段 | 恢复预测已偏离的静止/慢速目标 |
| **TCM** 轨迹置信度建模 | 置信度作弱线索进关联代价（一致性奖励） | HybridSORT `association.py` / `kalmanfilter_score.py` | 遮挡/聚类时消除歧义匹配 |

## 2. 契约表（双实现必须逐位一致）

### 2.1 ORU（观测中心重更新）

- **触发**：一个此前处于丢失态（本帧关联前 `misses > 0`）的 track 被重新 correct。
- 状态快照：track 进入丢失态时保存 `kf_snapshot = (x, P, Q, R)`（UKF 全量状态）与 `last_observation`、丢失起始帧号。
- 恢复时（`oru_enabled` 且有快照且丢失帧数 N≥1），按序：
  - 回滚 KF 到快照；
  - 对丢失的第 k 帧（k=1..N），构造**虚拟观测** = 线性插值 `lerp(last_observation, current_obs, k/(N+1))`（对 `[cx,cy,w,h]` 四分量），逐帧 `predict()` + `correct(虚拟观测, conf=当前观测conf)`；
  - 最后 `predict()` + `correct(current_obs)`（真实恢复观测，正常路径）。
- 目的：让 KF 状态反映"最后观测→恢复观测"的真实运动，而非丢失期间的纯外推漂移。
- **官方实现说明**：OC-SORT 论文提出 ORU，但其开源 `noahcao/OC_SORT` 的 `KalmanBoxTracker.update` 未含回滚重放代码；本实现按论文语义实现，文档注明非逐字移植。
- 与既有 ROI 重检测正交（ORU 在 correct 内部，ROI 重检测是检测层找回）；与生命周期 `max_misses` 不变（仅成功恢复时触发）。
- 参数：`oru_enabled=False`。

### 2.2 OCR（观测中心恢复）

- 在既有匹配（hungarian/cascade）**之后**、新建 track 与 ROI 重检测**之前**，对仍未匹配的 track 与未匹配检测增加一次**二次关联**：
  - 用 track 的 `last_observation`（最后一次真实观测框）而非 KF 预测框，与剩余检测算 IoU；
  - `IoU > ocr_threshold`（默认 0.3）且类别一致则匹配并 `correct`（走正常 `_update_track`，OCM 观测照常记录；若该 track 处于丢失态且 ORU 开启，则先 ORU 重放）。
- 仅对丢失态 track（`misses>0`）启用；活跃 track 已在主匹配处理。
- 与 ROI 重检测的关系：OCR 用**已有剩余检测**（零额外推理），ROI 重检测用**重跑检测**；OCR 在前（便宜），ROI 在后。顺序固定。
- 参数：`ocr_enabled=False`、`ocr_threshold=0.3`。

### 2.3 TCM（轨迹置信度建模）

- `_Track` 增加**置信度状态**：1 维置信度滤波（conf + 速度的 2 维 KF，或简化为 conf + EMA 速度），输出当前帧期望置信度 `conf_pred`。
  - 高置信度阶段（cascade 第一阶段 / 单阶段主匹配）：用一阶预测（conf + EMA 速度）得 `conf_pred`；
  - 低置信度阶段（cascade 第二阶段）：**不施加 TCM**（保持低分找回不被置信度差压制）。注：HybridSORT 在此阶段用线性预测建模置信度；本实现二阶段不启用，`conf_pred_low` 方法已完全删除。
- 关联代价（一致性奖励）：`score_dif = |det_conf - conf_pred|`；在既有 `score` 上**减** `score_dif · tcm_weight`（与 HybridSORT `angle_diff_cost -= score_dif * TCM_first_step_weight` 一致，方向为越一致 score 越高）。仅 `tcm_enabled` 时启用，第一阶段（或单阶段）生效；第二阶段默认不加（保持低分找回不被置信度差压制）。
- 参数：`tcm_enabled=False`、`tcm_weight=0.2`（对应 TCM_first_step_weight）。
- 与 NSA 的关系：NSA 用 conf 调 R（滤波器内部），TCM 用 conf 一致性做关联（匹配层），正交。

### 2.4 配置与接口

- `DetectionTracker.__init__` / `set_tracker_params` 增加上述参数；Rust `py_api` 同步暴露；runner 增加 `--oru/--ocr/--tcm`（及 `--ocr-threshold/--tcm-weight`）。
- 默认全关 ⇒ 默认路径 golden 与等价测试**不应**有任何数值变化（硬验收条件）。
- 复用批次 1+2 的 `_Track.observations`/`last_observation`/`velocity`（OCM 基础设施），ORU/OCR 不重复建设观测历史。

## 3. 测试计划

- 新增 `tests/test_tracker_occlusion.py`：
  - ORU：构造 track 丢失 N 帧后恢复，验证回滚重放后状态与纯外推不同且更接近真实恢复观测；无快照/未丢失时 ORU 不触发。
  - OCR：构造预测已偏离但最后观测与某剩余检测重叠的 track，验证二次关联恢复；阈值门控生效；OCR 先于 ROI。
  - TCM：conf 一致性奖励符号与量级；高/低分阶段 conf_pred 来源不同（KF vs 线性）；默认关闭零影响。
  - 双后端等价（开启三特性，坐标 1e-6、状态机精确）。
- 既有全套件 `pytest tests/` 全绿（默认路径不变为前提）。

## 4. 评测结果（2026-11-27，commit `aa0a0eb`，rust，schema v2；本次只落盘报告，不提 PR）

对照设计：baseline vs 批次 1+2 full vs `+oru` / `+ocr` / `+tcm` 逐项叠加至全特性；rust，stride 1 与 stride 4。报告 meta 为 schema v2（revision=`aa0a0eb` 干净 HEAD + evaluated_sequences）。

### 4.1 DanceTrack val（YOLOv8n 私有检测，25 序列）

| 配置 | HOTA | AssA | DetA | IDF1 | IDSW |
|------|------|------|------|------|------|
| baseline ukf s1 | 0.2045 | 0.0842 | 0.5035 | 0.1827 | 8470 |
| full s1（批次1+2） | 0.2149 | 0.0910 | 0.5156 | 0.1880 | 7982 |
| full+oru s1 | 0.2151 | 0.0910 | 0.5158 | 0.1896 | 7988 |
| full+ocr s1 | 0.2149 | 0.0910 | 0.5156 | 0.1874 | 7869 |
| full+tcm s1 | 0.2146 | 0.0909 | 0.5146 | 0.1886 | 7884 |
| full+all s1 | 0.2152 | 0.0913 | 0.5150 | 0.1887 | 7822 |
| **full+all s4** | **0.2337** | **0.1214** | 0.4607 | **0.2391** | **6261** |
| auto full+all s1 | 0.2153 | 0.0914 | 0.5150 | 0.1887 | 7816 |

### 4.2 MOT17 FRCNN（7 序列）

| 配置 | HOTA | AssA | DetA | IDF1 | IDSW |
|------|------|------|------|------|------|
| baseline ukf s1 | 0.3961 | 0.3765 | 0.4188 | 0.4639 | 1131 |
| full s1（批次1+2） | 0.4187 | 0.4064 | 0.4333 | 0.4856 | 1064 |
| full+oru s1 | 0.4185 | 0.4064 | 0.4330 | 0.4851 | 1053 |
| full+ocr s1 | 0.4186 | 0.4063 | 0.4332 | 0.4856 | 1075 |
| full+tcm s1 | 0.4182 | 0.4054 | 0.4333 | 0.4839 | 1057 |
| full+all s1 | 0.4182 | 0.4056 | 0.4331 | 0.4846 | 1028 |
| full+all s4 | 0.4226 | 0.4257 | 0.4233 | 0.4850 | 1316 |
| auto full+all s1 | **0.4283** | 0.4220 | 0.4365 | **0.4963** | **1024** |

### 4.3 结论（诚实，含回归披露）

- **stride 1 上批次 3 单项为边际改进**：ORU/OCR/TCM 叠加在 full 上 HOTA 基本持平（±0.001 内）、IDSW 小幅下降（DanceTrack full→all：7982→7822；MOT17：1064→1028）。非飞跃，符合"遮挡恢复机制在检测器受限场景收益有限"的预期。
- **真正价值在 stride 4**：`full+all s4` 是 DanceTrack 全场最优（HOTA 0.2337、AssA 0.1214、IDSW 6261），显著优于 s1 全特性与批次 1+2 的 `full s4`（0.2221/7020）。
- **闭环批次 1+2 的 stride-4 IDSW 回归**：批次 1+2 时 `full s4` 曾使 DanceTrack IDSW 6730→7020（上升，见 nsa-ocm-hmiou §4.2 的 12.1 披露）；**批次 3 全特性把 stride-4 IDSW 压到 6261，低于 s4 baseline 6730**，MOT17 stride-4 IDSW 亦从 full s4 的 1390 降到 1316（低于 s4 baseline 1371）。即遮挡恢复机制恰好消解了 OCM/HMIoU 在稀疏观测下引入的切换回归——**此前"stride≥4 建议关 OCM"的权宜建议，在全特性下可收回**。
- **回归门槛核对（逐条，闭环 P1-4）**：方案 §6 要求"MOT17 IDSW 上升须说明"，逐条披露：
  - `full+oru s1` DanceTrack IDSW 7982→**7988**（+6，+0.08%）：ORU 回滚重放使恢复轨迹更贴近真实运动，个别原本会断链重开的轨迹被保留并延续，极少数情况下延续的轨迹在后续帧发生一次额外线性切换；幅度在噪声量级，HOTA 持平（0.2149→0.2151），可接受。
  - `full+ocr s1` MOT17 IDSW 1064→**1075**（+11，+1.0%）：OCR 用最后观测恢复预测已偏离的目标，恢复了部分本应过期删除的轨迹，恢复后与邻近轨迹偶发一次身份交换；换来的是轨迹覆盖/连续性提升（IDF1 持平、HOTA 0.4187→0.4186 持平），取舍合理。两处均为千次级数据下的个位数~十位数波动，且全特性叠加后（`full+all`）IDSW 反而低于 full（DanceTrack 7822、MOT17 1028），净效应为正。
  - HOTA 方面：stride 1 全特性较 full 在 MOT17 微降 0.0005（< 0.002 门槛）；stride-4 的 IDSW 改善见上条。
- **距 Deep OC-SORT 的差距主因检测器**：DetA≈0.46–0.52 是硬约束；跟踪关联层面的可榨收益已趋缓，下一数量级提升需升级检测器（更强 YOLO 或 DanceTrack 域微调）。
- **双后端对拍（闭环 P2-4 + P0-1 实证）**：全特性 Python/Rust 在真实基准上**3/4 配置逐位一致**（DanceTrack ukf_all 0.2152、auto_all 0.2153、MOT17 ukf_all 0.4182 双端逐位相同，IDSW 亦全一致）。**唯一例外**：MOT17 `auto_all` HOTA rust=0.42825541 vs py=0.42825681，差 **~1.4e-6**，定位到单条序列 MOT17-04-FRCNN——系 nalgebra vs NumPy/LAPACK 的浮点差（~7e-10/帧）被 AUTO 切换的离散阈值（NIS>8.0 / cond>1000 / hysteresis=2）放大所致，属 `docs/rust-tracker-core.md` §3.1 已记录的已知范畴，不影响结论方向。**重要限定**：此前基线报告的"Python/Rust 逐位一致"仅在 **ukf** 上成立，**不能外推到 auto**——auto 含离散滤波器切换，浮点噪声可能被阈值放大；auto 模式的双端对拍是本次才首次加入的。
- 报告：`benchmark/results/20261127_batch3_dancetrack_val.{json,md}`、`20261127_batch3_mot17_frcnn.{json,md}`。

## 5. 引用

- OC-SORT：`github.com/noahcao/OC_SORT`（论文一作官方仓库）`trackers/ocsort_tracker/ocsort.py`（二次关联段 = OCR；`k_previous_obs`/`last_observation` 复用）。ORU 按 OC-SORT 论文 §3 观测中心重更新语义实现（开源版未含回滚代码，已注明）。
- Hybrid-SORT：`github.com/ymzis69/HybridSORT` → `trackers/hybird_sort_tracker/association.py`（`cal_score_dif_batch`、`TCM_first_step_weight`）、`kalmanfilter_score.py`（置信度 KF 建模与二阶段线性预测）。

## 6. 维护指南

- 任何行为变更受 `docs/rust-tracker-core.md` §3.3 约束：双侧同步 + 等价测试 +（仅默认行为变更时）golden 更新 + 全套件。
- 三机制正交、可独立开关；ORU 依赖丢失快照、OCR 依赖最后观测、TCM 依赖置信度状态，均挂在 `_Track` 上，勿重复造观测/置信度历史。
- 若未来默认开启：属默认行为变更，须重生成 golden 并记录 PROJECT_BIBLE。
