# 跟踪质量量化基准：MOT17 / MOT20 / DanceTrack

> **版本**：0.2.3 — 2026-11-27
> **范围**：批次 0 评测基建的基准选型与协议定型；约束后续所有滤波器/关联层改造（NSA-KF、OCM、HMIoU、TCM、ORU 等）的量化验收方式。不涉及跟踪器行为变更。
> **状态**：M0 主体完成——基建重建（34 passed + 1 skipped，含 TrackEval 端到端合成用例），DanceTrack val 与 MOT17 FRCNN 基线均已落盘（§6.1），双后端真实基准逐位一致；MOT20 暂未下载（泛化抽查可后补）。
>
> **0.2.0 变更**：数据集根目录定为 `G:\基准测试`（用户指定，脚本参数化）；MOTChallenge 网站已于 2026-04-16 转为静态存档、**提交评测通道关闭**，MOT17/20 test 改为"存档 leaderboard 横向参照"（§2、§7）；DanceTrack 官方源为 HF `noahcao/dancetrack`，本机 huggingface.co 不可达，经 hf-mirror.com 下载且**仅需 val.zip**（§2、§5.2）。
>
> **0.2.1 变更（PR #21 审计 P0-3 + P1-5）**：① 回归门槛统一为 HOTA 绝对回退 **0.002**（§6，取代 0.2 笔误）；② 评测口径护栏——`evaluate_trackeval.py` 期望序列默认 GT 全量（或 `--seqs`），tracker 结果缺序列**立即报错并列出**（不再静默交集收窄），报告 meta 写入 `evaluated_sequences`，`ensure_seqmap` 缺失才生成、不一致默认报错（仅 `--allow-seqmap-rewrite` 才覆写并记录 `seqmap_rewritten`）。
>
> **0.2.2 变更（PR#22 复核 15.1–15.3）**：① `git_revision` 加工作区脏指纹（代码路径、排除产物），产出代码可复现锚点；② 报告加 `report_schema_version=2` 并明确 §4 字段定义（revision=执行时 HEAD / tracker_revisions=各 run 产出代码）；③ `iou_threshold` 收为 per-tracker 集合 + 不一致告警。全部落盘报告在干净 commit `816a1c2` 上重跑重出为 schema v2，数字与初版逐位一致。
>
> **0.2.3 变更（PR#26 复核 N-1）**：§6.1"逐位一致"结论加限定——仅在 ukf 上成立，auto 模式的 AUTO 离散切换可能放大浮点噪声（批次 3 对拍实测 MOT17 auto HOTA 双端差 ~1.4e-6）。

---

## 1. 目标与背景

跟踪器改造路线图（滤波器层 → 关联层 → 遮挡恢复层）要求每一步都有可复现的量化对照，禁止"凭观感调参"。本文定型三件事：

1. **基准数据集**：MOT17 / MOT20 / DanceTrack（公开通用基准，可与 SORT 系文献数字对照）。
2. **检测协议**：私有检测端到端（纵向对照）与公开检测（横向对照）分开报告，不混用。
3. **验收规则**：基线先行、逐批次对照、回归门槛明确。

对标对象（Deep OC-SORT / Hybrid-SORT）的论文数字基于 YOLOX 私有检测，**与本项目 YOLOv8n 端到端结果不直接可比**；横向对照仅在 MOT17 公开检测协议下成立，且必须标注检测器差异。

## 2. 基准选型与角色分工

| 数据集 | Split | 本地可评 | 角色 | 选择理由 |
|--------|-------|----------|------|----------|
| **DanceTrack** | val（GT 公开） | ✅ | **主迭代基准** | 非线性运动 + 相似外观 + 遮挡，对 OCM / TCM / 方向代价最敏感；Hybrid-SORT 增益最大的数据集，最适合检验关联层改造收益 |
| **MOT17** | train（纵向对照用全序列或固定 train/val 半分） | ✅ | **主场景基准** | 监控行人场景与本项目最贴近；支持 DPM/FRCNN/SDP 公开检测，衔接 PR #18/#19 的 FRCNN 历史基线 |
| **MOT17** | test | ❌（GT 不公开，需提交 MOTChallenge 服务器） | 里程碑验证 | 仅阶段性提交，不作日常迭代 |
| **MOT20** | train | ✅（仅 4 序列，统计噪声大） | 泛化抽查 | 密集人群遮挡场景；阶段性检查，不作合入硬门槛 |
| **MOT20** | test | ❌（同 MOT17 test） | 里程碑验证 | 同上 |

约定：
- DanceTrack test 同样需服务器提交，日常一律用 **val**。
- MOT17 train 的纵向对照协议一经选定（全序列 vs 半分）即冻结写入基线报告，禁止中途更换口径制造"提升"。
- **2026-11-27 修订**：MOTChallenge 已停止运营（静态存档），MOT17/20 test 的服务器提交不再可行；test 行保留仅作语义说明，实际横向对照引用存档 leaderboard（2026-04-16 存档状态）与公开结果 CSV，不再安排提交动作。
- 数据集根目录：`G:\基准测试\`（`MOTChallenge\MOT17|20\`、`DanceTrack\`、`_downloads\` 临时包）；全部脚本以 `--root` 参数接收，禁止硬编码。

## 3. 检测协议（两条线，分开报告）

| 协议 | 用途 | 检测来源 | 可比性 |
|------|------|----------|--------|
| **私有检测端到端（主）** | 改造前后纵向对照 | 项目现行 YOLO 后端逐帧检测 | 仅纵向有效；与论文数字不可直接比 |
| **公开检测（MOT17 对照）** | 与 SORT 系公开数字粗略横向对照 | MOT17 官方 FRCNN 检测 | 横向参考；须标注检测器与论文 YOLOX 不同 |

硬性规则：**检测缓存复用**。私有检测一次性生成并缓存为 MOT det 格式 txt，后续所有跟踪实验复用同一缓存，消除检测抖动对跟踪对照的污染。检测器本身变更时缓存版本号递增，旧基线作废并重测。

## 4. 指标与报告契约

- **主指标 HOTA**（含 AssA / DetA 分解）；**IDF1 + IDSW** 看关联质量；MOTA 仅作参考（受检测影响大）。
- 每份报告必须标注：数据集/split、检测协议与缓存版本、`infer_stride`、滤波器类型（ukf/mcukf/manifold_ukf/auto）、后端（python/rust）、代码 revision。
- **stride 矩阵**：stride 1 必测；stride 4 抽测（跳帧预测是本项目特色路径，禁止只测 stride 1）。
- **双后端**：python / rust 各跑一次，按 `docs/rust-tracker-core.md` §3.1 数值契约核对一致（状态机精确、坐标 1e-6）；Rust 侧需先 `maturin develop --release` 重建。
- 报告落盘：`benchmark/results/<date>_<revision>_<tag>.json`（机读）+ 同名 `.md`（人读摘要），入库；数据集与检测缓存不入库。

**报告字段定义（`report_schema_version=2`，2026-11-27 起）**：

| 字段 | 含义 |
|------|------|
| `report_schema_version` | 报告格式版本。v1=初版（revision=产出 tracker 结果代码、无下述溯源字段）；**v2=现行** |
| `revision` | **评测执行时的代码 HEAD**（`git rev-parse HEAD` + 脏指纹）。代码路径（`ai`/`rust`/`benchmark`，排除 `results` 等产物）有未提交改动时追加 `-dirty` |
| `tracker_revisions` | 每个 tracker 各自 summary 记录的 revision——**该 run 产出跟踪结果时的代码**（含脏指纹），用于逐 run 复现 |
| `iou_threshold` / `iou_threshold_by_tracker` | 关联 IoU 门控阈值；单值为各 tracker 一致时的值，多 tracker 不一致时主字段置 `null` 并告警、以 per-tracker 集合为准 |
| `evaluated_sequences` | 实际参与评测的序列排序列表（口径可审计，防静默收窄） |
| `trackeval_version` | 经 `importlib.metadata` 获取的 TrackEval 版本 |

> 复现口径：照某 run 的 `tracker_revisions[tag]` checkout（须非 `-dirty`）重跑跟踪即可复现该 run 数字；`revision` 仅标识报告生成时刻。2026-11-27 已将 M0 基线与 v2 增强全部报告统一为重出为 schema v2（在干净 commit `816a1c2` 上重跑，数字与初版逐位一致）。

## 5. 评测基建（重建）

### 5.1 历史现状

`benchmark/` 曾存在 `evaluate_mot17_trackeval`、`prepare_mot17_public_data`、`mot17_tracking_experiment`、`mot17_third_party_baselines`、`mot_quality_e2e` 等脚本（现仅存 `__pycache__` 编译缓存），源文件已在清理中丢失。**不反编译 pyc**，按 TrackEval 标准结构重写，避免继承不可审计的逻辑。

### 5.2 组成（已实现）

| 组件 | 职责 |
|------|------|
| `benchmark/prepare_mot_data.py` | 数据集目录校验/下载指引、MOT17 公开检测整理、DanceTrack val 转换 |
| `benchmark/run_detection_cache.py` | 私有检测离线生成 det 缓存（版本化） |
| `benchmark/run_tracking_offline.py` | 离线 runner：逐帧读取 det → 直接驱动 `DetectionTracker.update()`（无 GUI/摄像头依赖），输出 MOT 格式结果 |
| `benchmark/evaluate_trackeval.py` | 调用 TrackEval 计算 HOTA/IDF1/MOTA/IDSW，汇总报告 |
| `benchmark/results/` | 基线与对照报告（入库） |

目录约定：数据集放 `data/MOTChallenge/`、`data/DanceTrack/`（git 排除）；TrackEval 作为外部工具依赖安装（pip 或子目录，不 vendored 魔改）。

实际落地（0.2.0）：数据根 `G:\基准测试\`（仓库外，天然不入库）；TrackEval 已安装为 git 版（`pip install git+https://github.com/JonathonLuiten/TrackEval.git`）。下载源：MOT17/MOT20 为 motchallenge.net 官方静态 zip；DanceTrack 为 HF `noahcao/dancetrack` 经 `hf-mirror.com`，M0 只需 `val.zip`（约 4.2GB），train/test 分包（约 13GB）暂不下载。

## 6. 基线与验收规则

1. **基线先行**：任何跟踪改造合入前，当前 main 必须在 DanceTrack val + MOT17 train（私有检测 + FRCNN 公开检测两协议）留有基线报告。
2. **逐批次对照**：批次 1（滤波器层）/ 批次 2（关联层）/ 批次 3（遮挡恢复）各自合入时附前后对照表；改善必须引用具体指标，禁止只写"效果更好"。
3. **回归门槛**：DanceTrack val HOTA 绝对回退 > 0.002（即 0.2 个百分点）、或 MOT17 IDSW 上升，必须在 PR 中给出解释与取舍依据；无说明不得合入。（0.2.1 修订：与 tracker-nsa-ocm-hmiou §4 对齐，统一为绝对值 0.002，取代早期 0.2 的笔误——基线 HOTA 仅 0.2~0.4，0.2 门槛等于无约束。）
4. **困难基线诚实披露**：MOTA 为负或极低的序列/配置，对照时禁止只引用单侧数字（沿用 mcukf-irls-correction §4 的披露惯例）。

### 6.1 基线落盘记录（2026-11-27，revision 32788480）

DanceTrack val / YOLOv8n 私有检测（det_yolov8n_v1，conf=0.25，imgsz=640，GPU）/ 全 25 序列：

| 配置 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|------|------|------|------|------|------|------|
| ukf / python / s1 | 0.2045 | 0.0842 | 0.5035 | 0.1827 | 0.5485 | 8470 |
| ukf / rust / s1 | 0.2045 | 0.0842 | 0.5035 | 0.1827 | 0.5485 | 8470 |
| auto / rust / s1 | 0.2046 | 0.0842 | 0.5040 | 0.1827 | 0.5485 | 8476 |
| ukf / rust / s4 | 0.2170 | 0.1050 | 0.4612 | 0.2128 | 0.4862 | 6730 |

结论：① Python/Rust 指标逐位一致，双后端等价在真实基准上成立——**限定：该"逐位一致"仅在 ukf 上成立；auto 模式含离散滤波器切换，浮点噪声可能被 AUTO 阈值放大**（批次 3 双后端对拍实测：MOT17 auto HOTA 双端差 ~1.4e-6，定位 MOT17-04，属 §3.1 已知范畴）；② **AssA≈0.08 是主要瓶颈**（关联质量极差，IDSW 高达 8470），印证"差距在关联层而非滤波器"的诊断；③ AUTO 在此基准上与 UKF 几乎无差别；④ stride 4 反而 HOTA/IDSW 略优（高检测噪声下减少误关联），DetA 略降。报告文件：`benchmark/results/20261127_baseline_dancetrack_val_*.json/.md`。

MOT17 train / FRCNN 公开检测 / 7 序列（02/04/05/09/10/11/13）：

| 配置 | HOTA | AssA | DetA | IDF1 | MOTA | IDSW |
|------|------|------|------|------|------|------|
| ukf / python / s1 | 0.3961 | 0.3765 | 0.4188 | 0.4639 | 0.4524 | 1131 |
| ukf / rust / s1 | 0.3961 | 0.3765 | 0.4188 | 0.4639 | 0.4524 | 1131 |
| auto / rust / s1 | 0.3984 | 0.3807 | 0.4188 | 0.4668 | 0.4516 | 1128 |
| ukf / rust / s4 | 0.4123 | 0.4117 | 0.4173 | 0.4766 | 0.4473 | 1371 |

结论：DetA≈0.42 受 FRCNN 公开检测与 miss-不输出策略共同限制；AssA≈0.38 仍显著低于 Deep OC-SORT 量级（0.6+），关联层是主改造对象；stride 4 的 HOTA 更高但 IDSW 更多（跳帧外推漂移与关联频率的权衡，后续 ORU/方向代价的直接靶点）。报告文件：`benchmark/results/20261127_baseline_mot17_frcnn.json/.md`。

## 7. 里程碑

| 里程碑 | 内容 | 验收 |
|--------|------|------|
| M0 | 基建重建 + 当前 main 基线落盘（DanceTrack val、MOT17 train 双协议；MOT20 train 抽查） | 报告入库，双后端一致 |
| M1 | 批次 1（NSA-KF / dt / EMA 解耦）后对照 | §6 规则 |
| M2 | 批次 2（马氏门控 + OCM + HMIoU）后对照 | §6 规则 |
| M3 | 批次 3（ORU / OCR / TCM）后对照 + 与 MOT17 存档 leaderboard 同协议条目横向参照 | §6 规则 + 参照条目存档 |

## 8. 维护指南

- 数据集、det 缓存、TrackEval 输出临时目录一律 git 排除；仅 `benchmark/results/` 报告入库。
- 协议口径（split、检测缓存版本、stride 矩阵）变更属破坏性变更：需更新本文版本号并重测基线。
- 离线 runner 复用 `DetectionTracker` 公开接口，不得为评测在跟踪器内加评测专用分支。
- 与 `docs/rust-tracker-core.md` §3.3 修改契约叠加适用：任何跟踪行为变更仍须双实现同步 + 等价测试 + golden 更新 + 全套件通过。
