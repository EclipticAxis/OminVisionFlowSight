# InferWorker Legacy 行为基线与 Pipeline 对拍契约

> **版本**：0.4.0（2026-11-20）
> **范围**：`tests/test_inferworker_pipeline_contract.py`、`visioncore/pipeline/adapters.py`、`ai/inference.py`（InferWorker）、`tests/test_inferworker_migration.py`、`tests/test_inferworker_resilience.py`、`docs/inferworker-pipeline-parity.md`（本文件）、`PROJECT_BIBLE.md`（§11.2 / §12.2 同步）；不改 `ai/tracker.py`、`rust/`
> **状态**：待合并；Legacy ↔ Pipeline 双模式对拍 22 项（11 scenario × 2 模式）全部转绿，run 级 TargetManager 单写证明 5 项 + 双槽位 slot 信号契约 2 项 + Pipeline runtime 复用与生命周期收口 11 项通过（对拍专项命令 137 项、含 Pipeline/core 扩展命令 168 项、全套件 1336 项通过）

## 目标与背景

VisionCore Pipeline（C8）正在逐里程碑迁移 Legacy 推理路径。迁移的前提是可证伪的
**行为等价定义**：先把生产 `ai.inference.InferWorker` 的 Legacy 行为固化为基线
测试，再让 Pipeline 模式在同一 scenario 下对拍转绿。

- **第 3 步（0.1.0）**：建立 Legacy 行为基线（8 项）。
- **第 4 步（0.2.0）**：核心 Pipeline 接缝对齐——schema 统一、扩展字段
  无损往返、redetect 回调对齐、TargetManager 单写——同一 `LegacyScenario`
  实例驱动 False/True 双模式，同一断言双方转绿。
- **第 4 步（0.3.0）**：抽取并复用共享后处理——Legacy 命中帧与
  `_run_pipeline_mode` 的"头部属性 → 身体手势 → 矩形 → 手部 →
  dedupe/display → 健康回退/恢复 → source/latency"内联链合并为单一私有方法
  `_postprocess_slot_detections`，两侧调用同一实现；新增 run 级双槽位
  `detection_ready` slot_id 信号契约（真实生产 QThread，非 mock 帧处理）。

- **第 4 步（0.4.0，本版）**：Pipeline 实例复用与生命周期收口——
  `_run_pipeline_mode` 从"每帧新建 Pipeline + 4 adapters 并
  initialize/shutdown"改为每 slot 懒创建并复用一个已 initialize 的
  runtime（`_SlotPipelineRuntime`）；补齐 redetect 回调逐帧重设、异常
  runtime 淘汰重建、运行中 unregister 的 retirement 收口、stop 批量
  清理契约与 11 项精确生命周期测试（spy/计数，无长 sleep）。

**明确边界**：本契约是 `_process_slot_frame` 级单元契约 + run 级单写契约，
**不是**摄像头采集→推理→显示的端到端集成测试（该缺口仍记录在
PROJECT_BIBLE §12.2）。不涉及 GUI、不启用 Pipeline 默认开关、不接入
orphan `inference_controller`。

## 测试架构

```
tests/test_inferworker_pipeline_contract.py
├── LegacyScenario          # 数据驱动 scenario（对拍复用入口）
│     raw_detections        #   backend._run_inference 输出（生产 schema）
│     tracker_output        #   tracker.update 输出
│     predicted_output      #   tracker.predict_step 输出（跳帧）
│     redetect_result / head_attrs / body_gestures
│     rectangle_raw / hand_raw / call_redetect
├── fakes（记录事件顺序 + 返回 scenario 数据）
│     FakeBackend / FakeTracker / FakeRectangleTracker
│     FakeHeadClassifier / FakeGestureRecognizer
│     FakeHandGestureRecognizer / FakeRectangleDetector / FakeHealthMonitor
└── build_worker(monkeypatch, scenario, ..., pipeline_enabled=False)
      真实 InferWorker（_initialize_backend 被禁，backend=cpu）
      + worker._model = object()（满足适配器 health_check，不加载模型）
      + 组件边界 fake + _debug_emit_reports=20（禁 debug HTTP）
      + pipeline_enabled=True 时走 _run_pipeline_mode 对拍路径
```

关键约束：

- **被测方法绝不 mock**：直接调用真实 `InferWorker._process_slot_frame`；
  只替换组件边界（模型推理、tracker、后处理器、health monitor、ROI 重检测）。
- **无 QThread / 无模型 / 无摄像头 / 无网络**：不 `start()` 线程，
  `monkeypatch` 禁掉 `_initialize_backend`；事件顺序通过共享 `events` 列表记录。
- **生产 schema**：全部检测 dict 使用 `x1/y1/x2/y2` + `confidence` / `label` /
  `class_id`（与 `Detection.to_dict()` / `DetectionTracker` 一致），
  **不含 `bbox` 键**；对拍双方必须同 schema。
- `final_dedupe`（模块级纯函数）与 `_denoise_frame` 仅包一层记录器，
  内部仍走生产实现。
- **双模式**：8 个 Legacy scenario 全部参数化为
  `@pytest.mark.parametrize("pipeline_enabled", [False, True])`，
  **同一断言**在两种模式下都必须通过；禁止削弱 Legacy 断言、禁止 xfail、
  禁止把输出改成 bbox。

## 契约

| # | 契约 | 基线/对拍测试 | 修改契约 |
|---|------|----------|----------|
| 1 | 命中帧处理顺序：denoise（可禁用）→ inference → tracker.update → update_appearance → head attributes → body gestures → rectangle → hand gestures → dedupe/display → health → source 标记 → `_last_detections` 缓存（与返回值为同一列表对象）；Pipeline 模式经 DetectorStage→TrackerStage→TargetStage→EventStage + 同一后处理链，**事件顺序与 Legacy 完全一致** | `test_hit_frame_full_feature_order` / `test_hit_frame_minimal_order_and_cache`（×2 模式） | 增删任一帧处理步骤必须同步顺序断言与本文档 |
| 2 | `redetect_enabled=True`：tracker 收到**非 None** `redetect_callback`；回调为 `InferWorker._make_redetect_callback(frame)` 工厂产物（Legacy 内联路径与 `InferWorkerTrackerAdapter` 共用同一单点），以（预测框, 原置信度）签名委托 `_redetect_person_in_roi`，帧对象与主推理输入为同一 ndarray；attempt/success/failed 计数精确（成功 +1/+1/0，失败 +1/0/+1） | `test_redetect_enabled_counts_success` / `test_redetect_enabled_counts_failure`（×2 模式） | 回调工厂签名或计数语义变更必须同步三个计数断言 |
| 3 | `redetect_enabled=False`：tracker 收到 `None` callback，**调用方式为仅传 frame**（`tracker.update(dicts, frame=...)`，与原约定一致）；redetect 计数保持 0 | `test_redetect_disabled_passes_none`（×2 模式） | 禁止在 disabled 路径注入任何回调；禁止改用后置 RedetectStage 代替 tracker 内回调 |
| 4 | 跳帧路径（`frame_id % stride != 0`）：不调用推理 / tracker.update / health / dedupe；输出 = `predict_step()` 的 person + 上次非 person（同一 dict 对象复用）；结果缓存为 `_last_detections`；**Pipeline 开启时与 Legacy 逐字段一致** | `test_skipped_frame_uses_prediction_and_last_non_person`（×2 模式） | 跳帧不得引入新的副作用步骤 |
| 5 | 扩展字段保留：`keypoints`、`polygon`（OBB）、`head_attrs`、`gestures`、`track_id`、`track_state` 等 bbox/confidence/class_id/label 之外的字段在最终输出中原样存在；dict↔core 往返**无损**（见"扩展字段无损往返"节） | `test_extension_fields_preserved`（×2 模式）+ `test_inferworker_migration.py` 往返专项 | 新增扩展字段必须在对拍与往返专项中同步验证 |
| 6 | 生产 schema：所有输出检测含 `x1/y1/x2/y2`，**不含 `bbox` 键**；适配器输入以 `x1/y1/x2/y2` 为准，`bbox` 旧键仅作可选兼容输入且绝不泄漏到输出 | `test_extension_fields_preserved`（×2 模式）+ 迁移测试 schema 专项 | 禁止让 `bbox` 混入输出 schema；禁止 bbox-only 假件 |
| 7 | source/latency 标记：每个输出检测含 `source_frame_id == frame_id`、`source_ts == submitted_ts`（精确相等）、`latency_ms`（同帧所有检测共享同一值，落在调用前后单调时钟区间内）；health 监控在 source 标记**之前**执行 | `test_source_mark_and_latency_contract`（×2 模式） | 标记时序或字段名变更必须同步 |
| 8 | **TargetManager 单写**：Pipeline 模式仅当 TargetStage 确实完成 `update_targets`（适配器 `updated` 标志）时，run() 尾部跳过 `_bypass_update_targets`；Legacy、跳帧、Pipeline 在 TargetStage 前失败并返回 `[]` 的路径仍执行**一次** bypass，保留过期清理语义；**禁止按 `pipeline_enabled` 盲跳** | `tests/test_inferworker_resilience.py` 5 项 run 级证明（成功一次 / 前置失败一次 / TargetStage 后失败仍一次 / Legacy 一次 / 后处理异常后下一 Legacy 帧仍一次） | 单写判定必须基于"该帧 update 是否完成"，不得基于模式开关；每帧开始必须清除上次标志 |
| 9 | **run 级 slot 信号契约**：真实生产 `InferWorker` QThread 下双槽位各提交帧，`detection_ready` 首参数必须为**提交槽位 slot_id**（不是 frame_id——每槽位首帧 frame_id 均为 1，槽位 0 首轮可判别）；结果内容与提交槽位严格一一对应（track_id / 帧内容双向核对），Legacy 同契约不回归；确定性有界等待、及时 stop | `test_detection_ready_slot_id_matches_submitting_slot`（×2 模式，run 级） | 信号签名或 emit 首参数语义变更必须同步该 run 级契约 |

## 第 4 步接缝对齐设计

### 生产 schema 统一（契约 #6）

`visioncore/pipeline/adapters.py` 的 dict↔core 转换以生产 schema 为准：

- 输入：`_extract_corners()` 优先取 `x1/y1/x2/y2`，旧 `bbox` 元组键仅作
  向后兼容 fallback（`KeyError` 仅在两者都缺失时抛出）。
- 输出：`_core_detection_to_dict()` 只产出
  `x1/y1/x2/y2/confidence/class_id/label` + 恢复的扩展字段，`bbox` 键绝不出现。
- `tests/test_inferworker_migration.py` 全部假件（`_MockDetectorBackend`、
  `_raise_process`、结构断言）已从失真的 bbox-only 改为生产 schema。

### 扩展字段无损往返（契约 #5）

`CoreDetection` 基类**不改**（四字段契约由
`tests/test_core_models.py::test_detection_has_four_fields` 锁定）。适配器内部
定义局部载体：

```python
@dataclass(frozen=True, slots=True, eq=False)
class _AdaptedDetection(CoreDetection):
    extras: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
```

- 入向：所有非基础字段（`track_id`/`track_state`/`keypoints`/`polygon`/
  `gesture`/`head_attrs`/...）**深拷贝**进 `extras`，另存私有键
  `__corners__`（原始角点）与 `__source_bbox__`（摄入时中心/宽高）。
- 出向：先按当前 core bbox 派生角点；若 `(b.x, b.y, b.w, b.h)` 与
  `__source_bbox__` 完全一致（未被任何 stage 修改），则回放 `__corners__`
  快照——角点值**位级精确**还原（角点→中心/宽高→角点本身不是 float 精确的，
  必须用快照）；若 core 基础字段被修改，则派生值映射回输出，extras 仍保留。
- 私有 `__*` 键永不泄漏到输出 dict；`extras` 不参与
  eq/hash（手写 `__eq__`/`__hash__`，对任意 `CoreDetection` 按四核心字段比较）。
- 无全局 id map、无 `ai` import 副作用（adapters 保持单向依赖）。

### redetect 对齐（契约 #2/#3）

- `InferWorker._make_redetect_callback(frame)` 为唯一回调工厂，Legacy 内联
  路径与 `InferWorkerTrackerAdapter` 共用；计数维护（attempt/success/failed）
  只有这一处实现。
- `InferWorkerTrackerAdapter.set_redetect_callback()`：启用时
  `tracker.update(dicts, redetect_callback=cb, frame=...)`；禁用时
  `tracker.update(dicts, frame=...)`——与 Legacy disabled 路径调用方式一致，
  兼容不接受 `redetect_callback` 形参的简单 fake tracker。
- 不接语义不同的后置 RedetectStage 代替 tracker 内回调。

### TargetManager 单写（契约 #8）

- `InferWorkerTargetManagerAdapter` 在 `update_targets` **成功返回后**置
  `updated = True`（抛异常则保持 False）。
- `_run_pipeline_mode` 在成功与异常两条出口都记录
  `self._pipeline_target_updated = tgt_adapter.updated`。
- `run()` 尾部：仅当 `_pipeline_target_updated` 为真才跳过 bypass 并重置；
  否则执行一次 `_bypass_update_targets`（Legacy / 跳帧 / Pipeline 前置失败）。
  判定基于"该帧 update 是否完成"，不基于 `pipeline_enabled`。

## 共享后处理抽取与 slot 信号契约（0.3.0）

### 共享后处理 `_postprocess_slot_detections`

- 单一私有方法承载后处理链：`_should_run_yolo_inference` 分支（头部属性 /
  身体手势）→ 矩形检测 → 手部手势 → `final_dedupe` + display 过滤 →
  健康监控回退/恢复 → `source_frame_id` / `source_ts` / `latency_ms` 标记。
- Legacy 命中帧（`_process_slot_frame` 内）与 `_run_pipeline_mode` 尾部
  **调用同一实现**，处理顺序、事件序列、返回值与 `_last_detections` 更新
  位置不变；健康日志文案按 `mode` 参数区分（legacy 记录三项回退说明 /
  还原详情，pipeline 带 `(pipeline mode)` 简短标记），其余行为两侧一致。
- 新增精确单测（均双模式参数化）：`test_shared_postprocess_single_implementation`
  （每命中帧恰好经共享方法一次）、`test_shared_postprocess_health_fallback_behavior_and_wording`
  （快照内容 + 三项回退生效 + 文案区分）、
  `test_shared_postprocess_health_recovery_behavior_and_wording`
  （快照精确还原 + 文案区分）。

### run 级双槽位 slot 信号契约

- 真实生产 QThread `run()`（`_process_slot_frame` 不 mock），两槽位注册、
  各提交帧；`detection_ready` 首参数判别：槽位 0 首帧 frame_id 同为 1，
  若误传 frame_id 首轮首参数即暴露；结果与槽位以 track_id 与全 0/全 1
  帧内容双向核对，覆盖单提交与双槽位同批两种调度。

## Pipeline runtime 复用与生命周期收口（0.4.0）

0.3.0 的差距项"Pipeline 实例复用"在本版收口。核心变化与契约如下。

### Runtime cache 与每帧复用契约

- 新增私有 `_SlotPipelineRuntime`（`ai/inference.py`）：打包一个已
  `initialize()` 的 `Pipeline`（DetectorStage → TrackerStage →
  TargetStage → EventStage）与 4 个适配器（det/trk/tgt/evt）。
- `InferWorker._pipeline_runtimes: dict[int, _SlotPipelineRuntime]`
  按 slot 懒创建：`_run_pipeline_mode` 首帧经
  `_get_or_create_pipeline_runtime` 创建并 initialize 一次，后续帧
  复用同一对象；**不同 slot 各自独立 runtime**。Legacy 模式不调用
  该方法，**不创建任何 runtime**。
- `Pipeline.initialize()` 只在创建时执行一次（成功帧绝不重复
  initialize，也**不得 shutdown**）；`set_pipeline_enabled(False)`
  保留 cache 内容但完全不触碰 runtime，Legacy 行为不变。

### 每帧重置契约（`_run_pipeline_mode` 帧头）

- 更新 det/trk adapter 当前帧引用（`set_frame_image(infer_frame)`）；
- **无论开关如何都重设 redetect 回调**：`_person_redetect_enabled`
  时挂 `_make_redetect_callback(infer_frame)` 的**新**回调（与 Legacy
  共用工厂），disabled 时必须 `set_redetect_callback(None)`——上一帧
  的旧回调不得残留到下一帧；
- 清零 tgt adapter `updated = False`（单写契约帧内语义）与
  trk/det adapter 的帧输出缓存（`last_tracked_dicts` /
  `last_raw_detections`）；
- 每帧新建 `PipelineContext`（携带当前 frame_id / timestamp /
  source_id / image）。

### 异常路径与单写语义（保持不变）

- `initialize` / `run` 异常保持既有行为：记
  `logging.warning("Pipeline mode error (falling back to empty): ...")`
  并返回 `[]`；`_pipeline_target_updated` 仍按帧记录
  `tgt_adapter.updated`（成功与异常两条出口各写一次，
  `_process_slot_frame` 帧首清零的语义不变）。
- 异常 runtime 必须 `shutdown` 并从 cache 淘汰（`_discard_pipeline_runtime`），
  **下一帧可重建**（重新 initialize）。后处理异常仍由 run() 的单槽位
  单帧防火带兜底——runtime 不被淘汰，下一帧继续复用。

### 生命周期收口（retirement / stop / unregister）

- **未运行**时 `unregister_slot`：直接 `shutdown_once` + 从 cache
  移除（锁内做状态变更，锁外执行 teardown）。
- **运行中** `unregister_slot`：不得与推理帧并发 shutdown——仅标记
  `_retired_pipeline_slots`，由 inference thread 在后续 run 循环顶部
  批量 shutdown + 淘汰（与帧处理同线程，天然串行）。
- **重新注册同 slot**（`register_slot`）取消 retirement 标记并保留
  原 runtime 复用。
- `stop()`：等待线程结束后 shutdown 全部 cache runtime 并清空（每个
  runtime 经 `shutdown_once` **至多一次**，单个 shutdown 异常记
  warning 但不中断其余清理）；线程仍在运行（wait 超时）时不得并发
  shutdown，保留 cache。
- `_SlotPipelineRuntime.shutdown_once` 幂等：重复调用、异常重入均不
  触发第二次实际 shutdown。

### 新增精确测试（11 项，spy/计数，不依赖长 sleep）

- `test_pipeline_runtime_reused_across_frames_same_slot`（contract）：
  同 slot 两帧复用同一 Pipeline/adapter、`initialize` 恰一次、
  成功路径无 shutdown；`set_pipeline_enabled(False)` 保留 cache。
- `test_pipeline_runtime_independent_across_slots`（contract）：
  双 slot runtime 互不共享，各 initialize 一次。
- `test_pipeline_redetect_callback_reset_each_frame`（contract）：
  enabled 逐帧新回调，disabled 清 `None`（旧回调不残留）。
- `test_pipeline_failing_runtime_shutdown_evicted_and_rebuilt`（contract）：
  run 异常 → `[]` + warning 语义、shutdown+淘汰、下一帧重建并重新
  initialize、新 runtime 不被关闭。
- `test_pipeline_legacy_mode_creates_no_runtime`（contract）：Legacy
  零 runtime、零 initialize/shutdown。
- `test_unregister_not_running_shuts_down_runtime_immediately`（contract）：
  未运行 unregister 直接 shutdown+移除，`shutdown_once` 幂等。
- `test_reregister_cancels_retirement_and_keeps_runtime`（contract）：
  运行中 unregister 走 retirement 标记，重注册取消并保留 runtime。
- `test_unregister_while_running_retires_runtime_and_run_loop_cleans`
  （resilience，run 级真实 QThread）：运行中不并发 shutdown，run 循环
  有界轮询内完成清理。
- `test_stop_shuts_down_all_runtimes_and_clears_cache_once`
  （resilience，run 级真实 QThread）：stop 后 cache 清空、每 runtime
  恰好 shutdown 一次、重复 stop 不重复关闭。
- `test_pipeline_runtime_retired_during_creation_shutdown_outside_lock`（contract）：
  runtime initialize 期间出现 retirement，锁外 shutdown、返回 `[]`，半成品不入 cache。
- `test_pipeline_postprocess_exception_retains_runtime_for_next_frame`（contract）：
  TargetStage 成功后的后处理异常不淘汰 runtime，下一帧继续复用且不重复 initialize。

## 维护指南

- 跑对拍：`venv\Scripts\python.exe -m pytest tests/test_inferworker_pipeline_contract.py tests/test_inferworker_resilience.py tests/test_inferworker_migration.py tests/test_pipeline.py tests/test_shadow_integration.py tests/test_core_models.py -q`（当前 168 项；其中 contract+resilience+migration+pipeline+shadow 五文件 137 项）。
- 新增帧处理步骤：先加进 Legacy 基线顺序断言（双模式同红同绿），再实现对 Pipeline 侧。
- 新增扩展字段：在对拍测试与 `_dict_to_core_detection` 往返专项中同步验证；
  非基础字段自动经 `extras` 往返，无需登记白名单。
- 单写契约：修改 TargetStage / bypass 任一侧时，必须保持 run 级 5 项证明同红同绿。
- 共享后处理：增删后处理步骤只改 `_postprocess_slot_detections` 一处；
  健康文案按 mode 区分是契约，新增模式需补文案断言。
- slot 信号契约：`detection_ready` 首参数语义（slot_id）变更必须同步
  run 级双槽位测试。
- runtime 生命周期：新增/调整帧处理步骤时保持每帧重置契约（redetect
  重设、updated 清零、新建 Context）与 9 项生命周期测试同红同绿；
  `shutdown_once` 的至多一次语义不得放宽；运行中 teardown 只能经
  retirement 由 inference thread 执行。
- 事件名单（`events`）是顺序契约的单一事实源；改名需全文件同步。
- 不引入真实模型 / 摄像头 / 网络依赖；`_debug_emit_reports` 必须保持 20。
- 默认开关：`pipeline_enabled` 必须保持默认 `False`（Legacy），本步不加 UI 开关。

## 当前差距（对拍已转绿；以下为剩余差距）

0.1.0 记录的 5 项对拍缺口已全部转绿：

1. ✅ 适配器 schema 与生产不一致（KeyError）——统一为生产 schema。
2. ✅ redetect 回调缺失——`_make_redetect_callback` 单点工厂共用。
3. ✅ 扩展字段 core 往返丢失——`_AdaptedDetection` 深拷贝 extras 无损往返。
4. ✅ TargetManager 双写——按帧内 `updated` 标志单写。
5. ✅ 跳帧路径未对拍——双模式同一断言转绿。

剩余差距（后续里程碑）：

0.4.0 已收口「Pipeline 实例复用」差距：`_run_pipeline_mode` 改为每 slot
懒创建并复用已 initialize 的 `_SlotPipelineRuntime`（详见上文 0.4.0
契约），生命周期（retirement / stop / 异常淘汰）契约化并转绿。

1. **端到端（采集→推理→显示）**：仍无集成测试覆盖完整链路（PROJECT_BIBLE §12.2）。
2. **全套件回归**：1336 项已通过（2026-11-20）；后续改动仍须保持 `pytest tests/` 全绿。

## 剩余边界摘要

| 边界 | 现状 |
|------|------|
| 端到端（采集→推理→显示） | 不在本契约范围，缺口仍在 §12.2 |
| Pipeline 对拍 | 11 scenario × 2 模式全部转绿（契约 #1–#7 + 共享后处理 3 项，#9 slot 信号为 run 级） |
| TargetManager 单写 | run 级 5 项证明通过（契约 #8） |
| Pipeline 默认开关 | 保持默认关闭；无 UI 开关 |
| Pipeline runtime 复用 | 0.4.0 收口：每 slot 懒创建复用，Legacy 零 runtime；生命周期契约 + 11 项测试转绿 |
| orphan inference_controller | 不使用、不修复、不接入 |
| `ai/tracker.py` / `rust/` | 未触碰；无需重建 |
