# InferWorker 单槽位单帧异常防火带

> **版本**：1.0.0（2026-11-20）
> **范围**：`ai/inference.py`（InferWorker）、`gui/main_window.py`、`tests/test_inferworker_resilience.py`（新增）
> **状态**：已合入 main（PR #4 → `b4286d3`，2026-11-20）；本地 6 项防火带新增测试与全套件 1336 项通过；后续单写、slot 与 runtime 契约见 [inferworker-pipeline-parity](inferworker-pipeline-parity.md)

## 目标与背景

生产用 `ai.inference.InferWorker` 是全局唯一的推理 QThread（1 个共享推理线程是
既定架构决策，见 PROJECT_BIBLE §16.4）。改动前，`run()` 主循环内每帧的
`tracker.update` / `update_appearance` / `predict_step` 及部分后处理没有顶层保护：
任一处抛异常都会逃出 `run()`，整个线程终止，**所有**采集位同时失去 AI 检测，
且 GUI 无任何提示（仅日志里一条未捕获异常）。

本设计为每帧处理加"防火带"：单槽位单帧的异常被就地兜住，只跳过该帧，线程与
其他槽位继续工作，同时按槽位节流上报错误与恢复，让 dashboard 可感知。

## 架构

### 机械提取 `_process_slot_frame`

`run()` 循环体中"每个 snapshot 的 Pipeline / Legacy / 跳帧三路径处理"原样提取为
独立方法（字节级一致，含 `#region debug-point A:infer-emit` 成功帧
`_debug_report` 调试契约）：

```python
def _process_slot_frame(self, slot_id, frame_id, submitted_ts, frame) -> list[dict]
```

- 默认行为、处理顺序、latency / source 标记（`source_frame_id` / `source_ts` /
  `latency_ms`）、`_last_detections` 更新全部不变；
- 成功帧的 `_debug_report("A", "ai/inference.py:run", "detection emitted", ...)`
  仍在 emit 前按原条件（`_debug_emit_reports < 20`）执行；异常帧不上报该点
  （与改动前"线程直接死亡"相比，调试契约对成功帧完全等价）。

### run() 防火带

```python
try:
    detections = self._process_slot_frame(slot_id, frame_id, submitted_ts, frame)
except Exception as exc:
    self._record_slot_failure(slot_id, frame_id, exc)
    continue
self._clear_slot_failure(slot_id)
self.detection_ready.emit(slot_id, detections)
self._bypass_update_targets(slot_id, detections, submitted_ts)
```

异常时**跳过该帧**的 `detection_ready` 与 `_bypass_update_targets`，保留
`_last_detections` 中的最后有效结果，继续同批其余槽位与后续帧；
**刻意不 emit 空列表**（空结果会清空 GUI 叠加与 TargetManager 目标状态）。

### 失败计数与节流上报

| 信号 | 签名 | 语义 |
|------|------|------|
| `error_occurred` | `(int, str)` | `(slot_id, message)`，message 含采集位/帧号/连续次数/异常 |
| `processing_recovered` | `(int)` | `(slot_id,)`，一轮连续失败后首次成功时发射一次 |

- 每槽位记录连续失败数（`_slot_failure_counts`）；**第 1 次及每第 30 次**
  （1, 30, 60, ...）连续失败才记录带 traceback 的 ERROR 并 emit
  `error_occurred`，避免异常风暴刷爆日志/GUI；
- 任一帧成功且该槽位此前有失败：清零计数、记录 INFO、emit
  `processing_recovered` 一次；
- `register_slot` / `unregister_slot` 同步清理该槽位计数（重新注册从第 1 次计）。

### GUI 联动

`MainWindow` 连接两个信号（AutoConnection → GUI 线程队列执行）：

- `error_occurred` → dashboard 显示"采集位 N：AI 异常 — <消息>"；
- `processing_recovered` → dashboard 显示"采集位 N：AI 已恢复"。

只更新状态文案，**不影响摄像头采集与其他槽位**；不做自动重启。

## 契约

| 项 | 约定 | 修改契约 |
|----|------|----------|
| 提取等价性 | `_process_slot_frame` 与原循环体字节级一致（112 行，机器比对 0 差异） | 改动任一路径（Pipeline/Legacy/跳帧）必须保持在该方法内，且同步更新 run 级测试 |
| 异常帧副作用 | 异常帧不得 emit `detection_ready`、不得调用 `_bypass_update_targets`、不得改动 `_last_detections` | 禁止改为 emit `[]` 或清空缓存 |
| 上报节流 | 第 1 次及每第 30 次连续失败才 ERROR + emit；成功一轮只 emit 一次 recovered | 调整节流倍数需同步测试断言 |
| 计数生命周期 | register/unregister 清零该槽位计数 | 新增槽位生命周期入口时必须同步清理 |
| 调试契约 | 成功帧 `debug-point A` 报告（字段/计数/位置串 `ai/inference.py:run`）不变 | 禁止删除或改条件 |
| 线程边界 | 防火带只保证"帧级不退出"；模型加载（`_initialize_backend` / `_warmup` / `_switch_model_if_needed`）与 `submit_frame` 的异常仍不在保护范围 | 剩余边界见下节 |

## 维护指南

- 新增每帧处理步骤：优先放进 `_process_slot_frame` 内对应路径，自动获得防火带；
  放在 run() 循环内的代码需自行评估异常语义。
- 剩余边界（本设计明确**不解决**，未来里程碑考虑）：
  1. 模型加载/切换阶段异常仍会终止线程（防火带只覆盖稳态帧处理）；
  2. 防火带不做线程自动重启，持续异常时该槽位持续跳帧（GUI 有节流提示）；
  3. `_HealthMonitor` 的健康回退与防火带相互独立，前者语义未变。
- 测试：`pytest tests/test_inferworker_resilience.py -q`（真实 QThread run 级，
  monkeypatch 禁模型加载；等待全部有界轮询，无 sleep 长等）。不触碰、不启用
  orphan 的 `inference_controller`。

## 剩余边界摘要

| 边界 | 现状 |
|------|------|
| 模型加载/切换异常 | 未覆盖（线程仍会终止） |
| 线程自动重启 | 不做（dashboard 节流提示代替） |
| 全套件回归 | 1336 项通过（2026-11-20） |
