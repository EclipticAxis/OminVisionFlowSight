# TrackerStage 设计文档 (Milestone C4)

> **版本**：C4 — 2026-07-25
> **范围**：新增 `visioncore/pipeline/stages/tracker_stage.py`，定义跟踪器阶段 + Tracker 接口 + DummyTracker
> **状态**：已实现，42 项单元测试全部通过，全量回归零破坏（18 个既有测试套件 + 42 项新测试）

---

## 1. 目标与背景

### 1.1 为什么需要 TrackerStage

C3 建立了检测阶段（DetectorStage：frame → detections），但 Pipeline 的**跟踪能力**仍未接入——`PipelineContext.tracks` 字段没有生产者。

当前跟踪逻辑耦合在 `ai/tracker.py`（~1086 行 DetectionTracker + UKF 族）与 `ai/inference.py` 中，直接硬编码卡尔曼/UKF 滤波器与匹配策略。若 Pipeline 直接依赖具体跟踪器，会产生：

- **算法耦合**：Pipeline 绑死到 ByteTrack/BoTSORT/Kalman 等具体算法
- **测试困难**：无法在不实例化滤波器的情况下测试跟踪阶段
- **切换成本**：换跟踪算法需修改 Pipeline 代码

TrackerStage 通过**依赖反转**解决：定义抽象 `Tracker` 接口，TrackerStage 依赖接口而非具体类，具体跟踪器在构造时注入。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **接口优先** | `TrackerStage(tracker)` 依赖 `Tracker` ABC，非具体类 |
| **算法不可知** | **禁止**依赖 ByteTrack / BoTSORT / OSTrack；AST 测试守护 |
| **薄适配器** | TrackerStage 自身无跟踪逻辑，全权委托给包装的 Tracker |
| **生命周期委托** | stage 的 initialize/process/shutdown/health_check 委托给 tracker |
| **REPLACE 语义** | tracker 输出是完整当前 track 集，**替换** context.tracks（非 extend） |
| **空检测不跳过** | `detections=[]` 是合法"仅预测"信号，**总是调用 update** |
| **零侵入** | 不修改 `ai/` `camera/` `gui/`，现有 DetectionTracker 继续运行不变 |

### 1.3 与 C1/C2/C3 的关系

```
C2 FrameSource → C3 DetectorStage → detections → C4 TrackerStage → tracks
                                                                 │
                                                                 ▼
                                              (C5 TargetMgmtStage 读 tracks 写 targets)
```

C4 是 Pipeline 跟踪链：读 C3 产出的 detections，关联跨帧成 tracks，为 C5 目标管理阶段提供输入。

### 1.4 与 DetectorStage 的关键差异

| 维度 | DetectorStage (C3) | TrackerStage (C4) |
|---|---|---|
| **输入** | `context.frame` (单帧) | `context.detections` (检测列表) |
| **输出** | `context.detections` (**extend** 累积) | `context.tracks` (**replace** 替换) |
| **无输入时** | frame=None → **跳过** (不调 detect) | detections=[] → **仍调 update** (仅预测) |
| **状态** | 无状态 (每帧独立检测) | **有状态** (跨帧维护 track 身份) |
| **接口方法** | `detect(frame)` | `update(detections)` |

**为何 tracker replace 而 detector extend？** 检测是**加法**的——多个检测器（人体+头部+手势）可各贡献检测，输出累积。跟踪是**状态**的——一个跟踪器在任意时刻持有**完整当前 track 集**，其输出是全状态，非加法贡献。链式两个跟踪器会让第二个替换第一个的输出（第二个成为"当前"跟踪器）。

**为何空检测不跳过？** 跟踪器即使无新检测也需**预测/推进**已有 track（卡尔曼预测、老化 lost_frames）。`update([])` 是"本帧无新检测，仅预测"的合法信号，不是 no-op。这与 DetectorStage 在 frame=None 时跳过形成对比——检测器无帧可检测是真 no-op，跟踪器无新检测仍有事可做。

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── __init__.py            # 包导出：Detector系(C3) + Tracker系(C4)
├── detector_stage.py      # (C3) Detector ABC + DetectorError + DummyDetector + DetectorStage
└── tracker_stage.py       # (C4) Tracker ABC + TrackerError + DummyTracker + TrackerStage
```

按 C4 规约，三类关注点（接口 + 测试双 + 阶段）集中于单文件 `tracker_stage.py`，用 `# ====` 分节。`__init__.py` 在 C3 基础上追加 tracker 导出（C3 的 detector 测试已改为子集断言以容忍包增长）。

---

## 3. Tracker 接口

### 3.1 类定义

```python
class Tracker(ABC):
    @abstractmethod
    def initialize(self) -> None: ...
    @abstractmethod
    def update(self, detections: list[Detection]) -> list[Track]: ...
    @abstractmethod
    def shutdown(self) -> None: ...
    @abstractmethod
    def health_check(self) -> bool: ...
```

### 3.2 四个方法

| 方法 | 语义 |
|---|---|
| `initialize()` | 获取/重置资源（清空 track 状态、初始化滤波器参数、重置 ID 计数器）。幂等——二次调用重置为全新状态。失败抛 `TrackerError`。 |
| `update(detections)` | 关联 `detections` 与现有 track，推进状态，返回**完整当前 track 集**。`detections` 可为空（"仅预测"信号）。返回列表归调用方所有，tracker 不得保留后改的引用。 |
| `shutdown()` | 释放资源。幂等，**绝不抛异常**。 |
| `health_check()` | 无副作用探针，返回 `True` 当且仅当已初始化且就绪。**绝不抛异常**。 |

### 3.3 算法不可知（强制）

Tracker 接口**不引用**任何具体算法：无 ByteTrack、无 BoTSORT、无 OSTrack、无 Kalman、无 UKF、无 filterpy、无 scipy。接口只说"跟踪器消费检测返回 track 列表"，不说"如何跟踪"。

AST 测试 `test_no_bytrack_botsort_ostrack_in_source` 解析 `tracker_stage.py` 源码，断言无任何 banned 后端的 import。运行时测试 `test_no_banned_backends_loaded_at_runtime` 断言导入 stages 包不加载 ultralytics/onnxruntime/cv2/filterpy/scipy。

### 3.4 update([]) 的"仅预测"语义

调用 `update([])`（空检测列表）是合法的"本帧无新检测，仅预测/推进现有 track"信号。跟踪器应：
- 推进所有现有 track 的卡尔曼/UKF 预测
- 递增各 track 的 `lost_frames`（无关联即视为丢失一帧）
- 移除超过 `max_misses` 的 track
- 返回（可能减少的）当前 track 集

`update([])` **不是** no-op，**不抛异常**。TrackerStage 据此在无检测的帧仍调用 update。

### 3.5 TrackerError

`TrackerError(RuntimeError)` 是结构化异常，跟踪器 *可选* 地用它标记可恢复故障（滤波器发散/NaN 协方差、匹配产生不一致状态、外观模型 OOM）。Pipeline **不**特殊捕获——和普通异常一样传出 `run()`，遵循 C1"诚实异常"契约。

---

## 4. TrackerStage

### 4.1 数据流

```
process(context):
    detections = context.detections          # 可为空
    tracks = tracker.update(detections)       # 总是调用，空=仅预测
    context.tracks.clear()                    # REPLACE：先清空
    context.tracks.extend(tracks)             # 再填入新 track 集
```

### 4.2 设计决策

| 决策 | 理由 |
|---|---|
| **REPLACE（clear+extend）非 rebind** | 保留 list 对象身份，外部引用能看到新内容；tracker 输出是完整状态 |
| **总是调用 update** | `detections=[]` 是"仅预测"合法信号；跟踪器即使无新检测也需推进/老化 track。对齐现有 DetectionTracker 的 `predict_step()` 语义 |
| **不跳过** | 与 DetectorStage（frame=None 跳过）形成对比——检测器无帧是真 no-op，跟踪器无新检测仍有事可做 |
| **clear 在 update 之后** | 若 update 抛异常，context.tracks 保持原状（不被清空），避免异常时丢失既有 track |
| **生命周期全委托** | stage 是薄适配器，无自身状态 |

### 4.3 构造

```python
TrackerStage(tracker: Tracker, *, name: str | None = None)
```

- `tracker` 必须是 `Tracker` 实例（接口强制）。`TypeError` 守护。
- `name` 默认 `"TrackerStage"`，可覆盖（多跟踪器链时区分）。

### 4.4 异常时不清空 tracks

关键实现细节：`process` 先调 `tracker.update()`，**成功后**才 `clear()`+`extend()`。若 `update` 抛异常，`clear()` 不执行，`context.tracks` 保持原状。测试 `test_process_exception_does_not_write_partial_tracks` 守护此语义——异常时不丢失既有 track。

---

## 5. DummyTracker

### 5.1 用途

**可配置测试跟踪器**——无滤波器、无匹配、无状态，返回预定的 `Track` 列表。用于：

1. **阶段测试**：验证 TrackerStage 生命周期委托与数据流，无需真实滤波器
2. **Pipeline 集成**：向 Pipeline 喂确定性 tracks，测试下游目标管理/投影阶段
3. **基准**：测量阶段开销，零跟踪成本

### 5.2 配置

| 参数 | 默认 | 作用 |
|---|---|---|
| `tracks` | `None` | 返回的 Track 列表；`None` 时生成 1 个合成 track（track_id=1, 居中 person） |
| `raise_on_update` | `None` | 异常实例，下次 `update()` 抛出（异常传播测试）；持续到清除 |

### 5.3 行为

- `update()` 返回**新列表**（调用方可安全修改），记录 `last_detections`（断言 stage 转发了什么）
- 计数器：`initialize_count`/`update_count`/`shutdown_count`
- `reset()` 清零计数与状态，保留 track 缓冲

---

## 6. 测试覆盖

### 6.1 统计

- **测试文件**：`tests/test_tracker_stage.py`
- **测试函数数**：42（全部通过）
- **运行方式**：`PYTHONPATH=F:/VisionBata F:/VisionBata/venv/Scripts/python.exe tests/test_tracker_stage.py`

### 6.2 覆盖类别

| 类别 | 测试数 | 覆盖 |
|---|---|---|
| 包导出 + ABC 契约 | 4 | tracker 名在 __all__；TrackerError 是 RuntimeError 子类；Tracker 不可实例化；缺方法子类失败 |
| DummyTracker | 8 | 默认 track；自定义；fresh list；记录 last_detections；生命周期+计数器；raise_on_update；持续到清除；reset |
| TrackerStage 构造 | 6 | 包装 tracker；默认名；自定义名；拒 None；拒非 Tracker；是 PipelineStage 子类 |
| 生命周期委托 | 4 | initialize/shutdown/health_check 委托；repr 报健康 |
| process 数据流 | 8 | 读 detections 写 tracks；自定义；**REPLACE 非 extend**；保留 list 身份；**空检测仍调 update**；update_count 递增；转发精确列表；无 detections 字段 |
| 异常传播 | 3 | TrackerError 传播；任意异常传播；**异常时不清空 tracks** |
| Pipeline 集成 | 3 | 端到端；生命周期一次/多次 run |
| 端到端 Detector→Tracker | 2 | C3→C4 链；无帧时检测跳过但跟踪仍运行 |
| 无 banned 后端 | 3 | AST 无 ByteTrack/etc；运行时不加载；接口仅 4 方法 |
| Replace vs extend 对比 | 1 | 明确对比 DetectorStage extend 与 TrackerStage replace |

### 6.3 关键测试

| 测试 | 验证 |
|---|---|
| `test_no_bytrack_botsort_ostrack_in_source` | AST 解析源码，断言无 banned 后端 import |
| `test_no_banned_backends_loaded_at_runtime` | 导入 stages 不加载 ultralytics/onnxruntime/cv2/filterpy/scipy |
| `test_process_replaces_not_extends_tracks` | REPLACE 语义：旧 track 被清空，仅留 tracker 输出 |
| `test_process_with_empty_detections_still_calls_update` | 空检测不跳过，update 仍调用（仅预测语义） |
| `test_process_exception_does_not_write_partial_tracks` | update 抛异常时 tracks 不被清空 |
| `test_process_preserves_list_identity` | clear+extend 保留 list 对象身份 |
| `test_end_to_end_detector_then_tracker` | C3 DetectorStage → C4 TrackerStage 全链 |
| `test_end_to_end_detector_tracker_no_frame` | 无帧：检测跳过，跟踪仍运行（update([])） |
| `test_tracker_replaces_while_detector_extends` | 明确对比两种语义 |

### 6.4 回归测试

全量回归零破坏——所有 18 个既有测试套件 + C4 新增全过：

```
test_tracker_stage.py (C4 新增)  : 42/42 passed
test_detector_stage.py (C3)      : 39/39 passed  (test_package_exports 改为子集断言以容忍包增长)
test_frame_source.py (C2)        : 55/55 passed
test_pipeline.py (C1)            : 56/56 passed
test_core_models.py              : 31/31 passed
test_target_state.py             : 36/36 passed
test_event_bus.py                : 77/77 passed
test_target_manager.py           : 76/76 passed
test_target_manager_events.py    : 29/29 passed
test_shadow_integration.py       : 12/12 passed
test_protocol_base.py            : 36/36 passed
test_debug_logger.py             : 20/20 passed
test_json_serializer.py          : 31/31 passed
test_cbor_serializer.py          : 31/31 passed
test_udp_adapter.py              : 23/23 passed
test_state_publisher.py          : 28/28 passed
test_shadow_publisher.py         : 36/36 passed
test_obb_stage2.py               : passed
test_matching.py                 : passed
────────────────────────────────────────────────────────────────
总计                             : 既有全过 + 42 新增，0 回归
```

**C3 测试调整说明**：`test_detector_stage.py` 的 `test_package_exports_public_api` 原用严格集合相等 `set(__all__) == {detector names}`，C4 合法扩展 `__all__` 后改为子集断言（检测器名**存在于** `__all__`）。这是 C3 测试过于严格的修正，非功能回归。

---

## 7. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `tracker_stage.py` | ✅ |
| TrackerStage 继承 PipelineStage | ✅ |
| 输入 detections，输出 tracks | ✅ |
| 依赖 Tracker Interface | ✅ Tracker ABC，4 方法 |
| 禁止 ByteTrack/BoTSORT/OSTrack | ✅ AST + 运行时双重守护 |
| 新增 DummyTracker | ✅ 返回固定 Track |
| 新增 `tests/test_tracker_stage.py` | ✅ 42 项测试 |
| 输出 `docs/tracker-stage.md` | ✅ 本文件 |
| 不修改 ai/camera/gui | ✅ 仅新增文件 + 更新 stages/__init__.py |

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/stages/tracker_stage.py` | Tracker ABC + TrackerError + DummyTracker + TrackerStage |
| `visioncore/pipeline/stages/__init__.py` | 更新：追加 tracker 导出（与 detector 并列） |
| `tests/test_tracker_stage.py` | 42 项单元测试 + 自定义 `raises` + 手动 runner |
| `tests/test_detector_stage.py` | 微调：`test_package_exports` 改为子集断言（容忍包增长） |
| `docs/tracker-stage.md` | 本设计文档 |

---

## 9. 未来扩展路线

### 9.1 真实跟踪器（C5）

| 跟踪器 | 文件 | 后端 |
|---|---|---|
| `KalmanTracker` | `stages/trackers/kalman_tracker.py` | 包装现有 `ai/tracker.py` DetectionTracker（线性卡尔曼） |
| `UkfTracker` | `stages/trackers/ukf_tracker.py` | 包装 UKF/MCUKF/Manifold UKF 族 |
| `ReidTracker` | `stages/trackers/reid_tracker.py` | 包装带外观特征的跟踪器 |

每个真实跟踪器是 `Tracker` 子类，是**唯一** import 其算法库（filterpy/scipy）的模块。TrackerStage 永不改变。

### 9.2 完整 Pipeline 链（C5+）

| 阶段 | 读 | 写 | 状态 |
|---|---|---|---|
| `CaptureStage` | FrameSource | `context.frame` | 待实现 |
| `DetectorStage` | `context.frame` | `context.detections` | ✅ C3 |
| `TrackerStage` | `context.detections` | `context.tracks` | ✅ C4 |
| `TargetMgmtStage` | `context.tracks` | `context.targets` | 待实现 |
| `ProjectionStage` | `context.targets` | `context.target_states` | 待实现 |

完整链：`Capture → Detect → Track → TargetMgmt → Project`，每阶段包装一个 VisionCore 抽象，全用接口注入。C4 完成了链的中段。

### 9.3 多跟踪器与融合（C6+）

- 多目标跟踪器（按类别分流：人/车/头各一跟踪器）
- 跨摄像头跟踪融合（ReID + 全局 ID 分配）
- 跟踪健康监控（`health_check` 接入 `_HealthMonitor`，发散自动回退）

---

## 10. 后续里程碑预告

- **C5**：真实检测器/跟踪器接入（包装现有 ai/ 后端）+ CaptureStage + TargetMgmtStage + ProjectionStage（完整链）
- **C6**：InferWorker 影子旁路——Pipeline 与现有 InferWorker 并行对比
- **D**：InferWorker 退役，Pipeline 转正
