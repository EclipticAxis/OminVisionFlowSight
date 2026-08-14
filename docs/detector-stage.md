# DetectorStage 设计文档 (Milestone C3)

> **版本**：C3 — 2026-07-25
> **范围**：新增 `visioncore/pipeline/stages/` 子包，定义检测器阶段 + Detector 接口 + DummyDetector
> **状态**：已实现，39 项单元测试全部通过，全量回归零破坏（17 个既有测试套件 + 39 项新测试）

---

## 1. 目标与背景

### 1.1 为什么需要 DetectorStage

C1 建立了 Pipeline 框架（Stage + Context + 执行器），C2 建立了输入源抽象（FrameSource → Frame）。但 Pipeline 的**检测能力**仍未接入——`PipelineContext.detections` 字段没有生产者。

当前检测逻辑耦合在 `ai/inference.py`（~1792 行单体 InferWorker）中，直接硬编码 YOLO/ONNX 后端调用。若 Pipeline 直接依赖具体检测器，会产生：

- **模型耦合**：Pipeline 绑死到 YOLO/RT-DETR 等具体架构
- **测试困难**：无法在不加载模型权重的情况下测试检测阶段
- **切换成本**：换检测后端需修改 Pipeline 代码

DetectorStage 通过**依赖反转**解决：定义抽象 `Detector` 接口，DetectorStage 依赖接口而非具体类，具体检测器在构造时注入。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **接口优先** | `DetectorStage(detector)` 依赖 `Detector` ABC，非具体类 |
| **模型不可知** | **禁止**依赖 YOLO / RT-DETR / GroundingDINO / SAM；AST 测试守护 |
| **薄适配器** | DetectorStage 自身无检测逻辑，全权委托给包装的 Detector |
| **生命周期委托** | stage 的 initialize/process/shutdown/health_check 委托给 detector |
| **诚实数据流** | 读 `context.frame` → 写 `context.detections`；frame=None 则跳过 |
| **零侵入** | 不修改 `ai/` `camera/` `gui/`，现有 InferWorker 继续运行不变 |

### 1.3 与 C1/C2 的关系

```
C2 FrameSource.read() ──> Frame ──> (FeederStage) ──> context.frame
                                                          │
                                                          ▼
                            C3 DetectorStage.process(context)
                                   │
                          detector.detect(frame)
                                   │
                          list[Detection]
                                   │
                                   ▼
                            context.detections
                                   │
                                   ▼
                    (C4 TrackingStage 读 detections 写 tracks)
```

C3 是 Pipeline 检测链的第一环：读 C2 产出的 Frame，产出 Detection，为 C4 跟踪阶段提供输入。

---

## 2. 文件结构

```
visioncore/pipeline/stages/
├── __init__.py            # 包导出：Detector, DetectorError, DetectorStage, DummyDetector
└── detector_stage.py      # Detector ABC + DetectorError + DummyDetector + DetectorStage
```

按 C3 规约，三类关注点（接口 + 测试双 + 阶段）集中于单文件 `detector_stage.py`，用 `# ====` 分节。`__init__.py` 仅做导出。未来真实检测器（C4+）作为 `Detector` 子类加入，可以是本子包的兄弟模块或独立包。

**不修改** `visioncore/pipeline/__init__.py`——stages 子包独立导出，保持 C1 框架包纯净（框架 vs 具体阶段分离）。

---

## 3. Detector 接口

### 3.1 类定义

```python
class Detector(ABC):
    @abstractmethod
    def initialize(self) -> None: ...
    @abstractmethod
    def detect(self, frame: Frame) -> list[Detection]: ...
    @abstractmethod
    def shutdown(self) -> None: ...
    @abstractmethod
    def health_check(self) -> bool: ...
```

### 3.2 四个方法

| 方法 | 语义 |
|---|---|
| `initialize()` | 获取资源（加载模型权重、创建 ONNX/TensorRT session、预热分配）。幂等。失败抛 `DetectorError`。 |
| `detect(frame)` | 对一帧运行检测，返回 `list[Detection]`（可为空）。`frame` 不可为 `None`（DetectorStage 在 frame=None 时跳过，不调用 detect）。 |
| `shutdown()` | 释放资源。幂等，**绝不抛异常**。 |
| `health_check()` | 无副作用探针，返回 `True` 当且仅当已初始化且就绪。**绝不抛异常**。 |

### 3.3 模型不可知（强制）

Detector 接口**不引用**任何具体架构：无 YOLO、无 RT-DETR、无 GroundingDINO、无 SAM、无 ultralytics、无 onnxruntime、无 cv2。接口只说"检测器检测一帧返回 Detection 列表"，不说"如何检测"。

AST 测试 `test_no_yolo_rt_detr_groundingdino_sam_in_source` 解析 `detector_stage.py` 源码，断言无任何 banned 后端的 import。运行时测试 `test_no_yolo_rt_detr_groundingdino_sam_loaded_at_runtime` 断言导入 stages 包不加载 ultralytics/onnxruntime/cv2。

### 3.4 DetectorError

`DetectorError(RuntimeError)` 是结构化异常，检测器 *可选* 地用它标记可恢复故障（模型 OOM、推理超时、NaN 分数）。Pipeline **不**特殊捕获——它和普通异常一样传出 `run()`，遵循 C1"诚实异常"契约。

---

## 4. DetectorStage

### 4.1 数据流

```
process(context):
    frame = context.frame
    if frame is None: skip (debug log)     # 合法的"无帧可检测"
    else:
        detections = detector.detect(frame)
        context.detections.extend(detections)  # 扩展，非替换
```

### 4.2 设计决策

| 决策 | 理由 |
|---|---|
| **extend 而非 replace** | 允许 Pipeline 链式多个检测阶段（如人体检测 + 头部检测），输出累积。单检测器在空 Context 上 extend == replace。 |
| **frame=None 跳过而非抛异常** | `None` 帧是合法的"暂时无帧"信号（源耗尽、丢帧），不是错误。跳过 + debug 日志，不中止 run。对齐 C2 的 `read()→None` 语义。 |
| **不修改 frame** | Frame 是 frozen dataclass，process 只读不写。测试 `test_process_does_not_mutate_frame` 守护。 |
| **生命周期全委托** | stage.initialize→detector.initialize，stage.shutdown→detector.shutdown，stage.health_check→detector.health_check。stage 是薄适配器，无自身状态。 |

### 4.3 构造

```python
DetectorStage(detector: Detector, *, name: str | None = None)
```

- `detector` 必须是 `Detector` 实例（接口强制，非 None、非任意对象）。`TypeError` 守护。
- `name` 默认 `"DetectorStage"`，可覆盖（多检测器链时区分）。

### 4.4 与 PipelineStage 的关系

DetectorStage 继承 PipelineStage，实现四方法。它是 C1 框架的第一个**真实**阶段（DummyStage 是测试双）。可在任意 Pipeline 中 `add_stage` 使用：

```python
p = Pipeline()
p.add_stage(DetectorStage(DummyDetector(), name="detect"))
with p:
    ctx = p.run(PipelineContext.empty())
```

---

## 5. DummyDetector

### 5.1 用途

**可配置测试检测器**——无模型、无 GPU、无 I/O，返回预定的 `Detection` 列表。用于：

1. **阶段测试**：验证 DetectorStage 生命周期委托与数据流，无需真实模型
2. **Pipeline 集成**：向 Pipeline 喂确定性 Detection，测试下游跟踪/投影阶段
3. **基准**：测量阶段开销，零推理成本

### 5.2 配置

| 参数 | 默认 | 作用 |
|---|---|---|
| `detections` | `None` | 返回的 Detection 列表；`None` 时生成 1 个合成检测（居中 person, score 0.9） |
| `raise_on_detect` | `None` | 异常实例，下次 `detect()` 抛出（异常传播测试）；持续到清除 |

### 5.3 行为

- `detect()` 返回**新列表**（调用方可安全修改），但**共享** Detection 实例（frozen/immutable，安全共享）
- 计数器：`initialize_count`/`detect_count`/`shutdown_count`
- `reset()` 清零计数与状态，保留检测缓冲

---

## 6. 测试覆盖

### 6.1 统计

- **测试文件**：`tests/test_detector_stage.py`
- **测试函数数**：39（全部通过）
- **运行方式**：`PYTHONPATH=F:/VisionBata F:/VisionBata/venv/Scripts/python.exe tests/test_detector_stage.py`

### 6.2 覆盖类别

| 类别 | 测试数 | 覆盖 |
|---|---|---|
| 包导出 + ABC 契约 | 4 | `__all__`；DetectorError 是 RuntimeError 子类；Detector 不可实例化；缺方法子类失败 |
| DummyDetector | 8 | 默认 person；自定义；fresh list；共享实例；生命周期+计数器；raise_on_detect；持续到清除；reset |
| DetectorStage 构造 | 6 | 包装 detector；默认名；自定义名；拒 None；拒非 Detector；是 PipelineStage 子类 |
| 生命周期委托 | 4 | initialize/shutdown/health_check 委托；repr 报健康 |
| process 数据流 | 6 | 读 frame 写 detections；自定义；扩展非替换；frame=None 跳过；detect_count 递增；不修改 frame |
| 异常传播 | 3 | DetectorError 传播；任意异常传播；异常不写部分 detections |
| Pipeline 集成 | 4 | 端到端；生命周期一次/多次 run；frame=None 跳过 |
| 端到端 source→detect | 1 | DummyFrameSource→FeederStage→DetectorStage 全链 |
| 无 banned 后端 | 3 | AST 无 YOLO/etc import；运行时不加载；接口仅 4 方法无模型特定属性 |

### 6.3 关键测试

| 测试 | 验证 |
|---|---|
| `test_no_yolo_rt_detr_groundingdino_sam_in_source` | AST 解析源码，断言无 banned 后端 import |
| `test_no_yolo_rt_detr_groundingdino_sam_loaded_at_runtime` | 导入 stages 不加载 ultralytics/onnxruntime/cv2 |
| `test_detector_interface_is_model_agnostic` | Detector 仅 4 方法（initialize/detect/shutdown/health_check），无模型特定属性 |
| `test_process_with_none_frame_skips` | frame=None 不抛、不调 detect、detections 不变 |
| `test_process_extends_existing_detections` | extend 而非 replace（多阶段链支持） |
| `test_end_to_end_source_feeds_detector` | C2 FrameSource → C3 DetectorStage 全链 |

### 6.4 回归测试

全量回归零破坏——所有 17 个既有测试套件 + C3 新增全过：

```
test_detector_stage.py (C3 新增)  : 39/39 passed
test_frame_source.py (C2)         : 55/55 passed
test_pipeline.py (C1)             : 56/56 passed
test_core_models.py               : 31/31 passed
test_target_state.py              : 36/36 passed
test_event_bus.py                 : 77/77 passed
test_target_manager.py            : 76/76 passed
test_target_manager_events.py     : 29/29 passed
test_shadow_integration.py        : 12/12 passed
test_protocol_base.py             : 36/36 passed
test_debug_logger.py              : 20/20 passed
test_json_serializer.py           : 31/31 passed
test_cbor_serializer.py           : 31/31 passed
test_udp_adapter.py               : 23/23 passed
test_state_publisher.py           : 28/28 passed
test_shadow_publisher.py          : 36/36 passed
test_obb_stage2.py                : passed
test_matching.py                  : passed
────────────────────────────────────────────────────────────────
总计                              : 既有全过 + 39 新增，0 回归
```

循环导入：2 个 stages 子模块独立导入全成功。副作用：导入 stages 不加载 ai/gui/camera/PyQt/cv2/ultralytics/onnxruntime/mediapipe/av。

---

## 7. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `visioncore/pipeline/stages/` 目录 | ✅ |
| `detector_stage.py` 定义 DetectorStage | ✅ 继承 PipelineStage |
| 构造 `DetectorStage(detector)` | ✅ 接口注入 |
| 读 `PipelineContext.frame` | ✅ |
| 输出 `PipelineContext.detections` | ✅ extend |
| Detector 必须使用接口 | ✅ Detector ABC，4 方法 |
| 不得依赖 YOLO/RT-DETR/GroundingDINO/SAM | ✅ AST + 运行时双重守护 |
| 新增 DummyDetector | ✅ 返回固定 Detection |
| 新增 `tests/test_detector_stage.py` | ✅ 39 项测试 |
| 输出 `docs/detector-stage.md` | ✅ 本文件 |
| 不修改 ai/camera/gui | ✅ 仅新增文件 |

---

## 8. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/pipeline/stages/__init__.py` | 包导出（4 名）+ 范围说明 |
| `visioncore/pipeline/stages/detector_stage.py` | Detector ABC + DetectorError + DummyDetector + DetectorStage |
| `tests/test_detector_stage.py` | 39 项单元测试 + 自定义 `raises` + 手动 runner |
| `docs/detector-stage.md` | 本设计文档 |

---

## 9. 未来扩展路线

### 9.1 真实检测器（C4）

| 检测器 | 文件 | 后端 |
|---|---|---|
| `OnnxDetector` | `stages/detectors/onnx_detector.py` | 包装现有 `ai/onnx_yolo.py`，实现 Detector 接口 |
| `Yolo26Detector` | `stages/detectors/yolo26_detector.py` | 包装 `ai/yolo26_backend.py` |
| `UhdDetector` | `stages/detectors/uhd_detector.py` | 包装 UHD 人体检测 |

每个真实检测器是 `Detector` 子类，是**唯一** import 其模型库（onnxruntime/ultralytics）的模块。DetectorStage 永不改变。

### 9.2 更多 Pipeline 阶段（C4+）

| 阶段 | 读 | 写 |
|---|---|---|
| `CaptureStage` | FrameSource.read() | `context.frame` |
| `TrackingStage` | `context.detections` | `context.tracks` |
| `TargetMgmtStage` | `context.tracks` | `context.targets` |
| `ProjectionStage` | `context.targets` | `context.target_states` |

完整链：`CaptureStage → DetectorStage → TrackingStage → TargetMgmtStage → ProjectionStage`，每阶段包装一个 VisionCore 抽象，全用接口注入。

### 9.3 多检测器链（C5）

Pipeline 可链式多个 DetectorStage（人体检测 + 头部属性 + 手势），`extend` 语义让输出累积。每个检测器独立初始化/关闭，互不干扰。

---

## 10. 后续里程碑预告

- **C4**：真实检测器接入（OnnxDetector/Yolo26Detector 包装现有 ai/ 后端）+ TrackingStage
- **C5**：完整 Pipeline 链（Capture→Detect→Track→TargetMgmt→Project）+ 多检测器链
- **C6**：InferWorker 影子旁路——Pipeline 与现有 InferWorker 并行运行对比验证
- **D**：InferWorker 退役，Pipeline 转正
