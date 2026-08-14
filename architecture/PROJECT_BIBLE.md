# PROJECT_BIBLE

> 生成时间：2026-08-10
> 生成依据：F:\VisionBata 全部核心源码逐文件阅读，不参考任何历史报告
> 版本：1.1.1

## 项目定位

VisionBata（VisionDataPlatform, VDP）是一个桌面级实时多摄像头视觉检测平台。它在一个 PyQt6 桌面应用中集成了多路摄像头采集、YOLO 目标检测、卡尔曼滤波跟踪、目标生命周期管理、手势/头部朝向属性识别、矩形区域检测和事件驱动通信，面向需要在单机上同时监控多路视频流并实时输出结构化检测结果的场景。

应用版本号 1.1.1，组织名 VisionBata，在 `main.py` 中通过 `app.setApplicationName("VisionDataPlatform")` 和 `app.setApplicationVersion("1.1.1")` 设定。

## 技术栈

| 层级 | 技术 | 版本要求 | 用途 |
|------|------|----------|------|
| GUI 框架 | PyQt6 | >=6.2.0 | 主界面、信号槽通信、多线程 |
| 数值计算 | numpy | >=1.21.0 | 矩阵运算、滤波器实现 |
| 科学计算 | scipy | >=1.11.0 | 匈牙利匹配、线性代数 |
| 视频采集 | av (PyAV) | >=10.0.0 | DirectShow 摄像头抓帧、H.264 编码录制 |
| 图像处理 | opencv-python | >=4.5.0 | 图像变换、降噪、可视化叠加 |
| 目标检测 | ultralytics | >=8.0.0 | YOLOv8/YOLO26 PyTorch 推理 |
| 深度学习 | torch | >=2.0.0 | PyTorch 后端模型加载与推理 |
| 手势识别 | mediapipe | >=0.10.0 | 手部关键点检测 |
| ONNX 推理 | onnx | >=1.22.0 | ONNX 模型格式支持 |
| GPU 加速 | onnxruntime-directml | >=1.24.0 | DirectML GPU 加速推理 |

运行环境为 Windows，使用 DirectShow 进行摄像头采集，DirectML 进行 GPU 推理加速。

## 双架构并行

项目同时维护两套架构，这是理解整个项目的关键。

### 现有运行架构（ai/ + gui/ + camera/ + common/）

这是当前实际运行的生产架构。它是一个紧耦合的 PyQt6 桌面应用，核心是 `InferWorker`——一个共享的 QThread，驱动整个推理循环。

数据流：`camera/capture.py` 从多路摄像头抓帧 → 存入 `InferWorker` 的 per-slot 缓冲区 → `InferWorker.run()` 循环消费缓冲区，对每个 slot 执行检测推理（YOLO）+ 跟踪更新（UKF）→ 通过 `detection_ready` PyQt 信号将结果回传主线程 → `gui/main_window.py` 接收信号并分发到对应的 `CameraCell` 渲染。

这套架构的特点是：所有逻辑通过 PyQt6 信号槽串联，状态分散在各个对象中，线程间通信依赖 Qt 的事件循环。它的优点是实时性好、延迟低；缺点是难以单元测试、模块间耦合紧密。

### VisionCore 新架构（visioncore/）

这是正在逐步替换现有架构的模块化设计。它将检测-跟踪-目标管理-事件发射拆解为流水线阶段（PipelineStage），通过共享上下文（PipelineContext）传递数据，通过 EventBus 发布生命周期事件，通过 TargetManager 管理目标状态机。

VisionCore 当前以影子模式（shadow mode）运行：`visioncore/pipeline/shadow.py` 将现有架构的数据转换为 VisionCore 格式，送入 VisionCore 流水线并行处理，但不干扰现有架构的输出。这允许在不影响生产功能的前提下验证新架构的正确性。

### 桥接层

`visioncore/pipeline/adapters.py` 提供从现有架构到 VisionCore 的数据适配，`visioncore/core/adapters.py` 提供数据模型级别的转换（如现有 Detection → VisionCore Detection），`visioncore/pipeline/migration.py` 提供迁移工具。

## 现有运行架构详解

### InferWorker（ai/inference.py）

InferWorker 是整个现有架构的心脏，继承自 QThread。它管理多路摄像头的推理缓冲区，在单个线程中轮转处理所有 slot 的帧。

**核心循环。** `run()` 方法先执行 `_warmup()` 预热模型，然后进入主循环：检查是否需要切换模型（`_switch_model_if_needed`），在 `_buffer_lock` 保护下取出所有 slot 的待处理帧并清空缓冲区，然后逐个 slot 处理。

**推理步幅。** `InferWorker` 支持推理步幅（`_infer_stride`）：每隔 N 帧执行一次完整推理，中间帧使用 `_run_prediction` 进行预测（仅靠卡尔曼滤波器预测目标位置，不执行检测）。这是性能优化手段——检测是计算密集的，而滤波器预测是轻量的。

**多后端支持。** InferWorker 支持多种推理后端：
- ONNX Runtime（DirectML GPU 加速）——通过 `ai/onnx_yolo_backend.py`
- YOLO26 PyTorch——通过 `ai/yolo26_backend.py`
- UHD 轻量人体检测——通过 `ai/uhd_backend.py`（64x64 静态 ONNX 模型）

**属性检测。** 除了主检测，InferWorker 还集成了手势识别（`gesture_recognizer.py`、`hand_gesture_recognizer.py`）、头部分类（`head_classifier.py`）和矩形检测（`rectangle_detector.py`），这些作为检测后处理附加到结果中。

**健康监控。** InferWorker 内置健康监控机制，通过 `inference_controller.py` 进行模型切换、推理步幅调整和异常恢复。

### DetectionTracker（ai/tracker.py）

DetectionTracker 是纯 NumPy 实现的目标跟踪器，不依赖任何外部跟踪库。它实现了完整的检测-跟踪管线：预测 → 匹配 → 更新 → 创建 → 删除。

**滤波器家族。** 支持三种 Unscented Kalman Filter 变体：
- UKF（标准无迹卡尔曼滤波）
- MCUKF（ Measurement-Conditioned UKF，根据观测质量自适应调整滤波器参数）
- ManifoldUKF（流形 UKF，用于非欧状态空间）

通过 `_FilterAdaptor` 适配器统一接口，可以在运行时切换滤波器类型。

**匹配策略。** 使用匈牙利算法（`scipy.optimize.linear_sum_assignment`）基于 IoU 进行检测-轨迹匹配。`_iou_threshold`（默认 0.25）控制匹配阈值，`_max_misses`（默认 4）控制轨迹的连续丢失帧数上限——超过此值轨迹被删除。

**ROI 重检测。** 当轨迹丢失时，在轨迹的最后已知位置周围执行 ROI 重检测（通过 `pipeline/stages/redetect/` 实现），以尝试恢复丢失的目标。

### 摄像头系统（camera/）

**enumerator.py** 枚举系统中的 DirectShow 摄像头设备，返回设备列表供 GUI 选择。

**capture.py** 实现多路摄像头采集。使用 PyAV 的 DirectShow 输入格式，为每个 slot 创建独立的采集线程。帧通过回调送入 `InferWorker` 的缓冲区。支持帧率控制和分辨率配置。

**recorder.py** 实现视频录制，使用 H.264 编码将摄像头画面写入 MP4 文件。录制文件以 `slot_{N}_{timestamp}.mp4` 格式命名。

### GUI 系统（gui/）

**main_window.py** 是主窗口，负责 UI 布局（4 路 CameraCell 网格）和信号槽调度。它接收 `InferWorker` 的 `detection_ready` 信号，将检测结果分发到对应的 CameraCell 渲染。同时管理设置对话框、录制控制、模型切换等用户交互。

**camera_cell.py** 是单摄像头画面单元格，负责渲染视频帧、绘制检测框和跟踪轨迹叠加层。

**settings_dialog.py** 提供模型选择、推理参数、摄像头配置的设置界面。

**theme.py** 定义全局主题和配色方案，**animations.py** 提供淡入淡出和脉冲反馈动画工具，**safe_widgets.py** 提供防崩溃的组件包装器，**splash_screen.py** 实现启动闪屏动画。

### 公共抽象层（common/）

`common/__init__.py` 定义了三个后端抽象接口：`CaptureBackend`（采集后端）、`DetectionBackend`（检测后端）、`TrackingBackend`（跟踪后端）。`pyav_capture.py` 和 `yolo_detector.py` 分别是 PyAV 采集和 YOLO 检测的具体实现。这层抽象允许在不修改上层代码的情况下替换底层实现。

## VisionCore 架构详解

### 核心数据模型（visioncore/core/）

VisionCore 的核心数据模型是一组 frozen + slotted 的 dataclass，设计为不可变、内存紧凑、线程安全。

**Detection（detection.py）** 由 `BBox`（归一化边界框：cx, cy, width, height）+ `score`（置信度）+ `class_id`（类别 ID）+ `label`（类别名称）构成。这是检测阶段输出的最小单元。

**Track（track.py）** 聚合一个 `track_id` + 一个 `Detection` + 滤波器状态。它回答"什么在移动、在哪里"的问题。Track 的生命周期通常较短——一个目标可能经历多个 Track 生命周期（丢失后重新创建）。

**Target（target.py）** 是最高层级的实体。它聚合 Track，携带应用级优先级和状态，持有可扩展的属性字典。Target 回答"我们在看什么、为什么看"的问题。Target 的 `target_id` 格式为 `S{slot_id}-T{NNNN}`（如 `S0-T0001`），全局唯一。TargetState 是一个 5 值枚举：

```
ACTIVE --(lost too long)----> LOST
LOST   --(re-detected)------> RECOVERED --> ACTIVE
ACTIVE --(locked)-----------> LOCKED
LOCKED --(released)---------> ACTIVE
any    --(removed)----------> REMOVED  (terminal)
```

**Frame（frame.py）** 是视频帧的抽象：`frame_id` + `timestamp` + `camera_id` + `ndarray`（像素数据）。

**Event（event.py）** 是事件记录：`event_type`（点分命名字符串，如 `"target.lost"`）+ `timestamp` + `payload`（事件特定数据字典）。Event 是系统审计和反应日志的基础单元。

**adapters.py** 提供现有架构与 VisionCore 之间的数据模型转换函数。

### 流水线系统（visioncore/pipeline/）

**PipelineStage（base.py）** 是所有处理阶段的抽象基类，定义 4 方法生命周期：`initialize()`（一次性资源获取，幂等）、`process(context)`（每帧处理，读写上下文）、`shutdown()`（一次性资源释放，逆序调用，幂等，不可抛出）、`health_check()`（无副作用就绪探针）。

**Pipeline（pipeline.py）** 是有序阶段执行器。`add_stage` 在初始化前添加阶段，`initialize` 顺序调用每个阶段的 `initialize`，`run` 顺序调用每个阶段的 `process`，`shutdown` 逆序调用每个阶段的 `shutdown`。Pipeline 支持上下文管理器协议（`with p: p.run(ctx)`），确保 `shutdown` 总是被调用。

**PipelineContext（context.py）** 是共享上下文，携带：`frame`（当前帧）、`detections`（检测列表）、`tracks`（轨迹列表）、`targets`（目标列表）、`target_states`（状态快照列表）、`timestamp`、`metadata`。阶段从上游字段读取、向下游字段写入。

**阶段实现（stages/）：**

| 阶段 | 输入 | 输出 | 说明 |
|------|------|------|------|
| DetectorStage | context.frame | context.detections | 调用检测器后端，输出 Detection 列表 |
| TrackerStage | context.detections | context.tracks | 调用跟踪器，输出 Track 列表 |
| TargetStage | context.tracks | context.targets + context.target_states | 调用 TargetManager，输出 Target 和 TargetState 快照 |
| EventStage | context.target_states | (无，sink) | 为每个 TargetState 发布一个 Event 到 EventBus |
| DenoiseStage | context.frame | context.frame (原地) | 双边滤波 / NL-Means 降噪 |
| AttributeGestureStage | context.detections | context.detections (增强) | 手势/头部朝向属性检测 |
| RectangleFilterStage | context.detections | context.detections (过滤) | 矩形区域过滤 |
| RedetectStage | context.targets | context.detections (补充) | ROI 重检测 |

**影子流水线（shadow.py）。** ShadowPipeline 以并行模式运行 VisionCore 流水线，从现有架构的数据中提取输入，但不将输出回写到现有架构。这允许在无风险的环境下验证新架构的正确性和性能。

**配置系统（config/pipeline_config.py）。** 从 JSON 文件解析流水线配置，定义阶段顺序和参数。配置文件位于 `config/visioncore_pipeline.json`。

### 事件总线（visioncore/eventbus/）

EventBus 是线程安全的发布/订阅中心，详细审计见同目录下的 EventBus 审计报告。核心要点：

- **锁内快照、锁外分发**：RLock 保护注册表，回调在锁外执行
- **双模订阅**：字符串事件类型匹配 + isinstance 类型安全匹配
- **异常隔离**：单个订阅者故障不影响其他订阅者
- **类型化事件层级**：BaseEvent + 5 个 Target 生命周期事件子类
- **DebugEventLogger**：可开关的全事件调试日志记录器

### 目标管理系统（visioncore/target_manager/）

**TargetManager（manager.py）** 是组合根，协调三个组件：
- `TargetStore`——线程安全 CRUD 容器
- `TargetLifecycleManager`——纯状态机（6 条转换规则）
- `EventBus`——可选的生命周期事件发布器

TargetManager 添加了：ID 生成（`S{slot}-T{NNNN}` 格式的单调计数器）、Store+Lifecycle 协调（转换由 Lifecycle 验证后持久化到 Store）、生命周期事件发布（通过单一出口 `_emit_lifecycle_event`）、批量对账（`update_targets` 一次调用同步目标集与新轨迹列表）、查询辅助（`get_active_targets` 等按状态过滤）。

**TargetLifecycleManager（lifecycle.py）** 是无状态纯状态机，定义 6 条转换规则的不可变表。`transition` 方法是 `@final` 的——子类不应覆盖它，而应覆盖命名方法（`mark_lost`、`lock` 等）。`can_transition`、`get_valid_transitions`、`is_terminal` 是 classmethod，不需要实例化即可查询。

**TargetStore（store.py）** 是线程安全的 CRUD 容器，使用自己的锁保护内部字典。

**converters.py** 提供 Track → Target 和 Target → TargetState 的转换函数。**target_id_factory.py** 是无状态的 ID 格式化器，`create_target_id(slot_id, counter)` 返回 `S{slot}-T{NNNN:04d}` 格式的字符串。

### 协议层（visioncore/protocol/）

协议层负责将 TargetState 快照传输到外部消费者。

**ProtocolAdapter（base.py）** 是抽象基类，定义 `publish` 和 `health_check` 接口。

**具体适配器：**
- `ConsoleAdapter`——输出到控制台（调试用）
- `NullAdapter`——丢弃所有数据（空操作）
- `UDPAdapter`——通过 UDP 数据报传输

**序列化（serialization/）：**
- `JSONSerializer`——JSON 文本序列化
- `CBORSerializer`——CBOR 二进制序列化（更紧凑、更快）

**发布器（publisher/）：**
- `StatePublisher`——将 TargetState 快照通过 ProtocolAdapter 发布
- `ShadowPublisher`——影子模式发布器，用于并行测试

### 插件系统（visioncore/plugin/）

**Plugin（base.py）** 是插件抽象基类，携带 `PluginMetadata`（名称、版本、描述）。**PluginRegistry（registry.py）** 是插件注册表，管理插件的注册、查询和生命周期。**Processor（processors/processor.py）** 是流水线阶段扩展点，允许第三方代码在不修改核心代码的情况下插入自定义处理阶段。

### 运行时管理（visioncore/runtime/）

**ModelRuntime（model/model_runtime.py）** 管理模型的加载、卸载和健康检查。**RuntimeState（state/runtime_state.py）** 是运行时状态快照，记录当前系统状态（活跃模型、处理帧率、内存使用等）。

### 输入源（visioncore/source/）

**FrameSource（base.py）** 是输入源抽象基类，定义 `open`/`read`/`close` 接口。**frame_source.py** 是具体实现，从摄像头或视频文件读取帧并转换为 Frame 对象。**dummy_source.py** 是测试用的虚拟输入源，生成合成帧。

### 状态快照（visioncore/state/）

**TargetState（target_state.py）** 是目标的投影快照，包含归一化坐标（cx, cy, width, height）、速度（vx, vy）、置信度、标签、时间戳、摄像头 ID 和元数据。它是 TargetStage 的输出、EventStage 的输入、ProtocolAdapter 的传输单元——是 VisionCore 数据流中"对外可见"的目标表示。

## 应用入口

`main.py` 是应用入口，执行以下步骤：

1. 安装全局异常钩子（`setup_global_exception_hook`），将未捕获异常写入 `error.log`
2. 配置高 DPI 缩放（`QT_ENABLE_HIGHDPI_SCALING=1`，`PassThrough` 舍入策略）
3. 创建 QApplication，设置 Segoe UI Variable Text 字体、Fusion 样式
4. 显示启动闪屏（SplashScreen）
5. 创建主窗口（MainWindow），闪屏结束后淡入显示
6. 进入 Qt 事件循环

`demo.py` 是一个最小可运行示例，演示单摄像头实时视觉闭环，不依赖完整 GUI，适合快速验证检测和跟踪功能。

## 测试体系

项目拥有 35 个测试文件，覆盖 VisionCore 架构的每个模块。测试不依赖 pytest（每个文件都有 `__main__` runner），可以独立运行。测试文件使用 Dummy 桩（DummyDetector、DummyTracker、DummyTargetManager、DummyEventBus、DummySource）隔离被测模块。

测试运行方式：
```
python tests/test_event_bus.py           # 独立运行
python -m pytest tests/ -v               # pytest 运行
```

`benchmark/benchmark_serializer.py` 提供 JSON vs CBOR 序列化性能基准。`tools/mot_eval.py` 提供多目标跟踪精度评测工具。

## 数据模型层级关系

```
Frame ──DetectorStage──> Detection ──TrackerStage──> Track ──TargetStage──> Target
 │                                                                         │
 │                                                                    TargetState (快照)
 │                                                                         │
 └─────────────────────────────────────────────────── EventStage ──> Event
                                                                         │
                                                                  EventBus.publish
```

Detection 是最底层的检测输出。Track 聚合 Detection 并添加跟踪 ID 和滤波器状态。Target 聚合 Track 并添加应用级状态和优先级。TargetState 是 Target 的投影快照，剥离了内部状态，只保留对外可见的坐标、速度和元数据。Event 是对 TargetState 变化的审计记录，通过 EventBus 发布给订阅者。

## 关键设计模式

**组合根（Composition Root）。** TargetManager 是典型的组合根——它在构造函数中组装 Store + Lifecycle + EventBus，对外暴露统一的接口，内部协调三个组件的交互。

**依赖倒置（Dependency Inversion）。** EventStage 定义自己的 EventBus ABC（最小 `publish` 接口），具体总线是注入的。PipelineStage 定义抽象接口，具体阶段是注入的。FrameSource 定义抽象接口，具体输入源是注入的。

**策略模式（Strategy）。** DetectionTracker 的 `_FilterAdaptor` 允许在运行时切换 UKF/MCUKF/ManifoldUKF 滤波器。InferWorker 支持在运行时切换 ONNX/YOLO26/UHD 推理后端。

**观察者模式（Observer）。** EventBus 是经典的观察者模式实现——生产者 publish 事件，消费者 subscribe 回调，总线负责路由和投递。

**影子模式（Shadow）。** ShadowPipeline 以并行模式运行新架构，不影响生产架构——这是一种渐进式迁移策略，允许在验证新架构正确性的同时保持系统稳定运行。

## 文件依赖关系

```
main.py
  └── gui/main_window.py
        ├── gui/camera_cell.py
        ├── gui/settings_dialog.py
        ├── gui/theme.py, animations.py, safe_widgets.py, splash_screen.py
        ├── ai/inference.py (InferWorker)
        │     ├── ai/tracker.py (DetectionTracker)
        │     ├── ai/onnx_yolo_backend.py / yolo26_backend.py / uhd_backend.py
        │     ├── ai/gesture_recognizer.py / hand_gesture_recognizer.py
        │     ├── ai/head_classifier.py
        │     ├── ai/rectangle_detector.py
        │     ├── ai/reid_backend.py
        │     ├── ai/inference_controller.py
        │     └── ai/model_task.py
        └── camera/capture.py
              ├── camera/enumerator.py
              ├── camera/recorder.py
              └── common/pyav_capture.py

visioncore/ (独立于现有架构，通过 adapters 桥接)
  ├── core/ (数据模型)
  ├── pipeline/ (流水线 + 阶段)
  │     └── stages/event_stage.py → eventbus/
  ├── eventbus/ (事件总线)
  ├── target_manager/ (目标管理 → eventbus/)
  ├── protocol/ (外部通信)
  ├── plugin/ (插件系统)
  ├── runtime/ (运行时管理)
  ├── source/ (输入源)
  └── state/ (状态快照)
```
