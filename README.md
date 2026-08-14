# VisionBata (VisionDataPlatform · VDP)

多路视频智能分析平台。基于 PyQt6 图形界面与 YOLO 检测体系，支持多路相机实时采集、目标检测 / 跟踪 / 识别、事件发布与可扩展的图像处理流水线。

> 当前项目遵循 MIT 许可，属于**私有**仓库，仅供内部团队使用。

## 功能特性

- **多路相机采集**：多路视频源并行采集与录制，支持 RTSP / 本地文件等输入。
- **目标检测与跟踪**：基于 YOLOv8 / YOLO26 系列，支持目标检测、关键点姿态、旋转框（OBB）与 ReID 跟踪。
- **多模型后端**：模型按文件名自动识别，支持 ONNX（OnnxYoloBackend）与 PyTorch（Yolo26Backend）两种推理后端，可无缝切换。
- **可扩展流水线**：`VisionCore` 新架构提供模块化的流水线（Pipeline）/ 阶段（Stage）体系，内置去噪、属性手势识别、事件、目标状态、重检等多个阶段，支持插件注册与影子（Shadow）旁路验证。
- **事件总线**：通过 Qt 信号-槽解耦的 EventBus，实现采集线程、推理线程、GUI 主线程与录制线程之间的异步通信。
- **多协议发布**：支持 JSON / CBOR 序列化，提供 UDP、Console 等适配器将目标状态对外发布。

## 架构

项目包含两套并行架构：

- **现有运行架构**：`ai/`、`gui/`、`camera/`、`common/`，当前实际运行的主链路。
- **VisionCore 新架构**：`visioncore/`，处于"影子旁路"阶段，不直接影响现有逻辑，用于逐步重构与验证。

**线程模型**：1 个 GUI 主线程 + 1 个共享推理线程 + N 个采集线程 + 视频录制线程池，通过 Qt 信号-槽机制解耦。

```
gui/       PyQt6 图形界面（主窗口、相机格、主题、动画）
camera/    相机采集 / 枚举 / 录制
ai/        推理控制、多模型后端、检测、跟踪、手势识别
common/    通用工具（PyAV 采集、YOLO 检测器）
visioncore/ 模块化流水线 / 事件总线 / 协议 / 插件 / 目标管理
config/    运行配置
models/    模型权重文件
tests/     pytest 单元测试
docs/      设计与审计文档
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行主程序
python main.py
```

> 提示：MediaPipe 手势识别在当前版本因兼容性问题默认关闭。

## 测试

```bash
pytest tests/
```

## 目录说明

- `docs/`：各阶段的设计、审计与里程碑报告。
- `architecture/`：项目架构梳理、事件总线审计与全量档案文档。

## 许可

本项目以 [MIT](./LICENSE) 许可开源。