# OminVisionFlowSight

> 桌面级多路视频智能分析平台 —— 在一台普通 PC 上，同时接入多路摄像头，实时完成目标检测、跟踪、姿态与属性识别，并将结构化结果对外发布。

**OminVisionFlowSight**（内部代号 *VisionDataPlatform*，简称 VDP）是一个基于 **PyQt6** 图形界面与 **YOLO** 检测体系的桌面应用。它通过"共享推理线程 + 跳帧卡尔曼预测"的设计，在 CPU / 集成显卡（DirectML）上即可流畅运行多路实时视觉分析，并提供模块化、可扩展的图像处理流水线架构（VisionCore）。

---

## 目录

- [项目背景](#项目背景)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [测试](#测试)
- [许可证](#许可证)
- [贡献指南](#贡献指南)
- [致谢](#致谢)

---

## 项目背景

在安防监控、工业质检、人机交互、出入口管理等场景中，多路视频的实时 AI 分析通常依赖 GPU 服务器集群，部署成本高、门槛高。本项目要解决的正是这些问题：

1. **多路摄像头同时 AI 分析**：通过共享单一推理线程，最多 4 路视频源并行采集与推理，避免 GPU/CPU 资源争抢，单台 PC + USB 摄像头即可运行。
2. **检测与跟踪一体化**：YOLO 检测 + 多滤波器卡尔曼跟踪，形成完整的目标生命周期管理（出现 → 跟踪 → 预测 → 丢失 → 重找回）。
3. **多模型后端灵活切换**：同一工程内支持 YOLOv8 / YOLO26（PyTorch）、ONNX Runtime + DirectML、UHD 轻量人体检测等多种后端，根据模型文件名自动识别。
4. **跳帧预测降低算力需求**：推理帧间隔（`infer_stride`）机制 + 卡尔曼预测填充中间帧，在维持视觉流畅度的同时大幅降低推理频率。
5. **目标消失后重找回**：基于卡尔曼预测的 ROI 裁剪重检测，自动找回被遮挡后重新出现的目标。
6. **运行时异常自适应恢复**：`_FilterAdaptor` + `_HealthMonitor` 双重机制，运行时自动检测并修正算法异常。

## 功能特性

- 📷 **多路相机采集**：多路视频源并行采集与录制，支持 RTSP / IP 摄像头 / USB 摄像头 / 本地文件等输入（PyAV DirectShow），多路 H.264 MP4 同步录像。
- 🎯 **多维度目标检测**：
  - YOLOv8 / YOLO26 通用目标检测（COCO 80 类）
  - OBB 有向目标检测（旋转框）
  - UHD 64×64 轻量人体检测（CPU 优先）
  - 矩形物体检测（Canny + 轮廓 + 线性卡尔曼跟踪）
- 🧭 **多目标跟踪**：纯 NumPy 实现的卡尔曼滤波体系，支持 UKF / MCUKF / Manifold UKF 与 AUTO 自适应切换，OBB polygon 关联跟踪。
- 🦴 **姿态与手势**：17 点 COCO 人体骨架绘制；基于几何规则的身体姿态手势识别（举手 / 摔倒 / 蹲下 / 双臂展开等）。
- 🧢 **头部属性识别**：帽子 / 口罩 / 墨镜 / 眼睛开闭 / 嘴巴开闭五维属性分类。
- 🌫️ **图像增强**：双边滤波 / NL-Means 去噪预处理，可选 ESPCN ×4 小目标超分辨率增强。
- 🔌 **可扩展流水线（VisionCore）**：模块化的 Pipeline / Stage 体系，内置去噪、检测、跟踪、目标状态、重检、事件等阶段，支持插件注册（Plugin Registry）与影子（Shadow）旁路验证，可在不影响主链路的情况下逐步重构。
- 📡 **事件总线与协议发布**：Qt 信号槽解耦的 EventBus 贯通采集 / 推理 / GUI / 录制线程；支持 JSON / CBOR 序列化，提供 UDP、Console 等适配器将目标状态对外发布。
- 🖥️ **专业暗色 GUI**：Fusion 风格 + 动画系统的深色主题界面，支持多相机网格布局、启动画面、设置对话框与大量功能开关。
- 🧪 **完善的测试**：`tests/` 下 40+ 个 pytest 单元测试，覆盖流水线各阶段、序列化、事件总线、目标管理等模块。

> 提示：MediaPipe 手部手指手势（OK / 点赞 / 数字 1~5 / 握拳等）在当前版本因版本兼容性问题默认关闭，身体姿态手势不受影响。

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.11 | 全部实现 |
| GUI | PyQt6 | 主窗口、相机网格、设置对话框、信号槽、动画 |
| 视觉 / 图像 | OpenCV, PyAV | 采集、预处理、去噪、矩形检测、录像 |
| 检测 | ultralytics, PyTorch, ONNX Runtime (DirectML) | YOLOv8 / YOLO26 检测、OBB、姿态、UHD、CHC 属性 |
| 数值计算 | NumPy, SciPy | 卡尔曼滤波（UKF / MCUKF / Manifold UKF）等算法 |
| 协议 | 自定义 JSON / CBOR 序列化 | 目标状态的结构化发布 |
| 测试 | pytest | 单元测试 |

> 环境说明：采集链路（PyAV DirectShow）与 `onnxruntime-directml` 依赖主要面向 **Windows**。在其他操作系统上运行时，可将 `onnxruntime-directml` 替换为通用 CPU 版 `onnxruntime`，采集后端也可能需要调整。

## 快速开始

### 环境要求

- Windows 10 / 11（推荐；DirectShow 采集与 DirectML 加速）
- Python 3.11
- [Git LFS](https://git-lfs.com/)（`models/` 目录中的模型权重以 Git LFS 存储）

### 安装步骤

```bash
# 1. 克隆仓库（含 LFS 模型权重）
git clone https://github.com/1465586445-jpg/OminVisionFlowSight.git
cd OminVisionFlowSight

# 2. 拉取模型权重（未安装 git-lfs 前 models/ 下仅有指针文件）
git lfs install
git lfs pull

# 3. 创建虚拟环境并安装依赖（Windows 示例）
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 启动主程序

```bash
# 图形界面（多路相机网格）
python main.py
```

Windows 用户也可运行随附的 `start.bat`（请在运行前按本机路径调整其中的 venv 位置）：

```bat
@echo off
cd /d "%~dp0"
"%~dp0venv\Scripts\python.exe" main.py
pause
```

首次启动后，可通过 GUI 的**设置对话框**开闭各项功能（人体检测、骨骼、手势、矩形检测、去噪、滤波器类型、ROI 重检、头部属性等）。

## 使用示例

除完整 GUI 外，项目提供最小可运行 Demo（单路摄像头闭环：采集 → 显示 → YOLO 检测 → 跟踪 → 叠加显示）：

```bash
# 默认参数启动
python demo.py

# 指定摄像头与模型
python demo.py --camera "HD Webcam" --model models/yolov8n.pt --conf 0.5
```

## 配置说明

- `config/visioncore_pipeline.json`：VisionCore 新架构的流水线阶段配置，例如：

  ```json
  {
    "stages": [
      {"type": "DummyStage", "params": {"name": "capture"}},
      {"type": "DenoiseStage", "params": {"name": "denoise"}},
      {"type": "DummyStage", "params": {"name": "detect"}}
    ]
  }
  ```

- `models/`：模型权重目录。文件按命名自动匹配推理后端（`.pt` → PyTorch 后端，`.onnx` → ONNX Runtime / DirectML 后端）。
- 其余运行参数（相机列表、功能开关、滤波器类型等）通过 GUI 设置对话框调整并持久化。

## 项目结构

```
OminVisionFlowSight/
├── ai/             # 主链路：推理控制、多模型后端、检测/跟踪/手势/头部属性/矩形检测
├── camera/         # 相机采集、枚举、录制
├── common/         # 通用工具（PyAV 采集、YOLO 检测器、目标数据结构）
├── gui/            # PyQt6 图形界面（主窗口、相机格、主题、动画、设置对话框）
├── visioncore/     # 新架构：模块化流水线 / 事件总线 / 协议 / 插件 / 目标管理（影子旁路）
├── config/         # 运行配置（visioncore_pipeline.json）
├── models/         # 模型权重文件（Git LFS 存储）
├── tests/          # pytest 单元测试
├── docs/           # 各阶段的设计、审计与里程碑报告
├── architecture/   # 架构梳理、审计报告与全量档案文档
├── tools/          # 辅助工具（如 MOT 评估脚本）
├── benchmark/      # 序列化性能基准
├── main.py         # GUI 主程序入口
├── demo.py         # 最小可运行单路 Demo
├── requirements.txt
├── LICENSE         # PolyForm Noncommercial License 1.0.0
└── .gitignore
```

## 测试

```bash
pytest tests/
```

## 许可证

本项目采用 **[PolyForm Noncommercial License 1.0.0](./LICENSE)**（原文见仓库根目录 `LICENSE` 文件，官方链接：<https://polyformproject.org/licenses/noncommercial/1.0.0>）。

在遵守许可证原文的前提下，我们对许可意图做出如下说明：

- ✅ **允许**：个人学习、研究、实验、爱好项目以及其他**非商业用途**下的使用、修改、分发，以及基于本项目创建衍生作品。
- 🚫 **核心限制**：使用者**在主观上不得以盈利为目的**，**在客观上不得通过本软件或其衍生作品直接获取商业利益**。
- ❤️ **关于打赏与捐赠**：开发者因提交 Pull Request、代码贡献、修改、创建分支或维护本项目而收到的**自愿赞助、打赏、捐赠**，属于主观上不以盈利为目的的行为，**不受本许可证限制**。
- 💼 **商业使用**：任何以商业产品、付费服务、SaaS、二次销售或其他直接盈利为目的的使用，均**不被本许可证允许**。如需商业授权，请联系项目作者单独协商。

> “非商业目的”等术语的正式定义以 `LICENSE` 原文为准（包括但不限于 Personal Uses 与 Noncommercial Organizations 等相关条款）。本说明仅用于帮助理解，不构成对许可证条款的修改。

## 贡献指南

欢迎通过 **Issue** 与 **Pull Request** 参与贡献！

1. Fork 本仓库；
2. 基于 `main` 创建特性分支（如 `feat/your-feature`）；
3. 提交代码并推送到你的 Fork，注意提交信息清晰规范；
4. 向本仓库发起 Pull Request，描述变更动机、实现方式与测试情况。

请注意：

- 参与贡献前请确认你已阅读并同意 [许可证](#许可证) 条款；
- 本项目因采用 **PolyForm Noncommercial License 1.0.0**，你的贡献同样遵循上述**非商业使用限制**——请勿将本软件或其衍生作品用于任何以盈利为目的的场景；
- 为开发者提供的**自愿赞助、打赏或捐赠**不受此限制（详见上文许可证说明）；
- 提交的代码应尽量附带或更新相应测试（`pytest tests/`）。

## 致谢

- [Ultralytics](https://github.com/ultralytics) — YOLOv8 / YOLO26 检测与姿态模型；
- [PINTO0309](https://github.com/PINTO0309) — 集成的 UHD 轻量人体检测与 CHC 头部属性 ONNX 模型；
- [PolyForm Project](https://polyformproject.org/) — PolyForm Noncommercial License 1.0.0；
- 以及所有在 `docs/` 与 `architecture/` 中留下设计与审计文档的贡献者。

## 截图

> 待补充：项目运行效果与 GUI 界面截图将在此处展示。

*（当前仓库暂无演示截图，欢迎通过 Issue 或 PR 分享你的运行截图。）*