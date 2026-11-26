# OminVisionFlowSight

> 桌面级多路视频智能分析平台 —— 在一台普通 PC 上，同时接入多路摄像头，实时完成目标检测、跟踪、姿态与属性识别，并将结构化结果对外发布。

[![License](https://img.shields.io/badge/license-PolyForm%20Noncommercial-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.11--3.12-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/tracker%20core-Rust%20(PyO3)-orange.svg)](./rust/omintrack)
[![Platform](https://img.shields.io/badge/platform-Windows-lightgrey.svg)](#技术栈)

**English**: OminVisionFlowSight is a desktop multi-camera video analytics platform that runs on a single PC — real-time YOLO detection, multi-filter Kalman tracking (with an optional **Rust core**, ~37x faster in benchmark), pose/gesture and head-attribute recognition, structured event publishing, and a modular VisionCore pipeline, all behind a PyQt6 GUI. No GPU server required: it runs on CPU or a DirectML iGPU.

---

## 目录

- [项目概览](#项目概览)
- [演示截图](#演示截图)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [配置说明](#配置说明)
- [项目结构](#项目结构)
- [测试](#测试)
- [文档](#文档)
- [许可证](#许可证)
- [贡献指南](#贡献指南)
- [致谢](#致谢)

---

## 项目概览

在安防监控、工业质检、人机交互、出入口管理等场景中，多路视频的实时 AI 分析通常依赖 GPU 服务器集群，部署成本高、门槛高。OminVisionFlowSight（内部代号 *VisionDataPlatform*，简称 VDP；QSettings 组织名与部分历史文档中的 *VisionBata* 为既有配置兼容保留，勿改名以免丢失用户配置）通过**共享推理线程 + 跳帧卡尔曼预测**的设计，在 CPU / 集成显卡（DirectML）上即可流畅完成最多 4 路视频源的并行采集、推理、跟踪与同步录像：

- **检测与跟踪一体化**：YOLO 检测 + 多滤波器卡尔曼跟踪，覆盖目标完整生命周期（出现 → 跟踪 → 预测 → 丢失 → 重找回）；
- **跟踪核心提供 Rust 实现（omintrack）**：与 Python 版逐帧等价，一行参数即可切换；
- **多模型后端灵活切换**：YOLOv8 / YOLO26（PyTorch）、ONNX Runtime + DirectML、UHD 轻量人体检测，按模型文件名自动识别；
- **模块化 VisionCore 架构**：流水线 / 事件总线 / 协议发布，便于二次开发与集成。

## 演示截图

> 🚧 待补充：计划展示（1）多路相机网格主界面、（2）检测 / 跟踪 / 骨架叠加效果、（3）设置对话框。
> 运行 `python main.py` 后截图，放入 `docs/images/` 并在本节引用即可（欢迎通过 Issue 或 PR 分享你的运行截图）。

## 功能特性

- 🚀 **Rust 跟踪内核（新）**：跟踪核心（UKF 滤波族 / AUTO 自适应 / 匹配引擎 / 重检测与生命周期管理）的 Rust 重写版，经 PyO3 暴露为 `omintrack.RustTracker`。与 Python 版**逐帧等价**（18 个等价性测试，数值容差 1e-6，ID / 状态机精确一致），基准实测**平均约 37x 加速**（20 tracks x 500 帧，3 滤波器 x 3 匹配策略）。默认 **auto**： omintrack 可导入即用 Rust，否则自动回退纯 Python（零构建开箱即用）；也可在设置对话框或经 `OMINTRACK_BACKEND` 环境变量手动指定，调用方零改动。
- 📷 **多路相机采集**：多路视频源并行采集与录制，支持 RTSP / IP 摄像头 / USB 摄像头 / 本地文件等输入（PyAV DirectShow），多路 H.264 MP4 同步录像。
- 🎯 **多维度目标检测**：
  - YOLOv8 / YOLO26 通用目标检测（COCO 80 类）
  - OBB 有向目标检测（旋转框，.pt 与 ONNX / DirectML 均支持）
  - UHD 64x64 轻量人体检测（CPU 优先）
  - 矩形物体检测（Canny + 轮廓 + 线性卡尔曼跟踪）
- 🧭 **多目标跟踪**：UKF / MCUKF / Manifold UKF 滤波族 + AUTO 自适应切换，OBB polygon 关联跟踪；跳帧（`infer_stride`）卡尔曼预测填充中间帧；目标消失后基于预测的 ROI 重检测自动找回。
- 🦴 **姿态与手势**：17 点 COCO 人体骨架绘制；基于几何规则的身体姿态手势识别（举手 / 摔倒 / 蹲下 / 双臂展开等），以及基于 MediaPipe Hands 的手部手指手势识别（OK / 点赞 / 数字 1~5 / 握拳等）。
- 🧢 **头部属性识别**：帽子 / 口罩 / 墨镜 / 眼睛开闭 / 嘴巴开闭五维属性分类。
- 🌫️ **图像增强**：双边滤波 / NL-Means 去噪预处理，可选 ESPCN x4 小目标超分辨率增强。
- 🔌 **可扩展流水线（VisionCore）**：模块化 Pipeline / Stage 体系，内置去噪、检测、跟踪、目标状态、重检、事件等阶段，支持插件注册（Plugin Registry）与影子（Shadow）旁路验证，可在不影响主链路的情况下逐步重构。
- 📡 **事件总线与协议发布**：Qt 信号槽解耦的 EventBus 贯通采集 / 推理 / GUI / 录制线程；支持 JSON / CBOR 序列化，提供 UDP、Console 等适配器将目标状态对外发布。
- 🖥️ **专业暗色 GUI**：Fusion 风格 + 动画系统的深色主题界面，支持多相机网格布局、启动画面、设置对话框与大量功能开关。
- 🧪 **完善的测试**：`tests/` 下 1429 个 pytest 用例，覆盖流水线各阶段、序列化、事件总线、跟踪器行为快照与 Python/Rust 等价性。

> 提示：手部手势代码使用 MediaPipe 旧式 Solutions API，依赖版本限定为 `mediapipe>=0.10.14,<0.10.15`；当前验证的 0.10.35 与 1.0.1 已移除该 API，升级前需先迁移至 Tasks API。

## 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| 语言 | Python 3.11 / Rust | 应用主体 / 跟踪内核 |
| GUI | PyQt6 | 主窗口、相机网格、设置对话框、信号槽、动画 |
| 视觉 / 图像 | OpenCV, PyAV | 采集、预处理、去噪、矩形检测、录像 |
| 检测 | ultralytics, PyTorch, ONNX Runtime (DirectML) | YOLOv8 / YOLO26 检测、OBB、姿态、UHD、CHC 属性 |
| 跟踪内核 | Rust + PyO3 + nalgebra（maturin 构建） | UKF / MCUKF / Manifold UKF、匹配引擎、轨迹生命周期 |
| 数值计算 | NumPy, SciPy | Python 侧滤波参考实现与帧相关计算 |
| 协议 | 自定义 JSON / CBOR 序列化 | 目标状态的结构化发布 |
| 测试 | pytest | 单元测试、golden 快照、Python/Rust 等价性对拍 |

> 环境说明：采集链路（PyAV DirectShow）与 `onnxruntime-directml` 依赖主要面向 **Windows**。其他操作系统可运行检测 / 跟踪核心逻辑，但采集与 DirectML 加速需自行适配。

## 快速开始

> **Python 版本**：需要 **Python 3.11 或 3.12**。依赖 `mediapipe>=0.10.14,<0.10.15` 仅提供 cp39–cp312 预编译包，Python 3.13+ 会在 `pip install` 阶段直接失败。

### 1. 克隆与安装

````bash
git clone https://github.com/EclipticAxis/OminVisionFlowSight.git
cd OminVisionFlowSight

# 创建虚拟环境
python -m venv venv
```

激活虚拟环境（按终端类型二选一）：

```powershell
# PowerShell（Windows 11 默认终端）
venv\Scripts\Activate.ps1
# 若提示“禁止运行脚本”，先执行：Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

```bat
:: CMD
venv\Scripts\activate.bat
```

安装依赖（未激活虚拟环境时，可用 `venv\Scripts\python.exe -m pip` 代替 `pip`）：

````bash
pip install -r requirements.txt
```

### 2. 启动主程序

````bash
# 图形界面（多路相机网格）；虚拟环境已激活时
python main.py

# 未激活时，用仓内解释器直接启动（效果相同，可避开激活问题）
venv\Scripts\python.exe main.py
```

Windows 用户也可直接运行随附的 `start.bat`。脚本通过 `%~dp0` 定位自身目录及仓内 `venv`，移动项目目录后无需修改路径。**注意**：`venv/` 不随仓库分发，新克隆后需先完成第 1 步创建虚拟环境并安装依赖；缺少 `venv` 时脚本会打印提示并回退到 PATH 上的 `python`：

```bat
start.bat
```

首次启动后，可通过 GUI 的**设置对话框**开闭各项功能（人体检测、骨骼、手势、矩形检测、去噪、滤波器类型、ROI 重检、头部属性、ReID（OSNet 行人重识别，默认启用，可在设置中关闭）等）。

### 3. Rust 跟踪内核（默认自动启用）

跟踪核心默认 **auto** 后端：`omintrack` 扩展可导入即用 Rust（基准实测平均约 37x
加速），未构建则自动回退纯 Python，行为一致、开箱即用。手动切换方式：

```python
tracker = DetectionTracker(backend="rust")   # 代码显式指定
# 或设置环境变量 OMINTRACK_BACKEND=python / rust / auto，覆盖所有默认构造点
```

也可以在 GUI **设置对话框 → 跟踪后端**中切换（对新接入/重置的相机生效）。
如需自行构建扩展（[Rust 工具链](https://rustup.rs/) + maturin）：

```powershell
pip install maturin
maturin develop --manifest-path rust\omintrack\Cargo.toml --release
```

构建细节与国内镜像加速见 [rust/omintrack/README.md](./rust/omintrack/README.md)。

## 使用示例

除完整 GUI 外，项目提供最小可运行 Demo（单路摄像头闭环：采集 → 显示 → YOLO 检测 → 跟踪 → 叠加显示）：

````bash
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
├── rust/omintrack/ # Rust 跟踪内核（PyO3 扩展 crate）
├── camera/         # 相机采集、枚举、录制
├── common/         # 通用工具（PyAV 采集、YOLO 检测器、目标数据结构）
├── gui/            # PyQt6 图形界面（主窗口、相机格、主题、动画、设置对话框）
├── visioncore/     # 新架构：模块化流水线 / 事件总线 / 协议 / 插件 / 目标管理（影子旁路）
├── config/         # 运行配置（visioncore_pipeline.json）
├── models/         # 模型权重文件（随仓库直接分发）
├── tests/          # pytest 单元测试（含跟踪器等价性与 golden 快照）
├── docs/           # 各阶段的设计、审计与里程碑报告
├── architecture/   # 架构梳理、审计报告与全量档案文档
├── tools/          # 辅助工具（如 MOT 评估脚本）
├── benchmark/      # 性能基准（序列化、跟踪器 Python/Rust 对比）
├── main.py         # GUI 主程序入口
├── demo.py         # 最小可运行单路 Demo
├── requirements.txt
├── LICENSE         # PolyForm Noncommercial License 1.0.0
└── .gitignore
```

## 测试

先安装开发依赖（含 pytest 与 Rust 扩展构建用 maturin）：`pip install -r requirements-dev.txt`。

```bash
pytest tests/                                          # 全部用例（1429 项）
pytest tests/test_hand_gesture_recognizer.py           # MediaPipe Hands API / 数组布局兼容性（8 例）
pytest tests/test_inferworker_pipeline_contract.py     # Legacy/Pipeline 帧处理与 runtime 契约（31 例）
pytest tests/test_inferworker_resilience.py            # 推理线程防火带 / 单写 / slot / 生命周期（15 例）
pytest tests/test_tracker_rust_equivalence.py          # Python/Rust 跟踪器逐帧等价性（18 例）
pytest tests/test_tracker_snapshot.py                  # 跟踪器行为 golden 快照（13 场景）
pytest tests/test_tracker_snapshot.py --update-golden  # 行为变更后再生成快照
python benchmark/benchmark_tracker_rust.py             # 跟踪器性能基准（Python vs Rust）
```

> 说明：全套件 1429 项已通过（2026-11-26）。架构守护用例通过 `tests/_runtime_checks.py` 在子进程中检查 import 副作用，并使用 UTF-8 避免 Windows 中文区域编码影响。

## 文档

`docs/` 与 `architecture/` 收录了完整的演进文档，精选入口：

- **架构与规划**：[战略路线图](./docs/strategic-roadmap.md)、[VisionCore 流水线基础](./docs/pipeline-foundation.md)、[SentinelTrack 2.0 架构蓝图](./docs/sentineltrack-2.0-architecture-blueprint.md)
- **跟踪与滤波**：[预测输出门控与轨迹生命周期](./docs/prediction-output-gating.md)、[MCUKF IRLS 修复契约](./docs/mcukf-irls-correction.md)、[MCUKF / Manifold UKF 方案](./docs/mcukf-manifold-ukf-plan.md)、[AUTO 自适应滤波](./docs/auto-filter-adaptive-plan.md)、[跟踪阶段](./docs/tracker-stage.md)、[Rust 内核设计契约](./docs/rust-tracker-core.md)、[Rust 构建说明](./rust/omintrack/README.md)
- **识别运行时**：[MediaPipe Hands 兼容性约束](./docs/mediapipe-hands-compatibility.md)
- **运行与维护**：[运行时元数据与启动脚本维护](./docs/runtime-maintenance.md)
- **GUI 与主题**：[squircle 连续曲率圆角](./docs/squircle-capsule-widgets.md)
- **性能**：[性能退化分析](./docs/performance-degradation-analysis.md)、[CBOR 基准](./docs/cbor-benchmark.md)
- **插件与协议**：[插件注册表](./docs/plugin-registry.md)、[状态发布](./docs/state-publisher.md)、[UDP 适配器](./docs/udp-adapter.md)

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

- **内容向 PR 必须沉淀文档**：按 [文档沉淀规范](./PROJECT_BIBLE.md#17-文档沉淀规范pr-契约)（§17）将设计意图与契约写入 `docs/` 并同步白皮书——PR 模板与 CI 会检查（完整流程见 [CONTRIBUTING.md](./CONTRIBUTING.md)）；
- 参与贡献前请确认你已阅读并同意 [许可证](#许可证) 条款；
- 本项目因采用 **PolyForm Noncommercial License 1.0.0**，你的贡献同样遵循上述**非商业使用限制**——请勿将本软件或其衍生作品用于任何以盈利为目的的场景；
- 为开发者提供的**自愿赞助、打赏或捐赠**不受此限制（详见上文许可证说明）；
- 提交的代码应尽量附带或更新相应测试（`pytest tests/`）。

## 致谢

- [Ultralytics](https://github.com/ultralytics) — YOLOv8 / YOLO26 检测与姿态模型；
- [PINTO0309](https://github.com/PINTO0309) — 集成的 UHD 轻量人体检测与 CHC 头部属性 ONNX 模型；
- [PyO3](https://pyo3.rs/) / [maturin](https://github.com/PyO3/maturin) / [nalgebra](https://nalgebra.org/) — Rust 跟踪内核的扩展绑定、构建与线性代数基础；
- [PolyForm Project](https://polyformproject.org/) — PolyForm Noncommercial License 1.0.0；
- 以及所有在 `docs/` 与 `architecture/` 中留下设计与审计文档的贡献者。
