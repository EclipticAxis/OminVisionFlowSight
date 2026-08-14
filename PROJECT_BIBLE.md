# VisionBata（VisionDataPlatform）项目白皮书

> **版本**：1.1.1  
> **最后更新**：2026-07-10  
> **文档类型**：项目全量技术白皮书（PROJECT BIBLE）  
> **目标读者**：新开发者、AI Agent、项目维护者  

---

## 目录

1. [项目概述](#1-项目概述)
2. [项目目标](#2-项目目标)
3. [核心功能](#3-核心功能)
4. [技术栈](#4-技术栈)
5. [系统架构](#5-系统架构)
6. [模块说明](#6-模块说明)
7. [数据流](#7-数据流)
8. [算法设计](#8-算法设计)
9. [配置系统](#9-配置系统)
10. [构建部署](#10-构建部署)
11. [当前状态](#11-当前状态)
12. [技术债务](#12-技术债务)
13. [风险分析](#13-风险分析)
14. [未来规划](#14-未来规划)
15. [开发建议](#15-开发建议)
16. [Agent 接入指南](#16-agent-接入指南)

---

## 1. 项目概述

### 1.1 系统定位

VisionBata（内部代号 VisionDataPlatform，简称 VDP）是一个**桌面级实时多路摄像头视觉检测平台**。它是一站式的视频监控智能分析工具，在一台普通 PC 上即可同时接入最多 4 路摄像头，实时执行 AI 目标检测、目标跟踪、手势识别、矩形物体检测、头部属性分类等多维度视觉分析。

### 1.2 适用场景

| 场景 | 说明 |
|------|------|
| **安防监控** | 实时人体检测与跟踪，多路画面同时监控 |
| **工业质检** | 矩形金属件/标牌检测、边缘完整性分析 |
| **人机交互** | 手势识别（举手、挥手、OK、点赞等）用于无接触控制 |
| **行为分析** | 摔倒检测、蹲下检测、人物姿态估计 |
| **出入口管理** | 人脸/头部属性（帽子、口罩、墨镜）检测 |
| **轻量部署** | 单台 PC + USB 摄像头即可运行，无需 GPU 集群 |

### 1.3 解决的核心问题

1. **多路摄像头同时 AI 分析**：共享一个推理线程，避免 GPU/CPU 资源争抢。
2. **检测与跟踪一体化**：YOLO 检测 + 卡尔曼跟踪形成完整的目标生命周期管理。
3. **多模型后端灵活切换**：同一工程内支持 YOLOv8、YOLO26、ONNX Runtime（DirectML GPU 加速）、UHD 轻量人体检测等多种后端，通过文件名自动识别。
4. **跳帧预测降低算力需求**：推理帧间隔（infer_stride）机制 + 卡尔曼预测填充，在维持视觉流畅度的同时大幅降低推理频率。
5. **目标消失后重找回**：基于卡尔曼预测的 ROI 裁剪重检测，自动找回被遮挡后重新出现的目标。
6. **异常检测自适应恢复**：_FilterAdaptor + _HealthMonitor 双重机制，运行时自动检测并修正算法异常。

---

## 2. 项目目标

### 2.1 主要目标

- 构建一个可扩展、模块化的视觉检测桌面应用
- 支持多种 AI 模型后端（PyTorch / ONNX / ONNX+DirectML）的无缝切换
- 提供企业级的卡尔曼跟踪能力（UKF/MCUKF/Manifold UKF + AUTO 自适应）
- 在 CPU 设备上保持实时性（≥15fps 推理 + 显示）

### 2.2 次要目标

- 支持多维度识别能力：目标检测、姿态估计、OBB 旋转框、矩形检测、手势识别、头部属性分类
- 提供完整录像能力（H.264 MP4）
- 提供专业级暗色主题 GUI
- 保持配置的持久化与可调整性

---

## 3. 核心功能

### 3.1 功能矩阵

| 维度 | 功能 | 状态 | 依赖 |
|------|------|------|------|
| **采集** | 多路 USB/IP 摄像头同时接入（最多 4 路） | ✅ 完整 | PyAV DirectShow |
| **检测** | YOLOv8/YOLO26 通用目标检测（80 类 COCO） | ✅ 完整 | ultralytics / ONNX |
| **检测** | OBB 有向目标检测（旋转框） | ✅ 完整 | YOLO26 .pt |
| **检测** | UHD 64x64 轻量人体检测（CPU 优先） | ✅ 完整 | PINTO0309 ONNX |
| **跟踪** | 多目标卡尔曼跟踪（UKF / MCUKF / Manifold UKF） | ✅ 完整 | 纯 NumPy |
| **跟踪** | AUTO 自适应滤波器切换 | ✅ 完整 | _FilterAdaptor |
| **跟踪** | OBB polygon 关联跟踪 | ✅ 完整 | 第二阶段 |
| **跟踪** | 跳帧预测填充（predicted detections） | ✅ 完整 | tracker.py |
| **重检测** | 卡尔曼预测 ROI 裁剪重检测 | ✅ 完整 | tracker + 回调 |
| **姿态** | 17 点 COCO 人体骨架 + 绘制 | ✅ 完整 | YOLOv8-pose |
| **手势** | 身体姿态手势（举手/摔倒/蹲下/双臂展开等） | ✅ 完整 | 纯几何 |
| **手势** | 手部手指手势（OK/点赞/数字 1~5/握拳等） | ⚠️ 禁用 | MediaPipe（版本兼容） |
| **矩形** | Canny + 轮廓矩形检测 + 线性卡尔曼跟踪 | ✅ 完整 | OpenCV |
| **头部属性** | 帽子/口罩/墨镜/眼睛开闭/嘴巴开闭 | ✅ 完整 | PINTO0309 CHC ONNX |
| **去噪** | 双边滤波 / NL-Means 预处理 | ✅ 完整 | OpenCV |
| **超分辨率** | ESPCN x4 小目标增强 | ✅ 完整（可选） | cv2.dnn_superres |
| **录像** | H.264 MP4 多路同时录制 | ✅ 完整 | PyAV |
| **GUI** | 暗色主题 Fusion 风格 + 动画系统 | ✅ 完整 | PyQt6 |

### 3.2 功能开关

大部分高级功能可通过 GUI 独立开关：
- 人体检测、骨骼绘制、手势模式
- 矩形检测（含灵敏度/颜色阈值）
- 去噪方法与强度
- 滤波器类型选择
- ROI 重检测参数
- 头部属性分类

---

## 4. 技术栈

### 4.1 核心依赖

| 类别 | 库 | 版本要求 | 用途 |
|------|-----|---------|------|
| **GUI 框架** | PyQt6 | ≥6.2.0 | 主窗口、摄像头网格、设置对话框、信号槽、Fusion 风格 |
| **推理引擎** | ultralytics | ≥8.0.0 | YOLOv8/YOLO26 .pt 推理 |
| **推理引擎** | onnxruntime-directml | ≥1.24.0 | ONNX 推理 + DirectML GPU 加速 |
| **推理引擎** | onnx | ≥1.22.0 | ONNX 模型解析 |
| **深度学习** | torch | ≥2.0.0 | PyTorch 后端 |
| **数值计算** | numpy | ≥1.21.0 | 卡尔曼滤波、数组运算、几何计算 |
| **图像处理** | opencv-python | ≥4.5.0 | Canny/轮廓/CLAHE、去噪、颜色转换、超分辨率 |
| **摄像头采集** | av (PyAV) | ≥10.0.0 | DirectShow 摄像头采集 + H.264 录像 |
| **手部手势** | mediapipe | ≥0.10.0 | 手部关键点（**当前版本兼容性问题，已禁用**） |

### 4.2 技术选型理由

| 选型 | 理由 |
|------|------|
| **PyQt6** | Windows 原生体验、丰富的信号槽机制、成熟的 Fusion 主题、QThread 线程模型 |
| **PyAV (av)** | 比 OpenCV VideoCapture 更稳定的 DirectShow 支持、原生 H.264 编码、帧级控制 |
| **纯 NumPy 卡尔曼** | 无外部滤波库依赖（不依赖 filterpy）、完全可控的参数调优、三种 UKF 变体统一接口 |
| **ONNX Runtime DirectML** | Windows 原生 GPU 加速、无需 CUDA、AMD/NVIDIA/Intel 通用 |
| **PINTO0309 模型** | 社区广泛验证的轻量 ONNX 模型、64x64 超低分辨率输入、CPU 友好 |

### 4.3 语言与运行环境

- **语言**：Python 3.11（venv 虚拟环境）
- **操作系统**：Windows 10/11（DirectShow 依赖）
- **硬件最低要求**：CPU 4 核、8GB RAM、USB 摄像头
- **推荐硬件**：带 DirectX 12 GPU（用于 DirectML 加速）

---

## 5. 系统架构

### 5.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                      GUI 表现层 (gui/)                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  MainWindow  │  │  CameraCell  │  │  SettingsDialog      │   │
│  │  主窗口调度   │  │  画面+叠加   │  │  QSettings 持久化     │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  theme.py    │  │ animations   │  │  safe_widgets/splash │   │
│  │  暗色主题     │  │  动画系统     │  │  交互辅助             │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      推理引擎层 (ai/)                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              InferWorker (QThread)  ← 核心推理线程         │   │
│  │  双缓冲 → 去噪 → 多后端推理 → 跟踪 → 手势/矩形/头部属性     │   │
│  └──────────────────────────────────────────────────────────┘   │
│  ┌───────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐   │
│  │ OnnxYolo  │ │ Yolo26   │ │ Uhd      │ │ HeadClassifier │   │
│  │ Backend   │ │ Backend  │ │ Backend  │ │ Backend        │   │
│  ├───────────┤ ├──────────┤ ├──────────┤ ├────────────────┤   │
│  │ ONNX+     │ │ ultralyt │ │ 64x64    │ │ Hat/Mask/      │   │
│  │ DirectML  │ │ .pt      │ │ ONNX     │ │ Eye/Mouth      │   │
│  └───────────┘ └──────────┘ └──────────┘ └────────────────┘   │
│  ┌───────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │DetectionTrack │  │ Rectangle    │  │ Gesture/Hand       │   │
│  │UKF/MCUKF/     │  │ Detector +   │  │ GestureRecognizer  │   │
│  │Manifold+AUTO  │  │ Tracker(KF)  │  │ (17点几何/MediaP)  │   │
│  └───────────────┘  └──────────────┘  └────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                     数据模型层 (ai/)                             │
│  ┌──────────────────┐  ┌──────────────────┐                     │
│  │  Detection       │  │  ModelTask       │                     │
│  │  (统一检测容器)    │  │  (DETECT/POSE/   │                     │
│  │  + polygon IoU   │  │   OBB 枚举)      │                     │
│  └──────────────────┘  └──────────────────┘                     │
├─────────────────────────────────────────────────────────────────┤
│                    摄像头采集层 (camera/)                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ CameraCapture│  │ VideoRecorder│  │ CameraInfo/scan      │   │
│  │ 多 worker 管理│  │ H.264 MP4    │  │ 三级设备枚举           │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                    硬件/系统层                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ 摄像头   │  │ 摄像头   │  │ 摄像头   │  │ 摄像头        │   │
│  │ USB #0   │  │ USB #1   │  │ USB #2   │  │ USB #3       │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 线程模型

```
┌─────────────────────────────────────────┐
│         QApplication 主线程 (GUI)         │
│  MainWindow → CameraCell × 4 → paintEvent│
│                    ▲                      │
│    detection_ready (QueuedConnection)     │
│                    │                      │
├────────────────────┼──────────────────────┤
│                    │                      │
│  ┌─────────────────┴───────────────────┐ │
│  │   InferWorker 线程 (推理线程)         │ │
│  │   共享1个，服务全部 4 个槽位           │ │
│  └─────────────────────────────────────┘ │
│                    ▲                      │
│      submit_frame (DirectConnection)      │
│                    │                      │
├────────────────────┼──────────────────────┤
│                    │                      │
│  ┌─────────────────┴───────────────────┐ │
│  │ SingleCameraWorker × N (采集线程)     │ │
│  │ 每个槽位 1 个 PyAV DirectShow 线程     │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘

录像: ThreadPoolExecutor(max_workers=4) 守护线程池
```

### 5.3 架构特点

| 特点 | 说明 |
|------|------|
| **共享推理** | 1 个 InferWorker 服务全部槽位，避免 GPU 争抢和多模型多实例内存爆炸 |
| **双缓冲** | 每槽位独立的 `_FrameSlot`，采集线程无锁写入，推理线程快照读取 |
| **信号槽解耦** | 采集→显示用 `QueuedConnection`；采集→推理用 `DirectConnection`（跨线程零拷贝） |
| **统一数据模型** | `Detection` dataclass 是所有后端输出的统一容器，`to_dict()` 兼容旧模块 |
| **自适应** | _FilterAdaptor + _HealthMonitor 双重运行时自适应，无需人工介入 |
| **纯 NumPy** | 所有跟踪算法不依赖外部滤波库，完全可控 |

---

## 6. 模块说明

### 6.1 入口与引导

#### `main.py` — 应用入口

| 属性 | 内容 |
|------|------|
| **职责** | 全局异常钩子注册 → 高 DPI 配置 → QApplication 创建 → SplashScreen → MainWindow → 事件循环 |
| **输入** | 命令行参数（未实际使用） |
| **输出** | 退出码（0 成功，1 异常） |
| **依赖** | PyQt6, logging |

关键函数：
- `setup_global_exception_hook()`：捕获所有未处理异常写入 `error.log`
- `configure_high_dpi()`：启用 `QT_ENABLE_HIGHDPI_SCALING`
- `create_application()`：QApplication（Segoe UI 字体 + Fusion 风格）
- `main()`：启动序列编排

#### `start.bat` — Windows 启动脚本

硬编码路径 `F:\VisionBata\venv\Scripts\python.exe main.py`。**注意**：若项目目录移动需同步修改。

---

### 6.2 GUI 层

#### `gui/main_window.py` — 主窗口 (1282 行)

| 属性 | 内容 |
|------|------|
| **职责** | 应用 UI 布局 + 信号槽调度中心 + 配置应用 |
| **UI 结构** | 顶栏（品牌+设置） + 侧栏（概览卡片/设备选择/运行控制/存储配置） + 2×2 CameraCell 网格 |
| **关键信号槽** | 摄像头选择 → worker 启动 → frame_ready → 分发到显示+推理+录像 |
| **输入** | 用户交互（按钮、下拉框、双击） + QSettings |
| **输出** | UI 状态更新 + 配置下发到 InferWorker/CameraCapture/VideoRecorder |

核心方法：
- `_on_camera_selected(slot_id, device_name)`：停旧启新 worker，建立信号槽链路
- `_on_ai_toggled(checked)`：动态 connect/disconnect 推理链路
- `_apply_ai_settings(cfg)`：从 SettingsDialog 获取配置并完整应用
- `closeEvent()`：有序关闭录像→摄像头→推理→阻断信号

#### `gui/camera_cell.py` — 单路画面显示 (1348 行)

| 属性 | 内容 |
|------|------|
| **职责** | 单路摄像头画面显示 + 检测框/骨架/多边形/标签叠加绘制 |
| **输入** | `update_frame(slot_id, rgb_ndarray)` + `update_detections(slot_id, list[dict])` |
| **输出** | Qt paintEvent → GPU 合成叠加层 |
| **缓存策略** | 背景缓存 / 占位缓存 / 叠加缓存 / 文本宽度缓存，避免无效重绘 |

绘制功能：
- 检测框：按实例着色（冷蓝系 8 色调色板）+ 预测/重检框虚线区分
- OBB 旋转框：polygon 叠加 + 发光边缘
- 人体骨架：分区着色 + 半透明填充 + 方块关键点 + 实例编号
- 标签：类别 + track_id + 置信度 + 手势 + 头部属性
- 覆盖信息：FPS、目标数、在线/离线徽章

#### `gui/settings_dialog.py` — 设置对话框 (976 行)

| 属性 | 内容 |
|------|------|
| **职责** | GUI 配置编辑 + QSettings 持久化管理 |
| **配置域** | 录像存储、AI 检测参数、模型切换、功能开关 |
| **配置键** | 约 30 个 QSettings 键（`org=VisionBata, app=VisionDataPlatform`） |

#### `gui/theme.py` — 暗色主题

| 属性 | 内容 |
|------|------|
| **职责** | 冷蓝灰暗色 Fusion QSS 主题定义 |
| **颜色系统** | 4 层背景 + 3 层边框 + 强调色 + 3 层文字 + 3 种状态色（约 30 色常量） |
| **STYLESHEET** | 约 720 行 QSS，覆盖所有组件 |

#### `gui/animations.py` — 动画系统

12 种常用动画快捷创建：淡入、滑入、序列动画、脉冲、呼吸灯、缩放脉冲、骨架屏微光效果。

#### `gui/safe_widgets.py` — 防滚轮抢焦点控件

Mixin 模式：仅在控件获焦时才响应滚轮事件，防止滚动面板时误改参数。

#### `gui/splash_screen.py` — 启动闪屏

2.2 秒动画：暗色卡片 + 信号线 + "VDP V1.1" + 进度条 + 脉冲指示器。

---

### 6.3 摄像头采集层

#### `camera/enumerator.py` — 摄像头枚举

三级枚举策略：
1. `QMediaDevices.videoInputs()`（Qt6 现代设备拓扑）
2. PyAV DirectShow 实际拉流验证（亮度 > 1.0 排除虚拟设备）
3. OpenCV `CAP_DSHOW` 索引回退

去重按 `(index, name)` 键。

#### `camera/capture.py` — 摄像头采集

每路摄像头对应一个 `SingleCameraWorker(QThread)`：
- `av.open("video={name}", format="dshow", ...)` → 循环解码 → `frame_ready.emit(slot_id, rgb_frame)`
- `CameraCapture` 管理所有 worker（停旧启新、全部停止）

#### `camera/recorder.py` — 录像器

- 每路独立 `_RecorderThread(threading.Thread)`：消费队列帧 → RGB→yuv420p → H.264 编码 → mux → MP4
- `VideoRecorder`：`ThreadPoolExecutor(max_workers=4)` 守护线程池
- 编码参数：`preset="ultrafast", tune="zerolatency", crf="23"`
- 命名格式：`slot_{slot_id}_{YYYYMMDD_HHMMSS}.mp4`

---

### 6.4 推理引擎层

#### `ai/inference.py` — InferWorker 核心推理线程 (~1790 行)

**这是整个项目最核心的模块。**

| 属性 | 内容 |
|------|------|
| **职责** | 共享推理线程：调度多后端推理 → 跟踪 → 手势/矩形/头部属性 → 输出 |
| **输入** | `submit_frame(slot_id, rgb_frame)` |
| **输出** | `detection_ready.emit(slot_id, detections)` |
| **线程** | QThread（独立于 GUI 线程） |

推理管线（每帧命中 stride 时）：
```
submit_frame → _buffers[slot]
    ↓
run() 快照所有槽位
    ↓
if frame_id % stride == 0:
    1. 可选去噪预处理 (_denoise_frame)
    2. 多后端推理 (_run_inference)
       ├─ OnnxYoloBackend (ONNX/DirectML)
       ├─ Yolo26Backend (.pt Detect/Pose/OBB)
       └─ UhdBackend (64x64 ONNX, CPU优先)
    3. DetectionTracker.update()
       + ROI 重检测回调
       + HSV 颜色直方图更新
    4. _attach_head_attributes (CHC 头部属性)
    5. _attach_gestures (身体姿态手势)
    6. _run_rectangle_detection (Canny + 轮廓)
    7. _run_hand_gesture_detection (MediaPipe, 已禁用)
    8. final_dedupe() 去重
    9. _HealthMonitor 健康检查 → 异常自动回退保守配置
else:
    tracker.predict_step() 卡尔曼填充
    + 复用上次非 person 检测结果
    
detection_ready.emit(slot_id, detections)
```

配置方法（约 20 个 setter）：
- `set_conf()`, `set_infer_stride()`, `set_smooth_alpha()`
- `set_filter_type()`, `set_denoise()`
- `set_person_redetect()`, `set_head_classifier()`
- `set_rectangle_detection_enabled()` 等矩形参数
- `register_slot()/unregister_slot()` 槽位管理

内部类：
- `_HealthMonitor`：3 秒滑动窗口监控人物置信度，连续低于 0.35 → 回退 denoise=none + filter=ukf + redetect=0
- `_FrameSlot`：双缓冲槽位（帧 + 元数据 + 锁）
- `SuperResEngine`：ESPCN x4 超分辨率引擎（可选）

模型识别函数：
- `_is_uhd_model(path)`：文件名含 `uhd`/`ultratinyod`
- `_is_yolo26_model(path)`：文件名含 `yolo26`/`yolov10`
- `_infer_model_task_from_path(path)`：`obb`→OBB, `pose`→POSE, 默认 DETECT

#### `ai/onnx_yolo_backend.py` — ONNX Runtime YOLO 后端

| 属性 | 内容 |
|------|------|
| **职责** | ONNX Format YOLO 推理（支持 DirectML GPU 加速） |
| **任务类型** | DETECT, POSE |
| **预处理** | letterbox 缩放 + 填充 + 归一化 [0,1] + CHW 转置 |
| **后处理** | 解码 cx/cy/w/h + NMS（DETECT）或 56 通道关键点（POSE） |
| **Provider** | DirectML 优先，CPU 回退 |

#### `ai/yolo26_backend.py` — YOLO26 .pt 后端

| 属性 | 内容 |
|------|------|
| **职责** | Ultralytics YOLO26 .pt 模型封装 |
| **任务类型** | DETECT, POSE, OBB |
| **OBB 输出** | `xyxyxyxy` 四边形 + 映射到归一化坐标 |

#### `ai/uhd_backend.py` — UHD 64x64 轻量人体检测

| 属性 | 内容 |
|------|------|
| **职责** | PINTO0309 UHD ONNX 后端（64x64 输入） |
| **模型** | `uhd_n_64x64_static.onnx`（含后处理） |
| **输入** | 64x64 RGB，INTER_NEAREST 拉伸 |
| **输出** | 仅 `person` 类别 |

#### `ai/head_classifier.py` — 头部属性分类

| 属性 | 内容 |
|------|------|
| **职责** | PINTO0309 CHC ONNX 后端 |
| **模型** | `chc_s_wo_fiqa.onnx`（无 FIQA 分支） |
| **输入** | 基于 person box 裁剪 head/eyes/mouth 三个 ROI |
| **输出** | hat, mask, sunglass, eye_open(L/R), mouth_open |
| **性能** | CPU 单目标 ~16-17ms，通过跳帧控制多目标延迟 |

---

### 6.5 跟踪与滤波层

#### `ai/tracker.py` — DetectionTracker（纯 NumPy 多目标跟踪器）

**这是项目算法核心。**

| 类 | 职责 |
|-----|------|
| `_DetectionUKF` | 基础无迹卡尔曼（8 维状态，17 sigma 点，Cholesky 分解） |
| `_DetectionMCUKF` | 最大相关熵 UKF（高斯核加权，抗离群值，不动点迭代） |
| `_DetectionManifoldUKF` | 流形 UKF（eigh 特征分解 + SPD 投影，P 永久正定） |
| `_FilterAdaptor` | AUTO 自适应切换（监控 NIS/缺失率/P条件数/Cholesky 失败） |
| `_Track` | 单条轨迹（滤波器 + 外观直方图 + OBB polygon 管理） |
| `DetectionTracker` | 主跟踪器（匈牙利匹配 + 轨迹生命周期管理） |

跟踪流程：
```
detections (list[Detection])
    ↓
1. 推进所有 track.predict() 卡尔曼预测
2. 计算 cost 矩阵 (IoU + 外观 + 中心距)
   优先 polygon IoU > bbox IoU
3. 匈牙利匹配 (scipy.optimize.linear_sum_assignment)
4. 匹配成功 → track.correct(detection) + 重置 miss
5. 未匹配检测 → 新建 track
6. 未匹配 track → ROI 重检测或 miss+1
7. 清理超期 track (miss > max_misses)
8. AUTO 模式评估 → 可能切换滤波器
   输出 list[dict] (含 track_id, track_state)
```

关键参数（已保守化）：
- `α=1e-3, β=2.0, κ=0`（UKF sigma 点参数）
- `P_MAX_POSITION=0.05, P_MAX_SIZE=0.02, P_MAX_VELOCITY=0.01`（按分量协方差上界）
- 自适应 Q/R（基于 NIS + 目标尺度）
- 速度限幅 `[-0.3, 0.3]`（归一化坐标）

#### `ai/rectangle_detector.py` — 矩形检测与跟踪

| 属性 | 内容 |
|------|------|
| **检测算法** | 灰度化 → CLAHE → 高斯模糊 → Canny 边缘 → 形态学闭合 → 轮廓检测 → 四边形近似 → 几何评分（角度 46% + 填充率 24% + 平行度 18% + 边缘 12%） |
| **跟踪算法** | 线性卡尔曼（8 维），IoU 62% + 中心距 26% + 尺寸 12% 匹配 |
| **灵敏度预设** | low / medium / high 三档 |

---

### 6.6 手势识别层

#### `ai/gesture_recognizer.py` — 身体姿态手势

基于 17 个 COCO 关键点的纯几何规则识别：
- `hands_up`（双手举起）、`left_hand_up`/`right_hand_up`
- `fallen`（摔倒）、`squat`（蹲下）
- `arms_spread`（双臂展开）、`arms_crossed`（手臂交叉）
- `hand_near_head`（手靠近头部）、`sideways`（侧身）

#### `ai/hand_gesture_recognizer.py` — 手部手指手势

基于 MediaPipe Hands 21 个关键点分类：
- `ok`、`thumbs_up`、`one`~`five`、`fist`、`open_palm`

**当前状态：已禁用**（mediapipe 版本兼容性问题导致 `mp.solutions.hands` 无法导入）。

---

### 6.7 数据模型层

#### `ai/detection.py` — 统一检测类型

```python
@dataclass
class Detection:
    class_id: int           # COCO 类别索引
    label: str              # 类别名称
    confidence: float       # 置信度 [0,1]
    bbox: tuple[float,...]  # (x1, y1, x2, y2) 归一化
    polygon: list | None    # OBB 四边形 [(x,y),...]
    keypoints: list | None  # 17 个 Keypoint
    track_id: int | None    # 跟踪 ID
    track_state: str | None # normal/predicted/redetected
```

几何工具函数：
- `polygon_iou(a, b)`：优先 shapely 精确 IoU，回退 bbox
- `polygon_center(poly)`：顶点平均值
- `polygon_to_bbox(poly)`：外接矩形

#### `ai/model_task.py` — 任务类型枚举

```python
class ModelTask(Enum):
    DETECT = auto()  # 通用目标检测
    POSE = auto()    # 姿态检测（17 关键点）
    OBB = auto()     # 有向目标检测（旋转框+多边形）
```

---

## 7. 数据流

### 7.1 端到端数据流图

```
┌──────────┐
│ USB 摄像头 │ 物理设备
└────┬─────┘
     │ PyAV dshow (DirectShow)
     ▼ BGR → RGB ndarray (640×480, uint8)
┌──────────────────┐
│ SingleCameraWorker│ QThread (每路1个)
│ frame_ready(slot, │
│   rgb_ndarray)    │
└───┬──────┬───────┘
    │      │
    │      ├──→ CameraCell.update_frame  [QueuedConnection]
    │      │      → QImage → paintEvent (~30fps GPU 合成)
    │      │         ├─ 背景缓存
    │      │         ├─ 视频帧 (支持水平翻转)
    │      │         └─ 检测叠加层
    │      │              ├─ 检测框 (实线/虚线)
    │      │              ├─ OBB polygon
    │      │              ├─ 人体骨架 + 填充
    │      │              ├─ 矩形 polygon
    │      │              └─ 标签信息
    │      │
    │      ├──→ InferWorker.submit_frame  [DirectConnection, AI开]
    │      │      → _buffers[slot].frame = frame (双缓冲)
    │      │
    │      └──→ VideoRecorder.push_frame  [录制开]
    │             → queue → ThreadPoolExecutor
    │                → H.264 encode → .mp4
    │
    ▼
┌─────────────────────────────────────────────┐
│            InferWorker.run()                 │
│  for each slot with buffered frame:         │
│                                              │
│  if frame_id % infer_stride == 0:           │
│    ┌─ _denoise_frame() [none/bilateral/nl]  │
│    ├─ _run_inference()                      │
│    │    └→ backend.predict() → list[Detection]
│    ├─ DetectionTracker.update(dets)         │
│    │    ├─ UKF/MCUKF/Manifold predict+correct│
│    │    ├─ _redetect_person_in_roi() [丢失]  │
│    │    └─ _FilterAdaptor 评估              │
│    ├─ _attach_head_attributes() [CHC]       │
│    ├─ _attach_gestures() [身体姿态]          │
│    ├─ _run_rectangle_detection()            │
│    ├─ final_dedupe()                        │
│    └─ _HealthMonitor.update()              │
│  else:                                      │
│    └─ tracker.predict_step() [卡尔曼填充]    │
│                                              │
│  detection_ready.emit(slot_id, detections)   │
└────────────────────┬────────────────────────┘
                     │ [QueuedConnection]
                     ▼
          CameraCell.update_detections()
            → 缓存 + 签名比对 + TTL 2.0s
```

### 7.2 坐标约定

- **全程归一化** `[0, 1]`：所有检测框、关键点、多边形顶点均归一化
- **绘制映射**：绘制时乘以控件宽/高还原像素坐标
- **水平翻转**：开启时 `x = 1.0 - x`
- **OBB polygon**：`[(x1,y1), (x2,y2), (x3,y3), (x4,y4)]` 四边形顶点

### 7.3 配置持久化

```
QSettings (org=VisionBata, app=VisionDataPlatform)
  ├─ recording/output_dir
  ├─ ai/conf_threshold, ai/infer_stride, ai/smooth_alpha
  ├─ ai/hide_delay_ms, ai/model_path
  ├─ ai/person_enabled, ai/skeleton_enabled
  ├─ ai/gesture_enabled, ai/gesture_mode
  ├─ ai/filter_type, ai/mcukf_kernel_sigma
  ├─ ai/denoise_method, ai/denoise_strength
  ├─ ai/person_redetect_max_retries, ai/roi_pad
  ├─ ai/head_classifier_enabled, ai/head_classifier_path, ai/head_classifier_stride
  └─ ai/rectangle_* (enabled/sensitivity/max_count/target_color/...)
```

---

## 8. 算法设计

### 8.1 算法矩阵

| 算法 | 位置 | 作用 | 复杂度 | 性能影响 |
|------|------|------|--------|----------|
| **无迹卡尔曼滤波 (UKF)** | `tracker.py::_DetectionUKF` | 非线性状态估计，跟踪 person bbox | O(n³) 8维 | 中等（Cholesky 分解） |
| **最大相关熵 UKF (MCUKF)** | `tracker.py::_DetectionMCUKF` | 抗离群值 UKF，高斯核加权观测残差 | O(n³ + k·n) | 较高（不动点迭代） |
| **流形 UKF (Manifold UKF)** | `tracker.py::_DetectionManifoldUKF` | P 矩阵永远正定，SPD 投影 | O(n³) + 2·eigh | 较高（特征分解） |
| **AUTO 自适应切换** | `tracker.py::_FilterAdaptor` | 运行时自动选择最优滤波器 | O(1) 评估 | 几乎为零 |
| **线性卡尔曼** | `rectangle_detector.py::_RectangleKalman` | 矩形框线性跟踪 | O(n³) 8维 | 低 |
| **匈牙利匹配** | `tracker.py::DetectionTracker` | 检测-跟踪最优分配 | O(n³) | 低（n 通常 < 20） |
| **NMS (非极大抑制)** | `onnx_yolo_backend.py::_nms` | 去除重叠检测框 | O(n²) | 低 |
| **Canny 边缘检测** | `rectangle_detector.py::RectangleDetector` | 矩形轮廓提取 | O(W·H) | 低（灰度图） |
| **多边形 IoU** | `detection.py::polygon_iou` | OBB 匹配的面积交并比 | O(n) shapely | 低 |
| **NL-Means 去噪** | `inference.py::_denoise_frame` | 非局部均值图像去噪 | O(W·H·patch²) | 高（>720p 降采样） |
| **双边滤波** | `inference.py::_denoise_frame` | 保边去噪 | O(W·H·d²) | 低 (~5ms) |
| **17 点姿态分析** | `gesture_recognizer.py::GestureRecognizer` | 几何规则手势分类 | O(1) 17点 | 几乎为零 |
| **HSV 颜色直方图** | `tracker.py::_Track` | 外观特征用于匹配 | O(W·H) 裁剪区域 | 低 |
| **SPD 投影（特征值裁剪）** | `tracker.py::_DetectionManifoldUKF` | 保证协方差正定 | O(n³) eigh | 高 |

### 8.2 UKF 滤波器族设计

```
_DetectionUKF (基类)
  ├─ 8维状态: [cx, cy, w, h, vx, vy, vw, vh]
  ├─ 17 sigma 点 (n=8, n+2n+1)
  ├─ 匀速运动模型 (dt=1)
  ├─ Cholesky 分解生成 sigma 点
  ├─ 自适应 Q/R
  ├─ 按分量 P 上界限制
  └─ 速度/尺度限幅

_DetectionMCUKF (_DetectionUKF)
  ├─ 高斯核加权观测残差 e_i
  ├─ 权重 w_i = exp(-e_i²/(2σ²))
  ├─ 不动点迭代 (默认3次)
  └─ Joseph 形式协方差更新

_DetectionManifoldUKF (_DetectionUKF)
  ├─ eigh 特征分解 P = V·diag(λ)·Vᵀ
  ├─ SPD 投影: λ ← clip(λ, 1e-8, P_MAX_i)
  └─ P := V·diag(λ)·Vᵀ

_FilterAdaptor
  ├─ 监控信号: NIS, miss_rate, P_cond, chol_fail
  ├─ 决策逻辑:
  │   cond>1000 或 chol_fail>2 → manifold_ukf
  │   NIS>8.0 或 miss_rate>50% → mcukf
  │   否则 → ukf
  └─ 迟滞: 连续 2 次评估一致才切换
```

### 8.3 ROI 重检测算法

```
person track 丢失 (IoU 匹配失败)
  ↓
卡尔曼预测框 (bbox_pred)
  ↓
扩展 ROI: pad = roi_pad × max(w, h)
  ↓
裁剪原图 + 低置信度 (conf × 0.7) 重跑 YOLO
  ↓
质量门控:
  ├─ IoU(pred, redet) > 0.3
  ├─ conf > original_conf × 0.6
  └─ scale_ratio ∈ [0.5, 1.5]
  ↓
通过 → track.correct(redet)
失败 → miss += 1 (等待下次重试或过期)
```

### 8.4 矩形检测评分系统

```
几何评分 = 0.46·angle_score + 0.24·fill_score + 0.18·parallel_score + 0.12·edge_score

angle_score:   四边夹角与 90° 的偏差
fill_score:    凸包面积 / 轮廓面积
parallel_score: 对边平行度
edge_score:    四边长度标准差
```

### 8.5 OBB Polygon 外推

```
跳帧时 polygon 跟随 bbox 变化外推：
  scale_x = w_new / w_old
  scale_y = h_new / h_old
  delta_cx = cx_new - cx_old
  delta_cy = cy_new - cy_old

for each vertex (vx, vy):
    vx' = (vx - cx_old) × scale_x + cx_new
    vy' = (vy - cy_old) × scale_y + cy_new
```

---

## 9. 配置系统

### 9.1 配置来源

| 来源 | 内容 | 优先级 |
|------|------|--------|
| QSettings | 用户通过 GUI 调整的持久化配置 | 最高（覆盖默认值） |
| 代码默认值 | 各模块的类常量/构造函数默认参数 | 最低 |
| 环境变量 | `QT_ENABLE_HIGHDPI_SCALING=1` | 启动时设定 |

### 9.2 关键可调整项

| 参数 | 默认值 | 范围 | 影响 |
|------|--------|------|------|
| `ai/conf_threshold` | 0.5 | 0.1–0.9 | YOLO 检测置信度阈值 |
| `ai/infer_stride` | 1 | 1–10 | 推理跳帧间隔（越大越省算力但跟踪预测多） |
| `ai/smooth_alpha` | 0.85 | 0–1 | 显示框平滑系数 |
| `ai/hide_delay_ms` | 2000 | 500–5000 | 目标消失后延迟隐藏时间 |
| `ai/filter_type` | ukf | ukf/mcukf/manifold_ukf/auto | 卡尔曼滤波器类型 |
| `ai/denoise_method` | none | none/bilateral/nl_means | 推理前去噪方法 |
| `ai/denoise_strength` | 1 | 1–10 | 去噪强度（保守映射） |
| `ai/person_redetect_max_retries` | 2 | 0–5 | 目标丢失后重检测尝试次数 |
| `ai/roi_pad` | 0.15 | 0–0.5 | ROI 裁剪扩展系数 |
| `ai/mcukf_kernel_sigma` | 0.25 | 0.05–2.0 | MCUKF 高斯核带宽 |
| `ai/head_classifier_stride` | 2 | 1–10 | 头部属性推理跳帧 |

### 9.3 配置传递路径

```
SettingsDialog (QSettings)
    ↓
MainWindow._apply_ai_settings(cfg)
    ↓
InferWorker.set_*() 方法
    ↓
遍历 _trackers[slot_id] 设置参数
    ↓
_clear_slot_caches() 清除缓存
```

---

## 10. 构建部署

### 10.1 依赖安装

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境 (Windows)
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 额外：ESPCN 超分辨率模型（可选）
# 下载 ESPCN_x4.pb 放入 models/ 目录
```

### 10.2 模型准备

| 文件 | 必须 | 说明 |
|------|------|------|
| `models/yolov8n.pt` | ✅ | 默认 YOLOv8 检测模型 |
| `models/yolov8n-pose.pt` | 推荐 | 姿态估计（骨骼+手势） |
| `models/yolov8n.onnx` | 可选 | ONNX 版本（DirectML 加速） |
| `models/yolo26n.pt` | 可选 | YOLO26 检测模型 |
| `models/yolo26n-obb.pt` | 可选 | OBB 旋转框检测 |
| `models/uhd_n_64x64_static.onnx` | 可选 | UHD 轻量人体检测 |
| `models/chc_s_wo_fiqa.onnx` | 可选 | 头部属性分类 |
| `models/ESPCN_x4.pb` | 可选 | 超分辨率模型 |

### 10.3 启动流程

```
1. python main.py (或双击 start.bat)
2. main() → setup_global_exception_hook()
3. configure_high_dpi() → QT_ENABLE_HIGHDPI_SCALING=1
4. create_application() → QApplication (Fusion 风格, Segoe UI 字体)
5. SplashScreen → 2.2s 动画
6. MainWindow → 扫描摄像头 → 恢复上次配置
7. 用户选择摄像头 → _on_camera_selected()
   → SingleCameraWorker 启动 → frame_ready 信号建立
8. 用户开启 AI → InferWorker 启动 → detection_ready 信号建立
9. app.exec() → Qt 事件循环
```

### 10.4 关闭流程

```
MainWindow.closeEvent()
  1. 停止所有录像 (VideoRecorder.stop_all)
  2. 停止所有摄像头 (CameraCapture.stop_all)
  3. 停止推理线程 (InferWorker.stop, wait 3s)
  4. blockSignals(True) 阻断残余信号
```

### 10.5 目录结构要求

```
F:\VisionBata\
├── main.py              # 入口（必须）
├── start.bat            # 启动脚本（路径可改）
├── venv/                # 虚拟环境（必须）
├── models/              # 模型文件（必须至少含 yolov8n.pt）
├── ai/                  # 推理模块（必须）
├── gui/                 # GUI 模块（必须）
├── camera/              # 采集模块（必须）
├── requirements.txt     # 依赖清单
└── recordings/          # 录像输出（自动创建）
```

---

## 11. 当前状态

### 11.1 版本信息

- **版本号**：1.1.1
- **最近重大更新**：2026-07-10（OBB Tracker 第二阶段完成）

### 11.2 已完成的重大特性

| 日期 | 特性 | 状态 |
|------|------|------|
| 2026-07-07 | 卡尔曼 ROI 重检测机制 | ✅ |
| 2026-07-08 | UKF 替换线性卡尔曼 | ✅ |
| 2026-07-08 | MCUKF + Manifold UKF | ✅ |
| 2026-07-08 | AUTO 自适应滤波器切换 | ✅ |
| 2026-07-08 | 图像去噪（双边/NL-Means） | ✅ |
| 2026-07-08 | 性能下降恢复（参数保守化） | ✅ |
| 2026-07-09 | PINTO0309 UHD + CHC 集成 | ✅ |
| 2026-07-09 | YOLO26 Detect / OBB 集成 | ✅ |
| 2026-07-09 | 识别失效修复（np.diag bug） | ✅ |
| 2026-07-10 | OBB Tracker 第二阶段（polygon 关联） | ✅ |

### 11.3 已知问题

| 问题 | 严重程度 | 状态 |
|------|----------|------|
| MediaPipe Hands 版本兼容性导致手部手势不可用 | 中 | 已禁用，待修复 |
| DirectML 某些 ONNX 模型可能 fallback CPU | 低 | 取决于 GPU 驱动 |
| UHD 64x64 模型在随机噪声上假阳性较多 | 低 | 依赖 tracker 置信度过滤 |
| `start.bat` 硬编码绝对路径 | 低 | 文档已知 |
| 测试覆盖率低（仅 OBB 第二阶段有测试） | 中 | 待补充 |

---

## 12. 技术债务

### 12.1 代码层面

| 问题 | 位置 | 影响 | 建议 |
|------|------|------|------|
| **`Detection` 双重形式** | 全局 | Detection 是 dataclass 但系统中同时使用 dict 形式（to_dict()），造成类型不统一 | 统一使用 Detection 对象，仅在序列化边界转换 |
| **`InferWorker` 过大** | `ai/inference.py` ~1790 行 | 单类承担推理/跟踪/手势/矩形/健康监控全部职责 | 拆分为调度器 + 独立管线（detection_pipeline, tracking_pipeline, attribute_pipeline） |
| **魔法字符串** | 各处 | 滤波器类型识别用字符串比较（如 `"manifold_ukf"`） | 引入 FilterType 枚举 |
| **DirectML/CPU provider 切换硬编码** | `ai/inference.py` | 多后端 provider 选择逻辑散落 | 引入 BackendFactory |
| **重复的 `_debug_report()`** | `gui/camera_cell.py`, `ai/inference.py` | 两个独立副本 | 抽取到公共工具模块 |
| **GUI 和逻辑未完全分离** | `gui/main_window.py` | 信号槽连接与业务逻辑混合 | 引入 ViewModel/Controller 层 |

### 12.2 测试层面

| 问题 | 说明 |
|------|------|
| **测试覆盖极低** | 仅 `tests/test_obb_stage2.py` 覆盖 polygon IoU 和 OBB 关联 |
| **无集成测试** | 无端到端测试验证采集→推理→显示的完整链路 |
| **无性能回归测试** | A/B 测试协议已设计（`docs/performance-degradation-analysis.md` 4.2 节）但未自动化 |

### 12.3 文档层面

| 问题 | 说明 |
|------|------|
| 设计文档分散在 `docs/` + 根目录多个 markdown | 可整合为统一的设计文档索引 |
| 部分文档已过时（如 `架构梳理报告.md` 日期为 2026-07-08） | 本白皮书为最终权威版本 |

---

## 13. 风险分析

### 13.1 当前优点

- **模块化良好**：采集、推理、跟踪、GUI 各层职责清晰，依赖方向单向
- **解耦合理**：通过信号槽实现采集→推理→显示的解耦，线程安全
- **易扩展**：新增模型后端只需实现 `predict(frame, conf) → list[Detection]` 接口
- **自适应强**：_FilterAdaptor + _HealthMonitor 双重自动恢复
- **参数保守化**：性能恢复后默认参数稳定可靠
- **纯 NumPy**：不依赖外部滤波库，零额外依赖引入
- **丰富的 GUI**：专业暗色主题 + 动画系统 + 骨架屏

### 13.2 当前风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| **InferWorker 单点故障** | 中 | 推理线程崩溃 → 全部摄像头无检测结果 | 当前有 `_HealthMonitor` 部分缓解，但未做线程重启 |
| **MediaPipe 兼容性** | 中 | 手部手势功能不可用 | 考虑替换为 ONNX 手势模型 |
| **PyAV DirectShow 稳定性** | 低 | 特定摄像头驱动可能导致采集中断 | `camera/enumerator.py` 三级枚举 + 拉流验证 |
| **ONNX 模型兼容性** | 中 | 非标准 ONNX 模型（如 PINTO0309 后处理）可能在新版 ORT 上行为变化 | 锁定 onnxruntime-directml 版本 |
| **无 GPU 时的性能** | 中 | CPU 下多路全功能推理可能低于实时 | 跳帧机制、UHD 64x64 轻量模型、功能开关 |
| **配置漂移** | 低 | QSettings 累积陈旧配置 | QSettings 本身稳定，键命名规范 |

### 13.3 单点故障点

| 组件 | 故障影响 | 严重程度 |
|------|----------|----------|
| InferWorker 线程 | 全部槽位无 AI 检测结果 | 高 |
| MainWindow | 整个应用崩溃 | 高 |
| 单个 SingleCameraWorker | 对应槽位无画面 | 低（不影响其他槽位） |

---

## 14. 未来规划

### 14.1 短期（1-2 周）

1. **修复 MediaPipe 手部手势**：升级 mediapipe 版本或替换为 ONNX 手势模型
2. **补充测试**：为 DetectionTracker、UKF 族、RectangleDetector 补充单元测试
3. **代码清理**：消除重复的 `_debug_report()`、统一 Detection dict/object 使用
4. **引入 FilterType 枚举**：消除魔法字符串

### 14.2 中期（1-2 月）

1. **InferWorker 拆分**：将 ~1790 行的单类拆分为独立管线模块
   - `DetectionPipeline`（去噪 + 推理 + 后处理）
   - `TrackingPipeline`（跟踪 + 重检测 + 外观）
   - `AttributePipeline`（手势 + 矩形 + 头部属性）
2. **健康监控增强**：增加推理线程自动重启机制
3. **模型热加载**：不重启推理线程即可切换模型
4. **性能基准测试**：建立自动化性能回归测试套件
5. **GPU 优先级调度**：DirectML provider 共享策略优化

### 14.3 长期（3-6 月）

1. **Web 远程监控**：新增 Flask/FastAPI 服务端，支持浏览器远程查看
2. **录像回放分析**：对已录制的 MP4 进行离线 AI 分析
3. **多人脸识别**：集成人脸检测 + 识别模型（如 ArcFace）
4. **事件报警系统**：自定义规则（如"区域入侵"、"人员倒地"）触发通知
5. **分布式部署**：多台 PC 的摄像头统一接入到中心管理端
6. **跨平台支持**：探索 Linux/macOS 兼容（需替换 DirectShow 为 V4L2/AVFoundation）

---

## 15. 开发建议

### 15.1 新模型接入指南

添加一个新的 AI 模型后端只需 3 步：

1. **实现后端类**（在 `ai/` 下新建文件）：

```python
class MyNewBackend:
    def __init__(self, model_path, ...):
        # 加载模型
        pass

    def predict(self, frame: np.ndarray, conf: float) -> list[Detection]:
        # 推理 → 返回 Detection 列表
        pass
```

2. **注册到 InferWorker**（在 `ai/inference.py` 中添加识别函数）：

```python
def _is_mynew_model(path: str) -> bool:
    return "mynew" in path.lower()

# 在 _initialize_backend() 中添加分支
if _is_mynew_model(path):
    return MyNewBackend(path)
```

3. **更新模型自动发现**（无需修改，models/ 目录扫描自动包含）

### 15.2 新功能接入指南

添加新的分析功能（类比头部属性分类的接入方式）：

1. 实现功能模块类
2. 在 `InferWorker` 中添加 `_attach_xxx()` 方法
3. 在 `SettingsDialog` 中添加配置控件
4. 在 `CameraCell` 中添加标签绘制
5. 在 `MainWindow._apply_ai_settings()` 中添加配置下发

### 15.3 编码规范

- 坐标统一归一化 `[0, 1]`，不混用像素坐标
- 检测结果统一使用 `Detection` dataclass 或 `to_dict()` 序列化
- 配置持久化使用 `QSettings(org=VisionBata, app=VisionDataPlatform)`
- 推理线程参数设置走 `InferWorker.set_*()` 方法
- 卡尔曼滤波保持纯 NumPy 实现，不引入外部滤波库

### 15.4 调试技巧

- `error.log`：所有未捕获异常自动记录（含时间戳 + traceback）
- 推理线程日志：`logging.getLogger("ai.inference")`
- 跟踪器日志：`logging.getLogger("ai.tracker")`
- CameraCell 调试报告：每 300 帧输出一次重绘统计

---

## 16. Agent 接入指南

### 16.1 关键文件索引

| 文件 | 必读 | 说明 |
|------|------|------|
| `main.py` | ✅ | 入口与启动流程 |
| `ai/inference.py` | ✅ | **核心**：InferWorker 推理线程 |
| `ai/tracker.py` | ✅ | **核心**：DetectionTracker + UKF 族 |
| `ai/detection.py` | ✅ | **核心**：Detection/Keypoint 数据模型 |
| `ai/model_task.py` | ✅ | ModelTask 枚举 |
| `gui/main_window.py` | ✅ | 主窗口 + 信号槽调度 |
| `gui/camera_cell.py` | 推荐 | 画面显示 + 叠加绘制 |
| `gui/settings_dialog.py` | 推荐 | 配置对话框 + QSettings 键 |
| `camera/capture.py` | 按需 | 摄像头采集 |
| `camera/recorder.py` | 按需 | 录像功能 |

### 16.2 快速理解路径

如果要让 AI Agent 快速理解项目，建议按以下顺序阅读：

1. 本白皮书（PROJECT_BIBLE.md）
2. `ai/detection.py`（理解核心数据类型）
3. `ai/tracker.py` 前 80 行（理解 UKF 基类和跟踪接口）
4. `ai/inference.py` 的 `run()` 方法（理解推理管线）
5. `gui/main_window.py` 的 `_on_camera_selected()` 方法（理解信号槽链路）

### 16.3 常见任务切入点

| 任务 | 切入文件 | 重点关注 |
|------|----------|----------|
| 添加新检测模型 | `ai/inference.py::_initialize_backend` + `_run_inference` | 模仿 `_is_uhd_model` 模式 |
| 修改跟踪算法 | `ai/tracker.py::_DetectionUKF` | UKF 参数在类常量中 |
| 调整 GUI 布局 | `gui/main_window.py::_build_control_panel` | QVBoxLayout 结构 |
| 修改检测框绘制 | `gui/camera_cell.py::_rebuild_overlay_cache` | QPainter 绘制逻辑 |
| 调整配置项 | `gui/settings_dialog.py::__init__` | QSettings 键 + 控件创建 |
| 添加手势 | `ai/gesture_recognizer.py::detect` | 17 点关键点的几何规则 |

### 16.4 关键设计决策

| 决策 | 原因 | 不应轻易更改 |
|------|------|-------------|
| 1 个共享推理线程 | 避免 GPU 资源争抢和多模型实例内存爆炸 | ✅ |
| 纯 NumPy 卡尔曼 | 零外部滤波库依赖，完全可控 | ✅ |
| 归一化坐标 | 分辨率无关，跨摄像头一致 | ✅ |
| QSettings 持久化 | 跨会话配置记忆 | ✅ |
| 信号槽解耦 | 线程安全，独立生命周期 | ✅ |
| Detection dict 兼容层 | 旧模块（手势/矩形/头部属性）仍使用 dict | 可逐步迁移为 Detection 对象 |

### 16.5 模型文件命名约定

| 命名模式 | 识别为 | 后端 |
|----------|--------|------|
| `*yolov8*` | YOLOv8 检测/姿态（默认） | OnnxYoloBackend 或 ultralytics |
| `*yolo26*` / `*yolov10*` | YOLO26（自动推断任务类型） | Yolo26Backend |
| `*obb*` | OBB 旋转框任务 | Yolo26Backend (OBB) |
| `*pose*` | 姿态估计任务 | OnnxYoloBackend/Yolo26Backend (POSE) |
| `*uhd*` / `*ultratinyod*` | UHD 人体检测 | UhdBackend |
| `*chc*` | 头部属性分类 | HeadClassificationBackend |

---

> **本文档为 VisionBata 项目的权威技术白皮书。所有新开发者、AI Agent 和维护者应首先阅读本文档以理解项目全貌。**
>
> 文档维护者：项目架构师（通过 WorkBuddy 自动生成）  
> 生成日期：2026-07-10
