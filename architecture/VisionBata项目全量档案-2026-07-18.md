# VisionBata 项目全量档案

> **打包日期**：2026-07-18
> **内容**：项目目录树 + PROJECT_BIBLE 技术白皮书 + EventBus 阶段验收审计报告
> **总 Python 源文件**：59 个，约 20,420 行
> **项目版本**：1.1.1

---

# 第一部分：项目目录树

> **范围**：`F:\VisionBata` 全量（不含 venv/、`.workbuddy/`、`__pycache__/`）

```
VisionBata/
│
├── main.py                                120行  应用入口：异常钩子 + 高DPI + QApplication 引导
├── start.bat                                      Windows 启动脚本（硬编码 venv 路径）
├── requirements.txt                               第三方依赖清单
├── error.log                                      ~17MB 运行错误日志
├── PROJECT_BIBLE.md                               项目权威技术白皮书
├── demo.py                                552行  演示脚本（独立于主应用）
├── debug-yolo-detection-overlay.md                 YOLO 检测叠加层调试记录
│
├── models/                                        模型文件（运行时依赖，非源码）
│   ├── yolov8n.pt                                 YOLOv8 检测模型
│   ├── yolov8n.onnx                               YOLOv8 ONNX 版本
│   ├── yolov8n-pose.pt                            YOLOv8 姿态估计
│   ├── yolov8n-pose.onnx                          YOLOv8-pose ONNX 版本
│   ├── yolo26n.pt                                 YOLO26 检测模型
│   ├── yolo26n-obb.pt                             YOLO26 OBB 旋转框模型
│   ├── yolo26s.pt                                 YOLO26 小模型
│   ├── uhd_n_64x64_static.onnx                    PINTO0309 UHD 人体检测
│   ├── chc_s_wo_fiqa.onnx                         PINTO0309 CHC 头部属性分类
│   └── test_obb.jpg                               OBB 测试图片
│
├── ai/                                            ★ 推理与识别核心（13文件, 5,527行）
│   ├── __init__.py                         1行    仅导出 InferWorker
│   ├── inference.py                    2,018行    ★ InferWorker 共享推理线程 + EventBus Shadow
│   ├── tracker.py                      1,423行    ★ DetectionTracker + UKF/MCUKF/Manifold + AUTO
│   ├── detection.py                      196行    Detection/Keypoint 统一数据模型 + polygon IoU
│   ├── model_task.py                      12行    ModelTask 枚举 (DETECT/POSE/OBB)
│   ├── onnx_yolo_backend.py              322行    ONNX Runtime/DirectML YOLO 后端
│   ├── yolo26_backend.py                 158行    YOLO26 .pt 后端（ultralytics）
│   ├── uhd_backend.py                    132行    PINTO0309 UHD 64x64 人体检测
│   ├── head_classifier.py                171行    CHC 头部属性分类
│   ├── reid_backend.py                   136行    ReID 重识别嵌入后端（mock 256维）
│   ├── rectangle_detector.py             549行    矩形检测 + 线性卡尔曼跟踪
│   ├── gesture_recognizer.py             180行    身体姿态手势（17 COCO 关键点几何）
│   └── hand_gesture_recognizer.py        229行    手部手势（MediaPipe Hands，已禁用）
│
├── gui/                                            PyQt6 界面层（8文件, 5,252行）
│   ├── __init__.py                         0行    空包初始化
│   ├── main_window.py                  1,310行    ★ MainWindow 主窗口 + 信号槽调度中心
│   ├── camera_cell.py                  1,358行    ★ CameraCell 单路画面 + 叠加绘制
│   ├── settings_dialog.py              1,133行    SettingsDialog 配置编辑 + QSettings 持久化
│   ├── theme.py                          799行    冷蓝灰暗色 Fusion QSS 主题
│   ├── animations.py                     465行    Qt 动画系统（12种快捷动画）
│   ├── splash_screen.py                  152行    SplashScreen 启动闪屏
│   └── safe_widgets.py                    35行    防滚轮抢焦点控件
│
├── camera/                                         采集与录像层（4文件, 591行）
│   ├── __init__.py                         3行    导出相机枚举/采集/录像组件
│   ├── enumerator.py                     289行    CameraInfo/scan_cameras 三级枚举
│   ├── capture.py                        137行    SingleCameraWorker(PyAV dshow) + CameraCapture
│   └── recorder.py                       162行    VideoRecorder(H.264 MP4) + 守护线程池
│
├── common/                                         公共抽象层（3文件, 320行）
│   ├── __init__.py                       109行    Protocol 接口 + Detection/DetectionTracker 重导出
│   ├── pyav_capture.py                   107行    PyAV 采集后端实现
│   └── yolo_detector.py                  104行    YOLO 检测后端实现
│
├── visioncore/                                     ★ 新架构重构层（20文件, 3,909行）
│   ├── __init__.py                        41行    包入口，版本标记
│   │
│   ├── core/                                      核心数据模型（6文件）
│   │   ├── __init__.py                    77行    统一导出
│   │   ├── detection.py                   82行    CoreDetection 数据类
│   │   ├── target.py                     139行    Target 数据类（frozen slots）
│   │   ├── track.py                       72行    Track 数据类
│   │   ├── frame.py                       72行    Frame 数据类
│   │   ├── event.py                       66行    Event 数据类
│   │   └── adapters.py                   308行    ai↔visioncore 桥接（TYPE_CHECKING）
│   │
│   ├── eventbus/                                  ★ 发布-订阅基础设施 — M3新增（6文件, 1,286行）
│   │   ├── __init__.py                    83行    包导出：EventBus + 6事件类
│   │   ├── bus.py                        480行    ★ EventBus 核心（RLock + 锁外分发 + 双模式订阅）
│   │   ├── events.py                     241行    类型化事件层级（BaseEvent + 5生命周期事件）
│   │   ├── subscriber.py                 175行    Subscriber dataclass + EventListener 别名
│   │   ├── dispatcher.py                  90行    Dispatcher 同步分发引擎（异常隔离）
│   │   └── debug_logger.py               217行    DebugEventLogger 可切换通配符调试观察者
│   │
│   └── target_manager/                            目标管理业务逻辑（6文件, 1,766行）
│       ├── __init__.py                    91行    统一导出 + EventBus 集成说明
│       ├── manager.py                    718行    ★ TargetManager + EventBus + 生命周期事件发布
│       ├── lifecycle.py                  319行    TargetLifecycleManager 纯状态机
│       ├── store.py                      208行    TargetStore 多槽位存储
│       ├── converters.py                 348行    Detection↔Track 转换器
│       └── target_id_factory.py           82行    TargetIdFactory 统一ID生成
│
├── tests/                                          ★ 测试（8文件, 3,849行, 260项）
│   ├── test_event_bus.py               1,112行    77项 EventBus 全量测试（★ 新增）
│   ├── test_target_manager.py            947行    76项 TargetManager 测试
│   ├── test_target_manager_events.py     492行    29项 生命周期事件测试（★ 新增）
│   ├── test_core_models.py               396行    31项 核心数据模型测试
│   ├── test_debug_logger.py              367行    20项 DebugEventLogger 测试（★ 新增）
│   ├── test_shadow_integration.py        282行    12项 Shadow 集成端到端测试（★ 新增）
│   ├── test_matching.py                  141行    8项 跟踪匹配策略测试
│   └── test_obb_stage2.py                112行    7项 OBB polygon IoU 测试
│
├── tools/                                          工具（1文件, 300行）
│   └── mot_eval.py                       300行    离线 MOT 评估工具
│
├── docs/                                           设计文档（12份）
│   ├── auto-filter-adaptive-plan.md                AUTO 自适应滤波方案
│   ├── denoise-bilateral-nlmeans-plan.md            去噪方案
│   ├── eventbus-audit-report.md                     EventBus 阶段验收审计
│   ├── mcukf-manifold-ukf-plan.md                   MCUKF/Manifold UKF 方案
│   ├── milestone2-audit-report.md                   Milestone 2 审计
│   ├── milestone2-cleanup-report.md                 Milestone 2 清理
│   ├── multislot-isolation-report.md                多槽位隔离修复
│   ├── performance-degradation-analysis.md          性能下降分析
│   ├── pinto0309-uhd-chc-integration-plan.md        PINTO0309 集成方案
│   ├── project-audit-report.md                      项目审计
│   ├── sentineltrack-2.0-architecture-blueprint.md  SentinelTrack 2.0 蓝图
│   └── strategic-roadmap.md                         战略路线图
│
├── architecture/                                   架构文档归档
│   ├── 架构审计报告-2026-07-16.md                    v1.0 审计
│   ├── 架构审计报告-2026-07-18.md                    v2.0 审计（含 EventBus）
│   ├── 架构总结-2026-07-14.md                        第三版总结
│   ├── 架构梳理报告-2026-07-08.md                    初版
│   ├── 架构梳理报告-2026-07-09.md                    第二版
│   ├── 项目目录树-2026-07-18.md                      目录树
│   ├── eventbus-audit-report.md                     EventBus 审计
│   ├── PROJECT_BIBLE.md                             白皮书
│   └── VisionBata项目全量档案-2026-07-18.md         本文件
│
└── recordings/                                     录像输出
    ├── slot_0_20260609_205202.mp4
    └── slot_0_20260707_022612.mp4
```

### 按目录规模汇总

| 目录 | 文件数 | 行数 | 占比 |
|------|--------|------|------|
| `ai/` | 13 | 5,527 | 27.1% |
| `gui/` | 8 | 5,252 | 25.7% |
| `visioncore/` | 20 | 3,909 | 19.1% |
| `tests/` | 8 | 3,849 | 18.9% |
| `camera/` | 4 | 591 | 2.9% |
| `common/` | 3 | 320 | 1.6% |
| `tools/` | 1 | 300 | 1.5% |
| 顶层脚本 | 2 | 672 | 3.3% |
| **总计** | **59** | **20,420** | **100%** |

---

# 第二部分：PROJECT_BIBLE — 项目技术白皮书

> **版本**：1.1.1
> **最后更新**：2026-07-10
> **目标读者**：新开发者、AI Agent、项目维护者

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
| **手部手势** | mediapipe | ≥0.10.0 | 手部关键点（当前版本兼容性问题，已禁用） |

### 4.2 技术选型理由

| 选型 | 理由 |
|------|------|
| **PyQt6** | Windows 原生体验、丰富的信号槽机制、成熟的 Fusion 主题、QThread 线程模型 |
| **PyAV (av)** | 比 OpenCV VideoCapture 更稳定的 DirectShow 支持、原生 H.264 编码、帧级控制 |
| **纯 NumPy 卡尔曼** | 无外部滤波库依赖（不依赖 filterpy）、完全可控的参数调优、三种 UKF 变体统一接口 |
| **ONNX Runtime DirectML** | Windows 原生 GPU 加速、无需 CUDA、AMD/NVIDIA/Intel 通用 |
| **PINTO0309 模型** | 社区广泛验证的轻量 ONNX 模型、64x64 超低分辨率输入、CPU 友好 |

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
QApplication 主线程 (GUI)
  MainWindow → CameraCell × 4 → paintEvent
                   ▲
    detection_ready (QueuedConnection)
                   │
  ┌────────────────┴───────────────────┐
  │   InferWorker 线程 (推理线程)       │
  │   共享1个，服务全部 4 个槽位         │
  └────────────────────────────────────┘
                   ▲
     submit_frame (DirectConnection)
                   │
  ┌────────────────┴───────────────────┐
  │ SingleCameraWorker × N (采集线程)   │
  │ 每个槽位 1 个 PyAV DirectShow 线程   │
  └────────────────────────────────────┘

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

## 6. 模块说明（核心模块节选）

### `ai/inference.py` — InferWorker 核心推理线程 (2,018行)

★ 整个项目最核心的模块。

推理管线（每帧命中 stride 时）：
1. 可选去噪预处理 (_denoise_frame)
2. 多后端推理 (_run_inference)：OnnxYoloBackend / Yolo26Backend / UhdBackend
3. DetectionTracker.update() + ROI 重检测 + HSV 外观
4. _attach_head_attributes (CHC 头部属性)
5. _attach_gestures (身体姿态手势)
6. _run_rectangle_detection (Canny + 轮廓)
7. final_dedupe() + _HealthMonitor 健康检查

内部类：`_HealthMonitor`（3秒滑动窗口置信度监控 → 异常自动回退）、`_FrameSlot`（双缓冲槽位）、`SuperResEngine`（ESPCN x4）

### `ai/tracker.py` — DetectionTracker 多目标跟踪器 (1,423行)

★ 项目算法核心。

| 类 | 职责 |
|-----|------|
| `_DetectionUKF` | 基础无迹卡尔曼（8 维状态，17 sigma 点，Cholesky 分解） |
| `_DetectionMCUKF` | 最大相关熵 UKF（高斯核加权，抗离群值，不动点迭代） |
| `_DetectionManifoldUKF` | 流形 UKF（eigh 特征分解 + SPD 投影，P 永久正定） |
| `_FilterAdaptor` | AUTO 自适应切换（监控 NIS/缺失率/P条件数/Cholesky 失败） |
| `_Track` | 单条轨迹（滤波器 + 外观直方图 + OBB polygon 管理） |
| `DetectionTracker` | 主跟踪器（匈牙利匹配 + 轨迹生命周期管理） |

关键参数（已保守化）：α=1e-3, β=2.0, κ=0；P_MAX_POSITION=0.05/P_MAX_SIZE=0.02/P_MAX_VELOCITY=0.01；自适应 Q/R；速度限幅 [-0.3, 0.3]

### `ai/detection.py` — 统一检测类型

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

---

## 7. 数据流

### 7.1 端到端数据流

```
USB 摄像头 → PyAV dshow → SingleCameraWorker
  ├─→ CameraCell.update_frame [QueuedConnection, 显示]
  ├─→ InferWorker.submit_frame [DirectConnection, 推理]
  │     → 去噪 → 推理 → Tracking → 属性附加 → 去重 → HealthMonitor
  │     → detection_ready.emit → CameraCell.update_detections
  │     → _bypass_update_targets → TargetManager → EventBus ★ Shadow
  └─→ VideoRecorder.push_frame [录制]
```

### 7.2 坐标约定

- **全程归一化 [0, 1]**：检测框、关键点、多边形顶点
- **绘制映射**：乘以控件宽高还原像素坐标
- **水平翻转**：x = 1.0 - x

### 7.3 配置持久化

QSettings (org=VisionBata, app=VisionDataPlatform)：
ai/conf_threshold / ai/infer_stride / ai/smooth_alpha / ai/filter_type /
ai/denoise_method / ai/matching_strategy / ai/head_classifier_enabled / ...（共39键）

---

## 8. 算法设计

### UKF 滤波器族

```
_DetectionUKF (基类)
  8维状态: [cx, cy, w, h, vx, vy, vw, vh]
  17 sigma 点 (n=8), Cholesky 分解, 自适应 Q/R
  ↓
_DetectionMCUKF: 高斯核加权 + 不动点迭代
_DetectionManifoldUKF: eigh 特征分解 + SPD 投影

_FilterAdaptor:
  cond>1000 或 chol_fail>2 → manifold_ukf
  NIS>8.0 或 miss_rate>50% → mcukf
  否则 → ukf
  迟滞: 连续2次评估一致才切换
```

### ROI 重检测算法

```
person track 丢失 → 卡尔曼预测框 → 扩展ROI → 裁剪+低置信度重跑YOLO
→ 质量门控(IoU>0.3, conf>original×0.6, scale∈[0.5,1.5])
→ 通过则 track.correct / 失败则 miss+=1
```

---

## 9. 技术债务

| 问题 | 位置 | 建议 |
|------|------|------|
| `Detection` 双重形式（dataclass + dict） | 全局 | 统一使用 Detection 对象 |
| `InferWorker` 过大 | `ai/inference.py` ~2,018行 | 拆分为独立管线 |
| 魔法字符串（filter/denoise/matching） | 各处 | 引入枚举类型 |
| 测试覆盖极低 | `ai/` 核心模块 | 补充 UKF/推理管线测试 |
| `common/` 依赖 `ai/`（逆依赖） | `common/__init__.py` | 重构接口方向 |

---

## 10. 开发指南

### 新模型接入 3 步：

1. 实现后端类：`predict(frame, conf) -> list[Detection]`
2. 注册到 InferWorker：`_is_xxx_model(path)` + `_initialize_backend` 分支
3. 无需修改模型自动发现（models/ 目录扫描）

### 新功能接入 5 步：

1. 实现功能模块类
2. InferWorker 加 `_attach_xxx()`
3. SettingsDialog 加配置控件
4. CameraCell 加标签绘制
5. MainWindow._apply_ai_settings() 加配置下发

### 模型文件命名约定

| 命名模式 | 识别为 | 后端 |
|----------|--------|------|
| `*yolov8*` | YOLOv8 | OnnxYoloBackend / ultralytics |
| `*yolo26*` / `*yolov10*` | YOLO26 | Yolo26Backend |
| `*obb*` | OBB 旋转框 | Yolo26Backend (OBB) |
| `*pose*` | 姿态估计 | OnnxYoloBackend / Yolo26Backend (POSE) |
| `*uhd*` / `*ultratinyod*` | UHD 人体检测 | UhdBackend |
| `*chc*` | 头部属性分类 | HeadClassificationBackend |

---

# 第三部分：EventBus 阶段验收审计报告

> **审计日期**：2026-07-17
> **审计范围**：`visioncore/eventbus/` + `visioncore/target_manager/manager.py` 的 EventBus 集成 + 全部相关测试
> **审计目标**：线程安全 / 重复订阅 / 取消订阅 / 事件顺序 / TargetManager 集成 / 异常处理 / 测试覆盖率

---

## E1. 架构评审

### E1.1 模块组成

| 模块 | 职责 | 状态 |
|------|------|------|
| `eventbus/bus.py` | `EventBus` 主体：subscribe/unsubscribe/publish/计数，RLock 保护，锁内快照+锁外分发 | ✅ 健康 |
| `eventbus/subscriber.py` | `Subscriber`：双模式匹配（字符串/类 isinstance），active 软删除 | ✅ 健康 |
| `eventbus/dispatcher.py` | `Dispatcher`：无状态同步分发，异常隔离 | ✅ 健康 |
| `eventbus/events.py` | `BaseEvent` + 5 个 Target 生命周期事件，frozen+slots+repr=False | ✅ 健康 |
| `eventbus/debug_logger.py` | `DebugEventLogger`：通配订阅，多行日志，enable/disable 幂等 | ✅ 健康 |
| `target_manager/manager.py` | `TargetManager`：`_emit_lifecycle_event` 单点发布，event_bus 可选注入 | ✅ 健康（补丁后） |

### E1.2 线程安全评审

| 组件 | 机制 | 评审结论 |
|------|------|----------|
| `EventBus` | `threading.RLock` 保护所有注册表操作；publish 锁内快照、锁外分发 | ✅ 正确。RLock 可重入，回调可安全重入 |
| `Subscriber.active` | 无锁 bool，跨线程读写 | ✅ CPython GIL 下原子且可见 |
| `Dispatcher` | 无状态，操作 bus 传入的 snapshot | ✅ 无并发风险 |
| `TargetManager._emit` | 调用 bus.publish（bus 自带锁） | ✅ 补丁后异常隔离 |
| `DebugEventLogger._enabled` | 无锁 bool，enable/disable 非原子 | ⚠️ P3：实际风险低 |

### E1.3 关键设计评审

- **重复订阅**：同 callback 同 type → 独立 Subscriber，publish 时各调用一次 ✅
- **取消订阅**：接受 Subscriber 或 str id，幂等；软删除 active=False ✅
- **事件顺序**：同步分发按订阅插入顺序 ✅

---

## E2. 发现的问题

### P1（重要，已修复）— `_emit_lifecycle_event` 未隔离异常

**位置**：`visioncore/target_manager/manager.py`

**现象**：事件构造或 bus.publish() 若抛异常，异常传播到 mark_lost/lock_target 等转换方法，导致返回异常而非 bool。

**补丁**：`_emit` 加 try/except，异常 logger.exception 吞掉，核心逻辑不受影响。

### P2（中等，已修复）— 事件 timestamp 来源不一致

**现象**：lock_target/mark_removed 用硬编码 0.0，create_target 用 last_seen。

**补丁**：lock_target/mark_removed 加可选 timestamp 参数（默认 0.0，向后兼容），update_targets sweep 调用传 timestamp。

### P2b（建议）— CreatedEvent timestamp 语义不准

create_target 用 `last_seen` 而非创建时间。留待后续。

### P3（轻微）— DebugEventLogger._enabled 非线程安全

并发的 enable/disable 可能重复订阅，实际风险低。留待后续。

### P4（轻微）— TargetManager 单目标方法不持锁

依赖 InferWorker 单线程推理假设。当前架构安全，留待后续。

---

## E3. 最终 EventBus 数据流

```
Tracker (ai/tracker.py) → detections
    ↓
InferWorker._bypass_update_targets
    self._event_bus = EventBus()  ← Shadow Integration (M3)
    detections_to_tracks(detections)
    self._target_mgr.update_targets(tracks)
    ↓
TargetManager
    create_target → _emit → TargetCreatedEvent  → EventBus.publish
    mark_lost     → _emit → TargetLostEvent     → EventBus.publish
    mark_recovered→ _emit → TargetRecoveredEvent→ EventBus.publish
    lock_target   → _emit → TargetLockedEvent   → EventBus.publish
    mark_removed  → _emit → TargetRemovedEvent  → EventBus.publish
    ↓
EventBus
    publish(event):
      with RLock: 快照 candidates → 释放锁
      dispatcher.dispatch(matching, event)  ← 锁外分发
    Subscriber.matches: isinstance(类) / "=="(字符串) / True(通配)
    Dispatcher: for sub in matching: sub.deliver(event)
    ↓
【当前无订阅者 — 管道验证通过，O(1) 空操作】
  (未来) DebugEventLogger / 告警 / UI高亮 / 录像标记
```

---

## E4. 验收结论

**总测试**：213 项全部通过

| 验收项 | 结论 |
|------|------|
| 线程安全 | ✅ 通过（EventBus RLock + 锁外分发） |
| 重复订阅 | ✅ 通过 |
| 取消订阅 | ✅ 通过（幂等 + 软删除） |
| 事件顺序 | ✅ 通过（同步分发 + 订阅顺序） |
| TargetManager 集成 | ✅ 通过（P1 异常隔离已补丁） |
| 异常处理 | ✅ 通过（dispatcher 回调隔离 + _emit 异常隔离） |
| 测试覆盖率 | ✅ 良好（213 项，含异常隔离/并发/边界） |

**P1 已修复并测试验证。P2 已修复（timestamp 一致性）。P3/P4 为低风险建议。EventBus 阶段验收通过。**

---

> **本档案由三份权威文档合并而成**：项目目录树（Part 1）、PROJECT_BIBLE 项目白皮书（Part 2）、EventBus 阶段验收审计报告（Part 3）。
> 生成日期：2026-07-18 | 零代码修改 | 归档位置：`F:\VisionBata\architecture\`
