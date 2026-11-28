# 深度视频（yolo26s-depth 单目深度估计）设计文档

> **版本**：0.3.0 — 2026-11-28
> **范围**：`ai/depth_worker.py`、`gui/main_window.py`、`gui/camera_cell.py`、`gui/settings_dialog.py`、`tools/benchmark_depth_video.py`、`tests/`（DepthWorker 52 项 + 显示/路由/错误/色标 10 项 + 侧栏联动 1 项）；模型文件 `models/yolo26s-depth.pt` 为部署依赖
> **状态**：0.2.0 已合入 main（PR #27 → `876dc07`）；0.3.0 与 PR28 审查修复待合并；63 项相关专项通过，完整套件 1590 项通过、1 项跳过；真实主窗口集成验证通过

---

## 1. 目标与背景

主链路目前的视觉理解全部基于 2D 检测/跟踪/姿态，无法回答"目标离我多远"。
本特性为每路摄像头提供**单目深度估计实时预览**：基于 ultralytics 的
yolo26s-depth 模型，把每帧画面转成一张伪彩色深度图并整幅替换该路画面显示。

定位与边界必须先说清：

- **这是单目深度估计，不是深度相机**。输出是网络从单张 RGB 图像估计的
  米制深度场，经固定或稳定自适应量程着色；数值存在场景先验与尺度偏差，
  文档与 UI 文案均不得将其表述为测距传感器精度。
- **深度模式与检测互斥（每路）**。深度推理与 YOLO 检测共享 GPU，同开会导致
  两边都掉帧；开启深度时该路暂停普通检测推理，检测框/骨骼/手势均不绘制，
  把算力让给深度。这是有意的资源取舍，不是缺陷。
- **只对米制深度做时域滤波**。不插帧、不与彩色帧混合；运动自适应 EMA
  先稳定深度数值，再用完整色带着色，避免无意义的颜色伪影和"假流畅"。

## 2. 架构

```
gui/main_window（“深度视频”开关，AI 功能面板；QSettings ai/depth_enabled）
    │ 前置条件：总 AI 开关 AND 深度开关 同时为真
    ▼
ai/depth_worker.DepthWorker(QThread)        ← 单实例，多路共享单模型
    │ 懒加载 models/yolo26s-depth.pt（仅首次有帧时在线程内加载；
    │ CUDA 可用 → device 0，否则 CPU；imgsz=640；输入 RGB）
    │ 每 slot 最新帧覆盖（old frame 丢弃）+ deque 公平轮询多路
    │ QMutex/QWaitCondition 唤醒
    ▼
result.depth.data（米制图）→ resize 到输入尺寸
    → 运动自适应 EMA（静态 alpha=0.25，突变 alpha=0.85）
    → 固定量程或 2%/98% 分位目标 + 慢速 EMA（alpha=0.05）
    → disparity=1/depth + gamma=0.5 → 默认 OpenCV Turbo（近暖远冷）
    → BGR 转 RGB
    ▼
信号 depth_range_ready(int slot, float vmin, float vmax)
    → 当前实际量程与色标
信号 depth_ready(int slot, object np.ndarray) ──► gui/camera_cell
    缓存最新深度 QImage；深度模式下整幅替换显示（非半透明叠加）
错误路径：error_occurred(int slot, str msg)，模型级错误 slot = -1
    ▼
模型级错误：GUI 自动关闭深度开关并回退原帧显示
单帧错误：GUI 保持深度模式，worker 丢弃该帧并继续后续帧
```

启停接线（`gui/main_window.py`）：

- **开启深度**：暂停该路普通检测推理——断开 `InferWorker.submit_frame`，
  检测框、骨骼、手势全部不画；`DepthWorker.register(slot)`。
  **不注销 `InferWorker` 槽位**，保留 tracker、track ID、速度和 ReID 状态。
- **关闭深度或关闭总 AI**：`DepthWorker.unregister(slot)`，画面恢复原始帧
  （camera_cell 持续接收原帧，所以恢复是即时的）；若总 AI 开关仍开，
  恢复该路 `InferWorker` 推理。
- **摄像头切换**：清理该 cell 的深度帧缓存，避免旧画面残留。
- **关闭/切换/分辨率变化**：清理对应深度 EMA 与量程状态，禁止跨会话串帧。
- **closeEvent**：停止 `DepthWorker`（`stop(wait_ms=30000)`，等待线程退出）。

## 3. 契约表

| 项 | 契约 |
|----|------|
| 工作线程 | `ai/depth_worker.py::DepthWorker(QThread)`，单实例多路共享单模型；独立于 `InferWorker`，互不复用 |
| 模型加载 | **懒加载**：仅首次有帧到达时在工作线程内加载 `models/yolo26s-depth.pt`；CUDA 可用用 device `0`，否则 CPU；`imgsz=640`；**不新增后台预热** |
| 依赖下限 | `ultralytics>=8.4.148`，与当前实测环境一致，确保提供 YOLO26 depth task 与 `Results.depth` |
| 输入/输出 | 输入 RGB 帧；原始米制深度先做运动自适应 EMA，再在米制量程内计算 disparity=1/depth，归一化应用 gamma=0.5 后着色；输出连续 `uint8` RGB 深度图，禁止直接混合彩色帧 |
| 时域稳定 | 静态像素 alpha=0.25；超过 `max(0.15m, 15%×previous)` 的突变像素 alpha=0.85；新出现像素直接使用当前值，无效像素不继承旧值 |
| 色带与量程 | 默认 Turbo + disparity + gamma=0.5，近处红橙、远处蓝紫；可选 Inferno/Plasma/Viridis 与固定 0–3/0–8/0–20 m；自动量程目标为 2%/98% 分位，范围 alpha=0.05，最小跨度 0.5 m；固定 0 米起档仅以 5%×vmax 作为着色近端，避免 1/0 奇点 |
| 帧调度 | 每 slot 仅保留最新帧（新帧覆盖旧帧，不排队积压）；多路经 deque 公平轮询；`QMutex`/`QWaitCondition` 唤醒与等待 |
| 公共接口 | `set_enabled(bool)` / `register(slot)` / `unregister(slot)` / `set_visualization(cmap, range_mode)` / `stop()` |
| 信号 | 成功时先发 `depth_range_ready(int slot, float vmin, float vmax)` 再发 `depth_ready(int slot, object image)`；per-slot tombstone 代际只影响对应采集位，cmap/量程变化使全部在飞旧帧失效；过期帧不写状态也不发射 |
| 显示语义 | 深度模式下整幅替换；色标使用当前实际米制量程调用同一 `colorize_depth`，缓存键为 `cmap+vmin+vmax`，逐像素与画面一致；色标锚定并随 `video_rect` 宽度收缩，量程未就绪时返回 None 并跳过绘制 |
| 配置持久化 | `ai/depth_colormap` 默认 `turbo`；`ai/depth_range_mode` 默认 `auto`，与 `ai/depth_enabled` 均通过 QSettings 保存 |
| 互斥语义 | 深度开启 ⇒ 仅断开该路 `InferWorker.submit_frame`，不注销/重建 tracker，检测框/骨骼/手势均不绘制；关闭深度或总 AI ⇒ 原帧立即恢复，总 AI 仍开时恢复 `InferWorker` |
| 信号连接 | `_disconnect_all` 循环移除历史重复槽；`_safe_direct_connect` 不使用非法 `UniqueConnection | DirectConnection` 位运算，每次路由先清理再建立至多一条直连；真实 Qt 信号回归测试锁定 |
| 开关与持久化 | 默认**关闭**；色带/量程选项仅在深度开关开启时展开，复用既有透明度动画契约；`ai/depth_enabled`、`ai/depth_colormap`、`ai/depth_range_mode` 均持久化 |
| 错误回退 | 模型级 `slot=-1` 错误自动关闭深度开关并回退原帧；单帧 `slot>=0` 错误只跳过当前帧、保留深度模式并继续；dashboard 按每路 1 秒时间窗节流；已停用后的 queued 模型错误不改写开关 |
| 数值与缓冲 | NaN/+Inf/-Inf 统一映射低端；`CameraCell.update_depth_frame` 对非连续 numpy 输入先转连续 RGB 缓冲再创建 QImage |
| 资源清理 | `DepthWorker` 由 MainWindow 持有 parent；摄像头切换清理该路深度帧；`closeEvent` 调 `DepthWorker.stop(wait_ms=30000)`，协作等待在飞推理结束，不使用 `QThread.terminate()` |
| 历史性能基线 | PR27 单路实测（RTX 5080 Laptop，imgsz 640）：推理+传输+着色中位 **15.6 ms（约 64 fps）**，不含 0.3.0 新增滤波与量程计算，不与本版直接作速度对比 |
| 0.3.0 实测 | 真实 ASUS FHD webcam + DepthWorker 核心链路约 **30.15 fps**；不同实时输入样本的配对颜色波动下降 **56.2%–78.4%**；真实主窗口与近暖远冷色标集成通过；未加入后台预热 |
| 可复现复核 | `tools/benchmark_depth_video.py` 支持视频/摄像头与 JSON 输出；当前 30 帧复核中位：推理 9.87ms、滤波 4.37ms、量程 1.28ms、raw+filtered 双着色 4.85ms；脚本 21.79fps 包含对照所需的双着色，产品只着色 filtered 一次 |
| 能力边界 | 单目深度估计 ≠ 深度相机：固定或自动显示量程都不是测量精度承诺；禁止在 UI/文档中暗示测距级准确度 |
| 录像语义 | 本特性只替换 GUI 预览；录像仍保存原始摄像头画面，不保存伪彩深度视频 |

## 4. 维护指南

- **改深度着色/量程**：量程（0–20 m）与 colormap（Inferno）在
  `ai/depth_worker.py` 的后处理段；改量程会影响所有路的显示语义，
  需同步更新本文档契约表。
- **改启停/互斥逻辑**：入口在 `gui/main_window.py` 的“深度视频”开关槽函数；
  注意三条不变式——(1) 深度开 ⇒ 该路 `InferWorker` 停止接收新帧但不销毁 tracker；
  (2) 恢复路径必须区分"总 AI 是否仍开"；(3) 错误回退只关深度开关，不动总 AI 开关。
  所有直连信号先 `_disconnect_all` 再连接，禁止恢复非法枚举位运算。
- **改显示**：`gui/camera_cell.py` 同时缓存原始 QImage 与最新深度 QImage，
  深度模式整幅替换；原帧持续接收是"关闭立即恢复"的前提，勿为省内存
  停收原帧。
- **模型部署**：`models/yolo26s-depth.pt` 是**部署依赖，不属于普通源码**；
  缺失时模型级错误经 `error_occurred(-1, ...)` 上报并触发自动回退，
  不得让异常逃逸到 GUI 线程。
- **测试维护**：DepthWorker 专项 52 项（`tests/test_depth_worker.py`）覆盖
  懒加载/设备选择/调度/信号/停止契约，改动 `ai/depth_worker.py` 后必跑；
  GUI 集成测试 10 项（`tests/test_depth_display.py`）覆盖开关互斥、真实 Qt 单连接、
  tracker 保留、错误回退、非连续帧与色标逐像素一致；侧栏新增深度面板联动测试。
  closeEvent 清理由真实主窗口集成冒烟验证。
  新增行为请先加测试再改实现。
- **实测复核**：运行 `venv\Scripts\python.exe tools\benchmark_depth_video.py --camera "ASUS FHD webcam" --frames 120`，或用 `--input <视频>`；`--output <path.json>` 可落盘报告。
