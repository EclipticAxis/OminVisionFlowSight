# 深度视频（yolo26s-depth 单目深度估计）设计文档

> **版本**：0.2.0 — 2026-11-28
> **范围**：`ai/depth_worker.py`、`gui/main_window.py`（深度开关、真实 Qt 信号路由）、`gui/camera_cell.py`、`gui/settings_dialog.py`、`requirements.txt`、`tests/`（DepthWorker 24 项 + 显示/路由/错误 9 项）；模型文件 `models/yolo26s-depth.pt` 为部署依赖
> **状态**：工作区已实现，待合并；PR27 审查修复已完成，33 项深度专项测试通过；完整套件 1560 项通过、1 项跳过

---

## 1. 目标与背景

主链路目前的视觉理解全部基于 2D 检测/跟踪/姿态，无法回答"目标离我多远"。
本特性为每路摄像头提供**单目深度估计实时预览**：基于 ultralytics 的
yolo26s-depth 模型，把每帧画面转成一张伪彩色深度图并整幅替换该路画面显示。

定位与边界必须先说清：

- **这是单目深度估计，不是深度相机**。输出是网络从单张 RGB 图像估计的
  米制深度场，经固定 0–20 m 量程着色；数值存在场景先验与尺度偏差，
  文档与 UI 文案均不得将其表述为测距传感器精度。
- **深度模式与检测互斥（每路）**。深度推理与 YOLO 检测共享 GPU，同开会导致
  两边都掉帧；开启深度时该路暂停普通检测推理，检测框/骨骼/手势均不绘制，
  把算力让给深度。这是有意的资源取舍，不是缺陷。
- **不做时域平滑**。不插帧、不与彩色帧混合，深度图逐帧独立着色整幅替换，
  避免引入无意义的颜色伪影和"假流畅"。

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
    → 固定 0–20 m metric 量程 → OpenCV Inferno 着色 → BGR 转 RGB
    ▼
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
- **closeEvent**：停止 `DepthWorker`（`stop(wait_ms=30000)`，等待线程退出）。

## 3. 契约表

| 项 | 契约 |
|----|------|
| 工作线程 | `ai/depth_worker.py::DepthWorker(QThread)`，单实例多路共享单模型；独立于 `InferWorker`，互不复用 |
| 模型加载 | **懒加载**：仅首次有帧到达时在工作线程内加载 `models/yolo26s-depth.pt`；CUDA 可用用 device `0`，否则 CPU；`imgsz=640` |
| 依赖下限 | `ultralytics>=8.4.148`，与当前实测环境一致，确保提供 YOLO26 depth task 与 `Results.depth` |
| 输入/输出 | 输入 RGB 帧；输出连续 `uint8` RGB 深度图（`result.depth.data` resize 到输入尺寸 → 固定 0–20 m metric 量程 → OpenCV Inferno 着色 → BGR→RGB） |
| 帧调度 | 每 slot 仅保留最新帧（新帧覆盖旧帧，不排队积压）；多路经 deque 公平轮询；`QMutex`/`QWaitCondition` 唤醒与等待 |
| 公共接口 | `set_enabled(bool)` / `register(slot)` / `unregister(slot)` / `stop()` |
| 信号 | `depth_ready(int slot, object image)`；`error_occurred(int slot, str message)`，**模型级错误 slot = -1** |
| 显示语义 | 深度模式下整幅替换（非叠加、非混合）；未收到首帧深度图前短暂显示原帧但不画检测框；不做插帧、不做时域平滑 |
| 互斥语义 | 深度开启 ⇒ 仅断开该路 `InferWorker.submit_frame`，不注销/重建 tracker，检测框/骨骼/手势均不绘制；关闭深度或总 AI ⇒ 原帧立即恢复，总 AI 仍开时恢复 `InferWorker` |
| 信号连接 | `_disconnect_all` 循环移除历史重复槽；`_safe_direct_connect` 不使用非法 `UniqueConnection | DirectConnection` 位运算，每次路由先清理再建立至多一条直连；真实 Qt 信号回归测试锁定 |
| 开关与持久化 | 默认**关闭**；GUI “深度视频”开关位于 AI 功能面板；生效条件 = 总 AI 开关 AND 深度开关；状态持久化于 QSettings 键 `ai/depth_enabled` |
| 错误回退 | 模型级 `slot=-1` 错误自动关闭深度开关并回退原帧；单帧 `slot>=0` 错误只跳过当前帧、保留深度模式并继续；dashboard 按每路 1 秒时间窗节流；已停用后的 queued 模型错误不改写开关 |
| 数值与缓冲 | NaN/+Inf/-Inf 统一映射低端；`CameraCell.update_depth_frame` 对非连续 numpy 输入先转连续 RGB 缓冲再创建 QImage |
| 资源清理 | `DepthWorker` 由 MainWindow 持有 parent；摄像头切换清理该路深度帧；`closeEvent` 调 `DepthWorker.stop(wait_ms=30000)`，协作等待在飞推理结束，不使用 `QThread.terminate()` |
| 性能基线 | 单路实测（RTX 5080 Laptop，imgsz 640）：推理+传输+着色中位 **15.6 ms（约 64 fps）**；默认采集 640×480@30。多路共享单模型，实际 fps 按总算力分配，**不承诺所有路 30 fps** |
| 能力边界 | 单目深度估计 ≠ 深度相机：0–20 m 为固定显示量程而非测量精度承诺；禁止在 UI/文档中暗示测距级准确度 |
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
- **测试维护**：DepthWorker 专项 24 项（`tests/test_depth_worker.py`）覆盖
  懒加载/设备选择/调度/信号/停止契约，改动 `ai/depth_worker.py` 后必跑；
  GUI 集成测试 9 项（`tests/test_depth_display.py`）覆盖开关互斥、真实 Qt 单连接、
  tracker 保留、错误回退和非连续帧；closeEvent 清理由真实主窗口集成冒烟验证。
  新增行为请先加测试再改实现。
