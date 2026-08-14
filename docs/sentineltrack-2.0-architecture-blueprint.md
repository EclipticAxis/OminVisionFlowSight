# SentinelTrack 2.0 Architecture Blueprint

> **文档类型**：技术架构蓝皮书（Architecture Blueprint）
> **版本**：2.0-draft-1
> **日期**：2026-07-11
> **角色**：首席架构师 / 架构评审委员会 / 资深软件工程师 / 平台架构负责人
> **目标**：指导未来 12~24 个月 SentinelTrack 2.0 架构重构的完整技术设计

---

## 目录

1. [总体架构设计](#1-总体架构设计)
2. [核心层详细拆分](#2-核心层详细拆分)
3. [插件系统设计](#3-插件系统设计)
4. [数据流架构](#4-数据流架构)
5. [Tracking Engine 深度设计](#5-tracking-engine-深度设计)
6. [跨平台架构](#6-跨平台架构)
7. [仓库重构方案](#7-仓库重构方案)
8. [架构决策记录（ADR）](#8-架构决策记录adr)

---

## 1. 总体架构设计

### 1.1 候选方案分析

#### 方案 A：Monolith（单体架构）

```
所有代码在一个进程、一个包中，模块间直接函数调用。
```

| 优点 | 缺点 |
|------|------|
| 开发简单，无通信开销 | 模块耦合严重，无法独立替换 |
| 调试方便 | 扩展需修改核心代码 |
| 部署简单 | 无法支持插件生态 |

**结论**：VDP 1.0 当前状态。不适合 2.0 目标。

#### 方案 B：Modular Monolith（模块化单体）

```
单进程，但模块边界严格，通过内部接口通信，无跨模块直接调用。
```

| 优点 | 缺点 |
|------|------|
| 边界清晰，可测试 | 无法运行时加载/卸载模块 |
| 无 IPC 开销 | 插件需编译期注册 |
| 部署简单（单包） | 无法支持第三方插件 |

**结论**：比 Monolith 好，但不满足插件市场和热加载需求。

#### 方案 C：Plugin-based Architecture（插件架构）

```
核心 + 插件管理器，插件通过注册接口加入系统，核心调度插件。
```

| 优点 | 缺点 |
|------|------|
| 运行时加载/卸载 | 核心仍可能膨胀 |
| 支持第三方插件 | 插件间通信需设计 |
| 可测试（插件独立测试） | 插件版本兼容复杂 |

**结论**：满足大部分需求，但核心边界定义不够严格。

#### 方案 D：Microkernel Architecture（微内核架构）

```
最小化核心（仅插件框架 + 事件总线 + 基础服务），所有功能以插件形式存在。
```

| 优点 | 缺点 |
|------|------|
| 核心极简，永不膨胀 | 设计复杂度高 |
| 完全可扩展 | 调试链路长（跨插件） |
| 插件可独立开发/测试/发布 | 性能有少量开销（事件总线） |
| 最适合平台化 | 需要严格的接口版本管理 |

**结论**：最适合 SentinelTrack 2.0 的长期平台化目标。

#### 方案 E：Event-driven Architecture（事件驱动架构）

```
所有组件通过事件总线通信，无直接调用。
```

| 优点 | 缺点 |
|------|------|
| 极度解耦 | 调试困难（事件溯源） |
| 天然异步 | 事件风暴风险 |
| 易于扩展 | 顺序保证复杂 |

**结论**：不适合作为主架构（实时视频处理需确定性顺序），但事件总线可作为微内核的通信机制。

#### 方案 F：Service-oriented Architecture（SOA / 微服务）

```
每个功能独立进程/服务，通过 RPC/消息队列通信。
```

| 优点 | 缺点 |
|------|------|
| 独立部署/扩展 | IPC 开销大（对实时视频不可接受） |
| 技术栈自由 | 运维复杂 |
| 故障隔离 | 帧传输序列化成本高 |

**结论**：不适合实时视觉处理（帧序列化开销不可接受）。可用于 Web 管理后台等非实时部分。

### 1.2 最终推荐方案

## **微内核 + 插件架构 + 进程内事件总线**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SentinelTrack 2.0 架构                            │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    微内核 (Microkernel)                       │   │
│  │                                                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│  │  │ Plugin       │  │ Event Bus    │  │ Config       │      │   │
│  │  │ Framework    │  │ (进程内)      │  │ Manager      │      │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │
│  │                                                             │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│  │  │ Stream       │  │ Inference    │  │ Tracking     │      │   │
│  │  │ Manager      │  │ Scheduler    │  │ Engine       │      │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │
│  │                                                             │   │
│  │  ┌──────────────┐  ┌──────────────┐                        │   │
│  │  │ Detection    │  │ Adaptive     │                        │   │
│  │  │ Model        │  │ Monitor      │                        │   │
│  │  └──────────────┘  └──────────────┘                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                      插件接口 (Plugin API)                           │
│                              │                                      │
│  ┌───────────────────────────┼──────────────────────────────────┐  │
│  │                    插件层 (Plugins)                            │  │
│  │                                                               │  │
│  │  Capture    Detector    Tracker     Attribute    Storage      │  │
│  │  ┌──────┐   ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    │  │
│  │  │Direct │   │YOLOv8│    │UKF   │    │Head  │    │H.264 │    │  │
│  │  │Show   │   │YOLO26│    │MCUKF │    │Attr  │    │MP4   │    │  │
│  │  │V4L2   │   │UHD   │    │Byte  │    │Gesture│   │Frames│    │  │
│  │  │RTSP   │   │ONNX  │    │Track │    │Rect  │    │DB    │    │  │
│  │  │GigE   │   │TRT   │    │Deep  │    │      │    │      │    │  │
│  │  │File   │   │SORT  │    │OSTrack│   │      │    │      │    │  │
│  │  └──────┘   └──────┘    └──────┘    └──────┘    └──────┘    │  │
│  │                                                               │  │
│  │  Preprocessor  Postprocessor  Accelerator  UI                 │  │
│  │  ┌──────┐      ┌──────┐       ┌──────┐     ┌──────┐         │  │
│  │  │Bilat │      │NMS   │       │CUDA  │     │PyQt6 │         │  │
│  │  │NL-Mns│      │Dedup │       │Direct│     │Web   │         │  │
│  │  │CLAHE │      │Smooth│       │ML    │     │CLI   │         │  │
│  │  │Super │      │Label │       │OpenCL│     │API   │         │  │
│  │  │Res   │      │      │       │NPU   │     │      │         │  │
│  │  └──────┘      └──────┘       └──────┘     └──────┘         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    平台抽象层 (Platform Abstraction)           │   │
│  │  Path Utils | Thread Model | GPU Factory | Logger | Timer   │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 推荐理由

1. **微内核保证核心永不膨胀**：核心层只有 7 个模块，每个职责单一。新功能只能以插件形式加入，从架构上杜绝 God Class 再现。

2. **插件架构支持生态**：第三方开发者可以编写 Capture/Detector/Tracker 等插件，通过 Python entry points 注册，无需修改核心代码。

3. **进程内事件总线兼顾解耦与性能**：不使用 RPC/消息队列（帧序列化成本不可接受），而是用进程内事件总线（pub/sub）实现插件间通信。延迟 <0.1ms，远低于帧间隔（33ms@30fps）。

4. **核心层不可替换部分极少**：只有插件框架、事件总线、配置管理是"真正的内核"。Stream Manager / Inference Scheduler / Tracking Engine 虽然在核心层，但它们本身也是"内置插件"——可以被替换或跳过。

5. **渐进式迁移可行**：VDP 1.0 的代码可以逐步迁移到 2.0 架构。先建立微内核 + 插件框架，然后将现有功能逐个包装为插件。

---

## 2. 核心层详细拆分

### 2.1 模块依赖图

```
                    ┌────────────────┐
                    │  Plugin        │
                    │  Framework     │ ◄──── 所有模块依赖（插件加载/注册）
                    └───────┬────────┘
                            │
                    ┌───────┴────────┐
                    │  Event Bus     │ ◄──── 所有模块依赖（事件发布/订阅）
                    └───────┬────────┘
                            │
                    ┌───────┴────────┐
                    │  Config        │ ◄──── 所有模块依赖（配置读取）
                    │  Manager       │
                    └───────┬────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
  ┌───────┴──────┐  ┌──────┴───────┐  ┌──────┴───────┐
  │  Stream      │  │  Detection   │  │  Adaptive    │
  │  Manager     │  │  Model       │  │  Monitor     │
  └───────┬──────┘  └──────┬───────┘  └──────┬───────┘
          │                │                 │
          │         ┌──────┴───────┐         │
          │         │  Inference   │         │
          └────────►│  Scheduler   │◄────────┘
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │  Tracking    │
                    │  Engine      │
                    └──────────────┘
```

**依赖方向规则**：
- 核心模块只能依赖更底层的核心模块（不可反向依赖）
- 核心模块不能依赖任何插件
- 插件可以依赖核心模块和事件总线
- 插件之间不能直接依赖（通过事件总线通信）

### 2.2 各模块详细设计

---

#### 2.2.1 Plugin Framework（插件框架）

**职责**：
- 插件发现（扫描 entry points / 目录）
- 插件加载（动态 import + 实例化）
- 插件注册（类型注册 + 实例注册）
- 插件生命周期管理（init → start → stop → destroy）
- 插件依赖解析（A 依赖 B，先加载 B）
- 插件版本兼容性检查
- 插件健康检查（心跳 + 状态报告）

**不负责**：
- 不负责具体插件的功能实现
- 不负责插件间通信（由 Event Bus 负责）
- 不负责插件配置存储（由 Config Manager 负责）

**输入**：
- 插件搜索路径（配置项）
- 插件清单文件（`plugin.toml` / entry points）
- 插件配置（从 Config Manager 注入）

**输出**：
- 已注册的插件实例（供其他模块查询）
- 插件状态事件（loaded / started / stopped / error）

**生命周期**：

```
discover() → load() → register() → init(config) → start()
                                                          │
                                                   [运行中]
                                                          │
                                              stop() → destroy()
```

**接口定义**：

```python
from typing import Protocol, runtime_checkable
from enum import Enum

class PluginState(Enum):
    DISCOVERED = "discovered"
    LOADED = "loaded"
    INITIALIZED = "initialized"
    STARTED = "started"
    STOPPED = "stopped"
    ERROR = "error"

@runtime_checkable
class Plugin(Protocol):
    """所有插件必须实现的基接口。"""
    
    @property
    def plugin_id(self) -> str:
        """唯一标识符，如 'sentineltrack.capture.directshow'。"""
        ...
    
    @property
    def plugin_version(self) -> str:
        """语义化版本号，如 '1.0.0'。"""
        ...
    
    @property
    def plugin_type(self) -> str:
        """插件类型，如 'capture' / 'detector' / 'tracker'。"""
        ...
    
    @property
    def api_version(self) -> str:
        """兼容的 SentinelTrack API 版本，如 '2.0'。"""
        ...
    
    def init(self, config: dict) -> None:
        """初始化插件（加载模型、打开设备等）。"""
        ...
    
    def start(self) -> None:
        """启动插件运行。"""
        ...
    
    def stop(self) -> None:
        """停止插件运行（可重启）。"""
        ...
    
    def destroy(self) -> None:
        """销毁插件（释放所有资源）。"""
        ...
    
    def health_check(self) -> dict:
        """返回健康状态。"""
        ...
```

**与其他模块关系**：
- 依赖：Event Bus（发布插件状态事件）、Config Manager（读取插件配置）
- 被依赖：所有核心模块和插件

---

#### 2.2.2 Event Bus（事件总线）

**职责**：
- 进程内 pub/sub 消息传递
- 事件类型注册与过滤
- 同步/异步事件分发
- 事件历史（可选，用于调试）

**不负责**：
- 不负责跨进程通信（非消息队列）
- 不负责事件持久化（非事件溯源）
- 不负责事件路由逻辑（由发布者指定 topic）

**输入**：事件消息（topic + payload + metadata）

**输出**：事件分发给订阅者

**事件类型定义**：

```python
from dataclasses import dataclass, field
from time import time
from typing import Any

@dataclass
class Event:
    topic: str              # 事件主题，如 "frame.captured"
    payload: Any            # 事件数据
    source: str             # 发布者 ID
    timestamp: float = field(default_factory=time)
    metadata: dict = field(default_factory=dict)

# 核心事件主题
EVENT_TOPICS = {
    # 流事件
    "stream.registered":    "流注册成功",
    "stream.unregistered":  "流注销",
    "stream.error":         "流错误",
    
    # 帧事件
    "frame.captured":       "帧采集完成",
    "frame.preprocessed":   "帧预处理完成",
    "frame.dropped":        "帧丢弃",
    
    # 检测事件
    "detection.ready":      "检测结果就绪",
    "detection.failed":     "检测失败",
    
    # 跟踪事件
    "track.updated":        "跟踪状态更新",
    "track.created":        "新轨迹创建",
    "track.deleted":        "轨迹删除",
    "track.lost":           "轨迹丢失",
    "track.redetected":     "轨迹重检测恢复",
    
    # 属性事件
    "attribute.attached":   "属性附加完成",
    
    # 存储事件
    "storage.written":      "数据写入完成",
    "storage.failed":       "写入失败",
    
    # 系统事件
    "plugin.loaded":        "插件加载",
    "plugin.started":       "插件启动",
    "plugin.stopped":       "插件停止",
    "plugin.error":         "插件错误",
    "config.changed":       "配置变更",
    "system.health":        "系统健康状态",
    "system.degraded":      "系统降级",
    "system.recovered":     "系统恢复",
}
```

**接口定义**：

```python
from typing import Callable, Protocol

class EventBus(Protocol):
    def publish(self, topic: str, payload: Any, source: str = "") -> None:
        """同步发布事件（当前线程内同步调用所有同步订阅者）。"""
        ...
    
    def publish_async(self, topic: str, payload: Any, source: str = "") -> None:
        """异步发布事件（放入队列，由事件线程分发）。"""
        ...
    
    def subscribe(self, topic: str, handler: Callable[[Event], None]) -> str:
        """订阅事件，返回订阅 ID。"""
        ...
    
    def unsubscribe(self, subscription_id: str) -> None:
        """取消订阅。"""
        ...
    
    def get_history(self, topic: str, limit: int = 100) -> list[Event]:
        """获取事件历史（调试用）。"""
        ...
```

**同步 vs 异步策略**：

| 事件类型 | 分发方式 | 理由 |
|----------|----------|------|
| frame.captured | 异步 | 采集线程不应被下游阻塞 |
| detection.ready | 异步 | 推理结果由消费者异步处理 |
| track.updated | 同步 | 跟踪更新需在当前帧周期内完成 |
| config.changed | 同步 | 配置变更需立即生效 |
| plugin.error | 异步 | 错误处理不应阻塞主流程 |

---

#### 2.2.3 Config Manager（配置管理器）

**职责**：
- 配置 schema 定义与校验
- 配置持久化（JSON / YAML / QSettings）
- 配置热更新（运行时修改无需重启）
- 配置导入/导出
- 场景预设管理

**不负责**：
- 不负责具体配置的业务含义（由使用方解释）
- 不负责配置的 UI 展示（由 UI 插件处理）

**输入**：配置键值对 + schema

**输出**：校验后的配置值 + 变更事件

**接口定义**：

```python
from typing import Any, Protocol
from pydantic import BaseModel

class ConfigSchema(BaseModel):
    """配置 schema 基类，插件通过继承此类定义自己的配置。"""
    pass

class ConfigManager(Protocol):
    def get(self, key: str, default: Any = None) -> Any:
        """读取配置值。"""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """设置配置值并触发变更事件。"""
        ...
    
    def get_plugin_config(self, plugin_id: str) -> dict:
        """获取指定插件的配置。"""
        ...
    
    def set_plugin_config(self, plugin_id: str, config: dict) -> None:
        """设置插件配置。"""
        ...
    
    def register_schema(self, plugin_id: str, schema: type[ConfigSchema]) -> None:
        """注册配置 schema（插件初始化时调用）。"""
        ...
    
    def export_config(self) -> dict:
        """导出全部配置。"""
        ...
    
    def import_config(self, config: dict) -> None:
        """导入配置。"""
        ...
    
    def save_preset(self, name: str) -> None:
        """保存当前配置为预设。"""
        ...
    
    def load_preset(self, name: str) -> None:
        """加载预设。"""
        ...
```

**配置层级**：

```
默认值 (代码内)
  ↓ 覆盖
全局配置 (config.json)
  ↓ 覆盖
插件配置 (plugins/<id>/config.json)
  ↓ 覆盖
运行时覆盖 (API / GUI 临时修改)
```

---

#### 2.2.4 Stream Manager（流管理器）

**职责**：
- 视频流生命周期管理（注册/注销/查询）
- 帧缓冲管理（双缓冲 + 丢弃策略）
- 帧分发（一帧多消费者）
- 流状态监控（fps / 延迟 / 丢帧率）
- 流同步（多路帧时间对齐，可选）

**不负责**：
- 不负责帧采集（由 CapturePlugin 负责）
- 不负责帧处理（由下游消费者负责）
- 不负责帧存储（由 StoragePlugin 负责）

**输入**：
- CapturePlugin 产生的帧（`Frame` 对象）
- 流配置（分辨率/帧率/缓冲策略）

**输出**：
- 帧快照（供消费者读取）
- 流状态事件

**数据模型**：

```python
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class Frame:
    """统一帧数据模型。"""
    stream_id: str              # 流 ID
    frame_id: int               # 帧序号（单调递增）
    timestamp: float            # 采集时间戳（秒）
    data: np.ndarray            # RGB ndarray (H, W, 3) uint8
    width: int                  # 原始宽度
    height: int                 # 原始高度
    metadata: dict = field(default_factory=dict)  # 附加元数据

@dataclass
class StreamConfig:
    """流配置。"""
    stream_id: str
    source: str                 # 采集插件 ID
    resolution: tuple[int, int] # 期望分辨率
    fps: int                    # 期望帧率
    buffer_strategy: str = "latest"  # latest / queue / drop
    buffer_size: int = 2        # 缓冲帧数
    sync_group: str | None = None    # 同步组（同组帧时间对齐）

@dataclass
class StreamStatus:
    """流状态。"""
    stream_id: str
    is_alive: bool
    current_fps: float
    avg_latency_ms: float
    dropped_frames: int
    total_frames: int
```

**接口定义**：

```python
class StreamManager(Protocol):
    def register_stream(self, config: StreamConfig) -> str:
        """注册新流，返回 stream_id。"""
        ...
    
    def unregister_stream(self, stream_id: str) -> None:
        """注销流。"""
        ...
    
    def get_frame(self, stream_id: str) -> Frame | None:
        """获取最新帧快照（非阻塞）。"""
        ...
    
    def get_streams(self) -> list[StreamStatus]:
        """获取所有流状态。"""
        ...
    
    def on_frame(self, stream_id: str, frame: Frame) -> None:
        """CapturePlugin 调用：提交新帧。"""
        ...
```

**缓冲策略**：

| 策略 | 行为 | 适用场景 |
|------|------|----------|
| `latest` | 只保留最新帧，旧帧丢弃 | 实时显示/推理 |
| `queue` | FIFO 队列，满则阻塞 | 录像（不丢帧） |
| `drop` | 满则丢弃新帧 | 低优先级消费者 |

---

#### 2.2.5 Inference Scheduler（推理调度器）

**职责**：
- 多路共享推理调度（1 个推理线程服务 N 路流）
- 跳帧控制（infer_stride）
- 推理后端路由（多 DetectorPlugin 并行或选择）
- 批处理调度（多路帧合并推理，可选）
- 推理性能监控

**不负责**：
- 不负责模型加载（由 DetectorPlugin 负责）
- 不负责后处理（由 PostprocessorPlugin 负责）
- 不负责跟踪（由 Tracking Engine 负责）

**输入**：
- Stream Manager 的帧快照
- 推理配置（stride / backend / conf）

**输出**：
- 检测结果（`list[Detection]`）+ 检测事件

**接口定义**：

```python
class InferenceScheduler(Protocol):
    def submit(self, stream_id: str, frame: Frame) -> None:
        """提交帧到推理队列。"""
        ...
    
    def register_detector(self, plugin_id: str, priority: int = 0) -> None:
        """注册检测器插件。"""
        ...
    
    def unregister_detector(self, plugin_id: str) -> None:
        """注销检测器插件。"""
        ...
    
    def set_stride(self, stream_id: str, stride: int) -> None:
        """设置跳帧间隔。"""
        ...
    
    def get_stats(self) -> dict:
        """获取推理统计（fps / latency / backlog）。"""
        ...
```

**调度算法**：

```
推理主循环 (InferenceScheduler.run):
  while running:
    for stream_id in registered_streams:
      frame = stream_manager.get_frame(stream_id)
      if frame is None:
        continue
      
      frame_id = frame.frame_id
      stride = config[stream_id].stride
      
      if frame_id % stride == 0:
        # 推理帧：执行完整推理
        detections = detector.predict(frame, conf)
        event_bus.publish("detection.ready", {
          "stream_id": stream_id,
          "frame_id": frame_id,
          "detections": detections
        })
      else:
        # 跳帧：不推理，由 Tracking Engine 用 predict_step 填充
        pass  # Tracking Engine 自行处理
    
    sleep(target_interval)  # 控制推理频率
```

---

#### 2.2.6 Tracking Engine（跟踪引擎）

**职责**：
- 多目标状态估计（预测 + 校正）
- 轨迹生命周期管理（创建/更新/删除）
- 检测-跟踪关联（匈牙利匹配 / IoU 匹配）
- ROI 重检测协调
- 多跟踪器共存（每流独立跟踪器）

**不负责**：
- 不负责检测（由 DetectorPlugin 负责）
- 不负责属性分析（由 AttributePlugin 负责）
- 不负责绘制（由 UIPlugin 负责）

**输入**：
- 检测结果（`list[Detection]`）
- 跳帧标记（是否推理帧）

**输出**：
- 跟踪结果（`list[Track]`）+ 跟踪事件

**数据模型**：

```python
@dataclass
class Track:
    """跟踪轨迹。"""
    track_id: int                   # 全局唯一 ID
    stream_id: str                  # 所属流
    state: TrackState               # normal / predicted / redetected / lost
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) 归一化
    velocity: tuple[float, float]   # (vx, vy) 归一化/帧
    confidence: float               # 最近观测置信度
    age: int                        # 总存活帧数
    hits: int                       # 成功匹配次数
    misses: int                     # 连续未匹配次数
    polygon: list[tuple[float, float]] | None  # OBB 四边形
    keypoints: list | None          # 17 COCO 关键点
    attributes: dict                # 附加属性（手势/头部等）
    filter_state: dict              # 滤波器内部状态（UKF 的 x, P）

class TrackState(Enum):
    NORMAL = "normal"           # 正常跟踪
    PREDICTED = "predicted"     # 跳帧预测
    REDETECTED = "redetected"   # 重检测恢复
    LOST = "lost"               # 丢失（待删除）
```

**接口定义**：

```python
class TrackingEngine(Protocol):
    def update(self, stream_id: str, detections: list[Detection]) -> list[Track]:
        """推理帧：用检测结果更新跟踪。"""
        ...
    
    def predict_step(self, stream_id: str) -> list[Track]:
        """跳帧：用卡尔曼预测填充。"""
        ...
    
    def register_tracker(self, stream_id: str, plugin_id: str) -> None:
        """为指定流注册跟踪器插件。"""
        ...
    
    def get_tracks(self, stream_id: str) -> list[Track]:
        """获取当前所有活跃轨迹。"""
        ...
    
    def reset(self, stream_id: str) -> None:
        """重置指定流的跟踪器。"""
        ...
```

---

#### 2.2.7 Detection Model（检测数据模型）

**职责**：
- 统一检测数据容器
- 几何工具函数（IoU / polygon / 坐标变换）
- 序列化/反序列化

**不负责**：
- 不负责检测逻辑
- 不负责跟踪逻辑

**数据模型**（已在 VDP 1.0 中定义，2.0 保持兼容）：

```python
@dataclass
class Detection:
    class_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) 归一化 [0,1]
    polygon: list[tuple[float, float]] | None  # OBB 四边形
    keypoints: list[Keypoint] | None  # 17 COCO 关键点
    track_id: int | None
    track_state: TrackState | None
    attributes: dict  # 附加属性（手势/头部属性等）
    
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, d: dict) -> "Detection": ...
    def has_polygon(self) -> bool: ...
    def polygon_iou_with(self, other: "Detection") -> float: ...
```

---

#### 2.2.8 Adaptive Monitor（自适应监控器）

**职责**：
- 运行时健康监控（置信度/NIS/条件数/丢帧率）
- 自动参数回退（性能下降时切换保守配置）
- 跟踪器热切换协调（UKF → MCUKF → Manifold）
- 系统降级/恢复事件发布

**不负责**：
- 不负责具体跟踪算法
- 不负责具体推理调度

**接口定义**：

```python
class AdaptiveMonitor(Protocol):
    def update(self, stream_id: str, metrics: dict) -> None:
        """更新健康指标。"""
        ...
    
    def is_healthy(self, stream_id: str) -> bool:
        """判断流是否健康。"""
        ...
    
    def get_recommendation(self, stream_id: str) -> dict:
        """获取推荐配置调整。"""
        ...
    
    def get_health_report(self) -> dict:
        """获取全局健康报告。"""
        ...
```

---

### 2.3 核心层 UML 结构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SentinelTrack Core                              │
│                                                                         │
│  ┌──────────────────┐     ┌──────────────────┐     ┌────────────────┐  │
│  │  «interface»     │     │  «interface»     │     │  «interface»   │  │
│  │  Plugin          │     │  EventBus        │     │  ConfigManager │  │
│  │  Framework       │     │                  │     │                │  │
│  ├──────────────────┤     ├──────────────────┤     ├────────────────┤  │
│  │ +discover()      │     │ +publish()       │     │ +get()         │  │
│  │ +load()          │     │ +publish_async() │     │ +set()         │  │
│  │ +register()      │     │ +subscribe()     │     │ +register_     │  │
│  │ +get_plugin()    │     │ +unsubscribe()   │     │   schema()     │  │
│  │ +list_plugins()  │     │ +get_history()   │     │ +export()      │  │
│  └────────┬─────────┘     └────────┬─────────┘     └───────┬────────┘  │
│           │                        │                       │           │
│           │          ┌─────────────┼───────────────┐       │           │
│           │          │             │               │       │           │
│           ▼          ▼             ▼               ▼       ▼           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                      «interface» StreamManager                    │  │
│  │  +register_stream(config) → stream_id                             │  │
│  │  +unregister_stream(stream_id)                                    │  │
│  │  +get_frame(stream_id) → Frame | None                             │  │
│  │  +on_frame(stream_id, frame)                                      │  │
│  │  +get_streams() → list[StreamStatus]                              │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                                 │                                      │
│                                 │ provides frames                      │
│                                 ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    «interface» InferenceScheduler                 │  │
│  │  +submit(stream_id, frame)                                        │  │
│  │  +register_detector(plugin_id, priority)                          │  │
│  │  +set_stride(stream_id, stride)                                   │  │
│  │  +get_stats() → dict                                              │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                                 │                                      │
│                                 │ provides detections                  │
│                                 ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     «interface» TrackingEngine                    │  │
│  │  +update(stream_id, detections) → list[Track]                     │  │
│  │  +predict_step(stream_id) → list[Track]                           │  │
│  │  +register_tracker(stream_id, plugin_id)                          │  │
│  │  +get_tracks(stream_id) → list[Track]                             │  │
│  │  +reset(stream_id)                                                │  │
│  └──────────────────────────────┬───────────────────────────────────┘  │
│                                 │                                      │
│                                 │ reports health                       │
│                                 ▼                                      │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                     «interface» AdaptiveMonitor                   │  │
│  │  +update(stream_id, metrics)                                      │  │
│  │  +is_healthy(stream_id) → bool                                    │  │
│  │  +get_recommendation(stream_id) → dict                            │  │
│  │  +get_health_report() → dict                                      │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────┐                                                    │
│  │  «dataclass»     │     ┌──────────────────┐  ┌──────────────────┐   │
│  │  Detection       │     │  «dataclass»     │  │  «dataclass»     │   │
│  │                  │     │  Frame           │  │  Track           │   │
│  ├──────────────────┤     ├──────────────────┤  ├──────────────────┤   │
│  │ class_id: int    │     │ stream_id: str   │  │ track_id: int    │   │
│  │ label: str       │     │ frame_id: int    │  │ stream_id: str   │   │
│  │ confidence: float│     │ timestamp: float │  │ state: TrackState│   │
│  │ bbox: tuple      │     │ data: ndarray    │  │ bbox: tuple      │   │
│  │ polygon: list?   │     │ width: int       │  │ velocity: tuple  │   │
│  │ keypoints: list? │     │ height: int      │  │ confidence: float│   │
│  │ track_id: int?   │     └──────────────────┘  │ age: int         │   │
│  │ attributes: dict │                           │ hits: int        │   │
│  └──────────────────┘                           │ misses: int      │   │
│                                                 │ polygon: list?   │   │
│                                                 │ keypoints: list? │   │
│                                                 │ attributes: dict │   │
│                                                 └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 插件系统设计

### 3.1 插件类型总览

| 插件类型 | 接口名 | 职责 | 数据流位置 |
|----------|--------|------|-----------|
| CapturePlugin | 采集 | 从设备/文件获取视频帧 | 数据流入口 |
| PreprocessorPlugin | 预处理 | 图像增强/去噪/超分 | 采集后、检测前 |
| DetectorPlugin | 检测 | AI 模型推理 | 预处理后、跟踪前 |
| TrackerPlugin | 跟踪 | 目标状态估计 | 检测后、属性前 |
| AttributePlugin | 属性 | 附加属性分析 | 跟踪后 |
| PostprocessorPlugin | 后处理 | 去重/平滑/标签 | 属性后、输出前 |
| StoragePlugin | 存储 | 录像/截图/数据持久化 | 后处理后 |
| UIPlugin | 界面 | 用户交互/显示 | 后处理后 |
| AcceleratorPlugin | 加速 | GPU/NPU 推理加速 | 检测器内部使用 |

### 3.2 各插件接口定义

#### 3.2.1 CapturePlugin

```python
class CapturePlugin(Plugin):
    """视频采集插件接口。"""
    
    plugin_type = "capture"
    
    def open(self, config: CaptureConfig) -> FrameStream:
        """
        打开采集源。
        
        Args:
            config: 采集配置（设备名/URL/文件路径/分辨率/帧率）
        
        Returns:
            FrameStream: 帧流对象，可迭代获取帧
        """
        ...
    
    def close(self, stream: FrameStream) -> None:
        """关闭采集源。"""
        ...
    
    def list_devices(self) -> list[DeviceInfo]:
        """列出可用设备（仅实时采集插件需要实现）。"""
        ...
    
    def is_realtime(self) -> bool:
        """是否为实时采集（USB/RTSP=True，File=False）。"""
        ...

@dataclass
class CaptureConfig:
    source: str               # 设备名/URL/文件路径
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    format: str = "rgb"       # rgb / bgr / gray
    buffer_size: int = 2

@dataclass
class DeviceInfo:
    device_id: str            # 设备唯一标识
    name: str                 # 显示名称
    type: str                 # usb / rtsp / file / gigevision
    capabilities: dict        # 支持的分辨率/帧率

class FrameStream:
    """帧流对象。"""
    
    def read(self) -> Frame | None:
        """读取下一帧（阻塞或非阻塞，取决于配置）。"""
        ...
    
    def __iter__(self) -> Iterator[Frame]:
        """迭代获取帧。"""
        ...
```

#### 3.2.2 PreprocessorPlugin

```python
class PreprocessorPlugin(Plugin):
    """图像预处理插件接口。"""
    
    plugin_type = "preprocessor"
    
    def process(self, frame: Frame) -> Frame:
        """
        处理帧，返回处理后的帧。
        
        注意:
        - 必须保持帧的 stream_id 和 frame_id
        - 可以修改 frame.data（图像数据）
        - 可以修改 frame.metadata（添加处理信息）
        - 不应修改原始帧（应返回新帧或副本）
        """
        ...
    
    def supports(self, frame: Frame) -> bool:
        """判断是否支持处理该帧（按分辨率/格式过滤）。"""
        ...
```

#### 3.2.3 DetectorPlugin

```python
class DetectorPlugin(Plugin):
    """目标检测插件接口。"""
    
    plugin_type = "detector"
    
    def load_model(self, model_path: str, **kwargs) -> None:
        """加载模型文件。"""
        ...
    
    def predict(self, frame: Frame, conf: float = 0.5) -> list[Detection]:
        """
        对帧执行检测推理。
        
        Args:
            frame: 输入帧
            conf: 置信度阈值
        
        Returns:
            list[Detection]: 检测结果列表
        """
        ...
    
    def supported_tasks(self) -> list[ModelTask]:
        """支持的任务类型（DETECT / POSE / OBB / SEGMENT）。"""
        ...
    
    def supported_labels(self) -> list[str]:
        """支持的标签列表（如 COCO 80 类）。"""
        ...
    
    def get_input_size(self) -> tuple[int, int]:
        """模型输入尺寸。"""
        ...
    
    def benchmark(self) -> dict:
        """性能基准（单帧推理时间）。"""
        ...
```

#### 3.2.4 TrackerPlugin

```python
class TrackerPlugin(Plugin):
    """目标跟踪插件接口。"""
    
    plugin_type = "tracker"
    
    def init_tracker(self, stream_id: str, config: TrackerConfig) -> None:
        """为指定流初始化跟踪器实例。"""
        ...
    
    def update(self, stream_id: str, detections: list[Detection]) -> list[Track]:
        """推理帧：用检测结果更新跟踪。"""
        ...
    
    def predict_step(self, stream_id: str) -> list[Track]:
        """跳帧：预测填充。"""
        ...
    
    def reset(self, stream_id: str) -> None:
        """重置指定流的跟踪器。"""
        ...
    
    def get_tracks(self, stream_id: str) -> list[Track]:
        """获取当前轨迹。"""
        ...
    
    def supports_polygon(self) -> bool:
        """是否支持 OBB polygon 跟踪。"""
        ...
    
    def supports_keypoints(self) -> bool:
        """是否支持关键点跟踪。"""
        ...

@dataclass
class TrackerConfig:
    filter_type: str = "ukf"         # ukf / mcukf / manifold_ukf / auto
    iou_threshold: float = 0.3       # 匹配阈值
    max_misses: int = 30             # 最大丢失帧数
    smooth_alpha: float = 0.85       # 平滑系数
    redetect_enabled: bool = True    # ROI 重检测
    redetect_max_retries: int = 2    # 重检测次数
    roi_pad: float = 0.15            # ROI 扩展系数
```

#### 3.2.5 AttributePlugin

```python
class AttributePlugin(Plugin):
    """属性分析插件接口。"""
    
    plugin_type = "attribute"
    
    def attach(
        self,
        detections: list[Detection],
        frame: Frame,
        stream_id: str
    ) -> list[Detection]:
        """
        为检测结果附加属性。
        
        Args:
            detections: 当前帧的检测结果（已跟踪）
            frame: 原始帧
            stream_id: 流 ID
        
        Returns:
            更新后的检测结果列表（detection.attributes 被填充）
        """
        ...
    
    def supported_attributes(self) -> list[str]:
        """支持的属性列表（如 ['hat', 'mask', 'sunglass']）。"""
        ...
    
    def get_stride(self) -> int:
        """属性分析的跳帧间隔（1=每帧，2=隔帧）。"""
        ...
```

#### 3.2.6 PostprocessorPlugin

```python
class PostprocessorPlugin(Plugin):
    """后处理插件接口。"""
    
    plugin_type = "postprocessor"
    
    def process(
        self,
        detections: list[Detection],
        stream_id: str
    ) -> list[Detection]:
        """
        后处理检测结果（去重/平滑/过滤/标签）。
        """
        ...
```

#### 3.2.7 StoragePlugin

```python
class StoragePlugin(Plugin):
    """存储插件接口。"""
    
    plugin_type = "storage"
    
    def open(self, config: StorageConfig) -> str:
        """打开存储会话，返回 session_id。"""
        ...
    
    def write_frame(self, session_id: str, frame: Frame, metadata: dict) -> None:
        """写入一帧。"""
        ...
    
    def close(self, session_id: str) -> None:
        """关闭存储会话。"""
        ...
    
    def list_sessions(self) -> list[StorageSession]:
        """列出存储会话。"""
        ...
    
    def read_session(self, session_id: str) -> Iterator[Frame]:
        """读取存储会话的帧（回放）。"""
        ...

@dataclass
class StorageConfig:
    output_dir: str
    format: str = "mp4"        # mp4 / h265 / frames / db
    codec: str = "h264"
    quality: str = "crf=23"
    metadata_enabled: bool = True
```

#### 3.2.8 UIPlugin

```python
class UIPlugin(Plugin):
    """用户界面插件接口。"""
    
    plugin_type = "ui"
    
    def render(
        self,
        streams: dict[str, Frame],
        tracks: dict[str, list[Track]]
    ) -> None:
        """
        渲染画面和跟踪结果。
        
        Args:
            streams: stream_id → 最新帧
            tracks: stream_id → 轨迹列表
        """
        ...
    
    def run_event_loop(self) -> None:
        """运行 UI 事件循环（阻塞）。"""
        ...
    
    def on_config_changed(self, key: str, value: Any) -> None:
        """配置变更通知。"""
        ...
    
    def show_error(self, message: str) -> None:
        """显示错误消息。"""
        ...
```

#### 3.2.9 AcceleratorPlugin

```python
class AcceleratorPlugin(Plugin):
    """硬件加速插件接口。"""
    
    plugin_type = "accelerator"
    
    def available(self) -> bool:
        """检查加速器是否可用。"""
        ...
    
    def get_provider(self) -> str:
        """返回 provider 名称（cuda / directml / opencl / npu / cpu）。"""
        ...
    
    def create_session(self, model_path: str) -> Any:
        """创建加速会话（加载模型到设备）。"""
        ...
    
    def infer(self, session: Any, input_data: np.ndarray) -> Any:
        """执行推理。"""
        ...
    
    def destroy_session(self, session: Any) -> None:
        """销毁加速会话。"""
        ...
    
    def benchmark(self, session: Any) -> dict:
        """性能基准。"""
        ...
```

### 3.3 插件生命周期

```
                    ┌──────────────┐
                    │  Discovered  │ ← Plugin Framework 扫描到插件
                    └──────┬───────┘
                           │ load()
                           ▼
                    ┌──────────────┐
                    │   Loaded     │ ← 模块导入成功
                    └──────┬───────┘
                           │ register()
                           ▼
                    ┌──────────────┐
                    │ Registered   │ ← 注册到插件管理器
                    └──────┬───────┘
                           │ init(config)
                           ▼
                    ┌──────────────┐
                    │ Initialized  │ ← 配置注入，资源准备
                    └──────┬───────┘
                           │ start()
                    ┌──────┴───────┐
                    │              │
                    ▼              ▼
            ┌──────────────┐  ┌──────────────┐
            │   Started    │  │   Error      │
            │  (运行中)     │  │  (错误状态)   │
            └──────┬───────┘  └──────┬───────┘
                   │ stop()          │ retry
                   ▼                 │
            ┌──────────────┐        │
            │  Stopped     │─────────┘
            └──────┬───────┘
                   │ destroy()
                   ▼
            ┌──────────────┐
            │  Destroyed   │ ← 资源释放
            └──────────────┘
```

### 3.4 插件注册机制

#### 3.4.1 Python Entry Points（主要机制）

```toml
# pyproject.toml
[project.entry-points."sentineltrack.capture"]
directshow = "sentineltrack_plugins.capture.directshow:DirectShowPlugin"
v4l2 = "sentineltrack_plugins.capture.v4l2:V4L2Plugin"
rtsp = "sentineltrack_plugins.capture.rtsp:RTSPPlugin"

[project.entry-points."sentineltrack.detector"]
yolov8_onnx = "sentineltrack_plugins.detector.yolov8_onnx:YOLOv8ONNXPlugin"
yolo26 = "sentineltrack_plugins.detector.yolo26:YOLO26Plugin"

[project.entry-points."sentineltrack.tracker"]
ukf = "sentineltrack_plugins.tracker.ukf:UKFTrackerPlugin"
bytetrack = "sentineltrack_plugins.tracker.bytetrack:ByteTrackPlugin"
```

#### 3.4.2 目录扫描（辅助机制）

```
~/.sentineltrack/plugins/
├── capture/
│   ├── my_custom_camera/
│   │   ├── plugin.toml      ← 插件清单
│   │   ├── camera.py
│   │   └── config_schema.py
│   └── ...
├── detector/
├── tracker/
└── ...
```

```toml
# plugin.toml
[plugin]
id = "my-custom-camera"
name = "My Custom Camera"
version = "1.0.0"
type = "capture"
api_version = "2.0"
author = "..."
description = "..."

[dependencies]
python = ">=3.10"
packages = ["opencv-python>=4.5.0"]

[config]
schema = "config_schema:CameraConfig"
```

### 3.5 配置注入方式

```python
# 插件通过 init() 接收配置
class MyDetectorPlugin(DetectorPlugin):
    def init(self, config: dict) -> None:
        # config 已通过 ConfigManager 校验
        self.model_path = config["model_path"]
        self.conf_threshold = config["conf_threshold"]
        self.device = config.get("device", "auto")
        
        # 加载模型
        self.model = self._load_model(self.model_path)
```

```python
# ConfigManager 自动注入插件配置
# config.json 结构:
{
    "plugins": {
        "sentineltrack.detector.yolov8_onnx": {
            "model_path": "models/yolov8n.onnx",
            "conf_threshold": 0.5,
            "device": "auto"
        },
        "sentineltrack.tracker.ukf": {
            "filter_type": "ukf",
            "iou_threshold": 0.3,
            "max_misses": 30
        }
    }
}
```

### 3.6 热加载策略

```python
class PluginFramework:
    def hot_reload(self, plugin_id: str) -> None:
        """热重载插件（不停机）。"""
        # 1. 保存旧插件状态
        old_plugin = self._plugins[plugin_id]
        old_state = old_plugin.health_check()
        
        # 2. 停止旧插件
        old_plugin.stop()
        
        # 3. 重新加载模块
        importlib.reload(sys.modules[old_plugin.__module__])
        
        # 4. 创建新实例
        new_plugin = self._create_instance(plugin_id)
        new_plugin.init(self._config.get_plugin_config(plugin_id))
        new_plugin.start()
        
        # 5. 迁移状态（如果支持）
        if hasattr(new_plugin, 'migrate_state'):
            new_plugin.migrate_state(old_state)
        
        # 6. 替换注册
        self._plugins[plugin_id] = new_plugin
        
        # 7. 发布事件
        self._event_bus.publish("plugin.reloaded", {
            "plugin_id": plugin_id
        })
```

**热加载限制**：
- Capture/Detector/Tracker 插件支持热加载
- Storage 插件不支持热加载（会中断录像）
- UI 插件不支持热加载（需重启 UI）
- Accelerator 插件不支持热加载（会中断推理）

### 3.7 版本兼容策略

```python
# 插件通过 api_version 声明兼容性
class MyPlugin(Plugin):
    api_version = "2.0"  # 兼容 SentinelTrack 2.0 API

# Plugin Framework 检查兼容性
class PluginFramework:
    _API_VERSIONS = {
        "2.0": ["2.0", "2.1", "2.2"],  # 2.0 兼容 2.x
        "3.0": ["3.0", "3.1"],          # 3.0 不兼容 2.x
    }
    
    def _check_compatibility(self, plugin: Plugin) -> bool:
        plugin_api = plugin.api_version
        core_api = self.CORE_API_VERSION
        compatible_versions = self._API_VERSIONS.get(core_api, [])
        return plugin_api in compatible_versions
```

**版本规则**：
- 主版本（2.0 → 3.0）：接口不兼容，插件需更新
- 次版本（2.0 → 2.1）：接口兼容，新增可选方法
- 修订版本（2.0.1 → 2.0.2）：Bug 修复，完全兼容

### 3.8 插件市场设计

```
┌─────────────────────────────────────────────────────────┐
│                  插件市场架构                             │
│                                                         │
│  ┌─────────────┐     ┌─────────────┐  ┌─────────────┐  │
│  │  Registry   │     │  Index      │  │  CDN        │  │
│  │  (插件注册)  │────►│  (索引服务)  │─►│  (包存储)   │  │
│  └─────────────┘     └─────────────┘  └─────────────┘  │
│         ▲                    ▲                  │       │
│         │                    │                  │       │
│  ┌──────┴──────┐      ┌──────┴──────┐    ┌──────┴────┐ │
│  │  Publisher  │      │  Search     │    │  Client   │ │
│  │  (开发者    │      │  API        │    │  (ST CLI) │ │
│  │   上传)     │      │  (查询/评分) │    │  (下载    │ │
│  └─────────────┘      └─────────────┘    │   /安装)  │ │
│                                          └───────────┘ │
└─────────────────────────────────────────────────────────┘
```

**CLI 安装命令**：

```bash
# 搜索插件
sentineltrack search capture rtsp

# 安装插件
sentineltrack install sentineltrack.capture.rtsp

# 列出已安装插件
sentineltrack list

# 更新插件
sentineltrack update --all

# 卸载插件
sentineltrack uninstall sentineltrack.capture.rtsp

# 发布插件
sentineltrack publish ./my-plugin/
```

---

## 4. 数据流架构

### 4.1 完整数据流图

```
┌──────────┐
│ 摄像头/   │ 物理设备
│ 文件/RTSP │
└────┬─────┘
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  CapturePlugin                                           │
│  (采集线程, 每流1个)                                      │
│  ├ DirectShow / V4L2 / RTSP / File                       │
│  └ frame.captured 事件 ──► EventBus                      │
└──────────────────────┬───────────────────────────────────┘
                       │ Frame
                       ▼
┌──────────────────────────────────────────────────────────┐
│  StreamManager                                           │
│  (核心线程)                                               │
│  ├ 双缓冲 (latest 策略)                                   │
│  ├ 帧分发 (一帧多消费者)                                   │
│  └ frame 事件 ──► EventBus                                │
└───┬──────────┬──────────┬───────────────────────────────┘
    │          │          │
    │ Frame    │ Frame    │ Frame
    │ (显示)    │ (推理)    │ (录像)
    ▼          ▼          ▼
┌────────┐ ┌──────────────────────────────────┐ ┌────────┐
│UIPlugin│ │  PreprocessorPipeline             │ │Storage │
│(显示)  │ │  (推理线程预处理)                  │ │Plugin  │
│        │ │  ├ PreprocessorPlugin (去噪)       │ │(录像)  │
│ ~30fps │ │  └ ► 预处理后的 Frame              │ │        │
│ 可丢帧  │ └──────────────┬───────────────────┘ │ 不丢帧  │
└────────┘                │                      └────────┘
                          ▼
┌──────────────────────────────────────────────────────────┐
│  InferenceScheduler                                       │
│  (共享推理线程, 1个服务N流)                                 │
│  ├ 跳帧控制 (frame_id % stride == 0)                      │
│  ├ DetectorPlugin.predict(frame, conf)                   │
│  │  └ AcceleratorPlugin.infer() (内部调用)                │
│  └ detection.ready 事件 ──► EventBus                      │
└──────────────────────────┬───────────────────────────────┘
                           │ list[Detection]
                           ▼
┌──────────────────────────────────────────────────────────┐
│  TrackingEngine                                           │
│  (推理线程内同步执行)                                      │
│  ├ 推理帧: TrackerPlugin.update(detections)               │
│  │  ├ UKF predict + correct                               │
│  │  ├ 匈牙利匹配                                           │
│  │  ├ ROI 重检测 (回调 DetectorPlugin)                     │
│  │  └ _FilterAdaptor 自适应切换                            │
│  ├ 跳帧: TrackerPlugin.predict_step()                     │
│  └ track.updated 事件 ──► EventBus                        │
└──────────────────────────┬───────────────────────────────┘
                           │ list[Track]
                           ▼
┌──────────────────────────────────────────────────────────┐
│  AttributePipeline                                        │
│  (推理线程内, 可跳帧)                                      │
│  ├ AttributePlugin.attach(detections, frame)              │
│  │  ├ 头部属性 (CHC)                                      │
│  │  ├ 手势识别                                            │
│  │  └ 矩形检测                                            │
│  └ attribute.attached 事件 ──► EventBus                   │
└──────────────────────────┬───────────────────────────────┘
                           │ list[Detection] (with attributes)
                           ▼
┌──────────────────────────────────────────────────────────┐
│  PostprocessorPipeline                                    │
│  (推理线程内同步执行)                                      │
│  ├ PostprocessorPlugin.process(detections)                │
│  │  ├ NMS 去重                                            │
│  │  ├ 框平滑                                              │
│  │  └ 标签过滤                                            │
│  └ ► 最终检测结果                                         │
└──────────────────────────┬───────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ EventBus │ │ UIPlugin │ │ Adaptive │
        │ (事件)   │ │ (绘制)   │ │ Monitor  │
        │          │ │          │ │ (监控)   │
        └──────────┘ └──────────┘ └──────────┘
```

### 4.2 同步/异步分析

| 阶段 | 同步/异步 | 理由 |
|------|----------|------|
| 采集 → StreamManager | 异步 | 采集线程不应被下游阻塞 |
| StreamManager → UI | 异步 | 显示可丢帧，不应阻塞推理 |
| StreamManager → 录像 | 异步 | 录像有独立队列，不丢帧 |
| StreamManager → 推理 | 异步 | 推理有独立调度，可跳帧 |
| 预处理 → 检测 | 同步 | 预处理结果直接传给检测，无中间缓冲 |
| 检测 → 跟踪 | 同步 | 检测结果直接传给跟踪，保持帧一致性 |
| 跟踪 → 属性 | 同步 | 属性依赖跟踪结果中的 track_id |
| 属性 → 后处理 | 同步 | 后处理需要完整检测结果 |
| 后处理 → UI | 异步 | UI 按自身节奏消费 |
| 后处理 → 监控 | 异步 | 监控非关键路径 |

### 4.3 丢帧/保序策略

| 阶段 | 允许丢帧 | 必须保序 | 策略 |
|------|----------|----------|------|
| 采集 → StreamManager | ✅ | ✅ | latest 策略，旧帧丢弃 |
| StreamManager → UI | ✅ | ❌ | 30fps 节流，可跳帧显示 |
| StreamManager → 录像 | ❌ | ✅ | queue 策略，FIFO 不丢 |
| StreamManager → 推理 | ✅ | ✅ | stride 控制，跳帧用预测填充 |
| 预处理 → 检测 | ❌ | ✅ | 同步，不丢帧 |
| 检测 → 跟踪 | ❌ | ✅ | 同步，不丢帧 |
| 跟踪 → 属性 | ✅ | ✅ | 可跳帧（stride 控制），但必须有序 |
| 属性 → 后处理 | ❌ | ✅ | 同步，不丢帧 |
| 后处理 → UI | ✅ | ❌ | UI 可跳过旧结果 |

### 4.4 推荐线程模型

```
┌─────────────────────────────────────────────────────────────────┐
│                        线程模型                                  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GUI 线程 (主线程)                                       │   │
│  │  ├ UIPlugin.render()                                    │   │
│  │  ├ 用户交互事件处理                                      │   │
│  │  └ 配置变更通知                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Capture     │  │ Capture     │  │ Capture     │  采集线程   │
│  │ Thread #0   │  │ Thread #1   │  │ Thread #2   │  (每流1个)  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘            │
│         │                │                │                    │
│         └────────────────┼────────────────┘                    │
│                          │                                      │
│                   ┌──────▼──────┐                               │
│                   │ Stream      │  核心线程                     │
│                   │ Manager     │  (帧缓冲+分发)                │
│                   └──────┬──────┘                               │
│                          │                                      │
│                   ┌──────▼──────────────────────────────┐      │
│                   │  Inference Thread (共享1个)          │      │
│                   │  ├ Preprocessor (同步)              │      │
│                   │  ├ Detector (同步)                  │      │
│                   │  ├ Tracker (同步)                   │      │
│                   │  ├ Attribute (同步, 可跳帧)         │      │
│                   │  └ Postprocessor (同步)            │      │
│                   └──────┬──────────────────────────────┘      │
│                          │                                      │
│                   ┌──────▼──────┐                               │
│                   │ Event Bus   │  事件线程                     │
│                   │ Dispatcher  │  (异步事件分发)               │
│                   └─────────────┘                               │
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Storage     │  │ Storage     │  │ Storage     │  录像线程   │
│  │ Thread #0   │  │ Thread #1   │  │ Thread #2   │  (每流1个)  │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Adaptive Monitor Thread (健康监控, 低优先级)             │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘

线程数 = 1(GUI) + N(采集) + 1(流管理) + 1(推理) + 1(事件) + M(录像) + 1(监控)
      = 5 + N + M
      典型 4 路: 5 + 4 + 4 = 13 线程
```

---

## 5. Tracking Engine 深度设计

### 5.1 当前 UKF 族设计评估

| 维度 | 评分 | 评估 |
|------|------|------|
| 算法正确性 | 8/10 | UKF/MCUKF/Manifold 实现正确，参数已保守化 |
| 工程质量 | 6/10 | 纯 NumPy 无依赖，但缺乏测试 |
| 可扩展性 | 3/10 | 硬编码在 DetectionTracker 中，无法替换 |
| 多跟踪器共存 | 0/10 | 不支持，只有一种跟踪器 |
| 插件化 | 0/10 | 完全未插件化 |

### 5.2 是否继续作为核心

**结论：UKF 族继续作为核心内置跟踪器，但重构为 TrackerPlugin 实现。**

理由：
1. UKF 族是 SentinelTrack 的技术壁垒（5/5），必须保留
2. 但不应硬编码在核心层，应作为"内置插件"实现 TrackerPlugin 接口
3. 核心层只保留 TrackingEngine 调度器，不包含具体跟踪算法

### 5.3 重构方案

```
当前 (VDP 1.0):
┌──────────────────────────────────────────┐
│  ai/tracker.py (1086 行)                 │
│  ├ _DetectionUKF                         │
│  ├ _DetectionMCUKF                       │
│  ├ _DetectionManifoldUKF                 │
│  ├ _FilterAdaptor                        │
│  ├ _Track                                │
│  └ DetectionTracker (硬编码 UKF 族)      │
└──────────────────────────────────────────┘

目标 (SentinelTrack 2.0):
┌──────────────────────────────────────────┐
│  sentineltrack-core/tracking/            │
│  ├ engine.py     ← TrackingEngine 调度器 │
│  └ types.py      ← Track, TrackState     │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│  sentineltrack-tracker-ukf/  (内置插件)  │
│  ├ plugin.py     ← UKFTrackerPlugin      │
│  ├ ukf.py        ← _DetectionUKF        │
│  ├ mcukf.py      ← _DetectionMCUKF      │
│  ├ manifold.py   ← _DetectionManifoldUKF│
│  ├ adaptor.py    ← _FilterAdaptor       │
│  └ track.py      ← _Track               │
└──────────────────────────────────────────┘
```

### 5.4 多跟踪器共存架构

```
┌──────────────────────────────────────────────────────────┐
│  TrackingEngine (核心调度器)                              │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Stream #0 → TrackerPlugin: UKF                  │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  Stream #1 → TrackerPlugin: ByteTrack           │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  Stream #2 → TrackerPlugin: DeepSORT            │    │
│  ├─────────────────────────────────────────────────┤    │
│  │  Stream #3 → TrackerPlugin: OSTrack (视觉跟踪)   │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  每流独立跟踪器实例，互不干扰                              │
│  统一 Track 数据模型输出                                  │
│  统一事件发布                                             │
└──────────────────────────────────────────────────────────┘
```

### 5.5 跟踪器对比与集成规划

| 跟踪器 | 类型 | 优势 | 集成优先级 | 集成方式 |
|--------|------|------|-----------|----------|
| **UKF 族** | 卡尔曼滤波 | 纯 NumPy，无依赖，AUTO 自适应 | P0（内置） | 核心保留 |
| **ByteTrack** | IoU 跟踪 | 轻量高速，低置信度二次匹配 | P1 | TrackerPlugin |
| **OC-SORT** | 卡尔曼+观测中心 | 抗遮挡，简单高效 | P2 | TrackerPlugin |
| **BoT-SORT** | ReID+IoU | 跨摄像头 ReID，高精度 | P2 | TrackerPlugin |
| **DeepSORT** | 深度特征+卡尔曼 | 经典方案，生态成熟 | P3 | TrackerPlugin |
| **OSTrack** | 视觉跟踪 (Transformer) | 单目标高精度，适合云台 | P3 | TrackerPlugin（特殊：单目标） |

### 5.6 混合跟踪架构（UKF + OSTrack）

对于云台场景，需要 UKF 状态估计 + OSTrack 视觉跟踪的融合：

```
┌──────────────────────────────────────────────────────────┐
│  混合跟踪模式 (Hybrid Tracking)                          │
│                                                          │
│  ┌─────────────┐       ┌─────────────┐                  │
│  │  UKF Tracker │       │  OSTrack    │                  │
│  │  (状态估计)  │       │  (视觉跟踪)  │                  │
│  │              │       │              │                  │
│  │ ├ 位置预测   │       │ ├ 模板匹配   │                  │
│  │ ├ 速度估计   │       │ ├ 外观更新   │                  │
│  │ └ 协方差     │       │ └ 置信度     │                  │
│  └──────┬───────┘       └──────┬───────┘                  │
│         │                      │                          │
│         └──────────┬───────────┘                          │
│                    │                                      │
│           ┌────────▼────────┐                             │
│           │  Fusion Layer   │                             │
│           │  ├ UKF 状态先验 │                             │
│           │  ├ OSTrack 观测 │                             │
│           │  ├ 融合校正     │                             │
│           │  └ 输出统一 Track│                             │
│           └─────────────────┘                             │
└──────────────────────────────────────────────────────────┘
```

**融合策略**：
- UKF 提供状态先验（位置/速度预测）
- OSTrack 提供视觉观测（模板匹配位置）
- 用 UKF 的 correct 步骤融合 OSTrack 观测
- OSTrack 模板在 UKF 预测位置附近裁剪
- 两者互补：UKF 抗遮挡，OSTrack 抗漂移

---

## 6. 跨平台架构

### 6.1 平台抽象层设计

```
┌─────────────────────────────────────────────────────────────┐
│  应用层 (平台无关)                                          │
│  ├ SentinelTrack Core (纯 Python + NumPy)                  │
│  ├ Tracker Plugins (纯 NumPy)                              │
│  └ Detector Plugins (ONNX/PyTorch)                         │
├─────────────────────────────────────────────────────────────┤
│  平台抽象层 (Platform Abstraction Layer, PAL)               │
│  ├ path_utils.py    ← 路径管理 (pathlib)                   │
│  ├ thread_model.py  ← 线程抽象 (threading/asyncio)         │
│  ├ gpu_factory.py   ← GPU Provider 选择                    │
│  ├ logger.py        ← 日志 (logging + RotatingFileHandler) │
│  ├ timer.py         ← 高精度计时                            │
│  └ platform_info.py ← 平台检测                              │
├─────────────────────────────────────────────────────────────┤
│  平台相关层 (Platform-Specific, 通过插件提供)                │
│  ├ Capture: DirectShow / V4L2 / AVFoundation / RTSP       │
│  ├ Accelerator: CUDA / DirectML / OpenCL / NPU            │
│  └ UI: PyQt6 / Web / CLI                                   │
├─────────────────────────────────────────────────────────────┤
│  操作系统                                                    │
│  ├ Windows 10/11 (x86_64, ARM64)                           │
│  ├ Linux (x86_64, ARM64, aarch64)                         │
│  ├ macOS (x86_64, ARM64) [远期]                            │
│  └ 嵌入式 (Jetson, QCS6490, RPi) [远期]                    │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 代码平台属性划分

#### 必须平台无关（核心代码）

| 模块 | 理由 |
|------|------|
| TrackingEngine | 纯数学，无平台依赖 |
| UKF/MCUKF/Manifold | 纯 NumPy |
| InferenceScheduler | 纯 Python 调度逻辑 |
| StreamManager | 纯 Python 缓冲管理 |
| Detection Model | 纯 dataclass |
| AdaptiveMonitor | 纯 Python 监控逻辑 |
| ConfigManager | 纯 Python + JSON |
| EventBus | 纯 Python |
| PluginFramework | 纯 Python + importlib |
| NMS / 几何工具 | 纯 NumPy |

#### 必须平台相关（插件代码）

| 模块 | Windows | Linux | macOS | 嵌入式 |
|------|---------|-------|-------|--------|
| 采集 | DirectShow | V4L2 | AVFoundation | V4L2 |
| GPU | DirectML | CUDA/OpenCL | Metal | NPU |
| UI | PyQt6 | PyQt6/Web | PyQt6/Web | Web |
| 路径 | %APPDATA% | ~/.config | ~/Library | /opt |

#### 平台抽象接口

```python
# platform_abstraction.py

import platform
import sys
from pathlib import Path
from enum import Enum

class Platform(Enum):
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    JETSON = "jetson"
    QCS6490 = "qcs6490"
    RASPBERRY_PI = "raspberry_pi"

def detect_platform() -> Platform:
    """检测当前平台。"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "windows":
        return Platform.WINDOWS
    elif system == "linux":
        # 检测特定嵌入式平台
        if Path("/proc/device-tree/model").exists():
            model = Path("/proc/device-tree/model").read_text()
            if "NVIDIA Jetson" in model:
                return Platform.JETSON
            elif "Raspberry Pi" in model:
                return Platform.RASPBERRY_PI
        if Path("/sys/class/qcom").exists():
            return Platform.QCS6490
        return Platform.LINUX
    elif system == "darwin":
        return Platform.MACOS
    else:
        return Platform.LINUX  # 默认按 Linux 处理

def get_config_dir() -> Path:
    """获取配置目录。"""
    p = detect_platform()
    if p == Platform.WINDOWS:
        return Path(os.environ.get("APPDATA", "")) / "SentinelTrack"
    elif p in (Platform.LINUX, Platform.JETSON, Platform.QCS6490, Platform.RASPBERRY_PI):
        return Path.home() / ".config" / "sentineltrack"
    elif p == Platform.MACOS:
        return Path.home() / "Library" / "Application Support" / "SentinelTrack"
    return Path(".")

def get_plugin_dir() -> Path:
    """获取插件目录。"""
    return get_config_dir() / "plugins"

def get_default_gpu_provider() -> str:
    """获取默认 GPU provider。"""
    p = detect_platform()
    if p == Platform.WINDOWS:
        return "directml"  # 优先 DirectML, 回退 CUDA
    elif p in (Platform.LINUX, Platform.JETSON):
        return "cuda"
    elif p == Platform.QCS6490:
        return "npu"
    elif p == Platform.MACOS:
        return "metal"  # 远期
    else:
        return "cpu"
```

### 6.3 目录结构建议

```
sentineltrack/                          ← Monorepo 根目录
├── pyproject.toml                      ← 项目配置
├── README.md
├── LICENSE
│
├── packages/                           ← 核心 Python 包
│   ├── sentineltrack-core/             ← 微内核 + 核心模块
│   │   ├── pyproject.toml
│   │   ├── sentineltrack/
│   │   │   ├── __init__.py
│   │   │   ├── core/                   ← 微内核
│   │   │   │   ├── __init__.py
│   │   │   │   ├── plugin_framework.py ← 插件框架
│   │   │   │   ├── event_bus.py        ← 事件总线
│   │   │   │   ├── config_manager.py   ← 配置管理
│   │   │   │   └── platform.py         ← 平台抽象
│   │   │   ├── stream/                 ← 流管理
│   │   │   │   ├── __init__.py
│   │   │   │   ├── manager.py          ← StreamManager
│   │   │   │   ├── buffer.py           ← 帧缓冲
│   │   │   │   └── types.py            ← Frame, StreamConfig
│   │   │   ├── inference/              ← 推理调度
│   │   │   │   ├── __init__.py
│   │   │   │   ├── scheduler.py        ← InferenceScheduler
│   │   │   │   └── pipeline.py         ← 推理管线
│   │   │   ├── tracking/               ← 跟踪引擎
│   │   │   │   ├── __init__.py
│   │   │   │   ├── engine.py           ← TrackingEngine
│   │   │   │   └── types.py            ← Track, TrackState
│   │   │   ├── adaptive/               ← 自适应监控
│   │   │   │   ├── __init__.py
│   │   │   │   └── monitor.py          ← AdaptiveMonitor
│   │   │   ├── model/                  ← 数据模型
│   │   │   │   ├── __init__.py
│   │   │   │   ├── detection.py        ← Detection
│   │   │   │   └── geometry.py         ← IoU, polygon 工具
│   │   │   └── interfaces/             ← 插件接口定义
│   │   │       ├── __init__.py
│   │   │       ├── plugin.py           ← Plugin 基接口
│   │   │       ├── capture.py          ← CapturePlugin
│   │   │       ├── detector.py         ← DetectorPlugin
│   │   │       ├── tracker.py          ← TrackerPlugin
│   │   │       ├── attribute.py        ← AttributePlugin
│   │   │       ├── preprocessor.py     ← PreprocessorPlugin
│   │   │       ├── postprocessor.py    ← PostprocessorPlugin
│   │   │       ├── storage.py          ← StoragePlugin
│   │   │       ├── ui.py               ← UIPlugin
│   │   │       └── accelerator.py      ← AcceleratorPlugin
│   │   └── tests/
│   │
│   ├── sentineltrack-tracker-ukf/      ← UKF 族跟踪器（内置插件）
│   │   ├── pyproject.toml
│   │   ├── sentineltrack_tracker_ukf/
│   │   │   ├── __init__.py
│   │   │   ├── plugin.py               ← UKFTrackerPlugin
│   │   │   ├── ukf.py                  ← _DetectionUKF
│   │   │   ├── mcukf.py                ← _DetectionMCUKF
│   │   │   ├── manifold.py             ← _DetectionManifoldUKF
│   │   │   ├── adaptor.py              ← _FilterAdaptor
│   │   │   └── track.py                ← _Track
│   │   └── tests/
│   │
│   ├── sentineltrack-plugins/          ← 官方插件集合
│   │   ├── pyproject.toml
│   │   ├── sentineltrack_plugins/
│   │   │   ├── capture/
│   │   │   │   ├── directshow/         ← Windows USB
│   │   │   │   ├── v4l2/              ← Linux USB
│   │   │   │   ├── rtsp/             ← IP Camera
│   │   │   │   ├── file/             ← 视频文件
│   │   │   │   └── synthetic/        ← 测试用合成流
│   │   │   ├── detector/
│   │   │   │   ├── yolov8_onnx/      ← YOLOv8 ONNX
│   │   │   │   ├── yolo26/           ← YOLO26 .pt
│   │   │   │   └── uhd/              ← UHD 64x64
│   │   │   ├── tracker/
│   │   │   │   ├── bytetrack/        ← ByteTrack
│   │   │   │   └── ocsort/           ← OC-SORT
│   │   │   ├── attribute/
│   │   │   │   ├── head_classifier/  ← 头部属性
│   │   │   │   ├── gesture/          ← 手势识别
│   │   │   │   └── rectangle/        ← 矩形检测
│   │   │   ├── preprocessor/
│   │   │   │   ├── denoise/          ← 去噪
│   │   │   │   └── super_res/        ← 超分辨率
│   │   │   ├── postprocessor/
│   │   │   │   ├── nms/              ← NMS 去重
│   │   │   │   └── smoothing/        ← 框平滑
│   │   │   ├── storage/
│   │   │   │   ├── h264_mp4/         ← H.264 MP4
│   │   │   │   └── frame_sequence/   ← 帧序列
│   │   │   ├── accelerator/
│   │   │   │   ├── cuda/            ← CUDA
│   │   │   │   ├── directml/        ← DirectML
│   │   │   │   └── opencl/          ← OpenCL
│   │   │   └── ui/
│   │   │       └── pyqt6/           ← PyQt6 桌面 UI
│   │   └── tests/
│   │
│   ├── sentineltrack-studio/          ← 桌面应用发行版
│   │   ├── pyproject.toml
│   │   ├── sentineltrack_studio/
│   │   │   ├── __init__.py
│   │   │   ├── app.py                ← 应用入口
│   │   │   ├── main_window.py        ← 主窗口
│   │   │   ├── camera_cell.py        ← 画面显示
│   │   │   ├── settings_dialog.py    ← 设置对话框
│   │   │   ├── theme.py              ← 主题
│   │   │   └── splash.py             ← 启动闪屏
│   │   └── assets/
│   │
│   ├── sentineltrack-web/            ← Web UI 发行版
│   │   ├── pyproject.toml
│   │   ├── sentineltrack_web/
│   │   │   ├── server.py             ← FastAPI 服务
│   │   │   ├── api/                  ← REST API
│   │   │   ├── websocket/            ← WebSocket 流推送
│   │   │   └── static/               ← 前端资源
│   │   └── frontend/                 ← React/Vue 前端
│   │
│   └── sentineltrack-cli/           ← CLI 工具
│       ├── pyproject.toml
│       ├── sentineltrack_cli/
│       │   ├── __init__.py
│       │   ├── main.py              ← CLI 入口
│       │   ├── commands/
│       │   │   ├── run.py           ← 运行
│       │   │   ├── install.py       ← 安装插件
│       │   │   ├── search.py        ← 搜索插件
│       │   │   └── config.py        ← 配置管理
│       │   └── utils.py
│       └── tests/
│
├── docs/                             ← 文档
│   ├── architecture/                 ← 架构文档
│   ├── plugin-development/           ← 插件开发指南
│   ├── api/                          ← API 文档
│   └── deployment/                   ← 部署指南
│
├── tools/                            ← 开发工具
│   ├── benchmark/                    ← 性能基准
│   ├── profiler/                     ← 性能分析
│   └── codegen/                      ← 代码生成
│
├── ci/                               ← CI/CD
│   ├── github-actions/
│   └── docker/
│
└── examples/                         ← 示例
    ├── basic_detection/
    ├── multi_camera/
    ├── custom_plugin/
    └── edge_deployment/
```

---

## 7. 仓库重构方案

### 7.1 Monorepo vs Multi-repo 分析

| 维度 | Monorepo | Multi-repo |
|------|----------|------------|
| 代码共享 | ✅ 简单（同一仓库） | ❌ 需 npm/pip 发布 |
| 版本同步 | ✅ 原子提交 | ❌ 需协调多仓库版本 |
| CI/CD | ⚠️ 需要路径触发 | ✅ 每仓库独立 |
| 权限管理 | ⚠️ 全员可见全部代码 | ✅ 按仓库隔离 |
| 插件市场 | ✅ 统一发布 | ⚠️ 每插件独立仓库 |
| 代码搜索 | ✅ 全局搜索 | ❌ 跨仓库搜索困难 |
| 新人上手 | ✅ clone 一次即可 | ❌ 需 clone 多个仓库 |
| 包发布 | ✅ 统一发布脚本 | ✅ 独立发布 |
| 社区贡献 | ⚠️ 仓库大可能吓退 | ✅ 小仓库更友好 |

### 7.2 推荐：Monorepo（初期）→ 拆分（成熟期）

**阶段 1（0-12 月）：Monorepo**

```
sentineltrack/                    ← 单一 Monorepo
├── packages/
│   ├── sentineltrack-core/
│   ├── sentineltrack-tracker-ukf/
│   ├── sentineltrack-plugins/
│   ├── sentineltrack-studio/
│   ├── sentineltrack-web/
│   └── sentineltrack-cli/
├── docs/
├── tools/
├── ci/
└── examples/
```

理由：
- 初期代码频繁变更，原子提交保证一致性
- 插件接口未稳定前，统一仓库便于快速迭代
- 新开发者 clone 一次即可获得全部代码
- 使用 `pip install -e packages/sentineltrack-core` 开发安装

**阶段 2（12-24 月）：选择性拆分**

```
sentineltrack/                    ← 核心 Monorepo（保留）
├── packages/
│   ├── sentineltrack-core/
│   ├── sentineltrack-tracker-ukf/
│   ├── sentineltrack-studio/
│   └── sentineltrack-cli/

sentineltrack-plugins-community/  ← 社区插件独立仓库
├── plugins/
│   ├── capture/
│   ├── detector/
│   └── ...

sentineltrack-web/                ← Web 前端独立仓库（技术栈不同）
```

理由：
- 核心包保持 Monorepo（接口已稳定）
- 社区插件独立（降低核心仓库体积）
- Web 前端独立（技术栈完全不同）

### 7.3 包发布策略

```python
# 使用 pip install -e 开发安装
pip install -e packages/sentineltrack-core
pip install -e packages/sentineltrack-tracker-ukf
pip install -e packages/sentineltrack-plugins
pip install -e packages/sentineltrack-studio

# 发布到 PyPI
# 每个包独立发布，但版本号同步
sentineltrack-core           == 2.0.0
sentineltrack-tracker-ukf    == 2.0.0
sentineltrack-plugins        == 2.0.0
sentineltrack-studio         == 2.0.0

# 用户安装
pip install sentineltrack-studio  # 自动依赖 core + tracker + plugins
pip install sentineltrack-core    # 只装核心（无头服务）
```

---

## 8. 架构决策记录（ADR）

### 8.1 未来三年最值得坚持的 10 项架构决策

#### ADR-001：微内核架构

**决策**：核心层只包含 7 个模块（Plugin Framework / Event Bus / Config Manager / Stream Manager / Inference Scheduler / Tracking Engine / Adaptive Monitor），所有功能以插件形式存在。

**理由**：从架构上杜绝 God Class 再现。核心永不膨胀，新功能只能以插件加入。

**坚持期限**：永久。

---

#### ADR-002：进程内事件总线，不用 RPC

**决策**：插件间通信使用进程内事件总线（pub/sub），不使用 RPC/消息队列。

**理由**：实时视频处理中帧序列化成本不可接受（一帧 640×480 RGB ≈ 900KB，RPC 序列化 >5ms）。进程内事件总线延迟 <0.1ms。

**坚持期限**：永久（除非转向分布式架构）。

---

#### ADR-003：跟踪引擎纯 NumPy 实现

**决策**：UKF/MCUKF/Manifold UKF 保持纯 NumPy 实现，不引入 PyTorch/TensorFlow 等深度学习框架作为跟踪依赖。

**理由**：纯 NumPy 意味着零外部依赖，可移植到任何 Python 环境（包括嵌入式 Python）。如果引入 PyTorch，部署体积从 ~50MB 增加到 ~2GB。

**坚持期限**：永久（UKF 族）。其他跟踪器（ByteTrack/DeepSORT）可用各自依赖。

---

#### ADR-004：统一归一化坐标 [0,1]

**决策**：所有检测框、关键点、多边形顶点均使用归一化坐标 [0,1]，绘制时映射到像素坐标。

**理由**：分辨率无关，跨摄像头一致，跨平台一致。640×480 和 1920×1080 的检测结果可直接比较。

**坚持期限**：永久。

---

#### ADR-005：插件接口使用 Python Protocol

**决策**：插件接口使用 `typing.Protocol`（结构化子类型），而非 ABC（抽象基类）。

**理由**：Protocol 不要求显式继承，第三方插件无需安装 sentineltrack-core 即可实现接口（只需结构匹配）。降低插件开发门槛。

**坚持期限**：永久（除非迁移到非 Python 语言）。

---

#### ADR-006：共享推理线程模型

**决策**：1 个推理线程服务所有流，通过跳帧 + 预测填充维持实时性。

**理由**：避免多推理实例的 GPU 内存爆炸。4 路 × YOLOv8 = 4GB 显存 vs 1 路 = 1GB。共享线程是资源效率最优解。

**坚持期限**：直到 GPU 显存不再是瓶颈（或批处理推理成熟后可改为批处理模式）。

---

#### ADR-007：配置 Schema 使用 Pydantic

**决策**：配置 schema 使用 Pydantic BaseModel 定义，支持类型校验、默认值、文档生成。

**理由**：Pydantic 是 Python 生态最成熟的 schema 验证库。自动生成 JSON Schema 可用于 UI 动态生成配置界面。与 FastAPI 生态一致。

**坚持期限**：永久（除非 Pydantic 停止维护）。

---

#### ADR-008：插件通过 Python Entry Points 注册

**决策**：插件通过 `pyproject.toml` 的 `[project.entry-points]` 注册，而非自定义扫描。

**理由**：Entry Points 是 Python 生态标准机制，pip install 自动注册，无需手动配置。与 setuptools/poetry/hatch 兼容。

**坚持期限**：永久（除非 Python 生态发生根本变化）。

---

#### ADR-009：每流独立跟踪器实例

**决策**：每个流拥有独立的跟踪器实例，流之间跟踪状态完全隔离。

**理由**：不同场景可能使用不同跟踪器（安防用 UKF，云台用 OSTrack）。流间隔离避免状态污染。支持多跟踪器共存。

**坚持期限**：永久。

---

#### ADR-010：核心层零 PyQt 依赖

**决策**：sentineltrack-core 不依赖 PyQt6 或任何 GUI 框架。GUI 作为 UIPlugin 实现。

**理由**：核心层无 GUI 依赖意味着可以无头运行（服务器/边缘设备）。PyQt6 体积 ~100MB，不适合嵌入式部署。

**坚持期限**：永久。

---

### 8.2 未来最容易成为技术债务的 10 项架构决策

#### 技术债务-001：事件总线无持久化

**决策**：事件总线是进程内的，事件不持久化。

**风险**：进程崩溃后事件丢失。无法做事件回放/审计。

**缓解**：关键状态（轨迹/配置）通过 ConfigManager 和 TrackingEngine 持久化，不依赖事件总线。事件总线只用于实时通知。

---

#### 技术债务-002：插件热加载不保证状态迁移

**决策**：插件热加载时通过 `migrate_state()` 迁移状态，但不是所有插件都实现。

**风险**：热加载后跟踪状态丢失，导致 ID 重置。

**缓解**：核心跟踪器（UKF）必须实现 `migrate_state()`。非核心插件热加载时接受状态丢失。

---

#### 技术债务-003：单进程架构

**决策**：所有模块在同一进程中运行。

**风险**：一个插件崩溃可能导致整个进程崩溃。

**缓解**：插件异常由 PluginFramework 捕获，发布 `plugin.error` 事件，不影响其他插件。关键路径（推理/跟踪）有 try/except 保护。

---

#### 技术债务-004：Python GIL 限制

**决策**：使用 Python 多线程（threading），受 GIL 限制。

**风险**：CPU 密集型任务（UKF 计算/NMS）无法真正并行。

**缓解**：NumPy 操作释放 GIL。推理（ONNX/PyTorch）释放 GIL。只有纯 Python 逻辑受 GIL 限制。如果成为瓶颈，可用 multiprocessing 或 C 扩展。

---

#### 技术债务-005：插件版本兼容性

**决策**：通过 `api_version` 字符串匹配检查兼容性。

**风险**：版本号是声明式的，不是验证式的。插件可能声明兼容但实际不兼容。

**缓解**：提供插件兼容性测试套件，插件发布前必须通过。社区插件标记 "verified" / "unverified"。

---

#### 技术债务-006：配置热更新可能不一致

**决策**：配置变更通过 ConfigManager 立即生效。

**风险**：配置在推理过程中变更，可能导致当前帧使用新配置但跟踪状态是旧配置的。

**缓解**：配置变更通过事件通知，插件在下一个帧周期开始时应用新配置。当前帧周期内配置不变。

---

#### 技术债务-007：事件总线无背压

**决策**：事件总线异步分发，无背压机制。

**风险**：如果消费者处理速度慢于生产者，事件队列无限增长，内存溢出。

**缓解**：事件队列设上限（默认 1000），满则丢弃最旧事件并发布 `event.dropped` 警告。

---

#### 技术债务-008：多跟踪器结果格式不统一

**决策**：所有跟踪器输出统一的 `Track` 数据模型。

**风险**：不同跟踪器可能有不同的附加信息（如 DeepSORT 的 embedding、OSTrack 的 template），统一模型可能丢失信息。

**缓解**：`Track.attributes` 字典存储跟踪器特有的附加信息。消费者按需读取。

---

#### 技术债务-009：跨平台 GPU 抽象复杂度

**决策**：通过 AcceleratorPlugin 抽象 GPU 差异。

**风险**：CUDA/DirectML/OpenCL/NPU 的行为不完全一致（精度/性能/支持的算子），抽象层可能泄漏。

**缓解**：AcceleratorPlugin 接口最小化（只暴露 infer），具体后端差异由 DetectorPlugin 适配。提供标准化的精度/性能测试套件验证一致性。

---

#### 技术债务-010：Monorepo 体量增长

**决策**：初期使用 Monorepo。

**风险**：随着社区插件增加，仓库体量膨胀，clone/build 变慢。

**缓解**：12 个月后拆分社区插件到独立仓库。核心包保持 Monorepo。使用 sparse checkout 减少开发时 clone 体积。

---

### 8.3 ADR 汇总表

| 编号 | 决策 | 类型 | 风险等级 |
|------|------|------|----------|
| ADR-001 | 微内核架构 | ✅ 坚持 | 低 |
| ADR-002 | 进程内事件总线 | ✅ 坚持 | 低 |
| ADR-003 | 跟踪引擎纯 NumPy | ✅ 坚持 | 低 |
| ADR-004 | 归一化坐标 | ✅ 坚持 | 低 |
| ADR-005 | Protocol 接口 | ✅ 坚持 | 低 |
| ADR-006 | 共享推理线程 | ✅ 坚持 | 中（性能瓶颈时需改批处理） |
| ADR-007 | Pydantic 配置 | ✅ 坚持 | 低 |
| ADR-008 | Entry Points 注册 | ✅ 坚持 | 低 |
| ADR-009 | 每流独立跟踪器 | ✅ 坚持 | 低 |
| ADR-010 | 核心零 GUI 依赖 | ✅ 坚持 | 低 |
| 债务-001 | 事件无持久化 | ⚠️ 风险 | 中 |
| 债务-002 | 热加载状态迁移 | ⚠️ 风险 | 中 |
| 债务-003 | 单进程架构 | ⚠️ 风险 | 中 |
| 债务-004 | GIL 限制 | ⚠️ 风险 | 低（NumPy 释放 GIL） |
| 债务-005 | 版本兼容性 | ⚠️ 风险 | 中 |
| 债务-006 | 配置热更新一致性 | ⚠️ 风险 | 低 |
| 债务-007 | 事件无背压 | ⚠️ 风险 | 中 |
| 债务-008 | 多跟踪器格式 | ⚠️ 风险 | 低 |
| 债务-009 | GPU 抽象泄漏 | ⚠️ 风险 | 中 |
| 债务-010 | Monorepo 体量 | ⚠️ 风险 | 低（可拆分） |

---

## 附录：技术选型清单

| 类别 | 技术 | 版本 | 用途 |
|------|------|------|------|
| 语言 | Python | ≥3.10 | 主语言（Protocol 需要 3.8+，match 需要 3.10+） |
| 数值计算 | NumPy | ≥1.21 | 卡尔曼滤波、数组运算 |
| 图像处理 | OpenCV | ≥4.5 | Canny/轮廓/颜色转换 |
| 推理引擎 | ONNX Runtime | ≥1.24 | ONNX 推理（多 provider） |
| 推理引擎 | PyTorch | ≥2.0 | ultralytics .pt 推理 |
| GUI 框架 | PyQt6 | ≥6.2 | 桌面 UI（UIPlugin） |
| Web 框架 | FastAPI | ≥0.100 | Web UI + REST API |
| WebSocket | websockets | ≥11.0 | 实时流推送 |
| 配置 | Pydantic | ≥2.0 | 配置 schema 校验 |
| 采集 | PyAV | ≥10.0 | DirectShow/V4L2/RTSP 采集 |
| 打包 | hatch/hatchling | ≥1.0 | Python 包构建 |
| 测试 | pytest | ≥7.0 | 单元/集成测试 |
| CI/CD | GitHub Actions | - | 自动化构建/测试/发布 |
| 容器 | Docker | - | 容器化部署 |
| 文档 | MkDocs Material | - | 文档网站 |

---

> **文档状态**：SentinelTrack 2.0 Architecture Blueprint Draft 1
> 
> **下一步**：架构评审委员会评审 → 修订 → 发布 v1.0 → 启动重构
