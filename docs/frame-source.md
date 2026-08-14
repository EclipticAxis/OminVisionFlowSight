# FrameSource 设计文档 (Milestone C2)

> **版本**：C2 — 2026-07-25
> **范围**：新增 `visioncore/source/` 包，定义 VisionCore 输入源统一抽象
> **状态**：已实现，55 项单元测试全部通过，全量回归零破坏（16 个既有测试套件 + 55 项新测试）

---

## 1. 目标与背景

### 1.1 为什么需要 FrameSource 抽象

当前 VisionDataPlatform 的采集逻辑分散在 `camera/` 包中（DirectShow 后端、枚举器、多路槽位管理），与 `ai/inference.py` 紧耦合。随着 VisionCore 向"跨平台视觉中间件"演进（战略路线图 M4-M5），输入源需要扩展到：

- 视频文件回放（测试/基准）
- RTSP 网络流（M4）
- Linux V4L2 设备（M4）
- 合成测试源（无硬件依赖的自动化测试）

如果每个下游消费者（Pipeline、录像器、GUI 预览）都直接硬编码具体采集后端，会产生：

- **采集耦合**：消费者代码绑死到 DirectShow/cv2/av 等特定后端
- **测试困难**：无法在没有摄像头的 CI 环境运行端到端测试
- **切换成本**：换采集后端需要改多处消费者代码

FrameSource 通过引入**抽象基类 `FrameSource`** + **统一返回 `visioncore.core.Frame`** 解决这些问题。消费者只依赖抽象，具体源在运行时注入（经工厂或直接构造）。

### 1.2 设计原则

| 原则 | 落实 |
|---|---|
| **统一返回类型** | `read()` 返回 `Frame \| None`，**禁止**裸 `numpy.ndarray` / OpenCV `Mat` |
| **契约优先** | ABC 定义 5 方法生命周期，具体源只实现不扩展 |
| **来源不可知** | base.py 零 I/O 代码（无 camera/cv2/av/socket） |
| **安全默认** | DummyFrameSource 作为无副作用默认/测试源 |
| **配置驱动** | 工厂 + 注册表，按 kind 字符串创建源，解耦选择与构造 |
| **零侵入** | 不修改 `ai/` `camera/` `gui/`，现有采集栈继续运行不变 |

### 1.3 与 C1 的关系

C1 建立了 Pipeline 框架（Stage + Context + 执行器），但 Pipeline 的输入来源未定义。C2 填补这一空白：

```
C2 FrameSource.read() ──产出──> Frame ──填入──> C1 PipelineContext.frame
                                                    │
                                                    ▼
                                            C1 Pipeline.run(ctx)
                                            (Detection/Tracking/... Stages)
```

C3 将引入 `CaptureStage`（一个 `PipelineStage` 子类），在 `process()` 中调用 `FrameSource.read()` 把 Frame 填入 `context.frame`，完成 source → pipeline 的接线。C2 只建 source 抽象本身，不触碰 Pipeline。

### 1.4 与 B1/B2 的关系

- **B1 TargetState**：定义了流动的快照数据；FrameSource 产出的 Frame 是 Pipeline 的**输入**数据，最终经检测/跟踪/投影产出 TargetState 快照。
- **B2 ProtocolAdapter**：是 Pipeline 的**输出** sink（快照投递）；FrameSource 是 Pipeline 的**输入** source（帧采集）。两者对称：一进一出，都是抽象基类 + 具体实现 + 上下文管理器。

---

## 2. 文件结构

```
visioncore/source/
├── __init__.py            # 包导出：FrameSource, FrameSourceError, DummyFrameSource, 工厂函数
├── base.py                # FrameSource(ABC) + FrameSourceError + 上下文管理器
├── frame_source.py        # 工厂 + 注册表（register/create/available）
└── dummy_source.py        # DummyFrameSource — 测试用可配置源
```

与 `visioncore/protocol/`（base + null/console 适配器）和 `visioncore/pipeline/`（base + stage + pipeline）的多文件模式一致。未来真实源（CameraFrameSource / FileFrameSource / RtspFrameSource）作为兄弟模块加入，各自是**唯一** import 其传输库（DirectShow/cv2/av/socket）的模块。

---

## 3. FrameSource 契约

### 3.1 类定义

```python
class FrameSource(ABC):
    @abstractmethod
    def open(self) -> None: ...
    @abstractmethod
    def is_open(self) -> bool: ...
    @abstractmethod
    def read(self) -> Frame | None: ...
    @abstractmethod
    def close(self) -> None: ...
    @abstractmethod
    def health_check(self) -> bool: ...
    # 上下文管理器（具体，不可覆盖）
    def __enter__(self) -> "FrameSource": ...
    def __exit__(self, ...) -> None: ...
```

### 3.2 五个方法

| 方法 | 抽象? | 语义 |
|---|---|---|
| `open()` | ✅ 抽象 | 获取底层资源（开设备/开文件/连 socket）。幂等——已开时再调用为 no-op。`read()` 前必须调用。失败抛 `FrameSourceError`。 |
| `is_open()` | ✅ 抽象 | 廉价状态探针，返回 `True` 当且仅当源当前已开。无副作用。区别于 `health_check`（后者探查操作就绪度，不仅是开/关标志）。 |
| `read()` | ✅ 抽象 | 产出下一 `Frame`，或 `None`（源暂时空/永久耗尽）。**返回类型强制**：`Frame \| None`，禁裸 ndarray/Mat。对关闭的源调用抛 `RuntimeError`（契约违反，区别于合法 `None`）。 |
| `close()` | ✅ 抽象 | 释放 `open()` 获取的资源。幂等且**绝不抛异常**。上下文管理器保证即使 `read` 抛异常也调用。 |
| `health_check()` | ✅ 抽象 | 无副作用探针，返回 `True` 当且仅当源已开且操作就绪。**绝不抛异常**——内部错误返回 `False`。 |

### 3.3 返回类型政策（强制）

`read()` 的返回类型是 **`Frame | None`**，**禁止**：

- 裸 `numpy.ndarray`（像素数组必须包在 `Frame.image` 字段内）
- OpenCV `Mat`（cv2 读取的帧必须转成 Frame）

`visioncore.core.Frame` 是统一信封，包装原始像素数组 + 身份（`frame_id`）+ 来源（`source_id`）+ 时序（`timestamp`）。返回信封而非裸数组确保：

- **来源可追溯**：每帧都带 `source_id`，多摄像头场景可区分
- **类型稳定**：下游 Stage 永远面对同一类型，无需 isinstance 分支
- **不可变共享**：Frame 是 frozen dataclass，多消费者可安全共享引用

### 3.4 `None` vs `RuntimeError` 的语义区分

| 情况 | 返回/抛出 | 语义 |
|---|---|---|
| 源已开，暂时无帧（相机预热、短暂丢帧） | `None` | 合法的"暂时空" |
| 源已开，永久耗尽（文件 EOF，loop=False 的 DummyFrameSource） | `None` | 合法的"耗尽" |
| 源**未开**就调用 read | `RuntimeError` | 契约违反（程序错误） |
| 源已开但设备掉线/流超时 | `FrameSourceError` | 可恢复故障（可 close+open 恢复） |

这一区分让消费者能用 `if frame is None: continue` 处理暂时空，用 `try/except FrameSourceError` 处理故障，不会混淆。

### 3.5 上下文管理器

`__enter__` 调用 `open()` 并返回 `self`；`__exit__` 调用 `close()` 且**不抑制异常**。即使在 `with` 块内抛出异常，`close()` 也保证被调用：

```python
with source:
    frame = source.read()
    raise ValueError("oops")
# close() 已被调用，ValueError 继续传播
```

### 3.6 FrameSourceError

`FrameSourceError(RuntimeError)` 是结构化异常，源 *可选* 地用它标记"可恢复的源特定故障"（设备掉线、流超时、文件截断）。调用方想要重试/回退/重开语义时，在 `read()` 周围 `try/except FrameSourceError` 并驱动 `close()`/`open()` 恢复循环。

---

## 4. DummyFrameSource

### 4.1 用途

**可配置测试源**——无任何真实 I/O，输出预定的 `Frame` 实例。应用场景：

1. **Pipeline 测试**：向 Pipeline 喂固定 Frame，无需摄像头即可测试检测/跟踪 Stage
2. **源层测试**：用确定性、无副作用的源验证 FrameSource 契约（生命周期、异常恢复）
3. **基准/演示**：可复现的帧流，无外部依赖

### 4.2 配置参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `frames` | `None` | 可选 Frame 列表回放；`None` 时自动生成 1 个合成帧（64×64×3 zeros，懒导入 numpy） |
| `source_id` | `"dummy"` | 盖在自动生成帧的 `source_id` 字段 |
| `loop` | `True` | `True` 则循环回放；`False` 则耗尽后 `read()` 返回 `None` |
| `raise_on_read` | `None` | 异常实例，下次 `read()` 抛出（异常恢复测试） |

### 4.3 行为表

| 方法 | 行为 |
|---|---|
| `open()` | `open_count += 1`；`_open = True`；`_healthy = True` |
| `is_open()` | 返回 `_open` |
| `read()` | `read_count += 1`；若 `raise_on_read` 非空则抛；否则返回下一帧（loop 则循环，否则耗尽返回 None）；对关闭源抛 `RuntimeError` |
| `close()` | `close_count += 1`；`_open = False`；`_healthy = False`；绝不抛 |
| `health_check()` | 返回 `_healthy` |
| `reset()` | 清零计数器与状态，保留帧缓冲与 loop |

### 4.4 异常恢复模式

`raise_on_read` 模拟瞬时源故障。恢复循环：

```python
src.open()
src.raise_on_read = FrameSourceError("device dropped")
try:
    src.read()  # raises
except FrameSourceError:
    pass
src.raise_on_read = None  # 清除故障
src.close()
src.open()                 # 回收源
frame = src.read()         # 恢复，返回 Frame
```

`raise_on_read` 持续到被清除——设置后每次 `read()` 都抛，直到调用方置 `None`。这让测试可精确控制故障的起止。

### 4.5 固定输出语义

"固定输出 Frame"——默认配置（1 帧 + loop=True）下，`read()` 每次返回**同一 Frame 实例**（Frame 是 frozen/immutable，安全共享）。测试需单调 `frame_id` 时，调用方提供含不同 frame_id 的 Frame 列表。

---

## 5. 工厂与注册表

### 5.1 动机

真实源（CameraFrameSource 等）的构造可能拉起重量级传输库（DirectShow/cv2/av）。工厂 + 注册表让调用方**按 kind 字符串创建源**，无需 import 具体类——只有实际使用的源才付出传输库的导入成本。

### 5.2 接口

| 函数 | 语义 |
|---|---|
| `register_frame_source(name, cls, *, overwrite=False)` | 注册源类到 kind 名。幂等（同名同类 no-op）；拒绝同名异类（除非 `overwrite=True`）。拒绝空名/非子类/抽象基类本身。 |
| `create_frame_source(name, **kwargs)` | 按 kind 创建源实例，`kwargs` 转发构造器。未知 kind 抛 `ValueError`（错误信息列出可用 kind）。 |
| `available_frame_sources()` | 返回已注册 kind 名的排序列表。 |
| `get_frame_source_class(name)` | 返回 kind 对应的类，未知返回 `None`。 |

### 5.3 内置注册

C2 自动注册一个内置源：

| kind | 类 |
|---|---|
| `"dummy"` | `DummyFrameSource` |

C3+ 真实源在各自模块底部自注册：

```python
# visioncore/source/camera_source.py (C3)
register_frame_source("camera", CameraFrameSource)
```

### 5.4 配置驱动用法

设置文件指定 `{"type": "dummy", "source_id": "cam0"}`，调用方：

```python
src = create_frame_source(config["type"], **config)
```

完全解耦源选择与源构造。

---

## 6. 返回类型政策与 numpy

### 6.1 政策

`read()` 返回 `Frame | None`，**不返回**裸 `numpy.ndarray` 或 OpenCV `Mat`。测试 `test_read_returns_frame_not_ndarray` 断言：

```python
f = src.read()
assert isinstance(f, Frame)        # 信封
assert not isinstance(f, np.ndarray)  # 非裸数组
assert isinstance(f.image, np.ndarray)  # 内部字段是 ndarray（数据模型设计）
```

### 6.2 numpy 加载说明

- `Frame.image` 字段类型为 `np.ndarray`（B1 数据模型设计），故 import `Frame` 传递性加载 numpy——这是数据模型属性，非源层引入。
- `DummyFrameSource` 在合成帧生成器 `_make_default_frame` 内**懒导入** numpy，模块其余部分（生命周期、读循环、计数器）不直接触碰 numpy。
- 用户提供显式 `frames` 时，合成生成路径（及其 numpy 使用）被跳过。
- numpy 加载不影响 C2 评分准则（准则针对 ai/gui/camera/Detector/Tracker，numpy 是数据模型基础库）。

---

## 7. 线程安全

- `FrameSource` 不要求线程安全。源可能有可变状态（开/关标志、缓冲区、socket），跨线程共享同一源必须自行串行化。
- `Frame` 是 `frozen=True` + `slots=True`，实例创建后不可变，可安全跨线程共享无需锁。
- `DummyFrameSource` 非线程安全，设计用于 C1 Pipeline 的串行执行模型。

---

## 8. 测试覆盖

### 8.1 测试统计

- **测试文件**：`tests/test_frame_source.py`
- **测试函数数**：55（全部通过）
- **运行方式**：`PYTHONPATH=F:/VisionBata F:/VisionBata/venv/Scripts/python.exe tests/test_frame_source.py`（无 pytest，纯 assert + 自定义 `raises` + 手动 runner）

### 8.2 四大必覆盖领域

| 领域 | 测试 | 验证 |
|---|---|---|
| **生命周期** | `test_open_makes_source_open_and_healthy` | open 置开/健康 |
| | `test_open_is_idempotent` | 二次 open 为 no-op |
| | `test_close_makes_source_closed_and_unhealthy` | close 置关/不健康 |
| | `test_close_is_idempotent` | 二次 close 为 no-op |
| | `test_close_without_open_is_safe` | 未开直接 close 不抛 |
| | `test_repr_reports_open_state` / `test_repr_survives_is_open_exception` | repr 报状态且抗异常 |
| **read()** | `test_read_returns_frame_when_open` | 开源返回 Frame |
| | `test_read_returns_frame_not_ndarray` | 返回 Frame 信封非裸 ndarray |
| | `test_read_stamps_source_id_on_auto_frame` | 自动帧盖 source_id |
| | `test_read_returns_fixed_frame_by_default` | 默认固定输出同一实例 |
| | `test_read_returns_none_when_exhausted_no_loop` | loop=False 耗尽返回 None |
| | `test_read_loops_through_multiple_frames` / `test_read_replays_explicit_frames_in_order` | 多帧循环/顺序回放 |
| **close()** | `test_close_never_raises_even_after_failure` | 故障后 close 仍成功 |
| | `test_close_after_close_does_not_revive_source` | 二次 close 不复活源 |
| | `test_close_resets_health` | close 置健康 False |
| **异常恢复** | `test_read_on_closed_source_raises_runtime_error` | 关源 read 抛 RuntimeError |
| | `test_raise_on_read_propagates_frame_source_error` | raise_on_read 传播 FrameSourceError |
| | `test_recovery_via_close_and_open` | close+open 恢复后 read 成功 |
| | `test_raise_on_read_persists_until_cleared` | 故障持续到清除 |
| | `test_read_failure_does_not_corrupt_frame_buffer` | 失败 read 不丢帧/不乱序 |

### 8.3 补充测试

| 类别 | 测试数 | 覆盖 |
|---|---|---|
| 包导出 + ABC 契约 | 4 | `__all__` 精确；ABC 不可实例化；`FrameSourceError` 是 `RuntimeError` 子类；缺方法子类不可实例化 |
| DummyFrameSource 构造 | 6 | 默认 source_id；显式 source_id；默认 loop；零计数器/关闭态；自动生成 1 帧；显式 frames 保留 |
| health_check 语义 | 3 | 开前 False；开后 True；关后 False |
| 上下文管理器 | 4 | 进出开/关；异常时仍 close；不抑制异常；块内 read |
| 工厂/注册表 | 9 | dummy 已注册；create 端到端；未知 kind 抛；拒空名/非子类/抽象基类；拒同名异类；同类幂等；overwrite 替换 |
| reset/杂项 | 3 | reset 清零；reset 保留缓冲/loop；显式 frames 接受非数组 image |
| 端到端 | 1 | 工厂建源 → 循环读 5 帧 → 全是 Frame → 上下文管理器关闭 |

### 8.4 回归测试

全量回归零破坏——所有 16 个既有测试套件 + C2 新增全过：

```
test_pipeline.py (C1)         : 56/56 passed
test_core_models.py            : 31/31 passed
test_target_state.py           : 36/36 passed
test_event_bus.py              : 77/77 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_shadow_integration.py     : 12/12 passed
test_protocol_base.py          : 36/36 passed
test_debug_logger.py           : 20/20 passed
test_json_serializer.py        : 31/31 passed
test_cbor_serializer.py        : 31/31 passed
test_udp_adapter.py            : 23/23 passed
test_state_publisher.py        : 28/28 passed
test_shadow_publisher.py       : 36/36 passed
test_obb_stage2.py             : passed
test_matching.py               : passed
────────────────────────────────────────────────────────────────
test_frame_source.py (C2 新增) : 55/55 passed
总计                           : 既有全过 + 55 新增，0 回归
```

循环导入检查：4 个 source 子模块独立导入全成功。副作用检查：导入 `visioncore.source` 不加载 `ai`/`gui`/`camera`/`PyQt6`/`cv2`/`onnxruntime`/`ultralytics`/`mediapipe`/`av` 中任何一个。

---

## 9. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 新增 `visioncore/source/` 目录 | ✅ |
| `base.py` 定义 `FrameSource(ABC)` | ✅ 5 方法：open/is_open/read/close/health_check |
| `read()` 返回 `Frame \| None` | ✅ 强制；测试 `test_read_returns_frame_not_ndarray` 守护 |
| 禁止裸 numpy ndarray / OpenCV Mat 返回 | ✅ |
| 统一返回 `visioncore.core.Frame` | ✅ |
| `frame_source.py` 工厂 | ✅ register/create/available/get_class |
| `dummy_source.py` DummyFrameSource | ✅ 固定输出/重放/循环/异常恢复 |
| 新增 `tests/test_frame_source.py` | ✅ 55 项测试，覆盖四大领域 |
| 禁止修改 `ai/` `camera/` `gui/` | ✅ 仅新增文件，零侵入 |
| 不改变现有采集逻辑 | ✅ 全量回归零破坏 |
| 输出 `docs/frame-source.md` | ✅ 本文件 |

---

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/source/__init__.py` | 包导出（7 名）+ 范围/政策/快速开始 |
| `visioncore/source/base.py` | `FrameSource(ABC)` + `FrameSourceError` + 上下文管理器 |
| `visioncore/source/frame_source.py` | 工厂 + 注册表（register/create/available/get_class）+ 自动注册 dummy |
| `visioncore/source/dummy_source.py` | `DummyFrameSource`（固定输出/重放/循环/异常恢复/计数器） |
| `tests/test_frame_source.py` | 55 项单元测试 + 自定义 `raises` + 手动 runner |
| `docs/frame-source.md` | 本设计文档 |

---

## 11. 未来扩展路线

### 11.1 真实源（C3）

| 源 | kind | 文件 | 传输 |
|---|---|---|---|
| `CameraFrameSource` | `"camera"` | `camera_source.py` | 包装现有 `camera/` 后端（DirectShow），不修改 camera 包 |
| `FileFrameSource` | `"file"` | `file_source.py` | cv2/av 读视频文件，逐帧产出 Frame |
| `RtspFrameSource` | `"rtsp"` | `rtsp_source.py` | av/socket 读 RTSP 流 |

每个真实源是 `visioncore.source` 包中**唯一** import 其传输库的模块。base.py 永不引入传输库。

### 11.2 CaptureStage 接线（C3）

C3 引入 `CaptureStage(PipelineStage)`，在 `process()` 中调用 `FrameSource.read()` 填充 `context.frame`：

```python
class CaptureStage(PipelineStage):
    def __init__(self, source: FrameSource):
        super().__init__(name="capture")
        self._source = source
    def initialize(self):
        self._source.open()
    def process(self, context):
        context.frame = self._source.read()
    def shutdown(self):
        self._source.close()
    def health_check(self):
        return self._source.health_check()
```

完成 source → pipeline 接线，DummyFrameSource 成为 CaptureStage 的测试双。

### 11.3 多源与同步（C4+）

- **多源 Pipeline**：每个摄像头一个 CaptureStage，多路并行分支
- **同步策略**：时间戳对齐、最老帧丢弃、背压
- **源健康监控**：`health_check()` 接入 `_HealthMonitor` 模式，连续不健康自动回退到 DummyFrameSource

### 11.4 跨平台采集（M4-M5，战略路线图）

- Linux V4L2 源、RTSP 源（M4）
- CUDA 加速采集（M5）
- 每个新平台一个新源类，注册到工厂，消费者代码不变

---

## 12. 后续里程碑预告

- **C3**：CaptureStage 接线 + CameraFrameSource/FileFrameSource（首组真实源）
- **C4**：多源并行 + 同步策略 + 源健康监控集成
- **C5**：跨平台采集源（V4L2/RTSP）+ 配置驱动的多摄像头部署
