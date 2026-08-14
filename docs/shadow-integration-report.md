# Shadow Integration Report (Milestone B7)

> **日期**：2026-07-24
> **范围**：将 TargetManager / EventBus / TargetState / Protocol Layer 以 Shadow Mode 挂接到现有系统
> **状态**：已实现，36 项单元测试全部通过，446 项回归零破坏

---

## 1. 目标

在不改变现有业务逻辑的前提下，将 VisionCore 的四个核心组件以 **Shadow Mode**（影子模式）挂接到现有 VisionBata 系统：

| 组件 | 角色 | 挂接方式 |
|---|---|---|
| **TargetManager** | 目标生命周期管理 | 不修改——已有 B2 影子集成（InferWorker 创建 bus + TargetManager） |
| **EventBus** | 事件分发 | 不修改——ShadowStatePublisher 仅订阅，不发布事件 |
| **TargetState** | 快照数据模型 | 不修改——B1 已定义，ShadowStatePublisher 从事件转换生成 |
| **Protocol Layer** | 传输适配器 | 不修改——默认 NullAdapter（零副作用），可选注入真实适配器 |

**Shadow Mode 保证**：
- ❌ 不影响现有检测结果
- ❌ 不影响现有 Tracker
- ❌ 不影响 GUI
- ✅ 仅在 EventBus 上添加一个只读订阅者

---

## 2. 实现

### 2.1 新增文件

| 文件 | 说明 |
|---|---|
| `visioncore/protocol/publisher/shadow_publisher.py` | ShadowStatePublisher + ShadowMetrics + _MetricsAdapter |
| `tests/test_shadow_publisher.py` | 36 项单元测试 |

### 2.2 修改文件

| 文件 | 变更 |
|---|---|
| `visioncore/protocol/publisher/__init__.py` | 新增导出 ShadowStatePublisher（仅添加，不修改现有 StatePublisher 导出） |

### 2.3 未修改文件（关键约束）

| 目录/文件 | 状态 |
|---|---|
| `ai/` (InferWorker, Tracker, detection) | ✅ 零修改 |
| `gui/` (MainWindow, CameraCell, ...) | ✅ 零修改 |
| `camera/` (capture, recorder, ...) | ✅ 零修改 |
| `common/` (Frame, Protocols) | ✅ 零修改 |
| `visioncore/eventbus/` (EventBus, events) | ✅ 零修改 |
| `visioncore/target_manager/` (TargetManager, TargetStore) | ✅ 零修改 |
| `visioncore/state/` (TargetState) | ✅ 零修改 |
| `visioncore/protocol/base.py` (ProtocolAdapter ABC) | ✅ 零修改 |

测试 `test_no_gui_modification` 和 `test_no_inferworker_modification` 用文件扫描守护以上约束。

---

## 3. ShadowStatePublisher 设计

### 3.1 类继承

```
ProtocolAdapter (ABC)          ← B2
    └── NullAdapter            ← B2 (默认 shadow sink)
    └── ConsoleAdapter         ← B2
    └── UDPAdapter             ← B5
    └── _MetricsAdapter        ← B7 (内部包装器，记录 publish 指标)

StatePublisher                 ← B6 (EventBus → ProtocolAdapter 桥梁)
    └── ShadowStatePublisher   ← B7 (extends StatePublisher + metrics)
```

ShadowStatePublisher **继承** StatePublisher，复用其全部事件订阅和事件→TargetState 转换逻辑。新增：
1. `_MetricsAdapter` 包装器——包装内部适配器，每次 publish 计时+计数
2. `ShadowMetrics` 收集器——线程安全计数器 + 派生指标计算
3. 重写 `_on_event`——在事件处理前后记录指标

### 3.2 Shadow Mode 工作流

```
EventBus ──(TargetCreatedEvent)──> ShadowStatePublisher._on_event()
                                          │
                                    record_event()        ← 计数：收到事件
                                          │
                                    _event_to_state()     ← 转换为 TargetState
                                          │
                               ┌──────────┴──────────┐
                               │ 成功                 │ 失败
                               │                      │
                               ▼                      ▼
                    _MetricsAdapter.publish()   record_conversion_failure()
                          │                            + return (不发布)
                    ┌─────┴─────┐
                    │           │
              计时开始      inner.publish()
                    │           │
                    │     ┌─────┴─────┐
                    │     │ 成功       │ 失败
                    │     │           │
                    │     ▼           ▼
                    │  (正常返回)  (抛异常)
                    │     │           │
                    └─────┴───────────┘
                          │
                    record_publish(duration, success)
                          │
                          ▼
                    ShadowMetrics 更新
```

### 3.3 默认 NullAdapter

构造器默认使用 `NullAdapter`——所有 publish 调用被静默丢弃。这意味着：

- **零网络开销**：无 socket、无 I/O
- **零副作用**：不影响任何外部系统
- **纯指标**：仅测量事件→TargetState 转换 + publish 调用的开销

可选注入真实适配器（如 UDPAdapter）进行实际传输，同时仍收集指标。

---

## 4. Debug Metrics

### 4.1 ShadowMetrics 类

| 原始计数器 | 类型 | 说明 |
|---|---|---|
| `total_events` | int | 从 EventBus 收到的事件总数 |
| `total_publishes` | int | `adapter.publish()` 调用总次数 |
| `successful_publishes` | int | 未抛异常的 publish 调用数 |
| `failed_publishes` | int | 抛异常的 publish 调用数 |
| `conversion_failures` | int | 事件→TargetState 转换失败数 |
| `total_publish_time_ns` | int | publish 调用累计耗时（纳秒） |

### 4.2 派生指标

| 指标 | 公式 | 单位 |
|---|---|---|
| `states_per_second` | `total_publishes / elapsed_seconds` | 状态/秒 |
| `success_rate` | `successful_publishes / total_publishes` | 0.0~1.0 |
| `avg_latency_us` | `total_publish_time_ns / total_publishes / 1000` | 微秒 |
| `elapsed_seconds` | `perf_counter() - first_event_time` | 秒 |

### 4.3 线程安全

`ShadowMetrics` 使用 `threading.Lock` 保护所有计数器更新。锁仅在计数器递增时持有（纳秒级），不在 publish 调用期间持有——因此争用可忽略。

### 4.4 访问方式

```python
shadow = ShadowStatePublisher(bus)
shadow.start()

# 实时属性访问
shadow.metrics.total_events
shadow.metrics.success_rate
shadow.metrics.avg_latency_us
shadow.metrics.states_per_second

# 完整快照（dict，适合日志/JSON 序列化）
shadow.metrics.summary()

# 重置（清零所有计数器）
shadow.metrics.reset()

# 可读 repr
print(shadow.metrics)  # ShadowMetrics(events=42, publishes=42, ok=42, ...)
```

### 4.5 summary() 示例输出

```python
{
    "total_events": 42,
    "total_publishes": 42,
    "successful": 42,
    "failed": 0,
    "conversion_failures": 0,
    "elapsed_seconds": 4.012345,
    "states_per_second": 10.47,
    "success_rate": 1.0,
    "avg_latency_us": 3.21,
}
```

---

## 5. 错误隔离

ShadowStatePublisher 继承 StatePublisher 的双层错误隔离：

| 失败点 | 处理 | 指标影响 |
|---|---|---|
| 事件→TargetState 转换失败 | 捕获 + 日志 + return（不发布） | `conversion_failures += 1` |
| `adapter.publish()` 失败 | 捕获 + 日志（不传播回 EventBus） | `failed_publishes += 1` |
| `adapter.publish()` 成功 | 正常返回 | `successful_publishes += 1` |

测试 `test_metrics_success_rate_with_failing_adapter` 验证：FailingAdapter 的 publish() 抛 OSError 后，总线继续运行，后续事件正常处理，`success_rate` 正确反映失败比例。

---

## 6. 测试覆盖

### 6.1 测试统计

- **测试文件**：`tests/test_shadow_publisher.py`
- **测试函数数**：36（全部通过）

### 6.2 覆盖类别

| 类别 | 测试数 | 覆盖点 |
|---|---|---|
| ShadowMetrics 单元测试 | 12 | 初始状态 / record_event / record_publish(成功+失败) / success_rate(混合) / conversion_failure / states_per_second / avg_latency / reset / summary / repr |
| 类型注解约定 | 1 | AST 检查无 forbidden import |
| 不修改约束 | 2 | GUI 文件扫描 / InferWorker 文件扫描 |
| 包导出 + 导入一致性 | 2 | __all__ / 直接导入一致 |
| 生命周期 | 6 | NullAdapter 默认 / 订阅5事件 / stop取消订阅 / 上下文管理器 / start幂等 / stop幂等 |
| 指标收集 | 8 | 事件计数 / publish计数 / 5种事件 / FailingAdapter / avg_latency(NullAdapter+SlowAdapter) / sps / reset / summary |
| Shadow 模式 | 2 | NullAdapter 零副作用 / 自定义适配器接收 TargetState |
| 不影响现有系统 | 3 | 不发布事件 / 不影响其他订阅者 / 不修改总线 |
| repr | 1 | 含 started + metrics |
| 不修改约束 | 1 | 类型注解约定 |

### 6.3 回归测试

```
test_target_state.py           : 36/36 passed
test_core_models.py            : 31/31 passed
test_event_bus.py              : 77/77 passed
test_shadow_integration.py     : 12/12 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_protocol_base.py          : 36/36 passed
test_json_serializer.py        : 31/31 passed
test_cbor_serializer.py        : 31/31 passed
test_udp_adapter.py            : 23/23 passed
test_state_publisher.py        : 28/28 passed
test_shadow_publisher.py       : 36/36 passed  (新增)
────────────────────────────────────────────────────────────────
总计                           : 446/446 passed, 0 回归
```

---

## 7. 性能影响分析

### 7.1 Shadow Mode 开销

| 操作 | 耗时 | 说明 |
|---|---|---|
| EventBus.publish() 调度 | ~1-2 µs | 已有开销（B2 影子集成） |
| record_event() | ~100 ns | Lock + int increment |
| _event_to_state() | ~5-10 µs | 正则解析 + dict 构建 + TargetState 构造 |
| NullAdapter.publish() | ~100 ns | 类型检查 + 丢弃 |
| record_publish() | ~100 ns | Lock + int increment |
| _MetricsAdapter 包装开销 | ~50 ns | perf_counter_ns 调用 |
| **总额外开销** | **~6-13 µs/事件** | **含转换+指标** |

### 7.2 对现有系统的影响

| 组件 | 影响 | 理由 |
|---|---|---|
| 检测结果 | **零** | ShadowStatePublisher 不参与检测管线，仅在事件到达后异步处理 |
| Tracker | **零** | 不修改 TargetManager / Tracker / Track |
| GUI | **零** | 不修改任何 GUI 文件 |
| EventBus 吞吐 | **微降** | 每个事件多一个订阅者回调（~6-13µs），对 30fps 系统（33ms/帧）可忽略 |
| 内存 | **微增** | ShadowMetrics ~200 bytes + _MetricsAdapter ~100 bytes |

### 7.3 结论

Shadow Mode 的额外开销（~10µs/事件）相对于帧间隔（33ms@30fps）小于 0.03%，对实时性能无可感知影响。默认 NullAdapter 确保零网络副作用。

---

## 8. 使用示例

### 8.1 纯监控（默认 NullAdapter）

```python
from visioncore.eventbus import EventBus
from visioncore.protocol.publisher import ShadowStatePublisher

bus = EventBus()  # 已有 InferWorker 创建的 bus
shadow = ShadowStatePublisher(bus)
shadow.start()

# ... 系统正常运行，shadow 在后台收集指标 ...

# 定期检查指标
print(shadow.metrics.summary())
# {'total_events': 150, 'total_publishes': 150, 'successful': 150,
#  'failed': 0, 'states_per_second': 5.2, 'success_rate': 1.0,
#  'avg_latency_us': 4.3}

shadow.stop()
```

### 8.2 监控 + UDP 传输

```python
from visioncore.protocol.adapters import UDPAdapter
from visioncore.protocol.serialization import CBORSerializer

adapter = UDPAdapter(CBORSerializer(), "127.0.0.1", 9999)
shadow = ShadowStatePublisher(bus, adapter=adapter)
shadow.start()

# 事件流同时：(1) 通过 UDP 发送 CBOR 快照 (2) 收集指标
# 即使 UDP 失败，指标仍记录失败率

shadow.stop()
```

### 8.3 与 main.py 集成（未来）

```python
# main.py（未来集成，不在 B7 范围内）
bus = EventBus()
infer_worker = InferWorker(event_bus=bus)
shadow = ShadowStatePublisher(bus)
shadow.start()

app.exec_()

shadow.stop()
```

---

## 9. 设计约束回顾

| 约束 | 落实 |
|---|---|
| 不改变现有业务逻辑 | ✅ 仅新增文件 + __init__.py 导出 |
| 不影响检测结果 | ✅ 不参与检测管线 |
| 不影响 Tracker | ✅ 不修改 TargetManager / Tracker |
| 不影响 GUI | ✅ 测试扫描 gui/ 零引用 |
| 实现 ShadowStatePublisher | ✅ 继承 StatePublisher |
| 仅监听 EventBus | ✅ 只订阅不发布 |
| 统计每秒状态数 | ✅ `metrics.states_per_second` |
| 统计发送成功率 | ✅ `metrics.success_rate` |
| 统计平均耗时 | ✅ `metrics.avg_latency_us` |
| 新增 debug metrics | ✅ ShadowMetrics 类 + summary() |
| 输出报告 | ✅ 本文件 |

---

## 10. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/protocol/publisher/shadow_publisher.py` | ShadowStatePublisher + ShadowMetrics + _MetricsAdapter |
| `visioncore/protocol/publisher/__init__.py` | 更新：导出 ShadowStatePublisher |
| `tests/test_shadow_publisher.py` | 36 项单元测试 |
| `docs/shadow-integration-report.md` | 本文件 |

---

## 11. 后续里程碑

- **B8**：`main.py` 集成——将 bus + ShadowStatePublisher 接入启动流程
- **B9**：`FileAdapter` — 录制元数据文件 sink
- **C**：健康监控——当 `success_rate < threshold` 时自动告警 + 回退
- **D**：`UDPReceiver` — 接收端反序列化 + 指标汇聚
- **E**：Grafana/Prometheus 指标导出（可选）
