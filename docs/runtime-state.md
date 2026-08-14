# Runtime State Migration Report (Milestone D9.4)

> **日期**：2026-08-04
> **范围**：从 `ai/inference.py` 提取内部缓存/统计/计时状态到 `visioncore/runtime/state/`
> **状态**：已实现，41 项测试全通过 + 全量回归零破坏（35 个套件）

---

## 1. 分析：源代码定位

### 1.1 Legacy 代码（`ai/inference.py`）

| 字段 | 行号 | 类别 | 功能 |
|---|---|---|---|
| `_last_detections` | L362 | 缓存 | per-slot 上次检测结果 |
| `_last_submit_ts_by_slot` | L363 | 缓存 | per-slot 上次提交时间戳 |
| `_head_attr_cache` | L398 | 缓存 | per-track 头部属性缓存（stride 过期） |
| `_debug_infer_reports` | L311 | 统计 | 推理调试报告计数 |
| `_debug_submit_reports` | L312 | 统计 | 提交调试报告计数 |
| `_debug_emit_reports` | L313 | 统计 | 发射调试报告计数 |
| `_infer_cycle_count` | L321 | 统计 | 推理周期计数 |
| `_sr_attempt_count` | L314 | 统计 | 超分辨率尝试计数 |
| `_sr_refined_count` | L316 | 统计 | 超分辨率成功计数 |
| `_redetect_attempt_count` | L369 | 统计 | 重检测尝试计数 |
| `_redetect_success_count` | L370 | 统计 | 重检测成功计数 |
| `_redetect_failed_count` | L371 | 统计 | 重检测失败计数 |
| `_warmup_done` | L322 | 计时 | 预热完成标志 |
| `_startup_guard_until` | L323 | 计时 | 启动保护截止时间 |
| `_startup_guard_seconds` | L324 | 计时 | 启动保护持续时间 |

### 1.2 迁移的组件

```
RuntimeState ──── 主状态容器
  ├─ Stats ──── 命名统计计数器（increment/get/set/reset/snapshot）
  ├─ SlotCache ──── per-slot 缓存（detections + timestamps）
  ├─ HeadAttrCache ──── per-track 头部属性缓存（stride 过期）
  ├─ warmup_done: bool
  ├─ startup_guard_until: float
  └─ enable_startup_guard() / is_startup_guard_active()
```

---

## 2. 文件结构

```
visioncore/runtime/state/
├── __init__.py          # 包导出
└── runtime_state.py     # RuntimeState, Stats, SlotCache, HeadAttrCache

tests/test_runtime_state.py  # 41 项测试
docs/runtime-state.md        # 本文档
```

---

## 3. 接口

### 3.1 Stats（统计计数器）

```python
class Stats:
    def increment(self, name, amount=1): ...
    def get(self, name) -> int | float: ...    # 0 if unset
    def set(self, name, value): ...
    def reset(self, name=None): ...            # None = reset all
    def snapshot(self) -> dict: ...
```

### 3.2 SlotCache（per-slot 缓存）

```python
class SlotCache:
    def set_detections(self, slot_id, detections): ...
    def get_detections(self, slot_id) -> list[dict]: ...
    def set_timestamp(self, slot_id, timestamp): ...
    def get_timestamp(self, slot_id) -> float | None: ...
    def clear(self): ...
    def clear_slot(self, slot_id): ...
    @property
    def slot_ids(self) -> list[int]: ...
```

### 3.3 HeadAttrCache（per-track 缓存）

```python
class HeadAttrCache:
    def __init__(self, stride=2): ...
    def get(self, track_id, frame_id) -> dict | None: ...  # None if expired
    def set(self, track_id, frame_id, attrs): ...
    def clear(self): ...
    @property
    def size(self) -> int: ...
```

### 3.4 RuntimeState（主容器）

```python
class RuntimeState:
    def __init__(self, startup_guard_seconds=3.0, head_attr_stride=2): ...
    stats: Stats
    slot_cache: SlotCache
    head_attr_cache: HeadAttrCache
    warmup_done: bool
    startup_guard_until: float
    def enable_startup_guard(self, seconds=None): ...
    def is_startup_guard_active(self) -> bool: ...
    def mark_warmup_done(self): ...
    def mark_warmup_needed(self): ...
    def record_redetect_attempt/success/failure(self): ...
    @property
    def redetect_stats(self) -> dict: ...
    def record_infer_cycle(self): ...
    @property
    def infer_cycle_count(self) -> int: ...
    def reset(self): ...
```

### 3.5 Legacy 代码对应关系

| Legacy 代码 | RuntimeState 等价物 |
|---|---|
| `_last_detections` | `state.slot_cache.set_detections()` / `get_detections()` |
| `_last_submit_ts_by_slot` | `state.slot_cache.set_timestamp()` / `get_timestamp()` |
| `_head_attr_cache` | `state.head_attr_cache.set()` / `get()` |
| `_debug_infer/submit/emit_reports` | `state.stats.increment("debug_*_reports")` |
| `_infer_cycle_count` | `state.record_infer_cycle()` |
| `_redetect_*_count` | `state.record_redetect_*()` |
| `_sr_*_count` | `state.stats.increment("sr_*")` |
| `_warmup_done` | `state.warmup_done` |
| `_startup_guard_until` | `state.startup_guard_until` |
| `_clear_slot_caches()` | `state.slot_cache.clear()` |

---

## 4. 设计决策

### 4.1 为什么使用类而非 dataclass？

`Stats` 和 `SlotCache` 使用 `@dataclass`（纯数据），`HeadAttrCache` 使用手动 `__init__`（因为需要 stride 参数化）。`RuntimeState` 使用手动 `__init__` + `__slots__`（组合多个子组件）。

### 4.2 为什么 Stats 使用命名计数器而非固定字段？

Legacy 代码有 10+ 个不同计数器。使用 `Stats.increment("name")` 而非固定属性，使得：
- 新增计数器无需修改类定义
- 计数器名称由调用者决定（松耦合）
- `snapshot()` 一次性导出所有计数器

### 4.3 为什么不包含 `_health_monitor`？

`_HealthMonitor` 包含业务逻辑（`is_healthy` 判断、`trigger_fallback` 行为），不是纯状态。D9.4 仅迁移纯数据状态。

---

## 5. 测试覆盖（41 项）

| 分组 | 测试数 | 关键测试 |
|---|---|---|
| 模块表面 | 4 | 可导入性 |
| Stats | 9 | increment/get/set/reset/snapshot/float 值 |
| SlotCache | 8 | detections/timestamps/clear/clear_slot/slot_ids/multiple |
| HeadAttrCache | 6 | set/get/stride 过期/缺失 track/clear/size |
| RuntimeState | 12 | defaults/warmup/startup guard/redetect stats/infer stats/reset/integration |
| 无业务逻辑 | 1 | 源码无 NMS/检测函数 |
| 零依赖 | 2 | 无 banned 导入/无 torch |

---

## 6. 零侵入声明

新增 `visioncore/runtime/state/` 包和 `tests/test_runtime_state.py`。**未修改任何既有文件**。全量回归：35 个测试套件全部通过。
