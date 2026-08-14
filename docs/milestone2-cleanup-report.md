# Milestone 2 审计低风险问题清理报告

> 日期：2026-07-16  
> 范围：VisionCore Alpha 重构 — 清理 Milestone 2 审计发现的 4 项低风险问题  
> 约束：禁止修改业务逻辑、禁止改变 TargetManager 行为

---

## 一、审计问题 → 修复对照

| # | 审计问题 | 等级 | 修复方案 | 状态 |
|---|----------|------|----------|------|
| 1 | `__init__.py` docstring 过时（声称 "not connected to InferWorker"） | 低 | 重写 docstring，描述 shadow bypass 集成现状 + 多 slot 隔离 + 新增导出 | ✅ |
| 2 | ID 格式不统一（converter `T-5` vs manager `T-0005`） | 低 | 新增 `target_id_factory.py`，统一 `create_target_id()` 入口 | ✅ |
| 3 | InferWorker bypass 未复用 converters | 低 | bypass 改为调用 `detections_to_tracks()` | ✅ |
| 4 | `tick()` 职责不明确 | 低 | 补充 docstring + TODO + EventBus 入口说明 | ✅ |

---

## 二、修改文件清单

| # | 文件 | 类型 | 说明 |
|---|------|------|------|
| 1 | `visioncore/target_manager/target_id_factory.py` | **新增** | 统一 ID 工厂 `create_target_id(slot_id, local_id) → "S{slot}-T{NNNN}"` |
| 2 | `visioncore/target_manager/converters.py` | **重写** | `track_to_target` 改用工厂；新增 `detection_to_track` / `detections_to_tracks`；docstring 对齐 |
| 3 | `visioncore/target_manager/manager.py` | **修改** | `_generate_id` 改用工厂；`tick()` 补充 docstring/TODO |
| 4 | `visioncore/target_manager/__init__.py` | **重写** | docstring 更新为已集成状态；导出 5 个新符号 |
| 5 | `ai/inference.py` | **修改** | bypass 改用 `detections_to_tracks`；移除手工构造；docstring 更新 |
| 6 | `tests/test_target_manager.py` | **修改** | 修正 2 项断言（padded 格式）+ 新增 6 项工厂/转换器测试 |

---

## 三、关键 Diff

### 3.1 新增 `target_id_factory.py`（核心）

```python
def create_target_id(slot_id: int, local_id: int) -> str:
    """生成 S{slot_id}-T{local_id:04d} 格式的统一目标 ID。"""
    return f"S{slot_id}-T{local_id:04d}"
```

**统一前**（两处独立生成，格式不同）：
```python
# converters.py — 无补零
tid = f"S{slot_id}-T{track.track_id}"        # → "S0-T7"

# manager.py — 4 位补零
tid = f"S{slot_id}-T{self._next_id:04d}"     # → "S0-T0007"
```

**统一后**（共用工厂，格式一致）：
```python
# converters.py
tid = create_target_id(slot_id, track.track_id)     # → "S0-T0007"

# manager.py
tid = create_target_id(slot_id, self._next_id)      # → "S0-T0007"
```

### 3.2 converters.py — 新增 dict→Track 转换器

```python
def detection_to_track(det: dict) -> Track | None:
    """检测 dict → Track，无 track_id 返回 None。"""
    tid = det.get("track_id")
    if tid is None:
        return None
    x1, y1, x2, y2 = float(det["x1"]), ...  # bbox 角点 → 中心+尺寸
    return Track(track_id=int(tid),
                 detection=Detection(bbox=BBox(x=cx, y=cy, w=w, h=h), ...))

def detections_to_tracks(detections: list[dict]) -> list[Track]:
    """批量转换，跳过无 track_id 的条目。"""
    return [t for t in (detection_to_track(d) for d in detections) if t is not None]
```

### 3.3 ai/inference.py — bypass 复用 converters

```python
# 修改前（手工构造 20 行）
tracks = []
for det in detections:
    tid = det.get("track_id")
    if tid is None: continue
    x1, y1, x2, y2 = ...
    core_det = CoreDetection(bbox=CoreBBox(...), ...)
    tracks.append(CoreTrack(track_id=int(tid), detection=core_det))

# 修改后（1 行）
tracks = detections_to_tracks(detections)
```

import 也从 3 个 VisionCore 类简化为 1 个函数：
```python
# 修改前
from visioncore.core.detection import BBox as CoreBBox
from visioncore.core.detection import Detection as CoreDetection
from visioncore.core.track import Track as CoreTrack
from visioncore.target_manager import TargetManager

# 修改后
from visioncore.target_manager import TargetManager, detections_to_tracks
```

### 3.4 manager.py — tick() docstring

```python
# 修改前
def tick(self, timestamp: float) -> None:
    """Periodic maintenance hook -- **skeleton only**.
    Logs a debug trace. Future phases will implement ...
    """

# 修改后（补充职责说明 + TODO + EventBus 入口）
def tick(self, timestamp: float) -> None:
    """Periodic maintenance hook.

    Called at a fixed cadence ... operates on **wall-clock time** and is
    responsible for housekeeping that does not depend on having fresh
    detections.

    Current behaviour (Milestone 2): no-op, debug log only.

    .. todo:: (Milestone 3 -- EventBus)
        * Stale sweep: iterate all targets across ALL slots ...
        * EventBus emission: target.lost / target.removed / target.recovered
        * Metrics snapshot: per-slot target counts for health dashboard
    """
```

### 3.5 `__init__.py` — docstring 更新

```python
# 修改前
"""...This package is a **skeleton**. It is fully functional in isolation
but is not connected to the existing InferWorker / tracker / GUI pipeline.
Future phases will wire the manager into the vision system..."""

# 修改后
"""...Integration status (Milestone 2):
    The package is wired into the live pipeline via a **shadow bypass**
    in InferWorker. After each frame's detection_ready signal,
    _bypass_update_targets converts detection dicts to Track objects
    (via detections_to_tracks) and feeds them to update_targets with
    the originating slot_id...
"""
```

导出从 5 个扩展到 8 个：
```python
__all__ = [
    "TargetStore", "TargetManager", "TargetLifecycleManager",
    "track_to_target", "tracks_to_targets",
    "detection_to_track", "detections_to_tracks",  # 新增
    "create_target_id",                             # 新增
]
```

---

## 四、风险分析

### 4.1 ID 格式变更影响

**变更**：`track_to_target` 输出从 `S0-T7`（无补零）→ `S0-T0007`（4 位补零）。

| 影响面 | 评估 | 说明 |
|--------|------|------|
| TargetManager 行为 | ✅ 无变化 | `create_target` 仍用单调计数器，输出 `S0-T0001` 不变 |
| 现有 61 项测试 | ✅ 全通过 | 这些测试走 manager 路径，格式本就是 padded |
| 新增 9 项 multi-slot 测试 | ✅ 2 项断言已修正 | `S0-T5` → `S0-T0005`，`S2-T1` → `S2-T0001` |
| doctest | ✅ 已同步 | converters.py 4 个示例更新 |
| InferWorker bypass | ✅ 无影响 | bypass 走 `update_targets` → `create_target`（manager 路径），不经过 converter |
| GUI | ✅ 无影响 | CameraCell 不显示 target_id |
| Tracker | ✅ 无影响 | 未修改 |

### 4.2 bypass 改用 converters 的影响

**变更**：bypass 从手工构造 `CoreTrack`/`CoreDetection`/`CoreBBox` 改为调用 `detections_to_tracks`。

| 风险 | 评估 |
|------|------|
| 行为等价性 | ✅ `detections_to_tracks` 的转换逻辑与原手工代码完全一致（bbox 角点→中心+尺寸、跳过无 track_id 的条目） |
| 异常处理 | ✅ 保留 `try/except Exception: pass`，旁路失败仍不影响推理 |
| 性能 | ✅ 无额外开销——函数调用 vs 内联循环，差异可忽略 |
| 依赖方向 | ✅ `ai.inference` → `visioncore.target_manager`（原有方向，未新增反向依赖） |

### 4.3 未改变项（约束遵守验证）

| 约束 | 验证 |
|------|------|
| 禁止修改业务逻辑 | ✅ 匹配/恢复/过期清理逻辑零改动；`update_targets` 算法不变 |
| 禁止改变 TargetManager 行为 | ✅ 公共 API 签名不变；`create_target`/`update_targets`/`mark_lost` 等语义不变；仅 `_generate_id` 内部改为调用工厂（输出格式不变） |
| 保持 GUI 不变 | ✅ gui/ 目录零修改 |
| 保持 InferWorker 输出不变 | ✅ `detection_ready.emit(slot_id, detections)` 信号格式不变 |
| 保持 Tracker 不变 | ✅ ai/tracker.py 零修改 |

### 4.4 残留风险

| 风险 | 等级 | 说明 |
|------|------|------|
| converter 与 manager 的 local_id 来源不同 | 低 | converter 用 `track_id`，manager 用单调计数器。两者通过 `(slot_id, track_id)` 元组键隔离，不依赖 target_id 字符串匹配，故无运行时风险。仅文档层面需注意区分。 |
| `detections_to_tracks` 未验证 dict 字段完整性 | 低 | 缺失字段用默认值（0.0/0/""），不抛异常。bypass 的 try/except 兜底。M3 可加 schema 校验。 |
| `tick()` 仍为 no-op | 低 | 按设计——M3 接入 EventBus 后实现。docstring 已明确 TODO。 |

---

## 五、验证结果

```
tests/test_target_manager.py  : 76/76 passed  (70 原有 + 6 新增)
tests/test_core_models.py     : 31/31 passed
target_id_factory doctests    :  4/4  passed
converters doctests           : 24/24 passed
target doctests               :  5/5  passed
InferWorker 导入              : OK
端到端验证                     : detections_to_tracks → update_targets → S0-T0001 / S1-T0002 共存
工厂统一性                    : create_target_id(0,1) == track_to_target(tid=1).target_id == "S0-T0001"
```
