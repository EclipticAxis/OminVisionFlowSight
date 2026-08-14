# 多 Slot 隔离审计修复报告

> 日期：2026-07-16  
> 范围：VisionCore TargetManager 多摄像头 slot 身份隔离  
> 审计问题：多个摄像头 slot 共用一个 TargetManager，slot0 track_id=1 与 slot1 track_id=1 身份冲突

---

## 一、现状结论

**slot_id 隔离基础设施在 Milestone 2 骨架中已就位**，核心机制正确：

| 层 | 文件 | 已有机制 |
|----|------|----------|
| 数据模型 | `visioncore/core/target.py` | `Target.slot_id: int = 0` 字段（7 字段之一） |
| 转换器 | `visioncore/target_manager/converters.py` | `track_to_target(slot_id=…)` 生成 `S{slot_id}-T{track_id}` |
| 管理器 | `visioncore/target_manager/manager.py` | `create_target(slot_id=…)` 生成 `S{slot_id}-T{NNNN}`；`update_targets` 按 `(slot_id, track_id)` 二元组建索引、按 slot 过滤未匹配清理 |
| 推理集成 | `ai/inference.py` | `_bypass_update_targets(slot_id, …)` 调用 `update_targets(tracks, slot_id=slot_id, …)` |

**隔离原理**：`update_targets` 内部索引键为 `dict[tuple[int, int], Target]`，即 `(slot_id, track_id)`。slot0/track_id=1 → 键 `(0,1)`；slot1/track_id=1 → 键 `(1,1)`。两键不同，不会互相覆盖。未匹配清理阶段 `if target.slot_id != slot_id: continue` 保证 slot0 的更新不触碰 slot1 的目标。

本次修复的是审计发现的 **4 项遗留缺陷**（非重写）。

---

## 二、修改文件

| # | 文件 | 修改类型 | 说明 |
|---|------|----------|------|
| 1 | `ai/inference.py` | **Bug 修复** | 移除 `_bypass_update_targets` 中 `if tracks:` 守卫，空检测帧也调用 `update_targets` 触发过期清理；更新过时 docstring |
| 2 | `visioncore/target_manager/manager.py` | **死代码清理** | `_generate_id` 重构为接受 `slot_id` 参数、返回 `S{slot_id}-T{NNNN}`；`create_target` 改为调用它（消除内联重复） |
| 3 | `visioncore/target_manager/converters.py` | **文档修复** | 映射表/示例从旧 `T-{track_id}` 更新为 `S{slot_id}-T{track_id}`；补充 slot_id 字段说明 |
| 4 | `visioncore/core/target.py` | **文档修复** | 示例从 `target_id="T-001"` 更新为 `target_id="S0-T1"` + `slot_id=0` |
| 5 | `tests/test_target_manager.py` | **新增测试** | +9 项多 slot 隔离测试（第 12 节） |

---

## 三、关键 Diff

### 3.1 ai/inference.py — 核心修复（移除 if tracks 守卫）

```python
# 修改前（第 2004-2005 行）
            if tracks:
                self._target_mgr.update_targets(tracks, slot_id=slot_id, timestamp=timestamp)

# 修改后
            # 即使 tracks 为空也调用 update_targets，触发该 slot 的过期清理
            self._target_mgr.update_targets(tracks, slot_id=slot_id, timestamp=timestamp)
```

**docstring 同步更新**：从 "slot_id: 摄像头槽位 id（当前未用于目标区分，预留）" 改为说明 `(slot_id, track_id)` 二元组索引隔离机制 + 空帧仍触发清理的设计意图。

### 3.2 visioncore/target_manager/manager.py — 死代码清理

```python
# 修改前（_generate_id 从未被调用，返回旧格式）
    def _generate_id(self) -> str:
        tid: str = f"T-{self._next_id:04d}"
        self._next_id += 1
        return tid

# create_target 内联了自己的 id 生成（重复代码）
        with self._lock:
            tid: str = f"S{slot_id}-T{self._next_id:04d}"
            self._next_id += 1
            ...

# 修改后（统一入口）
    def _generate_id(self, slot_id: int = 0) -> str:
        tid: str = f"S{slot_id}-T{self._next_id:04d}"
        self._next_id += 1
        return tid

# create_target 改为调用
        with self._lock:
            tid: str = self._generate_id(slot_id)
            ...
```

### 3.3 visioncore/target_manager/converters.py — 文档对齐

```python
# 映射表
- ``track_id`` (int)    ``target_id`` (str, ``f"T-{track_id}"``)
+ ``track_id`` (int)    ``target_id`` (str, ``f"S{slot_id}-T{track_id}"``)

# 示例
- >>> target.target_id
- 'T-7'
+ >>> target.target_id
+ 'S0-T7'
+ >>> target.slot_id
+ 0

# 批量示例
- ['T-1', 'T-2']
+ ['S0-T1', 'S0-T2']
```

### 3.4 tests/test_target_manager.py — 新增 9 项测试（第 12 节）

```
test_multislot_same_track_id_no_collision        — 两 slot 同 track_id 不冲突
test_multislot_update_does_not_cross_contaminate  — slot0 更新不影响 slot1
test_multislot_stale_only_affects_own_slot        — 过期清理按 slot 隔离
test_multislot_empty_input_triggers_cleanup       — 空帧触发清理（bypass 修复回归）
test_multislot_recover_is_slot_scoped             — 恢复按 (slot,track) 匹配
test_multislot_full_scenario                      — 双 slot 并行全流程
test_track_to_target_slot_id_in_id                — converter 生成 S{slot}-T{track}
test_tracks_to_targets_slot_scoped                — 批量 converter slot 一致性
test_multislot_monotonic_id_global_unique         — 4 slot × 3 track 全局唯一
```

---

## 四、风险分析

### 4.1 已消除风险

| 风险 | 等级 | 说明 |
|------|------|------|
| 空检测帧不清理过期目标 | **高** | `if tracks:` 守卫导致 slot 断流时目标永久残留。已移除。这是本次最实质性修复。 |
| 死代码 `_generate_id` 误导 | 低 | 返回旧 `T-NNNN` 格式，未来维护者可能误用。已重构统一。 |
| 文档与实现不一致 | 低 | converters/target docstring 显示旧格式，可能误导调用方。已对齐。 |

### 4.2 设计权衡（非风险，需知悉）

**两种 target_id 生成策略共存**：
- `track_to_target`（无状态转换器）：`S{slot_id}-T{track_id}` — 直接嵌入 track_id，自描述，适合一次性转换。
- `TargetManager.create_target`（有状态管理器）：`S{slot_id}-T{NNNN}` — 单调计数器，跨 track 生命周期稳定（一个 target 可跨越多次 track 丢失/重建）。

两者**均通过 `(slot_id, track_id)` 元组键实现隔离**，不会冲突。这是有意设计，非缺陷：target_id 设计为"稳定且区别于 track_id"（见 Target docstring），因此有状态路径用单调计数器而非 track_id 直映。

### 4.3 残留风险（本次未改，属 Milestone 3 范围）

| 风险 | 等级 | 建议 |
|------|------|------|
| 单 TargetManager 实例无 slot 级容量隔离 | 低 | 当前 4 slot 共享一个 store，目标总数无上限。M3 可考虑 per-slot LRU 或容量上限。 |
| bypass 异常被静默吞掉 | 低 | `except Exception: pass` 保证不影响推理，但丢失诊断信息。M3 可加 `logging.debug` 记录。 |
| `update_targets` 每帧全量扫描 store | 低 | O(n) 扫描，当前目标数小无影响。目标数 >100 时可加 slot 索引。 |
| 无 ReID 跨 slot 关联 | 中 | 不同 slot 看到同一物理人仍生成不同 target。这是正确行为（slot 是物理隔离的摄像头），跨 slot 关联属 M3 ReID 范围。 |
| GUI 未展示 target_id | 低 | CameraCell 仍显示 track_id，未显示 S{slot}-T{NNNN}。符合"保持 GUI 不变"约束。 |

### 4.4 兼容性验证

- **GUI 不变** ✓ — CameraCell/MainWindow 未修改，仍消费 `detection_ready(int, list[dict])`
- **InferWorker 输出不变** ✓ — `detection_ready.emit(slot_id, detections)` 信号格式未变，bypass 在 emit 之后执行
- **Tracker 不变** ✓ — `ai/tracker.py` DetectionTracker 未触碰
- **现有测试全通过** ✓ — 70/70 + 31/31 + 16 doctests

---

## 五、新增测试建议（已实现 + 未来建议）

### 5.1 已实现（本次，9 项，全部 PASS）

见上文 3.4 节。

### 5.2 未来建议（Milestone 3）

1. **bypass 端到端测试**：构造 mock InferWorker，喂入多 slot 帧，验证 TargetManager 内部状态正确（当前仅单元测试 update_targets，未覆盖 bypass dict→Track 转换路径）。
2. **并发多 slot 压测**：4 slot 同时 `register_slot` + `submit_frame`，验证 TargetManager 线程安全在真实负载下不退化。
3. **slot 注销清理测试**：`unregister_slot` 后该 slot 的残留目标是否需要主动清理（当前 unregister 不清 TargetManager，可能残留）。
4. **target_id 解析工具**：若下游需要从 `S0-T0001` 反解 slot_id，建议加 `Target.slot_id` 字段已满足（无需解析字符串），但可加 `parse_target_id(s) -> (slot_id, seq)` 工具函数供调试。
5. **跨 slot ReID 回归测试**：M3 接入 ReID 后，验证同物理人跨 slot 仍为不同 target（除非显式合并）。

---

## 六、验证结果

```
tests/test_target_manager.py  : 70/70 passed  (61 原有 + 9 新增)
tests/test_core_models.py     : 31/31 passed
converters.py doctests        : 16/16 passed
target.py doctests            :  5/5  passed
InferWorker 导入              : OK
多 slot 隔离手工验证           : S0-T0001 / S1-T0002 共存，无冲突
```
