# Milestone B1.1 — TargetState Contract Tightening (Hotfix)

> **日期**：2026-07-19
> **范围**：修复 B1 审计报告中的 P1 问题
> **修改文件**：`visioncore/state/target_state.py` + `tests/test_target_state.py`
> **状态**：已完成，36/36 测试通过，261 项回归零破坏

---

## 1. 背景

B1 审计报告（`docs/targetstate-audit-report.md`）在审计项 7（`to_dict`/`from_dict` 对称性）中标记了一项 P1 问题：

> `from_dict` 的 metadata 防御性拷贝守卫为：
> ```python
> if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
>     kwargs["metadata"] = dict(kwargs["metadata"])
> ```
> 当传入 `metadata=None` 或 `metadata="string"` 时，`isinstance(..., dict)` 为 `False`，跳过拷贝，原值直接传给构造器。由于 dataclass 构造器不做类型校验，最终 `ts.metadata` 是 `None` 或字符串——**违反 `dict[str, Any]` 类型注解**。

B1.1 通过在 `from_dict` 中显式校验 metadata 类型来收紧契约。

---

## 2. 修改内容

### 2.1 `visioncore/state/target_state.py` — `from_dict()` 方法

**修改位置**：`from_dict` 方法的 metadata 处理块（约 50 行）。

#### 修改前

```python
# Defensive copy of metadata: if the caller mutates the dict they
# passed in, the snapshot must not change. A shallow copy is
# sufficient -- nested mutation is the caller's responsibility.
if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
    kwargs["metadata"] = dict(kwargs["metadata"])
elif "metadata" not in kwargs:
    # Let the dataclass default_factory handle the missing case.
    pass
```

#### 修改后

```python
# Metadata type contract enforcement (B1.1).
#   - absent  : leave kwargs untouched -> dataclass default_factory
#               supplies a fresh empty dict.
#   - None    : reject. None is not a dict and would silently
#               violate the field's type annotation, breaking
#               downstream callers that expect .keys() / .items().
#   - non-dict: reject for the same reason.
#   - dict    : defensive shallow copy so the caller's dict
#               cannot leak into the constructed snapshot.
if "metadata" in kwargs:
    md_value = kwargs["metadata"]
    if md_value is None:
        raise TypeError(
            "from_dict: 'metadata' must be a dict, got None. "
            "Omit the key to use the default empty dict, or pass "
            "an empty dict explicitly."
        )
    if not isinstance(md_value, dict):
        raise TypeError(
            f"from_dict: 'metadata' must be a dict, got "
            f"{type(md_value).__name__}."
        )
    kwargs["metadata"] = dict(md_value)
# else: metadata absent -> default_factory handles it.
```

同时更新了 `from_dict` 的 docstring，新增 4 行类型的 ASCII 表格，明确每种情况的行为，并在 `Raises:` 段补充说明 metadata 类型违规也抛 `TypeError`。

### 2.2 新增的 4 项契约规则

| # | `metadata` 在 `data` 中的状态 | 行为 | 实现分支 |
|---|---|---|---|
| 1 | key 缺失 | ✅ **允许** — `default_factory` 供应新 `{}` | `if "metadata" not in kwargs` 走 else 分支 |
| 2 | `None` | ❌ **拒绝** — 抛 `TypeError`（错误信息含 "None" + 修复提示） | `if md_value is None: raise TypeError(...)` |
| 3 | 非 dict（str/list/int/tuple/set/float/object） | ❌ **拒绝** — 抛 `TypeError`（错误信息含实际类型名） | `if not isinstance(md_value, dict): raise TypeError(...)` |
| 4 | `dict` | ✅ **允许** — 防御性浅拷贝 | `kwargs["metadata"] = dict(md_value)` |

### 2.3 `tests/test_target_state.py` — 新增 4 项测试

| 测试函数 | 覆盖点 |
|---|---|
| `test_from_dict_reject_none_metadata` | 规则 2 — `metadata=None` 抛 TypeError，错误信息含 "metadata" + "None" |
| `test_from_dict_reject_non_dict_metadata` | 规则 3 — 7 种非 dict 类型（str/list/int/float/tuple/set/object）全部抛 TypeError，错误信息含 "metadata" + "dict" + 实际类型名 |
| `test_copy_with_chain` | 4 步链式 `copy_with` 累积覆盖（运动学传播场景），验证每步中间快照不被后续步骤污染 + metadata 在链中独立 |
| `test_dataclass_equality` | frozen dataclass 自动生成的 `__eq__` 语义：相同字段值相等、任一字段不同则不等、metadata 内容相等但引用不同仍相等、None 可选字段相等 |

---

## 3. 验证结果

### 3.1 新增测试

```
=== tests/test_target_state.py ===
  [PASS] test_from_dict_reject_none_metadata
  [PASS] test_from_dict_reject_non_dict_metadata
  [PASS] test_copy_with_chain
  [PASS] test_dataclass_equality
  ... (32 项原有测试全部保持通过)
=== 36/36 passed, 0 failed ===
```

### 3.2 回归测试

```
test_core_models.py            : 31/31 passed
test_event_bus.py              : 77/77 passed
test_shadow_integration.py     : 12/12 passed
test_target_manager.py         : 76/76 passed
test_target_manager_events.py  : 29/29 passed
test_target_state.py           : 36/36 passed   (+4 新增)
─────────────────────────────────────────────────
总计                           : 261/261 passed, 0 回归
```

### 3.3 Runtime 契约验证

直接运行 4 条规则的场景脚本：

```
=== B1.1 Hotfix: from_dict metadata contract ===
  Rule 1 (missing)      : metadata={}  (expect {})
  Rule 2 (None)         : TypeError raised, msg contains "None": True
  Rule 3 (all 7 types)  : TypeError raised for every non-dict, msg names actual type
  Rule 4 (dict)         : defensive copy works, snapshot intact: True
  Round-trip equality   : True
```

✅ 全部 4 条规则按预期工作。

---

## 4. 兼容性影响

### 4.1 向后兼容性分析

| 调用模式 | B1 行为 | B1.1 行为 | 兼容性 |
|---|---|---|---|
| `from_dict({"target_id":1, ...})`（无 metadata 键） | ✅ metadata={} | ✅ metadata={} | ✅ **完全兼容** |
| `from_dict({...,"metadata":{...}})`（dict） | ✅ 防御性拷贝 | ✅ 防御性拷贝 | ✅ **完全兼容** |
| `from_dict(to_dict(state))`（往返） | ✅ 相等 | ✅ 相等 | ✅ **完全兼容** |
| `from_dict({...,"metadata":None})` | ⚠️ 静默接受，metadata=None | ❌ 抛 TypeError | ⚠️ **破坏性变更** |
| `from_dict({...,"metadata":"str"})` | ⚠️ 静默接受，metadata="str" | ❌ 抛 TypeError | ⚠️ **破坏性变更** |

### 4.2 破坏性变更评估

**变更范围**：仅当调用方主动传入 `metadata=None` 或非 dict 值时才会触发新的 TypeError。

**实际影响**：

1. **正常生产路径**：零影响。所有合规生产者（InferWorker / TargetManager / 适配器）都通过 `to_dict()` 输出或直接构造 dict 传 metadata，不会传 None 或非 dict 值。
2. **测试代码**：零影响。`test_target_state.py` 原有 32 项测试无任何一项传 None 或非 dict metadata（已验证全部通过）。
3. **第三方/未来调用方**：如果有调用方依赖"B1 静默接受 None"的非契约行为，B1.1 会显式失败。**这是设计意图**——B1 的静默接受本就是审计发现的缺陷，显式失败优于静默污染。

**修复策略**：依赖此非契约行为的调用方应：
- 若想表达"无 metadata"：省略 metadata 键，或显式传 `{}`
- 若 metadata 来自不可信源：在调用 `from_dict` 前自行做 `or {}` 归一化

### 4.3 与既有 visioncore 模块的一致性

`visioncore.core.Event.payload` 字段也是 `dict[str, Any]`，且 `Event` 同样是 `frozen=True`。B1.1 的收紧使 `TargetState` 的 metadata 比 `Event.payload` **更严格**（Event 不做 from_dict 校验，因为 Event 没有 from_dict 方法）。这是 B1.1 的设计选择——`TargetState` 作为未来跨阶段、跨摄像头的数据流载体，对契约严格性的要求高于 `Event`。

### 4.4 未变更项

- ✅ 目录结构未变（`visioncore/state/` 仍只含 `__init__.py` + `target_state.py`）
- ✅ 架构未变（仍是 frozen dataclass，无协议/无业务逻辑/无 forbidden 依赖）
- ✅ `visioncore/__init__.py` 未变（顶层导出仍是枚举）
- ✅ `to_dict()` 未变
- ✅ `copy_with()` 未变
- ✅ `__repr__` 未变
- ✅ 14 个字段定义未变
- ✅ B1 审计报告中 P2/P3 项未在本 hotfix 范围内（P2 部分通过新增 4 测试覆盖；P3 父包 eager-load 留待 B2+）

---

## 5. B1 审计报告 P1 项关闭确认

| 审计报告原文 | B1.1 处理 | 状态 |
|---|---|---|
| "P1: `from_dict` 不校验 metadata 类型——`metadata=None` 或 `metadata="str"` 被静默接受，违反 `dict[str, Any]` 注解。建议 B2 之前修复（抛 TypeError 或归一化为 `{}`）。" | 在 `from_dict` 中新增显式类型校验，None 和非 dict 均抛 TypeError，dict 走防御性拷贝 | ✅ **已修复** |
| "P2: 补 3 个测试：非 dict metadata 抛错 / copy_with 链式 / 相等性语义" | 新增 4 项测试覆盖以上 3 项 + None metadata 单独测试 | ✅ **已补全** |

---

## 6. 后续行动项（B1.1 范围外）

| 优先级 | 项 | 责任里程碑 |
|---|---|---|
| P3 | 将 `visioncore/__init__.py` core 导入改 lazy（消除 numpy eager-load 副作用） | B2+ |
| P4 | 安装 `coverage.py` 做自动行覆盖率测量 | 项目级 |
| P5 | 考虑为 `confidence` 范围 `[0,1]` 加文档化测试（当前 docstring 明示"not validated"） | B2 可选 |

---

## 7. 文件清单

| 文件 | 变更 | 行数变化 |
|---|---|---|
| `visioncore/state/target_state.py` | 修改 `from_dict` 方法 + docstring | +约 30 行（含表格 + 校验逻辑） |
| `tests/test_target_state.py` | 新增 4 项测试 + 1 个新 section 注释 | +约 130 行 |
| `docs/targetstate-hotfix.md` | 新增（本文件） | — |

---

## 8. 结论

B1.1 hotfix 通过显式类型校验收紧了 `from_dict` 的 metadata 契约，消除了 B1 审计报告中的 P1 问题。修改局限在 `from_dict` 单方法 + 4 项新测试，零架构变更、零目录结构变更、零回归。破坏性变更仅影响依赖"B1 静默接受 None/非 dict metadata"非契约行为的调用方——而项目内不存在此类调用方。

**P1 关闭，可进入 B2。**
