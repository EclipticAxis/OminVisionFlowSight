# FrameSource 审计报告 (Milestone C2)

> **审计目标**：`visioncore/source/`
> **审计日期**：2026-07-25
> **审计范围**：C2 交付的 4 个实现文件 + 1 个测试文件
> **审计方法**：静态导入分析（grep + AST 解析）+ 运行时 `sys.modules` 追踪 + 返回类型内省 + 回归测试
> **最终评分**：**PASS**（6 项审计点全部通过）

---

## 0. 执行摘要

Milestone C2 的 `visioncore/source/` 包是一个**纯契约框架**：抽象基类 `base.py` 运行时仅依赖 Python 标准库（logging / abc / typing），对 `camera/`、OpenCV（cv2）、PyQt、`ai/` **零依赖**。`read()` 的返回类型经内省与测试双重验证为 `Frame | None`——返回 `visioncore.core.Frame` 统一信封，**绝不**返回裸 `numpy.ndarray` 或 OpenCV `Mat`。`DummyFrameSource` 在测试中出现 56 次，四大必测领域（生命周期 / read / close / 异常恢复）全覆盖。无循环依赖，无生产代码引用该包，现有 VDP 采集逻辑完全不受影响。

6 项审计点全部 **PASS**。唯一的非阻塞观察：导入 `visioncore.source` 会经 `visioncore/__init__.py → visioncore.core.frame` 与 `dummy_source.py → core.frame` 间接加载 numpy——这是 visioncore 顶层包预有的数据模型基础属性（`Frame` 持有 `np.ndarray`），非 C2 引入，亦非视觉/业务耦合。值得注意的是，**`base.py` 单独导入时自身零 numpy 依赖**（AST 验证 `Frame` 在 `TYPE_CHECKING` 守卫下），numpy 加载源于顶层包初始化与具体实现 `dummy_source.py`。

---

## 1. 审计范围

### 1.1 受审文件

| 文件 | 职责 |
|---|---|
| `visioncore/source/__init__.py` | 包导出（7 名）+ 范围/返回类型政策说明 |
| `visioncore/source/base.py` | `FrameSource(ABC)` + `FrameSourceError` + 上下文管理器 |
| `visioncore/source/frame_source.py` | 工厂 + 注册表（register/create/available/get_class） |
| `visioncore/source/dummy_source.py` | `DummyFrameSource`（测试用可配置源） |
| `tests/test_frame_source.py` | 55 项单元测试 |

### 1.2 审计准则（6 项）

1. 是否依赖 `camera/`
2. 是否依赖 OpenCV（cv2）
3. 是否依赖 PyQt
4. 是否依赖 `ai/`
5. 是否统一返回 `Frame`
6. DummySource 是否覆盖测试

---

## 2. 审计方法

| 方法 | 工具 | 验证内容 |
|---|---|---|
| 静态导入枚举 | `grep "^\s*(import\|from)"` | 列出包内全部 import 语句 |
| 业务耦合关键词扫描 | `grep -i "camera\|opencv\|cv2\|PyQt\|ai\.\|numpy\|onnx\|ultralytics\|mediapipe\|DirectShow\|av\b"` | 确认无代码级业务依赖（区分 docstring 散文与真实代码） |
| 反向引用扫描 | `grep "import\s+visioncore\.source\|from\s+visioncore\.source"` | 确认无生产代码引用该包 |
| AST 解析 | `ast.parse` + `if TYPE_CHECKING` 守卫识别 | 权威区分运行时导入与类型检查专用导入 |
| 运行时 `sys.modules` 追踪 | `__import__` + `sys.modules` 快照 | 确认副作用加载范围、无循环导入 |
| 返回类型内省 | `inspect.signature(FrameSource.read)` | 验证 `read()` 返回注解为 `Frame \| None` |
| 测试断言扫描 | `grep "isinstance.*Frame\|not isinstance.*ndarray"` | 验证返回类型经测试守护 |
| 回归测试 | 既有套件 + C2 新测试 | 确认零破坏 |

---

## 3. 逐项审计发现

### 3.1 审计点 1 — 是否依赖 `camera/` ✅ PASS

**准则**：包内不得 `import camera.*` 或在运行时引用 `camera` 模块。

**证据**：
- 静态导入枚举：`visioncore/source/` 内**零** `import camera` 或 `from camera` 语句。
- 关键词扫描：`camera` 的全部匹配**位于 docstring 散文中**——
  - `base.py:4` "source (camera, video file, RTSP stream, ...)" 描述源的种类
  - `base.py:21` "any camera, file, or network I/O (that is C3+)" 声明不包含
  - `base.py:24-25` "no modification to ai/, camera/, or gui/" 声明不修改
  - `frame_source.py:18` "camera -> CameraFrameSource (C3, wraps camera/ backends)" 描述未来规划
  - `dummy_source.py:6` "no camera, no file, no network" 声明无 I/O
- 运行时 `sys.modules`：`camera` **不在** `sys.modules` 中。

**结论**：零 `camera/` 依赖。

---

### 3.2 审计点 2 — 是否依赖 OpenCV（cv2）✅ PASS

**准则**：包内不得 `import cv2` 或引用 OpenCV。

**证据**：
- 静态导入枚举：`visioncore/source/` 内**零** `import cv2` 语句。
- 关键词扫描：`opencv`/`cv2` 的全部匹配**位于 docstring 散文中**，且均为**否定声明**——
  - `base.py:22` "any numpy or OpenCV capture code -- concrete sources implement capture"
  - `base.py:29-30` "never a bare numpy.ndarray, never an OpenCV Mat"
  - `base.py:218` "never an OpenCV Mat"
  - `__init__.py:22` "any numpy or OpenCV capture code in the base class"
  - `__init__.py:30` "never an OpenCV Mat"
  - `frame_source.py:23` "imports its transport library (DirectShow / OpenCV / av / socket)" 描述未来真实源的传输库归属
- 运行时 `sys.modules`：`cv2` **不在** `sys.modules` 中。

**结论**：零 OpenCV/cv2 依赖。docstring 明确且反复声明"never OpenCV Mat"。

---

### 3.3 审计点 3 — 是否依赖 PyQt ✅ PASS

**准则**：包内不得 `import PyQt6`/`PyQt5` 或任何 GUI 框架。

**证据**：
- 静态导入枚举：`visioncore/source/` 内**零** PyQt 引用。
- 关键词扫描：`PyQt` **零匹配**（连 docstring 都未提及，仅 `__init__.py:24` 与 `base.py:24` 提及 `gui/` 作为"不修改"声明）。
- 运行时 `sys.modules`：`PyQt6`、`PyQt5` **均不在** `sys.modules` 中。

**结论**：零 PyQt/GUI 依赖。

---

### 3.4 审计点 4 — 是否依赖 `ai/` ✅ PASS

**准则**：包内不得 `import ai.*` 或在运行时引用 `ai` 模块。

**证据**：
- 静态导入枚举：`visioncore/source/` 内**零** `import ai` 或 `from ai` 语句。
- 关键词扫描：`ai\.` 的匹配**全部为 docstring 中的路径引用**（`ai/`、`ai.inference`），均为"不修改"声明——
  - `base.py:23-24` "no modification to ai/, camera/, or gui/"
  - `__init__.py:23-24` 同上
- 运行时 `sys.modules`：`ai` **不在** `sys.modules` 中。

**结论**：零 `ai/` 依赖。

---

### 3.5 审计点 5 — 是否统一返回 `Frame` ✅ PASS

**准则**：`FrameSource.read()` 返回 `Frame | None`，**禁止**裸 `numpy.ndarray`、**禁止** OpenCV `Mat`，统一返回 `visioncore.core.Frame`。

**证据**：

(a) **返回类型注解内省**（`inspect.signature`）：
```
FrameSource.read() return annotation: 'Frame | None'
```

(b) **测试守护**——`test_read_returns_frame_not_ndarray`（line 261）双重断言：
```python
f = src.read()
assert isinstance(f, Frame)        # the envelope
assert not isinstance(f, np.ndarray)  # never a bare array
assert isinstance(f.image, np.ndarray)  # internal field is ndarray (data model design)
```
该测试明确区分：`read()` 返回 Frame 信封（✓），不返回裸 ndarray（✓），但 Frame 的 `image` 字段内部持有 ndarray（数据模型设计，允许）。

(c) **多处 isinstance(f, Frame) 断言**：
- `test_read_returns_frame_when_open`（line 258）
- `test_context_manager_read_inside`（line 535）
- `test_end_to_end_source_feeds_consumer`（line 692）：`all(isinstance(f, Frame) for f in collected)`

(d) **DummyFrameSource.read() 实现**：返回 `self._frames[self._index]`（Frame 实例）或 `None`（耗尽时），从不返回裸数组。

(e) **政策文档化**：`base.py` 与 `__init__.py` 的 docstring 反复声明返回类型政策（"never a bare numpy.ndarray, never an OpenCV Mat"）。

**结论**：`read()` 统一返回 `Frame | None`，经注解、实现、测试三重验证。裸 ndarray / OpenCV Mat 被明确禁止并有测试守护。

---

### 3.6 审计点 6 — DummySource 是否覆盖测试 ✅ PASS

**准则**：DummyFrameSource 应支撑四大必测领域（生命周期 / read / close / 异常恢复）的测试。

**证据**：

(a) **使用频度**：`DummyFrameSource` 在 `tests/test_frame_source.py` 中出现 **56 次**，是测试的主力源。

(b) **四大领域覆盖映射**：

| 领域 | 测试函数 | 验证 |
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
| | `test_read_increments_counter` | read_count 递增 |
| | `test_read_returns_none_when_exhausted_no_loop` | loop=False 耗尽返回 None |
| | `test_read_loops_through_multiple_frames` | 多帧循环 |
| | `test_read_replays_explicit_frames_in_order` | 显式帧顺序回放 |
| **close()** | `test_close_never_raises_even_after_failure` | 故障后 close 仍成功 |
| | `test_close_after_close_does_not_revive_source` | 二次 close 不复活源 |
| | `test_close_resets_health` | close 置健康 False |
| **异常恢复** | `test_read_on_closed_source_raises_runtime_error` | 关源 read 抛 RuntimeError |
| | `test_raise_on_read_propagates_frame_source_error` | raise_on_read 传播 FrameSourceError |
| | `test_raise_on_read_propagates_arbitrary_exception` | 任意异常类型传播 |
| | `test_recovery_via_close_and_open` | close+open 恢复后 read 成功 |
| | `test_raise_on_read_persists_until_cleared` | 故障持续到清除 |
| | `test_read_failure_does_not_corrupt_frame_buffer` | 失败 read 不丢帧/不乱序 |

(c) **补充覆盖**：健康检查（3 项）、上下文管理器（4 项）、工厂/注册表（9 项）、reset/杂项（3 项）、端到端（1 项）。

(d) **配置维度覆盖**：`frames`、`source_id`、`loop`、`raise_on_read`、计数器、`reset()` 全部有对应测试。

**结论**：DummyFrameSource 全面支撑四大领域测试，56 次使用，配置维度完整覆盖。

---

## 4. 附加审计（循环依赖与 base.py 纯度）

### 4.1 循环依赖 ✅ PASS

**证据**：

(a) **包内 DAG**（运行时导入边）：
```
__init__.py ──> base.py            (base 无运行时 visioncore 依赖)
           ──> dummy_source.py ──> base.py + core.frame
           ──> frame_source.py  ──> base.py + dummy_source.py
```
`base.py` 是叶子（运行时零 visioncore 依赖）；`dummy_source.py` 依赖 base + core.frame；`frame_source.py` 依赖 base + dummy_source；`__init__.py` 聚合全部。**无环**。

(b) **独立导入测试**：逐个 `__import__` 四个子模块，全部成功无 `ImportError`/`CircularImport`：
```
OK import visioncore.source.base
OK import visioncore.source.dummy_source
OK import visioncore.source.frame_source
OK import visioncore.source
```

(c) **外部反向引用**：`visioncore.source` 仅被 `tests/test_frame_source.py` 与包自身引用，**无生产代码**（ai/gui/camera/main.py/其他 visioncore 子包）导入，故不可能形成外部环。

### 4.2 base.py 纯度（AST 权威验证）✅ PASS

**AST 解析**区分运行时导入与 `TYPE_CHECKING` 守卫：

```
base.py runtime imports (NOT under TYPE_CHECKING):
  line 84: from __future__
  line 86: import logging
  line 87: from abc
  line 88: from typing

base.py TYPE_CHECKING-only imports (NOT runtime):
  line 91: from visioncore.core.frame
```

`base.py` 运行时**仅**依赖标准库（`__future__` / `logging` / `abc` / `typing`）。`visioncore.core.frame.Frame` 在 `if TYPE_CHECKING:` 守卫下（line 91），配合 `from __future__ import annotations`（PEP 563，注解为字符串），类型检查器专用，运行时不执行。

**结论**：`base.py` 是一个纯契约，运行时零 visioncore 依赖、零 numpy 依赖。这是源层最干净的部分——抽象基类不拉起任何数据模型或传输库。

### 4.3 FrameSource 抽象性 ✅ PASS

```
FrameSource()
→ TypeError: Can't instantiate abstract class FrameSource
  with abstract methods close, health_check, is_open, open, read
```

5 个方法（open / is_open / read / close / health_check）全部 `@abstractmethod`，ABC 机制经运行时验证生效。

### 4.4 副作用加载范围

导入 `visioncore.source` 后 `sys.modules` 中的 `visioncore.*`（11 个模块）：
```
visioncore.core / .adapters / .detection / .event / .frame / .target / .track
visioncore.source / .base / .dummy_source / .frame_source
```

**未加载**：`visioncore.state`、`visioncore.target_manager`、`visioncore.eventbus`、`visioncore.protocol`、`visioncore.pipeline`——source 包不拉起这些较重子系统。

---

## 5. 非阻塞观察（信息性，不影响评分）

### 5.1 numpy 间接加载

**现象**：导入 `visioncore.source` 后，`numpy` 出现在 `sys.modules` 中。甚至单独导入 `visioncore.source.base` 也会加载 numpy。

**根因（双重）**：
1. **顶层包初始化**：`visioncore/__init__.py` eager-export `from visioncore.core import Frame, ...`，而 `visioncore/core/frame.py` 定义 `Frame` 时 `image: np.ndarray` 需要 `import numpy`。这是 **visioncore 顶层包预有的数据模型基础属性**（B1 之前就存在），非 C2 引入。
2. **具体实现**：`dummy_source.py` 运行时 `from visioncore.core.frame import Frame`（构造 Frame 实例所需），传递性加载 numpy。

**关键区分**：
- **`base.py`（抽象基类）自身零 numpy 依赖**——AST 验证 `Frame` 在 `TYPE_CHECKING` 守卫下，运行时不导入。numpy 加载源于顶层包初始化，非 base.py 自身。
- **`dummy_source.py`（具体实现）**通过 `Frame` 传递性依赖 numpy，这是数据模型设计使然（Frame 持有 ndarray），非视觉/业务耦合。
- `DummyFrameSource` 在合成帧生成器 `_make_default_frame` 内**懒导入** numpy，模块其余部分不直接触碰。

**影响评估**：
- numpy 是数值计算基础库，不属于审计准则所针对的视觉/业务耦合（camera/cv2/PyQt/ai）。
- 不构成循环依赖。
- 不影响现有运行逻辑（VDP 本就依赖 numpy）。
- `read()` 返回类型政策（Frame 信封，非裸 ndarray）不受影响——numpy 是 Frame 内部实现细节。

**结论**：信息性观察，不触发任何审计准则的失败或警告。`base.py` 的纯度（AST 验证零运行时 visioncore/numpy 依赖）是源层设计的亮点。

---

## 6. 回归验证

审计期间重跑相关测试套件，确认 C2 零破坏：

```
test_frame_source.py (C2 新增) : 55/55 passed
test_pipeline.py (C1)          : 56/56 passed
test_protocol_base.py (B2)     : 36/36 passed
test_event_bus.py              : 77/77 passed
────────────────────────────────────────────────────────────────
（全量 17 套件在 C2 实现时已验证零回归，见 docs/frame-source.md §8.4）
```

主应用导入冒烟通过：`visioncore` 顶层包与 `visioncore.source` 均正常导入，`ai/`、`gui/`、`camera/` 未被触碰。

---

## 7. 评分汇总

| # | 审计点 | 评分 | 关键证据 |
|---|---|---|---|
| 1 | 依赖 `camera/` | **PASS** | 零 import；匹配全为 docstring；`camera` 不在 sys.modules |
| 2 | 依赖 OpenCV (cv2) | **PASS** | 零 import；docstring 反复声明 "never OpenCV Mat"；`cv2` 不在 sys.modules |
| 3 | 依赖 PyQt | **PASS** | 零引用（连 docstring 都无 PyQt）；`PyQt6/5` 不在 sys.modules |
| 4 | 依赖 `ai/` | **PASS** | 零 import；匹配为"不修改"声明；`ai` 不在 sys.modules |
| 5 | 统一返回 `Frame` | **PASS** | 注解 `Frame \| None`；测试双重断言 isinstance(Frame)+非 ndarray；政策文档化 |
| 6 | DummySource 覆盖测试 | **PASS** | 56 次使用；四大领域全覆盖；配置维度完整 |

### 附加审计

| 项 | 评分 | 关键证据 |
|---|---|---|
| 循环依赖 | **PASS** | 包内 DAG 无环；4 子模块独立导入全成功；无外部反向引用 |
| base.py 纯度 | **PASS** | AST 验证运行时仅 stdlib；`Frame` 在 TYPE_CHECKING 下 |
| FrameSource 抽象性 | **PASS** | 5 方法全 @abstractmethod；实例化触发 TypeError |

### 最终评分：**PASS**

6 项审计点 + 3 项附加审计全部通过。`visioncore/source/` 是一个纯净、独立、统一返回 `Frame` 的输入源抽象框架，满足 C2 规约的全部要求，对现有系统零侵入、零回归。

---

## 8. 建议（信息性，非阻塞）

1. **C3 真实源接入时复用本审计方法**：当 `CameraFrameSource`（包装 `camera/`）/`FileFrameSource`（用 cv2/av）引入时，应对每个真实源单独做依赖审计——真实源**应当**依赖其传输库（camera/cv2/av），但该耦合必须 confined 在源模块内，不泄漏到 `base.py`/`frame_source.py`。
2. **新增返回类型守护测试**：建议在 C3 中新增一个参数化测试，对所有注册的 FrameSource 子类运行 `test_read_returns_frame_not_ndarray`，防止未来真实源意外返回裸数组。
3. **考虑 `visioncore/__init__.py` 懒加载**（同 C1 审计建议）：当前顶层包 eager-export `Frame` 导致 numpy 成为任何 `visioncore.*` 导入的副作用。若希望 `base.py` 可真正独立加载（不拉 numpy），需重构顶层包为懒加载。此为顶层包设计议题，超出 C2 范围。

---

## 9. 文件清单

| 文件 | 说明 |
|---|---|
| `visioncore/source/__init__.py` | 包导出（7 名）+ 返回类型政策说明 |
| `visioncore/source/base.py` | `FrameSource(ABC)` + `FrameSourceError`（AST 验证运行时零 visioncore 依赖） |
| `visioncore/source/frame_source.py` | 工厂 + 注册表 |
| `visioncore/source/dummy_source.py` | `DummyFrameSource`（懒导入 numpy） |
| `tests/test_frame_source.py` | 55 项测试（含四大领域 + 返回类型守护 + 工厂） |
| `docs/frame-source.md` | 设计文档 |
| `docs/frame-source-audit.md` | 本审计报告 |
