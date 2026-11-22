# AGENTS.md — 仓库级 Agent 指南

本仓库由人类与 AI Agent 共同维护。开始任何任务前：

1. **先读 `PROJECT_BIBLE.md`**（权威白皮书），尤其：
   - §16 Agent 接入指南（关键文件索引、快速理解路径、任务切入点、设计决策）；
   - §17 文档沉淀规范（本文件第 2 条的完整定义）。

2. **内容向改动必须沉淀文档**（合并硬条件，CI `pr-docs-check` 拦截）：
   任何功能 / 行为 / 修复类改动，在 `docs/` 新增或更新设计文档
   （版本/范围/状态头 + 目标背景 + 契约表 + 维护指南），并同步 `PROJECT_BIBLE.md`。

3. **跟踪器是双实现**：`ai/tracker.py`（Python 参考）与 `rust/omintrack/`（Rust 性能）
   行为锁定等价。改任一侧前必读 `docs/rust-tracker-core.md` §3 契约；
   改完两侧同步 + `pytest tests/test_tracker_rust_equivalence.py`，
   行为变更另需 `pytest tests/test_tracker_snapshot.py --update-golden`。

4. **测试**：全套件 `pytest tests/` 必须全绿（当前 1341 项）；
   守护类用例经 `tests/_runtime_checks.py` 在子进程检查 import 副作用，勿改回进程内断言。

5. **环境**：Windows + Python 3.11（`venv/`）；
   Rust 扩展重建 `venv\Scripts\python.exe -m maturin develop --manifest-path rust\omintrack\Cargo.toml --release`（改过 `rust/` 后必须执行）；
   GUI 依赖 PyQt6；采集与 DirectML 仅 Windows。

6. **提交规范**：`<type>(<scope>): <summary>`（feat / fix / docs / test / chore），
   内容向提交在 body 列出契约要点。
