## 变更类型

- [ ] 内容向（新功能 / 行为变更 / 缺陷修复）—— **必须完成下方「文档沉淀」**
- [ ] 非内容向（纯格式化 / 注释 / 无行为变化的机械重构）—— 在下方说明豁免原因

## 文档沉淀（内容向 PR 必填）

> 本仓库约定：内容向 PR 必须把**隐性契约显性化**，按 `PROJECT_BIBLE.md` §17
> 沉淀进文档体系（对人及 AI Agent 机器可读）。CI `pr-docs-check` 会强制检查。

- [ ] 新增或更新了 `docs/<topic>.md` 设计文档（版本/范围/状态头 + 目标背景 + 契约表 + 维护指南）
- [ ] 同步了 `PROJECT_BIBLE.md`（§11.2 特性表；如涉及架构决策/切入点/文件索引，同步 §16）
- [ ] 涉及跟踪器：已阅读 `docs/rust-tracker-core.md` §3 契约，Python / Rust **两侧同步**，`pytest tests/test_tracker_rust_equivalence.py` 通过，行为变更已 `--update-golden`
- [ ] 涉及控件样式：只改 `gui/theme.py::SQUIRCLE_ASSETS` 并已重跑 `tools/gen_squircle_assets.py`（禁手改 PNG）

> 确认无需文档时，请在此处写明豁免原因（CI 也接受 PR 描述中出现 `[docs-skip]`）：
>
>

## 测试

- [ ] `pytest tests/` 全绿（或已注明失败的既有用例）
- [ ] 新增行为附带或更新了测试

## 变更说明

<!-- 动机、实现方式、影响面 -->
