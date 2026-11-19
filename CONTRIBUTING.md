# 贡献指南

欢迎通过 **Issue** 与 **Pull Request** 参与贡献！

## 流程

1. Fork 本仓库；
2. 基于 `main` 创建特性分支（如 `feat/your-feature`）；
3. 提交代码并推送到你的 Fork，提交信息清晰规范（`<type>(<scope>): <summary>`）；
4. 向本仓库发起 Pull Request，描述变更动机、实现方式与测试情况。

## 文档沉淀契约（内容向 PR 必须遵守）

本仓库约定：**任何内容向 PR（功能 / 行为变更 / 缺陷修复）都必须把隐性契约显性化**，
按 `PROJECT_BIBLE.md` §17 沉淀进文档体系：

1. 在 `docs/` 新增或更新设计文档（格式契约见 §17.3）；
2. 同步 `PROJECT_BIBLE.md`（§11 状态表，必要时 §16）；
3. PR 模板已内置检查清单，CI `pr-docs-check` 会强制执行（豁免需在 PR 描述声明）。

特殊域：

- **跟踪器**：Python（`ai/tracker.py`）与 Rust（`rust/omintrack/`）双实现锁定等价，
  两侧必须同步修改——先读 `docs/rust-tracker-core.md` §3；
- **控件样式**：只改 `gui/theme.py::SQUIRCLE_ASSETS` 并重跑 `tools/gen_squircle_assets.py`——
  见 `docs/squircle-capsule-widgets.md`。

## 测试

- 提交前确保 `pytest tests/` 全绿；
- 新增行为尽量附带或更新相应测试。

## 许可证

本项目采用 **PolyForm Noncommercial License 1.0.0**，你的贡献同样遵循非商业限制——
请勿将本软件或其衍生作品用于任何以盈利为目的的场景；为开发者提供的自愿赞助、
打赏或捐赠不受此限制（详见 `README.md` 许可证一节与 `LICENSE` 原文）。
