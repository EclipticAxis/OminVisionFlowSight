# Markdown 代码围栏配对检查（CommonMark §4.5）

> **版本**：1.0 — 2026-11-26
> **范围**：`tools/check_markdown_fences.py`（新增）、`tests/test_markdown_fences.py`（新增 17 项）、`.github/workflows/markdown-fences.yml`（新增）、`README.md`（围栏修复 + 计数/索引同步）、`PROJECT_BIBLE.md` / `AGENTS.md`（计数同步）
> **状态**：待合并（PR #20）；新增 17 项测试，全套件 1446 项通过。合并后须按 §17.3 回改本状态头与 §11.2 行

---

## 1. 目标与背景

PR #17（`71fb92c`）将 README 的 4 处 ` ```bash ` 起始围栏误改为四反引号（````bash），
闭合围栏仍为三反引号。按 CommonMark 规范，闭合围栏反引号数必须 ≥ 开启围栏，
导致自第 87 行起代码块**永不闭合**，README 后半部分（配置说明、项目结构、测试、
文档、许可证、贡献指南、致谢，约占全文 60%）在 GitHub 上全部渲染为等宽代码文本。

该缺陷从引入（11-23）到发现（11-26，PR #20）**跨越 3 个 PR**，原因是它只影响渲染：

- 不影响任何 pytest 用例（无测试读取 README）；
- 不在 pr-docs-check 的覆盖面内（该检查只管"文档是否沉淀"，不管 Markdown 结构）；
- 现有 Markdown linter（markdownlint / remark-lint）**不会**报未闭合围栏——
  CommonMark 允许 EOF 隐式闭合未关闭的 fence，规范层面它不是语法错误。

本仓库是 Markdown 重度用户（docs/ 沉淀契约 + PROJECT_BIBLE + AGENTS.md），
且由人类与 AI Agent 共同维护——围栏反引号数漂移是 LLM 改写文档的高频错误模式。
因此将"围栏配对"固化为机器检查：同一脚本服务本地守护与 CI 硬闸门。

## 2. 架构

```
tools/check_markdown_fences.py   # 唯一实现：扫描 + 判定 + CLI
        │
        ├─ tests/test_markdown_fences.py  # 守护用例：全仓 .md 平衡（本地全套件内）
        │                                  # + 15 项规则单测 + PR #17 回归
        └─ .github/workflows/markdown-fences.yml  # CI 硬闸门（ubuntu + Python 3.11）
```

三层共用同一份判定逻辑，无双实现漂移风险（对比：跟踪器 Python/Rust 双实现
需要等价性对拍，本检查刻意单实现）。

## 3. 契约表

### 3.1 判定规则（CommonMark §4.5 + 仓库级推论）

| # | 规则 | 说明 |
|---|------|------|
| 1 | 闭合围栏字符类型须与开启一致 | ` 对 `，`~` 对 `~`，交叉不闭合 |
| 2 | 闭合围栏长度 ≥ 开启围栏长度 | ```` `````` ```` 开启后 ` ``` ` 不闭合 |
| 3 | 闭合围栏不得携带 info string | 围栏序列之后仅允许空白；` ```bash ` 永远不能当闭合行 |
| 4 | EOF 时仍有开启围栏 → **违规** | CommonMark 允许 EOF 隐式闭合，但本仓库文档中它几乎必然是笔误，故提升为错误 |

辅助判定（与 CommonMark 对齐）：

- 围栏行 = 至多 3 空格缩进 + ≥3 个同种字符；4 空格缩进的围栏行是缩进代码块，不参与判定；
- 反引号开启围栏的 info string 不得含反引号（否则该行根本不是围栏行）；`~` 围栏无此限制；
- 合法嵌套：用 4 反引号包裹含 3 反引号示例的文档写法**不报错**（只守"最终必须闭合"底线，
  不对开启围栏内部的短围栏行报警——避免对合法嵌套误报）。

### 3.2 检查范围与输入

| 项 | 契约 |
|---|---|
| 默认输入 | `git ls-files -z -- "*.md"`（尊重 .gitignore，不含 venv/ 等未跟踪文件） |
| 非 ASCII 路径 | `-c core.quotepath=false` + `-z`，中文路径不做八进制转义 |
| 文件编码 | UTF-8 读取（容忍 BOM，`utf-8-sig`） |
| CLI 输出 | stdout/stderr 强制 UTF-8（Windows GBK 控制台防乱码） |
| 退出码 | 0 = 全部平衡；1 = 存在未闭合围栏或读取失败 |
| 报错格式 | 指向**开启行**，并列出"疑似闭合失败行 + 原因"（长度不足 / 字符不同 / 携带 info string） |

### 3.3 CI 触发契约

| 项 | 契约 |
|---|---|
| 事件 | `pull_request` + `push`（main） |
| paths 过滤 | `**.md`、`tools/check_markdown_fences.py`、`tests/test_markdown_fences.py`、workflow 自身 |
| 运行环境 | ubuntu-latest + Python 3.11（脚本零三方依赖，stdlib only） |

### 3.4 v1 明确不做

- 不解析 blockquote / list 嵌套前缀（`> ``` ` 行不参与围栏判定）——仓库现有文档无此用法；
  未来若需要，先扩展解析再考虑放行，不得直接关闭检查；
- 不做完整 Markdown lint（标题层级、空行等），守住围栏配对一条底线；
- 历史文档的状态头计数不回改（如 prediction-output-gating.md 的"1429 项"为其合并时事实）。

## 4. 维护指南

- **本地运行**：`python tools/check_markdown_fences.py`（检查全仓）；`python tools/check_markdown_fences.py a.md`（指定文件）。
- **新增 .md 文档**：无需任何登记，git 跟踪即被守护覆盖。
- **检查失败排查**：报错第一行指向未闭合的开启围栏；其下"疑似闭合失败"列表给出每个
  候选行的失败原因，按 CommonMark §4.5 三规则修正开启或闭合行即可。
- **合法嵌套被误报时**：确认外层围栏长度严格大于内层（4 包 3），且外层闭合行不带 info string；
  仍误报则说明命中 v1 未覆盖的解析场景（如 blockquote 嵌套），应扩展检查器而非豁免文件。
- **修改判定规则**：必须同步 `tests/test_markdown_fences.py` 单测与本文件 §3.1，
  并保证 `test_all_tracked_markdown_files_have_balanced_fences` 在全套件中通过。