# 运行时元数据与启动脚本维护

> **版本**：1.1.0（2026-11-21）
> **范围**：`main.py`、`start.bat`、`README.md`、`PROJECT_BIBLE.md`
> **状态**：已合入 main（PR #5 → `7391b57`，2026-11-21）；全套件 1336 项通过（2026-11-21）
>
> 历史：1.0.0 已合入 main（PR #3 → `d2d8349`，2026-11-19；当时全套件 1279 项通过）

## 1. 目标与背景

1.0.0（2026-11-19）：应用版本、白皮书和启动说明存在不同步：`main.py` 仍报告 1.1.1，白皮书正文已升级为 1.2.0；`start.bat` 已使用自定位路径，但文档仍描述为硬编码绝对路径。该次只校正元数据和文档，不改变启动流程。

1.1.0（2026-11-21）：GitHub 新克隆后 README「启动主程序」路径整体失效，根因有三：

1. `venv/` 被 `.gitignore` 排除，新克隆（或下载 ZIP）双击 `start.bat` 必然失败（找不到 `venv\Scripts\python.exe`），而 README 把它描述为可直接运行，未写明前置条件；
2. README 徽章标注 `python 3.11+`，但 `mediapipe>=0.10.14,<0.10.15` 仅提供 cp39–cp312 预编译包，Python 3.13+ 用户在 `pip install -r requirements.txt` 阶段即失败，主程序无从启动；
3. `venv\Scripts\activate` 仅适配 CMD；PowerShell（Windows 11 默认终端）在执行策略受限时激活失败，随后的 `python main.py` 落到全局解释器，报 `ModuleNotFoundError`。

本次让 `start.bat` 在缺少仓内 venv 时打印提示并回退 PATH 上的 `python`，README 快速开始补齐 Python 版本窗口、分终端激活命令与免激活启动方式；正常启动流程不变。

## 2. 契约

| # | 契约 | 维护约束 |
|---|------|----------|
| C1 | `QApplication.applicationVersion` 的配置值为 1.2.0 | 与 `PROJECT_BIBLE.md` §11.1 同步；后续发布版本时同时修改 |
| C2 | `start.bat` 使用 `%~dp0` 定位项目目录和仓内 `venv` | 不得改回机器相关的绝对路径；移动项目目录无需编辑脚本 |
| C3 | 启动失败时才执行 `pause` | 保留非零退出码的控制台错误，不改变正常退出行为 |
| C4 | README 的测试数量与当前套件一致 | 增删测试后同步 README、`AGENTS.md` 和实际 pytest 结果 |
| C5 | 仓内 `venv` 缺失时，`start.bat` 打印提示并回退到 PATH 上的 `python` | 提示文本必须保持纯 ASCII——cmd 中文代码页下脚本内 UTF-8 中文会显示为乱码；C2/C3 不变 |
| C6 | README 快速开始标注 Python 3.11/3.12 版本窗口，提供 PowerShell / CMD 两种激活方式与免激活启动命令 | `mediapipe` 版本窗口放宽并验证前，不得宣称支持 3.13+ |

## 3. 维护指南

- 修改启动脚本后，从项目目录之外调用脚本，确认仍能定位仓内 Python；不要通过启动完整 GUI 代替自动化回归。
- 模拟新克隆验证 `start.bat`：临时重命名 `venv/` 后运行脚本，应看到英文提示与 PATH 回退；恢复 `venv/` 后再确认正常启动。
- 修改版本时同步应用元数据和白皮书，避免把文档头部日期当作应用版本。
- 放宽 `mediapipe` 版本约束或验证新版 Python 后，同步 README 徽章、快速开始版本窗口、`PROJECT_BIBLE.md` §10.1 与本文件 C6。
