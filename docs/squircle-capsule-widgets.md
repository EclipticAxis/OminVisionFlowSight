# Squircle 连续曲率圆角 设计文档

> **版本**：1.0 — 2026-11-17
> **范围**：`gui/squircle.py`（矢量路径与贴图渲染）、`gui/theme.py`（`SQUIRCLE_ASSETS` 清单 + 胶囊 QSS 全面换贴图）、`tools/gen_squircle_assets.py`（资产再生成）、设置对话框修复、契约测试
> **状态**：已合入 main（PR #1 → `1624ad9`）；52 张 9-slice 资产随仓库分发；几何 + 样式/资产契约测试通过

---

## 1. 目标与背景

### 1.1 为什么是 squircle 而不是 border-radius

QSS `border-radius` 只能画普通圆弧：曲率在直边连接点从 0 突跳到 1/r（非 G2 连续），圆角与直边的过渡在视觉上生硬。本设计用 n=4 超椭圆（|x/a|^4+|y/b|^4=1）构造**连续曲率圆角**：超椭圆弧在直边连接点处曲率自然趋于 0，随后平滑升至 45° 最大再回落，整条路径 G2 连续——即 iOS/macOS 的 continuous corner（squircle）造型。

### 1.2 两种渲染路径的分工

| 路径 | 用途 | 原因 |
|---|---|---|
| **矢量路径**（`squircle_path()`，QPainter 实时绘制） | `camera_cell.py` 叠加 chip（label/device/badges/FPS/targets）等动态内容 | QPainter 自绘场景无 QSS，且尺寸/内容随帧变化 |
| **预渲染 9-slice 贴图**（QSS `border-image`） | 全部胶囊控件 QSS（按钮 / metric pill / 卡片 / combo / checkbox / tooltip 等） | QSS 无法表达超椭圆路径，只能用贴图；9-slice 保证任意尺寸下圆角不被拉伸变形 |

## 2. 架构

### 2.1 模块职责

- `gui/squircle.py`：**纯模块**（import 不依赖 QApplication，Qt 类型仅在函数体内使用）。`squircle_path(rect, radius, segments=10)` 构造覆盖 rect 的闭合 squircle 路径：每角为角点局部 2r×2r 正方形内的 1/4 超椭圆弧，采样为折线；`radius` clamp 到 `min(w,h)/2`（达到后即"跑道形"端点），退化输入安全。另有 `render_tile_to()`（9-slice 资产渲染）与 `asset_file()`（资产路径解析）。
- `gui/theme.py`：**单一定义源** `SQUIRCLE_ASSETS` 清单（逐项声明 bg/border 颜色与 radius），以及 `COLORS` / `RADIUS` / `SQ_TILE_SCALE` / `SQ_TILE_PAD` 常量；全部胶囊 QSS 规则改为引用对应贴图。
- `tools/gen_squircle_assets.py`：遍历清单、离屏（offscreen）渲染全部资产到 `gui/assets/squircle/`，**可重复运行（覆盖写）**。
- `gui/main_window.py`：矩形颜色 swatch 等在**运行时**动态渲染贴图（产物进 `_runtime/`，见 §3）。

### 2.2 关键常量

| 常量 | 值 | 含义 |
|---|---|---|
| `SUPERELLIPSE_N` | 4 | 超椭圆指数（经典 squircle） |
| `DEFAULT_SEGMENTS` | 10 | 每角采样段数；任意采样点与真实曲线偏差 < 0.1%·r（r=5px 时 < 0.005px，远低于像素精度） |
| `SQ_TILE_SCALE` | 4 | 贴图超采样（@4x 渲染、按逻辑尺寸引用，高分屏不糊） |

## 3. 资产契约（单一定义源）

1. **唯一事实源是 `gui/theme.py` 的 `SQUIRCLE_ASSETS`**：新增/修改控件圆角样式 = 改清单 + 重跑生成器，**禁止手改 PNG**；
2. 再生成命令：`python tools/gen_squircle_assets.py`（脚本自带 offscreen 平台设置与仓库根 sys.path 引导）；
3. 生成产物随仓库提交分发——克隆即可用，终端用户无需任何构建步骤；
4. `gui/assets/squircle/_runtime/` 是**运行时**动态贴图目录（如主窗口矩形色板），已 gitignore，**不提交**；
5. 资产命名即 QSS 引用名：`{角色}_{radius}px@4x.png`；契约测试（`tests/test_squircle.py`）校验**清单 / QSS 引用 / 磁盘文件**三方一致。

## 4. QSS 使用契约

- 胶囊控件一律引用对应贴图的 `border-image`，**不再写 `border-radius`**——两种圆角混用会破坏视觉统一；
- 设置对话框三条硬约束：行高统一 **36px**；滚动条保持可见 **5px**；输入行不做 fixed-height 挤压。PR #1 修复的两个回归（border-image band 把滚动条挤到 1px 使设置看似"折叠消失"、band+min-height 挤压导致输入文本被裁剪）正是违反本契约的教训；
- 新增控件样式前，先查清单是否已有同 bg/border/radius 的资产可复用，避免清单膨胀。

## 5. 维护指南

- 改圆角半径/颜色：改 `SQUIRCLE_ASSETS` → 跑生成器 → `pytest tests/test_squircle.py`；
- 改几何（N、采样段数）：`gui/squircle.py` 常量与几何契约测试同步更新；
- 叠加绘制（camera_cell 等 QPainter 场景）继续走矢量路径，**不要**为逐帧变化的内容生成贴图。
