# 侧栏 AI 子选项面板折叠/展开动画

> **版本**：1.1.0（2026-11-22）
> **范围**：`gui/main_window.py`（`_apply_options_panel_visibility`、`_build_control_panel`）、`tests/test_sidebar_options_animation.py`
> **状态**：已合入本地分支 `codex/fix-sidebar-options-animation`（2026-11-22，待 PR 合入后回改）；专项 5 项通过，全套件 1341 项通过

## 1. 目标与背景

用户反馈：侧栏“矩形识别 / 手势模式”子选项面板展开时严重卡顿掉帧，并伴随整个左侧栏抖动。定位到三个叠加根因：

1. **逐帧 `maximumHeight` 动画**：面板位于 `QScrollArea`（`widgetResizable=True`）内，每一帧高度变化都触发从面板到滚动区视口的整条布局链重排（4 张 squircle 卡片及全部子控件），160ms 内重排十余次；与视频帧 30fps 重绘叠加后表现为掉帧。
2. **滚动条出现/消失振荡**：展开时内容高度跨越视口阈值，5px 垂直滚动条出现 → 视口宽度 −5px → 全卡片宽度重排；高度变化与宽度变化互相耦合可逐帧振荡（滚动条出现→内容变窄→高度再变→滚动条消失→……），表现为整个侧栏持续水平抖动。
3. **无状态守卫**：`_sync_feature_toggle_states` 在任何功能开关变化时都会调用两个面板的可见性方法，目标状态未变也会停掉并重启动画。
4. **陈旧高度钳制（1.1.0 补充）**：`_animate_ai_feature_panel` 展开时把 AI 面板 `maximumHeight` 钳在展开瞬间的 `sizeHint`，子选项面板也用 `sizeHint` 钳制。若 AI 面板在选项收起状态下展开，之后再展开选项时两级钳制叠加，选项行被压成几像素高（视觉表现为文字重叠、控件挤在一起）。

## 2. 契约

| # | 契约 | 维护约束 |
|---|------|----------|
| C1 | 子选项面板折叠/展开采用“瞬时几何 + 透明度渐变”（展开 130ms / 收起 100ms，OutCubic），与 `_animate_ai_feature_panel` 同一模式 | 禁止改回逐帧 `maximumHeight` 高度动画 |
| C2 | 动画结束或被反向打断后必须落在稳定终态：可见性标志、解除高度钳制（展开 → `QWIDGETSIZE_MAX`，收起 → 0）、移除 `QGraphicsOpacityEffect`、句柄置 None | 中途反向切换不得残留半透明或错误几何 |
| C5 | 展开状态下两级面板（AI 面板与子选项面板）的 `maximumHeight` 必须为 `QWIDGETSIZE_MAX`，禁止用 `sizeHint` 数值钳制 | 钳制值会陈旧：内容变化后被旧值挤压（本次重叠 bug 根因） |
| C3 | 侧栏滚动区垂直滚动条常驻（`ScrollBarAlwaysOn`） | 不得改回 `ScrollBarAsNeeded`；滚动条出现/消失会引发视口宽度抖动 |
| C4 | 已处目标状态时调用可见性方法必须是 no-op（不创建动画）；判定用 `isHidden()`（部件自身标志），禁用 `isVisible()`（受祖先可见性影响） | `_sync_feature_toggle_states` 每次都会调用两个面板方法，no-op 是防动画重启的关键 |

## 3. 维护指南

- 若想恢复“生长”式展开，需先解决 `QScrollArea` 内逐帧 relayout 的成本（如离屏快照过渡），否则维持 C1 淡入方案。
- 展开目标高度唯一来源是 `panel.sizeHint().height()`；调整面板内部结构后无需改动画代码。
- 滚动条宽度样式（`theme.py` QScrollBar）调整与 C3 无关，常驻策略照宽度自动适配。
- 展开目标高度不再需要 `sizeHint()`：展开即 `QWIDGETSIZE_MAX`，由布局按内容自适应；收起钳到 0。
- 回归测试：`pytest tests/test_sidebar_options_animation.py`（5 项：展开/收起终态、重复调用 no-op、手势面板同等行为、陈旧钳制挤压回归）。

