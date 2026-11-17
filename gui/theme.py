from gui.squircle import asset_file


COLORS: dict[str, str] = {
    # === 基础背景层（山田凉冷蓝灰基底） ===
    "BG_PRIMARY": "#0B1020",
    "BG_SECONDARY": "#12182B",
    "BG_TERTIARY": "#1A2238",
    "BG_ELEVATED": "#222D47",
    "BG_GLASS": "rgba(18, 24, 43, 0.74)",
    "BG_GLASS_HOVER": "rgba(26, 34, 56, 0.9)",

    # === 边框层（低饱和蓝灰） ===
    "BORDER": "#27304A",
    "BORDER_LIGHT": "#32405F",
    "BORDER_ACCENT": "#46608E",
    "BORDER_GLOW": "rgba(111, 168, 255, 0.18)",

    # === 强调色（冷蓝） ===
    "ACCENT": "#6FA8FF",
    "ACCENT_HOVER": "#8CC2FF",
    "ACCENT_PRESSED": "#4D82D6",
    "ACCENT_GLOW": "rgba(111, 168, 255, 0.3)",
    "ACCENT_SUBTLE": "rgba(111, 168, 255, 0.14)",

    # === 文字色（冷白与灰蓝） ===
    "TEXT_PRIMARY": "#EAF1FF",
    "TEXT_SECONDARY": "#9AA8C7",
    "TEXT_TERTIARY": "#6E7A96",
    "TEXT_DISABLED": "#465067",
    "TEXT_MUTED": "#323B51",

    # === 状态色（克制） ===
    "DANGER": "#D86A7A",
    "DANGER_HOVER": "#E48695",
    "DANGER_PRESSED": "#BA5161",
    "DANGER_GLOW": "rgba(216, 106, 122, 0.24)",
    "SUCCESS": "#6FA38C",
    "SUCCESS_SUBTLE": "rgba(111, 163, 140, 0.14)",
    "WARNING": "#B9A36A",
    "WARNING_SUBTLE": "rgba(185, 163, 106, 0.14)",

    # === 滚动条 ===
    "SCROLLBAR_BG": "#12182B",
    "SCROLLBAR_HANDLE": "#27304A",
    "SCROLLBAR_HANDLE_HOVER": "#32405F",
}

C = COLORS

# === 阴影与发光效果（暖色系阴影） ===
SHADOWS = {
    "SM": "0 1px 2px rgba(7, 10, 18, 0.42)",
    "MD": "0 4px 12px rgba(7, 10, 18, 0.52)",
    "LG": "0 8px 24px rgba(7, 10, 18, 0.62)",
    "XL": "0 16px 48px rgba(7, 10, 18, 0.72)",
    "GLOW_ACCENT": "0 0 20px rgba(111, 168, 255, 0.18)",
    "GLOW_DANGER": "0 0 20px rgba(216, 106, 122, 0.18)",
}

# === 字体系统 ===
TYPOGRAPHY = {
    "H1": "font-size: 18pt; font-weight: 700; letter-spacing: -0.02em;",
    "H2": "font-size: 14pt; font-weight: 600; letter-spacing: -0.01em;",
    "H3": "font-size: 12pt; font-weight: 600;",
    "BODY": "font-size: 9.5pt; line-height: 1.5;",
    "CAPTION": "font-size: 8pt; letter-spacing: 0.02em;",
    "LABEL": "font-size: 8.5pt; font-weight: 500; letter-spacing: 0.01em;",
    "METRIC": "font-size: 9pt; font-weight: 600;",
}

# === 圆角半径（胶囊控件，按类别单独可调；视觉零变化默认值） ===
RADIUS = {
    "PILL": 5,      # QLabel#metricPill（含 accent/success/warning 变体）
    "BUTTON": 5,    # QPushButton / #secondaryCompact / #topBarSettings
    "CONTROL": 5,   # QComboBox / #featureCombo、QSpinBox/QDoubleSpinBox#featureCombo
    "CHECK": 5,     # QCheckBox::indicator、QCheckBox#featureSwitch::indicator、QCheckBox#topBarFlip
    "CARD": 5,      # #pathValue/#statusBanner/#heroCard/#sectionCard/#slotCard/#statRow/#aiFeaturePanel/#skeleton
    "GROUPBOX": 5,  # QGroupBox
    "TOOLTIP": 5,   # QToolTip
    "SMALL": 3,     # QCheckBox#topBarFlip::indicator（小指示器）
    "SCROLL": 2,    # QScrollBar track/handle
}

# === squircle（连续曲率圆角）资产清单（阶段 2 契约，单一定义源） ===
# 颜色一律写 COLORS 的键名字符串，由消费方（tools/gen_squircle_assets.py /
# 运行时动态生成）解析；radius 写 RADIUS 的键名字符串。
# border_width 为烘焙进 PNG 的描边宽度（CSS px，默认 1）；None 表示无边框线。
# gradient=(c1, c2) 时用对角渐变替代实底（如 heroCard）。
# 同底同边同半径的规则共用 token。
SQUIRCLE_ASSETS: dict[str, dict] = {
    # --- 指标胶囊 ---
    "PILL":         {"bg": "BG_TERTIARY",    "border": "BORDER_LIGHT", "radius": "PILL"},
    "PILL_ACCENT":  {"bg": "ACCENT_SUBTLE",  "border": "ACCENT",       "radius": "PILL"},
    "PILL_SUCCESS": {"bg": "SUCCESS_SUBTLE", "border": "SUCCESS",      "radius": "PILL"},
    "PILL_WARNING": {"bg": "WARNING_SUBTLE", "border": "WARNING",      "radius": "PILL"},
    # --- 卡片 / 面板 ---
    "PATH_VALUE":      {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "CARD"},
    "STATUS_BANNER":   {"bg": "BG_GLASS",    "border": "BORDER_ACCENT", "radius": "CARD"},
    "HERO_CARD":       {"bg": "#26241F",   "border": "BORDER_ACCENT", "radius": "CARD",
                        "gradient": ("#2D2A23", "#1F1E1B")},
    "CARD_SECONDARY":  {"bg": "BG_SECONDARY", "border": "BORDER",      "radius": "CARD"},
    "CARD":            {"bg": "BG_TERTIARY",  "border": "BORDER_LIGHT", "radius": "CARD"},
    "CARD_HOVER":      {"bg": "BG_ELEVATED",  "border": "BORDER_ACCENT", "radius": "CARD"},
    "SKELETON":        {"bg": "BG_TERTIARY",  "border": None,          "radius": "CARD"},
    # --- 顶部栏 ---
    "TOPBAR_BTN":         {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "BUTTON"},
    "TOPBAR_BTN_HOVER":   {"bg": "BG_ELEVATED", "border": "ACCENT",       "radius": "BUTTON"},
    "TOPBAR_BTN_PRESSED": {"bg": "ACCENT_PRESSED", "border": "ACCENT_PRESSED", "radius": "BUTTON"},
    "TOPBAR_FLIP":        {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "CHECK"},
    "TOPBAR_FLIP_HOVER":  {"bg": "BG_ELEVATED", "border": "ACCENT",       "radius": "CHECK"},
    "CHECK_SMALL":        {"bg": "BG_SECONDARY", "border": "BORDER_LIGHT", "radius": "SMALL"},
    "CHECK_SMALL_CHECKED": {"bg": "ACCENT", "border": "ACCENT", "radius": "SMALL"},
    # --- 复选框 ---
    "CHECK":         {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "CHECK", "border_width": 1.5},
    "CHECK_HOVER":   {"bg": "BG_ELEVATED", "border": "ACCENT",       "radius": "CHECK", "border_width": 1.5},
    "CHECK_CHECKED": {"bg": "ACCENT",      "border": "ACCENT",       "radius": "CHECK", "border_width": 1.5},
    "FEATURE_SWITCH":          {"bg": "BG_SECONDARY", "border": "BORDER_LIGHT", "radius": "CHECK"},
    "FEATURE_SWITCH_HOVER":    {"bg": "BG_ELEVATED",  "border": "ACCENT",       "radius": "CHECK"},
    "FEATURE_SWITCH_CHECKED":  {"bg": "ACCENT",       "border": "ACCENT",       "radius": "CHECK"},
    "FEATURE_SWITCH_DISABLED": {"bg": "BG_SECONDARY", "border": "BORDER",       "radius": "CHECK"},
    # --- 按钮 ---
    "BUTTON":         {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "BUTTON"},
    "BUTTON_HOVER":   {"bg": "BG_ELEVATED", "border": "ACCENT",       "radius": "BUTTON"},
    "BUTTON_PRESSED": {"bg": "ACCENT_PRESSED", "border": "ACCENT_PRESSED", "radius": "BUTTON"},
    "BUTTON_DISABLED": {"bg": "BG_SECONDARY", "border": "BORDER",     "radius": "BUTTON"},
    "PRIMARY":         {"bg": "ACCENT",         "border": None, "radius": "BUTTON"},
    "PRIMARY_HOVER":   {"bg": "ACCENT_HOVER",   "border": None, "radius": "BUTTON"},
    "PRIMARY_PRESSED": {"bg": "ACCENT_PRESSED", "border": None, "radius": "BUTTON"},
    "SECONDARY_PRESSED": {"bg": "BORDER", "border": "BORDER_LIGHT", "radius": "BUTTON"},
    "SECONDARY_COMPACT":        {"bg": "BG_SECONDARY", "border": "BORDER_LIGHT", "radius": "BUTTON"},
    "SECONDARY_COMPACT_HOVER":  {"bg": "BG_ELEVATED",  "border": "ACCENT",       "radius": "BUTTON"},
    "DANGER_BTN":        {"bg": "DANGER",        "border": None, "radius": "BUTTON"},
    "DANGER_BTN_HOVER":  {"bg": "DANGER_HOVER",  "border": None, "radius": "BUTTON"},
    "DANGER_BTN_PRESSED": {"bg": "DANGER_PRESSED", "border": None, "radius": "BUTTON"},
    "GHOST_HOVER":       {"bg": "BG_TERTIARY", "border": None, "radius": "BUTTON"},
    # --- 下拉 / 输入控件 ---
    "CONTROL":       {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "CONTROL"},
    "CONTROL_HOVER": {"bg": "BG_TERTIARY", "border": "ACCENT",       "radius": "CONTROL"},
    "FEATURE_COMBO":          {"bg": "BG_SECONDARY", "border": "BORDER_LIGHT", "radius": "CONTROL"},
    "FEATURE_COMBO_HOVER":    {"bg": "BG_SECONDARY", "border": "ACCENT",       "radius": "CONTROL"},
    "FEATURE_COMBO_DISABLED": {"bg": "BG_SECONDARY", "border": "BORDER",       "radius": "CONTROL"},
    "COMBO_VIEW":         {"bg": "BG_TERTIARY", "border": "BORDER",       "radius": "CONTROL"},
    "COMBO_VIEW_FEATURE": {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "CONTROL"},
    "COMBO_ITEM":          {"bg": "BG_TERTIARY", "border": None, "radius": "CONTROL"},
    "COMBO_ITEM_HOVER":    {"bg": "BG_ELEVATED", "border": None, "radius": "CONTROL"},
    "COMBO_ITEM_SELECTED": {"bg": "ACCENT",      "border": None, "radius": "CONTROL"},
    # --- 其余 ---
    "GROUPBOX":   {"bg": "BG_TERTIARY", "border": "BORDER_LIGHT", "radius": "GROUPBOX"},
    "TOOLTIP":    {"bg": "BG_TERTIARY", "border": "BORDER_ACCENT", "radius": "TOOLTIP"},
    "DIALOG":     {"bg": "BG_PRIMARY",  "border": None,           "radius": "CARD"},
}

# === squircle 瓦片几何（必须与 tools/gen_squircle_assets.py 的生成参数一致） ===
SQ_TILE_PAD = 1    # 瓦片透明边距（源像素组；scale=4 时 = 4px）
SQ_TILE_SCALE = 4  # 资产渲染倍率（文件名 @4x）


def squircle_slice(radius: int) -> int:
    """border-image 切片（源像素）：角片完整覆盖 squircle 角弧，中央片保持非零。"""
    return (SQ_TILE_PAD + radius) * SQ_TILE_SCALE - 1


def squircle_border_image(token: str, radius_key: str, bw: int | None = None) -> str:
    """胶囊控件的完整边框声明（QSS 单条规则内联使用）。

    背景/描边/圆角均已烘焙进 PNG；border-width 只决定九宫格条带在控件上的宽度
    （默认 radius+1，使烘焙的 1px 描边在屏幕上仍约为 1px）。
    注意：Qt Style Sheets 无法加载 file:/// URL，必须使用带引号的本地绝对路径。
    """
    r = RADIUS[radius_key]
    w = r + 1 if bw is None else bw
    img = asset_file(token, r).resolve().as_posix()
    return (
        f"background-color: transparent; border-style: solid; "
        f"border-width: {w}px; border-image: url(\"{img}\") "
        f"{squircle_slice(r)} stretch stretch;"
    )


STYLESHEET = f"""
/* === 全局基础 === */
QMainWindow {{
    background-color: {C['BG_PRIMARY']};
}}

QWidget {{
    background-color: {C['BG_PRIMARY']};
    color: {C['TEXT_PRIMARY']};
    font-family: "Segoe UI Variable Text", "Segoe UI", system-ui, -apple-system, sans-serif;
    font-size: 9.5pt;
    line-height: 1.5;
}}

/* === 文字样式 === */
QLabel {{
    background-color: transparent;
    color: {C['TEXT_PRIMARY']};
}}

QLabel#title {{
    font-size: 18pt;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: {C['ACCENT']};
    padding: 12px 0px 8px 0px;
}}

QLabel#subtitle {{
    font-size: 11pt;
    font-weight: 500;
    color: {C['TEXT_SECONDARY']};
    padding: 4px 0px 8px 0px;
}}

QLabel#quote {{
    font-size: 9pt;
    font-style: italic;
    color: {C['TEXT_SECONDARY']};
    padding: 6px 0px 12px 0px;
    line-height: 1.5;
}}

QLabel#sectionTitle {{
    font-size: 12pt;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: {C['TEXT_PRIMARY']};
    padding: 4px 0px;
}}

QLabel#sectionHint {{
    font-size: 8.5pt;
    color: {C['TEXT_SECONDARY']};
    padding: 2px 0px 8px 0px;
    line-height: 1.4;
}}

QLabel#slotTitle {{
    font-size: 10pt;
    font-weight: 600;
    color: {C['TEXT_PRIMARY']};
}}

QLabel#slotMeta {{
    font-size: 8.5pt;
    color: {C['TEXT_TERTIARY']};
    line-height: 1.4;
}}

QLabel#fieldLabel {{
    font-size: 8pt;
    font-weight: 500;
    letter-spacing: 0.01em;
    color: {C['TEXT_SECONDARY']};
    padding: 8px 0px 4px 0px;
}}

QLabel#pathValue {{
    {squircle_border_image('PATH_VALUE', 'CARD')}
    padding: 12px 14px;
    color: {C['TEXT_PRIMARY']};
    font-size: 9pt;
    font-weight: 500;
}}

/* === 状态标签 === */
QLabel#statusBanner {{
    {squircle_border_image('STATUS_BANNER', 'CARD')}
    padding: 14px;
    color: {C['TEXT_PRIMARY']};
    font-size: 9pt;
    font-weight: 500;
}}

QLabel#status {{
    font-size: 8.5pt;
    color: {C['TEXT_SECONDARY']};
}}

/* === 指标胶囊 === */
QLabel#metricPill {{
    {squircle_border_image('PILL', 'PILL')}
    padding: 8px 12px;
    color: {C['TEXT_PRIMARY']};
    font-size: 8.5pt;
    font-weight: 600;
}}

QLabel#metricPill#accent {{
    {squircle_border_image('PILL_ACCENT', 'PILL')}
    color: {C['ACCENT']};
}}

QLabel#metricPill#success {{
    {squircle_border_image('PILL_SUCCESS', 'PILL')}
    color: {C['SUCCESS']};
}}

QLabel#metricPill#warning {{
    {squircle_border_image('PILL_WARNING', 'PILL')}
    color: {C['WARNING']};
}}

/* === 统计行 === */
QLabel#statName {{
    background-color: transparent;
    color: {C['TEXT_SECONDARY']};
    font-size: 8.5pt;
    padding: 12px 14px;
}}

QLabel#statValue {{
    background-color: transparent;
    color: {C['TEXT_PRIMARY']};
    font-size: 9.5pt;
    font-weight: 600;
    padding: 12px 14px;
}}

/* === 布局容器 === */
QWidget#sidePanel {{
    background-color: {C['BG_SECONDARY']};
    border-right: 1px solid {C['BORDER']};
}}

QWidget#panelContent {{
    background-color: transparent;
}}

QWidget#cameraGrid {{
    background-color: {C['BG_PRIMARY']};
}}

/* === 顶部固定栏 === */
QFrame#topBar {{
    background-color: {C['BG_SECONDARY']};
    border-bottom: 1px solid {C['BORDER']};
}}

QLabel#topBarBrand {{
    background-color: transparent;
    font-size: 11pt;
    font-weight: 700;
    letter-spacing: -0.01em;
    color: {C['ACCENT']};
}}

QPushButton#topBarSettings {{
    {squircle_border_image('TOPBAR_BTN', 'BUTTON')}
    color: {C['TEXT_SECONDARY']};
    padding: 3px 16px;
    min-height: 20px;
    font-size: 9pt;
    font-weight: 500;
}}

QPushButton#topBarSettings:hover {{
    {squircle_border_image('TOPBAR_BTN_HOVER', 'BUTTON')}
    color: {C['ACCENT_HOVER']};
}}

QPushButton#topBarSettings:pressed {{
    {squircle_border_image('TOPBAR_BTN_PRESSED', 'BUTTON')}
    color: #1F1E1B;
}}

QCheckBox#topBarFlip {{
    {squircle_border_image('TOPBAR_FLIP', 'CHECK')}
    color: {C['TEXT_SECONDARY']};
    padding: 3px 12px;
    min-height: 20px;
    font-size: 9pt;
    font-weight: 500;
}}

QCheckBox#topBarFlip:hover {{
    {squircle_border_image('TOPBAR_FLIP_HOVER', 'CHECK')}
    color: {C['ACCENT_HOVER']};
}}

QCheckBox#topBarFlip::indicator {{
    width: 13px;
    height: 13px;
    {squircle_border_image('CHECK_SMALL', 'SMALL')}
}}

QCheckBox#topBarFlip::indicator:checked {{
    {squircle_border_image('CHECK_SMALL_CHECKED', 'SMALL')}
}}

/* === 卡片组件 === */
QFrame#heroCard {{
    {squircle_border_image('HERO_CARD', 'CARD')}
}}

QFrame#sectionCard {{
    {squircle_border_image('CARD_SECONDARY', 'CARD')}
}}

QFrame#slotCard {{
    {squircle_border_image('CARD', 'CARD')}
}}

QFrame#slotCard:hover {{
    {squircle_border_image('CARD_HOVER', 'CARD')}
}}

QFrame#statRow {{
    {squircle_border_image('CARD', 'CARD')}
}}

QFrame#aiFeaturePanel {{
    {squircle_border_image('CARD', 'CARD')}
}}

QCheckBox#featureSwitch {{
    background-color: transparent;
    color: {C['TEXT_PRIMARY']};
    spacing: 10px;
    padding: 6px 4px;
    font-size: 9pt;
    font-weight: 500;
}}

QCheckBox#featureSwitch:disabled {{
    color: {C['TEXT_DISABLED']};
}}

QCheckBox#featureSwitch::indicator {{
    width: 18px;
    height: 18px;
    {squircle_border_image('FEATURE_SWITCH', 'CHECK')}
}}

QCheckBox#featureSwitch::indicator:hover {{
    {squircle_border_image('FEATURE_SWITCH_HOVER', 'CHECK')}
}}

QCheckBox#featureSwitch::indicator:checked {{
    {squircle_border_image('FEATURE_SWITCH_CHECKED', 'CHECK')}
}}

QCheckBox#featureSwitch::indicator:disabled {{
    {squircle_border_image('FEATURE_SWITCH_DISABLED', 'CHECK')}
}}

QWidget#featureRow {{
    background-color: transparent;
}}

QWidget#gestureOptionsPanel, QWidget#rectangleOptionsPanel, QWidget#rectangleColorPanel {{
    background-color: transparent;
}}

QLabel#featureLabel {{
    background-color: transparent;
    color: {C['TEXT_SECONDARY']};
    font-size: 8.5pt;
    font-weight: 500;
}}

QComboBox#featureCombo {{
    {squircle_border_image('FEATURE_COMBO', 'CONTROL')}
    color: {C['TEXT_PRIMARY']};
    padding: 3px 10px;
    min-height: 20px;
    font-size: 8.5pt;
}}

QSpinBox#featureCombo, QDoubleSpinBox#featureCombo {{
    {squircle_border_image('FEATURE_COMBO', 'CONTROL')}
    color: {C['TEXT_PRIMARY']};
    padding: 5px 10px;
    min-height: 20px;
    font-size: 8.5pt;
}}

QSpinBox#featureCombo:hover, QDoubleSpinBox#featureCombo:hover {{
    {squircle_border_image('FEATURE_COMBO_HOVER', 'CONTROL')}
}}

QSpinBox#featureCombo:disabled, QDoubleSpinBox#featureCombo:disabled {{
    {squircle_border_image('FEATURE_COMBO_DISABLED', 'CONTROL')}
    color: {C['TEXT_DISABLED']};
}}

QComboBox#featureCombo:disabled {{
    {squircle_border_image('FEATURE_COMBO_DISABLED', 'CONTROL')}
    color: {C['TEXT_DISABLED']};
}}

QComboBox#featureCombo QAbstractItemView {{
    {squircle_border_image('COMBO_VIEW_FEATURE', 'CONTROL')}
    color: {C['TEXT_PRIMARY']};
    selection-background-color: {C['ACCENT']};
    selection-color: {C['BG_PRIMARY']};
    outline: none;
    padding: 4px;
}}

QComboBox#featureCombo QAbstractItemView::item {{
    {squircle_border_image('COMBO_ITEM', 'CONTROL')}
    color: {C['TEXT_PRIMARY']};
    padding: 6px 10px;
    min-height: 22px;
}}

QComboBox#featureCombo QAbstractItemView::item:hover {{
    {squircle_border_image('COMBO_ITEM_HOVER', 'CONTROL')}
}}

QComboBox#featureCombo QAbstractItemView::item:selected {{
    {squircle_border_image('COMBO_ITEM_SELECTED', 'CONTROL')}
    color: {C['BG_PRIMARY']};
}}

QComboBox#featureCombo QAbstractScrollArea {{
    background-color: {C['BG_TERTIARY']};
    border: none;
}}

QComboBox#featureCombo QAbstractScrollArea::corner {{
    background-color: {C['BG_TERTIARY']};
}}

/* === 按钮 === */
QPushButton {{
    {squircle_border_image('BUTTON', 'BUTTON')}
    color: {C['TEXT_PRIMARY']};
    padding: 6px 18px;
    font-weight: 500;
    font-size: 9pt;
    min-height: 24px;
}}

QPushButton:hover {{
    {squircle_border_image('BUTTON_HOVER', 'BUTTON')}
    color: {C['ACCENT_HOVER']};
}}

QPushButton:pressed {{
    {squircle_border_image('BUTTON_PRESSED', 'BUTTON')}
    color: {C['BG_PRIMARY']};
}}

QPushButton:disabled {{
    {squircle_border_image('BUTTON_DISABLED', 'BUTTON')}
    color: {C['TEXT_DISABLED']};
}}

QPushButton#primary {{
    {squircle_border_image('PRIMARY', 'BUTTON')}
    color: #1F1E1B;
    font-weight: 600;
    font-size: 9.5pt;
}}

QPushButton#primary:hover {{
    {squircle_border_image('PRIMARY_HOVER', 'BUTTON')}
    color: #1F1E1B;
}}

QPushButton#primary:pressed {{
    {squircle_border_image('PRIMARY_PRESSED', 'BUTTON')}
}}

QPushButton#secondary {{
    {squircle_border_image('BUTTON', 'BUTTON')}
    color: {C['TEXT_PRIMARY']};
}}

QPushButton#secondary:hover {{
    {squircle_border_image('BUTTON_HOVER', 'BUTTON')}
}}

QPushButton#secondary:pressed {{
    {squircle_border_image('SECONDARY_PRESSED', 'BUTTON')}
}}

QPushButton#secondaryCompact {{
    {squircle_border_image('SECONDARY_COMPACT', 'BUTTON')}
    color: {C['TEXT_SECONDARY']};
    padding: 4px 10px;
    min-height: 20px;
    font-size: 8.5pt;
}}

QPushButton#secondaryCompact:hover {{
    {squircle_border_image('SECONDARY_COMPACT_HOVER', 'BUTTON')}
    color: {C['TEXT_PRIMARY']};
}}

QPushButton#secondaryCompact:disabled {{
    {squircle_border_image('BUTTON_DISABLED', 'BUTTON')}
    color: {C['TEXT_DISABLED']};
}}

QPushButton#danger {{
    {squircle_border_image('DANGER_BTN', 'BUTTON')}
    color: #1F1E1B;
    font-weight: 600;
}}

QPushButton#danger:hover {{
    {squircle_border_image('DANGER_BTN_HOVER', 'BUTTON')}
}}

QPushButton#danger:pressed {{
    {squircle_border_image('DANGER_BTN_PRESSED', 'BUTTON')}
}}

QPushButton#ghost {{
    /* 无背景无描边，显式关闭继承自 QPushButton 的 border-image */
    border-image: none;
    border-style: none;
    background-color: transparent;
    color: {C['TEXT_SECONDARY']};
    padding: 8px 12px;
}}

QPushButton#ghost:hover {{
    {squircle_border_image('GHOST_HOVER', 'BUTTON')}
    color: {C['TEXT_PRIMARY']};
}}

/* === 下拉框 === */
QComboBox {{
    {squircle_border_image('CONTROL', 'CONTROL')}
    color: {C['TEXT_PRIMARY']};
    padding: 5px 14px;
    min-height: 28px;
    font-size: 9pt;
    font-weight: 500;
}}

QComboBox:hover {{
    {squircle_border_image('CONTROL_HOVER', 'CONTROL')}
}}

QComboBox::drop-down {{
    border: none;
    width: 28px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {C['TEXT_SECONDARY']};
    margin-right: 10px;
}}

QComboBox QAbstractItemView {{
    {squircle_border_image('COMBO_VIEW', 'CONTROL')}
    color: {C['TEXT_PRIMARY']};
    selection-background-color: {C['ACCENT']};
    selection-color: #1F1E1B;
    outline: none;
    padding: 6px;
}}

QComboBox QAbstractItemView::item {{
    padding: 8px 14px;
    min-height: 28px;
    {squircle_border_image('COMBO_ITEM', 'CONTROL')}
    color: {C['TEXT_PRIMARY']};
}}

QComboBox QAbstractItemView::item:hover {{
    {squircle_border_image('COMBO_ITEM_HOVER', 'CONTROL')}
}}

QComboBox QAbstractItemView::item:selected {{
    {squircle_border_image('COMBO_ITEM_SELECTED', 'CONTROL')}
    color: #1F1E1B;
}}

QComboBox QAbstractScrollArea {{
    background-color: {C['BG_TERTIARY']};
    border: none;
}}

QComboBox QAbstractScrollArea::corner {{
    background-color: {C['BG_TERTIARY']};
}}

/* === 分组框 === */
QGroupBox {{
    {squircle_border_image('GROUPBOX', 'GROUPBOX')}
    margin-top: 14px;
    padding: 18px 14px 14px 14px;
    font-weight: 600;
    color: {C['TEXT_SECONDARY']};
    font-size: 9pt;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 4px;
    padding: 0px 8px;
    color: {C['ACCENT']};
    background-color: {C['BG_TERTIARY']};
    font-weight: 600;
    font-size: 9pt;
}}

/* === 复选框 === */
QCheckBox {{
    background-color: transparent;
    color: {C['TEXT_PRIMARY']};
    spacing: 12px;
    padding: 8px 0px;
    font-size: 9.5pt;
    font-weight: 500;
}}

QCheckBox::indicator {{
    width: 22px;
    height: 22px;
    {squircle_border_image('CHECK', 'CHECK')}
}}

QCheckBox::indicator:hover {{
    {squircle_border_image('CHECK_HOVER', 'CHECK')}
}}

QCheckBox::indicator:checked {{
    {squircle_border_image('CHECK_CHECKED', 'CHECK')}
}}

QCheckBox::indicator:checked::after {{
    content: "✓";
    color: #1F1E1B;
    font-size: 12px;
    font-weight: bold;
}}

/* === 分隔线 === */
QFrame[frameShape="4"] {{
    background-color: {C['BORDER']};
    max-height: 1px;
    border: none;
}}

/* === 滚动区域 === */
QScrollArea {{
    background-color: transparent;
    border: none;
}}

QScrollArea#panelScroll {{
    background-color: transparent;
}}

/* === 滚动条 ===
   5px 细滚动条不使用 squircle border-image：border-width 条带会把 5px 宽
   内容挤压成 1px 细线，滑块几乎不可见（设置对话框曾因此"看起来内容缺失"）。
   纯色 + 2px 小圆角在 5px 宽度下与 squircle 视觉无差，是更合适的方案。 */
QScrollBar:vertical {{
    background-color: {C['SCROLLBAR_BG']};
    width: 5px;
    border-radius: {RADIUS['SCROLL']}px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background-color: {C['SCROLLBAR_HANDLE']};
    border-radius: {RADIUS['SCROLL']}px;
    min-height: 40px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {C['SCROLLBAR_HANDLE_HOVER']};
}}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {{
    background: none;
    height: 0px;
    border: none;
}}

QScrollBar:horizontal {{
    background-color: {C['SCROLLBAR_BG']};
    height: 5px;
    border-radius: {RADIUS['SCROLL']}px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background-color: {C['SCROLLBAR_HANDLE']};
    border-radius: {RADIUS['SCROLL']}px;
    min-width: 40px;
}}

QScrollBar::handle:horizontal:hover {{
    background-color: {C['SCROLLBAR_HANDLE_HOVER']};
}}

QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal,
QScrollBar::sub-page:horizontal {{
    background: none;
    width: 0px;
    border: none;
}}

/* === 工具提示 === */
QToolTip {{
    {squircle_border_image('TOOLTIP', 'TOOLTIP')}
    color: {C['TEXT_PRIMARY']};
    padding: 8px 12px;
    font-size: 8.5pt;
    font-weight: 500;
}}

/* === 骨架屏 === */
QWidget#skeleton {{
    {squircle_border_image('SKELETON', 'CARD')}
}}

/* === 呼吸灯状态 === */
QLabel#breathe {{
    background-color: transparent;
    color: {C['DANGER']};
    font-size: 8pt;
    font-weight: 600;
}}
"""


def get_stylesheet() -> str:
    return STYLESHEET


def get_color(name: str) -> str:
    """获取颜色值，支持动态获取"""
    return COLORS.get(name, "#000000")
