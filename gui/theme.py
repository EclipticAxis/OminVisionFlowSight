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
    background-color: {C['BG_TERTIARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 12px 14px;
    color: {C['TEXT_PRIMARY']};
    font-size: 9pt;
    font-weight: 500;
}}

/* === 状态标签 === */
QLabel#statusBanner {{
    background-color: {C['BG_GLASS']};
    border: 1px solid {C['BORDER_ACCENT']};
    border-radius: 5px;
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
    background-color: {C['BG_TERTIARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 8px 12px;
    color: {C['TEXT_PRIMARY']};
    font-size: 8.5pt;
    font-weight: 600;
}}

QLabel#metricPill#accent {{
    background-color: {C['ACCENT_SUBTLE']};
    border-color: {C['ACCENT']};
    color: {C['ACCENT']};
}}

QLabel#metricPill#success {{
    background-color: {C['SUCCESS_SUBTLE']};
    border-color: {C['SUCCESS']};
    color: {C['SUCCESS']};
}}

QLabel#metricPill#warning {{
    background-color: {C['WARNING_SUBTLE']};
    border-color: {C['WARNING']};
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
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_SECONDARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 6px 16px;
    min-height: 20px;
    font-size: 9pt;
    font-weight: 500;
}}

QPushButton#topBarSettings:hover {{
    background-color: {C['BG_ELEVATED']};
    border-color: {C['ACCENT']};
    color: {C['ACCENT_HOVER']};
}}

QPushButton#topBarSettings:pressed {{
    background-color: {C['ACCENT_PRESSED']};
    color: #1F1E1B;
    border-color: {C['ACCENT_PRESSED']};
}}

QCheckBox#topBarFlip {{
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_SECONDARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 5px 12px;
    min-height: 20px;
    font-size: 9pt;
    font-weight: 500;
}}

QCheckBox#topBarFlip:hover {{
    background-color: {C['BG_ELEVATED']};
    border-color: {C['ACCENT']};
    color: {C['ACCENT_HOVER']};
}}

QCheckBox#topBarFlip::indicator {{
    width: 13px;
    height: 13px;
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 3px;
    background-color: {C['BG_SECONDARY']};
}}

QCheckBox#topBarFlip::indicator:checked {{
    background-color: {C['ACCENT']};
    border-color: {C['ACCENT']};
}}

/* === 卡片组件 === */
QFrame#heroCard {{
    background-color: qlineargradient(
        x1:0, y1:0, x2:1, y2:1,
        stop:0 #2D2A23,
        stop:0.5 #26241F,
        stop:1 #1F1E1B
    );
    border: 1px solid {C['BORDER_ACCENT']};
    border-radius: 5px;
}}

QFrame#sectionCard {{
    background-color: {C['BG_SECONDARY']};
    border: 1px solid {C['BORDER']};
    border-radius: 5px;
}}

QFrame#slotCard {{
    background-color: {C['BG_TERTIARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
}}

QFrame#slotCard:hover {{
    border-color: {C['BORDER_ACCENT']};
    background-color: {C['BG_ELEVATED']};
}}

QFrame#statRow {{
    background-color: {C['BG_TERTIARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
}}

QFrame#aiFeaturePanel {{
    background-color: {C['BG_TERTIARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
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
    border-radius: 5px;
    border: 1px solid {C['BORDER_LIGHT']};
    background-color: {C['BG_SECONDARY']};
}}

QCheckBox#featureSwitch::indicator:hover {{
    border-color: {C['ACCENT']};
    background-color: {C['BG_ELEVATED']};
}}

QCheckBox#featureSwitch::indicator:checked {{
    background-color: {C['ACCENT']};
    border-color: {C['ACCENT']};
}}

QCheckBox#featureSwitch::indicator:disabled {{
    background-color: {C['BG_SECONDARY']};
    border-color: {C['BORDER']};
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
    background-color: {C['BG_SECONDARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 5px 10px;
    min-height: 20px;
    font-size: 8.5pt;
}}

QSpinBox#featureCombo, QDoubleSpinBox#featureCombo {{
    background-color: {C['BG_SECONDARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 5px 10px;
    min-height: 20px;
    font-size: 8.5pt;
}}

QSpinBox#featureCombo:hover, QDoubleSpinBox#featureCombo:hover {{
    border-color: {C['ACCENT']};
}}

QSpinBox#featureCombo:disabled, QDoubleSpinBox#featureCombo:disabled {{
    color: {C['TEXT_DISABLED']};
    border-color: {C['BORDER']};
}}

QComboBox#featureCombo:disabled {{
    color: {C['TEXT_DISABLED']};
    border-color: {C['BORDER']};
}}

QComboBox#featureCombo QAbstractItemView {{
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    selection-background-color: {C['ACCENT']};
    selection-color: {C['BG_PRIMARY']};
    outline: none;
    padding: 4px;
}}

QComboBox#featureCombo QAbstractItemView::item {{
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    padding: 6px 10px;
    min-height: 22px;
}}

QComboBox#featureCombo QAbstractItemView::item:hover {{
    background-color: {C['BG_ELEVATED']};
}}

QComboBox#featureCombo QAbstractItemView::item:selected {{
    background-color: {C['ACCENT']};
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
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 10px 18px;
    font-weight: 500;
    font-size: 9pt;
    min-height: 24px;
}}

QPushButton:hover {{
    background-color: {C['BG_ELEVATED']};
    border-color: {C['ACCENT']};
    color: {C['ACCENT_HOVER']};
}}

QPushButton:pressed {{
    background-color: {C['ACCENT_PRESSED']};
    color: {C['BG_PRIMARY']};
    border-color: {C['ACCENT_PRESSED']};
}}

QPushButton:disabled {{
    background-color: {C['BG_SECONDARY']};
    color: {C['TEXT_DISABLED']};
    border-color: {C['BORDER']};
}}

QPushButton#primary {{
    background-color: {C['ACCENT']};
    color: #1F1E1B;
    border: none;
    font-weight: 600;
    font-size: 9.5pt;
}}

QPushButton#primary:hover {{
    background-color: {C['ACCENT_HOVER']};
    color: #1F1E1B;
}}

QPushButton#primary:pressed {{
    background-color: {C['ACCENT_PRESSED']};
}}

QPushButton#secondary {{
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER_LIGHT']};
}}

QPushButton#secondary:hover {{
    background-color: {C['BG_ELEVATED']};
    border-color: {C['ACCENT']};
}}

QPushButton#secondary:pressed {{
    background-color: {C['BORDER']};
}}

QPushButton#secondaryCompact {{
    background-color: {C['BG_SECONDARY']};
    color: {C['TEXT_SECONDARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 4px 10px;
    min-height: 20px;
    font-size: 8.5pt;
}}

QPushButton#secondaryCompact:hover {{
    background-color: {C['BG_ELEVATED']};
    border-color: {C['ACCENT']};
    color: {C['TEXT_PRIMARY']};
}}

QPushButton#secondaryCompact:disabled {{
    color: {C['TEXT_DISABLED']};
    border-color: {C['BORDER']};
}}

QPushButton#danger {{
    background-color: {C['DANGER']};
    color: #1F1E1B;
    border: none;
    font-weight: 600;
}}

QPushButton#danger:hover {{
    background-color: {C['DANGER_HOVER']};
}}

QPushButton#danger:pressed {{
    background-color: {C['DANGER_PRESSED']};
}}

QPushButton#ghost {{
    background-color: transparent;
    color: {C['TEXT_SECONDARY']};
    border: none;
    padding: 8px 12px;
}}

QPushButton#ghost:hover {{
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
}}

/* === 下拉框 === */
QComboBox {{
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
    padding: 10px 14px;
    min-height: 28px;
    font-size: 9pt;
    font-weight: 500;
}}

QComboBox:hover {{
    border-color: {C['ACCENT']};
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
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER']};
    border-radius: 5px;
    selection-background-color: {C['ACCENT']};
    selection-color: #1F1E1B;
    outline: none;
    padding: 6px;
}}

QComboBox QAbstractItemView::item {{
    padding: 8px 14px;
    min-height: 28px;
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    border-radius: 5px;
}}

QComboBox QAbstractItemView::item:hover {{
    background-color: {C['BG_ELEVATED']};
}}

QComboBox QAbstractItemView::item:selected {{
    background-color: {C['ACCENT']};
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
    background-color: {C['BG_TERTIARY']};
    border: 1px solid {C['BORDER_LIGHT']};
    border-radius: 5px;
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
    border-radius: 5px;
    border: 1.5px solid {C['BORDER_LIGHT']};
    background-color: {C['BG_TERTIARY']};
}}

QCheckBox::indicator:hover {{
    border-color: {C['ACCENT']};
    background-color: {C['BG_ELEVATED']};
}}

QCheckBox::indicator:checked {{
    background-color: {C['ACCENT']};
    border-color: {C['ACCENT']};
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

/* === 滚动条 === */
QScrollBar:vertical {{
    background-color: {C['SCROLLBAR_BG']};
    width: 5px;
    border-radius: 2px;
    margin: 0px;
}}

QScrollBar::handle:vertical {{
    background-color: {C['SCROLLBAR_HANDLE']};
    border-radius: 2px;
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
    border-radius: 2px;
    margin: 0px;
}}

QScrollBar::handle:horizontal {{
    background-color: {C['SCROLLBAR_HANDLE']};
    border-radius: 2px;
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
    background-color: {C['BG_TERTIARY']};
    color: {C['TEXT_PRIMARY']};
    border: 1px solid {C['BORDER_ACCENT']};
    border-radius: 5px;
    padding: 8px 12px;
    font-size: 8.5pt;
    font-weight: 500;
}}

/* === 骨架屏 === */
QWidget#skeleton {{
    background-color: {C['BG_TERTIARY']};
    border-radius: 5px;
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
