import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QFrame,
    QScrollArea,
    QWidget,
)

from gui.animations import AnimationHelper
from gui.safe_widgets import WheelFocusComboBox, WheelFocusDoubleSpinBox, WheelFocusSpinBox
from gui.theme import COLORS as C


class SettingsDialog(QDialog):

    _SETTINGS_ORG = "VisionBata"
    _SETTINGS_APP = "VisionDataPlatform"
    _APP_VERSION = "VDP V1.1.1"
    _BUILD_DATE = "26/7/6"

    _KEY_OUTPUT_DIR = "recording/output_dir"
    _KEY_AI_CONF = "ai/conf_threshold"
    _KEY_AI_STRIDE = "ai/infer_stride"
    _KEY_AI_SMOOTH = "ai/smooth_alpha"
    _KEY_AI_HIDE_MS = "ai/hide_delay_ms"
    _KEY_AI_MODEL_PATH = "ai/model_path"
    _KEY_AI_PERSON_ENABLED = "ai/person_enabled"
    _KEY_AI_SKELETON_ENABLED = "ai/skeleton_enabled"
    _KEY_AI_GESTURE_ENABLED = "ai/gesture_enabled"
    _KEY_AI_GESTURE_MODE = "ai/gesture_mode"
    _KEY_AI_RECTANGLE_ENABLED = "ai/rectangle_enabled"
    _KEY_AI_RECTANGLE_SENSITIVITY = "ai/rectangle_sensitivity"
    _KEY_AI_RECTANGLE_MAX_COUNT = "ai/rectangle_max_count"
    _KEY_AI_RECTANGLE_TARGET_RGB = "ai/rectangle_target_color_rgb"
    _KEY_AI_RECTANGLE_TARGET_HSV = "ai/rectangle_target_color_hsv"
    _KEY_AI_RECTANGLE_COLOR_THRESHOLD = "ai/rectangle_color_threshold"
    _KEY_AI_PERSON_REDETECT_RETRIES = "ai/person_redetect_max_retries"
    _KEY_AI_PERSON_REDETECT_ROI_PAD = "ai/person_redetect_roi_pad"
    _KEY_AI_FILTER_TYPE = "ai/filter_type"
    _KEY_AI_MCUKF_KERNEL_SIGMA = "ai/mcukf_kernel_sigma"
    _KEY_AI_DENOISE_METHOD = "ai/denoise_method"
    _KEY_AI_DENOISE_STRENGTH = "ai/denoise_strength"
    _KEY_AI_MATCHING_STRATEGY = "ai/matching_strategy"
    _KEY_AI_IOU_THRESHOLD = "ai/iou_threshold"
    _KEY_AI_MAX_MISSES = "ai/max_misses"
    _KEY_AI_VELOCITY_CLIP = "ai/velocity_clip"
    _KEY_AI_HIGH_CONF_THRESH = "ai/high_conf_thresh"
    _KEY_AI_REDETECT_BUDGET = "ai/redetect_budget"
    _KEY_AI_REID_ENABLED = "ai/reid_enabled"
    _KEY_AI_REID_MIN_TRACKS = "ai/reid_min_tracks"
    _KEY_AI_REID_STRIDE = "ai/reid_stride"
    _KEY_AI_HEAD_CLASSIFIER_ENABLED = "ai/head_classifier_enabled"
    _KEY_AI_HEAD_CLASSIFIER_PATH = "ai/head_classifier_path"
    _KEY_AI_HEAD_CLASSIFIER_STRIDE = "ai/head_classifier_stride"
    _KEY_PIPELINE_ENABLED = "ai/pipeline_enabled"

    _DEFAULT_DIR = str(Path.cwd() / "recordings")
    _RECOMMENDED_CONF = 0.25
    _RECOMMENDED_STRIDE = 2
    _RECOMMENDED_SMOOTH = 0.85
    _RECOMMENDED_HIDE_MS = 1500
    _RECOMMENDED_REDETECT_RETRIES = 2
    _RECOMMENDED_REDETECT_ROI_PAD = 0.15
    _RECOMMENDED_FILTER_TYPE = "ukf"
    _RECOMMENDED_MCUKF_KERNEL_SIGMA = 0.25
    _RECOMMENDED_DENOISE_METHOD = "none"
    _RECOMMENDED_DENOISE_STRENGTH = 5
    _RECOMMENDED_MATCHING_STRATEGY = "hungarian"
    _RECOMMENDED_IOU_THRESHOLD = 0.25
    _RECOMMENDED_MAX_MISSES = 4
    _RECOMMENDED_VELOCITY_CLIP = 0.30
    _RECOMMENDED_HIGH_CONF_THRESH = 0.5
    _RECOMMENDED_REDETECT_BUDGET = 1
    _RECOMMENDED_REID_ENABLED = False
    _RECOMMENDED_REID_MIN_TRACKS = 3
    _RECOMMENDED_REID_STRIDE = 3
    _RECOMMENDED_HEAD_CLASSIFIER_ENABLED = False
    _RECOMMENDED_HEAD_CLASSIFIER_STRIDE = 2
    _DEFAULT_CONF = _RECOMMENDED_CONF
    _DEFAULT_STRIDE = _RECOMMENDED_STRIDE
    _DEFAULT_SMOOTH = _RECOMMENDED_SMOOTH
    _DEFAULT_HIDE_MS = _RECOMMENDED_HIDE_MS

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("系统设置")
        self.setFixedSize(520, 600)
        self.setModal(True)

        self._current_dir = self.load_output_dir()
        self._ai_conf = self.load_ai_conf()
        self._ai_stride = self.load_ai_stride()
        self._ai_smooth = self.load_ai_smooth()
        self._ai_hide_ms = self.load_ai_hide_ms()
        self._ai_redetect_retries = self.load_ai_person_redetect_retries()
        self._ai_redetect_roi_pad = self.load_ai_person_redetect_roi_pad()
        self._ai_filter_type = self.load_ai_filter_type()
        self._ai_mcukf_sigma = self.load_ai_mcukf_kernel_sigma()
        self._ai_denoise_method = self.load_ai_denoise_method()
        self._ai_denoise_strength = self.load_ai_denoise_strength()
        self._ai_matching_strategy = self.load_ai_matching_strategy()
        self._ai_iou_threshold = self.load_ai_iou_threshold()
        self._ai_max_misses = self.load_ai_max_misses()
        self._ai_velocity_clip = self.load_ai_velocity_clip()
        self._ai_high_conf_thresh = self.load_ai_high_conf_thresh()
        self._ai_redetect_budget = self.load_ai_redetect_budget()
        self._ai_reid_enabled = self.load_ai_reid_enabled()
        self._ai_reid_min_tracks = self.load_ai_reid_min_tracks()
        self._ai_reid_stride = self.load_ai_reid_stride()
        self._ai_model_path = self.load_ai_model_path()
        self._ai_head_classifier_enabled = self.load_ai_head_classifier_enabled()
        self._ai_head_classifier_path = self.load_ai_head_classifier_path()
        self._ai_head_classifier_stride = self.load_ai_head_classifier_stride()
        self._reset_confirming = False
        self._reset_confirm_timer = QTimer(self)
        self._reset_confirm_timer.setSingleShot(True)
        self._reset_confirm_timer.timeout.connect(self._cancel_reset_confirm)

        self._build_ui()
        self._apply_theme()
        self._open_animation = None
        self._close_animation = None
        self._closing_decision: str | None = None

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 18, 18, 16)
        root_layout.setSpacing(12)

        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("settingsContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 8, 4)
        layout.setSpacing(10)

        # --- 录像存储 ---
        title = QLabel("录像存储配置")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("视频保存位置")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        layout.addWidget(self._create_separator())

        path_label = QLabel("存储路径")
        path_label.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        layout.addWidget(path_label)

        path_row = QHBoxLayout()
        path_row.setSpacing(8)

        self._path_edit = QLineEdit(self._current_dir)
        self._path_edit.setReadOnly(True)
        self._path_edit.setFixedHeight(34)
        self._path_edit.setStyleSheet(
            f"background-color: {C['BG_TERTIARY']}; "
            f"color: {C['TEXT_PRIMARY']}; "
            f"border: 1px solid {C['BORDER_LIGHT']}; "
            f"border-radius: 5px; "
            f"padding: 6px 10px;"
            f"font-size: 9pt;"
            f"font-weight: 500;"
        )
        path_row.addWidget(self._path_edit, stretch=1)

        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse)
        path_row.addWidget(browse_btn)

        layout.addLayout(path_row)

        self._preview_label = QLabel(f"当前路径：{self._current_dir}")
        self._preview_label.setObjectName("status")
        self._preview_label.setWordWrap(True)
        self._preview_label.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 8pt;")
        layout.addWidget(self._preview_label)

        layout.addSpacing(6)

        # --- AI 检测设置 ---
        ai_title = QLabel("AI 检测设置")
        ai_title.setObjectName("title")
        layout.addWidget(ai_title)

        ai_subtitle = QLabel("推理与显示参数")
        ai_subtitle.setObjectName("subtitle")
        layout.addWidget(ai_subtitle)

        layout.addWidget(self._create_separator())

        layout.addLayout(self._create_spin_row("置信度阈值", self._ai_conf, 0.10, 0.90, 0.05, "_conf_spin"))
        layout.addLayout(self._create_spin_row_int("推理跳帧", self._ai_stride, 1, 10, 1, "_stride_spin", suffix=" (1=每帧, 2=每2帧)"))
        layout.addLayout(self._create_spin_row("平滑系数", self._ai_smooth, 0.0, 1.0, 0.05, "_smooth_spin"))
        layout.addLayout(self._create_spin_row_int("显示消失延迟", self._ai_hide_ms, 500, 5000, 100, "_hide_spin", suffix=" ms"))
        layout.addLayout(self._create_spin_row_int("重检测尝试", self._ai_redetect_retries, 0, 5, 1, "_redetect_retries_spin", suffix=" (0=关)"))
        layout.addLayout(self._create_spin_row("ROI 扩展系数", self._ai_redetect_roi_pad, 0.0, 0.5, 0.05, "_roi_pad_spin"))
        layout.addLayout(self._create_filter_type_row())
        layout.addLayout(self._create_spin_row("MCUKF 核带宽", self._ai_mcukf_sigma, 0.1, 2.0, 0.05, "_mcukf_sigma_spin"))
        layout.addLayout(self._create_matching_strategy_row())
        layout.addLayout(self._create_spin_row("IoU 匹配阈值", self._ai_iou_threshold, 0.10, 0.60, 0.05, "_iou_threshold_spin"))
        layout.addLayout(self._create_spin_row("最大丢失帧数", self._ai_max_misses, 1, 10, 1, "_max_misses_spin"))
        layout.addLayout(self._create_spin_row("速度限幅", self._ai_velocity_clip, 0.10, 0.50, 0.05, "_velocity_clip_spin"))
        layout.addLayout(self._create_spin_row("级联高置信阈值", self._ai_high_conf_thresh, 0.30, 0.80, 0.05, "_high_conf_thresh_spin"))
        layout.addLayout(self._create_spin_row("重检测预算/帧", self._ai_redetect_budget, 1, 5, 1, "_redetect_budget_spin"))
        layout.addLayout(self._create_spin_row("ReID 最小 track 数", self._ai_reid_min_tracks, 2, 10, 1, "_reid_min_tracks_spin"))
        layout.addLayout(self._create_spin_row("ReID 跳帧", self._ai_reid_stride, 1, 10, 1, "_reid_stride_spin"))
        layout.addLayout(self._create_denoise_row())
        layout.addLayout(self._create_reset_ai_row())

        layout.addSpacing(6)

        model_title = QLabel("模型切换")
        model_title.setObjectName("title")
        layout.addWidget(model_title)

        model_subtitle = QLabel("切换成功后保存")
        model_subtitle.setObjectName("subtitle")
        layout.addWidget(model_subtitle)

        layout.addWidget(self._create_separator())
        layout.addLayout(self._create_model_row())
        layout.addLayout(self._create_head_classifier_row())

        layout.addSpacing(6)
        layout.addWidget(self._create_version_info())

        layout.addStretch(1)

        scroll.setWidget(content)
        root_layout.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        confirm_btn = QPushButton("确定")
        confirm_btn.setObjectName("primary")
        confirm_btn.clicked.connect(self._on_confirm)
        btn_row.addWidget(confirm_btn)

        root_layout.addLayout(btn_row)

    def _create_spin_row(self, label: str, value: float, min_v: float, max_v: float, step: float, attr_name: str) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        spin = WheelFocusDoubleSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(value)
        spin.setFixedHeight(34)
        setattr(self, attr_name, spin)
        row.addWidget(spin)
        return row

    def _create_spin_row_int(self, label: str, value: int, min_v: int, max_v: int, step: int, attr_name: str, suffix: str = "") -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        spin = WheelFocusSpinBox()
        spin.setRange(min_v, max_v)
        spin.setSingleStep(step)
        spin.setValue(value)
        spin.setFixedHeight(34)
        setattr(self, attr_name, spin)
        row.addWidget(spin)

        if suffix:
            suf = QLabel(suffix)
            suf.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 8pt;")
            row.addWidget(suf)
        return row

    def _create_model_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel("AI 检测模型")
        lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        self._model_combo = WheelFocusComboBox()
        self._model_combo.setFixedHeight(34)
        self._model_combo.addItem("自动 / 默认 YOLO 模型", "")
        for path in self._discover_model_paths():
            self._model_combo.addItem(path.name, str(path))
        # 固定 YOLO26 选项（按约定解析为 models/yolo26n.pt 和 models/yolo26n-obb.pt）
        self._model_combo.addItem("YOLO26 Detect", "__yolo26_detect__")
        self._model_combo.addItem("YOLO26 OBB", "__yolo26_obb__")

        if self._ai_model_path:
            model_path = self._ai_model_path
            # 若保存的是真实 YOLO26 路径，映射回固定选项标识符
            name = Path(model_path).name.lower()
            if name == "yolo26n.pt":
                model_path = "__yolo26_detect__"
            elif name == "yolo26n-obb.pt":
                model_path = "__yolo26_obb__"
            index = self._model_combo.findData(model_path)
            if index < 0:
                self._model_combo.addItem(Path(model_path).name, model_path)
                index = self._model_combo.findData(model_path)
            self._model_combo.setCurrentIndex(index)
        row.addWidget(self._model_combo, stretch=1)

        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse_model)
        row.addWidget(browse_btn)
        return row

    def _create_head_classifier_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel("头部属性")
        lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        self._head_classifier_combo = WheelFocusComboBox()
        self._head_classifier_combo.setFixedHeight(34)
        self._head_classifier_combo.addItem("关闭", "")
        for path in self._discover_head_classifier_paths():
            self._head_classifier_combo.addItem(path.name, str(path))

        if self._ai_head_classifier_path:
            index = self._head_classifier_combo.findData(self._ai_head_classifier_path)
            if index < 0:
                self._head_classifier_combo.addItem(Path(self._ai_head_classifier_path).name, self._ai_head_classifier_path)
                index = self._head_classifier_combo.findData(self._ai_head_classifier_path)
            self._head_classifier_combo.setCurrentIndex(index)
        row.addWidget(self._head_classifier_combo, stretch=1)

        stride_spin = WheelFocusSpinBox()
        stride_spin.setRange(1, 10)
        stride_spin.setValue(self._ai_head_classifier_stride)
        stride_spin.setFixedHeight(34)
        stride_spin.setFixedWidth(60)
        stride_spin.setToolTip("分类跳帧（每 N 帧一次），降低多目标场景下的延迟")
        self._head_classifier_stride_spin = stride_spin
        row.addWidget(stride_spin)

        browse_btn = QPushButton("浏览...")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._on_browse_head_classifier_model)
        row.addWidget(browse_btn)
        return row

    @staticmethod
    def _discover_head_classifier_paths() -> list[Path]:
        models_dir = Path.cwd() / "models"
        if not models_dir.exists():
            return []
        return sorted(models_dir.glob("chc_*.onnx"), key=lambda p: p.name.lower())

    def _on_browse_head_classifier_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择头部属性模型",
            str(Path.cwd() / "models"),
            "CHC Models (chc_*.onnx);;ONNX Models (*.onnx);;All Files (*)",
        )
        if file_path:
            index = self._head_classifier_combo.findData(file_path)
            if index < 0:
                self._head_classifier_combo.addItem(Path(file_path).name, file_path)
                index = self._head_classifier_combo.findData(file_path)
            self._head_classifier_combo.setCurrentIndex(index)

    def _create_filter_type_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel("滤波器类型")
        lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        self._filter_type_combo = WheelFocusComboBox()
        self._filter_type_combo.setFixedHeight(34)
        self._filter_type_combo.addItem("UKF（标准无迹卡尔曼）", "ukf")
        self._filter_type_combo.addItem("MCUKF（最大相关熵，抗离群）", "mcukf")
        self._filter_type_combo.addItem("Manifold UKF（流形，SPD 稳定）", "manifold_ukf")
        self._filter_type_combo.addItem("AUTO（自适应切换）", "auto")

        index = self._filter_type_combo.findData(self._ai_filter_type)
        if index >= 0:
            self._filter_type_combo.setCurrentIndex(index)
        row.addWidget(self._filter_type_combo, stretch=1)
        return row

    def _create_matching_strategy_row(self) -> QHBoxLayout:
        """匹配策略下拉框。"""
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel("匹配策略")
        lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        self._matching_strategy_combo = WheelFocusComboBox()
        self._matching_strategy_combo.setFixedHeight(34)
        self._matching_strategy_combo.addItem("匈牙利（全局最优）", "hungarian")
        self._matching_strategy_combo.addItem("级联（ByteTrack 两阶段）", "cascade")
        self._matching_strategy_combo.addItem("贪婪（原始，回退用）", "greedy")

        index = self._matching_strategy_combo.findData(self._ai_matching_strategy)
        if index >= 0:
            self._matching_strategy_combo.setCurrentIndex(index)
        row.addWidget(self._matching_strategy_combo, stretch=1)
        return row

    def _create_denoise_row(self) -> QHBoxLayout:
        """去噪方法下拉框 + 强度 SpinBox 组合行。"""
        row = QHBoxLayout()
        row.setSpacing(8)

        lbl = QLabel("图像去噪")
        lbl.setStyleSheet(f"color: {C['TEXT_SECONDARY']}; font-size: 9pt;")
        lbl.setFixedWidth(100)
        row.addWidget(lbl)

        self._denoise_combo = WheelFocusComboBox()
        self._denoise_combo.setFixedHeight(34)
        self._denoise_combo.addItem("关闭", "none")
        self._denoise_combo.addItem("双边滤波（快）", "bilateral")
        self._denoise_combo.addItem("NL-Means（高质量）", "nl_means")
        index = self._denoise_combo.findData(self._ai_denoise_method)
        if index >= 0:
            self._denoise_combo.setCurrentIndex(index)
        row.addWidget(self._denoise_combo, stretch=1)

        self._denoise_strength_spin = WheelFocusSpinBox()
        self._denoise_strength_spin.setRange(1, 10)
        self._denoise_strength_spin.setValue(self._ai_denoise_strength)
        self._denoise_strength_spin.setFixedHeight(34)
        self._denoise_strength_spin.setFixedWidth(60)
        row.addWidget(self._denoise_strength_spin)

        return row

    def _create_reset_ai_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addStretch(1)

        self._reset_ai_btn = QPushButton("恢复推荐参数")
        self._reset_ai_btn.setObjectName("secondary")
        self._reset_ai_btn.clicked.connect(self._on_reset_ai_params)
        row.addWidget(self._reset_ai_btn)
        return row

    def _create_version_info(self) -> QFrame:
        card = QFrame()
        card.setObjectName("versionInfo")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel("版本信息")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        version = QLabel(f"当前版本    {self._APP_VERSION}")
        version.setObjectName("versionValue")
        layout.addWidget(version)

        build = QLabel(f"最后编译    {self._BUILD_DATE}")
        build.setObjectName("versionValue")
        layout.addWidget(build)
        return card

    @staticmethod
    def _discover_model_paths() -> list[Path]:
        models_dir = Path.cwd() / "models"
        if not models_dir.exists():
            return []
        paths = []
        for pattern in ("*.pt", "*.onnx"):
            paths.extend(models_dir.glob(pattern))
        return sorted(paths, key=lambda p: p.name.lower())

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            f"QDialog {{ background-color: {C['BG_PRIMARY']}; color: {C['TEXT_PRIMARY']}; border-radius: 5px; }} "
            f"QLineEdit {{ background-color: {C['BG_TERTIARY']}; color: {C['TEXT_PRIMARY']}; "
            f"border: 1px solid {C['BORDER_LIGHT']}; border-radius: 5px; padding: 8px 12px; }} "
            f"QScrollArea#settingsScroll {{ background-color: transparent; border: none; }} "
            f"QWidget#settingsContent {{ background-color: transparent; }} "
            f"QFrame#versionInfo {{ background-color: {C['BG_TERTIARY']}; border: 1px solid {C['BORDER_LIGHT']}; border-radius: 5px; }} "
            f"QLabel#versionValue {{ color: {C['TEXT_SECONDARY']}; font-size: 8.5pt; }} "
            f"QComboBox, QSpinBox, QDoubleSpinBox {{ background-color: {C['BG_TERTIARY']}; color: {C['TEXT_PRIMARY']}; "
            f"border: 1px solid {C['BORDER_LIGHT']}; border-radius: 5px; padding: 6px 10px; }} "
            f"QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {C['ACCENT']}; }} "
            f"QComboBox::drop-down {{ width: 26px; border: none; }} "
            f"QComboBox QAbstractItemView {{ background-color: {C['BG_TERTIARY']}; color: {C['TEXT_PRIMARY']}; "
            f"selection-background-color: {C['ACCENT']}; selection-color: {C['BG_PRIMARY']}; "
            f"border: 1px solid {C['BORDER_LIGHT']}; outline: none; padding: 4px; }} "
            f"QComboBox QAbstractItemView::item {{ background-color: {C['BG_TERTIARY']}; color: {C['TEXT_PRIMARY']}; "
            f"padding: 7px 10px; min-height: 24px; }} "
            f"QComboBox QAbstractItemView::item:hover {{ background-color: {C['BG_ELEVATED']}; }} "
            f"QComboBox QAbstractItemView::item:selected {{ background-color: {C['ACCENT']}; color: {C['BG_PRIMARY']}; }} "
            f"QComboBox QAbstractScrollArea {{ background-color: {C['BG_TERTIARY']}; border: none; }} "
            f"QComboBox QAbstractScrollArea::corner {{ background-color: {C['BG_TERTIARY']}; }} "
            f"QPushButton {{ background-color: {C['BG_TERTIARY']}; color: {C['TEXT_PRIMARY']}; "
            f"border: 1px solid {C['BORDER_LIGHT']}; border-radius: 5px; padding: 8px 16px; "
            f"font-weight: 500; font-size: 9pt; }} "
            f"QPushButton:hover {{ background-color: {C['BG_ELEVATED']}; border-color: {C['ACCENT']}; color: {C['ACCENT_HOVER']}; }} "
            f"QPushButton:pressed {{ background-color: {C['ACCENT_PRESSED']}; color: {C['BG_PRIMARY']}; border-color: {C['ACCENT_PRESSED']}; }} "
            f"QPushButton#primary {{ background-color: {C['ACCENT']}; color: {C['BG_PRIMARY']}; border: none; font-weight: 600; }} "
            f"QPushButton#primary:hover {{ background-color: {C['ACCENT_HOVER']}; }} "
            f"QPushButton#primary:pressed {{ background-color: {C['ACCENT_PRESSED']}; }}"
        )
        self._apply_model_combo_palette()

    def _apply_model_combo_palette(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Base, QColor(C["BG_TERTIARY"]))
        palette.setColor(QPalette.ColorRole.Window, QColor(C["BG_TERTIARY"]))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(C["TEXT_PRIMARY"]))
        palette.setColor(QPalette.ColorRole.Text, QColor(C["TEXT_PRIMARY"]))
        palette.setColor(QPalette.ColorRole.Button, QColor(C["BG_TERTIARY"]))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(C["TEXT_PRIMARY"]))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(C["ACCENT"]))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(C["BG_PRIMARY"]))

        view = self._model_combo.view()
        view.setPalette(palette)
        view.setAutoFillBackground(True)
        popup = view.parentWidget()
        if popup is not None:
            popup.setPalette(palette)
            popup.setAutoFillBackground(True)

    def _create_separator(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet(f"background-color: {C['BORDER']}; max-height: 1px;")
        return line

    def _on_browse(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择录像存储文件夹",
            self._current_dir,
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self._current_dir = directory
            self._path_edit.setText(directory)
            self._preview_label.setText(f"当前路径：{directory}")

    def _on_browse_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择 AI 检测模型",
            str(Path.cwd() / "models"),
            "AI Models (*.pt *.onnx);;All Files (*)",
        )
        if file_path:
            index = self._model_combo.findData(file_path)
            if index < 0:
                self._model_combo.addItem(Path(file_path).name, file_path)
                index = self._model_combo.findData(file_path)
            self._model_combo.setCurrentIndex(index)

    def _on_reset_ai_params(self) -> None:
        if not self._reset_confirming:
            self._reset_confirming = True
            self._reset_ai_btn.setText("再次点击确认")
            self._reset_confirm_timer.start(2000)
            return

        self._cancel_reset_confirm()
        self._restore_recommended_ai_params()

    def _cancel_reset_confirm(self) -> None:
        self._reset_confirming = False
        if hasattr(self, "_reset_ai_btn"):
            self._reset_ai_btn.setText("恢复推荐参数")

    def _restore_recommended_ai_params(self) -> None:
        self._conf_spin.setValue(self._RECOMMENDED_CONF)
        self._stride_spin.setValue(self._RECOMMENDED_STRIDE)
        self._smooth_spin.setValue(self._RECOMMENDED_SMOOTH)
        self._hide_spin.setValue(self._RECOMMENDED_HIDE_MS)
        self._redetect_retries_spin.setValue(self._RECOMMENDED_REDETECT_RETRIES)
        self._roi_pad_spin.setValue(self._RECOMMENDED_REDETECT_ROI_PAD)
        idx = self._filter_type_combo.findData(self._RECOMMENDED_FILTER_TYPE)
        if idx >= 0:
            self._filter_type_combo.setCurrentIndex(idx)
        self._mcukf_sigma_spin.setValue(self._RECOMMENDED_MCUKF_KERNEL_SIGMA)
        idx_dn = self._denoise_combo.findData(self._RECOMMENDED_DENOISE_METHOD)
        if idx_dn >= 0:
            self._denoise_combo.setCurrentIndex(idx_dn)
        self._denoise_strength_spin.setValue(self._RECOMMENDED_DENOISE_STRENGTH)
        idx_hc = self._head_classifier_combo.findData("")
        if idx_hc >= 0:
            self._head_classifier_combo.setCurrentIndex(idx_hc)
        self._head_classifier_stride_spin.setValue(self._RECOMMENDED_HEAD_CLASSIFIER_STRIDE)

    def _on_confirm(self) -> None:
        self._save_settings(self._current_dir)
        self._save_ai_settings(
            self._conf_spin.value(),
            self._stride_spin.value(),
            self._smooth_spin.value(),
            self._hide_spin.value(),
            self._redetect_retries_spin.value(),
            self._roi_pad_spin.value(),
            self._filter_type_combo.currentData() or "ukf",
            self._mcukf_sigma_spin.value(),
            self._denoise_combo.currentData() or "none",
            self._denoise_strength_spin.value(),
        )
        self._save_head_classifier_settings(
            bool(self._head_classifier_combo.currentData()),
            self._head_classifier_combo.currentData() or "",
            self._head_classifier_stride_spin.value(),
        )
        self._animate_close("accept")

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._open_animation = AnimationHelper.dialog_open(self, parent=self)
        self._open_animation.start()

    def reject(self) -> None:
        self._animate_close("reject")

    def _animate_close(self, decision: str) -> None:
        if self._closing_decision is not None:
            return
        self._closing_decision = decision
        self._close_animation = AnimationHelper.dialog_close(self, parent=self)

        def _finalize() -> None:
            if self._closing_decision == "accept":
                super(SettingsDialog, self).accept()
            else:
                super(SettingsDialog, self).reject()

        self._close_animation.finished.connect(_finalize)
        self._close_animation.start()

    def get_output_dir(self) -> str:
        return self._current_dir

    def get_ai_settings(self) -> dict:
        return {
            "conf": self._conf_spin.value(),
            "stride": self._stride_spin.value(),
            "smooth": self._smooth_spin.value(),
            "hide_ms": self._hide_spin.value(),
            "model_path": self._model_combo.currentData() or "",
            "person_redetect_retries": int(self._redetect_retries_spin.value()),
            "person_redetect_roi_pad": float(self._roi_pad_spin.value()),
            "filter_type": self._filter_type_combo.currentData() or "ukf",
            "mcukf_kernel_sigma": float(self._mcukf_sigma_spin.value()),
            "denoise_method": self._denoise_combo.currentData() or "none",
            "denoise_strength": int(self._denoise_strength_spin.value()),
            "matching_strategy": self._matching_strategy_combo.currentData() or "hungarian",
            "iou_threshold": float(self._iou_threshold_spin.value()),
            "max_misses": int(self._max_misses_spin.value()),
            "velocity_clip": float(self._velocity_clip_spin.value()),
            "high_conf_thresh": float(self._high_conf_thresh_spin.value()),
            "redetect_budget": int(self._redetect_budget_spin.value()),
            "reid_min_tracks": int(self._reid_min_tracks_spin.value()),
            "reid_stride": int(self._reid_stride_spin.value()),
            "head_classifier_enabled": bool(self._head_classifier_combo.currentData()),
            "head_classifier_path": self._head_classifier_combo.currentData() or "",
            "head_classifier_stride": int(self._head_classifier_stride_spin.value()),
        }

    def _save_settings(self, path: str) -> None:
        settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        settings.setValue(self._KEY_OUTPUT_DIR, path)
        logging.info("SettingsDialog: saved output dir = %s", path)

    def _save_ai_settings(
        self,
        conf: float,
        stride: int,
        smooth: float,
        hide_ms: int,
        redetect_retries: int = 2,
        redetect_roi_pad: float = 0.15,
        filter_type: str = "ukf",
        mcukf_kernel_sigma: float = 0.4,
        denoise_method: str = "none",
        denoise_strength: int = 5,
        matching_strategy: str = "hungarian",
        iou_threshold: float = 0.25,
        max_misses: int = 4,
        velocity_clip: float = 0.30,
        high_conf_thresh: float = 0.5,
        redetect_budget: int = 1,
        reid_min_tracks: int = 3,
        reid_stride: int = 3,
    ) -> None:
        settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        settings.setValue(self._KEY_AI_CONF, conf)
        settings.setValue(self._KEY_AI_STRIDE, stride)
        settings.setValue(self._KEY_AI_SMOOTH, smooth)
        settings.setValue(self._KEY_AI_HIDE_MS, hide_ms)
        settings.setValue(self._KEY_AI_PERSON_REDETECT_RETRIES, int(redetect_retries))
        settings.setValue(self._KEY_AI_PERSON_REDETECT_ROI_PAD, float(redetect_roi_pad))
        settings.setValue(self._KEY_AI_FILTER_TYPE, str(filter_type))
        settings.setValue(self._KEY_AI_MCUKF_KERNEL_SIGMA, float(mcukf_kernel_sigma))
        settings.setValue(self._KEY_AI_DENOISE_METHOD, str(denoise_method))
        settings.setValue(self._KEY_AI_DENOISE_STRENGTH, int(denoise_strength))
        settings.setValue(self._KEY_AI_MATCHING_STRATEGY, str(matching_strategy))
        settings.setValue(self._KEY_AI_IOU_THRESHOLD, float(iou_threshold))
        settings.setValue(self._KEY_AI_MAX_MISSES, int(max_misses))
        settings.setValue(self._KEY_AI_VELOCITY_CLIP, float(velocity_clip))
        settings.setValue(self._KEY_AI_HIGH_CONF_THRESH, float(high_conf_thresh))
        settings.setValue(self._KEY_AI_REDETECT_BUDGET, int(redetect_budget))
        settings.setValue(self._KEY_AI_REID_MIN_TRACKS, int(reid_min_tracks))
        settings.setValue(self._KEY_AI_REID_STRIDE, int(reid_stride))
        logging.info(
            "SettingsDialog: saved AI settings conf=%.2f stride=%d smooth=%.2f hide=%d redetect=%d roi_pad=%.2f filter=%s sigma=%.2f denoise=%s str=%d matching=%s iou=%.2f misses=%d",
            conf, stride, smooth, hide_ms, redetect_retries, redetect_roi_pad, filter_type, mcukf_kernel_sigma, denoise_method, denoise_strength, matching_strategy, iou_threshold, max_misses,
        )

    def _save_head_classifier_settings(
        self,
        enabled: bool,
        model_path: str,
        stride: int,
    ) -> None:
        settings = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        settings.setValue(self._KEY_AI_HEAD_CLASSIFIER_ENABLED, bool(enabled))
        settings.setValue(self._KEY_AI_HEAD_CLASSIFIER_PATH, str(model_path or ""))
        settings.setValue(self._KEY_AI_HEAD_CLASSIFIER_STRIDE, max(1, int(stride)))
        logging.info(
            "SettingsDialog: saved head classifier enabled=%s path=%s stride=%d",
            enabled, model_path or "<auto>", stride,
        )

    @staticmethod
    def save_ai_model_path(model_path: str) -> None:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        settings.setValue(SettingsDialog._KEY_AI_MODEL_PATH, model_path)
        logging.info("SettingsDialog: saved AI model path = %s", model_path or "<auto>")

    @staticmethod
    def load_output_dir() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        saved = settings.value(SettingsDialog._KEY_OUTPUT_DIR, "")
        if saved and Path(saved).is_absolute():
            return str(saved)
        return SettingsDialog._DEFAULT_DIR

    @staticmethod
    def load_ai_conf() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_CONF, SettingsDialog._DEFAULT_CONF)
        try:
            return float(val)
        except Exception:
            return SettingsDialog._DEFAULT_CONF

    @staticmethod
    def load_ai_stride() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_STRIDE, SettingsDialog._DEFAULT_STRIDE)
        try:
            return int(val)
        except Exception:
            return SettingsDialog._DEFAULT_STRIDE

    @staticmethod
    def load_ai_smooth() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_SMOOTH, SettingsDialog._DEFAULT_SMOOTH)
        try:
            return float(val)
        except Exception:
            return SettingsDialog._DEFAULT_SMOOTH

    @staticmethod
    def load_ai_hide_ms() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_HIDE_MS, SettingsDialog._DEFAULT_HIDE_MS)
        try:
            return int(val)
        except Exception:
            return SettingsDialog._DEFAULT_HIDE_MS

    @staticmethod
    def load_ai_person_redetect_retries() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_PERSON_REDETECT_RETRIES, SettingsDialog._RECOMMENDED_REDETECT_RETRIES)
        try:
            return int(val)
        except Exception:
            return SettingsDialog._RECOMMENDED_REDETECT_RETRIES

    @staticmethod
    def load_ai_person_redetect_roi_pad() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_PERSON_REDETECT_ROI_PAD, SettingsDialog._RECOMMENDED_REDETECT_ROI_PAD)
        try:
            return float(val)
        except Exception:
            return SettingsDialog._RECOMMENDED_REDETECT_ROI_PAD

    @staticmethod
    def load_ai_filter_type() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_FILTER_TYPE, SettingsDialog._RECOMMENDED_FILTER_TYPE)
        ft = str(val or "ukf")
        return ft if ft in ("ukf", "mcukf", "manifold_ukf", "auto") else "ukf"

    @staticmethod
    def load_ai_mcukf_kernel_sigma() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_MCUKF_KERNEL_SIGMA, SettingsDialog._RECOMMENDED_MCUKF_KERNEL_SIGMA)
        try:
            return float(val)
        except Exception:
            return SettingsDialog._RECOMMENDED_MCUKF_KERNEL_SIGMA

    @staticmethod
    def load_ai_denoise_method() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_DENOISE_METHOD, SettingsDialog._RECOMMENDED_DENOISE_METHOD)
        dm = str(val or "none")
        return dm if dm in ("none", "bilateral", "nl_means") else "none"

    @staticmethod
    def load_ai_denoise_strength() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_DENOISE_STRENGTH, SettingsDialog._RECOMMENDED_DENOISE_STRENGTH)
        try:
            return max(1, min(10, int(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_DENOISE_STRENGTH

    @staticmethod
    def load_ai_model_path() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_MODEL_PATH, "")
        return str(val or "")

    @staticmethod
    def load_ai_matching_strategy() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_MATCHING_STRATEGY, SettingsDialog._RECOMMENDED_MATCHING_STRATEGY)
        s = str(val or "hungarian")
        return s if s in ("greedy", "hungarian", "cascade") else "hungarian"

    @staticmethod
    def load_ai_iou_threshold() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_IOU_THRESHOLD, SettingsDialog._RECOMMENDED_IOU_THRESHOLD)
        try:
            return max(0.10, min(0.60, float(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_IOU_THRESHOLD

    @staticmethod
    def load_ai_max_misses() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_MAX_MISSES, SettingsDialog._RECOMMENDED_MAX_MISSES)
        try:
            return max(1, min(10, int(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_MAX_MISSES

    @staticmethod
    def load_ai_velocity_clip() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_VELOCITY_CLIP, SettingsDialog._RECOMMENDED_VELOCITY_CLIP)
        try:
            return max(0.10, min(0.50, float(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_VELOCITY_CLIP

    @staticmethod
    def load_ai_high_conf_thresh() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_HIGH_CONF_THRESH, SettingsDialog._RECOMMENDED_HIGH_CONF_THRESH)
        try:
            return max(0.30, min(0.80, float(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_HIGH_CONF_THRESH

    @staticmethod
    def load_ai_redetect_budget() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_REDETECT_BUDGET, SettingsDialog._RECOMMENDED_REDETECT_BUDGET)
        try:
            return max(1, min(5, int(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_REDETECT_BUDGET

    @staticmethod
    def load_ai_reid_enabled() -> bool:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_REID_ENABLED, SettingsDialog._RECOMMENDED_REID_ENABLED)
        return bool(val)

    @staticmethod
    def load_ai_reid_min_tracks() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_REID_MIN_TRACKS, SettingsDialog._RECOMMENDED_REID_MIN_TRACKS)
        try:
            return max(2, min(10, int(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_REID_MIN_TRACKS

    @staticmethod
    def load_ai_reid_stride() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_REID_STRIDE, SettingsDialog._RECOMMENDED_REID_STRIDE)
        try:
            return max(1, min(10, int(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_REID_STRIDE

    @staticmethod
    def load_ai_head_classifier_enabled() -> bool:
        return SettingsDialog._load_bool(SettingsDialog._KEY_AI_HEAD_CLASSIFIER_ENABLED, SettingsDialog._RECOMMENDED_HEAD_CLASSIFIER_ENABLED)

    @staticmethod
    def load_ai_head_classifier_path() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_HEAD_CLASSIFIER_PATH, "")
        return str(val or "")

    @staticmethod
    def load_ai_head_classifier_stride() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_HEAD_CLASSIFIER_STRIDE, SettingsDialog._RECOMMENDED_HEAD_CLASSIFIER_STRIDE)
        try:
            return max(1, min(10, int(val)))
        except Exception:
            return SettingsDialog._RECOMMENDED_HEAD_CLASSIFIER_STRIDE

    @staticmethod
    def load_ai_rectangle_enabled() -> bool:
        return SettingsDialog._load_bool(SettingsDialog._KEY_AI_RECTANGLE_ENABLED, False)

    @staticmethod
    def save_ai_rectangle_enabled(enabled: bool) -> None:
        SettingsDialog._save_bool(SettingsDialog._KEY_AI_RECTANGLE_ENABLED, enabled)

    @staticmethod
    def load_ai_rectangle_sensitivity() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = str(settings.value(SettingsDialog._KEY_AI_RECTANGLE_SENSITIVITY, "low") or "low").lower()
        return val if val in {"low", "medium", "high"} else "low"

    @staticmethod
    def save_ai_rectangle_sensitivity(level: str) -> None:
        normalized = str(level or "low").lower()
        if normalized not in {"low", "medium", "high"}:
            normalized = "low"
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        settings.setValue(SettingsDialog._KEY_AI_RECTANGLE_SENSITIVITY, normalized)

    @staticmethod
    def load_ai_rectangle_max_count() -> int:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_RECTANGLE_MAX_COUNT, 3)
        try:
            return max(1, min(10, int(val)))
        except Exception:
            return 3

    @staticmethod
    def save_ai_rectangle_max_count(count: int) -> None:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        settings.setValue(SettingsDialog._KEY_AI_RECTANGLE_MAX_COUNT, max(1, min(10, int(count))))

    @staticmethod
    def load_ai_rectangle_target_rgb() -> tuple[int, int, int] | None:
        return SettingsDialog._load_color_tuple(SettingsDialog._KEY_AI_RECTANGLE_TARGET_RGB)

    @staticmethod
    def load_ai_rectangle_target_hsv() -> tuple[int, int, int] | None:
        return SettingsDialog._load_hsv_tuple(SettingsDialog._KEY_AI_RECTANGLE_TARGET_HSV)

    @staticmethod
    def save_ai_rectangle_target_color(rgb: tuple[int, int, int] | None, hsv: tuple[int, int, int] | None) -> None:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        if rgb is None:
            settings.remove(SettingsDialog._KEY_AI_RECTANGLE_TARGET_RGB)
            settings.remove(SettingsDialog._KEY_AI_RECTANGLE_TARGET_HSV)
            return
        settings.setValue(SettingsDialog._KEY_AI_RECTANGLE_TARGET_RGB, SettingsDialog._format_color_tuple(rgb))
        if hsv is not None:
            settings.setValue(SettingsDialog._KEY_AI_RECTANGLE_TARGET_HSV, SettingsDialog._format_hsv_tuple(hsv))

    @staticmethod
    def load_ai_rectangle_color_threshold() -> float:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(SettingsDialog._KEY_AI_RECTANGLE_COLOR_THRESHOLD, 30.0)
        try:
            return max(1.0, min(100.0, float(val)))
        except Exception:
            return 30.0

    @staticmethod
    def save_ai_rectangle_color_threshold(threshold: float) -> None:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        settings.setValue(SettingsDialog._KEY_AI_RECTANGLE_COLOR_THRESHOLD, max(1.0, min(100.0, float(threshold))))

    @staticmethod
    def load_ai_person_enabled() -> bool:
        return SettingsDialog._load_bool(SettingsDialog._KEY_AI_PERSON_ENABLED, True)

    @staticmethod
    def save_ai_person_enabled(enabled: bool) -> None:
        SettingsDialog._save_bool(SettingsDialog._KEY_AI_PERSON_ENABLED, enabled)

    @staticmethod
    def load_ai_skeleton_enabled() -> bool:
        return SettingsDialog._load_bool(SettingsDialog._KEY_AI_SKELETON_ENABLED, True)

    @staticmethod
    def save_ai_skeleton_enabled(enabled: bool) -> None:
        SettingsDialog._save_bool(SettingsDialog._KEY_AI_SKELETON_ENABLED, enabled)

    @staticmethod
    def load_ai_gesture_enabled() -> bool:
        return SettingsDialog._load_bool(SettingsDialog._KEY_AI_GESTURE_ENABLED, False)

    @staticmethod
    def save_ai_gesture_enabled(enabled: bool) -> None:
        SettingsDialog._save_bool(SettingsDialog._KEY_AI_GESTURE_ENABLED, enabled)

    @staticmethod
    def load_ai_gesture_mode() -> str:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        raw = settings.value(SettingsDialog._KEY_AI_GESTURE_MODE, None)
        if raw is None:
            return "body" if SettingsDialog.load_ai_gesture_enabled() else "off"
        mode = str(raw or "off").lower()
        return mode if mode in {"off", "body", "hand", "all"} else "off"

    @staticmethod
    def save_ai_gesture_mode(mode: str) -> None:
        normalized = str(mode or "off").lower()
        if normalized not in {"off", "body", "hand", "all"}:
            normalized = "off"
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        settings.setValue(SettingsDialog._KEY_AI_GESTURE_MODE, normalized)
        settings.setValue(SettingsDialog._KEY_AI_GESTURE_ENABLED, normalized in {"body", "all"})

    @staticmethod
    def _load_bool(key: str, default: bool) -> bool:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        val = settings.value(key, default)
        if isinstance(val, bool):
            return val
        return str(val).lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _save_bool(key: str, enabled: bool) -> None:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        settings.setValue(key, bool(enabled))

    @staticmethod
    def load_pipeline_enabled() -> bool:
        """加载 Pipeline 模式开关（C8）。默认 False（Legacy Mode）。"""
        return SettingsDialog._load_bool(
            SettingsDialog._KEY_PIPELINE_ENABLED, False
        )

    @staticmethod
    def save_pipeline_enabled(enabled: bool) -> None:
        """持久化 Pipeline 模式开关（C8）。"""
        SettingsDialog._save_bool(
            SettingsDialog._KEY_PIPELINE_ENABLED, bool(enabled)
        )

    @staticmethod
    def _load_color_tuple(key: str) -> tuple[int, int, int] | None:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        raw = str(settings.value(key, "") or "")
        parts = raw.split(",")
        if len(parts) != 3:
            return None
        try:
            return tuple(max(0, min(255, int(part))) for part in parts)
        except Exception:
            return None

    @staticmethod
    def _load_hsv_tuple(key: str) -> tuple[int, int, int] | None:
        settings = QSettings(SettingsDialog._SETTINGS_ORG, SettingsDialog._SETTINGS_APP)
        raw = str(settings.value(key, "") or "")
        parts = raw.split(",")
        if len(parts) != 3:
            return None
        try:
            h, s, v = (int(part) for part in parts)
            return (max(0, min(359, h)), max(0, min(255, s)), max(0, min(255, v)))
        except Exception:
            return None

    @staticmethod
    def _format_color_tuple(values: tuple[int, int, int]) -> str:
        return ",".join(str(max(0, min(255, int(value)))) for value in values)

    @staticmethod
    def _format_hsv_tuple(values: tuple[int, int, int]) -> str:
        h, s, v = values
        return f"{max(0, min(359, int(h)))},{max(0, min(255, int(s)))},{max(0, min(255, int(v)))}"
