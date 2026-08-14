from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import json
import time
import urllib.request

from PyQt6.QtCore import QEasingCurve, QParallelAnimationGroup, QPropertyAnimation, Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QColor, QPalette
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ai.inference import InferWorker
from camera.capture import CameraCapture
from camera.enumerator import CameraInfo, scan_cameras
from camera.recorder import VideoRecorder
from gui.camera_cell import CameraCell
from gui.animations import AnimationHelper, Easing
from gui.safe_widgets import WheelFocusComboBox, WheelFocusDoubleSpinBox, WheelFocusSpinBox
from gui.settings_dialog import SettingsDialog
from gui.theme import get_stylesheet

if TYPE_CHECKING:
    # VisionCore 未来统一模型入口 —— 当前仅用于类型检查，不影响运行时。
    # MainWindow 的运行逻辑保持不变；这些类型为未来迁移至 VisionCore 架构做准备。
    from visioncore.core import Detection as CoreDetection
    from visioncore.core import Event as CoreEvent
    from visioncore.core import Frame as CoreFrame
    from visioncore.core import Target as CoreTarget
    from visioncore.core import Track as CoreTrack


def _debug_report(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    try:
        debug_server_url = "http://127.0.0.1:7777/event"
        debug_session_id = "yolo-detection-overlay"
        with open(".dbg/yolo-detection-overlay.env", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if line.startswith("DEBUG_SERVER_URL="):
                    debug_server_url = line.split("=", 1)[1]
                elif line.startswith("DEBUG_SESSION_ID="):
                    debug_session_id = line.split("=", 1)[1]
        urllib.request.urlopen(
            urllib.request.Request(
                debug_server_url,
                data=json.dumps(
                    {
                        "sessionId": debug_session_id,
                        "runId": "post-fix",
                        "hypothesisId": hypothesis_id,
                        "location": location,
                        "msg": f"[DEBUG] {msg}",
                        "data": data,
                        "ts": int(time.time() * 1000),
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.5,
        ).read()
    except Exception:
        pass


class MainWindow(QMainWindow):

    _MAX_SLOTS = 4

    @staticmethod
    def _resolve_special_model_path(path: str) -> str:
        """把 YOLO26 固定选项标识符解析为真实模型路径。"""
        if path == "__yolo26_detect__":
            return str(Path.cwd() / "models" / "yolo26n.pt")
        if path == "__yolo26_obb__":
            return str(Path.cwd() / "models" / "yolo26n-obb.pt")
        return path

    @staticmethod
    def _reverse_resolve_special_model_path(path: str) -> str:
        """把真实模型路径反向映射为固定选项标识符（用于持久化）。"""
        if not path:
            return path
        name = Path(path).name.lower()
        if name == "yolo26n.pt":
            return "__yolo26_detect__"
        if name == "yolo26n-obb.pt":
            return "__yolo26_obb__"
        return path

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VDP V1.1")
        self.setMinimumSize(1360, 820)

        self._is_recording = False
        self._is_ai_enabled = False
        self._cells: dict[int, CameraCell] = {}
        self._combos: dict[int, QComboBox] = {}
        self._camera_list: list[CameraInfo] = []
        self._sidebar_animations: list[QPropertyAnimation] = []
        self._ui_feedback_animations: list[object] = []
        self._sidebar_sections: list[QWidget] = []
        self._debug_connect_reports = 0
        self._first_show_done = False
        self._expanded_slot_id: int | None = None
        self._grid_layout: QGridLayout | None = None
        self._is_frame_flipped = True
        self._ai_feature_animation: QPropertyAnimation | None = None
        self._gesture_options_animation: QParallelAnimationGroup | None = None
        self._rectangle_options_animation: QParallelAnimationGroup | None = None

        self._recording_output_dir = SettingsDialog.load_output_dir()
        self._ai_conf = SettingsDialog.load_ai_conf()
        self._ai_stride = SettingsDialog.load_ai_stride()
        self._ai_smooth = SettingsDialog.load_ai_smooth()
        self._ai_hide_ms = SettingsDialog.load_ai_hide_ms()
        self._ai_model_path = SettingsDialog.load_ai_model_path()
        self._ai_person_enabled = SettingsDialog.load_ai_person_enabled()
        self._ai_skeleton_enabled = SettingsDialog.load_ai_skeleton_enabled()
        self._ai_gesture_mode = SettingsDialog.load_ai_gesture_mode()
        self._ai_rectangle_enabled = SettingsDialog.load_ai_rectangle_enabled()
        self._ai_rectangle_sensitivity = SettingsDialog.load_ai_rectangle_sensitivity()
        self._ai_rectangle_max_count = SettingsDialog.load_ai_rectangle_max_count()
        self._ai_rectangle_target_rgb = SettingsDialog.load_ai_rectangle_target_rgb()
        self._ai_rectangle_target_hsv = SettingsDialog.load_ai_rectangle_target_hsv()
        self._ai_rectangle_color_threshold = SettingsDialog.load_ai_rectangle_color_threshold()
        self._ai_head_classifier_enabled = SettingsDialog.load_ai_head_classifier_enabled()
        self._ai_head_classifier_path = SettingsDialog.load_ai_head_classifier_path()
        self._ai_head_classifier_stride = SettingsDialog.load_ai_head_classifier_stride()
        self._pending_ai_model_path: str | None = None

        self._camera_capture = CameraCapture()
        self._video_recorder = VideoRecorder(output_dir=self._recording_output_dir)
        try:
            resolved_model = self._resolve_special_model_path(self._ai_model_path)
            self._infer_worker = InferWorker(model_path=resolved_model or None, conf=self._ai_conf, infer_stride=self._ai_stride)
        except FileNotFoundError as err:
            logging.warning("Saved AI model is unavailable, falling back to auto model: %s", err)
            self._ai_model_path = ""
            SettingsDialog.save_ai_model_path("")
            self._infer_worker = InferWorker(conf=self._ai_conf, infer_stride=self._ai_stride)
        self._apply_feature_flags_to_worker()
        # C8: 从 QSettings 加载 Pipeline 模式开关（默认 False = Legacy）
        self._infer_worker.set_pipeline_enabled(
            SettingsDialog.load_pipeline_enabled()
        )
        self._infer_worker.set_head_classifier(
            self._ai_head_classifier_enabled,
            self._ai_head_classifier_path or None,
            self._ai_head_classifier_stride,
        )
        self._infer_worker.model_ready.connect(self._on_infer_model_ready)
        self._infer_worker.set_rectangle_sensitivity(self._ai_rectangle_sensitivity)
        self._infer_worker.set_rectangle_max_count(self._ai_rectangle_max_count)
        self._infer_worker.set_rectangle_target_color(self._ai_rectangle_target_rgb)
        self._infer_worker.set_rectangle_color_threshold(self._ai_rectangle_color_threshold)

        self._camera_list = scan_cameras()

        self._build_ui()

        self.setStyleSheet(get_stylesheet())
        self._apply_combo_dark_palette()
        self._refresh_dashboard("状态：待机")

        self._infer_worker.start()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._first_show_done:
            self._first_show_done = True
            QTimer.singleShot(60, self._animate_sidebar_sections)

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 固定顶部栏：不随侧边栏/网格滚动，最右侧预留设置入口
        top_bar = self._build_top_bar()
        root_layout.addWidget(top_bar)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        control_panel = self._build_control_panel()
        main_layout.addWidget(control_panel)

        grid_widget = self._build_grid_area()
        main_layout.addWidget(grid_widget, stretch=1)

        root_layout.addLayout(main_layout, stretch=1)

    def _build_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("topBar")
        bar.setFixedHeight(52)

        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 16, 0)
        layout.setSpacing(12)

        brand = QLabel("VDP V1.1")
        brand.setObjectName("topBarBrand")
        layout.addWidget(brand)

        layout.addStretch(1)

        self._flip_toggle = QCheckBox("画面翻转")
        self._flip_toggle.setObjectName("topBarFlip")
        self._flip_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._flip_toggle.setToolTip("水平镜像翻转全部画面")
        self._flip_toggle.setChecked(self._is_frame_flipped)
        self._flip_toggle.toggled.connect(self._on_flip_toggled)
        layout.addWidget(self._flip_toggle)

        # 设置入口：具体内容暂不确定，先预留按钮与回调骨架
        self._settings_btn = QPushButton("设置")
        self._settings_btn.setObjectName("topBarSettings")
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_btn.setToolTip("设置（即将上线）")
        self._settings_btn.clicked.connect(self._on_open_settings)
        layout.addWidget(self._settings_btn)

        return bar

    def _on_open_settings(self) -> None:
        self._play_feedback_animation(AnimationHelper.bass_pluck(self._settings_btn, parent=self))
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_dir = dialog.get_output_dir()
            self._recording_output_dir = new_dir
            self._video_recorder.set_output_dir(new_dir)
            self._storage_path_label.setText(new_dir)
            ai_cfg = dialog.get_ai_settings()
            self._apply_ai_settings(ai_cfg)
            self._refresh_dashboard("设置已更新")
        else:
            self._refresh_dashboard("已取消设置修改")

    def _on_flip_toggled(self, checked: bool) -> None:
        self._is_frame_flipped = checked
        for cell in self._cells.values():
            cell.set_frame_flipped(checked)
        self._refresh_dashboard("画面已水平翻转" if checked else "画面已恢复")

    def _build_control_panel(self) -> QWidget:
        outer = QWidget()
        outer.setObjectName("sidePanel")
        outer.setMinimumWidth(340)
        outer.setMaximumWidth(388)

        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("panelScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("panelContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(20, 20, 20, 24)
        layout.setSpacing(18)

        # hero 卡片与胶囊都放在滚动区内，与其他卡片一起滚动，融合在侧边栏里
        hero_card = self._build_overview_card()
        layout.addWidget(hero_card)
        self._sidebar_sections.append(hero_card)

        devices_card = self._build_devices_card()
        layout.addWidget(devices_card)
        self._sidebar_sections.append(devices_card)

        runtime_card = self._build_runtime_card()
        layout.addWidget(runtime_card)
        self._sidebar_sections.append(runtime_card)

        storage_card = self._build_storage_card()
        layout.addWidget(storage_card)
        self._sidebar_sections.append(storage_card)

        # 系统状态栏已删除：与 hero 卡的指标胶囊重复
        # 状态文本（_status_label）保留内部对象引用，避免其他位置引用空指针

        layout.addStretch(1)

        scroll.setWidget(content)
        outer_layout.addWidget(scroll, stretch=1)
        return outer

    def _build_overview_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("heroCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        title = QLabel("VDP V1.1")
        title.setObjectName("title")
        layout.addWidget(title)

        quote = QLabel("更清晰地接入设备、更稳定地管理采集、更直观地掌握状态。")
        quote.setObjectName("quote")
        quote.setWordWrap(True)
        layout.addWidget(quote)

        pills_row = QHBoxLayout()
        pills_row.setSpacing(8)

        self._active_pill = self._create_metric_pill("在线 0")
        self._mode_pill = self._create_metric_pill("AI 关闭")
        pills_row.addWidget(self._active_pill)
        pills_row.addWidget(self._mode_pill)

        layout.addLayout(pills_row)
        return card

    def _build_devices_card(self) -> QFrame:
        card, body = self._create_section_card("设备接入", "为每个采集位分配输入源")

        for slot_id in range(self._MAX_SLOTS):
            body.addWidget(self._build_slot_group(slot_id))

        return card

    def _build_runtime_card(self) -> QFrame:
        card, body = self._create_section_card("运行控制", "围绕检测、录制与设备刷新提供集中操作")

        self._ai_toggle = QCheckBox("启用 AI 检测引擎")
        self._ai_toggle.setObjectName("toggleSwitch")
        self._ai_toggle.toggled.connect(self._on_ai_toggled)
        body.addWidget(self._ai_toggle)

        self._ai_feature_panel = self._build_ai_feature_panel()
        body.addWidget(self._ai_feature_panel)

        self._record_btn = QPushButton("开始录制")
        self._record_btn.setObjectName("primary")
        self._record_btn.clicked.connect(self._on_record_toggled)
        body.addWidget(self._record_btn)

        self._refresh_btn = QPushButton("刷新摄像头")
        self._refresh_btn.setObjectName("secondary")
        self._refresh_btn.clicked.connect(self._on_refresh_cameras)
        body.addWidget(self._refresh_btn)

        return card

    def _build_ai_feature_panel(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("aiFeaturePanel")
        panel.setVisible(False)
        panel.setMaximumHeight(0)

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        self._person_toggle = self._create_feature_toggle("人体识别", self._ai_person_enabled)
        self._skeleton_toggle = self._create_feature_toggle("人体骨骼绑定", self._ai_skeleton_enabled and self._ai_person_enabled)
        self._gesture_options_panel = QWidget()
        self._gesture_options_panel.setObjectName("gestureOptionsPanel")
        self._gesture_options_panel.setVisible(False)
        self._gesture_options_panel.setMaximumHeight(0)
        gesture_options_layout = QVBoxLayout(self._gesture_options_panel)
        gesture_options_layout.setContentsMargins(4, 2, 4, 0)
        gesture_options_layout.setSpacing(8)

        self._gesture_mode_row = QWidget()
        self._gesture_mode_row.setObjectName("featureRow")
        gesture_layout = QHBoxLayout(self._gesture_mode_row)
        gesture_layout.setContentsMargins(4, 0, 4, 0)
        gesture_layout.setSpacing(8)
        gesture_label = QLabel("手势模式")
        gesture_label.setObjectName("featureLabel")
        gesture_layout.addWidget(gesture_label)
        self._gesture_mode_combo = WheelFocusComboBox()
        self._gesture_mode_combo.setObjectName("featureCombo")
        self._gesture_mode_combo.addItem("关闭", "off")
        self._gesture_mode_combo.addItem("人体姿态", "body")
        self._gesture_mode_combo.addItem("手部手指", "hand")
        self._gesture_mode_combo.addItem("全部", "all")
        gesture_index = self._gesture_mode_combo.findData(self._ai_gesture_mode)
        self._gesture_mode_combo.setCurrentIndex(max(0, gesture_index))
        self._gesture_mode_combo.currentIndexChanged.connect(self._on_gesture_mode_changed)
        gesture_layout.addWidget(self._gesture_mode_combo, stretch=1)
        self._rectangle_toggle = self._create_feature_toggle("矩形识别", self._ai_rectangle_enabled)

        self._rectangle_options_panel = QWidget()
        self._rectangle_options_panel.setObjectName("rectangleOptionsPanel")
        self._rectangle_options_panel.setVisible(False)
        self._rectangle_options_panel.setMaximumHeight(0)
        rectangle_options_layout = QVBoxLayout(self._rectangle_options_panel)
        rectangle_options_layout.setContentsMargins(4, 2, 4, 0)
        rectangle_options_layout.setSpacing(8)

        self._rectangle_sensitivity_row = QWidget()
        self._rectangle_sensitivity_row.setObjectName("featureRow")
        sensitivity_layout = QHBoxLayout(self._rectangle_sensitivity_row)
        sensitivity_layout.setContentsMargins(4, 0, 4, 0)
        sensitivity_layout.setSpacing(8)
        sensitivity_label = QLabel("矩形灵敏度")
        sensitivity_label.setObjectName("featureLabel")
        sensitivity_layout.addWidget(sensitivity_label)
        self._rectangle_sensitivity_combo = WheelFocusComboBox()
        self._rectangle_sensitivity_combo.setObjectName("featureCombo")
        self._rectangle_sensitivity_combo.addItem("低", "low")
        self._rectangle_sensitivity_combo.addItem("中", "medium")
        self._rectangle_sensitivity_combo.addItem("高", "high")
        sensitivity_index = self._rectangle_sensitivity_combo.findData(self._ai_rectangle_sensitivity)
        self._rectangle_sensitivity_combo.setCurrentIndex(max(0, sensitivity_index))
        self._rectangle_sensitivity_combo.currentIndexChanged.connect(self._on_rectangle_sensitivity_changed)
        sensitivity_layout.addWidget(self._rectangle_sensitivity_combo, stretch=1)

        self._rectangle_max_count_row = QWidget()
        self._rectangle_max_count_row.setObjectName("featureRow")
        max_count_layout = QHBoxLayout(self._rectangle_max_count_row)
        max_count_layout.setContentsMargins(4, 0, 4, 0)
        max_count_layout.setSpacing(8)
        max_count_label = QLabel("最多数量")
        max_count_label.setObjectName("featureLabel")
        max_count_layout.addWidget(max_count_label)
        self._rectangle_max_count_spin = WheelFocusSpinBox()
        self._rectangle_max_count_spin.setObjectName("featureCombo")
        self._rectangle_max_count_spin.setRange(1, 10)
        self._rectangle_max_count_spin.setValue(self._ai_rectangle_max_count)
        self._rectangle_max_count_spin.valueChanged.connect(self._on_rectangle_max_count_changed)
        max_count_layout.addWidget(self._rectangle_max_count_spin, stretch=1)

        self._rectangle_color_panel = QWidget()
        self._rectangle_color_panel.setObjectName("rectangleColorPanel")
        color_layout = QVBoxLayout(self._rectangle_color_panel)
        color_layout.setContentsMargins(0, 0, 0, 0)
        color_layout.setSpacing(12)

        color_row = QHBoxLayout()
        color_row.setContentsMargins(0, 0, 0, 0)
        color_row.setSpacing(8)
        color_label = QLabel("目标颜色")
        color_label.setObjectName("featureLabel")
        color_row.addWidget(color_label)
        self._rectangle_color_btn = QPushButton()
        self._rectangle_color_btn.setObjectName("colorSwatchButton")
        self._rectangle_color_btn.setFixedSize(44, 28)
        self._rectangle_color_btn.clicked.connect(self._on_rectangle_color_pick)
        color_row.addWidget(self._rectangle_color_btn)
        self._rectangle_color_clear_btn = QPushButton("清除")
        self._rectangle_color_clear_btn.setObjectName("secondaryCompact")
        self._rectangle_color_clear_btn.clicked.connect(self._on_rectangle_color_clear)
        color_row.addWidget(self._rectangle_color_clear_btn)
        color_row.addStretch(1)
        color_layout.addLayout(color_row)

        threshold_row = QHBoxLayout()
        threshold_row.setContentsMargins(0, 4, 0, 0)
        threshold_row.setSpacing(8)
        threshold_label = QLabel("色差阈值")
        threshold_label.setObjectName("featureLabel")
        threshold_row.addWidget(threshold_label)
        self._rectangle_color_threshold_spin = WheelFocusDoubleSpinBox()
        self._rectangle_color_threshold_spin.setObjectName("featureCombo")
        self._rectangle_color_threshold_spin.setRange(1.0, 100.0)
        self._rectangle_color_threshold_spin.setDecimals(1)
        self._rectangle_color_threshold_spin.setSingleStep(1.0)
        self._rectangle_color_threshold_spin.setValue(self._ai_rectangle_color_threshold)
        self._rectangle_color_threshold_spin.valueChanged.connect(self._on_rectangle_color_threshold_changed)
        threshold_row.addWidget(self._rectangle_color_threshold_spin, stretch=1)
        color_layout.addLayout(threshold_row)
        self._update_rectangle_color_swatch()

        self._person_toggle.toggled.connect(self._on_person_feature_toggled)
        self._skeleton_toggle.toggled.connect(self._on_skeleton_feature_toggled)
        self._rectangle_toggle.toggled.connect(self._on_rectangle_feature_toggled)

        layout.addWidget(self._person_toggle)
        layout.addWidget(self._skeleton_toggle)
        gesture_options_layout.addWidget(self._gesture_mode_row)
        layout.addWidget(self._gesture_options_panel)
        layout.addWidget(self._rectangle_toggle)
        rectangle_options_layout.addWidget(self._rectangle_sensitivity_row)
        rectangle_options_layout.addWidget(self._rectangle_max_count_row)
        rectangle_options_layout.addWidget(self._rectangle_color_panel)
        layout.addWidget(self._rectangle_options_panel)
        self._skeleton_toggle.setEnabled(self._ai_person_enabled)
        self._set_gesture_options_visible(self._ai_skeleton_enabled and self._ai_person_enabled, animated=False)
        self._set_rectangle_options_visible(self._ai_rectangle_enabled, animated=False)
        return panel

    @staticmethod
    def _create_feature_toggle(text: str, checked: bool) -> QCheckBox:
        toggle = QCheckBox(text)
        toggle.setObjectName("featureSwitch")
        toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        toggle.setChecked(checked)
        return toggle

    def _build_storage_card(self) -> QFrame:
        card, body = self._create_section_card("存储配置", "统一管理录像输出目录与路径展示")

        self._select_path_btn = QPushButton("选择存储路径")
        self._select_path_btn.setObjectName("secondary")
        self._select_path_btn.clicked.connect(self._on_select_storage_path)
        body.addWidget(self._select_path_btn)

        path_caption = QLabel("当前输出目录")
        path_caption.setObjectName("fieldLabel")
        body.addWidget(path_caption)

        self._storage_path_label = QLabel(self._recording_output_dir)
        self._storage_path_label.setObjectName("pathValue")
        self._storage_path_label.setWordWrap(True)
        body.addWidget(self._storage_path_label)

        return card

    def _create_section_card(self, title: str, description: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("sectionCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setObjectName("sectionHint")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label)

        return card, layout

    def _create_metric_pill(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("metricPill")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _create_stat_row(self, parent_layout: QVBoxLayout, name: str) -> QLabel:
        row = QFrame()
        row.setObjectName("statRow")

        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        name_label = QLabel(name)
        name_label.setObjectName("statName")
        layout.addWidget(name_label)

        value_label = QLabel("--")
        value_label.setObjectName("statValue")
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(value_label, stretch=1)

        parent_layout.addWidget(row)
        return value_label

    def _build_slot_group(self, slot_id: int) -> QFrame:
        group = QFrame()
        group.setObjectName("slotCard")

        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(12, 12, 12, 12)
        group_layout.setSpacing(8)

        title = QLabel(f"采集位 {slot_id + 1}")
        title.setObjectName("slotTitle")
        group_layout.addWidget(title)

        subtitle = QLabel("选择要接入当前视频卡片的摄像头设备")
        subtitle.setObjectName("slotMeta")
        subtitle.setWordWrap(True)
        group_layout.addWidget(subtitle)

        combo = WheelFocusComboBox()
        combo.addItem("未分配", "")
        for cam in self._camera_list:
            combo.addItem(cam.name, cam.name)
        combo.currentIndexChanged.connect(lambda idx, sid=slot_id: self._on_camera_selected(sid, idx))
        self._combos[slot_id] = combo
        group_layout.addWidget(combo)

        return group

    def _build_grid_area(self) -> QWidget:
        grid_widget = QWidget()
        grid_widget.setObjectName("cameraGrid")

        self._grid_layout = QGridLayout(grid_widget)
        self._grid_layout.setSpacing(16)
        self._grid_layout.setContentsMargins(18, 18, 18, 18)

        for slot_id in range(self._MAX_SLOTS):
            cell = CameraCell(slot_id, label=f"采集位 {slot_id + 1}")
            cell.set_frame_flipped(self._is_frame_flipped)
            cell.double_clicked.connect(self._on_cell_double_clicked)
            self._cells[slot_id] = cell
            row = slot_id // 2
            col = slot_id % 2
            self._grid_layout.addWidget(cell, row, col)

        return grid_widget

    def _on_cell_double_clicked(self, slot_id: int) -> None:
        cell = self._cells.get(slot_id)
        if cell is None or not cell.has_input_signal():
            return

        if self._expanded_slot_id == slot_id:
            self._restore_camera_grid()
            return
        self._expand_camera_cell(slot_id)

    def _expand_camera_cell(self, slot_id: int) -> None:
        if self._grid_layout is None:
            return

        self._expanded_slot_id = slot_id
        for sid, cell in self._cells.items():
            self._grid_layout.removeWidget(cell)
            if sid == slot_id:
                cell.setVisible(True)
                self._grid_layout.addWidget(cell, 0, 0, 2, 2)
            else:
                cell.setVisible(False)

        self._refresh_dashboard(f"采集位 {slot_id + 1}：已放大")

    def _restore_camera_grid(self) -> None:
        if self._grid_layout is None:
            return

        self._expanded_slot_id = None
        for sid, cell in self._cells.items():
            self._grid_layout.removeWidget(cell)
            cell.setVisible(True)
            self._grid_layout.addWidget(cell, sid // 2, sid % 2)

        self._refresh_dashboard("已恢复四宫格")

    def _animate_sidebar_sections(self) -> None:
        self._sidebar_animations.clear()

        # 使用渐进式动画序列，参考现代前端设计
        animations = AnimationHelper.staggered_slide_in(
            self._sidebar_sections,
            base_duration=500,
            stagger_delay=100,
            parent=self,
        )
        self._sidebar_animations.extend(animations)

    def _apply_combo_dark_palette(self) -> None:
        palette = QPalette()
        bg = QColor("#1A2238")
        fg = QColor("#EAF1FF")
        highlight = QColor("#6FA8FF")
        highlight_fg = QColor("#0B1020")

        palette.setColor(QPalette.ColorRole.Base, bg)
        palette.setColor(QPalette.ColorRole.Window, bg)
        palette.setColor(QPalette.ColorRole.WindowText, fg)
        palette.setColor(QPalette.ColorRole.Text, fg)
        palette.setColor(QPalette.ColorRole.Button, bg)
        palette.setColor(QPalette.ColorRole.ButtonText, fg)
        palette.setColor(QPalette.ColorRole.Highlight, highlight)
        palette.setColor(QPalette.ColorRole.HighlightedText, highlight_fg)
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#6E7A96"))

        combos = list(self._combos.values())
        if hasattr(self, "_gesture_mode_combo"):
            combos.append(self._gesture_mode_combo)
        if hasattr(self, "_rectangle_sensitivity_combo"):
            combos.append(self._rectangle_sensitivity_combo)

        for combo in combos:
            combo.view().setPalette(palette)
            combo.view().setAutoFillBackground(True)
            popup = combo.view().parentWidget()
            if popup is not None:
                popup.setPalette(palette)
                popup.setAutoFillBackground(True)

    def _sync_cell_states(self) -> None:
        active_ids = {slot_id for slot_id, _ in self._camera_capture.get_all_workers()}
        for slot_id, cell in self._cells.items():
            selected_name = self._combos[slot_id].currentData() or ""
            is_connected = slot_id in active_ids and bool(selected_name)
            cell.set_stream_state(
                connected=is_connected,
                device_name=selected_name or None,
                ai_enabled=self._is_ai_enabled and is_connected,
                recording=self._is_recording and is_connected,
            )

    def _refresh_dashboard(self, message: str | None = None) -> None:
        active_workers = self._camera_capture.get_all_workers()
        active_count = len(active_workers)

        # 状态信息合并到 hero 卡顶部的 3 个 metricPill
        # 已删除底部"系统状态"重复栏
        self._swap_metric_text(self._active_pill, f"在线 {active_count}/{self._MAX_SLOTS}")
        self._swap_metric_text(self._mode_pill, "AI 开启" if self._is_ai_enabled else "AI 关闭")

        if message is not None:
            logging.info("status: %s", message)

        self._sync_cell_states()

    def _swap_metric_text(self, label: QLabel, new_text: str) -> None:
        animation = AnimationHelper.fade_swap_label(label, new_text, duration=220, parent=self)
        if animation is None:
            return
        self._play_feedback_animation(animation)

    def _play_feedback_animation(self, animation) -> None:
        if animation is None:
            return
        self._ui_feedback_animations.append(animation)

        def _cleanup() -> None:
            try:
                self._ui_feedback_animations.remove(animation)
            except ValueError:
                pass

        animation.finished.connect(_cleanup)
        animation.start()

    def _animate_ai_feature_panel(self, expanded: bool) -> None:
        if self._ai_feature_animation is not None:
            self._ai_feature_animation.stop()
            self._ai_feature_animation = None

        panel = self._ai_feature_panel
        if expanded:
            panel.setVisible(True)
            panel.setMaximumHeight(panel.sizeHint().height())

        effect = panel.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(panel)
            panel.setGraphicsEffect(effect)

        opacity_anim = QPropertyAnimation(effect, b"opacity", self)
        opacity_anim.setDuration(130 if expanded else 100)
        opacity_anim.setStartValue(effect.opacity() if panel.isVisible() and not expanded else 0.0)
        opacity_anim.setEndValue(1.0 if expanded else 0.0)
        opacity_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        def _cleanup() -> None:
            if expanded:
                effect.setOpacity(1.0)
                panel.setGraphicsEffect(None)
            else:
                panel.setVisible(False)
                panel.setMaximumHeight(0)
                panel.setGraphicsEffect(None)
            self._ai_feature_animation = None

        opacity_anim.finished.connect(_cleanup)
        self._ai_feature_animation = opacity_anim
        opacity_anim.start()

    def _apply_feature_flags_to_worker(self) -> None:
        self._infer_worker.set_feature_flags(
            person=self._ai_person_enabled,
            skeleton=self._ai_skeleton_enabled,
            rectangle=self._ai_rectangle_enabled,
            gesture_mode=self._ai_gesture_mode,
        )
        self._infer_worker.set_rectangle_sensitivity(self._ai_rectangle_sensitivity)
        self._infer_worker.set_rectangle_max_count(self._ai_rectangle_max_count)
        self._infer_worker.set_rectangle_target_color(self._ai_rectangle_target_rgb)
        self._infer_worker.set_rectangle_color_threshold(self._ai_rectangle_color_threshold)

    def _save_feature_flags(self) -> None:
        SettingsDialog.save_ai_person_enabled(self._ai_person_enabled)
        SettingsDialog.save_ai_skeleton_enabled(self._ai_skeleton_enabled)
        SettingsDialog.save_ai_gesture_mode(self._ai_gesture_mode)
        SettingsDialog.save_ai_rectangle_enabled(self._ai_rectangle_enabled)
        SettingsDialog.save_ai_rectangle_sensitivity(self._ai_rectangle_sensitivity)
        SettingsDialog.save_ai_rectangle_max_count(self._ai_rectangle_max_count)
        SettingsDialog.save_ai_rectangle_target_color(self._ai_rectangle_target_rgb, self._ai_rectangle_target_hsv)
        SettingsDialog.save_ai_rectangle_color_threshold(self._ai_rectangle_color_threshold)

    def _sync_feature_toggle_states(self, save: bool = True) -> None:
        if not self._ai_person_enabled:
            self._ai_skeleton_enabled = False
        if not self._ai_skeleton_enabled:
            self._ai_gesture_mode = "off"

        self._skeleton_toggle.setEnabled(self._ai_person_enabled)
        self._set_gesture_options_visible(self._ai_skeleton_enabled and self._ai_person_enabled, animated=self._ai_feature_panel.isVisible())
        self._set_rectangle_options_visible(self._ai_rectangle_enabled, animated=self._ai_feature_panel.isVisible())
        for toggle, value in (
            (self._person_toggle, self._ai_person_enabled),
            (self._skeleton_toggle, self._ai_skeleton_enabled),
            (self._rectangle_toggle, self._ai_rectangle_enabled),
        ):
            toggle.blockSignals(True)
            toggle.setChecked(value)
            toggle.blockSignals(False)

        mode_index = self._gesture_mode_combo.findData(self._ai_gesture_mode)
        self._gesture_mode_combo.blockSignals(True)
        self._gesture_mode_combo.setCurrentIndex(max(0, mode_index))
        self._gesture_mode_combo.blockSignals(False)

        self._rectangle_max_count_spin.blockSignals(True)
        self._rectangle_max_count_spin.setValue(self._ai_rectangle_max_count)
        self._rectangle_max_count_spin.blockSignals(False)

        self._rectangle_color_threshold_spin.blockSignals(True)
        self._rectangle_color_threshold_spin.setValue(self._ai_rectangle_color_threshold)
        self._rectangle_color_threshold_spin.blockSignals(False)
        self._update_rectangle_color_swatch()

        if save:
            self._save_feature_flags()
        self._apply_feature_flags_to_worker()

    def _on_person_feature_toggled(self, checked: bool) -> None:
        self._ai_person_enabled = checked
        if not checked:
            self._ai_skeleton_enabled = False
        self._sync_feature_toggle_states()
        self._clear_all_detections()
        self._refresh_dashboard("人体识别：已启用" if checked else "人体识别：已关闭")

    def _on_skeleton_feature_toggled(self, checked: bool) -> None:
        self._ai_skeleton_enabled = checked and self._ai_person_enabled
        self._sync_feature_toggle_states()
        self._clear_all_detections()
        self._refresh_dashboard("骨骼绑定：已启用" if self._ai_skeleton_enabled else "骨骼绑定：已关闭")

    def _on_gesture_mode_changed(self, index: int) -> None:
        mode = str(self._gesture_mode_combo.itemData(index) or "off")
        if not (self._ai_person_enabled and self._ai_skeleton_enabled):
            mode = "off"
        self._ai_gesture_mode = mode
        self._sync_feature_toggle_states()
        self._clear_all_detections()
        self._refresh_dashboard(f"手势模式：{self._gesture_mode_combo.currentText()}")

    def _on_rectangle_feature_toggled(self, checked: bool) -> None:
        self._ai_rectangle_enabled = checked
        self._sync_feature_toggle_states()
        self._clear_all_detections()
        self._refresh_dashboard("矩形识别：已启用" if checked else "矩形识别：已关闭")

    def _on_rectangle_sensitivity_changed(self, index: int) -> None:
        level = self._rectangle_sensitivity_combo.itemData(index) or "low"
        self._ai_rectangle_sensitivity = str(level)
        SettingsDialog.save_ai_rectangle_sensitivity(self._ai_rectangle_sensitivity)
        self._infer_worker.set_rectangle_sensitivity(self._ai_rectangle_sensitivity)
        self._clear_all_detections()
        label = self._rectangle_sensitivity_combo.currentText()
        self._refresh_dashboard(f"矩形灵敏度：{label}")

    def _on_rectangle_max_count_changed(self, value: int) -> None:
        self._ai_rectangle_max_count = max(1, min(10, int(value)))
        SettingsDialog.save_ai_rectangle_max_count(self._ai_rectangle_max_count)
        self._infer_worker.set_rectangle_max_count(self._ai_rectangle_max_count)
        self._clear_all_detections()
        self._refresh_dashboard(f"矩形最多数量：{self._ai_rectangle_max_count}")

    def _on_rectangle_color_pick(self) -> None:
        initial = QColor(*(self._ai_rectangle_target_rgb or (244, 184, 96)))
        color = QColorDialog.getColor(initial, self, "选择矩形目标颜色")
        if not color.isValid():
            return
        self._ai_rectangle_target_rgb = (color.red(), color.green(), color.blue())
        self._ai_rectangle_target_hsv = self._rgb_to_hsv_tuple(self._ai_rectangle_target_rgb)
        SettingsDialog.save_ai_rectangle_target_color(self._ai_rectangle_target_rgb, self._ai_rectangle_target_hsv)
        self._infer_worker.set_rectangle_target_color(self._ai_rectangle_target_rgb)
        self._update_rectangle_color_swatch()
        self._clear_all_detections()
        r, g, b = self._ai_rectangle_target_rgb
        self._refresh_dashboard(f"矩形目标颜色：红 {r}，绿 {g}，蓝 {b}")

    def _on_rectangle_color_clear(self) -> None:
        self._ai_rectangle_target_rgb = None
        self._ai_rectangle_target_hsv = None
        SettingsDialog.save_ai_rectangle_target_color(None, None)
        self._infer_worker.set_rectangle_target_color(None)
        self._update_rectangle_color_swatch()
        self._clear_all_detections()
        self._refresh_dashboard("矩形目标颜色：已清除")

    def _on_rectangle_color_threshold_changed(self, value: float) -> None:
        self._ai_rectangle_color_threshold = max(1.0, min(100.0, float(value)))
        SettingsDialog.save_ai_rectangle_color_threshold(self._ai_rectangle_color_threshold)
        self._infer_worker.set_rectangle_color_threshold(self._ai_rectangle_color_threshold)
        self._clear_all_detections()
        self._refresh_dashboard(f"矩形色差阈值：{self._ai_rectangle_color_threshold:.1f}")

    def _set_gesture_options_visible(self, visible: bool, animated: bool = True) -> None:
        panel = self._gesture_options_panel
        if self._gesture_options_animation is not None:
            self._gesture_options_animation.stop()
            self._gesture_options_animation = None

        target_height = panel.sizeHint().height() if visible else 0
        if not animated:
            panel.setVisible(visible)
            panel.setMaximumHeight(target_height)
            return
        if visible:
            panel.setVisible(True)

        height_anim = QPropertyAnimation(panel, b"maximumHeight", self)
        height_anim.setDuration(160)
        height_anim.setStartValue(panel.maximumHeight())
        height_anim.setEndValue(target_height)
        height_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(height_anim)

        def _cleanup() -> None:
            panel.setVisible(visible)
            panel.setMaximumHeight(target_height)
            self._gesture_options_animation = None

        group.finished.connect(_cleanup)
        self._gesture_options_animation = group
        group.start()

    def _set_rectangle_options_visible(self, visible: bool, animated: bool = True) -> None:
        panel = self._rectangle_options_panel
        if self._rectangle_options_animation is not None:
            self._rectangle_options_animation.stop()
            self._rectangle_options_animation = None

        target_height = panel.sizeHint().height() if visible else 0
        if not animated:
            panel.setVisible(visible)
            panel.setMaximumHeight(target_height)
            return
        if visible:
            panel.setVisible(True)

        height_anim = QPropertyAnimation(panel, b"maximumHeight", self)
        height_anim.setDuration(160)
        height_anim.setStartValue(panel.maximumHeight())
        height_anim.setEndValue(target_height)
        height_anim.setEasingCurve(QEasingCurve.Type.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(height_anim)

        def _cleanup() -> None:
            panel.setVisible(visible)
            panel.setMaximumHeight(target_height)
            self._rectangle_options_animation = None

        group.finished.connect(_cleanup)
        self._rectangle_options_animation = group
        group.start()

    def _update_rectangle_color_swatch(self) -> None:
        if self._ai_rectangle_target_rgb is None:
            self._rectangle_color_btn.setText("未选")
            self._rectangle_color_btn.setStyleSheet(
                "QPushButton#colorSwatchButton { background-color: #1A2238; color: #9AA8C7; border: 1px solid #32405F; border-radius: 5px; }"
            )
            self._rectangle_color_clear_btn.setEnabled(False)
            return
        r, g, b = self._ai_rectangle_target_rgb
        text_color = "#0B1020" if (r * 0.299 + g * 0.587 + b * 0.114) > 150 else "#EAF1FF"
        self._rectangle_color_btn.setText("")
        self._rectangle_color_btn.setStyleSheet(
            f"QPushButton#colorSwatchButton {{ background-color: rgb({r}, {g}, {b}); color: {text_color}; border: 1px solid #EAF1FF; border-radius: 5px; }}"
        )
        self._rectangle_color_clear_btn.setEnabled(True)

    @staticmethod
    def _rgb_to_hsv_tuple(rgb: tuple[int, int, int] | None) -> tuple[int, int, int] | None:
        if rgb is None:
            return None
        color = QColor(*rgb)
        return (int(color.hue() if color.hue() >= 0 else 0), int(color.saturation()), int(color.value()))

    def _clear_all_detections(self) -> None:
        for slot_id, cell in self._cells.items():
            cell.update_detections(slot_id, [])

    @staticmethod
    def _safe_connect(signal, slot) -> None:
        try:
            signal.connect(slot, Qt.ConnectionType.UniqueConnection)
        except TypeError:
            try:
                signal.connect(slot)
            except TypeError:
                pass

    @staticmethod
    def _safe_direct_connect(signal, slot) -> None:
        try:
            signal.connect(slot, Qt.ConnectionType.UniqueConnection | Qt.ConnectionType.DirectConnection)
        except TypeError:
            try:
                signal.connect(slot, Qt.ConnectionType.DirectConnection)
            except TypeError:
                pass

    def _on_camera_selected(self, slot_id: int, combo_index: int) -> None:
        device_name = self._combos[slot_id].itemData(combo_index)

        if not device_name:
            if self._expanded_slot_id == slot_id:
                self._restore_camera_grid()
            self._camera_capture.stop_camera(slot_id)
            self._infer_worker.unregister_slot(slot_id)
            self._cells[slot_id].clear_stream()
            self._refresh_dashboard(f"采集位 {slot_id + 1}：已断开")
            return

        self._camera_capture.stop_camera(slot_id)
        self._cells[slot_id].clear_stream()
        self._infer_worker.register_slot(slot_id)
        self._camera_capture.start_camera(slot_id, device_name)

        worker = self._camera_capture.get_worker(slot_id)
        if worker is not None:
            # #region debug-point D:signal-connection
            if self._debug_connect_reports < 10:
                _debug_report(
                    "D",
                    "gui/main_window.py:_on_camera_selected",
                    "worker connected to cell",
                    {
                        "slot_id": slot_id,
                        "device_name": device_name,
                        "ai_enabled": self._is_ai_enabled,
                        "recording": self._is_recording,
                        "worker_id": id(worker),
                    },
                )
                self._debug_connect_reports += 1
            # #endregion
            self._safe_connect(worker.frame_ready, self._cells[slot_id].update_frame)
            if self._is_ai_enabled:
                self._safe_direct_connect(worker.frame_ready, self._infer_worker.submit_frame)
                self._safe_connect(self._infer_worker.detection_ready, self._cells[slot_id].update_detections)
            if self._is_recording:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str(Path(self._recording_output_dir) / f"slot_{slot_id}_{timestamp}.mp4")
                self._video_recorder.start_record(slot_id, path)
                self._safe_connect(worker.frame_ready, self._on_frame_for_recording)

        self._refresh_dashboard(f"采集位 {slot_id + 1}：{device_name}")

    def _on_ai_toggled(self, checked: bool) -> None:
        self._is_ai_enabled = checked
        if checked:
            self._sync_feature_toggle_states()
            self._set_gesture_options_visible(self._ai_skeleton_enabled and self._ai_person_enabled, animated=False)
            self._set_rectangle_options_visible(self._ai_rectangle_enabled, animated=False)
            self._animate_ai_feature_panel(True)
            self._infer_worker.prepare_startup_guard()
        else:
            self._animate_ai_feature_panel(False)

        for slot_id, worker in self._camera_capture.get_all_workers():
            if checked:
                self._safe_direct_connect(worker.frame_ready, self._infer_worker.submit_frame)
                self._infer_worker.register_slot(slot_id)
                self._safe_connect(self._infer_worker.detection_ready, self._cells[slot_id].update_detections)
            else:
                try:
                    worker.frame_ready.disconnect(self._infer_worker.submit_frame)
                except TypeError:
                    pass
                self._infer_worker.unregister_slot(slot_id)
                try:
                    self._infer_worker.detection_ready.disconnect(self._cells[slot_id].update_detections)
                except TypeError:
                    pass
                self._cells[slot_id].update_detections(slot_id, [])

        state_text = "预热中" if checked else "已禁用"
        self._refresh_dashboard(f"AI 检测：{state_text}")
        if checked:
            QTimer.singleShot(
                2000,
                lambda: self._refresh_dashboard("AI 检测：已启用") if self._is_ai_enabled else None,
            )

    def _on_record_toggled(self) -> None:
        if not self._is_recording:
            active_workers = self._camera_capture.get_all_workers()
            if not active_workers:
                self._refresh_dashboard("没有活动的摄像头可供录制")
                return

            for slot_id, worker in active_workers:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = str(Path(self._recording_output_dir) / f"slot_{slot_id}_{timestamp}.mp4")
                self._video_recorder.start_record(slot_id, path)
                self._safe_connect(worker.frame_ready, self._on_frame_for_recording)

            self._is_recording = True
            self._record_btn.setText("停止录制")
            self._record_btn.setObjectName("danger")
            self._record_btn.style().unpolish(self._record_btn)
            self._record_btn.style().polish(self._record_btn)
            self._play_feedback_animation(AnimationHelper.bass_pluck(self._record_btn, parent=self))
            self._refresh_dashboard("录制中...")
        else:
            for _, worker in self._camera_capture.get_all_workers():
                try:
                    worker.frame_ready.disconnect(self._on_frame_for_recording)
                except TypeError:
                    pass

            self._video_recorder.stop_all()
            self._is_recording = False
            self._record_btn.setText("开始录制")
            self._record_btn.setObjectName("primary")
            self._record_btn.style().unpolish(self._record_btn)
            self._record_btn.style().polish(self._record_btn)
            self._play_feedback_animation(AnimationHelper.bass_pluck(self._record_btn, parent=self))
            self._refresh_dashboard("录制已停止")

    def _on_select_storage_path(self) -> None:
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_dir = dialog.get_output_dir()
            self._recording_output_dir = new_dir
            self._video_recorder.set_output_dir(new_dir)
            self._storage_path_label.setText(new_dir)
            self._refresh_dashboard(f"录像路径已更新：{new_dir}")

            ai_cfg = dialog.get_ai_settings()
            self._apply_ai_settings(ai_cfg)

    def _apply_ai_settings(self, cfg: dict) -> None:
        self._ai_conf = cfg["conf"]
        self._ai_stride = cfg["stride"]
        self._ai_smooth = cfg["smooth"]
        self._ai_hide_ms = cfg["hide_ms"]
        requested_model_path = cfg.get("model_path", "") or ""

        self._infer_worker.set_conf(self._ai_conf)
        self._infer_worker.set_infer_stride(self._ai_stride)
        self._infer_worker.set_smooth_alpha(self._ai_smooth)
        # 透传 ROI 重检测配置
        redetect_retries = cfg.get("person_redetect_retries", 2)
        redetect_roi_pad = cfg.get("person_redetect_roi_pad", 0.15)
        self._infer_worker.set_person_redetect(
            enabled=int(redetect_retries) > 0,
            max_retries=int(redetect_retries),
            roi_pad=float(redetect_roi_pad),
        )
        # 透传滤波器类型配置
        filter_type = cfg.get("filter_type", "ukf")
        mcukf_sigma = cfg.get("mcukf_kernel_sigma", 0.4)
        self._infer_worker.set_filter_type(filter_type, kernel_sigma=float(mcukf_sigma))
        # 透传去噪配置
        denoise_method = cfg.get("denoise_method", "none")
        denoise_strength = cfg.get("denoise_strength", 5)
        self._infer_worker.set_denoise(denoise_method, strength=int(denoise_strength))
        # 透传跟踪器匹配参数
        self._infer_worker.set_tracker_params(
            matching_strategy=cfg.get("matching_strategy", "hungarian"),
            iou_threshold=float(cfg.get("iou_threshold", 0.25)),
            max_misses=int(cfg.get("max_misses", 4)),
            velocity_clip=float(cfg.get("velocity_clip", 0.30)),
            high_conf_thresh=float(cfg.get("high_conf_thresh", 0.5)),
        )
        # 透传重检测预算和 ReID 配置
        self._infer_worker.set_redetect_budget(int(cfg.get("redetect_budget", 1)))
        reid_min = int(cfg.get("reid_min_tracks", 3))
        self._infer_worker.set_reid_config(
            enabled=reid_min > 0,
            min_tracks=reid_min,
            stride=int(cfg.get("reid_stride", 3)),
        )

        # 透传头部属性分类器配置
        self._ai_head_classifier_enabled = bool(cfg.get("head_classifier_enabled", False))
        self._ai_head_classifier_path = cfg.get("head_classifier_path", "")
        self._ai_head_classifier_stride = int(cfg.get("head_classifier_stride", 2))
        self._infer_worker.set_head_classifier(
            self._ai_head_classifier_enabled,
            self._ai_head_classifier_path or None,
            self._ai_head_classifier_stride,
        )

        # 透传 Pipeline 模式开关 (C8) — 默认 False (Legacy Mode)
        self._infer_worker.set_pipeline_enabled(
            bool(cfg.get("pipeline_enabled", False))
        )

        self._apply_feature_flags_to_worker()

        # 同步 CameraCell 的 TTL
        CameraCell._BBOX_TTL = self._ai_hide_ms / 1000.0
        if self._resolve_special_model_path(requested_model_path) != self._resolve_special_model_path(self._ai_model_path):
            self._pending_ai_model_path = requested_model_path
            for slot_id, cell in self._cells.items():
                cell.update_detections(slot_id, [])
            self._infer_worker.request_model_switch(self._resolve_special_model_path(requested_model_path))
            self._refresh_dashboard("AI 模型切换中...")
            return

        self._refresh_dashboard(f"AI 设置已更新：conf={self._ai_conf:.2f} stride={self._ai_stride}")

    def _on_infer_model_ready(self, model_path: str, ok: bool, message: str) -> None:
        if ok:
            self._ai_model_path = self._reverse_resolve_special_model_path(model_path or "")
            self._pending_ai_model_path = None
            SettingsDialog.save_ai_model_path(self._ai_model_path)
            self._refresh_dashboard(message)
            return

        self._pending_ai_model_path = None
        self._refresh_dashboard(message)


    def _on_frame_for_recording(self, slot_id: int, frame: object) -> None:
        if self._is_recording:
            self._video_recorder.push_frame(slot_id, frame)

    def _on_refresh_cameras(self) -> None:
        self._camera_list = scan_cameras()

        for slot_id in range(self._MAX_SLOTS):
            combo = self._combos[slot_id]
            current_data = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("未分配", "")
            for cam in self._camera_list:
                combo.addItem(cam.name, cam.name)
            idx = combo.findData(current_data)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            combo.blockSignals(False)

        self._apply_combo_dark_palette()
        self._refresh_dashboard(f"已发现 {len(self._camera_list)} 个摄像头")

    def closeEvent(self, event: QCloseEvent) -> None:
        logging.info("MainWindow closeEvent triggered, shutting down all subsystems...")

        try:
            self._video_recorder.shutdown()
            logging.info("VideoRecorder stopped")
        except Exception as err:
            logging.error("Error stopping VideoRecorder: %s", err)

        try:
            self._camera_capture.stop_all()
            for cell in self._cells.values():
                cell.clear_stream()
            logging.info("CameraCapture stopped")
        except Exception as err:
            logging.error("Error stopping CameraCapture: %s", err)

        try:
            self._infer_worker.stop()
            logging.info("InferWorker stopped")
        except Exception as err:
            logging.error("Error stopping InferWorker: %s", err)

        try:
            for cell in self._cells.values():
                cell.blockSignals(True)
            for combo in self._combos.values():
                combo.blockSignals(True)
            self._ai_toggle.blockSignals(True)
            self._person_toggle.blockSignals(True)
            self._skeleton_toggle.blockSignals(True)
            self._gesture_mode_combo.blockSignals(True)
            self._rectangle_toggle.blockSignals(True)
            self._rectangle_max_count_spin.blockSignals(True)
            self._rectangle_color_threshold_spin.blockSignals(True)
            self._rectangle_color_btn.blockSignals(True)
            self._rectangle_color_clear_btn.blockSignals(True)
            self._record_btn.blockSignals(True)
        except Exception as err:
            logging.error("Error blocking signals: %s", err)

        logging.info("All subsystems shut down, accepting close event")
        event.accept()
