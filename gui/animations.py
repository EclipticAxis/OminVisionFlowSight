from PyQt6.QtCore import (
    QRect,
    QPropertyAnimation,
    QEasingCurve,
    QParallelAnimationGroup,
    QSequentialAnimationGroup,
    QTimer,
    QVariantAnimation,
    QObject,
)
from PyQt6.QtWidgets import QDialog, QGraphicsOpacityEffect, QWidget


class Easing:
    """封装常用缓动曲线，参考现代前端动画规范"""

    SMOOTH_DECEL = QEasingCurve.Type.OutCubic
    SMOOTH_ACCEL = QEasingCurve.Type.InCubic
    ELASTIC = QEasingCurve.Type.OutBack
    BOUNCE = QEasingCurve.Type.OutBounce
    LINEAR = QEasingCurve.Type.Linear

    # 更精细的缓动
    OUT_EXPO = QEasingCurve.Type.OutExpo
    IN_OUT_QUART = QEasingCurve.Type.InOutQuart
    OUT_QUINT = QEasingCurve.Type.OutQuint


class AnimationHelper:
    """动画辅助类，提供常用动画模式的快捷创建"""

    @staticmethod
    def fade_in(
        widget: QWidget,
        duration: int = 400,
        easing: QEasingCurve.Type = Easing.SMOOTH_DECEL,
        parent: QObject = None,
    ) -> QPropertyAnimation:
        """创建淡入动画"""
        from PyQt6.QtWidgets import QGraphicsOpacityEffect

        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", parent or widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(easing)

        def _cleanup() -> None:
            effect.setOpacity(1.0)
            widget.setGraphicsEffect(None)

        anim.finished.connect(_cleanup)
        return anim

    @staticmethod
    def fade_up(
        widget: QWidget,
        duration: int = 500,
        distance: int = 20,
        easing: QEasingCurve.Type = Easing.OUT_QUINT,
        parent: QObject = None,
    ) -> QPropertyAnimation:
        """创建向上淡入动画（位移 + 透明度）"""
        from PyQt6.QtWidgets import QGraphicsOpacityEffect

        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)

        # 先设置位置偏移（通过 margin 或直接使用 move，但这里用 opacity 为主）
        anim = QPropertyAnimation(effect, b"opacity", parent or widget)
        anim.setDuration(duration)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(easing)

        # 保存原始位置用于后续扩展
        widget.setProperty("_anim_original_pos", widget.pos())
        return anim

    @staticmethod
    def staggered_fade_in(
        widgets: list[QWidget],
        base_duration: int = 400,
        stagger_delay: int = 80,
        easing: QEasingCurve.Type = Easing.SMOOTH_DECEL,
        parent: QObject = None,
    ) -> list[QPropertyAnimation]:
        """创建渐进式淡入动画序列"""
        animations = []
        for index, widget in enumerate(widgets):
            anim = AnimationHelper.fade_in(
                widget,
                duration=base_duration,
                easing=easing,
                parent=parent,
            )
            QTimer.singleShot(index * stagger_delay, anim.start)
            animations.append(anim)
        return animations

    @staticmethod
    def fade_slide_in(
        widget: QWidget,
        duration: int = 420,
        x_offset: int = -14,
        easing: QEasingCurve.Type = Easing.OUT_QUINT,
        parent: QObject = None,
    ) -> QParallelAnimationGroup:
        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(0.0)
        widget.setGraphicsEffect(effect)

        end_pos = widget.pos()
        start_pos = end_pos
        start_pos.setX(end_pos.x() + x_offset)
        widget.move(start_pos)

        opacity = QPropertyAnimation(effect, b"opacity", parent or widget)
        opacity.setDuration(duration)
        opacity.setStartValue(0.0)
        opacity.setEndValue(1.0)
        opacity.setEasingCurve(easing)

        position = QPropertyAnimation(widget, b"pos", parent or widget)
        position.setDuration(duration)
        position.setStartValue(start_pos)
        position.setEndValue(end_pos)
        position.setEasingCurve(easing)

        group = QParallelAnimationGroup(parent or widget)
        group.addAnimation(opacity)
        group.addAnimation(position)

        def _cleanup() -> None:
            effect.setOpacity(1.0)
            widget.setGraphicsEffect(None)
            widget.move(end_pos)

        group.finished.connect(_cleanup)
        return group

    @staticmethod
    def staggered_slide_in(
        widgets: list[QWidget],
        base_duration: int = 420,
        stagger_delay: int = 80,
        parent: QObject = None,
    ) -> list[QParallelAnimationGroup]:
        animations = []
        for index, widget in enumerate(widgets):
            anim = AnimationHelper.fade_slide_in(widget, duration=base_duration, parent=parent)
            QTimer.singleShot(index * stagger_delay, anim.start)
            animations.append(anim)
        return animations

    @staticmethod
    def fade_swap_label(
        widget: QWidget,
        new_text: str,
        duration: int = 200,
        easing: QEasingCurve.Type = Easing.SMOOTH_DECEL,
        parent: QObject = None,
    ) -> QSequentialAnimationGroup | None:
        if getattr(widget, "text", None) is None:
            return None
        if widget.text() == new_text:
            return None

        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(1.0)
            widget.setGraphicsEffect(effect)

        fade_out = QPropertyAnimation(effect, b"opacity", parent or widget)
        fade_out.setDuration(max(80, duration // 2))
        fade_out.setStartValue(effect.opacity())
        fade_out.setEndValue(0.25)
        fade_out.setEasingCurve(easing)

        fade_in = QPropertyAnimation(effect, b"opacity", parent or widget)
        fade_in.setDuration(max(80, duration // 2))
        fade_in.setStartValue(0.25)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(easing)

        group = QSequentialAnimationGroup(parent or widget)
        group.addAnimation(fade_out)
        group.addAnimation(fade_in)

        def _apply_text() -> None:
            widget.setText(new_text)

        fade_out.finished.connect(_apply_text)

        def _cleanup() -> None:
            effect.setOpacity(1.0)
            widget.setGraphicsEffect(None)

        group.finished.connect(_cleanup)
        return group

    @staticmethod
    def pulse_opacity(
        widget: QWidget,
        min_opacity: float = 0.72,
        max_opacity: float = 1.0,
        duration: int = 180,
        parent: QObject = None,
    ) -> QSequentialAnimationGroup:
        effect = widget.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(widget)
            effect.setOpacity(1.0)
            widget.setGraphicsEffect(effect)

        down = QPropertyAnimation(effect, b"opacity", parent or widget)
        down.setDuration(max(60, duration // 2))
        down.setStartValue(effect.opacity())
        down.setEndValue(min_opacity)
        down.setEasingCurve(Easing.OUT_EXPO)

        up = QPropertyAnimation(effect, b"opacity", parent or widget)
        up.setDuration(max(60, duration // 2))
        up.setStartValue(min_opacity)
        up.setEndValue(max_opacity)
        up.setEasingCurve(Easing.OUT_EXPO)

        group = QSequentialAnimationGroup(parent or widget)
        group.addAnimation(down)
        group.addAnimation(up)

        def _cleanup() -> None:
            effect.setOpacity(1.0)
            widget.setGraphicsEffect(None)

        group.finished.connect(_cleanup)
        return group

    @staticmethod
    def bass_pluck(widget: QWidget, parent: QObject = None) -> QSequentialAnimationGroup:
        return AnimationHelper.pulse_opacity(
            widget,
            min_opacity=0.62,
            max_opacity=1.0,
            duration=150,
            parent=parent,
        )

    @staticmethod
    def dialog_open(
        dialog: QDialog,
        duration: int = 220,
        y_offset: int = 18,
        parent: QObject = None,
    ) -> QParallelAnimationGroup:
        end_rect = dialog.geometry()
        start_rect = QRect(end_rect.x(), end_rect.y() + y_offset, end_rect.width(), end_rect.height())

        dialog.setWindowOpacity(0.0)
        dialog.setGeometry(start_rect)

        opacity = QPropertyAnimation(dialog, b"windowOpacity", parent or dialog)
        opacity.setDuration(duration)
        opacity.setStartValue(0.0)
        opacity.setEndValue(1.0)
        opacity.setEasingCurve(Easing.OUT_QUINT)

        geometry = QPropertyAnimation(dialog, b"geometry", parent or dialog)
        geometry.setDuration(duration)
        geometry.setStartValue(start_rect)
        geometry.setEndValue(end_rect)
        geometry.setEasingCurve(Easing.OUT_QUINT)

        group = QParallelAnimationGroup(parent or dialog)
        group.addAnimation(opacity)
        group.addAnimation(geometry)
        return group

    @staticmethod
    def dialog_close(
        dialog: QDialog,
        duration: int = 180,
        y_offset: int = 14,
        parent: QObject = None,
    ) -> QParallelAnimationGroup:
        start_rect = dialog.geometry()
        end_rect = QRect(start_rect.x(), start_rect.y() + y_offset, start_rect.width(), start_rect.height())

        opacity = QPropertyAnimation(dialog, b"windowOpacity", parent or dialog)
        opacity.setDuration(duration)
        opacity.setStartValue(dialog.windowOpacity())
        opacity.setEndValue(0.0)
        opacity.setEasingCurve(Easing.IN_OUT_QUART)

        geometry = QPropertyAnimation(dialog, b"geometry", parent or dialog)
        geometry.setDuration(duration)
        geometry.setStartValue(start_rect)
        geometry.setEndValue(end_rect)
        geometry.setEasingCurve(Easing.IN_OUT_QUART)

        group = QParallelAnimationGroup(parent or dialog)
        group.addAnimation(opacity)
        group.addAnimation(geometry)
        return group

    @staticmethod
    def breathe_effect(
        widget: QWidget,
        min_opacity: float = 0.4,
        max_opacity: float = 1.0,
        duration: int = 2000,
        parent: QObject = None,
    ) -> QPropertyAnimation:
        """创建呼吸灯效果（循环）"""
        from PyQt6.QtWidgets import QGraphicsOpacityEffect

        effect = QGraphicsOpacityEffect(widget)
        effect.setOpacity(max_opacity)
        widget.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", parent or widget)
        anim.setDuration(duration)
        anim.setStartValue(max_opacity)
        anim.setEndValue(min_opacity)
        anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        anim.finished.connect(lambda: anim.setDirection(
            QPropertyAnimation.Direction.Backward if anim.direction() == QPropertyAnimation.Direction.Forward else QPropertyAnimation.Direction.Forward
        ) or anim.start())
        return anim

    @staticmethod
    def scale_pulse(
        widget: QWidget,
        min_scale: float = 1.0,
        max_scale: float = 1.05,
        duration: int = 600,
        parent: QObject = None,
    ) -> QVariantAnimation:
        """创建缩放脉冲效果"""
        anim = QVariantAnimation(parent or widget)
        anim.setDuration(duration)
        anim.setStartValue(min_scale)
        anim.setEndValue(max_scale)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)

        def on_value_changed(value):
            # 使用 QTransform 或 scale 方式，简化处理
            widget.setProperty("scale", value)
            widget.update()

        anim.valueChanged.connect(on_value_changed)
        anim.finished.connect(lambda: anim.setDirection(
            QPropertyAnimation.Direction.Backward if anim.direction() == QPropertyAnimation.Direction.Forward else QPropertyAnimation.Direction.Forward
        ) or anim.start())
        return anim


class HoverAnimation:
    """悬停动画辅助类，处理鼠标悬停时的微妙反馈"""

    def __init__(self, widget: QWidget, duration: int = 180):
        self.widget = widget
        self.duration = duration
        self._hover_progress = 0.0

        self._anim = QVariantAnimation(widget)
        self._anim.setDuration(duration)
        self._anim.setEasingCurve(Easing.SMOOTH_DECEL)
        self._anim.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value):
        self._hover_progress = float(value)
        self.widget.update()

    def animate_to(self, target: float):
        self._anim.stop()
        self._anim.setStartValue(self._hover_progress)
        self._anim.setEndValue(target)
        self._anim.start()

    def enter(self):
        self.animate_to(1.0)

    def leave(self):
        self.animate_to(0.0)

    @property
    def progress(self) -> float:
        return self._hover_progress


class SkeletonWidget(QWidget):
    """骨架屏加载组件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shimmer_offset = 0.0

        self._timer = QTimer(self)
        self._timer.setInterval(16)  # 60fps
        self._timer.timeout.connect(self._update_shimmer)
        self._timer.start()

    def _update_shimmer(self):
        self._shimmer_offset = (self._shimmer_offset + 0.02) % 2.0
        self.update()

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QLinearGradient, QColor, QBrush

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 绘制骨架背景
        bg_color = QColor(15, 22, 41, 180)
        painter.fillRect(self.rect(), bg_color)

        # 绘制微光效果
        gradient = QLinearGradient(
            self.rect().left() + self._shimmer_offset * self.width() - self.width(),
            0,
            self.rect().left() + self._shimmer_offset * self.width(),
            0,
        )
        gradient.setColorAt(0.0, QColor(15, 22, 41, 0))
        gradient.setColorAt(0.5, QColor(38, 73, 111, 40))
        gradient.setColorAt(1.0, QColor(15, 22, 41, 0))
        painter.fillRect(self.rect(), QBrush(gradient))

        painter.end()


def apply_hover_scale(
    widget: QWidget,
    scale_factor: float = 1.02,
    duration: int = 200,
) -> tuple[QVariantAnimation, QVariantAnimation]:
    """为 widget 应用悬停缩放效果，返回 (enter_anim, leave_anim)"""
    enter_anim = QVariantAnimation(widget)
    enter_anim.setDuration(duration)
    enter_anim.setStartValue(1.0)
    enter_anim.setEndValue(scale_factor)
    enter_anim.setEasingCurve(Easing.OUT_EXPO)

    leave_anim = QVariantAnimation(widget)
    leave_anim.setDuration(duration)
    leave_anim.setStartValue(scale_factor)
    leave_anim.setEndValue(1.0)
    leave_anim.setEasingCurve(Easing.OUT_EXPO)

    def apply_scale(value):
        # 这里需要配合自定义 paint 使用，暂存 scale 值
        widget.setProperty("_hover_scale", float(value))
        widget.update()

    enter_anim.valueChanged.connect(apply_scale)
    leave_anim.valueChanged.connect(apply_scale)

    return enter_anim, leave_anim
