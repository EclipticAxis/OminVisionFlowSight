from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox


class _WheelFocusMixin:
    def _init_wheel_focus(self) -> None:
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, event) -> None:
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()

    def leaveEvent(self, event) -> None:
        self.clearFocus()
        super().leaveEvent(event)


class WheelFocusComboBox(_WheelFocusMixin, QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_wheel_focus()


class WheelFocusSpinBox(_WheelFocusMixin, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_wheel_focus()


class WheelFocusDoubleSpinBox(_WheelFocusMixin, QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_wheel_focus()
