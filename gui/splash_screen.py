from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QWidget


class SplashScreen(QWidget):
    finished = pyqtSignal()

    _UI_RADIUS = 5

    _MESSAGES = (
        "checking cameras...",
        "warming up inference...",
        "tuning signal lines...",
        "ready, probably.",
    )

    def __init__(self, duration_ms: int = 2200, parent=None):
        super().__init__(parent)
        self._duration_ms = duration_ms
        self._progress = 0.0
        self._message_index = 0
        self._fade_anim: QPropertyAnimation | None = None

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.SplashScreen
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(620, 360)
        self.setWindowOpacity(0.0)

        self._tick = QTimer(self)
        self._tick.setInterval(16)
        self._tick.timeout.connect(self._advance)

        self._message_timer = QTimer(self)
        self._message_timer.setInterval(max(360, duration_ms // len(self._MESSAGES)))
        self._message_timer.timeout.connect(self._advance_message)

    def start(self) -> None:
        self._center_on_screen()
        self.show()

        self._fade_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_anim.setDuration(280)
        self._fade_anim.setStartValue(0.0)
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._fade_anim.start()

        self._tick.start()
        self._message_timer.start()
        QTimer.singleShot(self._duration_ms, self.finish)

    def finish(self) -> None:
        self._tick.stop()
        self._message_timer.stop()
        self._fade_anim = QPropertyAnimation(self, b"windowOpacity", self)
        self._fade_anim.setDuration(260)
        self._fade_anim.setStartValue(self.windowOpacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._fade_anim.finished.connect(self._finish)
        self._fade_anim.start()

    def _finish(self) -> None:
        self.hide()
        self.finished.emit()
        self.deleteLater()

    def _center_on_screen(self) -> None:
        screen = self.screen()
        if screen is None:
            return
        available = screen.availableGeometry()
        self.move(
            available.x() + (available.width() - self.width()) // 2,
            available.y() + (available.height() - self.height()) // 2,
        )

    def _advance(self) -> None:
        self._progress = min(1.0, self._progress + 16 / max(1, self._duration_ms))
        self.update()

    def _advance_message(self) -> None:
        self._message_index = min(len(self._MESSAGES) - 1, self._message_index + 1)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        card = QRectF(8, 8, self.width() - 16, self.height() - 16)
        gradient = QLinearGradient(card.topLeft(), card.bottomRight())
        gradient.setColorAt(0.0, QColor("#0B1020"))
        gradient.setColorAt(0.58, QColor("#12182B"))
        gradient.setColorAt(1.0, QColor("#1A2238"))
        painter.setPen(QPen(QColor(70, 96, 142, 150), 1.2))
        painter.setBrush(gradient)
        painter.drawRoundedRect(card, self._UI_RADIUS, self._UI_RADIUS)

        self._paint_signal_lines(painter, card)

        painter.setPen(QColor("#EAF1FF"))
        title_font = QFont("Segoe UI Variable Text", 30, QFont.Weight.Bold)
        painter.setFont(title_font)
        painter.drawText(QRectF(0, 92, self.width(), 52), Qt.AlignmentFlag.AlignCenter, "VDP V1.1")

        painter.setPen(QColor("#9AA8C7"))
        sub_font = QFont("Segoe UI Variable Text", 10, QFont.Weight.Medium)
        painter.setFont(sub_font)
        painter.drawText(
            QRectF(0, 148, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter,
            "visual capture suite / low-key signal mode",
        )

        msg_font = QFont("Consolas", 9, QFont.Weight.Normal)
        painter.setFont(msg_font)
        painter.setPen(QColor("#8FE0FF"))
        painter.drawText(
            QRectF(0, 232, self.width(), 26),
            Qt.AlignmentFlag.AlignCenter,
            self._MESSAGES[self._message_index],
        )

        track = QRectF(160, 276, 300, 5)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(39, 48, 74, 220))
        painter.drawRoundedRect(track, 2.5, 2.5)
        fill = QRectF(track.left(), track.top(), track.width() * self._progress, track.height())
        painter.setBrush(QColor("#6FA8FF"))
        painter.drawRoundedRect(fill, 2.5, 2.5)

        painter.end()

    def _paint_signal_lines(self, painter: QPainter, card: QRectF) -> None:
        line_pen = QPen(QColor(111, 168, 255, 42), 1.0)
        painter.setPen(line_pen)
        base_y = card.top() + 194
        start_x = card.left() + 98
        for row in range(4):
            y = base_y + row * 12
            painter.drawLine(int(start_x), int(y), int(card.right() - 98), int(y))

        pulse_x = start_x + (card.width() - 196) * self._progress
        pulse_pen = QPen(QColor(143, 224, 255, 160), 2.0)
        painter.setPen(pulse_pen)
        painter.drawLine(int(pulse_x), int(base_y - 8), int(pulse_x), int(base_y + 44))
