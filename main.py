import sys
import os
import logging
import traceback
from datetime import datetime
from pathlib import Path

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt


def setup_global_exception_hook():
    log_directory = Path(__file__).parent
    log_file_path = log_directory / "error.log"

    logging.basicConfig(
        filename=str(log_file_path),
        level=logging.ERROR,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        encoding="utf-8",
    )

    def global_exception_handler(exc_type, exc_value, exc_tb):
        if exc_type is KeyboardInterrupt:
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "=" * 80
        crash_report = (
            f"\n{separator}\n"
            f"UNCAUGHT EXCEPTION - {timestamp}\n"
            f"{separator}\n"
            f"Type: {exc_type.__module__}.{exc_type.__qualname__}\n"
            f"Value: {exc_value}\n\n"
            f"Traceback:\n{''.join(traceback.format_exception(exc_type, exc_value, exc_tb))}"
            f"{separator}\n"
        )

        logging.error(crash_report)

        try:
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                log_file.write(crash_report)
        except OSError:
            pass

        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = global_exception_handler


def configure_high_dpi():
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_SCALE_FACTOR_ROUNDING_POLICY"] = "PassThrough"


def create_application(argv: list[str]) -> QApplication:
    configure_high_dpi()

    app = QApplication(argv)
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    modern_font = QFont("Segoe UI Variable Text", 9)
    modern_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    modern_font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(modern_font)

    app.setStyle("Fusion")

    return app


def main() -> int:
    setup_global_exception_hook()

    app = create_application(sys.argv)
    app.setApplicationName("VisionDataPlatform")
    app.setApplicationVersion("1.1.1")
    app.setOrganizationName("VisionBata")

    try:
        from gui.main_window import MainWindow
        from gui.splash_screen import SplashScreen

        splash = SplashScreen()
        main_window = MainWindow()
        main_window.setWindowOpacity(0.0)

        def show_main_window() -> None:
            main_window.show()
            fade_in = QPropertyAnimation(main_window, b"windowOpacity", main_window)
            fade_in.setDuration(320)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
            main_window._play_feedback_animation(fade_in)

        splash.finished.connect(show_main_window)
        splash.start()
    except ImportError as import_error:
        logging.critical(
            "Failed to import MainWindow: %s. "
            "Ensure gui/main_window.py exists and is correctly implemented.",
            import_error,
        )
        return 1
    except Exception as runtime_error:
        logging.critical("Failed to initialize MainWindow: %s", runtime_error)
        raise

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
