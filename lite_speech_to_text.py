#!/usr/bin/env python3
"""Lite Speech-to-Text: interfaccia system tray con PyQt6."""

import json
import logging
import os
import signal
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QGroupBox, QHBoxLayout,
    QMessageBox, QPushButton, QRadioButton, QSystemTrayIcon, QMenu,
    QVBoxLayout,
)
from pynput import keyboard

from stt_core import (
    SUPPORTED_LANGUAGES, ENGINE_INFO, MIC_TEST_THRESHOLD,
    STTEngine, get_microphones, test_microphone, get_available_engines,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CONFIG_FILE = "config.json"


def get_base_dir() -> Path:
    """Directory base del progetto (o dell'exe se frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def load_config() -> dict | None:
    """Carica configurazione da config.json."""
    config_path = get_base_dir() / CONFIG_FILE
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def save_config(config: dict) -> None:
    """Salva configurazione in config.json."""
    config_path = get_base_dir() / CONFIG_FILE
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def create_icon(color: str) -> QIcon:
    """Crea icona circolare colorata per il system tray."""
    size = 64
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QColor(color))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(4, 4, size - 8, size - 8)
    painter.end()
    return QIcon(pixmap)


# --- Thread-safe bridge per segnali dal worker thread alla UI ---

class SignalBridge(QObject):
    """Ponte segnali tra thread di lavoro STTEngine e thread Qt principale."""
    status_changed = pyqtSignal(str)
    result_ready = pyqtSignal(str, float)
    error_occurred = pyqtSignal(str)


class ModelLoaderThread(QThread):
    """Thread per caricamento modello senza bloccare la UI."""
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, engine: STTEngine) -> None:
        super().__init__()
        self.engine = engine

    def run(self) -> None:
        try:
            self.engine.load_model()
            self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))


# --- Finestra Impostazioni ---

class SettingsDialog(QDialog):
    """Dialog per configurazione microfono, lingua, engine."""

    def __init__(self, parent=None, config: dict | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Lite Speech-to-Text - Impostazioni")
        self.setMinimumWidth(420)
        self.result_config: dict | None = None

        layout = QVBoxLayout(self)

        # Connessione
        conn_group = QGroupBox("Connessione")
        conn_layout = QHBoxLayout()
        self.offline_radio = QRadioButton("OFFLINE (raccomandato)")
        self.online_radio = QRadioButton("ONLINE (modelli mancanti)")
        self.offline_radio.setChecked(True)
        conn_layout.addWidget(self.offline_radio)
        conn_layout.addWidget(self.online_radio)
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)

        # Microfono
        mic_group = QGroupBox("Microfono")
        mic_layout = QHBoxLayout()
        self.mic_combo = QComboBox()
        self.mic_test_btn = QPushButton("Testa")
        self.mic_test_btn.clicked.connect(self._test_microphone)
        mic_layout.addWidget(self.mic_combo, 1)
        mic_layout.addWidget(self.mic_test_btn)
        mic_group.setLayout(mic_layout)
        layout.addWidget(mic_group)

        # Lingua
        lang_group = QGroupBox("Lingua")
        lang_layout = QHBoxLayout()
        self.lang_combo = QComboBox()
        for code, name in SUPPORTED_LANGUAGES.items():
            self.lang_combo.addItem(f"{name} ({code})", code)
        lang_layout.addWidget(self.lang_combo)
        lang_group.setLayout(lang_layout)
        layout.addWidget(lang_group)

        # Engine
        engine_group = QGroupBox("Engine")
        engine_layout = QHBoxLayout()
        self.engine_combo = QComboBox()
        engine_layout.addWidget(self.engine_combo)
        engine_group.setLayout(engine_layout)
        layout.addWidget(engine_group)

        # Bottoni
        btn_layout = QHBoxLayout()
        save_btn = QPushButton("Salva")
        cancel_btn = QPushButton("Annulla")
        save_btn.clicked.connect(self._save)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(save_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # Popola
        self._load_microphones()
        self.offline_radio.toggled.connect(self._update_engines)
        self._update_engines()

        if config:
            self._apply_config(config)

    def _load_microphones(self) -> None:
        self.mic_combo.clear()
        self._devices = get_microphones()
        for idx, name in self._devices:
            self.mic_combo.addItem(name, idx)

    def _update_engines(self) -> None:
        allow_internet = self.online_radio.isChecked()
        available = get_available_engines(allow_internet)
        current = self.engine_combo.currentData()
        self.engine_combo.clear()
        for engine_type, is_available in available.items():
            if is_available:
                info = ENGINE_INFO[engine_type]
                self.engine_combo.addItem(
                    f"{info['name']} - {info['desc']}", engine_type)
        if current:
            idx = self.engine_combo.findData(current)
            if idx >= 0:
                self.engine_combo.setCurrentIndex(idx)

    def _test_microphone(self) -> None:
        device_index = self.mic_combo.currentData()
        if device_index is None:
            return
        self.mic_test_btn.setEnabled(False)
        self.mic_test_btn.setText("Testando...")
        QApplication.processEvents()
        try:
            level = test_microphone(device_index)
            if level < MIC_TEST_THRESHOLD:
                QMessageBox.warning(
                    self, "Test microfono",
                    f"Segnale molto basso ({level:.4f}).\n"
                    "Verifica che il microfono sia attivo.")
            else:
                QMessageBox.information(
                    self, "Test microfono",
                    f"Microfono OK (livello: {level:.4f})")
        except OSError as e:
            QMessageBox.critical(
                self, "Errore", f"Errore apertura microfono: {e}")
        finally:
            self.mic_test_btn.setEnabled(True)
            self.mic_test_btn.setText("Testa")

    def _apply_config(self, config: dict) -> None:
        if config.get("allow_internet"):
            self.online_radio.setChecked(True)
        idx = self.mic_combo.findData(config.get("device_index"))
        if idx >= 0:
            self.mic_combo.setCurrentIndex(idx)
        idx = self.lang_combo.findData(config.get("lang_code"))
        if idx >= 0:
            self.lang_combo.setCurrentIndex(idx)
        self._update_engines()
        idx = self.engine_combo.findData(config.get("engine_type"))
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)

    def _save(self) -> None:
        if self.engine_combo.currentData() is None:
            QMessageBox.warning(
                self, "Errore",
                "Nessun engine disponibile.\n"
                "Prova la modalita' ONLINE per scaricare i modelli.")
            return
        self.result_config = {
            "allow_internet": self.online_radio.isChecked(),
            "device_index": self.mic_combo.currentData(),
            "lang_code": self.lang_combo.currentData(),
            "engine_type": self.engine_combo.currentData(),
        }
        self.accept()


# --- Applicazione System Tray ---

class TrayApp(QObject):
    """Applicazione system tray principale."""

    def __init__(self) -> None:
        super().__init__()
        self.engine: STTEngine | None = None
        self.listener: keyboard.Listener | None = None
        self._loader: ModelLoaderThread | None = None

        # Bridge segnali thread-safe
        self._bridge = SignalBridge()
        self._bridge.status_changed.connect(self._on_status)
        self._bridge.result_ready.connect(self._on_result)
        self._bridge.error_occurred.connect(self._on_error)

        # Icone
        self.icons = {
            "ready": create_icon("#4CAF50"),
            "recording": create_icon("#F44336"),
            "processing": create_icon("#FFC107"),
            "loading": create_icon("#9E9E9E"),
        }

        # System tray
        self.tray = QSystemTrayIcon(self.icons["loading"])
        self.tray.setToolTip("Lite Speech-to-Text")

        self._menu = QMenu()
        self.status_action = QAction("Stato: Avvio...", self._menu)
        self.status_action.setEnabled(False)
        self._menu.addAction(self.status_action)
        self._menu.addSeparator()
        settings_action = QAction("Impostazioni...", self._menu)
        settings_action.triggered.connect(self._show_settings)
        self._menu.addAction(settings_action)
        self._menu.addSeparator()
        quit_action = QAction("Esci", self._menu)
        quit_action.triggered.connect(self._quit)
        self._menu.addAction(quit_action)
        self.tray.setContextMenu(self._menu)
        self.tray.show()

        # Avvio
        config = load_config()
        if config:
            self._start_engine(config)
        else:
            self._show_settings(first_run=True)

    def _show_settings(self, first_run: bool = False) -> None:
        config = load_config()
        dialog = SettingsDialog(config=config)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.result_config:
            save_config(dialog.result_config)
            self._start_engine(dialog.result_config)
        elif first_run:
            self._quit()

    def _start_engine(self, config: dict) -> None:
        # Ferma engine precedente
        self._stop_listener()
        if self.engine:
            self.engine.close()

        self._set_status("loading", "Caricamento modello...")

        self.engine = STTEngine(config)
        # Collega callback dell'engine ai segnali Qt (thread-safe)
        self.engine.on_status = self._bridge.status_changed.emit
        self.engine.on_result = self._bridge.result_ready.emit
        self.engine.on_error = self._bridge.error_occurred.emit

        self._loader = ModelLoaderThread(self.engine)
        self._loader.finished.connect(self._on_model_loaded)
        self._loader.error.connect(self._on_model_error)
        self._loader.start()

    def _on_model_loaded(self) -> None:
        self._set_status("ready", "Pronto")
        self._start_listener()

    def _on_model_error(self, error: str) -> None:
        self._set_status("loading", "Errore")
        self.tray.showMessage(
            "Errore caricamento",
            error,
            QSystemTrayIcon.MessageIcon.Critical,
            5000,
        )

    def _start_listener(self) -> None:
        self.listener = keyboard.Listener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self.listener.start()

    def _stop_listener(self) -> None:
        if self.listener:
            self.listener.stop()
            self.listener = None

    def _on_key_press(self, key) -> bool:
        if not self.engine or self.engine._pasting:
            return True
        try:
            if key == keyboard.Key.ctrl_l:
                self.engine.start_recording()
        except Exception as e:
            log.error("Errore key_press: %s", e)
        return True

    def _on_key_release(self, key) -> bool:
        if not self.engine or self.engine._pasting:
            return True
        try:
            if key == keyboard.Key.ctrl_l:
                self.engine.stop_recording()
        except Exception as e:
            log.error("Errore key_release: %s", e)
        return True

    def _on_status(self, status: str) -> None:
        labels = {
            "ready": "Pronto",
            "recording": "Registrazione...",
            "processing": "Elaborazione...",
            "loading": "Caricamento...",
        }
        self._set_status(status, labels.get(status, status))

    def _on_result(self, text: str, elapsed: float) -> None:
        preview = text[:80] + "..." if len(text) > 80 else text
        self.tray.showMessage(
            f"Incollato ({elapsed:.1f}s)",
            preview,
            QSystemTrayIcon.MessageIcon.Information,
            3000,
        )

    def _on_error(self, error: str) -> None:
        self.tray.showMessage(
            "Errore",
            error,
            QSystemTrayIcon.MessageIcon.Warning,
            5000,
        )

    def _set_status(self, icon_key: str, label: str) -> None:
        if icon_key in self.icons:
            self.tray.setIcon(self.icons[icon_key])
        self.status_action.setText(f"Stato: {label}")
        engine_name = self.engine.engine_name if self.engine else ""
        tooltip = f"Lite STT - {label}"
        if engine_name:
            tooltip += f" ({engine_name})"
        self.tray.setToolTip(tooltip)

    def _quit(self) -> None:
        self._stop_listener()
        if self.engine:
            self.engine.close()
        self.tray.hide()
        QApplication.instance().quit()


def main() -> None:
    """Entry point dell'applicazione."""
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)

    # Ctrl+C nel terminale chiude l'app (PyQt su Windows non gestisce SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    tray_app = TrayApp()  # noqa: F841 (must keep reference)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
