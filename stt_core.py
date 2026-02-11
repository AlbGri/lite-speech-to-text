#!/usr/bin/env python3
"""Core STT: audio recording, trascrizione, gestione modelli."""

import json
import logging
import multiprocessing
import os
import platform
import sys
import threading
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
import pyaudio
import pyperclip
from pynput import keyboard

log = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "it": "Italiano",
    "en": "English",
    "es": "Espanol",
    "fr": "Francais",
    "de": "Deutsch",
    "pt": "Portugues",
}

AUDIO_CHUNK = 4096
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
MIN_RECORDING_DURATION = 0.3
MIC_TEST_DURATION = 1.5
MIC_TEST_THRESHOLD = 0.005

VOSK_MODEL_URLS = {
    "it": "https://alphacephei.com/vosk/models/vosk-model-small-it-0.22.zip",
    "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "es": "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip",
    "fr": "https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip",
    "de": "https://alphacephei.com/vosk/models/vosk-model-small-de-0.15.zip",
    "pt": "https://alphacephei.com/vosk/models/vosk-model-small-pt-0.3.zip",
}

ENGINE_INFO = {
    "whisper_small": {
        "name": "WHISPER SMALL",
        "model": "small",
        "desc": "1-2s, GPU",
        "device": "GPU CUDA",
    },
    "whisper_turbo": {
        "name": "WHISPER TURBO",
        "model": "large-v3-turbo",
        "desc": "3-5s, GPU, massima accuratezza (raccomandato)",
        "device": "GPU CUDA",
    },
    "vosk": {
        "name": "VOSK",
        "model": None,
        "desc": "0.5-1.7s, CPU",
        "device": "CPU",
    },
}


def get_base_dir() -> Path:
    """Directory base del progetto (o dell'exe se frozen)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def get_models_dir() -> Path:
    """Directory models/ per modelli STT."""
    return get_base_dir() / "models"


def detect_cpu_config() -> tuple[int, bool]:
    """Rileva CPU e restituisce (optimal_threads, is_amd)."""
    cpu_count = multiprocessing.cpu_count()
    is_amd = "AMD" in platform.processor().upper() or cpu_count >= 12
    if is_amd:
        threads = min(cpu_count // 2, 6)
    else:
        threads = min(cpu_count, 8)
    return threads, is_amd


def setup_environment(allow_internet: bool, threads: int) -> None:
    """Configura variabili d'ambiente per offline mode e threading."""
    if not allow_internet:
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE", "TOKENIZERS_OFFLINE",
                    "HF_HUB_DISABLE_TELEMETRY", "DISABLE_TELEMETRY"):
            os.environ[var] = "1"
    else:
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE", "TOKENIZERS_OFFLINE"):
            os.environ.pop(var, None)

    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_microphones() -> list[tuple[int, str]]:
    """Elenca microfoni WASAPI disponibili. Ritorna [(device_index, name)]."""
    p = pyaudio.PyAudio()
    try:
        wasapi_index = None
        for i in range(p.get_host_api_count()):
            api_info = p.get_host_api_info_by_index(i)
            if "wasapi" in api_info["name"].lower():
                wasapi_index = i
                break

        devices: list[tuple[int, str]] = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] <= 0:
                continue
            if wasapi_index is not None and info["hostApi"] != wasapi_index:
                continue
            devices.append((i, info["name"]))
        return devices
    finally:
        p.terminate()


def test_microphone(device_index: int) -> float:
    """Testa un microfono. Ritorna il livello medio (0.0 - 1.0)."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=AUDIO_CHUNK,
        )
        frames = []
        chunks_needed = int(AUDIO_RATE * MIC_TEST_DURATION / AUDIO_CHUNK) + 1
        for _ in range(chunks_needed):
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            frames.append(data)
        stream.stop_stream()
        stream.close()

        audio = np.frombuffer(b"".join(frames), dtype=np.int16)
        return float(np.abs(audio).mean() / 32768.0)
    finally:
        p.terminate()


def check_whisper_lib() -> bool:
    """Controlla se faster-whisper e' installata."""
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def find_whisper_model(model_name: str) -> str | None:
    """Cerca modello Whisper in models/. Ritorna path o None."""
    models_dir = get_models_dir()
    if not models_dir.exists():
        return None
    try:
        for item in models_dir.iterdir():
            if not (item.is_dir()
                    and "whisper" in item.name.lower()
                    and model_name in item.name.lower()):
                continue
            # Struttura HuggingFace cache: item/snapshots/<hash>/
            snapshots = item / "snapshots"
            if snapshots.exists():
                for snap in sorted(snapshots.iterdir(), reverse=True):
                    if snap.is_dir() and (snap / "model.bin").exists():
                        return str(snap)
            if (item / "model.bin").exists():
                return str(item)
    except OSError:
        pass
    return None


def check_vosk_lib() -> bool:
    """Controlla se vosk e' installata."""
    try:
        import vosk  # noqa: F401
        return True
    except ImportError:
        return False


def find_vosk_model(lang_code: str) -> str | None:
    """Cerca un modello Vosk per la lingua specificata."""
    base_dir = get_base_dir()
    for search_dir in [base_dir, base_dir / "models"]:
        if not search_dir.exists():
            continue
        try:
            for item in search_dir.iterdir():
                if (item.is_dir()
                        and "vosk-model" in item.name
                        and f"-{lang_code}" in item.name):
                    return str(item)
        except OSError:
            continue
    return None


def download_vosk_model(lang_code: str) -> str:
    """Scarica modello Vosk per la lingua, estrae in models/, rimuove zip.

    Returns:
        Path della directory del modello estratto.
    """
    url = VOSK_MODEL_URLS.get(lang_code)
    if not url:
        raise ValueError(f"Nessun modello Vosk disponibile per '{lang_code}'")

    models_dir = get_models_dir()
    models_dir.mkdir(exist_ok=True)
    zip_name = url.split("/")[-1]
    zip_path = models_dir / zip_name

    log.info("Download modello Vosk da %s...", url)
    urllib.request.urlretrieve(url, zip_path)

    log.info("Estrazione in %s...", models_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(models_dir)

    zip_path.unlink()
    log.info("Modello Vosk scaricato")

    model_path = find_vosk_model(lang_code)
    if not model_path:
        raise FileNotFoundError("Modello estratto ma non trovato in models/")
    return model_path


def get_available_engines(allow_internet: bool) -> dict[str, bool]:
    """Rileva engine disponibili. Ritorna {engine_type: disponibile}."""
    has_whisper = check_whisper_lib()
    return {
        "whisper_small": has_whisper and (
            find_whisper_model("small") is not None or allow_internet),
        "whisper_turbo": has_whisper and (
            find_whisper_model("large-v3-turbo") is not None
            or allow_internet),
        "vosk": check_vosk_lib() and (
            any(find_vosk_model(lc) for lc in SUPPORTED_LANGUAGES)
            or allow_internet),
    }


class STTEngine:
    """Engine STT: gestione audio, registrazione, trascrizione."""

    def __init__(self, config: dict) -> None:
        self.device_index: int = config["device_index"]
        self.lang_code: str = config["lang_code"]
        self.engine_type: str = config["engine_type"]
        self.allow_internet: bool = config.get("allow_internet", False)

        info = ENGINE_INFO[self.engine_type]
        self.engine_name: str = info["name"]
        self.model_name: str | None = info["model"]
        self.device_info: str = info["device"]

        self.threads, _ = detect_cpu_config()
        setup_environment(self.allow_internet, self.threads)

        # Stato
        self.is_recording = False
        self.audio_frames: list[bytes] = []
        self.start_time: float = 0.0
        self._processing = False
        self._pasting = False
        self.keyboard_controller = keyboard.Controller()
        self.model = None
        self.vosk_model = None

        # Callback per la UI (chiamati da thread di lavoro)
        self.on_status: Callable[[str], None] = lambda s: None
        self.on_result: Callable[[str, float], None] = lambda t, d: None
        self.on_error: Callable[[str], None] = lambda e: None

        # Stream audio persistente (evita hook timeout Windows)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=AUDIO_FORMAT,
            channels=AUDIO_CHANNELS,
            rate=AUDIO_RATE,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=AUDIO_CHUNK,
        )

    def load_model(self) -> None:
        """Carica il modello STT selezionato."""
        if self.engine_type in ("whisper_small", "whisper_turbo"):
            from faster_whisper import WhisperModel
            models_dir = get_models_dir()
            models_dir.mkdir(exist_ok=True)
            local_path = find_whisper_model(self.model_name)
            if local_path:
                self.model = WhisperModel(
                    local_path,
                    device="auto",
                    compute_type="auto",
                    num_workers=1,
                )
            else:
                self.model = WhisperModel(
                    self.model_name,
                    device="auto",
                    compute_type="auto",
                    num_workers=1,
                    download_root=str(models_dir),
                    local_files_only=not self.allow_internet,
                )
        elif self.engine_type == "vosk":
            import vosk
            model_path = find_vosk_model(self.lang_code)
            if not model_path and self.allow_internet:
                model_path = download_vosk_model(self.lang_code)
            if not model_path:
                raise FileNotFoundError(
                    f"Modello Vosk per '{self.lang_code}' non trovato. "
                    "Scarica da: https://alphacephei.com/vosk/models")
            self.vosk_model = vosk.Model(model_path)

    def start_recording(self) -> None:
        """Avvia registrazione audio."""
        if self.is_recording or self._processing:
            return
        self.is_recording = True
        self.audio_frames = []
        self.start_time = time.time()
        self.on_status("recording")
        thread = threading.Thread(target=self._record_loop, daemon=True)
        thread.start()

    def _record_loop(self) -> None:
        """Loop cattura audio in thread separato."""
        while self.is_recording:
            try:
                data = self.stream.read(
                    AUDIO_CHUNK, exception_on_overflow=False)
                self.audio_frames.append(data)
            except OSError:
                break

    def stop_recording(self) -> None:
        """Ferma registrazione e avvia elaborazione in thread."""
        if not self.is_recording:
            return
        self.is_recording = False
        duration = time.time() - self.start_time
        thread = threading.Thread(
            target=self._process_audio,
            args=(list(self.audio_frames), duration),
            daemon=True,
        )
        thread.start()

    def _process_audio(self, frames: list[bytes], duration: float) -> None:
        """Elabora audio in thread separato."""
        self._processing = True
        self.on_status("processing")
        try:
            if duration < MIN_RECORDING_DURATION or not frames:
                return

            t0 = time.perf_counter()
            audio_data = b"".join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32, copy=False) / 32768.0

            text = self._transcribe(audio_float)
            elapsed = time.perf_counter() - t0

            if text:
                text_clean = " ".join(text.split())
                if text_clean[0].islower():
                    text_clean = text_clean[0].upper() + text_clean[1:]
                if text_clean[-1] not in ".!?":
                    text_clean += "."
                self._paste_text(text_clean)
                self.on_result(text_clean, elapsed)
            else:
                log.info("Nessun testo rilevato (%.2fs)", elapsed)

        except Exception as e:
            log.error("Errore elaborazione: %s", e)
            self.on_error(str(e))
        finally:
            self._processing = False
            self.on_status("ready")

    def _transcribe(self, audio_float: np.ndarray) -> str:
        """Trascrive audio usando l'engine selezionato."""
        if self.engine_type in ("whisper_small", "whisper_turbo"):
            return self._transcribe_whisper(audio_float)
        if self.engine_type == "vosk":
            return self._transcribe_vosk(audio_float)
        return ""

    def _transcribe_whisper(self, audio_float: np.ndarray) -> str:
        """Trascrizione con Whisper."""
        beam = 5 if self.engine_type == "whisper_turbo" else 1
        segments, _ = self.model.transcribe(
            audio_float,
            language=self.lang_code,
            beam_size=beam,
            temperature=0.0,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
            no_speech_threshold=0.6,
            suppress_blank=True,
        )
        parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(parts).strip()

    def _transcribe_vosk(self, audio_float: np.ndarray) -> str:
        """Trascrizione con Vosk, con deduplicazione."""
        import vosk

        recognizer = vosk.KaldiRecognizer(self.vosk_model, AUDIO_RATE)
        audio_bytes = (audio_float * 32768).astype(np.int16).tobytes()
        recognizer.AcceptWaveform(audio_bytes)

        result = json.loads(recognizer.FinalResult())
        text = result.get("text", "").strip()
        if not text:
            return ""

        # Rimuovi ripetizioni immediate
        words = text.split()
        deduped: list[str] = []
        for word in words:
            if not deduped or word != deduped[-1]:
                deduped.append(word)

        # Rimuovi sequenze ripetute di 2-3 parole
        final: list[str] = []
        i = 0
        while i < len(deduped):
            final.append(deduped[i])
            if (i + 3 < len(deduped)
                    and deduped[i:i + 2] == deduped[i + 2:i + 4]):
                i += 2
                continue
            if (i + 5 < len(deduped)
                    and deduped[i:i + 3] == deduped[i + 3:i + 6]):
                i += 3
                continue
            i += 1

        return " ".join(final).strip()

    def _paste_text(self, text: str) -> None:
        """Copia in clipboard e incolla con Ctrl+V + Enter."""
        pyperclip.copy(text)
        self._pasting = True
        try:
            ctrl = keyboard.Key.ctrl_l
            v_key = keyboard.KeyCode.from_char("v")
            self.keyboard_controller.press(ctrl)
            self.keyboard_controller.press(v_key)
            self.keyboard_controller.release(v_key)
            self.keyboard_controller.release(ctrl)
            time.sleep(0.05)
            self.keyboard_controller.press(keyboard.Key.enter)
            self.keyboard_controller.release(keyboard.Key.enter)
            time.sleep(0.05)
        finally:
            self._pasting = False

    def close(self) -> None:
        """Rilascia risorse audio."""
        self.is_recording = False
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            if self.p:
                self.p.terminate()
                self.p = None
        except OSError:
            pass
