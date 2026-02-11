#!/usr/bin/env python3
"""Lite Speech-to-Text: trascrizione vocale con Whisper (GPU) e Vosk (CPU)."""

import json
import logging
import multiprocessing
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pyaudio
import pyperclip
from pynput import keyboard

logging.basicConfig(level=logging.INFO, format="%(message)s")
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


@contextmanager
def timer(label: str):
    """Misura e stampa il tempo di esecuzione di un blocco."""
    t0 = time.perf_counter()
    yield
    log.info("%s: %.2fs", label, time.perf_counter() - t0)


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


def choose_option(prompt: str, options: list[tuple[str, str]], default: str = "1") -> str:
    """Menu interattivo. Restituisce la chiave associata alla scelta."""
    for key, label in options:
        log.info("  %s. %s", key, label)
    print()
    while True:
        choice = input(f"{prompt}: ").strip() or default
        for key, label in options:
            if choice == key:
                return key
        log.warning("Scelta non valida")


def find_vosk_model(lang_code: str) -> str | None:
    """Cerca un modello Vosk per la lingua specificata."""
    if getattr(sys, "frozen", False):
        base_dir = Path(sys.executable).resolve().parent
    else:
        base_dir = Path(__file__).resolve().parent

    search_dirs = [
        base_dir,
        base_dir / "models",
        Path.home() / ".cache" / "vosk",
    ]

    for search_dir in search_dirs:
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


class LiteSpeechToText:
    """Trascrizione vocale con supporto Whisper (GPU) e Vosk (CPU)."""

    def __init__(self) -> None:
        log.info("LITE SPEECH-TO-TEXT")
        log.info("=" * 45)
        print()

        # Connessione
        log.info("CONNESSIONE INTERNET:")
        log.info("-" * 25)
        internet_choice = choose_option(
            "Scegli modalita' (1/2)",
            [("1", "OFFLINE - Nessuna connessione (raccomandato)"),
             ("2", "ONLINE - Per modelli mancanti")],
        )
        self.allow_internet = internet_choice == "2"
        log.info("Modalita' %s selezionata",
                 "ONLINE" if self.allow_internet else "OFFLINE")
        print()

        # CPU e ambiente
        self.threads, is_amd = detect_cpu_config()
        setup_environment(self.allow_internet, self.threads)
        if is_amd:
            log.info("Hardware: AMD %d cores -> %d threads",
                     multiprocessing.cpu_count(), self.threads)
        else:
            log.info("Hardware: %d threads", self.threads)
        print()

        # Microfono
        self.device_index = self._select_microphone()

        # Lingua
        self.lang_code = self._choose_language()

        # Stato
        self.is_recording = False
        self.audio_frames: list[bytes] = []
        self.model = None
        self.vosk_model = None
        self.engine_type: str = ""
        self.engine_name: str = ""
        self.expected_time: str = ""
        self.device_info: str = ""
        self.start_time: float = 0.0
        self.keyboard_controller = keyboard.Controller()
        self.esc_press_count = 0
        self.last_esc_press: float = 0.0
        self._pasting = False
        self._processing = False

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

        # Inizializzazione engine
        self._check_engines_and_choose()
        self._load_model()
        self._show_ready_message()

    def _select_microphone(self) -> int:
        """Elenca dispositivi di input audio (WASAPI), permette selezione e test."""
        log.info("MICROFONO:")
        log.info("-" * 25)

        p = pyaudio.PyAudio()
        try:
            # Cerca host API WASAPI (migliore qualita'/latenza su Windows)
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
                # Filtra per WASAPI se disponibile, altrimenti mostra tutti
                if wasapi_index is not None and info["hostApi"] != wasapi_index:
                    continue
                devices.append((i, info["name"]))

            if not devices:
                log.error("Nessun dispositivo di input trovato")
                sys.exit(1)

            for idx, (_, name) in enumerate(devices):
                log.info("  %d. %s", idx + 1, name)
            print()

            while True:
                choice = input(
                    f"Scegli microfono (1-{len(devices)}) [1]: "
                ).strip() or "1"
                try:
                    sel = int(choice) - 1
                    if 0 <= sel < len(devices):
                        break
                except ValueError:
                    pass
                log.warning("Scelta non valida")

            device_index, device_name = devices[sel]
            log.info("Microfono: [%d] %s", device_index, device_name)
            print()

            # Test microfono
            log.info("Test microfono (parla per %.1f secondi)...",
                     MIC_TEST_DURATION)
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
                chunks_needed = int(
                    AUDIO_RATE * MIC_TEST_DURATION / AUDIO_CHUNK) + 1
                for _ in range(chunks_needed):
                    data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                    frames.append(data)
                stream.stop_stream()
                stream.close()

                audio = np.frombuffer(b"".join(frames), dtype=np.int16)
                amplitude = np.abs(audio).mean() / 32768.0

                if amplitude < MIC_TEST_THRESHOLD:
                    log.warning(
                        "Segnale molto basso (%.4f). "
                        "Verifica che il microfono sia attivo.", amplitude)
                else:
                    log.info("Microfono OK (livello: %.4f)", amplitude)
            except OSError as e:
                log.error("Errore apertura microfono [%d]: %s",
                          device_index, e)
                sys.exit(1)

            print()
            return device_index
        finally:
            p.terminate()

    def _choose_language(self) -> str:
        """Selezione lingua per la trascrizione."""
        log.info("LINGUA:")
        log.info("-" * 25)
        options = [(str(i + 1), f"{name} ({code})")
                   for i, (code, name) in enumerate(SUPPORTED_LANGUAGES.items())]
        choice = choose_option(f"Scegli lingua (1-{len(options)})", options)
        lang_code = list(SUPPORTED_LANGUAGES.keys())[int(choice) - 1]
        log.info("Lingua: %s (%s)",
                 SUPPORTED_LANGUAGES[lang_code], lang_code)
        print()
        return lang_code

    def _check_engines_and_choose(self) -> None:
        """Verifica engines disponibili e permette la scelta."""
        log.info("ENGINES DISPONIBILI:")
        log.info("-" * 25)

        has_whisper_lib = self._check_whisper_lib()
        whisper_small = has_whisper_lib and (
            self._find_whisper_model("small") is not None or self.allow_internet)
        whisper_turbo = has_whisper_lib and (
            self._find_whisper_model("large-v3-turbo") is not None
            or self.allow_internet)
        vosk_available = self._check_vosk()

        log.info("Whisper Small: %s",
                 "disponibile" if whisper_small else "non trovato")
        log.info("Whisper Turbo: %s",
                 "disponibile" if whisper_turbo else "non trovato")
        log.info("Vosk: %s",
                 "disponibile" if vosk_available else "non trovato")
        print()

        options: list[tuple[str, str, str, str]] = []
        if whisper_small:
            options.append((str(len(options) + 1), "WHISPER SMALL",
                            "whisper_small", "1-2s, GPU"))
        if whisper_turbo:
            options.append((str(len(options) + 1), "WHISPER TURBO",
                            "whisper_turbo",
                            "3-5s, GPU, massima accuratezza (raccomandato)"))
        if vosk_available:
            options.append((str(len(options) + 1), "VOSK", "vosk",
                            "0.5-1.7s, CPU"))

        if not options:
            log.error("Nessun engine disponibile. "
                      "Installa: pip install faster-whisper vosk")
            sys.exit(1)

        log.info("Seleziona engine:")
        menu_opts = [(num, f"{name} - {desc}")
                     for num, name, _, desc in options]
        choice = choose_option(f"Scegli engine (1-{len(options)})", menu_opts)

        for num, name, engine, _desc in options:
            if choice == num:
                self.engine_type = engine
                self.engine_name = name
                log.info("Engine: %s", name)
                print()
                self._setup_engine_params()
                return

    def _check_whisper_lib(self) -> bool:
        """Controlla se la libreria faster-whisper e' installata."""
        try:
            import faster_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_models_dir(self) -> Path:
        """Restituisce la cartella models/ del progetto."""
        if getattr(sys, "frozen", False):
            base = Path(sys.executable).resolve().parent
        else:
            base = Path(__file__).resolve().parent
        return base / "models"

    def _find_whisper_model(self, model_name: str) -> str | None:
        """Cerca modello Whisper in models/.

        Naviga la struttura HuggingFace cache creata da download_root
        per trovare la directory con i file del modello.

        Returns:
            Path della directory del modello, None se non trovato.
        """
        models_dir = self._get_models_dir()
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
                # Directory con file modello diretto
                if (item / "model.bin").exists():
                    return str(item)
        except OSError:
            pass
        return None

    def _check_vosk(self) -> bool:
        """Controlla se vosk e' disponibile."""
        try:
            import vosk  # noqa: F401
            return True
        except ImportError:
            return False

    def _setup_engine_params(self) -> None:
        """Configura parametri per l'engine selezionato."""
        if self.engine_type == "whisper_small":
            self.model_name = "small"
            self.expected_time = "1-2 secondi"
            self.device_info = "GPU CUDA"
        elif self.engine_type == "whisper_turbo":
            self.model_name = "large-v3-turbo"
            self.expected_time = "3-5 secondi"
            self.device_info = "GPU CUDA"
        elif self.engine_type == "vosk":
            self.expected_time = "0.5-1.7 secondi"
            self.device_info = "CPU"

    def _load_model(self) -> None:
        """Carica il modello STT selezionato."""
        log.info("Caricamento %s...", self.engine_name)

        try:
            if self.engine_type in ("whisper_small", "whisper_turbo"):
                from faster_whisper import WhisperModel
                models_dir = self._get_models_dir()
                models_dir.mkdir(exist_ok=True)
                local_path = self._find_whisper_model(self.model_name)
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
                if not model_path:
                    log.error(
                        "Modello Vosk per '%s' non trovato. "
                        "Scarica da: https://alphacephei.com/vosk/models",
                        self.lang_code,
                    )
                    sys.exit(1)
                self.vosk_model = vosk.Model(model_path)
                log.info("Modello Vosk: %s", model_path)

            log.info("Engine %s caricato", self.engine_name)

        except Exception as e:
            log.error("Errore caricamento engine: %s", e)
            sys.exit(1)

    def _show_ready_message(self) -> None:
        """Mostra configurazione e controlli."""
        print()
        log.info("=" * 45)
        log.info("SISTEMA PRONTO")
        log.info("=" * 45)
        print()
        log.info("CONTROLLI:")
        log.info("  CTRL SINISTRO = Registra")
        log.info("  ESC x2 = Esci")
        print()
        log.info("CONFIGURAZIONE:")
        log.info("  Microfono: [%d]", self.device_index)
        log.info("  Internet: %s",
                 "abilitato" if self.allow_internet else "disabilitato")
        log.info("  Engine: %s", self.engine_name)
        log.info("  Device: %s", self.device_info)
        log.info("  Lingua: %s (%s)",
                 SUPPORTED_LANGUAGES[self.lang_code], self.lang_code)
        log.info("  Threads: %d", self.threads)
        log.info("  Tempo atteso: %s", self.expected_time)
        print()
        log.info("Pronto! Tieni premuto CTRL SINISTRO...")
        log.info("=" * 45)

    def transcribe_audio(self, audio_float: np.ndarray) -> str:
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
        """Trascrizione con Vosk, con deduplicazione post-processing."""
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

    def _start_recording(self) -> None:
        """Avvia registrazione audio."""
        if self.is_recording or self._processing:
            return
        self.is_recording = True
        self.audio_frames = []
        self.start_time = time.time()
        log.info("\nRegistrazione... (rilascia CTRL)")

        thread = threading.Thread(target=self._record_loop, daemon=True)
        thread.start()

    def _record_loop(self) -> None:
        """Loop di cattura audio in thread separato."""
        while self.is_recording:
            try:
                data = self.stream.read(
                    AUDIO_CHUNK, exception_on_overflow=False)
                self.audio_frames.append(data)
            except OSError:
                break

    def _stop_recording(self) -> None:
        """Ferma registrazione e avvia elaborazione in thread separato."""
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
        """Elabora audio in thread separato (non blocca il keyboard hook)."""
        self._processing = True
        try:
            self._process_audio_inner(frames, duration)
        finally:
            self._processing = False

    def _process_audio_inner(
        self, frames: list[bytes], duration: float
    ) -> None:
        """Logica di elaborazione audio."""
        log.info("\nElaborazione...")

        if duration < MIN_RECORDING_DURATION:
            log.info("Registrazione troppo breve")
            log.info("\nPronto")
            return

        if not frames:
            log.info("Nessun audio registrato")
            log.info("\nPronto")
            return

        total_start = time.perf_counter()

        try:
            audio_data = b"".join(frames)
            audio_np = np.frombuffer(audio_data, dtype=np.int16)
            audio_float = audio_np.astype(np.float32, copy=False) / 32768.0

            with timer(self.engine_name):
                text = self.transcribe_audio(audio_float)

            if text:
                text_clean = " ".join(text.split())
                if text_clean[0].islower():
                    text_clean = text_clean[0].upper() + text_clean[1:]
                if text_clean[-1] not in ".!?":
                    text_clean += "."

                log.info('\nTESTO: "%s"', text_clean)

                with timer("Output"):
                    pyperclip.copy(text_clean)
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
                    log.info("Incollato!")
            else:
                log.info("Nessun testo rilevato")

            log.info("\nTOTALE: %.2fs", time.perf_counter() - total_start)

        except Exception as e:
            log.error("Errore elaborazione: %s", e)

        log.info("\nPronto")

    def _close_audio(self) -> None:
        """Chiude le risorse audio."""
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

    def _on_esc(self) -> bool:
        """Gestisce doppio ESC per uscita."""
        now = time.time()
        if now - self.last_esc_press <= 2.0:
            self.esc_press_count += 1
        else:
            self.esc_press_count = 1
        self.last_esc_press = now
        log.info("\nESC (%d/2)", self.esc_press_count)

        if self.esc_press_count >= 2:
            log.info("Uscita...")
            self._close_audio()
            return False
        return True

    def _on_key_press(self, key) -> bool | None:
        """Handler pressione tasti."""
        if self._pasting:
            return True
        try:
            if key == keyboard.Key.ctrl_l:
                self._start_recording()
            elif key == keyboard.Key.esc:
                return self._on_esc()
        except Exception as e:
            log.error("Errore key_press: %s", e)
        return True

    def _on_key_release(self, key) -> bool | None:
        """Handler rilascio tasti."""
        if self._pasting:
            return True
        try:
            if key == keyboard.Key.ctrl_l:
                self._stop_recording()
        except Exception as e:
            log.error("Errore key_release: %s", e)
        return True

    def run(self) -> None:
        """Loop principale con listener tastiera."""
        try:
            with keyboard.Listener(
                on_press=self._on_key_press,
                on_release=self._on_key_release,
            ) as listener:
                listener.join()
        finally:
            self._close_audio()


def main() -> None:
    """Entry point dell'applicazione."""
    if getattr(sys, "frozen", False):
        os.chdir(Path(sys.executable).resolve().parent)

    try:
        app = LiteSpeechToText()
        app.run()
    except KeyboardInterrupt:
        log.info("\nInterruzione utente")
    except Exception as e:
        log.error("Errore: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
