# Lite Speech-to-Text

Trascrizione vocale in tempo reale con supporto multi-engine e multi-lingua.

Tieni premuto **CTRL sinistro** per registrare, rilascia per trascrivere. Il testo viene copiato in clipboard e incollato automaticamente.

## Engines

| Engine | Device | Tempo | Note |
|--------|--------|-------|------|
| Whisper Small | GPU CUDA | 1-2s | Buon compromesso velocita'/accuratezza |
| Whisper Medium | GPU CUDA | 6-7s | Massima accuratezza |
| Vosk | CPU | 0.5-1.7s | Leggero, completamente offline |

## Lingue supportate

Italiano, English, Espanol, Francais, Deutsch, Portugues.

Per Vosk, scaricare il modello corrispondente da [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) e posizionarlo nella cartella del progetto (o in `models/`).

## Setup

```bash
conda create -n speech-to-text python=3.12 -y
conda activate speech-to-text
pip install -r requirements.txt
```

### Requisiti di sistema

- **Vosk (CPU)**: qualsiasi CPU, ~2GB RAM
- **Whisper (GPU)**: NVIDIA con CUDA, ~8GB RAM, CUDA Toolkit installato

### Modello Vosk

Scaricare ed estrarre il modello nella cartella del progetto:

```
lite-speech-to-text/
  lite_speech_to_text.py
  vosk-model-small-it-0.22/   <-- esempio per italiano
```

## Utilizzo

```bash
conda activate speech-to-text
python lite_speech_to_text.py
```

### Controlli

- **CTRL sinistro** (tenere premuto): registra
- **CTRL sinistro** (rilasciare): trascrive e incolla
- **ESC x2** (entro 2 secondi): esci

## Build eseguibile

```bash
pip install pyinstaller
pyinstaller LiteSpeechToText.spec --noconfirm
```

L'eseguibile viene creato in `dist/LiteSpeechToText/`. Il modello Vosk va posizionato nella stessa cartella dell'exe.

Distribuzione:

```powershell
Compress-Archive -Path "dist\LiteSpeechToText\*" -DestinationPath "LiteSpeechToText-v1.0.0-windows.zip"
```
