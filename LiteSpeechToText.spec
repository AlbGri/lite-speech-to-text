# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec per Lite Speech-to-Text
# Build: pyinstaller LiteSpeechToText.spec --noconfirm
# Nota: i modelli vanno nella cartella models/ accanto all'exe

import os
import sys
from pathlib import Path

conda_env = os.environ.get("CONDA_PREFIX", "")

# DLL del conda env che PyInstaller potrebbe non trovare
binaries = []
if conda_env:
    dll_dir = Path(conda_env) / "Library" / "bin"
    for dll_name in ("ffi.dll", "liblzma.dll", "libbz2.dll", "libexpat.dll"):
        dll_path = dll_dir / dll_name
        if dll_path.exists():
            binaries.append((str(dll_path), "."))

# DLL native di vosk
import vosk as _vosk
_vosk_dir = os.path.dirname(_vosk.__file__)
for _dll in ("libvosk.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll"):
    _dll_path = os.path.join(_vosk_dir, _dll)
    if os.path.exists(_dll_path):
        binaries.append((_dll_path, "vosk"))

# onnxruntime: raccolta completa (DLL native + dati, richiesto da VAD filter)
from PyInstaller.utils.hooks import collect_all
ort_datas, ort_binaries, ort_hiddenimports = collect_all("onnxruntime")

a = Analysis(
    ["lite_speech_to_text.py", "stt_core.py"],
    pathex=[],
    binaries=binaries + ort_binaries,
    datas=ort_datas,
    hiddenimports=[
        "pyaudio",
        "numpy",
        "pynput",
        "pynput.keyboard",
        "pynput.keyboard._win32",
        "pyperclip",
        "vosk",
        "stt_core",
        "PyQt6",
        "PyQt6.QtCore",
        "PyQt6.QtGui",
        "PyQt6.QtWidgets",
        "faster_whisper",
        "ctranslate2",
    ] + ort_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "PIL",
        "pandas",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="LiteSpeechToText",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="LiteSpeechToText",
)
