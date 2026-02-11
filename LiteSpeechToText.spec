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

a = Analysis(
    ["lite_speech_to_text.py", "stt_core.py"],
    pathex=[],
    binaries=binaries,
    datas=[],
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
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "PIL",
        "scipy",
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
