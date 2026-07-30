# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller-Beschreibung (onedir).

Wird von build-windows.ps1 aufgerufen. Alle Schalter kommen über
Umgebungsvariablen, damit es nur eine Spec-Datei gibt:

  SF_NAME        Name des Ausgabeordners und der .exe
  SF_ENTRY       Einstiegsskript (Vorgabe run_app.py)
  SF_ROOT        Projektwurzel
  SF_NOGUI       "1" = tkinter nicht mitnehmen (reine Kommandozeilen-Variante)
  SF_WITHCUDA    "1" = torch/CUDA-Pfade mitsammeln
  SF_CONSOLE     "1" = Konsolenfenster behalten (Diagnose-Build)
"""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

root = Path(os.environ.get("SF_ROOT", os.getcwd())).resolve()
name = os.environ.get("SF_NAME", "StreamForge")
entry = os.environ.get("SF_ENTRY", "run_app.py")
no_gui = os.environ.get("SF_NOGUI", "0") == "1"
with_cuda = os.environ.get("SF_WITHCUDA", "0") == "1"
console = os.environ.get("SF_CONSOLE", "0") == "1" or no_gui

block_cipher = None

# --- Daten, die neben die .exe gehören ------------------------------------
datas = []
for candidate in ("THIRD-PARTY-NOTICES.md", "MODELS.md", "README.md"):
    source = root / candidate
    if source.is_file():
        datas.append((str(source), "."))

# huggingface_hub bringt eigene Datendateien mit
try:
    datas += collect_data_files("huggingface_hub")
except Exception:  # noqa: BLE001 – fehlendes Paket ist kein Abbruchgrund
    pass

# Piper wird BEWUSST NICHT eingebettet: piper-tts steht unter GPL-3.0 und
# bettet espeak-ng ein – im selben Prozess zöge das die gesamte Anwendung
# unter die GPL. Wer Piper einsetzen will, liefert es als eigenständiges
# Programm daneben aus (getrennter Prozess) und legt Lizenz und
# Quelltextangebot bei. Siehe MODELS.md und app/licensing.py.

# --- Versteckte Importe ---------------------------------------------------
hiddenimports = [
    "app",
    "app.accel",
    "app.compose",
    "app.config",
    "app.jobs",
    "app.licensing",
    "app.models",
    "app.nettrust",
    "app.paths",
    "app.pipeline_image",
    "app.pipeline_video",
    "app.pipeline_voice",
    "app.single_instance",
    "app.voice_profiles",
    "truststore",
    "certifi",
    "huggingface_hub",
    "onnxruntime",
]

if not no_gui:
    hiddenimports += [
        "app.gui",
        "app.gui.main_window",
        "app.gui.theme",
        "app.gui.widgets",
        "tkinter",
        "tkinter.ttk",
        "tkinter.filedialog",
        "tkinter.messagebox",
    ]

if with_cuda:
    # diffusers/transformers laden viele Module dynamisch nach.
    for package in ("diffusers", "transformers", "safetensors", "accelerate"):
        try:
            hiddenimports += collect_submodules(package)
        except Exception:  # noqa: BLE001
            pass
    try:
        datas += collect_data_files("diffusers")
        datas += collect_data_files("transformers", include_py_files=False)
    except Exception:  # noqa: BLE001
        pass

# --- Ausschlüsse: alles, was nur den Ordner aufbläht ----------------------
excludes = [
    # Kolors ruft beim Import torch.jit.script auf; das braucht Python-
    # Quelltext, den ein PyInstaller-Bundle nicht mitliefert. Die Anwendung
    # nutzt Kolors nicht – draußen lassen spart Platz und verhindert den
    # Fehler "TorchScript requires source access".
    "diffusers.pipelines.kolors",
    "matplotlib",
    "scipy.spatial.cKDTree",
    "pytest",
    "IPython",
    "notebook",
    "PySide6",
    "PyQt5",
    "PyQt6",
    "wx",
    "tests",
]
if no_gui:
    excludes += ["tkinter", "app.gui"]

analysis = Analysis(
    [str(root / entry)],
    pathex=[str(root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

icon_path = root / "packaging" / "app.ico"

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,  # UPX bricht CUDA-DLLs und triggert Virenscanner
    console=console,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(icon_path) if icon_path.is_file() else None,
)

collect = COLLECT(
    exe,
    analysis.binaries,
    analysis.zipfiles,
    analysis.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=name,
)
