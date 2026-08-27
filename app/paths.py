"""Pfad-Auflösung für eingefrorene (PyInstaller) und Entwicklungs-Umgebung.

Regel: Modelle, Ausgaben, Logs und Konfiguration liegen NIE im
``_MEIPASS``-Verzeichnis. Das wird bei jedem Start neu ausgepackt und beim
Beenden gelöscht – mehrere GB Modelldaten wären nach jedem Start weg.

Zwei Ablage-Modi:
  * portable  – alles neben der .exe (Marker-Datei ``portable.txt`` und
                schreibbares Verzeichnis)
  * benutzer  – ``%LOCALAPPDATA%\\StreamForge`` (Windows) bzw.
                ``~/.local/share/StreamForge`` (Linux/macOS-Fallback)
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from . import __app_name__

# ---------------------------------------------------------------------------
# Basisverzeichnisse – exakt das vom Auftrag vorgegebene Muster
# ---------------------------------------------------------------------------
if getattr(sys, "frozen", False):
    exe_dir = Path(sys.executable).resolve().parent
    bundle_dir = Path(getattr(sys, "_MEIPASS", exe_dir))
else:
    exe_dir = bundle_dir = Path(__file__).resolve().parent.parent

PORTABLE_MARKER = "portable.txt"

_data_dir_override: Path | None = None
# Einmal entschieden, dann festhalten. Ohne diesen Zwischenspeicher liefe
# bei JEDEM Pfadzugriff ein Schreibtest neben der .exe. Schlägt der auch nur
# einmal fehl – gesperrte Datei, Virenscanner, kurzzeitig volle Platte –,
# würde die Anwendung mitten im Betrieb auf %LOCALAPPDATA% umschalten. Für
# den Nutzer sähe das so aus, als wären Stimmprofile und Modelle plötzlich
# verschwunden.
_portable_cache: bool | None = None
_data_dir_cache: Path | None = None


def is_frozen() -> bool:
    """True, wenn das Programm als PyInstaller-Bundle läuft."""
    return bool(getattr(sys, "frozen", False))


def reset_caches() -> None:
    """Entscheidung neu treffen (Tests, Wechsel des Datenverzeichnisses)."""
    global _portable_cache, _data_dir_cache
    _portable_cache = None
    _data_dir_cache = None


def set_data_dir_override(path: str | os.PathLike[str] | None) -> None:
    """Datenverzeichnis erzwingen (CLI-Schalter ``--data-dir``)."""
    global _data_dir_override
    _data_dir_override = Path(path).expanduser().resolve() if path else None
    reset_caches()


def _is_writable(directory: Path) -> bool:
    """Schreibtest ohne Ausnahme nach außen – reine Ja/Nein-Antwort."""
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / f".write-probe-{os.getpid()}"
        probe.write_bytes(b"ok")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def is_portable() -> bool:
    """Portable-Modus: Marker neben der .exe UND Verzeichnis beschreibbar.

    Wird genau einmal ermittelt. Ein späterer Wechsel wäre schlimmer als
    jede Fehlentscheidung: die Anwendung würde plötzlich in einem anderen
    Verzeichnis nach Modellen und Stimmprofilen suchen.
    """
    global _portable_cache
    if _data_dir_override is not None:
        return False
    if _portable_cache is None:
        marker = exe_dir / PORTABLE_MARKER
        # Auch OHNE Marker portabel, wenn daneben schon ein data-Ordner
        # mit Inhalt liegt.
        #
        # Der Marker wird beim Bauen geschrieben. Bricht der Bau vorher
        # ab, stuende ein lauffaehiges Programm ohne Marker da - und es
        # legte einen ZWEITEN Datenbestand unter %LOCALAPPDATA% an,
        # waehrend Modelle und Stimmprofile daneben unberuehrt liegen.
        # Genau das ist einmal passiert.
        neben_exe = exe_dir / "data"
        hat_daten = False
        try:
            hat_daten = neben_exe.is_dir() and any(neben_exe.iterdir())
        except OSError:
            hat_daten = False
        _portable_cache = bool((marker.is_file() or hat_daten) and _is_writable(exe_dir))
    return _portable_cache


def _user_data_root() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / __app_name__
        return Path.home() / "AppData" / "Local" / __app_name__
    xdg = os.environ.get("XDG_DATA_HOME")
    if xdg:
        return Path(xdg) / __app_name__
    return Path.home() / ".local" / "share" / __app_name__


def data_dir() -> Path:
    """Wurzel aller veränderlichen Daten (Modelle, Config, Logs, Ausgabe).

    Ergebnis wird festgehalten: dieser Pfad darf sich während eines Laufs
    nicht ändern, sonst wandern Stimmprofile und Modelle scheinbar weg.
    """
    global _data_dir_cache
    if _data_dir_override is not None:
        return _data_dir_override
    if _data_dir_cache is None:
        _data_dir_cache = (exe_dir / "data") if is_portable() else _user_data_root()
    return _data_dir_cache


def ensure_dir(path: Path) -> Path:
    """Verzeichnis anlegen (idempotent) und zurückgeben."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Modell-Cache. Mehrere GB – niemals in _MEIPASS."""
    override = os.environ.get("STREAMFORGE_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return data_dir() / "models"


def hf_cache_dir() -> Path:
    """Hugging-Face-Cache. Liegt unterhalb von models_dir, damit ein
    Deinstallieren mit einem Ordner erledigt ist."""
    return models_dir() / "hf"


def outputs_dir() -> Path:
    return data_dir() / "output"


def logs_dir() -> Path:
    return data_dir() / "logs"


def temp_dir() -> Path:
    """Arbeitsverzeichnis für Zwischendateien (Frames, WAV-Stücke)."""
    return data_dir() / "tmp"


def config_path() -> Path:
    return data_dir() / "config.json"


def consent_path() -> Path:
    """Markerdatei für die Zustimmung zu Drittanbieter-Lizenzen."""
    return data_dir() / "consent.json"


def notices_path() -> Path:
    """THIRD-PARTY-NOTICES.md – im Bundle neben der .exe, sonst im Repo."""
    for candidate in (
        exe_dir / "THIRD-PARTY-NOTICES.md",
        bundle_dir / "THIRD-PARTY-NOTICES.md",
        exe_dir / "_internal" / "THIRD-PARTY-NOTICES.md",
    ):
        if candidate.is_file():
            return candidate
    return exe_dir / "THIRD-PARTY-NOTICES.md"


def tools_dir() -> Path:
    """Mitgelieferte Fremd-Programme (ffmpeg). Bundle zuerst, dann Repo."""
    for candidate in (
        exe_dir / "tools",
        exe_dir / "_internal" / "tools",
        bundle_dir / "tools",
    ):
        if candidate.is_dir():
            return candidate
    return exe_dir / "tools"


def ffmpeg_exe() -> Path | None:
    """ffmpeg suchen: mitgeliefert -> PATH. None, wenn nicht vorhanden.

    Kein Fehler bei Abwesenheit; der Aufrufer meldet das im Klartext.
    """
    exe_name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    override = os.environ.get("STREAMFORGE_FFMPEG")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate
        if candidate.is_dir() and (candidate / exe_name).is_file():
            return candidate / exe_name
    for candidate in (
        tools_dir() / "ffmpeg" / "bin" / exe_name,
        tools_dir() / "ffmpeg" / exe_name,
        tools_dir() / exe_name,
        exe_dir / exe_name,
    ):
        if candidate.is_file():
            return candidate
    from shutil import which  # lokal, damit der Modul-Import billig bleibt

    found = which("ffmpeg")
    return Path(found) if found else None


def instance_lock_path() -> Path:
    """Sperrdatei für die Einzelinstanz (nur Nicht-Windows-Pfad)."""
    return Path(tempfile.gettempdir()) / f"{__app_name__.lower()}.lock"


def bootstrap() -> None:
    """Alle Nutzerverzeichnisse anlegen. Einmal beim Start aufrufen."""
    for directory in (data_dir(), models_dir(), outputs_dir(), logs_dir(), temp_dir()):
        ensure_dir(directory)


def describe() -> str:
    """Mehrzeilige Übersicht für Diagnose und GUI-Hardwareseite."""
    lines = [
        f"frozen:        {is_frozen()}",
        f"exe_dir:       {exe_dir}",
        f"bundle_dir:    {bundle_dir}",
        f"portable:      {is_portable()}",
        f"data_dir:      {data_dir()}",
        f"models_dir:    {models_dir()}",
        f"outputs_dir:   {outputs_dir()}",
        f"config:        {config_path()}",
        f"ffmpeg:        {ffmpeg_exe() or 'nicht gefunden'}",
    ]
    return "\n".join(lines)
