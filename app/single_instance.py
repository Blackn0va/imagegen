"""Nur eine Instanz gleichzeitig.

Zwei Instanzen würden sich VRAM, Modell-Cache und Ausgabeordner streitig
machen – im schlimmsten Fall überschreiben sie sich gegenseitig Dateien.

Windows: benannter Kernel-Mutex. Den räumt der Kernel auch nach einem
Absturz auf, eine liegengebliebene Datei tut das nicht.
Sonst: ``fcntl.flock`` auf einer Datei im Temp-Verzeichnis.
Auf exotischen Plattformen lieber starten als fälschlich blockieren.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import sys
from typing import Any

from . import __app_name__, paths

log = logging.getLogger(__name__)

ERROR_ALREADY_EXISTS = 183

# Handles müssen prozessweit offen bleiben – als Modul-Global, nicht lokal.
_mutex_handle: Any = None
_lock_file: Any = None
_acquired = False


class InstanceGuard:
    """Ergebnis der Sperre.

    ``acquired`` False bedeutet: es läuft bereits eine Instanz.
    ``reason`` enthält immer eine Klartext-Begründung für die Anzeige.
    """

    def __init__(self, acquired: bool, reason: str, mechanism: str) -> None:
        self.acquired = acquired
        self.reason = reason
        self.mechanism = mechanism

    def __bool__(self) -> bool:
        return self.acquired

    def __repr__(self) -> str:  # pragma: no cover – Diagnose
        return f"InstanceGuard(acquired={self.acquired}, mechanism={self.mechanism!r})"


def _mutex_name(suffix: str = "") -> str:
    # "Local\\" – pro Anmeldesitzung, damit mehrere Benutzer parallel arbeiten
    # können (Terminalserver). "Global\\" wäre maschinenweit.
    name = f"Local\\{__app_name__}-single-instance"
    return f"{name}-{suffix}" if suffix else name


def _acquire_windows(suffix: str) -> InstanceGuard:
    import ctypes
    from ctypes import wintypes

    global _mutex_handle
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE

    handle = kernel32.CreateMutexW(None, False, _mutex_name(suffix))
    last_error = ctypes.get_last_error()
    if not handle:
        return InstanceGuard(
            True,
            f"Mutex konnte nicht erzeugt werden (Fehler {last_error}) – Start wird erlaubt.",
            "windows-mutex",
        )
    if last_error == ERROR_ALREADY_EXISTS:
        kernel32.CloseHandle(handle)
        return InstanceGuard(False, f"{__app_name__} läuft bereits.", "windows-mutex")
    _mutex_handle = handle  # offen halten bis Prozessende
    return InstanceGuard(True, "Einzelinstanz-Sperre gesetzt.", "windows-mutex")


def _acquire_posix(suffix: str) -> InstanceGuard:
    try:
        import fcntl
    except ImportError:
        return InstanceGuard(True, "Keine Sperrmechanik verfügbar – Start wird erlaubt.", "none")

    global _lock_file
    lock_path = paths.instance_lock_path()
    if suffix:
        lock_path = lock_path.with_name(f"{lock_path.stem}-{suffix}{lock_path.suffix}")
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(lock_path, "a+", encoding="utf-8")  # noqa: SIM115 – offen halten
    except OSError as exc:
        return InstanceGuard(
            True, f"Sperrdatei nicht nutzbar ({exc}) – Start wird erlaubt.", "flock"
        )

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return InstanceGuard(False, f"{__app_name__} läuft bereits.", "flock")

    try:
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
    except OSError:
        pass  # PID ist nur Diagnose, kein Teil der Sperre
    _lock_file = handle
    return InstanceGuard(True, "Einzelinstanz-Sperre gesetzt.", "flock")


def acquire(suffix: str = "") -> InstanceGuard:
    """Sperre holen. ``suffix`` trennt z. B. GUI von CLI-Läufen."""
    global _acquired
    if _acquired:
        return InstanceGuard(True, "Sperre in diesem Prozess bereits gesetzt.", "cached")
    try:
        guard = _acquire_windows(suffix) if os.name == "nt" else _acquire_posix(suffix)
    except Exception as exc:
        log.debug("Einzelinstanz-Prüfung fehlgeschlagen: %s", exc)
        return InstanceGuard(True, "Prüfung fehlgeschlagen – Start wird erlaubt.", "error")
    if guard.acquired:
        _acquired = True
        atexit.register(release)
    return guard


def release() -> None:
    """Sperre freigeben. Beim normalen Ende über atexit, sonst explizit."""
    global _mutex_handle, _lock_file, _acquired
    if _mutex_handle is not None:
        try:
            import ctypes

            ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(_mutex_handle)
        except Exception:
            pass
        _mutex_handle = None
    if _lock_file is not None:
        try:
            import fcntl

            fcntl.flock(_lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        with contextlib.suppress(Exception):
            _lock_file.close()
        _lock_file = None
    _acquired = False


def notify_already_running(guard: InstanceGuard, gui: bool = True) -> None:
    """Nutzer informieren, dass schon eine Instanz läuft – ohne Stacktrace."""
    message = (
        f"{guard.reason}\n\n"
        "Es kann nur eine Instanz laufen, weil sich zwei Instanzen "
        "Grafikspeicher und Ausgabeordner streitig machen würden.\n"
        "Wechsle zum bereits offenen Fenster."
    )
    print(message, file=sys.stderr)
    if not gui or os.name != "nt":
        return
    try:
        import ctypes

        # MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND
        ctypes.windll.user32.MessageBoxW(None, message, __app_name__, 0x40 | 0x10000)
    except Exception:
        pass
