"""Getrennte Laufzeit für Klonstimmen.

Chatterbox (MIT) kann Stimmen aus einer kurzen Referenzaufnahme nachbilden
und beherrscht Deutsch. Es lässt sich aber nicht in dieselbe Umgebung
installieren wie Bild und Video: es verlangt torch 2.6 ohne CUDA-Build,
diffusers 0.29 und transformers 5.x und würde die GPU-Beschleunigung und die
Videopipelines mit herunterziehen.

Deshalb läuft es in einem eigenen Interpreter und wird über die
Kommandozeile aufgerufen – dasselbe Muster wie bei ffmpeg. Dieses Modul
findet die Laufzeit, prüft sie und ruft den Arbeiter auf.

Suchreihenfolge:
  1. Umgebungsvariable STREAMFORGE_VOICE_PYTHON
  2. mitgeliefert:  <exe>/tools/voice-runtime/Scripts/python.exe
  3. Entwicklung:   <projekt>/.voice-venv/Scripts/python.exe
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from . import paths
from .accel import clean_error

log = logging.getLogger(__name__)

WORKER_NAME = "voice_worker.py"
CHECK_TIMEOUT = 90.0
# Kein Limit auf die Gesamtdauer: der erste Lauf lädt mehrere GB Modell.
# Abgebrochen wird nur bei echtem Stillstand – jede Ausgabe des Arbeiters
# gilt als Lebenszeichen.
IDLE_TIMEOUT = 600.0
PREPARE_IDLE_TIMEOUT = 1800.0


class VoiceRuntimeMissing(RuntimeError):
    """Klon-Laufzeit ist nicht eingerichtet."""


class VoiceRuntimeError(RuntimeError):
    """Der Arbeiter ist mit einem Fehler zurückgekommen."""


@dataclass(frozen=True)
class RuntimeInfo:
    python: Path
    worker: Path
    torch_version: str = ""
    cuda: bool = False
    multilingual: bool = False

    def label(self) -> str:
        gpu = "CUDA" if self.cuda else "CPU"
        sprachen = "mehrsprachig" if self.multilingual else "nur Englisch"
        return f"{gpu}, {sprachen}, torch {self.torch_version}"


def _creation_flags() -> int:
    return 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


def _worker_env() -> dict[str, str]:
    """Umgebung für den Arbeiter.

    Wichtig: Der Arbeiter lädt sein Modell sonst in den Standard-Cache unter
    %USERPROFILE%\\.cache\\huggingface – mehrere GB außerhalb des
    Anwendungsordners, die beim Deinstallieren liegen bleiben und im
    Portable-Betrieb auf dem falschen Laufwerk landen.
    """
    env = dict(os.environ)
    cache = paths.ensure_dir(paths.hf_cache_dir())
    env["HF_HOME"] = str(cache)
    env["HF_HUB_CACHE"] = str(cache / "hub")
    env["TRANSFORMERS_CACHE"] = str(cache / "hub")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return env


def python_path() -> Path | None:
    """Interpreter der Klon-Laufzeit suchen."""
    override = os.environ.get("STREAMFORGE_VOICE_PYTHON")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return candidate

    name = "python.exe" if os.name == "nt" else "python"
    sub = "Scripts" if os.name == "nt" else "bin"
    for base in (
        paths.exe_dir / "tools" / "voice-runtime",
        paths.exe_dir / "_internal" / "tools" / "voice-runtime",
        paths.exe_dir / ".voice-venv",
        paths.bundle_dir / "tools" / "voice-runtime",
    ):
        candidate = base / sub / name
        if candidate.is_file():
            return candidate
    return None


def worker_path() -> Path | None:
    """Arbeiter-Skript suchen (liegt bei der Laufzeit oder im Bundle)."""
    for candidate in (
        paths.exe_dir / "tools" / "voice-runtime" / WORKER_NAME,
        paths.exe_dir / "_internal" / "packaging" / WORKER_NAME,
        paths.bundle_dir / "packaging" / WORKER_NAME,
        paths.exe_dir / "packaging" / WORKER_NAME,
    ):
        if candidate.is_file():
            return candidate
    return None


_info_cache: RuntimeInfo | None = None


def available(refresh: bool = False) -> tuple[bool, str]:
    """Ist die Klon-Laufzeit einsatzbereit? Sonst Klartext-Begründung."""
    try:
        info = probe(refresh=refresh)
    except VoiceRuntimeMissing as exc:
        return False, str(exc)
    except VoiceRuntimeError as exc:
        return False, str(exc)
    return True, info.label()


def probe(refresh: bool = False) -> RuntimeInfo:
    """Laufzeit prüfen. Wirft VoiceRuntimeMissing/VoiceRuntimeError."""
    global _info_cache
    if _info_cache is not None and not refresh:
        return _info_cache

    python = python_path()
    worker = worker_path()
    if python is None or worker is None:
        raise VoiceRuntimeMissing(
            "Die Laufzeit für Klonstimmen ist nicht eingerichtet. "
            "Einrichten mit 'streamforge voice-runtime install' oder in der "
            "Oberfläche unter 'Stimme anlernen'. Ohne sie wird die "
            "Standardstimme verwendet."
        )

    try:
        proc = subprocess.run(
            [str(python), str(worker), "check"],
            check=False, capture_output=True, text=True,
            timeout=CHECK_TIMEOUT, creationflags=_creation_flags(),
            env=_worker_env(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VoiceRuntimeError(f"Klon-Laufzeit nicht startbar: {clean_error(exc)}") from exc

    data = _parse_json(proc.stdout)
    if not data.get("ok"):
        raise VoiceRuntimeError(
            "Klon-Laufzeit meldet: " + str(data.get("error") or clean_error(proc.stderr))
        )

    _info_cache = RuntimeInfo(
        python=python,
        worker=worker,
        torch_version=str(data.get("torch", "")),
        cuda=bool(data.get("cuda")),
        multilingual=bool(data.get("multilingual")),
    )
    return _info_cache


def _parse_json(text: str) -> dict[str, Any]:
    """Letzte JSON-Zeile aus der Ausgabe lesen (davor kann Geplapper stehen)."""
    for line in reversed((text or "").strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return {}


def synthesize(
    reference: Path,
    text: str,
    output: Path,
    language: str = "de",
    exaggeration: float = 0.5,
    cfg: float = 0.5,
    seed: int = 0,
    device: str = "auto",
    should_stop: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Sprache mit geklonter Stimme erzeugen. Abbrechbar."""
    info = probe()
    status = on_status or (lambda _t: None)
    paths.ensure_dir(output.parent)

    command = [
        str(info.python), str(info.worker), "synth",
        "--ref", str(reference),
        "--text", text,
        "--out", str(output),
        "--language", language,
        "--exaggeration", str(exaggeration),
        "--cfg", str(cfg),
        "--seed", str(int(seed)),
        "--device", device,
    ]
    log.debug("Klon-Laufzeit: %s", " ".join(command[:6]))
    status("Klonstimme wird erzeugt …")
    return _run_worker(command, status, should_stop, IDLE_TIMEOUT)


def _run_worker(
    command: list[str],
    status: Callable[[str], None],
    should_stop: Callable[[], bool] | None,
    idle_timeout: float,
) -> dict[str, Any]:
    """Arbeiter starten, Lebenszeichen mitlesen, sauber abbrechen können.

    Der erste Lauf lädt mehrere GB. Ein Limit auf die Gesamtdauer würde
    genau dann zuschlagen, wenn alles richtig läuft. Deshalb zählt nur, wie
    lange der Arbeiter **nichts** mehr von sich gibt.
    """
    import threading

    try:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, creationflags=_creation_flags(), env=_worker_env(),
        )
    except OSError as exc:
        raise VoiceRuntimeError(f"Klon-Laufzeit nicht startbar: {clean_error(exc)}") from exc

    last_signal = [time.time()]
    stderr_tail: list[str] = []
    stdout_parts: list[str] = []

    def pump_stderr() -> None:
        assert process.stderr is not None
        for raw in process.stderr:
            line = raw.strip()
            last_signal[0] = time.time()
            if not line:
                continue
            stderr_tail.append(line)
            if len(stderr_tail) > 40:
                stderr_tail.pop(0)
            # Ladebalken von huggingface_hub nicht durchreichen
            if "%|" not in line and len(line) < 160:
                status(line)

    def pump_stdout() -> None:
        assert process.stdout is not None
        for raw in process.stdout:
            last_signal[0] = time.time()
            stdout_parts.append(raw)

    threads = [threading.Thread(target=pump_stderr, daemon=True),
               threading.Thread(target=pump_stdout, daemon=True)]
    for thread in threads:
        thread.start()

    while process.poll() is None:
        if should_stop is not None and should_stop():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            from .jobs import JobCancelled

            raise JobCancelled("Klonstimme abgebrochen")
        if time.time() - last_signal[0] > idle_timeout:
            process.kill()
            raise VoiceRuntimeError(
                f"Klon-Laufzeit meldet sich seit {idle_timeout:g}s nicht mehr – abgebrochen."
            )
        time.sleep(0.2)

    for thread in threads:
        thread.join(timeout=5)

    data = _parse_json("".join(stdout_parts))
    if process.returncode != 0 or not data.get("ok"):
        raise VoiceRuntimeError(
            "Klonstimme fehlgeschlagen: "
            + str(data.get("error") or clean_error(" ".join(stderr_tail[-3:]) or "unbekannt"))
        )
    return data


def prepare(
    language: str = "de",
    device: str = "auto",
    should_stop: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Modell einmalig laden (mehrere GB). Danach ist die Synthese kurz."""
    info = probe()
    status = on_status or (lambda _t: None)
    status("Lade Klonstimmen-Modell – beim ersten Mal mehrere GB …")
    command = [str(info.python), str(info.worker), "prepare",
               "--language", language, "--device", device]
    return _run_worker(command, status, should_stop, PREPARE_IDLE_TIMEOUT)


# ---------------------------------------------------------------------------
# Einrichtung
# ---------------------------------------------------------------------------
def install(
    target: Path | None = None,
    cuda_index: str = "https://download.pytorch.org/whl/cu126",
    on_status: Callable[[str], None] | None = None,
    base_python: str | None = None,
) -> Path:
    """Klon-Laufzeit einrichten: eigenes Venv anlegen und chatterbox laden.

    Braucht einen Python-Interpreter auf dem Rechner. Im ausgelieferten
    Bundle wird die Laufzeit stattdessen mitgeliefert – dann ist dieser
    Schritt nicht nötig.
    """
    import sys

    status = on_status or (lambda _t: None)
    if paths.is_frozen() and base_python is None:
        raise VoiceRuntimeMissing(
            "In der ausgelieferten Fassung wird die Klon-Laufzeit mitgeliefert. "
            "Fehlt sie, bitte beim Anbieter melden – ein Nachinstallieren "
            "braucht Python auf diesem Rechner."
        )

    root = Path(target) if target else (paths.data_dir() / "voice-runtime")
    interpreter = base_python or sys.executable
    sub = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    venv_python = root / sub / name

    if not venv_python.is_file():
        status(f"Lege Umgebung an: {root}")
        subprocess.run([interpreter, "-m", "venv", str(root)], check=True,
                       creationflags=_creation_flags())

    status("Installiere chatterbox-tts (mehrere GB, dauert einige Minuten) …")
    subprocess.run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
                   check=True, creationflags=_creation_flags())
    # 'perth' (das Wasserzeichen von Chatterbox) braucht pkg_resources.
    # setuptools ab 81 liefert das nicht mehr mit, und Python-3.13-Venvs
    # bringen setuptools gar nicht erst mit – ohne diese Zeile scheitert das
    # Laden des Modells mit "'NoneType' object is not callable".
    subprocess.run([str(venv_python), "-m", "pip", "install", "setuptools<81"],
                   check=True, creationflags=_creation_flags())
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "chatterbox-tts",
         "--extra-index-url", cuda_index],
        check=True, creationflags=_creation_flags(),
    )

    worker = worker_path()
    if worker is not None:
        import shutil

        shutil.copy2(worker, root / WORKER_NAME)
    os.environ["STREAMFORGE_VOICE_PYTHON"] = str(venv_python)
    status("Klon-Laufzeit eingerichtet.")
    globals()["_info_cache"] = None
    return root


def describe() -> str:
    """Zustand für `info` und die Oberfläche."""
    ok, note = available()
    lines = [f"Klon-Laufzeit: {'bereit' if ok else 'nicht eingerichtet'}"]
    lines.append(f"  {note}")
    python = python_path()
    if python:
        lines.append(f"  Interpreter: {python}")
    worker = worker_path()
    if worker:
        lines.append(f"  Arbeiter:    {worker}")
    return "\n".join(lines)
