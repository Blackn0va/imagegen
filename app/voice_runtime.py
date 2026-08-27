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
  2. selbst eingerichtet: <daten>/voice-runtime/Scripts/python.exe
  3. mitgeliefert:        <exe>/tools/voice-runtime/Scripts/python.exe
  4. Entwicklung:         <projekt>/.voice-venv/Scripts/python.exe

Punkt 2 fehlte lange. Weil ``install()`` genau dorthin einrichtet, wurde
eine selbst eingerichtete Laufzeit nach dem naechsten Programmstart nicht
mehr gefunden -- in der Sitzung der Einrichtung ging es noch, da dort
STREAMFORGE_VOICE_PYTHON gesetzt wird.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import paths
from .accel import clean_error

log = logging.getLogger(__name__)

WORKER_NAME = "voice_worker.py"
CHECK_TIMEOUT = 120.0
FAST_CHECK_TIMEOUT = 20.0
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
    # None = noch nicht ermittelt (Schnellprüfung importiert torch nicht).
    cuda: bool | None = None
    multilingual: bool = False
    fast: bool = True

    def label(self) -> str:
        if self.cuda is None:
            gpu = "Gerät wird beim ersten Lauf ermittelt"
        else:
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
    orte = (
        # ZUERST der Ort, an den install() einrichtet. Fehlte er hier,
        # wurde eine selbst eingerichtete Laufzeit nach dem naechsten
        # Programmstart nie wieder gefunden.
        paths.data_dir() / "voice-runtime",
        paths.exe_dir / "tools" / "voice-runtime",
        paths.exe_dir / "_internal" / "tools" / "voice-runtime",
        paths.exe_dir / ".voice-venv",
        paths.bundle_dir / "tools" / "voice-runtime",
    )

    # Erster Durchgang: nur VOLLSTAENDIGE Umgebungen.
    #
    # Ein python.exe allein sagt nichts. Lagen zwei Umgebungen
    # nebeneinander - eine halb eingerichtete und die mitgelieferte -,
    # gewann die kaputte, und die Anwendung meldete "chatterbox fehlt",
    # obwohl die gute daneben lag.
    for base in orte:
        kandidat = base / sub / name
        if kandidat.is_file() and (base / "Lib" / "site-packages" / "chatterbox").is_dir():
            return kandidat

    # Zweiter Durchgang: irgendetwas ist besser als nichts -- die
    # Bereitschaftspruefung sagt dann im Klartext, was fehlt.
    for base in orte:
        kandidat = base / sub / name
        if kandidat.is_file():
            return kandidat
    return None


def _worker_source() -> Path | None:
    """Der mitgelieferte ``voice_worker.py``, um ihn zu kopieren.

    Getrennt von :func:`worker_path`: dort wird gesucht, wo er liegen
    SOLL, hier, wo er HERKOMMT.
    """
    for candidate in (
        paths.bundle_dir / "packaging" / WORKER_NAME,
        paths.exe_dir / "_internal" / "packaging" / WORKER_NAME,
        paths.exe_dir / "packaging" / WORKER_NAME,
        Path(__file__).resolve().parent.parent / "packaging" / WORKER_NAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def worker_path() -> Path | None:
    """Arbeiter-Skript suchen (liegt bei der Laufzeit oder im Bundle)."""
    for candidate in (
        paths.data_dir() / "voice-runtime" / WORKER_NAME,
        paths.exe_dir / "tools" / "voice-runtime" / WORKER_NAME,
        paths.exe_dir / "_internal" / "packaging" / WORKER_NAME,
        paths.bundle_dir / "packaging" / WORKER_NAME,
        paths.exe_dir / "packaging" / WORKER_NAME,
    ):
        if candidate.is_file():
            return candidate
    return None


_info_cache: RuntimeInfo | None = None
_state_cache: tuple[bool, str] | None = None


def _state_file() -> Path:
    return paths.data_dir() / "voice-runtime-state.json"


def _load_state() -> tuple[bool, str] | None:
    """Letztes Prüfergebnis von der Platte.

    Damit ist die Seite 'Stimme anlernen' auch beim allerersten Klick nach
    einem Neustart sofort da, statt auf einen Unterprozess zu warten.
    Gültig nur, solange der Interpreter unverändert ist.
    """
    try:
        data = json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    python = python_path()
    if not python or str(python) != data.get("python"):
        return None
    try:
        if abs(python.stat().st_mtime - float(data.get("python_mtime", 0))) > 1.0:
            return None
    except OSError:
        return None
    ok = bool(data.get("ok"))
    if not ok:
        # NUR gute Nachrichten merken.
        #
        # Ein "ok" spart die teure Prüfung. Ein "nicht ok" darf sich nicht
        # selbst festschreiben: es kann von einem Zeitlimit stammen, von
        # einem inzwischen behobenen Zustand oder von einer Fassung, die
        # den Ort noch nicht kannte. Genau das ist passiert -- nach der
        # Reparatur meldete die Anwendung weiter "nicht eingerichtet",
        # weil sie ihre eigene alte Antwort las.
        return None
    return True, str(data.get("note", ""))


def _save_state(ok: bool, note: str) -> None:
    python = python_path()
    if python is None:
        return
    try:
        paths.ensure_dir(_state_file().parent)
        _state_file().write_text(
            json.dumps(
                {
                    "ok": ok,
                    "note": note,
                    "python": str(python),
                    "python_mtime": python.stat().st_mtime,
                    "checked_at": time.time(),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass  # Zwischenspeicher ist Komfort, kein Muss


def cached_state() -> tuple[bool, str] | None:
    """Bekannter Zustand ohne jeden Unterprozess. None = noch unbekannt."""
    global _state_cache
    if _state_cache is None:
        _state_cache = _load_state()
    return _state_cache


def available(refresh: bool = False, full: bool = False) -> tuple[bool, str]:
    """Ist die Klon-Laufzeit einsatzbereit? Sonst Klartext-Begründung.

    Vorgabe ist die Schnellprüfung (Bruchteile einer Sekunde). Die volle
    Prüfung importiert torch und chatterbox und dauert auf einem kalten
    Dateisystem über eine Minute – die gehört niemals in den
    Oberflächen-Thread.
    """
    global _state_cache
    if not refresh:
        known = cached_state()
        # Der gemerkte Zustand gilt nur, solange Interpreter UND Arbeiter
        # wirklich liegen. Beides sind billige Dateiabfragen -- ohne sie
        # antwortet der Sprechpfad aus einem Zwischenspeicher, der die
        # Wirklichkeit längst überholt hat.
        if known is not None and python_path() is not None and worker_path() is not None:
            return known
        _state_cache = None
    try:
        info = probe(refresh=refresh, full=full)
    except (VoiceRuntimeMissing, VoiceRuntimeError) as exc:
        _state_cache = (False, str(exc))
        _save_state(False, str(exc))
        return _state_cache
    _state_cache = (True, info.label())
    _save_state(True, info.label())
    return _state_cache


def probe(refresh: bool = False, full: bool = False) -> RuntimeInfo:
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

    command = [str(python), str(worker), "check"]
    if full:
        command.append("--full")
    try:
        proc = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT if full else FAST_CHECK_TIMEOUT,
            creationflags=_creation_flags(),
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
        # In der Schnellprüfung ist das Gerät unbekannt; erst beim Laden
        # steht fest, ob CUDA wirklich greift.
        cuda=bool(data.get("cuda")) if "cuda" in data else None,
        multilingual=bool(data.get("multilingual")),
        fast=not full,
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
    reference: Path | None,
    text: str | Sequence[str],
    output: Path,
    language: str = "de",
    exaggeration: float = 0.5,
    cfg: float = 0.5,
    temperature: float = 0.8,
    seed: int = 0,
    device: str = "auto",
    should_stop: Callable[[], bool] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Sprache über Chatterbox erzeugen. Abbrechbar.

    Ohne ``reference`` spricht das Modell mit seiner **eingebauten**
    Stimme. Das ist kein Klon einer Person und braucht deshalb auch keine
    Einwilligung – anders als eine Referenzaufnahme, die eine reale
    Stimme nachbildet.
    """
    info = probe()
    status = on_status or (lambda _t: None)
    paths.ensure_dir(output.parent)

    # Mehrere Sätze gehen als Datei mit, damit das Modell nur einmal geladen
    # wird. Ein Aufruf je Satz kostete jedes Mal den vollen Ladevorgang.
    text_file: Path | None = None
    if isinstance(text, str):
        single = text
    else:
        saetze = [s for s in text if s.strip()]
        single = saetze[0] if len(saetze) == 1 else ""
        if len(saetze) > 1:
            text_file = (
                paths.ensure_dir(paths.temp_dir())
                / f"texte-{os.getpid()}-{int(time.time() * 1000)}.json"
            )
            text_file.write_text(json.dumps(saetze, ensure_ascii=False), encoding="utf-8")

    command = [
        str(info.python),
        str(info.worker),
        "synth",
        "--text",
        single,
        "--out",
        str(output),
        "--language",
        language,
        "--exaggeration",
        str(exaggeration),
        "--cfg",
        str(cfg),
        "--temperature",
        str(temperature),
        "--seed",
        str(int(seed)),
        "--device",
        device,
    ]
    if reference is not None:
        command += ["--ref", str(reference)]
    if text_file is not None:
        command += ["--text-file", str(text_file)]

    log.debug("Klon-Laufzeit: %s", " ".join(command[:6]))
    status("Klonstimme wird erzeugt …")
    try:
        return _run_worker(command, status, should_stop, IDLE_TIMEOUT)
    finally:
        if text_file is not None:
            text_file.unlink(missing_ok=True)


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
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            creationflags=_creation_flags(),
            env=_worker_env(),
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

    threads = [
        threading.Thread(target=pump_stderr, daemon=True),
        threading.Thread(target=pump_stdout, daemon=True),
    ]
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
    command = [
        str(info.python),
        str(info.worker),
        "prepare",
        "--language",
        language,
        "--device",
        device,
    ]
    return _run_worker(command, status, should_stop, PREPARE_IDLE_TIMEOUT)


# ---------------------------------------------------------------------------
# Einrichtung
# ---------------------------------------------------------------------------
# Kleinste Fassung, mit der chatterbox-tts laeuft.
MIN_PYTHON = (3, 11)


def _taugt(kandidat: Sequence[str]) -> str:
    """Fassungsnummer, wenn dieser Aufruf ein brauchbares Python startet."""
    try:
        fertig = subprocess.run(
            [*kandidat, "-c", "import sys;print('%d.%d' % sys.version_info[:2])"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=_creation_flags(),
        )
    except Exception:
        return ""
    if fertig.returncode != 0:
        return ""
    text = (fertig.stdout or "").strip()
    try:
        teile = tuple(int(t) for t in text.split("."))
    except ValueError:
        return ""
    return text if teile >= MIN_PYTHON else ""


def find_system_python() -> tuple[list[str], str]:
    """Ein Python auf diesem Rechner, das ``venv`` anlegen kann.

    Rückgabe: (Aufruf als Liste, Fassungsnummer). Leere Liste heißt: keins
    gefunden. Der Aufruf ist eine Liste, weil der Windows-Py-Launcher als
    ``py -3.12`` kommt und nicht als einzelner Pfad.

    Ohne diese Suche endete die Einrichtung im gebauten Programm in einer
    Sackgasse -- obwohl auf den meisten Rechnern längst ein Python liegt.
    """
    import shutil

    kandidaten: list[list[str]] = []

    # Der Py-Launcher kennt alle eingetragenen Fassungen; neueste zuerst.
    if os.name == "nt" and shutil.which("py"):
        kandidaten += [["py", f"-3.{minor}"] for minor in (13, 12, 11)]

    for name in ("python3", "python"):
        pfad = shutil.which(name)
        if pfad:
            kandidaten.append([pfad])

    # Uebliche Installationsstellen, falls nichts im Suchpfad steht.
    if os.name == "nt":
        basis = Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Python"
        if basis.is_dir():
            for ordner in sorted(basis.glob("Python3*"), reverse=True):
                exe = ordner / "python.exe"
                if exe.is_file():
                    kandidaten.append([str(exe)])

    gesehen: set[tuple[str, ...]] = set()
    for kandidat in kandidaten:
        schluessel = tuple(kandidat)
        if schluessel in gesehen:
            continue
        gesehen.add(schluessel)
        fassung = _taugt(kandidat)
        if fassung:
            log.info("Python für die Klon-Laufzeit: %s (%s)", " ".join(kandidat), fassung)
            return kandidat, fassung
    return [], ""


def install_possible() -> tuple[bool, str]:
    """Lässt sich die Klon-Laufzeit hier einrichten? Mit Begründung.

    Getrennt von ``install()``, damit die Oberfläche das *vorher* wissen
    kann und einen Knopf anbietet statt einer Enttäuschung hinterher.
    """
    aufruf, fassung = find_system_python()
    if aufruf:
        return True, f"Python {fassung} gefunden ({' '.join(aufruf)})."
    return False, (
        "Auf diesem Rechner ist kein Python ab 3.11 zu finden. Die "
        "Klon-Laufzeit braucht eines, um ihre eigene Umgebung anzulegen. "
        "Nach der Installation von python.org (Haken bei 'Add to PATH') "
        "geht es ohne weiteres Zutun."
    )


class VoiceServer:
    """Ein laufender Arbeiter, der sein Modell geladen hält.

    Ohne ihn kostet jeder Satz das vollständige Laden des Modells --
    gemessen rund 35 s. Mit ihm kostet es das einmal, danach nur noch die
    Rechenzeit für den Satz selbst.

    Verständigt wird sich zeilenweise über JSON. Der Prozess ist bewusst
    schlicht gehalten: stirbt er, wird beim nächsten Satz ein neuer
    gestartet, und wenn das auch nicht geht, fällt der Aufrufer auf den
    Einzelaufruf zurück. Ein Gespräch darf nicht daran scheitern, dass
    ein Hilfsprozess weg ist.
    """

    # So lange darf das erste Laden dauern (Modell kommt von der Platte).
    START_TIMEOUT = 300.0
    # So lange darf ein einzelner Satz dauern.
    SENTENCE_TIMEOUT = 300.0

    def __init__(self, language: str = "de", device: str = "auto") -> None:
        self.language = language
        self.device = device
        self._proc: Any = None
        self.info: dict[str, Any] = {}

    # -- Leben ---------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def crashed(self) -> bool:
        """Lief einmal und ist jetzt weg.

        Zu unterscheiden von "noch nie gestartet": ein frisch angelegter
        Arbeiter läuft ebenfalls nicht, ist aber völlig in Ordnung. Wer
        beides gleich behandelt, legt bei jedem Satz einen neuen an – und
        holt sich damit das Modellladen zurück, das der Dauerbetrieb
        gerade abschaffen soll.
        """
        return self._proc is not None and self._proc.poll() is not None

    def start(self, on_status: Callable[[str], None] | None = None) -> None:
        """Prozess starten und warten, bis das Modell steht."""
        if self.running:
            return
        status = on_status or (lambda _t: None)
        info = probe()
        status("Stimmmodell wird geladen (einmalig) …")
        self._proc = subprocess.Popen(
            [
                str(info.python),
                str(info.worker),
                "serve",
                "--language",
                self.language,
                "--device",
                self.device,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
            env=_worker_env(),
            creationflags=_creation_flags(),
        )
        antwort = self._read(self.START_TIMEOUT)
        if not antwort.get("ok"):
            self.stop()
            raise VoiceRuntimeError(
                antwort.get("error") or "Das Stimmmodell hat sich nicht gemeldet."
            )
        self.info = antwort
        status(f"Stimme bereit ({antwort.get('device', '?')}).")

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('{"command":"quit"}\n')
                proc.stdin.flush()
                proc.wait(timeout=10)
        except Exception as exc:
            log.debug("Arbeiter reagiert nicht: %s", clean_error(exc))
        finally:
            if proc.poll() is None:
                with contextlib.suppress(Exception):
                    proc.kill()

    # -- Sprechen ------------------------------------------------------
    def speak(
        self,
        texts: Sequence[str],
        output: Path,
        reference: Path | None = None,
        language: str = "",
        exaggeration: float = 0.5,
        cfg: float = 0.5,
        temperature: float = 0.8,
        seed: int = 0,
        on_status: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Sätze sprechen lassen. Startet den Prozess, falls nötig."""
        if not self.running:
            self.start(on_status)
        paths.ensure_dir(output.parent)

        auftrag: dict[str, Any] = {
            "texts": list(texts),
            "out": str(output),
            "language": language or self.language,
            "exaggeration": float(exaggeration),
            "cfg": float(cfg),
            "temperature": float(temperature),
        }
        if reference is not None:
            auftrag["ref"] = str(reference)
        if seed:
            auftrag["seed"] = int(seed)

        try:
            self._proc.stdin.write(json.dumps(auftrag, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except Exception as exc:
            self.stop()
            raise VoiceRuntimeError(f"Stimmmodell nicht erreichbar: {clean_error(exc)}") from exc

        antwort = self._read(self.SENTENCE_TIMEOUT)
        if not antwort.get("ok"):
            raise VoiceRuntimeError(antwort.get("error") or "Sprachausgabe fehlgeschlagen.")
        return antwort

    # -- Hilfen --------------------------------------------------------
    def _read(self, timeout: float) -> dict[str, Any]:
        """Nächste JSON-Zeile lesen. Alles andere ist Geschwätz.

        Fremdbibliotheken schreiben Warnungen und Fortschrittsbalken; die
        dürfen die Verständigung nicht stören.
        """
        ende = time.monotonic() + timeout
        while time.monotonic() < ende:
            if self._proc is None or self._proc.stdout is None:
                return {"ok": False, "error": "Kein Arbeiter."}
            zeile = self._proc.stdout.readline()
            if not zeile:
                code = self._proc.poll()
                return {"ok": False, "error": f"Arbeiter beendet (Code {code})."}
            zeile = zeile.strip()
            if not zeile.startswith("{"):
                continue
            try:
                return json.loads(zeile)
            except ValueError:
                continue
        return {"ok": False, "error": f"Keine Antwort binnen {timeout:.0f}s."}


def install(
    target: Path | None = None,
    cuda_index: str = "https://download.pytorch.org/whl/cu126",
    on_status: Callable[[str], None] | None = None,
    base_python: str | None = None,
) -> Path:
    """Klon-Laufzeit einrichten: eigenes Venv anlegen und chatterbox laden.

    Braucht ein Python ab 3.11 auf dem Rechner. Im gebauten Programm wird
    danach gesucht (:func:`find_system_python`), weil dessen eigener
    Interpreter keine Umgebungen anlegen kann. Ob es klappen wird, sagt
    :func:`install_possible` **vor** dem Versuch.

    Liegt die Laufzeit schon mitgeliefert neben der .exe, ist dieser
    Schritt nicht nötig.
    """
    import sys

    status = on_status or (lambda _t: None)

    # Welcher Interpreter legt die Umgebung an?
    #
    # Im gebauten Programm ist ``sys.executable`` die .exe und kein
    # Python -- damit lässt sich kein venv anlegen. Statt das als
    # Sackgasse zu melden, wird auf dem Rechner nachgesehen: meistens
    # liegt dort längst ein Python.
    if base_python:
        aufruf = [base_python]
    elif paths.is_frozen():
        aufruf, fassung = find_system_python()
        if not aufruf:
            _moeglich, grund = install_possible()
            raise VoiceRuntimeMissing(grund)
        status(f"Python {fassung} gefunden – richte damit ein.")
    else:
        aufruf = [sys.executable]

    root = Path(target) if target else (paths.data_dir() / "voice-runtime")
    sub = "Scripts" if os.name == "nt" else "bin"
    name = "python.exe" if os.name == "nt" else "python"
    venv_python = root / sub / name

    if not venv_python.is_file():
        status(f"Lege Umgebung an: {root}")
        subprocess.run(
            [*aufruf, "-m", "venv", str(root)], check=True, creationflags=_creation_flags()
        )

    status("Installiere chatterbox-tts (mehrere GB, dauert einige Minuten) …")
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        check=True,
        creationflags=_creation_flags(),
    )
    # 'perth' (das Wasserzeichen von Chatterbox) braucht pkg_resources.
    # setuptools ab 81 liefert das nicht mehr mit, und Python-3.13-Venvs
    # bringen setuptools gar nicht erst mit – ohne diese Zeile scheitert das
    # Laden des Modells mit "'NoneType' object is not callable".
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "setuptools<81"],
        check=True,
        creationflags=_creation_flags(),
    )
    subprocess.run(
        [
            str(venv_python),
            "-m",
            "pip",
            "install",
            "chatterbox-tts",
            "--extra-index-url",
            cuda_index,
        ],
        check=True,
        creationflags=_creation_flags(),
    )

    # Den Arbeiter mitnehmen. Gesucht wird dort, wo er HERKOMMT -- nicht
    # mit worker_path(), das sucht, wo er liegen SOLL, und faende bei
    # einer frischen Einrichtung nichts.
    quelle = _worker_source()
    if quelle is not None:
        import shutil

        try:
            shutil.copy2(quelle, root / WORKER_NAME)
            status(f"Arbeiter kopiert: {WORKER_NAME}")
        except OSError as exc:
            log.warning("Arbeiter nicht kopierbar: %s", clean_error(exc))
    else:
        log.warning("Kein %s zum Kopieren gefunden.", WORKER_NAME)

    os.environ["STREAMFORGE_VOICE_PYTHON"] = str(venv_python)
    globals()["_info_cache"] = None
    globals()["_state_cache"] = None

    # Nachpruefen statt behaupten. Ohne diese Pruefung meldete die
    # Einrichtung "fertig" und die Anwendung danach "nicht eingerichtet" --
    # der Nutzer hatte mehrere Gigabyte geladen und stand vor derselben
    # Meldung wie vorher.
    if python_path() is None or worker_path() is None:
        fehlt = "der Interpreter" if python_path() is None else WORKER_NAME
        raise VoiceRuntimeMissing(
            f"Eingerichtet nach {root}, aber {fehlt} wird danach nicht "
            "gefunden. So ist die Laufzeit nicht nutzbar."
        )

    # Die gemerkte Antwort von vorher wegräumen: sie sagt "nicht
    # eingerichtet" und würde die frische Einrichtung überstimmen.
    with contextlib.suppress(OSError):
        _state_file().unlink(missing_ok=True)

    status("Klon-Laufzeit eingerichtet und geprüft.")
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
