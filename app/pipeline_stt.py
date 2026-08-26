"""Spracherkennung (Whisper) über faster-whisper.

Der letzte fehlende Baustein für das Telefonieren: aus dem aufgenommenen
Redebeitrag wird Text, den das Sprachmodell versteht.

Warum faster-whisper und nicht das Original von OpenAI: es läuft über
CTranslate2, ist auf der CPU mehrfach schneller und bringt CUDA von sich
aus mit – dieselbe Schnittstelle für beide Wege. Für ein Gespräch zählt
genau das: ein Satz muss in Bruchteilen einer Sekunde erkannt sein, sonst
entstehen Pausen, die kein Mensch aushält.

Die Laufzeit ist **optional**. Fehlt sie, bleibt das Telefonat
abgeschaltet und sagt warum.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import models
from .accel import Backend, clean_error
from .config import AppConfig

log = logging.getLogger(__name__)

INSTALL_HINT = "pip install faster-whisper"
BUILD_HINT = ".\\build-windows.ps1 -Clean -WithCall $true"

# Genauigkeit je Rechenweg. int8 auf der CPU ist der übliche Kompromiss:
# spürbar schneller, kaum schlechter. Auf der Karte float16.
COMPUTE_CPU = "int8"
COMPUTE_CUDA = "float16"


class SttUnavailable(RuntimeError):
    """Laufzeit oder Modell fehlt – Text enthält die Anleitung."""

    expected = True


@dataclass(frozen=True)
class Transcript:
    """Ergebnis einer Erkennung."""

    text: str
    language: str
    seconds: float
    elapsed_s: float
    model_key: str
    device: str

    @property
    def empty(self) -> bool:
        return not self.text.strip()

    def speed_factor(self) -> float:
        """Wie viel schneller als Echtzeit. Über 1 heißt: hält Schritt."""
        return self.seconds / self.elapsed_s if self.elapsed_s > 0 else 0.0


def _nachruesten() -> str:
    import sys

    if getattr(sys, "frozen", False):
        return (
            "Dies ist ein gebautes Programm mit eigenem Python – ein "
            f"'pip install' wirkt hier nicht. Neu bauen mit: {BUILD_HINT}"
        )
    return f"Nachrüsten: {INSTALL_HINT}"


def runtime_available() -> tuple[bool, str]:
    """Ist faster-whisper benutzbar?"""
    import importlib.util

    if importlib.util.find_spec("faster_whisper") is None:
        return False, f"faster-whisper fehlt. {_nachruesten()}"
    try:
        import faster_whisper
    except Exception as exc:
        return False, f"faster_whisper nicht ladbar ({clean_error(exc)}). {_nachruesten()}"
    fassung = getattr(faster_whisper, "__version__", "unbekannt")
    return True, f"faster-whisper {fassung} vorhanden."


def cuda_available() -> bool:
    """Kann CTranslate2 auf der Grafikkarte rechnen?"""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


def device_for(plan: Any = None) -> tuple[str, str]:
    """Gerät und Genauigkeit wählen. Rückgabe: (Gerät, Genauigkeit).

    Die Karte wird nur genommen, wenn CTranslate2 sie wirklich sieht –
    ein ``device="cuda"`` ohne Karte wirft mitten im Gespräch.
    """
    will_cuda = plan is None or getattr(plan, "backend", None) == Backend.CUDA
    if will_cuda and cuda_available():
        return "cuda", COMPUTE_CUDA
    return "cpu", COMPUTE_CPU


def available_models() -> list[models.ModelSpec]:
    return [spec for spec in models.REGISTRY.values() if spec.task is models.Task.STT]


def _weights_spec(spec: models.ModelSpec) -> models.ModelSpec:
    """Nur die CTranslate2-Dateien laden, nicht das ganze Repo."""
    return spec


def ensure_weights(config: AppConfig, spec: models.ModelSpec, context) -> Path:
    return models.ensure_local(
        _weights_spec(spec),
        allow_download=config.allow_model_download,
        on_status=context.status,
        should_stop=context.should_stop,
        allow_conditional=True,
        offline=config.offline_mode,
        on_progress=lambda done, total: context.progress(
            (done / total) if total else 0.0,
            f"{done / (1024**2):.0f} MB von {total / (1024**2):.0f} MB",
        ),
    )


class SpeechToText:
    """Ein geladenes Whisper-Modell.

    Bleibt geladen, solange das Gespräch läuft – ein Neuladen je Satz
    würde jede Antwort um Sekunden verzögern.
    """

    def __init__(self, config: AppConfig, spec: models.ModelSpec | None = None) -> None:
        self.config = config
        self.spec = spec or models.resolve(
            getattr(config, "stt_model", "") or models.DEFAULTS[models.Task.STT]
        )
        self._modell: Any = None
        self.device = ""
        self.compute = ""

    def load(self, context, plan: Any = None) -> None:
        ok, grund = runtime_available()
        if not ok:
            raise SttUnavailable(grund)

        ordner = ensure_weights(self.config, self.spec, context)
        self.device, self.compute = device_for(plan)

        from faster_whisper import WhisperModel

        context.status(f"Lade Spracherkennung ({self.spec.title}, {self.device}) …")
        begonnen = time.time()
        try:
            self._modell = WhisperModel(
                str(ordner),
                device=self.device,
                compute_type=self.compute,
                cpu_threads=self.config.cpu_threads or 0,
            )
        except Exception as exc:
            # Häufigster Fall: CUDA gemeldet, cuDNN fehlt. Auf CPU
            # ausweichen ist besser als ein abgebrochenes Gespräch.
            if self.device == "cuda":
                log.warning("Whisper auf CUDA fehlgeschlagen: %s", clean_error(exc))
                context.status("Grafikkarte für die Spracherkennung nicht nutzbar – CPU.")
                self.device, self.compute = "cpu", COMPUTE_CPU
                try:
                    self._modell = WhisperModel(str(ordner), device="cpu", compute_type=COMPUTE_CPU)
                except Exception as zweiter:
                    raise SttUnavailable(
                        f"Spracherkennung nicht ladbar: {clean_error(zweiter)}"
                    ) from zweiter
            else:
                raise SttUnavailable(f"Spracherkennung nicht ladbar: {clean_error(exc)}") from exc
        context.status(f"Spracherkennung bereit ({time.time() - begonnen:.0f} s, {self.device}).")

    def transcribe(self, wav: Path, language: str = "") -> Transcript:
        """Aufnahme zu Text. Wirft nur, wenn nichts geladen ist."""
        if self._modell is None:
            raise SttUnavailable("Es ist kein Spracherkennungsmodell geladen.")

        begonnen = time.time()
        sprache = (language or self.config.language or "").strip() or None
        segmente, info = self._modell.transcribe(
            str(wav),
            language=sprache,
            beam_size=1,  # Gespräch: Tempo vor letzter Genauigkeit
            vad_filter=True,  # Stille am Rand wegschneiden
            condition_on_previous_text=False,
        )
        text = " ".join(teil.text.strip() for teil in segmente).strip()
        return Transcript(
            text=text,
            language=str(getattr(info, "language", sprache or "")),
            seconds=float(getattr(info, "duration", 0.0)),
            elapsed_s=time.time() - begonnen,
            model_key=self.spec.key,
            device=self.device,
        )

    def unload(self) -> None:
        self._modell = None


def describe() -> str:
    """Zustandsbericht für Diagnose und Oberfläche."""
    ok, grund = runtime_available()
    zeilen = [grund]
    if ok:
        zeilen.append(f"  Grafikkarte nutzbar: {'ja' if cuda_available() else 'nein'}")
    for spec in available_models():
        zustand = "geladen" if models.is_downloaded(spec) else "nicht geladen"
        zeilen.append(
            f"  {spec.key:<16} {zustand:<13} {spec.approx_size_mb / 1024:.1f} GB  {spec.title}"
        )
    return "\n".join(zeilen)
