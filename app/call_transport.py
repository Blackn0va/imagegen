"""Woher die Stimme kommt und wohin sie geht.

Das Telefonat selbst -- zuhoeren, verstehen, antworten, sprechen -- ist
immer dasselbe. Verschieden ist nur der Weg: entweder das Mikrofon dieses
Rechners, oder ein Bot in einem Discord-Sprachkanal, in dem mehrere Leute
sitzen.

Diese Datei trennt beides. ``CallSession`` kennt nur noch einen
``Transport`` und muss nicht wissen, ob dahinter PortAudio oder Discord
steckt. Ohne diese Trennung waere jede Zeile im Gespraechsablauf mit einem
``if discord:`` durchsetzt -- und genau dort schleichen sich die Fehler
ein, die man erst im Gespraech merkt.

Ein Transport schuldet vier Dinge:

    open()    bereitstellen, was es braucht (Geraet oeffnen, Bot anmelden)
    listen()  einen Redebeitrag aufnehmen -> WAV in 16 kHz Mono
    play()    eine WAV-Datei hoerbar machen
    close()   aufraeumen

Alles andere -- Spracherkennung, Sprachmodell, Sprachausgabe -- bleibt
gemeinsam.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TransportInfo:
    """Was dieser Weg kann und was ihm fehlt."""

    key: str  # "lokal" | "discord"
    title: str
    ready: bool
    reason: str = ""
    # Mehrere Sprecher moeglich? Bei Discord ja, am eigenen Mikrofon nein.
    multi_speaker: bool = False
    details: tuple[str, ...] = field(default_factory=tuple)

    def label(self) -> str:
        return f"{self.title} – {'bereit' if self.ready else self.reason}"


@runtime_checkable
class Transport(Protocol):
    """Der Weg, ueber den gesprochen und gehoert wird."""

    def open(self, context: Any) -> None: ...

    def listen(
        self,
        target: Path,
        on_level: Callable[[float, bool], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_threshold: Callable[[float], None] | None = None,
    ) -> tuple[Path | None, float]:
        """Einen Redebeitrag aufnehmen.

        Rueckgabe: (WAV-Datei in 16 kHz Mono, Sekunden Sprache).
        ``None`` heisst: nichts Verwertbares -- dann soll die
        Spracherkennung nicht auf Rauschen losgelassen werden.
        """
        ...

    def play(self, wav: Path) -> None: ...

    def stop_playback(self) -> None:
        """Laufende Wiedergabe abbrechen (jemand faellt ins Wort)."""
        ...

    def close(self) -> None: ...

    def speaker_hint(self) -> str:
        """Wer zuletzt gesprochen hat, soweit bekannt. Sonst leer."""
        ...


# ---------------------------------------------------------------------------
# Lokal: Mikrofon und Lautsprecher dieses Rechners
# ---------------------------------------------------------------------------
class LocalTransport:
    """Der bisherige Weg: PortAudio ueber ``audio_io``."""

    def __init__(self, config: Any) -> None:
        from . import audio_io

        self.config = config
        self._audio = audio_io
        self._playback = audio_io.Playback()

    # -- Zustand ------------------------------------------------------
    @staticmethod
    def info(config: Any) -> TransportInfo:
        from . import audio_io

        ok, grund = audio_io.available()
        einzelheiten: tuple[str, ...] = ()
        if ok:
            mikros = audio_io.useful_devices(inputs=True)
            einzelheiten = tuple(f"Mikrofon: {g.short_label()}" for g in mikros[:3])
        return TransportInfo(
            key="lokal",
            title="Dieser Rechner",
            ready=ok,
            reason=grund,
            multi_speaker=False,
            details=einzelheiten,
        )

    # -- Ablauf -------------------------------------------------------
    def open(self, context: Any) -> None:
        ok, grund = self._audio.available()
        if not ok:
            raise self._audio.AudioUnavailable(grund)
        geraet = self._geraet(getattr(self.config, "call_input_device", -1))
        name = "Systemvorgabe"
        if geraet is not None:
            treffer = next(
                (g for g in self._audio.devices() if g.index == geraet),
                None,
            )
            if treffer is not None:
                name = treffer.short_label()
        context.status(f"Mikrofon bereit: {name}")

    def listen(
        self,
        target: Path,
        on_level: Callable[[float, bool], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_threshold: Callable[[float], None] | None = None,
    ) -> tuple[Path | None, float]:
        return self._audio.record_turn(
            target,
            on_level=on_level,
            should_stop=should_stop,
            on_threshold=on_threshold,
            device=self._geraet(getattr(self.config, "call_input_device", -1)),
            gain=float(getattr(self.config, "call_input_gain", 1.0) or 1.0),
            silence_seconds=float(getattr(self.config, "call_silence_seconds", 1.0) or 1.0),
        )

    def play(self, wav: Path) -> None:
        self._playback.play(
            wav, device=self._geraet(getattr(self.config, "call_output_device", -1))
        )

    def stop_playback(self) -> None:
        self._playback.stop()

    def close(self) -> None:
        self._playback.stop()

    def speaker_hint(self) -> str:
        return ""

    # -- Hilfen -------------------------------------------------------
    @staticmethod
    def _geraet(wert) -> int | None:
        """-1 (oder leer) heisst Systemvorgabe, sonst die Geraetenummer."""
        try:
            nummer = int(wert)
        except (TypeError, ValueError):
            return None
        return None if nummer < 0 else nummer


# ---------------------------------------------------------------------------
# Auswahl
# ---------------------------------------------------------------------------
def available_transports(config: Any) -> list[TransportInfo]:
    """Alle Wege mit ihrem Zustand – ohne etwas zu laden."""
    from . import pipeline_discord

    return [LocalTransport.info(config), pipeline_discord.DiscordTransport.info(config)]


def create_transport(config: Any) -> Transport:
    """Den eingestellten Weg bauen.

    Fail-closed: Ist "discord" eingestellt, aber nicht einsatzbereit, wird
    NICHT stillschweigend auf das Mikrofon zurueckgefallen. Wer den Bot
    erwartet, soll nicht versehentlich in sein eigenes Zimmer sprechen --
    und umgekehrt soll nichts in einen Kanal gesendet werden, wenn der
    Bediener das gerade nicht will.
    """
    modus = str(getattr(config, "call_mode", "lokal") or "lokal").lower()
    if modus == "discord":
        from . import pipeline_discord

        return pipeline_discord.DiscordTransport(config)
    return LocalTransport(config)
