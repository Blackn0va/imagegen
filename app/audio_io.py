"""Mikrofon aufnehmen und Ton ausgeben.

Für das Telefonieren mit der KI gebraucht: der Regelkreis ist Mikrofon →
Spracherkennung → Sprachmodell → Sprachausgabe → Lautsprecher. Alles
davon lief bisher als Datei; live fehlte beides.

``sounddevice`` ist eine **optionale** Abhängigkeit (PortAudio). Fehlt
sie, sagt das Telefonat das im Klartext und bleibt abgeschaltet – die
Anwendung startet trotzdem.

Erkennung des Sprechendes ohne fremdes VAD-Modell: gemessen wird die
Lautstärke je Block. Nach ``SILENCE_SECONDS`` Ruhe gilt der Redebeitrag
als beendet. Das ist absichtlich einfach – ein Modell dafür wäre ein
weiterer Download und eine weitere Fehlerquelle, und für ein Gespräch am
Schreibtisch trägt die Energie zuverlässig.
"""

from __future__ import annotations

import logging
import threading
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

INSTALL_HINT = "pip install sounddevice"
BUILD_HINT = ".\\build-windows.ps1 -Clean -WithCall $true"

# Aufnahme in 16 kHz Mono: genau das, was Whisper erwartet. Höher
# aufzunehmen und danach herunterzurechnen kostet nur Zeit.
SAMPLE_RATE = 16_000
BLOCK_MS = 30
BLOCK_SAMPLES = SAMPLE_RATE * BLOCK_MS // 1000

# Schwellen für die Sprech-Erkennung.
SILENCE_SECONDS = 1.0  # so lange Ruhe = Redebeitrag zu Ende
MAX_TURN_SECONDS = 60.0  # Notbremse gegen ein offenes Mikrofon
MIN_SPEECH_SECONDS = 0.35  # kürzeres gilt als Huster, nicht als Beitrag
CALIBRATION_SECONDS = 0.6  # Grundrauschen zu Beginn messen
NOISE_FACTOR = 3.0  # so viel über dem Grundrauschen gilt als Sprache
MIN_THRESHOLD = 0.004  # Untergrenze für sehr stille Mikrofone


class AudioUnavailable(RuntimeError):
    """Kein Audiogerät nutzbar – Text enthält die Anleitung."""

    expected = True


def _nachruesten() -> str:
    import sys

    if getattr(sys, "frozen", False):
        return (
            "Dies ist ein gebautes Programm mit eigenem Python – ein "
            f"'pip install' wirkt hier nicht. Neu bauen mit: {BUILD_HINT}"
        )
    return f"Nachrüsten: {INSTALL_HINT}"


def available() -> tuple[bool, str]:
    """Ist Aufnahme und Wiedergabe möglich?"""
    import importlib.util

    from .accel import clean_error

    if importlib.util.find_spec("sounddevice") is None:
        return False, f"sounddevice fehlt. {_nachruesten()}"
    try:
        import sounddevice as sd
    except Exception as exc:
        return False, f"sounddevice nicht ladbar ({clean_error(exc)}). {_nachruesten()}"
    try:
        geraete = sd.query_devices()
    except Exception as exc:
        return False, f"Kein Audiogerät abfragbar: {clean_error(exc)}"
    eingang = [d for d in geraete if int(d.get("max_input_channels", 0)) > 0]
    ausgang = [d for d in geraete if int(d.get("max_output_channels", 0)) > 0]
    if not eingang:
        return False, "Kein Mikrofon gefunden."
    if not ausgang:
        return False, "Kein Wiedergabegerät gefunden."
    return True, f"{len(eingang)} Mikrofon(e), {len(ausgang)} Wiedergabegerät(e)."


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    inputs: int
    outputs: int

    def label(self) -> str:
        art = []
        if self.inputs:
            art.append("Eingang")
        if self.outputs:
            art.append("Ausgang")
        return f"[{self.index}] {self.name} ({', '.join(art)})"


def devices() -> list[DeviceInfo]:
    """Alle Audiogeräte. Leere Liste, wenn nichts abfragbar ist."""
    try:
        import sounddevice as sd

        return [
            DeviceInfo(
                index=index,
                name=str(eintrag.get("name", "?")),
                inputs=int(eintrag.get("max_input_channels", 0)),
                outputs=int(eintrag.get("max_output_channels", 0)),
            )
            for index, eintrag in enumerate(sd.query_devices())
        ]
    except Exception as exc:
        log.debug("Geräteliste nicht abrufbar: %s", exc)
        return []


def rms(block) -> float:
    """Lautstärke eines Blocks als quadratisches Mittel."""
    import numpy as np

    if block is None or len(block) == 0:
        return 0.0
    werte = np.asarray(block, dtype="float32").ravel()
    return float(np.sqrt(np.mean(np.square(werte)))) if werte.size else 0.0


def threshold_from_noise(pegel: list[float]) -> float:
    """Auslöseschwelle aus dem gemessenen Grundrauschen ableiten.

    Ohne diese Messung löst ein rauschendes Mikrofon sofort aus und ein
    sehr leises nie. Getrennt von der Aufnahme, damit die Regel ohne
    Audiogerät prüfbar ist.
    """
    if not pegel:
        return MIN_THRESHOLD
    mittel = sum(pegel) / len(pegel)
    return max(MIN_THRESHOLD, mittel * NOISE_FACTOR)


def record_turn(
    target: Path,
    on_level: Callable[[float, bool], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    silence_seconds: float = SILENCE_SECONDS,
    max_seconds: float = MAX_TURN_SECONDS,
    device: int | None = None,
) -> tuple[Path | None, float]:
    """Einen Redebeitrag aufnehmen, bis Ruhe eintritt.

    Rückgabe: (Datei, Sekunden Sprache). ``None`` heißt: es wurde nichts
    Verwertbares gesprochen – dann soll der Anrufer die Spracherkennung
    nicht auf Rauschen loslassen.

    ``on_level`` bekommt (Pegel, spricht_gerade) je Block, damit die
    Oberfläche einen Aussteuerungsbalken zeigen kann.
    """
    ok, grund = available()
    if not ok:
        raise AudioUnavailable(grund)

    import numpy as np
    import sounddevice as sd

    bloecke: list = []
    stille_blocks = 0
    sprach_blocks = 0
    schwelle = MIN_THRESHOLD
    grundrauschen: list[float] = []
    noetige_stille = max(1, int(silence_seconds * 1000 / BLOCK_MS))
    hoechstzahl = int(max_seconds * 1000 / BLOCK_MS)
    kalibrier_blocks = int(CALIBRATION_SECONDS * 1000 / BLOCK_MS)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=BLOCK_SAMPLES,
        device=device,
    ) as strom:
        for zaehler in range(hoechstzahl):
            if should_stop is not None and should_stop():
                break
            block, _ueberlauf = strom.read(BLOCK_SAMPLES)
            pegel = rms(block)

            # Erst das Grundrauschen messen, dann zuhören.
            if zaehler < kalibrier_blocks:
                grundrauschen.append(pegel)
                if on_level is not None:
                    on_level(pegel, False)
                continue
            if zaehler == kalibrier_blocks:
                schwelle = threshold_from_noise(grundrauschen)
                log.debug("Auslöseschwelle %.4f", schwelle)

            spricht = pegel >= schwelle
            if on_level is not None:
                on_level(pegel, spricht)

            if spricht:
                sprach_blocks += 1
                stille_blocks = 0
                bloecke.append(np.array(block, copy=True))
            elif sprach_blocks:
                # Ruhe NACH Sprache mit aufnehmen, damit das Wortende nicht
                # abgeschnitten wird.
                stille_blocks += 1
                bloecke.append(np.array(block, copy=True))
                if stille_blocks >= noetige_stille:
                    break

    gesprochen = sprach_blocks * BLOCK_MS / 1000.0
    if not bloecke or gesprochen < MIN_SPEECH_SECONDS:
        return None, gesprochen

    daten = np.concatenate(bloecke).ravel()
    write_wav_float(target, daten, SAMPLE_RATE)
    return target, gesprochen


def write_wav_float(target: Path, samples, sample_rate: int = SAMPLE_RATE) -> Path:
    """Fließkomma-Aufnahme als 16-Bit-WAV schreiben."""
    import numpy as np

    from . import paths

    paths.ensure_dir(target.parent)
    werte = np.clip(np.asarray(samples, dtype="float32"), -1.0, 1.0)
    ganz = (werte * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as datei:
        datei.setnchannels(1)
        datei.setsampwidth(2)
        datei.setframerate(int(sample_rate))
        datei.writeframes(ganz.tobytes())
    return target


class Playback:
    """Wiedergabe, die sich abbrechen lässt.

    Abbrechbar ist Pflicht: wer der KI ins Wort fällt, will nicht warten,
    bis sie ihren Satz zu Ende gesprochen hat.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    def play(self, wav: Path, device: int | None = None) -> float:
        """WAV abspielen. Rückgabe: tatsächlich gespielte Sekunden."""
        ok, grund = available()
        if not ok:
            raise AudioUnavailable(grund)

        import numpy as np
        import sounddevice as sd

        self._stop.clear()
        with wave.open(str(wav), "rb") as datei:
            rate = datei.getframerate()
            breite = datei.getsampwidth()
            kanaele = datei.getnchannels()
            roh = datei.readframes(datei.getnframes())

        if breite != 2:
            raise AudioUnavailable(f"{wav.name}: nur 16-Bit-WAV wird abgespielt.")
        daten = np.frombuffer(roh, dtype="<i2").astype("float32") / 32768.0
        if kanaele > 1:
            daten = daten.reshape(-1, kanaele)

        begonnen = time.time()
        blocklaenge = max(256, int(rate * 0.05))
        with sd.OutputStream(
            samplerate=rate, channels=kanaele, dtype="float32", device=device
        ) as strom:
            for anfang in range(0, len(daten), blocklaenge):
                if self._stop.is_set():
                    break
                strom.write(np.ascontiguousarray(daten[anfang : anfang + blocklaenge]))
        return time.time() - begonnen


def describe() -> str:
    """Zustandsbericht für Diagnose und Oberfläche."""
    ok, grund = available()
    zeilen = [grund]
    if ok:
        for geraet in devices():
            if geraet.inputs or geraet.outputs:
                zeilen.append(f"  {geraet.label()}")
    return "\n".join(zeilen)
