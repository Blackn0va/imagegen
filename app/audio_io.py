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

# Verstärkung des Mikrofons. 1,0 heißt unverändert.
#
# Nach oben begrenzt, weil ein Faktor über 20 nur noch das Rauschen des
# Vorverstärkers lauter macht – wer so viel braucht, hat ein Problem am
# Gerät, das keine Rechnung löst.
MIN_GAIN = 1.0
MAX_GAIN = 20.0

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
    hostapi: str = ""
    default_rate: int = 0

    def label(self) -> str:
        art = []
        if self.inputs:
            art.append("Eingang")
        if self.outputs:
            art.append("Ausgang")
        teile = ", ".join(art)
        if self.hostapi:
            teile += f", {self.hostapi}"
        return f"[{self.index}] {self.name} ({teile})"

    def short_label(self) -> str:
        """Kurzform für die Oberfläche – Name und Schnittstelle."""
        if self.hostapi:
            return f"{self.name} · {self.hostapi}"
        return self.name


def devices() -> list[DeviceInfo]:
    """Alle Audiogeräte. Leere Liste, wenn nichts abfragbar ist."""
    try:
        import sounddevice as sd

        apis = sd.query_hostapis()
        gefunden = []
        for index, eintrag in enumerate(sd.query_devices()):
            nummer = int(eintrag.get("hostapi", -1))
            api = str(apis[nummer]["name"]) if 0 <= nummer < len(apis) else ""
            gefunden.append(
                DeviceInfo(
                    index=index,
                    name=str(eintrag.get("name", "?")),
                    inputs=int(eintrag.get("max_input_channels", 0)),
                    outputs=int(eintrag.get("max_output_channels", 0)),
                    hostapi=api,
                    default_rate=int(eintrag.get("default_samplerate", 0) or 0),
                )
            )
        return gefunden
    except Exception as exc:
        log.debug("Geräteliste nicht abrufbar: %s", exc)
        return []


# Raten, die als Ausweichlösung geprüft werden. Reihenfolge zählt: je
# näher an 16 kHz, desto weniger muss hinterher gerechnet werden.
FALLBACK_RATES = (16_000, 48_000, 44_100, 32_000, 22_050, 8_000)


def _device_default_rate(device: int | None, eingang: bool) -> int:
    """Vorgaberate eines Geräts, 0 wenn unbekannt."""
    try:
        import sounddevice as sd

        eintrag = sd.query_devices(
            device if device is not None else None, "input" if eingang else "output"
        )
        return int(eintrag.get("default_samplerate", 0) or 0)
    except Exception:
        return 0


def pick_input_rate(device: int | None = None, wunsch: int = SAMPLE_RATE) -> int:
    """Eine Aufnahmerate finden, die das Gerät wirklich annimmt.

    Ohne diese Aushandlung bricht das Telefonat auf WASAPI- und
    WDM-Geräten sofort ab: die laufen im Shared Mode fest auf 48 kHz und
    melden für 16 kHz 'Invalid sample rate [PaErrorCode -9997]'. Nur
    MME/DirectSound rechnen selbst um.

    Rückgabe ist die Rate, mit der der Strom geöffnet werden kann – nicht
    zwingend die gewünschte. Wer 16 kHz braucht, rechnet danach mit
    ``resample`` herunter.
    """
    try:
        import sounddevice as sd
    except Exception:
        return wunsch

    kandidaten: list[int] = [wunsch]
    vorgabe = _device_default_rate(device, eingang=True)
    if vorgabe:
        kandidaten.append(vorgabe)
    kandidaten.extend(FALLBACK_RATES)

    geprueft: set[int] = set()
    for rate in kandidaten:
        rate = int(rate)
        if rate <= 0 or rate in geprueft:
            continue
        geprueft.add(rate)
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype="float32")
            if rate != wunsch:
                log.info("Mikrofon kann keine %d Hz – nehme %d Hz.", wunsch, rate)
            return rate
        except Exception:
            continue
    # Nichts hat gepasst: mit dem Wunsch öffnen und den PortAudio-Fehler
    # sprechen lassen, statt hier eine eigene Vermutung zu erfinden.
    return wunsch


def pick_output_rate(device: int | None = None, wunsch: int = SAMPLE_RATE) -> int:
    """Dasselbe für die Wiedergabe.

    Betrifft echte Dateien: die Windows-Stimmen liefern 22050 Hz, Bark
    24000 Hz – beides lehnt ein WASAPI-Gerät ab.
    """
    try:
        import sounddevice as sd
    except Exception:
        return wunsch

    kandidaten: list[int] = [wunsch]
    vorgabe = _device_default_rate(device, eingang=False)
    if vorgabe:
        kandidaten.append(vorgabe)
    kandidaten.extend(FALLBACK_RATES)

    geprueft: set[int] = set()
    for rate in kandidaten:
        rate = int(rate)
        if rate <= 0 or rate in geprueft:
            continue
        geprueft.add(rate)
        try:
            sd.check_output_settings(device=device, samplerate=rate, channels=1, dtype="float32")
            return rate
        except Exception:
            continue
    return wunsch


def resample(samples, von: int, nach: int):
    """Abtastrate umrechnen.

    Mit ``scipy`` über ``resample_poly`` (Polyphase, mit Tiefpass). Ohne
    scipy über einen Kastenfilter plus lineare Interpolation: nicht so
    sauber, aber es verhindert das gröbste Aliasing beim Herunterrechnen.
    Ganz ohne Filter würden die Frequenzen über der halben Zielrate als
    Störtöne zurückfalten – Whisper hört dann Wörter, die niemand gesagt
    hat.
    """
    import numpy as np

    werte = np.asarray(samples, dtype="float32").ravel()
    von, nach = int(von), int(nach)
    if von == nach or werte.size == 0 or von <= 0 or nach <= 0:
        return werte

    try:
        from math import gcd

        from scipy.signal import resample_poly

        teiler = gcd(von, nach)
        return resample_poly(werte, nach // teiler, von // teiler).astype("float32")
    except Exception as exc:  # scipy fehlt oder mag die Längen nicht
        log.debug("resample_poly nicht nutzbar (%s) – einfacher Weg.", exc)

    if nach < von:
        # Kastenfilter über so viele Abtastwerte, wie zusammenfallen.
        breite = max(1, round(von / nach))
        if breite > 1:
            kern = np.ones(breite, dtype="float32") / breite
            werte = np.convolve(werte, kern, mode="same").astype("float32")
    laenge = max(1, round(werte.size * nach / von))
    alt = np.arange(werte.size, dtype="float64")
    neu = np.linspace(0, werte.size - 1, laenge, dtype="float64")
    return np.interp(neu, alt, werte).astype("float32")


def apply_gain(block, gain: float):
    """Abtastwerte verstärken, ohne sie zu übersteuern.

    Hart abschneiden würde Sprache verzerren und Whisper das Erkennen
    erschweren. Deshalb wird nur bis knapp unter Vollausschlag skaliert:
    Werte, die danach über 1,0 lägen, werden weich begrenzt.
    """
    import numpy as np

    werte = np.asarray(block, dtype="float32")
    if gain is None or abs(gain - 1.0) < 1e-3 or werte.size == 0:
        return werte
    faktor = float(max(MIN_GAIN, min(MAX_GAIN, gain)))
    verstaerkt = werte * faktor
    # Weiche Begrenzung: unterhalb 0,9 unverändert, darüber sanft in die
    # Sättigung. tanh liefert genau das, ohne Knacken an der Kante.
    spitze = float(np.max(np.abs(verstaerkt))) if verstaerkt.size else 0.0
    if spitze > 0.9:
        verstaerkt = np.tanh(verstaerkt * 1.1) * 0.95
    return verstaerkt.astype("float32")


# Sammelgeräte: Weiterleitungen auf das, was Windows gerade als Vorgabe
# führt. Genau das ist der Eintrag "Systemvorgabe" – doppelt braucht es
# das nicht.
_SAMMEL_NAMEN = (
    "soundmapper",
    "sound mapper",
    "primärer soundaufnahmetreiber",
    "primary sound capture",
    "primärer soundtreiber",
    "primary sound driver",
)

# Rückflüsse des Ausgangs. Kein Mikrofon, sondern das, was aus den
# Lautsprechern kommt. Für ein Gespräch falsch, für Mitschnitt manchmal
# gewollt – deshalb nicht weg, nur nach hinten.
_RUECKFLUSS = ("stream out", "loopback", "stereomix", "stereo mix", "was aufnehmen")

# Reihenfolge der Schnittstellen bei gleichem Gerät. MME steht vorn, weil
# es jede Abtastrate selbst umrechnet; WASAPI verlangt seine 48 kHz.
_API_RANG = {"MME": 0, "Windows DirectSound": 1, "Windows WASAPI": 2}


def _kern_name(name: str) -> str:
    """Gerätenamen auf das Wesentliche kürzen, um Dubletten zu finden.

    Windows hängt Kanalnummern an ("3- A50 Mic") und PortAudio kürzt
    lange Namen je nach Schnittstelle verschieden ab. Verglichen wird
    deshalb nur der Anfang ohne Zusätze.
    """
    text = name.lower().strip()
    # führende Kanalnummer "3- " entfernen
    teile = text.split("- ", 1)
    if len(teile) == 2 and teile[0].strip().rstrip("-").isdigit():
        text = teile[1]
    # Klammerzusatz abschneiden: "Mikrofon (NVIDIA Broadcast)" -> beides
    # behalten, aber Leerraum vereinheitlichen
    return " ".join(text.replace("(", " ").replace(")", " ").split())[:28]


def is_loopback(name: str) -> bool:
    text = name.lower()
    return any(wort in text for wort in _RUECKFLUSS)


def useful_devices(
    inputs: bool = True,
    include_all: bool = False,
) -> list[DeviceInfo]:
    """Geräte, die für ein Gespräch taugen – ohne Dubletten und Sackgassen.

    ``include_all=True`` gibt alles zurück (nur sortiert), für den Fall,
    dass die Auswahl doch zu eng war.
    """
    alle = [g for g in devices() if (g.inputs if inputs else g.outputs)]
    if include_all:
        return sorted(alle, key=lambda g: (_API_RANG.get(g.hostapi, 9), g.index))

    brauchbar = []
    for geraet in alle:
        if geraet.hostapi not in GOOD_HOSTAPIS:
            continue  # WDM-KS lässt sich nicht blockierend lesen
        klein = geraet.name.lower()
        if any(wort in klein for wort in _SAMMEL_NAMEN):
            continue  # deckt "Systemvorgabe" bereits ab
        brauchbar.append(geraet)

    # Dubletten: dasselbe Gerät über mehrere Schnittstellen. Die mit dem
    # besten Rang gewinnt.
    beste: dict[str, DeviceInfo] = {}
    for geraet in sorted(brauchbar, key=lambda g: (_API_RANG.get(g.hostapi, 9), g.index)):
        beste.setdefault(_kern_name(geraet.name), geraet)

    return sorted(
        beste.values(),
        key=lambda g: (is_loopback(g.name), _API_RANG.get(g.hostapi, 9), g.index),
    )


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


# Schnittstellen, die für ein Telefonat taugen. Windows WDM-KS steht
# bewusst nicht dabei: PortAudio kann darüber nur asynchron lesen und
# meldet beim Öffnen "Blocking API not supported yet".
GOOD_HOSTAPIS = ("MME", "Windows DirectSound", "Windows WASAPI")


def _current_device(device: int | None) -> DeviceInfo | None:
    if device is None or device < 0:
        return None
    for geraet in devices():
        if geraet.index == device:
            return geraet
    return None


def suggest_input(like: DeviceInfo | None = None) -> DeviceInfo | None:
    """Ein Mikrofon vorschlagen, das erfahrungsgemäß funktioniert.

    Bevorzugt dasselbe Gerät über eine brauchbare Schnittstelle – wer
    "Kopfhörermikrofon" über WDM-KS gewählt hat, will dasselbe Mikrofon,
    nur über MME. Sonst irgendein Eingang mit brauchbarer Schnittstelle.
    """
    kandidaten = [g for g in devices() if g.inputs and g.hostapi in GOOD_HOSTAPIS]
    if not kandidaten:
        return None
    if like is not None:
        # Namen vergleichen ohne die Kanalnummer davor ("3- A50 Mic").
        kern = like.name.lower().strip()
        for geraet in kandidaten:
            if geraet.name.lower().strip() == kern:
                return geraet
        for geraet in kandidaten:
            if kern[:12] and kern[:12] in geraet.name.lower():
                return geraet
    return kandidaten[0]


def _ersatz_hinweis(device: int | None) -> str:
    """Satz mit einem konkreten Ersatzgerät, sonst leer."""
    ersatz = suggest_input(_current_device(device))
    if ersatz is None:
        return ""
    return f" Nimm stattdessen '[{ersatz.index}] {ersatz.short_label()}'."


def _mic_problem(exc: Exception, device: int | None, rate: int) -> str:
    """Aus einem PortAudio-Fehler eine Meldung machen, die weiterhilft.

    PortAudio meldet Dinge wie "Unanticipated host error [PaErrorCode
    -9999]". Damit kann niemand etwas anfangen. Hier steht, was zu tun
    ist – mit dem Namen eines Geräts, das stattdessen geht.
    """
    from .accel import clean_error

    text = clean_error(exc)
    klein = text.lower()
    aktuell = _current_device(device)
    name = aktuell.short_label() if aktuell else "Systemvorgabe"

    if "blocking api" in klein or "-9999" in text:
        return (
            f"Mikrofon '{name}' lässt sich nicht direkt auslesen "
            "(WDM-KS kann das nicht)." + _ersatz_hinweis(device)
        )
    if "sample rate" in klein or "-9997" in text:
        return f"Mikrofon '{name}' nimmt {rate} Hz nicht an." + _ersatz_hinweis(device)
    if "-9996" in text or "invalid device" in klein:
        return f"Mikrofon '{name}' ist nicht (mehr) verfügbar. Geräteliste neu laden."
    if "-9985" in text or "device unavailable" in klein:
        return f"Mikrofon '{name}' ist von einem anderen Programm belegt." + _ersatz_hinweis(device)
    return f"Mikrofon '{name}' lässt sich nicht öffnen: {text}"


def record_turn(
    target: Path,
    on_level: Callable[[float, bool], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    silence_seconds: float = SILENCE_SECONDS,
    max_seconds: float = MAX_TURN_SECONDS,
    device: int | None = None,
    on_threshold: Callable[[float], None] | None = None,
    gain: float = 1.0,
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

    # Rate aushandeln statt fordern – siehe pick_input_rate.
    rate = pick_input_rate(device, SAMPLE_RATE)
    block_samples = max(1, int(rate * BLOCK_MS / 1000))

    bloecke: list = []
    stille_blocks = 0
    sprach_blocks = 0
    schwelle = MIN_THRESHOLD
    grundrauschen: list[float] = []
    noetige_stille = max(1, int(silence_seconds * 1000 / BLOCK_MS))
    hoechstzahl = int(max_seconds * 1000 / BLOCK_MS)
    kalibrier_blocks = int(CALIBRATION_SECONDS * 1000 / BLOCK_MS)

    try:
        strom = sd.InputStream(
            samplerate=rate,
            channels=1,
            dtype="float32",
            blocksize=block_samples,
            device=device,
        )
    except Exception as exc:
        raise AudioUnavailable(_mic_problem(exc, device, rate)) from exc

    with strom:
        for zaehler in range(hoechstzahl):
            if should_stop is not None and should_stop():
                break
            block, _ueberlauf = strom.read(block_samples)
            # Verstärken, bevor irgendetwas anderes passiert: Anzeige,
            # Schwelle und die Datei für Whisper sollen dasselbe Signal
            # sehen.
            block = apply_gain(block, gain)
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
                # Die Oberfläche zeigt die Schwelle als Marke im Pegel –
                # ohne sie sieht man einen Ausschlag und weiß trotzdem
                # nicht, warum nichts als Sprache gilt.
                if on_threshold is not None:
                    on_threshold(schwelle)

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
    # Whisper erwartet 16 kHz. Wurde höher aufgenommen, wird jetzt
    # heruntergerechnet – einmal, statt bei jedem Block.
    if rate != SAMPLE_RATE:
        daten = resample(daten, rate, SAMPLE_RATE)
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

        # Auch hier aushandeln: die Windows-Stimmen liefern 22050 Hz, ein
        # WASAPI-Gerät nimmt nur 48000 – ohne Umrechnung bricht die
        # Wiedergabe mit demselben Fehler ab wie die Aufnahme.
        ziel_rate = pick_output_rate(device, rate)
        if ziel_rate != rate:
            if kanaele > 1:
                daten = np.column_stack(
                    [resample(daten[:, k], rate, ziel_rate) for k in range(kanaele)]
                )
            else:
                daten = resample(daten, rate, ziel_rate)
            rate = ziel_rate

        begonnen = time.time()
        blocklaenge = max(256, int(rate * 0.05))
        try:
            strom = sd.OutputStream(
                samplerate=rate, channels=kanaele, dtype="float32", device=device
            )
        except Exception as exc:
            from .accel import clean_error

            raise AudioUnavailable(
                f"Wiedergabe lässt sich nicht öffnen: {clean_error(exc)}"
            ) from exc
        with strom:
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
