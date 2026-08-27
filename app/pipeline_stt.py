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
    # Warum nichts erkannt wurde -- etwa "zu leise". Ohne diesen Grund
    # steht im Gespraech nur "keine Antwort", und niemand weiss warum.
    note: str = ""

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


def cuda_reason() -> tuple[bool, str]:
    """Kann CTranslate2 auf der Grafikkarte rechnen -- und wenn nicht, warum?

    Frueher schluckte diese Pruefung jede Ausnahme und gab ein stummes
    ``False`` zurueck. Ein fehlendes cudnn_ops64_9.dll, ein Treiber, der
    nicht antwortet, ein belegter Kontext -- alles sah gleich aus, und
    das Gespraech lief auf der CPU weiter, ohne dass irgendwo stand,
    warum. Wer "alles ueber GPU" eingestellt hat, sieht dann nur, dass
    es langsam ist.

    Deshalb kommt der Grund mit zurueck.
    """
    try:
        import ctranslate2
    except Exception as exc:
        return False, f"CTranslate2 nicht ladbar ({clean_error(exc)})"

    try:
        anzahl = int(ctranslate2.get_cuda_device_count())
    except Exception as exc:
        # Hier landen die interessanten Faelle: fehlende CUDA- oder
        # cuDNN-Bibliotheken, ein Treiber, der nicht antwortet.
        return False, f"CUDA-Abfrage fehlgeschlagen ({clean_error(exc)})"

    if anzahl <= 0:
        return False, "CTranslate2 sieht keine CUDA-Karte"
    return True, f"{anzahl} CUDA-Karte(n)"


def cuda_available() -> bool:
    """Kann CTranslate2 auf der Grafikkarte rechnen?"""
    geht, grund = cuda_reason()
    if not geht:
        log.info("Spracherkennung kann die Grafikkarte nicht nutzen: %s", grund)
    return geht


def device_for(plan: Any = None) -> tuple[str, str]:
    """Gerät und Genauigkeit wählen. Rückgabe: (Gerät, Genauigkeit).

    Der Backend-Plan wird hier NICHT als Verbot gelesen, sondern nur als
    Wunsch. Der Grund ist ein Fehler, der ein ganzes Telefonat auf die
    CPU zwang: der Plan richtet sich nach dem **Bildmodell**. Ist SDXL
    nicht heruntergeladen, meldet die Backend-Kette "CUDA nicht bereit"
    und stellt auf CPU – obwohl die Karte da ist und Whisper eigene,
    längst geladene Gewichte hat.

    Maßgeblich ist deshalb, ob CTranslate2 die Karte wirklich sieht. Nur
    wenn jemand ausdrücklich ein anderes Gerät erzwingt (``device``
    in der Konfiguration steht auf "cpu"), bleibt es dabei.

    Die Karte wird nur genommen, wenn CTranslate2 sie wirklich sieht –
    ein ``device="cuda"`` ohne Karte wirft mitten im Gespräch.
    """
    # Hat der Bediener das Backend ausdrücklich festgelegt, gilt das auch
    # hier – ``forced`` unterscheidet die feste Wahl von der automatischen.
    if getattr(plan, "forced", False) and getattr(plan, "backend", None) != Backend.CUDA:
        return "cpu", COMPUTE_CPU

    geht, grund = cuda_reason()
    if geht:
        return "cuda", COMPUTE_CUDA
    # Nicht stumm ausweichen: wer "alles ueber GPU" eingestellt hat,
    # soll lesen koennen, woran es lag.
    log.warning("Spracherkennung rechnet auf der CPU: %s", grund)
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


# Übliche Sprachaufnahmen liegen hier. Darunter tut sich Whisper schwer.
# Darunter ist es keine Sprache, sondern Rauschen (siehe audio_io).
MIN_SPRACHE_RMS = 0.008

# Was Whisper aus Rauschen erfindet.
#
# Das Modell ist auf Untertiteln trainiert und gibt bei fehlender Sprache
# immer dieselben Abspann-Floskeln aus. An echten Anrufaufnahmen
# gemessen: "Thanks for watching!" und "Bis zum naechsten Mal." - gesagt
# hatte der Anrufer nichts davon. Eine erfundene Frage ist schlimmer als
# gar keine: das Sprachmodell antwortet dann auf etwas, das niemand
# gesagt hat.
ERFUNDEN = frozenset(
    {
        "thanks for watching",
        "thank you for watching",
        "bis zum nächsten mal",
        "bis zum naechsten mal",
        "vielen dank",
        "vielen dank für die aufmerksamkeit",
        "untertitel im auftrag des zdf",
        "untertitelung des zdf",
        "untertitel von stephanie geiges",
        "copyright wdr",
        "das war's",
        "das wars",
        "tschüss",
        "amara.org",
        "subtitles by the amara.org community",
        "so",
        "ende",
    }
)

# Bis hierher ist eine Aufnahme zu kurz, um mehr als eine Floskel zu
# enthalten. In einem langen Beitrag kann "vielen Dank" durchaus
# gefallen sein - bei zwei Sekunden Rauschen nicht.
KURZ_S = 4.0

# Ab dem Wievielfachen des RMS ein Ausschlag als Impuls gilt.
#
# Sprache erreicht typisch das Vier- bis Sechsfache ihres RMS. Was
# deutlich darueber liegt, ist in einer Anrufaufnahme fast immer etwas
# anderes: eine Tastatur, ein Klicken, ein Knacken bei Paketverlust.
# Solche Ausreisser deckelten frueher die Verstaerkung fuer die GANZE
# Aufnahme (gemessen: 1,1-fach statt der noetigen 3,2-fach).
IMPULS_GRENZE = 8.0


def _ist_erfunden(text: str, sekunden: float) -> bool:
    """Sieht das nach einer erfundenen Floskel aus?"""
    if not text or sekunden > KURZ_S:
        return False
    sauber = text.strip().lower().rstrip(".!?…").strip()
    return sauber in ERFUNDEN


ZIEL_RMS = 0.08
MAX_VERSTAERKUNG = 20.0
MAX_SPITZE = 0.95


def _pegel(quelle: Path) -> float | None:
    """Effektivwert der Aufnahme (0..1). ``None``, wenn nicht lesbar."""
    import audioop
    import contextlib
    import wave

    try:
        with contextlib.closing(wave.open(str(quelle), "rb")) as ein:
            daten = ein.readframes(ein.getnframes())
            breite = ein.getsampwidth()
    except (wave.Error, OSError):
        return None
    if not daten:
        return 0.0
    return audioop.rms(daten, breite) / 32768.0


def _gekappt(daten: bytes, breite: int, grenze: float) -> bytes:
    """Ausschlaege oberhalb der Grenze begrenzen.

    Hart gekappt, nicht weich: fuer die Spracherkennung ist das
    unkritisch, weil ein Impuls ohnehin keine Sprache trug. Weich zu
    begrenzen wuerde hier nur Rechenzeit kosten.
    """
    import array

    typ = {1: "b", 2: "h", 4: "i"}.get(breite)
    if typ is None:
        return daten
    werte = array.array(typ)
    werte.frombytes(daten)
    hoechst = int(grenze * 32768)
    for i, wert in enumerate(werte):
        if wert > hoechst:
            werte[i] = hoechst
        elif wert < -hoechst:
            werte[i] = -hoechst
    return werte.tobytes()


def _angehoben(quelle: Path) -> Path | None:
    """Eine zu leise Aufnahme lauter machen – als Kopie.

    Gemessen an einem echten Anruf: RMS 0,014 bis 0,028. Der Stille-Filter
    verwarf das vollständig, und ohne ihn halluzinierte Whisper. Angehoben
    auf ein übliches Maß wird daraus verwertbares Material.

    Rückgabe: Pfad der lauteren Kopie, oder ``None`` wenn nichts nötig
    war (oder nichts möglich).
    """
    import audioop
    import contextlib
    import tempfile
    import wave

    try:
        with contextlib.closing(wave.open(str(quelle), "rb")) as ein:
            kanaele, breite, rate = ein.getnchannels(), ein.getsampwidth(), ein.getframerate()
            daten = ein.readframes(ein.getnframes())
    except (wave.Error, OSError) as exc:
        log.debug("Aufnahme nicht lesbar: %s", clean_error(exc))
        return None

    if not daten:
        return None

    rms = audioop.rms(daten, breite) / 32768.0
    if rms >= ZIEL_RMS or rms <= 0.0:
        return None  # laut genug oder still

    faktor = min(ZIEL_RMS / rms, MAX_VERSTAERKUNG)

    # Einzelne Impulse zuerst kappen.
    #
    # Eine Discord-Aufnahme ist leise Sprache mit einzelnen lauten
    # Ausschlaegen: Tastatur, Klicken, Knacken bei Paketverlust. Ohne
    # Kappen bestimmt diese Handvoll Abtastwerte ueber die Spitze den
    # Pegel der ganzen Aufnahme - gemessen 1,1-fach statt der noetigen
    # 3,2-fach, und die Erkennung fand nichts.
    spitze = audioop.max(daten, breite) / 32768.0
    grenze = min(1.0, IMPULS_GRENZE * rms)
    gekappt = False
    if spitze > grenze > 0 and ZIEL_RMS / rms > MAX_SPITZE / spitze:
        # Nur kappen, wenn die Spitze wirklich im Weg steht - sonst bleibt
        # die Aufnahme unangetastet.
        daten = _gekappt(daten, breite, grenze)
        spitze = audioop.max(daten, breite) / 32768.0
        gekappt = True

    # Nicht übersteuern: die Spitze begrenzt den Faktor mit.
    if spitze > 0:
        faktor = min(faktor, MAX_SPITZE / spitze)
    if faktor <= 1.05:
        return None

    try:
        lauter = audioop.mul(daten, breite, faktor)
    except audioop.error as exc:
        log.debug("Anheben fehlgeschlagen: %s", clean_error(exc))
        return None

    ziel = Path(tempfile.gettempdir()) / f"sf-laut-{quelle.stem}.wav"
    try:
        with contextlib.closing(wave.open(str(ziel), "wb")) as aus:
            aus.setnchannels(kanaele)
            aus.setsampwidth(breite)
            aus.setframerate(rate)
            aus.writeframes(lauter)
    except (wave.Error, OSError) as exc:
        log.debug("Lautere Fassung nicht schreibbar: %s", clean_error(exc))
        return None

    if gekappt:
        log.info(
            "Aufnahme war leise (RMS %.4f) – Impulse gekappt, %.1f-fach angehoben.",
            rms,
            faktor,
        )
    else:
        log.info("Aufnahme war leise (RMS %.4f) – %.1f-fach angehoben.", rms, faktor)
    return ziel


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

        # Ist da überhaupt Sprache drin?
        #
        # Whisper erfindet aus Rauschen ganze Sätze -- "Thanks for
        # watching!", "Bis zum nächsten Mal." sind die bekanntesten. Eine
        # erfundene Antwort ist schlimmer als gar keine, denn das
        # Sprachmodell antwortet dann auf etwas, das niemand gesagt hat.
        pegel = _pegel(Path(wav))
        if pegel is not None and pegel < MIN_SPRACHE_RMS:
            log.info("Aufnahme enthält keine Sprache (RMS %.4f).", pegel)
            return Transcript(
                text="",
                language=sprache or "",
                seconds=0.0,
                elapsed_s=time.time() - begonnen,
                model_key=self.spec.key,
                device=self.device,
                note=(
                    f"Zu leise (Pegel {pegel:.3f}). Näher ans Mikrofon, "
                    "lauter sprechen oder die Verstärkung erhöhen."
                ),
            )

        # Zu leise Aufnahmen anheben, bevor sie in die Erkennung gehen.
        # Die Datei im Gesprächsordner bleibt unverändert – sie soll
        # klingen wie aufgenommen.
        quelle = _angehoben(Path(wav)) or Path(wav)

        segmente, info = self._modell.transcribe(
            str(quelle),
            language=sprache,
            beam_size=1,  # Gespräch: Tempo vor letzter Genauigkeit
            vad_filter=True,  # Stille am Rand wegschneiden
            vad_parameters={
                # Vorgabe 0,5 ist zu streng. Gemessen an einem echten
                # Anruf (RMS 0,014-0,028, Spitzen 0,10-0,24) verwarf der
                # Filter die GANZE Aufnahme, und im Gespräch stand
                # dreimal "keine Antwort".
                "threshold": 0.25,
                # Kurze Pausen im Satz nicht als Ende werten.
                "min_silence_duration_ms": 500,
            },
            condition_on_previous_text=False,
            # Eine Temperatur statt sechs.
            #
            # Whisper arbeitet bei schlechten Ergebnissen die Leiter
            # 0,0 -> 0,2 -> 0,4 -> 0,6 -> 0,8 -> 1,0 ab. Jede Stufe ist
            # ein vollstaendiger Durchlauf. An einem Discord-Anruf
            # gemessen kostete das ueber zehn Sekunden -- und keine der
            # Stufen fand etwas, weil in der Aufnahme nichts war.
            #
            # Im Gespraech zaehlt die Leitung. Wer eine Datei abtippen
            # laesst, bekommt die Leiter weiterhin (dort ist temperature
            # nicht gesetzt).
            temperature=0.0,
        )
        text = " ".join(teil.text.strip() for teil in segmente).strip()

        # Zweiter Versuch ohne Filter.
        #
        # Verwirft er trotzdem alles, ist ein leerer Text das schlechteste
        # Ergebnis: der Anrufer redet, die Gegenseite schweigt ohne Grund.
        # Lieber ein Wort zu viel erkennen als eine stumme Leitung.
        #
        # Aber nur, wenn ueberhaupt Sprache drin sein KANN: bei einem
        # Pegel unter MIN_SPRACHE_RMS ist da Rauschen, und der zweite
        # Durchlauf kostet dann die volle Zeit fuer ein Ergebnis, das es
        # nicht gibt. Gemessen: RMS 0,0059, zehn Sekunden, nichts.
        zu_leise = pegel is not None and pegel < MIN_SPRACHE_RMS
        if not text and zu_leise:
            log.info(
                "Kein zweiter Versuch: Pegel %.4f liegt unter %.4f – da ist keine Sprache.",
                pegel,
                MIN_SPRACHE_RMS,
            )
        elif not text:
            log.info("Stille-Filter verwarf alles – zweiter Versuch ohne ihn.")
            segmente, info = self._modell.transcribe(
                str(quelle),
                language=sprache,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
            )
            text = " ".join(teil.text.strip() for teil in segmente).strip()

        # Erfundene Floskeln verwerfen.
        #
        # Whisper ist auf Untertiteln trainiert und gibt bei Rauschen
        # ohne Sprache immer dieselben Abspann-Floskeln aus. An echten
        # Anrufaufnahmen gemessen: "Thanks for watching!" und "Bis zum
        # nächsten Mal." -- gesagt hatte der Anrufer nichts davon.
        dauer = float(getattr(info, "duration", 0.0) or 0.0)
        if _ist_erfunden(text, dauer):
            log.info("Verworfen, weil aus Rauschen erfunden: %r", text)
            text = ""
            hinweis = (
                "Nichts verstanden – die Aufnahme enthält keine Sprache. "
                "Näher ans Mikrofon, lauter sprechen oder die Verstärkung erhöhen."
            )
        elif not text:
            hinweis = (
                f"Nichts verstanden (Pegel {pegel:.3f})."
                if pegel is not None
                else "Nichts verstanden."
            ) + " Näher ans Mikrofon oder die Verstärkung erhöhen."
        else:
            hinweis = ""

        return Transcript(
            note=hinweis,
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
