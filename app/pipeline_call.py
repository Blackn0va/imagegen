"""Telefonieren mit der KI.

Verbindet vier Teile, die es einzeln schon gab, zu einem Gespräch:

    Mikrofon → Spracherkennung → Sprachmodell → Sprachausgabe → Lautsprecher
    (audio_io)   (pipeline_stt)   (pipeline_chat)  (pipeline_voice)  (audio_io)

Die Stimme ist frei wählbar: eine der mitgelieferten oder eine **selbst
angelernte** aus den Stimmprofilen. Beides läuft über dieselbe
``VoiceRequest`` – für den Gesprächskreis macht es keinen Unterschied.

**Dateien aus dem Gespräch.** Gesprochene Antworten sind flüchtig; Code
und Listen aus einem Telefonat wären es auch. Deshalb wird jede Antwort
zusätzlich schriftlich geführt: die Mitschrift als Markdown, und jeder
Code-Block als eigene Datei mit passender Endung. Gesprochen wird nur der
Fließtext – Quelltext vorzulesen ist sinnlos, und die Sprachausgabe
bräuchte dafür Minuten.

Jedes Teil ist einzeln abschaltbar und meldet im Klartext, wenn es fehlt.
Ein Gespräch ohne Spracherkennung gibt es nicht – dann bleibt der Chat.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import accel, audio_io, call_transport, models, paths, pipeline_chat, pipeline_stt
from .accel import clean_error
from .config import AppConfig

log = logging.getLogger(__name__)

# Endungen je Sprachangabe im Code-Block. Alles Unbekannte wird .txt –
# eine falsche Endung ist schlimmer als eine neutrale.
CODE_SUFFIX = {
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "json": ".json",
    "html": ".html",
    "css": ".css",
    "bash": ".sh",
    "sh": ".sh",
    "powershell": ".ps1",
    "ps1": ".ps1",
    "sql": ".sql",
    "yaml": ".yaml",
    "yml": ".yaml",
    "csv": ".csv",
    "markdown": ".md",
    "md": ".md",
    "c": ".c",
    "cpp": ".cpp",
    "java": ".java",
    "csharp": ".cs",
    "cs": ".cs",
    "go": ".go",
    "rust": ".rs",
    "rs": ".rs",
    "xml": ".xml",
}

_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.S)

# Was gesprochen wird, muss anders klingen als was geschrieben wird.
CALL_SYSTEM_PROMPT = (
    "Du führst ein Telefongespräch auf Deutsch. Antworte kurz und "
    "gesprochen – zwei bis vier Sätze, keine Aufzählungen, keine "
    "Überschriften. Sprich natürlich, wie am Telefon. "
    "Wenn Quelltext oder eine längere Liste verlangt wird, gib sie in "
    "einem Markdown-Block aus und sage dazu nur einen Satz wie 'Ich habe "
    "dir den Code in eine Datei gelegt.' – der Block wird nicht "
    "vorgelesen, sondern gespeichert. "
    "Erfinde nichts; sage, wenn du etwas nicht weißt."
)


class CallUnavailable(RuntimeError):
    """Ein Teil des Gesprächskreises fehlt – Text nennt welcher."""

    expected = True


@dataclass
class CallTurn:
    """Ein Zug im Gespräch: Frage, Antwort, was dabei entstand."""

    frage: str = ""
    antwort: str = ""
    gesprochen: str = ""
    aufnahme: Path | None = None
    stimme: Path | None = None
    dateien: tuple[Path, ...] = ()
    stt_sekunden: float = 0.0
    denk_sekunden: float = 0.0
    tts_sekunden: float = 0.0
    # Wie lange es vom Beginn der Antwort bis zum ersten hoerbaren Ton
    # dauerte. Die Zahl, die am Telefon wirklich zaehlt.
    erster_ton_s: float = 0.0
    zeitpunkt: float = field(default_factory=time.time)


@dataclass
class CallReadiness:
    """Was für ein Gespräch fehlt – jede Stufe einzeln."""

    audio: tuple[bool, str]
    stt: tuple[bool, str]
    chat: tuple[bool, str]
    # Vierte Stufe: ohne Stimme gibt es kein Gespräch, nur Text. Sie
    # fehlte, obwohl der Modul-Docstring die Kette bis zum Lautsprecher
    # nennt.
    voice: tuple[bool, str] = (True, "")

    @property
    def ready(self) -> bool:
        return self.audio[0] and self.stt[0] and self.chat[0] and self.voice[0]

    def problems(self) -> list[str]:
        return [grund for ok, grund in (self.audio, self.stt, self.chat, self.voice) if not ok]

    def report(self) -> str:
        zeilen = []
        for name, (ok, grund) in (
            ("Mikrofon/Ton", self.audio),
            ("Spracherkennung", self.stt),
            ("Sprachmodell", self.chat),
            ("Sprachausgabe", self.voice),
        ):
            zeilen.append(f"  {name:<16} {'ok  ' if ok else 'FEHLT'}  {grund}")
        return "\n".join(zeilen)


def _modell_da(schluessel: str, was: str) -> tuple[bool, str]:
    """Liegt dieses Modell wirklich auf der Platte?

    Ein vorhandenes Paket sagt nichts über die Gewichte. Ohne diese
    Prüfung meldete die Bereitschaftsanzeige „ok" für Modelle, die erst
    mehrere Gigabyte nachladen müssten -- und im Offline-Betrieb gar
    nicht kommen können.
    """
    if not schluessel:
        return False, f"Kein {was} gewählt."
    try:
        spec = models.resolve(schluessel)
    except Exception as exc:
        return False, f"{was} '{schluessel}' unbekannt ({clean_error(exc)})."
    try:
        if models.is_downloaded(spec):
            return True, f"{spec.title} liegt bereit."
    except Exception as exc:
        return True, f"{spec.title} (nicht prüfbar: {clean_error(exc)})"
    groesse = float(getattr(spec, "approx_size_mb", 0) or 0)
    wieviel = f" ({groesse / 1024:.1f} GB)" if groesse >= 1024 else ""
    return False, f"{spec.title} ist noch nicht geladen{wieviel}."


def readiness(config: AppConfig | None = None) -> CallReadiness:
    """Prüfen, ob ein Gespräch möglich ist – ohne etwas zu laden.

    Ohne ``config`` bleibt es bei der reinen Paketprüfung (so wie
    früher). Mit ``config`` wird zusätzlich gefragt, ob die gewählten
    Modelle überhaupt auf der Platte liegen -- sonst meldet diese
    Funktion „bereit" für ein Gespräch, das erst Gigabyte nachlädt.
    """
    audio = audio_io.available()
    stt = pipeline_stt.runtime_available()
    chat = pipeline_chat.runtime_available()
    voice: tuple[bool, str] = (True, "Windows-Stimme vorhanden.")

    if config is None:
        return CallReadiness(audio=audio, stt=stt, chat=chat, voice=voice)

    # Pakete da, aber liegen auch die Gewichte?
    if stt[0]:
        stt_ok, stt_grund = _modell_da(
            str(getattr(config, "stt_model", "") or ""), "Spracherkennungsmodell"
        )
        if not stt_ok:
            stt = (False, stt_grund)
    if chat[0]:
        chat_ok, chat_grund = _modell_da(brain_model(config), "Denkmodell")
        if not chat_ok:
            chat = (False, chat_grund)

    # Sprachausgabe: die gewählte Stimme muss sprechen können.
    with contextlib.suppress(Exception):
        gewaehlt = str(getattr(config, "call_voice", "") or "")
        katalog = voice_catalog(config)
        treffer = next((v for v in katalog if v.key == gewaehlt), None)
        if treffer is None:
            nutzbar = [v for v in katalog if v.ready]
            voice = (
                (True, f"{nutzbar[0].label} (Vorauswahl).")
                if nutzbar
                else (False, "Keine Stimme nutzbar.")
            )
        elif treffer.ready:
            voice = (True, f"{treffer.label}.")
        else:
            voice = (False, f"{treffer.label}: {treffer.note or 'nicht nutzbar'}")

    return CallReadiness(audio=audio, stt=stt, chat=chat, voice=voice)


# ---------------------------------------------------------------------------
# Stimmenauswahl
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VoiceChoice:
    """Eine wählbare Stimme: mitgeliefert oder selbst angelernt."""

    key: str  # Modellschlüssel, Profil-Slug oder Windows-Stimme
    label: str
    is_profile: bool
    speaker: str = ""
    # Windows-Stimme statt Modell. Schnell genug fuers Gespraech: rund
    # 0,4 s je Satz gegenueber 20 s bei Bark.
    is_sapi: bool = False

    # --- Beschreibende Felder für die Auswahl -------------------------
    # Ohne die sieht eine Stimme, die sofort spricht, genauso aus wie
    # eine, die erst 5 GB lädt und dann 20 s je Satz braucht.
    provider: str = "windows"  # windows | modell | angelernt
    engine: str = ""  # sapi | kokoro | bark | piper | clone
    model_key: str = ""  # Schlüssel im Modellverzeichnis
    ready: bool = True  # kann sofort sprechen
    seconds_per_sentence: float = 0.5  # gemessener Richtwert
    size_mb: float = 0.0  # Download, 0 = keiner
    license_id: str = ""
    note: str = ""  # was der Auswahl noch fehlt
    # Rechnet diese Stimme auf der Grafikkarte? Nur die torch-Motoren
    # koennen das. Windows-Stimmen und Piper nicht -- Piper haengt an
    # onnxruntime, und das mitgelieferte Paket kennt gemessen nur
    # CPU- und Azure-Ausfuehrung, keinen CUDA-Weg.
    on_gpu: bool = False
    # Wie natürlich das Ergebnis klingt (1 blechern .. 5 sehr natürlich).
    # Der Grund, warum jemand eine Stimme überhaupt wechselt.
    quality: int = 2

    @property
    def speed_label(self) -> str:
        """Kurzform der Geschwindigkeit für die Auswahlliste."""
        if self.seconds_per_sentence <= 1.0:
            return "sofort"
        # Die Sekundenzahl sagt bereits alles. Ein zusaetzliches
        # "langsam" davor ist doppelt und sprengt die Zeilenbreite in
        # der Auswahlliste.
        return f"~{self.seconds_per_sentence:.0f} s/Satz"

    @property
    def quality_label(self) -> str:
        return QUALITY_WORDS.get(self.quality, "")

    @property
    def size_label(self) -> str:
        if not self.size_mb:
            return ""
        if self.size_mb >= 1024:
            return f"{self.size_mb / 1024:.1f} GB"
        return f"{self.size_mb:.0f} MB"

    def short_label(self) -> str:
        """Name für die Auswahlliste – kurz genug zum Lesen.

        Vorher stand hier alles auf einmal: Titel, Tempo, Größe, zwei
        Lizenzen und der Fehlergrund. Der längste Eintrag hatte 160
        Zeichen; in einer Auswahlliste ist davon nichts zu erkennen.
        """
        name = self.label
        # Alles ab der ersten Klammer oder dem ersten Gedankenstrich ist
        # Beschreibung, nicht Name.
        for trenner in (" (", " – ", " - "):
            if trenner in name:
                name = name.split(trenner, 1)[0]
                break
        if not self.ready:
            # Zwischen "muss erst geladen werden" und "geht gar nicht"
            # unterscheiden – das eine kostet Zeit, das andere ist ein
            # Verbot oder ein fehlendes Paket.
            hinweis = (self.note or "").lower()
            if "lädt" in hinweis or "laedt" in hinweis:
                return f"{name} — {self.size_label or 'Download'} laden"
            return f"{name} — nicht nutzbar"
        # Der Klang zuerst: er ist der Grund, warum jemand die Stimme
        # überhaupt wechselt. Danach, was sie kostet – beides gehört in
        # die Zeile, sonst wählt man Klang ohne zu wissen, was er dauert.
        if self.on_gpu:
            return f"{name} — {self.quality_label}, GPU, {self.speed_label}"
        if self.seconds_per_sentence > 10:
            return f"{name} — {self.quality_label}, langsam"
        return f"{name} — {self.quality_label}, sofort"

    def describe(self) -> str:
        """Eine Zeile mit allem, was zur gewählten Stimme zu wissen ist."""
        # Der Rechenweg ist ein Vorzug -- aber nur bei einer Stimme, die
        # auch sprechen kann. Bei einer gesperrten waere er eine
        # Empfehlung fuer etwas, das nicht geht.
        teile = [] if not self.ready else ["Grafikkarte" if self.on_gpu else "Hauptprozessor"]
        if self.quality_label:
            teile.insert(0, f"Klang: {self.quality_label}")
        teile.append(self.speed_label)
        if self.size_label:
            teile.append(self.size_label)
        if self.license_id:
            teile.append(self.license_id)
        if not self.ready and self.note:
            teile.append(self.note)
        return " · ".join(teile)

    def configure(self, config: AppConfig) -> AppConfig:
        """Die Einstellungen auf diese Stimme umstellen.

        ``apply()`` reicht nicht: es setzt nur Sprecher und Profil in der
        Anfrage. WELCHE Pipeline entsteht, entscheidet dagegen die
        Konfiguration (``voice_model``, ``voice_cloning_enabled``). Ohne
        diesen Schritt wählte man am Telefon "Bark" und sprach weiter mit
        dem Modell aus den Einstellungen.
        """
        if self.is_profile:
            return config.with_values(voice_cloning_enabled=True, voice_profile=self.key)
        if self.provider == "modell" and self.model_key:
            return config.with_values(voice_cloning_enabled=False, voice_model=self.model_key)
        if self.engine == "clone":
            # Eingebaute Chatterbox-Stimme: Klon-Laufzeit ohne Profil.
            return config.with_values(voice_cloning_enabled=False, voice_profile="")
        return config

    def apply(self, request: Any) -> Any:
        """Auf eine ``VoiceRequest`` anwenden."""
        from dataclasses import replace as _replace

        if self.is_profile:
            return _replace(request, profile_slug=self.key, speaker="")
        # Bei SAPI benennt der Sprecher die Windows-Stimme, nicht ein Modell.
        return _replace(request, profile_slug="", speaker=self.speaker or self.key)


# Gemessene Richtwerte je Laufzeit (Sekunden für einen Satz mittlerer
# Länge, auf diesem Rechner). Sie stehen hier und nicht im Text, damit die
# Auswahl sie anzeigen kann, ohne dass jemand nachschlagen muss.
ENGINE_SPEED = {
    "sapi": 0.5,
    "kokoro": 1.5,
    "piper": 1.0,
    "bark": 20.0,
    "clone": 12.0,
}

# Dieselben Motoren auf der Grafikkarte. Nur die torch-gestützten haben
# dort überhaupt etwas zu gewinnen: eine Windows-Stimme kennt keine GPU,
# und Piper hängt an onnxruntime, dessen mitgeliefertes Paket gemessen nur
# CPU- und Azure-Ausführung anbietet -- keinen CUDA-Weg.
#
# Richtwerte, keine Messung auf diesem Rechner: die Stimmmodelle sind noch
# nicht geladen. Deshalb erscheinen sie in der Oberfläche mit "~".
# Sekunden je Satz auf der Grafikkarte.
#
# Gemessen, nicht geschaetzt. Fuer die Klonstimme auf einer RTX 4070 Ti:
# 4,4 bis 6,3 Sekunden bei freier Karte, rund 10 im laufenden Gespraech,
# wo Sprachmodell und Spracherkennung mitrechnen. Hier stand einmal 2,0 -
# das versprach etwas, das nie eintrat, und die Wartezeit sah nach einem
# Fehler aus statt nach dem Normalfall.
ENGINE_SPEED_GPU = {
    "bark": 3.0,
    "clone": 6.0,
    "kokoro": 0.6,
}

# Welcher Motor kann die Grafikkarte überhaupt nutzen?
GPU_ENGINES = frozenset(ENGINE_SPEED_GPU)

# Wie natürlich klingt das Ergebnis? 1 = blechern, 5 = kaum von einer
# Aufnahme zu unterscheiden. Das ist eine Eigenschaft des Verfahrens und
# hängt nicht am Rechner -- eine Windows-Stimme wird auf einer schnelleren
# Karte nicht besser, sie kommt nur früher.
#
# Ohne diese Angabe fehlte der Auswahl genau das, wonach gesucht wird:
# "sofort" stand bei der schlechtesten Stimme, und die guten sahen wegen
# ihrer Ladezeit nach der schlechteren Wahl aus.
ENGINE_QUALITY = {
    "sapi": 2,  # Formantsynthese, unüberhörbar künstlich
    "piper": 3,  # neuronal, aber tonlos in der Betonung
    "kokoro": 4,  # natürlich – spricht jedoch kein Deutsch
    "bark": 4,  # Sprachmelodie, Atmen; greift selten daneben
    "clone": 5,  # die eigene Stimme aus eigener Aufnahme
}

# Ab hier stockt ein Gespräch. Wer auf eine Antwort länger wartet als
# etwa fünf Sekunden je Satz, telefoniert nicht mehr, sondern wartet.
TELEFON_GRENZE_S = 5.0

QUALITY_WORDS = {
    1: "sehr künstlich",
    2: "künstlich",
    3: "sauber, aber tonlos",
    4: "natürlich",
    5: "sehr natürlich",
}


def engine_speed(engine: str, on_gpu: bool = False) -> float:
    """Sekunden je Satz für diesen Motor auf diesem Rechenweg."""
    if on_gpu and engine in ENGINE_SPEED_GPU:
        return ENGINE_SPEED_GPU[engine]
    return ENGINE_SPEED.get(engine, 5.0)


def voice_on_gpu(engine: str, config: Any = None) -> bool:
    """Läuft dieser Motor hier tatsächlich auf der Grafikkarte?

    Zwei Bedingungen, die beide gelten müssen: der Motor muss es können
    UND die Karte muss da sein. Nur eins von beidem zu prüfen führt zu
    der Anzeige, über die sich zu Recht beschwert wurde -- "GPU"
    dranstehen und CPU rechnen.
    """
    if engine not in GPU_ENGINES:
        return False
    try:
        # ``torch_cuda_hint`` statt ``torch_cuda_available``: letzteres
        # importiert torch, und das kostet gemessen 2,5 s -- mitten im
        # Aufbau der Telefon-Seite, also im Oberflächen-Thread. Genau
        # dadurch stand das Fenster beim Öffnen mehrere Sekunden.
        #
        # Der Hinweis liest nur torch/version.py (ein paar hundert Byte)
        # und gibt ein bereits geprüftes Ergebnis zurück, sobald eines
        # vorliegt. Für eine Anzeige ist das genau richtig.
        #
        # Liefert (ja/nein, Begründung) – nur das erste Feld zählt hier.
        # Das Tupel als Ganzes ist immer wahr; genau so entsteht die
        # Anzeige "GPU", während in Wirklichkeit die CPU rechnet.
        vorhanden, _grund = accel.torch_cuda_hint()
        return bool(vorhanden)
    except Exception as exc:  # pragma: no cover – Anzeige darf nie stören
        log.debug("Grafikkarte nicht prüfbar: %s", exc)
        return False


ENGINE_NAMES = {
    "sapi": "Windows",
    "kokoro": "Kokoro",
    "piper": "Piper",
    "bark": "Bark",
    "clone": "Klonstimme",
}


def brain_model(config: AppConfig) -> str:
    """Schlüssel des Denkmodells für das Gespräch.

    Das Telefonat darf ein anderes Modell benutzen als die Chat-Seite:
    dort zählt Bildverstehen, hier die Antwortzeit. Leer heißt "dasselbe
    wie im Chat", damit eine bestehende Einstellung weiter gilt und
    niemand zweimal dasselbe einstellen muss.
    """
    eigen = str(getattr(config, "call_chat_model", "") or "").strip()
    if eigen:
        return eigen
    return str(getattr(config, "chat_model", "") or "")


def brain_choices() -> list[tuple[str, str]]:
    """Wählbare Denkmodelle als (Schlüssel, Beschriftung).

    Beschriftet wird mit dem, was für die Wahl zählt: ob das Modell da
    ist. Ein Modell, das erst zwei Gigabyte lädt, soll nicht wie eines
    aussehen, das sofort antwortet.
    """
    fertig: list[tuple[str, str]] = []
    for spec in models.by_task(models.Task.CHAT, include_blocked=False):
        da = False
        with contextlib.suppress(Exception):
            da = models.is_downloaded(spec)
        groesse = float(getattr(spec, "approx_size_mb", 0) or 0)
        if da:
            zusatz = "bereit"
        elif groesse >= 1024:
            zusatz = f"{groesse / 1024:.1f} GB laden"
        elif groesse:
            zusatz = f"{groesse:.0f} MB laden"
        else:
            zusatz = "lädt beim ersten Mal"
        fertig.append((spec.key, f"{spec.title} — {zusatz}"))
    return fertig


def voice_catalog(config: AppConfig) -> list[VoiceChoice]:
    """Alle wählbaren Stimmen, mit Angabe was sie kosten.

    Reihenfolge: was sofort spricht, steht vorn. Wer telefonieren will,
    soll nicht als Erstes auf eine Stimme stoßen, die erst fünf Gigabyte
    lädt und dann zwanzig Sekunden je Satz braucht.
    """
    from . import models, pipeline_voice, voice_profiles

    auswahl: list[VoiceChoice] = []

    # --- Windows-Stimmen: sofort da, kein Download, keine Lizenzfrage ---
    try:
        from . import pipeline_sapi

        for stimme in pipeline_sapi.voices():
            auswahl.append(
                VoiceChoice(
                    key=f"sapi:{stimme.name}",
                    label=f"{stimme.name} ({stimme.culture})",
                    is_profile=False,
                    speaker=stimme.name,
                    is_sapi=True,
                    provider="windows",
                    engine="sapi",
                    ready=True,
                    seconds_per_sentence=engine_speed("sapi"),
                    quality=ENGINE_QUALITY["sapi"],
                    license_id="Windows-Bestandteil",
                )
            )
    except Exception as exc:
        log.debug("Windows-Stimmen nicht abfragbar: %s", exc)

    # --- Eingebaute Stimme der Klon-Laufzeit ------------------------------
    #
    # Chatterbox kann auch OHNE Referenzaufnahme sprechen; dann nimmt es
    # seine eigene, synthetische Stimme. Das ist kein Klon einer Person,
    # verlangt also keine Einwilligung -- und ist zugleich die beste
    # Stimme, die ohne Download zu haben ist.
    try:
        from . import voice_runtime

        bereit, grund = voice_runtime.available()
        auswahl.append(
            VoiceChoice(
                key="chatterbox:builtin",
                label="Chatterbox (eingebaute Stimme, deutsch)",
                is_profile=False,
                speaker="",
                is_sapi=False,
                provider="modell",
                engine="clone",
                ready=bool(bereit),
                on_gpu=voice_on_gpu("clone", config),
                seconds_per_sentence=engine_speed("clone", voice_on_gpu("clone", config)),
                quality=ENGINE_QUALITY["clone"],
                license_id="MIT",
                note="" if bereit else grund,
            )
        )
    except Exception as exc:
        log.debug("Klon-Laufzeit nicht abfragbar: %s", exc)

    # --- Angelernte Stimmen ---------------------------------------------
    #
    # Ohne Klon-Laufzeit kann kein Profil sprechen. Das einmal vorab
    # klären: sonst steht eine stumme Profilstimme wegen ihrer hohen
    # Klangnote ganz oben und wird zur Vorauswahl.
    klon_bereit = False
    klon_grund = "Klon-Laufzeit nicht eingerichtet"
    try:
        from . import voice_runtime as _vr

        klon_bereit, klon_grund = _vr.available()
    except Exception as exc:  # pragma: no cover – Anzeige darf nie stören
        log.debug("Klon-Laufzeit nicht abfragbar: %s", exc)

    try:
        for profil in voice_profiles.list_profiles():
            # Fail-closed: ohne Einwilligung ist ein Profil nicht nutzbar,
            # und das gilt am Telefon genauso wie sonst.
            nutzbar, _grund = profil.usable_for_synthesis()
            if not nutzbar:
                continue
            auswahl.append(
                VoiceChoice(
                    key=profil.slug,
                    label=f"Angelernt: {profil.display_name}",
                    is_profile=True,
                    provider="angelernt",
                    engine="clone",
                    ready=bool(klon_bereit),
                    note="" if klon_bereit else klon_grund,
                    on_gpu=voice_on_gpu("clone", config),
                    seconds_per_sentence=engine_speed("clone", voice_on_gpu("clone", config)),
                    quality=ENGINE_QUALITY["clone"],
                    license_id="Einwilligung dokumentiert",
                )
            )
    except Exception as exc:
        log.debug("Stimmprofile nicht lesbar: %s", exc)

    # --- Modellstimmen ---------------------------------------------------
    #
    # Jedes eingetragene Stimmmodell wird angeboten, aber ehrlich: was
    # nicht geladen ist, sagt das, und was eine Lizenzfrage aufwirft
    # (Piper steht unter GPL-3.0) ebenfalls.
    for spec in models.REGISTRY.values():
        # Task ist ein Enum; sein Wert ist die verlässliche Angabe.
        if getattr(getattr(spec, "task", None), "value", "") != "voice":
            continue
        motor = pipeline_voice.engine_for(spec.repo_id)
        geladen = False
        with contextlib.suppress(Exception):
            geladen = models.is_downloaded(spec)

        laufzeit_ok, laufzeit_grund = (True, "")
        try:
            laufzeit_ok, laufzeit_grund = pipeline_voice.engine_available(motor)
        except Exception as exc:
            laufzeit_ok, laufzeit_grund = False, str(exc)

        # Ein Motor ohne Umsetzung kann nicht sprechen, auch wenn seine
        # Pakete da sind. Das gehört gesagt -- sonst lädt jemand 330 MB
        # und bekommt danach die Attrappe zu hören.
        umgesetzt = motor in getattr(pipeline_voice, "IMPLEMENTED_ENGINES", frozenset())
        if not umgesetzt:
            laufzeit_ok = False
            laufzeit_grund = "in dieser Fassung noch nicht umgesetzt"

        gesperrt = ""
        try:
            models.check_allowed(spec, allow_conditional=True)
        except Exception as exc:
            gesperrt = str(exc)

        hinweis = ""
        if gesperrt:
            hinweis = "Zustimmung nötig"
        elif not laufzeit_ok:
            hinweis = laufzeit_grund
        elif not geladen:
            hinweis = "lädt beim ersten Mal"

        auf_gpu = voice_on_gpu(motor, config)
        name = ENGINE_NAMES.get(motor, motor or spec.key)
        # "Bark: Bark (große Fassung)" liest sich albern. Das Präfix nur
        # setzen, wenn der Titel die Laufzeit nicht schon nennt.
        titel = spec.title
        if name.lower() not in titel.lower():
            titel = f"{name}: {titel}"
        auswahl.append(
            VoiceChoice(
                key=spec.key,
                label=titel,
                is_profile=False,
                speaker=getattr(config, "voice_speaker", "") or "",
                is_sapi=False,
                provider="modell",
                engine=motor,
                model_key=spec.key,
                ready=bool(geladen and laufzeit_ok and not gesperrt),
                on_gpu=auf_gpu,
                seconds_per_sentence=engine_speed(motor, auf_gpu),
                quality=ENGINE_QUALITY.get(motor, 3),
                size_mb=float(getattr(spec, "approx_size_mb", 0) or 0),
                license_id=str(getattr(spec, "license_id", "")),
                note=hinweis,
            )
        )

    # Zuerst, was sprechen kann; darin die beste Stimme zuerst.
    #
    # Vorher entschied die Ladezeit, und damit stand die blecherne
    # Windows-Stimme ganz oben, während die guten unten standen. Wer eine
    # Stimme sucht, sucht nach Klang -- ein einmaliger Download ist dafür
    # ein hinnehmbarer Preis, ein dauerhaft schlechter Klang nicht.
    auswahl.sort(
        key=lambda v: (
            not v.ready,
            # Telefontauglichkeit steht ueber dem Klang. Eine Stimme, die
            # zwölf Sekunden je Satz braucht, macht ein Gespraech
            # unmoeglich -- da nuetzt der schoenste Klang nichts. Genau
            # dieser Fall tritt ein, wenn dieselbe Stimme mangels
            # Grafikkarte auf dem Hauptprozessor landet.
            v.seconds_per_sentence > TELEFON_GRENZE_S,
            -v.quality,
            v.seconds_per_sentence,
            not v.on_gpu,
            v.label,
        )
    )
    return auswahl


def voice_advice(config: AppConfig) -> str:
    """Hinweis, wenn eine deutlich bessere Stimme bereitläge.

    Beantwortet die Frage, die sonst als Urteil über die ganze Anwendung
    endet: "die Stimmen hören sich alle nicht gut an". Sie tun es, solange
    kein Stimmmodell geladen ist -- dann bleiben nur die Windows-Stimmen,
    und die sind Formantsynthese aus den Neunzigern.
    """
    katalog = voice_catalog(config)
    nutzbar = [v for v in katalog if v.ready]
    beste_jetzt = max((v.quality for v in nutzbar), default=0)

    # Was ließe sich mit einem Download erreichen? Nur zählen, was
    # danach auch wirklich spricht: ein nicht umgesetzter Motor bleibt
    # stumm, ganz gleich wie gut er klänge.
    ladbar = [
        v
        for v in katalog
        if not v.ready
        and v.quality > beste_jetzt
        and ("lädt" in (v.note or "").lower() or "laedt" in (v.note or "").lower())
    ]
    if not ladbar:
        return ""

    ziel = max(ladbar, key=lambda v: (v.quality, -v.size_mb))
    name = ziel.label.split(" (")[0].split(" – ")[0]
    wo = " und rechnet auf der Grafikkarte" if ziel.on_gpu else ""
    return (
        f"Die beste geladene Stimme klingt {QUALITY_WORDS.get(beste_jetzt, '?')}. "
        f"{name} klingt {ziel.quality_label}{wo} – "
        f"{ziel.size_label} einmalig laden."
    )


def voice_choices(config: AppConfig) -> list[VoiceChoice]:
    """Auswahl für die Oberfläche – kurze Namen.

    Die Einzelheiten holt die Oberfläche über ``describe()``, sobald eine
    Stimme gewählt ist. Behält den alten Namen, damit vorhandene Aufrufe
    weiterlaufen.
    """
    from dataclasses import replace as _replace

    fertig: list[VoiceChoice] = []
    gesehen: set[str] = set()
    for stimme in voice_catalog(config):
        kurz = stimme.short_label()
        # Zweimal dieselbe Kurzform (Hedda und Hedda Desktop) wuerde die
        # Zuordnung in der Oberflaeche zerstoeren.
        if kurz in gesehen:
            kurz = f"{kurz} ({stimme.engine or stimme.provider})"
        gesehen.add(kurz)
        fertig.append(_replace(stimme, label=kurz))
    return fertig


# ---------------------------------------------------------------------------
# Antwort aufteilen: was gesprochen wird, was Datei wird
# ---------------------------------------------------------------------------
def split_answer(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Antwort in Sprechtext und Code-Blöcke trennen.

    Rückgabe: (was vorgelesen wird, [(Sprache, Code), …]).

    Quelltext wird **nicht** vorgelesen: eine Funktion mit zwanzig Zeilen
    dauert gesprochen über eine Minute und ist danach niemandem im
    Gedächtnis. Sie gehört in eine Datei.
    """
    bloecke: list[tuple[str, str]] = []
    rest: list[str] = []
    position = 0
    for treffer in _FENCE.finditer(text):
        rest.append(text[position : treffer.start()])
        bloecke.append((treffer.group(1).strip().lower(), treffer.group(2)))
        position = treffer.end()
    rest.append(text[position:])

    gesprochen = " ".join(" ".join(rest).split())
    # Restliche Auszeichnung entfernen – gesprochene Sternchen sind Unsinn.
    gesprochen = re.sub(r"[*_`#]+", "", gesprochen).strip()
    return gesprochen, bloecke


def _slug(text: str, laenge: int = 32) -> str:
    roh = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text.lower())
    return "-".join(teil for teil in roh.split("-") if teil)[:laenge] or "antwort"


def write_artifacts(
    ordner: Path, frage: str, bloecke: list[tuple[str, str]], nummer: int
) -> list[Path]:
    """Code-Blöcke als Dateien ablegen. Rückgabe: geschriebene Pfade."""
    geschrieben: list[Path] = []
    paths.ensure_dir(ordner)
    for index, (sprache, code) in enumerate(bloecke, start=1):
        endung = CODE_SUFFIX.get(sprache, ".txt")
        name = f"{nummer:02d}-{_slug(frage)}"
        if len(bloecke) > 1:
            name += f"-{index}"
        ziel = ordner / f"{name}{endung}"
        zaehler = 2
        while ziel.exists():
            ziel = ordner / f"{name}-{zaehler}{endung}"
            zaehler += 1
        try:
            ziel.write_text(code.rstrip("\n") + "\n", encoding="utf-8")
        except OSError as exc:
            log.warning("Datei nicht schreibbar: %s", clean_error(exc))
            continue
        geschrieben.append(ziel)
    return geschrieben


# ---------------------------------------------------------------------------
# Das Gespräch
# ---------------------------------------------------------------------------
class CallSession:
    """Ein laufendes Telefonat.

    Hält Spracherkennung, Sprachmodell und Sprachausgabe geladen. Alles
    drei neu zu laden würde je Zug Sekunden kosten.
    """

    def __init__(
        self,
        config: AppConfig,
        plan: Any,
        voice: VoiceChoice | None = None,
        chat_spec: models.ModelSpec | None = None,
        persona_key: str = "",
    ) -> None:
        self.config = config
        self.plan = plan
        self.voice = voice
        self.turns: list[CallTurn] = []
        self.started_at = time.time()
        self.folder = paths.ensure_dir(
            config.resolved_output_dir() / "telefonate" / time.strftime("%Y%m%d-%H%M%S")
        )
        self._stt = pipeline_stt.SpeechToText(config)
        self._chat = pipeline_chat.ChatSession(
            config, chat_spec or models.resolve(brain_model(config))
        )
        self._voice_pipeline: Any = None
        # Der Weg, ueber den gehoert und gesprochen wird: eigenes Mikrofon
        # oder ein Bot in einem Discord-Sprachkanal. Siehe call_transport.
        self._transport = call_transport.create_transport(config)
        self.persona_key = persona_key
        self._speech = None  # laufende Sprach-Warteschlange
        # Rückfall auf eine Windows-Stimme, falls die gewählte nichts
        # liefert. Wird erst gebaut, wenn er gebraucht wird, und die
        # Meldung dazu kommt nur einmal je Gespräch.
        self._notfall_pipeline: Any = None
        self._notfall_gemeldet = False
        self.notes: list[str] = []

    # --- Aufbau -------------------------------------------------------
    def open(self, context) -> None:
        """Alle Teile bereitstellen. Fehlt eines, wird es gesagt."""
        stand = readiness(self.config)
        if not stand.ready:
            raise CallUnavailable("Telefonieren ist nicht möglich:\n" + "\n".join(stand.problems()))

        # Erst der Weg: ohne Mikrofon oder ohne Bot braucht das Laden der
        # Modelle gar nicht erst anzufangen.
        self._transport.open(context)

        # ACHTUNG, die Reihenfolge ist nicht beliebig: das Sprachmodell
        # MUSS vor der Spracherkennung auf die Karte.
        #
        # Gemessen, zweimal derselbe Code, nur getauscht:
        #
        #     llama.cpp -> Whisper -> erkennen      geht
        #     Whisper -> llama.cpp -> erkennen      Prozess stirbt
        #
        # llama.cpp richtet sich beim Laden seinen eigenen CUDA-Kontext
        # ein. CTranslate2 - die Maschine hinter faster-whisper - legt
        # seine cuBLAS-Handles dagegen schon beim Laden an. Kommt
        # llama.cpp danach, zeigen sie ins Leere:
        #
        #     RuntimeError: CUDA failed with error invalid resource handle
        #
        # und mit etwas Pech stirbt der Prozess ohne Traceback. Das war
        # das Bild beim Anrufen: "das Modell laedt und dann Absturz".
        context.status("Sprachmodell wird geladen …")
        # Sparsam laden: kein Bildteil, kurzer Kontext.
        #
        # Am Telefon teilen sich DREI Dinge die Karte - Sprachmodell,
        # Spracherkennung und Klonstimme. Gemessen mit Bildverstehen:
        #
        #     Modell geladen        3191 MiB
        #     nach einer Antwort    9385 MiB   (+6194)
        #
        # ohne Bildteil und mit kurzem Kontext:
        #
        #     Modell geladen        2967 MiB
        #     nach einer Antwort    3015 MiB   (+48)
        #
        # 6,4 GB Unterschied. Vorher war die Karte zu 96,5 % belegt bei
        # 3-39 % Auslastung - sie lagerte aus, statt zu rechnen, und ein
        # gesprochener Satz dauerte 14 bis 32 Sekunden statt 5 bis 6.
        #
        # Gezeigt wird am Telefon ohnehin nichts.
        self._chat.call_mode = True
        self._chat.load(context)
        # Persona setzt den Ton; am Telefon zusaetzlich kurz und gesprochen.
        if self.persona_key:
            self._chat.set_persona(self.persona_key, for_call=True)
        else:
            self._chat.system_prompt = CALL_SYSTEM_PROMPT

        context.status("Spracherkennung wird geladen …")
        try:
            self._stt.load(context, self.plan)
        except BaseException as exc:
            # Ein kaputter CUDA-Kontext darf kein Gespraech verhindern.
            #
            # Beobachtet: "CUDA failed with error context is destroyed" -
            # der vorherige Anruf hatte den Kontext beim Absturz kaputt
            # hinterlassen, und jeder weitere Versuch scheiterte gleich.
            # Auf der CPU ist die Erkennung langsamer, aber sie laeuft.
            if "cuda" not in clean_error(exc).lower():
                raise
            context.status(f"Grafikkarte nicht nutzbar ({clean_error(exc)}) – Erkennung auf CPU.")
            log.warning("STT auf CPU ausgewichen: %s", clean_error(exc))
            self._cuda_aufraeumen()
            nur_cpu = accel.BackendPlan(backend=accel.Backend.CPU)
            self._stt.load(context, nur_cpu)

        context.status("Sprachausgabe wird vorbereitet …")
        # Die Auswahl auf die Konfiguration übertragen, BEVOR die Pipeline
        # daraus gebaut wird -- sonst ist die Wahl wirkungslos.
        if self.voice is not None:
            self.config = self.voice.configure(self.config)
        if self.voice is not None and self.voice.is_sapi:
            from . import pipeline_sapi

            self._voice_pipeline = pipeline_sapi.build_pipeline(self.config, self.plan)
        elif self.voice is not None and self.voice.engine == "clone":
            # Klon-Laufzeit, mit oder ohne Profil. Ohne Profil spricht
            # Chatterbox mit seiner eingebauten Stimme; die Prüfung auf
            # Einwilligung greift nur, wenn wirklich eine reale Stimme
            # nachgebildet wird (siehe pipeline_voice).
            from . import pipeline_voice

            self._voice_pipeline = pipeline_voice.ChatterboxVoicePipeline(self.config, self.plan)
        else:
            from . import pipeline_voice

            self._voice_pipeline = pipeline_voice.create_voice_pipeline(self.config, self.plan)
        # Ausdrücklich sagen, was worauf rechnet. Ein Gespräch, das auf
        # der CPU läuft, obwohl eine Karte im Rechner steckt, merkt man
        # sonst nur an der Wartezeit – und sucht die Ursache im falschen
        # Teil.
        # Stimmmodell JETZT laden, nicht beim ersten Satz.
        #
        # Gemessen: das Laden kostet je nach Zustand 25 bis 140 Sekunden.
        # Faellt es in die erste Antwort, steht das Gespraech genau dann,
        # wenn der Anrufer eine Antwort erwartet. Hier stoert es nicht --
        # hier wird ohnehin verbunden.
        if self.voice is not None and self.voice.engine == "clone":
            from . import pipeline_voice

            try:
                pipeline_voice.warmup_voice(self.config, self.plan, context)
            except Exception as exc:
                # Kein Grund abzubrechen: der erste Satz laedt dann eben
                # selbst. Nur sagen, damit die Wartezeit erklaerbar ist.
                context.status(f"Stimme wird beim ersten Satz geladen ({clean_error(exc)}).")

        context.status("Verbunden. Sprich einfach los.")
        context.status(self.hardware_summary())

    def hardware_summary(self) -> str:
        """Eine Zeile: welcher Teil rechnet wo.

        Die drei Teile eines Gesprächs laufen unabhängig voneinander auf
        CPU oder GPU. Ohne diese Zeile sieht man nur, dass es langsam
        ist, aber nicht welcher Teil bremst.
        """
        teile: list[str] = []

        # Der Weg zuerst: er entscheidet, wessen Stimme überhaupt ankommt.
        # Beim eigenen Mikrofon ist das der Normalfall und braucht keine
        # Erwähnung; beim Bot ist es die wichtigste Angabe der Zeile.
        if str(getattr(self.config, "call_mode", "lokal") or "lokal") == "discord":
            teile.append("Ton: Discord-Kanal")

        geraet = getattr(self._stt, "device", "") or "?"
        teile.append(f"Verstehen: {geraet.upper()}")

        # Welches Modell denkt, gehört dazu: seit das Telefonat ein
        # eigenes haben kann, ist "GPU" allein keine Auskunft mehr.
        try:
            from . import pipeline_chat

            wo = "GPU" if pipeline_chat.gpu_offload_possible() else "CPU"
        except Exception:
            wo = "?"
        name = brain_model(self.config) or "?"
        with contextlib.suppress(Exception):
            name = models.resolve(name).title
        # Am Telefon wird der Bildteil nicht geladen (er belegt 6,4 GB,
        # die Erkennung und Stimme brauchen). Der Zusatz im Modellnamen
        # waere hier also eine Zusage, die nicht gilt.
        for zusatz in (" (sieht Bilder)", " (sieht Bilder, klein)"):
            if name.endswith(zusatz):
                name = name[: -len(zusatz)]
                break
        teile.append(f"Denken: {name} ({wo})")

        motor = getattr(self.voice, "engine", "") if self.voice else ""
        if motor == "sapi":
            teile.append("Sprechen: Windows-Stimme (CPU, ~0,5 s/Satz)")
        elif motor == "piper":
            # Piper haengt an onnxruntime; das mitgelieferte Paket kennt
            # gemessen keinen CUDA-Weg. "GPU" waere hier eine Falschaussage.
            teile.append("Sprechen: Piper (CPU, ~1 s/Satz)")
        elif motor:
            gpu = accel.speech_backend(self.plan) if self.plan is not None else ""
            auf_gpu = gpu == accel.Backend.CUDA
            name = ENGINE_NAMES.get(motor, motor)
            tempo = engine_speed(motor, auf_gpu)
            teile.append(f"Sprechen: {name} ({'GPU' if auf_gpu else 'CPU'}, ~{tempo:.0f} s/Satz)")

        return " · ".join(teile)

    # --- Ein Zug ------------------------------------------------------
    def discard_recording(self, wav: Path | None) -> None:
        """Eine Aufnahme wegwerfen, wenn sie nicht behalten werden soll.

        Der Haken "Aufnahmen der Kanalstimmen behalten" stand auf AUS und
        trug den Hinweis, fremde Stimmen aufzuzeichnen brauche deren
        Einverständnis -- gelesen wurde er nirgends. Die Stimmen aller
        Kanalteilnehmer blieben also dauerhaft auf der Platte liegen.

        Das ist keine Kleinigkeit: es ist genau die Zusage, auf die sich
        der Betreiber gegenüber den Beteiligten beruft (§ 201 StGB,
        DSGVO). Ein Haken, der eine Zusage macht und nichts tut, ist
        schlimmer als gar keiner.
        """
        if wav is None:
            return
        # Nur im Discord-Weg geht es um fremde Stimmen. Am eigenen
        # Mikrofon spricht der Bediener selbst.
        if str(getattr(self.config, "call_mode", "lokal") or "lokal") != "discord":
            return
        if bool(getattr(self.config, "discord_keep_audio", False)):
            return
        try:
            Path(wav).unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Aufnahme nicht löschbar: %s", clean_error(exc))

    def listen(
        self,
        on_level: Callable[[float, bool], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_threshold: Callable[[float], None] | None = None,
    ) -> tuple[Path | None, float]:
        """Zuhören, bis der Gesprächspartner fertig ist."""
        ziel = self.folder / f"frage-{len(self.turns) + 1:02d}.wav"
        return self._transport.listen(
            ziel,
            on_level=on_level,
            should_stop=should_stop,
            on_threshold=on_threshold,
        )

    def answer(
        self,
        aufnahme: Path,
        context,
        on_token: Callable[[str], None] | None = None,
        speak: bool = True,
    ) -> CallTurn:
        """Aufnahme verstehen, antworten, Antwort sprechen und ablegen."""
        zug = CallTurn(aufnahme=aufnahme)

        # 1. Verstehen
        mitschrift = self._stt.transcribe(aufnahme, language=self.config.language)
        zug.frage = mitschrift.text
        zug.stt_sekunden = mitschrift.elapsed_s
        # Verstanden ist verstanden: die Aufnahme fremder Stimmen wird
        # jetzt weggeworfen, sofern sie nicht ausdruecklich behalten
        # werden soll. Vorher blieb sie IMMER liegen - entgegen dem, was
        # der Haken im Discord-Dialog zusagt.
        self.discard_recording(aufnahme)
        if str(getattr(self.config, "call_mode", "lokal") or "lokal") == "discord" and not bool(
            getattr(self.config, "discord_keep_audio", False)
        ):
            # Auch aus der Mitschrift nehmen: ein Verweis auf eine Datei,
            # die es nicht mehr gibt, ist schlimmer als kein Verweis.
            zug.aufnahme = None
        if mitschrift.empty:
            zug.antwort = ""
            self.turns.append(zug)
            return zug

        # 2. Antworten – und dabei schon sprechen
        #
        # Der Satzteiler laeuft MIT dem Token-Strom: sobald der erste Satz
        # steht, geht er an die Sprachausgabe, waehrend das Modell weiter
        # schreibt. Ohne das entstuende erst die ganze Antwort, dann die
        # ganze Sprachausgabe - bei vier Saetzen zehn Sekunden Stille.
        from .speech_stream import SentenceSplitter, SpeechQueue

        begonnen = time.time()
        teiler = SentenceSplitter()
        nummer = len(self.turns) + 1
        sprecher: SpeechQueue | None = None
        erster_ton: list[float] = []

        if speak:
            sprecher = SpeechQueue(
                synth=lambda satz, index: self._synth_sentence(satz, nummer, index, context),
                play=lambda wav: self._play(wav, erster_ton),
                on_error=lambda text: context.status(f"Antwort nur schriftlich: {text}"),
            )
            sprecher.start()
            self._speech = sprecher

        def weiterreichen(stueck: str) -> None:
            if on_token is not None:
                on_token(stueck)
            if sprecher is not None and not sprecher.stopped:
                for satz in teiler.feed(stueck):
                    sprecher.say(satz)

        antwort = self._chat.ask(
            zug.frage,
            on_token=weiterreichen,
            should_stop=context.should_stop,
            temperature=self.config.chat_temperature,
            max_tokens=self.config.chat_max_tokens,
        )
        zug.antwort = antwort.text
        zug.denk_sekunden = time.time() - begonnen

        if sprecher is not None and not sprecher.stopped:
            for satz in teiler.finish():
                sprecher.say(satz)

        # 3. Trennen: Sprechtext und Dateien
        gesprochen, bloecke = split_answer(zug.antwort)
        zug.gesprochen = gesprochen
        if bloecke:
            zug.dateien = tuple(write_artifacts(self.folder, zug.frage, bloecke, nummer))

        # 4. Sprachausgabe zu Ende bringen
        if sprecher is not None:
            sprecher.wait()
            if sprecher.gesprochen:
                zug.stimme = sprecher.gesprochen[0]
            zug.tts_sekunden = time.time() - begonnen
            zug.erster_ton_s = erster_ton[0] - begonnen if erster_ton else 0.0
            self._speech = None

        self.turns.append(zug)
        self.save_transcript()
        return zug

    def _synth_sentence(self, satz: str, zug: int, index: int, context) -> Path | None:
        """Einen einzelnen Satz zu einer WAV-Datei machen."""
        from . import pipeline_voice

        anfrage = pipeline_voice.VoiceRequest.from_config(
            self.config,
            satz,
            output_dir=self.folder / "stimme",
            name_hint=f"a{zug:02d}-{index:02d}",
            split_sentences=False,  # ist bereits ein einzelner Satz
        )
        if self.voice is not None:
            anfrage = self.voice.apply(anfrage)
        try:
            ergebnis = self._voice_pipeline.synthesize(anfrage, context)
        except Exception as exc:
            return self._notfall_stimme(satz, anfrage, exc, context)
        # Ein Platzhalterton ist KEIN Erfolg.
        #
        # Die Attrappe liefert eine gültige WAV-Datei mit einer
        # Tonfolge – für diese Prüfung sah das aus wie eine gelungene
        # Sprachausgabe, und der Anrufer hörte statt einer Stimme ein
        # Rauschen. Am Telefon ist das der schlechteste aller Ausgänge:
        # es klingt nach Defekt und nennt keinen Grund.
        if getattr(ergebnis, "dummy", False):
            return self._notfall_stimme(
                satz,
                anfrage,
                RuntimeError(
                    "; ".join(ergebnis.notes) if ergebnis.notes else "Stimmmodell nicht verfügbar"
                ),
                context,
            )
        if ergebnis.audio and Path(ergebnis.audio).is_file():
            return Path(ergebnis.audio)
        # Kein Ton und kein Fehler: die Attrappen-Pipeline meldet so, dass
        # ihr das Modell fehlt.
        return self._notfall_stimme(satz, anfrage, None, context)

    def _notfall_stimme(self, satz: str, anfrage, fehler, context) -> Path | None:
        """Auf eine Windows-Stimme ausweichen, wenn sonst nichts kommt.

        Stille ist am Telefon der schlimmste Ausgang: der Anrufer weiß
        nicht, ob die Gegenseite nachdenkt, hängt oder weg ist. Lieber
        eine andere Stimme als keine – aber einmal ausgesprochen, damit
        niemand rätselt, warum es plötzlich anders klingt.
        """
        from . import pipeline_sapi

        if self._notfall_pipeline is None:
            ok, grund = pipeline_sapi.available()
            if not ok:
                if not self._notfall_gemeldet:
                    self._notfall_gemeldet = True
                    context.status(f"Sprachausgabe nicht möglich: {grund}")
                return None
            self._notfall_pipeline = pipeline_sapi.build_pipeline(self.config, self.plan)

        if not self._notfall_gemeldet:
            self._notfall_gemeldet = True
            stimme = pipeline_sapi.best_voice(self.config.language)
            grund = clean_error(fehler) if fehler is not None else "kein Ton erzeugt"
            hinweis = (
                f"Gewählte Stimme liefert nichts ({grund}). "
                f"Es spricht jetzt die Windows-Stimme "
                f"'{stimme.name if stimme else 'Systemvorgabe'}'."
            )
            log.warning("%s", hinweis)
            context.status(hinweis)
            self.notes.append(hinweis)

        from dataclasses import replace as _replace

        # Der Sprechername der Modelle ("v2/de_speaker_3") sagt einer
        # Windows-Stimme nichts – leer lassen, dann wählt SAPI passend.
        ersatz = _replace(anfrage, profile_slug="", speaker="")
        try:
            ergebnis = self._notfall_pipeline.synthesize(ersatz, context)
        except Exception as exc:
            log.warning("Auch die Windows-Stimme scheitert: %s", clean_error(exc))
            return None
        if ergebnis.audio and Path(ergebnis.audio).is_file():
            return Path(ergebnis.audio)
        return None

    def _play(self, wav: Path, erster_ton: list[float]) -> None:
        """Eine Satz-Datei abspielen und den ersten Ton stempeln."""
        if not erster_ton:
            erster_ton.append(time.time())
        self._transport.play(wav)

    @staticmethod
    def _geraet(wert) -> int | None:
        """-1 (oder leer) heisst Systemvorgabe, sonst die Geraetenummer."""
        try:
            nummer = int(wert)
        except (TypeError, ValueError):
            return None
        return None if nummer < 0 else nummer

    def _speak_ganz(self, text: str, context) -> Path | None:
        """Text sprechen und abspielen. Rückgabe: WAV-Datei."""
        from . import pipeline_voice

        anfrage = pipeline_voice.VoiceRequest.from_config(
            self.config,
            text,
            output_dir=self.folder,
            name_hint=f"antwort-{len(self.turns) + 1:02d}",
        )
        if self.voice is not None:
            anfrage = self.voice.apply(anfrage)
        ergebnis = self._voice_pipeline.synthesize(anfrage, context)
        # Auch hier gilt: ein Platzhalterton wird nicht abgespielt. Er
        # klingt nach Defekt und nennt keinen Grund.
        if getattr(ergebnis, "dummy", False):
            ersatz = self._notfall_stimme(
                text,
                anfrage,
                RuntimeError(
                    "; ".join(ergebnis.notes) if ergebnis.notes else "Stimmmodell nicht verfügbar"
                ),
                context,
            )
            if ersatz is not None:
                self._transport.play(ersatz)
            return ersatz
        if ergebnis.audio and Path(ergebnis.audio).is_file():
            self._transport.play(Path(ergebnis.audio))
            return Path(ergebnis.audio)
        return None

    def interrupt(self) -> None:
        """Die KI unterbrechen – wie am Telefon dazwischenreden.

        Stoppt beides: den laufenden Satz UND die noch wartenden. Ohne das
        Leeren der Warteschlange spraeche sie die restlichen Saetze weiter.
        """
        self._transport.stop_playback()
        if self._speech is not None:
            self._speech.stop()

    # --- Mitschrift ---------------------------------------------------
    def transcript(self) -> str:
        """Gespräch als Markdown."""
        zeilen = [
            f"# Telefonat vom {time.strftime('%d.%m.%Y %H:%M', time.localtime(self.started_at))}",
            "",
            f"Stimme: {self.voice.label if self.voice else 'Vorgabe'}  ",
            f"Sprachmodell: {self._chat.spec.title}  ",
            f"Spracherkennung: {self._stt.spec.title} ({self._stt.device})",
            "",
        ]
        for nummer, zug in enumerate(self.turns, start=1):
            zeilen.append(f"## {nummer}. Du")
            zeilen.append(zug.frage or "_nichts verstanden_")
            zeilen.append("")
            zeilen.append(f"## {nummer}. Assistent")
            zeilen.append(zug.antwort or "_keine Antwort_")
            if zug.dateien:
                zeilen.append("")
                zeilen.append("Dateien aus diesem Zug:")
                zeilen.extend(f"- `{p.name}`" for p in zug.dateien)
            zeilen.append("")
        return "\n".join(zeilen)

    def save_transcript(self) -> Path:
        """Mitschrift nach jedem Zug schreiben – nicht erst am Ende.

        Ein Gespräch kann abbrechen; die bisherigen Antworten sollen dann
        trotzdem auf der Platte liegen.
        """
        ziel = self.folder / "mitschrift.md"
        try:
            ziel.write_text(self.transcript(), encoding="utf-8")
        except OSError as exc:
            log.warning("Mitschrift nicht schreibbar: %s", clean_error(exc))
        return ziel

    def artifacts(self) -> list[Path]:
        """Alle Dateien aus dem Gespräch."""
        gesammelt: list[Path] = []
        for zug in self.turns:
            gesammelt.extend(zug.dateien)
        return gesammelt

    def close(self) -> None:
        """Auflegen. Jeder Schritt einzeln – nichts reißt den Rest mit.

        Vorher lief das ungekapselt, und ein CUDA-Fehler beim Entladen
        beendete die ganze Anwendung:

            RuntimeError: CUDA failed with error context is destroyed

        Spracherkennung, Sprachmodell und Stimme halten je einen eigenen
        CUDA-Kontext. Gibt einer seinen frei, während ein anderer noch
        daran hängt, kracht es. Und weil danach nichts mehr aufgeräumt
        wurde, scheiterte auch der nächste Anruf.
        """

        def schritt(was: str, tu) -> None:
            try:
                tu()
            except BaseException as exc:  # auch CUDA-Abbrüche
                log.warning("%s beim Auflegen: %s", was, clean_error(exc))

        schritt("Wiedergabe stoppen", self.interrupt)
        schritt("Mitschrift sichern", self.save_transcript)

        # Den Weg zuerst schliessen: bei Discord meldet sich damit der Bot
        # aus dem Kanal ab. Bliebe er sitzen, sitzt er auch nach dem
        # Auflegen noch da und hoert weiter mit.
        schritt("Verbindung schliessen", self._transport.close)

        # Dann die Unterprozesse. Sie halten ihren CUDA-Kontext für sich;
        # sie zuerst zu beenden nimmt Druck von den beiden Nutzern im
        # eigenen Prozess.
        def stimme_aus() -> None:
            from . import pipeline_voice

            pipeline_voice.shutdown_voice_servers()

        schritt("Stimm-Arbeiter beenden", stimme_aus)
        self._voice_pipeline = None

        # Zuletzt die beiden CUDA-Nutzer im eigenen Prozess, einzeln und
        # mit Aufräumen dazwischen.
        schritt("Spracherkennung entladen", self._stt.unload)
        schritt("Speicher freigeben", self._cuda_aufraeumen)
        schritt("Sprachmodell entladen", self._chat.unload)
        schritt("Speicher freigeben", self._cuda_aufraeumen)

    @staticmethod
    def _cuda_aufraeumen() -> None:
        """Belegten Grafikspeicher freigeben, falls torch geladen ist.

        Nur wenn torch ohnehin schon im Speicher liegt – ihn dafür zu
        importieren würde beim Auflegen Sekunden kosten.
        """
        import sys

        torch = sys.modules.get("torch")
        if torch is None:
            return
        with contextlib.suppress(Exception):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def describe(config: AppConfig | None = None) -> str:
    """Zustandsbericht für Diagnose und Oberfläche.

    Mit config wird auch geprüft, ob die gewählten Modelle wirklich
    auf der Platte liegen – ohne sie bliebe es bei der Paketprüfung, und
    der Bericht meldete „bereit" für ein Gespräch, das erst Gigabyte
    nachladen müsste.
    """
    stand = readiness(config)
    zeilen = ["== Telefonieren ==", ""]
    zeilen.append(stand.report())
    zeilen.append("")
    zeilen.append("Bereit." if stand.ready else "Nicht möglich – siehe oben.")
    return "\n".join(zeilen)
