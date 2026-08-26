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

from . import audio_io, call_transport, models, paths, pipeline_chat, pipeline_stt
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

    @property
    def ready(self) -> bool:
        return self.audio[0] and self.stt[0] and self.chat[0]

    def problems(self) -> list[str]:
        return [grund for ok, grund in (self.audio, self.stt, self.chat) if not ok]

    def report(self) -> str:
        zeilen = []
        for name, (ok, grund) in (
            ("Mikrofon/Ton", self.audio),
            ("Spracherkennung", self.stt),
            ("Sprachmodell", self.chat),
        ):
            zeilen.append(f"  {name:<16} {'ok  ' if ok else 'FEHLT'}  {grund}")
        return "\n".join(zeilen)


def readiness() -> CallReadiness:
    """Prüfen, ob ein Gespräch möglich ist – ohne etwas zu laden."""
    return CallReadiness(
        audio=audio_io.available(),
        stt=pipeline_stt.runtime_available(),
        chat=pipeline_chat.runtime_available(),
    )


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

    @property
    def speed_label(self) -> str:
        """Kurzform der Geschwindigkeit für die Auswahlliste."""
        if self.seconds_per_sentence <= 1.0:
            return "sofort"
        if self.seconds_per_sentence <= 5.0:
            return f"~{self.seconds_per_sentence:.0f} s/Satz"
        return f"langsam, ~{self.seconds_per_sentence:.0f} s/Satz"

    def describe(self) -> str:
        """Mehrzeilige Auskunft für die Oberfläche."""
        teile = [self.speed_label]
        if self.size_mb:
            teile.append(
                f"{self.size_mb / 1024:.1f} GB"
                if self.size_mb >= 1024
                else f"{self.size_mb:.0f} MB"
            )
        if self.license_id:
            teile.append(self.license_id)
        if not self.ready and self.note:
            teile.append(self.note)
        return " · ".join(teile)

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

ENGINE_NAMES = {
    "sapi": "Windows",
    "kokoro": "Kokoro",
    "piper": "Piper",
    "bark": "Bark",
    "clone": "Klonstimme",
}


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
                    seconds_per_sentence=ENGINE_SPEED["sapi"],
                    license_id="Windows-Bestandteil",
                )
            )
    except Exception as exc:
        log.debug("Windows-Stimmen nicht abfragbar: %s", exc)

    # --- Angelernte Stimmen ---------------------------------------------
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
                    ready=True,
                    seconds_per_sentence=ENGINE_SPEED["clone"],
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

        name = ENGINE_NAMES.get(motor, motor or spec.key)
        auswahl.append(
            VoiceChoice(
                key=spec.key,
                label=f"{name}: {spec.title}",
                is_profile=False,
                speaker=getattr(config, "voice_speaker", "") or "",
                is_sapi=False,
                provider="modell",
                engine=motor,
                model_key=spec.key,
                ready=bool(geladen and laufzeit_ok and not gesperrt),
                seconds_per_sentence=ENGINE_SPEED.get(motor, 5.0),
                size_mb=float(getattr(spec, "approx_size_mb", 0) or 0),
                license_id=str(getattr(spec, "license_id", "")),
                note=hinweis,
            )
        )

    # Sofort brauchbare zuerst, danach nach Geschwindigkeit.
    auswahl.sort(key=lambda v: (not v.ready, v.seconds_per_sentence, v.label))
    return auswahl


def voice_choices(config: AppConfig) -> list[VoiceChoice]:
    """Auswahl für die Oberfläche – mit Kosten in der Beschriftung.

    Behält den alten Namen, damit vorhandene Aufrufe weiterlaufen.
    """
    fertig: list[VoiceChoice] = []
    for stimme in voice_catalog(config):
        from dataclasses import replace as _replace

        beschriftung = f"{stimme.label} – {stimme.describe()}"
        fertig.append(_replace(stimme, label=beschriftung))
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
            config, chat_spec or models.resolve(config.chat_model)
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
        """Alle drei Teile bereitstellen. Fehlt eines, wird es gesagt."""
        stand = readiness()
        if not stand.ready:
            raise CallUnavailable("Telefonieren ist nicht möglich:\n" + "\n".join(stand.problems()))

        # Erst der Weg: ohne Mikrofon oder ohne Bot braucht das Laden der
        # Modelle gar nicht erst anzufangen.
        self._transport.open(context)

        context.status("Spracherkennung wird geladen …")
        self._stt.load(context, self.plan)

        context.status("Sprachmodell wird geladen …")
        self._chat.load(context)
        # Persona setzt den Ton; am Telefon zusaetzlich kurz und gesprochen.
        if self.persona_key:
            self._chat.set_persona(self.persona_key, for_call=True)
        else:
            self._chat.system_prompt = CALL_SYSTEM_PROMPT

        context.status("Sprachausgabe wird vorbereitet …")
        if self.voice is not None and self.voice.is_sapi:
            from . import pipeline_sapi

            self._voice_pipeline = pipeline_sapi.build_pipeline(self.config, self.plan)
        else:
            from . import pipeline_voice

            self._voice_pipeline = pipeline_voice.create_voice_pipeline(self.config, self.plan)
        context.status("Verbunden. Sprich einfach los.")

    # --- Ein Zug ------------------------------------------------------
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
        self.interrupt()
        self.save_transcript()
        # Den Weg zuerst schliessen: bei Discord meldet sich damit der Bot
        # aus dem Kanal ab. Bliebe er sitzen, sitzt er auch nach dem
        # Auflegen noch da und hoert weiter mit.
        try:
            self._transport.close()
        except Exception as exc:
            log.warning("Verbindung nicht sauber geschlossen: %s", clean_error(exc))
        self._stt.unload()
        self._chat.unload()
        self._voice_pipeline = None


def describe() -> str:
    """Zustandsbericht für Diagnose und Oberfläche."""
    stand = readiness()
    zeilen = ["== Telefonieren ==", ""]
    zeilen.append(stand.report())
    zeilen.append("")
    zeilen.append("Bereit." if stand.ready else "Nicht möglich – siehe oben.")
    return "\n".join(zeilen)
