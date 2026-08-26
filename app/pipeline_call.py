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

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import audio_io, models, paths, pipeline_chat, pipeline_stt
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

    def apply(self, request: Any) -> Any:
        """Auf eine ``VoiceRequest`` anwenden."""
        from dataclasses import replace as _replace

        if self.is_profile:
            return _replace(request, profile_slug=self.key, speaker="")
        # Bei SAPI benennt der Sprecher die Windows-Stimme, nicht ein Modell.
        return _replace(request, profile_slug="", speaker=self.speaker or self.key)


def voice_choices(config: AppConfig) -> list[VoiceChoice]:
    """Alle Stimmen: erst die angelernten, dann die mitgelieferten.

    Angelernte zuerst, weil wer eine eigene Stimme angelernt hat, sie
    auch benutzen will – und weil ihre Zahl klein ist.
    """
    from . import voice_profiles

    auswahl: list[VoiceChoice] = []

    # Windows-Stimmen zuerst: sie antworten in Bruchteilen einer Sekunde.
    # Ein Modell wie Bark klingt natuerlicher, braucht aber rund zwanzig
    # Sekunden je Satz - am Telefon unbrauchbar.
    try:
        from . import pipeline_sapi

        for stimme in pipeline_sapi.voices():
            auswahl.append(
                VoiceChoice(
                    key=f"sapi:{stimme.name}",
                    label=f"{stimme.label()} – schnell",
                    is_profile=False,
                    speaker=stimme.name,
                    is_sapi=True,
                )
            )
    except Exception as exc:
        log.debug("Windows-Stimmen nicht abfragbar: %s", exc)

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
                )
            )
    except Exception as exc:
        log.debug("Stimmprofile nicht lesbar: %s", exc)

    # Mitgelieferte Sprecher des gewählten Stimmmodells.
    vorgabe = getattr(config, "voice_speaker", "") or "default"
    auswahl.append(
        VoiceChoice(key=vorgabe, label=f"Vorgabe ({vorgabe})", is_profile=False, speaker=vorgabe)
    )
    for sprecher in ("v2/de_speaker_3", "v2/de_speaker_6", "v2/de_speaker_9"):
        if sprecher != vorgabe:
            auswahl.append(
                VoiceChoice(
                    key=sprecher,
                    label=f"Stimme {sprecher.rsplit('_', 1)[-1]}",
                    is_profile=False,
                    speaker=sprecher,
                )
            )
    return auswahl


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
        self._playback = audio_io.Playback()
        self.persona_key = persona_key
        self._speech = None  # laufende Sprach-Warteschlange

    # --- Aufbau -------------------------------------------------------
    def open(self, context) -> None:
        """Alle drei Teile bereitstellen. Fehlt eines, wird es gesagt."""
        stand = readiness()
        if not stand.ready:
            raise CallUnavailable("Telefonieren ist nicht möglich:\n" + "\n".join(stand.problems()))

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
        return audio_io.record_turn(
            ziel,
            on_level=on_level,
            should_stop=should_stop,
            on_threshold=on_threshold,
            device=self._geraet(getattr(self.config, "call_input_device", -1)),
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
        ergebnis = self._voice_pipeline.synthesize(anfrage, context)
        if ergebnis.audio and Path(ergebnis.audio).is_file():
            return Path(ergebnis.audio)
        return None

    def _play(self, wav: Path, erster_ton: list[float]) -> None:
        """Eine Satz-Datei abspielen und den ersten Ton stempeln."""
        if not erster_ton:
            erster_ton.append(time.time())
        self._playback.play(
            wav, device=self._geraet(getattr(self.config, "call_output_device", -1))
        )

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
            self._playback.play(
                Path(ergebnis.audio),
                device=getattr(self.config, "call_output_device", None) or None,
            )
            return Path(ergebnis.audio)
        return None

    def interrupt(self) -> None:
        """Die KI unterbrechen – wie am Telefon dazwischenreden.

        Stoppt beides: den laufenden Satz UND die noch wartenden. Ohne das
        Leeren der Warteschlange spraeche sie die restlichen Saetze weiter.
        """
        self._playback.stop()
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
