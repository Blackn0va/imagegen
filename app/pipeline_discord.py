"""Telefonieren ueber einen Discord-Bot.

Statt ins eigene Mikrofon zu sprechen, sitzt ein Bot im Sprachkanal: alle
Anwesenden reden mit dem lokalen Sprachmodell, und seine Antwort ist fuer
alle hoerbar. Das Rechnen bleibt auf diesem Rechner -- es verlaesst kein
Audio den PC ausser dem, was ohnehin in den Kanal gesprochen wird.

WAS GEHT UND WAS NICHT
======================

Seit dem 2. Maerz 2026 verschluesselt Discord alle Sprachkanaele Ende zu
Ende (**DAVE**, ein MLS-Verfahren). Ausgenommen sind nur **Stage-Kanaele**.

Das trifft nicht die Anwendung, sondern die gesamte Python-Landschaft:
weder ``discord.py`` (kann Empfang ohnehin nicht), noch
``discord-ext-voice-recv``, noch ``py-cord`` entschluesseln DAVE im
Empfangspfad. Wer es trotzdem versucht, bekommt ``OpusError: corrupted
stream`` oder Rauschen -- die Pakete sind noch verschluesselt, wenn sie
beim Decoder ankommen.

Daraus folgt fuer diese Anwendung:

    Sprechkanal (normal)   Bot spricht: JA     Bot hoert zu: NEIN
    Stage-Kanal            Bot spricht: JA     Bot hoert zu: JA

Beides wird beim Verbinden geprueft und im Klartext gesagt. Ein Bot, der
im Kanal sitzt und stumm bleibt, weil niemand weiss warum, ist schlimmer
als eine Meldung.

RECHTLICHES
===========

Fremde Stimmen aufzunehmen ist kein technisches, sondern ein rechtliches
Problem. Discord kennt kein Recht "Sprache empfangen": wer ``CONNECT``
hat, bekommt die Stroeme. Die Schranke muss die Anwendung setzen:

  * Beim Betreten wird im Textkanal **angesagt**, dass mitgehoert wird.
  * ``/optout`` verwirft den Strom eines Teilnehmers.
  * Mitschnitt ist **aus**, solange ihn niemand ausdruecklich einschaltet
    (``discord_keep_audio``).
  * In Deutschland ist das Aufnehmen des nichtoeffentlich gesprochenen
    Wortes ohne Einwilligung nach § 201 StGB strafbar. Ein privater
    Discord-Kanal ist nichtoeffentlich.

Discords Entwicklerbedingungen verlangen ausserdem, empfangene Daten
weder weiterzugeben noch damit Modelle zu trainieren. Ein lokales Modell,
das nur antwortet, tut beides nicht -- Feinabstimmung auf Kanalmitschnitte
waere ein Verstoss.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import paths, secrets_store
from .accel import clean_error
from .call_transport import TransportInfo

log = logging.getLogger(__name__)

# Schluessel im verschluesselten Geheimnis-Speicher.
TOKEN_KEY = "discord_bot_token"

# Was Discord liefert und erwartet: 48 kHz, stereo, 16 Bit.
DISCORD_RATE = 48_000
DISCORD_CHANNELS = 2
DISCORD_WIDTH = 2

# Was Whisper braucht.
TARGET_RATE = 16_000

# Rechte, die der Bot in der Einladung braucht.
# VIEW_CHANNEL 1024 | CONNECT 1048576 | SPEAK 2097152 | USE_VAD 33554432
PERMISSIONS = 36701696
# Zusaetzlich fuer Stage-Kanaele: REQUEST_TO_SPEAK (1 << 32)
PERMISSIONS_STAGE = PERMISSIONS + 4294967296

INSTALL_HINT = 'pip install "discord.py[voice]" discord-ext-voice-recv'
BUILD_HINT = ".\\build-windows.ps1 -Clean -WithDiscord $true"

# Ansage beim Betreten. Steht hier und nicht verstreut im Code, damit der
# Wortlaut an einer Stelle nachlesbar ist.
JOIN_NOTICE = (
    "🎙️ **Hinweis:** Ich höre in diesem Kanal mit und beantworte, was gesagt "
    "wird, mit einem Sprachmodell auf einem privaten Rechner. Die Sprache "
    "verlässt diesen Rechner nicht und wird nicht zum Training verwendet.\n"
    "Wer nicht möchte, dass sein Beitrag verarbeitet wird, schreibt "
    "`!optout` – dann wird sein Ton verworfen. `!optin` hebt das wieder auf."
)


class DiscordUnavailable(RuntimeError):
    """Discord-Weg nicht nutzbar – der Text enthält die Anleitung."""

    expected = True


# ---------------------------------------------------------------------------
# Verfügbarkeit
# ---------------------------------------------------------------------------
def _nachruesten() -> str:
    import sys

    if getattr(sys, "frozen", False):
        return (
            "Dies ist ein gebautes Programm mit eigenem Python – ein "
            f"'pip install' wirkt hier nicht. Neu bauen mit: {BUILD_HINT}"
        )
    return f"Nachrüsten: {INSTALL_HINT}"


def runtime_available() -> tuple[bool, str]:
    """Sind die Pakete da? Prüft flach, lädt nichts."""
    import importlib.util

    for modul, name in (
        ("discord", "discord.py"),
        ("nacl", "PyNaCl"),
    ):
        try:
            if importlib.util.find_spec(modul) is None:
                return False, f"{name} fehlt. {_nachruesten()}"
        except Exception as exc:
            return False, f"{name} nicht prüfbar ({clean_error(exc)}). {_nachruesten()}"

    try:
        if importlib.util.find_spec("discord.ext.voice_recv") is None:
            return (
                True,
                "discord.py vorhanden, aber ohne discord-ext-voice-recv – "
                "der Bot kann sprechen, aber nicht zuhören.",
            )
    except Exception:
        return True, "discord.py vorhanden; Empfang nicht prüfbar."
    return True, "discord.py mit Empfangs-Erweiterung vorhanden."


def receive_possible() -> tuple[bool, str]:
    """Kann diese Installation Sprache empfangen?

    Getrennt von ``runtime_available``, weil der Empfang an einer zweiten
    Bedingung hängt, die kein Paket löst: seit dem 2. März 2026
    verschlüsselt Discord alle Sprachkanäle Ende zu Ende. Nur
    Stage-Kanäle sind ausgenommen.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("discord.ext.voice_recv") is None:
            return False, "discord-ext-voice-recv fehlt – ohne das kein Empfang."
    except Exception as exc:
        return False, f"Empfangs-Erweiterung nicht prüfbar: {clean_error(exc)}"
    return True, (
        "Empfang nur in Stage-Kanälen. Normale Sprachkanäle sind seit dem "
        "2.3.2026 Ende-zu-Ende verschlüsselt (DAVE); keine Python-Bibliothek "
        "kann das entschlüsseln."
    )


def token() -> str:
    """Bot-Token aus dem verschlüsselten Speicher."""
    try:
        return secrets_store.get_secret(TOKEN_KEY)
    except Exception as exc:
        log.warning("Bot-Token nicht lesbar: %s", clean_error(exc))
        return ""


def set_token(value: str) -> str:
    return secrets_store.set_secret(TOKEN_KEY, value.strip())


def invite_url(app_id: str, stage: bool = False) -> str:
    """Einladungs-Adresse für den Bot."""
    rechte = PERMISSIONS_STAGE if stage else PERMISSIONS
    return (
        f"https://discord.com/api/oauth2/authorize?client_id={app_id}"
        f"&permissions={rechte}&scope=bot%20applications.commands"
    )


# ---------------------------------------------------------------------------
# Audio umrechnen
# ---------------------------------------------------------------------------
@dataclass
class _Resampler:
    """48 kHz Stereo -> 16 kHz Mono, zustandsbehaftet.

    Der Zustand von ``audioop.ratecv`` muss über die Aufrufe erhalten
    bleiben. Ohne ihn entsteht an jeder Blockgrenze ein Knacken, und
    Whisper hört Silben, die niemand gesagt hat.
    """

    state: Any = None

    def to_whisper(self, pcm: bytes) -> bytes:
        import audioop

        mono = audioop.tomono(pcm, DISCORD_WIDTH, 0.5, 0.5)
        umgerechnet, self.state = audioop.ratecv(
            mono, DISCORD_WIDTH, 1, DISCORD_RATE, TARGET_RATE, self.state
        )
        return umgerechnet

    def reset(self) -> None:
        self.state = None


def wav_for_discord(quelle: Path, ziel: Path) -> Path:
    """Eine WAV-Datei in das Format bringen, das Discord erwartet.

    Discord will 48 kHz, stereo, 16 Bit. Die Windows-Stimmen liefern
    22050 Hz mono. 22050 auf 48000 ist das Verhältnis 147:320 – dabei
    interpoliert ``ratecv`` nur linear, was bei Sprache hörbar wird.
    Deshalb erzeugt ``pipeline_sapi`` auf Wunsch gleich 48 kHz; diese
    Funktion ist der Rückfall für alles andere.
    """
    import audioop
    import wave

    with wave.open(str(quelle), "rb") as ein:
        kanaele = ein.getnchannels()
        breite = ein.getsampwidth()
        rate = ein.getframerate()
        daten = ein.readframes(ein.getnframes())

    if breite != DISCORD_WIDTH:
        daten = audioop.lin2lin(daten, breite, DISCORD_WIDTH)
        breite = DISCORD_WIDTH
    if rate != DISCORD_RATE:
        daten, _ = audioop.ratecv(daten, breite, kanaele, rate, DISCORD_RATE, None)
    if kanaele == 1:
        daten = audioop.tostereo(daten, breite, 1.0, 1.0)
    elif kanaele > DISCORD_CHANNELS:
        daten = audioop.tomono(daten, breite, 0.5, 0.5)
        daten = audioop.tostereo(daten, breite, 1.0, 1.0)

    paths.ensure_dir(ziel.parent)
    with wave.open(str(ziel), "wb") as aus:
        aus.setnchannels(DISCORD_CHANNELS)
        aus.setsampwidth(DISCORD_WIDTH)
        aus.setframerate(DISCORD_RATE)
        aus.writeframes(daten)
    return ziel


# ---------------------------------------------------------------------------
# Der Bot
# ---------------------------------------------------------------------------
@dataclass
class SpeechChunk:
    """Ein zusammenhängender Redebeitrag eines Teilnehmers."""

    user_id: int
    user_name: str
    pcm16k: bytes
    seconds: float


class DiscordBot:
    """Hält die Verbindung, in einem eigenen Faden.

    Discord läuft auf asyncio, die Anwendung nicht. Statt beides zu
    vermischen bekommt der Bot einen eigenen Faden mit eigener
    Ereignisschleife; ausgetauscht wird über Warteschlangen. Das ist die
    Grenze, an der sonst Zustandsfehler entstehen, die nur unter Last
    auftreten.
    """

    def __init__(self, config: Any) -> None:
        self.config = config
        self._thread: threading.Thread | None = None
        self._loop: Any = None
        self._client: Any = None
        self._voice: Any = None
        self._bereit = threading.Event()
        self._fehler: str = ""
        self._stop = threading.Event()

        # Fertige Redebeiträge, die auf Verarbeitung warten.
        self.speech: queue.Queue[SpeechChunk] = queue.Queue()
        # Wer nicht verarbeitet werden möchte.
        self.opted_out: set[int] = set()
        self.channel_name: str = ""
        self.is_stage: bool = False
        self.receiving: bool = False
        self.last_speaker: str = ""
        self._puffer: dict[int, list] = {}
        self._resampler: dict[int, _Resampler] = {}
        self._letzter_ton: dict[int, float] = {}
        self.notes: list[str] = field(default_factory=list)  # type: ignore[assignment]
        self.notes = []

    # -- Start und Ende ------------------------------------------------
    def start(self, timeout: float = 30.0) -> None:
        """Anmelden und dem Sprachkanal beitreten. Blockiert bis fertig."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._bereit.clear()
        self._fehler = ""
        self._thread = threading.Thread(target=self._lauf, daemon=True, name="discord-bot")
        self._thread.start()
        if not self._bereit.wait(timeout):
            self.stop()
            raise DiscordUnavailable(
                self._fehler
                or f"Discord hat sich in {timeout:g}s nicht gemeldet. "
                "Token, Netzwerk und Kanal-ID prüfen."
            )
        if self._fehler:
            self.stop()
            raise DiscordUnavailable(self._fehler)

    def stop(self) -> None:
        self._stop.set()
        schleife = self._loop
        klient = self._client
        if schleife is not None and klient is not None:
            try:
                import asyncio

                asyncio.run_coroutine_threadsafe(klient.close(), schleife)
            except Exception as exc:
                log.debug("Abmelden fehlgeschlagen: %s", clean_error(exc))
        faden = self._thread
        if faden is not None:
            faden.join(timeout=8.0)
        self._thread = None
        self._loop = None
        self._client = None
        self._voice = None

    # -- Der Faden -----------------------------------------------------
    def _lauf(self) -> None:
        import asyncio

        try:
            schleife = asyncio.new_event_loop()
            asyncio.set_event_loop(schleife)
            self._loop = schleife
            schleife.run_until_complete(self._starte())
        except Exception as exc:
            self._fehler = clean_error(exc)
            log.warning("Discord-Faden beendet: %s", self._fehler)
        finally:
            self._bereit.set()

    async def _starte(self) -> None:
        import discord

        bot_token = token()
        if not bot_token:
            self._fehler = (
                "Kein Bot-Token hinterlegt. In den Einstellungen unter "
                "'Telefonieren über Discord' eintragen."
            )
            return

        # Nur die Absichten anfordern, die wirklich gebraucht werden.
        # GUILD_MEMBERS und MESSAGE_CONTENT sind privilegiert und muessen
        # im Entwicklerportal freigeschaltet werden -- ohne sie kommt man
        # aus, solange Slash-Befehle statt Praefix-Befehlen genutzt werden.
        intents = discord.Intents.none()
        intents.guilds = True
        intents.voice_states = True
        intents.guild_messages = True

        self._client = discord.Client(intents=intents)
        klient = self._client

        @klient.event
        async def on_ready() -> None:  # pragma: no cover – braucht Verbindung
            try:
                await self._betrete_kanal()
            except Exception as exc:
                self._fehler = clean_error(exc)
            finally:
                self._bereit.set()

        try:
            await klient.start(bot_token)
        except Exception as exc:
            text = clean_error(exc)
            if "improper token" in text.lower() or "401" in text:
                text = "Der Bot-Token wird abgelehnt. Im Entwicklerportal neu erzeugen."
            self._fehler = text
            self._bereit.set()

    async def _betrete_kanal(self) -> None:  # pragma: no cover – braucht Verbindung
        import discord

        kanal_id = str(getattr(self.config, "discord_channel_id", "") or "").strip()
        if not kanal_id.isdigit():
            raise DiscordUnavailable(
                "Keine gültige Kanal-ID. Im Discord-Client mit Rechtsklick auf "
                "den Sprachkanal → 'ID kopieren' (Entwicklermodus muss an sein)."
            )
        kanal = self._client.get_channel(int(kanal_id))
        if kanal is None:
            raise DiscordUnavailable(
                f"Kanal {kanal_id} nicht gefunden. Ist der Bot auf dem Server "
                "und darf er den Kanal sehen?"
            )
        if not isinstance(kanal, discord.VoiceChannel | discord.StageChannel):
            raise DiscordUnavailable(f"'{getattr(kanal, 'name', kanal_id)}' ist kein Sprachkanal.")

        self.channel_name = getattr(kanal, "name", kanal_id)
        self.is_stage = isinstance(kanal, discord.StageChannel)

        # Empfang nur versuchen, wo er möglich ist.
        empfang_moeglich, grund = receive_possible()
        klasse = None
        if empfang_moeglich and self.is_stage:
            try:
                from discord.ext import voice_recv

                klasse = voice_recv.VoiceRecvClient
            except Exception as exc:
                self.notes.append(f"Empfang nicht ladbar: {clean_error(exc)}")

        self._voice = await kanal.connect(cls=klasse) if klasse else await kanal.connect()

        if klasse is not None:
            try:
                from discord.ext import voice_recv

                self._voice.listen(voice_recv.BasicSink(self._on_audio))
                self.receiving = True
            except Exception as exc:
                self.notes.append(f"Zuhören fehlgeschlagen: {clean_error(exc)}")

        if not self.receiving:
            self.notes.append(
                "Der Bot spricht, hört aber nicht zu. "
                + (
                    "Dieser Kanal ist Ende-zu-Ende verschlüsselt (DAVE); "
                    "für Empfang einen Stage-Kanal verwenden."
                    if not self.is_stage
                    else grund
                )
            )

        await self._sage_an(kanal)

    async def _sage_an(self, kanal) -> None:  # pragma: no cover – braucht Verbindung
        """Im Textkanal ansagen, dass mitgehört wird.

        Keine Höflichkeit, sondern Voraussetzung: fremde Sprache ohne
        Wissen der Sprechenden zu verarbeiten ist in Deutschland nach
        § 201 StGB strafbar und nach DSGVO ohne Rechtsgrundlage.
        """
        if not self.receiving:
            return
        text_kanal = None
        for kandidat in getattr(getattr(kanal, "guild", None), "text_channels", []):
            if kandidat.permissions_for(kanal.guild.me).send_messages:
                text_kanal = kandidat
                break
        if text_kanal is None:
            self.notes.append(
                "Kein Textkanal zum Ansagen gefunden – bitte die Anwesenden "
                "selbst darauf hinweisen, dass mitgehört wird."
            )
            return
        try:
            await text_kanal.send(JOIN_NOTICE)
        except Exception as exc:
            self.notes.append(f"Ansage nicht gesendet: {clean_error(exc)}")

    # -- Empfang -------------------------------------------------------
    def _on_audio(self, user: Any, data: Any) -> None:  # pragma: no cover
        """Ein 20-ms-Paket von einem Teilnehmer.

        Läuft im Discord-Faden. Hier wird nur gesammelt und umgerechnet;
        alles Weitere passiert im Gesprächsfaden.
        """
        if user is None or self._stop.is_set():
            return
        uid = int(getattr(user, "id", 0))
        if uid in self.opted_out:
            return

        pcm = getattr(data, "pcm", b"")
        if not pcm:
            return

        umrechner = self._resampler.setdefault(uid, _Resampler())
        try:
            stueck = umrechner.to_whisper(pcm)
        except Exception as exc:
            log.debug("Umrechnung fehlgeschlagen: %s", clean_error(exc))
            return

        self._puffer.setdefault(uid, []).append(stueck)
        self._letzter_ton[uid] = time.time()
        self.last_speaker = str(getattr(user, "display_name", "") or getattr(user, "name", ""))

    def collect_finished(self, silence_seconds: float) -> list[SpeechChunk]:
        """Beiträge einsammeln, bei denen die Redepause lang genug war.

        Wird vom Gesprächsfaden regelmäßig aufgerufen. Die Trennung nach
        Sprechpausen passiert hier und nicht im Discord-Faden, damit der
        nie durch Arbeit aufgehalten wird – sonst reißt der Ton.
        """
        fertig: list[SpeechChunk] = []
        jetzt = time.time()
        for uid in list(self._puffer):
            letzter = self._letzter_ton.get(uid, 0.0)
            if jetzt - letzter < silence_seconds:
                continue
            stuecke = self._puffer.pop(uid, [])
            self._letzter_ton.pop(uid, None)
            self._resampler.pop(uid, None)
            if not stuecke:
                continue
            roh = b"".join(stuecke)
            sekunden = len(roh) / float(TARGET_RATE * 2)
            if sekunden < 0.35:
                continue  # Huster, kein Beitrag
            fertig.append(
                SpeechChunk(
                    user_id=uid,
                    user_name=self.last_speaker or str(uid),
                    pcm16k=roh,
                    seconds=sekunden,
                )
            )
        return fertig

    # -- Senden --------------------------------------------------------
    def play_file(self, wav: Path, timeout: float = 120.0) -> None:
        """Eine WAV-Datei im Kanal abspielen. Blockiert bis fertig."""
        if self._voice is None:
            raise DiscordUnavailable("Nicht mit einem Sprachkanal verbunden.")
        import asyncio

        import discord

        fertig = threading.Event()

        async def spiele() -> None:
            quelle = discord.FFmpegPCMAudio(str(wav))
            if self._voice.is_playing():
                self._voice.stop()
            self._voice.play(quelle, after=lambda _e: fertig.set())

        try:
            asyncio.run_coroutine_threadsafe(spiele(), self._loop).result(timeout=10)
        except Exception as exc:
            raise DiscordUnavailable(f"Wiedergabe nicht möglich: {clean_error(exc)}") from exc
        fertig.wait(timeout)

    def stop_playing(self) -> None:
        if self._voice is None:
            return
        import asyncio

        async def halt() -> None:
            if self._voice.is_playing():
                self._voice.stop()

        try:
            asyncio.run_coroutine_threadsafe(halt(), self._loop)
        except Exception as exc:
            log.debug("Abbrechen fehlgeschlagen: %s", clean_error(exc))


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
class DiscordTransport:
    """Der Weg über Discord, als ``Transport`` für ``CallSession``."""

    def __init__(self, config: Any) -> None:
        self.config = config
        self.bot = DiscordBot(config)
        self._zaehler = 0

    # -- Zustand -------------------------------------------------------
    @staticmethod
    def info(config: Any) -> TransportInfo:
        einzelheiten: list[str] = []

        # Die rechtliche Schranke steht VOR der technischen. Ob ein Paket
        # fehlt, ist eine Frage der Einrichtung; ob fremde Stimmen
        # verarbeitet werden duerfen, ist keine.
        #
        # Fail-closed: ohne die ausdrueckliche Bestaetigung, dass die
        # Einwilligung der Beteiligten eingeholt wird, bleibt der Weg zu.
        # Fremde Stimmen ohne deren Wissen zu verarbeiten ist in
        # Deutschland nach § 201 StGB strafbar; das darf nicht davon
        # abhaengen, ob jemand einen Hinweistext gelesen hat.
        if not bool(getattr(config, "discord_consent_confirmed", False)):
            return TransportInfo(
                key="discord",
                title="Discord-Bot",
                ready=False,
                reason=(
                    "Einwilligung nicht bestätigt. Unter 'Discord einrichten' "
                    "bestätigen, dass die Beteiligten einverstanden sind."
                ),
                multi_speaker=True,
            )
        ok, grund = runtime_available()
        if not ok:
            return TransportInfo(
                key="discord", title="Discord-Bot", ready=False, reason=grund, multi_speaker=True
            )
        if not token():
            return TransportInfo(
                key="discord",
                title="Discord-Bot",
                ready=False,
                reason="Kein Bot-Token hinterlegt.",
                multi_speaker=True,
            )
        kanal = str(getattr(config, "discord_channel_id", "") or "").strip()
        if not kanal.isdigit():
            return TransportInfo(
                key="discord",
                title="Discord-Bot",
                ready=False,
                reason="Keine gültige Kanal-ID.",
                multi_speaker=True,
            )

        empfang, empfangsgrund = receive_possible()
        einzelheiten.append(grund)
        einzelheiten.append(
            "Zuhören: nur in Stage-Kanälen" if empfang else f"Zuhören: {empfangsgrund}"
        )
        return TransportInfo(
            key="discord",
            title="Discord-Bot",
            ready=True,
            reason="bereit",
            multi_speaker=True,
            details=tuple(einzelheiten),
        )

    # -- Ablauf --------------------------------------------------------
    def open(self, context: Any) -> None:
        # Auch hier pruefen, nicht nur in der Oberflaeche: die Sitzung
        # laesst sich auch ueber die Kommandozeile starten.
        stand = self.info(self.config)
        if not stand.ready:
            raise DiscordUnavailable(stand.reason)
        ok, grund = runtime_available()
        if not ok:
            raise DiscordUnavailable(grund)
        context.status("Bot meldet sich bei Discord an …")
        self.bot.start()
        art = "Stage-Kanal" if self.bot.is_stage else "Sprachkanal"
        context.status(f"Im {art} '{self.bot.channel_name}'.")
        for hinweis in self.bot.notes:
            context.status(hinweis)

    def listen(
        self,
        target: Path,
        on_level: Callable[[float, bool], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        on_threshold: Callable[[float], None] | None = None,
    ) -> tuple[Path | None, float]:
        """Auf den nächsten fertigen Redebeitrag warten."""
        if not self.bot.receiving:
            # Nicht endlos warten, wenn nie etwas kommen kann.
            raise DiscordUnavailable(
                "Dieser Kanal liefert keinen Ton an den Bot. " + " ".join(self.bot.notes)
            )

        from . import audio_io

        stille = float(getattr(self.config, "discord_silence_seconds", 1.4) or 1.4)
        while not (should_stop and should_stop()):
            fertige = self.bot.collect_finished(stille)
            if fertige:
                # Bei mehreren gleichzeitig: der längste Beitrag gewinnt.
                beitrag = max(fertige, key=lambda c: c.seconds)
                import numpy as np

                werte = np.frombuffer(beitrag.pcm16k, dtype="<i2").astype("float32") / 32768.0
                audio_io.write_wav_float(target, werte, TARGET_RATE)
                self._letzter_sprecher = beitrag.user_name
                if on_level is not None:
                    on_level(float(np.sqrt(np.mean(np.square(werte)))) if werte.size else 0.0, True)
                return target, beitrag.seconds
            time.sleep(0.1)
        return None, 0.0

    def play(self, wav: Path) -> None:
        self._zaehler += 1
        ziel = wav.with_name(f"{wav.stem}-discord.wav")
        # Kommt die Datei schon im Zielformat (SAPI spricht im
        # Discord-Modus gleich 48 kHz stereo), entfaellt die Umwandlung.
        try:
            import wave

            with wave.open(str(wav), "rb") as pruef:
                passt = (
                    pruef.getframerate() == DISCORD_RATE
                    and pruef.getnchannels() == DISCORD_CHANNELS
                    and pruef.getsampwidth() == DISCORD_WIDTH
                )
            if passt:
                self.bot.play_file(wav)
                return
        except Exception as exc:
            log.debug("Format nicht prüfbar: %s", clean_error(exc))

        try:
            fertig = wav_for_discord(wav, ziel)
        except Exception as exc:
            log.warning("Umwandlung für Discord fehlgeschlagen: %s", clean_error(exc))
            fertig = wav
        self.bot.play_file(fertig)

    def stop_playback(self) -> None:
        self.bot.stop_playing()

    def close(self) -> None:
        self.bot.stop()

    def speaker_hint(self) -> str:
        return getattr(self, "_letzter_sprecher", "")


def describe() -> str:
    """Zustandsbericht für Diagnose und Oberfläche."""
    ok, grund = runtime_available()
    zeilen = [f"Laufzeit: {grund}"]
    _empfang, empfangsgrund = receive_possible() if ok else (False, "Laufzeit fehlt")
    zeilen.append(f"Empfang:  {empfangsgrund}")
    zustand = secrets_store.info(TOKEN_KEY)
    zeilen.append(f"Token:    {zustand.label()}")
    zeilen.append(f"Rechte:   {PERMISSIONS} (Stage: {PERMISSIONS_STAGE})")
    if os.name != "nt":
        zeilen.append("Hinweis:  Token wird ohne DPAPI im Klartext abgelegt.")
    return "\n".join(zeilen)
