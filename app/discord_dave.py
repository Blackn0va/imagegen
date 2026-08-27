"""Die fehlende Hälfte der Ende-zu-Ende-Verschlüsselung.

Seit dem 2. März 2026 verschlüsselt Discord Sprachkanäle Ende zu Ende
(**DAVE**, ein MLS-Verfahren). Der Ton ist damit zweifach verpackt:

    RTP-Paket
      └─ Transportschlüssel (xsalsa20 / aead_xchacha20)   Server ↔ Client
           └─ DAVE-Schlüssel (MLS-Gruppe)            Teilnehmer ↔ Teilnehmer
                └─ Opus-Bild

``discord.py`` beherrscht beide Schichten, aber nur beim **Senden**
(``voice_client.py``: ``dave_session.encrypt_opus``); empfangen kann es
gar nicht. ``discord-ext-voice-recv`` empfängt, löst aber nur die
Transportschicht: in seinem ``reader.py`` steht direkt hinter
``decrypt_rtp`` schon der Opus-Decoder – kein DAVE dazwischen. Was dort
ankommt, ist also noch verschlüsselt, und Opus meldet ``corrupted
stream``.

Beide Hälften sind vorhanden, sie sind nur nicht verbunden. Diese Datei
legt den fehlenden Schritt dazwischen. Damit hört der Bot in **normalen
Sprachkanälen** zu; der frühere Umweg über Stage-Kanäle entfällt.

WARUM EIN UMHÜLLEN UND KEIN EIGENER READER
==========================================

``AudioReader`` baut seinen ``PacketDecryptor`` selbst und startet den
Lesefaden im selben Aufruf. Einen eigenen Reader zu unterhalten hieße,
die RTP-Zerlegung, die Sprecher-Zuordnung und die Jitter-Pufferung aus
``voice_recv`` nachzubauen und bei jeder Aktualisierung nachzuziehen.
Umhüllt wird deshalb nur die eine Methode, die genau eine Aufgabe hat:
aus einem Paket Nutzdaten machen. Bricht die Fremdbibliothek diesen
Namen, meldet sich :func:`attach` mit einer klaren Ursache, statt still
zu versagen.

Angefasst wird immer nur die **Instanz** dieses einen Bots, nie die
Klasse – ein anderer Client im selben Prozess bleibt unberührt.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .accel import clean_error

log = logging.getLogger(__name__)

# Was Discord als "hier spricht gerade niemand" sendet. Diese Pakete
# tragen keine DAVE-Verpackung und dürfen unberührt weiterlaufen.
OPUS_SILENCE = b"\xf8\xff\xfe"


@dataclass
class DaveStats:
    """Mitschrift darüber, was aus den Paketen wurde.

    Ohne diese Zahlen ist ein stummer Bot nicht von einem Bot zu
    unterscheiden, dem niemand etwas sagt. Genau diese Verwechslung hat
    den Discord-Weg vorher unbrauchbar gemacht.
    """

    entschluesselt: int = 0
    durchgereicht: int = 0
    ohne_sprecher: int = 0
    fehlgeschlagen: int = 0
    letzter_fehler: str = ""

    @property
    def verworfen(self) -> int:
        return self.ohne_sprecher + self.fehlgeschlagen

    def summary(self) -> str:
        if not any((self.entschluesselt, self.durchgereicht, self.verworfen)):
            return "Noch kein Ton empfangen."
        teile = [f"{self.entschluesselt} entschlüsselt"]
        if self.durchgereicht:
            teile.append(f"{self.durchgereicht} unverschlüsselt")
        if self.ohne_sprecher:
            teile.append(f"{self.ohne_sprecher} ohne Sprecherzuordnung")
        if self.fehlgeschlagen:
            teile.append(f"{self.fehlgeschlagen} nicht entschlüsselbar")
        text = ", ".join(teile)
        if self.fehlgeschlagen and self.letzter_fehler:
            text += f" (zuletzt: {self.letzter_fehler})"
        return text

    def healthy(self) -> bool:
        """Kommt genug an, um daraus Sprache zu machen?

        Ein paar Ausfälle sind normal: zwischen dem ersten Paket eines
        Sprechers und der Meldung, welche Kennung dazugehört, liegen
        einige Bilder. Dauerhaft überwiegende Ausfälle sind es nicht.
        """
        brauchbar = self.entschluesselt + self.durchgereicht
        if brauchbar == 0:
            return False
        return self.fehlgeschlagen <= brauchbar


def _nachruesten() -> str:
    """Wie davey nachzurüsten ist – im Bündel anders als im Quellbaum."""
    import sys

    if getattr(sys, "frozen", False):
        return (
            "Dies ist ein gebautes Programm mit eigenem Python; neu bauen mit "
            ".\\build-windows.ps1 -WithDiscord $true"
        )
    return "Nachrüsten: pip install davey"


def available() -> tuple[bool, str]:
    """Sind die Teile da, um DAVE zu entschlüsseln?"""
    import importlib.util

    try:
        if importlib.util.find_spec("davey") is None:
            return (
                False,
                f"davey fehlt – ohne das kein Empfang in verschlüsselten Kanälen. {_nachruesten()}",
            )
    except Exception as exc:
        return False, f"davey nicht prüfbar: {clean_error(exc)}"

    try:
        import davey
    except Exception as exc:
        return False, f"davey nicht ladbar: {clean_error(exc)}"

    if not hasattr(davey.DaveSession, "decrypt"):
        return False, "Diese davey-Fassung kann nicht entschlüsseln (kein 'decrypt')."
    return True, f"DAVE-Fassung {getattr(davey, 'DAVE_PROTOCOL_VERSION', '?')} einsatzbereit."


def attach(voice_client: Any) -> DaveStats:
    """Den DAVE-Schritt in einen laufenden Empfang einhängen.

    Muss **nach** ``listen()`` aufgerufen werden, weil der Reader erst
    dort entsteht. Die ersten Pakete können dabei durchrutschen; das
    fällt nicht auf, weil ohnehin bis zur ersten Sprechpause gesammelt
    wird, bevor irgendetwas ausgewertet wird.

    Wirft :class:`RuntimeError`, wenn der Einhängepunkt fehlt – lieber
    eine deutliche Meldung als ein Bot, der ohne Grund schweigt.
    """
    leser = getattr(voice_client, "_reader", None)
    if leser is None:
        raise RuntimeError("Kein Lesefaden vorhanden – wurde listen() aufgerufen?")
    entschluessler = getattr(leser, "decryptor", None)
    if entschluessler is None or not hasattr(entschluessler, "decrypt_rtp"):
        raise RuntimeError(
            "discord-ext-voice-recv hat seinen Aufbau geändert: "
            "'decryptor.decrypt_rtp' nicht gefunden. Ohne diesen Punkt "
            "lässt sich die Ende-zu-Ende-Verschlüsselung nicht auflösen."
        )

    # Erst nach der Aufbauprüfung, damit ein fehlendes davey nicht die
    # deutlichere Meldung über einen geänderten Fremdaufbau verdeckt.
    try:
        import davey
    except ImportError as exc:
        raise RuntimeError(
            "davey fehlt – ohne diese Bibliothek lässt sich die "
            "Ende-zu-Ende-Verschlüsselung der Sprachkanäle nicht auflösen. "
            f"{_nachruesten()}"
        ) from exc

    zahlen = DaveStats()
    vorher = entschluessler.decrypt_rtp
    audio = davey.MediaType.audio

    def decrypt_rtp(packet: Any) -> bytes:
        # Schritt 1: Transportschicht – das kann voice_recv bereits.
        daten = vorher(packet)

        # Schritt 2: DAVE. Nur wenn der Kanal wirklich verschlüsselt ist;
        # sonst liegt hier schon fertiges Opus.
        zustand = getattr(voice_client, "_connection", None)
        sitzung = getattr(zustand, "dave_session", None)
        if sitzung is None or not getattr(sitzung, "ready", False):
            zahlen.durchgereicht += 1
            return daten
        if daten == OPUS_SILENCE or not daten:
            return daten

        # DAVE entschlüsselt je Absender. Solange unbekannt ist, wer
        # hinter einer Tonspur steckt, ist das Paket nicht zu öffnen.
        kennung = voice_client._ssrc_to_id.get(packet.ssrc)
        if kennung is None:
            zahlen.ohne_sprecher += 1
            return OPUS_SILENCE

        try:
            klar = sitzung.decrypt(kennung, audio, daten)
        except Exception as exc:
            # Verwerfen, nicht durchreichen: verschlüsselte Bytes im
            # Opus-Decoder erzeugen Rauschen, und Rauschen wird von der
            # Spracherkennung zu Wörtern gemacht, die niemand gesagt hat.
            zahlen.fehlgeschlagen += 1
            zahlen.letzter_fehler = clean_error(exc)
            if zahlen.fehlgeschlagen in (1, 50) or zahlen.fehlgeschlagen % 500 == 0:
                log.warning(
                    "DAVE-Entschlüsselung fehlgeschlagen (%dx): %s",
                    zahlen.fehlgeschlagen,
                    zahlen.letzter_fehler,
                )
            return OPUS_SILENCE

        zahlen.entschluesselt += 1
        return klar

    entschluessler.decrypt_rtp = decrypt_rtp  # type: ignore[method-assign]
    log.info("DAVE-Entschlüsselung eingehängt.")
    return zahlen


def privacy_code(voice_client: Any) -> str:
    """Der Prüfcode des Kanals, sofern verschlüsselt.

    Discord zeigt denselben Code allen Teilnehmern an. Stimmt er
    überein, redet niemand dazwischen. Für die Anwesenden ist das die
    einzige Möglichkeit nachzuprüfen, dass der Bot in derselben
    geschützten Runde sitzt wie sie.
    """
    try:
        return str(getattr(voice_client, "voice_privacy_code", "") or "")
    except Exception as exc:
        log.debug("Prüfcode nicht lesbar: %s", clean_error(exc))
        return ""
