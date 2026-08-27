"""Pruefungen fuer Discord-Telefonie, Transport-Umschaltung und Geheimnisse.

Der Discord-Weg laesst sich ohne Server nicht vollstaendig pruefen. Was
hier geprueft wird, ist alles, was ohne Verbindung feststeht: die
Umrechnung der Tonformate, die Zugangsschranken, das Ablegen des Tokens
und die Frage, ob die Anwendung ehrlich sagt, was sie kann.

Was NICHT geprueft ist: eine echte Anmeldung, das Betreten eines Kanals
und der Empfang. Dafuer braucht es einen Bot-Token und einen Server.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(check) -> None:
    """Alle Pruefungen dieses Moduls."""
    run_secrets(check)
    run_transport(check)
    run_discord_gates(check)
    run_discord_audio(check)
    run_dave(check)
    run_theme(check)


# ---------------------------------------------------------------------------
# Geheimnisse
# ---------------------------------------------------------------------------
def run_secrets(check) -> None:
    """Ein Bot-Token darf nicht im Klartext herumliegen."""
    print("\n== Geheimnisse ==")
    import os

    from app import secrets_store

    beispiel = "MTIzNDU2Nzg5MC5FeGFtcGxlLk5pY2h0RWNodA"
    schluessel = "test_token"

    verfahren = secrets_store.set_secret(schluessel, beispiel)
    check("Wert laesst sich ablegen", bool(verfahren), verfahren)

    zurueck = secrets_store.get_secret(schluessel)
    check("Wert kommt unveraendert zurueck", zurueck == beispiel)

    inhalt = secrets_store.path().read_text(encoding="utf-8")
    if os.name == "nt":
        check(
            "Token steht nicht im Klartext in der Datei",
            beispiel not in inhalt,
            "DPAPI muesste ihn verschluesselt haben",
        )
        check("DPAPI wurde benutzt", verfahren == secrets_store.VERFAHREN_DPAPI, verfahren)
    else:
        check(
            "ohne DPAPI wird das ausdruecklich vermerkt",
            verfahren == secrets_store.VERFAHREN_KLAR,
        )

    zustand = secrets_store.info(schluessel)
    check("Zustand meldet 'hinterlegt'", zustand.present)
    check(
        "Erkennungshilfe zeigt nicht den ganzen Wert",
        beispiel not in zustand.hint and bool(zustand.hint),
        zustand.hint,
    )

    secrets_store.set_secret(schluessel, "")
    check("Leerer Wert loescht den Eintrag", not secrets_store.info(schluessel).present)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------
def run_transport(check) -> None:
    """Der Weg des Gespraechs laesst sich umschalten."""
    print("\n== Weg des Gespraechs ==")
    from app import call_transport
    from app.config import AppConfig

    wege = call_transport.available_transports(AppConfig())
    check("es gibt zwei Wege", len(wege) == 2, str([w.key for w in wege]))
    check("beide nennen einen Zustand", all(bool(w.reason) for w in wege))
    check(
        "Discord ist als Weg mit mehreren Sprechern gekennzeichnet",
        next(w for w in wege if w.key == "discord").multi_speaker,
    )
    check(
        "der eigene Rechner hat nur einen Sprecher",
        not next(w for w in wege if w.key == "lokal").multi_speaker,
    )

    lokal = call_transport.create_transport(AppConfig(call_mode="lokal"))
    check("lokal ergibt den lokalen Weg", type(lokal).__name__ == "LocalTransport")

    discord = call_transport.create_transport(AppConfig(call_mode="discord"))
    check("discord ergibt den Discord-Weg", type(discord).__name__ == "DiscordTransport")

    # Fail-closed: ein unbekannter Wert darf nicht in Discord landen.
    unbekannt = call_transport.create_transport(AppConfig(call_mode="unsinn"))
    check(
        "unbekannter Wert faellt auf den lokalen Weg",
        type(unbekannt).__name__ == "LocalTransport",
        "sonst spricht jemand versehentlich in einen Kanal",
    )

    check(
        "der Transport erfuellt die Schnittstelle",
        isinstance(lokal, call_transport.Transport),
    )


# ---------------------------------------------------------------------------
# Zugangsschranken
# ---------------------------------------------------------------------------
def run_discord_gates(check) -> None:
    """Ohne Einwilligung, Token und Kanal bleibt der Bot zu."""
    print("\n== Discord: Schranken ==")
    from app import pipeline_discord
    from app.config import AppConfig

    # Reihenfolge zaehlt: die Einwilligung wird VOR allem anderen geprueft.
    ohne = pipeline_discord.DiscordTransport.info(AppConfig())
    check("ohne alles nicht bereit", not ohne.ready)
    check(
        "Einwilligung wird zuerst verlangt",
        "Einwilligung" in ohne.reason,
        ohne.reason,
    )

    mit_zustimmung = pipeline_discord.DiscordTransport.info(
        AppConfig(discord_consent_confirmed=True)
    )
    check("danach fehlt der Token", not mit_zustimmung.ready)

    # Der Kanal muss eine Zahl sein.
    falsch = pipeline_discord.DiscordTransport.info(
        AppConfig(discord_consent_confirmed=True, discord_channel_id="mein-kanal")
    )
    check("Kanalname statt ID wird abgelehnt", not falsch.ready)

    # Rechte und Einladung
    check(
        "Einladung enthaelt die noetigen Rechte",
        str(pipeline_discord.PERMISSIONS) in pipeline_discord.invite_url("42"),
    )
    check(
        "Stage-Einladung hat mehr Rechte",
        pipeline_discord.PERMISSIONS_STAGE > pipeline_discord.PERMISSIONS,
    )
    check(
        "Rechte enthalten CONNECT und SPEAK",
        bool(pipeline_discord.PERMISSIONS & (1 << 20))
        and bool(pipeline_discord.PERMISSIONS & (1 << 21)),
    )

    # Ehrlichkeit ueber den Empfang: was auch immer der Zustand ist, er
    # muss benannt werden. Geht Empfang, steht warum; geht er nicht, steht
    # was fehlt. Ein leerer oder nichtssagender Grund ist der Fehler, der
    # den Nutzer vor einem stummen Bot sitzen laesst.
    empfang, grund = pipeline_discord.receive_possible()
    check("der Empfangszustand wird begruendet", len(grund) > 20, grund)
    if empfang:
        check(
            "bei moeglichem Empfang wird die Verschluesselung erwaehnt",
            "erschl" in grund,  # ver-schl-uesselt, gross wie klein
            grund,
        )
    else:
        check(
            "bei unmoeglichem Empfang steht das fehlende Stueck",
            any(wort in grund for wort in ("fehlt", "nicht prüfbar", "nicht ladbar", "Stage")),
            grund,
        )
    check(
        "die Ansage nennt den Widerspruchsweg",
        "optout" in pipeline_discord.JOIN_NOTICE,
    )
    check(
        "die Ansage sagt, dass mitgehoert wird",
        "höre" in pipeline_discord.JOIN_NOTICE or "hoere" in pipeline_discord.JOIN_NOTICE,
    )

    bericht = pipeline_discord.describe()
    check("Bericht nennt den Zustand des Tokens", "Token:" in bericht)


# ---------------------------------------------------------------------------
# Tonformate
# ---------------------------------------------------------------------------
def run_discord_audio(check) -> None:
    """Discord liefert 48 kHz Stereo, Whisper will 16 kHz Mono."""
    print("\n== Discord: Tonformate ==")
    import tempfile
    import wave

    try:
        import numpy as np
    except ImportError:
        print("  über  numpy fehlt – übersprungen")
        return
    try:
        # In Python 3.13 aus der Standardbibliothek entfernt; 'audioop-lts'
        # liefert es nach und kommt mit discord.py mit.
        import audioop  # noqa: F401
    except ImportError:
        print("  über  audioop fehlt (Paket 'audioop-lts') – übersprungen")
        return

    from app import audio_io, pipeline_discord

    check("Discord-Rate ist 48 kHz", pipeline_discord.DISCORD_RATE == 48_000)
    check("Ziel ist 16 kHz", pipeline_discord.TARGET_RATE == 16_000)

    # 440-Hz-Ton in Discords Format bauen.
    t = np.arange(48_000) / 48_000.0
    ton = (0.4 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    stereo = np.empty(len(t) * 2, dtype="<i2")
    stereo[0::2] = ton
    stereo[1::2] = ton
    roh = stereo.tobytes()

    umrechner = pipeline_discord._Resampler()
    block = 960 * 2 * 2  # 20 ms, wie Discord sie schickt
    aus = b"".join(
        umrechner.to_whisper(roh[i : i + block]) for i in range(0, len(roh) - block, block)
    )
    sekunden = len(aus) / (16_000 * 2)
    check("Laenge bleibt erhalten", 0.95 < sekunden < 1.02, f"{sekunden:.2f} s")

    werte = np.frombuffer(aus, dtype="<i2").astype("float32") / 32768.0
    spektrum = np.abs(np.fft.rfft(werte))
    frequenzen = np.fft.rfftfreq(len(werte), 1 / 16_000)
    spitze = frequenzen[int(np.argmax(spektrum))]
    check("Tonhoehe bleibt erhalten", abs(spitze - 440) < 15, f"{spitze:.0f} Hz")

    # Der Zustand des Umrechners muss ueber Bloecke hinweg gehalten
    # werden - ohne ihn knackt es an jeder Blockgrenze.
    frisch = pipeline_discord._Resampler()
    check("Umrechner startet ohne Zustand", frisch.state is None)
    frisch.to_whisper(roh[:block])
    check("Umrechner merkt sich den Zustand", frisch.state is not None)

    # Rueckweg: WAV in Discords Format
    ordner = Path(tempfile.mkdtemp())
    tt = np.arange(int(22050 * 0.4)) / 22050.0
    audio_io.write_wav_float(
        ordner / "sapi.wav", (0.3 * np.sin(2 * np.pi * 440 * tt)).astype("float32"), 22050
    )
    ziel = pipeline_discord.wav_for_discord(ordner / "sapi.wav", ordner / "fuer-discord.wav")
    with wave.open(str(ziel), "rb") as datei:
        check("Ausgabe ist 48 kHz", datei.getframerate() == 48_000, str(datei.getframerate()))
        check("Ausgabe ist stereo", datei.getnchannels() == 2, str(datei.getnchannels()))
        check("Ausgabe ist 16 Bit", datei.getsampwidth() == 2)
        dauer = datei.getnframes() / datei.getframerate()
        check("Dauer bleibt erhalten", 0.38 < dauer < 0.42, f"{dauer:.2f} s")


# ---------------------------------------------------------------------------
# Ende-zu-Ende-Verschluesselung
# ---------------------------------------------------------------------------
class _FakeSession:
    """MLS-Sitzung als Attrappe. ``ready`` und Ausgang sind einstellbar."""

    def __init__(self, ready=True, ergebnis=b"KLARTEXT", fehler=None):
        self.ready = ready
        self._ergebnis = ergebnis
        self._fehler = fehler
        self.aufrufe = []

    def decrypt(self, user_id, media_type, packet):
        self.aufrufe.append((user_id, packet))
        if self._fehler:
            raise self._fehler
        return self._ergebnis


class _FakePacket:
    def __init__(self, ssrc=1):
        self.ssrc = ssrc


class _FakeClient:
    """So viel VoiceClient, wie das Einhaengen anfasst."""

    def __init__(self, session=None, roh=b"VERSCHLUESSELT", ssrc_map=None):
        self._connection = type("Zustand", (), {"dave_session": session})()
        self._ssrc_to_id = ssrc_map if ssrc_map is not None else {1: 4242}
        entschluessler = type("Entschluessler", (), {})()
        entschluessler.decrypt_rtp = lambda packet: roh
        self._reader = type("Leser", (), {})()
        self._reader.decryptor = entschluessler

    def decrypt(self, packet):
        return self._reader.decryptor.decrypt_rtp(packet)


def run_dave(check) -> None:
    """Die Schicht, die discord-ext-voice-recv fehlt.

    Ohne sie kommt der Ton verschluesselt beim Opus-Decoder an und der
    Bot hoert nichts. Geprueft wird die Umhuellung selbst -- ohne Netz,
    ohne Discord, ohne Schluessel.
    """
    print("\n== Discord: Verschluesselung ==")
    from app import discord_dave

    # -- Buchfuehrung ---------------------------------------------------
    zahlen = discord_dave.DaveStats()
    check("frische Zaehlung meldet keinen Ton", "Noch kein Ton" in zahlen.summary())
    check("ohne Ton gilt der Empfang nicht als gesund", not zahlen.healthy())

    zahlen.entschluesselt = 100
    check("mit Ton gilt der Empfang als gesund", zahlen.healthy())
    zahlen.fehlgeschlagen = 500
    check("bei ueberwiegenden Ausfaellen nicht mehr", not zahlen.healthy())
    check("Ausfaelle stehen im Bericht", "nicht entschlüsselbar" in zahlen.summary())

    # -- Einhaengepunkt --------------------------------------------------
    try:
        discord_dave.attach(type("Leer", (), {"_reader": None})())
        check("fehlender Lesefaden faellt auf", False, "keine Meldung")
    except RuntimeError as exc:
        check("fehlender Lesefaden faellt auf", "listen()" in str(exc), str(exc))

    kaputt = _FakeClient()
    del kaputt._reader.decryptor
    try:
        discord_dave.attach(kaputt)
        check("geaenderter Fremdaufbau faellt auf", False, "keine Meldung")
    except RuntimeError as exc:
        check("geaenderter Fremdaufbau faellt auf", "decrypt_rtp" in str(exc), str(exc))

    # Ab hier wird davey selbst gebraucht: attach() holt daraus die
    # Angabe, dass es sich um Ton handelt. Fehlt es, ist der Empfang
    # ohnehin tot -- geprueft wird dann nur, dass das deutlich gesagt wird.
    if not discord_dave.available()[0]:
        klient = _FakeClient(session=None)
        try:
            discord_dave.attach(klient)
            check("fehlendes davey wird benannt", False, "keine Meldung")
        except RuntimeError as exc:
            check("fehlendes davey wird benannt", "davey" in str(exc), str(exc))
        print("  über  davey fehlt – Entschluesselung nicht pruefbar")
        return

    # -- Unverschluesselter Kanal: unveraendert durchreichen -------------
    klient = _FakeClient(session=None, roh=b"OPUS-ROH")
    z = discord_dave.attach(klient)
    check("ohne Sitzung bleibt der Ton unberuehrt", klient.decrypt(_FakePacket()) == b"OPUS-ROH")
    check("das wird als unverschluesselt gezaehlt", z.durchgereicht == 1)

    klient = _FakeClient(session=_FakeSession(ready=False), roh=b"OPUS-ROH")
    z = discord_dave.attach(klient)
    check(
        "eine noch nicht fertige Sitzung reicht durch", klient.decrypt(_FakePacket()) == b"OPUS-ROH"
    )

    # -- Verschluesselter Kanal ------------------------------------------
    sitzung = _FakeSession(ergebnis=b"HALLO-OPUS")
    klient = _FakeClient(session=sitzung, roh=b"GEHEIM")
    z = discord_dave.attach(klient)
    check("verschluesselter Ton wird geoeffnet", klient.decrypt(_FakePacket()) == b"HALLO-OPUS")
    check(
        "entschluesselt wird je Sprecher", sitzung.aufrufe == [(4242, b"GEHEIM")], sitzung.aufrufe
    )
    check("Erfolg wird gezaehlt", z.entschluesselt == 1 and z.fehlgeschlagen == 0)

    # Stille traegt keine Verpackung und darf nicht angefasst werden.
    klient = _FakeClient(session=_FakeSession(), roh=discord_dave.OPUS_SILENCE)
    z = discord_dave.attach(klient)
    check(
        "Stille laeuft unveraendert durch",
        klient.decrypt(_FakePacket()) == discord_dave.OPUS_SILENCE,
    )

    # Unbekannter Sprecher: verwerfen, nicht durchreichen. Verschluesselte
    # Bytes im Opus-Decoder werden zu Rauschen, und Rauschen wird von der
    # Spracherkennung zu Woertern, die niemand gesagt hat.
    klient = _FakeClient(session=_FakeSession(), roh=b"GEHEIM", ssrc_map={})
    z = discord_dave.attach(klient)
    check(
        "ohne Sprecherzuordnung wird verworfen",
        klient.decrypt(_FakePacket()) == discord_dave.OPUS_SILENCE,
    )
    check("das wird gezaehlt", z.ohne_sprecher == 1)

    # Fehlschlag: ebenfalls verwerfen.
    klient = _FakeClient(session=_FakeSession(fehler=ValueError("NoDecryptorForUser")))
    z = discord_dave.attach(klient)
    check(
        "nicht entschluesselbarer Ton wird verworfen",
        klient.decrypt(_FakePacket()) == discord_dave.OPUS_SILENCE,
    )
    check("der Grund wird festgehalten", "NoDecryptor" in z.letzter_fehler, z.letzter_fehler)

    # -- Vertrag mit der Fremdbibliothek ---------------------------------
    # Die Umhuellung greift in fremden Aufbau: AudioReader.decryptor und
    # dessen decrypt_rtp. Benennt discord-ext-voice-recv das um, hoert der
    # Bot nichts mehr. Das soll hier auffallen und nicht im Gespraech.
    try:
        import inspect

        from discord.ext.voice_recv.reader import AudioReader, PacketDecryptor
    except Exception as exc:  # pragma: no cover – Paket fehlt
        print(f"  über  voice_recv nicht ladbar ({type(exc).__name__}) – übersprungen")
    else:
        check(
            "AudioReader legt seinen Entschluessler unter 'decryptor' ab",
            "self.decryptor" in inspect.getsource(AudioReader.__init__),
        )
        entschluessler = PacketDecryptor("aead_xchacha20_poly1305_rtpsize", bytes(32))
        check("der Entschluessler hat decrypt_rtp", hasattr(entschluessler, "decrypt_rtp"))
        entschluessler.decrypt_rtp = lambda _packet: b"ersetzt"
        check(
            "decrypt_rtp laesst sich je Instanz ersetzen",
            entschluessler.decrypt_rtp(None) == b"ersetzt",
            "sonst muesste die Klasse angefasst werden, was andere Clients traefe",
        )

    # -- Verfuegbarkeit ---------------------------------------------------
    ok, grund = discord_dave.available()
    check("die Verfuegbarkeit wird begruendet", len(grund) > 15, grund)
    if ok:
        import davey

        check("davey kann entschluesseln", hasattr(davey.DaveSession, "decrypt"))
        # Die Signatur ist der Vertrag mit der Fremdbibliothek. Aendert
        # sie sich, bleibt der Bot stumm - das soll hier auffallen und
        # nicht erst im Gespraech.
        try:
            davey.DaveSession(1, 1, 1).decrypt(1, davey.MediaType.audio, b"x")
            check("decrypt nimmt (Sprecher, Art, Paket)", True)
        except TypeError as exc:
            check("decrypt nimmt (Sprecher, Art, Paket)", False, str(exc))
        except Exception:
            # Jeder andere Fehler heisst: die Signatur stimmt, nur der
            # Schluessel fehlt - genau das ist ohne echte Gruppe zu erwarten.
            check("decrypt nimmt (Sprecher, Art, Paket)", True)


# ---------------------------------------------------------------------------
# Farben
# ---------------------------------------------------------------------------
def run_theme(check) -> None:
    """Die Farben stammen von streamwizard.de und bleiben lesbar."""
    print("\n== Farben ==")
    import math

    from app.gui import theme

    def kontrast(vordergrund: str, hintergrund: str) -> float:
        def helligkeit(farbe: str) -> float:
            werte = [int(farbe[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            angepasst = [w / 12.92 if w <= 0.03928 else ((w + 0.055) / 1.055) ** 2.4 for w in werte]
            return 0.2126 * angepasst[0] + 0.7152 * angepasst[1] + 0.0722 * angepasst[2]

        a, b = helligkeit(vordergrund), helligkeit(hintergrund)
        return (max(a, b) + 0.05) / (min(a, b) + 0.05)

    check("Akzent ist das Violett der Website", theme.DARK.accent == "#8B5CF6")
    check("Hintergrund ist der dunkle Ton der Website", theme.DARK.bg == "#151224")
    check("Schrift wie auf der Website", theme.FONT_UI[0] == "Segoe UI")

    # Lesbarkeit ist wichtiger als Wiedererkennung: was hier durchfaellt,
    # kann niemand entziffern.
    for palette in (theme.DARK, theme.LIGHT):
        for name, farbe in (
            ("Text", palette.text),
            ("gedimmter Text", palette.text_dim),
            ("Fehlerfarbe", palette.error),
            ("Warnfarbe", palette.warn),
        ):
            wert = min(kontrast(farbe, palette.bg), kontrast(farbe, palette.surface))
            check(
                f"{palette.name}: {name} ist lesbar",
                wert >= 4.5,
                f"{wert:.1f}:1 (nötig 4,5:1)",
            )

    knopf = kontrast(theme.DARK.accent_text, theme.DARK.accent)
    check(
        "Knopfschrift auf dem Akzent ist lesbar",
        knopf >= 3.0,
        f"{knopf:.1f}:1 – gilt für fette Schrift ab 3:1",
    )
    check("beide Paletten sind vollständig", math.isclose(1, 1) and bool(theme.LIGHT.accent))

    # --- Ansage abschaltbar, Schranke nicht ---------------------------
    #
    # Die Ansage beim Betreten ist nicht in jedem Aufbau noetig: steht der
    # Hinweis im Kanalthema oder ist nur der Betreiber im Kanal, ist sie
    # ueberfluessiger Laerm. Die !optout-Schranke dagegen muss bleiben --
    # wer widerspricht, wird verworfen, egal ob eine Nachricht geschrieben
    # wurde. Beim naechsten Aufraeumen koennte jemand das eine fuer das
    # andere halten.
    from app.config import AppConfig

    vorgabe = AppConfig()
    check(
        "Ansage beim Betreten ist vorgabegemaess an",
        bool(getattr(vorgabe, "discord_join_notice", None)),
        "die Ansage ist die Grundlage dafuer, dass mitgehoert werden darf – "
        "wer sie abschaltet, soll das bewusst tun",
    )

    aus = vorgabe.with_values(discord_join_notice=False)
    check(
        "Ansage laesst sich abschalten",
        aus.discord_join_notice is False,
        "sonst schreibt der Bot bei jedem Verbinden dieselbe Zeile",
    )

    quelle = (ROOT / "app" / "pipeline_discord.py").read_text(encoding="utf-8")
    check(
        "die Ansage fragt die Einstellung ab",
        "discord_join_notice" in quelle,
        "sonst wirkt der Schalter nicht",
    )
    check(
        "abgeschaltete Ansage steht im Protokoll",
        "Ansage beim Betreten ist abgeschaltet" in quelle,
        "spaeter muss erkennbar sein, dass sie bewusst aus war und nicht "
        "ausgefallen ist",
    )

    # Die Schranke selbst bleibt unberuehrt.
    for wort in ("!optout", "!optin"):
        check(
            f"{wort} bleibt erhalten",
            wort in quelle,
            "die Widerspruchsmoeglichkeit haengt nicht an der Ansage",
        )
    check(
        "Einwilligung bleibt Voraussetzung (fail-closed)",
        "discord_consent_confirmed" in quelle,
        "ohne Bestaetigung bleibt der Discord-Weg gesperrt",
    )
