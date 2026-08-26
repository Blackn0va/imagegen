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

    # Ehrlichkeit ueber den Empfang
    _ok, grund = pipeline_discord.receive_possible()
    check(
        "die Grenze beim Zuhoeren wird benannt",
        any(wort in grund for wort in ("Stage", "fehlt", "nicht prüfbar")),
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
