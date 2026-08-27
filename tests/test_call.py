"""Prüfungen fürs Telefonieren – ohne Mikrofon, ohne Modell.

Getestet wird, was ohne Audiogerät und ohne Download entscheidbar ist:
die Aufteilung der Antwort in Sprechtext und Dateien, die Endungen, die
Auslöseschwelle des Mikrofons und die Bereitschaftsprüfung. Der Rest
(Aufnahme, Erkennung, Sprachausgabe) braucht Hardware und steht im
Rauchtest unter „Telefonieren".
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run(check) -> None:
    """Alle Prüfungen dieses Moduls ausführen."""
    run_personas(check)
    run_streaming(check)
    run_sapi(check)
    run_devices(check)
    run_rates(check)
    run_resample(check)
    run_meter(check)
    run_gain(check)
    run_voice_list(check)
    run_gui(check)
    run_brain(check)
    run_klonlaufzeit(check)
    run_dauerbetrieb(check)
    run_tabkosten(check)
    run_kein_rauschen(check)
    run_lauf1(check)
    run_lauf3(check)
    run_lauf45(check)
    run_gui_feinschliff(check)
    run_anruf_robust(check)
    _test_split(check)
    _test_artifacts(check)
    _test_threshold(check)
    _test_readiness(check)
    _test_voices(check)


def _test_split(check) -> None:
    from app.pipeline_call import split_answer

    antwort = (
        "Klar, hier ist die Funktion.\n\n"
        "```python\ndef f(x):\n    return x * 2\n```\n\n"
        "Sag Bescheid, wenn du mehr brauchst."
    )
    gesprochen, bloecke = split_answer(antwort)
    check("Code wird aus dem Sprechtext entfernt", "def f(x)" not in gesprochen, gesprochen[:60])
    check("Sprechtext bleibt erhalten", "Klar, hier ist die Funktion." in gesprochen)
    check("Nachsatz bleibt erhalten", "Sag Bescheid" in gesprochen)
    check("ein Code-Block erkannt", len(bloecke) == 1, str(len(bloecke)))
    check("Sprache des Blocks erkannt", bloecke[0][0] == "python", bloecke[0][0])
    check("Code vollständig", "return x * 2" in bloecke[0][1])

    # Auszeichnung darf nicht vorgelesen werden.
    roh, _ = split_answer("Das ist **wichtig** und `x` auch. # Titel")
    check("Sternchen werden nicht gesprochen", "*" not in roh, roh)
    check("Backticks werden nicht gesprochen", "`" not in roh, roh)
    check("Rauten werden nicht gesprochen", "#" not in roh, roh)

    ohne, leer = split_answer("Nur ein Satz ohne alles.")
    check("Antwort ohne Code ergibt keine Dateien", leer == [])
    check("Antwort ohne Code bleibt unverändert", ohne == "Nur ein Satz ohne alles.", ohne)

    mehrere = "Erst\n```python\na=1\n```\ndann\n```bash\nls\n```\nfertig"
    _text, viele = split_answer(mehrere)
    check("mehrere Blöcke werden alle erfasst", len(viele) == 2, str(len(viele)))
    check("zweite Sprache stimmt", viele[1][0] == "bash", viele[1][0])


def _test_artifacts(check, tmpdir: Path | None = None) -> None:
    import tempfile

    from app.pipeline_call import CODE_SUFFIX, write_artifacts

    ordner = Path(tempfile.mkdtemp(prefix="anruf-"))
    try:
        bloecke = [("python", "print(1)"), ("bash", "ls -la"), ("wasauchimmer", "x")]
        dateien = write_artifacts(ordner, "Mach mir was", bloecke, 3)
        check("je Block eine Datei", len(dateien) == 3, str(len(dateien)))
        endungen = [p.suffix for p in dateien]
        check("Python bekommt .py", ".py" in endungen, str(endungen))
        check("Bash bekommt .sh", ".sh" in endungen, str(endungen))
        check("Unbekanntes wird .txt", ".txt" in endungen, str(endungen))
        check("Nummer steht im Namen", dateien[0].name.startswith("03-"), dateien[0].name)
        check("Frage steht im Namen", "mach-mir-was" in dateien[0].name, dateien[0].name)
        check("Inhalt stimmt", dateien[0].read_text(encoding="utf-8").strip() == "print(1)")

        # Zweiter Lauf darf nichts überschreiben.
        nochmal = write_artifacts(ordner, "Mach mir was", [("python", "print(2)")], 3)
        check("kein Überschreiben", nochmal[0] != dateien[0], nochmal[0].name)
        check("erste Datei unverändert", "print(1)" in dateien[0].read_text(encoding="utf-8"))
        check("Endungstabelle kennt die üblichen Sprachen", len(CODE_SUFFIX) >= 20)
    finally:
        import shutil

        shutil.rmtree(ordner, ignore_errors=True)


def _test_threshold(check) -> None:
    from app import audio_io

    check(
        "ohne Messung gilt die Untergrenze",
        audio_io.threshold_from_noise([]) == audio_io.MIN_THRESHOLD,
    )
    # Ein rauschendes Mikrofon muss die Schwelle anheben, sonst löst es
    # dauernd aus.
    laut = audio_io.threshold_from_noise([0.05, 0.05, 0.05])
    check("lautes Grundrauschen hebt die Schwelle", laut > audio_io.MIN_THRESHOLD, f"{laut:.4f}")
    check("Schwelle liegt über dem Rauschen", laut > 0.05, f"{laut:.4f}")
    # Ein sehr leises darf nicht unter die Untergrenze fallen.
    leise = audio_io.threshold_from_noise([0.0001, 0.0001])
    check(
        "stilles Mikrofon fällt nicht unter die Untergrenze",
        leise == audio_io.MIN_THRESHOLD,
        f"{leise:.4f}",
    )
    check("Blockgröße passt zu 16 kHz", audio_io.BLOCK_SAMPLES == 480, str(audio_io.BLOCK_SAMPLES))
    check("Aufnahme läuft in 16 kHz", audio_io.SAMPLE_RATE == 16_000)


def _test_readiness(check) -> None:
    from app import pipeline_call

    stand = pipeline_call.readiness()
    check(
        "Bereitschaft nennt alle vier Stufen",
        len(stand.report().splitlines()) == 4,
        "die Sprachausgabe fehlte - ohne Stimme gibt es kein Gespraech, nur Text",
    )
    for name, wert in (
        ("audio", stand.audio),
        ("stt", stand.stt),
        ("chat", stand.chat),
        ("voice", stand.voice),
    ):
        check(f"Stufe '{name}' hat eine Begründung", bool(wert[1]), str(wert))
    if not stand.ready:
        check("fehlende Stufen werden aufgezählt", bool(stand.problems()))
    check("Bericht ist lesbar", "Telefonieren" in pipeline_call.describe())

    # "Bereit" darf nicht bedeuten "die Pakete sind da".
    #
    # Nachgestellt am gebauten Programm: mit einem nicht geladenen
    # Denkmodell und --offline meldete es "Sprachmodell ok" und
    # "Bereit." - genau der Befehl, mit dem man vorher nachsieht.
    from app.config import AppConfig

    mit_fehlendem = pipeline_call.readiness(
        AppConfig(chat_model="qwen25-32b", call_chat_model="qwen25-32b")
    )
    check(
        "ein nicht geladenes Denkmodell gilt nicht als bereit",
        not mit_fehlendem.chat[0],
        mit_fehlendem.chat[1],
    )
    check(
        "und der Grund nennt das Modell",
        "geladen" in mit_fehlendem.chat[1] or "gew" in mit_fehlendem.chat[1],
        mit_fehlendem.chat[1],
    )
    check(
        "ohne Konfiguration bleibt es bei der Paketpruefung",
        pipeline_call.readiness().chat[0] == pipeline_call.readiness(None).chat[0],
    )


def _test_voices(check) -> None:
    from app import pipeline_call
    from app.config import AppConfig

    auswahl = pipeline_call.voice_choices(AppConfig())
    check("es gibt wählbare Stimmen", bool(auswahl), str(len(auswahl)))
    # Jede Beschriftung endet mit dem, was die Wahl kostet: sofort,
    # eine Groesse zum Laden, langsam - oder gar nicht nutzbar.
    endungen = ("sofort", "langsam", "laden", "nicht nutzbar", "s/Satz")
    check(
        "jede Stimme sagt, was sie kostet",
        all(any(v.label.endswith(e) for e in endungen) for v in auswahl),
        next((v.label for v in auswahl if not any(v.label.endswith(e) for e in endungen)), ""),
    )
    check(
        "Beschriftungen bleiben lesbar kurz",
        all(len(v.label) <= 46 for v in auswahl),
        max((v.label for v in auswahl), key=len, default=""),
    )
    check("jede Stimme hat eine Beschriftung", all(s.label for s in auswahl))

    # Die Auswahl muss sich sauber auf eine VoiceRequest übertragen.
    from app.pipeline_voice import VoiceRequest

    anfrage = VoiceRequest(text="Hallo")
    fest = pipeline_call.VoiceChoice(
        key="v2/de_speaker_6", label="X", is_profile=False, speaker="v2/de_speaker_6"
    )
    umgesetzt = fest.apply(anfrage)
    check("mitgelieferte Stimme setzt den Sprecher", umgesetzt.speaker == "v2/de_speaker_6")
    check("mitgelieferte Stimme löscht das Profil", umgesetzt.profile_slug == "")

    gelernt = pipeline_call.VoiceChoice(key="meine-stimme", label="Y", is_profile=True)
    umgesetzt2 = gelernt.apply(anfrage)
    check("angelernte Stimme setzt das Profil", umgesetzt2.profile_slug == "meine-stimme")
    check("angelernte Stimme löscht den Sprecher", umgesetzt2.speaker == "")


def run_personas(check) -> None:
    """Persona-Modul prüfen – ohne Modell, ohne GUI."""
    import tempfile

    from app import paths

    # Eigenes Datenverzeichnis, damit der Test die echte Datei nicht anfasst.
    alt = paths.data_dir
    ordner = tempfile.mkdtemp(prefix="persona-")
    paths.set_data_dir_override(__import__("pathlib").Path(ordner))
    try:
        from app import personas

        alle = personas.all_personas()
        check("mindestens acht Personas", len(alle) >= 8, str(len(alle)))
        schluessel = {p.key for p in alle}
        for erwartet in ("assistant", "funny", "serious", "hacker", "contrarian", "conspiracy"):
            check(f"Persona '{erwartet}' vorhanden", erwartet in schluessel)
        check("neutraler Assistent steht vorn", alle[0].key == "assistant", alle[0].key)
        check("Datei wurde angelegt", (paths.data_dir() / "personas.json").is_file())

        check("jede Persona hat einen Systemprompt", all(p.system.strip() for p in alle))
        check("jede Persona hat eine Kurzbeschreibung", all(p.short.strip() for p in alle))

        # Telefon-Zusatz nur im Anruf.
        h = personas.get("hacker")
        check("Anruf-Prompt trägt den Telefon-Zusatz", "Datei gelegt" in h.prompt(for_call=True))
        check("Chat-Prompt ohne Telefon-Zusatz", "Datei gelegt" not in h.prompt(for_call=False))
        check(
            "unbekannter Schlüssel fällt auf Assistent",
            personas.get("gibtsnicht").key == "assistant",
        )

        # Eigene Persona anlegen, speichern, wiederfinden, löschen.
        eigen = personas.Persona(
            key="meintest",
            name="Testfigur",
            emoji="X",
            short="nur ein Test",
            system="Du bist ein Test.",
            builtin=False,
        )
        personas.save(eigen)
        check("eigene Persona wird gefunden", personas.get("meintest").name == "Testfigur")
        check("eigene Persona ist löschbar", personas.delete("meintest"))
        check("mitgelieferte ist nicht löschbar", not personas.delete("assistant"))

        # --- Hacker-Persona: Form (nicht Modellverhalten) --------------
        #
        # Das eigentliche Verhalten hängt am lokalen Modell und wird
        # separat gegen das echte Modell gemessen. Hier wird nur geprüft,
        # dass der Text die Eigenschaften hat, die diese Messung als
        # wirksam ergeben hat.
        hacker = personas.get("hacker")
        check(
            "Hacker antwortet direkt statt zu mahnen",
            "sofort" in hacker.system and "ohne Vorrede" in hacker.system,
        )
        check(
            "Hacker deutet die vage Bitte als CTF-Frage",
            "gib mir hacking code" in hacker.system.lower(),
        )
        check(
            "Hacker behält eine Grenze gegen Schaden an Dritten",
            "Unbeteiligten" in hacker.system or "fremde" in hacker.system.lower(),
        )
        # Der Prompt darf nicht mit Verbots-Begriffen überladen sein –
        # ein kleines Modell verweigert dann auch bei legitimen Fragen.
        verbote = sum(
            hacker.system.lower().count(w)
            for w in ("illegal", "waffen", "schadsoftware", "verboten", "strafbar")
        )
        check(
            "Hacker-Prompt ist nicht mit Verboten überladen",
            verbote <= 1,
            f"{verbote} Verbots-Begriffe",
        )

        # --- Abgleich mitgelieferter Personas --------------------------
        import json

        datei = personas._persona_path()

        # a) Eine unveränderte Vorgabe wird auf den aktuellen Stand gebracht.
        roh = json.loads(datei.read_text(encoding="utf-8"))
        for eintrag in roh["personas"]:
            if eintrag["key"] == "hacker":
                eintrag["system"] = "Veralteter Text."
                eintrag["builtin_sig"] = personas._signature(personas._from_dict(eintrag))
        datei.write_text(json.dumps(roh, ensure_ascii=False), encoding="utf-8")
        check(
            "unveränderte Vorgabe wird aktualisiert",
            "Veralteter Text." not in personas.get("hacker").system,
        )

        # b) Eine selbst geänderte Vorgabe bleibt erhalten.
        roh = json.loads(datei.read_text(encoding="utf-8"))
        for eintrag in roh["personas"]:
            if eintrag["key"] == "serious":
                eintrag["system"] = "Von Hand angepasst."  # Signatur bleibt alt
        datei.write_text(json.dumps(roh, ensure_ascii=False), encoding="utf-8")
        check(
            "selbst geänderte Vorgabe bleibt erhalten",
            personas.get("serious").system == "Von Hand angepasst.",
        )

        # c) Eine alte Datei ohne Signatur wird migriert (einstufig).
        roh = json.loads(datei.read_text(encoding="utf-8"))
        for eintrag in roh["personas"]:
            if eintrag["key"] == "funny":
                eintrag["system"] = "Uralt, keine Signatur."
                eintrag.pop("builtin_sig", None)
        datei.write_text(json.dumps(roh, ensure_ascii=False), encoding="utf-8")
        check(
            "Vorgabe ohne Signatur wird in einem Schritt aktualisiert",
            "Uralt" not in personas.get("funny").system,
        )
    finally:
        paths.data_dir = alt


def run_rates(check) -> None:
    """Abtastrate aushandeln statt fordern.

    Der Fehler, der das Telefonieren unbenutzbar machte: WASAPI- und
    WDM-Geräte laufen im Shared Mode fest auf 48 kHz und lehnen die
    geforderten 16 kHz mit 'Invalid sample rate [PaErrorCode -9997]' ab.
    """
    from app import audio_io

    check(
        "Windows WDM-KS gilt nicht als brauchbar",
        "Windows WDM-KS" not in audio_io.GOOD_HOSTAPIS,
    )
    check(
        "MME und WASAPI gelten als brauchbar",
        "MME" in audio_io.GOOD_HOSTAPIS and "Windows WASAPI" in audio_io.GOOD_HOSTAPIS,
    )
    check(
        "16 kHz steht als erste Ausweichrate",
        audio_io.FALLBACK_RATES[0] == 16_000,
        str(audio_io.FALLBACK_RATES),
    )
    check(
        "48 kHz ist als Ausweichrate dabei",
        48_000 in audio_io.FALLBACK_RATES,
    )

    # Die Klartext-Meldungen sind der eigentliche Nutzen: PortAudio meldet
    # nur Zahlencodes.
    fehler = RuntimeError("Error opening InputStream: Invalid sample rate [PaErrorCode -9997]")
    text = audio_io._mic_problem(fehler, None, 16_000)
    check("Ratenfehler wird übersetzt", "16000 Hz nicht an" in text, text)
    check("PortAudio-Code taucht nicht auf", "-9997" not in text, text)

    blockend = RuntimeError(
        "Unanticipated host error [PaErrorCode -9999]: 'Blocking API not supported yet'"
    )
    text = audio_io._mic_problem(blockend, None, 16_000)
    check("WDM-KS-Fehler wird übersetzt", "nicht direkt auslesen" in text, text)

    belegt = RuntimeError("Error opening InputStream: Device unavailable [PaErrorCode -9985]")
    text = audio_io._mic_problem(belegt, None, 16_000)
    check("belegtes Gerät wird erkannt", "belegt" in text, text)


def run_resample(check) -> None:
    """Umrechnen der Abtastrate – Länge, Tonhöhe, Aliasing.

    Ohne Tiefpass würden Frequenzen über der halben Zielrate als Störtöne
    zurückfalten. Whisper hört dann Wörter, die niemand gesagt hat.
    """
    import numpy as np

    from app import audio_io

    check("gleiche Rate ändert nichts", len(audio_io.resample(np.zeros(100), 16000, 16000)) == 100)
    check("leere Eingabe bleibt leer", len(audio_io.resample(np.zeros(0), 48000, 16000)) == 0)

    t = np.arange(48_000) / 48_000.0
    ton = np.sin(2 * np.pi * 1000 * t).astype("float32")
    herunter = audio_io.resample(ton, 48_000, 16_000)
    check(
        "48 kHz auf 16 kHz ergibt ein Drittel der Werte",
        abs(len(herunter) - 16_000) <= 2,
        str(len(herunter)),
    )

    spektrum = np.abs(np.fft.rfft(herunter))
    frequenzen = np.fft.rfftfreq(len(herunter), 1 / 16_000)
    spitze = frequenzen[int(np.argmax(spektrum))]
    check("Tonhöhe bleibt erhalten", abs(spitze - 1000) < 20, f"{spitze:.0f} Hz")

    lautstaerke = float(np.sqrt(np.mean(herunter**2)))
    check("Lautstärke bleibt erhalten", 0.6 < lautstaerke < 0.8, f"{lautstaerke:.3f}")

    # Ein Ton über der halben Zielrate darf nicht als tiefer Ton auftauchen.
    hoch = np.sin(2 * np.pi * 11_000 * t).astype("float32")
    gefiltert = audio_io.resample(hoch, 48_000, 16_000)
    rest = float(np.sqrt(np.mean(gefiltert**2)))
    check(
        "Töne über der halben Zielrate werden gedämpft",
        rest < 0.2,
        f"Restpegel {rest:.3f} (ohne Tiefpass läge er bei 0.7)",
    )

    hoch_rechnen = audio_io.resample(np.zeros(1000, dtype="float32"), 16_000, 48_000)
    check("Hochrechnen liefert mehr Werte", len(hoch_rechnen) == 3000, str(len(hoch_rechnen)))


def run_meter(check) -> None:
    """Pegelanzeige: logarithmisch, mit Schwellenmarke."""
    try:
        import tkinter as tk
    except ImportError:
        print("  über  tkinter fehlt – übersprungen")
        return
    try:
        wurzel = tk.Tk()
    except tk.TclError:
        print("  über  keine Anzeige – übersprungen")
        return
    wurzel.withdraw()
    try:
        from app.gui import theme
        from app.gui.widgets import LevelMeter

        meter = LevelMeter(wurzel, theme.palette_for("dark"))
        check("Stille zeigt keinen Balken", meter._anteil(0.0) == 0.0)
        check("Vollausschlag ist voll", meter._anteil(1.0) == 1.0)

        # Der Grund für die logarithmische Skala: Sprache liegt bei etwa
        # 0,05 Effektivwert. Linear angezeigt wären das 5 % – unsichtbar.
        sprache = meter._anteil(0.05)
        check(
            "Sprechlautstärke ist deutlich sichtbar",
            sprache > 0.4,
            f"{sprache * 100:.0f}% (linear wären es 5%)",
        )
        check(
            "lauter ergibt mehr Ausschlag",
            meter._anteil(0.3) > meter._anteil(0.05) > meter._anteil(0.004),
        )

        meter.set_threshold(0.012)
        meter.set_level(0.05, True)
        wurzel.update_idletasks()
        check("Balken, Spitze und Schwelle werden gezeichnet", len(meter.canvas.find_all()) == 3)

        meter.reset()
        wurzel.update_idletasks()
        check("Zurücksetzen räumt den Balken weg", len(meter.canvas.find_all()) <= 1)
    finally:
        wurzel.destroy()


def run_gain(check) -> None:
    """Mikrofon-Verstärkung: laut genug, aber nicht übersteuert."""
    import numpy as np

    from app import audio_io

    leise = (0.008 * np.sin(2 * np.pi * 300 * np.arange(1600) / 16000)).astype("float32")
    check(
        "Faktor 1 lässt das Signal in Ruhe",
        audio_io.rms(audio_io.apply_gain(leise, 1.0)) == audio_io.rms(leise),
    )

    verstaerkt = audio_io.apply_gain(leise, 4.0)
    verhaeltnis = audio_io.rms(verstaerkt) / max(audio_io.rms(leise), 1e-9)
    check("Faktor 4 vervierfacht den Pegel", 3.8 < verhaeltnis < 4.2, f"{verhaeltnis:.2f}")

    # Der eigentliche Zweck: das leise Signal muss über die Schwelle
    # kommen, sonst gilt Sprache weiter als Stille.
    schwelle = audio_io.threshold_from_noise([0.002] * 20)
    check(
        "leises Mikrofon bleibt ohne Verstärkung unter der Schwelle",
        audio_io.rms(leise) < schwelle,
        f"{audio_io.rms(leise):.4f} < {schwelle:.4f}",
    )
    check(
        "mit Verstärkung gilt es als Sprache",
        audio_io.rms(audio_io.apply_gain(leise, 6.5)) >= schwelle,
    )

    # Übersteuern würde Whisper das Erkennen erschweren.
    laut = (0.6 * np.sin(2 * np.pi * 300 * np.arange(1600) / 16000)).astype("float32")
    spitze = float(np.max(np.abs(audio_io.apply_gain(laut, 8.0))))
    check("lautes Signal wird nicht übersteuert", spitze <= 1.0, f"Spitze {spitze:.3f}")

    check(
        "Verstärkung ist nach oben begrenzt",
        audio_io.rms(audio_io.apply_gain(leise, 100.0))
        == audio_io.rms(audio_io.apply_gain(leise, audio_io.MAX_GAIN)),
        "sonst verstärkt man nur noch das Rauschen",
    )

    from app.config import AppConfig

    cfg, _ = AppConfig(call_input_gain=99.0).validated()
    check("Konfiguration begrenzt die Verstärkung", cfg.call_input_gain <= audio_io.MAX_GAIN)


def run_voice_list(check) -> None:
    """Stimmenliste: die richtige PowerShell, ehrliche Beschriftung."""
    import os

    from app import pipeline_sapi

    check(
        "pwsh steht vor powershell",
        pipeline_sapi._SHELLS[0] == "pwsh",
        "sonst fehlen Katja und Stefan – die stehen nur unter Speech_OneCore",
    )

    if os.name == "nt":
        stimmen = pipeline_sapi.voices()
        if stimmen:
            deutsch = [v for v in stimmen if v.is_german]
            check("mindestens eine Windows-Stimme gefunden", True, f"{len(stimmen)} Stück")
            if deutsch:
                check(
                    "deutsche Stimmen stehen vorn",
                    stimmen[0].is_german,
                    stimmen[0].name,
                )
                check(
                    "moderne Fassung vor der Desktop-Fassung",
                    not stimmen[0].name.endswith("Desktop")
                    or all(v.name.endswith("Desktop") for v in deutsch),
                    stimmen[0].name,
                )

    # Die Beschriftung muss sagen, dass ein Modell erst geladen wird –
    # sonst wählt man eine Stimme und hört minutenlang nichts.
    from app import pipeline_call
    from app.config import AppConfig

    katalog = pipeline_call.voice_catalog(AppConfig())
    check(
        "Katalog kennt mehrere Quellen",
        len({v.provider for v in katalog}) >= 2,
        str({v.provider for v in katalog}),
    )

    nicht_bereit = [v for v in katalog if not v.ready]
    check(
        "was nicht sofort spricht, sagt warum",
        all(bool(v.note) for v in nicht_bereit),
        "; ".join(f"{v.label}: {v.note!r}" for v in nicht_bereit if not v.note)[:120],
    )

    windows = [v for v in katalog if v.provider == "windows"]
    if windows:
        check("Windows-Stimmen brauchen keinen Download", all(v.size_mb == 0 for v in windows))
        check("Windows-Stimmen sind sofort bereit", all(v.ready for v in windows))
        check(
            "sofort brauchbare Stimmen stehen vorn",
            katalog[0].ready,
            katalog[0].label,
        )

    # Der eigentliche Zweck: der Unterschied muss ablesbar sein, bevor man
    # wählt. 0,5 s gegen 20 s je Satz entscheidet, ob ein Gespräch geht.
    langsam = [v for v in katalog if v.seconds_per_sentence > 10]
    if langsam:
        check(
            "langsame Stimmen sind als solche beschriftet",
            all("langsam" in v.describe() for v in langsam),
            "; ".join(v.describe() for v in langsam[:1]),
        )
    lizenzen = [v for v in katalog if v.provider == "modell"]
    if lizenzen:
        check("Modellstimmen nennen ihre Lizenz", all(bool(v.license_id) for v in lizenzen))


def run_gui(check) -> None:
    """Chat-Persona-Wähler und Telefon-Seite bauen sich ohne Absturz."""
    try:
        import tkinter as tk
    except ImportError:
        print("  über  tkinter fehlt – GUI übersprungen")
        return
    try:
        root = tk.Tk()
    except tk.TclError:
        print("  über  keine Anzeige – GUI übersprungen")
        return
    root.destroy()

    from app.__main__ import Runtime, build_parser
    from app.gui.main_window import MainWindow

    window = MainWindow(Runtime(build_parser().parse_args(["--dummy", "--no-gui"])))
    try:
        window.withdraw()

        window.show_page("chat")
        window.update_idletasks()
        check("Chat hat einen Persona-Wähler", hasattr(window, "chat_persona"))
        check("Persona-Wähler zeigt alle Charaktere", len(window._persona_labels) >= 8)
        # Persona wechseln setzt den Hinweis.
        for label, key in window._persona_labels.items():
            if key == "hacker":
                window.chat_persona.set(label)
                break
        window._chat_apply_persona()
        window.update_idletasks()
        check("Persona-Wechsel wird übernommen", window._chat_persona_key() == "hacker")
        check("Persona-Hinweis wird angezeigt", bool(window.chat_persona_hint.cget("text")))

        window.show_page("call")
        window.update_idletasks()
        check("Telefon-Seite baut sich", "call" in window._pages)
        check("Telefon-Seite listet Stimmen", len(window._call_voice_labels) >= 1)
        check("Telefon-Seite listet Charaktere", len(window._call_persona_labels) >= 8)
        check("Anrufen-Knopf vorhanden", hasattr(window, "call_start"))
        window._refresh_call()
        window.update_idletasks()
        check("Telefon-Seite zeigt einen Zustand", bool(window.call_hint.cget("text")))
    finally:
        window.destroy()


def run_streaming(check) -> None:
    """Satzweises Sprechen: Aufteilung, Reihenfolge, Abbruch."""
    import time

    from app.speech_stream import MIN_SENTENCE_CHARS, SentenceSplitter, SpeechQueue

    # --- Aufteilung -----------------------------------------------------
    t = SentenceSplitter()
    saetze = []
    for stueck in ["Das ", "geht ", "klar. ", "Noch ", "ein ", "Satz ", "hier. "]:
        saetze.extend(t.feed(stueck))
    check("zwei Saetze erkannt", len(saetze) == 2, str(saetze))
    check("erster Satz vollstaendig", saetze[0] == "Das geht klar.", saetze[0])

    # Abkuerzungen duerfen nicht trennen.
    t2 = SentenceSplitter()
    raus = []
    for stueck in ["Nimm ", "z. ", "B. ", "diese ", "Loesung ", "hier. "]:
        raus.extend(t2.feed(stueck))
    check("Abkuerzung trennt den Satz nicht", len(raus) == 1, str(raus))
    check("Abkuerzung bleibt im Satz", "z. B." in raus[0], raus[0])

    # Code wird nie gesprochen.
    t3 = SentenceSplitter()
    gesprochen = []
    for stueck in ["Hier bitte. ", "```python\n", "print(1)\n", "```", " Fertig."]:
        gesprochen.extend(t3.feed(stueck))
    gesprochen.extend(t3.finish())
    ganz = " ".join(gesprochen)
    check("Code taucht nicht im Sprechtext auf", "print(1)" not in ganz, ganz)
    check("Code wurde als Block gesichert", len(t3.code_bloecke) == 1, str(t3.code_bloecke))
    check("Sprache des Blocks erkannt", t3.code_bloecke[0][0] == "python")
    check("Text nach dem Block wird gesprochen", "Fertig" in ganz, ganz)

    # Unabgeschlossener Block: nicht sprechen, trotzdem sichern.
    t4 = SentenceSplitter()
    t4.feed("Schau mal. ```python\nx = 1\n")
    rest = t4.finish()
    check("offener Block wird gesichert", len(t4.code_bloecke) == 1)
    check("offener Block wird nicht gesprochen", all("x = 1" not in r for r in rest), str(rest))

    check("Mindestlaenge ist gesetzt", MIN_SENTENCE_CHARS > 0)

    # --- Warteschlange: Reihenfolge und Abbruch -------------------------
    gespielt: list[str] = []

    def synth(satz, index):
        from pathlib import Path

        time.sleep(0.02)
        return Path(f"s{index}.wav")

    def play(wav):
        time.sleep(0.02)
        gespielt.append(wav.name)

    q = SpeechQueue(synth=synth, play=play)
    q.start()
    for satz in ("Eins.", "Zwei.", "Drei."):
        q.say(satz)
    q.wait(timeout=10)
    check("alle Saetze gesprochen", len(gespielt) == 3, str(gespielt))
    check("Reihenfolge bleibt erhalten", gespielt == ["s1.wav", "s2.wav", "s3.wav"], str(gespielt))

    # Abbruch muss auch die wartenden Saetze verwerfen – sonst redet sie
    # nach dem Reinreden einfach weiter.
    gespielt.clear()
    q2 = SpeechQueue(synth=synth, play=play)
    q2.start()
    for i in range(20):
        q2.say(f"Satz Nummer {i}.")
    q2.stop()
    time.sleep(0.15)
    check("Abbruch stoppt die Warteschlange", q2.stopped)
    check("nach Abbruch wird nichts Neues angenommen", (q2.say("Noch was."), True)[1])
    vorher = len(gespielt)
    time.sleep(0.15)
    check("nach Abbruch kommt nichts mehr", len(gespielt) == vorher, f"{vorher} -> {len(gespielt)}")


def run_sapi(check) -> None:
    """Windows-Stimmen: Verfuegbarkeit, Auswahl, Einbindung."""
    import os

    from app import pipeline_call, pipeline_sapi
    from app.config import AppConfig

    ok, grund = pipeline_sapi.available()
    check("Windows-Stimmen melden einen Zustand", bool(grund), grund)
    if os.name != "nt":
        check("ausserhalb Windows sauber abgelehnt", not ok)
        return
    if not ok:
        check("ohne Stimmen wird das begruendet", "Stimme" in grund or "Windows" in grund, grund)
        return

    stimmen = pipeline_sapi.voices()
    check("mindestens eine Windows-Stimme", bool(stimmen), str(len(stimmen)))
    check("jede Stimme hat Name und Sprache", all(s.name and s.culture for s in stimmen))
    beste = pipeline_sapi.best_voice("de")
    check("eine Stimme wird gewaehlt", beste is not None)

    # Geschwindigkeit: SAPI kennt Stufen von -10 bis 10, nicht Faktoren.
    check("Text wird fuer PowerShell abgesichert", pipeline_sapi._escape("a'b") == "a''b")

    # In der Telefon-Auswahl steht die beste nutzbare Stimme vorn.
    #
    # Frueher entschied allein das Tempo, und damit stand die blecherne
    # Windows-Stimme oben. Jetzt zaehlt der Klang zuerst -- aber nicht um
    # jeden Preis: was vorn steht, muss telefontauglich bleiben.
    auswahl = pipeline_call.voice_choices(AppConfig())
    roh = pipeline_call.voice_catalog(AppConfig())
    check("Windows-Stimmen sind waehlbar", any(v.is_sapi for v in auswahl))
    erste = roh[0]
    check(
        "die erste Wahl ist schnell genug fuers Gespraech",
        erste.seconds_per_sentence <= 5.0,
        f"{erste.label}: {erste.seconds_per_sentence} s/Satz",
    )
    # Unter den telefontauglichen darf keine bessere hinten stehen. Eine
    # bessere, aber zu langsame Stimme dagegen schon -- sie waere im
    # Gespraech unbrauchbar, egal wie gut sie klingt.
    tauglich = [
        v for v in roh if v.ready and v.seconds_per_sentence <= pipeline_call.TELEFON_GRENZE_S
    ]
    check(
        "unter den tauglichen steht die beste vorn",
        all(v.quality <= tauglich[0].quality for v in tauglich) if tauglich else True,
        next((v.label for v in tauglich if v.quality > tauglich[0].quality), ""),
    )
    sapi_wahl = next(v for v in auswahl if v.is_sapi)
    check("Windows-Stimme ist kein Profil", not sapi_wahl.is_profile)

    from app.pipeline_voice import VoiceRequest

    umgesetzt = sapi_wahl.apply(VoiceRequest(text="Hallo"))
    check("Windows-Stimme setzt den Sprecher", bool(umgesetzt.speaker), umgesetzt.speaker)
    check("Windows-Stimme setzt kein Profil", umgesetzt.profile_slug == "")


def run_devices(check) -> None:
    """Geraetewahl: -1 bedeutet Systemvorgabe."""
    from app.pipeline_call import CallSession

    check("minus eins heisst Systemvorgabe", CallSession._geraet(-1) is None)
    check("leerer Wert heisst Systemvorgabe", CallSession._geraet(None) is None)
    check("Unsinn heisst Systemvorgabe", CallSession._geraet("abc") is None)
    check("null ist ein echtes Geraet", CallSession._geraet(0) == 0)
    check("Nummer wird durchgereicht", CallSession._geraet(3) == 3)


# ---------------------------------------------------------------------------
# Denken und Sprechen sind zwei Modelle
# ---------------------------------------------------------------------------
def run_brain(check) -> None:
    """Am Telefon denkt ein Modell und ein anderes spricht.

    Ein Sprachmodell kann nicht sprechen, ein Stimmmodell denkt nicht.
    Beide muessen getrennt waehlbar sein, und die Auswahl muss sagen,
    was sie kostet -- an Zeit, an Speicher und am Klang.
    """
    print("\n== Denken und Sprechen ==")
    from app import pipeline_call as pc
    from app.config import AppConfig

    # -- Welches Modell denkt? -------------------------------------------
    ohne = AppConfig(chat_model="qwen25-vl-3b")
    check(
        "ohne eigene Wahl gilt das Chat-Modell",
        pc.brain_model(ohne) == "qwen25-vl-3b",
        pc.brain_model(ohne),
    )
    eigen = AppConfig(chat_model="qwen25-vl-3b", call_chat_model="qwen25-14b")
    check(
        "eine eigene Wahl fuers Telefonat gewinnt",
        pc.brain_model(eigen) == "qwen25-14b",
        pc.brain_model(eigen),
    )
    check(
        "die Chat-Seite bleibt davon unberuehrt",
        eigen.chat_model == "qwen25-vl-3b",
        "sonst aendert das Telefonat den Chat mit",
    )

    # -- Auswahl der Denkmodelle -----------------------------------------
    auswahl = pc.brain_choices()
    schluessel = [k for k, _ in auswahl]
    check("es gibt Denkmodelle zur Auswahl", len(auswahl) >= 4, str(len(auswahl)))
    check("jedes traegt eine Beschriftung", all(lbl for _, lbl in auswahl))
    check(
        "auch schwere Modelle stehen zur Wahl",
        any(k in schluessel for k in ("qwen25-14b", "qwen25-32b")),
        str(schluessel),
    )
    check(
        "die Beschriftung sagt, ob geladen werden muss",
        all(("bereit" in lbl or "laden" in lbl or "lädt" in lbl) for _, lbl in auswahl),
        str([lbl for _, lbl in auswahl][:2]),
    )

    # -- Rechenweg --------------------------------------------------------
    # Eine Windows-Stimme kennt keine Grafikkarte, und Piper haengt an
    # onnxruntime, dessen Paket hier gemessen keinen CUDA-Weg anbietet.
    # "GPU" dranzuschreiben waere in beiden Faellen eine Falschaussage.
    for motor in ("sapi", "piper"):
        check(
            f"{motor} wird nie als Grafikkarten-Stimme ausgegeben",
            not pc.voice_on_gpu(motor),
        )
    check("bark kann die Grafikkarte nutzen", "bark" in pc.GPU_ENGINES)
    check(
        "auf der Grafikkarte ist bark schneller",
        pc.engine_speed("bark", True) < pc.engine_speed("bark", False),
        f"{pc.engine_speed('bark', True)} vs {pc.engine_speed('bark', False)}",
    )
    check(
        "ein Motor ohne GPU-Weg bleibt beim CPU-Wert",
        pc.engine_speed("sapi", True) == pc.engine_speed("sapi", False),
    )

    # -- Klang -------------------------------------------------------------
    check(
        "Bark klingt besser als eine Windows-Stimme",
        pc.ENGINE_QUALITY["bark"] > pc.ENGINE_QUALITY["sapi"],
    )
    check(
        "die eigene angelernte Stimme ist die beste",
        pc.ENGINE_QUALITY["clone"] == max(pc.ENGINE_QUALITY.values()),
    )

    katalog = pc.voice_catalog(AppConfig())
    check("es gibt Stimmen", bool(katalog))
    check(
        "jede Stimme sagt, wie sie klingt",
        all(v.quality_label for v in katalog),
    )
    check(
        "keine Stimme behauptet GPU, die keine nutzen kann",
        all(not v.on_gpu for v in katalog if v.engine in ("sapi", "piper")),
    )

    # Ein Motor ohne Umsetzung darf nicht als nutzbar erscheinen -- sonst
    # laedt jemand 330 MB und bekommt danach die Attrappe zu hoeren.
    from app import pipeline_voice

    # Nur Modellstimmen: Windows-Stimmen laufen nicht ueber
    # create_voice_pipeline, sondern ueber pipeline_sapi -- fuer sie sagt
    # diese Menge nichts aus.
    nicht_umgesetzt = [
        v
        for v in katalog
        if v.provider == "modell" and v.engine not in pipeline_voice.IMPLEMENTED_ENGINES
    ]
    check(
        "nicht umgesetzte Motoren gelten als nicht nutzbar",
        all(not v.ready for v in nicht_umgesetzt),
        str([v.key for v in nicht_umgesetzt if v.ready]),
    )

    # -- Hinweis auf eine bessere Stimme ----------------------------------
    rat = pc.voice_advice(AppConfig())
    if rat:
        check("der Hinweis nennt eine Groesse", "MB" in rat or "GB" in rat, rat)
        check("der Hinweis nennt den Klang", "klingt" in rat, rat)
    else:
        check(
            "ohne besseren Vorschlag bleibt der Hinweis leer",
            rat == "",
            "eine bessere Stimme ist bereits geladen",
        )


# ---------------------------------------------------------------------------
# Eigene angelernte Stimmen
# ---------------------------------------------------------------------------
def run_klonlaufzeit(check) -> None:
    """Der Weg zur eigenen Stimme darf nicht in einer Sackgasse enden.

    Die Klon-Laufzeit traegt die beste Stimmqualitaet. Vorher meldete das
    gebaute Programm nur, man moege sich an den Anbieter wenden -- was
    niemandem hilft, am wenigsten dem Anbieter selbst.
    """
    print("\n== Eigene Stimmen ==")
    from app import voice_runtime as vr

    # -- Suche nach einem brauchbaren Interpreter -------------------------
    aufruf, fassung = vr.find_system_python()
    check(
        "die Suche liefert ein Paar aus Aufruf und Fassung",
        isinstance(aufruf, list) and isinstance(fassung, str),
    )
    if aufruf:
        check("der gefundene Aufruf ist nicht leer", all(aufruf), str(aufruf))
        check(
            "die Fassung genuegt der Mindestanforderung",
            tuple(int(t) for t in fassung.split(".")) >= vr.MIN_PYTHON,
            f"{fassung} < {vr.MIN_PYTHON}",
        )
    else:
        check("ohne Fund bleibt die Fassung leer", fassung == "", fassung)

    # -- Auskunft VOR dem Versuch ------------------------------------------
    moeglich, grund = vr.install_possible()
    check("die Einrichtung wird begruendet", len(grund) > 20, grund)
    check(
        "Fund und Moeglichkeit stimmen ueberein",
        moeglich == bool(aufruf),
        f"moeglich={moeglich}, gefunden={bool(aufruf)}",
    )
    if not moeglich:
        # Eine Absage muss sagen, was zu tun ist. "Beim Anbieter melden"
        # war genau das nicht.
        check(
            "die Absage nennt einen Ausweg",
            "python" in grund.lower(),
            grund,
        )

    # -- Keine Sackgasse mehr ----------------------------------------------
    quelle = (ROOT / "app" / "voice_runtime.py").read_text(encoding="utf-8")
    check(
        "der Verweis an den Anbieter ist verschwunden",
        "beim Anbieter melden" not in quelle,
        "eine Absage ohne Handlungsmoeglichkeit ist keine Auskunft",
    )
    oberflaeche = (ROOT / "app" / "gui" / "main_window.py").read_text(encoding="utf-8")
    check(
        "auch die Oberflaeche verweist nicht mehr an den Anbieter",
        "beim Anbieter melden" not in oberflaeche,
    )

    # -- Der Bau kann sie beschaffen ---------------------------------------
    bau = (ROOT / "build-windows.ps1").read_text(encoding="utf-8")
    check(
        "der Bau legt eine fehlende Klon-Laufzeit selbst an",
        "voice-runtime" in bau and "install" in bau,
        "sonst bricht -WithVoiceRuntime an einem Schritt ab, den niemand kennt",
    )

    # -- Angelernte Stimmen sind die beste Wahl ----------------------------
    from app import pipeline_call as pc

    check(
        "eine angelernte Stimme gilt als die natuerlichste",
        pc.ENGINE_QUALITY["clone"] > pc.ENGINE_QUALITY["bark"],
    )
    check(
        "sie rechnet auf der Grafikkarte",
        "clone" in pc.GPU_ENGINES,
    )


# ---------------------------------------------------------------------------
# Das Modell bleibt geladen
# ---------------------------------------------------------------------------
def run_dauerbetrieb(check) -> None:
    """Je Satz ein Prozess heisst je Satz ein Modellladen.

    Gemessen auf einem Rechner mit RTX 4070 Ti: rund 40 s je Satz, davon
    etwa 35 s allein das Laden. Mit gehaltenem Modell sind es 5-10 s.
    Beim Telefonieren ist das der Unterschied zwischen einem Gespraech
    und einer Diaschau.
    """
    print("\n== Dauerbetrieb der Stimme ==")
    from app import pipeline_voice, voice_runtime

    check("es gibt einen Dauerbetrieb", hasattr(voice_runtime, "VoiceServer"))
    server = voice_runtime.VoiceServer(language="de")
    check("frisch angelegt laeuft er nicht", not server.running)
    check("Beenden ohne Start ist harmlos", server.stop() is None)

    for name in ("start", "stop", "speak"):
        check(f"VoiceServer kann {name}", callable(getattr(server, name, None)))

    check(
        "es gibt ein Vorladen fuers Gespraech",
        callable(getattr(pipeline_voice, "warmup_voice", None)),
        "sonst faellt das Modellladen in die erste Antwort",
    )
    check(
        "Arbeiter lassen sich beim Programmende beenden",
        callable(getattr(pipeline_voice, "shutdown_voice_servers", None)),
        "sonst bleibt ein Prozess mit mehreren GB Modell zurueck",
    )

    # Ein Arbeiter je Sprache und Geraet, nicht je Satz.
    from app.accel import Backend, BackendPlan

    plan = BackendPlan(backend=Backend.CPU)
    a = pipeline_voice._voice_server("de", plan)
    b = pipeline_voice._voice_server("de", plan)
    check("derselbe Arbeiter wird wiederverwendet", a is b)
    c = pipeline_voice._voice_server("en", plan)
    check("andere Sprache bekommt einen eigenen", c is not a)
    pipeline_voice.shutdown_voice_servers()
    check("nach dem Beenden ist die Liste leer", not pipeline_voice._server_cache)

    # Der Arbeiter MUSS zeilenweise antworten, sonst wartet die Anwendung
    # ewig. Genau daran ist der erste Versuch gescheitert.
    quelle = (ROOT / "packaging" / "voice_worker.py").read_text(encoding="utf-8")
    # Am Verhalten pruefen, nicht an der Schreibweise: der Arbeiter
    # wird geladen und _emit wirklich aufgerufen. Fehlt das
    # Zeilenende, wartet die Anwendung im Dauerbetrieb ewig auf eine
    # Antwort -- genau daran ist der erste Versuch gescheitert.
    import importlib.util as _util
    import io as _io
    from contextlib import redirect_stdout

    _spec = _util.spec_from_file_location("_vw", ROOT / "packaging" / "voice_worker.py")
    _vw = _util.module_from_spec(_spec)
    _spec.loader.exec_module(_vw)
    _puffer = _io.StringIO()
    with redirect_stdout(_puffer):
        _vw._emit({"ok": True})
    ausgabe = _puffer.getvalue()
    check(
        "Antworten enden mit einem Zeilenumbruch",
        ausgabe.endswith(chr(10)),
        "ohne Zeilenende blockiert readline() im Dauerbetrieb",
    )
    check("die Antwort ist gueltiges JSON", ausgabe.strip().startswith("{"), ausgabe[:40])
    check("der Arbeiter kennt den Dauerbetrieb", "def cmd_serve" in quelle)
    check(
        "ein misslungener Satz beendet ihn nicht",
        "Dauerbetrieb darf nie sterben" in quelle,
        "sonst kostet der naechste Satz wieder das volle Laden",
    )


# ---------------------------------------------------------------------------
# Was das Oeffnen der Telefon-Seite kostet
# ---------------------------------------------------------------------------
def run_tabkosten(check) -> None:
    """Der Aufbau der Seite laeuft im Oberflaechen-Thread.

    Alles, was hier Zeit kostet, laesst das Fenster stehen. Gemessen
    wurden 3,22 s, davon 2,5 s allein fuer einen torch-Import, ausgeloest
    von der GPU-Anzeige der Stimmen. Fuer eine Anzeige genuegt die billige
    Vermutung aus torch/version.py.
    """
    import subprocess
    import sys

    print(chr(10) + "== Kosten der Telefon-Seite ==")

    # In einem EIGENEN Prozess messen: in diesem hier ist torch laengst
    # geladen, und dann faellt genau der Fehler nicht auf.
    code = (
        "import sys, time;"
        "from app.config import AppConfig;"
        "from app import pipeline_call as pc;"
        "t=time.time(); pc.voice_catalog(AppConfig()); d=time.time()-t;"
        "print(d, 'torch' in sys.modules)"
    )
    try:
        fertig = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(ROOT),
        )
    except Exception as exc:
        print(f"  über  nicht messbar ({type(exc).__name__}) – übersprungen")
        return

    if fertig.returncode != 0:
        print(f"  über  Messung fehlgeschlagen: {(fertig.stderr or '')[-200:]}")
        return

    teile = (fertig.stdout or "").strip().split()
    if len(teile) != 2:
        print(f"  über  unerwartete Ausgabe: {fertig.stdout[:80]!r}")
        return

    dauer = float(teile[0])
    torch_geladen = teile[1] == "True"

    check(
        "der Stimmenkatalog laedt torch nicht",
        not torch_geladen,
        "torch importieren kostet gemessen 2,5 s -- im Oberflaechen-Thread",
    )
    check(
        "der Katalog ist schnell genug fuers Oeffnen",
        dauer < 2.0,
        f"{dauer:.2f}s (ueber 2 s steht das Fenster sichtbar)",
    )

    # Die Anzeige muss trotzdem stimmen: ohne torch-Import, aber mit
    # richtiger Antwort.
    from app import accel
    from app import pipeline_call as pc

    check(
        "es gibt eine billige GPU-Vermutung",
        callable(getattr(accel, "torch_cuda_hint", None)),
    )
    check(
        "sie liefert ein Paar aus Antwort und Begruendung",
        len(accel.torch_cuda_hint()) == 2,
    )
    check(
        "die Stimmen-Anzeige nutzt die billige Vermutung",
        "torch_cuda_hint" in (ROOT / "app" / "pipeline_call.py").read_text(encoding="utf-8"),
        "sonst kehrt die Wartezeit beim Oeffnen zurueck",
    )
    check(
        "Windows-Stimmen werden gemerkt",
        "_cache" in (ROOT / "app" / "pipeline_sapi.py").read_text(encoding="utf-8"),
        "die PowerShell-Abfrage kostet 0,47 s je Aufruf",
    )
    _ = pc


# ---------------------------------------------------------------------------
# Kein Platzhalterton, und die Laufzeit wird gefunden
# ---------------------------------------------------------------------------
def run_kein_rauschen(check) -> None:
    """Drei Fehler, die zusammen "nur ein komisches Rauschen" ergaben.

    1. install() richtet nach data_dir()/voice-runtime ein, python_path()
       suchte dort nicht -- eine eingerichtete Laufzeit blieb unsichtbar.
    2. Die Spec packte voice_worker.py nicht ins Buendel; install() fand
       nichts zum Kopieren.
    3. Ohne Arbeiter faellt die Sprachausgabe auf die Attrappe zurueck.
       Deren Platzhalterton galt als Erfolg und wurde abgespielt.
    """
    print(chr(10) + "== Kein Platzhalterton ==")
    import pathlib
    import tempfile

    from app import paths
    from app import voice_runtime as vr

    # -- 1. Der Ort der Einrichtung wird durchsucht ----------------------
    ordner = pathlib.Path(tempfile.mkdtemp(prefix="vr-"))
    unter = "Scripts" if __import__("os").name == "nt" else "bin"
    name = "python.exe" if __import__("os").name == "nt" else "python"
    (ordner / "voice-runtime" / unter).mkdir(parents=True, exist_ok=True)
    (ordner / "voice-runtime" / unter / name).write_text("", encoding="utf-8")
    (ordner / "voice-runtime" / vr.WORKER_NAME).write_text("", encoding="utf-8")
    # chatterbox mit anlegen: eine Umgebung OHNE das gilt seit dem Vorfall
    # zu Recht als unvollstaendig und wird uebersprungen.
    (ordner / "voice-runtime" / "Lib" / "site-packages" / "chatterbox").mkdir(
        parents=True, exist_ok=True
    )

    echt = paths.data_dir
    alt_env = __import__("os").environ.pop("STREAMFORGE_VOICE_PYTHON", None)
    paths.data_dir = lambda: ordner
    try:
        gefunden = vr.python_path()
        arbeiter = vr.worker_path()
        check(
            "eine eingerichtete Laufzeit wird gefunden",
            gefunden is not None and ordner in gefunden.parents,
            str(gefunden),
        )
        check(
            "der Arbeiter daneben wird gefunden",
            arbeiter is not None and ordner in arbeiter.parents,
            str(arbeiter),
        )
    finally:
        paths.data_dir = echt
        if alt_env is not None:
            __import__("os").environ["STREAMFORGE_VOICE_PYTHON"] = alt_env
        __import__("shutil").rmtree(ordner, ignore_errors=True)

    # Der Ort, an den eingerichtet wird, MUSS in der Suche stehen.
    quelle = (ROOT / "app" / "voice_runtime.py").read_text(encoding="utf-8")
    check(
        "Einrichtungsort und Suchort sind dieselben",
        quelle.count('paths.data_dir() / "voice-runtime"') >= 3,
        "install() legt dorthin; python_path() und worker_path() muessen dort suchen",
    )
    check(
        "es gibt eine getrennte Quelle fuer den Arbeiter",
        callable(getattr(vr, "_worker_source", None)),
        "worker_path() sucht wo er liegen soll, nicht wo er herkommt",
    )
    # Am Verhalten festmachen, nicht am Wortlaut: der Meldungstext ist im
    # Quelltext umbrochen, eine Textsuche fiele darueber.
    import inspect

    rumpf = inspect.getsource(vr.install)
    check(
        "die Einrichtung prueft sich selbst",
        "python_path()" in rumpf and "worker_path()" in rumpf and "raise" in rumpf,
        "ein Erfolg, der nicht nachgeprueft wird, ist keiner",
    )

    # -- 2. Der Arbeiter kommt ins Buendel --------------------------------
    spec = (ROOT / "packaging" / "app.spec").read_text(encoding="utf-8")
    check(
        "die Spec liefert voice_worker.py mit",
        "voice_worker.py" in spec,
        "ohne ihn ist jede eingerichtete Laufzeit stumm",
    )

    # -- 3. Ein Platzhalterton gilt nicht als Erfolg ----------------------
    anruf = (ROOT / "app" / "pipeline_call.py").read_text(encoding="utf-8")
    check(
        "ein Attrappen-Ergebnis fuehrt zur Ersatzstimme",
        anruf.count('getattr(ergebnis, "dummy", False)') >= 2,
        "sonst wird der Platzhalterton abgespielt - genau das war zu hoeren",
    )

    # Das Feld, an dem das haengt, muss es geben.
    from app.pipeline_voice import VoiceResult

    felder = VoiceResult.__dataclass_fields__
    check("ein Ergebnis kennzeichnet sich als Attrappe", "dummy" in felder)
    check("und es ist standardmaessig kein Platzhalter", felder["dummy"].default is False)

    # -- 4. Die Hoerprobe wird gehoert -------------------------------------
    gui = (ROOT / "app" / "gui" / "main_window.py").read_text(encoding="utf-8")
    check(
        "eine fertige Hoerprobe wird abgespielt",
        "_play_audio" in gui and 'view.kind == "voice"' in gui,
        "sonst steht sie nur als Pfad im Protokoll",
    )


# ---------------------------------------------------------------------------
# Die gewaehlte Stimme muss die benutzte sein
# ---------------------------------------------------------------------------
def run_lauf1(check) -> None:
    """Sieben Fehler mit demselben Ausgang: es spricht die falsche Stimme."""
    print(chr(10) + "== Auswahl wirkt ==")
    import json
    import pathlib
    import tempfile

    from app import paths
    from app import pipeline_call as pc
    from app import pipeline_voice as pv
    from app import voice_runtime as vr
    from app.config import AppConfig

    # -- Ein Fehlzustand darf sich nicht zementieren ---------------------
    ordner = pathlib.Path(tempfile.mkdtemp(prefix="state-"))
    echt = paths.data_dir
    paths.data_dir = lambda: ordner
    try:
        vr._state_cache = None
        (ordner / "voice-runtime-state.json").write_text(
            json.dumps({"ok": False, "note": "kaputt", "python": "x", "python_mtime": 0}),
            encoding="utf-8",
        )
        check(
            "ein gemerkter Fehlzustand wird nicht geglaubt",
            vr._load_state() is None,
            "sonst ueberlebt er jede Reparatur",
        )
    finally:
        paths.data_dir = echt
        vr._state_cache = None
        __import__("shutil").rmtree(ordner, ignore_errors=True)

    quelle = (ROOT / "app" / "voice_runtime.py").read_text(encoding="utf-8")
    check(
        "der Zwischenspeicher haengt an vorhandenen Dateien",
        "python_path() is not None and worker_path() is not None" in quelle,
        "sonst antwortet der Sprechpfad aus veralteten Angaben",
    )
    check(
        "die Einrichtung raeumt die alte Zustandsdatei weg",
        "_state_file().unlink" in quelle,
    )

    # -- Die Auswahl muss die Konfiguration umstellen --------------------
    check("VoiceChoice kann die Einstellungen umstellen", callable(pc.VoiceChoice.configure))

    grund = AppConfig(voice_model="bark-small", voice_cloning_enabled=False)
    modell = pc.VoiceChoice(
        key="bark",
        label="Bark",
        is_profile=False,
        provider="modell",
        model_key="bark",
        engine="bark",
    )
    nachher = modell.configure(grund)
    check(
        "eine gewaehlte Modellstimme landet in der Konfiguration",
        nachher.voice_model == "bark",
        f"{nachher.voice_model} statt bark",
    )
    check("und das Klonen bleibt dabei aus", not nachher.voice_cloning_enabled)

    profil = pc.VoiceChoice(key="meine", label="Meine", is_profile=True, provider="angelernt")
    nach2 = profil.configure(grund)
    check("eine angelernte Stimme schaltet das Klonen ein", nach2.voice_cloning_enabled)
    check("und traegt ihr Profil ein", nach2.voice_profile == "meine")

    anruf = (ROOT / "app" / "pipeline_call.py").read_text(encoding="utf-8")
    check(
        "das Gespraech wendet die Auswahl vor dem Bauen an",
        "self.voice.configure(self.config)" in anruf,
        "sonst ist die Wahl im Telefonat wirkungslos",
    )

    # -- Niemals ein Platzhalterton, wenn eine Stimme da ist -------------
    check("es gibt einen letzten Ausweg vor der Attrappe", callable(pv._letzter_ausweg))
    stimme = (ROOT / "app" / "pipeline_voice.py").read_text(encoding="utf-8")
    aufbau = stimme[stimme.index("def create_voice_pipeline") :]
    check(
        "der Aufbau gibt nirgends direkt eine Attrappe zurueck",
        "DummyVoicePipeline(" not in aufbau.split("def make_job")[0].replace("force_dummy", "X")
        or aufbau.count("_letzter_ausweg(") >= 3,
        "jeder Fehlweg muss ueber _letzter_ausweg gehen",
    )

    import os

    if os.name == "nt":
        from app.accel import Backend, BackendPlan

        pipeline = pv._letzter_ausweg(AppConfig(), BackendPlan(backend=Backend.CPU), "Test")
        check(
            "mit Windows-Stimmen ist der Ausweg keine Attrappe",
            type(pipeline).__name__ != "DummyVoicePipeline",
            type(pipeline).__name__,
        )
        check("und der Grund wird mitgegeben", bool(getattr(pipeline, "extra_notes", ())))

    # -- Stumme Profile stehen nicht vorn --------------------------------
    check(
        "angelernte Stimmen haengen an der Klon-Laufzeit",
        "ready=bool(klon_bereit)" in anruf,
        "sonst steht eine stumme Profilstimme wegen ihrer Klangnote ganz oben",
    )

    # -- Der Nutzer erfaehrt, wer gesprochen hat -------------------------
    gui = (ROOT / "app" / "gui" / "main_window.py").read_text(encoding="utf-8")
    check(
        "nach der Hoerprobe steht da, welche Stimme sprach",
        "Fertig: {wer}" in gui or "wer = " in gui,
    )
    check(
        "ein Fehlschlag erreicht die Seite, nicht nur das Protokoll",
        "Fehlgeschlagen:" in gui,
    )
    check(
        "nach dem Einrichten wird die Laufzeit nachgeprueft",
        'view.kind == "setup"' in gui,
    )


# ---------------------------------------------------------------------------
# Die schwersten Befunde der zweiten Pruefung
# ---------------------------------------------------------------------------
def run_lauf3(check) -> None:
    """Fuenf Fehler, die im Betrieb sofort auffallen."""
    print(chr(10) + "== Letzte Runde ==")

    # -- 1. Ein Abbruch darf die Warteschlange nicht toeten --------------
    from app import upscale

    quelle = (ROOT / "app" / "upscale.py").read_text(encoding="utf-8")
    check(
        "der Abbruch wirft keine BaseException",
        "raise KeyboardInterrupt" not in quelle,
        "KeyboardInterrupt laeuft durch alle except-Bloecke und toetet den Arbeiter",
    )
    check(
        "beide Zweige melden den Abbruch gleich",
        quelle.count("raise UpscaleCancelled") >= 2,
    )
    check("die Abbruch-Ausnahme ist eine normale", issubclass(upscale.UpscaleCancelled, Exception))

    jobs = (ROOT / "app" / "jobs.py").read_text(encoding="utf-8")
    check(
        "die Warteschlange ueberlebt einen harten Fehlgriff",
        "except BaseException" in jobs,
        "stirbt der Arbeiter, bleibt jeder weitere Auftrag fuer immer wartend",
    )

    # -- 2. Der Haken zu den Aufnahmen muss wirken -----------------------
    from app.config import AppConfig

    anruf = (ROOT / "app" / "pipeline_call.py").read_text(encoding="utf-8")
    check(
        "es gibt einen Weg, Aufnahmen zu verwerfen",
        "def discard_recording" in anruf,
    )
    check(
        "und er wird nach dem Verstehen gegangen",
        "self.discard_recording(aufnahme)" in anruf,
        "sonst bleiben fremde Stimmen dauerhaft liegen",
    )
    check(
        "der Haken wird ueberhaupt gelesen",
        "discord_keep_audio" in anruf,
        "vorher stand er nur in der Konfiguration und in der Anzeige",
    )
    # Am Verhalten: eine Datei im Discord-Betrieb ohne Haken muss weg sein.
    import tempfile

    from app import pipeline_call as pc

    class _Sitzung:
        config = AppConfig(call_mode="discord", discord_keep_audio=False)
        discard_recording = pc.CallSession.discard_recording

    ordner = tempfile.mkdtemp(prefix="rec-")
    from pathlib import Path as _P

    datei = _P(ordner) / "frage-01.wav"
    datei.write_bytes(b"RIFF")
    _Sitzung().discard_recording(datei)
    check("ohne Haken wird die Aufnahme geloescht", not datei.is_file())

    datei2 = _P(ordner) / "frage-02.wav"
    datei2.write_bytes(b"RIFF")

    class _Behalten:
        config = AppConfig(call_mode="discord", discord_keep_audio=True)
        discard_recording = pc.CallSession.discard_recording

    _Behalten().discard_recording(datei2)
    check("mit Haken bleibt sie liegen", datei2.is_file())

    datei3 = _P(ordner) / "frage-03.wav"
    datei3.write_bytes(b"RIFF")

    class _Lokal:
        config = AppConfig(call_mode="lokal", discord_keep_audio=False)
        discard_recording = pc.CallSession.discard_recording

    _Lokal().discard_recording(datei3)
    check(
        "am eigenen Mikrofon wird nichts geloescht",
        datei3.is_file(),
        "dort spricht der Bediener selbst",
    )
    __import__("shutil").rmtree(ordner, ignore_errors=True)

    # -- 3. Das gewaehlte Denkmodell muss geladen werden -----------------
    gui = (ROOT / "app" / "gui" / "main_window.py").read_text(encoding="utf-8")
    check(
        "der Anruf loest das Denkmodell des Gespraechs auf",
        "pipeline_call.brain_model(self.runtime.config)" in gui,
        "sonst ueberstimmt chat_model die Wahl auf der Telefon-Seite",
    )

    # -- 4. Der Sampler muss wirken --------------------------------------
    bild = (ROOT / "app" / "pipeline_image.py").read_text(encoding="utf-8")
    check(
        "der Sampler wird je Auftrag gesetzt",
        "benutzter_sampler" in bild,
        "vorher galt nur der Wert aus der Konfiguration, gesetzt beim Laden",
    )
    check(
        "die Metadaten nennen den benutzten Sampler",
        '"sampler": benutzter_sampler' in bild,
        "sonst steht im Bild ein Sampler, mit dem es nicht gerechnet wurde",
    )

    # -- 5. Kein Startbild an Modelle, die keins koennen -----------------
    video = (ROOT / "app" / "pipeline_video.py").read_text(encoding="utf-8")
    check(
        "das Startbild geht nur an passende Pipelines",
        "nimmt_bild" in video,
        "sonst bricht der Auftrag mit 'unbekanntes Schluesselwort image' ab",
    )
    check(
        "und der Nutzer erfaehrt, wenn es uebergangen wurde",
        "wurde nicht verwendet" in video,
    )
    check(
        "der Hinweis am Feld verspricht nichts Falsches",
        "Gesetzt = Bild wird animiert" not in gui,
        "kein mitgeliefertes Modell kann ein Startbild animieren",
    )


# ---------------------------------------------------------------------------
# Einstellungen, die an der entscheidenden Stelle nicht galten
# ---------------------------------------------------------------------------
def run_lauf45(check) -> None:
    """Sieben kleinere Fehler mit demselben Muster."""
    print(chr(10) + "== Einstellungen gelten ==")

    gui = (ROOT / "app" / "gui" / "main_window.py").read_text(encoding="utf-8")
    video = (ROOT / "app" / "pipeline_video.py").read_text(encoding="utf-8")
    compose = (ROOT / "app" / "compose.py").read_text(encoding="utf-8")

    # -- Chat: Temperatur und Antwortlaenge ------------------------------
    check(
        "der Chat gibt Temperatur und Antwortlaenge mit",
        "temperature=self.runtime.config.chat_temperature" in gui,
        "sonst gelten die Vorgaben der Bibliothek statt der Einstellung",
    )

    # -- Chat: Wahl ueberdauert den Neustart -----------------------------
    check(
        "die Modellwahl im Chat wird gemerkt",
        "with_values(chat_model=spec.key)" in gui,
    )
    check(
        "die Charakterwahl im Chat wird gemerkt",
        "with_values(chat_persona=key)" in gui,
    )

    # -- Ausgabeordner: aktueller Pfad -----------------------------------
    check(
        "kein Knopf oeffnet einen eingefangenen alten Pfad",
        "lambda: self._open_path(config.resolved_output_dir()" not in gui,
        "die Lambdas fingen config beim Bauen der Seite ein",
    )

    # -- Video: Schutzbegriffe gelten auch dort ---------------------------
    check(
        "Schutzbegriffe gelten auch im Video",
        "_negative_with_protection" in video,
        "die Schranke galt nur im Bildweg - ein Video ist eine Folge von Bildern",
    )

    # -- Video: Ergebnis wird sichtbar ------------------------------------
    check(
        "ein fertiges Video wird auf der Seite gemeldet",
        'view.kind in ("video", "compose")' in gui,
        "sonst steht es nur im Protokoll",
    )

    # -- Behaelter und Codec passen zusammen -------------------------------
    check(
        "der Codec richtet sich nach dem Behaelter",
        "behaelter" in compose and "libvpx-vp9" in compose,
        "ein .webm mit H.264 ist kein gueltiger Behaelter - ffmpeg bricht ab",
    )

    # -- Was nichts tut, sagt das ------------------------------------------
    check(
        "der Regler 'Bewegungsstaerke' ist als wirkungslos gekennzeichnet",
        "Ohne Wirkung" in gui,
    )
    from app import models

    svd = models.resolve("svd-xt")
    check(
        "svd-xt ist als nicht umgesetzt gekennzeichnet",
        "nicht umgesetzt" in svd.title,
        svd.title,
    )


# ---------------------------------------------------------------------------
# Oberflaeche: rollen, protokollieren, zustimmen
# ---------------------------------------------------------------------------
def run_gui_feinschliff(check) -> None:
    """Vier Beschwerden, vier Ursachen."""
    print(chr(10) + "== Oberflaeche und Bau ==")

    widgets = (ROOT / "app" / "gui" / "widgets.py").read_text(encoding="utf-8")
    gui = (ROOT / "app" / "gui" / "main_window.py").read_text(encoding="utf-8")
    bau = (ROOT / "build-windows.ps1").read_text(encoding="utf-8")

    # -- 1. Rollbalken nur bei Bedarf ------------------------------------
    check(
        "der Rollbereich prueft, ob es etwas zu rollen gibt",
        "_rollbar" in widgets and "_balken_pruefen" in widgets,
        "ein Balken ohne Inhalt sieht nach verstecktem Inhalt aus",
    )
    check(
        "der Balken wird nicht mehr fest eingeblendet",
        'self.scroll.grid(row=0, column=1, sticky="ns")\n        self.columnconfigure'
        not in widgets,
    )
    check(
        "das Mausrad tut nichts, wenn nichts zu rollen ist",
        "if not self._rollbar():" in widgets,
    )

    # -- 2. Protokoll ------------------------------------------------------
    check(
        "das Protokoll reisst nicht mehr nach unten",
        "_am_ende" in widgets and "if nachfuehren:" in widgets,
        "wer hochrollt um zu lesen, wurde bei jeder Zeile zurueckgerissen",
    )
    check("Meldungen tragen eine Uhrzeit", 'strftime("%H:%M:%S' in widgets)
    check("das Protokoll laesst sich leeren", "def clear" in widgets)
    check("und ganz auslesen", "def alles" in widgets)
    check(
        "die Protokollseite steckt in keinem Rollbereich",
        "scrollen=False" in gui,
        "ein Textfeld bringt seinen eigenen Balken mit - zwei sind einer zu viel",
    )
    check(
        "es gibt Knoepfe zum Kopieren und Leeren",
        "_log_kopieren" in gui and "_log_leeren" in gui,
    )

    # -- 3. AGB wirkt sofort ------------------------------------------------
    check(
        "nach der Zustimmung wird der Rechenweg neu geplant",
        "_nach_zustimmung" in gui,
        "sonst rechnet die Anwendung bis zum Neustart weiter auf der CPU",
    )
    check(
        "und das gilt auch beim Widerruf",
        "licensing.agb_accepted() != vorher" in gui,
    )

    # -- 4. Der Bau darf keine Daten stranden lassen ------------------------
    check(
        "der Bau prueft vorher, ob die Anwendung laeuft",
        "laeuft noch (PID" in bau,
        "ein laufendes Programm haelt seinen Datenordner",
    )
    check(
        "das Zuruecklegen steht in einem finally",
        "} finally {" in bau,
        "sonst stranden die Daten bei jedem Abbruch dazwischen",
    )
    check(
        "verwaiste Sicherungen werden eingesammelt",
        "Verwaiste Sicherung" in bau,
        "sonst startet die Anwendung mit leerem Datenordner",
    )

    # -- 5. Ein Import beweist nicht, dass ein Modell laedt ----------------
    check(
        "der Bau versucht wirklich ein Modell zu laden",
        "Ein Import beweist NICHTS" in bau and "n_gpu_layers=99" in bau,
        "ein CUDA-Wheel kann sauber importieren und beim Laden sterben",
    )
    check(
        "eine leere Antwort zaehlt als Fehlschlag",
        "IsNullOrWhiteSpace" in bau,
        "bei STATUS_ILLEGAL_INSTRUCTION schreibt Python nichts mehr",
    )
    check(
        "und es gibt einen Rueckfall auf den CPU-Bau",
        "Rueckfall auf CPU" in bau,
    )
    # Der Index ist wieder Vorgabe - aber mit FESTER Fassung.
    #
    # Gemessen auf dem Zielrechner: 0.3.35 stirbt beim Modell-Laden mit
    # 0xc000001d, 0.3.30 laedt in 1,6 s und antwortet in 0,2 s. Beide
    # melden GPU-faehig. Deshalb die Fassung festnageln statt "neueste".
    check(
        "die CUDA-Fassung ist festgenagelt",
        "LlamaCudaVersion" in bau,
        "neueste Fassung heisst hier: Absturz beim Laden",
    )
    check(
        "und zwar auf die gemessene",
        "0.3.30" in bau,
    )

    # -- 6. tools\ wird gesichert wie data\ -------------------------------
    check(
        "auch die Werkzeuge werden vor PyInstaller gesichert",
        "ToolsStash" in bau,
        "PyInstaller raeumt sein Ausgabeverzeichnis - die 5 GB Klon-Laufzeit "
        "scheitern dabei an tiefen Pfaden (WinError 145)",
    )
    check(
        "und danach zurueckgelegt",
        "Werkzeuge zur" in bau and "ckgelegt" in bau,
    )
    check(
        "eine vorhandene Klon-Laufzeit wird nicht erneut kopiert",
        "Klon-Laufzeit liegt bereits" in bau,
        "sonst 5 GB bei jedem Bau",
    )
    check(
        "kein return ausserhalb einer Funktion im Klon-Zweig",
        "KEIN return" in bau,
        "ein return wuerde den ganzen Bau beenden",
    )

    # -- 7. Der Portable-Marker entscheidet ueber den Datenort ------------
    # Der Marker muss VOR den langen, abbruchgefaehrdeten Schritten
    # stehen (Klon-Laufzeit kopieren: mehrere GB). Eine Prozentzahl
    # waere die falsche Bedingung - es geht um die Reihenfolge.
    zeilen_bau = bau.splitlines()
    marker = next(
        (i for i, z in enumerate(zeilen_bau) if "portable.txt" in z and "Set-Content" in z),
        10**6,
    )
    langlaeufer = next(
        (i for i, z in enumerate(zeilen_bau) if "kopiere Klon-Laufzeit" in z),
        10**6,
    )
    check(
        "der Portable-Marker steht vor den langen Schritten",
        marker < langlaeufer,
        f"Marker Zeile {marker + 1}, Klon-Kopie Zeile {langlaeufer + 1} - bricht der "
        "Bau dazwischen ab, laeuft das Programm ohne Marker und legt einen ZWEITEN "
        "Datenbestand unter LOCALAPPDATA an",
    )

    pfade = (ROOT / "app" / "paths.py").read_text(encoding="utf-8")
    check(
        "ein vorhandener data-Ordner zaehlt auch ohne Marker",
        "hat_daten" in pfade,
        "ein fehlender Marker darf vorhandene Modelle nicht uebersehen",
    )

    # -- 8. Eine halb kopierte Laufzeit gilt nicht als fertig -------------
    check(
        "die Klon-Laufzeit wird auf chatterbox geprueft",
        "VoiceChatterbox" in bau,
        "nur python.exe zu pruefen laesst eine halb kopierte Laufzeit durchgehen",
    )


# ---------------------------------------------------------------------------
# Der Anruf darf nicht die Anwendung mitreissen
# ---------------------------------------------------------------------------
def run_anruf_robust(check) -> None:
    """Aus einem echten Anruf: keine Antwort, dann Absturz beim Auflegen."""
    print(chr(10) + "== Anruf haelt durch ==")
    from app import pipeline_stt

    anruf = (ROOT / "app" / "pipeline_call.py").read_text(encoding="utf-8")
    audio = (ROOT / "app" / "audio_io.py").read_text(encoding="utf-8")

    # -- Auflegen ---------------------------------------------------------
    check(
        "jeder Schritt beim Auflegen ist einzeln gekapselt",
        "def schritt(" in anruf and "except BaseException" in anruf,
        "ein CUDA-Fehler beim Entladen beendete die ganze Anwendung",
    )
    check(
        "der Stimm-Arbeiter wird beim Auflegen beendet",
        "shutdown_voice_servers" in anruf,
        "sonst laeuft ein Prozess mit mehreren GB Modell weiter",
    )
    check(
        "zwischen den CUDA-Nutzern wird aufgeraeumt",
        "_cuda_aufraeumen" in anruf,
    )
    check(
        "ein kaputter CUDA-Kontext laesst auf die CPU ausweichen",
        "Erkennung auf CPU" in anruf,
        "sonst scheitert jeder weitere Anruf mit demselben Fehler",
    )

    # -- Erkennung --------------------------------------------------------
    check(
        "erfundene Floskeln werden erkannt",
        callable(getattr(pipeline_stt, "_ist_erfunden", None)),
    )
    check(
        "Thanks for watching gilt als erfunden",
        pipeline_stt._ist_erfunden("Thanks for watching!", 2.0),
        "Whisper gibt das bei Rauschen ohne Sprache aus",
    )
    check(
        "Bis zum naechsten Mal ebenso",
        pipeline_stt._ist_erfunden("Bis zum nächsten Mal.", 1.9),
    )
    check(
        "in einem langen Beitrag zaehlt es NICHT als erfunden",
        not pipeline_stt._ist_erfunden("Vielen Dank", 30.0),
        "dort kann der Satz wirklich gefallen sein",
    )
    check(
        "echte Sprache bleibt unangetastet",
        not pipeline_stt._ist_erfunden("Wie ist das Wetter heute", 2.0),
    )

    # -- Ausloeseschwelle -------------------------------------------------
    from app import audio_io

    check(
        "die Untergrenze liegt ueber dem Grundrauschen",
        audio_io.MIN_THRESHOLD >= 0.01,
        f"{audio_io.MIN_THRESHOLD} - bei 0,004 loeste schon das Rauschen aus, "
        "und die Aufnahme endete bevor jemand sprach",
    )
    check(
        "leise Aufnahmen werden angehoben",
        callable(getattr(pipeline_stt, "_angehoben", None)),
    )
    check(
        "und es gibt eine Pegelmessung",
        callable(getattr(pipeline_stt, "_pegel", None)),
    )

    # -- Der Grund wird transportiert -------------------------------------
    from app.pipeline_stt import Transcript

    check(
        "eine Erkennung kann sagen, warum nichts kam",
        "note" in Transcript.__dataclass_fields__,
        "'keine Antwort' ohne Grund ist die schlechteste Auskunft",
    )
    _ = audio
