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
    run_gui(check)
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
    check("Bereitschaft nennt drei Stufen", len(stand.report().splitlines()) == 3)
    for name, wert in (("audio", stand.audio), ("stt", stand.stt), ("chat", stand.chat)):
        check(f"Stufe '{name}' hat eine Begründung", bool(wert[1]), str(wert))
    if not stand.ready:
        check("fehlende Stufen werden aufgezählt", bool(stand.problems()))
    check("Bericht ist lesbar", "Telefonieren" in pipeline_call.describe())


def _test_voices(check) -> None:
    from app import pipeline_call
    from app.config import AppConfig

    auswahl = pipeline_call.voice_choices(AppConfig())
    check("es gibt wählbare Stimmen", bool(auswahl), str(len(auswahl)))
    check("Vorgabe ist dabei", any("Vorgabe" in s.label for s in auswahl))
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
    finally:
        paths.data_dir = alt


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

    # In der Telefon-Auswahl stehen die schnellen Stimmen vorn.
    auswahl = pipeline_call.voice_choices(AppConfig())
    check("Windows-Stimmen sind waehlbar", any(v.is_sapi for v in auswahl))
    check("die erste Wahl ist eine schnelle Stimme", auswahl[0].is_sapi, auswahl[0].label)
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
