# Änderungsverlauf

Format angelehnt an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
Versionierung nach [SemVer](https://semver.org/lang/de/).

## [Unveröffentlicht]

### Hinzugefügt – Schwarz-Weiß einfärben
- Neuer Bearbeitungsmodus `colorize` in Oberfläche, CLI (`streamforge
  colorize`) und `EditRequest`. Nutzt das bereits geladene Bildmodell, kein
  zusätzlicher Download.
- Das Bild geht entsättigt durch img2img; danach übernimmt
  `merge_luminance()` über YCbCr nur den Farbanteil (Cb, Cr), die Helligkeit
  (Y) kommt unverändert aus der Vorlage. Details bleiben Pixel für Pixel
  erhalten, an harten Kanten entstehen keine Farbsäume.
- Der Farbanteil wird pro Pixel so weit zurückgenommen, wie es nötig ist,
  damit die Rückrechnung nach RGB im Wertebereich bleibt. Ohne das schneidet
  die Umrechnung bei kräftiger Farbe auf sehr dunklen oder sehr hellen
  Stellen ab – und Abschneiden verschiebt genau die Helligkeit, die
  erhalten bleiben soll (gemessen: bis zu 33 Stufen Abweichung).
- Prompt ist freiwillig; ohne Angabe greifen `COLORIZE_PROMPT` und
  `COLORIZE_NEGATIVE`. Der wirklich benutzte Prompt steht als `prompt_used`
  in den Metadaten, sonst wäre das Ergebnis nicht nachstellbar.
- Neue Einstellungen: `image_colorize_strength` (0,55),
  `image_colorize_keep_luminance` (an).

### Hinzugefügt – Diamond-Painting-Vorlage
- Neues Modul `app/diamond.py` und Modus `diamond` in Oberfläche und CLI
  (`streamforge diamond`). Reine Bildrechnung: kein Modell, keine
  Grafikkarte, kein Download.
- Je Ausgangsbild entstehen drei Dateien: Vorlage mit Raster, Symbolen und
  Koordinaten alle zehn Steine, Farbtafel und Farbliste als Text mit
  Stückzahlen und fertiger Größe in Zentimetern.
- Verkleinert wird mit Flächenmittel (`Image.BOX`), nicht mit Lanczos:
  Lanczos schwingt an Kanten über und erzeugt Farbsäume, die es im Motiv
  nicht gibt und die hinterher Plätze in der Farbliste belegen.
- Kein Dithering – Streuung erzeugt einzelne Fremdsteine mitten in einer
  Fläche.
- Farben mit einem Abstand unter `MIN_COLOR_DISTANCE` werden zusammengelegt.
  Ohne das liefert die Farbreduktion denselben Himmel mehrfach (`#96C3E6`,
  `#96C3E5`, `#96C3E7`); ausgedruckt ist das nicht unterscheidbar. Es können
  dadurch weniger Farben herauskommen als angefordert.
- Neue Einstellungen: `diamond_stones` (100), `diamond_colors` (24),
  `diamond_cell_px` (18), `diamond_shape` (`round`), `diamond_symbols` (an).

### Behoben – Speicher lief nach mehreren Bildern voll
- `release_memory()` gibt den Zwischenspeicher von CUDA frei, ohne das
  Modell zu entladen. Der Allokator gibt geholte Blöcke nicht von selbst
  zurück; bleibt das Modell zwischen den Aufträgen liegen (Vorgabe), wuchs
  die Belegung über mehrere Bilder durch Verschnitt, bis das nächste Bild
  nicht mehr hineinpasste. torch wird dabei nicht nachgeladen, wenn es gar
  nicht im Spiel ist.
- `generate()` und `edit()` geben Bild und Pipeline-Ausgabe innerhalb der
  Schleife frei. Vorher lag das fertige Bild noch im Speicher, während das
  nächste der Reihe gerechnet wurde.
- Neues `finish_pipeline()` räumt nach **jedem** Auftrag auf, nicht nur beim
  Entladen. Bei gehaltenem Modell setzt `free_between_jobs()` zusätzlich die
  Auslagerung über `maybe_free_model_hooks()` zurück.
- `JobQueue` lässt den Handler nach dem Auftrag los – die Closure hielt
  Konfiguration, Backend-Plan und die vollständige Anfrage fest – und
  begrenzt die Zahl der erledigten Aufträge auf 200.

### Hinzugefügt – Inhalte für Erwachsene
- **Nacktheit und erotische Darstellungen sind Vorgabe.** Die
  Inhaltsprüfung des Modells (`safety_checker`, den SD 1.5 mitbringt und der
  sonst jedes betroffene Bild schwärzt) wird abgeschaltet. SDXL und FLUX
  bringen keine solche Komponente mit.
- Abschalten über Einstellungen → „Inhalte für Erwachsene", über
  `nsfw_enabled: false` oder je Aufruf mit `--no-nsfw`.
- Keine Freigabekette: die Anwendung läuft lokal und wird nicht
  weitergegeben, also gibt es niemanden, dem gegenüber eine Zustimmung zu
  dokumentieren wäre. Eine Einstellung genügt.
- Der tatsächlich verwendete Negativ-Prompt steht jetzt in den
  Bild-Metadaten (`negative_prompt_used`) – sonst wäre ein Bild mit
  angehängten Schutzbegriffen nicht reproduzierbar.

### Hinzugefügt – Modelle für Nacktheit und Pornografie
- Sieben geprüfte Feinabstimmungen in der Registrierung: `pony-v6`,
  `noobai-xl`, `realvis-xl`, `juggernaut-xl`, `nsfw-gen`,
  `realistic-vision`, `dreamshaper`. Jeder Eintrag ist gegen die
  Hugging-Face-API geprüft – Repo vorhanden, Format, Lizenzangabe der
  Modellkarte und die Größe **nach** dem Dateifilter aus `select_files()`
  (nicht die Roh-GB des Repos, die bei RealVisXL 25,8 GB betragen, nach
  Filter aber 6,5 GB).
- Abdeckung von 2,6 GB / 3 GB VRAM (`dreamshaper`) bis 8 GB
  (`nsfw-gen`), realistisch wie Anime.
- **Einzeldatei-Checkpoints** (`.safetensors` statt diffusers-Ordner) laufen
  jetzt. Das ist die übliche Bauart auf Sammelplattformen und war der Grund,
  warum Pony V6 – das stärkste Modell dieser Auswahl – bisher nicht nutzbar
  war. Neue `ModelSpec`-Felder: `single_file`, `single_file_class`,
  `single_file_config`.
- Für `from_single_file` braucht diffusers die Bauplan-Dateien eines
  Referenz-Repos. Der Weg über den Hugging-Face-Cache scheitert unter
  Windows ohne Entwicklermodus mit „WinError 1314: Dem Client fehlt ein
  erforderliches Recht“ (Symlinks). Deshalb holt `models.ensure_reference_config()`
  die Dateien mit dem vorhandenen eigenen Downloader in
  `models/configs/<repo>` – 3 MB, danach ist der Ladevorgang offline-fähig
  (nachgemessen mit `--offline`).

### Hinzugefügt – Inhaltssperre (nicht abschaltbar)
- Neues Modul `app/contentgate.py`. Es lehnt Aufträge ab, die Begriffe für
  Minderjährige mit sexuellen Begriffen verbinden – geprüft werden Prompt
  **und** Negativ-Prompt, bei Bild, Video und Bearbeiten, jeweils **vor**
  dem Laden des Modells. Erfasst sind deutsche und englische Begriffe,
  Altersangaben unter 18 („12 years old", „14 jahre alt") und einfache
  Verschleierungen (`n4ked t3en`).
- Auf Fehlalarme ausgelegt, nicht auf maximale Reichweite: `nude woman, 25
  years old` läuft, `a child playing football` läuft, `nude woman, kindness
  in her eyes` läuft, `lolita fashion dress, adult model` läuft. `girl`,
  `boy` und `young` stehen bewusst nicht auf der Liste. `kind` wird als
  ganzes Wort geprüft (sonst träfe es „kindness"), deutsche
  Zusammensetzungen wie „kinderzimmer" über den Wortanfang, deutsche
  Beugungen wie „nacktes"/„erotische" über eine eigene Präfixliste.
- `nsfw_block_minors` lässt sich nicht abschalten: ein `false` in der
  Konfiguration wird beim Laden zurückgesetzt und gemeldet (gleiche Bauart
  wie `voice_require_consent`).
- Bei zugelassenen Erwachsenen-Inhalten hängt die Anwendung Schutzbegriffe
  an den Negativ-Prompt (`nsfw_protective_negative`, abschaltbar).
- Warteschlange unterscheidet jetzt bewusste Ablehnungen von Programm-
  fehlern (`exc.expected`): Warnung statt Stacktrace im Protokoll.

### Hinzugefügt – Bestehende Bilder bearbeiten
- **Neue Seite „Bild bearbeiten"** und zwei Unterbefehle (`edit`, `upscale`)
  mit drei Modi:
  - *Vergrößern* um Faktor 2, 4 oder 8 über **Real-ESRGAN**. Die
    Netzarchitektur (RRDBNet) ist in `app/upscale.py` in reinem torch
    nachgebaut, damit kein weiteres Paket mit eigener Lizenz und eigener
    torch-Fassung dazukommt. Fehlen die Gewichte oder passen sie nicht, wird
    **Lanczos** benutzt und das Verfahren steht im Ergebnis.
  - *Nach Prompt umarbeiten* (img2img) mit einstellbarer Stärke.
  - *Bereich ersetzen* (Inpainting) über eine Maskendatei.
- Mehrere Dateien je Auftrag; das Ausgangsbild wird nie überschrieben, der
  Zieldateiname enthält den Namen der Quelle.
- Vorschau des Ausgangs- und des Ergebnisbildes in der Oberfläche, Knopf
  „Ergebnis weiterbearbeiten" auf der Bildseite.
- Grafikspeicher: img2img und Inpainting entstehen aus den **bereits
  geladenen** Modulen des Bildmodells (Konstruktor mit `pipe.components`).
  Gemessen: 0,0 s und kein zusätzlicher Grafikspeicher. Der naheliegende Weg
  `DiffusionPipeline.from_pipe` kopiert dagegen die Gewichte – im Test 280 s
  und 5 GB extra für dasselbe Ergebnis; er bleibt nur Rückfallebene.
- Inpainting lieferte 1024x1024 statt der Größe der Vorlage: ohne `width`
  und `height` nehmen die Inpaint-Pipelines ihre eigene Vorgabe und skalieren
  das Bild hoch. Beide Angaben werden jetzt gesetzt, sofern die Pipeline sie
  kennt (die Signatur entscheidet, nicht eine Modell-Liste).
- Das Vergrößern rechnet kachelweise und halbiert bei Speichermangel
  selbsttätig die Kachelgröße, bevor es sich beschwert.
- Neue Einstellungen: `upscale_factor`, `upscale_tile`, `upscale_use_model`,
  `upscale_refine`, `image_edit_strength`, `image_edit_refine_strength`.
  `upscale_model` steht jetzt auf `realesrgan-x4` statt leer.

### Behoben – Startzeit
- **Der Start wartete auf den torch-Import.** `Runtime.__init__` fragte über
  die Backend-Kette `torch.cuda.is_available()` – dafür muss torch geladen
  werden (gemessen: 2,1 s warm, **18,0 s** beim ersten Start einer Sitzung),
  und zwar bevor überhaupt ein Fenster erschien. Der Start beantwortet die
  Frage jetzt aus `torch/version.py` (eine Textdatei), die echte Prüfung
  läuft im Hintergrund, sobald das Fenster steht, und korrigiert die Anzeige.
- **Zwei PowerShell-Prozesse für Grafikkarten und NPUs** sind jetzt einer,
  und der Hardware-Bericht wird als `hardware-cache.json` im Datenordner
  behalten. Beim nächsten Start steht er sofort zur Verfügung; die
  Neuerkennung läuft im Hintergrund.
- Gemessen auf dem Entwicklungsrechner: Aufbau der Laufzeit von **2,8 s
  (warm) bzw. ~19 s (kalt) auf 0,20 s**.
- Stellt sich im Hintergrund heraus, dass CUDA doch nicht nutzbar ist, stellt
  die Bild-Pipeline beim Laden auf CPU und `float32` um, statt beim ersten
  `.to("cuda")` abzustürzen.

### Behoben
- **Stimmprofile verschwanden mitten im Betrieb.** `is_portable()` machte bei
  jedem Pfadzugriff einen Schreibtest neben der .exe. Schlug der auch nur
  einmal fehl (gesperrte Datei, Virenscanner), wechselte das Datenverzeichnis
  von `<exe>\data` nach `%LOCALAPPDATA%\StreamForge` – Profile und Modelle
  waren scheinbar weg. Die Entscheidung wird jetzt einmal getroffen und
  festgehalten.
- **Verfahren „Nachtrainieren" war eine Sackgasse.** Der Dialog bot es an,
  aber es ist nicht umgesetzt und verlangte 600 s Material; solche Profile
  ließen sich nie anlernen und meldeten stattdessen „zu wenig Material".
  Die Auswahl ist raus, bestehende Profile lassen sich mit einem Knopf
  (oder `voice-profile set-mode`) auf „Referenz" umstellen.
- Nach jedem Auffrischen war die Auswahl im Profilbaum weg, weshalb der
  nächste Knopfdruck „zuerst ein Profil auswählen" meldete. Auswahl bleibt
  jetzt erhalten.

### Behoben (Vorlauf)
- **Oberfläche stand beim Öffnen von „Stimme anlernen" über eine Minute.**
  Die Seite prüfte die Klon-Laufzeit im Oberflächen-Thread, und diese Prüfung
  importierte im Unterprozess torch und chatterbox (gemessen: 81 s, davon
  17 s allein für torch). Jetzt: Schnellprüfung ohne Importe (0,16 s),
  Ergebnis auf Platte zwischengespeichert, Nachprüfung im Hintergrund.
  Seitenwechsel liegen damit bei höchstens 130 ms.
- Rückgaben aus Hintergrundarbeit liefen über `after()` aus einem
  Fremd-Thread – bei tkinter nicht threadsicher („main thread is not in main
  loop"). Sie gehen jetzt über dieselbe Ereignispumpe wie die Aufträge.
- Klonstimmen luden das mehrere GB große Modell **je Satz neu**, weil pro
  Satz ein eigener Prozess startete. Jetzt gehen alle Sätze in einem Aufruf
  mit; drei Sätze brauchen damit einen Ladevorgang statt drei.

### Hinzugefügt
- **Aufnahmen verwalten**: eigene Liste je Profil mit Dauer, Abtastrate und
  Grund, falls eine Aufnahme unbrauchbar ist; Hinzufügen, Entfernen und
  Ordner öffnen. Nach dem Hinzufügen wird die Referenz automatisch neu
  aufgebaut – das ist der „weiter anlernen"-Schritt.
- **Stimme verfeinern**: Ausdruck, Führung, Streuung und Referenzlänge je
  Profil einstellbar und gespeichert, dazu eine Hörprobe auf Knopfdruck.
- Die Referenz entsteht jetzt aus **mehreren** Aufnahmen (die längsten, bis
  die Ziellänge erreicht ist) statt nur aus der längsten – mehr Laute und
  Tonhöhen ergeben eine treffendere Stimme.
- Zebrastreifen in den Listen, Zustandsfarben für die Laufzeit-Anzeige,
  Anzeige offener Aufträge in der Fußzeile, größere Zeilenhöhe.

### Offen
- Echtes Nachtrainieren von Stimmen (`finetune`) – bricht derzeit mit klarer
  Meldung ab, statt ein wertloses Artefakt zu schreiben.
- DirectML-Zweig: ONNX-Export als eigener, abbrechbarer Auftrag.
- Maskenwerkzeug in der Oberfläche: eine Maske fürs Inpainting muss derzeit
  in einem Bildprogramm gemalt und als Datei ausgewählt werden.
- Entscheidung zur Sprachausgabe: bei Bark (MIT) bleiben oder Piper als
  eigenständiges Programm einbinden (GPL-3.0, siehe MODELS.md).

## [0.1.0] – 2026-07-30

Erste vollständige Fassung: Bild und Sprache erzeugen echte Ergebnisse,
Video ist eingebaut und wartet auf ein geladenes Videomodell.

### Hinzugefügt
- **Bild**: diffusers-Pipeline für SD 1.5, SDXL und FLUX; Sampler-Wahl,
  fp16, Attention-Slicing, VAE-Tiling, automatische Auslagerung bei knappem
  Grafikspeicher, Abbruch pro Diffusionsschritt, PNG mit Prompt/Seed/Modell
  in den Metadaten.
- **Video**: Wan 2.1, CogVideoX und AnimateDiff; Einzelbilder werden über
  ffmpeg zusammengesetzt.
- **Sprache**: Bark (MIT) als Vorgabe – kann Deutsch, anders als Kokoro.
  Piper ist eingebaut, aber wegen GPL-3.0 der Laufzeit fail-closed gesperrt.
- **Klonstimmen**: Chatterbox (MIT) in einer getrennten Laufzeit, als eigener
  Prozess aufgerufen. Anlernen erzeugt eine aufbereitete Referenzaufnahme.
- **Stimmprofile** mit Einwilligungs-Nachweis (Name, Zweck, Datum,
  Prüfsumme des Wortlauts); ohne gültigen Nachweis wird weder angelernt noch
  erzeugt. Löschen ist der Widerrufsweg.
- **Hardware-Erkennung**: NVIDIA über `nvidia-smi`, AMD/Intel über CIM, NPUs
  über PnP und OpenVINO; Eignungsstufe mit Modellempfehlung vor dem Download.
- **Backend-Kette** CUDA → DirectML → CPU mit Erststart-Bremse: im
  Auto-Modus wird kein mehrere GB großer Export beim ersten Start angestoßen.
- **Warteschlange** mit Fortschritt, Abbruch und gedrosselter Fehlerausgabe.
- **Modellverwaltung**: Registrierung mit Lizenzstufe (frei / bedingt /
  gesperrt), abbrechbarer Streaming-Download mit Wiederaufnahme,
  Vollständigkeits-Marker, `prune` und `verify`.
- **Lizenz-Tore**, fail-closed: NVIDIA-Laufzeit, Stimmklonen und Piper
  brauchen ausdrückliche Zustimmung.
- **AGB** mit Zustimmung beim ersten Start; Fassung über Text-Hash, geänderter
  Wortlaut erzwingt neue Zustimmung.
- **ffmpeg** als LGPL-Build eingebettet, mit Prüfung am Binary gegen
  GPL-/nonfree-Bestandteile.
- **Oberfläche** (tkinter) mit zehn Seiten, Einzelinstanz-Sperre,
  TLS-Vertrauensanker über den Windows-Zertifikatspeicher.
- **Build-Skript** mit Schaltern, Schutz der Nutzerdaten beim Bauen und
  Abbruch bei einem CPU-Wheel im GPU-Build.
- **Rauchtest** mit 46 Prüfungen, ohne Netz und ohne GPU lauffähig.

### Behoben
- CPU-Wheel im GPU-Build: `requirements.txt` vor dem CUDA-Index installiert,
  wodurch pip das cu-Wheel übersprang. Reihenfolge umgestellt, Gegenprobe
  eingebaut.
- Modell-Download ließ sich nicht abbrechen: `snapshot_download` reicht
  `tqdm_class` nur an die Dateizähler-Leiste weiter. Eigener Streaming-
  Downloader mit Abbruch je Block.
- Halb geladene Modelle galten als vollständig; jetzt Teil- und Fertig-Marker.
- Dateifilter zog fp32- und fp16-Gewichte, `.bin`-Doppelungen und OpenVINO
  mit: 46 GB statt 6,5 GB je SDXL-Repo.
- `UnicodeEncodeError` auf cp1252-Konsolen; Ausgabe wird auf UTF-8 gestellt.
- „USB Input Device" wurde als NPU erkannt (Teilstring), virtuelle Monitore
  als GPU.
- `--dummy` wirkte nicht, weil es nie an die Pipelines gereicht wurde.
- Bundle scheiterte an `AutoPipelineForText2Image`: dessen Zuordnungstabelle
  zieht Kolors mit, das beim Import `torch.jit.script` ruft und Quelltext
  braucht.
- Bark-Absturz „tensors on different devices" durch `.to(cuda)` zusammen mit
  Auslagerung.
- Klonstimme lief in ein Zeitlimit, weil der 6-GB-Modelldownload in das
  Synthese-Fenster fiel; Vorbereitung ist jetzt ein eigener Schritt und der
  Wachhund misst Stillstand statt Gesamtdauer.
- Wasserzeichen-Paket `perth` scheiterte an fehlendem `pkg_resources`
  (setuptools ab 81 liefert es nicht mehr mit).
- Klonstimme schrieb Float-WAV, das die Standardbibliothek nicht lesen kann;
  jetzt 16-Bit-PCM.
- Modelle der Klon-Laufzeit landeten im Benutzerprofil statt im
  Anwendungsordner.
- Build-Skript hätte beim Bauen die Nutzerdaten gelöscht und war unter
  Windows PowerShell 5.1 nicht lesbar (UTF-8 ohne BOM).
