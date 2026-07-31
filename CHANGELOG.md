# Änderungsverlauf

Format angelehnt an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
Versionierung nach [SemVer](https://semver.org/lang/de/).

## [Unveröffentlicht]

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
- Bild-zu-Bild, Inpainting, Hochskalieren (`realesrgan-x4`).
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
