# StreamForge Studio

Lokale Erzeugung von **Bild, Video und Sprache** auf dem Rechner des Nutzers.
Keine Cloud, keine API-Schlüssel. Windows-Desktop, Auslieferung als
PyInstaller-Bundle (`--onedir`), Nutzer braucht kein Python.

**Stand:** Bild und Sprache erzeugen echte Ergebnisse, Video ist eingebaut und
wartet nur auf ein heruntergeladenes Videomodell. Fällt eine Voraussetzung
aus (Modell fehlt, Lizenz gesperrt, Bibliothek nicht installiert), schaltet
die Anwendung auf eine Platzhalter-Attrappe um **und schreibt den Grund in
die Ausgabe** – ein Farbverlauf ohne Erklärung sieht sonst wie ein Fehler aus.

| Bereich | Umsetzung | Vorgabemodell |
|---|---|---|
| Bild | diffusers (SD 1.5, SDXL, FLUX) | `sdxl-base` |
| Video | diffusers (Wan 2.1, CogVideoX, AnimateDiff) | `wan-t2v-1.3b` |
| Sprache | Bark über transformers | `bark-small` (MIT, kann Deutsch) |
| Stimme anlernen | Profile, Aufnahmen, Einwilligung – Anlernen noch Attrappe | `chatterbox` |

## Schnellstart (Entwicklung)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m app info                 # Hardware, Backend, Pfade, Lizenzen
python -m app                      # Oberfläche
python -m app image "ein Leuchtturm im Sturm"
python -m app voice "Guten Abend zusammen."
```

GPU-Pfad (NVIDIA). **Reihenfolge ist entscheidend** – torch zuerst über den
CUDA-Index, sonst installiert pip das CPU-Wheel von PyPI und betrachtet
`torch>=2.6,<3` danach als erfüllt. Ergebnis wären CUDA-DLLs neben einer
Anwendung, die trotzdem auf der CPU rechnet:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Die letzte Zeile muss `…+cu126 12.6 True` zeigen, nicht `…+cpu None False`.
`build-windows.ps1` erledigt das in dieser Reihenfolge und bricht ab, wenn
doch ein CPU-Wheel im Venv landet.

## Bauen

```powershell
.\build-windows.ps1 -Clean                                   # Vollbau mit CUDA + Modell
.\build-windows.ps1 -SkipModelDownload -WithCuda:$false      # schneller Testbau
.\build-windows.ps1 -NoGui                                   # nur Kommandozeile
.\build-windows.ps1 -Model ssd-1b -FfmpegDir C:\ffmpeg-lgpl\bin
```

Ergebnis: `dist\StreamForge\StreamForge.exe`.

Schalter: `-Python`, `-Name`, `-Model`, `-WithCuda`, `-SkipModelDownload`,
`-NoGui`, `-Clean`, `-FfmpegDir`, `-CudaIndexUrl`, `-Console`.

Zwei Sicherungen sind eingebaut, weil beides schon schiefgegangen ist:

* **Nutzerdaten überleben den Bau.** Im Portable-Modus liegen Modelle,
  Konfiguration und Ausgaben in `dist\<Name>\data` – schnell zweistellige GB.
  PyInstaller räumt den Ausgabeordner mit `--noconfirm` ab, deshalb wird
  `data\` vorher weggetragen und danach zurückgelegt (auch bei Abbruch).
  `-Clean` löscht dagegen absichtlich alles.
* **Kein stiller CPU-Fehlbau.** Nach der Installation wird geprüft, ob
  wirklich ein cu-Wheel im Venv liegt; sonst bricht der Bau mit Begründung
  ab, statt eine .exe mit CUDA-DLLs und CPU-torch auszuliefern.

Nach Codeänderungen muss neu gebaut werden – die .exe enthält eine Kopie des
Programms, nicht den Ordner `app\`.

## Aufbau

```
app/
  __main__.py         CLI, Argumente, Startreihenfolge, Logging
  config.py           eingefrorene Dataclass, JSON + Umgebungsvariablen
  paths.py            frozen vs. Entwicklung, portable vs. %LOCALAPPDATA%
  accel.py            DLL-Suchpfad, GPU/NPU/CPU-Erkennung, Backend-Kette
  models.py           Registrierung mit Lizenzstufe, Download, Cache
  pipeline_image.py   Text -> Bild        (Schnittstelle + Attrappe)
  pipeline_video.py   Bild/Text -> Video  (Schnittstelle + Attrappe)
  pipeline_voice.py   Text -> Sprache     (Schnittstelle + Attrappe)
  voice_profiles.py   Stimme anlernen: Aufnahmen, Einwilligung, Artefakte
  compose.py          ffmpeg: Video schreiben, Ton muxen (LGPL-Pfad)
  jobs.py             Warteschlange, Fortschritt, Abbruch, Drosselung
  licensing.py        Zustimmung zu Drittanbieter-Lizenzen, fail-closed
  single_instance.py  Kernel-Mutex (Windows) bzw. flock
  nettrust.py         truststore -> certifi
  gui/                tkinter-Oberfläche (Seiten, Stil, Bausteine)
run_app.py            dünner Starter für PyInstaller
build-windows.ps1     Build mit Schaltern
packaging/app.spec    PyInstaller-Beschreibung (über Umgebungsvariablen)
```

## Wie die Basis die harten Punkte löst

**Pfade unter PyInstaller.** `paths.py` unterscheidet `exe_dir` und
`bundle_dir`. Modelle, Konfiguration, Ausgaben und Logs liegen **nie** im
`_MEIPASS`-Verzeichnis, sondern im Portable-Ordner `data\` neben der .exe
(Marker `portable.txt`) oder in `%LOCALAPPDATA%\StreamForge`.

**GPU-DLLs vor dem ersten Modell-Laden.** `accel.prepare_gpu_dll_path()` ist
idempotent (Modul-Global), sucht in beiden Welten – `exe_dir\cuda`,
`exe_dir\_internal\cuda` und die `bin`-Ordner der `nvidia-*-cu12`-Pip-Pakete
über `importlib.util.find_spec("nvidia")` – und wird als Erstes in
`run_app.py` und `__main__.main()` aufgerufen, vor jedem torch-Import.

**VRAM entscheidet.** `nvidia-smi` mit 2 s Zeitlimit, `check=False`,
fail-soft. Fehlt es, gilt CPU – kein Fehler. Aus dem VRAM entsteht eine
Eignungsstufe (`CapabilityTier`), die dem Nutzer **vor** dem Download sagt,
was der Rechner schafft. AMD/Intel-GPUs und NPUs kommen über CIM- bzw.
PnP-Abfragen und OpenVINO dazu. Fremdfehler werden über `clean_error()`
einzeilig und auf 240 Zeichen gekürzt angezeigt.

**Backend-Kette mit Erststart-Bremse.** Reihenfolge CUDA (float16) →
DirectML (AMD/Intel/NPU) → CPU. Jeder Fehlschlag steht im Klartext in
`BackendPlan.report()`. Entscheidend: im Auto-Modus wird ein Beschleuniger
nur genommen, wenn sein Modell **bereits konvertiert** vorliegt oder gar kein
sofort lauffähiges Modell existiert. Sonst wird übersprungen und im Klartext
erklärt, wie man die Beschleunigung bewusst einschaltet (Einstellungen →
Gerät = `dml`). Damit startet der erste Programmstart keinen mehrere GB
großen ONNX-Export, der wie ein Absturz aussieht.

**Konfiguration.** `AppConfig` ist `@dataclass(frozen=True)`. Varianten nur
über `dataclasses.replace()` (`with_values`). Unbekannte JSON-Schlüssel
werden ignoriert und protokolliert, Umgebungsvariablen `STREAMFORGE_<FELD>`
überschreiben, `validated()` zieht Werte in gültige Bereiche und rundet
Auflösungen auf Vielfache von 8. Speichern läuft atomar.

**Download.** Eigener Streaming-Downloader statt `snapshot_download`: dieses
gibt `tqdm_class` nur an die Dateizähler-Leiste weiter, nicht an den
Byte-Strom – ein Abbruch hätte erst nach der nächsten fertigen Datei
gegriffen, bei mehreren GB also praktisch nie. Stattdessen wird jede Datei
blockweise (1 MB) geladen, der Abbruch pro Block geprüft und über
`DownloadCancelled(RuntimeError)` gemeldet, die nirgends von einem
allgemeinen `except Exception` geschluckt wird. Geschrieben wird nach
`<datei>.part` und erst danach per `os.replace` umbenannt – ein Abbruch
hinterlässt also nie eine halbe Zieldatei, und ein vorhandener Rest wird per
Range-Request fortgesetzt. Solange ein Download läuft, liegt ein Teil-Marker
im Modellordner; erst am Ende entsteht `.streamforge-complete.json` mit
Dateiliste und Größen (`verify_local()`). Damit gilt ein halb geladenes
Modell nicht fälschlich als vorhanden. Kurznamen (`sdxl-base`,
`wan-t2v-1.3b`) bilden auf Repo-IDs ab.

**Warteschlange.** Ein Arbeiter-Thread (einstellbar, Vorgabe 1 – mehr
Aufträge würden sich den VRAM teilen). `submit()` liefert eine Auftrags-ID,
`should_stop()` geht in jede lange Schleife, Fortschritt läuft über Rückrufe.
Gleiche Fehlermeldungen werden gedrosselt (`_ErrorThrottle`). Beim Beenden
wird die Queue geschlossen und die Threads werden gejoint.

**Einzelinstanz.** Windows: benannter Kernel-Mutex (`CreateMutexW`,
`ERROR_ALREADY_EXISTS = 183`), den der Kernel auch nach einem Absturz
aufräumt. Sonst `fcntl.flock`. Handle bleibt prozessweit offen. Auf
exotischen Plattformen wird eher gestartet als fälschlich blockiert.

**TLS.** `truststore` nutzt den Windows-Zertifikatspeicher, damit
TLS-prüfende Virenscanner und Firmen-Proxys keine Downloads blockieren;
`certifi` ist die mitgelieferte Rückfallebene.

**Lizenzen, fail-closed.** Proprietäre Laufzeiten (CUDA/cuDNN) und das
Stimmklonen sind Komponenten mit Zustimmungspflicht. Die Zustimmung liegt als
`consent.json` neben der Konfiguration und gilt nur für die zugestimmte
Fassung. Ohne Zustimmung wird nicht geladen, sondern auf den freien Pfad
zurückgefallen – sichtbar in der Oberfläche. `THIRD-PARTY-NOTICES.md` liegt
der Auslieferung bei.

## Stimme anlernen

Ein Profil ist ein Ordner unter `<daten>\voices\<slug>\` mit `profile.json`,
`samples\` und `artifacts\`.

```powershell
streamforge licenses accept voice-cloning
streamforge voice-profile create --name "Sprecher A" --speaker "Vorname Nachname" `
    --purpose "Trailer-Vertonung" --mode zero_shot --yes
streamforge voice-profile add-sample sprecher-a --file .\aufnahme.wav
streamforge voice-profile train sprecher-a
streamforge voice "Text mit angelernter Stimme" --profile sprecher-a
```

### Warum die Klonstimme eine eigene Laufzeit hat

`chatterbox-tts` (MIT, kann Deutsch) verlangt **torch 2.6**, **diffusers 0.29**
und **transformers 5.x**. In derselben Umgebung wie Bild und Video
installiert, stuft pip torch auf eine Fassung ohne CUDA-Build herunter und
diffusers unter die Version, die Wan 2.1 und CogVideoX brauchen – die
GPU-Beschleunigung und die Videopipelines wären weg.

Deshalb läuft die Klonstimme in einer **getrennten Umgebung** und wird über
die Kommandozeile aufgerufen, genau wie ffmpeg
(`app/voice_runtime.py`, `packaging/voice_worker.py`).

```powershell
streamforge voice-runtime status     # Zustand
streamforge voice-runtime install    # eigene Umgebung anlegen (~5,4 GB)
streamforge voice-runtime prepare    # Modell einmalig laden (~6 GB)
```

`prepare` ist bewusst ein eigener Schritt: der erste Modellabruf lädt rund
6 GB, und das darf nicht in das Zeitfenster der Sprachausgabe fallen. Der
Wachhund misst deshalb auch keine Gesamtdauer, sondern nur **Stillstand** –
jede Ausgabe des Arbeiters gilt als Lebenszeichen.

Das Modell landet in `<daten>\models\hf`, nicht im Benutzerprofil: der
Arbeiter bekommt `HF_HOME` gesetzt, sonst lägen 6 GB außerhalb des
Anwendungsordners.

In der Oberfläche: **Stimme anlernen → Laufzeit einrichten**. Beim Bau
mitliefern: `-WithVoiceRuntime $true` (kopiert `.voice-venv` nach
`tools\voice-runtime`).

**Fehlt die Laufzeit, kommt kein Platzhalterton mehr**, sondern die echte
Standardstimme plus Hinweis, warum nicht geklont wurde.

### Was „Anlernen" bei Zero-Shot bedeutet

Es wird nicht nachtrainiert. Aus dem Rohmaterial entsteht eine saubere
Referenzaufnahme (`artifacts/reference.wav`): Stille entfernt, Mono,
24 kHz, Lautheit angeglichen, auf 20 Sekunden begrenzt – über das
mitgelieferte ffmpeg. Genau diese Datei bekommt das Modell zur Laufzeit.
Der Modus `finetune` bricht mit klarer Meldung ab, statt ein wertloses
Artefakt zu schreiben; echtes Nachtrainieren ist noch nicht umgesetzt.

Zwei Sperren, beide fail-closed:

1. Ohne gültigen Einwilligungs-Nachweis (Name der sprechenden Person, Zweck,
   Datum, Prüfsumme des Wortlauts) wird nicht angelernt und nicht erzeugt.
2. Ohne Mindestmenge brauchbarer Aufnahmen wird nicht angelernt – 10 s für
   Zero-Shot, 10 min für echtes Nachtrainieren. Jede Aufnahme wird auf Dauer,
   Abtastrate und Kanalzahl geprüft.

Profil löschen ist der Widerrufsweg – es entfernt Aufnahmen und Artefakte.

## Kommandozeile

```
streamforge info
streamforge models list | table | installed | download <name> | remove <name>
streamforge image "<prompt>" [--steps N --width N --height N --seed N --batch N]
streamforge video "<prompt>" [--frames N --fps N --audio ton.wav --keep-frames]
streamforge voice "<text>" [--profile <slug> --speed 1.0]
streamforge voice-profile list|create|add-sample|train|delete
streamforge licenses list|accept|revoke <komponente>
```

Global: `--config`, `--data-dir`, `--device auto|cuda|dml|cpu`, `--offline`,
`--dummy`, `--no-gui`, `--no-single-instance`, `-v`/`-vv`.

Rückgabewerte: `0` ok, `1` Fehler, `3` läuft bereits, `4` abgebrochen.

## ffmpeg und AGB

**ffmpeg ist eingebettet.** Der Build holt einen **LGPL**-Build
(`win64-lgpl`), prüft am Binary, dass weder `--enable-gpl` noch
`--enable-nonfree` gesetzt ist, und legt `ffmpeg.exe` und `ffprobe.exe` nach
`dist\<Name>\tools\ffmpeg\bin\`. Dazu kommen `LICENSE.txt` und
`HERKUNFT.txt` (Fassung, Quelle, SHA-256). Aufgerufen wird ffmpeg als
eigener Prozess, nicht gelinkt – die Anwendung bleibt proprietär.

Schalter: `-WithFfmpeg $false` lässt es weg, `-FfmpegDir <pfad>` nimmt einen
eigenen Build (wird ebenfalls geprüft), `-FfmpegUrl` zeigt auf eine andere
Quelle. Ein GPL-Build bricht den Bau ab.

**AGB.** `AGB.md` liegt der Auslieferung bei und wird beim ersten Start
angezeigt. Die Zustimmung ist Voraussetzung für die Nutzung: ohne sie
beendet sich die Anwendung. Der Zustimmen-Knopf wird erst frei, wenn der
Text bis zum Ende gerollt wurde. Die Fassung ergibt sich aus dem
Text-Hash – wird der Wortlaut geändert, muss erneut zugestimmt werden.
Jederzeit erreichbar über **Lizenzen → AGB lesen**.

```powershell
streamforge agb            # Text anzeigen
streamforge agb status     # Fassung und Zustand
streamforge agb accept     # bestätigen
streamforge agb revoke     # widerrufen
```

Der mitgelieferte AGB-Text ist ein **Entwurf** mit Platzhaltern
(`[ANBIETER]`, `[GERICHTSSTAND]` …) und muss vor dem Verkauf juristisch
geprüft werden.

## Rauchtest

```powershell
python tests\smoke.py
```

Läuft ohne Netz und ohne GPU, legt alles in einem Temp-Ordner an und prüft 46
Punkte: Pfadtrennung, Konfigurations-Validierung, Warteschlange samt Abbruch
und Fehlerdrosselung, Backend-Kette mit Erststart-Bremse, Lizenz-Tore,
Einwilligungs-Nachweis für Stimmprofile (auch der Fall „Nachweis nachträglich
verändert") und die drei Attrappen-Pipelines. Rückgabe 0 = bestanden.

## Abnahmekriterien

| # | Kriterium | Umsetzung |
|---|---|---|
| 1 | Ohne Python und ohne NVIDIA-Treiber → CPU mit klarer Meldung | `resolve_backend` schreibt jeden Fehlschlag in `attempts`, Oberfläche zeigt den Bericht |
| 2 | Mit NVIDIA-GPU → CUDA ohne Zutun | Auto-Modus, `nvidia-smi` + `torch.cuda`, DLL-Pfad vorher gesetzt |
| 3 | Download mitten im Laden abbrechbar, keine halbe Datei | `DownloadCancelled` im tqdm-Ersatz, `_cleanup_incomplete()` |
| 4 | Zweiter Start meldet „läuft bereits“ | `single_instance.acquire()` + `notify_already_running()`, Exit 3 |
| 5 | Laufender Auftrag abbrechbar, ohne den Prozess zu töten | `Job.request_cancel()`, `should_stop()`, `JobCancelled` |
| 6 | Fehlendes `nvidia-smi`/`ffmpeg`/Modell → Meldung statt Stacktrace | fail-soft Sonden, `FfmpegMissing`, `clean_error()`; Konsole wird auf UTF-8 gestellt, damit Umlaute und Pfeile auf cp1252-Konsolen keinen `UnicodeEncodeError` auslösen |
| 7 | `THIRD-PARTY-NOTICES.md` liegt im Auslieferungsordner | Build kopiert sie und warnt, wenn sie fehlt |

## Speicherplatz: Modelle aufräumen

Ein SDXL-Repo führt dieselben Gewichte mehrfach (fp32 und fp16, dazu `.bin`,
OpenVINO, Einzeldatei-Checkpoints). Ungefiltert sind das **46 GB** statt der
gebrauchten **6,5 GB**. Der Filter in `models.select_files()` nimmt nur die
Komponenten aus `model_index.json`, davon die fp16-Variante, ohne Duplikate.

Ältere, zu groß geladene Modelle nachträglich aufräumen:

```powershell
streamforge models prune sdxl-base --dry-run   # nur anzeigen
streamforge models prune sdxl-base             # aufräumen
streamforge models verify sdxl-base            # Vollständigkeit prüfen
```

In der Oberfläche: **Modelle → Aufräumen**.

## Nächster Schritt

1. `voice_profiles.train_profile`: echte Sprecher-Einbettung statt
   Platzhalter (Chatterbox/OpenVoice), Nachtraining als `finetune`-Zweig.
2. DirectML-Zweig: ONNX-Export als eigener, abbrechbarer Auftrag mit
   Fortschritt, Ablage in `models.converted_dir()` – danach greift die
   Erststart-Bremse nicht mehr.
3. Entscheidung zur Sprachausgabe (siehe unten), danach ggf. Piper als
   eigenständiges Programm einbinden.
4. Bild-zu-Bild, Inpainting und Hochskalieren (`realesrgan-x4`).

## Bekannte Grenzen dieses Stands

* **Sprachausgabe braucht eine Entscheidung.** Vorgabe ist Bark (MIT,
  unbedenklich, aber langsam und 1,8 GB). Die schnellere Alternative Piper
  ist eingebaut, aber gesperrt: `piper-tts` steht unter **GPL-3.0** und
  bettet espeak-ng ein – im selben Prozess erfasst das die gesamte verkaufte
  Anwendung. Wege und Empfehlung stehen in [MODELS.md](MODELS.md).
* **Stimme anlernen** verwaltet Aufnahmen und Einwilligungen vollständig,
  das Anlernen selbst schreibt aber noch ein Platzhalter-Artefakt.
* DirectML ist erkannt und geplant, der ONNX-Export fehlt noch.
* ffmpeg wird nicht automatisch beschafft – bewusst, damit kein GPL-Build in
  eine verkaufte Anwendung gerät.
* Versionen in `requirements*.txt` haben Untergrenzen. Vor dem Release
  `pip freeze` und festnageln.
