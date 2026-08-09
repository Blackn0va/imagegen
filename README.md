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
| Bild bearbeiten | img2img und Inpainting über dieselben Gewichte | `sdxl-base` |
| Bild vergrößern | Real-ESRGAN (torch), sonst Lanczos | `realesrgan-x4` |
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
  pipeline_image.py   Text -> Bild, Bild -> Bild, Inpainting, Vergrößern
  upscale.py          Real-ESRGAN (RRDBNet in torch) + Lanczos-Rückfallebene
  contentgate.py      Inhaltssperre: keine sexualisierten Minderjährigen
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

### Stimme verfeinern

Auf der Seite **Stimme anlernen** hat jedes Profil eigene Regler, die
gespeichert werden – eine einmal gut eingestellte Stimme klingt danach immer
gleich:

| Regler | Wirkung |
|---|---|
| Ausdruck | niedrig: ruhig und sachlich · hoch: betont, mehr Melodie |
| Führung | niedrig hält sich näher an Tempo und Rhythmus der Referenz |
| Streuung | niedrig: gleichmäßig · hoch: lebendiger, aber unruhiger |
| Referenzlänge | wie viel Material in die Referenz fließt (10–30 s) |

**Speichern und verfeinern** übernimmt die Werte und baut die Referenz neu
auf, wenn sich die Länge geändert hat. **Hörprobe erzeugen** liefert einen
kurzen Satz mit den aktuellen Einstellungen.

Mehr Aufnahmen helfen: die Referenz wird aus den längsten brauchbaren
Aufnahmen zusammengesetzt, bis die Ziellänge erreicht ist. Verschiedene
Sätze decken mehr Laute und Tonhöhen ab als eine einzelne lange Aufnahme.

### Weitere Aufnahmen nachlegen

Auf der Seite **Stimme anlernen** hat jedes Profil eine Aufnahmenliste mit
Dauer, Abtastrate und – falls unbrauchbar – dem Grund. **Aufnahmen
hinzufügen …** übernimmt neue Dateien und baut die Referenz gleich neu auf.
Genau das ist das „weiter anlernen": mehr und verschiedenartiges Material
ergibt eine treffendere Stimme.

Was dabei **nicht** passiert: das Modell selbst wird nicht nachtrainiert.
`chatterbox-tts` liefert keinen Trainingscode mit. Das frühere Verfahren
„Nachtrainieren" war deshalb eine Sackgasse (es verlangte 600 s Material und
konnte nie fertig werden) und ist aus der Auswahl entfernt. Bestehende
Profile in diesem Zustand lassen sich umstellen:

```powershell
streamforge voice-profile set-mode <slug> --mode zero_shot
```

In der Oberfläche: **Auf Referenz umstellen**.

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
streamforge edit <datei...> --prompt "<text>" [--mode img2img|inpaint --mask maske.png
                            --strength 0.45 --steps N --guidance N --seed N --max-side N]
streamforge upscale <datei...> [--scale 2|4|8 --no-model --tile 512 --refine
                            --prompt "<text>" --strength 0.25 --max-side N]
streamforge video "<prompt>" [--frames N --fps N --audio ton.wav --keep-frames]
streamforge voice "<text>" [--profile <slug> --speed 1.0]
streamforge voice-profile list|create|add-sample|train|delete
streamforge licenses list|accept|revoke <komponente>
```

Global: `--config`, `--data-dir`, `--device auto|cuda|dml|cpu`, `--offline`,
`--dummy`, `--no-nsfw`, `--no-gui`, `--no-single-instance`, `-v`/`-vv`.

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

Läuft ohne Netz und ohne GPU, legt alles in einem Temp-Ordner an und prüft 122
Punkte: Pfadtrennung, Konfigurations-Validierung, Warteschlange samt Abbruch
und Fehlerdrosselung, Backend-Kette mit Erststart-Bremse, Lizenz-Tore,
Einwilligungs-Nachweis für Stimmprofile (auch der Fall „Nachweis nachträglich
verändert"), die drei Attrappen-Pipelines, Vergrößern, Bearbeiten und die
Inhaltssperre.
Das Real-ESRGAN-Netz wird dabei gegen selbst erzeugte Gewichte geprüft –
Aufbau, Größenableitung und Kachelweg, ohne Download. Rückgabe 0 = bestanden.

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

## Bestehende Bilder bearbeiten

Seite **Bild bearbeiten** (oder `streamforge edit` / `streamforge upscale`).
Das Ausgangsbild wird nie überschrieben – es entsteht immer eine neue Datei,
deren Name die Quelle enthält.

| Modus | Was passiert | Braucht |
|---|---|---|
| Vergrößern | Real-ESRGAN rekonstruiert Kanten, sonst Lanczos | `realesrgan-x4` (250 MB) oder nichts |
| Nach Prompt umarbeiten | img2img über das Bildmodell | Bildmodell + Prompt |
| Bereich ersetzen | Inpainting, weiße Maskenfläche wird neu gerechnet | Bildmodell + Prompt + Maske |

Zum Grafikspeicher: img2img und Inpainting bauen ihre Pipeline aus den
**bereits geladenen** Modulen des Bildmodells (Konstruktor mit
`pipe.components`). Es wird nichts ein zweites Mal geladen – gemessen 0,0 s
und 0 MB zusätzlich, gegenüber 280 s und rund 5 GB, wenn man stattdessen
`from_pipe` nimmt (das kopiert die Gewichte). `from_pipe` bleibt nur die
Rückfallebene. Vergrößern läuft kachelweise; reicht der Speicher trotzdem
nicht, halbiert sich die Kachelgröße automatisch, und erst danach gibt es
eine Meldung.

Fehlt Real-ESRGAN oder passen die Gewichte nicht, wird **Lanczos** benutzt und
das Verfahren steht im Ergebnis – ein weiches Bild ist besser als ein
abgebrochener Auftrag.

## Inhalte für Erwachsene

**An, ohne Zutun.** Die Inhaltsprüfung der Modelle ist abgeschaltet, Nacktheit
und erotische Darstellungen laufen durch:

```powershell
streamforge image "nude woman, 30 years old, oil painting"
```

Abschalten geht über **Einstellungen → Inhalte für Erwachsene**, über
`nsfw_enabled: false` in der Konfiguration oder je Aufruf mit `--no-nsfw`.
Dann greift wieder der `safety_checker`, den SD 1.5 mitbringt und der sonst
jedes als nicht jugendfrei eingestufte Bild schwärzt. SDXL und FLUX bringen
keine solche Komponente mit – dort gab es nie etwas abzuschalten.

Zusätzlich hängt die Anwendung Schutzbegriffe an den Negativ-Prompt
(`nsfw_protective_negative`, abschaltbar) und schreibt den tatsächlich
verwendeten Negativ-Prompt als `negative_prompt_used` in die Bild-Metadaten –
sonst ließe sich ein Bild nicht noch einmal genauso erzeugen.

### Was gesperrt bleibt

`app/contentgate.py` lehnt Aufträge ab, die Begriffe für Minderjährige mit
sexuellen Begriffen verbinden – geprüft werden Prompt **und** Negativ-Prompt,
vor dem Laden des Modells, bei Bild, Video und Bearbeiten. Ein
`nsfw_block_minors: false` in der Konfiguration wird beim Laden zurückgesetzt
(gleiche Bauart wie `voice_require_consent`).

Erwachsenendarstellungen sind davon nicht betroffen. Die Wortlisten sind
darauf ausgelegt, keine Fehlalarme zu erzeugen – eine Sperre, die bei
harmlosen Motiven ständig zuschlägt, wird ausgebaut und schützt dann nichts:

| läuft | wird abgelehnt |
|---|---|
| `nude woman, 25 years old` | `nude child` |
| `a child playing football` | `nacktes kleinkind` |
| `nude woman, kindness in her eyes` | `sexy schoolgirl, 14 years old` |
| `lolita fashion dress, adult model` | `lolita, nude` |
| `topless adult woman, minorca island` | `n4ked t3en` |

`girl`, `boy` und `young` stehen bewusst nicht auf der Liste; `kind` wird als
ganzes Wort geprüft (sonst träfe es „kindness“), deutsche Zusammensetzungen
wie „kinderzimmer“ über den Wortanfang.

Das ist eine Prüfung auf **Text**, keine Bildprüfung – eine Untergrenze, kein
Schloss.

### Modellwahl

Sieben geprüfte Feinabstimmungen sind eingetragen – die Basismodelle können
Nacktheit, sind darauf aber nicht abgestimmt:

| Modell | Basis | Größe | VRAM | Stärke |
|---|---|---|---|---|
| `pony-v6` | SDXL | 6,5 GB | ab 6 GB | stärkste Prompt-Treue, explizit |
| `noobai-xl` | SDXL | 6,5 GB | ab 6 GB | Anime/Manga, Danbooru-Tags |
| `realvis-xl` | SDXL | 6,5 GB | ab 6 GB | fotorealistische Menschen |
| `juggernaut-xl` | SDXL | 6,5 GB | ab 6 GB | fotorealistisch, kräftiges Licht |
| `nsfw-gen` | SDXL | 8,0 GB | ab 6 GB | direkt auf explizite Motive trainiert |
| `realistic-vision` | SD 1.5 | 5,1 GB | ab 4 GB | fotorealistisch, sparsam |
| `dreamshaper` | SD 1.5 | 2,6 GB | ab 3 GB | kleinster Eintrag, Allrounder |

```powershell
streamforge models download pony-v6
# in der Oberfläche: Modelle -> Als Bildmodell setzen
```

**Pony V6** erwartet Wertungs-Marker am Prompt-Anfang, sonst sind die
Ergebnisse deutlich schwächer:
`score_9, score_8_up, score_7_up, <eigentlicher Prompt>`

Details zu Lizenzen und Einzeldatei-Checkpoints stehen in
[MODELS.md](MODELS.md). Jedes weitere Repo lässt sich ohne Codeänderung
nachladen (`streamforge models download <besitzer>/<repo>`); es läuft dann
als **CONDITIONAL** („Lizenz nicht geprüft“).

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
4. Maskenwerkzeug in der Oberfläche: eine Maske muss derzeit in einem
   Bildprogramm gemalt und als Datei ausgewählt werden.

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
