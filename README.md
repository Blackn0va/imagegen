# StreamForge Studio

**Bild, Video und Sprache lokal erzeugen – ohne Cloud, ohne API-Schlüssel.**

Windows-Desktop-Anwendung. Alles rechnet auf dem eigenen Rechner: kein Konto,
kein Upload, keine laufenden Kosten. Ausgeliefert als PyInstaller-Bundle, der
Nutzer braucht kein Python.

<img width="1915" height="1026" alt="Oberfläche von StreamForge Studio" src="https://github.com/user-attachments/assets/d9053b7e-10c3-496c-9005-595a299f15a0" />

## Was es kann

| Bereich | Umsetzung | Vorgabemodell |
|---|---|---|
| **Bild erzeugen** | diffusers (SD 1.5, SDXL, FLUX) | `sdxl-base` |
| **Bild umarbeiten** | img2img und Inpainting über dieselben Gewichte | `sdxl-base` |
| **Vergrößern** | Real-ESRGAN (torch), sonst Lanczos | `realesrgan-x4` |
| **Schwarz-Weiß einfärben** | Farbe vom Modell, Helligkeit aus der Vorlage | `sdxl-base` |
| **Diamond-Painting-Vorlage** | Steinraster mit Symbolen und DMC-Nummern | – (nur Pillow) |
| **Video** | diffusers (Wan 2.1, CogVideoX, AnimateDiff) | `wan-t2v-1.3b` |
| **Sprache** | Bark über transformers | `bark-small` |
| **Stimme anlernen** | Profile, Aufnahmen, Einwilligung – Anlernen noch Attrappe | `chatterbox` |

Fehlt eine Voraussetzung (Modell, Lizenz, Bibliothek), schaltet die Anwendung
auf eine Attrappe um **und schreibt den Grund in die Ausgabe** – ein
Farbverlauf ohne Erklärung sieht sonst wie ein Fehler aus.

## Bauen

```powershell
.\build-windows.ps1 -Clean
```

Ergebnis: `dist\StreamForge\StreamForge.exe`.

Weitere Schalter: `-Python`, `-Name`, `-Model`, `-WithCuda`,
`-SkipModelDownload`, `-NoGui`, `-FfmpegDir`, `-CudaIndexUrl`, `-Console`.

Nutzerdaten (`data\`, oft zweistellige GB) werden vor dem Bau weggetragen und
danach zurückgelegt; `-Clean` löscht sie absichtlich. Landet ein CPU-Wheel im
Venv, bricht der Bau ab, statt eine .exe mit CUDA-DLLs und CPU-torch
auszuliefern.

## Entwicklung

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m app info                            # Hardware, Backend, Pfade
python -m app                                 # Oberfläche
python -m app image "ein Leuchtturm im Sturm"
python -m app colorize omas-foto.jpg
python -m app diamond motiv.jpg --stones 100
```

GPU-Pfad (NVIDIA) – **Reihenfolge entscheidet**, sonst installiert pip das
CPU-Wheel und hält `torch>=2.6` danach für erfüllt:

```powershell
pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision torchaudio
pip install -r requirements-cuda.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Muss `…+cu126 True` zeigen, nicht `…+cpu False`.

## Kommandozeile

```
streamforge info
streamforge models list | table | installed | access <name> | verify <name>
                        | download <name> | remove <name> | prune <name>
streamforge image "<prompt>" [--steps N --width N --height N --seed N --batch N]
streamforge edit <datei...> --prompt "<text>" [--mode img2img|inpaint --mask maske.png
                        --strength 0.45 --steps N --guidance N --seed N --max-side N]
streamforge upscale <datei...> [--scale 2|4|8 --no-model --tile 512 --refine
                        --prompt "<text>" --strength 0.25 --max-side N]
streamforge colorize <datei...> [--prompt "<farbwunsch>" --strength 0.55 --steps N
                        --guidance N --seed N --free-luminance --max-side N]
streamforge diamond <datei...> [--stones 100 --colors 24 --shape round|square
                        --cell 18 --no-symbols --no-dmc --max-side N]
streamforge video "<prompt>" [--frames N --fps N --audio ton.wav --keep-frames]
streamforge voice "<text>" [--profile <slug> --speed 1.0]
streamforge voice-profile list|create|add-sample|train|delete
streamforge licenses list|accept|revoke <komponente>
```

Global: `--config`, `--data-dir`, `--device auto|cuda|dml|cpu`, `--offline`,
`--dummy`, `--no-nsfw`, `--no-gui`, `--no-single-instance`, `-v`/`-vv`.

Rückgabe: `0` ok, `1` Fehler, `3` läuft bereits, `4` abgebrochen.

## Bilder bearbeiten

Seite **Bild bearbeiten** oder die Befehle oben. Das Ausgangsbild wird nie
überschrieben. Die Seite zeigt nur, was zur gewählten Aufgabe gehört, und sagt
vorher, was herauskommt (Zielauflösung, Rastermaße, Steinzahl, cm).

| Modus | Was passiert | Braucht |
|---|---|---|
| Vergrößern | Real-ESRGAN, sonst Lanczos | `realesrgan-x4` (250 MB) oder nichts |
| Nach Prompt umarbeiten | img2img | Bildmodell + Prompt |
| Bereich ersetzen | Inpainting, weiße Maskenfläche wird neu gerechnet | Bildmodell + Prompt + Maske |
| Schwarz-Weiß einfärben | Farbe vom Modell, Helligkeit aus der Vorlage | Bildmodell, Prompt freiwillig |
| Diamond-Painting-Vorlage | Steinraster, Farbtafel, Farbliste | nur Pillow |

**Einfärben:** das Bild geht entsättigt durch img2img, übernommen wird über
YCbCr nur der Farbanteil – die Helligkeit stammt unverändert aus der Vorlage.
Details bleiben erhalten, keine Farbsäume. `--free-luminance` schaltet das ab.

**Diamond Painting:** kein Modell, kein Download, in Sekunden fertig. Je Bild
Vorlage, Farbtafel und Farbliste. Farben werden auf die 445 bestellbaren
DMC-Steinnummern abgebildet (`--no-dmc` schaltet auf Bildfarben um, dann aber
nicht bestellbar), zu ähnliche Töne zusammengelegt – es kommen also unter
Umständen weniger Farben heraus als angefordert. Die RGB-Werte in
[app/dmc.py](app/dmc.py) sind Näherungen; vor großen Bestellungen mit der
Farbkarte des Anbieters abgleichen.

**Speicher:** img2img und Inpainting bauen ihre Pipeline aus den bereits
geladenen Modulen – 0 MB zusätzlich statt rund 5 GB mit `from_pipe`.
Vergrößern läuft kachelweise und halbiert die Kachel bei Speichermangel.

## Modelle

```powershell
streamforge models table            # Lizenzstufen, kommerziell nutzbar?
streamforge models access <name>    # Zugang und Platz prüfen, ohne Download
streamforge models download <name>
streamforge models prune <name>     # Altlasten aus zu weiten Filtern
```

Manche Repos sind **gated**: Dateiliste öffentlich, Dateien nicht. FLUX.1
gehört dazu. Bedingungen auf der Modellseite annehmen, Token erzeugen
(Settings → Access Tokens, Rolle `read`), dann:

```powershell
setx HF_TOKEN "hf_..."      # oder: huggingface-cli login
```

Ein Konfigurationsfeld dafür gibt es bewusst nicht – ein Token gehört nicht im
Klartext in eine Einstellungsdatei. Abgebrochene Downloads bleiben liegen und
setzen beim nächsten Anlauf fort.

Lizenzstufen und Auflagen je Modell: [MODELS.md](MODELS.md).

## Inhalte für Erwachsene

**An, ohne Zutun.** Die Inhaltsprüfung der Modelle ist abgeschaltet, Nacktheit
und erotische Darstellungen laufen durch. Abschalten über Einstellungen,
`nsfw_enabled: false` oder `--no-nsfw`.

Nicht abschaltbar ist die Sperre in `app/contentgate.py`: Aufträge, die
Begriffe für Minderjährige mit sexuellen Begriffen verbinden, werden
abgelehnt – Prompt **und** Negativ-Prompt, vor dem Laden des Modells.
`nsfw_block_minors: false` wird beim Laden zurückgesetzt.

| läuft | wird abgelehnt |
|---|---|
| `nude woman, 25 years old` | `nude child` |
| `a child playing football` | `sexy schoolgirl, 14 years old` |
| `lolita fashion dress, adult model` | `lolita, nude` |

Eine Textprüfung, keine Bildprüfung – Untergrenze, kein Vollschutz.

## Stimme

Bark ist Vorgabe (MIT, unbedenklich, aber langsam und 1,8 GB). Piper wäre
schneller, ist aber gesperrt: `piper-tts` steht unter GPL-3.0 und bettet
espeak-ng ein – im selben Prozess erfasst das die gesamte Anwendung.

Stimmprofile verwalten Aufnahmen und Einwilligungen vollständig; ohne
dokumentierte Einwilligung der sprechenden Person wird ein Profil nicht
benutzt, und das ist nicht abschaltbar. Das Anlernen selbst schreibt noch ein
Platzhalter-Artefakt.

## ffmpeg

Wird **nicht** automatisch beschafft – sonst geriete leicht ein GPL-Build in
eine verkaufte Anwendung. Pfad über `-FfmpegDir` beim Bau oder in den
Einstellungen. Fehlt es, sagt die Anwendung das im Klartext.

## Rauchtest

```powershell
python tests\smoke.py
```

Ohne Netz und ohne GPU, 267 Prüfungen, alles in einem Temp-Ordner. Rückgabe
`0` = bestanden. Die Oberflächen-Prüfungen überspringen sich ohne Anzeige.

## Aufbau

```
app\
  __main__.py         CLI, Startreihenfolge, Logging
  config.py           eingefrorene Dataclass, JSON + Umgebungsvariablen
  paths.py            portable vs. %LOCALAPPDATA%
  accel.py            DLL-Suchpfad, GPU/CPU-Erkennung, Backend-Kette
  models.py           Registrierung, Lizenzstufe, Download, Cache
  pipeline_image.py   Bild erzeugen, umarbeiten, inpainten, einfärben
  upscale.py          Real-ESRGAN (RRDBNet in torch) + Lanczos
  diamond.py          Diamond-Painting-Vorlage
  dmc.py              DMC-Farbtabelle (489 Farben, 445 als Stein)
  contentgate.py      Inhaltssperre
  pipeline_video.py   Video
  pipeline_voice.py   Sprache
  voice_profiles.py   Stimme anlernen
  compose.py          ffmpeg (LGPL-Pfad)
  jobs.py             Warteschlange, Fortschritt, Abbruch
  licensing.py        Zustimmung zu Drittanbieter-Lizenzen, fail-closed
  single_instance.py  Kernel-Mutex
  nettrust.py         truststore -> certifi
  gui\                tkinter-Oberfläche
```

## Grenzen dieses Stands

* Stimme anlernen schreibt noch ein Platzhalter-Artefakt.
* DirectML ist erkannt, der ONNX-Export fehlt.
* Kein Maskenwerkzeug – Masken müssen extern gemalt werden.
* Versionen in `requirements*.txt` haben Untergrenzen; vor dem Release
  `pip freeze` und festnageln.

## Lizenz

Proprietär, siehe [LICENSE](LICENSE). Bestandteile Dritter:
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Nutzungsbedingungen:
[AGB.md](AGB.md).
