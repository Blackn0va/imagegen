# StreamForge Studio

**Bild, Video und Sprache lokal erzeugen – ohne Cloud, ohne API-Schlüssel.**

Windows-Desktop-Anwendung. Alles rechnet auf dem eigenen Rechner: kein Konto,
kein Upload, keine laufenden Kosten. Ausgeliefert als PyInstaller-Bundle, der
Nutzer braucht kein Python.

<img width="1915" height="1026" alt="Oberfläche von StreamForge Studio" src="https://github.com/user-attachments/assets/d9053b7e-10c3-496c-9005-595a299f15a0" />

## Was es kann

| Bereich | Umsetzung | Vorgabemodell |
|---|---|---|
| **Chat / Code-Writer** | llama.cpp (GGUF), liest eingefügte Bilder | `qwen25-vl-3b` |
| **Telefonieren** | Whisper hört zu, Sprachmodell antwortet, eigene Stimme spricht | `whisper-small` |
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

Bringt standardmäßig alles mit: CUDA, ONNX/OpenVINO (AMD-/Intel-GPU, NPU)
und die Chat-Laufzeit. Abschaltbar über `-WithOnnx $false` / `-WithChat $false`.

Weitere Schalter: `-Python`, `-Name`, `-Model`, `-WithCuda`,
`-SkipModelDownload`, `-NoGui`, `-FfmpegDir`, `-CudaIndexUrl`, `-Console`,
`-WithOnnx` (ONNX/OpenVINO für AMD-/Intel-GPU und NPU mitliefern),
`-PurgeData` (löscht mit `-Clean` auch die geladenen Modelle).

Nutzerdaten (`data\`, oft zweistellige GB) werden vor dem Bau weggetragen und
danach zurückgelegt – auch bei `-Clean`. Modelle überleben also jeden Neubau.
Wer sie wirklich weghaben will, nimmt zusätzlich `-PurgeData`. Landet ein CPU-Wheel im
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
streamforge npu                     # NPU-/Beschleuniger-Diagnose
streamforge chat [--info] ["<frage>"] [--model <name> --persona <name> --image bild.png]
streamforge call [--info --voice <stimme> --list-voices --list-devices --turns N]
streamforge models list | table | installed | access <name> | verify <name>
                        | download <name> | remove <name> | prune <name>
                        | convert <name> --backend dml|openvino
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

Global: `--config`, `--data-dir`, `--device auto|cuda|dml|openvino|cpu`, `--offline`,
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

**Maske malen:** bei „Bereich ersetzen" öffnet *Maske malen …* das Bild direkt
in der Anwendung – malen, radieren, Mausrad für die Pinselgröße, Strg+Z für
zurück. Kein zweites Programm nötig. Gespeichert wird in voller Auflösung des
Originals, auch wenn die Arbeitsfläche verkleinert dargestellt ist.

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

## Chat und Code-Writer

Läuft lokal über **llama.cpp** mit GGUF-Modellen. Gemessen ist das auf
Intel-CPUs rund doppelt so schnell wie OpenVINO – und die NPU ist für
Sprachmodelle der falsche Baustein, CPU und iGPU schlagen sie.

| Modell | Größe | Bilder | Tempo |
|---|---|---|---|
| `qwen25-vl-3b` *(Vorgabe)* | 3,4 GB | **ja** | **10 Token/s gemessen** |
| `qwen25-coder-3b` | 2,0 GB | nein | etwas schneller (kleiner) |
| `qwen25-coder-7b` | 4,7 GB | nein | grob halbes Tempo |
| `llama32-3b` | 2,1 GB | nein | etwa wie coder-3b |

Gemessen auf einem i9-10850K ohne Grafikkarte: Laden 2 s (Modell lokal),
**10,4 Token/s**, Antwort strömt Stück für Stück. Ein Bild kostet einmalig
rund 11 s fürs Kodieren, danach läuft die Antwort normal. Auf einem
Notebook-Prozessor eher weniger – die Zahlen sind Anhaltspunkte, keine
Zusage.

Die Seite **Chat**: Code kommt in Blöcken mit Sprachangabe und
**Kopierknopf**, Markdown wird dargestellt (Überschriften, Listen, fett,
`inline`). Bilder per **Strg+V** einfügen oder über „Bild …" wählen –
mit `qwen25-vl-3b` liest das Modell sie wirklich. Bei einem Modell ohne
Bildverständnis sagt die Anwendung das, statt das Bild zu verwerfen.

Enter sendet, Umschalt+Enter macht eine neue Zeile.

Die Laufzeit ist optional. In der Entwicklung **immer** mit fertigem Wheel
installieren – sonst will pip aus Quelltext bauen und braucht CMake und die
MSVC-Build-Tools:

```powershell
pip install llama-cpp-python --prefer-binary `
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Der Bau macht das selbst. Schlägt es fehl, läuft der Bau weiter und der Chat
meldet die fehlende Laufzeit im Klartext – ein Nebenteil darf die .exe nicht
kosten.

### Chat auf der Grafikkarte (CUDA)

Mit NVIDIA-Karte ist der Chat **rund zehnmal schneller**. Gemessen auf einer
RTX 4070 Ti mit Qwen2.5-VL 3B:

| | Token/s |
|---|---|
| CPU (i9-10850K) | 10,4 |
| **CUDA, alle 37 Schichten auf der Karte** | **105** |

Dafür braucht es ein CUDA-Wheel von llama-cpp-python — das übliche Wheel von
PyPI ist CPU-only. Der Bau erledigt alles Weitere:

```powershell
.uild-windows.ps1 -Clean -LlamaCudaWheel "<URL>"
```

Passende URL wählen: sie muss zur **Python-Fassung** des Bau-Venvs und zur
**CUDA-Fassung** passen. Für Python 3.13 gibt es keine offiziellen
CUDA-Wheels; die Quelle unten ist eine Fremdquelle (Lieferketten-Entscheidung,
deshalb nichts fest verdrahtet):

```
https://github.com/JamePeng/llama-cpp-python/releases/download/
  v0.3.46-cu128-win-20260808/
  llama_cpp_python-0.3.46%2Bcu128-cp313-cp313-win_amd64.whl
```

Für Python 3.12 reicht der offizielle Index
(`abetlen.github.io/llama-cpp-python/whl/cu124`).

Zwei Dinge, die sonst still danebengehen und die der Bau jetzt erledigt:

- **Backends registrieren sich nicht von selbst.** Ab llama.cpp 0.3.4x liegt
  jedes Backend als eigene DLL vor und wird erst zur Laufzeit geladen — der
  Lader braucht den Pfad ausdrücklich. Ohne das registriert sich *keines*,
  auch nicht die CPU. Erledigt `pipeline_chat.load_backends()`.
- **`ggml-cuda.dll` ist gegen `cudart`/`cublas` gelinkt.** Fehlen die im
  selben Ordner, wird das CUDA-Backend stillschweigend übersprungen. Der Bau
  legt sie aus torch daneben.

Prüfen, ob es greift:

```powershell
streamforge chat --info
```

Muss `GPU-Offload:  JA` und die Karte zeigen. Steht dort `NEIN`, rechnet der
Chat auf der CPU — unabhängig von `chat_gpu_layers`.

**Bildverständnis kostet Tempo:** derselbe Lauf ohne Bildteil schafft 96
Token/s, mit Bildteil 38 im Kaltlauf (warm 105). Wer nur Code schreibt, fährt
mit `qwen25-coder-3b` schneller.

### Charaktere (Personas)

Chat und Telefon haben einen **Charakter-Wähler**: sachlicher Assistent,
Lustiger, Ernster, Hacker (offensive Security für autorisierte Arbeit),
Querdenker, Verschwörungs-Erzähler (als Spiel, gekennzeichnet), Mentor,
Ideengeber, Stoiker, Pirat.

Eine Persona ändert nur den **Ton**, keine Sperre: die Inhaltssperre und die
Lizenz-/Einwilligungstore greifen unabhängig weiter. Die Charaktere liegen in
`personas.json` im Datenverzeichnis – **editierbar**, eigene lassen sich
dazulegen, ohne neu zu bauen.

```powershell
streamforge chat --list-personas
streamforge chat "Erklär mir Rekursion" --persona pirate
```

## Telefonieren mit der KI

Ein Gespräch statt Tippen. Der Kreis:

```
Mikrofon → Whisper → Sprachmodell → Sprachausgabe → Lautsprecher
```

```powershell
streamforge call --info           # prüft alle drei Stufen
streamforge call --list-voices    # wählbare Stimmen
streamforge call                  # Gespräch starten, Strg+C legt auf
```

**Stimme wählen:** jede mitgelieferte oder eine **selbst angelernte** aus den
Stimmprofilen (`--voice <slug>`). Angelernte stehen oben in der Liste. Ohne
dokumentierte Einwilligung bleibt ein Profil gesperrt – auch am Telefon.

**Dateien aus dem Gespräch.** Gesprochenes ist flüchtig, Code darf es nicht
sein. Jede Antwort wird deshalb doppelt geführt:

- **Gesprochen** wird nur der Fließtext. Quelltext vorzulesen ist sinnlos.
- **Geschrieben** wird alles: `mitschrift.md` mit dem ganzen Verlauf und jeder
  Code-Block als eigene Datei mit passender Endung (`.py`, `.ps1`, `.sql` …).

Alles landet unter `output	elefonate\<Zeitstempel>\`. Die Mitschrift wird
nach **jedem** Zug geschrieben, nicht erst am Ende – ein abgebrochenes
Gespräch soll nicht alles mitnehmen.

**Wann ist der Redebeitrag zu Ende?** Gemessen wird die Lautstärke; nach einer
Sekunde Ruhe antwortet die KI. Die Auslöseschwelle wird zu Beginn aus dem
Grundrauschen des Mikrofons bestimmt – ohne das löst ein rauschendes Mikrofon
dauernd aus und ein sehr leises nie. Einstellbar über `call_silence_seconds`.

**Die Stimme wählen** – oben auf der Seite oder mit `--voice`:

| Art | Tempo | Klang |
|---|---|---|
| **Windows-Stimmen** (Vorgabe) | ~0,4 s je Satz | robotisch, aber sofort |
| Bark, Chatterbox u. a. | ~20 s je Satz | natürlich, fürs Telefon zu langsam |
| Angelernte Stimmprofile | je nach Modell | deine eigene Stimme |

Windows bringt deutsche Stimmen mit (Hedda, Katja, Stefan) – kein Download,
keine Lizenzfrage. Deshalb stehen sie oben und sind Vorwahl. Ein Modell wie
Bark klingt besser, braucht aber rund zwanzig Sekunden je Satz; das ist für
eine Datei in Ordnung, für ein Gespräch nicht.

**Mikrofon und Wiedergabe** lassen sich auf der Seite auswählen; die Wahl
wird sofort gespeichert. *Mikrofon testen* nimmt kurz auf und zeigt den
Pegel – damit man nicht erst im Sprachmodell sucht, wenn das Mikrofon stumm war.

**Gesprochen wird satzweise, während das Modell noch schreibt.** Sobald der
erste Satz steht, spricht die Stimme ihn, während die Antwort weiterläuft.
Ohne das käme der erste Ton erst, wenn die ganze Antwort erzeugt *und* die
ganze Sprachausgabe fertig ist.

Gemessen auf einer RTX 4070 Ti mit `whisper-small`, `qwen25-vl-3b` und der
Windows-Stimme Hedda:

| Schritt | Zeit |
|---|---|
| Spracherkennung (4,8 s Audio) | 0,58 s — **8× Echtzeit** |
| Antwort des Sprachmodells | 2,2 s |
| **Frage bis erster Ton** | **2,3 s** |
| ganzer Zug, drei Sätze gesprochen | 3,4 s |

Zum Vergleich derselbe Zug mit Bark: erster Ton nach **20,8 s**.

Die Laufzeiten sind optional (`-WithCall`, Vorgabe an). Fehlen sie, sagt
`call --info`, welche Stufe fehlt, und der Chat bleibt trotzdem nutzbar.

## AMD-/Intel-GPU und Intel-NPU

torch bedient unter Windows nur NVIDIA und CPU. Für alles andere gibt es
zwei Laufzeiten, die **eigene Gewichte** brauchen – das Modell wird einmalig
exportiert.

| Backend | Laufzeit | Gerät |
|---|---|---|
| `dml` | ONNX Runtime, `DmlExecutionProvider` | AMD- und Intel-GPU |
| `openvino` | OpenVINO | Intel-GPU **und NPU** |

Beide sind optional und nicht vorinstalliert – wer eine NVIDIA-Karte hat,
braucht sie nicht.

**Im gebauten Programm** muss die Laufzeit beim Bauen dabei sein. Die .exe
bringt ihren eigenen Python mit und sieht nichts, was hinterher per
`pip install` in ein System-Python gelegt wird:

```powershell
.uild-windows.ps1 -Clean -WithOnnx $true
```

**In der Entwicklung** genügt pip:

```powershell
pip install "optimum[onnxruntime]"          # DirectML
pip install "optimum[openvino]" openvino    # Intel-GPU / NPU
```

Danach einmalig konvertieren:

```powershell
streamforge models convert sdxl-base --backend openvino
```

Danach in den Einstellungen **Gerät = `openvino`** (bzw. `dml`) wählen. Das
OpenVINO-Zielgerät lässt sich festlegen; leer heißt NPU vor GPU vor CPU.

Fehlt Laufzeit oder Konvertat, wird das im Klartext gemeldet und auf CPU
gerechnet – nie stillschweigend unter falschem Namen. Prüfen mit:

```powershell
streamforge npu
```

FLUX läuft auf diesem Weg **nicht** – optimum hat dafür keine Pipeline.
Dann CPU oder CUDA.

Zur Erwartung: eine NPU ist auf kleine, quantisierte Netze ausgelegt. Sie
ist sparsam, bei Diffusionsmodellen aber langsamer als eine dedizierte
Grafikkarte; oft lohnt `GPU` statt `NPU`.

## Modelle

Alle Modelle sind nutzbar – auch die mit eingeschränkter kommerzieller
Lizenz. Diese Anwendung wird privat betrieben und nicht verkauft; dafür
erlauben die Lizenzen das. Die Freigabe wird beim ersten Start gesetzt und
protokolliert, ein Widerruf hält:

```powershell
streamforge licenses revoke private-use   # wieder sperren
streamforge licenses list                 # Stand und Auflagen ansehen
```

**Damit darf die gebaute Anwendung nicht weitergegeben oder verkauft
werden** – die eingeschränkten Modelle wären dann Teil eines
kommerziellen Produkts.

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
benutzt, und das ist nicht abschaltbar.

Anlernen läuft als **Zero-Shot**: aus dem Rohmaterial wird eine saubere,
einkanalige, normalisierte Referenzaufnahme gebaut – genau das, was das Modell
zur Laufzeit braucht. Kein Platzhalter, sondern das Verfahren. Echtes
Nachtrainieren (`finetune`) ist nicht umgesetzt und lehnt mit klarer Meldung
ab, statt ein wertloses Artefakt zu schreiben.

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
  accel.py            DLL-Suchpfad, GPU/NPU/CPU-Erkennung, Backend-Kette
  models.py           Registrierung, Lizenzstufe, Download, Cache
  pipeline_image.py   Bild erzeugen, umarbeiten, inpainten, einfärben
  pipeline_onnx.py    DirectML und OpenVINO (Intel-GPU und NPU)
  upscale.py          Real-ESRGAN (RRDBNet in torch) + Lanczos
  diamond.py          Diamond-Painting-Vorlage
  dmc.py              DMC-Farbtabelle (489 Farben, 445 als Stein)
  pipeline_chat.py    Chat/Code-Writer (llama.cpp, GGUF)
  pipeline_stt.py     Spracherkennung (Whisper)
  pipeline_call.py    Telefonieren: Kreis aus Zuhören, Denken, Sprechen
  audio_io.py         Mikrofon und Wiedergabe
  personas.py         Gespraechscharaktere (editierbare JSON)
  speech_stream.py    satzweises Sprechen waehrend der Antwort
  pipeline_sapi.py    Windows-Stimmen (schnell, fuers Telefonat)
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

* Nachtrainieren einer Stimme (`finetune`) ist nicht umgesetzt; Zero-Shot ist
  es und genügt für den Zweck.
* Der ONNX-/OpenVINO-Pfad ist eingebaut, aber auf keinem Gerät mit AMD-GPU,
  Intel-iGPU oder NPU gegengeprüft – nur die Ablehnungen sind getestet.
* Versionen in `requirements*.txt` haben Untergrenzen; vor dem Release
  `pip freeze` und festnageln.

## Lizenz

Proprietär, siehe [LICENSE](LICENSE). Bestandteile Dritter:
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). Nutzungsbedingungen:
[AGB.md](AGB.md).
