# Gedächtnis

Was bei der Arbeit an StreamForge Studio teuer gelernt wurde und sich weder aus
dem Quelltext noch aus der Git-Historie ablesen lässt. Gedacht als erstes, was
man liest, bevor man hier etwas ändert.

Kein Ersatz für `README.md` (Bedienung), `CHANGELOG.md` (was wann geschah) oder
`CONTRIBUTING.md` (Regeln). Hier steht nur, was sonst jedes Mal neu herausgefunden
werden müsste.

---

## Was dieses Projekt ist

`imagegen` heißt das Verzeichnis, **StreamForge Studio** die Anwendung. Der Name
führt in die Irre: es ist längst keine reine Bilderzeugung mehr, sondern eine
proprietäre Windows-Desktop-Anwendung (Python 3.13, tkinter/ttk, PyInstaller
`--onedir`) für lokale Erzeugung von **Bild, Video, Sprache und Text**. Alles
rechnet auf dem Rechner des Nutzers; nichts geht in eine Cloud.

Bestandteile: SDXL/Real-ESRGAN (Bild), faster-whisper (Verstehen),
llama.cpp/GGUF (Antworten), SAPI/Piper/Bark/Kokoro (Sprechen), Discord-Bot
(Telefonieren im Sprachkanal).

Erklärtes Ziel: **Release-Reife**. Deshalb zählen Rechtstexte, Lizenzhinweise und
ehrliche Fehlermeldungen als Teil der Arbeit, nicht als Beiwerk.

### Nicht zu verwechseln mit StreamWizard

Die globale `~/.claude/CLAUDE.md` beschreibt **StreamWizard** – ein
Java-21-Maven-Mehrmodulprojekt (obsbotjava, DiscordServer, admintool …) mit JDA,
Twitch4J und einer PHP-Website. Das ist ein **anderes Projekt**.

| | gilt hier |
|---|---|
| Verhaltens- und Sicherheitsregeln aus CLAUDE.md | **ja** – Umlaute ausnahmslos (ä ö ü, nie ae/oe/ue), keine Secrets im Code, nie selbstständig committen, nie ungefragt auf `main` pushen, fail-closed bei Auth/Lizenz, externe Dienste nur auf ausdrückliche Aufforderung |
| Modultabelle, Java-Tech-Stack, `mvn`-Befehle, YubiKey-Signing, HMAC-Endpunkt-Regel | **nein** – hier gibt es kein Maven und kein PHP |

Gebaut wird mit `build-windows.ps1`, nicht mit Maven.

---

## Bauen: zwei Schalter, die keine Kleinigkeit sind

```powershell
.\build-windows.ps1 -Clean -WithPiper -LlamaCudaWheel "<URL des CUDA-Wheels>"
```

Ohne diese beiden Schalter fehlen genau die Dinge, die am häufigsten bemängelt
wurden – und beide fehlen **ohne deutliche Fehlermeldung**:

- **`-WithPiper`** bringt die Stimme *Thorsten*. Vorgabe **aus**, und zwar
  absichtlich: piper-tts steht unter **GPL-3.0** und bettet espeak-ng ein. Eine
  damit gebaute Fassung darf **nicht weitergegeben** werden. Für den privaten
  Betrieb folgenlos – für ein Release nicht.
- **`-LlamaCudaWheel`** bringt Chat auf der GPU. Ohne sie rechnet er auf der CPU,
  rund zehnmal langsamer. Die URL wird in `.llama-cuda-wheel.txt` gemerkt;
  einmal angeben genügt.
- `-WithDiscord` ist Vorgabe **an** und bringt `davey` mit, das den Ton aus
  verschlüsselten Sprachkanälen öffnet.
- `-Clean` löscht **keine** Modelle und kein Bau-Venv. Dafür gibt es
  `-PurgeData`, `-FreshVenv`, `-PurgeCache`.

Tests laufen zweimal, weil die Umgebungen unterschiedlich bestückt sind:

```powershell
.build-venv\Scripts\python.exe tests\smoke.py   # mehr Prüfungen, alle Pakete da
python tests\smoke.py                            # System-Python, ohne davey/audioop
```

---

## Messen statt annehmen

Die wichtigste Arbeitsregel hier. Vier Fälle, in denen eine plausible Annahme
falsch war und das Messen den Unterschied machte:

- **Discord-Empfang** galt als unmöglich (Ende-zu-Ende-Verschlüsselung DAVE, nur
  Stage-Kanäle). Nachgemessen an den installierten Paketen: `davey.DaveSession`
  hat sehr wohl ein `decrypt` – es wurde von `discord-ext-voice-recv` nur nie
  aufgerufen. Die Grenze war eine fehlende Verbindung, keine Grenze.
- **`collect_dynamic_libs('davey')`** sollte die Bibliothek ins Bündel holen.
  Gemessen: **null Treffer**, für `nacl` ebenso. Ein `.pyd` ist für PyInstaller
  ein Modul, keine Bibliothek. Was wirkt, ist `collect_submodules()`.
- **Eine Persona-Verbesserung** wurde am echten 3B-Modell bei Temperatur 0
  A/B-getestet. Die erste, „gründlichere" Fassung war messbar **schlechter**.
- **Sprachausgabe lief auf CPU statt GPU**, obwohl anders eingestellt. Die
  Ursache lag nicht dort, wo sie vermutet wurde (`device_for()` las den
  Bildmodell-Plan).

Das Ergebnis einer Messung gehört in den Code – als Kommentar mit der gemessenen
Zahl und, wo möglich, als Test. Sonst wird dieselbe Annahme in einem halben Jahr
wieder geglaubt.

---

## Fallen, die hier schon zugeschnappt sind

### Heredocs zerstören Escape-Folgen

Dateien **nicht** über Bash-Heredocs oder Inline-`python -c` schreiben. Über
Git-Bash auf Windows werden Escape-Folgen ausgewertet, die literal gemeint waren.
Belegte Schäden:

| gemeint | geschrieben | Folge |
|---|---|---|
| `build\ffmpeg-dl` | `build\<0x0c>fmpeg-dl` | Unsinn im README |
| `nvidia\*\bin\cudart64_*.dll` | `nvidia\*<0x08>in\cudart64_*.dll` | **CUDA-Laufzeit kam nie mit** |
| `print("\n== …")` | echter Zeilenumbruch im String | kaputter Quelltext |

Der mittlere Fall überlebte mehrere Sitzungen unbemerkt: das Suchmuster traf nie
etwas, die CUDA-DLLs landeten nicht neben `ggml-cuda.dll`, und der Chat rechnete
still auf der CPU. Ein **zweiter, unabhängiger Weg** zu „Antworten: CPU" trotz
CUDA-Wheel.

Stattdessen: mit einem Editor-Werkzeug schreiben. Wo ein Skript nötig ist, als
`.py`-Datei anlegen, `io.open(..., encoding="utf-8")` zum Lesen und Schreiben,
Steuerzeichen über `chr(8)` aufbauen statt als Escape. `tests/smoke.py` prüft
inzwischen alle `.md`, `.ps1`, `.py` und `.spec` auf Steuerzeichen – dieser
Wächter darf nicht wieder auf einzelne Dateien eingeengt werden. Vorher prüfte er
nur Markdown, obwohl sein eigener Kommentar `build-windows` als Beispiel nannte.

### Kleine Modelle verstärken Verbotsbegriffe

Beim Abstimmen der Personas gegen das mitgelieferte 3B-Modell (Temperatur 0,
mehrere Fassungen A/B-gemessen): **das Modell greift Negativ-Begriffe auf, statt
sie als Grenze zu lesen.** Je mehr „Schadsoftware", „Waffen", „illegal" im
Systemprompt stehen, desto eher verweigert es auch bei erlaubten Fragen.

Was wirkt: positiver Rahmen, „auf jede konkrete Frage antwortest du sofort", und
**eine kurze** Grenze statt einer langen Liste.

Der kurze Grenzsatz **wirkt messbar** – ohne ihn ließ das Modell
Erpressungssoftware gegen eine Klinik durch. Er darf nicht als „unnötige
Vorsicht" entfernt werden.

Persona-Änderungen immer gegen das echte Modell messen, mit **beiden** Fragearten:
legitime Fälle (müssen durchkommen) und echter Schaden (muss abgelehnt bleiben).
Den Prompt nie „zur Sicherheit" mit weiteren Verboten anreichern.

### Verbesserungen, die den Nutzer nie erreichen

Mitgelieferte Personas wurden nur beim allerersten Start herausgeschrieben;
danach gewann immer die Datei. Wer die Anwendung schon nutzte, behielt den alten
Text – eine Verbesserung im Quelltext kam nie an. `write_defaults()` gleicht
jetzt über eine Inhalts-Signatur ab: unveränderte Vorgabe wird aktualisiert,
selbst geänderte bleibt.

Dieselbe Klasse Fehler lauert überall dort, wo eine Vorgabe einmalig in eine
Nutzerdatei geschrieben wird. Bei solchen Änderungen immer prüfen: **kommt das
bei jemandem an, der die Anwendung schon benutzt hat?**

---

## Stimmen: warum alles blechern klang

Gemessen am 2026-08-26: **kein einziges Stimmmodell war heruntergeladen.**
Übrig blieben die Windows-Stimmen (SAPI) – Formantsynthese aus den
Neunzigern. Wer daraus schließt, die Anwendung klinge eben so, liegt falsch;
die guten Stimmen waren nur nie da.

Dazu zeigte die Vorgabe `models.DEFAULTS[Task.VOICE]` auf **kokoro** – ein
Modell ohne Umsetzung (`create_voice_pipeline` kennt nur piper, bark, clone),
das obendrein **kein Deutsch** spricht.

| Stimme | Lizenz | Deutsch | Grafikkarte | Klang |
|---|---|---|---|---|
| Windows (SAPI) | Windows | ja | nein | künstlich |
| Piper (Thorsten) | **GPL-3.0-Laufzeit** | ja | **nein** | sauber, tonlos |
| **Bark / bark-small** | **MIT** | ja | **ja** | natürlich |
| Kokoro | Apache-2.0 | **nein** | – | nicht umgesetzt |
| Angelernt (Chatterbox) | MIT | ja | ja | sehr natürlich |

Piper kann **nicht** auf die Grafikkarte: das mitgelieferte onnxruntime
bietet gemessen nur `AzureExecutionProvider` und `CPUExecutionProvider`.
Für einen CUDA-Weg bräuchte es `onnxruntime-gpu` – und Piper bliebe GPL.

**Die beste Stimme ist die eingebaute von Chatterbox.** Sie braucht weder
eine Referenzaufnahme noch einen Download: deutsch, auf der Grafikkarte,
gemessen 4,84 s Ton in rund 7 s (`device: cuda, multilingual: true`), MIT.
Sie steht in der Auswahl an erster Stelle.

Wichtig ist die Trennung, die dahintersteckt:

- **mit** Referenz → Klon einer realen Person. Einwilligung zwingend,
  fail-closed. Nicht aufweichen.
- **ohne** Referenz → synthetische Stimme des Modells. Gehört niemandem,
  also ist nichts einzuwilligen.

Danach kommt **bark-small** (MIT, 1,8 GB Download). Eine **selbst
angelernte** Stimme klingt am persönlichsten und ist rechtlich sauber,
weil die Einwilligung die eigene ist – kostet aber eine Aufnahme.

**Klang schlägt nicht Telefontauglichkeit.** Ohne Grafikkarte fällt
Chatterbox auf 12 s/Satz; damit ist kein Gespräch möglich, egal wie gut es
klingt. `TELEFON_GRENZE_S = 5.0` schiebt alles darüber ans Listenende.
Aufgefallen ist das nur, weil die Prüfungen in **zwei** Umgebungen laufen
(Bau-Venv mit CUDA, System-Python ohne) – diese Doppelung beibehalten.

**Fallstrick, der zweimal zuschlug:** `accel.torch_cuda_available()` liefert
ein Tupel `(ja/nein, Begründung)`. Das Tupel als Ganzes ist immer wahr – wer
es direkt als Bedingung nimmt, zeigt „GPU" an, während die CPU rechnet.

---

## Ein Bedienelement, das nichts tut, ist schlimmer als keins

Reihenweise gefunden – und alle nach demselben Muster: der Wert wird
gespeichert, aber an der Stelle, die zählt, nie gelesen.

| Element | schrieb nach | gelesen wurde |
|---|---|---|
| „Denken" (Telefon) | `call_chat_model` | `chat_model` (überstimmt) |
| Sampler (Bild) | `request.sampler` | nur die **Metadaten** |
| „Aufnahmen behalten" (Discord) | `discord_keep_audio` | **nirgends** |
| Startbild (Video) | `init_image` | an Modelle, die es nicht können |

Der Sampler-Fall ist besonders heimtückisch: im PNG stand der gewählte,
aber nicht benutzte Sampler. Wer zwei Sampler verglich, verglich
identische Bilder und suchte den Unterschied bei sich.

Der Discord-Fall ist der schwerste: der Haken sagte zu, fremde Stimmen
**nicht** zu behalten, und tat nichts. Genau darauf beruft sich der
Betreiber gegenüber den Beteiligten (§ 201 StGB).

**Regel:** Wer ein Feld baut, sucht die Stelle, die es liest – und wenn es
keine gibt, baut er das Feld nicht. Ein Test, der prüft, dass jedes
Konfigurationsfeld irgendwo ausgewertet wird, wäre das nächste
sinnvolle Netz.

## Eine BaseException kann die ganze Anwendung stilllegen

`raise KeyboardInterrupt` als Abbruchmeldung im Vergrößern lief durch
**alle** `except Exception`-Blöcke bis aus dem Arbeiter-Thread heraus. Der
Thread wird nur einmal gestartet; danach blieb jeder weitere Auftrag für
immer auf „wartend", ohne dass irgendwo ein Fehler stand.

Zwei Lehren:

1. Abbrüche **nie** über `KeyboardInterrupt`/`SystemExit` melden – dafür
   gibt es eigene Ausnahmen (`UpscaleCancelled`, `JobCancelled`).
2. Eine Arbeiterschleife, die nur einmal startet, muss `BaseException`
   fangen. Ein einzelner Fehlgriff darf einen Auftrag kosten, nie die
   Warteschlange.

---

## Wo etwas hingeschrieben wird, muss auch gesucht werden

Der teuerste Fehler dieses Projekts bisher: `install()` richtete die
Klon-Laufzeit nach `data_dir()/voice-runtime` ein, `python_path()` suchte
dort **nicht**. Mehrere Gigabyte heruntergeladen, ausgepackt, voll
funktionsfähig – und die Anwendung meldete unverändert „nicht
eingerichtet".

Besonders tückisch: **in der Sitzung der Einrichtung funktionierte es**,
weil `install()` `STREAMFORGE_VOICE_PYTHON` setzt. Erst nach dem Neustart
war es weg. Das lässt den Fehler wie etwas anderes aussehen.

Bei allem, was an einem Ort erzeugt und an einem anderen gesucht wird:
**beide Listen nebeneinander legen.** Ein Test hält jetzt fest, dass
`data_dir()/voice-runtime` in Einrichtung *und* Suche vorkommt.

Zwei Folgefehler derselben Kette:

- Die Spec packte `voice_worker.py` gar nicht ins Bündel – eine perfekt
  eingerichtete Laufzeit blieb stumm, weil das auszuführende Skript
  fehlte. Getrennte Prozesse brauchen ihr Skript **im Bündel**, auch wenn
  es nie in diesem Python läuft.
- `worker_path()` sucht, **wo er liegen soll**; `_worker_source()` sucht,
  **wo er herkommt**. Beides mit derselben Funktion zu erledigen scheitert
  bei einer frischen Einrichtung.

## Ein Platzhalterton ist kein Erfolg

`DummyVoicePipeline` liefert eine gültige WAV-Datei mit einer Tonfolge.
Wer nur prüft, ob *eine Datei* entstand, spielt sie ab – und der Nutzer
hört „ein komisches Rauschen, keine Stimme". Am Telefon ist das der
schlechteste Ausgang: es klingt nach Defekt und nennt keinen Grund.

`VoiceResult.dummy` kennzeichnet das. **Immer auswerten**, bevor Ton
abgespielt wird; Ersatz ist die Windows-Stimme, die immer da ist.

Faustregel: eine Attrappe darf im laufenden Betrieb **nie hörbar oder
sichtbar** werden. Sie ist ein Werkzeug für Tests, kein Ausgabeformat.

---

## Ein Import beweist nicht, dass es läuft

Zielrechner: **Intel i9-10850K** (Comet Lake) — AVX2 ja, **AVX-512 und
AVX-VNNI nein** — mit einer RTX 4070 Ti.

Das CUDA-Wheel von `abetlen.github.io/.../whl/cu124` importiert dort
sauber, `ggml_cuda_init` meldet die Grafikkarte, `llama_supports_gpu_offload()`
gibt `True`. Und dann:

```
Llama(model_path=...) → OSError [WinError -1073741795]   (0xc000001d)
```

`0xc000001d` = STATUS_ILLEGAL_INSTRUCTION. Die betroffene Rechenschleife
läuft erst beim **Laden eines Modells** — jede Prüfung, die vorher
aufhört, geht daran vorbei.

**Es gibt für diese Kombination kein brauchbares CUDA-Wheel**: cu121–cu123
führen keine, nur cu124, und das crasht. Der Chat läuft dort auf der CPU.

Regel: eine Laufzeit gilt erst als brauchbar, wenn sie ihre **eigentliche
Arbeit** einmal gemacht hat — Modell laden, nicht nur importieren. Der Bau
prüft das jetzt und fällt sonst auf den CPU-Bau zurück.

---

## Ein fehlgeschlagener Import leert das Bündel lautlos

PyInstallers `collect_submodules(paket)` **importiert** das Paket. Scheitert
der Import, liefert es **0 Module** — ohne Fehler, ohne Warnung. Im Bündel
fehlt das Paket dann vollständig, und erst das fertige Programm meldet
„nicht ladbar".

So geschehen bei `llama_cpp`: `llama.dll` aus dem CUDA-Bau braucht
`cudart64_12.dll`, `cublas64_12.dll` **und** `cublasLt64_12.dll` daneben.
Das Kopiermuster kannte nur die ersten beiden. Gemessen: 0 statt 25 Module.

**Die Falle beim Prüfen:** ein Test über `pipeline_chat` meldete
`GPU-Offload: True`, weil dessen `prepare_gpu_dll_path()` die DLL-Pfade
vorher zurechtlegt. PyInstaller importiert **ohne** diese Hilfe. Wer über
die Anwendung prüft, sieht den Fehler nicht.

Regel: ein Paket, das ins Bündel soll, muss **für sich allein**
importierbar sein — `python -c "import paket"`, ohne Anwendungscode
drumherum. Der Bau prüft das jetzt.

## Ein laufendes Programm blockiert seinen eigenen Datenordner

Ein hängender `StreamForge.exe` ließ `Move-Item` auf `dist\StreamForge\data`
scheitern und brach den Bau ab — mitten im Umbau, mit halb geleertem
Bündel. Vor jedem Bau prüfen, ob noch eine Instanz läuft.

---

## Nichts Teures im Oberflächen-Thread

Der Aufbau einer Seite läuft im Oberflächen-Thread; alles, was dort Zeit
kostet, lässt das Fenster stehen. Gemessen für die Telefon-Seite: **3,22 s**,
davon 2,5 s allein ein `import torch`, ausgelöst von der GPU-Anzeige der
Stimmen.

Faustregel: **für eine Anzeige nie die harte Prüfung nehmen.**

| Zweck | richtig |
|---|---|
| anzeigen, ob GPU da ist | `accel.torch_cuda_hint()` – liest `torch/version.py` |
| wirklich rechnen | `accel.torch_cuda_available()` – importiert torch |
| Klon-Laufzeit anzeigen | `voice_runtime.cached_state()` |
| Klon-Laufzeit benutzen | `voice_runtime.available(refresh=True)` |

Dasselbe Muster gab es schon zweimal: der ffmpeg-Test beim Start und die
volle Prüfung der Klon-Laufzeit. Beide wurden in den Hintergrund verlegt.

Solche Kosten **in einem eigenen Prozess** messen – im Testprozess sind
die schweren Module längst geladen, und dann fällt genau dieser Fehler
nicht auf.

---

## Das Stimmmodell muss geladen bleiben

Chatterbox lädt beim Start **rund 35 Sekunden** (6 GB). Wird die
Sprachausgabe je Satz aufgerufen, fällt das je Satz an – eine Antwort aus
drei Sätzen dauert dann über zwei Minuten, und der Klang springt, weil
jeder Satz aus einem frischen Prozess kommt.

Deshalb: `voice_worker.py serve` hält das Modell, `VoiceServer` spricht
zeilenweise über JSON damit. Gemessen über den echten Weg der Anwendung –
Zug 1: 140 s (mit Laden), **Zug 2: 6,9 s**.

Drei Fallen, die dabei alle zugeschnappt sind:

1. **Ohne Zeilenende blockiert `readline()` ewig.** `_emit` schrieb keins;
   beim Einzelaufruf fällt das nicht auf, im Dauerbetrieb hängt alles.
2. **„Läuft nicht" ist nicht „kaputt".** Ein frisch angelegter Arbeiter
   läuft auch nicht – ihn deshalb wegzuwerfen holt das Modellladen
   zurück. `crashed` unterscheidet beides.
3. **Eine Quelltextsuche prüft die Schreibweise, nicht die Wirkung** – und
   fällt selbst der Escape-Falle zum Opfer. Solche Prüfungen am Verhalten
   festmachen: Modul laden, Funktion aufrufen, Ausgabe ansehen.

Das Modell wird beim **Anrufstart** geladen (`warmup_voice`), nicht beim
ersten Satz. Beim Programmende beenden (`shutdown_voice_servers`), sonst
bleibt ein Prozess mit mehreren GB zurück.

---

## Eigene angelernte Stimmen

Die beste Klangqualität liefert eine **selbst angelernte** Stimme über
Chatterbox (MIT, mehrsprachig, CUDA). Sie läuft in einer **getrennten
Umgebung** (`tools/voice-runtime` neben der .exe, `.voice-venv` im
Quellbaum) – chatterbox-tts verlangt torch 2.6, ältere diffusers und
transformers und würde die GPU-Beschleunigung für Bild und Video
mitreißen, wenn es in dieselbe Umgebung käme.

Zwei Wege, sie zu bekommen, und beide müssen offen bleiben:

1. **Mitliefern** beim Bauen: `-WithVoiceRuntime $true`. Legt sie an,
   falls sie fehlt (mehrere GB), und kopiert sie neben die .exe.
2. **Per Klick** in der Anwendung unter *Stimme anlernen*. Das gebaute
   Programm kann selbst kein venv anlegen – sein `sys.executable` ist die
   .exe. Deshalb sucht `voice_runtime.find_system_python()` ein Python ab
   3.11 auf dem Rechner und richtet damit ein.

**Nie wieder:** eine Absage der Form „bitte beim Anbieter melden". Der
Nutzer *ist* der Anbieter, und selbst wenn nicht, ist eine Absage ohne
Handlungsmöglichkeit keine Auskunft. Ein Test hält fest, dass diese
Formulierung verschwunden bleibt.

Ohne Klon-Laufzeit **und** ohne geladenes Stimmmodell bleiben nur die
Windows-Stimmen – siehe [Stimmen](#stimmen-warum-alles-blechern-klang).

---

## Denken und Sprechen sind zwei Modelle

Ein Sprachmodell kann nicht sprechen, ein Stimmmodell denkt nicht. Am
Telefon laufen deshalb zwei Modelle: `call_chat_model` denkt (leer =
dasselbe wie im Chat), das gewählte Stimmmodell spricht das Ergebnis aus.
Beide sind auf der Telefon-Seite getrennt einstellbar und sollen es
bleiben – vorher hing das Telefonat still an der Chat-Seite, und wer dort
umstellte, änderte das Telefonat ungewollt mit.

Für schwere Denkmodelle gilt die Speichergrenze der Karte: 14B in Q4
(≈ 9 GB) passt auf 12 GB vollständig, 32B (≈ 20 GB) nicht mehr und teilt
sich mit dem Hauptprozessor. Das gehört vor die Wahl gesagt, nicht danach.

---

## Wer zuerst auf die Grafikkarte darf

Zwei Tage Fehlersuche steckten in einer vertauschten Reihenfolge. Gemessen,
zweimal derselbe Code:

| Reihenfolge | Ergebnis |
|---|---|
| llama.cpp → Whisper → erkennen | läuft |
| Whisper → llama.cpp → erkennen | **Prozess stirbt** |

llama.cpp richtet sich beim Laden seinen **eigenen CUDA-Kontext** ein.
CTranslate2 – die Maschine hinter faster-whisper – legt seine cuBLAS-Handles
dagegen schon **beim Laden des Modells** an. Kommt llama.cpp danach, zeigen
diese Handles ins Leere:

    RuntimeError: CUDA failed with error invalid resource handle

und mit etwas Pech stirbt der Prozess ohne Traceback. Beim Anrufen sah das so
aus: „das Modell lädt und dann Absturz". Beim Auflegen so: „CUDA failed with
error context is destroyed".

`open()` in `pipeline_call.py` lud genau in der tödlichen Richtung. Beim Lesen
sieht man das nicht – beide Reihenfolgen wirken vernünftig. Deshalb steht der
Grund jetzt als Kommentar im Code **und** als Prüfung im Rauchtest.

Was diese Suche so lange gemacht hat: jeder isolierte Test lief durch. Erst der
Vergleich **beider Richtungen im selben Versuchsaufbau** zeigte es. Ein
einzelner erfolgreicher Test beweist bei CUDA-Koexistenz nichts – es braucht
den Gegenversuch.

Wer eine dritte CUDA-Bibliothek dazunimmt: **zuerst laden, was seinen eigenen
Kontext aufbaut.**

## Nutzerdaten am Stück bewegen, nie Datei für Datei

`Move-Item` arbeitet bei Verzeichnissen rekursiv. Trifft es unterwegs auf einen
Pfad jenseits von 260 Zeichen – und die Klon-Laufzeit bringt reichlich davon mit
(`onnx/backend/test/data/node/…`) – bricht es **mitten im Umbau** ab. Gemessen an
einem echten Fehlschlag lag danach:

    dist\StreamForge\data       logs, models, output, tmp, voice-runtime, voices
    build\data-stash-61a465ba   config.json, secrets.json, personas.json, …

Der Datenbestand war auf zwei Orte zerrissen. Wäre PyInstaller weitergelaufen,
hätte es die eine Hälfte gelöscht und die andere wäre in `build\` verschollen.

`[System.IO.Directory]::Move` benennt statt zu kopieren: ein einziger Vorgang
auf Dateisystemebene, unteilbar, ohne Pfadlängenproblem und ohne Zeitverlust bei
23 GB. Nur wenn das nicht geht, wird auf robocopy ausgewichen. Scheitert beides:
abbrechen, **bevor** PyInstaller den Ordner leerräumt.

Zwei weitere Löcher derselben Art:

- Der **Rückweg verweigerte**, sobald PyInstaller selbst ein `data\` angelegt
  hatte (die vorgeladenen Modelle als Erstausstattung). Er schrieb nur eine
  Warnung und ließ 23,8 GB im Stash liegen. Jetzt wird zusammengeführt, die
  Nutzerdaten gewinnen.
- **Verwaiste Sicherungen** wurden nur unter `<Repo>\` gesucht. Der
  PyInstaller-Schritt legt sie aber unter `build\` ab – genau dort, wo es
  schiefging.

Merksatz: *Ein Schutz, der im Fehlerfall nur warnt, ist kein Schutz.* Wer den
Bau nicht Zeile für Zeile liest, hält seine Sachen für gelöscht.

## Was nicht gelockert werden darf

### Discord-Empfang: fragil und rechtlich gebunden

`app/discord_dave.py` greift in fremden Aufbau ein: es umhüllt
`voice_client._reader.decryptor.decrypt_rtp` aus `discord-ext-voice-recv` und
schiebt die DAVE-Entschlüsselung dazwischen. Discord verschlüsselt Sprache
zweifach; die Fremdbibliothek löst nur die äußere Schicht.

1. **Wartung.** Benennt `discord-ext-voice-recv` `decryptor` oder `decrypt_rtp`
   um, hört der Bot nichts mehr. Deshalb wirft `attach()` eine deutliche Meldung
   statt still zu versagen, und ein Test hält den Vertrag fest. Diese
   Absicherung nicht entfernen – ohne sie ist der Ausfall unsichtbar.
2. **Recht.** Fremde Stimmen ohne Einwilligung zu verarbeiten ist in Deutschland
   nach **§ 201 StGB** strafbar. Der Discord-Weg ist deshalb **fail-closed**:
   ohne ausdrückliche Bestätigung im Einrichtungsdialog bleibt er gesperrt, auch
   über die Kommandozeile. Der Bot sagt beim Betreten im Textkanal an, dass
   mitgehört wird; `!optout` verwirft den Ton eines Teilnehmers; Mitschnitt ist
   Vorgabe aus.

Nicht entschlüsselbarer Ton wird **verworfen**, nicht durchgereicht:
verschlüsselte Bytes im Opus-Decoder werden zu Rauschen, und Rauschen macht die
Spracherkennung zu Wörtern, die niemand gesagt hat.

### Ein Widerruf muss halten

`revoke()` löschte die Zustimmung, statt sie als widerrufen zu vermerken – der
Nachtrag beim nächsten Start hätte sie stillschweigend zurückgeholt. Der Eintrag
bleibt jetzt mit `accepted_at = 0` stehen. Bei allem, was Zustimmung verwaltet:
**Abwesenheit ist kein Widerruf.**

---

## Offen vor einer Veröffentlichung

In den Rechtstexten stehen Platzhalter, die ausgefüllt werden müssen:

| Datei | fehlt |
|---|---|
| `AGB.md` | `[Anschrift eintragen]`, `[E-Mail-Adresse eintragen]`, `[Gerichtsstand eintragen]` |
| `LICENSE` | `[ANSCHRIFT eintragen]`, `[E-MAIL eintragen]` |
| `SECURITY.md` | `[SICHERHEIT-E-MAIL eintragen]`, `[FRIST eintragen]` |

Diese Angaben sind rechtsverbindlich und personenbezogen – sie zu erfinden wäre
schlimmer, als sie offen zu lassen. Ein Test prüft, dass alle Platzhalter
**einheitlich erkennbar** sind (Wortlaut „eintragen"), damit sie vor einem
Release auffindbar bleiben und nicht mit ausgeliefert werden. Das Format
beibehalten.

Ebenfalls vor einer Weitergabe zu klären: eine mit `-WithPiper` gebaute Fassung
steht unter GPL-3.0 und darf nicht weitergegeben werden.

---

## Gestaltung

Farben und Schrift sind an **https://streamwizard.de/** angelehnt (die Webseite
gehört zum anderen Projekt des Nutzers und ist hier nur Vorlage): Akzent
`#8B5CF6`, Hintergrund `#151224`, Fläche `#221D3D`, Segoe UI. Die Werte stehen in
`app/gui/theme.py`, der Unterstützen-Knopf zeigt auf
`https://streamwizard.de/unterstuetzen`.

Die Fehlerfarbe wurde bewusst von `#EF4444` auf `#F87171` angehoben: der
Website-Ton erreichte auf dem dunklen Hintergrund keine 4,5:1 und wäre schlecht
lesbar gewesen. **Lesbarkeit geht vor Wiedererkennung**; Tests prüfen die
Kontrastverhältnisse.
