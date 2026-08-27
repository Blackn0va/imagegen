# Änderungsverlauf

Format angelehnt an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/),
Versionierung nach [SemVer](https://semver.org/lang/de/).

## [Unveröffentlicht]

### Behoben – Ein abgebrochener Bau ließ die Anwendung den Datenordner wechseln
**Der schwerste Fehler dieser Nacht – er hat zwei Stimmprofile gekostet.**

`portable.txt` entscheidet, wo die Anwendung ihre Daten sucht: neben der
.exe oder unter `%LOCALAPPDATA%`. Geschrieben wurde die Datei in **Zeile
1065 von 1089** – ganz am Ende des Baus.

Bricht der Bau vorher ab (bei uns: `WinError 145`, weil PyInstaller die
5 GB Klon-Laufzeit nicht löschen konnte), steht ein lauffähiges Programm
**ohne Marker** da. Es legt dann einen **zweiten** Datenbestand unter
`%LOCALAPPDATA%\StreamForge` an: neue Modelle, neue Stimmprofile, neue
Konfiguration. Der alte Bestand bleibt im Portable-Ordner liegen und geht
beim nächsten Bau unter. Für den Bediener sieht das aus, als sei alles
gelöscht worden.

- Der Marker wird jetzt **direkt nach PyInstaller** geschrieben, vor allen
  langen Schritten. Ein Test hält die Reihenfolge fest.
- `paths.is_portable()` erkennt einen portablen Ordner auch **ohne**
  Marker, wenn daneben ein `data`-Ordner mit Inhalt liegt. Ein fehlender
  Marker darf nie dazu führen, dass vorhandene Modelle übersehen werden.

### Behoben – `tools\` wurde nicht gesichert, `data\` schon
- PyInstaller räumt sein Ausgabeverzeichnis leer. Darin lag die 4,8 GB
  große Klon-Laufzeit mit sehr tiefen Pfaden
  (`onnx/backend/test/data/node/...`) – das Löschen scheitert dort mit
  `WinError 145` und brach den ganzen Bau ab.
- `tools\` wird jetzt genauso zur Seite gelegt und zurückgeholt wie
  `data\`, auch nach einem Fehlschlag. Nebengewinn: die Klon-Laufzeit
  muss nicht mehr bei jedem Bau neu kopiert werden.

### Behoben – Eine halb kopierte Laufzeit galt als fertig
- Die Prüfung „liegt schon da" sah nur nach `Scripts\python.exe`. Eine
  aus einem abgebrochenen Bau halb kopierte Laufzeit hat die auch – ihr
  fehlte aber `chatterbox`. Das Programm meldete danach „Klon-Laufzeit
  nicht eingerichtet", obwohl 4,8 GB dort lagen.
- Geprüft wird jetzt auf das Paket, auf das es ankommt.


### Zurückgenommen – Der CUDA-Index als Vorgabe war ein Fehler
Ich hatte den cu124-Index zur Vorgabe gemacht, weil ein Trockenlauf ein
482-MB-Wheel lieferte und `import llama_cpp` samt `ggml_cuda_init`
durchlief. Beides war wahr und beides genügte nicht:

```
Llama(model_path=...) → OSError [WinError -1073741795]   (0xc000001d)
```

`0xc000001d` ist **STATUS_ILLEGAL_INSTRUCTION**. Das Wheel nutzt
Befehlssätze (AVX-512, AVX-VNNI), die ein i9-10850K (Comet Lake) nicht
kennt. Der Import merkt davon nichts, weil die betroffene Rechenschleife
erst beim **Laden eines Modells** läuft. Andere Indizes gibt es nicht:
cu121–cu123 führen keine Wheels.

Für diese Rechnerkombination existiert also kein brauchbares CUDA-Wheel.
Der Index ist nicht mehr Vorgabe; wer eine passende CPU hat, gibt ihn an.

**Die eigentliche Lehre steckt in der Prüfung.** Der Bau prüfte nur, ob
sich `llama_cpp` importieren lässt — und das tat es. Jetzt wird ein
wirklich vorhandenes Modell geladen (`n_gpu_layers=99`, kleinstes GGUF im
Datenordner). Schlägt das fehl, fällt der Bau selbsttätig auf den CPU-Bau
zurück. Eine leere Antwort zählt dabei als Fehlschlag: bei einem harten
Abbruch kommt Python nicht mehr zum Schreiben.


### Behoben – `-Clean` ließ Nutzerdaten stranden
- Der Bau verschiebt `data` zur Seite, löscht das Bündel und legt sie
  zurück. Bricht dazwischen etwas ab – etwa eine laufende Anwendung, die
  den Ordner hält –, blieben die Daten in einem `.data-stash-XXXX` liegen,
  und der nächste Start fand einen **leeren Datenordner**. Für den
  Bediener sieht das aus, als seien Modelle und Einstellungen gelöscht.
- Genau das war passiert: Konfiguration, Zustimmung und der verschlüsselte
  Discord-Token lagen in einem verwaisten Stash.
- Drei Änderungen: der Bau **bricht vorher ab**, wenn die Anwendung noch
  läuft (mit PID in der Meldung); das Zurücklegen steht in einem
  `finally`; und verwaiste Sicherungen werden beim nächsten Lauf
  eingesammelt statt liegen gelassen.

### Behoben – Nach der AGB-Zustimmung blieb es bis zum Neustart bei der CPU
- Der Rechenweg hängt an `licensing.proprietary_gpu_allowed()`. Der
  AGB-Dialog gab die Zustimmung, schloss sich – und niemand plante neu.
  Es sah aus, als hätte das Zustimmen nichts bewirkt.
- Jetzt wird sofort neu geplant, der neue Rechenweg ins Protokoll
  geschrieben und die abhängigen Seiten aufgefrischt. Gilt auch für den
  Widerruf.

### Behoben – Rollbalken, wo es nichts zu rollen gibt
- `ScrollArea` zeigte den Balken **immer**, auch wenn der Inhalt
  vollständig hineinpasst. Ein Balken ohne Inhalt sieht nach verstecktem
  Inhalt aus. Er erscheint jetzt nur bei Bedarf, und das Mausrad tut
  nichts, wenn es nichts zu rollen gibt.
- Die **Protokollseite** steckte in einem Rollbereich, obwohl das Textfeld
  seinen eigenen mitbringt – ein Rollbereich im Rollbereich. Das Rad
  wirkte mal hier, mal dort, und das Feld bekam nie die volle Höhe.
  `_page_frame(..., scrollen=False)` ist der Weg für Seiten, die selbst
  rollen.

### Geändert – Das Protokoll lässt sich benutzen
- Es sprang bei **jeder** Zeile ans Ende. Wer hochrollte, um eine Meldung
  zu lesen, wurde beim nächsten Eintrag zurückgerissen. Jetzt wird nur
  nachgeführt, wenn die Ansicht ohnehin unten steht.
- Jede Meldung trägt eine **Uhrzeit** – vorher sah man nicht, ob etwas
  gerade eben oder vor einer Stunde passierte.
- Neu: „Alles kopieren" und „Leeren", dazu Strg+A und Strg+C im Feld.


### Behoben – Ein fehlendes `cublasLt64_12.dll` leerte das halbe Bündel
Beim ersten Bau mit CUDA fehlte `llama_cpp` **vollständig** im Ergebnis;
das fertige Programm meldete „llama_cpp nicht ladbar" und weder Chat noch
Telefonieren liefen. Eine Kette aus vier Gliedern:

1. `llama.dll` aus dem CUDA-Bau braucht `cudart64_12.dll`,
   `cublas64_12.dll` **und** `cublasLt64_12.dll` daneben. Das
   Kopiermuster kannte nur die ersten beiden.
2. Fehlt eine davon, scheitert schon `import llama_cpp`.
3. PyInstallers `collect_submodules('llama_cpp')` **importiert** das Paket
   – und liefert bei einem Fehlschlag **0 Module**, ohne Fehlermeldung.
4. Also landete nichts im Bündel. Gemessen: 0 statt 25 Module.

Besonders heimtückisch: ein Test über `pipeline_chat` meldete `GPU-Offload:
True`, weil dessen `prepare_gpu_dll_path()` die Pfade vorher zurechtlegt.
PyInstaller importiert **ohne** diese Hilfe – der Fehler war also
unsichtbar, solange man ihn über die Anwendung prüfte.

Behoben: `cublasLt64_*` in beiden Kopierzweigen, und der Bau prüft jetzt
nach, ob `llama_cpp` **für sich allein** importierbar ist. Eine Kopie, die
niemand nachrechnet, ist keine Zusicherung.


### Behoben – Einstellungen, die an der entscheidenden Stelle nicht galten
- **Chat: Temperatur und Antwortlänge** wurden nicht mitgegeben – im
  Chat-Fenster galten die Vorgaben der Bibliothek, während die
  Einstellungsseite andere Werte anzeigte. Der Telefonweg machte es längst
  richtig.
- **Chat: Modell- und Charakterwahl** wurden nie gespeichert und waren beim
  nächsten Start wieder weg.
- **„Ausgabeordner öffnen"** öffnete den Pfad von vor der
  Einstellungsänderung: vier Knöpfe fingen `config` beim Bauen der Seite
  ein statt die aktuelle Konfiguration zu lesen.
- **Schutzbegriffe galten nur im Bildweg.** Ein Video ist eine Folge von
  Bildern – dass die Schranke dort nicht griff, war eine Lücke.
- **Fertige Videos blieben unsichtbar**: `_on_job_finished` hatte keinen
  Zweig für `video`/`compose`, das Ergebnis stand nur im Protokoll.
- **Behälter und Codec passten nicht zusammen**: ein `.webm` bekam H.264,
  ein `.mov` konnte VP9 bekommen – ffmpeg bricht dabei ab. Der Wunschcodec
  richtet sich jetzt am Behälter aus.

### Geändert – Was nichts tut, sagt das jetzt
- Der Regler **„Bewegungsstärke"** wird von keiner der drei lieferbaren
  Video-Pipelines ausgewertet. Statt weiter Steuerung vorzutäuschen, steht
  das jetzt am Feld.
- **`svd-xt`** ist als „nicht umgesetzt" gekennzeichnet – es ist ein
  Bild-zu-Video-Modell, und diesen Weg gibt es im Programm nicht.


### Behoben – Ein Abbruch legte die ganze Anwendung lahm
- `_run_tiled` meldete den Abbruch mit `raise KeyboardInterrupt`. Das ist
  eine **BaseException**: sie wird weder von `upscale_image` noch von
  `run_upscale` noch von `JobQueue._run_job` gefangen und läuft aus dem
  Arbeiter-Thread heraus. Der Thread stirbt und wird nirgends neu
  gestartet – danach bleibt **jeder** weitere Auftrag (Bild, Video,
  Stimme, Download) für immer auf „wartend". Die Anwendung wirkt
  eingefroren, ohne dass ein Fehler erscheint.
- Betroffen war jedes Bild über der Kachelgröße (Vorgabe 512), also
  praktisch jedes. Jetzt `UpscaleCancelled` wie im Zweig daneben.
- Zusätzlich fängt `_worker_loop` jetzt `BaseException`: ein einzelner
  Fehlgriff kostet höchstens einen Auftrag, nie die Warteschlange.

### Behoben – Der Haken zu den Discord-Aufnahmen war eine leere Zusage
- „Aufnahmen der Kanalstimmen behalten" stand per Vorgabe auf **aus** und
  trug den Hinweis, fremde Stimmen aufzuzeichnen brauche deren
  Einverständnis. Gelesen wurde `discord_keep_audio` **nirgends** – die
  Stimmen aller Kanalteilnehmer landeten immer dauerhaft als WAV auf der
  Platte, gelöscht wurden sie nie.
- Das ist genau die Zusage, auf die sich der Betreiber gegenüber den
  Beteiligten beruft (§ 201 StGB, DSGVO). Ein Haken, der etwas verspricht
  und nichts tut, ist schlimmer als keiner.
- Neu: `discard_recording()` wirft die Aufnahme direkt nach dem Verstehen
  weg, sofern sie nicht ausdrücklich behalten werden soll. Nur im
  Discord-Weg – am eigenen Mikrofon spricht der Bediener selbst.

### Behoben – Weitere wirkungslose Bedienelemente
- **„Denken" auf der Telefon-Seite** wurde übergangen: `_call_start` gab
  ein `chat_spec` aus `config.chat_model` vor, und weil das immer gesetzt
  ist, kam `brain_model()` nie zum Zug. Die Statuszeile nannte trotzdem
  das gewählte Modell – angezeigt A, geladen B.
- **Die Sampler-Auswahl** (beide Bildseiten) tat nichts: der Scheduler kam
  allein aus `config.image_sampler` und wurde nur beim Laden gesetzt.
  `request.sampler` wurde ausschließlich in die **Metadaten** geschrieben –
  im PNG stand also der gewählte, aber nicht benutzte Sampler, und das
  Bild war nicht reproduzierbar. Jetzt gilt der Sampler je Auftrag, und
  die Metadaten nennen den tatsächlich benutzten (bei FLUX
  „flow-matching").
- **Das Startbild fürs Video** ging bedingungslos an die Pipeline. Alle
  drei lieferbaren Modelle sind reines Text-zu-Video; eine
  ImageToVideo-Klasse kommt im Programm nicht vor. Der Auftrag brach mit
  „unbekanntes Schlüsselwort image" ab. Jetzt wird geprüft, ob die
  Pipeline ein Bild annimmt – sonst läuft der Text durch und es wird
  gesagt, dass das Bild übergangen wurde. Der Hinweis am Feld verspricht
  nicht mehr „Bild wird animiert".


### Behoben – GPU-Chat brauchte keine handgesuchte Wheel-Adresse
- Im Bauskript stand, die offiziellen CUDA-Indizes von `llama-cpp-python`
  führten für Python 3.13 keine Wheels („geprüft: cu121–cu125 ohne
  cp313"). Der Schluss war falsch: gesucht wurde nach `cp313`, das Wheel
  trägt aber gar keine Python-Bindung.
- Trockenlauf gegen den cu124-Index (Seite des Autors, keine Fremdquelle):
  `llama_cpp_python-0.3.35-py3-none-win_amd64.whl`, **482,7 MB**.
  `py3-none` läuft auf 3.13, und 482 MB statt rund 10 MB ist der Bau mit
  CUDA.
- Neu: `-LlamaCudaIndex` mit diesem Index als Vorgabe. Ist `-WithCuda`
  gesetzt und keine feste Adresse angegeben, wird er von selbst genommen;
  schlägt das fehl, bleibt der CPU-Weg. Eine per `-LlamaCudaWheel`
  angegebene Adresse hat weiterhin Vorrang.
- Damit rechnet der Chat nicht mehr stillschweigend auf dem
  Hauptprozessor, nur weil beim Bauen ein Schalter fehlte.


### Behoben – Die gewählte Stimme war nicht die gesprochene
Aus einer Werkstatt-Prüfung mit 40 Agenten; jeder Befund einzeln
gegengeprüft. Sieben Fehler mit demselben Ausgang.

- **Ein gemerkter Fehlzustand zementierte sich selbst.** `_load_state()`
  hielt auch ein gespeichertes „nicht eingerichtet" für gültig. Nach der
  Reparatur las die Anwendung ihre eigene alte Antwort und meldete
  weiterhin den Fehler. Jetzt werden **nur positive Zustände** gemerkt –
  ein „ok" spart die teure Prüfung, ein „nicht ok" darf sich nicht
  festschreiben. `install()` räumt die Zustandsdatei zusätzlich weg.
- **Der Zwischenspeicher überholte die Wirklichkeit.** `available()` gilt
  jetzt nur noch, solange Interpreter *und* Arbeiter wirklich liegen –
  zwei billige Dateiabfragen.
- **`VoiceChoice.apply()` setzte nur Sprecher und Profil**, nicht aber
  `voice_model`. Welche Pipeline entsteht, entscheidet die Konfiguration –
  wer im Telefonat „Bark" wählte, sprach weiter mit dem Modell aus den
  Einstellungen. Neu: `VoiceChoice.configure()`, angewandt **vor** dem
  Bauen der Pipeline.
- **Angelernte Stimmen galten ungeprüft als bereit.** Ohne Klon-Laufzeit
  kann kein Profil sprechen; wegen ihrer hohen Klangnote stand eine stumme
  Profilstimme trotzdem ganz oben und wurde zur Vorauswahl.
- **Die Attrappe war der letzte Rückfall.** Drei Wege in
  `create_voice_pipeline` endeten beim Platzhalterton, sobald ein Motor
  fehlte. Neu: `_letzter_ausweg()` nimmt die **Windows-Stimme** – die ist
  überall da, braucht keinen Download und sagt wenigstens den Text. Die
  Attrappe bleibt nur, wenn selbst dort keine Stimme installiert ist.
- **`engine_available("clone")` antwortete aus dem Zwischenspeicher.** Der
  Sprechpfad prüft jetzt einmal je Programmlauf wirklich nach.

### Behoben – Der Nutzer sah nicht, was passiert war
- `tune_hint` blieb für immer auf „Hörprobe läuft" stehen. Jetzt steht
  dort, **wer gesprochen hat**: die angelernte Stimme, eine Ersatzstimme
  oder ein Platzhalterton – genau die Antwort auf „warum klingt das nicht
  nach mir?".
- Fehlgeschlagene Aufträge landeten ausschließlich im Protokoll, das eine
  eigene Seite ist. Sie erreichen jetzt auch die Seite, auf der geklickt
  wurde.
- Nach dem Einrichten der Klon-Laufzeit wird nachgeprüft, ob sie nun
  gefunden wird – vorher stand die Seite weiter auf „nicht eingerichtet",
  obwohl der Auftrag „fertig" meldete.


### Behoben – Angelernte Stimmen erzeugten nur ein Rauschen
Drei Fehler in einer Kette; jeder allein genügt, um jede Klonstimme
unbrauchbar zu machen.

1. **Die Laufzeit wurde nie gefunden.** `install()` richtet nach
   `data_dir()/voice-runtime` ein – `python_path()` und `worker_path()`
   suchten dort **nicht**, nur unter `exe_dir/tools`,
   `exe_dir/_internal/tools`, `exe_dir/.voice-venv` und
   `bundle_dir/tools`. Wer die Laufzeit einrichtete, lud mehrere Gigabyte
   und bekam danach dieselbe Meldung wie vorher: „nicht eingerichtet".
   *In der Sitzung der Einrichtung ging es noch*, weil `install()`
   `STREAMFORGE_VOICE_PYTHON` setzt – nach dem Neustart war das weg.
   Am Rechner des Nutzers nachgeprüft: die Umgebung war vollständig und
   einsatzbereit (torch 2.6.0+cu126, chatterbox 0.1.7, mehrsprachig).
2. **Der Arbeiter fehlte im Bündel.** Die PyInstaller-Spec packte
   `voice_worker.py` gar nicht ein. `install()` will ihn in die neue
   Umgebung kopieren, fand aber nichts – und ohne ihn ist selbst eine
   perfekt eingerichtete Laufzeit stumm.
3. **Der Platzhalterton galt als Erfolg.** Ohne Arbeiter fällt die
   Sprachausgabe auf `DummyVoicePipeline` zurück, und die liefert eine
   gültige WAV-Datei mit einer Tonfolge. `_synth_sentence` prüfte nur, ob
   *eine Datei* entstand – also wurde das Rauschen abgespielt, statt auf
   die Windows-Stimme auszuweichen. `VoiceResult.dummy` gab es längst, es
   wurde nur nicht ausgewertet.

Behoben: `data_dir()/voice-runtime` steht jetzt an **erster** Stelle beider
Suchlisten; die Spec liefert `voice_worker.py` mit; ein Attrappen-Ergebnis
führt zur Ersatzstimme statt zum Platzhalterton. `_worker_source()` sucht,
**wo der Arbeiter herkommt** (getrennt von `worker_path()`, das sucht, wo
er liegen soll). `install()` prüft am Ende selbst nach, ob die Laufzeit
danach gefunden wird – ein Erfolg, der nicht nachgeprüft wird, ist keiner.

### Behoben – Die Hörprobe wurde erzeugt, aber nie abgespielt
- „Hörprobe erzeugen" legte die Datei nach `data/output/audio/` und
  schrieb eine Zeile „Ausgabe: …" ins Protokoll. Mehr nicht. Wer den Knopf
  drückt, erwartet zu hören – und rätselt sonst, ob überhaupt etwas
  passiert ist.
- Fertige Stimm-Aufträge werden jetzt abgespielt (im Hintergrund, nie im
  Oberflächen-Thread), mit dem Dateinamen im Hinweistext. Schlägt das
  Abspielen fehl, wird der Pfad genannt.


### Behoben – Die Telefon-Seite ließ das Fenster stehen
- Gemessen: der Aufbau kostete **3,22 s**, davon rund 2,5 s allein ein
  `import torch` – ausgelöst von der GPU-Anzeige der Stimmen
  (`accel.torch_cuda_available`). Das lief im Oberflächen-Thread.
- `accel.torch_cuda_hint()` gibt es genau dafür: es liest nur
  `torch/version.py` (ein paar hundert Byte) und liefert ein bereits
  geprüftes Ergebnis, sobald eines vorliegt. Für eine **Anzeige** ist das
  richtig; die harte Prüfung gehört an die Stelle, wo wirklich gerechnet
  wird.
- Ergebnis: **3,22 s → 0,47 s**, und torch wird beim Aufbau gar nicht mehr
  geladen. Die verbliebene halbe Sekunde ist die einmalige
  PowerShell-Abfrage der Windows-Stimmen; sie wird bereits gemerkt.
- Eine Prüfung misst das in einem **eigenen Prozess** – im Testprozess ist
  torch längst geladen, und genau dann fiele der Fehler nicht auf.

### Behoben – Der Hinweis am Regler „Führung" stand verkehrt herum
- Dort stand: *„Niedrig hält sich näher an Tempo und Rhythmus der
  Referenz."* Bei Chatterbox ist es umgekehrt: ein **hoher** Wert bindet
  streng an die Referenz und klingt schnell gepresst, ein niedriger (etwa
  0,3) gibt freieres Tempo und wirkt natürlicher.
- Wer eine natürlichere Stimme suchte, wurde vom Text also in die falsche
  Richtung geschickt.
- Der Knopf „Hörprobe erzeugen" profitiert vom Dauerbetrieb: die erste
  Probe kostet das Modellladen, jede weitere rund 10 s. Damit lassen sich
  die Regler überhaupt erst hörend vergleichen.


### Behoben – Das Stimmmodell wurde bei JEDEM Satz neu geladen
- `_synth_sentence` rief die Sprachausgabe je Satz auf. Bei Chatterbox
  startet das jedes Mal einen eigenen Prozess, der das **6 GB große Modell
  neu lädt**. Der Arbeiter kann Sätze bündeln (`--text-file`, genau dafür
  gebaut) – der Anrufweg nutzte das nie.
- Gemessen auf einer RTX 4070 Ti: drei Sätze in einem Aufruf ergeben
  11,4 s Ton und brauchen 51 s; davon sind rund 16 s Rechnen und **35 s
  Modellladen**. Je Satz aufgerufen sind das über zwei Minuten für eine
  Antwort – und der Klang springt zwischen den Sätzen, weil jeder aus
  einem frischen Prozess kommt.
- Neu: `voice_worker.py serve` hält das Modell geladen und nimmt Sätze
  zeilenweise als JSON entgegen; `voice_runtime.VoiceServer` spricht damit.
  Gemessen über den echten Weg der Anwendung:

  | | vorher | jetzt |
  |---|---|---|
  | Zug 1 (mit Laden) | ~140 s | 140 s |
  | **Zug 2** | **~140 s** | **6,9 s** |

- Das Modell wird beim **Anrufstart** geladen (`warmup_voice`), nicht beim
  ersten Satz – die Wartezeit fällt damit in die Verbindungsphase statt
  mitten ins Gespräch.
- Ein Arbeiter je Sprache und Rechenweg, nicht je Satz. Beim Programmende
  wird er beendet (`shutdown_voice_servers`), sonst bliebe ein Prozess mit
  mehreren GB Modell zurück. Stirbt er, fällt die Sprachausgabe auf den
  Einzelaufruf zurück und vermerkt den Grund – lieber langsam sprechen als
  gar nicht.

### Behoben – Drei Fehler beim Umbau, gefunden durch die Prüfungen
- `_emit` schrieb **kein Zeilenende**. Beim Einzelaufruf folgenlos, im
  Dauerbetrieb blockiert `readline()` dadurch für immer. Die Prüfung dazu
  ruft `_emit` jetzt wirklich auf, statt im Quelltext nach der
  Schreibweise zu suchen – die Suche war selbst der Escape-Falle zum Opfer
  gefallen und konnte nie zutreffen.
- Ein **frisch angelegter** Arbeiter gilt ebenfalls als „läuft nicht" und
  wurde deshalb bei jedem Satz verworfen – womit das Modellladen
  zurückgekommen wäre. Neu unterscheidet `crashed` zwischen „nie
  gestartet" und „gestorben".
- Nach dem Zweig für die eingebaute Stimme wurde weiter auf
  `decision.profile` zugegriffen, das es dort nicht gibt: ohne Profil wäre
  die Sprachausgabe mit einem `NameError` gestorben.


### Hinzugefügt – Eine Stimme, die wirklich gut klingt
- Chatterbox kann auch **ohne Referenzaufnahme** sprechen; dann nimmt es
  seine eigene, synthetische Stimme. Der Arbeiter verlangte `--ref` bisher
  zwingend und setzte `audio_prompt_path` immer – damit war diese
  Möglichkeit verdeckt.
- Gemessen auf diesem Rechner: **deutsch, auf der Grafikkarte**, 4,84 s Ton
  in rund 7 s erzeugt, `{"device": "cuda", "multilingual": true}`. **Ohne
  jeden Download** – die 6 GB des Modells liegen bereits im
  Zwischenspeicher. Damit steht sie als *„Chatterbox — sehr natürlich,
  GPU"* an erster Stelle der Auswahl, vor allen Windows-Stimmen.
- Sauber getrennt von der Klonstimme: **mit** Referenz wird die Stimme
  einer realen Person nachgebildet – Einwilligung zwingend, fail-closed,
  unverändert. **Ohne** Referenz ist es eine synthetische Stimme, die
  niemandem gehört; dort ist auch nichts einzuwilligen.

### Behoben – Klang darf Telefontauglichkeit nicht schlagen
- Die Sortierung stellte den Klang über alles. Fehlt die Grafikkarte,
  fällt dieselbe Stimme auf 12 s/Satz zurück – und stand trotzdem vorn.
  Ein Gespräch ist damit unmöglich, egal wie gut sie klingt.
- Neu: `TELEFON_GRENZE_S = 5.0`. Was darüber liegt, rutscht ans Ende;
  unter den telefontauglichen entscheidet weiterhin der Klang. Aufgefallen
  ist das nur, weil die Prüfungen in **zwei** Umgebungen laufen – mit und
  ohne CUDA.


### Behoben – Eigene angelernte Stimmen endeten in einer Sackgasse
- Das gebaute Programm meldete beim Einrichten der Klon-Laufzeit: *„In der
  ausgelieferten Fassung gehört die Klon-Laufzeit zum Lieferumfang. Fehlt
  sie, bitte beim Anbieter melden."* Das ist keine Auskunft, sondern ein
  Sackgasse – am wenigsten hilfreich für den Anbieter selbst. Ohne die
  Laufzeit lassen sich **keine eigenen Stimmen anlernen oder verwenden**,
  und die tragen die beste Klangqualität.
- Der Weg war längst da: `install()` nimmt einen `base_python` entgegen.
  Es fehlte nur die Suche nach einem Interpreter. Neu:
  `voice_runtime.find_system_python()` prüft Py-Launcher (3.13/3.12/3.11),
  `PATH` und die üblichen Installationsstellen; `install_possible()` sagt
  **vor** dem Versuch, ob es klappen wird.
- Der Knopf richtet jetzt auch im gebauten Programm ein. Fehlt Python,
  öffnet er auf Wunsch python.org – statt an den Anbieter zu verweisen.
- `-WithVoiceRuntime` **brach ab**, wenn `.voice-venv` nicht schon von Hand
  angelegt war, mit einer Anleitung, was man vorher hätte tun sollen. Ein
  Schalter, der etwas mitliefern soll, beschafft es jetzt selbst.
  `voice-runtime install` nimmt dafür ein neues `--target`.
- Der Hinweis ohne Laufzeit sagt jetzt, was dadurch fehlt (eigene Stimmen)
  statt nur „fällt auf die Standardstimme zurück".


### Hinzugefügt – Denken und Sprechen sind zwei getrennt wählbare Modelle
- Am Telefon denkt ein Sprachmodell und ein **anderes** spricht die
  Antwort aus. Bisher war nur das zweite einstellbar; das erste kam
  stillschweigend von der Chat-Seite. Wer dort umstellte, änderte
  ungewollt das Telefonat mit.
- Neu auf der Telefon-Seite: **Denken** neben **Stimme**. `call_chat_model`
  gilt, wenn gesetzt; leer heißt weiterhin „dasselbe wie im Chat", damit
  bestehende Einstellungen gelten.
- Schwere Denkmodelle ergänzt: **Qwen2.5 14B** (Q4 ≈ 9 GB, passt auf einer
  12-GB-Karte noch vollständig auf die Grafikkarte) und **Qwen2.5 32B**
  (Q4 ≈ 20 GB, teilt sich zwischen Karte und Hauptprozessor auf – möglich,
  aber spürbar langsamer). Beide Apache-2.0.
- Die Hardware-Zeile nennt das Denkmodell beim Namen statt nur „GPU/CPU".

### Behoben – Die guten Stimmen waren unerreichbar, und niemand sagte es
- Gemessen: **kein einziges Stimmmodell war je geladen**. Übrig blieben
  die Windows-Stimmen (Formantsynthese aus den Neunzigern) – daher der
  Eindruck, die Anwendung klinge grundsätzlich blechern.
- `models.DEFAULTS[Task.VOICE]` zeigte auf **kokoro** – ein Modell, für das
  es *keine Umsetzung* gibt (`create_voice_pipeline` kennt nur piper, bark
  und clone) und das **kein Deutsch** spricht (en, es, fr, hi, it, ja, pt,
  zh). Vorgabe ist jetzt **bark-small**: MIT, deutsche Sprecher, rechnet
  auf der Grafikkarte.
- Ein Motor ohne Umsetzung erscheint nicht mehr als nutzbar. Vorher hätte
  jemand 330 MB geladen und danach die Attrappe gehört.
- **Neu: die Auswahl sagt, wie eine Stimme klingt** („künstlich",
  „natürlich", „sehr natürlich"). Vorher stand dort Tempo, Größe und
  Lizenz – alles außer dem, wonach gesucht wird. Sortiert wird jetzt nach
  Klang statt nach Ladezeit; ein einmaliger Download ist ein hinnehmbarer
  Preis, dauerhaft schlechter Klang nicht.
- Beim Öffnen der Telefon-Seite wird ungefragt gemeldet, wenn eine
  deutlich bessere Stimme bereitläge, samt Größe.

### Behoben – „GPU" stand da, wo CPU rechnete
- `torch_cuda_available()` liefert `(ja/nein, Begründung)`. Das Tupel als
  Ganzes ist **immer wahr** – so entsteht die Anzeige „GPU", während der
  Hauptprozessor rechnet. Jetzt wird das erste Feld ausgewertet.
- Das Tempo hängt am Rechenweg: „Bark 20 s/Satz" ist ein CPU-Wert. Auf der
  Grafikkarte ist es ein Vielfaches schneller (`ENGINE_SPEED_GPU`). Weil in
  der Auswahl der CPU-Wert stand, sah die einzige gute deutsche Stimme wie
  die schlechteste Wahl aus.
- Gemessen: das mitgelieferte **onnxruntime kennt keinen CUDA-Weg**
  (`['AzureExecutionProvider', 'CPUExecutionProvider']`). Piper kann daher
  nicht auf die Grafikkarte – das wird jetzt so gesagt, statt „GPU"
  zu behaupten.


### Hinzugefügt – Der Discord-Bot hört jetzt auch im normalen Sprachkanal
- Bisher konnte der Bot sprechen, aber nur in Stage-Kanälen zuhören. Der
  Grund lag nicht bei Discord, sondern an einer nicht verbundenen Stelle:
  Discord verschlüsselt Sprache zweifach – Transportschlüssel zwischen
  Server und Client, DAVE-Schlüssel zwischen den Teilnehmern.
  `discord-ext-voice-recv` löst nur die erste Schicht und reicht das
  Ergebnis direkt an den Opus-Decoder, wo es noch verschlüsselt ankommt.
  `davey` – ohnehin mitgeliefert – kann die zweite Schicht lösen, wurde im
  Empfangspfad aber nie aufgerufen.
- Neu: [`app/discord_dave.py`](app/discord_dave.py) legt den fehlenden
  Schritt dazwischen. Umhüllt wird nur `decryptor.decrypt_rtp`, und zwar
  je Instanz, nicht auf der Klasse – ein anderer Client im selben Prozess
  bleibt unberührt. Ändert die Fremdbibliothek diesen Namen, meldet sich
  `attach()` mit klarer Ursache, statt still zu versagen; ein Test hält
  den Vertrag (`AudioReader.decryptor`, `decrypt_rtp`, je Instanz
  ersetzbar) fest.
- **Nicht entschlüsselbarer Ton wird verworfen, nicht durchgereicht.**
  Verschlüsselte Bytes im Opus-Decoder werden zu Rauschen, und Rauschen
  macht die Spracherkennung zu Wörtern, die niemand gesagt hat.
- Kommt Ton an, der sich nicht öffnen lässt, bricht das Zuhören nach 15 s
  mit Begründung ab, statt endlos zu warten. Ein Zählwerk (`DaveStats`)
  unterscheidet die beiden Fälle, die sich sonst gleich anfühlen: niemand
  redet, oder es redet jemand und es kommt nicht an.
- Im Stage-Kanal wird der Bot zum Sprecher gemacht (`suppress=False`) –
  vorher spielte er dort ins Leere.

### Geändert – Der Pegel gilt für beide Wege
- Die Pegelanzeige war als „lokal" eingestuft und verschwand im
  Discord-Modus. Dort ist sie wichtiger als am eigenen Mikrofon: sie ist
  die einzige Rückmeldung, dass Ton aus dem Kanal ankommt **und** sich
  entschlüsseln lässt. Sie bleibt jetzt sichtbar und heißt dort „Kanal".
- Der Pegel wird im Discord-Modus laufend gemeldet, nicht erst am Ende
  eines Beitrags – vorher stand er während des ganzen Redens auf null.
- Die Hardware-Zeile nennt jetzt den Weg (`Ton: Discord-Kanal`). Sie ist
  die Zeile, die zuletzt CPU-statt-GPU verraten hat; sie soll auch
  verraten, wessen Stimme überhaupt ankommt.
- Beim Betreten wird gesagt, ob der Bot zuhört, und der Prüfcode der
  Verschlüsselung ausgegeben – Discord zeigt allen denselben Code.

### Behoben – Ein Backspace verhinderte, dass die CUDA-Laufzeit mitkam
- In `build-windows.ps1` stand im Suchmuster für die CUDA-DLLs
  `nvidia\*<0x08>in\cudart64_*.dll` statt `nvidia\*\bin\cudart64_*.dll`:
  beim Schreiben über ein Skript war `\b` als Backspace ausgewertet
  worden. Das Muster traf nie etwas.
- Folge: die CUDA-Laufzeit aus den `nvidia`-Paketen landete nicht neben
  `ggml-cuda.dll`. Fehlten passende DLLs in `torch\lib`, lud ggml-cuda
  still nicht und der Chat rechnete auf der CPU – ohne Fehlermeldung.
  Das ist ein zweiter, vom `device_for()`-Fehler unabhängiger Weg zu
  „Antworten: CPU" trotz CUDA-Wheel.
- Der Steuerzeichen-Wächter in den Tests prüfte nur sechs
  Markdown-Dateien, obwohl sein eigener Kommentar `build-windows` als
  Beispiel nannte. Er läuft jetzt über alle `.md`, `.ps1`, `.py` und
  `.spec` (52 Dateien) und prüft zusätzlich, dass dieser eine Pfad
  vollständig ist – repariert steht dort ein gültiger Pfad, das
  Steuerzeichen allein genügt als Merkmal also nicht.

### Behoben – `davey` wäre womöglich nicht im Bündel gelandet
- Die Spec sicherte davey über `collect_dynamic_libs()`. Gemessen liefert
  das für davey wie für nacl **null Treffer**: ein `.pyd` ist für
  PyInstaller ein Modul, keine Bibliothek. Ersetzt durch
  `collect_submodules()`, das `davey.davey` findet, plus expliziter
  Hidden-Import.
- Der Build prüft jetzt davey wie libopus. Beide Ausfälle sehen sonst
  gleich aus: ein Bot, der ohne Fehlermeldung nichts hört.
- `attach()` meldet ein fehlendes davey als verständlichen Fehler mit
  Nachrüst-Anleitung statt als rohen `ModuleNotFoundError`.


### Geändert – Hacker-Persona hilft, statt pauschal abzulehnen
- Die Persona verweigerte legitime Sicherheitsfragen mit einer
  Standard-Absage. Der Text wurde gegen das mitgelieferte 3B-Modell
  empirisch neu abgestimmt (Temperatur 0, mehrere Fassungen gemessen).
- Erkenntnis, die im Wortlaut steckt: ein kleines Modell greift
  Negativ-Begriffe auf – je mehr „Schadsoftware", „Waffen", „illegal" im
  Prompt stehen, desto eher verweigert es auch bei erlaubten Fragen.
  Deshalb positiver Rahmen, „auf jede konkrete Frage sofort", eine kurze
  statt langer Grenze. Gemessen: SQLi, ROP, XSS, Portscanner, Buffer
  Overflow werden zuverlässig beantwortet (6/6).
- Die Grenze bleibt und wirkt: Ransomware gegen eine Klinik, der Angriff
  auf das Konto einer benannten Person, Waffenbau und der Einbruch ins
  fremde WLAN werden weiter abgelehnt.
- **Fehler behoben, der die Verbesserung nie ankommen ließ:** mitgelieferte
  Personas wurden nur beim allerersten Start herausgeschrieben; danach
  gewann immer die Datei. Wer die App schon nutzte, behielt den alten
  Text. Jetzt gleicht `write_defaults()` beim Start ab: eine unveränderte
  Vorgabe wird über ihre Inhalts-Signatur erkannt und aktualisiert, eine
  selbst geänderte bleibt. Alte Dateien ohne Signatur werden einstufig
  migriert.
- Nebenbei ein zweiter Fehler: `revoke()` löschte die Zustimmung, statt
  sie als widerrufen zu vermerken – ein Widerruf wäre beim nächsten Start
  stillschweigend zurückgeholt worden. Der Eintrag bleibt jetzt mit
  `accepted_at = 0` stehen. `sync_agb_coverage()` trägt beim Start nach,
  was die AGB abdecken, aber noch offen ist (betrifft alle, die vor der
  Sammelzustimmung zugestimmt haben) – ohne einen Widerruf zu übergehen.


### Hinzugefügt – Telefonieren über einen Discord-Bot
- Der Weg des Gesprächs ist umschaltbar: eigenes Mikrofon oder ein Bot im
  Sprachkanal. `call_transport.py` trennt den Weg vom Ablauf, damit nicht
  jede Zeile im Gesprächskreis ein `if discord:` bekommt.
- **Der Empfang hatte zunächst eine Grenze:** seit dem 2.3.2026
  verschlüsselt Discord Sprachkanäle Ende zu Ende (DAVE); Zuhören ging nur
  im Stage-Kanal. *Richtigstellung (siehe oben):* die Annahme, keine
  Python-Bibliothek könne das lösen, galt für `discord-ext-voice-recv`,
  nicht insgesamt – `davey` kann es, war im Empfangspfad nur nicht
  angeschlossen. Seit `discord_dave.py` hört der Bot in normalen
  Sprachkanälen mit.
- Fail-closed: ohne ausdrückliche Bestätigung, dass die Einwilligung der
  Beteiligten eingeholt wird, bleibt der Discord-Weg gesperrt – geprüft
  vor allem anderen, auch beim Start über die Kommandozeile. Der Bot sagt
  beim Betreten im Textkanal an, dass mitgehört wird; `!optout` verwirft
  den Ton eines Teilnehmers. Mitschnitt ist Vorgabe aus.
- Neuer `secrets_store`: der Bot-Token liegt nicht in `config.json`, sondern
  verschlüsselt in `secrets.json` (Windows-DPAPI, an das Benutzerkonto
  gebunden). Angezeigt wird nur, ob einer hinterlegt ist, plus vier Zeichen
  zum Wiedererkennen.
- Tonformate: Discord liefert 48 kHz Stereo, Whisper braucht 16 kHz Mono.
  Die Umrechnung läuft über `audioop` mit gehaltenem Zustand – ohne den
  knackt es an jeder Blockgrenze. Nachgemessen: Tonhöhe und Pegel bleiben
  erhalten.
- Für den Rückweg spricht SAPI im Discord-Modus gleich in 48 kHz Stereo.
  22050 auf 48000 ist das Verhältnis 147:320; dabei bliebe nur lineare
  Interpolation, und die hört man.
- `-WithDiscord` im Bau (Vorgabe an). Die libopus-DLL muss nach
  `_internal/discord/bin/` – `discord.opus._load_default()` sucht sie hart
  dort. Im Wurzelordner bleibt der Bot stumm, ohne Fehlermeldung.

### Hinzugefügt – AGB
- `AGB.md` fehlte, deshalb war der Zustimmungsdialog beim ersten Start leer.
  Jetzt 199 Zeilen in 15 Abschnitten. Die Fassungskennung ist ein Hash des
  Textes: ändert sich der Wortlaut, muss neu zugestimmt werden.
- Offene Platzhalter: Anschrift, E-Mail und Gerichtsstand sind mit
  `[… eintragen]` markiert und vor einer Weitergabe auszufüllen.

### Geändert – Farben und Gestaltung nach streamwizard.de
- Palette aus dem CSS der Website übernommen: `#8B5CF6` als Akzent,
  `#151224`/`#1F1B36` als Grund, dazu die Zustandsfarben. Die
  halbdurchsichtigen Kartenflächen der Website sind ausgerechnet, weil
  tkinter weder Verlauf noch Transparenz kennt.
- Das Fehlerrot der Website (`#EF4444`) kam auf der dunklen Karte nur auf
  4,3:1 – unter der Lesbarkeitsschwelle. Im Dunkelmodus steht deshalb
  `#F87171` (5,8:1), dieselbe Farbreihe. Kontraste sind als Prüfung
  festgehalten.
- Neuer Knopf **Unterstützen** unten in der Navigation
  (https://streamwizard.de/unterstuetzen).

### Geändert – nur noch brauchbare Audiogeräte in der Auswahl
- Ungefiltert waren es auf diesem Rechner 19 Mikrofone und 28 Ausgänge:
  dasselbe Headset dreimal (MME, DirectSound, WASAPI), dazu Sammelgeräte
  und acht WDM-KS-Einträge, über die PortAudio gar nicht lesen kann.
- Übrig bleiben 3 bzw. 4 echte Möglichkeiten. Ein gespeichertes Gerät
  bleibt sichtbar, auch wenn der Filter es sonst ausblenden würde – sonst
  springt die Wahl beim Öffnen still zurück. Haken *alle Geräte* zeigt alles.

### Geändert – Stimmen mit Angabe, was sie kosten
- Neuer `voice_catalog()` führt Windows-Stimmen, Modellstimmen und
  angelernte Stimmen zusammen und nennt zu jeder Geschwindigkeit, Größe,
  Lizenz und was ihr noch fehlt. Sofort brauchbare stehen vorn.
- Vorher sahen fünf Windows-Stimmen (0,5 s/Satz) und drei Bark-Sprecher
  (20 s/Satz, Modell nicht geladen) gleich aus.


### Geändert – Start dauert 288 ms statt 2,8 s
- `pipeline_onnx.runtime_available()` importierte `optimum.onnxruntime`,
  nur um zu prüfen, ob es da ist. Gemessen 3,8 s, und gefragt wird das beim
  Start zweimal (DirectML, OpenVINO), bevor das Fenster erscheint. Jetzt
  wird mit `find_spec()` nur nachgesehen (Millisekunden) und das Ergebnis
  gemerkt; wirklich geladen wird erst beim Export, wo die Zeit neben dem
  Modell nicht auffällt.
- `_report_startup()` rief `compose.available()` – das startet das
  220-MB-ffmpeg, um seine Fassung zu lesen (2,4 s). Läuft jetzt im
  Hintergrund wie die übrigen teuren Prüfungen.
- Zusammen: `Runtime()` 4038 → 203 ms, Fenster sichtbar nach 2790 → 288 ms.
  Zwei Prüfungen in der Testsammlung halten das fest.

### Geändert – AGB-Zustimmung deckt alle Lizenzpunkte ab
- Wer beim ersten Start die AGB bestätigt, musste auf der Lizenzseite
  dieselben Auflagen noch einmal einzeln abhaken. Zur Wahl stand dort
  nichts anderes – die AGB nennen sie bereits im Wortlaut.
- `accept_agb()` bestätigt jetzt alle registrierten Komponenten mit.
  Protokolliert wird weiter jede einzeln, mit dem Vermerk „über die
  AGB-Zustimmung mitbestätigt". Ein Widerruf auf der Lizenzseite hält, bis
  die AGB erneut bestätigt werden.
- Nicht berührt: die dokumentierte Einwilligung je angelernter Stimme. Die
  wird pro Profil erhoben und bleibt die eigentliche Schranke.

### Behoben – nur eine von fünf Windows-Stimmen war wählbar
- Die Stimmenabfrage lief über `powershell` (5.1, .NET Framework). Die
  meldet 2 Stimmen. `pwsh` (7, .NET) meldet 5 – Katja und Stefan stehen
  unter `Speech_OneCore\Voices`, und nur die .NET-Portierung von
  System.Speech liest diesen Registry-Pfad mit. Es wird jetzt `pwsh`
  bevorzugt, `powershell` bleibt Rückfall.
- Stimmen werden sortiert: deutsche zuerst, die alten „Desktop"-Fassungen
  ans Ende.
- Die Modellstimmen (Bark) waren wählbar, obwohl das Modell nicht geladen
  war – wer eine nahm, hörte nichts. Sie tragen jetzt „lädt beim ersten
  Mal" bzw. „langsam (~20 s/Satz)" im Namen.
- Scheitert die gewählte Stimme trotzdem, wird auf eine Windows-Stimme
  ausgewichen statt stumm zu bleiben – einmal je Gespräch ausgesprochen.
  Am Telefon ist Stille der schlimmste Ausgang.

### Hinzugefügt – Verstärkungsregler fürs Mikrofon
- Headsets liefern oft Effektivwerte um 0,008; die Auslöseschwelle liegt
  bei 0,006 – Sprache gilt dann als Stille. Der Windows-Regler dafür sitzt
  drei Menüs tief und wirkt geräteweit.
- Neuer Regler auf der Telefon-Seite (1× bis 20×, mit dB-Anzeige). Wirkt
  direkt nach dem Einlesen, also auf Anzeige, Sprech-Erkennung und die
  Datei für Whisper gleichermaßen.
- Weiche Begrenzung statt hartem Abschneiden: ab 0,9 geht es über `tanh`
  in die Sättigung. Hartes Clipping würde Sprache verzerren und Whisper das
  Erkennen erschweren.
- Der Mikrofontest rechnet aus dem gemessenen Spitzenpegel aus, welche
  Verstärkung fehlt, und nennt sie.

### Behoben – Auswahlfelder standen seitenweit unterschiedlich
- `ComboRow` legte seine Auswahl mit `sticky="w"` in eine Spalte, die
  mitwächst – `EntryRow` und `TextRow` nutzen `"ew"`. Die Comboboxen
  blieben schmal, daneben klaffte Leerraum. Betraf 17 Stellen auf sieben
  Seiten.
- Die Chat-Kopfzeile hatte „Modell" und „Charakter" in getrennten Frames,
  beide mit eigener gewichteter Spalte. Da die Beschriftungen verschieden
  breit sind, begannen die Felder an verschiedenen x-Positionen. Jetzt ein
  gemeinsames Raster (nachgemessen: 0 px Abweichung).
- Neue Prüfung `_test_page_layout()` baut jede Seite und meldet jedes Feld,
  das in einer wachsenden Spalte schmal bleibt.

### Hinzugefügt – Hinweis, wenn der Chat auf der CPU rechnet
- Gemessen mit Qwen2.5-VL 3B auf der CPU: erstes Token nach 2,7 s, Antwort
  (44 Wörter) nach 7,6 s. Auf der GPU rund zehnmal schneller. Im Telefonat
  ist das der Unterschied zwischen Gespräch und Warteschleife.
- Chat- und Telefon-Seite sagen es jetzt im Klartext, mit dem Namen der
  gefundenen Karte – bisher stand es nur im Protokoll.
- `build-windows.ps1` merkt sich die URL des CUDA-Wheels in
  `.llama-cuda-wheel.txt` und nimmt sie beim nächsten Bau von selbst. Wird
  ohne gebaut, obwohl `-WithCuda` gilt, warnt das Skript. Automatisch
  auflösen geht nicht: die offiziellen CUDA-Indizes von llama-cpp-python
  führen für Python 3.13 keine Wheels (geprüft: cu121–cu125 ohne cp313,
  cu126 antwortet nicht).

### Geändert – `-Clean` baut neu, statt alles neu zu laden
- `-Clean` löschte `build\`, `dist\` und das Venv. Von den vier
  betroffenen Sammlungen war genau eine ein Bauartefakt. Mit weg waren: die
  pip-Pakete im Venv (torch mit CUDA rund 8 GB), der Modell-Zwischenspeicher
  unter `build\stage-models` (SDXL 6,6 GB) und der ffmpeg-Download. Ein
  "nur neu bauen" kostete dadurch über 15 GB Download.
- Jetzt entfernt `-Clean` nur den PyInstaller-Arbeitsordner und das fertige
  Bundle. Die Nutzerdaten darin werden vorher weggetragen und danach
  zurückgelegt.
- Wer eine der Sammlungen wirklich weghaben will, sagt es einzeln:
  `-PurgeData` (Modelle und Konfiguration), `-FreshVenv` (Pakete),
  `-PurgeCache` (vorgeladene Modelle und ffmpeg).
- Das Skript schreibt beim Aufräumen dazu, was stehen bleibt und mit welchem
  Schalter es fiele – und nennt die Größe dessen, was es entfernt.

### Behoben – Telefonieren brach beim Anruf sofort ab
- `record_turn()` forderte fest 16 kHz. Geräte über Windows WASAPI und WDM
  laufen im Shared Mode fest auf 48 kHz und lehnen das mit
  `Invalid sample rate [PaErrorCode -9997]` ab – der Anruf endete, bevor er
  begann. Neu handelt `pick_input_rate()` die Rate mit dem Gerät aus; die
  Aufnahme wird danach einmal auf die 16 kHz gerechnet, die Whisper braucht.
  Nachgemessen: derselbe Satz wird über 48 kHz wortgleich erkannt wie direkt.
- Dieselbe Falle bei der Wiedergabe: die Windows-Stimmen liefern 22050 Hz,
  ein WASAPI-Gerät nimmt nur 48000. `pick_output_rate()` rechnet um.
- `resample()` filtert vor dem Herunterrechnen (scipy `resample_poly`, ohne
  scipy Kastenfilter). Ohne Tiefpass falten Frequenzen über der halben
  Zielrate als Störtöne zurück, und Whisper hört Wörter, die niemand
  gesagt hat.
- PortAudio-Fehler kommen als Klartext an, nicht als Zahlencode. Bei einem
  nicht auslesbaren Gerät (`Windows WDM-KS`) wird ein konkreter Ersatz
  genannt – dasselbe Mikrofon über eine Schnittstelle, die funktioniert.

### Geändert – Telefon-Seite aufgeräumt
- Alle Auswahlfelder liegen jetzt in einem Raster mit festen
  Beschriftungsspalten. Vorher hatten die Kopfzeilen `weight=1` auf Spalten
  gesetzt, deren Inhalt mit `sticky="w"` links klebte – eine der beschwerten
  Spalten gab es gar nicht. Die Felder standen dadurch unterschiedlich weit
  links. Nachgemessen: Abweichung jetzt 0 px statt sichtbar versetzt.
- Neue `LevelMeter`-Anzeige statt `ttk.Progressbar`: logarithmische Skala mit
  Schwellenmarke und Spitzenhalter. Sprechlautstärke (Effektivwert um 0,05)
  füllt jetzt gut die Hälfte des Balkens statt 5 %.
- Die Geräteliste zeigt die Schnittstelle mit an, sortiert brauchbare nach
  vorn und markiert die übrigen. Neuer Knopf *Geräte neu laden* für
  Headsets, die nach dem Start eingesteckt werden.
- Der Pegel wird nicht mehr aus dem Aufnahme-Thread gezeichnet (rund 33
  `after()`-Aufrufe je Sekunde aus einem Fremdthread), sondern im Tk-Takt
  alle 50 ms abgeholt.


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
  DMC-Nummern, Stückzahlen und fertiger Größe in Zentimetern.
- **DMC-Farbabgleich** (neues Modul `app/dmc.py`, 489 Farben). Jede
  Bildfarbe wird auf die nächstgelegene bestellbare DMC-Farbe abgebildet –
  beschränkt auf die 445 Nummern, die es beim Diamond Painting als Stein
  gibt, nicht auf alle Garnfarben. Ohne das ist eine Vorlage nicht
  bestellbar: Steine werden nach Nummer verkauft, nicht nach Hexwert.
  Abschaltbar über `--no-dmc` bzw. `diamond_use_dmc: false`.
- Zwei Bildfarben, die auf dieselbe DMC-Nummer fallen, verschmelzen zu
  einer. Der Auftrag meldet, wenn dadurch weniger Farben herauskommen als
  angefordert.
- Die RGB-Werte der Tabelle sind Näherungen – ein Harzstein hat kein
  definiertes sRGB. Jede erzeugte Farbliste weist darauf hin und nennt den
  Abgleich mit der Farbkarte des Anbieters als Pflichtschritt vor einer
  großen Bestellung.
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

### Hinzugefügt – Maskenwerkzeug in der Oberfläche
- Neues Modul `app/gui/mask_editor.py`. Bei „Bereich ersetzen" öffnet
  *Maske malen …* das Bild direkt in der Anwendung: malen mit links,
  radieren mit rechts, Mausrad für die Pinselgröße, Strg+Z für zurück,
  dazu „Alles füllen" und „Leeren".
- Damit entfällt der letzte Schritt, für den ein zweites Programm nötig
  war – bisher musste die Maske extern gemalt und als Datei ausgewählt
  werden.
- Zwei Auflösungen laufen parallel: die Anzeige ist auf 900 px verkleinert,
  damit auch ein 6000-Pixel-Bild flüssig zu bemalen ist; die Maske wird
  zusätzlich in voller Größe des Originals geführt und auch so
  gespeichert. Eine verkleinerte Maske hochzurechnen gäbe ausgefranste
  Kanten – genau an der Naht zwischen altem und neuem Bild.
- Rückgängig arbeitet über die Striche, nicht über Kopien der Maske: eine
  Kopie einer 6000x4000-Maske sind 24 MB, zehn davon 240 MB nur für die
  Rücknahme.
- Leere und vollständig gefüllte Masken werden abgelehnt – beide würden
  nichts bewirken bzw. gehören in den img2img-Modus. Die Prüfung
  (`problem()`) ist vom Dialog getrennt und damit ohne Fenster testbar.
- Neue Zeile `widgets.ButtonRow` – eine Schaltfläche als Formularzeile, die
  sich wie die Eingabezeilen aufgabenabhängig aus- und einblendet.

### Behoben – README beschrieb das Stimm-Anlernen falsch
- Dort stand, Anlernen schreibe „noch ein Platzhalter-Artefakt". Tatsächlich
  ist Zero-Shot vollständig umgesetzt: `build_reference()` baut aus dem
  Rohmaterial eine saubere, einkanalige, normalisierte Referenzaufnahme –
  genau das, was das Modell zur Laufzeit braucht. Nicht umgesetzt ist nur
  `finetune`, und das lehnt mit klarer Meldung ab.

### Geändert – Oberfläche baut sich nach der gewählten Aufgabe um
- Bisher standen auf der Seite **Bild bearbeiten** alle vier Karten
  gleichzeitig da; nicht passende Felder wurden nur ausgegraut. Der leere
  Platz blieb, und man sah immer alles auf einmal.
- Neue Mischklasse `widgets.Visible`: Zeilen blenden sich vollständig aus
  – Beschriftung, Eingabe und Hinweistext. Über `grid_remove()` bleiben
  die Rasterangaben erhalten, die Zeilen darunter rücken nach.
- `Card.set_visible()` blendet ganze Karten aus. Sichtbar ist jetzt nur
  noch, was zur Aufgabe gehört: „Vergrößern" beim Vergrößern, „Einfärben"
  beim Einfärben, „Diamond Painting" bei der Vorlage.
- Auch innerhalb der Karte folgen die Zeilen der Aufgabe: die Maske gibt
  es nur beim Ersetzen, Prompt und Sampler nur dort, wo ein Modell rechnet,
  der Formatwähler nicht bei der Vorlage (die schreibt immer PNG).
  „Nachschärfen" holt die Modellregler bei Bedarf dazu.
- `_set_rows_state()` bleibt für Felder, die sichtbar, aber gerade nicht
  bedienbar sein sollen.

### Hinzugefügt – Vorschau auf das Ergebnis vor dem Start
- Neue Zeile unter den Eingaben rechnet mit dem gewählten Bild aus, was
  herauskommt – und aktualisiert sich live beim Ändern der Zahlen:
  - Vergrößern: `Ergebnis: 2560x1920 px (4.9 MP)`
  - Diffusionsmodi: Zielgröße samt Rundung auf ein Vielfaches von 8
  - Vorlage: `Raster: 100x74 Steine = 7400 Stück · fertig 28.0 x 20.7 cm
    (rund, 2.8 mm) · bis zu 24 Farben · Vorlage 1824x1356 px`
- Die Höchstkante wird eingerechnet. Ohne lesbares Bild bleibt die Zeile
  leer – eine erfundene Zahl wäre schlechter als keine.

### Behoben – Download von FLUX.1 schlug ohne brauchbare Meldung fehl
- **Ursache:** FLUX.1 ist auf Hugging Face ein zugangsbeschränktes Repo
  (`gated: auto`). Die Metadaten sind öffentlich, die Dateien nicht. Ohne
  Token lief die Dateiliste durch, der Auftrag startete – und starb erst
  beim ersten Dateiabruf an einem nackten `HTTPError: 401`.
- Neue Vorabprüfung `repo_access()`: Zugang wird geklärt, **bevor** ein
  Byte fließt. Fehlt er, kommt `ModelAccessDenied` mit den drei Schritten
  (Bedingungen annehmen, Token erzeugen, `HF_TOKEN` setzen).
- Neuer Befehl `streamforge models access <name>` prüft Zugang, Größe und
  Plattenplatz, ohne etwas zu laden. Rückgabe 1, wenn etwas fehlt.
- `classify_hub_error()` übersetzt 401/403, 404, 429, 5xx, volle Platte und
  Verbindungsabbrüche in Klartext mit dem jeweils nächsten Schritt.
- Kein Token in der Konfiguration – gelesen wird nur `HF_TOKEN` bzw. die
  Datei aus `huggingface-cli login`. Ein Token gehört nicht im Klartext in
  eine Einstellungsdatei.

### Behoben – Fortsetzung eines abgebrochenen Downloads war wirkungslos
- `_download_file()` baut Wiederaufnahme über `.part` und Range-Requests
  auf, der Fehlerpfad in `download()` löschte diese Teile aber sofort
  wieder. Bei FLUX (24 GB) fing damit jeder Verbindungsabriss wieder bei
  null an. `_cleanup_incomplete()` behält angefangene Dateien jetzt;
  weggeräumt wird nur auf ausdrücklichen Wunsch (`keep_parts=False`).
  Das Modell gilt weiterhin als unvollständig – dafür sorgen Teil-Marker
  und `_looks_partial()`.
- Jede Datei bekommt bis zu vier Versuche mit wachsender Pause
  (`_download_file_resilient`). Wiederholt wird nur, was sich durch
  Wiederholen ändern kann – 401, 404 und volle Platte brechen sofort ab.
- Vor dem Start wird der Plattenplatz geprüft (`check_disk_space`), inkl.
  2 GB Reserve. Vorher lief ein 24-GB-Download bis zur vollen Platte.
- Abbruch und Fehler melden jetzt, wie viel liegen bleibt und dass ein
  neuer Anlauf dort fortsetzt.

### Behoben – Handlungsanweisungen wurden von der Fehleranzeige zerstört
- `accel.clean_error()` zog Zeilenumbrüche zusammen und kürzte auf 240
  Zeichen – bei einer mehrzeiligen Anleitung fiel genau der Teil weg, der
  sagt, was zu tun ist. Eigene Ablehnungen mit `expected = True`
  (Inhaltssperre, Lizenztor, Repo-Zugang) behalten Wortlaut und Zeilen;
  Fremdfehler werden weiterhin eingedampft.
- `ModelBlocked` und `ModelAccessDenied` tragen jetzt `expected`.

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
