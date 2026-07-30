# Modelle und Lizenzen

Die Anwendung wird **verkauft**. Deshalb gilt: nur Modelle, bei denen die
kommerzielle Nutzung eindeutig erlaubt ist, stehen als Vorgabe zur Verfügung.
Modelle mit Bedingungen sind gekennzeichnet und müssen einzeln freigegeben
werden. Nicht-kommerzielle Modelle sind im Quelltext bewusst als `DENIED`
eingetragen – damit sichtbar bleibt, *warum* sie fehlen, und der Download
verweigert wird (`app/models.py`, `check_allowed`).

Stand der Prüfung: 2026-07-29. **Vor jedem Release neu prüfen** – Anbieter
ändern Lizenzen rückwirkend für neue Downloads (bei Stability AI 2024
passiert).

## Übersicht

| Modell | Aufgabe | Lizenz | kommerziell erlaubt? | Auflagen |
|---|---|---|---|---|
| `sdxl-base` (stabilityai/stable-diffusion-xl-base-1.0) | Bild | CreativeML Open RAIL++-M | ja | Nutzungsbeschränkungen aus Anhang A an Endkunden weitergeben; Lizenzkopie beilegen |
| `ssd-1b` (segmind/SSD-1B) | Bild | Apache-2.0 | ja | Namensnennung im Lizenzhinweis |
| `sd15` (stable-diffusion-v1-5/stable-diffusion-v1-5) | Bild | CreativeML Open RAIL-M | ja | Nutzungsbeschränkungen an Endkunden weitergeben |
| `flux-schnell` (black-forest-labs/FLUX.1-schnell) | Bild | Apache-2.0 | ja | Namensnennung |
| `sdxl-turbo` (stabilityai/sdxl-turbo) | Bild | Stability AI Community License | **bedingt** | Umsatzgrenze; darüber Enterprise-Lizenz; „Powered by Stability AI“; Registrierung |
| `wan-t2v-1.3b` (Wan-AI/Wan2.1-T2V-1.3B) | Video | Apache-2.0 | ja | Namensnennung |
| `cogvideox-2b` (THUDM/CogVideoX-2b) | Video | Apache-2.0 | ja | Namensnennung; **nur die 2B-Fassung**, CogVideoX-5B hat eine eigene Lizenz |
| `animatediff` (guoyww/animatediff-motion-adapter-v1-5-3) | Video | Apache-2.0 | ja | Basismodell SD 1.5 bringt RAIL-Beschränkungen mit |
| `svd-xt` (stabilityai/stable-video-diffusion-img2vid-xt) | Video | Stability AI Community License | **bedingt** | Umsatzgrenze; Namensnennung |
| `bark-small` (suno/bark-small) | Stimme | MIT | ja | Namensnennung — **Vorgabe für Deutsch** |
| `bark` (suno/bark) | Stimme | MIT | ja | Namensnennung |
| `kokoro` (hexgrad/Kokoro-82M) | Stimme | Apache-2.0 | ja | Namensnennung — **kann kein Deutsch** (en/es/fr/it/pt/ja/zh) |
| `piper` (rhasspy/piper-voices) | Stimme | Stimmen CC0/CC-BY-4.0, **Laufzeit `piper-tts`: GPL-3.0** | **bedingt** | siehe Abschnitt „Piper und die GPL“ unten |
| `chatterbox` (ResembleAI/chatterbox) | Stimme klonen | MIT | ja | Einwilligung der sprechenden Person; Wasserzeichen nicht entfernen; **eigene Laufzeit nötig** |
| `openvoice-v2` (myshell-ai/OpenVoiceV2) | Stimme klonen | MIT | ja | Einwilligung der sprechenden Person |
| `xtts-v2` (coqui/XTTS-v2) | Stimme klonen | Coqui Public Model License | **nein** | kommerzielle Nutzung ausgeschlossen – gesperrt |
| `f5-tts` (SWivid/F5-TTS) | Stimme klonen | Code MIT, Gewichte CC-BY-NC-4.0 | **nein** | Gewichte auf NC-Datensatz trainiert – gesperrt |
| `realesrgan-x4` (Real-ESRGAN x4) | Hochskalieren | BSD-3-Clause | ja | Lizenztext und Namensnennung beilegen |

Die Tabelle wird zur Laufzeit aus dem Quelltext erzeugt:

```
streamforge models table
```

## Bewusst nicht aufgenommen

| Modell | Grund |
|---|---|
| FLUX.1-dev | Nicht-kommerzielle Lizenz. Nur `FLUX.1-schnell` (Apache-2.0) ist verkäuflich. |
| Stable Diffusion 3.x / 3.5 | Stability AI Community License mit Umsatzgrenze und Registrierung – nicht eindeutig „ja“. |
| CogVideoX-5B | Eigene Lizenz, nicht Apache-2.0 wie die 2B-Fassung. |
| LTX-Video | Eigene „Open Weights“-Lizenz mit Einschränkungen; nicht eindeutig. **Rückfrage nötig, falls gewünscht.** |
| Bark (suno/bark) | MIT, aber Stimmen-Presets mit unklarer Herkunft; Stimmklonen ohne Einwilligungspfad. |
| StyleTTS2-Checkpoints | Trainingsdaten-Beschränkungen bei den veröffentlichten Gewichten. |
| Tortoise-TTS | Modellherkunft/Datensatz nicht sauber dokumentiert. |

## Empfehlung nach Hardware

Die Anwendung nennt das vor dem Download (`app/accel.py`, `CapabilityTier`):

| Stufe | VRAM | Bild | Video | Stimme |
|---|---|---|---|---|
| Nur CPU | – | `sd15`, `ssd-1b`, ≤768 px, wenige Schritte | nicht sinnvoll | `bark-small` (langsam), `piper` (schnell, GPL) |
| Einstieg | 4–6 GB | `ssd-1b`, `sd15` | `animatediff`, sehr kurz | `bark-small` |
| Mittelklasse | 8–11 GB | `sdxl-base` | `wan-t2v-1.3b`, `cogvideox-2b` | alle |
| Oberklasse | 12–23 GB | `sdxl-base`, `flux-schnell` | `wan-t2v-1.3b` in höherer Auflösung | alle |
| Profi | ab 24 GB | alle | alle | alle |

Gemessen auf einer RTX 4070 Ti (12 GB): SDXL 1024×1024, 25 Schritte,
fp16 — rund 40 Sekunden inklusive Modellladen, danach je Bild deutlich
schneller, weil die Pipeline im Speicher bleibt.

## Speicherbedarf: Dateifilter beachten

Ein SDXL-Repo enthält dieselben Gewichte mehrfach (fp32 **und** fp16, dazu
`.bin`, OpenVINO und Einzeldatei-Checkpoints). Ungefiltert landen **46 GB**
auf der Platte, gebraucht werden **6,5 GB**. Die Auswahl in
`models.select_files()` nimmt deshalb nur: die in `model_index.json`
genannten Komponenten, davon die fp16-Variante, keine Duplikate, keine
fremden Laufzeiten.

Bereits zu groß geladene Modelle aufräumen:

```
streamforge models prune sdxl-base --dry-run
streamforge models prune sdxl-base
```

## Piper und die GPL — Entscheidung nötig

Beim Einbau ist aufgefallen, dass die **Stimmen** und die **Laufzeit**
getrennt zu bewerten sind:

* Die Stimmgewichte sind sauber: **Thorsten (deutsch) ist CC0**, HFC ist
  CC-BY-4.0. Kommerziell kein Problem.
* Das Python-Paket **`piper-tts` steht unter GPL-3.0-or-later** (Projekt
  `piper1-gpl`) und bettet **espeak-ng** ein, das ebenfalls GPL-3.0 ist.

Wird `piper-tts` in denselben Prozess importiert, erfasst die GPL nach
verbreiteter Auslegung die gesamte Anwendung. Für eine verkaufte,
proprietäre Anwendung ist das nicht tragbar. Deshalb:

* **Vorgabe ist jetzt `bark-small` (MIT)** — deckt Deutsch ab, keine
  Lizenzfrage, läuft auf GPU wie CPU.
* Der Piper-Zweig ist implementiert, aber **fail-closed gesperrt**: erst
  nach Zustimmung zur Komponente `piper-gpl` (Anwendung → Lizenzen) nutzbar,
  und `piper-tts` wird bewusst **nicht** mitgebündelt.

Drei Wege stehen offen — bitte entscheiden:

| Weg | Aufwand | Lizenzlage |
|---|---|---|
| **A: bei Bark bleiben** (empfohlen) | keiner, läuft | MIT, unbedenklich |
| **B: Piper als eigenes Programm** ausliefern und über die Kommandozeile aufrufen | ein Tag Arbeit (Prozessaufruf statt Import, GPL-Beilagen) | vertretbar: getrennte Programme, GPL nur für Piper — Lizenztext und Quelltextangebot beilegen |
| **C: Piper im selben Prozess** | keiner | **nicht empfohlen**: GPL-3.0 erfasst die gesamte Anwendung |

Gemessen auf diesem Rechner (RTX 4070 Ti, i9-10850K), beide Wege geprüft:

| | Piper (Thorsten) | Bark small |
|---|---|---|
| Ausgabe im Test | 7,1 s Sprache | 6,8 s Sprache |
| Abtastrate | 22,05 kHz | 24 kHz |
| Platzbedarf | 60 MB je Stimme | 1,7 GB |
| Rechenzeit | Sekundenbruchteile, CPU | deutlich länger, GPU sinnvoll |
| Lizenz | Stimme CC0 — **Laufzeit GPL-3.0** | MIT |

## Klonstimmen laufen getrennt

`chatterbox-tts` verlangt torch 2.6, diffusers 0.29 und transformers 5.x.
Zusammen mit Bild und Video installiert, stuft pip torch auf eine Fassung
ohne CUDA-Build herunter und diffusers unter die Version, die Wan 2.1 und
CogVideoX brauchen. Deshalb liegt die Klonstimme in einer eigenen Umgebung
(`.voice-venv`, rund 5,4 GB) und wird als eigener Prozess aufgerufen.

Zwei Stolpersteine, die beim Einbau aufgefallen sind:

* Das Wasserzeichen-Paket `perth` braucht `pkg_resources`. Python-3.13-Venvs
  bringen kein setuptools mit, und setuptools ab Fassung 81 liefert
  `pkg_resources` nicht mehr. Ohne `setuptools<81` scheitert das Laden mit
  `'NoneType' object is not callable`. Das Wasserzeichen wird bewusst nicht
  umgangen – es ist Lizenzauflage.
* `torchaudio.save` schreibt Float-WAV (Formatkennung 3). Das kann die
  Standardbibliothek nicht lesen; die Ausgabe wird deshalb als 16-Bit-PCM
  geschrieben.

## Stimmklonen: zwei getrennte Fragen

1. **Modell-Lizenz** – Chatterbox und OpenVoice v2 sind MIT, also verkäuflich.
2. **Persönlichkeitsrecht** – unabhängig davon braucht jede geklonte Stimme
   die Einwilligung der sprechenden Person. Die Anwendung erzwingt das:
   * Komponente `voice-cloning` muss freigegeben sein (`app/licensing.py`)
   * jedes Profil trägt einen Nachweis mit Name, Zweck, Datum und Prüfsumme
     des Wortlauts (`SpeakerConsent`)
   * ohne gültigen Nachweis wird das Profil nicht angelernt und nicht
     verwendet – Rückfall auf die Standardstimme
   * Löschen des Profils ist der Widerrufsweg

Bei kommerzieller Nutzung fremder Stimmen zusätzlich prüfen: schriftliche
Einwilligung, Zweckbindung, Vergütung, Widerrufsfrist. Das ist eine
Vertragsfrage, keine Softwarefrage.

## Offene Punkte für Rückfrage

* **LTX-Video** aufnehmen? Lizenz ist nicht eindeutig genug für eine
  Vorgabe – bitte entscheiden.
* **Piper-Stimmen**: soll die Auslieferung nur MIT/CC0-Stimmen enthalten
  (empfohlen) oder auch CC-BY mit Namensnennung im Impressum?
* **Stability-Modelle** (`sdxl-turbo`, `svd-xt`): nur sinnvoll, wenn der
  Jahresumsatz unter der Grenze der Community License liegt und die
  Registrierung erfolgt. Sonst weglassen.
