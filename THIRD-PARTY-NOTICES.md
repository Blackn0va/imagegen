# Hinweise zu Bestandteilen Dritter

StreamForge Studio enthält und verwendet Software und Modelle Dritter. Diese
Datei liegt jeder Auslieferung bei (`build-windows.ps1` kopiert sie neben die
.exe) und wird in der Anwendung unter **Lizenzen** verlinkt.

Stand: 2026-07-29 · Anwendungsversion 0.1.0

Rechtlicher Rahmen: StreamForge Studio selbst ist proprietär. Die unten
genannten Bestandteile behalten ihre jeweilige Lizenz. Wo eine Lizenz die
Weitergabe des Lizenztexts verlangt, ist der Text über die angegebene Adresse
erreichbar und Teil dieser Auslieferung.

---

## 1. Python-Laufzeit und Standardbibliothek

| Bestandteil | Lizenz | Adresse |
|---|---|---|
| CPython | PSF License Agreement 2.0 | https://docs.python.org/3/license.html |
| Tcl/Tk (über tkinter, GUI) | Tcl/Tk License (BSD-artig) | https://www.tcl.tk/software/tcltk/license.html |

Die Oberfläche verwendet bewusst tkinter/Tcl-Tk statt Qt. Qt/PySide würde
LGPL-Pflichten oder eine kommerzielle Lizenz auslösen.

## 2. Python-Pakete

| Paket | Lizenz | Adresse |
|---|---|---|
| huggingface_hub | Apache-2.0 | https://github.com/huggingface/huggingface_hub |
| diffusers | Apache-2.0 | https://github.com/huggingface/diffusers |
| transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| accelerate | Apache-2.0 | https://github.com/huggingface/accelerate |
| safetensors | Apache-2.0 | https://github.com/huggingface/safetensors |
| tokenizers | Apache-2.0 | https://github.com/huggingface/tokenizers |
| sentencepiece | Apache-2.0 | https://github.com/google/sentencepiece |
| PyTorch (torch, torchvision, torchaudio) | BSD-3-Clause | https://github.com/pytorch/pytorch/blob/main/LICENSE |
| ONNX Runtime / onnxruntime-directml | MIT | https://github.com/microsoft/onnxruntime |
| NumPy | BSD-3-Clause | https://numpy.org/doc/stable/license.html |
| Pillow | MIT-CMU (HPND) | https://github.com/python-pillow/Pillow/blob/main/LICENSE |
| SoundFile | BSD-3-Clause | https://github.com/bastibe/python-soundfile |
| libsndfile (durch SoundFile eingebunden) | LGPL-2.1-or-later | http://www.mega-nerd.com/libsndfile/ |
| truststore | MIT | https://github.com/sethmlarson/truststore |
| certifi | MPL-2.0 (Zertifikatsdaten: MPL-2.0) | https://github.com/certifi/python-certifi |
| packaging | Apache-2.0 / BSD-2-Clause | https://github.com/pypa/packaging |
| PyInstaller (nur Build) | GPL-2.0-or-later **mit Bootloader-Ausnahme** | https://github.com/pyinstaller/pyinstaller/blob/develop/COPYING.txt |
| llama-cpp-python (Chat) | MIT | https://github.com/abetlen/llama-cpp-python |
| faster-whisper (Spracherkennung) | MIT | https://github.com/SYSTRAN/faster-whisper |
| CTranslate2 | MIT | https://github.com/OpenNMT/CTranslate2 |
| sounddevice | MIT | https://github.com/spatialaudio/python-sounddevice |
| PortAudio (durch sounddevice eingebunden) | MIT | https://www.portaudio.com/license.html |
| discord.py (Discord-Bot) | MIT | https://github.com/Rapptz/discord.py |
| discord-ext-voice-recv | MIT | https://github.com/imayhaveborkedit/discord-ext-voice-recv |
| PyNaCl | Apache-2.0 | https://github.com/pyca/pynacl |
| libsodium (durch PyNaCl eingebunden) | ISC | https://github.com/jedisct1/libsodium |
| davey (DAVE-Verschlüsselung) | MIT | https://github.com/Snazzah/davey |
| audioop-lts | PSF-2.0 | https://github.com/AbstractUmbra/audioop |
| libopus (durch discord.py mitgeliefert) | BSD-3-Clause | https://opus-codec.org/license/ |
| aiohttp | Apache-2.0 | https://github.com/aio-libs/aiohttp |

Zu PyInstaller: die Ausnahme erlaubt ausdrücklich, damit erzeugte Bundles
proprietär auszuliefern. PyInstaller selbst wird nicht mitgeliefert.

Zu libsndfile (LGPL): dynamisch gebunden, unverändert. Auf Anfrage wird die
Bibliothek in der verwendeten Fassung samt Quelltext bereitgestellt; sie kann
durch eine eigene Fassung ersetzt werden.

Zu libopus (BSD-3-Clause): die DLL liegt unverändert so bei, wie discord.py
sie in seinem Paket ausliefert (`discord/bin/libopus-0.x64.dll`), samt der
Datei `COPYING`. Sie wird nur geladen, wenn der Discord-Weg genutzt wird.

Zu audioop-lts (PSF-2.0): führt das Modul `audioop` fort, das mit Python 3.13
aus der Standardbibliothek entfernt wurde. Es rechnet die Tonformate zwischen
Discord (48 kHz Stereo) und der Spracherkennung (16 kHz Mono) um.

## 3. NVIDIA-Laufzeit (nur im GPU-Build)

Wird nur mitgeliefert, wenn `build-windows.ps1 -WithCuda $true` läuft. Die
Bibliotheken liegen dann im Unterordner `cuda\`.

| Bestandteil | Lizenz | Adresse |
|---|---|---|
| CUDA Runtime, cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER, NVTX | NVIDIA Software License Agreement / CUDA EULA | https://docs.nvidia.com/cuda/eula/index.html |
| cuDNN | NVIDIA cuDNN Software License Agreement | https://docs.nvidia.com/deeplearning/cudnn/sla/index.html |

Auflagen, die diese Auslieferung erfüllt:

* Weitergabe ausschließlich eingebettet in diese Anwendung, nicht als
  eigenständiges SDK.
* Der Lizenztext ist über die oben genannten Adressen erreichbar und in
  dieser Datei benannt.
* Der Endkunde stimmt **vor** dem ersten Laden ausdrücklich zu
  (Anwendung → Lizenzen). Ohne Zustimmung werden die Bibliotheken nicht
  geladen; die Anwendung rechnet dann auf der CPU und sagt das im Klartext.

## 3a. Bewusst NICHT mitgelieferte Pakete

| Paket | Lizenz | Grund |
|---|---|---|
| piper-tts (Projekt `piper1-gpl`, enthält espeak-ng) | GPL-3.0-or-later | In denselben Prozess geladen würde die GPL die gesamte Anwendung erfassen. Der Programmzweig existiert, ist aber gesperrt (Zustimmung zur Komponente `piper-gpl` nötig) und das Paket wird nicht gebündelt. Zulässiger Weg wäre die Auslieferung als eigenständiges Programm samt Lizenztext und Quelltextangebot. |
| Coqui XTTS-v2, F5-TTS | CPML bzw. CC-BY-NC | nicht-kommerziell, siehe MODELS.md |

## 4. ffmpeg (mitgeliefert)

ffmpeg liegt der Auslieferung als **eigenständiges Programm** unter
`tools\ffmpeg\bin\` bei (`ffmpeg.exe`, `ffprobe.exe`). Die genaue Fassung,
die Bezugsquelle und die SHA-256-Prüfsumme stehen dort in `HERKUNFT.txt`,
der Lizenztext in `LICENSE.txt`.

| Bestandteil | Lizenz | Adresse |
|---|---|---|
| ffmpeg (Build ohne GPL-Bestandteile, mit `--enable-version3`) | LGPL-3.0-or-later | https://ffmpeg.org/legal.html |
| openh264 (H.264-Encoder, Vorgabe für Video) | BSD-2-Clause | https://github.com/cisco/openh264/blob/master/LICENSE |
| libvpx (VP8/VP9) | BSD-3-Clause | https://chromium.googlesource.com/webm/libvpx/+/master/LICENSE |
| libaom, dav1d, SVT-AV1 (AV1) | BSD-2/BSD-3 | https://aomedia.googlesource.com/aom/ |
| libopus, libvorbis, libmp3lame | BSD-3 / BSD-3 / LGPL-2.0-or-later | https://opus-codec.org/ |

Bezug des mitgelieferten Builds: https://github.com/BtbN/FFmpeg-Builds
(Variante `win64-lgpl`).

**Wie die Lizenz eingehalten wird:**

* Es wird ausschließlich ein Build **ohne** `--enable-gpl` und **ohne**
  `--enable-nonfree` ausgeliefert. `libx264`, `libx265` und `libxvid` sind
  im mitgelieferten Build ausdrücklich abgeschaltet. Das Build-Skript prüft
  das am Binary und bricht sonst ab; zusätzlich warnt die Anwendung zur
  Laufzeit (`app/compose.py`, `probe()`), und die Codec-Vorgabe meidet
  GPL-Encoder.
* ffmpeg wird **nicht gelinkt**, sondern als eigener Prozess über die
  Kommandozeile aufgerufen. Die Anwendung ist damit kein abgeleitetes Werk.
* Der Lizenztext (LGPL-3.0) liegt unverändert bei.
* Der Quelltext der ausgelieferten Fassung wird auf Anfrage in
  maschinenlesbarer Form bereitgestellt (Angabe der Fassung aus
  `tools\ffmpeg\HERKUNFT.txt`).
* Die Binärdateien werden unverändert weitergegeben; ein Austausch durch
  eine eigene ffmpeg-Fassung ist möglich, indem die Dateien in
  `tools\ffmpeg\bin\` ersetzt werden.

## 5. Modelle

Modelle werden nicht in die .exe kompiliert, sondern liegen als Dateien im
Modell-Cache. Die vollständige Tabelle mit Lizenz, kommerzieller Erlaubnis und
Auflagen steht in **MODELS.md** und wird in der Anwendung unter *Modelle*
angezeigt.

Namensnennung für mitgelieferte Modelle:

* Stable Diffusion XL 1.0 Base — Stability AI, CreativeML Open RAIL++-M
* Segmind SSD-1B — Segmind, Apache-2.0
* Stable Diffusion 1.5 — Runway/Stability AI, CreativeML Open RAIL-M
* FLUX.1 [schnell] — Black Forest Labs, Apache-2.0
* Wan 2.1 T2V 1.3B — Alibaba Wan-AI, Apache-2.0
* CogVideoX-2b — Zhipu AI / THUDM, Apache-2.0
* AnimateDiff Motion Adapter — Yuwei Guo u. a., Apache-2.0
* Kokoro-82M — hexgrad, Apache-2.0
* Bark / Bark small — Suno, MIT
* Piper-Stimmen — Thorsten Müller (CC0-1.0), HFC (CC-BY-4.0); die Laufzeit
  `piper-tts` steht unter GPL-3.0 und wird nicht mitgeliefert
* Chatterbox — Resemble AI, MIT
* OpenVoice v2 — MyShell AI, MIT
* Real-ESRGAN — Xintao Wang u. a., BSD-3-Clause

Die RAIL-Lizenzen (Open RAIL-M / RAIL++-M) enthalten Nutzungsbeschränkungen
(Anhang A). Diese Beschränkungen sind Teil der Endkunden-Bedingungen und
müssen mit weitergegeben werden.

## 6. Erzeugte Inhalte und Stimmen

* Für erzeugte Bilder, Videos und Sprachaufnahmen gilt die Lizenz des
  jeweils genutzten Modells samt Nutzungsbeschränkungen.
* Stimmklonen ist nur mit Einwilligung der sprechenden Person zulässig. Die
  Anwendung verlangt und speichert dazu einen Nachweis und verweigert ohne
  diesen die Nutzung des Profils.
* Chatterbox versieht erzeugte Sprache mit einem Wasserzeichen des
  Herstellers. Dieses darf nicht entfernt werden.

## 7. Bezug von Quelltexten

Für die unter LGPL stehenden Bestandteile (libsndfile, ffmpeg) wird der
Quelltext der ausgelieferten Fassung auf Anfrage in maschinenlesbarer Form
bereitgestellt. Anfrage über die im Impressum genannte Adresse, unter Angabe
der Anwendungsversion aus *Hardware → Bericht*.
