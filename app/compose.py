"""Muxen und Zusammensetzen über ffmpeg.

Lizenz-Hinweis (wichtig, weil die Anwendung verkauft wird):
ffmpeg wird als **LGPL-Build ohne GPL-Bestandteile** ausgeliefert. Ein
GPL-Build (mit libx264, libx265, GPL-Filtern) würde die gesamte Anwendung
unter die GPL zwingen. Deshalb ist die Codec-Vorgabe ``libopenh264``
(Cisco, BSD-2) beziehungsweise ``libvpx-vp9``/``mpeg4`` als Rückfallebene.

Alle Aufrufe sind fail-soft und abbrechbar: fehlt ffmpeg, kommt eine
verständliche Meldung statt eines Stacktrace.
"""

from __future__ import annotations

import logging
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .accel import clean_error
from .jobs import JobContext

log = logging.getLogger(__name__)

# Reihenfolge der Rückfallebenen – bewusst ohne libx264 (GPL).
CODEC_FALLBACKS: dict[str, tuple[str, ...]] = {
    "libopenh264": ("libopenh264", "h264_nvenc", "h264_qsv", "h264_amf", "libvpx-vp9", "mpeg4"),
    "libvpx-vp9": ("libvpx-vp9", "libvpx", "mpeg4"),
    "mpeg4": ("mpeg4",),
}
GPL_CODECS = ("libx264", "libx265", "libxvid")


class FfmpegMissing(RuntimeError):
    """ffmpeg ist nicht vorhanden."""


class FfmpegError(RuntimeError):
    """ffmpeg lief, brach aber mit Fehler ab."""


class FfmpegCancelled(RuntimeError):
    """Lauf wurde abgebrochen."""


@dataclass(frozen=True)
class FfmpegInfo:
    path: Path
    version: str
    encoders: tuple[str, ...] = ()
    gpl_build: bool = False

    def has(self, encoder: str) -> bool:
        return encoder in self.encoders


_info_cache: FfmpegInfo | None = None


def _creation_flags() -> int:
    import os

    return 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW


def probe(refresh: bool = False) -> FfmpegInfo:
    """ffmpeg suchen und Fähigkeiten lesen. Wirft FfmpegMissing."""
    global _info_cache
    if _info_cache is not None and not refresh:
        return _info_cache

    binary = paths.ffmpeg_exe()
    if binary is None:
        raise FfmpegMissing(
            "ffmpeg wurde nicht gefunden. Lege einen LGPL-Build nach "
            f"{paths.tools_dir() / 'ffmpeg'} oder trage ihn in den PATH ein. "
            "Ohne ffmpeg können Videos nicht geschrieben und Ton nicht "
            "gemuxt werden – Bilder und Sprache funktionieren trotzdem."
        )

    version = "unbekannt"
    encoders: list[str] = []
    gpl = False
    try:
        result = subprocess.run(
            [str(binary), "-hide_banner", "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=_creation_flags(),
        )
        first = (result.stdout or "").splitlines()
        if first:
            version = first[0].strip()
        gpl = "--enable-gpl" in (result.stdout or "")
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("ffmpeg -version fehlgeschlagen: %s", exc)

    try:
        result = subprocess.run(
            [str(binary), "-hide_banner", "-encoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=_creation_flags(),
        )
        for line in (result.stdout or "").splitlines():
            match = re.match(r"^\s*[A-Z.]{6}\s+(\S+)", line)
            if match:
                encoders.append(match.group(1))
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("ffmpeg -encoders fehlgeschlagen: %s", exc)

    _info_cache = FfmpegInfo(binary, version, tuple(encoders), gpl)
    if gpl:
        log.warning(
            "Der gefundene ffmpeg-Build ist mit --enable-gpl gebaut. Für die "
            "kommerzielle Auslieferung einen LGPL-Build verwenden."
        )
    return _info_cache


def available() -> bool:
    try:
        probe()
        return True
    except FfmpegMissing:
        return False


def pick_codec(wanted: str, info: FfmpegInfo | None = None) -> tuple[str, str]:
    """Codec wählen. Rückgabe (codec, Hinweis)."""
    info = info or probe()
    if wanted in GPL_CODECS:
        return (
            "libopenh264" if info.has("libopenh264") else "mpeg4",
            f"'{wanted}' ist ein GPL-Encoder und wird für die Auslieferung nicht "
            "genutzt – es wird ein LGPL-taugliches Ziel gewählt.",
        )
    for candidate in CODEC_FALLBACKS.get(wanted, (wanted, "mpeg4")):
        if info.has(candidate):
            note = "" if candidate == wanted else f"'{wanted}' fehlt, '{candidate}' wird genutzt."
            return candidate, note
    return "mpeg4", f"Weder '{wanted}' noch Alternativen vorhanden – 'mpeg4' wird versucht."


def _parse_progress(line: str) -> tuple[str, str] | None:
    if "=" not in line:
        return None
    key, _, value = line.partition("=")
    return key.strip(), value.strip()


def run_ffmpeg(
    args: Sequence[str],
    total_seconds: float = 0.0,
    context: JobContext | None = None,
    label: str = "ffmpeg",
    timeout: float = 3600.0,
) -> str:
    """ffmpeg starten, Fortschritt melden, auf Abbruch reagieren.

    ``args`` ohne Programmnamen. Rückgabe: gesammelte Ausgabe (für Diagnose).
    """
    info = probe()
    command = [
        str(info.path),
        "-hide_banner",
        "-nostdin",
        "-y",
        "-progress",
        "pipe:1",
        "-nostats",
        *args,
    ]
    log.debug("%s: %s", label, " ".join(command))

    started = time.time()
    tail: list[str] = []
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=_creation_flags(),
        )
    except OSError as exc:
        raise FfmpegError(f"{label} konnte nicht gestartet werden: {clean_error(exc)}") from exc

    try:
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.strip()
            if not line:
                continue
            tail.append(line)
            if len(tail) > 60:  # nur das Ende behalten, sonst frisst es Speicher
                tail.pop(0)

            parsed = _parse_progress(line)
            if parsed and context is not None:
                key, value = parsed
                if key == "out_time_ms" and total_seconds > 0:
                    try:
                        done = int(value) / 1_000_000.0
                        context.progress(
                            min(1.0, done / total_seconds),
                            f"{label}: {done:.1f}s / {total_seconds:.1f}s",
                        )
                    except ValueError:
                        pass
                elif key == "frame" and total_seconds <= 0:
                    context.status(f"{label}: Bild {value}")

            if context is not None and context.should_stop():
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise FfmpegCancelled(f"{label} abgebrochen")

            if time.time() - started > timeout:
                process.kill()
                raise FfmpegError(f"{label} hat das Zeitlimit von {timeout:g}s überschritten.")
    finally:
        with_output = "\n".join(tail)
        if process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()

    if process.returncode not in (0, None):
        raise FfmpegError(
            f"{label} endete mit Code {process.returncode}: {clean_error(with_output)}"
        )
    return with_output


def media_duration(path: Path) -> float:
    """Dauer einer Mediendatei in Sekunden. 0.0, wenn nicht ermittelbar."""
    info = probe()
    probe_binary = info.path.with_name(info.path.name.replace("ffmpeg", "ffprobe"))
    if probe_binary.is_file():
        try:
            result = subprocess.run(
                [
                    str(probe_binary),
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
                creationflags=_creation_flags(),
            )
            return float((result.stdout or "0").strip() or 0.0)
        except (OSError, ValueError, subprocess.SubprocessError):
            return 0.0
    # Ohne ffprobe: aus der ffmpeg-Ausgabe lesen
    try:
        result = subprocess.run(
            [str(info.path), "-hide_banner", "-i", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=_creation_flags(),
        )
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr or "")
        if match:
            hours, minutes, seconds = match.groups()
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (OSError, subprocess.SubprocessError):
        pass
    return 0.0


def frames_to_video(
    frames_dir: Path,
    pattern: str,
    fps: int,
    output: Path,
    crf: int = 20,
    codec: str = "libopenh264",
    audio: Path | None = None,
    context: JobContext | None = None,
) -> Path:
    """Einzelbilder zu einem Video zusammensetzen, optional mit Ton."""
    info = probe()
    # Den Wunschcodec am Behälter ausrichten, BEVOR gewählt wird.
    #
    # Ein .webm mit H.264 oder ein .mov mit VP9 ist kein gültiger
    # Behälter -- ffmpeg bricht dann ab. Vorher ging der Wunsch
    # ungeprüft durch, und die Behälterwahl in der Oberfläche führte zu
    # einem Fehlschlag statt zu einem Video.
    behaelter = output.suffix.lower()
    if behaelter == ".webm" and codec not in ("libvpx-vp9", "libvpx"):
        codec = "libvpx-vp9"
    elif behaelter == ".mov" and codec.startswith("libvpx"):
        codec = "libopenh264"

    chosen, note = pick_codec(codec, info)
    if note and context is not None:
        context.log(note)
    paths.ensure_dir(output.parent)

    args: list[str] = ["-framerate", str(max(1, fps)), "-i", str(frames_dir / pattern)]
    if audio is not None and Path(audio).is_file():
        args += ["-i", str(audio)]

    args += ["-c:v", chosen, "-pix_fmt", "yuv420p"]
    if chosen in ("libopenh264",):
        # libopenh264 kennt kein CRF – Bitrate aus dem CRF-Wunsch ableiten.
        bitrate = max(800, int(9000 - crf * 150))
        args += ["-b:v", f"{bitrate}k"]
    elif chosen.startswith("libvpx"):
        args += ["-crf", str(crf), "-b:v", "0"]
    elif chosen in ("h264_nvenc", "h264_qsv", "h264_amf"):
        args += ["-cq", str(crf)]
    else:
        args += ["-q:v", str(max(1, min(31, crf // 2)))]

    if audio is not None and Path(audio).is_file():
        args += ["-c:a", "aac", "-b:a", "192k", "-shortest"]

    args.append(str(output))
    total = 0.0  # Dauer ist über die Bildzahl nicht direkt bekannt
    frame_count = len(list(frames_dir.glob(pattern.replace("%05d", "*"))))
    if frame_count and fps:
        total = frame_count / float(fps)
    run_ffmpeg(args, total_seconds=total, context=context, label="Video schreiben")
    return output


def mux(
    video: Path,
    audio: Path,
    output: Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    normalize: bool = True,
    loop_audio: bool = False,
    context: JobContext | None = None,
) -> Path:
    """Video und Ton zusammenlegen. Video wird nicht neu kodiert (copy)."""
    if not Path(video).is_file():
        raise FileNotFoundError(f"Videodatei fehlt: {video}")
    if not Path(audio).is_file():
        raise FileNotFoundError(f"Tondatei fehlt: {audio}")
    paths.ensure_dir(output.parent)

    args: list[str] = []
    if loop_audio:
        args += ["-stream_loop", "-1"]
    args += [
        "-i",
        str(video),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        audio_codec,
        "-b:a",
        audio_bitrate,
    ]
    if normalize:
        # loudnorm ist LGPL-tauglich und liefert gleichmäßige Lautheit.
        args += ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    args += ["-shortest", str(output)]

    total = media_duration(Path(video))
    run_ffmpeg(args, total_seconds=total, context=context, label="Ton muxen")
    return output


def concat_audio(parts: Sequence[Path], output: Path, context: JobContext | None = None) -> Path:
    """Mehrere Tonstücke aneinanderhängen (Satzweise Sprachausgabe)."""
    files = [Path(p) for p in parts if Path(p).is_file()]
    if not files:
        raise FileNotFoundError("Keine Tondateien zum Zusammenhängen.")
    if len(files) == 1:
        paths.ensure_dir(output.parent)
        output.write_bytes(files[0].read_bytes())
        return output

    listing = output.with_suffix(".txt")
    paths.ensure_dir(output.parent)
    listing.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in files) + "\n", encoding="utf-8"
    )
    try:
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(output)],
            context=context,
            label="Ton zusammenhängen",
        )
    finally:
        listing.unlink(missing_ok=True)
    return output


def describe() -> str:
    """Zustandsbericht für die GUI-Hardwareseite und `--info`."""
    try:
        info = probe()
    except FfmpegMissing as exc:
        return str(exc)
    codec, note = pick_codec("libopenh264", info)
    lines = [
        f"ffmpeg:   {info.path}",
        f"Version:  {info.version}",
        f"Encoder:  {len(info.encoders)} verfügbar, gewählt: {codec}",
    ]
    if note:
        lines.append(f"Hinweis:  {note}")
    if info.gpl_build:
        lines.append("WARNUNG:  GPL-Build erkannt – für den Verkauf einen LGPL-Build ausliefern.")
    return "\n".join(lines)
