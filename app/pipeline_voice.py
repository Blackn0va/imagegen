"""Text -> Sprache, inklusive angelernter Stimmprofile.

Basis: Schnittstelle plus Attrappe. Die Attrappe schreibt eine echte
WAV-Datei (Formant-artige Töne, Länge aus dem Text abgeleitet) – damit sind
Vertonung, Muxen und Warteschlange vollständig prüfbar.

Wichtig für den Verkauf: geklonte Stimmen laufen nur mit dokumentierter
Einwilligung der sprechenden Person. Die Prüfung sitzt hier und ist
fail-closed – ohne Nachweis wird auf die Standardstimme zurückgefallen.
"""

from __future__ import annotations

import array
import contextlib
import logging
import math
import struct
import time
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

from . import accel, models, paths, voice_profiles
from .accel import BackendPlan, clean_error
from .config import AppConfig
from .jobs import JobContext

log = logging.getLogger(__name__)

# Grobe Sprechgeschwindigkeit für die Längenschätzung der Attrappe.
CHARS_PER_SECOND = 14.0


@dataclass(frozen=True)
class VoiceRequest:
    text: str
    speaker: str = "default"
    profile_slug: str = ""  # gesetzt = angelernte Stimme verwenden
    language: str = "de"
    speed: float = 1.0
    pitch: float = 0.0
    volume_db: float = 0.0
    sample_rate: int = 24000
    output_dir: Path | None = None
    file_format: str = "wav"
    split_sentences: bool = True
    name_hint: str = ""

    @staticmethod
    def from_config(config: AppConfig, text: str, **overrides: Any) -> "VoiceRequest":
        request = VoiceRequest(
            text=text,
            speaker=config.voice_speaker,
            profile_slug=config.voice_profile if config.voice_cloning_enabled else "",
            language=config.language,
            speed=config.voice_speed,
            pitch=config.voice_pitch,
            volume_db=config.voice_volume_db,
            sample_rate=config.voice_sample_rate,
            output_dir=config.resolved_output_dir() / "audio",
            file_format=config.audio_format,
            split_sentences=config.voice_split_sentences,
        )
        return replace(request, **{k: v for k, v in overrides.items() if hasattr(request, k)})

    def estimated_seconds(self) -> float:
        chars = max(1, len(self.text))
        return max(0.6, chars / (CHARS_PER_SECOND * max(0.4, self.speed)))


@dataclass(frozen=True)
class VoiceResult:
    audio: Path
    seconds: float
    sample_rate: int
    backend: str
    model_key: str
    profile_slug: str
    elapsed_s: float
    dummy: bool = False
    notes: tuple[str, ...] = ()


class VoicePipeline(ABC):
    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        self.config = config
        self.plan = plan
        self.model = models.resolve(config.voice_model) if config.voice_model else None
        self._loaded = False
        # Von der Fabrik gesetzt, z. B. "Klonstimme nicht verfügbar".
        self.extra_notes: tuple[str, ...] = ()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, context: JobContext) -> None: ...

    @abstractmethod
    def synthesize(self, request: VoiceRequest, context: JobContext) -> VoiceResult: ...

    def unload(self) -> None:
        self._loaded = False

    def describe(self) -> str:
        model = self.model.repo_id if self.model else "kein Modell gewählt"
        return f"{type(self).__name__}: Modell={model}, Backend={self.plan.label}"


# ---------------------------------------------------------------------------
# WAV schreiben (stdlib)
# ---------------------------------------------------------------------------
def write_wav(path: Path, samples: Sequence[int], sample_rate: int) -> Path:
    paths.ensure_dir(path.parent)
    data = array.array("h", samples)
    with contextlib.closing(wave.open(str(path), "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(data.tobytes())
    return path


def _gain(volume_db: float) -> float:
    return 10.0 ** (volume_db / 20.0)


def render_placeholder_speech(
    text: str,
    seconds: float,
    sample_rate: int,
    pitch: float,
    volume_db: float,
    context: JobContext | None = None,
) -> list[int]:
    """Sprachähnliche Tonfolge: Grundton + zwei Formanten + Silbentakt."""
    total = int(seconds * sample_rate)
    base = 130.0 * (2.0 ** (pitch / 12.0))  # Halbtöne
    gain = _gain(volume_db)
    syllable_hz = 4.0  # Silben pro Sekunde
    samples: list[int] = []
    report_every = max(1, total // 20)
    seed = sum(text.encode("utf-8")) % 97

    for index in range(total):
        if context is not None and index % report_every == 0:
            context.raise_if_cancelled()
            context.progress_steps(index, total, "Sprachausgabe")
        t = index / sample_rate
        # Silbenhüllkurve, damit es nicht wie ein Dauerton klingt
        envelope = 0.35 + 0.65 * max(0.0, math.sin(math.pi * syllable_hz * t) ** 2)
        drift = 1.0 + 0.02 * math.sin(2 * math.pi * 0.7 * t + seed)
        value = (
            0.55 * math.sin(2 * math.pi * base * drift * t)
            + 0.25 * math.sin(2 * math.pi * base * 2.4 * t)
            + 0.12 * math.sin(2 * math.pi * base * 3.8 * t)
        )
        # sanftes Ein- und Ausblenden gegen Knacken
        fade = min(1.0, index / (0.02 * sample_rate + 1), (total - index) / (0.02 * sample_rate + 1))
        samples.append(int(max(-1.0, min(1.0, value * envelope * fade * gain)) * 26000))
    if context is not None:
        context.progress_steps(total, total, "Sprachausgabe")
    return samples


def output_path(request: VoiceRequest, suffix: str | None = None) -> Path:
    directory = request.output_dir or (paths.outputs_dir() / "audio")
    paths.ensure_dir(directory)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    hint = request.name_hint or request.text
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in hint.lower())
    slug = "-".join(part for part in slug.split("-") if part)[:40] or "sprache"
    return directory / f"{stamp}_{slug}.{suffix or 'wav'}"


# ---------------------------------------------------------------------------
# Stimmprofil auflösen (fail-closed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProfileDecision:
    slug: str
    allowed: bool
    reason: str
    profile: voice_profiles.VoiceProfile | None = None


def resolve_profile(request: VoiceRequest, config: AppConfig) -> ProfileDecision:
    """Angelernte Stimme prüfen. Ohne Einwilligung: Standardstimme."""
    if not request.profile_slug:
        return ProfileDecision("", False, "Standardstimme (kein Profil gewählt).")
    if not config.voice_cloning_enabled:
        return ProfileDecision(
            request.profile_slug, False,
            "Stimmklonen ist in den Einstellungen ausgeschaltet – Standardstimme wird genutzt.",
        )
    profile = voice_profiles.load_profile(request.profile_slug)
    if profile is None:
        return ProfileDecision(
            request.profile_slug, False,
            f"Stimmprofil '{request.profile_slug}' nicht gefunden – Standardstimme wird genutzt.",
        )
    usable, reason = profile.usable_for_synthesis()
    if not usable:
        return ProfileDecision(profile.slug, False, reason + " Standardstimme wird genutzt.", profile)
    return ProfileDecision(profile.slug, True, reason, profile)


# ---------------------------------------------------------------------------
# Echte Umsetzungen
# ---------------------------------------------------------------------------
_voice_cache: dict[tuple, Any] = {}


def _split_sentences(text: str, limit: int = 220) -> list[str]:
    """Text in Sätze zerlegen – lange Absätze werden sonst abgeschnitten."""
    import re

    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)
    return chunks or [text.strip() or "."]


class PiperVoicePipeline(VoicePipeline):
    """Piper: ONNX-Sprachsynthese, läuft schnell auf der CPU.

    Deutsche Vorgabe (Thorsten, CC0). Bringt espeak-ng als Lautschrift mit,
    braucht also keine externe Installation.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        super().__init__(config, plan)
        self._voice = None
        self._voice_name = ""

    def _find_voice_file(self, root: Path, wanted: str) -> Path | None:
        candidates = sorted(root.rglob("*.onnx"))
        if not candidates:
            return None
        if wanted:
            for path in candidates:
                if path.stem == wanted:
                    return path
            for path in candidates:
                if wanted.lower() in path.stem.lower():
                    return path
        language = (self.config.language or "de").lower()
        for path in candidates:
            if path.stem.lower().startswith(language):
                return path
        return candidates[0]

    def load(self, context: JobContext) -> None:
        key = ("piper", self.config.voice_speaker)
        cached = _voice_cache.get(key)
        if cached is not None:
            self._voice, self._voice_name = cached
            self._loaded = True
            return

        assert self.model is not None
        try:
            root = models.ensure_local(
                self.model,
                allow_download=self.config.allow_model_download,
                on_progress=lambda done, total: context.progress(
                    (done / total) if total else 0.0,
                    f"Download {done / (1024 ** 2):.0f} MB von {total / (1024 ** 2):.0f} MB",
                ),
                on_status=context.status,
                should_stop=context.should_stop,
                allow_conditional=True,
                offline=self.config.offline_mode,
                workers_hint=self.config.download_workers,
            )
        except models.DownloadCancelled as exc:
            from .jobs import JobCancelled

            raise JobCancelled(str(exc)) from exc

        voice_file = self._find_voice_file(Path(root), self.config.voice_speaker)
        if voice_file is None:
            raise RuntimeError(
                f"Keine Piper-Stimme in {root} gefunden. Modell erneut laden "
                "(Modelle → Piper → Herunterladen)."
            )
        context.status(f"Lade Stimme {voice_file.stem} …")
        from piper import PiperVoice

        # use_cuda absichtlich aus: Piper ist auf der CPU schnell genug und
        # belegt so keinen Grafikspeicher, den Bild/Video brauchen.
        self._voice = PiperVoice.load(voice_file, use_cuda=False)
        self._voice_name = voice_file.stem
        _voice_cache[key] = (self._voice, self._voice_name)
        self._loaded = True

    def synthesize(self, request: VoiceRequest, context: JobContext) -> VoiceResult:
        if not self._loaded:
            self.load(context)
        started = time.time()
        notes: list[str] = []

        decision = resolve_profile(request, self.config)
        if decision.slug:
            notes.append(decision.reason)
        if decision.allowed:
            notes.append(
                "Piper nutzt feste Stimmen – für ein angelerntes Profil auf ein "
                "Klon-Modell umstellen (Einstellungen → Stimme)."
            )
        if request.pitch:
            notes.append("Piper kennt keine Tonhöhenverschiebung – Wert ignoriert.")

        from piper import SynthesisConfig

        syn = SynthesisConfig(
            length_scale=1.0 / max(0.4, request.speed),  # größer = langsamer
            volume=min(4.0, 10.0 ** (request.volume_db / 20.0)),
            normalize_audio=True,
        )

        chunks = _split_sentences(request.text) if request.split_sentences else [request.text]
        samples = bytearray()
        rate = request.sample_rate
        for index, sentence in enumerate(chunks):
            context.raise_if_cancelled()
            context.progress_steps(index, len(chunks), f"Satz {index + 1}/{len(chunks)}")
            for audio in self._voice.synthesize(sentence, syn_config=syn):
                context.raise_if_cancelled()
                samples.extend(audio.audio_int16_bytes)
                rate = audio.sample_rate
            # kurze Pause zwischen den Sätzen, sonst klingt es gehetzt
            if index < len(chunks) - 1:
                samples.extend(b"\x00\x00" * int(rate * 0.18))
        context.progress_steps(len(chunks), len(chunks), "fertig")

        target = output_path(request, suffix="wav")
        _write_wav_bytes(target, bytes(samples), rate)
        context.log(f"geschrieben: {target}")

        return VoiceResult(
            audio=target,
            seconds=len(samples) / 2 / max(1, rate),
            sample_rate=rate,
            backend="cpu",
            model_key=self.model.key if self.model else "piper",
            profile_slug="",
            elapsed_s=time.time() - started,
            dummy=False,
            notes=tuple(list(self.extra_notes) + notes + [f"Stimme: {self._voice_name}"]),
        )

    def unload(self) -> None:
        self._voice = None
        self._loaded = False
        _voice_cache.clear()


class BarkVoicePipeline(VoicePipeline):
    """Bark über transformers – mehrsprachig, auch Deutsch.

    Deutlich langsamer als Piper, dafür ausdrucksstärker und mit
    Sprecher-Vorgaben (``v2/de_speaker_3``).
    """

    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        super().__init__(config, plan)
        self._model = None
        self._processor = None
        # Gerät, auf das die Eingaben müssen. Bei aktiver Auslagerung ist das
        # NICHT model.device – accelerate hält die Gewichte dann im RAM.
        self._exec_device = "cpu"

    def load(self, context: JobContext) -> None:
        key = ("bark", self.model.repo_id if self.model else "", self.plan.backend)
        cached = _voice_cache.get(key)
        if cached is not None:
            self._processor, self._model, self._exec_device = cached
            self._loaded = True
            return

        assert self.model is not None
        try:
            path = models.ensure_local(
                self.model,
                allow_download=self.config.allow_model_download,
                on_status=context.status,
                should_stop=context.should_stop,
                allow_conditional=True,
                offline=self.config.offline_mode,
                workers_hint=self.config.download_workers,
            )
        except models.DownloadCancelled as exc:
            from .jobs import JobCancelled

            raise JobCancelled(str(exc)) from exc

        context.status(f"Lade {self.model.title} …")
        import torch
        from transformers import AutoProcessor, BarkModel

        dtype = accel.torch_dtype(self.plan) if self.plan.backend == accel.Backend.CUDA else None
        self._processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self._model = BarkModel.from_pretrained(
            path, local_files_only=True, **({"torch_dtype": dtype} if dtype else {})
        )
        if self.plan.backend == accel.Backend.CUDA:
            index = self.plan.device_index
            self._exec_device = f"cuda:{index}"
            # Entweder ganz auf die GPU ODER Auslagerung – niemals beides.
            # enable_cpu_offload() platziert die Teilmodelle selbst; ein
            # vorheriges .to(cuda) führt zu "tensors on different devices".
            if self.config.cpu_offload:
                try:
                    self._model.enable_cpu_offload(gpu_id=index)
                except Exception as exc:  # noqa: BLE001 – dann eben ganz auf die GPU
                    log.debug("Bark-Auslagerung nicht möglich: %s", exc)
                    self._model = self._model.to(self._exec_device)
            else:
                self._model = self._model.to(self._exec_device)
        else:
            self._exec_device = "cpu"
            self._model = self._model.to("cpu")

        self._model.eval()
        _voice_cache[key] = (self._processor, self._model, self._exec_device)
        self._loaded = True

    def synthesize(self, request: VoiceRequest, context: JobContext) -> VoiceResult:
        if not self._loaded:
            self.load(context)
        import numpy as np
        import torch

        started = time.time()
        notes: list[str] = []
        decision = resolve_profile(request, self.config)
        if decision.slug:
            notes.append(decision.reason)

        preset = request.speaker if request.speaker.startswith("v2/") else None
        if preset is None:
            language = (request.language or "de").lower()[:2]
            preset = f"v2/{language}_speaker_3"
            notes.append(f"Sprecher-Vorgabe: {preset}")

        chunks = _split_sentences(request.text, limit=180)
        pieces: list[Any] = []
        rate = int(self._model.generation_config.sample_rate)

        for index, sentence in enumerate(chunks):
            context.raise_if_cancelled()
            context.progress_steps(index, len(chunks), f"Satz {index + 1}/{len(chunks)}")
            inputs = self._processor(sentence, voice_preset=preset)
            inputs = {k: v.to(self._exec_device) if hasattr(v, "to") else v
                      for k, v in inputs.items()}
            with torch.inference_mode():
                audio = self._model.generate(**inputs, do_sample=True)
            piece = audio.detach().cpu().float().numpy().squeeze()
            pieces.append(piece)
            pieces.append(np.zeros(int(rate * 0.2), dtype=piece.dtype))

        context.progress_steps(len(chunks), len(chunks), "fertig")
        waveform = np.concatenate(pieces) if pieces else np.zeros(1)
        gain = 10.0 ** (request.volume_db / 20.0)
        peak = float(np.max(np.abs(waveform))) or 1.0
        waveform = np.clip(waveform / peak * 0.95 * gain, -1.0, 1.0)
        samples = (waveform * 32767.0).astype(np.int16)

        target = output_path(request, suffix="wav")
        _write_wav_bytes(target, samples.tobytes(), rate)
        context.log(f"geschrieben: {target}")

        if abs(request.speed - 1.0) > 0.05:
            notes.append("Bark kennt keine Geschwindigkeitsregelung – Wert ignoriert.")

        return VoiceResult(
            audio=target,
            seconds=len(samples) / max(1, rate),
            sample_rate=rate,
            backend=self.plan.backend,
            model_key=self.model.key if self.model else "bark",
            profile_slug="",
            elapsed_s=time.time() - started,
            dummy=False,
            notes=tuple(list(self.extra_notes) + notes),
        )

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._loaded = False
        _voice_cache.clear()
        from .pipeline_image import _clear_pipeline_cache

        _clear_pipeline_cache()


class ChatterboxVoicePipeline(VoicePipeline):
    """Klonstimme über Chatterbox – läuft in einer getrennten Laufzeit.

    Der eigentliche Aufruf geht über ``voice_runtime`` an einen eigenen
    Prozess. Grund: chatterbox-tts verlangt torch 2.6 ohne CUDA-Build und
    ältere diffusers/transformers und würde Bild und Video mit herunterziehen.
    """

    def load(self, context: JobContext) -> None:
        from . import voice_runtime

        ok, note = voice_runtime.available()
        if not ok:
            raise RuntimeError(note)
        context.status(f"Klon-Laufzeit bereit ({note}).")
        self._loaded = True

    def synthesize(self, request: VoiceRequest, context: JobContext) -> VoiceResult:
        from . import voice_runtime

        if not self._loaded:
            self.load(context)
        started = time.time()
        notes: list[str] = []

        decision = resolve_profile(request, self.config)
        if not decision.allowed or decision.profile is None:
            # Fail-closed: ohne gültiges Profil samt Einwilligung wird nicht
            # geklont. Der Aufrufer bekommt den Grund, keine stille Ausgabe.
            raise RuntimeError(decision.reason)

        reference = reference_clip(decision.profile)
        if reference is None:
            raise RuntimeError(
                f"Für '{decision.profile.display_name}' gibt es keine brauchbare "
                "Referenzaufnahme. Aufnahme hinzufügen und erneut anlernen."
            )
        notes.append(f"Referenz: {reference.name}")

        target = output_path(request, suffix="wav")
        chunks = _split_sentences(request.text, limit=280) if request.split_sentences \
            else [request.text]

        # Alle Sätze in EINEM Aufruf. Ein Prozess je Satz würde das mehrere
        # GB große Modell jedes Mal neu laden – bei fünf Sätzen fünfmal.
        context.progress(0.05, f"{len(chunks)} Satz/Sätze, Modell wird geladen …")
        voice_runtime.synthesize(
            reference=reference,
            text=chunks,
            output=target,
            language=request.language or "de",
            # Feinschliff steckt im Profil, nicht in der Anfrage – damit
            # klingt eine einmal eingestellte Stimme immer gleich.
            exaggeration=decision.profile.exaggeration,
            cfg=decision.profile.cfg_weight,
            temperature=decision.profile.temperature,
            seed=abs(hash(decision.profile.slug)) % (2**31) if self.config.seed_locked else 0,
            device="cpu" if self.plan.backend == accel.Backend.CPU else "auto",
            should_stop=context.should_stop,
            on_status=context.status,
        )
        context.progress(1.0, "fertig")
        context.log(f"geschrieben: {target}")

        seconds, rate = _wav_info(target)
        speaker = decision.profile.consent.speaker_name if decision.profile.consent else ""
        notes.append(f"Stimme: {decision.profile.display_name} (Einwilligung: {speaker})")
        notes.append(
            f"Feinschliff: Ausdruck {decision.profile.exaggeration:.2f}, "
            f"Führung {decision.profile.cfg_weight:.2f}, "
            f"Streuung {decision.profile.temperature:.2f}"
        )
        if request.speed != 1.0 or request.pitch:
            notes.append("Chatterbox regelt Tempo und Tonhöhe nicht – Werte ignoriert.")

        return VoiceResult(
            audio=target,
            seconds=seconds,
            sample_rate=rate,
            backend=self.plan.backend,
            model_key=self.model.key if self.model else "chatterbox",
            profile_slug=decision.profile.slug,
            elapsed_s=time.time() - started,
            dummy=False,
            notes=tuple(notes),
        )


def reference_clip(profile: voice_profiles.VoiceProfile) -> Path | None:
    """Referenzaufnahme für das Klonen: bevorzugt das Anlern-Artefakt."""
    prepared = profile.artifacts_dir / "reference.wav"
    if prepared.is_file():
        return prepared
    usable = [s for s in profile.samples() if s.usable and s.path.suffix.lower() == ".wav"]
    if usable:
        # längste brauchbare Aufnahme nehmen
        return max(usable, key=lambda s: s.seconds).path
    return None


def _wav_info(path: Path) -> tuple[float, int]:
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            rate = handle.getframerate() or 24000
            return handle.getnframes() / float(rate), rate
    except (wave.Error, OSError):
        return 0.0, 24000


def _concat_wavs(parts: Sequence[Path], target: Path) -> Path:
    """WAV-Stücke aneinanderhängen (gleiches Format vorausgesetzt)."""
    paths.ensure_dir(target.parent)
    first = parts[0]
    with contextlib.closing(wave.open(str(first), "rb")) as handle:
        params = handle.getparams()
        frames = [handle.readframes(handle.getnframes())]
    pause = b"\x00" * int(params.framerate * 0.2) * params.sampwidth * params.nchannels
    for piece in parts[1:]:
        with contextlib.closing(wave.open(str(piece), "rb")) as handle:
            frames.append(pause)
            frames.append(handle.readframes(handle.getnframes()))
    with contextlib.closing(wave.open(str(target), "wb")) as out:
        out.setparams(params)
        out.writeframes(b"".join(frames))
    return target


def _write_wav_bytes(path: Path, data: bytes, sample_rate: int) -> Path:
    """Rohdaten (int16, mono) als WAV schreiben."""
    paths.ensure_dir(path.parent)
    with contextlib.closing(wave.open(str(path), "wb")) as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(data)
    return path


# ---------------------------------------------------------------------------
# Attrappe
# ---------------------------------------------------------------------------
class DummyVoicePipeline(VoicePipeline):
    """Attrappe: erzeugt eine hörbare Platzhalter-Tonspur.

    ``reason`` sagt dem Nutzer, WARUM er einen Platzhalter bekommt – ohne
    das ist der Unterschied zur echten Ausgabe nicht erkennbar.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan, reason: str = "") -> None:
        super().__init__(config, plan)
        self.reason = reason

    def load(self, context: JobContext) -> None:
        name = self.model.key if self.model else "keins"
        context.status(f"Attrappe aktiv – kein Stimmmodell wird geladen ({name}).")
        if self.reason:
            context.log(f"Grund für die Attrappe: {self.reason}")
        self._loaded = True

    def synthesize(self, request: VoiceRequest, context: JobContext) -> VoiceResult:
        if not self._loaded:
            self.load(context)
        started = time.time()
        notes: list[str] = ["Platzhalter-Sprachausgabe, keine echte Stimme."]
        if self.reason:
            notes.append(f"Grund: {self.reason}")
        if self.model:
            notes.append(f"Gewähltes Modell: {self.model.repo_id} ({self.model.license_id}).")

        decision = resolve_profile(request, self.config)
        notes.append(decision.reason)
        if decision.allowed and decision.profile is not None:
            context.status(f"Nutze angelernte Stimme '{decision.profile.display_name}'.")
            # Aus dem Sprechernamen einen anderen Grundton ableiten, damit der
            # Unterschied hörbar ist.
            pitch_shift = (sum(decision.profile.slug.encode("utf-8")) % 9) - 4
        else:
            context.status("Nutze Standardstimme.")
            pitch_shift = 0.0

        text = request.text.strip() or "Kein Text angegeben."
        seconds = min(120.0, VoiceRequest(text=text, speed=request.speed).estimated_seconds())
        samples = render_placeholder_speech(
            text=text,
            seconds=seconds,
            sample_rate=request.sample_rate,
            pitch=request.pitch + pitch_shift,
            volume_db=request.volume_db,
            context=context,
        )
        target = output_path(request, suffix="wav")
        write_wav(target, samples, request.sample_rate)
        context.log(f"geschrieben: {target}")

        if request.file_format != "wav":
            notes.append(
                f"Format '{request.file_format}' wird beim Muxen über ffmpeg erzeugt; "
                "die Attrappe schreibt WAV."
            )

        return VoiceResult(
            audio=target,
            seconds=seconds,
            sample_rate=request.sample_rate,
            backend=self.plan.backend,
            model_key=self.model.key if self.model else "",
            profile_slug=decision.slug if decision.allowed else "",
            elapsed_s=time.time() - started,
            dummy=True,
            notes=tuple(notes),
        )


def engine_for(repo_id: str) -> str:
    """Welche Laufzeit gehört zu diesem Modell?"""
    lowered = repo_id.lower()
    if "piper" in lowered:
        return "piper"
    if "bark" in lowered:
        return "bark"
    if "kokoro" in lowered:
        return "kokoro"
    if "chatterbox" in lowered or "openvoice" in lowered:
        return "clone"
    return "unbekannt"


def engine_available(engine: str) -> tuple[bool, str]:
    """Ist die Laufzeit installiert? Sonst Klartext-Begründung."""
    import importlib.util

    if engine == "clone":
        # Klonstimmen laufen in einer eigenen Umgebung, nicht in dieser.
        from . import voice_runtime

        return voice_runtime.available()

    needed = {
        "piper": ("piper", "onnxruntime"),
        "bark": ("torch", "transformers", "numpy"),
        "kokoro": ("kokoro", "torch"),
    }.get(engine)
    if needed is None:
        return False, f"Für '{engine}' gibt es noch keine Umsetzung – Attrappe wird genutzt."
    for package in needed:
        if importlib.util.find_spec(package) is None:
            return False, f"Paket '{package}' fehlt – Attrappe wird genutzt."
    return True, ""


def create_voice_pipeline(
    config: AppConfig,
    plan: BackendPlan,
    force_dummy: bool = False,
) -> VoicePipeline:
    if force_dummy:
        return DummyVoicePipeline(config, plan, "Attrappen-Betrieb erzwungen (--dummy).")
    if not config.voice_model:
        return DummyVoicePipeline(config, plan, "Kein Stimmmodell gewählt.")
    try:
        key = config.voice_clone_model if config.voice_cloning_enabled else config.voice_model
        model = models.resolve(key)
        models.check_allowed(model, allow_conditional=True)
    except Exception as exc:  # noqa: BLE001 – Lizenzsperre verständlich melden
        reason = clean_error(exc)
        log.warning("Stimmmodell nicht verwendbar: %s", reason)
        return DummyVoicePipeline(config, plan, reason)

    engine = engine_for(model.repo_id)
    ok, reason = engine_available(engine)

    # Klonstimme gewählt, aber Laufzeit fehlt: NICHT auf den Tongenerator
    # zurückfallen – das klingt für den Nutzer wie ein Defekt. Stattdessen
    # die echte Standardstimme nehmen und den Grund nennen.
    if engine == "clone" and not ok:
        fallback = config.with_values(voice_cloning_enabled=False)
        base = models.resolve(fallback.voice_model) if fallback.voice_model else None
        base_engine = engine_for(base.repo_id) if base else ""
        base_ok, base_reason = engine_available(base_engine) if base_engine else (False, "")
        note = (f"Klonstimme '{model.key}' nicht nutzbar: {reason} "
                f"Es wird die Standardstimme verwendet.")
        log.warning("%s", note)
        if base_ok and base_engine == "bark":
            pipeline = BarkVoicePipeline(fallback, plan)
            pipeline.extra_notes = (note,)
            return pipeline
        if base_ok and base_engine == "piper":
            pipeline = PiperVoicePipeline(fallback, plan)
            pipeline.extra_notes = (note,)
            return pipeline
        return DummyVoicePipeline(config, plan, note + f" Auch die Standardstimme fehlt: {base_reason}")

    if not ok:
        text = (f"Modell '{model.key}' braucht die Laufzeit '{engine}': {reason}")
        log.warning("%s", text)
        return DummyVoicePipeline(config, plan, text)

    if engine == "piper":
        return PiperVoicePipeline(config, plan)
    if engine == "bark":
        return BarkVoicePipeline(config, plan)
    if engine == "clone":
        return ChatterboxVoicePipeline(config, plan)
    return DummyVoicePipeline(
        config, plan, f"Für '{engine}' gibt es noch keine Umsetzung.",
    )


def make_job(config: AppConfig, plan: BackendPlan, request: VoiceRequest,
             force_dummy: bool = False):
    def handler(context: JobContext) -> VoiceResult:
        pipeline = create_voice_pipeline(config, plan, force_dummy=force_dummy)
        try:
            return pipeline.synthesize(request, context)
        finally:
            if not config.keep_model_loaded:
                pipeline.unload()

    return handler


def make_training_job(config: AppConfig, plan: BackendPlan, slug: str):
    """Handler zum Anlernen einer Stimme (siehe voice_profiles.train_profile)."""

    def handler(context: JobContext) -> voice_profiles.TrainingResult:
        request = voice_profiles.TrainingRequest(
            slug=slug,
            epochs=config.voice_training_epochs,
            batch_size=config.voice_training_batch,
            learning_rate=config.voice_training_lr,
        )
        return voice_profiles.train_profile(request, context, backend=plan.backend)

    return handler
