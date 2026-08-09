"""Bild/Text -> Video.

Basis: Schnittstelle plus Attrappe. Die Attrappe schreibt echte Einzelbilder
und lässt sie – wenn ffmpeg vorhanden ist – zu einem Video zusammensetzen.
Fehlt ffmpeg, bleiben die Einzelbilder liegen und es gibt eine verständliche
Meldung statt eines Stacktrace.
"""

from __future__ import annotations

import logging
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from . import accel, compose, contentgate, models, paths
from .accel import BackendPlan, clean_error
from .config import AppConfig
from .jobs import JobContext
from .pipeline_image import render_placeholder, write_png

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class VideoRequest:
    prompt: str
    negative_prompt: str = ""
    init_image: Path | None = None  # gesetzt = Bild-zu-Video
    width: int = 832
    height: int = 480
    frames: int = 49
    fps: int = 16
    steps: int = 30
    guidance: float = 5.0
    motion: float = 1.0
    seed: int = -1
    output_dir: Path | None = None
    container: str = "mp4"
    crf: int = 20
    codec: str = "libopenh264"
    audio_file: Path | None = None  # direkt mitvertonen
    keep_frames: bool = False
    name_hint: str = ""

    @staticmethod
    def from_config(config: AppConfig, prompt: str, **overrides: Any) -> "VideoRequest":
        request = VideoRequest(
            prompt=prompt,
            width=config.video_width,
            height=config.video_height,
            frames=config.video_frames,
            fps=config.video_fps,
            steps=config.video_steps,
            guidance=config.video_guidance,
            motion=config.video_motion,
            output_dir=config.resolved_output_dir() / "videos",
            container=config.video_container,
            crf=config.video_crf,
            codec=config.video_codec,
        )
        return replace(request, **{k: v for k, v in overrides.items() if hasattr(request, k)})

    @property
    def duration_s(self) -> float:
        return self.frames / float(max(1, self.fps))


@dataclass(frozen=True)
class VideoResult:
    video: Path | None
    frames_dir: Path | None
    frame_count: int
    fps: int
    seed: int
    backend: str
    model_key: str
    elapsed_s: float
    dummy: bool = False
    notes: tuple[str, ...] = ()


class VideoPipeline(ABC):
    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        self.config = config
        self.plan = plan
        self.model = models.resolve(config.video_model) if config.video_model else None
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, context: JobContext) -> None: ...

    @abstractmethod
    def generate(self, request: VideoRequest, context: JobContext) -> VideoResult: ...

    def unload(self) -> None:
        self._loaded = False

    def describe(self) -> str:
        model = self.model.repo_id if self.model else "kein Modell gewählt"
        return f"{type(self).__name__}: Modell={model}, Backend={self.plan.label}"


def _frames_dir(request: VideoRequest, stamp: str) -> Path:
    base = request.output_dir or (paths.outputs_dir() / "videos")
    return paths.ensure_dir(base / f"frames_{stamp}")


class DiffusersVideoPipeline(VideoPipeline):
    """Text/Bild zu Video über diffusers.

    Deckt Wan 2.1, CogVideoX und AnimateDiff ab. Welche Pipeline-Klasse
    gebraucht wird, steht in der ``model_index.json`` – deshalb reicht
    ``DiffusionPipeline.from_pretrained``. AnimateDiff ist der Sonderfall:
    dort ist das Repo nur ein Bewegungs-Adapter, der auf ein SD-1.5-Modell
    aufgesetzt wird.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        super().__init__(config, plan)
        self._pipe = None
        self._family = ""

    def _cache_key(self) -> tuple:
        repo = self.model.repo_id if self.model else ""
        return ("video", repo, self.plan.backend, self.plan.compute_type)

    def load(self, context: JobContext) -> None:
        from .pipeline_image import _clear_pipeline_cache, _pipeline_cache, _try

        key = self._cache_key()
        cached = _pipeline_cache.get(key)
        if cached is not None:
            self._pipe = cached
            self._family = _video_family(cached)
            self._loaded = True
            context.status("Videomodell liegt bereits im Speicher.")
            return

        if self.model is None:
            raise RuntimeError("Kein Videomodell gewählt (Einstellungen → Modelle).")

        def on_progress(done: int, total: int) -> None:
            context.progress((done / total) if total else 0.0,
                             f"Download {done / (1024 ** 2):.0f} MB von {total / (1024 ** 2):.0f} MB")

        try:
            path = models.ensure_local(
                self.model,
                allow_download=self.config.allow_model_download,
                on_progress=on_progress,
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
        context.raise_if_cancelled()

        import torch
        from diffusers import DiffusionPipeline

        dtype = accel.torch_dtype(self.plan)
        # Video-VAEs rechnen in float32 deutlich stabiler; bei bfloat16-fähigen
        # Karten ist bf16 die bessere Wahl als fp16 (weniger Überläufe).
        if self.plan.backend == accel.Backend.CUDA and dtype == torch.float16:
            try:
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
            except Exception:  # noqa: BLE001
                pass

        if "animatediff" in self.model.repo_id.lower():
            pipe = self._load_animatediff(path, dtype, context)
        else:
            kwargs: dict[str, Any] = {"torch_dtype": dtype, "local_files_only": True}
            if self.model.variant:
                try:
                    pipe = DiffusionPipeline.from_pretrained(path, variant=self.model.variant,
                                                             **kwargs)
                except Exception:  # noqa: BLE001 – ohne Variante erneut versuchen
                    pipe = DiffusionPipeline.from_pretrained(path, **kwargs)
            else:
                pipe = DiffusionPipeline.from_pretrained(path, **kwargs)

        self._family = _video_family(pipe)
        context.raise_if_cancelled()

        # Video ist der speicherhungrigste Pfad – Auslagerung ist hier die
        # Vorgabe, nicht die Ausnahme.
        if self.plan.backend == accel.Backend.CUDA:
            if self.config.cpu_offload or self.config.gpu_low_impact:
                _try(pipe, "enable_model_cpu_offload", device=f"cuda:{self.plan.device_index}")
            else:
                pipe.to(f"cuda:{self.plan.device_index}")
        else:
            pipe.to("cpu")

        _try(pipe, "enable_vae_slicing")
        _try(pipe, "enable_vae_tiling")
        if self.config.attention_slicing:
            _try(pipe, "enable_attention_slicing")
        _try(pipe, "set_progress_bar_config", disable=True)

        _clear_pipeline_cache(keep=None)
        if self.config.keep_model_loaded:
            _pipeline_cache[key] = pipe
        self._pipe = pipe
        self._loaded = True
        context.status(f"{self.model.title} bereit ({self.plan.label}).")

    def _load_animatediff(self, path, dtype, context: JobContext):
        """AnimateDiff: Bewegungs-Adapter auf ein SD-1.5-Modell setzen."""
        from diffusers import AnimateDiffPipeline, MotionAdapter

        base_spec = models.resolve("sd15")
        context.status("AnimateDiff braucht zusätzlich Stable Diffusion 1.5 …")
        base_path = models.ensure_local(
            base_spec,
            allow_download=self.config.allow_model_download,
            on_status=context.status,
            should_stop=context.should_stop,
            allow_conditional=True,
            offline=self.config.offline_mode,
        )
        adapter = MotionAdapter.from_pretrained(path, torch_dtype=dtype, local_files_only=True)
        return AnimateDiffPipeline.from_pretrained(
            base_path, motion_adapter=adapter, torch_dtype=dtype, local_files_only=True,
            variant=base_spec.variant or None,
        )

    def generate(self, request: VideoRequest, context: JobContext) -> VideoResult:
        if not self._loaded:
            self.load(context)
        import torch

        started = time.time()
        seed = request.seed if request.seed >= 0 else int(started) % (2**31)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        notes: list[str] = []

        width, height = _snap_video(request.width), _snap_video(request.height)
        frames = _snap_frames(request.frames, self._family)
        if frames != request.frames:
            notes.append(f"Bildzahl auf {frames} angepasst (Modell verlangt 4n+1).")
        steps = max(1, request.steps)

        generator = torch.Generator(device="cpu").manual_seed(seed)

        def callback(pipe, step_index, timestep, callback_kwargs):
            context.raise_if_cancelled()
            context.progress((step_index + 1) / steps,
                             f"Diffusion {step_index + 1}/{steps}")
            return callback_kwargs

        call_kwargs: dict[str, Any] = {
            "prompt": request.prompt,
            "num_inference_steps": steps,
            "guidance_scale": request.guidance,
            "generator": generator,
            "callback_on_step_end": callback,
        }
        if request.negative_prompt:
            call_kwargs["negative_prompt"] = request.negative_prompt

        if self._family == "animatediff":
            call_kwargs["num_frames"] = frames
        else:
            call_kwargs.update({"width": width, "height": height, "num_frames": frames})

        if request.init_image is not None and Path(request.init_image).is_file():
            if "image" in getattr(self._pipe, "_callback_tensor_inputs", []) or True:
                from PIL import Image

                call_kwargs["image"] = Image.open(request.init_image).convert("RGB")

        context.status(f"Rechne {frames} Bilder bei {width}x{height} …")
        try:
            with torch.inference_mode():
                output = self._pipe(**call_kwargs)
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            from .pipeline_image import _clear_pipeline_cache

            _clear_pipeline_cache()
            raise RuntimeError(
                f"Grafikspeicher reicht für {width}x{height} mit {frames} Bildern nicht. "
                "Kleinere Auflösung, weniger Bilder oder 'Modellteile auslagern' wählen."
            ) from exc

        images = output.frames[0]
        context.status(f"{len(images)} Bilder erzeugt – schreibe Video …")

        frames_dir = _frames_dir(request, stamp)
        for index, image in enumerate(images):
            context.raise_if_cancelled()
            image.save(frames_dir / f"frame_{index:05d}.png")

        video: Path | None = None
        try:
            video = compose.frames_to_video(
                frames_dir=frames_dir,
                pattern="frame_%05d.png",
                fps=request.fps,
                output=(request.output_dir or paths.outputs_dir() / "videos")
                / f"{stamp}_{_slug(request.prompt)}.{request.container}",
                crf=request.crf,
                codec=request.codec,
                audio=request.audio_file,
                context=context,
            )
        except compose.FfmpegMissing as exc:
            notes.append(str(exc))
            context.log(str(exc))
        except compose.FfmpegError as exc:
            notes.append(f"ffmpeg-Fehler: {clean_error(exc)}")
            context.log(f"ffmpeg-Fehler: {clean_error(exc)}")

        keep = request.keep_frames or video is None
        if not keep:
            shutil.rmtree(frames_dir, ignore_errors=True)

        return VideoResult(
            video=video,
            frames_dir=frames_dir if keep else None,
            frame_count=len(images),
            fps=request.fps,
            seed=seed,
            backend=self.plan.backend,
            model_key=self.model.key if self.model else "",
            elapsed_s=time.time() - started,
            dummy=False,
            notes=tuple(notes),
        )

    def unload(self) -> None:
        from .pipeline_image import _clear_pipeline_cache

        self._pipe = None
        self._loaded = False
        _clear_pipeline_cache()


def _video_family(pipe) -> str:
    name = type(pipe).__name__.lower()
    for family in ("wan", "cogvideox", "animatediff", "ltx", "mochi", "hunyuan"):
        if family in name:
            return family
    return "unbekannt"


def _snap_video(value: int) -> int:
    return max(256, (int(value) // 16) * 16)


def _snap_frames(frames: int, family: str) -> int:
    """Videomodelle brauchen 4n+1 Bilder (zeitlicher VAE-Faktor 4)."""
    frames = max(9, int(frames))
    if family in ("animatediff",):
        return min(64, frames)
    return ((frames - 1) // 4) * 4 + 1


def _slug(text: str) -> str:
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in text.lower())
    return "-".join(part for part in slug.split("-") if part)[:40] or "video"


class DummyVideoPipeline(VideoPipeline):
    """Attrappe: Einzelbilder mit wanderndem Farbverlauf, dann ffmpeg.

    ``reason`` erklärt, warum kein echtes Video entsteht.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan, reason: str = "") -> None:
        super().__init__(config, plan)
        self.reason = reason

    def load(self, context: JobContext) -> None:
        name = self.model.key if self.model else "keins"
        context.status(f"Attrappe aktiv – kein Videomodell wird geladen ({name}).")
        if self.reason:
            context.log(f"Grund für die Attrappe: {self.reason}")
        self._loaded = True

    def generate(self, request: VideoRequest, context: JobContext) -> VideoResult:
        if not self._loaded:
            self.load(context)
        started = time.time()
        seed = request.seed if request.seed >= 0 else int(started) % 100000
        stamp = time.strftime("%Y%m%d-%H%M%S")
        frames_dir = _frames_dir(request, stamp)
        notes: list[str] = ["Platzhaltervideo, kein echtes Modellergebnis."]
        if self.reason:
            notes.append(f"Grund: {self.reason}")
        if self.model:
            notes.append(f"Gewähltes Modell: {self.model.repo_id} ({self.model.license_id}).")

        # Kleine Auflösung für die Attrappe – volle Größe wäre nur Wartezeit.
        width = min(request.width, 512)
        height = min(request.height, 288)
        if (width, height) != (request.width, request.height):
            notes.append(
                f"Attrappe rechnet in {width}x{height} statt {request.width}x{request.height}."
            )

        written: list[Path] = []
        total = max(1, request.frames)
        for index in range(total):
            context.raise_if_cancelled()
            context.progress_steps(index + 1, total, f"Bild {index + 1}/{total}")
            rows = render_placeholder(
                width, height, f"{request.prompt}#{index}", seed + index * 7
            )
            target = frames_dir / f"frame_{index:05d}.png"
            written.append(write_png(target, width, height, rows))

        context.status("Setze Einzelbilder zu einem Video zusammen …")
        video: Path | None = None
        try:
            video = compose.frames_to_video(
                frames_dir=frames_dir,
                pattern="frame_%05d.png",
                fps=request.fps,
                output=(request.output_dir or paths.outputs_dir() / "videos")
                / f"{stamp}_platzhalter.{request.container}",
                crf=request.crf,
                codec=request.codec,
                audio=request.audio_file,
                context=context,
            )
        except compose.FfmpegMissing as exc:
            notes.append(str(exc))
            context.log(str(exc))
        except compose.FfmpegError as exc:
            notes.append(f"ffmpeg-Fehler: {clean_error(exc)}")
            context.log(f"ffmpeg-Fehler: {clean_error(exc)}")

        keep = request.keep_frames or video is None
        if not keep:
            shutil.rmtree(frames_dir, ignore_errors=True)

        return VideoResult(
            video=video,
            frames_dir=frames_dir if keep else None,
            frame_count=len(written),
            fps=request.fps,
            seed=seed,
            backend=self.plan.backend,
            model_key=self.model.key if self.model else "",
            elapsed_s=time.time() - started,
            dummy=True,
            notes=tuple(notes),
        )


def create_video_pipeline(
    config: AppConfig,
    plan: BackendPlan,
    force_dummy: bool = False,
) -> VideoPipeline:
    if force_dummy:
        return DummyVideoPipeline(config, plan, "Attrappen-Betrieb erzwungen (--dummy).")
    if not config.video_model:
        return DummyVideoPipeline(config, plan, "Kein Videomodell gewählt.")
    try:
        model = models.resolve(config.video_model)
        models.check_allowed(model, allow_conditional=True)
    except Exception as exc:  # noqa: BLE001
        reason = clean_error(exc)
        log.warning("Videomodell nicht verwendbar: %s", reason)
        return DummyVideoPipeline(config, plan, reason)

    from .pipeline_image import diffusers_available

    ok, reason = diffusers_available()
    if not ok:
        log.warning("Echte Video-Pipeline nicht möglich: %s", reason)
        return DummyVideoPipeline(config, plan, reason)
    return DiffusersVideoPipeline(config, plan)


def make_job(config: AppConfig, plan: BackendPlan, request: VideoRequest,
             force_dummy: bool = False):
    def handler(context: JobContext) -> VideoResult:
        # Gleiche Sperre wie beim Bild – ein Video ist nur eine Folge davon.
        contentgate.enforce(request.prompt, request.negative_prompt)
        pipeline = create_video_pipeline(config, plan, force_dummy=force_dummy)
        try:
            return pipeline.generate(request, context)
        finally:
            if not config.keep_model_loaded:
                pipeline.unload()

    return handler
