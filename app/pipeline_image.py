"""Text -> Bild.

In dieser Basis nur Schnittstelle plus Attrappe. Die Attrappe schreibt ein
echtes PNG (Verlauf, aus dem Prompt abgeleitete Farben) – damit lassen sich
Warteschlange, Fortschritt, Abbruch, Dateinamen und Vertonung vollständig
testen, ohne mehrere GB Modell zu laden.

Der zweite Schritt ersetzt ``DummyImagePipeline`` durch eine
diffusers-Umsetzung; ``ImagePipeline`` bleibt unverändert.
"""

from __future__ import annotations

import hashlib
import logging
import math
import random
import struct
import time
import zlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from . import accel, models, paths
from .accel import BackendPlan, clean_error
from .config import AppConfig
from .jobs import JobContext

log = logging.getLogger(__name__)

SAMPLERS = ("euler_a", "euler", "dpmpp_2m", "ddim", "unipc", "lcm")


# ---------------------------------------------------------------------------
# Anfrage / Ergebnis
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    guidance: float = 6.0
    sampler: str = "euler_a"
    seed: int = -1  # -1 = zufällig
    batch: int = 1
    output_dir: Path | None = None
    file_format: str = "png"
    jpeg_quality: int = 92
    name_hint: str = ""

    @staticmethod
    def from_config(config: AppConfig, prompt: str, **overrides: Any) -> "ImageRequest":
        request = ImageRequest(
            prompt=prompt,
            negative_prompt=config.image_negative_prompt,
            width=config.image_width,
            height=config.image_height,
            steps=config.image_steps,
            guidance=config.image_guidance,
            sampler=config.image_sampler,
            batch=config.image_batch,
            output_dir=config.resolved_output_dir() / "images",
            file_format=config.image_format,
            jpeg_quality=config.image_jpeg_quality,
        )
        return replace(request, **{k: v for k, v in overrides.items() if hasattr(request, k)})

    def resolved_seed(self) -> int:
        return self.seed if self.seed >= 0 else random.randint(0, 2**31 - 1)


@dataclass(frozen=True)
class ImageResult:
    files: tuple[Path, ...]
    seed: int
    backend: str
    model_key: str
    elapsed_s: float
    width: int
    height: int
    steps: int
    dummy: bool = False
    notes: tuple[str, ...] = ()

    def first(self) -> Path | None:
        return self.files[0] if self.files else None


# ---------------------------------------------------------------------------
# Schnittstelle
# ---------------------------------------------------------------------------
class ImagePipeline(ABC):
    """Vertrag für jede Bild-Umsetzung.

    ``load()`` darf Minuten dauern und muss ``context`` für Fortschritt und
    Abbruch benutzen. ``generate()`` darf ohne vorheriges ``load()``
    aufgerufen werden – dann lädt es selbst nach.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        self.config = config
        self.plan = plan
        self.model = models.resolve(config.image_model)
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self, context: JobContext) -> None:
        """Modell in den Speicher holen (inklusive Download, falls erlaubt)."""

    @abstractmethod
    def generate(self, request: ImageRequest, context: JobContext) -> ImageResult:
        """Bild(er) erzeugen."""

    def unload(self) -> None:
        """Speicher freigeben. Vorgabe: nichts zu tun."""
        self._loaded = False

    def describe(self) -> str:
        return (
            f"{type(self).__name__}: Modell={self.model.key} "
            f"({self.model.repo_id}), Backend={self.plan.label}"
        )


# ---------------------------------------------------------------------------
# Hilfsfunktionen für Dateien
# ---------------------------------------------------------------------------
def output_path(
    request: ImageRequest,
    index: int,
    seed: int,
    suffix: str | None = None,
) -> Path:
    """Zieldatei bilden: Zeitstempel + Kurzform des Prompts + Seed."""
    directory = request.output_dir or (paths.outputs_dir() / "images")
    paths.ensure_dir(directory)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    hint = request.name_hint or request.prompt
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in hint.lower())
    slug = "-".join(part for part in slug.split("-") if part)[:48] or "bild"
    ext = suffix or request.file_format or "png"
    name = f"{stamp}_{slug}_s{seed}"
    if request.batch > 1:
        name += f"_{index + 1:02d}"
    return directory / f"{name}.{ext}"


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


def write_png(path: Path, width: int, height: int, rows: Sequence[bytes]) -> Path:
    """Minimaler PNG-Schreiber (RGB8) – ohne Fremdbibliothek.

    Wird nur von der Attrappe benutzt; die echte Umsetzung speichert über
    Pillow, weil dort auch JPEG/WebP und Metadaten gebraucht werden.
    """
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + row for row in rows)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(raw, 6))
        + _png_chunk(b"IEND", b"")
    )
    paths.ensure_dir(path.parent)
    path.write_bytes(payload)
    return path


def _palette_from_text(text: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    top = (digest[0], digest[1], digest[2])
    bottom = (digest[3], digest[4], digest[5])
    return top, bottom


def render_placeholder(
    width: int,
    height: int,
    text: str,
    seed: int,
    context: JobContext | None = None,
    steps: int = 0,
) -> list[bytes]:
    """Verlauf plus leichte Struktur – erkennbar als Platzhalter."""
    top, bottom = _palette_from_text(f"{text}:{seed}")
    rows: list[bytes] = []
    report_every = max(1, height // 20)
    for y in range(height):
        if context is not None and y % report_every == 0:
            context.raise_if_cancelled()
        ratio = y / max(1, height - 1)
        base = [int(top[i] + (bottom[i] - top[i]) * ratio) for i in range(3)]
        wave = int(12 * math.sin((y / max(8, height / 12)) + seed % 7))
        row = bytearray()
        for x in range(width):
            shade = int(18 * math.sin((x / max(8, width / 16)) + ratio * 3.0))
            row.extend(
                bytes(
                    (
                        max(0, min(255, base[0] + wave + shade)),
                        max(0, min(255, base[1] + shade)),
                        max(0, min(255, base[2] - wave + shade)),
                    )
                )
            )
        rows.append(bytes(row))
    return rows


# ---------------------------------------------------------------------------
# Echte Umsetzung über diffusers
# ---------------------------------------------------------------------------
# Ein geladenes SDXL belegt mehrere GB. Ohne Zwischenspeicher würde jeder
# Auftrag das Modell neu von der Platte holen – Minuten statt Sekunden.
_pipeline_cache: dict[tuple, Any] = {}


def _clear_pipeline_cache(keep: tuple | None = None) -> None:
    import gc

    for key in list(_pipeline_cache):
        if key == keep:
            continue
        _pipeline_cache.pop(key, None)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 – Aufräumen darf nie werfen
        pass


def _scheduler_for(sampler: str, current) -> Any:
    """Sampler-Kurzname auf einen diffusers-Scheduler abbilden."""
    import diffusers

    mapping = {
        "euler_a": ("EulerAncestralDiscreteScheduler", {}),
        "euler": ("EulerDiscreteScheduler", {}),
        "dpmpp_2m": ("DPMSolverMultistepScheduler",
                     {"algorithm_type": "dpmsolver++", "use_karras_sigmas": True}),
        "ddim": ("DDIMScheduler", {}),
        "unipc": ("UniPCMultistepScheduler", {}),
        "lcm": ("LCMScheduler", {}),
    }
    name, extra = mapping.get(sampler, mapping["euler_a"])
    cls = getattr(diffusers, name, None)
    if cls is None:
        return current
    return cls.from_config(current.config, **extra)


def _free_vram_mb(device_index: int) -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free, _total = torch.cuda.mem_get_info(device_index)
        return int(free / (1024 * 1024))
    except Exception:  # noqa: BLE001
        return 0


class DiffusersImagePipeline(ImagePipeline):
    """Text zu Bild über diffusers.

    Deckt SD 1.5, SDXL und FLUX ab – welche Klasse gebraucht wird, steht in
    der ``model_index.json`` des Modells, das übernimmt
    ``AutoPipelineForText2Image``.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        super().__init__(config, plan)
        self._pipe: Any = None
        self._family = ""

    # --- Laden -------------------------------------------------------------
    def _cache_key(self) -> tuple:
        return (self.model.repo_id, self.plan.backend, self.plan.compute_type,
                self.config.cpu_offload)

    def load(self, context: JobContext) -> None:
        key = self._cache_key()
        cached = _pipeline_cache.get(key)
        if cached is not None:
            self._pipe = cached
            self._family = _family_of(cached)
            self._loaded = True
            context.status("Modell liegt bereits im Speicher.")
            return

        # Modell auf der Platte sicherstellen (lädt bei Bedarf herunter)
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

        context.progress(0.0, f"Lade {self.model.title} …")
        context.raise_if_cancelled()

        import torch
        from diffusers import DiffusionPipeline

        dtype = accel.torch_dtype(self.plan)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "local_files_only": True,
        }

        # DiffusionPipeline statt AutoPipelineForText2Image: Auto zieht die
        # Zuordnungstabelle ALLER Pipelines mit, darunter Kolors. Dessen
        # Text-Encoder ruft beim Import torch.jit.script auf, und das braucht
        # den Python-Quelltext – im PyInstaller-Bundle liegen aber nur .pyc.
        # Ergebnis wäre "Can't get source ... TorchScript requires source access".
        # DiffusionPipeline lädt nur die Klasse aus model_index.json.
        #
        # Die Gewichte liegen als fp16-Variante vor (siehe models.select_files);
        # ohne variant="fp16" sucht diffusers die Vollfassung und scheitert.
        pipe = None
        errors: list[str] = []
        for variant in ((self.model.variant,) if self.model.variant else ()) + (None,):
            try:
                kwargs = dict(load_kwargs)
                if variant:
                    kwargs["variant"] = variant
                pipe = DiffusionPipeline.from_pretrained(path, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001 – nächste Variante probieren
                errors.append(f"variant={variant}: {clean_error(exc)}")
        if pipe is None:
            raise RuntimeError(
                f"{self.model.title} konnte nicht geladen werden. " + " | ".join(errors[-2:])
            )

        context.raise_if_cancelled()
        self._family = _family_of(pipe)

        # Scheduler nur bei den Modellen tauschen, die klassische Sampler
        # nutzen. FLUX rechnet mit Flow-Matching – dort wäre es falsch.
        if self._family != "flux":
            try:
                pipe.scheduler = _scheduler_for(self.config.image_sampler, pipe.scheduler)
            except Exception as exc:  # noqa: BLE001 – Vorgabe behalten
                context.log(f"Sampler '{self.config.image_sampler}' nicht nutzbar: "
                            f"{clean_error(exc)}")

        # Speicherverhalten
        needed_mb = self.model.min_vram_mb or 6000
        free_mb = _free_vram_mb(self.plan.device_index)
        offload = self.config.cpu_offload or (
            self.plan.backend == accel.Backend.CUDA and free_mb and free_mb < needed_mb
        )

        if self.plan.backend == accel.Backend.CUDA:
            if offload:
                context.status(
                    f"Wenig freier Grafikspeicher ({free_mb} MB) – Modellteile werden "
                    "bei Bedarf ausgelagert. Das ist langsamer, aber es läuft."
                )
                pipe.enable_model_cpu_offload(device=f"cuda:{self.plan.device_index}")
            else:
                pipe.to(f"cuda:{self.plan.device_index}")
        else:
            pipe.to("cpu")
            threads = self.config.cpu_threads
            if threads > 0:
                torch.set_num_threads(threads)

        if self.config.attention_slicing or self.config.gpu_low_impact:
            with _quiet():
                pipe.enable_attention_slicing()
        if self.config.vae_tiling:
            with _quiet():
                _try(pipe, "enable_vae_tiling")
        _try(pipe, "set_progress_bar_config", disable=True)

        if self.config.keep_model_loaded:
            _clear_pipeline_cache(keep=None)  # nur eine Pipeline gleichzeitig
            _pipeline_cache[key] = pipe
        self._pipe = pipe
        self._loaded = True
        context.status(f"{self.model.title} bereit ({self.plan.label}).")

    # --- Erzeugen ----------------------------------------------------------
    def generate(self, request: ImageRequest, context: JobContext) -> ImageResult:
        if not self._loaded:
            self.load(context)
        import torch

        started = time.time()
        seed = request.resolved_seed()
        notes: list[str] = []
        batch = max(1, request.batch)
        steps = max(1, request.steps)
        width, height = _snap(request.width), _snap(request.height)
        if (width, height) != (request.width, request.height):
            notes.append(f"Größe auf {width}x{height} gerundet (Vielfaches von 8).")

        guidance = request.guidance
        negative = request.negative_prompt or None
        if self._family == "flux":
            # FLUX.1-schnell ist auf 1–4 Schritte ohne Führung destilliert.
            if guidance > 0:
                notes.append("FLUX [schnell] arbeitet ohne Führung – CFG auf 0 gesetzt.")
                guidance = 0.0
            if steps > 8:
                notes.append(f"FLUX [schnell] braucht 1–4 Schritte – {steps} auf 4 gekürzt.")
                steps = 4
            if negative:
                notes.append("FLUX kennt keinen Negativ-Prompt – Eingabe wird ignoriert.")
                negative = None

        files: list[Path] = []
        total_units = batch * steps

        for index in range(batch):
            context.raise_if_cancelled()
            image_seed = seed + index
            generator = torch.Generator(device="cpu").manual_seed(image_seed)
            offset = index * steps

            def callback(pipe, step_index, timestep, callback_kwargs, _offset=offset):
                # Einziger Punkt, an dem ein laufender Diffusionslauf
                # abgebrochen werden kann – diffusers hat keinen Rückgabewert
                # dafür, deshalb über eine Ausnahme.
                context.raise_if_cancelled()
                done = _offset + step_index + 1
                context.progress(done / total_units,
                                 f"Bild {index + 1}/{batch}, Schritt {step_index + 1}/{steps}")
                return callback_kwargs

            call_kwargs: dict[str, Any] = {
                "prompt": request.prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "generator": generator,
                "callback_on_step_end": callback,
            }
            if negative is not None:
                call_kwargs["negative_prompt"] = negative
            if self._family == "flux":
                call_kwargs["max_sequence_length"] = 256

            context.status(f"Bild {index + 1}/{batch} wird gerechnet …")
            try:
                with torch.inference_mode():
                    output = self._pipe(**call_kwargs)
            except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
                _clear_pipeline_cache()
                raise RuntimeError(
                    f"Grafikspeicher reicht für {width}x{height} nicht aus. "
                    "Kleinere Auflösung wählen, Anzahl auf 1 setzen oder in den "
                    "Einstellungen 'Modellteile auslagern' einschalten."
                ) from exc

            image = output.images[0]
            target = output_path(request, index, image_seed, suffix=request.file_format)
            _save_image(image, target, request, image_seed, self.model.repo_id, steps, guidance)
            files.append(target)
            context.log(f"geschrieben: {target}")

        return ImageResult(
            files=tuple(files),
            seed=seed,
            backend=self.plan.backend,
            model_key=self.model.key,
            elapsed_s=time.time() - started,
            width=width,
            height=height,
            steps=steps,
            dummy=False,
            notes=tuple(notes),
        )

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False
        _clear_pipeline_cache()


def _family_of(pipe: Any) -> str:
    name = type(pipe).__name__.lower()
    if "flux" in name:
        return "flux"
    if "xl" in name:
        return "sdxl"
    if "stablediffusion3" in name:
        return "sd3"
    return "sd"


def _snap(value: int) -> int:
    return max(256, (int(value) // 8) * 8)


def _try(obj: Any, method: str, *args: Any, **kwargs: Any) -> None:
    """Optionale Pipeline-Funktion aufrufen, falls vorhanden."""
    function = getattr(obj, method, None)
    if callable(function):
        try:
            function(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 – Komfortfunktion, kein Muss
            log.debug("%s fehlgeschlagen: %s", method, exc)


class _quiet:
    """Kontext, der Ausnahmen aus Komfortfunktionen schluckt."""

    def __enter__(self) -> "_quiet":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None:
            log.debug("Speicheroption nicht verfügbar: %s", exc)
        return True


def _save_image(image, target: Path, request: ImageRequest, seed: int,
                repo_id: str, steps: int, guidance: float) -> Path:
    """Bild speichern – mit Erzeugungsdaten in den Metadaten."""
    paths.ensure_dir(target.parent)
    suffix = target.suffix.lower()
    metadata = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt,
        "seed": str(seed),
        "steps": str(steps),
        "guidance": str(guidance),
        "sampler": request.sampler,
        "model": repo_id,
        "software": "StreamForge Studio",
    }
    if suffix == ".png":
        from PIL import PngImagePlugin

        info = PngImagePlugin.PngInfo()
        for key, value in metadata.items():
            info.add_text(key, str(value))
        image.save(target, format="PNG", pnginfo=info, optimize=False)
    elif suffix in (".jpg", ".jpeg"):
        image.convert("RGB").save(target, format="JPEG", quality=request.jpeg_quality,
                                  subsampling=0)
    elif suffix == ".webp":
        image.save(target, format="WEBP", quality=request.jpeg_quality)
    else:
        image.save(target)
    return target


# ---------------------------------------------------------------------------
# Attrappe
# ---------------------------------------------------------------------------
class DummyImagePipeline(ImagePipeline):
    """Attrappe: schreibt ein echtes PNG, lädt aber kein Modell.

    ``reason`` erklärt, warum kein echtes Bild entsteht – ein Farbverlauf
    ohne Begründung sieht sonst wie ein Fehler aus.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan, reason: str = "") -> None:
        super().__init__(config, plan)
        self.reason = reason

    def load(self, context: JobContext) -> None:
        context.status(f"Attrappe aktiv – kein Modell wird geladen ({self.model.key}).")
        if self.reason:
            context.log(f"Grund für die Attrappe: {self.reason}")
        total = 8
        for step in range(total):
            context.raise_if_cancelled()
            context.progress_steps(step + 1, total, "Vorbereitung")
            time.sleep(0.02)
        self._loaded = True

    def generate(self, request: ImageRequest, context: JobContext) -> ImageResult:
        if not self._loaded:
            self.load(context)
        started = time.time()
        seed = request.resolved_seed()
        files: list[Path] = []
        notes = ["Platzhalterbild, kein echtes Modellergebnis."]
        if self.reason:
            notes.append(f"Grund: {self.reason}")
        notes.append(f"Gewähltes Modell: {self.model.repo_id} ({self.model.license_id}).")

        for index in range(max(1, request.batch)):
            context.raise_if_cancelled()
            context.status(f"Bild {index + 1}/{request.batch}")
            # Diffusionsschritte simulieren, damit Fortschritt/Abbruch stimmen
            for step in range(max(1, request.steps)):
                context.raise_if_cancelled()
                context.progress_steps(
                    index * request.steps + step + 1,
                    max(1, request.batch) * max(1, request.steps),
                    f"Schritt {step + 1}/{request.steps}",
                )
                time.sleep(0.01)
            rows = render_placeholder(
                request.width, request.height, request.prompt, seed + index, context, request.steps
            )
            target = output_path(request, index, seed + index, suffix="png")
            files.append(write_png(target, request.width, request.height, rows))
            context.log(f"geschrieben: {target}")

        return ImageResult(
            files=tuple(files),
            seed=seed,
            backend=self.plan.backend,
            model_key=self.model.key,
            elapsed_s=time.time() - started,
            width=request.width,
            height=request.height,
            steps=request.steps,
            dummy=True,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Fabrik
# ---------------------------------------------------------------------------
def diffusers_available() -> tuple[bool, str]:
    """Sind torch und diffusers nutzbar? Klartext-Begründung, wenn nicht."""
    import importlib.util

    for package in ("torch", "diffusers", "transformers", "PIL"):
        if importlib.util.find_spec(package) is None:
            return False, f"Paket '{package}' fehlt – es wird die Attrappe genutzt."
    return True, ""


def create_image_pipeline(
    config: AppConfig,
    plan: BackendPlan,
    force_dummy: bool = False,
) -> ImagePipeline:
    """Passende Umsetzung wählen: echt, sonst Attrappe mit Begründung.

    Der Aufrufer erkennt die Attrappe an ``ImageResult.dummy``.
    """
    if force_dummy:
        return DummyImagePipeline(config, plan, "Attrappen-Betrieb erzwungen (--dummy).")

    try:
        model = models.resolve(config.image_model)
        models.check_allowed(model, allow_conditional=True)
    except Exception as exc:  # noqa: BLE001 – Lizenzsperre verständlich melden
        reason = clean_error(exc)
        log.warning("Bildmodell nicht verwendbar: %s", reason)
        return DummyImagePipeline(config, plan, reason)

    ok, reason = diffusers_available()
    if not ok:
        log.warning("Echte Bild-Pipeline nicht möglich: %s", reason)
        return DummyImagePipeline(config, plan, reason)

    return DiffusersImagePipeline(config, plan)


def make_job(config: AppConfig, plan: BackendPlan, request: ImageRequest,
             force_dummy: bool = False):
    """Handler für die Warteschlange (``jobs.JobQueue.submit``)."""

    def handler(context: JobContext) -> ImageResult:
        pipeline = create_image_pipeline(config, plan, force_dummy=force_dummy)
        try:
            return pipeline.generate(request, context)
        finally:
            if not config.keep_model_loaded:
                pipeline.unload()

    return handler
