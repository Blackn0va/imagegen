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
from typing import Any, Callable, Sequence

from . import accel, contentgate, models, paths, upscale
from .accel import BackendPlan, clean_error
from .config import AppConfig
from .jobs import JobContext

log = logging.getLogger(__name__)

SAMPLERS = ("euler_a", "euler", "dpmpp_2m", "ddim", "unipc", "lcm")

# Was mit einem bestehenden Bild passieren kann.
EDIT_MODES: tuple[str, ...] = ("img2img", "inpaint", "upscale")
EDIT_MODE_LABELS = {
    "img2img": "Nach Prompt umarbeiten",
    "inpaint": "Bereich ersetzen (Maske)",
    "upscale": "Vergrößern",
}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")


# ---------------------------------------------------------------------------
# Inhalte für Erwachsene
# ---------------------------------------------------------------------------
def adult_content_allowed(config: AppConfig) -> tuple[bool, str]:
    """Dürfen Inhalte für Erwachsene erzeugt werden?

    Eine Einstellung, keine Zustimmungskette: die Anwendung läuft lokal
    und wird nicht weitergegeben. Unabhängig davon bleibt die Sperre gegen
    Darstellungen Minderjähriger aktiv – die sitzt in ``contentgate``.
    """
    if not getattr(config, "nsfw_enabled", True):
        return False, "In den Einstellungen abgeschaltet."
    return True, ""


def safety_checker_kwargs(config: AppConfig, model_dir: Path) -> tuple[dict[str, Any], str]:
    """Lade-Argumente für die Inhaltsprüfung des Modells.

    Die Prüfung steckt als eigene Komponente im Repo – SD 1.5 hat eine,
    SDXL und FLUX nicht. Sie schwärzt jedes Bild, das sie für nicht
    jugendfrei hält. Rückgabe: (Argumente, Begründung im Klartext).

    Getrennt von ``load()``, damit die Entscheidung ohne geladenes Modell
    prüfbar ist.
    """
    if not (Path(model_dir) / "safety_checker").is_dir():
        return {}, "Dieses Modell bringt keine eigene Inhaltsprüfung mit."
    allowed, reason = adult_content_allowed(config)
    if not allowed:
        return {}, f"Inhaltsprüfung des Modells bleibt aktiv: {reason}"
    if not getattr(config, "nsfw_disable_safety_checker", True):
        return {}, "Inhaltsprüfung des Modells bleibt auf Wunsch aktiv."
    return (
        {"safety_checker": None, "requires_safety_checker": False},
        "Inhaltsprüfung des Modells abgeschaltet. "
        "Die Sperre gegen Darstellungen Minderjähriger bleibt aktiv.",
    )


def _negative_with_protection(config: AppConfig, negative: str | None) -> str | None:
    """Schutzbegriffe anhängen, solange Erwachsenen-Inhalte frei sind."""
    allowed, _reason = adult_content_allowed(config)
    if not (allowed and getattr(config, "nsfw_protective_negative", True)):
        return negative
    return contentgate.with_protective_negative(negative or "", active=True)


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
class EditRequest:
    """Auftrag für ein **bestehendes** Bild.

    Ein Auftrag kann mehrere Dateien umfassen – beim Vergrößern ist das der
    Regelfall. ``mode`` entscheidet, welche Felder überhaupt gelesen werden;
    die übrigen bleiben auf ihrer Vorgabe und stören nicht.
    """

    sources: tuple[Path, ...] = ()
    mode: str = "img2img"
    prompt: str = ""
    negative_prompt: str = ""
    mask: Path | None = None
    strength: float = 0.45
    steps: int = 25
    guidance: float = 6.0
    sampler: str = "euler_a"
    seed: int = -1
    # --- nur beim Vergrößern ---
    factor: int = 2
    use_model: bool = True
    tile: int = 512
    refine: bool = False  # nach dem Vergrößern mit dem Bildmodell nachschärfen
    refine_strength: float = 0.25
    # --- Ablage ---
    max_side: int = 0  # 0 = Ausgangsgröße behalten
    output_dir: Path | None = None
    file_format: str = "png"
    jpeg_quality: int = 92
    name_hint: str = ""

    @staticmethod
    def from_config(config: AppConfig, sources: Sequence[Path], **overrides: Any) -> "EditRequest":
        request = EditRequest(
            sources=tuple(Path(item) for item in sources),
            negative_prompt=config.image_negative_prompt,
            strength=config.image_edit_strength,
            steps=config.image_steps,
            guidance=config.image_guidance,
            sampler=config.image_sampler,
            factor=config.upscale_factor,
            use_model=config.upscale_use_model,
            tile=config.upscale_tile,
            refine=config.upscale_refine,
            refine_strength=config.image_edit_refine_strength,
            output_dir=config.resolved_output_dir() / "images",
            file_format=config.image_format,
            jpeg_quality=config.image_jpeg_quality,
        )
        clean = {k: v for k, v in overrides.items() if hasattr(request, k)}
        if "sources" in clean:
            clean["sources"] = tuple(Path(item) for item in clean["sources"])
        return replace(request, **clean)

    def resolved_seed(self) -> int:
        return self.seed if self.seed >= 0 else random.randint(0, 2**31 - 1)

    def needs_model(self) -> bool:
        """Wird das Bildmodell (diffusers) gebraucht?"""
        if self.mode in ("img2img", "inpaint"):
            return True
        return self.mode == "upscale" and self.refine

    def validated(self) -> list[str]:
        """Fehlende Angaben als Klartext. Leere Liste = alles da."""
        problems: list[str] = []
        if not self.sources:
            problems.append("Kein Ausgangsbild gewählt.")
        for source in self.sources:
            if not Path(source).is_file():
                problems.append(f"Datei nicht gefunden: {source}")
        if self.mode not in EDIT_MODES:
            problems.append(f"Unbekannter Modus: {self.mode}")
        if self.mode in ("img2img", "inpaint") and not self.prompt.strip():
            problems.append("Für das Umarbeiten wird ein Prompt gebraucht.")
        if self.mode == "inpaint":
            if self.mask is None:
                problems.append("Für 'Bereich ersetzen' fehlt die Maske.")
            elif not Path(self.mask).is_file():
                problems.append(f"Maske nicht gefunden: {self.mask}")
            elif len(self.sources) > 1:
                problems.append(
                    "Eine Maske passt nur zu einem Bild – bitte einzeln bearbeiten."
                )
        return problems


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

    def edit(self, request: "EditRequest", context: JobContext) -> ImageResult:
        """Bestehende Bilder umarbeiten. Vorgabe: nicht unterstützt."""
        raise RuntimeError(
            f"{type(self).__name__} kann bestehende Bilder nicht bearbeiten."
        )

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


_EDIT_TAGS = {"img2img": "bearbeitet", "inpaint": "ersetzt", "upscale": "gross"}


def edit_output_path(request: "EditRequest", source: Path, seed: int,
                     suffix: str | None = None) -> Path:
    """Zieldatei für ein bearbeitetes Bild.

    Der Name der Quelle bleibt sichtbar – bei zwanzig vergrößerten Bildern
    ist ein reiner Zeitstempel wertlos.
    """
    directory = request.output_dir or (paths.outputs_dir() / "images")
    paths.ensure_dir(directory)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    stem = request.name_hint or Path(source).stem
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in stem.lower())
    slug = "-".join(part for part in slug.split("-") if part)[:48] or "bild"
    tag = _EDIT_TAGS.get(request.mode, request.mode)
    ext = suffix or request.file_format or "png"
    candidate = directory / f"{stamp}_{slug}_{tag}_s{seed}.{ext}"
    # Mehrere Dateien in derselben Sekunde: durchnummerieren statt überschreiben.
    counter = 2
    while candidate.exists():
        candidate = directory / f"{stamp}_{slug}_{tag}_s{seed}_{counter:02d}.{ext}"
        counter += 1
    return candidate


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
@dataclass
class _PipelineBundle:
    """Alle Aufgaben eines Modells samt der Speicherentscheidung.

    ``pipes`` enthält text2img und – sobald gebraucht – die daraus
    abgeleiteten Pipelines für img2img und Inpainting. Sie teilen sich
    dieselben Gewichte, belegen also zusammen nur einmal Speicher.
    ``offload`` muss mitwandern: eine abgeleitete Pipeline darf die Frage
    „auslagern oder nicht" nicht neu beantworten, sonst widerspricht sie
    der Basis, die dieselben Module benutzt.
    """

    offload: bool = False
    pipes: dict[str, Any] = field(default_factory=dict)


# Ein geladenes SDXL belegt mehrere GB. Ohne Zwischenspeicher würde jeder
# Auftrag das Modell neu von der Platte holen – Minuten statt Sekunden.
_pipeline_cache: dict[tuple, _PipelineBundle] = {}


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
        self._offload = False
        self._bundle = _PipelineBundle()

    # --- Laden -------------------------------------------------------------
    def _cache_key(self) -> tuple:
        return (self.model.repo_id, self.plan.backend, self.plan.compute_type,
                self.config.cpu_offload)

    def _effective_plan(self, context: JobContext) -> BackendPlan:
        """Backend gegen die Wirklichkeit prüfen.

        Beim Start wird CUDA nur aus den Metadaten des torch-Wheels
        vermutet, damit das Fenster nicht auf den torch-Import wartet. Hier
        ist torch geladen: stimmt die Vermutung nicht, wird auf CPU
        umgestellt, statt beim ersten ``.to('cuda')`` abzustürzen.
        """
        if self.plan.backend != accel.Backend.CUDA:
            return self.plan
        ok, note = accel.torch_cuda_available()
        if ok:
            return self.plan
        context.log(
            f"CUDA ist doch nicht nutzbar – es wird auf der CPU gerechnet. {note}"
        )
        return replace(self.plan, backend=accel.Backend.CPU, compute_type="float32")

    def load(self, context: JobContext) -> None:
        key = self._cache_key()
        cached = _pipeline_cache.get(key)
        if cached is not None:
            self._bundle = cached
            self._pipe = cached.pipes["text2img"]
            self._offload = cached.offload
            self._family = _family_of(self._pipe)
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

        from diffusers import DiffusionPipeline

        self.plan = self._effective_plan(context)
        dtype = accel.torch_dtype(self.plan)
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": True,
            "local_files_only": True,
        }

        # Die Sperre gegen Darstellungen Minderjähriger bleibt davon
        # unberührt – die sitzt vor dem Laden (contentgate.enforce).
        safety_kwargs, safety_note = safety_checker_kwargs(self.config, Path(path))
        load_kwargs.update(safety_kwargs)
        if safety_kwargs or "bleibt aktiv" in safety_note:
            context.log(safety_note)

        if self.model.is_single_file:
            pipe = self._load_single_file(Path(path), load_kwargs, context)
            self._after_load(pipe, key, context)
            return

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

        self._after_load(pipe, key, context)

    def _load_single_file(self, directory: Path, load_kwargs: dict[str, Any],
                          context: JobContext) -> Any:
        """Einzeldatei-Checkpoint laden (eine .safetensors statt Ordner).

        Die meisten guten Feinabstimmungen werden so verteilt. diffusers
        kann das über ``from_single_file``, braucht dafür aber die
        Bauplan-Dateien (Konfigurationen der Bestandteile) eines
        Referenz-Repos – zusammen einige hundert KB, die einmalig geladen
        und danach im Hugging-Face-Cache liegen.
        """
        import diffusers

        target = directory / self.model.single_file
        if not target.is_file():
            treffer = sorted(directory.rglob("*.safetensors"))
            if not treffer:
                raise RuntimeError(
                    f"{self.model.title}: Datei {self.model.single_file} fehlt in "
                    f"{directory}."
                )
            target = treffer[0]

        cls = getattr(diffusers, self.model.single_file_class, None)
        if cls is None:
            raise RuntimeError(
                f"diffusers kennt keine Klasse {self.model.single_file_class}."
            )

        kwargs: dict[str, Any] = {
            "torch_dtype": load_kwargs.get("torch_dtype"),
            "local_files_only": True,
        }
        for name in ("safety_checker", "requires_safety_checker"):
            if name in load_kwargs:
                kwargs[name] = load_kwargs[name]

        if self.model.single_file_config:
            from .jobs import JobCancelled

            try:
                kwargs["config"] = str(models.ensure_reference_config(
                    self.model.single_file_config,
                    allow_download=self.config.allow_model_download,
                    on_status=context.status,
                    should_stop=context.should_stop,
                    offline=self.config.offline_mode,
                ))
            except models.DownloadCancelled as exc:
                raise JobCancelled(str(exc)) from exc

        context.status(f"Lade {target.name} ({target.stat().st_size / 1024**3:.1f} GB) …")
        try:
            return cls.from_single_file(str(target), **kwargs)
        except Exception as exc:  # noqa: BLE001 – verständlich melden
            raise RuntimeError(
                f"{self.model.title} konnte nicht geladen werden: {clean_error(exc)}"
            ) from exc

    def _after_load(self, pipe: Any, key: tuple, context: JobContext) -> None:
        """Gemeinsamer Abschluss beider Ladewege."""
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

        self._place(pipe, context, announce=True)

        self._bundle = _PipelineBundle(offload=self._offload, pipes={"text2img": pipe})
        if self.config.keep_model_loaded:
            _clear_pipeline_cache(keep=None)  # nur ein Modell gleichzeitig
            _pipeline_cache[key] = self._bundle
        self._pipe = pipe
        self._loaded = True
        context.status(f"{self.model.title} bereit ({self.plan.label}).")

    def _place(self, pipe: Any, context: JobContext, announce: bool = False,
               decide: bool = True) -> None:
        """Pipeline auf das Gerät legen und die Speicheroptionen setzen.

        ``decide=False`` übernimmt die Entscheidung der Basis-Pipeline. Das
        ist bei abgeleiteten Pipelines Pflicht: sie teilen sich dieselben
        Module, und der freie Grafikspeicher ist zu diesem Zeitpunkt bereits
        durch das geladene Modell belegt – neu gemessen käme immer
        „auslagern“ heraus, obwohl die Basis fest auf der GPU liegt.
        """
        import torch

        if decide:
            needed_mb = self.model.min_vram_mb or 6000
            free_mb = _free_vram_mb(self.plan.device_index)
            self._offload = self.config.cpu_offload or (
                self.plan.backend == accel.Backend.CUDA and free_mb and free_mb < needed_mb
            )
        else:
            free_mb = 0

        if self.plan.backend == accel.Backend.CUDA:
            if self._offload:
                if announce:
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

    def task_pipeline(self, task: str, context: JobContext) -> Any:
        """Pipeline für ``text2img`` | ``img2img`` | ``inpaint``.

        Die abgeleiteten Pipelines übernehmen die **selben** Gewichte
        (``from_pipe`` bzw. ``components``) – es wird also nichts erneut
        geladen und kein zusätzlicher Grafikspeicher belegt.
        """
        if not self._loaded:
            self.load(context)
        if task == "text2img":
            return self._pipe
        existing = self._bundle.pipes.get(task)
        if existing is not None:
            return existing

        import diffusers

        base_name = type(self._pipe).__name__
        target_name = _task_class_name(base_name, task)
        cls = getattr(diffusers, target_name, None)
        if cls is None:
            raise RuntimeError(
                f"{self.model.title} kann '{EDIT_MODE_LABELS.get(task, task)}' nicht: "
                f"diffusers kennt keine Klasse {target_name}. Wähle ein anderes Bildmodell."
            )

        context.status(f"Bereite '{EDIT_MODE_LABELS.get(task, task)}' vor …")
        errors: list[str] = []
        derived = None
        # Reihenfolge ist gemessen, nicht geraten: der Konstruktor mit
        # ``components`` reicht dieselben Modul-Objekte weiter und ist
        # sofort fertig. ``from_pipe`` legte im Test eine zweite Kopie der
        # Gewichte an – 280 s und zusätzlich 5 GB Grafikspeicher für
        # dasselbe Ergebnis. Es bleibt nur als Rückfallebene.
        for build in (lambda: cls(**self._pipe.components), lambda: cls.from_pipe(self._pipe)):
            try:
                derived = build()
                break
            except Exception as exc:  # noqa: BLE001 – nächsten Weg probieren
                errors.append(clean_error(exc))
        if derived is None:
            raise RuntimeError(
                f"{target_name} konnte nicht aufgebaut werden: {' | '.join(errors[-2:])}"
            )

        self._place(derived, context, decide=False)
        _try(derived, "set_progress_bar_config", disable=True)
        self._bundle.pipes[task] = derived
        return derived

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

        steps, guidance, negative, flux_notes = _flux_adjust(
            self._family, steps, request.guidance,
            _negative_with_protection(self.config, request.negative_prompt) or None,
        )
        notes.extend(flux_notes)

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
            # Der wirklich benutzte Negativ-Prompt kann länger sein als der
            # eingegebene (Schutzbegriffe). Ohne ihn ließe sich das Bild
            # nicht noch einmal genauso erzeugen.
            _save_image(image, target, request, image_seed, self.model.repo_id, steps, guidance,
                        extra={"negative_prompt_used": negative or ""})
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

    # --- Bearbeiten --------------------------------------------------------
    def diffuse(
        self,
        task: str,
        image: Any,
        prompt: str,
        context: JobContext,
        mask: Any = None,
        negative: str | None = None,
        strength: float = 0.45,
        steps: int = 25,
        guidance: float = 6.0,
        seed: int = 0,
        progress: Callable[[float, str], None] | None = None,
    ) -> tuple[Any, list[str]]:
        """Einen Durchlauf img2img/inpaint. Rückgabe: (Bild, Hinweise)."""
        import torch

        pipe = self.task_pipeline(task, context)
        steps, guidance, negative, notes = _flux_adjust(
            self._family, max(1, int(steps)), guidance,
            _negative_with_protection(self.config, negative) or None,
        )
        strength = max(0.01, min(1.0, float(strength)))
        # img2img rechnet nur den Bruchteil der Schritte, der zur Stärke
        # gehört – sonst stimmt die Fortschrittsanzeige nicht.
        expected = max(1, int(round(steps * strength)))

        def callback(_pipe, step_index, _timestep, callback_kwargs):
            context.raise_if_cancelled()
            if progress is not None:
                progress(min(1.0, (step_index + 1) / expected),
                         f"Schritt {min(step_index + 1, expected)}/{expected}")
            return callback_kwargs

        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": image,
            "strength": strength,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "generator": generator,
            "callback_on_step_end": callback,
        }
        if mask is not None:
            call_kwargs["mask_image"] = mask
        if negative is not None:
            call_kwargs["negative_prompt"] = negative
        # Inpainting-Pipelines nehmen ohne Größenangabe ihre eigene Vorgabe
        # (bei SDXL 1024) und skalieren die Vorlage darauf hoch. Aus einem
        # 512er Bild käme ein 1024er zurück. img2img kennt die beiden
        # Angaben nicht – deshalb wird die Signatur gefragt, nicht geraten.
        import inspect

        accepted = inspect.signature(pipe.__call__).parameters
        if "width" in accepted and "height" in accepted:
            call_kwargs["width"] = image.width
            call_kwargs["height"] = image.height
        if self._family == "flux":
            call_kwargs["max_sequence_length"] = 256

        try:
            with torch.inference_mode():
                output = pipe(**call_kwargs)
        except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
            _clear_pipeline_cache()
            raise RuntimeError(
                f"Grafikspeicher reicht für {image.width}x{image.height} nicht aus. "
                "Setze eine Höchstkante (z. B. 1024), verringere die Stärke oder "
                "schalte in den Einstellungen 'Modellteile auslagern' ein."
            ) from exc
        return output.images[0], notes

    def edit(self, request: EditRequest, context: JobContext) -> ImageResult:
        """Bestehende Bilder umarbeiten oder einen Bereich ersetzen."""
        if not self._loaded:
            self.load(context)

        started = time.time()
        seed = request.resolved_seed()
        task = "inpaint" if request.mode == "inpaint" else "img2img"
        notes: list[str] = []
        files: list[Path] = []
        sources = list(request.sources)
        last_size = (0, 0)

        for index, source in enumerate(sources):
            context.raise_if_cancelled()
            context.status(f"{Path(source).name} ({index + 1}/{len(sources)})")
            image, prepared_notes = _prepare_source(source, request.max_side)
            notes.extend(f"{Path(source).name}: {note}" for note in prepared_notes)
            mask = _prepare_mask(request.mask, image) if task == "inpaint" else None

            def progress(fraction: float, text: str, _i=index) -> None:
                context.progress((_i + fraction) / len(sources), text)

            image_seed = seed + index
            negative_used = _negative_with_protection(self.config, request.negative_prompt) or ""
            result, run_notes = self.diffuse(
                task, image, request.prompt, context,
                mask=mask,
                negative=negative_used,
                strength=request.strength,
                steps=request.steps,
                guidance=request.guidance,
                seed=image_seed,
                progress=progress,
            )
            notes.extend(run_notes)
            target = edit_output_path(request, Path(source), image_seed,
                                      suffix=request.file_format)
            _save_image(result, target, request, image_seed, self.model.repo_id,
                        request.steps, request.guidance,
                        extra={"source": str(source), "mode": request.mode,
                               "strength": f"{request.strength:.2f}",
                               "negative_prompt_used": negative_used})
            files.append(target)
            last_size = (result.width, result.height)
            context.log(f"geschrieben: {target}")

        return ImageResult(
            files=tuple(files),
            seed=seed,
            backend=self.plan.backend,
            model_key=self.model.key,
            elapsed_s=time.time() - started,
            width=last_size[0],
            height=last_size[1],
            steps=request.steps,
            dummy=False,
            notes=tuple(dict.fromkeys(notes)),
        )

    def unload(self) -> None:
        self._pipe = None
        self._bundle = _PipelineBundle()
        self._loaded = False
        _clear_pipeline_cache()


def _task_class_name(base_name: str, task: str) -> str:
    """Klassenname der abgeleiteten Pipeline aus dem der Basis ableiten.

    diffusers benennt durchgehend nach demselben Muster:
    ``StableDiffusionXLPipeline`` -> ``StableDiffusionXLImg2ImgPipeline``.
    Über die Namensregel statt über eine Tabelle, damit neue Modellfamilien
    ohne Anpassung funktionieren.
    """
    suffix = {"img2img": "Img2ImgPipeline", "inpaint": "InpaintPipeline"}[task]
    stem = base_name
    for known in ("Img2ImgPipeline", "InpaintPipeline", "Pipeline"):
        if stem.endswith(known):
            stem = stem[: -len(known)]
            break
    return stem + suffix


def _flux_adjust(family: str, steps: int, guidance: float,
                 negative: str | None) -> tuple[int, float, str | None, list[str]]:
    """FLUX [schnell] ist destilliert: 1–4 Schritte, keine Führung, kein Negativ."""
    notes: list[str] = []
    if family != "flux":
        return steps, guidance, negative, notes
    if guidance > 0:
        notes.append("FLUX [schnell] arbeitet ohne Führung – CFG auf 0 gesetzt.")
        guidance = 0.0
    if steps > 8:
        notes.append(f"FLUX [schnell] braucht 1–4 Schritte – {steps} auf 4 gekürzt.")
        steps = 4
    if negative:
        notes.append("FLUX kennt keinen Negativ-Prompt – Eingabe wird ignoriert.")
        negative = None
    return steps, guidance, negative, notes


def _prepare_source(source: Path, max_side: int) -> tuple[Any, list[str]]:
    """Ausgangsbild laden, begrenzen und auf ein Vielfaches von 8 bringen."""
    image = upscale.open_image(source).convert("RGB")
    notes: list[str] = []
    image, limited = upscale.fit_to_max_side(image, max_side)
    if limited:
        notes.append(f"auf Höchstkante {max_side} px verkleinert.")
    image, snapped = upscale.snap_to_multiple(image, 8)
    if snapped:
        notes.append(f"Größe auf {image.width}x{image.height} gerundet (Vielfaches von 8).")
    return image, notes


def _prepare_mask(mask_path: Path | None, image: Any) -> Any:
    """Maske laden: weiß = ersetzen, schwarz = behalten. Größe wird angepasst."""
    if mask_path is None:
        raise RuntimeError("Für 'Bereich ersetzen' wird eine Maske gebraucht.")
    mask = upscale.open_image(mask_path).convert("L")
    if (mask.width, mask.height) != (image.width, image.height):
        from PIL import Image as PilImage

        mask = mask.resize((image.width, image.height), PilImage.NEAREST)
    return mask


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


def _save_image(image, target: Path, request: ImageRequest | EditRequest, seed: int,
                repo_id: str, steps: int, guidance: float,
                extra: dict[str, str] | None = None) -> Path:
    """Bild speichern – mit Erzeugungsdaten in den Metadaten.

    ``request`` kann ein ``ImageRequest`` oder ein ``EditRequest`` sein;
    gelesen werden nur die Felder, die beide haben.
    """
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
    metadata.update(extra or {})
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

    def edit(self, request: EditRequest, context: JobContext) -> ImageResult:
        """Attrappe: legt eine Kopie an, damit Ablauf und Dateinamen prüfbar sind."""
        started = time.time()
        seed = request.resolved_seed()
        files: list[Path] = []
        notes = ["Kopie des Ausgangsbildes, kein echtes Modellergebnis."]
        if self.reason:
            notes.append(f"Grund: {self.reason}")
        last_size = (0, 0)

        for index, source in enumerate(request.sources):
            context.raise_if_cancelled()
            context.progress_steps(index + 1, max(1, len(request.sources)),
                                   Path(source).name)
            image = upscale.open_image(source).convert("RGB")
            if request.mode == "upscale" and request.factor > 1:
                image, method = upscale.upscale_image(image, factor=request.factor)
                notes.append(f"{Path(source).name}: {method}")
            target = edit_output_path(request, Path(source), seed + index,
                                      suffix=request.file_format)
            _save_image(image, target, request, seed + index, self.model.repo_id,
                        request.steps, request.guidance,
                        extra={"source": str(source), "mode": request.mode})
            files.append(target)
            last_size = (image.width, image.height)
            context.log(f"geschrieben: {target}")

        return ImageResult(
            files=tuple(files),
            seed=seed,
            backend=self.plan.backend,
            model_key=self.model.key,
            elapsed_s=time.time() - started,
            width=last_size[0],
            height=last_size[1],
            steps=request.steps,
            dummy=True,
            notes=tuple(notes),
        )


# ---------------------------------------------------------------------------
# Vergrößern
# ---------------------------------------------------------------------------
def _upscale_device(plan: BackendPlan) -> tuple[str, bool]:
    """Gerät und Genauigkeit für Real-ESRGAN. CUDA nur, wenn wirklich da."""
    if plan.backend == accel.Backend.CUDA:
        ok, _note = accel.torch_cuda_available()
        if ok:
            return f"cuda:{plan.device_index}", True
    return "cpu", False


def _ensure_upscale_weights(config: AppConfig, factor: int,
                            context: JobContext) -> tuple[Path | None, str]:
    """Gewichte für Real-ESRGAN bereitstellen. Ohne sie bleibt Lanczos."""
    from .jobs import JobCancelled

    key = config.upscale_model or models.DEFAULTS[models.Task.UPSCALE]
    try:
        spec = models.resolve(key)
        models.check_allowed(spec, allow_conditional=True)
    except Exception as exc:  # noqa: BLE001 – Lizenzsperre ist kein Absturz
        return None, f"Vergrößerungsmodell nicht nutzbar: {clean_error(exc)}"

    def on_progress(done: int, total: int) -> None:
        context.progress((done / total) if total else 0.0,
                         f"Download {done / (1024 ** 2):.0f} MB von {total / (1024 ** 2):.0f} MB")

    try:
        directory = models.ensure_local(
            spec,
            allow_download=config.allow_model_download,
            on_progress=on_progress,
            on_status=context.status,
            should_stop=context.should_stop,
            allow_conditional=True,
            offline=config.offline_mode,
            workers_hint=config.download_workers,
        )
    except models.DownloadCancelled as exc:
        raise JobCancelled(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 – ohne Modell wird Lanczos benutzt
        return None, f"{spec.title} nicht verfügbar: {clean_error(exc)}"

    found = upscale.weights_for(directory, factor)
    if found is None:
        return None, f"Keine Gewichtsdatei in {directory}."
    return found, ""


def run_upscale(config: AppConfig, plan: BackendPlan, request: EditRequest,
                context: JobContext, force_dummy: bool = False) -> ImageResult:
    """Bilder vergrößern, auf Wunsch anschließend mit dem Bildmodell nachschärfen."""
    from .jobs import JobCancelled

    started = time.time()
    seed = request.resolved_seed()
    notes: list[str] = []
    files: list[Path] = []
    sources = list(request.sources)

    weights: Path | None = None
    if request.use_model and not force_dummy:
        weights, note = _ensure_upscale_weights(config, request.factor, context)
        if note:
            notes.append(note + " Es wird Lanczos benutzt (weicher, aber sofort).")
    elif not request.use_model:
        notes.append("Modell abgewählt – es wird Lanczos benutzt.")

    device, half = _upscale_device(plan) if weights else ("cpu", False)
    refine = request.refine and bool(request.prompt.strip()) and not force_dummy
    pipeline: DiffusersImagePipeline | None = None
    if refine:
        candidate = create_image_pipeline(config, plan)
        if isinstance(candidate, DiffusersImagePipeline):
            pipeline = candidate
        else:
            refine = False
            notes.append(
                "Nachschärfen übersprungen: das Bildmodell steht nicht zur Verfügung."
            )
    elif request.refine and not request.prompt.strip():
        notes.append("Nachschärfen übersprungen: dafür wird ein Prompt gebraucht.")

    last_size = (0, 0)
    try:
        for index, source in enumerate(sources):
            context.raise_if_cancelled()
            context.status(f"{Path(source).name} ({index + 1}/{len(sources)})")
            image = upscale.open_image(source)
            image, limited = upscale.fit_to_max_side(image, request.max_side)
            if limited:
                notes.append(
                    f"{Path(source).name}: vorher auf Höchstkante {request.max_side} px gebracht."
                )

            # Fortschritt: erst vergrößern, dann (halb) nachschärfen.
            share = 0.5 if refine else 1.0

            def progress(fraction: float, text: str, _i=index, _s=share) -> None:
                context.progress((_i + fraction * _s) / len(sources), text)

            try:
                result, method = upscale.upscale_image(
                    image,
                    factor=request.factor,
                    weights=weights,
                    device=device,
                    half=half,
                    tile=request.tile,
                    on_progress=progress,
                    should_stop=context.should_stop,
                )
            except upscale.UpscaleCancelled as exc:
                raise JobCancelled(str(exc)) from exc
            notes.append(f"{Path(source).name}: {method} → {result.width}x{result.height}")

            if refine and pipeline is not None:
                snapped, _snap_note = upscale.snap_to_multiple(result.convert("RGB"), 8)

                def refine_progress(fraction: float, text: str, _i=index) -> None:
                    context.progress((_i + 0.5 + fraction * 0.5) / len(sources), text)

                refined, run_notes = pipeline.diffuse(
                    "img2img", snapped, request.prompt, context,
                    negative=request.negative_prompt,
                    strength=request.refine_strength,
                    steps=request.steps,
                    guidance=request.guidance,
                    seed=seed + index,
                    progress=refine_progress,
                )
                notes.extend(run_notes)
                result = refined

            target = edit_output_path(request, Path(source), seed + index,
                                      suffix=request.file_format)
            _save_image(result, target, request, seed + index,
                        config.upscale_model or "lanczos", request.steps, request.guidance,
                        extra={"source": str(source), "mode": "upscale",
                               "factor": str(request.factor)})
            files.append(target)
            last_size = (result.width, result.height)
            context.log(f"geschrieben: {target}")
    finally:
        if not config.keep_model_loaded:
            upscale.unload()
            if pipeline is not None:
                pipeline.unload()

    return ImageResult(
        files=tuple(files),
        seed=seed,
        backend=plan.backend,
        model_key=config.upscale_model or "lanczos",
        elapsed_s=time.time() - started,
        width=last_size[0],
        height=last_size[1],
        steps=request.steps,
        dummy=force_dummy,
        notes=tuple(dict.fromkeys(notes)),
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
        # Vor dem Laden prüfen: mehrere GB Modell zu holen und danach
        # abzulehnen wäre die schlechteste Reihenfolge.
        contentgate.enforce(request.prompt, request.negative_prompt)
        pipeline = create_image_pipeline(config, plan, force_dummy=force_dummy)
        try:
            return pipeline.generate(request, context)
        finally:
            if not config.keep_model_loaded:
                pipeline.unload()

    return handler


def make_edit_job(config: AppConfig, plan: BackendPlan, request: EditRequest,
                  force_dummy: bool = False):
    """Handler für das Bearbeiten eines bestehenden Bildes.

    Deckt alle drei Modi ab. Fehlende Angaben werden vorab geprüft und als
    eine verständliche Meldung geworfen, nicht als Fehler aus PIL.
    """

    def handler(context: JobContext) -> ImageResult:
        problems = request.validated()
        if problems:
            raise RuntimeError(" ".join(problems))
        contentgate.enforce(request.prompt, request.negative_prompt)

        if request.mode == "upscale":
            return run_upscale(config, plan, request, context, force_dummy=force_dummy)

        pipeline = create_image_pipeline(config, plan, force_dummy=force_dummy)
        try:
            return pipeline.edit(request, context)
        finally:
            if not config.keep_model_loaded:
                pipeline.unload()

    return handler
