"""Bildmodelle über ONNX Runtime (DirectML) und OpenVINO (Intel-GPU/NPU).

Der torch-Pfad in ``pipeline_image`` deckt NVIDIA und CPU ab. Alles
andere – AMD-Karten, Intel-iGPUs, Intel-NPUs – wird von torch unter
Windows nicht bedient. Dafür gibt es zwei Laufzeiten, und beide brauchen
**andere Gewichte** als diffusers: das Modell muss einmalig exportiert
werden.

Exportiert und ausgeführt wird über ``optimum`` von Hugging Face, nicht
von Hand. Ein selbstgebauter Export von UNet, VAE und Textencodern müsste
jede Modellfamilie einzeln nachbilden und bei jeder diffusers-Fassung
nachgezogen werden – das ist genau die Arbeit, die optimum bereits leistet
und pflegt.

Zwei Laufzeiten, ein Ablauf:

  * ``dml``      → ``optimum.onnxruntime`` mit ``DmlExecutionProvider``
  * ``openvino`` → ``optimum.intel`` mit Gerät ``NPU``, ``GPU`` oder ``CPU``

Beide sind **optionale** Abhängigkeiten. Fehlen sie, wird das im Klartext
gesagt und das Backend gilt als nicht lauffähig – niemals wird
stillschweigend auf die CPU ausgewichen und dabei etwas anderes angezeigt.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from . import models, paths
from .accel import Backend, BackendPlan, clean_error
from .config import AppConfig
from .jobs import JobContext
from .pipeline_image import (
    EditRequest,
    ImagePipeline,
    ImageRequest,
    ImageResult,
    _negative_with_protection,
    _prepare_mask,
    _prepare_source,
    _save_image,
    _snap,
    edit_output_path,
    output_path,
    release_memory,
)

log = logging.getLogger(__name__)

# Welche Modellfamilie über welche Klasse läuft. Die Namen folgen dem
# Schema von optimum; gesucht wird zur Laufzeit über getattr, damit eine
# Umbenennung in einer neuen Fassung eine verständliche Meldung ergibt
# statt eines ImportError beim Start.
_CLASS_NAMES = {
    Backend.DML: {
        "sdxl": ("ORTStableDiffusionXLPipeline", "ORTStableDiffusionXLImg2ImgPipeline", None),
        "sd": (
            "ORTStableDiffusionPipeline",
            "ORTStableDiffusionImg2ImgPipeline",
            "ORTStableDiffusionInpaintPipeline",
        ),
    },
    Backend.OPENVINO: {
        "sdxl": ("OVStableDiffusionXLPipeline", "OVStableDiffusionXLImg2ImgPipeline", None),
        "sd": (
            "OVStableDiffusionPipeline",
            "OVStableDiffusionImg2ImgPipeline",
            "OVStableDiffusionInpaintPipeline",
        ),
    },
}

_RUNTIME_MODULES = {
    Backend.DML: "optimum.onnxruntime",
    Backend.OPENVINO: "optimum.intel",
}

_INSTALL_HINT = {
    Backend.DML: "pip install optimum[onnxruntime] onnxruntime-directml",
    Backend.OPENVINO: "pip install optimum[openvino] openvino",
}

# Familien, die diese Laufzeiten nicht abdecken. Lieber vorher ablehnen als
# nach zwanzig Minuten Export mit einem Fehler aus optimum enden.
UNSUPPORTED_FAMILIES = {"flux"}


class ExportUnavailable(RuntimeError):
    """Laufzeit fehlt oder Modell passt nicht – mit Anleitung im Text."""

    expected = True


# ---------------------------------------------------------------------------
# Verfügbarkeit
# ---------------------------------------------------------------------------
def _nachruesten(backend: str) -> str:
    """Wie die Laufzeit nachzurüsten ist – je nach Betriebsart verschieden.

    Im eingefrorenen Bundle ist ``pip install`` der **falsche** Rat: die
    Anwendung bringt ihren eigenen Python mit und sieht nichts, was
    hinterher in ein System-Python gelegt wird. Dort hilft nur ein neuer
    Bau mit der Laufzeit an Bord.
    """
    import sys

    if getattr(sys, "frozen", False):
        return (
            "Dies ist ein gebautes Programm mit eigenem Python – ein "
            "'pip install' im System-Python wirkt hier nicht. Die Laufzeit "
            "muss beim Bauen dabei sein: "
            ".\\build-windows.ps1 -Clean -WithOnnx $true"
        )
    return f"Nachrüsten: {_INSTALL_HINT[backend]}"


# Ergebnis je Backend merken. Die Frage wird beim Start mehrfach gestellt
# (Backend-Kette, Modelltabelle, Diagnose) und ändert sich zur Laufzeit nicht.
_runtime_cache: dict[str, tuple[bool, str]] = {}


def runtime_available(backend: str, deep: bool = False) -> tuple[bool, str]:
    """Ist die Laufzeit für dieses Backend benutzbar?

    Vorgabe ist die **billige** Auskunft: es wird nur nachgesehen, ob das
    Modul auffindbar ist. Der frühere Vollimport kostete beim Start
    gemessene 3,8 s – ``optimum.onnxruntime`` zieht transformers und torch
    mit, und gefragt wird das, bevor das Fenster überhaupt steht.

    ``deep=True`` lädt wirklich. Das gehört an die Stelle, wo gerechnet
    wird: dort fällt die Zeit neben dem Modell nicht auf, und ein
    kaputtes Paket soll dann auffliegen.
    """
    import importlib.util

    module = _RUNTIME_MODULES.get(backend)
    if module is None:
        return False, f"Für '{backend}' gibt es keinen ONNX-Weg."

    schluessel = f"{backend}:{'tief' if deep else 'flach'}"
    gemerkt = _runtime_cache.get(schluessel)
    if gemerkt is not None:
        return gemerkt

    ergebnis: tuple[bool, str]
    try:
        if importlib.util.find_spec("optimum") is None:
            ergebnis = (False, f"optimum fehlt. {_nachruesten(backend)}")
        elif importlib.util.find_spec(module) is None:
            ergebnis = (False, f"{module} fehlt. {_nachruesten(backend)}")
        elif not deep:
            ergebnis = (True, f"{module} vorhanden.")
        else:
            importlib.import_module(module)
            ergebnis = (True, f"{module} geladen.")
    except Exception as exc:
        # find_spec wirft bei kaputten Paketen ebenfalls.
        ergebnis = (
            False,
            f"{module} nicht ladbar ({clean_error(exc)}). {_nachruesten(backend)}",
        )

    _runtime_cache[schluessel] = ergebnis
    return ergebnis


def forget_runtime_cache() -> None:
    """Merker verwerfen – nach einer Nachinstallation zur Laufzeit."""
    _runtime_cache.clear()


def family_of(spec: models.ModelSpec) -> str:
    """Modellfamilie aus dem Bezeichner ableiten (sdxl, sd, flux)."""
    text = f"{spec.key} {spec.repo_id} {spec.title}".lower()
    if "flux" in text:
        return "flux"
    if "xl" in text or "sdxl" in text:
        return "sdxl"
    return "sd"


def supported(spec: models.ModelSpec, backend: str) -> tuple[bool, str]:
    """Kann dieses Modell auf diesem Backend laufen?"""
    family = family_of(spec)
    if family in UNSUPPORTED_FAMILIES:
        return False, (
            f"{spec.title} gehört zur Familie '{family}'. Dafür gibt es in "
            "optimum keine ONNX-/OpenVINO-Pipeline. Wähle ein SD- oder "
            "SDXL-Modell, oder rechne dieses Modell auf CPU bzw. CUDA."
        )
    if backend not in _CLASS_NAMES:
        return False, f"Backend '{backend}' kennt keinen ONNX-Weg."
    return True, ""


def _pipeline_class(backend: str, family: str, task: str) -> Any:
    """Klasse aus optimum holen. Fehlt sie, verständlich abbrechen."""
    import importlib

    slot = {"text2img": 0, "img2img": 1, "inpaint": 2}[task]
    name = _CLASS_NAMES[backend][family][slot]
    if name is None:
        raise ExportUnavailable(
            f"'{task}' ist für diese Modellfamilie über {backend} nicht "
            "umgesetzt. Für diesen Schritt auf CPU oder CUDA wechseln."
        )
    module = importlib.import_module(_RUNTIME_MODULES[backend])
    cls = getattr(module, name, None)
    if cls is None:
        raise ExportUnavailable(
            f"{_RUNTIME_MODULES[backend]} kennt '{name}' nicht – die "
            "installierte optimum-Fassung passt nicht. Nachrüsten: "
            f"{_INSTALL_HINT[backend]}"
        )
    return cls


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def is_exported(spec: models.ModelSpec, backend: str) -> bool:
    """Liegt ein fertiges Konvertat vor?"""
    return models.is_converted(spec, backend)


def export(
    config: AppConfig,
    spec: models.ModelSpec,
    backend: str,
    context: JobContext,
) -> Path:
    """Modell einmalig nach ONNX bzw. OpenVINO exportieren.

    Läuft als Auftrag, ist also abbrechbar und meldet Fortschritt. Der
    Export dauert Minuten und legt mehrere GB an – deshalb wird er nie
    beiläufig angestoßen, sondern immer ausdrücklich verlangt.
    """
    from .jobs import JobCancelled

    # Hier wirklich laden: der Export dauert ohnehin Minuten, und ein
    # kaputtes Paket soll jetzt auffliegen, nicht mitten im Lauf.
    ok, reason = runtime_available(backend, deep=True)
    if not ok:
        raise ExportUnavailable(reason)
    can, why = supported(spec, backend)
    if not can:
        raise ExportUnavailable(why)

    target = models.converted_dir(spec, backend)
    if is_exported(spec, backend):
        context.status(f"{spec.title} liegt für {backend} bereits konvertiert vor.")
        return target

    # Quelle muss lokal liegen: exportiert wird von der Platte, nicht
    # nebenbei aus dem Netz.
    context.status("Stelle das Ausgangsmodell bereit …")
    source = models.ensure_local(
        spec,
        allow_download=config.allow_model_download,
        on_status=context.status,
        should_stop=context.should_stop,
        allow_conditional=True,
        offline=config.offline_mode,
    )

    context.raise_if_cancelled()
    free_ok, free_note = models.check_disk_space(target, spec.approx_size_mb * 1024 * 1024)
    if not free_ok:
        raise RuntimeError(free_note)

    cls = _pipeline_class(backend, family_of(spec), "text2img")
    context.status(
        f"Exportiere {spec.title} nach {backend}. Das dauert mehrere Minuten und läuft nur einmal."
    )
    context.progress(0.05, "Modell laden")

    started = time.time()
    kwargs: dict[str, Any] = {"export": True}
    if backend == Backend.OPENVINO:
        # Beim Export noch kein Gerät festlegen – die Gewichte sind
        # geräteunabhängig, gewählt wird erst beim Laden.
        kwargs["compile"] = False
    if spec.variant:
        # Der lokale Snapshot enthält nur die gewünschte Variante (z. B.
        # fp16, siehe models.select_files). Ohne diesen Hinweis sucht
        # diffusers nach der Vollfassung und scheitert.
        kwargs["variant"] = spec.variant

    try:
        pipe = cls.from_pretrained(str(source), **kwargs)
        context.raise_if_cancelled()
        context.progress(0.7, "Konvertat schreiben")
        paths.ensure_dir(target)
        pipe.save_pretrained(str(target))
    except JobCancelled:
        raise
    except Exception as exc:
        # Halbe Konvertate sind wertlos und würden beim nächsten Start als
        # „fertig" gelten – deshalb weg damit.
        _discard(target)
        raise RuntimeError(
            f"Export von {spec.title} nach {backend} fehlgeschlagen: {clean_error(exc)}"
        ) from exc
    finally:
        release_memory(deep=True)

    if not models.is_converted(spec, backend):
        _discard(target)
        raise RuntimeError(
            f"Der Export nach {target} hat keine brauchbaren Gewichte "
            "hinterlassen. Das Konvertat wurde verworfen."
        )

    context.progress(1.0, "fertig")
    context.status(f"{spec.title} für {backend} bereit ({time.time() - started:.0f} s).")
    return target


def _discard(target: Path) -> None:
    """Unvollständiges Konvertat entfernen – nie werfend."""
    import shutil

    try:
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
    except OSError as exc:  # pragma: no cover – Aufräumen darf nie stören
        log.debug("Konvertat nicht entfernbar: %s", exc)


def make_export_job(config: AppConfig, spec: models.ModelSpec, backend: str):
    """Handler für die Warteschlange."""

    def handler(context: JobContext) -> Path:
        return export(config, spec, backend, context)

    return handler


# ---------------------------------------------------------------------------
# Ausführung
# ---------------------------------------------------------------------------
class OnnxImagePipeline(ImagePipeline):
    """Bildmodell über ONNX Runtime bzw. OpenVINO.

    Hält dieselbe Schnittstelle wie ``DiffusersImagePipeline``, damit der
    Rest der Anwendung nichts vom Unterschied wissen muss.
    """

    def __init__(self, config: AppConfig, plan: BackendPlan) -> None:
        super().__init__(config, plan)
        self._pipes: dict[str, Any] = {}
        self._family = family_of(self.model)
        self._device = ""

    # --- Laden --------------------------------------------------------
    def load(self, context: JobContext) -> None:
        ok, reason = runtime_available(self.plan.backend)
        if not ok:
            raise ExportUnavailable(reason)
        can, why = supported(self.model, self.plan.backend)
        if not can:
            raise ExportUnavailable(why)
        if not is_exported(self.model, self.plan.backend):
            raise ExportUnavailable(
                f"{self.model.title} ist für {self.plan.backend} nicht "
                "konvertiert. Einmalig ausführen: "
                f"'streamforge models convert {self.model.key} "
                f"--backend {self.plan.backend}'."
            )
        self._pipes.clear()
        self._get(context, "text2img")
        self._loaded = True

    def _get(self, context: JobContext, task: str) -> Any:
        """Pipeline je Aufgabe laden und behalten."""
        existing = self._pipes.get(task)
        if existing is not None:
            return existing

        cls = _pipeline_class(self.plan.backend, self._family, task)
        directory = str(models.converted_dir(self.model, self.plan.backend))
        kwargs: dict[str, Any] = {}

        if self.plan.backend == Backend.DML:
            kwargs["provider"] = "DmlExecutionProvider"
        else:
            from .accel import openvino_target

            device, note = openvino_target(getattr(self.config, "openvino_device", ""))
            if not device:
                raise ExportUnavailable(note)
            self._device = device
            kwargs["device"] = device
            context.status(f"OpenVINO rechnet auf {device}. {note}")

        context.status(f"Lade {self.model.title} ({self.plan.backend}) …")
        try:
            pipe = cls.from_pretrained(directory, **kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"{self.model.title} ließ sich für {self.plan.backend} nicht "
                f"laden: {clean_error(exc)}"
            ) from exc
        self._pipes[task] = pipe
        return pipe

    # --- Erzeugen -----------------------------------------------------
    def _call(
        self,
        pipe: Any,
        context: JobContext,
        call_kwargs: dict[str, Any],
        steps: int,
        progress,
    ) -> Any:
        """Pipeline aufrufen und dabei Fortschritt und Abbruch bedienen.

        optimum-Pipelines kennen je nach Fassung ``callback_on_step_end``
        oder das ältere ``callback``/``callback_steps``. Welche, wird an
        der Signatur abgelesen statt geraten – sonst bricht der Aufruf mit
        einem unverständlichen TypeError ab.
        """
        import inspect

        accepted = inspect.signature(pipe.__call__).parameters

        if "callback_on_step_end" in accepted:

            def modern(_pipe, step_index, _timestep, kwargs):
                context.raise_if_cancelled()
                if progress is not None:
                    progress(
                        min(1.0, (step_index + 1) / max(1, steps)), f"Schritt {step_index + 1}"
                    )
                return kwargs

            call_kwargs["callback_on_step_end"] = modern
        elif "callback" in accepted:

            def legacy(step_index, _timestep, _latents):
                context.raise_if_cancelled()
                if progress is not None:
                    progress(
                        min(1.0, (step_index + 1) / max(1, steps)), f"Schritt {step_index + 1}"
                    )

            call_kwargs["callback"] = legacy
            if "callback_steps" in accepted:
                call_kwargs["callback_steps"] = 1

        for name in list(call_kwargs):
            if name not in accepted and name not in ("prompt",):
                # Unbekannte Angaben still fallen lassen wäre schlecht, aber
                # ein harter Fehler wegen einer Feinheit auch. Der Hinweis
                # landet im Log, der Auftrag läuft weiter.
                log.debug("%s kennt '%s' nicht – wird weggelassen.", type(pipe).__name__, name)
                call_kwargs.pop(name)
        return pipe(**call_kwargs)

    def generate(self, request: ImageRequest, context: JobContext) -> ImageResult:
        if not self._loaded:
            self.load(context)
        started = time.time()
        seed = request.resolved_seed()
        width, height = _snap(request.width), _snap(request.height)
        steps = max(1, request.steps)
        batch = max(1, request.batch)
        notes: list[str] = [f"Gerechnet über {self.plan.label}."]
        if self._device:
            notes.append(f"OpenVINO-Gerät: {self._device}.")
        files: list[Path] = []

        pipe = self._get(context, "text2img")
        negative = _negative_with_protection(self.config, request.negative_prompt) or None

        for index in range(batch):
            context.raise_if_cancelled()
            image_seed = seed + index

            def progress(fraction: float, text: str, _i=index) -> None:
                context.progress((_i + fraction) / batch, text)

            call_kwargs: dict[str, Any] = {
                "prompt": request.prompt,
                "width": width,
                "height": height,
                "num_inference_steps": steps,
                "guidance_scale": request.guidance,
                "generator": _generator(image_seed),
            }
            if negative is not None:
                call_kwargs["negative_prompt"] = negative

            context.status(f"Bild {index + 1}/{batch} wird gerechnet …")
            output = self._call(pipe, context, call_kwargs, steps, progress)
            image = output.images[0]
            target = output_path(request, index, image_seed, suffix=request.file_format)
            _save_image(
                image,
                target,
                request,
                image_seed,
                self.model.repo_id,
                steps,
                request.guidance,
                extra={
                    "backend": self.plan.backend,
                    "device": self._device,
                    "negative_prompt_used": negative or "",
                },
            )
            files.append(target)
            context.log(f"geschrieben: {target}")
            del image, output
            if index + 1 < batch:
                release_memory()

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

    def edit(self, request: EditRequest, context: JobContext) -> ImageResult:
        if not self._loaded:
            self.load(context)
        started = time.time()
        seed = request.resolved_seed()
        task = "inpaint" if request.mode == "inpaint" else "img2img"
        pipe = self._get(context, task)

        notes: list[str] = [f"Gerechnet über {self.plan.label}."]
        files: list[Path] = []
        sources = list(request.sources)
        last_size = (0, 0)
        colorize = request.mode == "colorize"
        prompt_used = request.effective_prompt()
        strength_used = request.effective_strength()

        for index, source in enumerate(sources):
            context.raise_if_cancelled()
            context.status(f"{Path(source).name} ({index + 1}/{len(sources)})")
            image, prepared = _prepare_source(source, request.max_side)
            notes.extend(f"{Path(source).name}: {note}" for note in prepared)
            if colorize:
                from .pipeline_image import is_grayscale, merge_luminance

                if not is_grayscale(image):
                    notes.append(f"{Path(source).name}: Vorlage ist nicht schwarz-weiß.")
                image = image.convert("L").convert("RGB")

            def progress(fraction: float, text: str, _i=index) -> None:
                context.progress((_i + fraction) / len(sources), text)

            image_seed = seed + index
            negative = _negative_with_protection(self.config, request.effective_negative()) or ""
            call_kwargs: dict[str, Any] = {
                "prompt": prompt_used,
                "image": image,
                "strength": strength_used,
                "num_inference_steps": request.steps,
                "guidance_scale": request.guidance,
                "generator": _generator(image_seed),
                "negative_prompt": negative,
            }
            if task == "inpaint":
                call_kwargs["mask_image"] = _prepare_mask(request.mask, image)

            output = self._call(
                pipe, context, call_kwargs, max(1, round(request.steps * strength_used)), progress
            )
            result = output.images[0]
            if colorize and request.keep_luminance:
                from .pipeline_image import merge_luminance

                result = merge_luminance(image, result)

            target = edit_output_path(request, Path(source), image_seed, suffix=request.file_format)
            _save_image(
                result,
                target,
                request,
                image_seed,
                self.model.repo_id,
                request.steps,
                request.guidance,
                extra={
                    "source": str(source),
                    "mode": request.mode,
                    "backend": self.plan.backend,
                    "device": self._device,
                    "negative_prompt_used": negative,
                },
            )
            files.append(target)
            last_size = (result.width, result.height)
            context.log(f"geschrieben: {target}")
            del result, image, output
            if index + 1 < len(sources):
                release_memory()

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
        self._pipes.clear()
        self._loaded = False
        release_memory(deep=True)

    def free_between_jobs(self) -> None:
        release_memory(deep=True)


def _generator(seed: int) -> Any:
    """Zufallsquelle. numpy, weil die Laufzeiten kein torch voraussetzen."""
    try:
        import numpy as np

        return np.random.RandomState(int(seed) % (2**32))
    except ImportError:  # pragma: no cover – numpy ist Pflichtabhängigkeit
        return None


def describe() -> str:
    """Zustandsbericht für Diagnose und Oberfläche."""
    import sys

    lines: list[str] = []
    if getattr(sys, "frozen", False):
        lines.append(
            "Betriebsart: gebautes Programm mit eigenem Python. Laufzeiten "
            "müssen beim Bauen dabei sein – 'pip install' wirkt hier nicht."
        )
    for backend in (Backend.DML, Backend.OPENVINO):
        ok, reason = runtime_available(backend)
        lines.append(f"{backend}: {'verfügbar' if ok else 'nicht verfügbar'} – {reason}")
    return "\n".join(lines)
