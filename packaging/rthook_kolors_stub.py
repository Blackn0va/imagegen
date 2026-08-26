"""PyInstaller-Laufzeithaken: Stub für diffusers.pipelines.kolors.

Warum das nötig ist:
- diffusers.pipelines.auto_pipeline importiert ALLE Pipelines beim
  Modulimport – auch Kolors (``from .kolors import KolorsPipeline, ...``).
- Kolors hängt an ``@torch.jit.script`` (text_encoder.py), und TorchScript
  braucht Python-Quelltext, den ein eingefrorenes Bundle nicht mitliefert.
  Deshalb ist das Modul in app.spec bewusst exkludiert.
- optimum (OpenVINO-/ONNX-Export) importiert auto_pipeline beim Start;
  scheitert der Kolors-Import, bleibt das Task-Mapping leer und der Export
  endet mit ``KeyError: 'text-to-image'``.

Der Haken läuft vor dem eigentlichen Programm und legt einen leichten
Ersatz in sys.modules. Dadurch bauen sich die Auto-Mappings korrekt auf,
ohne dass der echte Kolors-Code (und TorchScript) geladen wird. Wer die
Platzhalter dennoch instanziiert, bekommt eine klare Fehlermeldung –
die Anwendung selbst nutzt Kolors nicht.
"""

import sys
import types


class _KolorsUnavailable:
    """Platzhalter für Kolors-Pipelineklassen (im Bundle nicht enthalten)."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "Kolors-Pipelines sind in diesem Paket nicht enthalten "
            "(TorchScript benötigt Quelltext, den ein eingefrorenes "
            "Programm nicht mitliefert)."
        )


def _install_stub() -> None:
    base = "diffusers.pipelines.kolors"
    if base in sys.modules:
        return

    package = types.ModuleType(base)
    package.__path__ = []  # als Paket kennzeichnen, sonst schlagen Unterimporte fehl
    package.KolorsPipeline = _KolorsUnavailable
    package.KolorsImg2ImgPipeline = _KolorsUnavailable
    package.KolorsPAGPipeline = _KolorsUnavailable

    pipeline_output = types.ModuleType(base + ".pipeline_output")
    pipeline_output.KolorsPipelineOutput = _KolorsUnavailable

    text_encoder = types.ModuleType(base + ".text_encoder")
    text_encoder.ChatGLMModel = _KolorsUnavailable

    tokenizer = types.ModuleType(base + ".tokenizer")
    tokenizer.ChatGLMTokenizer = _KolorsUnavailable

    # Die PAG-Variante wird von diffusers.pipelines.pag direkt importiert
    # (from .pipeline_pag_kolors import KolorsPAGPipeline) – ohne diesen
    # Stub scheitert der Import von auto_pipeline trotzdem.
    pag_kolors = types.ModuleType("diffusers.pipelines.pag.pipeline_pag_kolors")
    pag_kolors.KolorsPAGPipeline = _KolorsUnavailable

    sys.modules[base] = package
    sys.modules[base + ".pipeline_output"] = pipeline_output
    sys.modules[base + ".text_encoder"] = text_encoder
    sys.modules[base + ".tokenizer"] = tokenizer
    sys.modules["diffusers.pipelines.pag.pipeline_pag_kolors"] = pag_kolors


_install_stub()
