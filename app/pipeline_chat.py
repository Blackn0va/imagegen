"""Chat und Code-Writer über llama.cpp (GGUF).

Warum llama.cpp und nicht OpenVINO oder ONNX: für Sprachmodelle auf
Intel-CPUs ist llama.cpp in Messungen rund doppelt so schnell, und die
NPU ist für diese Last der falsche Baustein – sie ist auf kleine,
quantisierte Faltungsnetze ausgelegt, nicht auf autoregressive Textgabe.
Entscheidend ist deshalb die Modellgröße: 3B in Q4_K_M liefern auf einem
Rechner ohne Grafikkarte 20+ Token/s, 7B fallen auf etwa 4.

Bilder: Vision-Modelle bringen neben den Gewichten eine zweite Datei mit
(``mmproj``), die den Bildteil enthält. Fehlt sie, kann das Modell ein
eingefügtes Bild nicht lesen – dann wird das gesagt, statt das Bild
stillschweigend zu verwerfen.

Die Laufzeit ist eine **optionale** Abhängigkeit. Fehlt sie, meldet der
Chat das im Klartext und bleibt abgeschaltet.
"""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import models
from .accel import clean_error
from .config import AppConfig

log = logging.getLogger(__name__)

INSTALL_HINT = "pip install llama-cpp-python"
BUILD_HINT = ".\\build-windows.ps1 -Clean -WithChat $true"

# Bilder werden vor dem Anhängen begrenzt. Ein 12-Megapixel-Foto bläht den
# Kontext auf und bringt gegenüber der kurzen Kante nichts – die Bildteile
# dieser Modelle arbeiten ohnehin mit Kacheln um 448 bis 1024 Pixel.
MAX_IMAGE_SIDE = 1024

# Vorlagen für den Bildteil, in der Reihenfolge, in der gesucht wird. Die
# Namen wandern zwischen llama-cpp-python-Fassungen; gesucht wird deshalb
# über getattr, damit eine Umbenennung eine lesbare Meldung ergibt statt
# eines ImportError.
_VISION_HANDLERS = (
    "Qwen25VLChatHandler",
    "Qwen2VLChatHandler",
    "MiniCPMv26ChatHandler",
    "Llava16ChatHandler",
    "Llava15ChatHandler",
)

# Der Zusatz zu den Code-Blöcken muss die Einschränkung „nur wenn wirklich
# Code kommt" enthalten. Ohne sie packen kleine Modelle auch eine reine
# Textantwort in einen ```python-Block – gemessen an Qwen2.5-VL 3B.
SYSTEM_PROMPT = (
    "Du bist ein hilfsbereiter Assistent für Programmierung und allgemeine "
    "Fragen. Antworte knapp und genau, auf Deutsch. "
    "Wenn deine Antwort Quelltext enthält, setze NUR den Quelltext in einen "
    "Markdown-Block mit Sprachangabe (```python). Fließtext, Aufzählungen "
    "und Erklärungen stehen außerhalb solcher Blöcke. "
    "Erfinde nichts – sage, wenn du etwas nicht weißt."
)


class ChatUnavailable(RuntimeError):
    """Laufzeit oder Modell fehlt – Text enthält die Anleitung."""

    expected = True


# ---------------------------------------------------------------------------
# Verfügbarkeit
# ---------------------------------------------------------------------------
def _nachruesten() -> str:
    """Wie nachzurüsten ist – im Bundle hilft pip nicht."""
    import sys

    if getattr(sys, "frozen", False):
        return (
            "Dies ist ein gebautes Programm mit eigenem Python – ein "
            f"'pip install' wirkt hier nicht. Neu bauen mit: {BUILD_HINT}"
        )
    return f"Nachrüsten: {INSTALL_HINT}"


def runtime_available() -> tuple[bool, str]:
    """Ist llama.cpp benutzbar?"""
    import importlib.util

    if importlib.util.find_spec("llama_cpp") is None:
        return False, f"llama-cpp-python fehlt. {_nachruesten()}"
    try:
        import llama_cpp
    except Exception as exc:
        return False, f"llama_cpp nicht ladbar ({clean_error(exc)}). {_nachruesten()}"
    return True, f"llama.cpp {getattr(llama_cpp, '__version__', 'unbekannt')} vorhanden."


def available_models() -> list[models.ModelSpec]:
    """Alle eingetragenen Chat-Modelle."""
    return [spec for spec in models.REGISTRY.values() if spec.task is models.Task.CHAT]


def describe() -> str:
    _ok, reason = runtime_available()
    lines = [reason]
    for spec in available_models():
        geladen = "geladen" if models.is_downloaded(_weights_spec(spec)) else "nicht geladen"
        sieht = "sieht Bilder" if spec.sees_images else "nur Text"
        lines.append(f"  {spec.key:<18} {geladen:<13} {sieht}  {spec.title}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gewichte bereitstellen
# ---------------------------------------------------------------------------
def _weights_spec(spec: models.ModelSpec) -> models.ModelSpec:
    """Spec auf die zwei GGUF-Dateien eingrenzen.

    Die Repos enthalten oft ein Dutzend Quantisierungen – ohne diesen
    Filter lädt man zehnmal dasselbe Modell in verschiedenen Güten.
    """
    wanted = tuple(name for name in (spec.gguf_file, spec.mmproj_file) if name)
    return replace(spec, allow_patterns=wanted) if wanted else spec


def ensure_weights(config: AppConfig, spec: models.ModelSpec, context) -> tuple[Path, Path | None]:
    """Gewichte sicherstellen. Rückgabe: (Modelldatei, Bildteil oder None)."""
    directory = models.ensure_local(
        _weights_spec(spec),
        allow_download=config.allow_model_download,
        on_status=context.status,
        should_stop=context.should_stop,
        allow_conditional=True,
        offline=config.offline_mode,
        on_progress=lambda done, total: context.progress(
            (done / total) if total else 0.0,
            f"{done / (1024**2):.0f} MB von {total / (1024**2):.0f} MB",
        ),
    )

    def finde(name: str) -> Path | None:
        if not name:
            return None
        direkt = directory / name
        if direkt.is_file():
            return direkt
        return next(iter(sorted(directory.rglob(name))), None)

    gewichte = finde(spec.gguf_file)
    if gewichte is None:
        raise ChatUnavailable(
            f"{spec.gguf_file} liegt nicht in {directory}. Der Download ist "
            "unvollständig – 'models remove' und erneut laden."
        )
    return gewichte, finde(spec.mmproj_file)


# ---------------------------------------------------------------------------
# Nachrichten
# ---------------------------------------------------------------------------
@dataclass
class ChatMessage:
    """Eine Nachricht im Verlauf."""

    role: str  # "user" | "assistant" | "system"
    text: str = ""
    images: tuple[Path, ...] = ()
    created_at: float = field(default_factory=time.time)

    def to_api(self, with_images: bool) -> dict[str, Any]:
        """In das Format von llama.cpp bringen.

        Ohne Bildteil wird der Bildhinweis als Text mitgegeben statt das
        Bild wegzulassen – sonst antwortet das Modell auf eine Frage, die
        es nur halb gesehen hat, ohne dass jemand merkt warum.
        """
        if not self.images:
            return {"role": self.role, "content": self.text}
        if not with_images:
            namen = ", ".join(p.name for p in self.images)
            hinweis = f"[{len(self.images)} Bild(er) angehängt: {namen} – dieses Modell sieht keine Bilder]"
            return {"role": self.role, "content": f"{self.text}\n{hinweis}".strip()}

        teile: list[dict[str, Any]] = []
        for bild in self.images:
            teile.append({"type": "image_url", "image_url": {"url": _data_uri(bild)}})
        if self.text.strip():
            teile.append({"type": "text", "text": self.text})
        return {"role": self.role, "content": teile}


def _data_uri(path: Path) -> str:
    """Bild verkleinert als data:-URI. Nie werfend für die Oberfläche."""
    from PIL import Image

    with Image.open(path) as bild:
        bild.load()
        bild = bild.convert("RGB")
        if max(bild.size) > MAX_IMAGE_SIDE:
            faktor = MAX_IMAGE_SIDE / float(max(bild.size))
            bild = bild.resize(
                (max(1, int(bild.width * faktor)), max(1, int(bild.height * faktor))),
                Image.LANCZOS,
            )
        puffer = io.BytesIO()
        bild.save(puffer, format="JPEG", quality=88)
    roh = base64.b64encode(puffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{roh}"


# ---------------------------------------------------------------------------
# Sitzung
# ---------------------------------------------------------------------------
class ChatSession:
    """Ein geladenes Modell samt Verlauf.

    Das Modell bleibt geladen, solange die Sitzung lebt – ein GGUF neu
    einzulesen kostet Sekunden bis Minuten und würde jede Antwort
    verzögern.
    """

    def __init__(self, config: AppConfig, spec: models.ModelSpec) -> None:
        self.config = config
        self.spec = spec
        self.history: list[ChatMessage] = []
        self._llama: Any = None
        self._handler: Any = None
        self._lock = threading.Lock()
        self.sees_images = False

    # --- Laden --------------------------------------------------------
    def load(self, context) -> None:
        ok, reason = runtime_available()
        if not ok:
            raise ChatUnavailable(reason)

        gewichte, bildteil = ensure_weights(self.config, self.spec, context)
        import llama_cpp

        kwargs: dict[str, Any] = {
            "model_path": str(gewichte),
            "n_ctx": int(self.spec.context_tokens or 4096),
            "n_threads": self.config.cpu_threads or None,
            "verbose": False,
        }
        if self.spec.chat_format:
            kwargs["chat_format"] = self.spec.chat_format

        if bildteil is not None:
            handler_cls = self._vision_handler(llama_cpp)
            if handler_cls is None:
                context.status(
                    "Diese llama.cpp-Fassung bringt keinen passenden Bildteil "
                    "mit – der Chat läuft als reiner Textchat."
                )
            else:
                self._handler = handler_cls(clip_model_path=str(bildteil), verbose=False)
                kwargs["chat_handler"] = self._handler
                # Bilder fressen Kontext: ein Foto belegt schnell mehrere
                # hundert Token. Ohne Aufschlag reicht der Platz nach zwei
                # Bildern nicht mehr für die Antwort.
                kwargs["n_ctx"] = max(kwargs["n_ctx"], 8192)
                self.sees_images = True

        context.status(f"Lade {self.spec.title} …")
        started = time.time()
        try:
            self._llama = llama_cpp.Llama(**kwargs)
        except Exception as exc:
            raise ChatUnavailable(
                f"{self.spec.title} ließ sich nicht laden: {clean_error(exc)}"
            ) from exc
        context.status(
            f"{self.spec.title} bereit ({time.time() - started:.0f} s, "
            f"{'mit' if self.sees_images else 'ohne'} Bildverständnis)."
        )

    @staticmethod
    def _vision_handler(llama_cpp: Any) -> Any:
        """Passenden Bildteil in dieser llama.cpp-Fassung suchen."""
        try:
            from llama_cpp import llama_chat_format
        except Exception:
            return None
        for name in _VISION_HANDLERS:
            handler = getattr(llama_chat_format, name, None)
            if handler is not None:
                return handler
        return None

    # --- Fragen -------------------------------------------------------
    def ask(
        self,
        text: str,
        images: Sequence[Path] = (),
        on_token: Callable[[str], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> ChatMessage:
        """Frage stellen und Antwort strömen lassen.

        ``on_token`` bekommt jedes Stück, sobald es da ist – ohne das
        wirkt der Chat bei 4 Token/s wie eingefroren.
        """
        if self._llama is None:
            raise ChatUnavailable("Es ist kein Modell geladen.")

        frage = ChatMessage(role="user", text=text, images=tuple(images))
        self.history.append(frage)

        nachrichten = [{"role": "system", "content": SYSTEM_PROMPT}]
        nachrichten += [m.to_api(self.sees_images) for m in self._im_kontext()]

        stuecke: list[str] = []
        with self._lock:
            try:
                strom = self._llama.create_chat_completion(
                    messages=nachrichten,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                    stream=True,
                )
                for happen in strom:
                    if should_stop is not None and should_stop():
                        break
                    delta = happen.get("choices", [{}])[0].get("delta", {})
                    stueck = delta.get("content") or ""
                    if not stueck:
                        continue
                    stuecke.append(stueck)
                    if on_token is not None:
                        on_token(stueck)
            except Exception as exc:
                # Die Frage bleibt im Verlauf stehen – sonst wäre nach einem
                # Fehler nicht mehr nachvollziehbar, worauf er sich bezog.
                raise RuntimeError(f"Antwort fehlgeschlagen: {clean_error(exc)}") from exc

        antwort = ChatMessage(role="assistant", text="".join(stuecke))
        self.history.append(antwort)
        return antwort

    def _im_kontext(self, hoechstens: int = 20) -> Iterable[ChatMessage]:
        """Die jüngsten Nachrichten. Ältere fallen aus dem Fenster.

        Ohne Begrenzung wächst die Anfrage mit jedem Zug, bis der Kontext
        überläuft – und llama.cpp bricht dann mitten in einer Antwort ab.
        """
        return self.history[-hoechstens:]

    def clear(self) -> None:
        self.history.clear()

    def unload(self) -> None:
        self._llama = None
        self._handler = None
        self.sees_images = False

    # --- Verlauf sichern ----------------------------------------------
    def transcript(self) -> str:
        """Verlauf als Markdown – zum Kopieren und Ablegen."""
        zeilen: list[str] = [f"# Chat mit {self.spec.title}", ""]
        for nachricht in self.history:
            wer = {"user": "Du", "assistant": "Assistent"}.get(nachricht.role, nachricht.role)
            zeilen.append(f"## {wer}")
            if nachricht.images:
                zeilen.append("_Bilder: " + ", ".join(p.name for p in nachricht.images) + "_")
            zeilen.append(nachricht.text)
            zeilen.append("")
        return "\n".join(zeilen)


def make_chat_job(session: ChatSession, text: str, images: Sequence[Path], on_token, **kwargs):
    """Handler für die Warteschlange – hält die Oberfläche frei."""

    def handler(context) -> ChatMessage:
        if session._llama is None:
            session.load(context)
        context.status("Antwort wird geschrieben …")
        return session.ask(
            text,
            images,
            on_token=on_token,
            should_stop=context.should_stop,
            **kwargs,
        )

    return handler
