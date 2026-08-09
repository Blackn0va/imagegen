"""Modell-Registrierung, Download, Cache und Repo-Auflösung.

Kern dieses Moduls ist die **Lizenz-Kennzeichnung**. Die Anwendung wird
verkauft, also darf kein Modell mit nicht-kommerzieller Lizenz in den
Standardpfad geraten. Jeder Eintrag trägt seine Lizenz im Quelltext; die
Stufe ``Commercial`` entscheidet, ob überhaupt geladen werden darf:

  ALLOWED     – kommerziell klar erlaubt, wird als Vorgabe angeboten
  CONDITIONAL – erlaubt, aber mit Bedingung (Umsatzgrenze, Registrierung,
                Namensnennung). Nur nach ausdrücklicher Zustimmung.
  DENIED      – nicht-kommerziell. Wird bewusst mitgeführt, damit klar ist,
                warum das Modell fehlt – Download wird verweigert.

Die vollständige Tabelle liegt zusätzlich in MODELS.md.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import paths
from .accel import Backend, ModelReadiness, clean_error

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (bytes_done, bytes_total)
StatusCallback = Callable[[str], None]
StopCallback = Callable[[], bool]


class DownloadCancelled(RuntimeError):
    """Download wurde abgebrochen.

    Wird durch alle Ebenen durchgereicht und darf NICHT von einem
    allgemeinen ``except Exception`` geschluckt werden – wer fängt, muss
    sie erneut werfen.
    """


class ModelBlocked(RuntimeError):
    """Modell ist für den kommerziellen Einsatz gesperrt."""


class Task(StrEnum):
    IMAGE = "image"
    VIDEO = "video"
    VOICE = "voice"
    VOICE_CLONE = "voice_clone"
    UPSCALE = "upscale"


class Commercial(StrEnum):
    ALLOWED = "allowed"
    CONDITIONAL = "conditional"
    DENIED = "denied"

    def label(self) -> str:
        return {
            Commercial.ALLOWED: "ja",
            Commercial.CONDITIONAL: "ja, mit Bedingung",
            Commercial.DENIED: "nein",
        }[self]


@dataclass(frozen=True)
class ModelSpec:
    key: str  # Kurzname für die Konfiguration
    repo_id: str  # Hugging-Face-Repo
    task: Task
    title: str
    license_id: str
    license_url: str
    commercial: Commercial
    obligations: tuple[str, ...] = ()
    approx_size_mb: int = 0
    min_vram_mb: int = 0  # 0 = läuft auch ohne GPU
    revision: str = "main"
    # "fp16" = halbe Genauigkeit bevorzugen, wo das Repo beide Fassungen
    # anbietet. Spart bei SDXL rund 7 GB gegenüber fp32.
    variant: str = "fp16"
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    consent_component: str = ""  # Schlüssel in licensing.COMPONENTS
    notes: str = ""
    # --- Einzeldatei-Checkpoints (Civitai-Bauart) --------------------------
    # Viele der besten Feinabstimmungen liegen nicht als diffusers-Ordner
    # vor, sondern als eine einzige .safetensors. Dafür braucht diffusers
    # ``from_single_file`` plus die Bauplan-Dateien eines Referenz-Repos.
    single_file: str = ""  # Dateiname im Repo, "" = normaler Ordner
    single_file_class: str = "StableDiffusionXLPipeline"
    single_file_config: str = "stabilityai/stable-diffusion-xl-base-1.0"

    @property
    def is_single_file(self) -> bool:
        return bool(self.single_file)

    @property
    def approx_size_gb(self) -> float:
        return round(self.approx_size_mb / 1024.0, 1)

    def label(self) -> str:
        size = f"{self.approx_size_gb:g} GB" if self.approx_size_mb else "Größe unbekannt"
        return f"{self.title} ({size}, Lizenz: {self.license_id})"


# ---------------------------------------------------------------------------
# Vorgaben für Dateifilter: keine doppelten Gewichte mitziehen
#
# Ohne diese Filter lädt ein SDXL-Repo rund 46 GB statt 6,5 GB: fp32- UND
# fp16-Gewichte, dieselben Gewichte nochmal als .bin und als OpenVINO/ONNX,
# dazu die Einzeldatei-Checkpoints im Wurzelverzeichnis, die das
# diffusers-Ordnerformat gar nicht benutzt.
# ---------------------------------------------------------------------------
_DIFFUSERS_IGNORE = (
    "*.ckpt",
    "*.pt",
    "*.msgpack",
    "*.h5",
    "*.onnx",
    "*.onnx_data",
    "*_fp32.safetensors",
    "*.gguf",
    "*.md",
)

# Dateiendungen, die nur Doppelungen sind, WENN daneben ein safetensors liegt.
_REDUNDANT_SUFFIXES = (".bin", ".pth", ".pt", ".ckpt", ".h5", ".msgpack")

# Namensbestandteile fremder Laufzeiten – für diese Anwendung nutzlos.
_FOREIGN_RUNTIME_HINTS = (
    "openvino",
    "flax_model",
    "tf_model",
    "rust_model",
    "model.onnx",
    "coreml",
    "tensorrt",
    "_ov_",
    "/onnx/",
)

# Bilder und Beispiele im Repo – Doku, kein Modell.
_DOC_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".wav", ".md")

# ---------------------------------------------------------------------------
# Registrierung
# ---------------------------------------------------------------------------
REGISTRY: dict[str, ModelSpec] = {}


def _add(spec: ModelSpec) -> ModelSpec:
    REGISTRY[spec.key] = spec
    return spec


# --- Bild -------------------------------------------------------------------
_add(
    ModelSpec(
        key="sdxl-base",
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        task=Task.IMAGE,
        title="Stable Diffusion XL 1.0 Base",
        license_id="CreativeML Open RAIL++-M",
        license_url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md",
        commercial=Commercial.ALLOWED,
        obligations=(
            "Nutzungsbeschränkungen aus Anhang A müssen an Endkunden weitergegeben werden.",
            "Lizenzkopie beilegen.",
        ),
        approx_size_mb=6_900,
        min_vram_mb=6_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("sdxl", "xl"),
        notes="Vorgabe für Bild. Läuft auf CPU, dort aber langsam.",
    )
)

_add(
    ModelSpec(
        key="ssd-1b",
        repo_id="segmind/SSD-1B",
        task=Task.IMAGE,
        title="Segmind SSD-1B (destilliertes SDXL, 50 % kleiner)",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/segmind/SSD-1B",
        commercial=Commercial.ALLOWED,
        obligations=("Namensnennung im Lizenzhinweis.",),
        approx_size_mb=4_100,
        min_vram_mb=4_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("ssd", "sdxl-small"),
        notes="Gute Wahl für 4–6 GB VRAM.",
    )
)

_add(
    ModelSpec(
        key="sd15",
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        task=Task.IMAGE,
        title="Stable Diffusion 1.5",
        license_id="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5",
        commercial=Commercial.ALLOWED,
        obligations=("Nutzungsbeschränkungen an Endkunden weitergeben.",),
        approx_size_mb=2_200,
        min_vram_mb=3_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("sd", "1.5"),
        notes="Kleinste sinnvolle Basis, Grundlage für AnimateDiff.",
    )
)

_add(
    ModelSpec(
        key="flux-schnell",
        repo_id="black-forest-labs/FLUX.1-schnell",
        task=Task.IMAGE,
        title="FLUX.1 [schnell]",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        commercial=Commercial.ALLOWED,
        obligations=("Namensnennung im Lizenzhinweis.",),
        approx_size_mb=23_800,
        min_vram_mb=12_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("flux", "schnell"),
        notes="Beste Qualität der freien Modelle. FLUX.1-dev ist NICHT kommerziell nutzbar.",
    )
)

_add(
    ModelSpec(
        key="sdxl-turbo",
        repo_id="stabilityai/sdxl-turbo",
        task=Task.IMAGE,
        title="SDXL-Turbo (1–4 Schritte)",
        license_id="Stability AI Community License",
        license_url="https://stability.ai/community-license-agreement",
        commercial=Commercial.CONDITIONAL,
        obligations=(
            "Kommerziell nur unter einer Jahresumsatz-Grenze (Community License) – "
            "darüber ist eine Enterprise-Lizenz von Stability AI nötig.",
            "Namensnennung 'Powered by Stability AI'.",
            "Selbstauskunft/Registrierung bei Stability AI erforderlich.",
        ),
        approx_size_mb=6_900,
        min_vram_mb=6_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("turbo",),
        consent_component="",
        notes="NICHT Vorgabe. Erst nach Prüfung der Umsatzgrenze freischalten.",
    )
)

# --- Video ------------------------------------------------------------------
_add(
    ModelSpec(
        key="wan-t2v-1.3b",
        repo_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        task=Task.VIDEO,
        title="Wan 2.1 Text-zu-Video 1.3B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B",
        commercial=Commercial.ALLOWED,
        obligations=("Namensnennung im Lizenzhinweis.",),
        approx_size_mb=8_200,
        min_vram_mb=8_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("wan", "wan2.1"),
        notes="Vorgabe für Video. 480p, wenige Sekunden, läuft ab 8 GB VRAM.",
    )
)

_add(
    ModelSpec(
        key="cogvideox-2b",
        repo_id="THUDM/CogVideoX-2b",
        task=Task.VIDEO,
        title="CogVideoX 2B",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/THUDM/CogVideoX-2b",
        commercial=Commercial.ALLOWED,
        obligations=("Namensnennung im Lizenzhinweis.",),
        approx_size_mb=10_500,
        min_vram_mb=8_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("cogvideo", "cog"),
        notes="Nur die 2B-Fassung ist Apache-2.0. CogVideoX-5B hat eine eigene Lizenz.",
    )
)

_add(
    ModelSpec(
        key="animatediff",
        repo_id="guoyww/animatediff-motion-adapter-v1-5-3",
        task=Task.VIDEO,
        title="AnimateDiff Motion-Adapter (für SD 1.5)",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3",
        commercial=Commercial.ALLOWED,
        obligations=("Basismodell SD 1.5 bringt seine RAIL-Beschränkungen mit.",),
        approx_size_mb=1_700,
        min_vram_mb=4_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("animate",),
        notes="Sparsamster Videopfad – funktioniert mit 4–6 GB VRAM.",
    )
)

_add(
    ModelSpec(
        key="svd-xt",
        repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
        task=Task.VIDEO,
        title="Stable Video Diffusion XT (Bild zu Video)",
        license_id="Stability AI Community License",
        license_url="https://stability.ai/community-license-agreement",
        commercial=Commercial.CONDITIONAL,
        obligations=(
            "Umsatzgrenze der Community License beachten, darüber Enterprise-Lizenz.",
            "Namensnennung 'Powered by Stability AI'.",
        ),
        approx_size_mb=9_600,
        min_vram_mb=10_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("svd",),
        notes="NICHT Vorgabe.",
    )
)

# --- Stimme -----------------------------------------------------------------
_add(
    ModelSpec(
        key="kokoro",
        repo_id="hexgrad/Kokoro-82M",
        task=Task.VOICE,
        title="Kokoro 82M (Text zu Sprache)",
        license_id="Apache-2.0",
        license_url="https://huggingface.co/hexgrad/Kokoro-82M",
        commercial=Commercial.ALLOWED,
        obligations=("Namensnennung im Lizenzhinweis.",),
        approx_size_mb=330,
        min_vram_mb=0,
        # Doku, Beispiele und Bilder sind für den Betrieb nicht nötig.
        ignore_patterns=("*.md", "eval/**", "samples/**", "*.jpeg", "*.png"),
        aliases=("tts", "kokoro-82m"),
        notes="Vorgabe für Sprache. Läuft schnell auf der CPU.",
    )
)

_add(
    ModelSpec(
        key="bark-small",
        repo_id="suno/bark-small",
        task=Task.VOICE,
        title="Bark small (mehrsprachig, auch Deutsch)",
        license_id="MIT",
        license_url="https://huggingface.co/suno/bark-small",
        commercial=Commercial.ALLOWED,
        obligations=("Namensnennung im Lizenzhinweis.",),
        approx_size_mb=1_800,
        min_vram_mb=3_000,
        variant="",
        ignore_patterns=("*.md", "*.png", "*.jpg"),
        aliases=("bark", "de-tts"),
        notes=(
            "Vorgabe für deutsche Sprachausgabe. Kokoro kann kein Deutsch. "
            "Sprecher über Vorgaben wie 'v2/de_speaker_3'."
        ),
    )
)

_add(
    ModelSpec(
        key="bark",
        repo_id="suno/bark",
        task=Task.VOICE,
        title="Bark (große Fassung, mehrsprachig)",
        license_id="MIT",
        license_url="https://huggingface.co/suno/bark",
        commercial=Commercial.ALLOWED,
        obligations=("Namensnennung im Lizenzhinweis.",),
        approx_size_mb=5_200,
        min_vram_mb=6_000,
        variant="",
        ignore_patterns=("*.md", "*.png", "*.jpg"),
        aliases=("bark-large",),
        notes="Bessere Qualität als bark-small, braucht mehr Speicher.",
    )
)

_add(
    ModelSpec(
        key="piper",
        repo_id="rhasspy/piper-voices",
        task=Task.VOICE,
        title="Piper – Thorsten (deutsch, CC0) und HFC (englisch)",
        license_id="Stimmen: CC0-1.0 / CC-BY-4.0 – ABER Laufzeit piper-tts: GPL-3.0",
        license_url="https://huggingface.co/rhasspy/piper-voices",
        commercial=Commercial.CONDITIONAL,
        obligations=(
            "Die Stimmgewichte selbst sind frei (Thorsten CC0, HFC CC-BY-4.0).",
            "Das Problem ist die Laufzeit: 'piper-tts' steht unter GPL-3.0 und "
            "bettet espeak-ng ein. In denselben Prozess geladen zieht das die "
            "gesamte verkaufte Anwendung unter die GPL.",
            "Nur zulässig, wenn Piper als eigenständiges Programm (eigener Prozess) "
            "ausgeliefert wird – mit Lizenztext und Quelltextangebot.",
            "Weitere Piper-Stimmen NICHT ungeprüft nachladen – einzelne Datensätze "
            "im selben Repo sind nicht-kommerziell (z. B. Blizzard-basierte Stimmen).",
        ),
        consent_component="piper-gpl",
        approx_size_mb=140,
        min_vram_mb=0,
        variant="",
        # Bewusst eng: nur geprüfte Stimmen, nicht das ganze Repo (mehrere GB).
        allow_patterns=(
            "de/de_DE/thorsten/medium/*",
            "de/de_DE/thorsten/high/*",
            "en/en_US/hfc_female/medium/*",
        ),
        aliases=("piper-voices", "thorsten"),
        notes=(
            "Schnellste deutsche Sprachausgabe (CPU, ~60 MB je Stimme), aber wegen "
            "der GPL-Laufzeit NICHT die Vorgabe. Vorgabe ist bark-small (MIT)."
        ),
    )
)

_add(
    ModelSpec(
        key="chatterbox",
        repo_id="ResembleAI/chatterbox",
        task=Task.VOICE_CLONE,
        title="Chatterbox TTS (Stimme aus Beispielaufnahme)",
        license_id="MIT",
        license_url="https://huggingface.co/ResembleAI/chatterbox",
        commercial=Commercial.ALLOWED,
        obligations=(
            "Einwilligung der sprechenden Person erforderlich (Persönlichkeitsrecht).",
            "Erzeugte Sprache enthält ein Wasserzeichen des Herstellers – nicht entfernen.",
        ),
        approx_size_mb=2_100,
        min_vram_mb=6_000,
        aliases=("clone", "chatter"),
        consent_component="voice-cloning",
        notes="Zero-Shot-Klonen aus ~10 s Referenzaufnahme.",
    )
)

_add(
    ModelSpec(
        key="openvoice-v2",
        repo_id="myshell-ai/OpenVoiceV2",
        task=Task.VOICE_CLONE,
        title="OpenVoice v2 (Stimmfarbe übertragen)",
        license_id="MIT",
        license_url="https://huggingface.co/myshell-ai/OpenVoiceV2",
        commercial=Commercial.ALLOWED,
        obligations=("Einwilligung der sprechenden Person erforderlich.",),
        approx_size_mb=1_200,
        min_vram_mb=4_000,
        aliases=("openvoice",),
        consent_component="voice-cloning",
        notes="Überträgt Stimmfarbe auf eine Basis-Stimme; sparsamer als Chatterbox.",
    )
)

_add(
    ModelSpec(
        key="xtts-v2",
        repo_id="coqui/XTTS-v2",
        task=Task.VOICE_CLONE,
        title="Coqui XTTS v2",
        license_id="Coqui Public Model License (nicht kommerziell)",
        license_url="https://coqui.ai/cpml",
        commercial=Commercial.DENIED,
        obligations=("Kommerzielle Nutzung ausgeschlossen.",),
        approx_size_mb=1_900,
        min_vram_mb=6_000,
        aliases=("xtts",),
        notes="Bewusst gesperrt. Sieht frei aus, ist es nicht.",
    )
)

_add(
    ModelSpec(
        key="f5-tts",
        repo_id="SWivid/F5-TTS",
        task=Task.VOICE_CLONE,
        title="F5-TTS",
        license_id="Programmcode MIT, Gewichte CC-BY-NC-4.0",
        license_url="https://huggingface.co/SWivid/F5-TTS",
        commercial=Commercial.DENIED,
        obligations=("Gewichte auf NC-Datensatz trainiert – kommerziell gesperrt.",),
        approx_size_mb=1_400,
        min_vram_mb=6_000,
        aliases=("f5",),
        notes="Bewusst gesperrt.",
    )
)

# --- Feinabstimmungen für Inhalte für Erwachsene ----------------------------
# Alle Angaben hier sind gegen die Hugging-Face-API geprüft: Repo existiert,
# Format (diffusers-Ordner oder Einzeldatei), Lizenzangabe der Modellkarte
# und die Größe NACH dem Dateifilter aus select_files().
#
# Die Basismodelle weiter oben können Nacktheit, sind darauf aber nicht
# abgestimmt. Diese hier sind es – Anatomie, Hauttöne und Posen sitzen
# deutlich besser.
_add(
    ModelSpec(
        key="pony-v6",
        repo_id="AstraliteHeart/pony-diffusion-v6",
        task=Task.IMAGE,
        title="Pony Diffusion V6 XL (sehr starke Prompt-Treue, auch explizit)",
        license_id="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/AstraliteHeart/pony-diffusion-v6",
        commercial=Commercial.ALLOWED,
        obligations=(
            "Nutzungsbeschränkungen aus Anhang A weitergeben (u. a. Verbot der "
            "Ausbeutung Minderjähriger).",
        ),
        approx_size_mb=6_620,
        min_vram_mb=6_000,
        variant="",
        allow_patterns=("v6.safetensors",),
        single_file="v6.safetensors",
        single_file_class="StableDiffusionXLPipeline",
        aliases=("pony", "ponyxl"),
        notes=(
            "Einzeldatei-Checkpoint. Erwartet Wertungs-Marker am Prompt-Anfang: "
            "'score_9, score_8_up, score_7_up'. Ohne sie sind die Ergebnisse "
            "deutlich schwächer. Beim ersten Laden werden einige hundert KB "
            "Bauplan-Dateien von Hugging Face geholt."
        ),
    )
)

_add(
    ModelSpec(
        key="noobai-xl",
        repo_id="Laxhar/noobai-XL-1.1",
        task=Task.IMAGE,
        title="NoobAI-XL 1.1 (Anime/Manga, explizite Darstellungen)",
        license_id="Fair AI Public License 1.0-SD",
        license_url="https://freedevproject.org/faipl-1.0-sd/",
        commercial=Commercial.CONDITIONAL,
        obligations=(
            "Abwandlungen des Modells müssen unter derselben Lizenz und öffentlich "
            "weitergegeben werden (copyleft-artig).",
            "Bei entgeltlicher Weitergabe eines abgeleiteten Modells sind die "
            "Bedingungen der Lizenz zu prüfen.",
        ),
        approx_size_mb=6_620,
        min_vram_mb=6_000,
        variant="",
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("noob", "noobai"),
        notes="Anime-Basis mit sehr breitem Tag-Vokabular (Danbooru-Stil).",
    )
)

_add(
    ModelSpec(
        key="realvis-xl",
        repo_id="SG161222/RealVisXL_V4.0",
        task=Task.IMAGE,
        title="RealVisXL V4.0 (fotorealistische Menschen)",
        license_id="CreativeML Open RAIL++-M",
        license_url="https://huggingface.co/SG161222/RealVisXL_V4.0",
        commercial=Commercial.ALLOWED,
        obligations=("Nutzungsbeschränkungen aus Anhang A weitergeben.",),
        approx_size_mb=6_620,
        min_vram_mb=6_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("realvis", "realvisxl"),
        notes="Fotorealistisch, gute Haut und Anatomie. Vorgabe für realistische Akte.",
    )
)

_add(
    ModelSpec(
        key="juggernaut-xl",
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        task=Task.IMAGE,
        title="Juggernaut XL v9 (fotorealistisch, kräftige Beleuchtung)",
        license_id="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/RunDiffusion/Juggernaut-XL-v9",
        commercial=Commercial.ALLOWED,
        obligations=("Nutzungsbeschränkungen aus Anhang A weitergeben.",),
        approx_size_mb=6_620,
        min_vram_mb=6_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("juggernaut", "jugg"),
        notes="Zweite Meinung neben RealVisXL – anderer Look, gleiche Klasse.",
    )
)

_add(
    ModelSpec(
        key="realistic-vision",
        repo_id="SG161222/Realistic_Vision_V6.0_B1_noVAE",
        task=Task.IMAGE,
        title="Realistic Vision V6.0 (SD 1.5, fotorealistisch, sparsam)",
        license_id="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE",
        commercial=Commercial.ALLOWED,
        obligations=("Nutzungsbeschränkungen aus Anhang A weitergeben.",),
        approx_size_mb=5_229,
        min_vram_mb=4_000,
        variant="",
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("realvision", "rv6"),
        notes=(
            "Auf SD 1.5 aufgebaut: läuft schon mit 4 GB VRAM und ist bei 512–768 px "
            "am stärksten. Für kleine Karten die beste Wahl."
        ),
    )
)

_add(
    ModelSpec(
        key="dreamshaper",
        repo_id="Lykon/dreamshaper-8",
        task=Task.IMAGE,
        title="DreamShaper 8 (SD 1.5, Allrounder, kleinster Eintrag)",
        license_id="CreativeML Open RAIL-M",
        license_url="https://huggingface.co/Lykon/dreamshaper-8",
        commercial=Commercial.ALLOWED,
        obligations=("Nutzungsbeschränkungen aus Anhang A weitergeben.",),
        approx_size_mb=2_615,
        min_vram_mb=3_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("ds8", "dreamshaper8"),
        notes="Nur 2,6 GB. Zwischen Fotorealismus und Illustration, sehr genügsam.",
    )
)

_add(
    ModelSpec(
        key="nsfw-gen",
        repo_id="UnfilteredAI/NSFW-gen-v2",
        task=Task.IMAGE,
        title="NSFW-gen v2 (ausdrücklich auf explizite Darstellungen abgestimmt)",
        license_id="unbekannt – Modellkarte nennt nur 'other'",
        license_url="https://huggingface.co/UnfilteredAI/NSFW-gen-v2",
        commercial=Commercial.CONDITIONAL,
        obligations=(
            "Die Modellkarte nennt keine benannte Lizenz. Vor einer Weitergabe "
            "eigener Ergebnisse selbst prüfen.",
        ),
        approx_size_mb=8_179,
        min_vram_mb=6_000,
        ignore_patterns=_DIFFUSERS_IGNORE,
        aliases=("nsfwgen",),
        notes="Direkt auf explizite Motive trainiert, dafür stilistisch enger.",
    )
)


# --- Nachbearbeitung --------------------------------------------------------
_add(
    ModelSpec(
        key="realesrgan-x4",
        repo_id="ai-forever/Real-ESRGAN",
        task=Task.UPSCALE,
        title="Real-ESRGAN x4 (Hochskalieren)",
        license_id="BSD-3-Clause",
        license_url="https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE",
        commercial=Commercial.ALLOWED,
        obligations=("Lizenztext und Namensnennung beilegen.",),
        approx_size_mb=250,
        min_vram_mb=2_000,
        # Nur die Gewichte. Das Repo enthält daneben Beispielbilder, die der
        # Doku-Filter zwar ohnehin verwirft – ausdrücklich ist es klarer.
        allow_patterns=("*.pth",),
        variant="",
        aliases=("esrgan", "upscale"),
        notes=(
            "Vergrößert bestehende Bilder um Faktor 2, 4 oder 8. Läuft auch auf "
            "der CPU, dort langsamer. Ohne dieses Modell wird Lanczos benutzt."
        ),
    )
)


DEFAULTS: dict[Task, str] = {
    Task.IMAGE: "sdxl-base",
    Task.VIDEO: "wan-t2v-1.3b",
    Task.VOICE: "kokoro",
    Task.VOICE_CLONE: "chatterbox",
    Task.UPSCALE: "realesrgan-x4",
}


# ---------------------------------------------------------------------------
# Auflösung
# ---------------------------------------------------------------------------
def by_task(task: Task, include_blocked: bool = False) -> list[ModelSpec]:
    items = [s for s in REGISTRY.values() if s.task is task]
    if not include_blocked:
        items = [s for s in items if s.commercial is not Commercial.DENIED]
    return sorted(items, key=lambda s: (s.commercial.value, s.approx_size_mb))


def resolve(name: str) -> ModelSpec:
    """Kurzname, Alias oder rohe Repo-ID auflösen.

    Unbekannte Repo-IDs werden akzeptiert, aber als CONDITIONAL mit
    unbekannter Lizenz markiert – der Nutzer muss dann selbst prüfen.
    """
    key = (name or "").strip()
    if not key:
        raise KeyError("Kein Modellname angegeben.")
    lowered = key.lower()
    if lowered in REGISTRY:
        return REGISTRY[lowered]
    for spec in REGISTRY.values():
        if lowered == spec.repo_id.lower() or lowered in {a.lower() for a in spec.aliases}:
            return spec
    if "/" in key:
        return ModelSpec(
            key=key,
            repo_id=key,
            task=Task.IMAGE,
            title=f"Eigenes Modell {key}",
            license_id="unbekannt – selbst prüfen",
            license_url=f"https://huggingface.co/{key}",
            commercial=Commercial.CONDITIONAL,
            obligations=(
                "Lizenz dieses Repos wurde nicht geprüft. Vor dem Verkauf "
                "eigener Ausgaben Lizenz lesen.",
            ),
            ignore_patterns=_DIFFUSERS_IGNORE,
            notes="Nicht Teil der geprüften Auslieferung.",
        )
    raise KeyError(f"Unbekanntes Modell: {name}")


def license_table(include_blocked: bool = True) -> str:
    """Markdown-Tabelle: Modell | Aufgabe | Lizenz | kommerziell | Auflagen."""
    header = (
        "| Modell | Aufgabe | Lizenz | kommerziell erlaubt? | Auflagen |\n|---|---|---|---|---|"
    )
    rows: list[str] = []
    for spec in sorted(REGISTRY.values(), key=lambda s: (s.task.value, s.key)):
        if not include_blocked and spec.commercial is Commercial.DENIED:
            continue
        obligations = "<br>".join(spec.obligations) or "–"
        rows.append(
            f"| `{spec.key}` ({spec.repo_id}) | {spec.task.value} | {spec.license_id} "
            f"| {spec.commercial.label()} | {obligations} |"
        )
    return "\n".join([header, *rows])


# ---------------------------------------------------------------------------
# Cache-Pfade und Vorhandensein
# ---------------------------------------------------------------------------
def local_dir(spec: ModelSpec) -> Path:
    """Zielverzeichnis im Modell-Cache (ein Ordner je Repo)."""
    safe = spec.repo_id.replace("/", "__")
    return paths.models_dir() / spec.task.value / safe


def converted_dir(spec: ModelSpec, backend: str) -> Path:
    """Ablage für backend-spezifische Konvertate (ONNX für DirectML)."""
    return local_dir(spec).with_name(local_dir(spec).name + f".{backend}")


COMPLETE_MARKER = ".streamforge-complete.json"
PARTIAL_MARKER = ".streamforge-partial"


def _dir_has_weights(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    patterns = ("*.safetensors", "*.bin", "*.onnx", "*.pth", "*.gguf")
    return any(next(directory.rglob(pattern), None) is not None for pattern in patterns)


def _has_partial(directory: Path) -> bool:
    """Angefangener Download: Marker oder .part-Datei vorhanden."""
    if not directory.is_dir():
        return False
    if (directory / PARTIAL_MARKER).is_file():
        return True
    return next(directory.rglob("*" + PART_SUFFIX), None) is not None


def _read_marker(directory: Path) -> dict[str, Any] | None:
    marker = directory / COMPLETE_MARKER
    if not marker.is_file():
        return None
    try:
        import json

        data = json.loads(marker.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _write_complete_marker(directory: Path, spec: ModelSpec, files: Sequence[RemoteFile]) -> None:
    import json
    import time

    payload = {
        "repo_id": spec.repo_id,
        "revision": spec.revision,
        "completed_at": time.time(),
        "files": [{"name": item.name, "size": item.size} for item in files],
    }
    (directory / COMPLETE_MARKER).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (directory / PARTIAL_MARKER).unlink(missing_ok=True)


def is_downloaded(spec: ModelSpec) -> bool:
    """Ohne Netz prüfen, ob das Modell **vollständig** vorliegt.

    Ein abgebrochener Download hinterlässt einen Teil-Marker. Ohne diese
    Unterscheidung würde ein halb geladenes Modell als vorhanden gelten und
    erst beim Laden mit einer unverständlichen Meldung scheitern.
    """
    directory = local_dir(spec)
    if not directory.is_dir() or _has_partial(directory):
        return False
    if _read_marker(directory) is not None:
        return True
    # Kein Marker: von Hand abgelegtes Modell – Gewichte genügen.
    return _dir_has_weights(directory)


def verify_local(spec: ModelSpec) -> tuple[bool, list[str]]:
    """Vollständigkeit gegen den Marker prüfen (ohne Netz)."""
    directory = local_dir(spec)
    problems: list[str] = []
    if not directory.is_dir():
        return False, [f"Verzeichnis fehlt: {directory}"]
    if _has_partial(directory):
        problems.append("Angefangener Download – bitte erneut laden (wird fortgesetzt).")
    marker = _read_marker(directory)
    if marker is None:
        if not _dir_has_weights(directory):
            problems.append("Keine Gewichtsdateien gefunden.")
        return (not problems), problems
    for entry in marker.get("files", []):
        name = str(entry.get("name", ""))
        size = int(entry.get("size", 0) or 0)
        path = directory / name
        if not path.is_file():
            problems.append(f"fehlt: {name}")
        elif size and path.stat().st_size != size:
            problems.append(f"Größe weicht ab: {name}")
    return (not problems), problems


def is_converted(spec: ModelSpec, backend: str) -> bool:
    return _dir_has_weights(converted_dir(spec, backend))


def disk_usage_mb(spec: ModelSpec) -> int:
    directory = local_dir(spec)
    if not directory.is_dir():
        return 0
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except OSError:
            continue
    return int(total / (1024 * 1024))


def readiness(spec: ModelSpec) -> dict[str, ModelReadiness]:
    """Auskunft für die Backend-Kette (Erststart-Bremse in accel.py).

    CUDA und CPU nutzen dieselben diffusers-Gewichte, brauchen also keine
    Konvertierung. DirectML läuft über ONNX – dafür muss exportiert werden,
    und genau dieser Export darf im Auto-Modus nicht beim ersten Start
    losrennen.
    """
    downloaded = is_downloaded(spec)
    dml_ready = is_converted(spec, Backend.DML)
    return {
        Backend.CUDA: ModelReadiness(
            ready=downloaded,
            needs_conversion=False,
            note="" if downloaded else "Modell noch nicht heruntergeladen.",
        ),
        Backend.CPU: ModelReadiness(
            ready=downloaded,
            needs_conversion=False,
            note="" if downloaded else "Modell noch nicht heruntergeladen.",
        ),
        Backend.DML: ModelReadiness(
            ready=dml_ready,
            needs_conversion=not dml_ready,
            note="ONNX-Export vorhanden." if dml_ready else "ONNX-Export fehlt.",
        ),
    }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1024 * 1024  # 1 MB – Abbruch wird je Block geprüft
PART_SUFFIX = ".part"


@dataclass(frozen=True)
class RemoteFile:
    """Eine Datei im Repo samt bekannter Größe."""

    name: str
    size: int = 0


def _matches(name: str, allow: Sequence[str], ignore: Sequence[str]) -> bool:
    """Dateifilter wie bei huggingface_hub (fnmatch, '**' inklusive)."""
    from fnmatch import fnmatch

    def hit(pattern: str) -> bool:
        # "de/**" soll auch "de/x/y.onnx" treffen
        return fnmatch(name, pattern) or (pattern.endswith("/**") and name.startswith(pattern[:-2]))

    if allow and not any(hit(pattern) for pattern in allow):
        return False
    return not (ignore and any(hit(pattern) for pattern in ignore))


def _strip_variant(stem: str) -> str:
    """'model.fp16' -> 'model'. Erkennt die üblichen diffusers-Varianten."""
    for variant in ("fp16", "bf16", "fp32", "16", "8bit"):
        if stem.endswith("." + variant):
            return stem[: -(len(variant) + 1)]
    return stem


def select_files(
    spec: ModelSpec,
    names_with_size: Sequence[tuple[str, int]],
    components: set[str] | None = None,
) -> list[RemoteFile]:
    """Aus allen Repo-Dateien die auswählen, die wirklich gebraucht werden.

    Reihenfolge der Regeln:
      1. eigene allow/ignore-Muster des Modells
      2. fremde Laufzeiten (OpenVINO/ONNX/Flax/TF) und Doku raus
      3. Einzeldatei-Checkpoints im Wurzelverzeichnis raus, wenn das Repo
         im diffusers-Ordnerformat vorliegt
      4. nur Komponenten, die model_index.json nennt
      5. Gewichte doppelt (.bin neben .safetensors) raus
      6. von zwei Genauigkeiten die gewünschte Variante behalten
    """
    all_names = {name for name, _ in names_with_size}
    folder_format = "model_index.json" in all_names
    kept: dict[str, int] = {}

    for name, size in names_with_size:
        lowered = name.lower()
        if not _matches(name, spec.allow_patterns, spec.ignore_patterns):
            continue
        if any(hint in lowered for hint in _FOREIGN_RUNTIME_HINTS):
            continue
        if lowered.endswith(_DOC_SUFFIXES):
            continue
        if folder_format and "/" not in name and name.endswith((".safetensors", ".ckpt", ".pt")):
            # sd_xl_base_1.0.safetensors & Co. – das Ordnerformat lädt sie nicht.
            continue
        if components is not None and "/" in name:
            top = name.split("/", 1)[0]
            if top not in components:
                continue
        kept[name] = size

    # 5. Doppelte Gewichte: .bin/.pth entfernen, wenn safetensors daneben liegt
    for name in list(kept):
        path = Path(name)
        if path.suffix.lower() in _REDUNDANT_SUFFIXES:
            stem = _strip_variant(path.stem)
            parent = path.parent.as_posix()
            prefix = "" if parent in (".", "") else parent + "/"
            has_safetensors = any(
                other.startswith(prefix)
                and other.endswith(".safetensors")
                and _strip_variant(Path(other).stem) == stem
                for other in kept
            )
            if has_safetensors:
                kept.pop(name, None)

    # 6. Variante wählen: fp16 schlägt die Vollfassung, wenn beides da ist
    if spec.variant:
        for name in list(kept):
            path = Path(name)
            if path.suffix.lower() != ".safetensors":
                continue
            if _strip_variant(path.stem) != path.stem:
                continue  # ist bereits eine Variante
            parent = path.parent.as_posix()
            prefix = "" if parent in (".", "") else parent + "/"
            variant_name = f"{prefix}{path.stem}.{spec.variant}.safetensors"
            if variant_name in kept:
                kept.pop(name, None)

    return [RemoteFile(name=name, size=size) for name, size in sorted(kept.items())]


def _fetch_components(spec: ModelSpec, session: Any) -> set[str] | None:
    """Komponentenordner aus model_index.json lesen.

    SDXL-Repos enthalten zusätzliche Ordner (vae_1_0, vae_decoder, …), die
    die Pipeline nie lädt. Die Datei ist wenige Kilobyte groß – ein Abruf
    vorab spart mehrere GB Download.
    """
    try:
        from huggingface_hub import hf_hub_url

        url = hf_hub_url(spec.repo_id, "model_index.json", revision=spec.revision)
        response = session.get(url, headers=_hf_headers(), timeout=(10, 30))
        if response.status_code != 200:
            return None
        index = response.json()
    except Exception as exc:
        log.debug("model_index.json nicht lesbar: %s", clean_error(exc))
        return None
    if not isinstance(index, dict):
        return None
    components = {
        key
        for key, value in index.items()
        if not key.startswith("_") and isinstance(value, (list, tuple))
    }
    return components or None


def list_remote_files(spec: ModelSpec, timeout: float = 20.0) -> tuple[list[RemoteFile], str]:
    """Dateiliste mit Größen holen und auf das Nötige eindampfen."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(spec.repo_id, revision=spec.revision, files_metadata=True)
    except Exception as exc:
        return [], f"Dateiliste nicht abrufbar: {clean_error(exc)}"

    raw: list[tuple[str, int]] = []
    for sibling in info.siblings or []:
        name = str(getattr(sibling, "rfilename", "") or "")
        if not name or name.endswith("/"):
            continue
        raw.append((name, int(getattr(sibling, "size", 0) or 0)))

    components: set[str] | None = None
    if any(name == "model_index.json" for name, _ in raw):
        session = _http_session()
        try:
            components = _fetch_components(spec, session)
        finally:
            with contextlib.suppress(Exception):
                session.close()

    files = select_files(spec, raw, components)
    if not files:
        return [], "Keine passende Datei im Repo (Filter zu streng?)."
    return files, ""


def estimate_size_mb(spec: ModelSpec, timeout: float = 20.0) -> tuple[int, str]:
    """Gesamtgröße vorab schätzen (Warnung VOR dem Download)."""
    files, note = list_remote_files(spec, timeout)
    if note:
        return spec.approx_size_mb, f"{note} Schätzwert genutzt."
    total = sum(item.size for item in files)
    if not total:
        return spec.approx_size_mb, "Repo nennt keine Dateigrößen, Schätzwert genutzt."
    return int(total / (1024 * 1024)), ""


def _cleanup_incomplete(*directories: Path) -> int:
    """Halbe Dateien entfernen, damit ein Abbruch nichts hinterlässt."""
    removed = 0
    for directory in directories:
        if not directory or not directory.exists():
            continue
        for pattern in ("**/*" + PART_SUFFIX, "**/*.incomplete", "**/*.lock"):
            for path in directory.glob(pattern):
                try:
                    if path.is_file():
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
    return removed


def _http_session():
    """Sitzung von huggingface_hub verwenden (Proxy, Retries, Auth-Header)."""
    try:
        from huggingface_hub.utils import get_session

        return get_session()
    except Exception:
        import requests

        return requests.Session()


def _hf_headers() -> dict[str, str]:
    try:
        from huggingface_hub.utils import build_hf_headers

        return dict(build_hf_headers())
    except Exception:
        return {}


def _download_file(
    spec: ModelSpec,
    remote: RemoteFile,
    target_dir: Path,
    session: Any,
    state: dict[str, int],
    on_progress: ProgressCallback | None,
    should_stop: StopCallback | None,
    lock: Any = None,
) -> Path:
    """Eine Datei streamen: erst .part, dann os.replace.

    Damit hinterlässt ein Abbruch nie eine halbe Zieldatei, und ein
    vorhandener Rest wird über einen Range-Request fortgesetzt.
    """
    from huggingface_hub import hf_hub_url

    destination = target_dir / remote.name
    part = destination.with_name(destination.name + PART_SUFFIX)
    paths.ensure_dir(destination.parent)

    def advance(amount: int) -> None:
        """Fortschritt hochzählen – threadsicher, wenn ein Lock übergeben wurde."""
        if lock is not None:
            with lock:
                state["done"] += amount
                done, total = state["done"], state["total"]
        else:
            state["done"] += amount
            done, total = state["done"], state["total"]
        if on_progress is not None:
            on_progress(done, total)

    # Bereits vollständig vorhanden?
    if destination.is_file() and remote.size and destination.stat().st_size == remote.size:
        advance(remote.size)
        return destination

    headers = _hf_headers()
    resume_from = part.stat().st_size if part.is_file() else 0
    if resume_from and remote.size and resume_from < remote.size:
        headers["Range"] = f"bytes={resume_from}-"
    elif resume_from:
        # Größe unbekannt oder Rest zu groß – neu anfangen.
        part.unlink(missing_ok=True)
        resume_from = 0

    url = hf_hub_url(spec.repo_id, remote.name, revision=spec.revision)
    response = session.get(url, headers=headers, stream=True, timeout=(10, 60))
    if response.status_code == 416:  # Range nicht erfüllbar -> neu laden
        part.unlink(missing_ok=True)
        resume_from = 0
        headers.pop("Range", None)
        response = session.get(url, headers=headers, stream=True, timeout=(10, 60))
    response.raise_for_status()

    # Fortsetzen nur, wenn der Server den Range-Request auch bestätigt (206).
    resume_ok = resume_from > 0 and response.status_code == 206
    if resume_ok:
        mode = "ab"
        advance(resume_from)
    else:
        mode = "wb"
        resume_from = 0

    try:
        with open(part, mode) as handle:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if should_stop is not None and should_stop():
                    handle.flush()
                    raise DownloadCancelled("Download abgebrochen")
                if not chunk:
                    continue
                handle.write(chunk)
                advance(len(chunk))
    finally:
        response.close()

    os.replace(part, destination)
    return destination


def check_allowed(spec: ModelSpec, allow_conditional: bool = False) -> None:
    """Lizenztor. DENIED wird immer verweigert – fail-closed."""
    if spec.commercial is Commercial.DENIED:
        raise ModelBlocked(
            f"{spec.title} steht unter '{spec.license_id}' und ist für den "
            "kommerziellen Einsatz gesperrt. Wähle ein Modell mit freier Lizenz."
        )
    if spec.commercial is Commercial.CONDITIONAL and not allow_conditional:
        raise ModelBlocked(
            f"{spec.title} ist nur unter Bedingungen kommerziell nutzbar "
            f"({spec.license_id}). Auflagen: {'; '.join(spec.obligations) or 'siehe Lizenz'}. "
            "Erst nach ausdrücklicher Freigabe verwendbar."
        )
    if spec.consent_component:
        from . import licensing

        result = licensing.gate(spec.consent_component)
        if not result.allowed:
            raise ModelBlocked(result.reason)


def download(
    spec: ModelSpec | str,
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
    should_stop: StopCallback | None = None,
    allow_conditional: bool = False,
    offline: bool = False,
    workers_hint: int = 4,
) -> Path:
    """Modell in den Cache laden. Abbrechbar, ohne halbe Dateien.

    Wirft ``DownloadCancelled`` bei Abbruch und ``ModelBlocked`` bei
    Lizenzsperre. Beide bewusst als eigene Typen, damit sie nicht im
    allgemeinen Fehlerpfad verschwinden.
    """
    model = resolve(spec) if isinstance(spec, str) else spec
    check_allowed(model, allow_conditional=allow_conditional)

    target = local_dir(model)
    status = on_status or (lambda _text: None)

    if is_downloaded(model):
        status(f"{model.title} liegt bereits vor.")
        return target

    if offline:
        raise FileNotFoundError(
            f"{model.title} fehlt und der Offline-Modus ist aktiv. Erwartet in {target}."
        )

    try:
        import huggingface_hub  # noqa: F401 – nur Verfügbarkeit prüfen
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub ist nicht installiert – Modell-Download nicht möglich."
        ) from exc

    from . import nettrust

    nettrust.install()

    # Eigener Streaming-Download statt snapshot_download: nur so lässt sich
    # der Byte-Fortschritt melden UND zeitnah abbrechen. snapshot_download
    # gibt tqdm_class nur an die Dateizähler-Leiste weiter, nicht an den
    # Byte-Strom – ein Abbruch hätte dort erst nach der nächsten fertigen
    # Datei gegriffen, bei mehreren GB also praktisch nie.
    remote_files, list_note = list_remote_files(model)
    if list_note:
        raise RuntimeError(f"Download von {model.repo_id} nicht möglich: {list_note}")

    total_bytes = sum(item.size for item in remote_files) or model.approx_size_mb * 1024 * 1024
    status(
        f"Lade {model.title} – {len(remote_files)} Datei(en), "
        f"etwa {total_bytes / (1024**3):.1f} GB."
    )

    paths.ensure_dir(target)
    # Teil-Marker: solange er liegt, gilt das Modell als unvollständig.
    (target / PARTIAL_MARKER).write_text(f"{model.repo_id}@{model.revision}\n", encoding="utf-8")
    session = _http_session()
    state = {"done": 0, "total": int(total_bytes)}
    progress_lock = threading.Lock()
    counter = {"files": 0}

    # Parallel laden: Repos wie Bark bestehen aus hunderten kleinen Dateien.
    # Nacheinander kostet die Latenz je Datei mehr als die Übertragung selbst.
    workers = max(1, min(int(workers_hint), len(remote_files)))

    def fetch(remote: RemoteFile) -> None:
        if should_stop is not None and should_stop():
            raise DownloadCancelled("Download abgebrochen")
        _download_file(
            model, remote, target, session, state, on_progress, should_stop, lock=progress_lock
        )
        with progress_lock:
            counter["files"] += 1
            done_files = counter["files"]
        if done_files % 5 == 0 or done_files == len(remote_files):
            status(f"[{done_files}/{len(remote_files)}] {remote.name}")

    try:
        if workers == 1:
            for remote in remote_files:
                fetch(remote)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(fetch, remote) for remote in remote_files]
                try:
                    for future in as_completed(futures):
                        future.result()  # Ausnahme des Arbeiters hier hochreichen
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
    except DownloadCancelled:
        removed = _cleanup_incomplete(target)
        status(f"Abgebrochen. {removed} unvollständige Datei(en) entfernt.")
        raise  # niemals schlucken
    except Exception as exc:
        _cleanup_incomplete(target)
        raise RuntimeError(
            f"Download von {model.repo_id} fehlgeschlagen: {clean_error(exc)}"
        ) from exc
    finally:
        with contextlib.suppress(Exception):
            session.close()

    _write_complete_marker(target, model, remote_files)
    status(f"{model.title} bereit.")
    return target


def ensure_local(
    spec: ModelSpec | str,
    allow_download: bool = True,
    on_progress: ProgressCallback | None = None,
    on_status: StatusCallback | None = None,
    should_stop: StopCallback | None = None,
    allow_conditional: bool = False,
    offline: bool = False,
    workers_hint: int = 4,
) -> Path:
    """Pfad zum Modell liefern; bei Bedarf und Erlaubnis herunterladen."""
    model = resolve(spec) if isinstance(spec, str) else spec
    target = local_dir(model)
    if is_downloaded(model):
        return target
    if not allow_download:
        raise FileNotFoundError(
            f"{model.title} fehlt und Download ist ausgeschaltet. Erwartet in {target}."
        )
    return download(
        model,
        on_progress=on_progress,
        on_status=on_status,
        should_stop=should_stop,
        allow_conditional=allow_conditional,
        offline=offline,
    )


# Bauplan-Dateien eines diffusers-Repos: alles, was kein Gewicht ist.
_CONFIG_ALLOW = (
    "model_index.json",
    "*/config.json",
    "*/scheduler_config.json",
    "*/tokenizer_config.json",
    "*/special_tokens_map.json",
    "*/vocab.json",
    "*/merges.txt",
    "*/tokenizer.json",
    "*/preprocessor_config.json",
)
_CONFIG_IGNORE = (
    "*.safetensors",
    "*.bin",
    "*.ckpt",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.msgpack",
    "*.h5",
    "*.gguf",
)


def config_dir(repo_id: str) -> Path:
    """Ablage für die Bauplan-Dateien eines Referenz-Repos."""
    return paths.models_dir() / "configs" / repo_id.replace("/", "__")


def ensure_reference_config(
    repo_id: str,
    allow_download: bool = True,
    on_status: StatusCallback | None = None,
    should_stop: StopCallback | None = None,
    offline: bool = False,
) -> Path:
    """Bauplan eines Referenz-Repos lokal bereitstellen (nur JSON/TXT).

    Einzeldatei-Checkpoints enthalten nur Gewichte, keine Angaben darüber,
    wie die Bestandteile zusammengesetzt sind. diffusers holt die sonst
    über den Hugging-Face-Cache – und der legt unter Windows Symlinks an,
    was ohne Entwicklermodus mit „WinError 1314: Dem Client fehlt ein
    erforderliches Recht“ abbricht. Deshalb derselbe eigene Downloader wie
    für alle anderen Dateien, in ein Verzeichnis dieser Anwendung.

    Zusammen wenige hundert Kilobyte.
    """
    target = config_dir(repo_id)
    if (target / "model_index.json").is_file():
        return target

    if offline or not allow_download:
        raise FileNotFoundError(
            f"Der Bauplan für {repo_id} fehlt und darf nicht geladen werden. "
            f"Erwartet in {target}. Einmal mit Netzzugang laden, danach geht es offline."
        )

    spec = ModelSpec(
        key=repo_id,
        repo_id=repo_id,
        task=Task.IMAGE,
        title=f"Bauplan {repo_id}",
        license_id="nur Konfigurationsdateien",
        license_url="",
        commercial=Commercial.ALLOWED,
        variant="",
        allow_patterns=_CONFIG_ALLOW,
        ignore_patterns=_CONFIG_IGNORE,
    )
    remote, note = list_remote_files(spec)
    if note or not remote:
        raise RuntimeError(f"Bauplan für {repo_id} nicht abrufbar: {note or 'leer'}")

    paths.ensure_dir(target)
    session = _http_session()
    state = {"done": 0, "total": sum(f.size for f in remote) or 1}
    if on_status is not None:
        on_status(f"Hole Bauplan für {repo_id} ({len(remote)} Dateien) …")
    try:
        for item in remote:
            if should_stop is not None and should_stop():
                raise DownloadCancelled("Abgebrochen")
            _download_file(spec, item, target, session, state, None, should_stop)
    finally:
        with contextlib.suppress(Exception):
            session.close()
    return target


def _local_components(directory: Path) -> set[str] | None:
    """Komponentenordner aus der lokalen model_index.json."""
    index_file = directory / "model_index.json"
    if not index_file.is_file():
        return None
    try:
        import json

        index = json.loads(index_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict):
        return None
    components = {
        key
        for key, value in index.items()
        if not key.startswith("_") and isinstance(value, (list, tuple))
    }
    return components or None


def prune_local(spec: ModelSpec | str, dry_run: bool = False) -> tuple[int, int, list[str]]:
    """Überflüssige Dateien aus einem bereits geladenen Modell entfernen.

    Für Modelle, die mit einem älteren, zu großzügigen Filter geladen
    wurden: entfernt fp32-Doppelungen, .bin neben .safetensors, fremde
    Laufzeiten und Einzeldatei-Checkpoints. Spart bei SDXL rund 39 GB.

    Rückgabe: (Anzahl Dateien, freigegebene MB, Namen).
    """
    model = resolve(spec) if isinstance(spec, str) else spec
    directory = local_dir(model)
    if not directory.is_dir():
        return 0, 0, []

    entries: list[tuple[str, int]] = []
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        name = path.relative_to(directory).as_posix()
        if name.startswith(".streamforge"):
            continue
        try:
            entries.append((name, path.stat().st_size))
        except OSError:
            continue

    keep = {item.name for item in select_files(model, entries, _local_components(directory))}
    removed: list[str] = []
    freed = 0
    for name, size in entries:
        if name in keep:
            continue
        removed.append(name)
        freed += size
        if not dry_run:
            try:
                (directory / name).unlink()
            except OSError as exc:
                log.warning("Konnte %s nicht löschen: %s", name, clean_error(exc))

    if not dry_run and removed:
        # Leere Ordner hinterlassen sonst Verwirrung im Modellordner.
        for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
            if path.is_dir() and not any(path.iterdir()):
                with contextlib.suppress(OSError):
                    path.rmdir()
        kept_files = [RemoteFile(name=name, size=size) for name, size in entries if name in keep]
        _write_complete_marker(directory, model, kept_files)

    return len(removed), int(freed / (1024 * 1024)), removed


def remove(spec: ModelSpec | str) -> int:
    """Modell aus dem Cache löschen. Gibt freigegebene MB zurück."""
    model = resolve(spec) if isinstance(spec, str) else spec
    freed = disk_usage_mb(model)
    for directory in (local_dir(model), converted_dir(model, Backend.DML)):
        if directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
    return freed


def installed() -> list[tuple[ModelSpec, int]]:
    """Alle vorhandenen Modelle mit Belegung in MB."""
    result: list[tuple[ModelSpec, int]] = []
    for spec in REGISTRY.values():
        if is_downloaded(spec):
            result.append((spec, disk_usage_mb(spec)))
    return sorted(result, key=lambda item: -item[1])


def fits_hardware(spec: ModelSpec, vram_mb: int, ram_mb: int) -> tuple[bool, str]:
    """Passt das Modell auf diese Maschine? Antwort VOR dem Download."""
    if spec.min_vram_mb and vram_mb >= spec.min_vram_mb:
        return True, f"Passt: {vram_mb // 1024} GB VRAM vorhanden."
    if not spec.min_vram_mb:
        return True, "Läuft auch ohne GPU."
    if ram_mb >= spec.min_vram_mb * 2:
        return True, (
            f"Zu wenig VRAM ({vram_mb // 1024} GB), läuft aber über den "
            "Arbeitsspeicher – deutlich langsamer."
        )
    return False, (
        f"Braucht etwa {spec.min_vram_mb // 1024} GB VRAM, vorhanden sind "
        f"{vram_mb // 1024} GB. Nimm ein kleineres Modell."
    )
