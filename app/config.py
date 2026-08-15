"""Konfiguration als eingefrorene Dataclass.

Regeln:
  * niemals mutieren – Varianten über ``dataclasses.replace()``
  * unbekannte JSON-Schlüssel werden ignoriert, nicht als Fehler behandelt
    (alte Konfigurationen dürfen nach einem Update weiter laden)
  * Umgebungsvariablen ``STREAMFORGE_<FELD>`` überschreiben die Datei
  * Speichern läuft atomar (tmp + os.replace), damit ein Absturz keine
    halbe Konfiguration hinterlässt
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

from . import paths

log = logging.getLogger(__name__)

ENV_PREFIX = "STREAMFORGE_"

DEVICE_CHOICES = ("auto", "cuda", "dml", "openvino", "cpu")
# Zielgerät innerhalb von OpenVINO. Leer = automatische Wahl
# (iGPU vor NPU vor CPU; die NPU ist bei Diffusionsmodellen langsam und
# ihr Treiber neigt zu harten Abstürzen, siehe accel.OPENVINO_DEVICE_ORDER).
OPENVINO_DEVICES = ("", "NPU", "GPU", "CPU")
COMPUTE_CHOICES = ("float16", "bfloat16", "float32", "int8")
IMAGE_FORMATS = ("png", "jpg", "webp")
UPSCALE_FACTORS = (2, 4, 8)
# Steinformen fürs Diamond Painting. Doppelt zu app.diamond gehalten, damit
# die Konfiguration ohne Pillow-Modul prüfbar bleibt.
DIAMOND_SHAPES = ("round", "square")
VIDEO_CONTAINERS = ("mp4", "webm", "mov")
AUDIO_FORMATS = ("wav", "flac", "mp3")


@dataclass(frozen=True)
class AppConfig:
    # --- Allgemein ---------------------------------------------------------
    output_dir: str = "output"
    language: str = "de"
    theme: str = "dark"  # dark | light
    log_level: str = "INFO"

    # --- Gerät / Beschleunigung -------------------------------------------
    device: str = "auto"  # auto | cuda | dml | cpu
    device_index: int = 0
    compute_type: str = "float16"
    gpu_low_impact: bool = True  # Rechner bleibt bedienbar
    attention_slicing: bool = True
    vae_tiling: bool = True
    cpu_offload: bool = False  # Modellteile bei Bedarf in den RAM auslagern
    cpu_threads: int = 0  # 0 = automatisch
    seed_locked: bool = False

    # --- Modelle / Download -----------------------------------------------
    image_model: str = "sdxl-base"
    video_model: str = "wan-t2v-1.3b"
    # bark-small (MIT) ist die Vorgabe: Kokoro kann kein Deutsch, und die
    # schnellere Piper-Laufzeit steht unter GPL-3.0 – für eine verkaufte,
    # proprietäre Anwendung nicht tragbar (siehe licensing.py, 'piper-gpl').
    voice_model: str = "bark-small"
    upscale_model: str = "realesrgan-x4"
    allow_model_download: bool = True
    offline_mode: bool = False
    download_workers: int = 4
    prefer_safetensors: bool = True

    # --- Bild --------------------------------------------------------------
    image_width: int = 1024
    image_height: int = 1024
    image_steps: int = 25
    image_guidance: float = 6.0
    image_sampler: str = "euler_a"
    image_batch: int = 1
    image_format: str = "png"
    image_jpeg_quality: int = 92
    image_negative_prompt: str = ""
    image_clip_skip: int = 0

    # --- Inhalte für Erwachsene --------------------------------------------
    # Vorgabe: an. Die Anwendung läuft lokal und wird nicht weitergegeben,
    # also gibt es niemanden, dem gegenüber eine Freigabe zu dokumentieren
    # wäre. Abschalten schaltet die Inhaltsprüfung der Modelle wieder ein.
    nsfw_enabled: bool = True
    nsfw_disable_safety_checker: bool = True
    nsfw_protective_negative: bool = True
    # Sperre gegen sexualisierte Darstellungen Minderjähriger. Bewusst nicht
    # abschaltbar – siehe contentgate.py und validated().
    nsfw_block_minors: bool = True

    # --- Bild bearbeiten und vergrößern ------------------------------------
    # Stärke = wie stark das Modell vom Ausgangsbild abweichen darf.
    # 0 = nichts ändert sich, 1 = das Ausgangsbild wird völlig überschrieben.
    image_edit_strength: float = 0.45
    image_edit_refine_strength: float = 0.25
    # Einfärben: höhere Stärke als beim Umarbeiten, weil die Helligkeit
    # hinterher aus der Vorlage zurückgeholt wird – das Modell kann also
    # kräftig zugreifen, ohne dass Details verloren gehen.
    image_colorize_strength: float = 0.55
    image_colorize_keep_luminance: bool = True
    # --- Diamond Painting ---------------------------------------------------
    # Breite des Rasters in Steinen. 100 ergibt bei runden Steinen (2,8 mm)
    # rund 28 cm Bildbreite – das gängige Format.
    diamond_stones: int = 100
    diamond_colors: int = 24
    diamond_cell_px: int = 18
    diamond_shape: str = "round"
    diamond_symbols: bool = True
    # Farben auf bestellbare DMC-Nummern abbilden. Aus wäre die Vorlage
    # zwar farbtreuer, aber niemand verkauft Steine nach Hexwert.
    diamond_use_dmc: bool = True

    upscale_factor: int = 2
    upscale_tile: int = 512
    upscale_use_model: bool = True
    upscale_refine: bool = False

    # --- Video -------------------------------------------------------------
    video_width: int = 832
    video_height: int = 480
    video_frames: int = 49
    video_fps: int = 16
    video_steps: int = 30
    video_guidance: float = 5.0
    video_motion: float = 1.0
    video_container: str = "mp4"
    video_crf: int = 20
    video_codec: str = "libopenh264"  # LGPL-freundlich, siehe compose.py
    video_interpolate: bool = False

    # --- Stimme ------------------------------------------------------------
    # Bark-Sprechervorgabe; bei Piper wäre es z. B. "de_DE-thorsten-medium".
    voice_speaker: str = "v2/de_speaker_3"
    voice_speed: float = 1.0
    voice_pitch: float = 0.0
    voice_volume_db: float = 0.0
    voice_sample_rate: int = 24000
    audio_format: str = "wav"
    voice_split_sentences: bool = True

    # --- Stimme anlernen (Klonen) -----------------------------------------
    # Fail-closed: ohne dokumentierte Einwilligung des Sprechers wird kein
    # geklontes Profil verwendet. Details in licensing.py / voice_profiles.py.
    voice_cloning_enabled: bool = False
    voice_clone_model: str = "chatterbox"
    voice_profile: str = ""  # aktives angelerntes Profil
    voice_training_epochs: int = 20
    voice_training_batch: int = 8
    voice_training_lr: float = 1e-4
    voice_min_sample_seconds: float = 6.0
    voice_require_consent: bool = True  # bewusst nicht abschaltbar über GUI

    # --- Vertonung / Muxen -------------------------------------------------
    mux_audio_codec: str = "aac"
    mux_audio_bitrate: str = "192k"
    mux_normalize_audio: bool = True
    mux_loop_audio: bool = False

    # --- Warteschlange / Oberfläche ---------------------------------------
    job_workers: int = 1  # 1 = VRAM wird nicht zerrissen
    keep_model_loaded: bool = True
    # Leer = automatisch. Sonst NPU, GPU oder CPU erzwingen.
    openvino_device: str = ""
    auto_open_output: bool = False
    show_advanced: bool = False
    error_throttle_seconds: float = 5.0

    # ------------------------------------------------------------------
    # Abgeleitete Pfade
    # ------------------------------------------------------------------
    def resolved_output_dir(self) -> Path:
        """Ausgabeordner. Relative Angaben liegen unter dem Datenverzeichnis."""
        candidate = Path(os.path.expandvars(self.output_dir)).expanduser()
        if candidate.is_absolute():
            return candidate
        return paths.data_dir() / candidate

    # ------------------------------------------------------------------
    # Validierung
    # ------------------------------------------------------------------
    def validated(self) -> tuple[AppConfig, list[str]]:
        """Werte in gültige Bereiche zwingen. Gibt Konfiguration + Meldungen."""
        problems: list[str] = []
        changes: dict[str, Any] = {}

        def clamp(name: str, low, high) -> None:
            value = getattr(self, name)
            if value < low:
                changes[name] = low
                problems.append(f"{name}={value} zu klein, auf {low} gesetzt.")
            elif value > high:
                changes[name] = high
                problems.append(f"{name}={value} zu groß, auf {high} gesetzt.")

        def choice(name: str, allowed: tuple[str, ...]) -> None:
            """Auswahlfeld prüfen, ohne an Groß-/Kleinschreibung zu scheitern.

            Die meisten Auswahlen sind klein geschrieben, die Geräte von
            OpenVINO heißen aber "NPU" und "GPU". Verglichen wird deshalb
            unabhängig von der Schreibweise, und gemeldet wird nur, wenn
            sich der Wert wirklich ändert – eine „Korrektur"-Meldung bei
            jedem Start ist Lärm, kein Hinweis.
            """
            value = str(getattr(self, name))
            passend = {entry.lower(): entry for entry in allowed}
            treffer = passend.get(value.lower())
            if treffer is None:
                changes[name] = allowed[0]
                problems.append(f"{name}='{value}' unbekannt, '{allowed[0]}' wird genutzt.")
            elif treffer != value:
                changes[name] = treffer

        choice("device", DEVICE_CHOICES)
        choice("openvino_device", OPENVINO_DEVICES)
        choice("compute_type", COMPUTE_CHOICES)
        choice("image_format", IMAGE_FORMATS)
        choice("diamond_shape", DIAMOND_SHAPES)
        choice("video_container", VIDEO_CONTAINERS)
        choice("audio_format", AUDIO_FORMATS)

        clamp("device_index", 0, 15)
        clamp("image_width", 256, 4096)
        clamp("image_height", 256, 4096)
        clamp("image_steps", 1, 150)
        clamp("image_guidance", 0.0, 30.0)
        clamp("image_batch", 1, 16)
        clamp("image_jpeg_quality", 40, 100)
        clamp("image_edit_strength", 0.05, 1.0)
        clamp("image_edit_refine_strength", 0.05, 1.0)
        clamp("image_colorize_strength", 0.05, 1.0)
        # Grenzen wie in app.diamond: darüber wird die Vorlage unbezahlbar
        # groß, darunter ist nichts mehr zu erkennen.
        clamp("diamond_stones", 20, 400)
        clamp("diamond_colors", 2, 48)
        clamp("diamond_cell_px", 8, 48)
        clamp("upscale_tile", 0, 4096)
        clamp("video_width", 256, 1920)
        clamp("video_height", 256, 1088)
        clamp("video_frames", 8, 241)
        clamp("video_fps", 4, 60)
        clamp("video_steps", 1, 100)
        clamp("video_crf", 0, 51)
        clamp("voice_speed", 0.4, 2.5)
        clamp("voice_pitch", -12.0, 12.0)
        clamp("voice_volume_db", -30.0, 12.0)
        clamp("voice_training_epochs", 1, 500)
        clamp("voice_training_batch", 1, 64)
        clamp("job_workers", 1, 4)
        clamp("download_workers", 1, 16)
        clamp("cpu_threads", 0, 256)
        clamp("error_throttle_seconds", 0.5, 120.0)

        # Diffusionsmodelle brauchen Vielfache von 8 (VAE-Faktor).
        for name in ("image_width", "image_height", "video_width", "video_height"):
            value = int(changes.get(name, getattr(self, name)))
            snapped = max(256, (value // 8) * 8)
            if snapped != value:
                changes[name] = snapped
                problems.append(f"{name}={value} auf Vielfaches von 8 gerundet ({snapped}).")

        # Vergrößerungsfaktor: nur 2, 4 und 8 sind sinnvolle Netzfaktoren.
        factor = int(changes.get("upscale_factor", self.upscale_factor))
        if factor not in UPSCALE_FACTORS:
            nearest = min(UPSCALE_FACTORS, key=lambda value: abs(value - factor))
            changes["upscale_factor"] = nearest
            problems.append(f"upscale_factor={factor} unbekannt, {nearest} wird genutzt.")

        # Kachelgröße 0 heißt "ohne Kacheln" – dazwischen muss genug Platz
        # für die Überlappung bleiben, sonst rechnet jede Kachel nur Rand.
        tile = int(changes.get("upscale_tile", self.upscale_tile))
        if 0 < tile < 96:
            changes["upscale_tile"] = 96
            problems.append(f"upscale_tile={tile} zu klein, auf 96 gesetzt.")

        # Jugendschutz-Sperre darf nicht per Konfiguration ausgehebelt werden.
        # Gleiche Bauart wie voice_require_consent: wer die Datei von Hand
        # ändert, bekommt den Wert beim nächsten Laden zurückgesetzt.
        if not self.nsfw_block_minors:
            changes["nsfw_block_minors"] = True
            problems.append(
                "nsfw_block_minors kann nicht abgeschaltet werden – "
                "sexualisierte Darstellungen Minderjähriger bleiben gesperrt "
                "(§ 184b StGB, gilt auch für computererzeugte Bilder)."
            )

        # Einwilligungspflicht darf nicht per Konfiguration ausgehebelt werden.
        if not self.voice_require_consent:
            changes["voice_require_consent"] = True
            problems.append(
                "voice_require_consent kann nicht abgeschaltet werden – "
                "Stimmklonen braucht immer einen Einwilligungs-Nachweis."
            )

        if self.offline_mode and self.allow_model_download:
            changes["allow_model_download"] = False
            problems.append("Offline-Modus aktiv – Modell-Download deaktiviert.")

        return (replace(self, **changes) if changes else self), problems

    # ------------------------------------------------------------------
    # Serialisierung
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False, sort_keys=True)

    def with_values(self, **kwargs: Any) -> AppConfig:
        """Variante erzeugen – unbekannte Felder werden verworfen."""
        known = {f.name for f in fields(self)}
        clean = {k: v for k, v in kwargs.items() if k in known}
        return replace(self, **clean)

    def save(self, path: Path | None = None) -> Path:
        """Atomar schreiben: erst .tmp, dann os.replace."""
        target = Path(path) if path else paths.config_path()
        paths.ensure_dir(target.parent)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        os.replace(tmp, target)
        log.debug("Konfiguration gespeichert: %s", target)
        return target


# ---------------------------------------------------------------------------
# Laden
# ---------------------------------------------------------------------------
def _coerce(value: Any, template: Any) -> Any:
    """JSON-Wert auf den Typ des Vorgabewerts bringen. Fehler -> Vorgabe."""
    if isinstance(template, bool):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "ja", "on"):
            return True
        if text in ("0", "false", "no", "nein", "off"):
            return False
        return template
    try:
        if isinstance(template, int):
            return int(float(value))
        if isinstance(template, float):
            return float(value)
        if isinstance(template, str):
            return str(value)
    except (TypeError, ValueError):
        return template
    return value


def from_mapping(
    data: Mapping[str, Any], base: AppConfig | None = None
) -> tuple[AppConfig, list[str]]:
    """Konfiguration aus einem Wörterbuch. Unbekannte Schlüssel: ignorieren."""
    base = base or AppConfig()
    defaults = {f.name: getattr(base, f.name) for f in fields(base)}
    values: dict[str, Any] = {}
    ignored: list[str] = []
    for key, raw in data.items():
        if key not in defaults:
            ignored.append(key)
            continue
        values[key] = _coerce(raw, defaults[key])
    notes = [f"Unbekannter Konfigurationsschlüssel ignoriert: {k}" for k in sorted(ignored)]
    return replace(base, **values), notes


def apply_env(config: AppConfig) -> tuple[AppConfig, list[str]]:
    """Umgebungsvariablen ``STREAMFORGE_<FELD>`` anwenden."""
    values: dict[str, Any] = {}
    notes: list[str] = []
    for f in fields(config):
        env_key = ENV_PREFIX + f.name.upper()
        if env_key in os.environ:
            values[f.name] = _coerce(os.environ[env_key], getattr(config, f.name))
            notes.append(f"{f.name} aus Umgebungsvariable {env_key} übernommen.")
    return (replace(config, **values) if values else config), notes


def load(path: Path | None = None, use_env: bool = True) -> tuple[AppConfig, list[str]]:
    """Konfiguration laden. Fehlt oder kaputt: Vorgaben + Meldung."""
    target = Path(path) if path else paths.config_path()
    notes: list[str] = []
    config = AppConfig()

    if target.is_file():
        try:
            raw = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                config, parse_notes = from_mapping(raw, config)
                notes.extend(parse_notes)
            else:
                notes.append(f"{target.name} enthält kein Objekt – Vorgaben werden genutzt.")
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            from .accel import clean_error

            notes.append(
                f"{target.name} nicht lesbar ({clean_error(exc)}) – Vorgaben werden genutzt."
            )
    else:
        notes.append(f"Keine Konfiguration gefunden – Vorgaben werden genutzt ({target}).")

    if use_env:
        config, env_notes = apply_env(config)
        notes.extend(env_notes)

    config, problems = config.validated()
    notes.extend(problems)
    return config, notes


def load_or_create(path: Path | None = None) -> tuple[AppConfig, list[str]]:
    """Wie load(), schreibt aber eine fehlende Datei einmalig heraus."""
    target = Path(path) if path else paths.config_path()
    existed = target.is_file()
    config, notes = load(target)
    if not existed:
        try:
            config.save(target)
            notes.append(f"Neue Konfiguration angelegt: {target}")
        except OSError as exc:
            from .accel import clean_error

            notes.append(f"Konfiguration konnte nicht geschrieben werden: {clean_error(exc)}")
    return config, notes
