"""Stimmprofile – Aufnahmen verwalten, Einwilligung führen, Anlernen anstoßen.

Ein Profil ist ein Ordner unter ``<daten>/voices/<slug>/``:

    profile.json      Metadaten inklusive Einwilligungs-Nachweis
    samples/          Referenzaufnahmen (WAV)
    artifacts/        Ergebnis des Anlernens (Sprecher-Einbettung, Adapter)

Fail-closed an zwei Stellen:
  * ohne gültigen Einwilligungs-Nachweis wird weder angelernt noch erzeugt
  * ohne Mindestmenge brauchbarer Aufnahmen wird nicht angelernt

Die eigentliche Anlern-Fachlogik ist in dieser Basis eine Attrappe; die
Schnittstelle steht aber fest (siehe ``train_profile``).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shutil
import time
import wave
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from . import paths
from .accel import clean_error
from .licensing import SpeakerConsent, voice_clone_gate

log = logging.getLogger(__name__)

PROFILE_FILE = "profile.json"
SAMPLES_DIR = "samples"
ARTIFACTS_DIR = "artifacts"

MIN_SAMPLE_SECONDS = 3.0  # kürzere Schnipsel sind fürs Anlernen wertlos
MIN_TOTAL_SECONDS_CLONE = 10.0  # Zero-Shot-Klonen (Chatterbox/OpenVoice)
MIN_TOTAL_SECONDS_FINETUNE = 600.0  # echtes Nachtrainieren (Piper)
SUPPORTED_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


class ProfileState(StrEnum):
    DRAFT = "draft"  # angelegt, Aufnahmen fehlen oder Einwilligung fehlt
    READY = "ready"  # bereit zum Anlernen
    TRAINED = "trained"  # Artefakt vorhanden, nutzbar
    BLOCKED = "blocked"  # Einwilligung fehlt/zurückgezogen

    def label(self) -> str:
        return {
            ProfileState.DRAFT: "Entwurf",
            ProfileState.READY: "bereit zum Anlernen",
            ProfileState.TRAINED: "angelernt",
            ProfileState.BLOCKED: "gesperrt",
        }[self]


class TrainingMode(StrEnum):
    ZERO_SHOT = "zero_shot"  # Referenzaufnahme wird direkt genutzt
    FINETUNE = "finetune"  # Modell wird nachtrainiert – NOCH NICHT UMGESETZT

    def min_seconds(self) -> float:
        return (
            MIN_TOTAL_SECONDS_CLONE
            if self is TrainingMode.ZERO_SHOT
            else MIN_TOTAL_SECONDS_FINETUNE
        )

    @property
    def available(self) -> bool:
        """Ist das Verfahren nutzbar?

        'finetune' wäre echtes Nachtrainieren – dafür liefert chatterbox-tts
        keinen Trainingscode mit. Profile in diesem Modus verlangten 600 s
        Material und ließen sich nie anlernen; das sah wie ein Fehler aus.
        """
        return self is TrainingMode.ZERO_SHOT

    def label(self) -> str:
        return {
            TrainingMode.ZERO_SHOT: "Referenz (sofort nutzbar)",
            TrainingMode.FINETUNE: "Nachtrainieren (nicht verfügbar)",
        }[self]


@dataclass(frozen=True)
class SampleInfo:
    path: Path
    seconds: float
    sample_rate: int
    channels: int
    usable: bool
    note: str = ""


@dataclass
class VoiceProfile:
    slug: str
    display_name: str
    model_key: str = "chatterbox"
    mode: TrainingMode = TrainingMode.ZERO_SHOT
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    consent: SpeakerConsent | None = None
    state: ProfileState = ProfileState.DRAFT
    artifact: str = ""  # Dateiname im artifacts-Ordner
    language: str = "de"
    notes: str = ""
    # Feinschliff der Klangfarbe. Werte gelten je Profil, damit eine einmal
    # gut eingestellte Stimme reproduzierbar bleibt.
    exaggeration: float = 0.5  # 0.3 ruhig … 1.0 sehr ausdrucksstark
    cfg_weight: float = 0.5  # niedriger = näher am Referenztempo
    temperature: float = 0.8  # niedriger = gleichmäßiger, höher = lebendiger
    reference_seconds: float = 20.0  # Zielmenge Material für die Referenz

    # --- Pfade -------------------------------------------------------------
    @property
    def root(self) -> Path:
        return profiles_root() / self.slug

    @property
    def samples_dir(self) -> Path:
        return self.root / SAMPLES_DIR

    @property
    def artifacts_dir(self) -> Path:
        return self.root / ARTIFACTS_DIR

    @property
    def profile_file(self) -> Path:
        return self.root / PROFILE_FILE

    @property
    def artifact_path(self) -> Path | None:
        if not self.artifact:
            return None
        candidate = self.artifacts_dir / self.artifact
        return candidate if candidate.is_file() else None

    # --- Aufnahmen ---------------------------------------------------------
    def samples(self) -> list[SampleInfo]:
        if not self.samples_dir.is_dir():
            return []
        result: list[SampleInfo] = []
        for path in sorted(self.samples_dir.iterdir()):
            if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
                result.append(inspect_sample(path))
        return result

    def total_seconds(self) -> float:
        return sum(s.seconds for s in self.samples() if s.usable)

    # --- Prüfungen ---------------------------------------------------------
    def consent_ok(self) -> tuple[bool, str]:
        result = voice_clone_gate(self.consent)
        return result.allowed, result.reason

    def training_ready(self) -> tuple[bool, list[str]]:
        """Kann angelernt werden? Liste aller Hindernisse im Klartext."""
        problems: list[str] = []
        if not self.mode.available:
            problems.append(
                "Verfahren 'Nachtrainieren' ist nicht umgesetzt – Profil auf "
                "'Referenz' umstellen. Dafür genügen 10–30 Sekunden Material."
            )
        ok, reason = self.consent_ok()
        if not ok:
            problems.append(reason)
        usable = [s for s in self.samples() if s.usable]
        if not usable:
            problems.append("Keine brauchbare Aufnahme vorhanden.")
        total = sum(s.seconds for s in usable)
        needed = self.mode.min_seconds()
        if total < needed:
            problems.append(
                f"Zu wenig Material: {total:.1f}s vorhanden, {needed:.0f}s nötig "
                f"für '{self.mode.value}'."
            )
        return (not problems), problems

    def usable_for_synthesis(self) -> tuple[bool, str]:
        ok, reason = self.consent_ok()
        if not ok:
            return False, reason
        if not [s for s in self.samples() if s.usable]:
            return False, "Keine brauchbare Aufnahme vorhanden."
        if not self.mode.available:
            return False, (
                "Verfahren 'Nachtrainieren' ist nicht umgesetzt – Profil auf 'Referenz' umstellen."
            )
        return True, "Referenzaufnahme vorhanden."

    def refresh_state(self) -> ProfileState:
        ok, _ = self.consent_ok()
        if not ok:
            self.state = ProfileState.BLOCKED
        elif self.artifact_path is not None:
            self.state = ProfileState.TRAINED
        elif self.training_ready()[0]:
            self.state = ProfileState.READY
        else:
            self.state = ProfileState.DRAFT
        return self.state

    # --- Serialisierung ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "display_name": self.display_name,
            "model_key": self.model_key,
            "mode": self.mode.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "state": self.state.value,
            "artifact": self.artifact,
            "language": self.language,
            "notes": self.notes,
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "temperature": self.temperature,
            "reference_seconds": self.reference_seconds,
            "consent": self.consent.to_dict() if self.consent else None,
        }

    def save(self) -> Path:
        paths.ensure_dir(self.root)
        paths.ensure_dir(self.samples_dir)
        paths.ensure_dir(self.artifacts_dir)
        self.updated_at = time.time()
        self.refresh_state()
        tmp = self.profile_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.profile_file)
        return self.profile_file

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> VoiceProfile | None:
        slug = str(data.get("slug") or "").strip()
        if not slug:
            return None
        try:
            mode = TrainingMode(str(data.get("mode", TrainingMode.ZERO_SHOT.value)))
        except ValueError:
            mode = TrainingMode.ZERO_SHOT
        consent_raw = data.get("consent")
        consent = (
            SpeakerConsent.from_dict(consent_raw) if isinstance(consent_raw, Mapping) else None
        )
        profile = VoiceProfile(
            slug=slug,
            display_name=str(data.get("display_name") or slug),
            model_key=str(data.get("model_key") or "chatterbox"),
            mode=mode,
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
            consent=consent,
            artifact=str(data.get("artifact") or ""),
            language=str(data.get("language") or "de"),
            notes=str(data.get("notes") or ""),
            exaggeration=float(data.get("exaggeration", 0.5) or 0.5),
            cfg_weight=float(data.get("cfg_weight", 0.5) or 0.5),
            temperature=float(data.get("temperature", 0.8) or 0.8),
            reference_seconds=float(data.get("reference_seconds", 20.0) or 20.0),
        )
        profile.refresh_state()
        return profile


# ---------------------------------------------------------------------------
# Ablage
# ---------------------------------------------------------------------------
def profiles_root() -> Path:
    return paths.data_dir() / "voices"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return slug or f"stimme-{int(time.time())}"


def list_profiles() -> list[VoiceProfile]:
    root = profiles_root()
    if not root.is_dir():
        return []
    result: list[VoiceProfile] = []
    for directory in sorted(root.iterdir()):
        profile = load_profile(directory.name)
        if profile is not None:
            result.append(profile)
    return result


def load_profile(slug: str) -> VoiceProfile | None:
    file = profiles_root() / slug / PROFILE_FILE
    if not file.is_file():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        log.warning("Stimmprofil %s nicht lesbar: %s", slug, clean_error(exc))
        return None
    return VoiceProfile.from_dict(data) if isinstance(data, dict) else None


def create_profile(
    display_name: str,
    consent: SpeakerConsent,
    model_key: str = "chatterbox",
    mode: TrainingMode = TrainingMode.ZERO_SHOT,
    language: str = "de",
    notes: str = "",
) -> VoiceProfile:
    """Profil anlegen. Ohne gültigen Einwilligungs-Nachweis: Abbruch."""
    if consent is None or not consent.is_valid():
        raise ValueError(
            "Stimmprofil braucht einen gültigen Einwilligungs-Nachweis "
            "(Name der sprechenden Person, Zweck, Datum)."
        )
    slug = slugify(display_name)
    if (profiles_root() / slug).exists():
        slug = f"{slug}-{int(time.time()) % 10000}"
    profile = VoiceProfile(
        slug=slug,
        display_name=display_name.strip() or slug,
        model_key=model_key,
        mode=mode,
        consent=consent,
        language=language,
        notes=notes,
    )
    profile.save()
    log.info("Stimmprofil angelegt: %s (%s)", profile.display_name, profile.slug)
    return profile


def set_mode(slug: str, mode: TrainingMode) -> bool:
    """Verfahren eines Profils wechseln.

    Gedacht für Profile, die versehentlich auf 'Nachtrainieren' stehen: die
    ließen sich nie anlernen, weil sie 600 s Material verlangten.
    """
    profile = load_profile(slug)
    if profile is None:
        return False
    profile.mode = mode
    profile.save()
    log.info("Stimmprofil %s auf Verfahren '%s' umgestellt", slug, mode.value)
    return True


def delete_profile(slug: str) -> bool:
    """Profil samt Aufnahmen löschen – auch der Weg für einen Widerruf."""
    directory = profiles_root() / slug
    if not directory.is_dir():
        return False
    shutil.rmtree(directory, ignore_errors=True)
    log.info("Stimmprofil gelöscht: %s", slug)
    return not directory.exists()


def revoke_consent(slug: str, delete_data: bool = True) -> bool:
    """Einwilligung widerrufen: Profil sperren und Aufnahmen entfernen."""
    profile = load_profile(slug)
    if profile is None:
        return False
    if delete_data:
        return delete_profile(slug)
    profile.consent = None
    profile.state = ProfileState.BLOCKED
    profile.save()
    return True


# ---------------------------------------------------------------------------
# Aufnahmen prüfen und übernehmen
# ---------------------------------------------------------------------------
def inspect_sample(path: Path) -> SampleInfo:
    """Aufnahme prüfen – Dauer, Rate, Kanäle. Ohne Fremdbibliothek für WAV."""
    if path.suffix.lower() != ".wav":
        # Andere Formate werden erst beim Anlernen über ffmpeg gewandelt.
        size_mb = path.stat().st_size / (1024 * 1024) if path.is_file() else 0
        return SampleInfo(
            path=path,
            seconds=0.0,
            sample_rate=0,
            channels=0,
            usable=size_mb > 0.05,
            note="Dauer erst nach Umwandlung mit ffmpeg bekannt.",
        )
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as handle:
            frames = handle.getnframes()
            rate = handle.getframerate() or 1
            channels = handle.getnchannels()
            seconds = frames / float(rate)
    except (wave.Error, OSError, EOFError) as exc:
        return SampleInfo(path, 0.0, 0, 0, False, f"WAV nicht lesbar: {clean_error(exc)}")

    problems: list[str] = []
    if seconds < MIN_SAMPLE_SECONDS:
        problems.append(f"nur {seconds:.1f}s – mindestens {MIN_SAMPLE_SECONDS:.0f}s nötig")
    if rate < 16000:
        problems.append(f"Abtastrate {rate} Hz zu niedrig (mindestens 16000 Hz)")
    if channels > 2:
        problems.append(f"{channels} Kanäle – Mono oder Stereo erwartet")
    return SampleInfo(
        path=path,
        seconds=seconds,
        sample_rate=rate,
        channels=channels,
        usable=not problems,
        note="; ".join(problems),
    )


def add_sample(profile: VoiceProfile, source: Path | str) -> SampleInfo:
    """Aufnahme in das Profil kopieren (Original bleibt unangetastet)."""
    src = Path(source).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"Aufnahme nicht gefunden: {src}")
    if src.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Format {src.suffix} wird nicht unterstützt. Erlaubt: {', '.join(SUPPORTED_SUFFIXES)}"
        )
    paths.ensure_dir(profile.samples_dir)
    target = profile.samples_dir / src.name
    counter = 1
    while target.exists():
        target = profile.samples_dir / f"{src.stem}-{counter}{src.suffix}"
        counter += 1
    shutil.copy2(src, target)
    info = inspect_sample(target)
    profile.save()
    return info


def remove_sample(profile: VoiceProfile, name: str) -> bool:
    target = profile.samples_dir / name
    if not target.is_file():
        return False
    target.unlink()
    profile.save()
    return True


# ---------------------------------------------------------------------------
# Anlernen (Attrappe – Schnittstelle steht)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrainingRequest:
    slug: str
    epochs: int = 20
    batch_size: int = 8
    learning_rate: float = 1e-4
    keep_intermediate: bool = False


@dataclass(frozen=True)
class TrainingResult:
    slug: str
    artifact: Path
    seconds_used: float
    mode: TrainingMode
    elapsed_s: float
    backend: str = "cpu"
    dummy: bool = True


# Chatterbox arbeitet am besten mit einer sauberen Referenz von 7–20 s.
# Mehr Material hilft kaum, kostet aber Rechenzeit bei jedem Satz.
REFERENCE_MAX_SECONDS = 30.0
REFERENCE_SAMPLE_RATE = 24000
MAX_REFERENCE_SAMPLES = 5  # mehr Spuren bringen nichts, machen ffmpeg nur langsam


def build_reference(profile: VoiceProfile, context=None) -> tuple[Path, float, list[str]]:
    """Referenzaufnahme für das Klonen erzeugen.

    Das ist die eigentliche Arbeit beim Zero-Shot-Verfahren: aus dem
    Rohmaterial eine saubere, einkanalige, normalisierte Aufnahme machen.
    Läuft über das mitgelieferte ffmpeg; fehlt es, wird die beste
    WAV-Aufnahme unverändert übernommen.
    """
    from . import compose

    notes: list[str] = []
    usable = [s for s in profile.samples() if s.usable]
    if not usable:
        raise RuntimeError("Keine brauchbare Aufnahme vorhanden.")

    # Mehrere Aufnahmen ergeben eine bessere Referenz als eine einzige:
    # unterschiedliche Sätze decken mehr Laute und Tonhöhen ab. Genommen
    # werden die längsten, bis das Ziel erreicht ist.
    target_seconds = max(
        MIN_TOTAL_SECONDS_CLONE, min(float(profile.reference_seconds), REFERENCE_MAX_SECONDS)
    )
    ranked = sorted(usable, key=lambda s: s.seconds or 0.0, reverse=True)
    chosen: list[SampleInfo] = []
    collected = 0.0
    for sample in ranked[:MAX_REFERENCE_SAMPLES]:
        chosen.append(sample)
        collected += sample.seconds or 0.0
        if collected >= target_seconds:
            break

    paths.ensure_dir(profile.artifacts_dir)
    target = profile.artifacts_dir / "reference.wav"

    try:
        compose.probe()
    except compose.FfmpegMissing:
        source = chosen[0]
        if source.path.suffix.lower() != ".wav":
            raise RuntimeError("Ohne ffmpeg können nur WAV-Aufnahmen verwendet werden.") from None
        shutil.copy2(source.path, target)
        notes.append("ffmpeg fehlt – eine Aufnahme wurde unverändert übernommen.")
        return target, source.seconds, notes

    # Stille am Anfang/Ende weg, auf Mono und feste Rate, Lautheit angleichen.
    cleanup = (
        "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-45dB:"
        "stop_periods=-1:stop_duration=0.4:stop_threshold=-45dB,"
        "loudnorm=I=-18:TP=-2:LRA=9,"
        f"aresample={REFERENCE_SAMPLE_RATE}"
    )
    args: list[str] = []
    for sample in chosen:
        args += ["-i", str(sample.path)]

    if len(chosen) == 1:
        args += ["-af", cleanup]
    else:
        # Erst jede Spur einzeln auf Mono und gleiche Rate bringen, dann
        # aneinanderhängen – sonst lehnt concat unterschiedliche Formate ab.
        parts = "".join(
            f"[{index}:a]aformat=channel_layouts=mono,aresample={REFERENCE_SAMPLE_RATE}[a{index}];"
            for index in range(len(chosen))
        )
        inputs = "".join(f"[a{index}]" for index in range(len(chosen)))
        graph = f"{parts}{inputs}concat=n={len(chosen)}:v=0:a=1[joined];[joined]{cleanup}[out]"
        args += ["-filter_complex", graph, "-map", "[out]"]

    args += ["-t", str(target_seconds), "-ac", "1", "-c:a", "pcm_s16le", str(target)]
    compose.run_ffmpeg(args, context=context, label="Referenz aufbereiten")

    quelle = ", ".join(s.path.name for s in chosen)
    notes.append(
        f"Referenz aus {len(chosen)} Aufnahme(n) ({quelle}): Stille entfernt, "
        f"Mono, {REFERENCE_SAMPLE_RATE} Hz, auf {target_seconds:.0f} s begrenzt."
    )
    length = collected
    try:
        with contextlib.closing(wave.open(str(target), "rb")) as handle:
            length = handle.getnframes() / float(handle.getframerate() or 1)
    except (wave.Error, OSError):
        pass
    return target, length, notes


def train_profile(request: TrainingRequest, context, backend: str = "cpu") -> TrainingResult:
    """Stimme anlernen.

    Zero-Shot (Chatterbox/OpenVoice): es wird nicht nachtrainiert, sondern
    eine saubere Referenzaufnahme erzeugt – genau die braucht das Modell
    zur Laufzeit. Das ist kein Platzhalter, sondern das Verfahren.

    Finetune: echtes Nachtrainieren ist noch nicht umgesetzt; der Auftrag
    bricht mit klarer Meldung ab, statt ein wertloses Artefakt zu schreiben.
    """
    profile = load_profile(request.slug)
    if profile is None:
        raise FileNotFoundError(f"Stimmprofil '{request.slug}' nicht gefunden.")

    ready, problems = profile.training_ready()
    if not ready:
        raise RuntimeError("Anlernen nicht möglich: " + " | ".join(problems))

    started = time.time()
    context.status(f"Bereite '{profile.display_name}' auf …")

    if profile.mode is TrainingMode.FINETUNE:
        raise RuntimeError(
            "Echtes Nachtrainieren ist noch nicht umgesetzt. Profil auf "
            "'zero_shot' umstellen – dafür genügen 10–20 Sekunden Material, "
            "und das Ergebnis ist sofort nutzbar."
        )

    context.progress(0.2, "Aufnahmen prüfen")
    reference, seconds, notes = build_reference(profile, context)
    context.progress(0.8, "Kenndaten schreiben")

    artifact = profile.artifacts_dir / "speaker.json"
    artifact.write_text(
        json.dumps(
            {
                "slug": profile.slug,
                "display_name": profile.display_name,
                "mode": profile.mode.value,
                "model_key": profile.model_key,
                "reference": reference.name,
                "reference_seconds": round(seconds, 2),
                "trained_at": time.time(),
                "backend": backend,
                "consent_speaker": profile.consent.speaker_name if profile.consent else "",
                "notes": notes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    profile.artifact = artifact.name
    profile.save()
    for note in notes:
        context.log(note)
    context.progress(1.0, "fertig")

    return TrainingResult(
        slug=profile.slug,
        artifact=reference,
        seconds_used=seconds,
        mode=profile.mode,
        elapsed_s=time.time() - started,
        backend=backend,
        dummy=False,
    )


def describe_profiles() -> str:
    """Übersicht für CLI und GUI."""
    profiles = list_profiles()
    if not profiles:
        return "Keine Stimmprofile vorhanden."
    lines: list[str] = []
    for profile in profiles:
        usable, reason = profile.usable_for_synthesis()
        lines.append(
            f"[{profile.state.label()}] {profile.display_name} ({profile.slug}) – "
            f"{profile.mode.value}, {profile.total_seconds():.1f}s Material"
        )
        speaker = profile.consent.speaker_name if profile.consent else "kein Nachweis"
        lines.append(f"    Einwilligung: {speaker}")
        lines.append(f"    Nutzbar:      {'ja' if usable else 'nein'} – {reason}")
    return "\n".join(lines)
