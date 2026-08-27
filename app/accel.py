"""Hardware-Erkennung, DLL-Suchpfad und Backend-Kette.

Zwei Aufgaben:

1. ``prepare_gpu_dll_path()`` – MUSS vor dem ersten Import von torch,
   diffusers oder onnxruntime laufen. Windows findet die CUDA-DLLs sonst
   nicht, weil sie im Bundle neben der .exe bzw. in den ``nvidia-*-cu12``
   Pip-Paketen liegen und nicht im PATH stehen.
2. Erkennung von GPU / NPU / CPU inklusive VRAM und Ableitung einer
   Eignungsstufe. Alle Sonden sind fail-soft: fehlt ein Werkzeug
   (``nvidia-smi``, PowerShell, OpenVINO), ist das kein Fehler, sondern
   nur eine fehlende Information.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import json
import logging
import os
import platform
import re
import subprocess
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import Any

from . import paths

log = logging.getLogger(__name__)

NVIDIA_SMI_TIMEOUT = 2.0
POWERSHELL_TIMEOUT = 6.0
ERROR_TEXT_LIMIT = 240

# Fassung des Hardware-Zwischenspeichers auf der Platte. Hochzählen, sobald
# sich das Format ändert – ein alter Eintrag wird dann verworfen statt
# falsch gedeutet.
HARDWARE_CACHE_VERSION = 2

# Idempotenz-Schalter: prepare_gpu_dll_path() darf beliebig oft aufgerufen
# werden, arbeitet aber nur beim ersten Mal.
_prepared = False
_prepared_dirs: list[str] = []


# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------
def _creation_flags() -> int:
    """CREATE_NO_WINDOW – sonst blitzt bei jeder Sonde ein Konsolenfenster auf."""
    if os.name == "nt":
        return 0x08000000
    return 0


def clean_error(error: BaseException | str, limit: int = ERROR_TEXT_LIMIT) -> str:
    """Fremdbibliotheks-Fehler für die Anzeige säubern.

    Zeilenumbrüche raus, Mehrfach-Leerzeichen zusammenziehen, auf ``limit``
    Zeichen kürzen. Tracebacks von diffusers/onnxruntime sind sonst
    hunderte Zeichen lang und zerstören jedes Layout.

    Ausnahme: eigene Ablehnungen, die ``expected`` tragen (Inhaltssperre,
    Lizenztor, fehlender Repo-Zugang). Deren Wortlaut ist absichtlich
    gesetzt und enthält die Handlungsanweisung – Zeilen zusammenzuziehen
    und nach 240 Zeichen abzuschneiden würde genau den Teil abschneiden,
    der dem Bediener sagt, was zu tun ist.
    """
    if isinstance(error, BaseException) and getattr(error, "expected", False):
        return str(error).strip()
    text = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def _run(cmd: Sequence[str], timeout: float) -> tuple[int, str, str]:
    """Externes Programm aufrufen. Nie werfen – Rückgabe (code, out, err)."""
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=_creation_flags(),
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except FileNotFoundError:
        return 127, "", f"{cmd[0]} nicht gefunden"
    except subprocess.TimeoutExpired:
        return 124, "", f"{cmd[0]} hat nach {timeout:g}s nicht geantwortet"
    except OSError as exc:  # z. B. fehlende Rechte
        return 1, "", clean_error(exc)


# ---------------------------------------------------------------------------
# 1. DLL-Suchpfad
# ---------------------------------------------------------------------------
def _pip_nvidia_dirs() -> Iterator[Path]:
    """bin/lib-Ordner der ``nvidia-*-cu12`` Pip-Pakete (Entwicklungsmodus)."""
    try:
        spec = importlib.util.find_spec("nvidia")
    except (ImportError, ValueError):
        return
    if spec is None:
        return
    roots = [Path(p) for p in (spec.submodule_search_locations or [])]
    for root in roots:
        if not root.is_dir():
            continue
        for component in sorted(root.iterdir()):
            if not component.is_dir():
                continue
            for leaf in ("bin", "lib", Path("lib") / "x64"):
                candidate = component / leaf
                if candidate.is_dir():
                    yield candidate


def _torch_lib_dir() -> Iterator[Path]:
    """torch/lib enthält cudnn/cublas, wenn torch als cu-Wheel installiert ist."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return
    if spec is None or not spec.submodule_search_locations:
        return
    for location in spec.submodule_search_locations:
        candidate = Path(location) / "lib"
        if candidate.is_dir():
            yield candidate


def _candidate_dirs() -> Iterator[Path]:
    """Kandidaten in beiden Welten: gebündelt neben der .exe UND Pip-Pakete."""
    # Bundle – build-windows.ps1 legt die Laufzeit nach exe_dir/cuda ab.
    yield paths.exe_dir / "cuda"
    yield paths.exe_dir / "_internal" / "cuda"
    yield paths.exe_dir / "_internal" / "torch" / "lib"
    if paths.bundle_dir != paths.exe_dir:
        yield paths.bundle_dir / "cuda"
        yield paths.bundle_dir / "torch" / "lib"
    # Entwicklung
    yield from _pip_nvidia_dirs()
    yield from _torch_lib_dir()


def prepare_gpu_dll_path() -> list[str]:
    """DLL-Suchpfad für CUDA/cuDNN setzen. Idempotent, nie werfend."""
    global _prepared
    if _prepared:
        return []
    added: list[str] = []
    seen: set[str] = set()
    for directory in _candidate_dirs():
        key = str(directory).lower()
        if key in seen:
            continue
        seen.add(key)
        if not directory.is_dir():
            continue
        # AttributeError: Nicht-Windows. OSError: Pfad verschwunden. In
        # beiden Fällen bleibt der PATH-Eintrag darunter trotzdem richtig.
        with contextlib.suppress(OSError, AttributeError):
            os.add_dll_directory(str(directory))
        os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
        added.append(str(directory))
    _prepared = True
    _prepared_dirs.extend(added)
    if added:
        log.debug("DLL-Suchpfad erweitert: %s", "; ".join(added))
    return added


def prepared_dll_dirs() -> list[str]:
    """Was prepare_gpu_dll_path() tatsächlich hinzugefügt hat (Diagnose)."""
    return list(_prepared_dirs)


# ---------------------------------------------------------------------------
# 2. Datenmodell der Hardware-Erkennung
# ---------------------------------------------------------------------------
class Vendor(str):
    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    APPLE = "apple"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GpuDevice:
    index: int
    name: str
    vendor: str = Vendor.UNKNOWN
    total_vram_mb: int = 0
    source: str = ""  # woher die Information kommt (nvidia-smi, cim, torch)

    @property
    def vram_gb(self) -> float:
        return round(self.total_vram_mb / 1024.0, 1)

    def label(self) -> str:
        vram = f"{self.vram_gb:g} GB VRAM" if self.total_vram_mb else "VRAM unbekannt"
        return f"[{self.index}] {self.name} – {vram}"


@dataclass(frozen=True)
class NpuDevice:
    name: str
    source: str = ""


@dataclass(frozen=True)
class CpuInfo:
    name: str = "unbekannt"
    cores_physical: int = 0
    cores_logical: int = 0
    ram_mb: int = 0
    arch: str = ""

    @property
    def ram_gb(self) -> float:
        return round(self.ram_mb / 1024.0, 1)

    def label(self) -> str:
        return f"{self.name} – {self.cores_logical or '?'} Threads, {self.ram_gb:g} GB RAM"


class CapabilityTier(IntEnum):
    """Eignungsstufe. Bestimmt Modellvorschläge und Warnungen VOR dem Download."""

    CPU_ONLY = 0  # keine nutzbare GPU
    ENTRY = 1  # ~4–6 GB VRAM
    MID = 2  # ~8–11 GB VRAM
    HIGH = 3  # ~12–23 GB VRAM
    ULTRA = 4  # ab 24 GB VRAM


@dataclass(frozen=True)
class TierAdvice:
    tier: CapabilityTier
    title: str
    image_max_side: int
    video_ok: bool
    video_max_frames: int
    voice_ok: bool
    text: str


_TIER_ADVICE: dict[CapabilityTier, TierAdvice] = {
    CapabilityTier.CPU_ONLY: TierAdvice(
        tier=CapabilityTier.CPU_ONLY,
        title="Nur CPU",
        image_max_side=768,
        video_ok=False,
        video_max_frames=0,
        voice_ok=True,
        text=(
            "Keine nutzbare GPU erkannt. Bilder sind möglich, brauchen aber "
            "Minuten pro Bild – nutze ein kleines Modell und wenige Schritte. "
            "Video ist auf der CPU nicht sinnvoll (Stunden pro Clip). "
            "Sprache läuft auf der CPU gut."
        ),
    ),
    CapabilityTier.ENTRY: TierAdvice(
        tier=CapabilityTier.ENTRY,
        title="Einstieg (4–6 GB VRAM)",
        image_max_side=768,
        video_ok=False,
        video_max_frames=0,
        voice_ok=True,
        text=(
            "GPU vorhanden, aber knapper VRAM. Bilder bis 768 px laufen mit "
            "einem kompakten Modell. Video nur mit sehr kurzen Clips und "
            "aktivem VRAM-Sparmodus – rechne mit Abbrüchen wegen Speicher."
        ),
    ),
    CapabilityTier.MID: TierAdvice(
        tier=CapabilityTier.MID,
        title="Mittelklasse (8–11 GB VRAM)",
        image_max_side=1024,
        video_ok=True,
        video_max_frames=49,
        voice_ok=True,
        text=(
            "Solide Basis. Bilder bis 1024 px, kurze Videos (2–3 Sekunden) "
            "mit einem kleinen Videomodell. VRAM-Sparmodus anlassen."
        ),
    ),
    CapabilityTier.HIGH: TierAdvice(
        tier=CapabilityTier.HIGH,
        title="Oberklasse (12–23 GB VRAM)",
        image_max_side=1536,
        video_ok=True,
        video_max_frames=81,
        voice_ok=True,
        text=(
            "Bilder in hoher Auflösung, Videos mehrere Sekunden. "
            "Große Bildmodelle laufen, ggf. mit Auslagerung einzelner Teile."
        ),
    ),
    CapabilityTier.ULTRA: TierAdvice(
        tier=CapabilityTier.ULTRA,
        title="Profi (ab 24 GB VRAM)",
        image_max_side=2048,
        video_ok=True,
        video_max_frames=161,
        voice_ok=True,
        text=(
            "Alle mitgelieferten Modelle laufen ohne Kompromisse. "
            "VRAM-Sparmodus kann für mehr Geschwindigkeit aus bleiben."
        ),
    ),
}


def tier_for_vram(total_vram_mb: int) -> CapabilityTier:
    """Eignungsstufe aus dem VRAM der stärksten GPU ableiten."""
    if total_vram_mb >= 24_000:
        return CapabilityTier.ULTRA
    if total_vram_mb >= 12_000:
        return CapabilityTier.HIGH
    if total_vram_mb >= 8_000:
        return CapabilityTier.MID
    if total_vram_mb >= 3_500:
        return CapabilityTier.ENTRY
    return CapabilityTier.CPU_ONLY


def advice_for_tier(tier: CapabilityTier) -> TierAdvice:
    return _TIER_ADVICE[tier]


@dataclass(frozen=True)
class HardwareReport:
    gpus: tuple[GpuDevice, ...] = ()
    npus: tuple[NpuDevice, ...] = ()
    cpu: CpuInfo = field(default_factory=CpuInfo)
    os_name: str = ""
    notes: tuple[str, ...] = ()  # Klartext-Meldungen, auch über Fehlschläge

    # --- abgeleitete Angaben ------------------------------------------------
    @property
    def best_gpu(self) -> GpuDevice | None:
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda gpu: (gpu.total_vram_mb, -gpu.index))

    @property
    def nvidia_gpus(self) -> tuple[GpuDevice, ...]:
        return tuple(g for g in self.gpus if g.vendor == Vendor.NVIDIA)

    @property
    def tier(self) -> CapabilityTier:
        best = self.best_gpu
        if best is None:
            return CapabilityTier.CPU_ONLY
        return tier_for_vram(best.total_vram_mb)

    @property
    def advice(self) -> TierAdvice:
        return advice_for_tier(self.tier)


# ---------------------------------------------------------------------------
# 3. Sonden
# ---------------------------------------------------------------------------
def _vendor_from_name(name: str) -> str:
    lowered = name.lower()
    if "nvidia" in lowered or "geforce" in lowered or "quadro" in lowered or "tesla" in lowered:
        return Vendor.NVIDIA
    if "amd" in lowered or "radeon" in lowered or "gfx" in lowered:
        return Vendor.AMD
    if "intel" in lowered or "arc" in lowered or "iris" in lowered or "uhd" in lowered:
        return Vendor.INTEL
    if "apple" in lowered:
        return Vendor.APPLE
    return Vendor.UNKNOWN


def probe_nvidia_smi(timeout: float = NVIDIA_SMI_TIMEOUT) -> tuple[list[GpuDevice], str]:
    """NVIDIA-GPUs samt VRAM über nvidia-smi. Fehlt das Werkzeug: kein Fehler."""
    code, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        timeout,
    )
    if code != 0:
        if code == 127:
            return [], "nvidia-smi nicht vorhanden – kein NVIDIA-Treiber installiert."
        return [], f"nvidia-smi fehlgeschlagen: {clean_error(err or out or 'unbekannt')}"

    devices: list[GpuDevice] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
            vram = int(float(parts[2]))
        except ValueError:
            continue
        devices.append(
            GpuDevice(
                index=index,
                name=parts[1],
                vendor=Vendor.NVIDIA,
                total_vram_mb=vram,
                source="nvidia-smi",
            )
        )
    if not devices:
        return [], "nvidia-smi lieferte keine GPU-Zeile."
    return devices, ""


# Virtuelle Anzeigegeräte tauchen als Grafikkarte auf, rechnen aber nichts.
_VIRTUAL_ADAPTER_HINTS = (
    "virtual",
    "remote",
    "basic render",
    "basic display",
    "idd",
    "parsec",
    "meta ",
    "oculus",
    "spacedesk",
    "citrix",
    "vnc",
    "mirror",
    "duet",
)


def _is_virtual_adapter(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _VIRTUAL_ADAPTER_HINTS)


# Wortgrenzen sind Pflicht: 'NPU' steckt sonst als Teilstring in
# 'USB Input Device' und liefert falsche Treffer.
_NPU_PATTERN = (
    r"(^|[^A-Za-z])NPU([^A-Za-z]|$)|AI Boost|Neural Processor|"
    r"Neural Engine|Movidius|(^|[^A-Za-z])VPU([^A-Za-z]|$)"
)

# Ein PowerShell-Start kostet je nach Rechner 0,3–1,5 s. Grafikkarten und
# NPUs wurden früher in zwei getrennten Prozessen abgefragt – das war die
# Hälfte der Startzeit für zwei Zeilen Text. Jetzt: ein Prozess, ein
# Ergebnis, für die Dauer des Programmlaufs gemerkt.
_windows_devices_cache: tuple[list[GpuDevice], list[NpuDevice], list[str]] | None = None


def _probe_windows_devices(
    refresh: bool = False,
) -> tuple[list[GpuDevice], list[NpuDevice], list[str]]:
    """Grafikkarten und NPUs in EINEM PowerShell-Aufruf holen.

    AdapterRAM ist bei >4 GB unbrauchbar und wird deshalb nur als
    Untergrenze verwendet.
    """
    global _windows_devices_cache
    if _windows_devices_cache is not None and not refresh:
        return _windows_devices_cache
    if os.name != "nt":
        _windows_devices_cache = ([], [], [])
        return _windows_devices_cache

    script = (
        "$ErrorActionPreference='SilentlyContinue';"
        "Get-CimInstance Win32_VideoController | "
        'ForEach-Object { "GPU|$($_.Name)|$($_.AdapterRAM)" };'
        "Get-CimInstance Win32_PnPEntity | "
        f"Where-Object {{ $_.Name -match '{_NPU_PATTERN}' }} | "
        'ForEach-Object { "NPU|$($_.Name)" }'
    )
    code, out, err = _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        POWERSHELL_TIMEOUT,
    )
    if code != 0:
        note = f"Geräte-Abfrage fehlgeschlagen: {clean_error(err or out)}"
        _windows_devices_cache = ([], [], [note])
        return _windows_devices_cache

    gpus: list[GpuDevice] = []
    npus: list[NpuDevice] = []
    for line in out.splitlines():
        line = line.strip()
        kind, _, rest = line.partition("|")
        if not rest:
            continue
        if kind == "NPU":
            npus.append(NpuDevice(name=rest.strip(), source="pnp"))
            continue
        if kind != "GPU":
            continue
        name, _, ram_text = rest.partition("|")
        name = name.strip()
        if not name or _is_virtual_adapter(name):
            continue
        vram_mb = 0
        try:
            ram_bytes = int(ram_text.strip() or 0)
            if ram_bytes > 0:
                vram_mb = int(ram_bytes / (1024 * 1024))
        except ValueError:
            vram_mb = 0
        gpus.append(
            GpuDevice(
                index=len(gpus),
                name=name,
                vendor=_vendor_from_name(name),
                total_vram_mb=vram_mb,
                source="cim",
            )
        )
    _windows_devices_cache = (gpus, npus, [])
    return _windows_devices_cache


def _probe_windows_display_adapters(refresh: bool = False) -> tuple[list[GpuDevice], str]:
    """AMD/Intel-GPUs über CIM."""
    gpus, _npus, notes = _probe_windows_devices(refresh=refresh)
    return gpus, "; ".join(notes)


def npu_outlook(cpu_name: str) -> str:
    """Kann dieser Prozessor überhaupt eine NPU haben?

    „Keine erkannt" beantwortet die falsche Frage. Der Bediener will
    wissen, ob die Erkennung versagt oder ob schlicht keine NPU verbaut
    ist – das sind zwei völlig verschiedene Lagen, und nur eine davon ist
    ein Fehler.

    Geprüft werden die eindeutigen Namensmerkmale der Baureihen, die eine
    NPU mitbringen. Ist keines dabei, wird das als „vermutlich keine"
    gemeldet und nicht als Tatsache – Prozessornamen sind keine
    verlässliche Bauteilliste.
    """
    raw = cpu_name or ""
    if not raw.strip():
        return "Prozessor unbekannt – keine Aussage möglich."
    # Marken-Beiwerk entfernen: der echte Name lautet "Intel(R) Core(TM)
    # Ultra 7 155H", dort steht "(TM)" mitten zwischen "Core" und "Ultra".
    # Ohne diese Bereinigung greift keine Suche nach "core ultra".
    name = re.sub(r"\((?:r|tm)\)|™|®", " ", raw.lower())
    name = " ".join(name.split())
    if "core ultra" in name or "core(tm) ultra" in name:
        return "Core Ultra bringt eine NPU mit (Intel AI Boost)."
    if "ryzen ai" in name:
        return "Ryzen AI bringt eine NPU mit (AMD XDNA)."
    if "snapdragon" in name:
        return "Snapdragon X bringt eine NPU mit (Hexagon)."
    if re.search(r"ryzen.*\b[78]\d{3}(hs|u|h)\b", name):
        return "Diese Ryzen-Baureihe kann eine NPU haben (XDNA, je nach Modell)."
    if "intel" in name or "core" in name:
        return (
            "Diese Intel-Baureihe hat keine NPU – die gibt es erst ab "
            "Core Ultra (Meteor Lake, Ende 2023)."
        )
    if "ryzen" in name or "amd" in name:
        return "Diese AMD-Baureihe hat keine NPU – die gibt es ab Ryzen 7040 bzw. Ryzen AI."
    return "Vermutlich keine NPU verbaut."


def npu_reason(cpu_name: str) -> str:
    """Klartext, warum keine NPU gemeldet wird – Hardware oder Treiber."""
    outlook = npu_outlook(cpu_name)
    if importlib.util.find_spec("openvino") is None:
        outlook += (
            " Für Intel-NPUs wird zusätzlich OpenVINO gebraucht ('pip install "
            "openvino'); ohne das bleibt sie auch dann unsichtbar, wenn sie da ist."
        )
    return outlook


def probe_npus(refresh: bool = False) -> tuple[list[NpuDevice], str]:
    """NPU-Erkennung: Windows-PnP-Namen und – falls installiert – OpenVINO."""
    _gpus, pnp, probe_notes = _probe_windows_devices(refresh=refresh)
    found: list[NpuDevice] = list(pnp)
    notes: list[str] = list(probe_notes)

    if importlib.util.find_spec("openvino") is not None:
        try:  # OpenVINO ist optional – Import darf den Start nicht kippen
            import openvino as ov  # type: ignore

            for device in ov.Core().available_devices:
                if str(device).upper().startswith("NPU"):
                    found.append(NpuDevice(name=f"OpenVINO {device}", source="openvino"))
        except Exception as exc:
            notes.append(f"OpenVINO-Abfrage fehlgeschlagen: {clean_error(exc)}")

    # Doppelte Namen zusammenfassen
    unique: dict[str, NpuDevice] = {}
    for device in found:
        unique.setdefault(device.name.lower(), device)
    return list(unique.values()), "; ".join(notes)


def npu_diagnosis(refresh: bool = True) -> str:
    """Vollständiger Bericht zur NPU – für die Ferndiagnose.

    Zeigt jede Stufe einzeln: Prozessor, rohe Gerätenamen aus Windows,
    OpenVINO, ONNX-Provider. So ist ablesbar, **wo** es hakt – ob Windows
    das Gerät nicht meldet, der Treiber fehlt oder nur die Laufzeit nicht
    installiert ist. „Keine erkannt" allein sagt das nicht.
    """
    lines: list[str] = ["== NPU-Diagnose ==", ""]

    cpu = probe_cpu()
    lines.append(f"Prozessor:   {cpu.name}")
    lines.append(f"Einschätzung: {npu_outlook(cpu.name)}")
    lines.append("")

    lines.append("-- Windows-Geräte (PnP) --")
    _gpus, pnp, notes = _probe_windows_devices(refresh=refresh)
    if pnp:
        for device in pnp:
            lines.append(f"  gefunden: {device.name}")
    else:
        lines.append("  kein Gerät, dessen Name auf eine NPU passt.")
        lines.append(f"  Suchmuster: {_NPU_PATTERN}")
        lines.append(
            "  Falls die NPU im Gerätemanager unter anderem Namen steht, "
            "diesen Namen melden – dann wird das Muster erweitert."
        )
    for note in notes:
        lines.append(f"  Hinweis: {note}")
    lines.append("")

    lines.append("-- OpenVINO (Weg zur Intel-NPU) --")
    if importlib.util.find_spec("openvino") is None:
        lines.append("  nicht installiert. Ohne OpenVINO bleibt eine Intel-NPU unsichtbar.")
        lines.append("  Nachrüsten: pip install openvino")
    else:
        try:
            import openvino as ov  # type: ignore

            devices = list(ov.Core().available_devices)
            lines.append(f"  Fassung: {getattr(ov, '__version__', 'unbekannt')}")
            lines.append(f"  Geräte:  {', '.join(devices) or 'keine'}")
            if not any(str(d).upper().startswith("NPU") for d in devices):
                lines.append("  Kein NPU-Gerät dabei – meist fehlt der NPU-Treiber von Intel.")
        except Exception as exc:
            lines.append(f"  Abfrage fehlgeschlagen: {clean_error(exc)}")
    lines.append("")

    lines.append("-- ONNX Runtime --")
    providers, note = onnx_providers(refresh=refresh)
    lines.append(f"  Provider: {', '.join(providers) or 'keine'}")
    if note:
        lines.append(f"  Hinweis:  {note}")
    lines.append("")

    lines.append("-- Rechenpfad --")
    try:
        from . import pipeline_onnx

        for line in pipeline_onnx.describe().splitlines():
            lines.append(f"  {line}")
    except Exception as exc:  # pragma: no cover – Diagnose darf nie kippen
        lines.append(f"  Laufzeit-Prüfung fehlgeschlagen: {clean_error(exc)}")
    lines.append("")
    import sys

    laufzeit = (
        "ein Bau mit '-WithOnnx $true' (im gebauten Programm wirkt 'pip install' nicht)"
        if getattr(sys, "frozen", False)
        else "die Laufzeit ('pip install \"optimum[openvino]\" openvino')"
    )
    lines.append(
        f"  Die NPU wird über OpenVINO angesprochen. Nötig sind: {laufzeit}, "
        "ein einmaliger Export ('streamforge models convert <modell> "
        "--backend openvino') und Gerät = 'openvino' in den Einstellungen."
    )
    lines.append(
        "  Zur Erwartung: eine NPU ist auf kleine, quantisierte Netze "
        "ausgelegt. Sie ist sparsam, aber bei Diffusionsmodellen langsamer "
        "als eine dedizierte Grafikkarte. Wo eine Intel-iGPU vorhanden ist, "
        "lohnt oft 'GPU' statt 'NPU'."
    )
    return "\n".join(lines)


def _windows_ram_mb() -> int:
    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(MemoryStatusEx)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys / (1024 * 1024))
    except (OSError, AttributeError):
        pass
    return 0


def _cpu_name_windows() -> str:
    try:
        import winreg  # nur unter Windows vorhanden

        key_path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
    except (OSError, ImportError, ValueError):
        return ""


def probe_cpu() -> CpuInfo:
    """CPU-Name, Kerne und RAM – ohne Fremdbibliothek."""
    name = ""
    if os.name == "nt":
        name = _cpu_name_windows()
    if not name:
        name = platform.processor() or platform.machine() or "unbekannt"

    logical = os.cpu_count() or 0
    physical = 0
    try:
        # Python 3.13+: bevorzugte Zählung. Vorher: Fallback auf logisch.
        physical = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except AttributeError:
        physical = 0

    ram_mb = 0
    if os.name == "nt":
        ram_mb = _windows_ram_mb()
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            ram_mb = int(page_size * pages / (1024 * 1024))
        except (ValueError, OSError, AttributeError):
            ram_mb = 0

    return CpuInfo(
        name=name,
        cores_physical=physical or logical,
        cores_logical=logical,
        ram_mb=ram_mb,
        arch=platform.machine(),
    )


# --- Laufzeit-Sonden (importieren Fremdbibliotheken, daher gecacht) --------
_torch_cuda_cache: tuple[bool, str] | None = None
_onnx_cache: tuple[tuple[str, ...], str] | None = None


def torch_cuda_hint() -> tuple[bool, str]:
    """Schnelle Vermutung, OHNE torch zu importieren.

    ``import torch`` kostet beim ersten Start dieser Sitzung bis zu 18
    Sekunden – so lange darf das Fenster nicht auf sich warten lassen. Die
    Auskunft steckt aber schon in ``torch/version.py``: dort trägt jedes
    Wheel seine CUDA-Fassung ein (``cuda = '12.8'``) bzw. ``None`` beim
    CPU-Wheel. Das ist eine Textdatei von wenigen hundert Byte.

    Ist bereits ein geprüftes Ergebnis vorhanden, gewinnt dieses – die
    Vermutung ist nur die Auskunft für den Start.
    """
    if _torch_cuda_cache is not None:
        return _torch_cuda_cache
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return False, "torch ist nicht installiert."

    for location in spec.submodule_search_locations or []:
        version_file = Path(location) / "version.py"
        if not version_file.is_file():
            continue
        try:
            text = version_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(r"^\s*cuda\s*(?::[^=]+)?=\s*(.+)$", text, re.MULTILINE)
        if match is None:
            continue
        value = match.group(1).strip()
        if value.startswith(("'", '"')):
            return True, f"torch-Wheel mit CUDA {value.strip(chr(39) + chr(34))} (vorläufig)."
        return False, "torch ist als CPU-Wheel installiert (kein CUDA)."

    # Keine Auskunft: torch ist da, die Fassung unbekannt. Optimistisch
    # weitermachen und die Prüfung dem Hintergrund überlassen – ein
    # falsches Ja wird beim Laden des Modells abgefangen.
    return True, "torch vorhanden, CUDA-Fassung noch nicht geprüft (vorläufig)."


def torch_cuda_verified() -> bool:
    """Wurde ``torch.cuda.is_available()`` in diesem Lauf schon wirklich gefragt?"""
    return _torch_cuda_cache is not None


def torch_cuda_available(refresh: bool = False) -> tuple[bool, str]:
    """Prüft torch + CUDA. Importiert torch – erst bei Bedarf aufrufen."""
    global _torch_cuda_cache
    if _torch_cuda_cache is not None and not refresh:
        return _torch_cuda_cache
    if importlib.util.find_spec("torch") is None:
        _torch_cuda_cache = (False, "torch ist nicht installiert.")
        return _torch_cuda_cache
    prepare_gpu_dll_path()
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            reason = (
                "torch ist installiert, meldet aber keine CUDA-GPU "
                "(CPU-Wheel oder fehlender NVIDIA-Treiber)."
            )
            _torch_cuda_cache = (False, reason)
        else:
            count = torch.cuda.device_count()
            _torch_cuda_cache = (True, f"CUDA verfügbar, {count} Gerät(e).")
    except Exception as exc:
        _torch_cuda_cache = (False, f"torch-Import fehlgeschlagen: {clean_error(exc)}")
    return _torch_cuda_cache


def onnx_providers(refresh: bool = False) -> tuple[tuple[str, ...], str]:
    """Verfügbare ONNX-Runtime-Provider (für DirectML auf AMD/Intel/NPU)."""
    global _onnx_cache
    if _onnx_cache is not None and not refresh:
        return _onnx_cache
    if importlib.util.find_spec("onnxruntime") is None:
        _onnx_cache = ((), "onnxruntime ist nicht installiert.")
        return _onnx_cache
    prepare_gpu_dll_path()
    try:
        import onnxruntime as ort  # type: ignore

        providers = tuple(ort.get_available_providers())
        _onnx_cache = (providers, "")
    except Exception as exc:
        _onnx_cache = ((), f"onnxruntime-Import fehlgeschlagen: {clean_error(exc)}")
    return _onnx_cache


def directml_available() -> tuple[bool, str]:
    providers, note = onnx_providers()
    if note:
        return False, note
    if "DmlExecutionProvider" in providers:
        return True, "DirectML-Provider vorhanden."
    return False, (
        "Kein DmlExecutionProvider – Paket onnxruntime-directml fehlt oder Windows-Version zu alt."
    )


# ---------------------------------------------------------------------------
# 4. Gesamtbericht
# ---------------------------------------------------------------------------
_report_cache: HardwareReport | None = None
_report_from_disk = False


def _hardware_cache_path() -> Path:
    return paths.data_dir() / "hardware-cache.json"


def _machine_fingerprint(cpu: CpuInfo) -> str:
    """Kennung der Maschine aus billig ermittelbaren Angaben.

    Reicht, um einen Zwischenspeicher zu verwerfen, der von einem anderen
    Rechner stammt (portable Installation auf einem USB-Stick).
    """
    return "|".join(
        (
            platform.system(),
            platform.release(),
            cpu.name,
            str(cpu.cores_logical),
            str(cpu.ram_mb),
        )
    )


def _report_to_dict(report: HardwareReport, fingerprint: str) -> dict[str, Any]:
    return {
        "version": HARDWARE_CACHE_VERSION,
        "fingerprint": fingerprint,
        "written_at": time.time(),
        "os_name": report.os_name,
        "cpu": {
            "name": report.cpu.name,
            "cores_physical": report.cpu.cores_physical,
            "cores_logical": report.cpu.cores_logical,
            "ram_mb": report.cpu.ram_mb,
            "arch": report.cpu.arch,
        },
        "gpus": [
            {
                "index": gpu.index,
                "name": gpu.name,
                "vendor": gpu.vendor,
                "total_vram_mb": gpu.total_vram_mb,
                "source": gpu.source,
            }
            for gpu in report.gpus
        ],
        "npus": [{"name": npu.name, "source": npu.source} for npu in report.npus],
        "notes": list(report.notes),
    }


def _report_from_dict(data: Mapping[str, Any]) -> HardwareReport | None:
    if int(data.get("version", 0)) != HARDWARE_CACHE_VERSION:
        return None
    try:
        cpu_raw = data.get("cpu") or {}
        cpu = CpuInfo(
            name=str(cpu_raw.get("name", "unbekannt")),
            cores_physical=int(cpu_raw.get("cores_physical", 0) or 0),
            cores_logical=int(cpu_raw.get("cores_logical", 0) or 0),
            ram_mb=int(cpu_raw.get("ram_mb", 0) or 0),
            arch=str(cpu_raw.get("arch", "")),
        )
        gpus = tuple(
            GpuDevice(
                index=int(item.get("index", 0) or 0),
                name=str(item.get("name", "")),
                vendor=str(item.get("vendor", Vendor.UNKNOWN)),
                total_vram_mb=int(item.get("total_vram_mb", 0) or 0),
                source=str(item.get("source", "")),
            )
            for item in data.get("gpus") or []
        )
        npus = tuple(
            NpuDevice(name=str(item.get("name", "")), source=str(item.get("source", "")))
            for item in data.get("npus") or []
        )
    except (AttributeError, TypeError, ValueError):
        return None
    return HardwareReport(
        gpus=gpus,
        npus=npus,
        cpu=cpu,
        os_name=str(data.get("os_name", "")),
        notes=tuple(str(note) for note in data.get("notes") or ()),
    )


def _load_hardware_cache(fingerprint: str) -> HardwareReport | None:
    """Bericht des letzten Laufs lesen. Fehler sind nie fatal."""
    target = _hardware_cache_path()
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        log.debug("Hardware-Zwischenspeicher nicht lesbar: %s", clean_error(exc))
        return None
    if not isinstance(data, dict) or data.get("fingerprint") != fingerprint:
        return None
    return _report_from_dict(data)


def _save_hardware_cache(report: HardwareReport, fingerprint: str) -> None:
    target = _hardware_cache_path()
    try:
        paths.ensure_dir(target.parent)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(_report_to_dict(report, fingerprint), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, target)
    except OSError as exc:
        log.debug("Hardware-Zwischenspeicher nicht schreibbar: %s", clean_error(exc))


def hardware_report_is_cached() -> bool:
    """True, wenn der aktuelle Bericht aus der Datei des letzten Laufs stammt."""
    return _report_from_disk


def hardware_report(refresh: bool = False, allow_cache: bool = True) -> HardwareReport:
    """Vollständige Hardware-Erkennung. Ergebnis wird gecacht.

    ``allow_cache`` erlaubt den Bericht des letzten Laufs von der Platte.
    Das spart beim Start die PowerShell-Abfragen; der Aufrufer stößt
    danach eine Auffrischung im Hintergrund an (``refresh=True``).
    """
    global _report_cache, _report_from_disk
    if _report_cache is not None and not refresh:
        return _report_cache

    cpu = probe_cpu()  # billig: Registry-Wert und ein ctypes-Aufruf
    fingerprint = _machine_fingerprint(cpu)

    if allow_cache and not refresh:
        cached = _load_hardware_cache(fingerprint)
        if cached is not None:
            _report_cache = cached
            _report_from_disk = True
            return cached

    notes: list[str] = []
    gpus, nvidia_note = probe_nvidia_smi()
    if nvidia_note:
        notes.append(nvidia_note)

    others, other_note = _probe_windows_display_adapters(refresh=refresh)
    if other_note:
        notes.append(other_note)

    # NVIDIA-Einträge aus der CIM-Liste verwerfen – nvidia-smi ist genauer.
    known = {gpu.name.lower() for gpu in gpus}
    next_index = len(gpus)
    for device in others:
        if device.vendor == Vendor.NVIDIA and gpus:
            continue
        if device.name.lower() in known:
            continue
        gpus.append(replace(device, index=next_index))
        known.add(device.name.lower())
        next_index += 1

    npus, npu_note = probe_npus(refresh=refresh)
    if npu_note:
        notes.append(npu_note)

    report = HardwareReport(
        gpus=tuple(gpus),
        npus=tuple(npus),
        cpu=cpu,
        os_name=f"{platform.system()} {platform.release()}",
        notes=tuple(notes),
    )
    _report_cache = report
    _report_from_disk = False
    _save_hardware_cache(report, fingerprint)
    return report


def describe_hardware(report: HardwareReport | None = None) -> str:
    """Menschenlesbarer Bericht für CLI (`--info`) und GUI-Hardwareseite."""
    report = report or hardware_report()
    lines: list[str] = [f"System:  {report.os_name}", f"CPU:     {report.cpu.label()}"]
    if report.gpus:
        for gpu in report.gpus:
            lines.append(f"GPU:     {gpu.label()}  ({gpu.source})")
    else:
        lines.append("GPU:     keine erkannt")
    if report.npus:
        for npu in report.npus:
            lines.append(f"NPU:     {npu.name}  ({npu.source})")
        # Erkannt heißt nicht benutzt – das muss dabeistehen, sonst wartet
        # jemand auf eine Beschleunigung, die es noch nicht gibt.
        lines.append("         (erkannt, aber noch nicht als Rechenpfad nutzbar)")
    else:
        lines.append("NPU:     keine erkannt")
        lines.append(f"         {npu_reason(report.cpu.name if report.cpu else '')}")

    advice = report.advice
    lines.append("")
    lines.append(f"Einstufung: {advice.title}")
    lines.append(advice.text)
    if report.notes:
        lines.append("")
        lines.append("Hinweise:")
        lines.extend(f"  - {note}" for note in report.notes)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 5. Backend-Kette mit Erststart-Bremse
# ---------------------------------------------------------------------------
class Backend(str):
    CUDA = "cuda"
    DML = "dml"
    # OpenVINO ist der einzige Weg zur Intel-NPU und zugleich der beste zur
    # Intel-iGPU. Eigenes Backend, weil es eine andere Laufzeit, andere
    # Gewichte und eine eigene Geräteauswahl (NPU/GPU/CPU) mitbringt.
    OPENVINO = "openvino"
    CPU = "cpu"
    AUTO = "auto"


# Reihenfolge im Auto-Modus: dedizierte NVIDIA-Karte schlägt DirectML,
# DirectML schlägt OpenVINO. OpenVINO steht vor CPU, weil es selbst auf
# einer iGPU noch deutlich schneller rechnet als reine CPU.
BACKEND_ORDER: tuple[str, ...] = (
    Backend.CUDA,
    Backend.DML,
    Backend.OPENVINO,
    Backend.CPU,
)

BACKEND_LABELS = {
    Backend.CUDA: "CUDA (NVIDIA, float16)",
    # Nicht „und NPU": DirectML spricht DirectX-12-Geräte an, also
    # Grafikkarten. NPUs werden darüber nicht angesteuert – dafür wäre
    # OpenVINO oder der QNN-Provider nötig.
    Backend.DML: "DirectML (AMD/Intel-GPU)",
    Backend.OPENVINO: "OpenVINO (Intel-GPU und NPU)",
    Backend.CPU: "CPU",
}

# Geräte, die OpenVINO ansprechen kann, in der Reihenfolge, in der sie
# genommen werden. NPU zuerst: wo sie vorhanden ist, ist sie sparsamer als
# die iGPU und für Dauerlast die bessere Wahl.
# Automatische Gerätewahl: die iGPU steht bewusst vor der NPU. NPUs sind
# auf kleine quantisierte Netze ausgelegt; bei Diffusionsmodellen sind sie
# langsamer als die iGPU, und der NPU-Treiber stürzt beim Kompilieren
# großer VAE-Graphen erfahrungsgemäß hart ab (Prozess weg, kein Traceback).
# Wer die NPU ausdrücklich will, wählt sie in den Einstellungen fest.
OPENVINO_DEVICE_ORDER: tuple[str, ...] = ("GPU", "NPU", "CPU")


@dataclass(frozen=True)
class ModelReadiness:
    """Ist für dieses Backend ein sofort lauffähiges Modell vorhanden?

    ``needs_conversion`` = ein mehrere GB großer Export/Quantisierungslauf
    wäre nötig. Genau das darf im Auto-Modus NICHT beim ersten Start
    passieren – die Anwendung wirkt sonst eingefroren.
    """

    ready: bool = False
    needs_conversion: bool = False
    note: str = ""


ReadinessProvider = Callable[[str], ModelReadiness]


@dataclass(frozen=True)
class BackendAttempt:
    backend: str
    accepted: bool
    reason: str


@dataclass(frozen=True)
class BackendPlan:
    backend: str
    device_index: int = 0
    compute_type: str = "float32"
    attempts: tuple[BackendAttempt, ...] = ()
    notes: tuple[str, ...] = ()
    forced: bool = False  # True = Nutzer hat das Backend fest gewählt

    @property
    def label(self) -> str:
        return BACKEND_LABELS.get(self.backend, self.backend)

    def report(self) -> str:
        """Klartext: was probiert wurde und warum es nicht genommen wurde."""
        lines = [f"Gewähltes Backend: {self.label} (compute={self.compute_type})"]
        for attempt in self.attempts:
            mark = "OK  " if attempt.accepted else "nein"
            lines.append(
                f"  [{mark}] {BACKEND_LABELS.get(attempt.backend, attempt.backend)}: {attempt.reason}"
            )
        lines.extend(f"  Hinweis: {note}" for note in self.notes)
        return "\n".join(lines)


def _readiness(
    provider: Mapping[str, ModelReadiness] | ReadinessProvider | None, backend: str
) -> ModelReadiness:
    if provider is None:
        # Ohne Auskunft: annehmen, dass nichts konvertiert werden muss.
        return ModelReadiness(ready=True, needs_conversion=False)
    if callable(provider):
        try:
            return provider(backend)
        except Exception as exc:
            return ModelReadiness(ready=False, needs_conversion=False, note=clean_error(exc))
    return provider.get(backend, ModelReadiness())


def _check_cuda(
    report: HardwareReport, allow_proprietary: bool, quick: bool = False
) -> tuple[bool, str]:
    if not report.nvidia_gpus:
        return False, "Keine NVIDIA-GPU erkannt (nvidia-smi lieferte nichts)."
    if not allow_proprietary:
        return False, (
            "NVIDIA-Laufzeit (CUDA/cuDNN) nicht freigegeben – Lizenz-Zustimmung "
            "fehlt. Unter Einstellungen → Lizenzen zustimmen."
        )
    if quick:
        return torch_cuda_hint()
    ok, note = torch_cuda_available()
    return ok, note


def _check_dml(report: HardwareReport, quick: bool = False) -> tuple[bool, str]:
    if os.name != "nt":
        return False, "DirectML gibt es nur unter Windows."
    non_nvidia = [g for g in report.gpus if g.vendor != Vendor.NVIDIA]
    if not non_nvidia and not report.npus and not report.gpus:
        return False, "Kein DirectML-fähiges Gerät erkannt."
    if quick and _onnx_cache is None:
        # onnxruntime zu importieren kostet beim Start Zeit. Für die
        # vorläufige Antwort genügt, ob das Paket überhaupt da ist.
        if importlib.util.find_spec("onnxruntime") is None:
            return False, "onnxruntime ist nicht installiert."
        return True, "onnxruntime vorhanden, Provider noch nicht geprüft (vorläufig)."
    return directml_available()


_openvino_cache: tuple[tuple[str, ...], str] | None = None


def openvino_devices(refresh: bool = False) -> tuple[tuple[str, ...], str]:
    """Von OpenVINO gemeldete Geräte, z. B. ('NPU', 'GPU', 'CPU')."""
    global _openvino_cache
    if _openvino_cache is not None and not refresh:
        return _openvino_cache
    if importlib.util.find_spec("openvino") is None:
        _openvino_cache = ((), "OpenVINO ist nicht installiert (pip install openvino).")
        return _openvino_cache
    try:
        import openvino as ov  # type: ignore

        devices = tuple(str(d) for d in ov.Core().available_devices)
        _openvino_cache = (devices, "" if devices else "OpenVINO meldet kein Gerät.")
    except Exception as exc:
        _openvino_cache = ((), f"OpenVINO-Abfrage fehlgeschlagen: {clean_error(exc)}")
    return _openvino_cache


def openvino_target(preferred: str = "") -> tuple[str, str]:
    """Gerät für OpenVINO wählen. Rückgabe: (Gerät, Begründung).

    Ein leeres Gerät heißt: OpenVINO ist nicht benutzbar. ``preferred``
    erlaubt die feste Wahl aus der Konfiguration; steht das Gerät nicht
    bereit, wird das gesagt und nicht stillschweigend ersetzt.
    """
    devices, note = openvino_devices()
    if not devices:
        return "", note or "Kein OpenVINO-Gerät verfügbar."

    def match(prefix: str) -> str:
        for device in devices:
            if device.upper().startswith(prefix):
                return device
        return ""

    if preferred:
        found = match(preferred.upper())
        if found:
            return found, f"{found} wie eingestellt."
        return "", (
            f"Gerät '{preferred}' ist für OpenVINO nicht verfügbar. "
            f"Gemeldet werden: {', '.join(devices)}."
        )
    for prefix in OPENVINO_DEVICE_ORDER:
        found = match(prefix)
        if found:
            return found, f"{found} gewählt (verfügbar: {', '.join(devices)})."
    return devices[0], f"{devices[0]} gewählt (verfügbar: {', '.join(devices)})."


def _check_openvino(report: HardwareReport, quick: bool = False) -> tuple[bool, str]:
    """Ist OpenVINO nutzbar? Fail-soft, nie werfend."""
    if quick and _openvino_cache is None:
        if importlib.util.find_spec("openvino") is None:
            return False, "OpenVINO ist nicht installiert (pip install openvino)."
        return True, "OpenVINO vorhanden, Geräte noch nicht geprüft (vorläufig)."
    device, note = openvino_target()
    if not device:
        return False, note
    return True, note


def _check_for(
    backend: str, report: HardwareReport, allow_proprietary: bool, quick: bool
) -> tuple[bool, str]:
    """Verfügbarkeitsprüfung je Backend an einer Stelle.

    Vorher stand die Weiche als Bedingung an zwei Stellen im Ablauf – ein
    drittes Backend hätte beide ändern müssen und wäre an einer davon
    vergessen worden.
    """
    if backend == Backend.CUDA:
        return _check_cuda(report, allow_proprietary, quick=quick)
    if backend == Backend.DML:
        return _check_dml(report, quick=quick)
    if backend == Backend.OPENVINO:
        return _check_openvino(report, quick=quick)
    return True, "Immer verfügbar."


def resolve_backend(
    config,
    readiness: Mapping[str, ModelReadiness] | ReadinessProvider | None = None,
    report: HardwareReport | None = None,
    allow_proprietary: bool = True,
    quick: bool = False,
) -> BackendPlan:
    """Backend-Kette auflösen: CUDA -> DirectML -> CPU.

    Jeder Fehlschlag landet als Klartext in ``attempts``. Im Auto-Modus gilt
    die Erststart-Bremse: ein Beschleuniger wird nur genommen, wenn sein
    Modell bereits konvertiert vorliegt ODER gar kein sofort lauffähiges
    Modell existiert.

    ``quick=True`` beantwortet die CUDA-Frage aus den Metadaten des
    torch-Wheels, statt torch zu importieren. Für den Start gedacht: der
    Import kostet Sekunden bis zu einer halben Minute. Der Aufrufer muss
    danach mit ``quick=False`` nachprüfen (im Hintergrund).
    """
    report = report or hardware_report()
    requested = str(getattr(config, "device", "auto") or "auto").lower()
    device_index = int(getattr(config, "device_index", 0) or 0)
    wanted_compute = str(getattr(config, "compute_type", "float16") or "float16")
    attempts: list[BackendAttempt] = []
    notes: list[str] = []

    def compute_for(backend: str) -> str:
        if backend == Backend.CUDA:
            return (
                wanted_compute
                if wanted_compute in {"float16", "bfloat16", "float32"}
                else "float16"
            )
        if backend == Backend.DML:
            return "float16"
        return "float32" if wanted_compute not in {"int8", "float32"} else wanted_compute

    # --- feste Wahl des Nutzers -------------------------------------------
    if requested in (Backend.CUDA, Backend.DML, Backend.OPENVINO, Backend.CPU):
        if requested == Backend.CPU:
            attempts.append(BackendAttempt(Backend.CPU, True, "Vom Nutzer festgelegt."))
            return BackendPlan(
                Backend.CPU, 0, compute_for(Backend.CPU), tuple(attempts), tuple(notes), forced=True
            )
        ok, reason = _check_for(requested, report, allow_proprietary, quick)
        attempts.append(BackendAttempt(requested, ok, reason or "Vom Nutzer festgelegt."))
        if ok:
            state = _readiness(readiness, requested)
            # Fest eingestellt heißt nicht lauffähig. Ein Backend, dessen
            # Gewichte fehlen, würde sonst als gewählt gemeldet und dann
            # stillschweigend auf der CPU rechnen – der Nutzer wartet auf
            # eine Beschleunigung, die nie kommt.
            if not state.ready and not state.needs_conversion:
                attempts.append(
                    BackendAttempt(
                        requested,
                        False,
                        state.note or "Für dieses Backend fehlen die Gewichte.",
                    )
                )
                notes.append(
                    f"{BACKEND_LABELS.get(requested, requested)} ist fest "
                    f"eingestellt, aber nicht lauffähig: "
                    f"{state.note or 'Gewichte fehlen.'} Es wird auf CPU gerechnet."
                )
                attempts.append(BackendAttempt(Backend.CPU, True, "Rückfallebene."))
                return BackendPlan(
                    Backend.CPU,
                    0,
                    compute_for(Backend.CPU),
                    tuple(attempts),
                    tuple(notes),
                    forced=True,
                )
            if state.needs_conversion:
                notes.append(
                    "Für dieses Backend muss das Modell einmalig konvertiert "
                    "werden. Das dauert mehrere Minuten und lädt mehrere GB – "
                    "bewusst gewählt, läuft also jetzt los."
                )
            return BackendPlan(
                requested,
                device_index,
                compute_for(requested),
                tuple(attempts),
                tuple(notes),
                forced=True,
            )
        notes.append(
            f"{BACKEND_LABELS.get(requested, requested)} ist fest eingestellt, "
            "aber nicht verfügbar – es wird auf CPU zurückgefallen."
        )
        attempts.append(BackendAttempt(Backend.CPU, True, "Rückfallebene."))
        return BackendPlan(
            Backend.CPU, 0, compute_for(Backend.CPU), tuple(attempts), tuple(notes), forced=True
        )

    # --- Auto-Modus --------------------------------------------------------
    cpu_state = _readiness(readiness, Backend.CPU)
    have_ready_fallback = cpu_state.ready and not cpu_state.needs_conversion

    for backend in (Backend.CUDA, Backend.DML, Backend.OPENVINO):
        ok, reason = _check_for(backend, report, allow_proprietary, quick)
        if not ok:
            attempts.append(BackendAttempt(backend, False, reason))
            continue

        state = _readiness(readiness, backend)
        # Nicht lauffähig heißt nicht wählbar – unabhängig davon, ob eine
        # Rückfallebene bereitsteht. Ein Backend, für das weder Gewichte
        # vorliegen noch ein Export möglich ist, wird durch Warten nicht
        # lauffähig; es auszuwählen hieße, am Ende still auf der CPU zu
        # rechnen und dabei etwas anderes anzuzeigen.
        if not state.ready and not state.needs_conversion:
            attempts.append(BackendAttempt(backend, False, state.note or "Nicht lauffähig."))
            continue

        if state.needs_conversion and have_ready_fallback:
            # Erststart-Bremse. Kommt aus einem echten Vorfall: der Export
            # sieht wie ein Absturz aus, der Nutzer bricht ab.
            attempts.append(
                BackendAttempt(
                    backend,
                    False,
                    "Verfügbar, aber das Modell liegt für dieses Backend noch "
                    "nicht konvertiert vor. Im Auto-Modus wird kein mehrere GB "
                    "großer Export beim Start angestoßen.",
                )
            )
            notes.append(
                f"{BACKEND_LABELS.get(backend, backend)} bewusst einschalten: "
                f"Einstellungen → Gerät = '{backend}'. Dann läuft die einmalige "
                "Konvertierung mit Fortschrittsanzeige und lässt sich abbrechen."
            )
            continue

        attempts.append(BackendAttempt(backend, True, reason or "Verfügbar."))
        return BackendPlan(
            backend, device_index, compute_for(backend), tuple(attempts), tuple(notes)
        )

    attempts.append(BackendAttempt(Backend.CPU, True, "Immer verfügbar – Rückfallebene."))
    if not report.gpus:
        notes.append(
            "Es läuft auf der CPU, weil keine GPU erkannt wurde. "
            "Das ist kein Fehler, nur langsamer."
        )
    return BackendPlan(Backend.CPU, 0, compute_for(Backend.CPU), tuple(attempts), tuple(notes))


def torch_device_string(plan: BackendPlan) -> str:
    """Plan in eine torch-Gerätezeichenkette übersetzen."""
    if plan.backend == Backend.CUDA:
        return f"cuda:{plan.device_index}"
    if plan.backend == Backend.DML:
        return "dml"
    return "cpu"


def speech_backend(plan: BackendPlan) -> str:
    """Welches Backend fuer Sprache (STT und Sprachausgabe) gilt.

    Der ``BackendPlan`` richtet sich nach dem **Bildmodell**: ist SDXL
    nicht heruntergeladen, meldet die Kette "CUDA nicht bereit" und
    stellt auf CPU. Fuer Sprache ist das falsch -- Whisper und Bark haben
    eigene, laengst geladene Gewichte, und die Karte ist da.

    Genau das zwang ein ganzes Telefonat auf die CPU, obwohl eine
    RTX 4070 Ti im Rechner steckte.

    Maßgeblich ist deshalb, ob torch die Karte wirklich sieht. Nur eine
    ausdrueckliche Festlegung des Bedieners (``forced``) wird beachtet.
    """
    if getattr(plan, "forced", False):
        return plan.backend
    ok, _grund = torch_cuda_available()
    return Backend.CUDA if ok else plan.backend


def torch_dtype(plan: BackendPlan):
    """torch-dtype passend zum Plan. Importiert torch – nur bei Bedarf."""
    import torch  # type: ignore

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "int8": torch.float32,  # int8 wird über Quantisierung erreicht
    }
    return mapping.get(plan.compute_type, torch.float32)
