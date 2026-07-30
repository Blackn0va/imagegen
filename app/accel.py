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

import ctypes
import importlib.util
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field, replace
from enum import IntEnum
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence

from . import paths

log = logging.getLogger(__name__)

NVIDIA_SMI_TIMEOUT = 2.0
POWERSHELL_TIMEOUT = 6.0
ERROR_TEXT_LIMIT = 240

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
    """
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
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
        try:
            os.add_dll_directory(str(directory))
        except (OSError, AttributeError):
            # AttributeError: Nicht-Windows. OSError: Pfad verschwunden.
            pass
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
        return (
            f"{self.name} – {self.cores_logical or '?'} Threads, "
            f"{self.ram_gb:g} GB RAM"
        )


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
    "virtual", "remote", "basic render", "basic display", "idd", "parsec",
    "meta ", "oculus", "spacedesk", "citrix", "vnc", "mirror", "duet",
)


def _is_virtual_adapter(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _VIRTUAL_ADAPTER_HINTS)


def _probe_windows_display_adapters() -> tuple[list[GpuDevice], str]:
    """AMD/Intel-GPUs über CIM. AdapterRAM ist bei >4 GB unbrauchbar und
    wird deshalb nur als Untergrenze verwendet."""
    if os.name != "nt":
        return [], ""
    script = (
        "Get-CimInstance Win32_VideoController | "
        "ForEach-Object { \"$($_.Name)|$($_.AdapterRAM)\" }"
    )
    code, out, err = _run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        POWERSHELL_TIMEOUT,
    )
    if code != 0:
        return [], f"Grafikkarten-Abfrage fehlgeschlagen: {clean_error(err or out)}"
    devices: list[GpuDevice] = []
    for position, line in enumerate(out.splitlines()):
        line = line.strip()
        if not line or "|" not in line:
            continue
        name, _, ram_text = line.partition("|")
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
        devices.append(
            GpuDevice(
                index=position,
                name=name,
                vendor=_vendor_from_name(name),
                total_vram_mb=vram_mb,
                source="cim",
            )
        )
    return devices, ""


def probe_npus() -> tuple[list[NpuDevice], str]:
    """NPU-Erkennung: Windows-PnP-Namen und – falls installiert – OpenVINO."""
    found: list[NpuDevice] = []
    notes: list[str] = []

    if os.name == "nt":
        # Wortgrenzen sind Pflicht: 'NPU' steckt sonst als Teilstring in
        # 'USB Input Device' und liefert falsche Treffer.
        pattern = (
            r"(^|[^A-Za-z])NPU([^A-Za-z]|$)|AI Boost|Neural Processor|"
            r"Neural Engine|Movidius|(^|[^A-Za-z])VPU([^A-Za-z]|$)"
        )
        script = (
            "Get-CimInstance Win32_PnPEntity | "
            f"Where-Object {{ $_.Name -match '{pattern}' }} | "
            "ForEach-Object { $_.Name }"
        )
        code, out, err = _run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            POWERSHELL_TIMEOUT,
        )
        if code == 0:
            for line in out.splitlines():
                name = line.strip()
                if name:
                    found.append(NpuDevice(name=name, source="pnp"))
        else:
            notes.append(f"NPU-Abfrage fehlgeschlagen: {clean_error(err or out)}")

    if importlib.util.find_spec("openvino") is not None:
        try:  # OpenVINO ist optional – Import darf den Start nicht kippen
            import openvino as ov  # type: ignore

            for device in ov.Core().available_devices:
                if str(device).upper().startswith("NPU"):
                    found.append(NpuDevice(name=f"OpenVINO {device}", source="openvino"))
        except Exception as exc:  # noqa: BLE001 – Fremdbibliothek, fail-soft
            notes.append(f"OpenVINO-Abfrage fehlgeschlagen: {clean_error(exc)}")

    # Doppelte Namen zusammenfassen
    unique: dict[str, NpuDevice] = {}
    for device in found:
        unique.setdefault(device.name.lower(), device)
    return list(unique.values()), "; ".join(notes)


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
    except Exception as exc:  # noqa: BLE001 – Import kann an DLLs scheitern
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
    except Exception as exc:  # noqa: BLE001
        _onnx_cache = ((), f"onnxruntime-Import fehlgeschlagen: {clean_error(exc)}")
    return _onnx_cache


def directml_available() -> tuple[bool, str]:
    providers, note = onnx_providers()
    if note:
        return False, note
    if "DmlExecutionProvider" in providers:
        return True, "DirectML-Provider vorhanden."
    return False, (
        "Kein DmlExecutionProvider – Paket onnxruntime-directml fehlt "
        "oder Windows-Version zu alt."
    )


# ---------------------------------------------------------------------------
# 4. Gesamtbericht
# ---------------------------------------------------------------------------
_report_cache: HardwareReport | None = None


def hardware_report(refresh: bool = False) -> HardwareReport:
    """Vollständige Hardware-Erkennung. Ergebnis wird gecacht."""
    global _report_cache
    if _report_cache is not None and not refresh:
        return _report_cache

    notes: list[str] = []
    gpus, nvidia_note = probe_nvidia_smi()
    if nvidia_note:
        notes.append(nvidia_note)

    others, other_note = _probe_windows_display_adapters()
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

    npus, npu_note = probe_npus()
    if npu_note:
        notes.append(npu_note)

    report = HardwareReport(
        gpus=tuple(gpus),
        npus=tuple(npus),
        cpu=probe_cpu(),
        os_name=f"{platform.system()} {platform.release()}",
        notes=tuple(notes),
    )
    _report_cache = report
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
    else:
        lines.append("NPU:     keine erkannt")

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
    CPU = "cpu"
    AUTO = "auto"


BACKEND_ORDER: tuple[str, ...] = (Backend.CUDA, Backend.DML, Backend.CPU)

BACKEND_LABELS = {
    Backend.CUDA: "CUDA (NVIDIA, float16)",
    Backend.DML: "DirectML (AMD/Intel-GPU und NPU)",
    Backend.CPU: "CPU",
}


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
            lines.append(f"  [{mark}] {BACKEND_LABELS.get(attempt.backend, attempt.backend)}: {attempt.reason}")
        lines.extend(f"  Hinweis: {note}" for note in self.notes)
        return "\n".join(lines)


def _readiness(provider: Mapping[str, ModelReadiness] | ReadinessProvider | None,
               backend: str) -> ModelReadiness:
    if provider is None:
        # Ohne Auskunft: annehmen, dass nichts konvertiert werden muss.
        return ModelReadiness(ready=True, needs_conversion=False)
    if callable(provider):
        try:
            return provider(backend)
        except Exception as exc:  # noqa: BLE001 – Auskunft darf nie kippen
            return ModelReadiness(ready=False, needs_conversion=False,
                                  note=clean_error(exc))
    return provider.get(backend, ModelReadiness())


def _check_cuda(report: HardwareReport, allow_proprietary: bool) -> tuple[bool, str]:
    if not report.nvidia_gpus:
        return False, "Keine NVIDIA-GPU erkannt (nvidia-smi lieferte nichts)."
    if not allow_proprietary:
        return False, (
            "NVIDIA-Laufzeit (CUDA/cuDNN) nicht freigegeben – Lizenz-Zustimmung "
            "fehlt. Unter Einstellungen → Lizenzen zustimmen."
        )
    ok, note = torch_cuda_available()
    return ok, note


def _check_dml(report: HardwareReport) -> tuple[bool, str]:
    if os.name != "nt":
        return False, "DirectML gibt es nur unter Windows."
    non_nvidia = [g for g in report.gpus if g.vendor != Vendor.NVIDIA]
    if not non_nvidia and not report.npus and not report.gpus:
        return False, "Kein DirectML-fähiges Gerät erkannt."
    return directml_available()


def resolve_backend(
    config,
    readiness: Mapping[str, ModelReadiness] | ReadinessProvider | None = None,
    report: HardwareReport | None = None,
    allow_proprietary: bool = True,
) -> BackendPlan:
    """Backend-Kette auflösen: CUDA -> DirectML -> CPU.

    Jeder Fehlschlag landet als Klartext in ``attempts``. Im Auto-Modus gilt
    die Erststart-Bremse: ein Beschleuniger wird nur genommen, wenn sein
    Modell bereits konvertiert vorliegt ODER gar kein sofort lauffähiges
    Modell existiert.
    """
    report = report or hardware_report()
    requested = str(getattr(config, "device", "auto") or "auto").lower()
    device_index = int(getattr(config, "device_index", 0) or 0)
    wanted_compute = str(getattr(config, "compute_type", "float16") or "float16")
    attempts: list[BackendAttempt] = []
    notes: list[str] = []

    def compute_for(backend: str) -> str:
        if backend == Backend.CUDA:
            return wanted_compute if wanted_compute in {"float16", "bfloat16", "float32"} else "float16"
        if backend == Backend.DML:
            return "float16"
        return "float32" if wanted_compute not in {"int8", "float32"} else wanted_compute

    # --- feste Wahl des Nutzers -------------------------------------------
    if requested in (Backend.CUDA, Backend.DML, Backend.CPU):
        if requested == Backend.CPU:
            attempts.append(BackendAttempt(Backend.CPU, True, "Vom Nutzer festgelegt."))
            return BackendPlan(Backend.CPU, 0, compute_for(Backend.CPU),
                               tuple(attempts), tuple(notes), forced=True)
        ok, reason = (
            _check_cuda(report, allow_proprietary)
            if requested == Backend.CUDA
            else _check_dml(report)
        )
        attempts.append(BackendAttempt(requested, ok, reason or "Vom Nutzer festgelegt."))
        if ok:
            state = _readiness(readiness, requested)
            if state.needs_conversion:
                notes.append(
                    "Für dieses Backend muss das Modell einmalig konvertiert "
                    "werden. Das dauert mehrere Minuten und lädt mehrere GB – "
                    "bewusst gewählt, läuft also jetzt los."
                )
            return BackendPlan(requested, device_index, compute_for(requested),
                               tuple(attempts), tuple(notes), forced=True)
        notes.append(
            f"{BACKEND_LABELS.get(requested, requested)} ist fest eingestellt, "
            "aber nicht verfügbar – es wird auf CPU zurückgefallen."
        )
        attempts.append(BackendAttempt(Backend.CPU, True, "Rückfallebene."))
        return BackendPlan(Backend.CPU, 0, compute_for(Backend.CPU),
                           tuple(attempts), tuple(notes), forced=True)

    # --- Auto-Modus --------------------------------------------------------
    cpu_state = _readiness(readiness, Backend.CPU)
    have_ready_fallback = cpu_state.ready and not cpu_state.needs_conversion

    for backend in (Backend.CUDA, Backend.DML):
        ok, reason = (
            _check_cuda(report, allow_proprietary)
            if backend == Backend.CUDA
            else _check_dml(report)
        )
        if not ok:
            attempts.append(BackendAttempt(backend, False, reason))
            continue

        state = _readiness(readiness, backend)
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
        return BackendPlan(backend, device_index, compute_for(backend),
                           tuple(attempts), tuple(notes))

    attempts.append(BackendAttempt(Backend.CPU, True, "Immer verfügbar – Rückfallebene."))
    if not report.gpus:
        notes.append(
            "Es läuft auf der CPU, weil keine GPU erkannt wurde. "
            "Das ist kein Fehler, nur langsamer."
        )
    return BackendPlan(Backend.CPU, 0, compute_for(Backend.CPU),
                       tuple(attempts), tuple(notes))


def torch_device_string(plan: BackendPlan) -> str:
    """Plan in eine torch-Gerätezeichenkette übersetzen."""
    if plan.backend == Backend.CUDA:
        return f"cuda:{plan.device_index}"
    if plan.backend == Backend.DML:
        return "dml"
    return "cpu"


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
