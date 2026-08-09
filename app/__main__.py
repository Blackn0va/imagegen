"""CLI-Einstieg. Startet die GUI oder erledigt Aufgaben ohne Oberfläche.

Reihenfolge beim Start ist bewusst festgelegt:
  1. DLL-Suchpfad setzen (vor jedem Modellbibliothek-Import)
  2. Argumente lesen, Datenverzeichnis festlegen, Logging einrichten
  3. TLS-Vertrauensanker setzen
  4. Einzelinstanz-Sperre holen
  5. Konfiguration laden, Hardware erkennen, Backend planen
  6. GUI oder Unterbefehl ausführen
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __app_display_name__, __version__, accel, paths

log = logging.getLogger("app")

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_ALREADY_RUNNING = 3
EXIT_CANCELLED = 4


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def configure_console_encoding() -> None:
    """Konsolenausgabe auf UTF-8 stellen.

    Die deutsche Windows-Konsole läuft standardmäßig auf cp1252. Ein
    Pfeil oder ein Anführungszeichen in einer Meldung würde dort einen
    ``UnicodeEncodeError`` auslösen – der Endkunde sähe einen Stacktrace
    statt der Meldung. ``errors="replace"`` fängt auch den Rest ab.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            # Umgeleitete Ströme (Pipe, Datei) können sich sperren – dann
            # bleibt es bei der Vorgabe, aber ohne Absturz.
            pass


def setup_logging(level: str = "INFO", to_file: bool = True) -> Path | None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(getattr(logging, level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)

    if not to_file:
        return None
    try:
        directory = paths.ensure_dir(paths.logs_dir())
        target = directory / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            target, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(threadName)s: %(message)s")
        )
        root.addHandler(file_handler)
        return target
    except OSError as exc:
        log.warning("Logdatei nicht schreibbar: %s", accel.clean_error(exc))
        return None


# ---------------------------------------------------------------------------
# Argumente
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="streamforge",
        description=f"{__app_display_name__} – lokale Erzeugung von Bild, Video und Sprache.",
    )
    parser.add_argument(
        "--version", action="version", version=f"{__app_display_name__} {__version__}"
    )
    parser.add_argument("--config", type=Path, default=None, help="Pfad zur Konfigurationsdatei")
    parser.add_argument("--data-dir", type=Path, default=None, help="Datenverzeichnis erzwingen")
    parser.add_argument("--device", choices=("auto", "cuda", "dml", "cpu"), default=None)
    parser.add_argument("--offline", action="store_true", help="kein Netzzugriff, kein Download")
    parser.add_argument("--dummy", action="store_true", help="Attrappen erzwingen (Testbetrieb)")
    parser.add_argument(
        "--no-nsfw",
        action="store_true",
        help="Inhaltsprüfung der Modelle eingeschaltet lassen "
        "(Erwachsenen-Inhalte sind sonst zugelassen)",
    )
    parser.add_argument(
        "--no-single-instance",
        action="store_true",
        help="Einzelinstanz-Sperre überspringen (Diagnose)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    parser.add_argument("--no-gui", action="store_true", help="ohne Oberfläche starten")

    sub = parser.add_subparsers(dest="command")

    sub.add_parser("gui", help="Oberfläche starten (Vorgabe)")
    sub.add_parser("info", help="Hardware, Pfade, Backend und Lizenzen anzeigen")

    p_models = sub.add_parser("models", help="Modelle verwalten")
    p_models.add_argument(
        "action",
        choices=("list", "table", "download", "remove", "installed", "prune", "verify"),
    )
    p_models.add_argument("name", nargs="?", default="")
    p_models.add_argument(
        "--allow-conditional", action="store_true", help="Modelle mit Lizenzbedingung zulassen"
    )
    p_models.add_argument(
        "--dry-run", action="store_true", help="bei 'prune' nur anzeigen, nichts löschen"
    )

    p_image = sub.add_parser("image", help="Bild erzeugen")
    p_image.add_argument("prompt")
    p_image.add_argument("--steps", type=int, default=None)
    p_image.add_argument("--width", type=int, default=None)
    p_image.add_argument("--height", type=int, default=None)
    p_image.add_argument("--seed", type=int, default=-1)
    p_image.add_argument("--batch", type=int, default=None)

    p_edit = sub.add_parser("edit", help="Bestehendes Bild nach Prompt umarbeiten")
    p_edit.add_argument("files", nargs="+", type=Path, help="Ausgangsbild(er)")
    p_edit.add_argument("--prompt", required=True, help="was entstehen soll")
    p_edit.add_argument("--negative", default=None, help="Negativ-Prompt")
    p_edit.add_argument("--mode", choices=("img2img", "inpaint"), default="img2img")
    p_edit.add_argument(
        "--mask",
        type=Path,
        default=None,
        help="Maske für 'inpaint': weiß = ersetzen, schwarz = behalten",
    )
    p_edit.add_argument(
        "--strength",
        type=float,
        default=None,
        help="0,05 bis 1,0 – wie weit vom Ausgangsbild abweichen",
    )
    p_edit.add_argument("--steps", type=int, default=None)
    p_edit.add_argument("--guidance", type=float, default=None)
    p_edit.add_argument("--seed", type=int, default=-1)
    p_edit.add_argument(
        "--max-side", type=int, default=None, help="Ausgangsbild vorher auf diese Kante begrenzen"
    )

    p_up = sub.add_parser("upscale", help="Bestehende Bilder vergrößern")
    p_up.add_argument("files", nargs="+", type=Path)
    p_up.add_argument("--scale", type=int, choices=(2, 4, 8), default=None)
    p_up.add_argument("--no-model", action="store_true", help="nur Lanczos, kein Real-ESRGAN")
    p_up.add_argument("--tile", type=int, default=None, help="Kachelgröße, 0 = ohne")
    p_up.add_argument(
        "--refine",
        action="store_true",
        help="danach mit dem Bildmodell nachschärfen (braucht --prompt)",
    )
    p_up.add_argument("--prompt", default="", help="Prompt für das Nachschärfen")
    p_up.add_argument("--strength", type=float, default=None, help="Stärke beim Nachschärfen")
    p_up.add_argument(
        "--max-side", type=int, default=None, help="Ausgangsbild vorher auf diese Kante begrenzen"
    )

    p_color = sub.add_parser("colorize", help="Schwarz-Weiß-Bilder einfärben")
    p_color.add_argument("files", nargs="+", type=Path, help="Ausgangsbild(er)")
    p_color.add_argument(
        "--prompt",
        default="",
        help="Farbwunsch, z. B. 'red dress, blue sky'. Leer = allgemeine Vorgabe",
    )
    p_color.add_argument("--negative", default=None, help="Negativ-Prompt")
    p_color.add_argument(
        "--strength",
        type=float,
        default=None,
        help="0,05 bis 1,0 – wie kräftig das Modell Farbe setzen darf",
    )
    p_color.add_argument("--steps", type=int, default=None)
    p_color.add_argument("--guidance", type=float, default=None)
    p_color.add_argument("--seed", type=int, default=-1)
    p_color.add_argument(
        "--free-luminance",
        action="store_true",
        help="Helligkeit nicht aus der Vorlage zurückholen – das Modell darf "
        "auch Kanten und Details ändern",
    )
    p_color.add_argument(
        "--max-side", type=int, default=None, help="Ausgangsbild vorher auf diese Kante begrenzen"
    )

    p_diamond = sub.add_parser("diamond", help="Diamond-Painting-Vorlage aus einem Bild")
    p_diamond.add_argument("files", nargs="+", type=Path, help="Ausgangsbild(er)")
    p_diamond.add_argument(
        "--stones", type=int, default=None, help="Breite des Rasters in Steinen (20 bis 400)"
    )
    p_diamond.add_argument("--colors", type=int, default=None, help="Anzahl der Farben (2 bis 48)")
    p_diamond.add_argument("--shape", choices=("round", "square"), default=None, help="Steinform")
    p_diamond.add_argument(
        "--cell", type=int, default=None, help="Kantenlänge eines Kästchens in Pixeln (8 bis 48)"
    )
    p_diamond.add_argument(
        "--no-symbols", action="store_true", help="nur Farbfelder, keine Symbole"
    )
    p_diamond.add_argument(
        "--max-side", type=int, default=None, help="Ausgangsbild vorher auf diese Kante begrenzen"
    )

    p_video = sub.add_parser("video", help="Video erzeugen")
    p_video.add_argument("prompt")
    p_video.add_argument("--frames", type=int, default=None)
    p_video.add_argument("--fps", type=int, default=None)
    p_video.add_argument("--audio", type=Path, default=None, help="Tonspur mitvertonen")
    p_video.add_argument("--keep-frames", action="store_true")

    p_voice = sub.add_parser("voice", help="Sprache erzeugen")
    p_voice.add_argument("text")
    p_voice.add_argument("--profile", default="", help="angelerntes Stimmprofil")
    p_voice.add_argument("--speed", type=float, default=None)

    p_profile = sub.add_parser("voice-profile", help="Stimmprofile verwalten (Stimme anlernen)")
    p_profile.add_argument(
        "action", choices=("list", "create", "add-sample", "train", "delete", "set-mode")
    )
    p_profile.add_argument("slug", nargs="?", default="")
    p_profile.add_argument("--name", default="", help="Anzeigename des Profils")
    p_profile.add_argument("--speaker", default="", help="Name der sprechenden Person")
    p_profile.add_argument("--purpose", default="", help="Zweck der Verwendung")
    p_profile.add_argument("--granted-by", default="", help="wer die Einwilligung eingeholt hat")
    p_profile.add_argument(
        "--evidence", default="", help="Verweis auf die schriftliche Einwilligung"
    )
    p_profile.add_argument("--mode", choices=("zero_shot", "finetune"), default="zero_shot")
    p_profile.add_argument("--file", type=Path, default=None, help="Aufnahme für add-sample")
    p_profile.add_argument("--yes", action="store_true", help="Rückfragen überspringen")

    p_runtime = sub.add_parser("voice-runtime", help="Laufzeit für Klonstimmen prüfen/einrichten")
    p_runtime.add_argument(
        "action", nargs="?", default="status", choices=("status", "install", "prepare")
    )
    p_runtime.add_argument("--python", default="", help="Basis-Interpreter für das Venv")

    p_agb = sub.add_parser("agb", help="AGB anzeigen und bestätigen")
    p_agb.add_argument(
        "action", nargs="?", default="show", choices=("show", "status", "accept", "revoke")
    )

    p_lic = sub.add_parser("licenses", help="Lizenz-Zustimmungen verwalten")
    p_lic.add_argument("action", choices=("list", "accept", "revoke"))
    p_lic.add_argument("keys", nargs="*", default=[])

    return parser


# ---------------------------------------------------------------------------
# Laufzeit-Aufbau
# ---------------------------------------------------------------------------
class Runtime:
    """Gemeinsamer Zustand für GUI und CLI."""

    def __init__(self, args: argparse.Namespace) -> None:
        from . import config as config_module
        from . import licensing, models, nettrust

        self.args = args
        self.trust = nettrust.install()
        self.config, self.config_notes = config_module.load_or_create(args.config)

        overrides: dict[str, Any] = {}
        if args.device:
            overrides["device"] = args.device
        if args.offline:
            overrides["offline_mode"] = True
            overrides["allow_model_download"] = False
        if getattr(args, "no_nsfw", False):
            overrides["nsfw_enabled"] = False
        if overrides:
            self.config = self.config.with_values(**overrides)
            self.config, extra = self.config.validated()
            self.config_notes.extend(extra)

        # Beim Start nur der billige Weg: gespeicherter Hardware-Bericht und
        # eine Backend-Vermutung ohne torch-Import. Beides wird über
        # refresh_hardware()/refine_backend() nachgezogen, sobald das
        # Fenster steht – vorher warteten hier bis zu 20 Sekunden.
        self.hardware = accel.hardware_report()
        self.models = models
        self.licensing = licensing

        self.plan = self._resolve_plan(quick=True)
        self.plan_provisional = True

        from .jobs import JobQueue

        self.queue = JobQueue(
            workers=self.config.job_workers,
            error_throttle_seconds=self.config.error_throttle_seconds,
        )

    # --- Backend -----------------------------------------------------------
    def _resolve_plan(self, quick: bool) -> accel.BackendPlan:
        spec = self.models.resolve(self.config.image_model)
        return accel.resolve_backend(
            self.config,
            readiness=self.models.readiness(spec),
            report=self.hardware,
            allow_proprietary=self.licensing.proprietary_gpu_allowed(),
            quick=quick,
        )

    def refine_backend(self) -> tuple[accel.BackendPlan, bool]:
        """Backend endgültig festlegen. Importiert torch – nie im GUI-Thread.

        Rückgabe: (Plan, hat sich gegenüber der Vermutung etwas geändert).
        """
        previous = self.plan
        self.plan = self._resolve_plan(quick=False)
        self.plan_provisional = False
        changed = (previous.backend, previous.compute_type) != (
            self.plan.backend,
            self.plan.compute_type,
        )
        return self.plan, changed

    def refresh_hardware(self) -> accel.HardwareReport:
        """Hardware neu erkennen (PowerShell-Abfragen) – nie im GUI-Thread."""
        self.hardware = accel.hardware_report(refresh=True)
        return self.hardware

    def force_dummy(self) -> bool:
        return bool(self.args.dummy)

    def shutdown(self) -> None:
        self.queue.shutdown(wait=True, timeout=20)


def _print_notes(notes: Sequence[str], prefix: str = "Hinweis") -> None:
    for note in notes:
        print(f"{prefix}: {note}")


# ---------------------------------------------------------------------------
# Auftrag in der CLI ausführen
# ---------------------------------------------------------------------------
def _run_job_and_wait(runtime: Runtime, kind: str, title: str, handler) -> int:
    from .jobs import JobEvent, JobState

    last_line = ""

    def listener(event: JobEvent) -> None:
        nonlocal last_line
        if event.event in ("progress", "status"):
            percent = int(event.job.fraction * 100)
            line = f"\r[{percent:3d}%] {event.text[:80]:<80}"
            if line != last_line:
                sys.stdout.write(line)
                sys.stdout.flush()
                last_line = line
        elif event.event == "log":
            sys.stdout.write("\n" + event.text + "\n")
        elif event.event == "finished":
            sys.stdout.write(f"\r[{event.job.state.label():^12}] {event.text[:80]}\n")

    runtime.queue.subscribe(listener)
    job_id = runtime.queue.submit(kind, title, handler)
    try:
        while True:
            view = runtime.queue.get(job_id)
            if view is None or view.state.finished:
                break
            time.sleep(0.15)
    except KeyboardInterrupt:
        print("\nAbbruch angefordert …")
        runtime.queue.cancel(job_id)
        while True:
            view = runtime.queue.get(job_id)
            if view is None or view.state.finished:
                break
            time.sleep(0.2)

    view = runtime.queue.get(job_id)
    if view is None:
        return EXIT_ERROR
    if view.state is JobState.CANCELLED:
        return EXIT_CANCELLED
    if view.state is JobState.FAILED:
        print(f"Fehlgeschlagen: {view.error}", file=sys.stderr)
        return EXIT_ERROR

    result = view.result
    for attribute in ("files", "video", "audio", "artifact"):
        value = getattr(result, attribute, None)
        if value:
            items = value if isinstance(value, (list, tuple)) else [value]
            for item in items:
                print(f"Ausgabe: {item}")
    for note in getattr(result, "notes", ()) or ():
        print(f"Hinweis: {note}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Unterbefehle
# ---------------------------------------------------------------------------
def cmd_info(runtime: Runtime) -> int:
    from . import compose, contentgate, pipeline_image, upscale, voice_profiles

    print(f"{__app_display_name__} {__version__}")
    print("\n== Pfade ==")
    print(paths.describe())
    print("\n== Hardware ==")
    print(accel.describe_hardware(runtime.hardware))
    print("\n== Backend ==")
    print(runtime.plan.report())
    print("\n== ffmpeg ==")
    print(compose.describe())
    print("\n== Vergrößern ==")
    print(upscale.describe())
    print("\n== TLS ==")
    print(f"{runtime.trust.label()} – {runtime.trust.detail}")
    print("\n== AGB ==")
    _agb_text, agb_version = runtime.licensing.agb_text()
    accepted = runtime.licensing.agb_accepted()
    print(
        f"Fassung {agb_version}: "
        + ("zugestimmt" if accepted else "offen – bestätigen mit 'streamforge agb accept'")
    )
    print(f"Datei: {runtime.licensing.agb_path()}")
    print("\n== Inhalte für Erwachsene ==")
    allowed, reason = pipeline_image.adult_content_allowed(runtime.config)
    print("zugelassen" if allowed else f"aus – {reason}")
    print(contentgate.describe())
    print("\n== Lizenzen ==")
    print(runtime.licensing.summary())
    print("\n== Stimmprofile ==")
    print(voice_profiles.describe_profiles())
    if runtime.config_notes:
        print()
        _print_notes(runtime.config_notes, "Konfiguration")
    return EXIT_OK


def cmd_models(runtime: Runtime, args: argparse.Namespace) -> int:
    models = runtime.models
    if args.action == "table":
        print(models.license_table())
        return EXIT_OK

    if args.action == "list":
        best = runtime.hardware.best_gpu
        vram = best.total_vram_mb if best else 0
        ram = runtime.hardware.cpu.ram_mb
        for task in models.Task:
            print(f"\n== {task.value} ==")
            for spec in models.by_task(task, include_blocked=True):
                fits, reason = models.fits_hardware(spec, vram, ram)
                state = "vorhanden" if models.is_downloaded(spec) else "nicht geladen"
                mark = {"allowed": "frei", "conditional": "bedingt", "denied": "GESPERRT"}[
                    spec.commercial.value
                ]
                print(f"  {spec.key:<16} [{mark:<9}] {state:<12} {spec.label()}")
                print(f"      Hardware: {'ok' if fits else 'zu klein'} – {reason}")
        return EXIT_OK

    if args.action == "installed":
        entries = models.installed()
        if not entries:
            print("Keine Modelle im Cache.")
            return EXIT_OK
        for spec, size in entries:
            print(f"{spec.key:<16} {size / 1024:6.1f} GB  {models.local_dir(spec)}")
        return EXIT_OK

    if not args.name:
        print("Modellname fehlt.", file=sys.stderr)
        return EXIT_ERROR

    if args.action == "remove":
        freed = models.remove(args.name)
        print(f"{freed / 1024:.1f} GB freigegeben.")
        return EXIT_OK

    if args.action == "verify":
        spec = models.resolve(args.name)
        ok, problems = models.verify_local(spec)
        print(
            f"{spec.key}: {'vollständig' if ok else 'unvollständig'} "
            f"({models.disk_usage_mb(spec) / 1024:.1f} GB in {models.local_dir(spec)})"
        )
        for problem in problems[:20]:
            print(f"  - {problem}")
        return EXIT_OK if ok else EXIT_ERROR

    if args.action == "prune":
        # Räumt Modelle auf, die mit einem älteren, zu weiten Filter geladen
        # wurden: fp32-Doppelungen, .bin neben .safetensors, ONNX/OpenVINO.
        spec = models.resolve(args.name)
        before = models.disk_usage_mb(spec)
        count, freed, names = models.prune_local(spec, dry_run=args.dry_run)
        verb = "würde entfernen" if args.dry_run else "entfernt"
        print(
            f"{spec.key}: {verb} {count} Datei(en), {freed / 1024:.1f} GB "
            f"(vorher {before / 1024:.1f} GB)"
        )
        for name in names[:15]:
            print(f"  - {name}")
        if len(names) > 15:
            print(f"  … und {len(names) - 15} weitere")
        return EXIT_OK

    # download
    spec = models.resolve(args.name)

    def handler(context) -> Any:
        def on_progress(done: int, total: int) -> None:
            context.progress(
                (done / total) if total else 0.0,
                f"{done / (1024 * 1024):.0f} MB von {total / (1024 * 1024):.0f} MB",
            )

        try:
            return models.download(
                spec,
                on_progress=on_progress,
                on_status=context.status,
                should_stop=context.should_stop,
                allow_conditional=args.allow_conditional,
                offline=runtime.config.offline_mode,
            )
        except models.DownloadCancelled as exc:
            from .jobs import JobCancelled

            raise JobCancelled(str(exc)) from exc

    return _run_job_and_wait(runtime, "download", f"Download {spec.key}", handler)


def cmd_image(runtime: Runtime, args: argparse.Namespace) -> int:
    from . import pipeline_image

    overrides = {
        k: v
        for k, v in (
            ("steps", args.steps),
            ("width", args.width),
            ("height", args.height),
            ("batch", args.batch),
            ("seed", args.seed),
        )
        if v is not None
    }
    request = pipeline_image.ImageRequest.from_config(runtime.config, args.prompt, **overrides)
    handler = pipeline_image.make_job(
        runtime.config, runtime.plan, request, force_dummy=runtime.force_dummy()
    )
    return _run_job_and_wait(runtime, "image", f"Bild: {args.prompt[:40]}", handler)


def cmd_edit(runtime: Runtime, args: argparse.Namespace) -> int:
    """Bestehende Bilder umarbeiten (img2img) oder einen Bereich ersetzen."""
    from . import pipeline_image

    overrides: dict[str, Any] = {
        "mode": args.mode,
        "prompt": args.prompt,
        "seed": args.seed,
        "mask": args.mask,
    }
    for name, value in (
        ("negative_prompt", args.negative),
        ("strength", args.strength),
        ("steps", args.steps),
        ("guidance", args.guidance),
        ("max_side", args.max_side),
    ):
        if value is not None:
            overrides[name] = value

    request = pipeline_image.EditRequest.from_config(runtime.config, args.files, **overrides)
    problems = request.validated()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return EXIT_ERROR
    handler = pipeline_image.make_edit_job(
        runtime.config, runtime.plan, request, force_dummy=runtime.force_dummy()
    )
    return _run_job_and_wait(runtime, "edit", f"Bearbeiten: {args.files[0].name}", handler)


def cmd_upscale(runtime: Runtime, args: argparse.Namespace) -> int:
    """Bilder vergrößern – mit Real-ESRGAN, sonst Lanczos."""
    from . import pipeline_image

    overrides: dict[str, Any] = {
        "mode": "upscale",
        "prompt": args.prompt,
        "use_model": not args.no_model,
        "refine": args.refine,
    }
    for name, value in (
        ("factor", args.scale),
        ("tile", args.tile),
        ("refine_strength", args.strength),
        ("max_side", args.max_side),
    ):
        if value is not None:
            overrides[name] = value

    request = pipeline_image.EditRequest.from_config(runtime.config, args.files, **overrides)
    problems = request.validated()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return EXIT_ERROR
    handler = pipeline_image.make_edit_job(
        runtime.config, runtime.plan, request, force_dummy=runtime.force_dummy()
    )
    title = f"Vergrößern x{request.factor}: {args.files[0].name}"
    return _run_job_and_wait(runtime, "edit", title, handler)


def cmd_colorize(runtime: Runtime, args: argparse.Namespace) -> int:
    """Schwarz-Weiß-Bilder einfärben.

    Läuft über dieselbe img2img-Pipeline wie ``edit``. Der Unterschied
    steckt in der Nachbehandlung: die Helligkeit kommt aus der Vorlage
    zurück, vom Modell bleibt nur die Farbe.
    """
    from . import pipeline_image

    overrides: dict[str, Any] = {
        "mode": "colorize",
        "prompt": args.prompt,
        "seed": args.seed,
        "keep_luminance": not args.free_luminance,
    }
    for name, value in (
        ("negative_prompt", args.negative),
        ("colorize_strength", args.strength),
        ("steps", args.steps),
        ("guidance", args.guidance),
        ("max_side", args.max_side),
    ):
        if value is not None:
            overrides[name] = value

    request = pipeline_image.EditRequest.from_config(runtime.config, args.files, **overrides)
    problems = request.validated()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return EXIT_ERROR
    handler = pipeline_image.make_edit_job(
        runtime.config, runtime.plan, request, force_dummy=runtime.force_dummy()
    )
    title = f"Einfärben: {args.files[0].name}"
    if len(args.files) > 1:
        title += f" (+{len(args.files) - 1})"
    return _run_job_and_wait(runtime, "edit", title, handler)


def cmd_diamond(runtime: Runtime, args: argparse.Namespace) -> int:
    """Diamond-Painting-Vorlagen erzeugen. Kein Modell, keine Grafikkarte."""
    from . import pipeline_image

    overrides: dict[str, Any] = {
        "mode": "diamond",
        "diamond_symbols": not args.no_symbols,
    }
    for name, value in (
        ("diamond_stones", args.stones),
        ("diamond_colors", args.colors),
        ("diamond_shape", args.shape),
        ("diamond_cell_px", args.cell),
        ("max_side", args.max_side),
    ):
        if value is not None:
            overrides[name] = value

    request = pipeline_image.EditRequest.from_config(runtime.config, args.files, **overrides)
    problems = request.validated()
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return EXIT_ERROR
    handler = pipeline_image.make_edit_job(runtime.config, runtime.plan, request)
    title = f"Diamond-Vorlage: {args.files[0].name}"
    if len(args.files) > 1:
        title += f" (+{len(args.files) - 1})"
    return _run_job_and_wait(runtime, "edit", title, handler)


def cmd_video(runtime: Runtime, args: argparse.Namespace) -> int:
    from . import pipeline_video

    overrides: dict[str, Any] = {"keep_frames": args.keep_frames}
    if args.frames is not None:
        overrides["frames"] = args.frames
    if args.fps is not None:
        overrides["fps"] = args.fps
    if args.audio is not None:
        overrides["audio_file"] = args.audio
    request = pipeline_video.VideoRequest.from_config(runtime.config, args.prompt, **overrides)
    handler = pipeline_video.make_job(
        runtime.config, runtime.plan, request, force_dummy=runtime.force_dummy()
    )
    return _run_job_and_wait(runtime, "video", f"Video: {args.prompt[:40]}", handler)


def cmd_voice(runtime: Runtime, args: argparse.Namespace) -> int:
    from . import pipeline_voice

    overrides: dict[str, Any] = {}
    config = runtime.config
    if args.profile:
        # Ausdrückliche Profilangabe gilt als Absicht – die Einwilligungs-
        # prüfung in resolve_profile() bleibt davon unberührt.
        config = config.with_values(voice_cloning_enabled=True, voice_profile=args.profile)
        overrides["profile_slug"] = args.profile
    if args.speed is not None:
        overrides["speed"] = args.speed
    request = pipeline_voice.VoiceRequest.from_config(config, args.text, **overrides)
    handler = pipeline_voice.make_job(
        config, runtime.plan, request, force_dummy=runtime.force_dummy()
    )
    return _run_job_and_wait(runtime, "voice", f"Sprache: {args.text[:40]}", handler)


def cmd_voice_profile(runtime: Runtime, args: argparse.Namespace) -> int:
    from . import licensing, pipeline_voice, voice_profiles

    if args.action == "list":
        print(voice_profiles.describe_profiles())
        return EXIT_OK

    if args.action == "create":
        gate = licensing.gate("voice-cloning")
        if not gate.allowed:
            print(gate.reason, file=sys.stderr)
            print("Zustimmen mit: streamforge licenses accept voice-cloning", file=sys.stderr)
            return EXIT_ERROR
        speaker = args.speaker or args.name
        if not speaker:
            print("--speaker fehlt: Name der sprechenden Person ist Pflicht.", file=sys.stderr)
            return EXIT_ERROR
        if not args.yes:
            print(
                "Bestätige, dass eine Einwilligung der genannten Person vorliegt.\n"
                "Ohne Einwilligung ist das Anlernen einer fremden Stimme nicht zulässig.\n"
                "Wiederhole den Befehl mit --yes."
            )
            return EXIT_ERROR
        consent = licensing.SpeakerConsent.create(
            speaker_name=speaker,
            purpose=args.purpose or "Sprachausgabe in eigenen Produktionen",
            granted_by=args.granted_by or "Bediener",
            self_recorded=not args.granted_by,
            evidence_note=args.evidence,
        )
        profile = voice_profiles.create_profile(
            display_name=args.name or speaker,
            consent=consent,
            model_key=runtime.config.voice_clone_model,
            mode=voice_profiles.TrainingMode(args.mode),
            language=runtime.config.language,
        )
        print(f"Profil angelegt: {profile.slug} ({profile.root})")
        print(f"Aufnahmen ablegen in: {profile.samples_dir}")
        return EXIT_OK

    if not args.slug:
        print("Profil-Kennung (slug) fehlt.", file=sys.stderr)
        return EXIT_ERROR

    if args.action == "add-sample":
        if args.file is None:
            print("--file fehlt.", file=sys.stderr)
            return EXIT_ERROR
        profile = voice_profiles.load_profile(args.slug)
        if profile is None:
            print(f"Profil '{args.slug}' nicht gefunden.", file=sys.stderr)
            return EXIT_ERROR
        info = voice_profiles.add_sample(profile, args.file)
        state = "brauchbar" if info.usable else f"nicht brauchbar ({info.note})"
        print(f"{info.path.name}: {info.seconds:.1f}s, {info.sample_rate} Hz – {state}")
        print(f"Gesamt: {profile.total_seconds():.1f}s")
        return EXIT_OK

    if args.action == "set-mode":
        # Rettungsweg für Profile, die auf dem nicht umgesetzten
        # Nachtrainieren stehen und sich deshalb nie anlernen ließen.
        ziel = voice_profiles.TrainingMode(args.mode)
        if not voice_profiles.set_mode(args.slug, ziel):
            print(f"Profil '{args.slug}' nicht gefunden.", file=sys.stderr)
            return EXIT_ERROR
        print(f"{args.slug}: Verfahren jetzt '{ziel.value}' ({ziel.label()}).")
        return EXIT_OK

    if args.action == "delete":
        if voice_profiles.delete_profile(args.slug):
            print("Profil samt Aufnahmen gelöscht.")
            return EXIT_OK
        print("Profil nicht gefunden.", file=sys.stderr)
        return EXIT_ERROR

    # train
    profile = voice_profiles.load_profile(args.slug)
    if profile is None:
        print(f"Profil '{args.slug}' nicht gefunden.", file=sys.stderr)
        return EXIT_ERROR
    ready, problems = profile.training_ready()
    if not ready:
        for problem in problems:
            print(f"Blockiert: {problem}", file=sys.stderr)
        return EXIT_ERROR
    handler = pipeline_voice.make_training_job(runtime.config, runtime.plan, args.slug)
    return _run_job_and_wait(runtime, "train", f"Stimme anlernen: {profile.display_name}", handler)


def cmd_voice_runtime(runtime: Runtime, args: argparse.Namespace) -> int:
    """Klonstimmen laufen in einer eigenen Umgebung – hier wird sie verwaltet."""
    from . import voice_runtime

    if args.action == "status":
        print(voice_runtime.describe())
        ok, _note = voice_runtime.available()
        return EXIT_OK if ok else EXIT_ERROR

    if args.action == "prepare":
        # Modell einmalig laden, damit die erste Sprachausgabe nicht in ein
        # Zeitlimit läuft (mehrere GB Download).
        try:
            data = voice_runtime.prepare(on_status=lambda t: print(f"  {t}"))
        except Exception as exc:
            print(f"Vorbereitung fehlgeschlagen: {accel.clean_error(exc)}", file=sys.stderr)
            return EXIT_ERROR
        print(
            f"Bereit: Gerät {data.get('device')}, "
            f"{'mehrsprachig' if data.get('multilingual') else 'nur Englisch'}"
        )
        return EXIT_OK

    print("Die Laufzeit für Klonstimmen wird getrennt installiert, weil sie")
    print("ältere Fassungen von torch und diffusers verlangt und sonst die")
    print("GPU-Beschleunigung für Bild und Video zerstören würde.")
    try:
        target = voice_runtime.install(
            on_status=lambda text: print(f"  {text}"),
            base_python=args.python or None,
        )
    except Exception as exc:
        print(f"Einrichtung fehlgeschlagen: {accel.clean_error(exc)}", file=sys.stderr)
        return EXIT_ERROR
    print(f"Fertig: {target}")
    print(voice_runtime.describe())
    return EXIT_OK


def cmd_agb(runtime: Runtime, args: argparse.Namespace) -> int:
    """AGB lesen und bestätigen – ohne Zustimmung keine Nutzung."""
    licensing = runtime.licensing
    text, version = licensing.agb_text()
    accepted = licensing.agb_accepted()

    if args.action == "status":
        print(f"AGB Fassung {version}: {'zugestimmt' if accepted else 'offen'}")
        print(f"Datei: {licensing.agb_path()}")
        return EXIT_OK if accepted else EXIT_ERROR
    if args.action == "accept":
        licensing.accept_agb("über die Kommandozeile bestätigt")
        print(f"AGB Fassung {version} bestätigt.")
        return EXIT_OK
    if args.action == "revoke":
        licensing.revoke_agb()
        print("Zustimmung zurückgezogen. Die Anwendung darf nicht genutzt werden.")
        return EXIT_OK

    print(text)
    print()
    print(f"Fassung {version}: {'zugestimmt' if accepted else 'noch nicht zugestimmt'}")
    if not accepted:
        print("Bestätigen mit: streamforge agb accept")
    return EXIT_OK


def cmd_licenses(runtime: Runtime, args: argparse.Namespace) -> int:
    licensing = runtime.licensing
    if args.action == "list":
        print(licensing.summary())
        return EXIT_OK
    if not args.keys:
        print(
            "Keine Komponente angegeben. Verfügbar: " + ", ".join(sorted(licensing.COMPONENTS)),
            file=sys.stderr,
        )
        return EXIT_ERROR
    if args.action == "accept":
        changed = licensing.store().accept(args.keys, note="über CLI bestätigt")
        print("Zugestimmt: " + (", ".join(changed) or "nichts"))
        return EXIT_OK
    removed = licensing.store().revoke(args.keys)
    print("Zurückgezogen: " + (", ".join(removed) or "nichts"))
    return EXIT_OK


# ---------------------------------------------------------------------------
# Einstieg
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    # 0. Ausgabe-Kodierung, bevor irgendetwas gedruckt wird.
    configure_console_encoding()

    # 1. DLL-Suchpfad zuerst – vor jedem Import von torch/onnxruntime.
    accel.prepare_gpu_dll_path()

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.data_dir:
        paths.set_data_dir_override(args.data_dir)
    paths.bootstrap()

    level = "DEBUG" if args.verbose >= 2 else ("INFO" if args.verbose == 1 else "WARNING")
    setup_logging(level)

    command = args.command or ("info" if args.no_gui else "gui")
    wants_gui = command == "gui" and not args.no_gui

    # 4. Einzelinstanz – nur die GUI blockiert, CLI-Läufe nicht.
    if wants_gui and not args.no_single_instance:
        from . import single_instance

        guard = single_instance.acquire()
        if not guard:
            single_instance.notify_already_running(guard, gui=True)
            return EXIT_ALREADY_RUNNING

    try:
        runtime = Runtime(args)
    except Exception as exc:
        print(f"Start fehlgeschlagen: {accel.clean_error(exc)}", file=sys.stderr)
        log.exception("Start fehlgeschlagen")
        return EXIT_ERROR

    # Ohne Oberfläche gibt es niemanden, der die Nachprüfung im Hintergrund
    # anstößt – und wer rechnet, importiert torch ohnehin gleich. Also hier
    # sofort: erst Hardware neu erkennen, dann das Backend festklopfen.
    # Im Attrappen-Betrieb entfällt beides: dort wird kein Modell geladen.
    if (
        not wants_gui
        and not args.dummy
        and command
        in ("info", "image", "edit", "upscale", "colorize", "video", "voice", "voice-profile")
    ):
        if command == "info":
            runtime.refresh_hardware()
        runtime.refine_backend()

    try:
        if wants_gui:
            from .gui.main_window import run_gui

            return run_gui(runtime)
        if command == "info":
            return cmd_info(runtime)
        if command == "models":
            return cmd_models(runtime, args)
        if command == "image":
            return cmd_image(runtime, args)
        if command == "edit":
            return cmd_edit(runtime, args)
        if command == "upscale":
            return cmd_upscale(runtime, args)
        if command == "colorize":
            return cmd_colorize(runtime, args)
        if command == "diamond":
            return cmd_diamond(runtime, args)
        if command == "video":
            return cmd_video(runtime, args)
        if command == "voice":
            return cmd_voice(runtime, args)
        if command == "voice-profile":
            return cmd_voice_profile(runtime, args)
        if command == "voice-runtime":
            return cmd_voice_runtime(runtime, args)
        if command == "agb":
            return cmd_agb(runtime, args)
        if command == "licenses":
            return cmd_licenses(runtime, args)
        print(f"Unbekannter Befehl: {command}", file=sys.stderr)
        return EXIT_ERROR
    except KeyboardInterrupt:
        print("\nAbgebrochen.")
        return EXIT_CANCELLED
    except Exception as exc:
        print(f"Fehler: {accel.clean_error(exc)}", file=sys.stderr)
        log.exception("Unbehandelter Fehler")
        return EXIT_ERROR
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
