"""Rauchtest ohne Netz und ohne GPU.

Prüft die Abnahmekriterien, die sich lokal prüfen lassen:
Pfade, Konfiguration, Warteschlange samt Abbruch, Backend-Kette mit
Erststart-Bremse, Lizenz-Tore, Stimmprofil-Einwilligung und die drei
Attrappen-Pipelines.

Aufruf:  python tests\\smoke.py
Rückgabe 0 = alles bestanden.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import accel, paths  # noqa: E402
from app.__main__ import configure_console_encoding  # noqa: E402

configure_console_encoding()

failures: list[str] = []
checks = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if condition:
        print(f"  ok    {name}")
    else:
        failures.append(f"{name}: {detail}")
        print(f"  FEHL  {name} – {detail}")


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="streamforge-smoke-"))
    paths.set_data_dir_override(workspace)
    paths.bootstrap()
    print(f"Arbeitsverzeichnis: {workspace}")

    try:
        _test_paths()
        _test_config()
        _test_jobs()
        _test_backend_chain()
        _test_licensing()
        _test_voice_profiles()
        _test_pipelines()
        _test_single_instance()
    finally:
        shutil.rmtree(workspace, ignore_errors=True)

    print()
    if failures:
        print(f"{len(failures)} von {checks} Prüfungen fehlgeschlagen:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print(f"Alle {checks} Prüfungen bestanden.")
    return 0


def _test_paths() -> None:
    print("\n== Pfade ==")
    check("Datenverzeichnis liegt außerhalb von _MEIPASS",
          paths.bundle_dir not in paths.data_dir().parents,
          f"{paths.data_dir()} unter {paths.bundle_dir}")
    check("Ausgabeordner angelegt", paths.outputs_dir().is_dir())
    check("ffmpeg-Suche wirft nicht", paths.ffmpeg_exe() is None or paths.ffmpeg_exe().is_file())


def _test_config() -> None:
    print("\n== Konfiguration ==")
    import json

    from app import config as config_module

    target = paths.data_dir() / "config-smoke.json"
    target.write_text(json.dumps({
        "image_steps": 9,
        "image_width": 1023,
        "unbekannter_schluessel": True,
        "voice_require_consent": False,
        "device": "quantenkern",
    }), encoding="utf-8")

    config, notes = config_module.load(target, use_env=False)
    check("bekannter Wert übernommen", config.image_steps == 9)
    check("Auflösung auf Vielfaches von 8 gerundet", config.image_width == 1016,
          str(config.image_width))
    check("unbekannter Schlüssel ignoriert",
          any("unbekannter_schluessel" in note for note in notes))
    check("Einwilligungspflicht nicht abschaltbar", config.voice_require_consent is True)
    check("ungültiges Gerät auf 'auto' gesetzt", config.device == "auto", config.device)
    check("Konfiguration ist unveränderlich",
          _is_frozen(config), "replace() wäre nötig")
    check("Variante ändert das Original nicht",
          config.with_values(image_steps=3).image_steps == 3 and config.image_steps == 9)


def _is_frozen(config) -> bool:
    try:
        config.image_steps = 1  # type: ignore[misc]
    except Exception:
        return True
    return False


def _test_jobs() -> None:
    print("\n== Warteschlange ==")
    from app.jobs import JobQueue, JobState

    queue = JobQueue(workers=1, error_throttle_seconds=5.0)
    logged: list[str] = []
    queue.subscribe(lambda event: logged.append(event.event))

    def endless(context) -> None:
        while True:
            context.raise_if_cancelled()
            context.log_error("gleich", "immer dieselbe Meldung")
            time.sleep(0.01)

    job_id = queue.submit("test", "Endlos", endless)
    time.sleep(0.4)
    check("Auftrag läuft", queue.get(job_id).state is JobState.RUNNING)
    queue.cancel(job_id)
    deadline = time.time() + 5
    while time.time() < deadline and not queue.get(job_id).state.finished:
        time.sleep(0.05)
    check("Abbruch beendet den Auftrag", queue.get(job_id).state is JobState.CANCELLED)
    check("Fehlermeldung gedrosselt", logged.count("log") <= 2, f"{logged.count('log')} Meldungen")

    after = queue.submit("test", "Danach", lambda ctx: "wert")
    deadline = time.time() + 5
    while time.time() < deadline and not queue.get(after).state.finished:
        time.sleep(0.05)
    check("Prozess arbeitet nach Abbruch weiter", queue.get(after).result == "wert")

    def boom(context) -> None:
        raise ValueError("Absicht: Fehlerpfad\nmit Zeilenumbruch")

    failing = queue.submit("test", "Fehler", boom)
    deadline = time.time() + 5
    while time.time() < deadline and not queue.get(failing).state.finished:
        time.sleep(0.05)
    view = queue.get(failing)
    check("Fehler wird festgehalten", view.state is JobState.FAILED)
    check("Fehlertext einzeilig", "\n" not in view.error, repr(view.error))
    queue.shutdown(wait=True, timeout=10)
    check("Queue schließt sauber", not queue._started)  # noqa: SLF001 – Absicht im Test


def _test_backend_chain() -> None:
    print("\n== Backend-Kette ==")
    from app.accel import (Backend, CpuInfo, GpuDevice, HardwareReport,
                           ModelReadiness, Vendor, resolve_backend)
    from app.config import AppConfig

    report = HardwareReport(
        gpus=(GpuDevice(0, "Radeon RX 7800 XT", Vendor.AMD, 16384, "test"),),
        cpu=CpuInfo("Test-CPU", 8, 16, 32768, "AMD64"),
        os_name="Windows 11",
    )
    accel._onnx_cache = (("DmlExecutionProvider",), "")  # noqa: SLF001
    accel._torch_cuda_cache = (False, "kein NVIDIA-Treiber")  # noqa: SLF001

    needs_export = {
        Backend.CUDA: ModelReadiness(ready=True),
        Backend.CPU: ModelReadiness(ready=True),
        Backend.DML: ModelReadiness(ready=False, needs_conversion=True),
    }
    plan = resolve_backend(AppConfig(device="auto"), needs_export, report)
    check("Erststart-Bremse: Auto nimmt CPU statt Export", plan.backend == Backend.CPU,
          plan.backend)
    check("Bremse erklärt den Weg zur GPU",
          any("dml" in note for note in plan.notes), str(plan.notes))

    converted = dict(needs_export)
    converted[Backend.DML] = ModelReadiness(ready=True, needs_conversion=False)
    check("Fertiges Konvertat wird genommen",
          resolve_backend(AppConfig(device="auto"), converted, report).backend == Backend.DML)

    nothing = {
        Backend.CUDA: ModelReadiness(ready=False),
        Backend.CPU: ModelReadiness(ready=False),
        Backend.DML: ModelReadiness(ready=False, needs_conversion=True),
    }
    check("Ohne lauffähiges Modell greift die Bremse nicht",
          resolve_backend(AppConfig(device="auto"), nothing, report).backend == Backend.DML)

    forced = resolve_backend(AppConfig(device="cuda"), needs_export, report)
    check("Erzwungenes CUDA ohne NVIDIA fällt auf CPU", forced.backend == Backend.CPU)
    check("Fehlschlag steht im Klartext",
          any(not attempt.accepted and attempt.reason for attempt in forced.attempts))

    check("Fehlertext gekürzt", len(accel.clean_error("x" * 900)) <= 240)
    check("Fehlertext einzeilig", "\n" not in accel.clean_error("a\nb\nc"))
    accel._onnx_cache = None  # noqa: SLF001
    accel._torch_cuda_cache = None  # noqa: SLF001


def _test_licensing() -> None:
    print("\n== Lizenzen ==")
    from app import licensing

    store = licensing.ConsentStore(paths.consent_path()).load()
    check("ohne Zustimmung gesperrt", not store.is_accepted("nvidia-cuda"))
    store.accept("nvidia-cuda")
    check("Zustimmung wird gespeichert",
          licensing.ConsentStore(paths.consent_path()).load().is_accepted("nvidia-cuda"))
    store.revoke("nvidia-cuda")
    check("Widerruf wirkt",
          not licensing.ConsentStore(paths.consent_path()).load().is_accepted("nvidia-cuda"))
    check("Tor liefert Begründung", bool(licensing.gate("voice-cloning").reason))
    check("THIRD-PARTY-NOTICES vorhanden", paths.notices_path().is_file(),
          str(paths.notices_path()))


def _test_voice_profiles() -> None:
    print("\n== Stimmprofile ==")
    from app import licensing, voice_profiles

    licensing.store().accept("voice-cloning")
    try:
        voice_profiles.create_profile("Ohne Nachweis", consent=None)  # type: ignore[arg-type]
        check("Profil ohne Einwilligung abgelehnt", False, "wurde angelegt")
    except ValueError:
        check("Profil ohne Einwilligung abgelehnt", True)

    consent = licensing.SpeakerConsent.create("Test Sprecher", "Rauchtest", "Prüfer")
    profile = voice_profiles.create_profile("Rauchtest Stimme", consent)
    check("Profil angelegt", profile.profile_file.is_file())
    ready, problems = profile.training_ready()
    check("ohne Material kein Anlernen", not ready and bool(problems), str(problems))

    from app.pipeline_voice import write_wav

    sample = paths.temp_dir() / "probe.wav"
    write_wav(sample, [0] * (24000 * 12), 24000)
    info = voice_profiles.add_sample(profile, sample)
    check("Aufnahme geprüft und übernommen", info.usable and info.seconds > 11, info.note)
    check("Anlernen jetzt möglich", voice_profiles.load_profile(profile.slug).training_ready()[0])

    manipulated = voice_profiles.load_profile(profile.slug)
    assert manipulated is not None and manipulated.consent is not None
    from dataclasses import replace as dc_replace

    manipulated.consent = dc_replace(manipulated.consent, declaration="anderer Wortlaut")
    manipulated.save()
    reloaded = voice_profiles.load_profile(profile.slug)
    check("veränderter Nachweis sperrt das Profil",
          reloaded is not None and reloaded.state is voice_profiles.ProfileState.BLOCKED,
          str(reloaded.state if reloaded else None))
    check("gesperrtes Profil nicht nutzbar", not reloaded.usable_for_synthesis()[0])
    check("Löschen entfernt alle Daten",
          voice_profiles.delete_profile(profile.slug) and not profile.root.exists())
    licensing.store().revoke("voice-cloning")


def _test_pipelines() -> None:
    print("\n== Attrappen-Pipelines ==")
    from app.accel import Backend, BackendPlan
    from app.config import AppConfig
    from app.jobs import JobQueue
    from app import pipeline_image, pipeline_video, pipeline_voice

    config = AppConfig(image_width=256, image_height=192, image_steps=3, video_frames=8,
                       video_fps=8)
    plan = BackendPlan(Backend.CPU, 0, "float32")
    queue = JobQueue(workers=1)

    def run(kind: str, handler) -> object:
        job_id = queue.submit(kind, kind, handler)
        deadline = time.time() + 60
        while time.time() < deadline and not queue.get(job_id).state.finished:
            time.sleep(0.05)
        return queue.get(job_id)

    image_request = pipeline_image.ImageRequest.from_config(config, "rauchtest",
                                                           width=256, height=192, steps=3)
    view = run("image", pipeline_image.make_job(config, plan, image_request, force_dummy=True))
    check("Bild-Attrappe schreibt PNG",
          view.result is not None and view.result.files and view.result.files[0].is_file(),
          view.message)
    if view.result and view.result.files:
        import struct

        head = view.result.files[0].read_bytes()[:24]
        width, height = struct.unpack(">II", head[16:24])
        check("PNG hat die verlangte Größe", (width, height) == (256, 192), f"{width}x{height}")

    voice_request = pipeline_voice.VoiceRequest.from_config(config, "Kurzer Rauchtest.")
    view = run("voice", pipeline_voice.make_job(config, plan, voice_request, force_dummy=True))
    check("Stimm-Attrappe schreibt WAV",
          view.result is not None and view.result.audio.is_file(), view.message)
    if view.result:
        import wave

        with wave.open(str(view.result.audio)) as handle:
            check("WAV ist lesbar und nicht leer", handle.getnframes() > 1000)

    video_request = pipeline_video.VideoRequest.from_config(config, "rauchtest",
                                                           frames=8, fps=8)
    view = run("video", pipeline_video.make_job(config, plan, video_request, force_dummy=True))
    ok = view.result is not None and view.result.frame_count == 8
    check("Video-Attrappe erzeugt Einzelbilder", ok,
          str(view.result.frame_count if view.result else view.message))
    if view.result and view.result.video is None:
        check("fehlendes ffmpeg wird verständlich gemeldet",
              any("ffmpeg" in note for note in view.result.notes), str(view.result.notes))
    queue.shutdown(wait=True, timeout=10)


def _test_single_instance() -> None:
    print("\n== Einzelinstanz ==")
    from app import single_instance

    guard = single_instance.acquire(suffix="smoke")
    check("Sperre wird gesetzt", guard.acquired, guard.reason)
    check("zweiter Aufruf im selben Prozess meldet die Sperre",
          single_instance.acquire(suffix="smoke").acquired)
    single_instance.release()


if __name__ == "__main__":
    raise SystemExit(main())
