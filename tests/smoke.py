"""Rauchtest ohne Netz und ohne GPU.

Prüft die Abnahmekriterien, die sich lokal prüfen lassen:
Pfade, Konfiguration, Warteschlange samt Abbruch, Backend-Kette mit
Erststart-Bremse, Lizenz-Tore, Stimmprofil-Einwilligung und die drei
Attrappen-Pipelines.

Aufruf:  python tests\\smoke.py
Rückgabe 0 = alles bestanden.
"""

from __future__ import annotations

import errno
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import accel, paths
from app.__main__ import configure_console_encoding

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
        _test_upscale()
        _test_image_edit()
        _test_colorize()
        _test_diamond()
        _test_gui_visibility()
        _test_mask_editor()
        _test_gui_edit_page()
        _test_chat()
        _test_telefonieren()
        _test_private_use()
        _test_onnx_backends()
        _test_download_hardening()
        _test_memory_hygiene()
        _test_content_gate()
        _test_model_registry()
        _test_build_script()
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
    check(
        "Datenverzeichnis liegt außerhalb von _MEIPASS",
        paths.bundle_dir not in paths.data_dir().parents,
        f"{paths.data_dir()} unter {paths.bundle_dir}",
    )
    check("Ausgabeordner angelegt", paths.outputs_dir().is_dir())
    check("ffmpeg-Suche wirft nicht", paths.ffmpeg_exe() is None or paths.ffmpeg_exe().is_file())


def _test_config() -> None:
    print("\n== Konfiguration ==")
    import json

    from app import config as config_module

    target = paths.data_dir() / "config-smoke.json"
    target.write_text(
        json.dumps(
            {
                "image_steps": 9,
                "image_width": 1023,
                "unbekannter_schluessel": True,
                "voice_require_consent": False,
                "device": "quantenkern",
            }
        ),
        encoding="utf-8",
    )

    config, notes = config_module.load(target, use_env=False)
    check("bekannter Wert übernommen", config.image_steps == 9)
    check(
        "Auflösung auf Vielfaches von 8 gerundet",
        config.image_width == 1016,
        str(config.image_width),
    )
    check(
        "unbekannter Schlüssel ignoriert", any("unbekannter_schluessel" in note for note in notes)
    )
    check("Einwilligungspflicht nicht abschaltbar", config.voice_require_consent is True)
    check("ungültiges Gerät auf 'auto' gesetzt", config.device == "auto", config.device)
    check("Konfiguration ist unveränderlich", _is_frozen(config), "replace() wäre nötig")
    check(
        "Variante ändert das Original nicht",
        config.with_values(image_steps=3).image_steps == 3 and config.image_steps == 9,
    )


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
    check("Queue schließt sauber", not queue._started)


def _test_backend_chain() -> None:
    print("\n== Backend-Kette ==")
    from app.accel import (
        Backend,
        CpuInfo,
        GpuDevice,
        HardwareReport,
        ModelReadiness,
        Vendor,
        resolve_backend,
    )
    from app.config import AppConfig

    report = HardwareReport(
        gpus=(GpuDevice(0, "Radeon RX 7800 XT", Vendor.AMD, 16384, "test"),),
        cpu=CpuInfo("Test-CPU", 8, 16, 32768, "AMD64"),
        os_name="Windows 11",
    )
    accel._onnx_cache = (("DmlExecutionProvider",), "")
    accel._torch_cuda_cache = (False, "kein NVIDIA-Treiber")

    needs_export = {
        Backend.CUDA: ModelReadiness(ready=True),
        Backend.CPU: ModelReadiness(ready=True),
        Backend.DML: ModelReadiness(ready=False, needs_conversion=True),
    }
    plan = resolve_backend(AppConfig(device="auto"), needs_export, report)
    check(
        "Erststart-Bremse: Auto nimmt CPU statt Export", plan.backend == Backend.CPU, plan.backend
    )
    check(
        "Bremse erklärt den Weg zur GPU", any("dml" in note for note in plan.notes), str(plan.notes)
    )

    converted = dict(needs_export)
    converted[Backend.DML] = ModelReadiness(ready=True, needs_conversion=False)
    check(
        "Fertiges Konvertat wird genommen",
        resolve_backend(AppConfig(device="auto"), converted, report).backend == Backend.DML,
    )

    nothing = {
        Backend.CUDA: ModelReadiness(ready=False),
        Backend.CPU: ModelReadiness(ready=False),
        Backend.DML: ModelReadiness(ready=False, needs_conversion=True),
    }
    check(
        "Ohne lauffähiges Modell greift die Bremse nicht",
        resolve_backend(AppConfig(device="auto"), nothing, report).backend == Backend.DML,
    )

    # Ein Backend ohne Gewichte UND ohne möglichen Export darf weder im
    # Auto-Modus gewählt noch fest eingestellt durchgelassen werden. Sonst
    # meldet die Anwendung "DirectML" und rechnet still auf der CPU.
    unmoeglich = {
        Backend.CUDA: ModelReadiness(ready=False, needs_conversion=False, note="keine NVIDIA"),
        Backend.CPU: ModelReadiness(ready=True),
        Backend.DML: ModelReadiness(
            ready=False, needs_conversion=False, note="ONNX-Export ist nicht umgesetzt"
        ),
    }
    auto_plan = resolve_backend(AppConfig(device="auto"), unmoeglich, report)
    check(
        "Auto wählt kein Backend ohne möglichen Export",
        auto_plan.backend == Backend.CPU,
        auto_plan.backend,
    )
    fest = resolve_backend(AppConfig(device="dml"), unmoeglich, report)
    check(
        "fest eingestelltes, nicht lauffähiges Backend fällt auf CPU",
        fest.backend == Backend.CPU,
        fest.backend,
    )
    check(
        "der Rückfall wird begründet",
        any("nicht lauffähig" in note for note in fest.notes),
        str(fest.notes),
    )
    # Auch ohne bereite Rückfallebene darf Unmögliches nicht gewählt werden.
    ohne_cpu = dict(unmoeglich)
    ohne_cpu[Backend.CPU] = ModelReadiness(ready=False, needs_conversion=False)
    check(
        "ohne Rückfallebene bleibt es trotzdem bei CPU",
        resolve_backend(AppConfig(device="auto"), ohne_cpu, report).backend == Backend.CPU,
    )

    # NPU-Einschätzung nach Prozessorname – (TM) und (R) dürfen nicht stören.
    check(
        "Core Ultra wird als NPU-fähig erkannt",
        "NPU mit" in accel.npu_outlook("Intel(R) Core(TM) Ultra 7 155H"),
        accel.npu_outlook("Intel(R) Core(TM) Ultra 7 155H"),
    )
    check(
        "ältere Intel-Baureihe wird als NPU-los erkannt",
        "keine NPU" in accel.npu_outlook("Intel(R) Core(TM) i9-10850K"),
    )
    check("Ryzen AI wird erkannt", "NPU mit" in accel.npu_outlook("AMD Ryzen AI 9 365"))
    check("leerer Name wirft nicht", bool(accel.npu_outlook("")))

    forced = resolve_backend(AppConfig(device="cuda"), needs_export, report)
    check("Erzwungenes CUDA ohne NVIDIA fällt auf CPU", forced.backend == Backend.CPU)
    check(
        "Fehlschlag steht im Klartext",
        any(not attempt.accepted and attempt.reason for attempt in forced.attempts),
    )

    check("Fehlertext gekürzt", len(accel.clean_error("x" * 900)) <= 240)
    check("Fehlertext einzeilig", "\n" not in accel.clean_error("a\nb\nc"))
    accel._onnx_cache = None
    accel._torch_cuda_cache = None


def _test_licensing() -> None:
    print("\n== Lizenzen ==")
    from app import licensing

    store = licensing.ConsentStore(paths.consent_path()).load()
    check("ohne Zustimmung gesperrt", not store.is_accepted("nvidia-cuda"))
    store.accept("nvidia-cuda")
    check(
        "Zustimmung wird gespeichert",
        licensing.ConsentStore(paths.consent_path()).load().is_accepted("nvidia-cuda"),
    )
    store.revoke("nvidia-cuda")
    check(
        "Widerruf wirkt",
        not licensing.ConsentStore(paths.consent_path()).load().is_accepted("nvidia-cuda"),
    )
    check("Tor liefert Begründung", bool(licensing.gate("voice-cloning").reason))
    check(
        "THIRD-PARTY-NOTICES vorhanden", paths.notices_path().is_file(), str(paths.notices_path())
    )


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
    check(
        "veränderter Nachweis sperrt das Profil",
        reloaded is not None and reloaded.state is voice_profiles.ProfileState.BLOCKED,
        str(reloaded.state if reloaded else None),
    )
    check("gesperrtes Profil nicht nutzbar", not reloaded.usable_for_synthesis()[0])
    check(
        "Löschen entfernt alle Daten",
        voice_profiles.delete_profile(profile.slug) and not profile.root.exists(),
    )
    licensing.store().revoke("voice-cloning")


def _test_pipelines() -> None:
    print("\n== Attrappen-Pipelines ==")
    from app import pipeline_image, pipeline_video, pipeline_voice
    from app.accel import Backend, BackendPlan
    from app.config import AppConfig
    from app.jobs import JobQueue

    config = AppConfig(
        image_width=256, image_height=192, image_steps=3, video_frames=8, video_fps=8
    )
    plan = BackendPlan(Backend.CPU, 0, "float32")
    queue = JobQueue(workers=1)

    def run(kind: str, handler) -> object:
        job_id = queue.submit(kind, kind, handler)
        deadline = time.time() + 60
        while time.time() < deadline and not queue.get(job_id).state.finished:
            time.sleep(0.05)
        return queue.get(job_id)

    image_request = pipeline_image.ImageRequest.from_config(
        config, "rauchtest", width=256, height=192, steps=3
    )
    view = run("image", pipeline_image.make_job(config, plan, image_request, force_dummy=True))
    check(
        "Bild-Attrappe schreibt PNG",
        view.result is not None and view.result.files and view.result.files[0].is_file(),
        view.message,
    )
    if view.result and view.result.files:
        import struct

        head = view.result.files[0].read_bytes()[:24]
        width, height = struct.unpack(">II", head[16:24])
        check("PNG hat die verlangte Größe", (width, height) == (256, 192), f"{width}x{height}")

    voice_request = pipeline_voice.VoiceRequest.from_config(config, "Kurzer Rauchtest.")
    view = run("voice", pipeline_voice.make_job(config, plan, voice_request, force_dummy=True))
    check(
        "Stimm-Attrappe schreibt WAV",
        view.result is not None and view.result.audio.is_file(),
        view.message,
    )
    if view.result:
        import wave

        with wave.open(str(view.result.audio)) as handle:
            check("WAV ist lesbar und nicht leer", handle.getnframes() > 1000)

    video_request = pipeline_video.VideoRequest.from_config(config, "rauchtest", frames=8, fps=8)
    view = run("video", pipeline_video.make_job(config, plan, video_request, force_dummy=True))
    ok = view.result is not None and view.result.frame_count == 8
    check(
        "Video-Attrappe erzeugt Einzelbilder",
        ok,
        str(view.result.frame_count if view.result else view.message),
    )
    if view.result and view.result.video is None:
        check(
            "fehlendes ffmpeg wird verständlich gemeldet",
            any("ffmpeg" in note for note in view.result.notes),
            str(view.result.notes),
        )
    queue.shutdown(wait=True, timeout=10)


def _make_test_image(target: Path, width: int = 96, height: int = 64):
    """Kleines Testbild schreiben. Ohne Pillow gibt es None zurück."""
    try:
        from PIL import Image
    except ImportError:
        return None
    image = Image.new("RGB", (width, height))
    image.putdata(
        [((x * 3) % 256, (y * 5) % 256, (x + y) % 256) for y in range(height) for x in range(width)]
    )
    image.save(target)
    return image


def _test_upscale() -> None:
    print("\n== Vergrößern ==")
    from app import upscale

    ok, reason = upscale.pillow_available()
    check("Pillow vorhanden", ok, reason)
    if not ok:
        return

    source = paths.temp_dir() / "upscale-quelle.png"
    _make_test_image(source)

    image = upscale.open_image(source)
    doubled, method = upscale.upscale_image(image, factor=2)
    check(
        "Lanczos verdoppelt die Kantenlänge",
        (doubled.width, doubled.height) == (192, 128),
        f"{doubled.width}x{doubled.height}",
    )
    check("Verfahren wird benannt", "Lanczos" in method, method)

    limited, changed = upscale.fit_to_max_side(image, 48)
    check("Höchstkante wird eingehalten", changed and max(limited.size) == 48, str(limited.size))
    snapped, _ = upscale.snap_to_multiple(upscale.lanczos_resize(image, target=(101, 67)), 8)
    check(
        "Größe wird auf Vielfaches von 8 gerundet",
        snapped.width % 8 == 0 and snapped.height % 8 == 0,
        str(snapped.size),
    )

    # Kaputte Gewichte dürfen den Auftrag nicht kippen, sondern auf Lanczos fallen.
    broken = paths.temp_dir() / "kaputt.pth"
    broken.write_bytes(b"kein torch-modell")
    result, method = upscale.upscale_image(image, factor=2, weights=broken)
    check(
        "unbrauchbare Gewichte fallen auf Lanczos zurück",
        (result.width, result.height) == (192, 128) and "Lanczos" in method,
        method,
    )

    _test_esrgan_net()


def _test_esrgan_net() -> None:
    """RRDBNet gegen selbst erzeugte Gewichte prüfen.

    Damit ist belegt, dass Aufbau, Ableitung der Netzgröße aus den
    Gewichten, das Laden mit ``strict=True`` und der Kachelweg zusammen
    funktionieren – ohne mehrere hundert MB herunterzuladen.
    """
    try:
        import torch
    except ImportError:
        print("  über  torch fehlt – Netzprüfung übersprungen")
        return
    from app import upscale

    net_class = upscale._build_modules()
    reference = net_class(scale=4, num_feat=8, num_block=2, num_grow_ch=4)
    weights = paths.temp_dir() / "mini-esrgan.pth"
    torch.save({"params_ema": reference.state_dict()}, weights)

    state = upscale._state_dict_from(
        torch.load(str(weights), map_location="cpu", weights_only=True)
    )
    scale, num_feat, num_block, num_grow_ch = upscale.describe_weights(state)
    check(
        "Netzgröße wird aus den Gewichten abgeleitet",
        (scale, num_feat, num_block, num_grow_ch) == (4, 8, 2, 4),
        f"{scale}/{num_feat}/{num_block}/{num_grow_ch}",
    )

    source = paths.temp_dir() / "netz-quelle.png"
    _make_test_image(source, 64, 48)
    image = upscale.open_image(source)

    whole, method_whole = upscale.upscale_image(image, factor=4, weights=weights, tile=0)
    check(
        "Netz vergrößert um den Faktor 4",
        (whole.width, whole.height) == (256, 192),
        f"{whole.width}x{whole.height}",
    )
    check("Netz wird als Verfahren gemeldet", "Real-ESRGAN" in method_whole, method_whole)

    tiled, _ = upscale.upscale_image(image, factor=4, weights=weights, tile=32)
    check(
        "Kachelweg liefert dieselbe Größe",
        (tiled.width, tiled.height) == (whole.width, whole.height),
        f"{tiled.width}x{tiled.height}",
    )

    import numpy as np

    difference = float(
        np.abs(np.asarray(tiled, dtype=np.float32) - np.asarray(whole, dtype=np.float32)).mean()
    )
    check(
        "Kacheln erzeugen keine sichtbaren Nähte",
        difference < 6.0,
        f"mittlere Abweichung {difference:.2f}",
    )
    upscale.unload()


def _test_image_edit() -> None:
    print("\n== Bild bearbeiten ==")
    from app import pipeline_image, upscale
    from app.accel import Backend, BackendPlan
    from app.config import AppConfig
    from app.jobs import JobQueue

    if not upscale.pillow_available()[0]:
        print("  über  Pillow fehlt – übersprungen")
        return

    config = AppConfig()
    plan = BackendPlan(Backend.CPU, 0, "float32")
    source = paths.temp_dir() / "bearbeiten-quelle.png"
    _make_test_image(source, 80, 80)

    leer = pipeline_image.EditRequest.from_config(config, [], mode="img2img")
    check(
        "ohne Datei wird abgelehnt",
        any("Ausgangsbild" in p for p in leer.validated()),
        str(leer.validated()),
    )
    ohne_prompt = pipeline_image.EditRequest.from_config(config, [source], mode="img2img")
    check(
        "img2img ohne Prompt wird abgelehnt",
        any("Prompt" in p for p in ohne_prompt.validated()),
        str(ohne_prompt.validated()),
    )
    ohne_maske = pipeline_image.EditRequest.from_config(
        config, [source], mode="inpaint", prompt="etwas"
    )
    check(
        "inpaint ohne Maske wird abgelehnt",
        any("Maske" in p for p in ohne_maske.validated()),
        str(ohne_maske.validated()),
    )

    check(
        "Klassenname für img2img wird abgeleitet",
        pipeline_image._task_class_name("StableDiffusionXLPipeline", "img2img")
        == "StableDiffusionXLImg2ImgPipeline",
    )
    check(
        "Klassenname für inpaint wird abgeleitet",
        pipeline_image._task_class_name("FluxPipeline", "inpaint") == "FluxInpaintPipeline",
    )

    queue = JobQueue(workers=1)
    request = pipeline_image.EditRequest.from_config(
        config, [source], mode="upscale", factor=2, use_model=False
    )
    job_id = queue.submit("edit", "upscale", pipeline_image.make_edit_job(config, plan, request))
    deadline = time.time() + 60
    while time.time() < deadline and not queue.get(job_id).state.finished:
        time.sleep(0.05)
    view = queue.get(job_id)
    result = view.result
    check(
        "Vergrößern schreibt eine neue Datei",
        result is not None and result.files and result.files[0].is_file(),
        view.message,
    )
    check("Ausgangsdatei bleibt unverändert", source.is_file())
    if result and result.files:
        check(
            "Ergebnis hat die doppelte Kantenlänge",
            (result.width, result.height) == (160, 160),
            f"{result.width}x{result.height}",
        )
    queue.shutdown(wait=True, timeout=10)


def _test_colorize() -> None:
    print("\n== Einfärben ==")
    from app import pipeline_image, upscale
    from app.accel import Backend, BackendPlan
    from app.config import AppConfig
    from app.jobs import JobQueue

    if not upscale.pillow_available()[0]:
        print("  über  Pillow fehlt – übersprungen")
        return

    from PIL import Image

    config = AppConfig()
    plan = BackendPlan(Backend.CPU, 0, "float32")

    check("'colorize' ist ein bekannter Modus", "colorize" in pipeline_image.EDIT_MODES)
    check(
        "'colorize' hat eine Beschriftung",
        bool(pipeline_image.EDIT_MODE_LABELS.get("colorize")),
    )

    # --- Vorgaben ----------------------------------------------------------
    source = paths.temp_dir() / "einfaerben-quelle.png"
    _make_test_image(source, 64, 64)

    ohne_prompt = pipeline_image.EditRequest.from_config(config, [source], mode="colorize")
    check(
        "Einfärben ohne Prompt wird angenommen",
        not any("Prompt" in problem for problem in ohne_prompt.validated()),
        str(ohne_prompt.validated()),
    )
    check(
        "ohne Prompt greift die Vorgabe",
        ohne_prompt.effective_prompt() == pipeline_image.COLORIZE_PROMPT,
    )
    check(
        "ohne Negativ-Prompt greift die Vorgabe",
        ohne_prompt.effective_negative() == pipeline_image.COLORIZE_NEGATIVE,
    )
    check("Einfärben braucht das Bildmodell", ohne_prompt.needs_model())
    check(
        "Einfärben benutzt die eigene Stärke",
        ohne_prompt.effective_strength() == config.image_colorize_strength,
    )

    eigener = pipeline_image.EditRequest.from_config(
        config, [source], mode="colorize", prompt="rotes Kleid"
    )
    check("eigener Prompt schlägt die Vorgabe", eigener.effective_prompt() == "rotes Kleid")

    # img2img bleibt unberührt: dort ist der Prompt weiter Pflicht.
    umarbeiten = pipeline_image.EditRequest.from_config(config, [source], mode="img2img")
    check(
        "img2img verlangt weiterhin einen Prompt",
        any("Prompt" in problem for problem in umarbeiten.validated()),
        str(umarbeiten.validated()),
    )
    check("img2img bekommt keine Einfärb-Vorgabe", umarbeiten.effective_prompt() == "")

    # --- Graustufen erkennen ----------------------------------------------
    grau = Image.new("RGB", (32, 32))
    grau.putdata([(value * 8 % 256,) * 3 for value in range(32 * 32)])
    bunt = Image.new("RGB", (32, 32), (200, 40, 40))
    check("Graubild wird als schwarz-weiß erkannt", pipeline_image.is_grayscale(grau))
    check("Farbbild wird nicht als schwarz-weiß erkannt", not pipeline_image.is_grayscale(bunt))

    # --- Luminanz-Rückführung ---------------------------------------------
    verschmolzen = pipeline_image.merge_luminance(grau, bunt)
    check(
        "Ergebnis behält die Größe der Vorlage",
        (verschmolzen.width, verschmolzen.height) == (grau.width, grau.height),
        f"{verschmolzen.width}x{verschmolzen.height}",
    )
    vorher = list(grau.convert("YCbCr").getchannel(0).getdata())
    nachher = list(verschmolzen.convert("YCbCr").getchannel(0).getdata())
    # Kleine Abweichung ist Rundung beim Weg über RGB, keine Verfälschung.
    groesste = max(abs(a - b) for a, b in zip(vorher, nachher, strict=True))
    check(
        "Helligkeit stammt aus der Vorlage",
        groesste <= 3,
        f"größte Abweichung {groesste}",
    )
    check("Farbe stammt aus dem Modellergebnis", not pipeline_image.is_grayscale(verschmolzen))

    # Abweichende Größe des Modellergebnisses wird zurechtgerückt.
    grosses_bunt = Image.new("RGB", (64, 64), (40, 200, 60))
    angepasst = pipeline_image.merge_luminance(grau, grosses_bunt)
    check(
        "größeres Modellergebnis wird auf die Vorlage gebracht",
        (angepasst.width, angepasst.height) == (32, 32),
        f"{angepasst.width}x{angepasst.height}",
    )

    # --- Auftrag durch die Warteschlange ----------------------------------
    # Attrappe erzwungen: der Rauchtest lädt nie ein Modell herunter.
    queue = JobQueue(workers=1)
    request = pipeline_image.EditRequest.from_config(config, [source], mode="colorize")
    job_id = queue.submit(
        "edit",
        "colorize",
        pipeline_image.make_edit_job(config, plan, request, force_dummy=True),
    )
    deadline = time.time() + 60
    while time.time() < deadline and not queue.get(job_id).state.finished:
        time.sleep(0.05)
    view = queue.get(job_id)
    result = view.result
    check(
        "Einfärben schreibt eine neue Datei",
        result is not None and result.files and result.files[0].is_file(),
        view.message,
    )
    if result and result.files:
        check(
            "Dateiname trägt die Kennung 'farbig'",
            "farbig" in result.files[0].name,
            result.files[0].name,
        )
    check("Ausgangsdatei bleibt unverändert", source.is_file())
    queue.shutdown(wait=True, timeout=10)


def _test_diamond() -> None:
    print("\n== Diamond Painting ==")
    from app import diamond, pipeline_image, upscale
    from app.accel import Backend, BackendPlan
    from app.config import AppConfig
    from app.jobs import JobQueue

    if not upscale.pillow_available()[0]:
        print("  über  Pillow fehlt – übersprungen")
        return

    config = AppConfig()
    plan = BackendPlan(Backend.CPU, 0, "float32")

    check("'diamond' ist ein bekannter Modus", "diamond" in pipeline_image.EDIT_MODES)
    check(
        "'diamond' hat eine Beschriftung",
        bool(pipeline_image.EDIT_MODE_LABELS.get("diamond")),
    )

    source = paths.temp_dir() / "diamond-quelle.png"
    _make_test_image(source, 120, 90)

    anfrage = pipeline_image.EditRequest.from_config(config, [source], mode="diamond")
    check("Vorlage braucht kein Bildmodell", not anfrage.needs_model())
    check("Vorlage braucht keinen Prompt", not anfrage.validated(), str(anfrage.validated()))

    # --- Raster ------------------------------------------------------------
    image = upscale.open_image(source).convert("RGB")
    layout = diamond.build_plan(image, stones=40, colors=8, shape="round")
    check("Rasterbreite wie gewünscht", layout.columns == 40, str(layout.columns))
    check(
        "Seitenverhältnis bleibt erhalten",
        abs(layout.rows - 30) <= 1,
        f"{layout.columns}x{layout.rows}",
    )
    check("Farbzahl wird eingehalten", len(layout.colors) <= 8, str(len(layout.colors)))
    check(
        "jede Farbe hat ein eigenes Symbol",
        len({color.symbol for color in layout.colors}) == len(layout.colors),
    )
    check(
        "Steinzahl passt zum Raster",
        sum(color.count for color in layout.colors) == layout.total_stones,
        f"{sum(c.count for c in layout.colors)} statt {layout.total_stones}",
    )
    check(
        "häufigste Farbe steht vorn",
        all(
            layout.colors[i].count >= layout.colors[i + 1].count
            for i in range(len(layout.colors) - 1)
        ),
    )
    check(
        "jede Zelle trägt eine gültige Farbe",
        all(0 <= value < len(layout.colors) for row in layout.grid for value in row),
    )

    # Kern der Vorlage: keine zwei Farben, die auf Papier gleich aussehen.
    # Ohne das Zusammenlegen liefert die Farbreduktion denselben Himmel
    # mehrfach – man kauft dann drei Tütchen derselben Farbe.
    dichteste = min(
        (
            diamond.color_distance(links.rgb, rechts.rgb)
            for position, links in enumerate(layout.colors)
            for rechts in layout.colors[position + 1 :]
        ),
        default=999.0,
    )
    check(
        "keine zwei Farben liegen zu dicht beieinander",
        dichteste >= diamond.MIN_COLOR_DISTANCE,
        f"engster Abstand {dichteste:.1f} < {diamond.MIN_COLOR_DISTANCE}",
    )
    check(
        "gleiche Farbe hat den Abstand null",
        diamond.color_distance((10, 20, 30), (10, 20, 30)) == 0.0,
    )

    # --- DMC-Abgleich -------------------------------------------------------
    from app import dmc

    check("DMC-Tabelle ist gefüllt", len(dmc.COLORS) > 400, str(len(dmc.COLORS)))
    check(
        "Steinfarben sind eine Teilmenge der Garnfarben",
        0 < len(dmc.STONE_COLORS) <= len(dmc.COLORS),
        f"{len(dmc.STONE_COLORS)} von {len(dmc.COLORS)}",
    )
    check("jede Nummer kommt nur einmal vor", len(dmc.BY_CODE) == len(dmc.COLORS))
    check(
        "alle Steinnummern stehen in der Tabelle",
        all(code in dmc.BY_CODE for code in dmc.DIAMOND_CODES),
    )
    # Stichproben gegen die veröffentlichte Farbkarte.
    for code, erwartet in (("310", (0, 0, 0)), ("B5200", (255, 255, 255))):
        farbe = dmc.resolve_code(code)
        check(
            f"DMC {code} hat den erwarteten Farbwert",
            farbe is not None and farbe.rgb == erwartet,
            str(farbe),
        )
    check(
        "Schreibweise 5200 findet Snow White",
        (dmc.resolve_code("5200") or dmc.resolve_code("310")).code == "B5200",
    )
    check("unbekannte Nummer liefert nichts", dmc.resolve_code("99999") is None)
    check("Schwarz trifft DMC 310", dmc.nearest((0, 0, 0)).code == "310")
    check("Weiß trifft eine weiße Nummer", dmc.nearest((255, 255, 255)).rgb == (255, 255, 255))
    check(
        "nur bestellbare Steinfarben werden vorgeschlagen",
        all(
            dmc.nearest(probe).code in dmc.DIAMOND_CODES
            for probe in ((12, 34, 56), (200, 30, 40), (250, 250, 200), (90, 150, 80))
        ),
    )

    mit_dmc = diamond.build_plan(image, stones=40, colors=8, shape="round", use_dmc=True)
    check("Vorlage meldet DMC-Betrieb", mit_dmc.uses_dmc())
    check(
        "jede Farbe trägt eine bestellbare Nummer",
        all(color.dmc_code in dmc.DIAMOND_CODES for color in mit_dmc.colors),
        str([c.dmc_code for c in mit_dmc.colors]),
    )
    check(
        "angezeigte Farbe ist die DMC-Farbe",
        all(dmc.BY_CODE[color.dmc_code].rgb == color.rgb for color in mit_dmc.colors),
    )
    check(
        "keine Nummer doppelt in der Farbliste",
        len({color.dmc_code for color in mit_dmc.colors}) == len(mit_dmc.colors),
    )
    check("Bestellkennung nennt die Nummer", mit_dmc.colors[0].order_label().startswith("DMC "))

    ohne_dmc = diamond.build_plan(image, stones=40, colors=8, shape="round", use_dmc=False)
    check("ohne Abgleich keine Nummern", not ohne_dmc.uses_dmc())
    check(
        "ohne Abgleich bleibt der Hexwert die Kennung",
        ohne_dmc.colors[0].order_label().startswith("#"),
    )

    dmc_text = diamond.legend_text(mit_dmc, source)
    check("Farbliste nennt das Farbsystem DMC", "Farbsystem:   DMC" in dmc_text)
    check(
        "Farbliste nennt jede DMC-Nummer",
        all(color.dmc_code in dmc_text for color in mit_dmc.colors),
    )
    check("Farbliste erklärt den Bestellweg", "nach DMC-Nummer" in dmc_text)
    check(
        "Farbliste ohne DMC warnt vor dem Bestellen",
        "nicht bestellbar" in diamond.legend_text(ohne_dmc, source),
    )

    # Grenzen greifen, statt eine unbezahlbar große Vorlage zu bauen.
    riesig = diamond.build_plan(image, stones=9000, colors=999, shape="square")
    check("Rasterbreite wird begrenzt", riesig.columns <= diamond.MAX_STONES, str(riesig.columns))
    check(
        "Farbzahl wird begrenzt",
        len(riesig.colors) <= diamond.MAX_COLORS,
        str(len(riesig.colors)),
    )

    # --- Maße ---------------------------------------------------------------
    breite_mm, _hoehe_mm = layout.size_mm()
    check(
        "fertige Größe wird aus der Steingröße gerechnet",
        abs(breite_mm - 40 * 2.8) < 0.01,
        f"{breite_mm:.1f} mm",
    )
    check("Größenangabe ist lesbar", "cm" in layout.size_cm_text(), layout.size_cm_text())

    # --- Zeichnen -----------------------------------------------------------
    chart = diamond.render_chart(layout, cell_px=10, symbols=True)
    check(
        "Vorlage ist größer als das Raster (Rand für die Nummern)",
        chart.width > layout.columns * 10 and chart.height > layout.rows * 10,
        f"{chart.width}x{chart.height}",
    )
    sheet = diamond.render_legend(layout, cell_px=10)
    check("Farbtafel wird gezeichnet", sheet.width > 0 and sheet.height > 0)

    text = diamond.legend_text(layout, source)
    check("Farbliste nennt die Steinzahl", str(layout.total_stones) in text)
    check("Farbliste nennt jede Farbe", all(c.hex_code in text for c in layout.colors))

    # --- Auftrag durch die Warteschlange ------------------------------------
    queue = JobQueue(workers=1)
    request = pipeline_image.EditRequest.from_config(
        config, [source], mode="diamond", diamond_stones=40, diamond_colors=6, diamond_cell_px=8
    )
    job_id = queue.submit("edit", "diamond", pipeline_image.make_edit_job(config, plan, request))
    deadline = time.time() + 60
    while time.time() < deadline and not queue.get(job_id).state.finished:
        time.sleep(0.05)
    view = queue.get(job_id)
    result = view.result
    check(
        "Auftrag schreibt drei Dateien",
        result is not None and len(result.files) == 3,
        view.message,
    )
    if result and result.files:
        check("alle Dateien liegen auf der Platte", all(item.is_file() for item in result.files))
        namen = [item.name for item in result.files]
        check("Vorlage trägt die Kennung 'diamond'", any("diamond" in n for n in namen), str(namen))
        check("Farbtafel wird geschrieben", any("farbtafel" in n for n in namen), str(namen))
        check("Farbliste wird geschrieben", any("farbliste" in n for n in namen), str(namen))
    check("Ausgangsdatei bleibt unverändert", source.is_file())
    queue.shutdown(wait=True, timeout=10)


def _test_gui_visibility() -> None:
    print("\n== Oberfläche: Sichtbarkeit ==")
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("  über  tkinter fehlt – übersprungen")
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"  über  keine Anzeige verfügbar ({exc}) – übersprungen")
        return
    root.withdraw()
    try:
        from app.gui import theme
        from app.gui.widgets import Card, CheckRow, ComboRow, SliderRow, SpinRow, TextRow

        palette = theme.palette_for("dark")
        theme.apply(root, palette)
        frame = ttk.Frame(root)
        frame.grid()

        zeilen = {
            "Spin": SpinRow(frame, 0, "Zahl", 5, 0, 10, 1, hint="Hinweis dazu"),
            "Slider": SliderRow(frame, 2, "Regler", 0.5, 0.0, 1.0, hint="Hinweis dazu"),
            "Combo": ComboRow(frame, 4, "Auswahl", ["a", "b"], "a", hint="Hinweis dazu"),
            "Check": CheckRow(frame, 6, "Haken", True, hint="Hinweis dazu"),
            "Text": TextRow(frame, 8, "Text", palette, hint="Hinweis dazu"),
        }
        root.update_idletasks()

        for name, zeile in zeilen.items():
            check(f"{name}-Zeile ist zuerst sichtbar", zeile.is_visible())
            zeile.set_visible(False)
            root.update_idletasks()
            check(f"{name}-Zeile lässt sich ausblenden", not zeile.is_visible())
            # Beschriftung und Hinweis müssen mitgehen, sonst bleibt ein
            # beschriftetes Nichts stehen.
            sichtbare = [w for w in zeile.__dict__.get("_cells", ()) if w.winfo_manager()]
            check(f"{name}-Zeile blendet auch Beschriftung und Hinweis aus", not sichtbare)
            zeile.set_visible(True)
            root.update_idletasks()
            check(f"{name}-Zeile kommt unverändert zurück", zeile.is_visible())

        # Karten verschwinden ganz, nicht nur ihr Inhalt.
        karte = Card(frame, palette, "Titel", "Untertitel")
        karte.grid(row=20, column=0, sticky="ew")
        root.update_idletasks()
        check("Karte ist zuerst sichtbar", bool(karte.winfo_manager()))
        karte.set_visible(False)
        root.update_idletasks()
        check("Karte lässt sich ausblenden", not karte.winfo_manager())
        karte.set_visible(True)
        root.update_idletasks()
        check("Karte kommt zurück", bool(karte.winfo_manager()))

        # Ausgeblendete Zeilen dürfen keinen Platz mehr belegen.
        messfeld = ttk.Frame(root)
        messfeld.grid()
        proben = [SpinRow(messfeld, i * 2, f"Z{i}", 1, 0, 9, 1) for i in range(4)]
        root.update_idletasks()
        voll = messfeld.winfo_reqheight()
        for zeile in proben[1:]:
            zeile.set_visible(False)
        root.update_idletasks()
        schmal = messfeld.winfo_reqheight()
        check(
            "ausgeblendete Zeilen belegen keinen Platz mehr",
            schmal < voll,
            f"{schmal} px statt vorher {voll} px",
        )
    finally:
        root.destroy()


def _test_mask_editor() -> None:
    print("\n== Maskeneditor ==")
    try:
        import tkinter as tk
    except ImportError:
        print("  über  tkinter fehlt – übersprungen")
        return
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"  über  keine Anzeige verfügbar ({exc}) – übersprungen")
        return

    from PIL import Image

    from app.gui import theme
    from app.gui.mask_editor import MaskEditor
    from app.pipeline_image import _prepare_mask

    root.withdraw()
    try:
        palette = theme.palette_for("dark")
        theme.apply(root, palette)
        quelle = paths.temp_dir() / "maske-quelle.png"
        Image.new("RGB", (1600, 1200), (40, 90, 140)).save(quelle)

        editor = MaskEditor(root, palette, quelle)

        class _Event:
            def __init__(self, x: int, y: int) -> None:
                self.x, self.y = x, y

        check(
            "Maske behält die Größe des Originals",
            editor.full_size == (1600, 1200),
            str(editor.full_size),
        )
        check(
            "Anzeige wird auf Arbeitsgröße verkleinert",
            max(editor.view_size) <= 900,
            str(editor.view_size),
        )
        check("leere Maske markiert nichts", editor._coverage() == 0.0)

        editor.brush.set(60)
        editor._start(_Event(100, 100), erase=False)
        for x in range(100, 400, 10):
            editor._drag(_Event(x, 150))
        editor._finish(_Event(400, 150))
        gemalt = editor._coverage()
        check("Malen markiert Fläche", gemalt > 0.0, f"{gemalt * 100:.2f} %")
        check("Strich wird festgehalten", len(editor._strokes) == 1)

        editor._start(_Event(200, 150), erase=True)
        editor._drag(_Event(260, 150))
        editor._finish(_Event(260, 150))
        radiert = editor._coverage()
        check("Radieren nimmt Fläche weg", radiert < gemalt, f"{radiert * 100:.2f} %")

        editor._undo()
        check(
            "Rückgängig stellt den Stand davor her",
            abs(editor._coverage() - gemalt) < 0.001,
            f"{editor._coverage() * 100:.2f} % statt {gemalt * 100:.2f} %",
        )

        editor._fill_all()
        check("'Alles füllen' markiert alles", editor._coverage() > 0.99)
        # Ganz gefüllt ist so sinnlos wie leer – dafür gibt es img2img.
        check("volle Maske wird beanstandet", "ganze Bild" in editor.problem())
        editor._clear()
        check("'Leeren' setzt zurück", editor._coverage() == 0.0)
        check("leere Maske wird beanstandet", "kein Bereich" in editor.problem())

        editor._start(_Event(300, 300), erase=False)
        editor._drag(_Event(500, 400))
        editor._finish(_Event(500, 400))
        check("teilweise Maske ist in Ordnung", editor.problem() == "", editor.problem())
        ziel = editor.write_mask()
        check("Maske wird geschrieben", ziel.is_file(), str(ziel))

        if ziel is not None:
            with Image.open(ziel) as maske:
                maske.load()
                check(
                    "gespeicherte Maske hat die Größe des Originals",
                    maske.size == (1600, 1200),
                    str(maske.size),
                )
                werte = set(maske.convert("L").getdata())
            check(
                "Maske ist rein schwarz-weiß",
                werte <= {0, 255},
                f"{len(werte)} verschiedene Werte",
            )
            # Der eigentliche Zweck: die Pipeline muss sie annehmen.
            vorbereitet = _prepare_mask(ziel, Image.new("RGB", (1024, 768)))
            check(
                "Inpaint-Vorbereitung nimmt die Maske an",
                vorbereitet.size == (1024, 768) and vorbereitet.mode == "L",
                f"{vorbereitet.size} {vorbereitet.mode}",
            )
    finally:
        root.destroy()


def _test_gui_edit_page() -> None:
    print("\n== Oberfläche: Bearbeiten-Seite ==")
    try:
        import tkinter as tk
    except ImportError:
        print("  über  tkinter fehlt – übersprungen")
        return
    try:
        probe = tk.Tk()
    except tk.TclError as exc:
        print(f"  über  keine Anzeige verfügbar ({exc}) – übersprungen")
        return
    probe.destroy()

    from app.__main__ import Runtime, build_parser
    from app.gui.main_window import MainWindow

    quelle = paths.temp_dir() / "gui-quelle.png"
    _make_test_image(quelle, 640, 480)

    window = MainWindow(Runtime(build_parser().parse_args(["--dummy", "--no-gui"])))
    try:
        window.withdraw()
        window.show_page("imageedit")
        window.update_idletasks()
        nach_schluessel = {v: k for k, v in window._edit_mode_labels.items()}

        def modus(key: str) -> None:
            window.edit_mode.var.set(nach_schluessel[key])
            window._update_edit_mode()
            window.update_idletasks()

        karten = {
            "upscale": window.edit_up_card,
            "colorize": window.edit_color_card,
            "diamond": window.edit_diamond_card,
        }
        # Kern der Sache: je Aufgabe steht genau die passende Karte da,
        # nicht alle vier ausgegraut nebeneinander.
        for key, eigene in karten.items():
            modus(key)
            fremde = [
                name for name, karte in karten.items() if name != key and karte.winfo_manager()
            ]
            check(f"{key}: eigene Karte sichtbar", bool(eigene.winfo_manager()))
            check(f"{key}: fremde Karten verschwunden", not fremde, str(fremde))
        modus("img2img")
        check(
            "img2img zeigt keine der Sonderkarten",
            not any(karte.winfo_manager() for karte in karten.values()),
        )

        # Zeilen folgen der Aufgabe.
        modus("inpaint")
        check("inpaint zeigt die Maske", window.edit_mask.is_visible())
        modus("img2img")
        check("img2img blendet die Maske aus", not window.edit_mask.is_visible())
        modus("diamond")
        check("Vorlage zeigt keinen Prompt", not window.edit_prompt.is_visible())
        check("Vorlage zeigt keine Stärke", not window.edit_strength.is_visible())
        check("Vorlage zeigt keinen Formatwähler", not window.edit_format.is_visible())

        # Nachschärfen holt die Modellregler dazu.
        modus("upscale")
        check("Vergrößern zeigt zunächst keinen Prompt", not window.edit_prompt.is_visible())
        window.edit_refine.var.set(True)
        window._update_edit_mode()
        window.update_idletasks()
        check("Nachschärfen blendet den Prompt ein", window.edit_prompt.is_visible())
        check("Nachschärfen blendet seine Stärke ein", window.edit_refine_strength.is_visible())
        window.edit_refine.var.set(False)

        # Vorschau rechnet mit dem gewählten Bild.
        window._set_edit_sources([quelle])
        window.update_idletasks()
        modus("upscale")
        text = window.edit_estimate.cget("text")
        check("Vergrößern sagt die Zielgröße vorher", "1280x960" in text, text)
        modus("diamond")
        text = window.edit_estimate.cget("text")
        check("Vorlage nennt Raster und Steinzahl", "Steine" in text and "Stück" in text, text)
        check("Vorlage nennt die fertige Größe in cm", "cm" in text, text)
        window.edit_diamond_stones.var.set("120")
        window.update_idletasks()
        text = window.edit_estimate.cget("text")
        check("Vorschau folgt der geänderten Steinzahl", "120x90" in text, text)

        window._set_edit_sources([])
        window.update_idletasks()
        check("ohne Bild bleibt die Vorschau leer", not window.edit_estimate.cget("text"))
    finally:
        window.destroy()


def _test_telefonieren() -> None:
    print("\n== Telefonieren ==")
    from tests import test_call

    test_call.run(check)


def _test_chat() -> None:
    print("\n== Chat und Code-Writer ==")
    from app import models, pipeline_chat

    specs = pipeline_chat.available_models()
    check("Chat-Modelle eingetragen", len(specs) >= 3, str(len(specs)))
    check("Vorgabe ist eingetragen", models.DEFAULTS[models.Task.CHAT] in models.REGISTRY)
    vorgabe = models.resolve(models.DEFAULTS[models.Task.CHAT])
    check("Vorgabe sieht Bilder", vorgabe.sees_images, vorgabe.key)
    check("jedes Chat-Modell nennt eine GGUF-Datei", all(s.gguf_file for s in specs))
    check(
        "nur Vision-Modelle haben einen Bildteil",
        all(s.sees_images == bool(s.mmproj_file) for s in specs),
    )
    # Auf 16 GB RAM muss die Vorgabe passen – ein 7B-Modell als Vorgabe
    # wäre auf dieser Hardware unbenutzbar langsam.
    check("Vorgabe bleibt unter 4 GB", vorgabe.approx_size_mb < 4096, str(vorgabe.approx_size_mb))

    # Der Filter auf die zwei Dateien ist entscheidend: die Repos enthalten
    # ein Dutzend Quantisierungen.
    gefiltert = pipeline_chat._weights_spec(vorgabe)
    check("Download wird auf die GGUF-Dateien begrenzt", len(gefiltert.allow_patterns) == 2)
    check("Gewichtsdatei steht im Filter", vorgabe.gguf_file in gefiltert.allow_patterns)
    check("Bildteil steht im Filter", vorgabe.mmproj_file in gefiltert.allow_patterns)

    ok, grund = pipeline_chat.runtime_available()
    if not ok:
        check("fehlende Laufzeit nennt den Weg", "install" in grund or "bauen" in grund, grund)

    # --- Nachrichten ----------------------------------------------------
    nachricht = pipeline_chat.ChatMessage(role="user", text="Was ist das?")
    check("Text-Nachricht bleibt einfach", nachricht.to_api(True)["content"] == "Was ist das?")

    mit_bild = pipeline_chat.ChatMessage(
        role="user", text="Was ist das?", images=(Path("foto.png"),)
    )
    ohne_sicht = mit_bild.to_api(False)
    check(
        "ohne Bildverständnis wird das Bild benannt, nicht verschwiegen",
        "sieht keine Bilder" in str(ohne_sicht["content"]),
        str(ohne_sicht["content"])[:80],
    )

    # --- Markdown-Anzeige -----------------------------------------------
    try:
        import tkinter as tk
    except ImportError:
        print("  über  tkinter fehlt – Anzeige übersprungen")
        return
    try:
        root = tk.Tk()
    except tk.TclError:
        print("  über  keine Anzeige – Anzeige übersprungen")
        return
    try:
        root.withdraw()
        from app.gui import theme
        from app.gui.chat_view import ChatView

        palette = theme.palette_for("dark")
        theme.apply(root, palette)
        kopiert: list[str] = []
        view = ChatView(root, palette, on_copy=kopiert.append)
        view.grid()
        view.render_markdown(
            'Ein **Test** mit `inline` und:\n\n```python\nprint("hallo")\n```\n\n'
            "# Titel\n- Punkt eins\n"
        )
        root.update_idletasks()

        inhalt = view.text.get("1.0", "end")
        check("Code-Block wird erkannt", view._blocks == 1, str(view._blocks))
        check("Zaunzeichen verschwinden", "```" not in inhalt)
        check("Code steht im Text", 'print("hallo")' in inhalt)
        check("Sprache wird benannt", "python" in inhalt)
        for tag in ("code_block", "code_inline", "fett", "h1", "liste"):
            check(f"Auszeichnung '{tag}' gesetzt", bool(view.text.tag_ranges(tag)))

        from tkinter import ttk as _ttk

        knoepfe = [w for w in view.text.winfo_children() if isinstance(w, _ttk.Button)]
        check("Code-Block hat einen Kopierknopf", len(knoepfe) == 1, str(len(knoepfe)))
        if knoepfe:
            knoepfe[0].invoke()
            check(
                "Kopieren liefert genau den Code",
                bool(kopiert) and kopiert[0].strip() == 'print("hallo")',
                repr(kopiert[:1]),
            )
    finally:
        root.destroy()


def _test_private_use() -> None:
    print("\n== Private Nutzung ==")
    from app import licensing, models
    from app.models import Commercial

    eingeschraenkt = [s for s in models.REGISTRY.values() if s.commercial is not Commercial.ALLOWED]
    check("es gibt eingeschränkte Modelle", bool(eingeschraenkt), str(len(eingeschraenkt)))

    def freigegeben() -> int:
        anzahl = 0
        for spec in eingeschraenkt:
            try:
                models.check_allowed(spec)
                anzahl += 1
            except models.ModelBlocked:
                pass
        return anzahl

    war_an = licensing.private_use_accepted()
    try:
        licensing.revoke_private_use()
        check("Vorgabe ist gesperrt (fail-closed)", freigegeben() == 0, str(freigegeben()))
        check("Zustimmung fehlt zunächst", not licensing.private_use_accepted())

        # Die Sperrmeldung muss den Weg nennen, nicht nur das Nein.
        gesperrt = next(s for s in eingeschraenkt if s.commercial is not Commercial.ALLOWED)
        try:
            models.check_allowed(gesperrt)
            meldung = ""
        except models.ModelBlocked as exc:
            meldung = str(exc)
        check("Sperrmeldung nennt die Freischaltung", "Private Nutzung" in meldung, meldung[:90])

        licensing.accept_private_use()
        check("Zustimmung wird festgehalten", licensing.private_use_accepted())
        offen = freigegeben()
        check("nach Zustimmung sind Modelle frei", offen > 0, str(offen))

        # Die Stimm-Einwilligung ist ein anderes Tor und darf sich davon
        # nicht öffnen lassen – dort geht es um Persönlichkeitsrecht.
        mit_consent = [s for s in eingeschraenkt if s.consent_component]
        if mit_consent:
            noch_zu = 0
            for spec in mit_consent:
                try:
                    models.check_allowed(spec)
                except models.ModelBlocked:
                    noch_zu += 1
            check(
                "Einwilligungs-Tor bleibt trotz Freischaltung zu",
                noch_zu == len(mit_consent),
                f"{noch_zu} von {len(mit_consent)}",
            )

        licensing.revoke_private_use()
        check("Widerruf sperrt wieder", freigegeben() == 0, str(freigegeben()))

        # Auflagen dürfen durch die Freischaltung nicht verschwinden.
        bauteil = licensing.COMPONENTS[licensing.PRIVATE_USE_COMPONENT]
        check("Freischaltung nennt Auflagen", len(bauteil.obligations) >= 3)
        text = " ".join(bauteil.obligations)
        check("Auflage nennt die Grenze zur Kommerzialisierung", "Geld verdient" in text)
        check("Auflage verbietet Weitergabe der Anwendung", "weitergegeben" in text)
    finally:
        if war_an:
            licensing.accept_private_use()
        else:
            licensing.revoke_private_use()


def _test_onnx_backends() -> None:
    print("\n== ONNX / OpenVINO ==")
    from app import models, pipeline_onnx
    from app.accel import Backend, CpuInfo, GpuDevice, HardwareReport, Vendor, resolve_backend
    from app.config import DEVICE_CHOICES, AppConfig

    spec = models.resolve("sdxl-base")

    # --- Familien -----------------------------------------------------------
    check("SDXL wird als sdxl erkannt", pipeline_onnx.family_of(spec) == "sdxl")
    check(
        "FLUX wird als flux erkannt",
        pipeline_onnx.family_of(models.resolve("flux-schnell")) == "flux",
    )
    moeglich, grund = pipeline_onnx.supported(models.resolve("flux-schnell"), Backend.DML)
    check("FLUX wird für ONNX abgelehnt", not moeglich)
    check("Ablehnung nennt den Ausweg", "CPU bzw. CUDA" in grund, grund)
    check("SDXL wird zugelassen", pipeline_onnx.supported(spec, Backend.DML)[0])
    check(
        "unbekanntes Backend hat keinen ONNX-Weg",
        not pipeline_onnx.supported(spec, "quantenkern")[0],
    )

    # --- Laufzeit fehlt -> ehrliche Meldung ---------------------------------
    for backend in (Backend.DML, Backend.OPENVINO):
        vorhanden, hinweis = pipeline_onnx.runtime_available(backend)
        if not vorhanden:
            check(
                f"{backend}: fehlende Laufzeit nennt den Befehl",
                "pip install" in hinweis,
                hinweis,
            )
        else:
            check(f"{backend}: Laufzeit gemeldet", bool(hinweis))

    # --- Bereitschaft -------------------------------------------------------
    zustand = models.readiness(spec)
    for backend in (Backend.DML, Backend.OPENVINO):
        s = zustand[backend]
        check(
            f"{backend}: ohne Konvertat nicht bereit",
            not s.ready,
            f"ready={s.ready}",
        )
        check(f"{backend}: Begründung vorhanden", bool(s.note), s.note)
        # Ohne Laufzeit darf keine Konvertierung angeboten werden – sie
        # könnte gar nicht laufen.
        if not pipeline_onnx.runtime_available(backend)[0]:
            check(
                f"{backend}: ohne Laufzeit wird keine Konvertierung angeboten",
                not s.needs_conversion,
            )

    # Liegt ein Konvertat, gilt das Backend als bereit.
    ziel = models.converted_dir(spec, Backend.OPENVINO)
    ziel.mkdir(parents=True, exist_ok=True)
    (ziel / "openvino_model.bin").write_bytes(b"x" * 2048)
    try:
        mit = models.readiness(spec)[Backend.OPENVINO]
        check("vorhandenes Konvertat macht bereit", mit.ready, mit.note)
    finally:
        shutil.rmtree(ziel, ignore_errors=True)

    # --- Geräteauswahl in OpenVINO -----------------------------------------
    import app.accel as accel

    accel._openvino_cache = (("CPU", "GPU", "NPU"), "")
    geraet, notiz = accel.openvino_target()
    # GPU vor NPU: eine NPU ist auf kleine, quantisierte Netze ausgelegt
    # und bei Diffusionsmodellen langsamer als die iGPU. Wer die NPU
    # trotzdem will (sparsamer, Dauerlast), stellt sie fest ein.
    check(
        "ohne Vorgabe wird die schnellere GPU genommen",
        geraet == accel.OPENVINO_DEVICE_ORDER[0],
        f"{geraet} – {notiz}",
    )
    check("feste Wahl NPU wird beachtet", accel.openvino_target("NPU")[0] == "NPU")
    check("feste Wahl GPU wird beachtet", accel.openvino_target("GPU")[0] == "GPU")
    fehlend, warum = accel.openvino_target("NPU2")
    check("unverfügbares Wunschgerät wird abgelehnt", fehlend == "")
    check("Ablehnung listet die vorhandenen Geräte", "CPU" in warum, warum)
    accel._openvino_cache = ((), "OpenVINO ist nicht installiert.")
    check("ohne Gerät bleibt die Wahl leer", accel.openvino_target()[0] == "")

    # --- Backend-Kette kennt OpenVINO --------------------------------------
    check("'openvino' ist eine gültige Geräteeinstellung", "openvino" in DEVICE_CHOICES)
    check("OpenVINO steht in der Reihenfolge", Backend.OPENVINO in accel.BACKEND_ORDER)
    check("OpenVINO hat eine Beschriftung", bool(accel.BACKEND_LABELS.get(Backend.OPENVINO)))

    report = HardwareReport(
        gpus=(GpuDevice(0, "Intel(R) Arc(TM) Graphics", Vendor.INTEL, 8192, "cim"),),
        cpu=CpuInfo("Intel(R) Core(TM) Ultra 7 155H", 8, 16, 32768, "AMD64"),
        os_name="Windows 11",
    )
    accel._openvino_cache = ((), "OpenVINO ist nicht installiert.")
    accel._torch_cuda_cache = (False, "kein NVIDIA-Treiber")
    accel._onnx_cache = (("DmlExecutionProvider", "CPUExecutionProvider"), "")
    ohne_laufzeit = {
        Backend.CUDA: models.ModelReadiness(ready=False, needs_conversion=False),
        Backend.CPU: models.ModelReadiness(ready=True),
        Backend.DML: models.ModelReadiness(ready=False, needs_conversion=False, note="optimum"),
        Backend.OPENVINO: models.ModelReadiness(
            ready=False, needs_conversion=False, note="optimum"
        ),
    }
    for gewaehlt in ("auto", "dml", "openvino"):
        plan = resolve_backend(AppConfig(device=gewaehlt), ohne_laufzeit, report)
        check(
            f"device={gewaehlt} fällt ohne Laufzeit auf CPU",
            plan.backend == Backend.CPU,
            plan.backend,
        )

    # Mit Konvertat wird OpenVINO auch gewählt.
    accel._openvino_cache = (("NPU", "GPU", "CPU"), "")
    bereit = dict(ohne_laufzeit)
    bereit[Backend.OPENVINO] = models.ModelReadiness(ready=True, needs_conversion=False)
    check(
        "mit Konvertat und Gerät wird OpenVINO gewählt",
        resolve_backend(AppConfig(device="openvino"), bereit, report).backend == Backend.OPENVINO,
    )
    accel._openvino_cache = None


def _test_download_hardening() -> None:
    print("\n== Download-Härtung ==")
    from app import models
    from app.accel import clean_error

    spec = models.resolve("flux-schnell")

    class _Response:
        def __init__(self, code: int) -> None:
            self.status_code = code

    class _HubError(Exception):
        def __init__(self, code: int) -> None:
            super().__init__(f"{code} Client Error")
            self.response = _Response(code)

    # --- Fehler werden unterschieden ---------------------------------------
    gesperrt = models.classify_hub_error(_HubError(401), spec)
    check("401 nennt den Zugang als Ursache", "zugangsbeschränkt" in gesperrt, gesperrt[:80])
    check("401 nennt die Modellseite", spec.repo_id in gesperrt)
    check("401 erklärt den Token-Weg", "HF_TOKEN" in gesperrt)
    check(
        "403 wird wie 401 behandelt",
        "zugangsbeschränkt" in models.classify_hub_error(_HubError(403), spec),
    )
    fehlt = models.classify_hub_error(_HubError(404), spec)
    check("404 nennt das fehlende Repo", "gibt es auf Hugging Face nicht" in fehlt, fehlt[:80])
    check(
        "429 nennt die Bremse",
        "bremst" in models.classify_hub_error(_HubError(429), spec),
    )
    check(
        "500 nennt den Serverfehler",
        "Serverfehler" in models.classify_hub_error(_HubError(503), spec),
    )
    voll = OSError(errno.ENOSPC, "No space left on device")
    check("volle Platte wird erkannt", "Kein Platz" in models.classify_hub_error(voll, spec))

    class ConnectionError_(Exception):
        pass

    netz = models.classify_hub_error(ConnectionError_("abgerissen"), spec)
    check("Netzfehler nennt die Fortsetzung", "setzt dort fort" in netz, netz[:80])

    # --- Wiederholen nur, wo es etwas bringt --------------------------------
    check("401 wird nicht wiederholt", not models._is_retryable(_HubError(401)))
    check("404 wird nicht wiederholt", not models._is_retryable(_HubError(404)))
    check("volle Platte wird nicht wiederholt", not models._is_retryable(voll))
    check("Abbruch wird nicht wiederholt", not models._is_retryable(models.DownloadCancelled("x")))
    check("Serverfehler wird wiederholt", models._is_retryable(_HubError(502)))
    check("Bremse wird wiederholt", models._is_retryable(_HubError(429)))
    check("Netzfehler wird wiederholt", models._is_retryable(ConnectionError_("weg")))

    # --- Absichtliche Ablehnungen behalten ihren Wortlaut -------------------
    lang = models.ModelAccessDenied("Zeile eins\nZeile zwei mit Anleitung\n" + "x" * 400)
    gezeigt = clean_error(lang)
    check("Zugangsfehler behält die Zeilen", "\n" in gezeigt)
    check("Zugangsfehler wird nicht gekürzt", "…" not in gezeigt and len(gezeigt) > 300)
    check("Zugangsfehler gilt als erwartet", getattr(lang, "expected", False))
    check("Lizenzsperre gilt als erwartet", getattr(models.ModelBlocked("x"), "expected", False))
    # Fremde Fehler werden weiterhin eingedampft.
    fremd = clean_error(ValueError("a\nb\n" + "y" * 500))
    check("Fremdfehler bleibt einzeilig", "\n" not in fremd)
    check("Fremdfehler wird gekürzt", fremd.endswith("…"))

    # --- Angefangene Dateien überleben den Fehlerpfad -----------------------
    ordner = paths.temp_dir() / "download-reste"
    ordner.mkdir(parents=True, exist_ok=True)
    teil = ordner / ("gross.safetensors" + models.PART_SUFFIX)
    teil.write_bytes(b"x" * 2048)
    (ordner / "alt.lock").write_text("", encoding="utf-8")
    entfernt = models._cleanup_incomplete(ordner)
    check("Sperrdatei wird entfernt", entfernt == 1, str(entfernt))
    check("angefangene Datei bleibt liegen", teil.is_file())
    check("angefangene Bytes werden gemeldet", models.resumable_bytes(ordner) == 2048)
    models._cleanup_incomplete(ordner, keep_parts=False)
    check("ausdrückliches Verwerfen räumt auch die Teile weg", not teil.is_file())

    # --- Platzprüfung -------------------------------------------------------
    genug, _note = models.check_disk_space(ordner, 1024)
    check("kleiner Bedarf passt immer", genug)
    knapp, knapp_note = models.check_disk_space(ordner, 900 * 1024**4)
    check("unmöglicher Bedarf wird abgelehnt", not knapp)
    check("Ablehnung nennt Zahlen", "GB frei" in knapp_note, knapp_note)


def _test_memory_hygiene() -> None:
    print("\n== Speicher ==")
    from app import pipeline_image
    from app.jobs import JobQueue, _spent_handler

    # Aufräumen darf torch nicht nachladen, wenn es gar nicht im Spiel ist.
    vorher = "torch" in sys.modules
    pipeline_image.release_memory(deep=True)
    check(
        "Aufräumen lädt torch nicht nach",
        ("torch" in sys.modules) == vorher,
        "torch wurde beim Aufräumen importiert",
    )

    queue = JobQueue(workers=1)
    haltepunkt: list[object] = ["belegt"]

    def handler(_context) -> str:
        return "fertig"

    handler.beweis = haltepunkt  # type: ignore[attr-defined]
    job_id = queue.submit("test", "Speicher", handler)
    deadline = time.time() + 20
    while time.time() < deadline and not queue.get(job_id).state.finished:
        time.sleep(0.02)
    check("Auftrag läuft durch", queue.get(job_id).state.finished)
    with queue._lock:
        job = queue._jobs[job_id]
        check(
            "Handler wird nach dem Auftrag losgelassen",
            job.handler is _spent_handler,
            type(job.handler).__name__,
        )

    # Erledigte Aufträge werden gedeckelt.
    for index in range(12):
        queue.submit("test", f"füller {index}", lambda _c: None)
    deadline = time.time() + 20
    while time.time() < deadline and any(not view.state.finished for view in queue.snapshot()):
        time.sleep(0.02)
    with queue._lock:
        queue._prune_finished(keep=5)
        verbleibend = len(queue._jobs)
    check("erledigte Aufträge werden gedeckelt", verbleibend == 5, str(verbleibend))
    queue.shutdown(wait=True, timeout=10)


def _test_content_gate() -> None:
    print("\n== Inhaltssperre ==")
    from app import contentgate, pipeline_image
    from app.config import AppConfig

    # 1. Erwachsenen-Inhalte sind Vorgabe, lassen sich aber abschalten.
    freigeschaltet = AppConfig()
    check("Erwachsenen-Inhalte sind Vorgabe", freigeschaltet.nsfw_enabled)
    check(
        "Vorgabe wird als zugelassen gemeldet",
        pipeline_image.adult_content_allowed(freigeschaltet)[0],
    )
    config = freigeschaltet.with_values(nsfw_enabled=False)
    check("abschaltbar", not pipeline_image.adult_content_allowed(config)[0])

    # 2. Erwachsenendarstellungen dürfen nicht mitgesperrt werden.
    for prompt in (
        "nude woman, 25 years old, studio light",
        "erotic photo of an adult couple",
        "nackte frau, 30 jahre alt, atelier",
    ):
        check(
            f"erlaubt: {prompt[:34]}",
            contentgate.inspect(prompt)[0],
            contentgate.inspect(prompt)[1],
        )

    # 3. Harmlose Prompts mit Kindern bleiben erlaubt.
    check(
        "erlaubt: Kind ohne sexuellen Zusammenhang",
        contentgate.inspect("a child playing football in a park")[0],
    )

    # 3b. Keine Fehlalarme durch Wortbestandteile. Eine Sperre, die bei
    #     erwachsenen Motiven grundlos zuschlägt, wird ausgebaut.
    for prompt in (
        "nude woman, kindness in her eyes",
        "naked adult, kindly lit portrait",
        "lolita fashion dress, adult model, fully clothed",
        "topless adult woman, minorca island",
    ):
        erlaubt, grund = contentgate.inspect(prompt)
        check(f"kein Fehlalarm: {prompt[:34]}", erlaubt, grund)

    # 4. Kombination Minderjährig + sexuell wird abgelehnt.
    for prompt in (
        "nude child",
        "naked teen girl",
        "nacktes kind",
        "sexy schoolgirl, 14 years old",
        "erotic, 12 jahre alt",
        "loli",
        "n4ked t3en",
        "lolita, nude",
        "nacktes kleinkind",
        "erotische schülerin",
        "nude, kinderzimmer",
    ):
        erlaubt, _grund = contentgate.inspect(prompt)
        check(f"abgelehnt: {prompt[:34]}", not erlaubt, "wurde durchgelassen")

    # 5. Umgehung über den Negativ-Prompt zieht nicht.
    erlaubt, _ = contentgate.inspect("beautiful portrait", "nude child")
    check("Negativ-Prompt wird mitgeprüft", not erlaubt)

    # 6. Auftrag wird geworfen, nicht still ignoriert.
    try:
        contentgate.enforce("naked toddler")
        geworfen = False
    except contentgate.BlockedContent:
        geworfen = True
    check("enforce() wirft BlockedContent", geworfen)

    # 7. Sperre lässt sich nicht über die Konfiguration abschalten.
    gedreht, meldungen = AppConfig(nsfw_block_minors=False).validated()
    check("nsfw_block_minors wird zurückgesetzt", gedreht.nsfw_block_minors, "blieb abgeschaltet")
    check(
        "Rücksetzung wird gemeldet",
        any("nsfw_block_minors" in m for m in meldungen),
        str(meldungen),
    )

    # 8. Inhaltsprüfung des Modells: nur mit Freigabe abschalten.
    modell = paths.temp_dir() / "modell-mit-pruefung"
    (modell / "safety_checker").mkdir(parents=True, exist_ok=True)
    ohne = paths.temp_dir() / "modell-ohne-pruefung"
    ohne.mkdir(parents=True, exist_ok=True)

    kwargs, _grund = pipeline_image.safety_checker_kwargs(config, modell)
    check("ohne Freigabe bleibt die Inhaltsprüfung an", kwargs == {}, str(kwargs))
    kwargs, _grund = pipeline_image.safety_checker_kwargs(freigeschaltet, modell)
    check(
        "mit Freigabe wird die Inhaltsprüfung abgeschaltet",
        kwargs.get("safety_checker", "fehlt") is None
        and kwargs.get("requires_safety_checker") is False,
        str(kwargs),
    )
    kwargs, _grund = pipeline_image.safety_checker_kwargs(
        freigeschaltet.with_values(nsfw_disable_safety_checker=False), modell
    )
    check("auf Wunsch bleibt sie trotz Freigabe an", kwargs == {}, str(kwargs))
    kwargs, _grund = pipeline_image.safety_checker_kwargs(freigeschaltet, ohne)
    check("Modell ohne Prüfung bekommt keine Zusatzargumente", kwargs == {}, str(kwargs))

    # 9. Schutzbegriffe landen im Negativ-Prompt, aber nur einmal.
    ergaenzt = contentgate.with_protective_negative("blurry")
    check("Schutzbegriffe werden angehängt", "blurry" in ergaenzt and "child" in ergaenzt, ergaenzt)
    check("keine Doppelung", contentgate.with_protective_negative(ergaenzt) == ergaenzt)
    check(
        "kein 'school uniform' im Schutz-Negativ",
        "school uniform" not in contentgate.PROTECTIVE_NEGATIVE,
        "würde erwachsene Cosplay-Motive beschneiden",
    )


def _test_model_registry() -> None:
    """Die Feinabstimmungen für Erwachsenen-Inhalte. Ohne Netz."""
    print("\n== Modell-Registrierung ==")
    from app import models

    erwartet = (
        "pony-v6",
        "noobai-xl",
        "realvis-xl",
        "juggernaut-xl",
        "nsfw-gen",
        "realistic-vision",
        "dreamshaper",
    )
    fehlend = [k for k in erwartet if k not in models.REGISTRY]
    check("Feinabstimmungen eingetragen", not fehlend, f"fehlt: {fehlend}")

    for alias, ziel in (
        ("pony", "pony-v6"),
        ("noob", "noobai-xl"),
        ("realvis", "realvis-xl"),
        ("rv6", "realistic-vision"),
    ):
        check(
            f"Alias '{alias}' löst auf",
            models.resolve(alias).key == ziel,
            models.resolve(alias).key,
        )

    for key in erwartet:
        spec = models.REGISTRY[key]
        check(
            f"{key}: Größe und VRAM gesetzt",
            spec.approx_size_mb > 0 and spec.min_vram_mb > 0,
            f"{spec.approx_size_mb} MB / {spec.min_vram_mb} MB",
        )
        check(
            f"{key}: kein gesperrtes Modell",
            spec.commercial is not models.Commercial.DENIED,
            spec.commercial.value,
        )

    pony = models.REGISTRY["pony-v6"]
    check("Pony V6 ist ein Einzeldatei-Checkpoint", pony.is_single_file)
    check(
        "Einzeldatei nur über allow_patterns geladen",
        pony.allow_patterns == ("v6.safetensors",),
        str(pony.allow_patterns),
    )
    check("Ordner-Modelle sind keine Einzeldatei", not models.REGISTRY["realvis-xl"].is_single_file)
    check(
        "Bauplan liegt im Datenverzeichnis",
        models.config_dir(pony.single_file_config).parent == paths.models_dir() / "configs",
        str(models.config_dir(pony.single_file_config)),
    )

    # Der Bauplan-Filter darf keine Gewichte durchlassen – sonst lädt eine
    # Konfigurationsabfrage versehentlich mehrere GB.
    roh = [
        ("model_index.json", 600),
        ("unet/config.json", 1800),
        ("unet/diffusion_pytorch_model.fp16.safetensors", 5_000_000_000),
        ("tokenizer/merges.txt", 500),
        ("vae/diffusion_pytorch_model.bin", 300_000_000),
    ]
    spec = models.ModelSpec(
        key="x",
        repo_id="a/b",
        task=models.Task.IMAGE,
        title="x",
        license_id="x",
        license_url="",
        commercial=models.Commercial.ALLOWED,
        variant="",
        allow_patterns=models._CONFIG_ALLOW,
        ignore_patterns=models._CONFIG_IGNORE,
    )
    namen = {f.name for f in models.select_files(spec, roh, None)}
    check(
        "Bauplan-Filter nimmt nur Konfiguration",
        namen == {"model_index.json", "unet/config.json", "tokenizer/merges.txt"},
        str(sorted(namen)),
    )


def _test_build_script() -> None:
    """Das Bauskript darf beim Aufräumen nichts Geladenes vernichten.

    Beide Fehler waren real und teuer. Erst löschte '-Clean' das fertige
    Bundle samt data\\models, weil die Rettung der Nutzerdaten erst
    hundert Zeilen später kam. Danach zeigte sich der größere Teil: das
    Venv (torch mit CUDA rund 8 GB), der Modell-Zwischenspeicher (SDXL
    6,6 GB) und der ffmpeg-Download flogen ebenfalls mit. Ein "nur neu
    bauen" kostete damit über 15 GB Download.

    Geprüft wird deshalb, was NICHT im Löschzweig steht, und in welcher
    Reihenfolge gerettet wird.
    """
    print("\n== Bauskript ==")
    skript = ROOT / "build-windows.ps1"
    if not skript.is_file():
        check("Bauskript vorhanden", False, str(skript))
        return
    text = skript.read_text(encoding="utf-8-sig")

    # --- Reihenfolge: retten, dann löschen, dann zurücklegen ---------
    stash = text.find("$CleanStash = Join-Path $Root")
    bundle_weg = text.find('Remove-Tree $DistDir "fertiges Bundle"')
    zurueck = text.find("Move-Item -Path $CleanStash")
    check("Clean sichert die Nutzerdaten", stash > 0)
    check("Clean entfernt das fertige Bundle", bundle_weg > 0)
    check(
        "Sicherung läuft VOR dem Löschen",
        0 < stash < bundle_weg,
        f"stash={stash} loeschen={bundle_weg}",
    )
    check(
        "Rückgabe läuft NACH dem Löschen",
        bundle_weg < zurueck,
        f"loeschen={bundle_weg} zurueck={zurueck}",
    )

    # --- Was -Clean in Ruhe lassen muss ------------------------------
    clean_block = text[text.find("if ($Clean) {") : text.find("if ($FreshVenv) {")]
    check(
        "Clean fasst das Venv nicht an",
        "$VenvDir" not in clean_block,
        "sonst werden ~8 GB pip-Pakete neu geladen",
    )
    check(
        "Clean fasst den Modell-Zwischenspeicher nicht an",
        "Remove-Tree $StageCache" not in clean_block,
        "sonst wird SDXL (6,6 GB) neu geladen",
    )
    check(
        "Clean fasst den ffmpeg-Zwischenspeicher nicht an",
        "Remove-Tree $FfmpegCache" not in clean_block,
    )
    check(
        "Clean entfernt den PyInstaller-Arbeitsordner",
        "Remove-Tree $PyiWork" in clean_block,
        "ohne das übernimmt der neue Bau alte Ergebnisse",
    )

    # --- Die Schalter für die harten Fälle ---------------------------
    for schalter, zweck in (
        ("$PurgeData", "Nutzerdaten"),
        ("$FreshVenv", "Venv"),
        ("$PurgeCache", "Zwischenspeicher"),
    ):
        check(
            f"-{schalter[1:]} als bewusster Ausweg für {zweck}",
            f"[switch]{schalter}" in text,
        )
    check(
        "FreshVenv löscht wirklich das Venv",
        "Remove-Tree $VenvDir" in text,
    )
    check(
        "PurgeCache löscht beide Zwischenspeicher",
        "Remove-Tree $StageCache" in text and "Remove-Tree $FfmpegCache" in text,
    )
    check(
        "PurgeData überspringt die Sicherung",
        "(-not $PurgeData)" in text,
    )
    check(
        "PurgeData warnt vor dem Datenverlust",
        'Write-Warning "-PurgeData:' in text,
    )
    check(
        "das Skript sagt, was stehen bleibt",
        "Venv bleibt bestehen" in text and "Zwischenspeicher bleiben bestehen" in text,
    )
    check(
        "verwaiste Sicherung wird gemeldet",
        "abgebrochenen Lauf" in text,
    )
    check(
        "Zwischenlager ist in .gitignore",
        ".data-stash-*/" in (ROOT / ".gitignore").read_text(encoding="utf-8"),
    )
    liesmich = (ROOT / "README.md").read_text(encoding="utf-8")
    check(
        "README verspricht keine Löschung mehr",
        "`-Clean` löscht sie absichtlich" not in liesmich,
    )


def _test_single_instance() -> None:
    print("\n== Einzelinstanz ==")
    from app import single_instance

    guard = single_instance.acquire(suffix="smoke")
    check("Sperre wird gesetzt", guard.acquired, guard.reason)
    check(
        "zweiter Aufruf im selben Prozess meldet die Sperre",
        single_instance.acquire(suffix="smoke").acquired,
    )
    single_instance.release()


if __name__ == "__main__":
    raise SystemExit(main())
