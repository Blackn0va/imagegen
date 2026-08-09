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
        _test_download_hardening()
        _test_memory_hygiene()
        _test_content_gate()
        _test_model_registry()
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
