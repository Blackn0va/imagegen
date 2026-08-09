"""Arbeiter für Klonstimmen – läuft in einer EIGENEN Umgebung.

Warum ein eigener Prozess: chatterbox-tts zieht torch 2.6 (ohne CUDA-Build),
diffusers 0.29 und transformers 5.x nach. In derselben Umgebung wie Bild und
Video würde das die GPU-Beschleunigung und die Videopipelines zerstören.
Deshalb liegt die Klon-Laufzeit in einem getrennten Interpreter und wird über
die Kommandozeile aufgerufen – dasselbe Muster wie bei ffmpeg.

Aufruf (die Anwendung macht das selbst):

    python voice_worker.py synth --ref referenz.wav --text "..." \
        --out ausgabe.wav [--language de] [--exaggeration 0.5] [--cfg 0.5] \
        [--seed 0] [--device cuda]

    python voice_worker.py check      # meldet Bereitschaft als JSON

Ausgabe auf stdout ist JSON, damit die Anwendung sie sicher auswerten kann.
Fortschritt und Meldungen gehen auf stderr.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from pathlib import Path

# Sprachcodes, die die mehrsprachige Fassung kennt.
MULTILINGUAL = {
    "ar",
    "da",
    "de",
    "el",
    "en",
    "es",
    "fi",
    "fr",
    "he",
    "hi",
    "it",
    "ja",
    "ko",
    "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "tr",
    "zh",
}


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False))
    sys.stdout.flush()


def cmd_check_fast() -> int:
    """Bereitschaft prüfen, ohne etwas zu importieren.

    Nur Dateisystem und Paket-Metadaten – Bruchteile einer Sekunde statt
    über einer Minute. Ob CUDA nutzbar ist, steht hier bewusst nicht drin;
    das kostet einen torch-Import und wird erst beim Laden ermittelt.
    """
    import importlib.metadata as metadata
    import importlib.util as util

    info: dict = {"ok": False, "mode": "fast"}
    missing = []
    for package in ("torch", "chatterbox"):
        try:
            found = util.find_spec(package) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(package)
    if missing:
        info["error"] = f"{', '.join(missing)} fehlt in dieser Umgebung."
        _emit(info)
        return 1

    for name, package in (("torch", "torch"), ("chatterbox", "chatterbox-tts")):
        with contextlib.suppress(Exception):
            info[name + "_version"] = metadata.version(package)
    info["torch"] = info.get("torch_version", "")

    # Mehrsprachigkeit an der Datei erkennen: ein Import von
    # chatterbox.mtl_tts würde das ganze Paket laden.
    try:
        spec = util.find_spec("chatterbox")
        roots = list(getattr(spec, "submodule_search_locations", []) or [])
        info["multilingual"] = any((Path(root) / "mtl_tts.py").is_file() for root in roots)
    except Exception:
        info["multilingual"] = False

    # pkg_resources wird vom Wasserzeichen-Paket 'perth' gebraucht; fehlt es,
    # scheitert erst das Modellladen mit "'NoneType' object is not callable".
    try:
        info["pkg_resources"] = util.find_spec("pkg_resources") is not None
    except (ImportError, ValueError):
        info["pkg_resources"] = False
    if not info["pkg_resources"]:
        info["error"] = (
            "setuptools<81 fehlt – das Wasserzeichen-Paket 'perth' braucht pkg_resources."
        )
        _emit(info)
        return 1

    info["ok"] = True
    _emit(info)
    return 0


def _pick_device(requested: str) -> str:
    import torch

    if requested == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def cmd_check(args: argparse.Namespace) -> int:
    # Schnellprüfung ist die Vorgabe: sie schaut nur nach, ob die Pakete da
    # sind. Der volle Import von torch und chatterbox dauert auf diesem
    # Rechner über eine Minute – das darf die Oberfläche nie blockieren.
    if not getattr(args, "full", False):
        return cmd_check_fast()

    info: dict = {"ok": False, "mode": "full"}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = bool(torch.cuda.is_available())
    except Exception as exc:
        info["error"] = f"torch fehlt: {exc}"
        _emit(info)
        return 1
    try:
        import chatterbox  # noqa: F401
        from chatterbox.tts import ChatterboxTTS  # noqa: F401

        info["chatterbox"] = True
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401

            info["multilingual"] = True
        except Exception:
            info["multilingual"] = False
        info["ok"] = True
    except Exception as exc:
        info["error"] = f"chatterbox fehlt: {exc}"
    _emit(info)
    return 0 if info["ok"] else 1


def _load_model(device: str, language: str):
    """Modell holen. Beim ersten Mal lädt das mehrere GB herunter."""
    model = None
    multilingual = False
    if language != "en":
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            print("lade mehrsprachiges Modell ...", file=sys.stderr, flush=True)
            model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            multilingual = True
        except Exception as exc:
            print(f"mehrsprachige Fassung nicht verfuegbar: {exc}", file=sys.stderr, flush=True)
    if model is None:
        from chatterbox.tts import ChatterboxTTS

        print("lade englisches Modell ...", file=sys.stderr, flush=True)
        model = ChatterboxTTS.from_pretrained(device=device)
    return model, multilingual


def cmd_prepare(args: argparse.Namespace) -> int:
    """Modell einmalig laden, damit die spätere Synthese kurz ist.

    Getrennt vom Erzeugen, weil der erste Download mehrere GB groß ist und
    sonst in das Zeitlimit der Sprachausgabe fällt.
    """
    device = _pick_device(args.device)
    language = (args.language or "de").lower()[:2]
    model, multilingual = _load_model(device, language)
    _emit(
        {
            "ok": True,
            "device": device,
            "multilingual": multilingual,
            "sample_rate": int(getattr(model, "sr", 0) or 0),
        }
    )
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    import torch
    import torchaudio

    reference = Path(args.ref)
    if not reference.is_file():
        _emit({"ok": False, "error": f"Referenzaufnahme fehlt: {reference}"})
        return 2

    device = _pick_device(args.device)
    language = (args.language or "de").lower()[:2]

    if args.seed:
        torch.manual_seed(args.seed)

    # Mehrere Sätze in EINEM Aufruf: das Modell wiegt mehrere GB, und ein
    # Prozess je Satz würde es jedes Mal neu laden. Aus 5 Sätzen würden so
    # fünf Ladevorgänge statt einem.
    texts = [args.text]
    if getattr(args, "text_file", ""):
        try:
            geladen = json.loads(Path(args.text_file).read_text(encoding="utf-8"))
            if isinstance(geladen, list) and geladen:
                texts = [str(item) for item in geladen if str(item).strip()]
        except (OSError, ValueError) as exc:
            _emit({"ok": False, "error": f"Textliste nicht lesbar: {exc}"})
            return 2

    model, multilingual = _load_model(device, language)

    kwargs = {
        "audio_prompt_path": str(reference),
        "exaggeration": float(args.exaggeration),
        "cfg_weight": float(args.cfg),
        "temperature": float(getattr(args, "temperature", 0.8)),
    }
    if multilingual:
        kwargs["language_id"] = language if language in MULTILINGUAL else "en"

    stuecke = []
    for index, satz in enumerate(texts, start=1):
        # Auf stderr, damit die Anwendung Fortschritt sieht und der
        # Stillstands-Wachhund nicht zuschlägt.
        print(f"erzeuge Satz {index}/{len(texts)} ...", file=sys.stderr, flush=True)
        stuecke.append(model.generate(satz, **kwargs))

    if len(stuecke) == 1:
        wav = stuecke[0]
    else:
        pause = torch.zeros(1, int(model.sr * 0.22))
        teile = []
        for index, stueck in enumerate(stuecke):
            teil = stueck.detach().cpu()
            if teil.dim() == 1:
                teil = teil.unsqueeze(0)
            if index:
                teile.append(pause)
            teile.append(teil)
        wav = torch.cat(teile, dim=-1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Als 16-Bit-PCM speichern. torchaudio schreibt sonst Float-WAV
    # (Formatkennung 3); das kann die Standardbibliothek nicht lesen, und
    # das Zusammenhängen mehrerer Sätze würde scheitern.
    audio = wav.detach().cpu().clamp(-1.0, 1.0)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    torchaudio.save(str(out), audio, model.sr, encoding="PCM_S", bits_per_sample=16)

    _emit(
        {
            "ok": True,
            "output": str(out),
            "sentences": len(texts),
            "sample_rate": int(model.sr),
            "seconds": round(float(wav.shape[-1]) / float(model.sr), 2),
            "device": device,
            "multilingual": multilingual,
            "language": language,
        }
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="voice_worker")
    sub = parser.add_subparsers(dest="command", required=True)

    chk = sub.add_parser("check")
    chk.add_argument(
        "--full",
        action="store_true",
        help="mit Import von torch/chatterbox (langsam, prüft wirklich alles)",
    )

    prep = sub.add_parser("prepare")
    prep.add_argument("--language", default="de")
    prep.add_argument("--device", default="auto")

    p = sub.add_parser("synth")
    p.add_argument("--ref", required=True)
    p.add_argument("--text", default="")
    p.add_argument(
        "--text-file",
        dest="text_file",
        default="",
        help="JSON-Liste mehrerer Sätze – EIN Modellladen für alle",
    )
    p.add_argument("--out", required=True)
    p.add_argument("--language", default="de")
    p.add_argument("--exaggeration", default=0.5)
    p.add_argument("--cfg", default=0.5)
    p.add_argument("--temperature", default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    try:
        if args.command == "check":
            return cmd_check(args)
        if args.command == "prepare":
            return cmd_prepare(args)
        return cmd_synth(args)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    sys.exit(main())
