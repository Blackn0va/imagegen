"""Bilder vergrößern – mit Modell (Real-ESRGAN) oder ohne (Lanczos).

Zwei Wege, weil beide gebraucht werden:

  * **Real-ESRGAN** rekonstruiert Kanten und Struktur. Die Gewichte liegen
    im Modell ``realesrgan-x4`` (BSD-3-Clause, kommerziell frei). Die
    Netzarchitektur (RRDBNet) ist hier in reinem torch nachgebaut – so
    kommt kein weiteres Paket dazu, das eine eigene Lizenz und eine eigene
    torch-Fassung mitbringen würde.
  * **Lanczos** über Pillow. Kein Modell, keine GPU, sofort verfügbar.
    Weicher als Real-ESRGAN, aber nie ein Grund zu warten.

Große Bilder werden gekachelt gerechnet. Ohne Kacheln braucht ein
4000x4000er Bild bei Faktor 4 mehr Grafikspeicher, als eine übliche Karte
hat – und ein Absturz mitten im Auftrag ist das schlechteste Ergebnis.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .accel import clean_error

log = logging.getLogger(__name__)

UPSCALE_FACTORS: tuple[int, ...] = (2, 4, 8)
DEFAULT_TILE = 512
TILE_OVERLAP = 16
MIN_TILE = 96

ProgressCallback = Callable[[float, str], None]
StopCallback = Callable[[], bool]


class UpscaleError(RuntimeError):
    """Vergrößern nicht möglich – mit Klartext-Begründung."""


class UpscaleUnavailable(UpscaleError):
    """Das Modell ist nicht benutzbar (fehlt, passt nicht, fremdes Format).

    Getrennt von ``UpscaleError``, weil hier auf Lanczos ausgewichen wird –
    ein weiches Ergebnis ist besser als ein abgebrochener Auftrag. Echte
    Fehler (Speichermangel) bleiben ``UpscaleError`` und brechen ab.
    """


class UpscaleCancelled(RuntimeError):
    """Abbruch durch den Bediener. Darf nicht als Fehler gemeldet werden."""


# ---------------------------------------------------------------------------
# Pillow
# ---------------------------------------------------------------------------
def pillow_available() -> tuple[bool, str]:
    import importlib.util

    if importlib.util.find_spec("PIL") is None:
        return False, "Pillow ist nicht installiert – Bildbearbeitung nicht möglich."
    return True, ""


def open_image(path: Path) -> Any:
    """Bild laden. Wirft ``UpscaleError`` mit lesbarem Grund."""
    ok, reason = pillow_available()
    if not ok:
        raise UpscaleError(reason)
    from PIL import Image, UnidentifiedImageError

    target = Path(path)
    if not target.is_file():
        raise UpscaleError(f"Datei nicht gefunden: {target}")
    try:
        image = Image.open(target)
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise UpscaleError(f"{target.name} ist kein lesbares Bild: {clean_error(exc)}") from exc
    return image


def lanczos_resize(image: Any, factor: float = 2.0, target: tuple[int, int] | None = None) -> Any:
    """Rein rechnerisch vergrößern. Immer verfügbar, nie schnell falsch."""
    from PIL import Image

    if target is None:
        target = (
            max(1, round(image.width * factor)),
            max(1, round(image.height * factor)),
        )
    return image.resize(target, Image.LANCZOS)


def fit_to_max_side(image: Any, max_side: int) -> tuple[Any, bool]:
    """Auf eine Höchstkante begrenzen. Gibt (Bild, wurde verkleinert) zurück."""
    if max_side <= 0:
        return image, False
    longest = max(image.width, image.height)
    if longest <= max_side:
        return image, False
    ratio = max_side / float(longest)
    size = (max(1, int(image.width * ratio)), max(1, int(image.height * ratio)))
    return lanczos_resize(image, target=size), True


def snap_to_multiple(image: Any, multiple: int = 8) -> tuple[Any, bool]:
    """Kantenlängen auf ein Vielfaches bringen – Diffusionsmodelle brauchen das."""
    width = max(multiple, (image.width // multiple) * multiple)
    height = max(multiple, (image.height // multiple) * multiple)
    if (width, height) == (image.width, image.height):
        return image, False
    return lanczos_resize(image, target=(width, height)), True


# ---------------------------------------------------------------------------
# Real-ESRGAN: Netz
# ---------------------------------------------------------------------------
def _build_modules():
    """RRDBNet aufbauen. Erst beim Aufruf, weil torch teuer zu importieren ist."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        """Fünf Faltungen, jede sieht alle vorherigen Ausgaben."""

        def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            # 0.2 ist die Restgewichtung aus der Veröffentlichung; ohne sie
            # driftet der Wertebereich über die Tiefe des Netzes weg.
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat: int, num_grow_ch: int = 32) -> None:
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        """Aufbau und Namen exakt wie in den veröffentlichten Gewichten."""

        def __init__(
            self,
            num_in_ch: int = 3,
            num_out_ch: int = 3,
            scale: int = 4,
            num_feat: int = 64,
            num_block: int = 23,
            num_grow_ch: int = 32,
        ) -> None:
            super().__init__()
            self.scale = scale
            if scale == 2:
                num_in_ch *= 4
            elif scale == 1:
                num_in_ch *= 16
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if scale == 8:
                self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            if self.scale == 2:
                feat = F.pixel_unshuffle(x, downscale_factor=2)
            elif self.scale == 1:
                feat = F.pixel_unshuffle(x, downscale_factor=4)
            else:
                feat = x
            feat = self.conv_first(feat)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            if self.scale == 8:
                feat = self.lrelu(
                    self.conv_up3(F.interpolate(feat, scale_factor=2, mode="nearest"))
                )
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    return RRDBNet


def _state_dict_from(payload: Any) -> dict:
    """Gewichte aus der .pth holen – die Ablage unterscheidet sich je Quelle."""
    if isinstance(payload, dict):
        for key in ("params_ema", "params", "state_dict", "model"):
            inner = payload.get(key)
            if isinstance(inner, dict) and inner:
                return inner
        return payload
    raise UpscaleUnavailable("Unbekanntes Format der Gewichtsdatei.")


def _shape_of(state: dict, name: str):
    tensor = state.get(name)
    return tuple(tensor.shape) if tensor is not None else None


def describe_weights(state: dict) -> tuple[int, int, int, int]:
    """Aus den Gewichten ableiten, wie das Netz gebaut werden muss.

    Rückgabe: (scale, num_feat, num_block, num_grow_ch). Die Datei ist die
    Wahrheit – geraten wird nichts, sonst passt am Ende kein Tensor.
    """
    first = _shape_of(state, "conv_first.weight")
    if first is None:
        raise UpscaleUnavailable(
            "Gewichtsdatei enthält kein 'conv_first' – kein Real-ESRGAN-Modell."
        )
    num_feat, in_ch = int(first[0]), int(first[1])
    if in_ch == 12:
        scale = 2
    elif in_ch == 48:
        scale = 1
    else:
        scale = 8 if "conv_up3.weight" in state else 4

    blocks = 0
    for key in state:
        if key.startswith("body."):
            part = key.split(".", 2)[1]
            if part.isdigit():
                blocks = max(blocks, int(part) + 1)
    if blocks == 0:
        raise UpscaleUnavailable("Gewichtsdatei enthält keine RRDB-Blöcke.")

    grow = _shape_of(state, "body.0.rdb1.conv1.weight")
    num_grow_ch = int(grow[0]) if grow else 32
    return scale, num_feat, blocks, num_grow_ch


_net_cache: dict[tuple, Any] = {}


def _load_net(weights: Path, device: str, half: bool):
    """Netz einmal je (Datei, Gerät, Genauigkeit) laden und behalten."""
    key = (str(weights), device, half)
    cached = _net_cache.get(key)
    if cached is not None:
        return cached

    import torch

    try:
        payload = torch.load(str(weights), map_location="cpu", weights_only=True)
    except TypeError:
        # torch < 2.6 kennt weights_only noch nicht als Vorgabe-Parameter.
        payload = torch.load(str(weights), map_location="cpu")
    except Exception as exc:
        raise UpscaleUnavailable(
            f"Gewichte {weights.name} nicht ladbar: {clean_error(exc)}"
        ) from exc

    state = _state_dict_from(payload)
    scale, num_feat, num_block, num_grow_ch = describe_weights(state)
    net_class = _build_modules()
    net = net_class(scale=scale, num_feat=num_feat, num_block=num_block, num_grow_ch=num_grow_ch)
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise UpscaleUnavailable(
            f"Gewichte {weights.name} passen nicht zum Netz: {clean_error(exc)}"
        ) from exc
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    net = net.to(device)
    if half:
        net = net.half()
    _net_cache.clear()  # immer nur ein Netz im Speicher
    _net_cache[key] = (net, scale)
    return _net_cache[key]


def unload() -> None:
    """Netz freigeben (nach dem Auftrag, wenn nichts im Speicher bleiben soll)."""
    _net_cache.clear()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Gewichte auf der Platte finden
# ---------------------------------------------------------------------------
def weights_for(directory: Path, factor: int) -> Path | None:
    """Passende .pth im Modellordner suchen: erst exakt, dann irgendeine."""
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates = sorted(directory.rglob("*.pth"))
    if not candidates:
        return None
    exact = [p for p in candidates if f"x{factor}" in p.name.lower()]
    if exact:
        return exact[0]

    # Kein Netz für genau diesen Faktor – das nächstgrößere nehmen und
    # anschließend herunterrechnen ist immer noch besser als Lanczos.
    def rank(path: Path) -> tuple[int, str]:
        for known in (8, 4, 2):
            if f"x{known}" in path.name.lower():
                return (abs(known - factor), path.name)
        return (99, path.name)

    return sorted(candidates, key=rank)[0]


def model_ready(directory: Path) -> tuple[bool, str]:
    """Liegen brauchbare Gewichte vor?"""
    found = weights_for(Path(directory), 4)
    if found is None:
        return False, "Keine Real-ESRGAN-Gewichte im Modellordner."
    return True, f"Gewichte vorhanden: {found.name}"


# ---------------------------------------------------------------------------
# Kachelweise Berechnung
# ---------------------------------------------------------------------------
def _to_tensor(image: Any, device: str, half: bool):
    import numpy as np
    import torch

    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)
    return tensor.half() if half else tensor


def _to_image(tensor: Any) -> Any:
    import numpy as np
    from PIL import Image

    array = tensor.squeeze(0).float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8), mode="RGB")


def _run_tiled(
    net,
    tensor,
    scale: int,
    tile: int,
    on_progress: ProgressCallback | None,
    should_stop: StopCallback | None,
):
    """Bild kachelweise durch das Netz schicken und wieder zusammensetzen.

    Jede Kachel wird mit Überlappung gerechnet und danach auf ihren
    eigentlichen Bereich beschnitten – ohne diesen Rand entstehen an den
    Kachelgrenzen sichtbare Nähte.
    """
    import torch

    _batch, channels, height, width = tensor.shape
    if tile <= 0 or (height <= tile and width <= tile):
        if should_stop is not None and should_stop():
            raise UpscaleCancelled("Vergrößern abgebrochen")
        with torch.inference_mode():
            return net(tensor)

    output = torch.zeros(
        (1, channels, height * scale, width * scale),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    columns = math.ceil(width / tile)
    rows = math.ceil(height / tile)
    total = columns * rows
    done = 0

    for row in range(rows):
        for column in range(columns):
            if should_stop is not None and should_stop():
                # NICHT KeyboardInterrupt: das ist eine BaseException und
                # läuft durch alle except-Blöcke bis aus dem Arbeiter-
                # Thread der Warteschlange heraus. Der Thread stirbt, wird
                # nirgends neu gestartet, und danach bleibt jeder weitere
                # Auftrag für immer auf "wartend" -- die Anwendung wirkt
                # eingefroren, ohne dass ein Fehler erscheint.
                raise UpscaleCancelled("Vergrößern abgebrochen.")
            x0, y0 = column * tile, row * tile
            x1, y1 = min(x0 + tile, width), min(y0 + tile, height)
            # Rand mit Überlappung, aber nie über das Bild hinaus
            px0, py0 = max(0, x0 - TILE_OVERLAP), max(0, y0 - TILE_OVERLAP)
            px1, py1 = min(width, x1 + TILE_OVERLAP), min(height, y1 + TILE_OVERLAP)

            with torch.inference_mode():
                patch = net(tensor[:, :, py0:py1, px0:px1])

            # Im Ergebnis den Überlappungsrand wieder abschneiden
            cut_left = (x0 - px0) * scale
            cut_top = (y0 - py0) * scale
            cut_right = cut_left + (x1 - x0) * scale
            cut_bottom = cut_top + (y1 - y0) * scale
            output[:, :, y0 * scale : y1 * scale, x0 * scale : x1 * scale] = patch[
                :, :, cut_top:cut_bottom, cut_left:cut_right
            ]
            del patch

            done += 1
            if on_progress is not None:
                on_progress(done / total, f"Kachel {done}/{total}")
    return output


def _net_pass(
    image: Any,
    weights: Path,
    device: str,
    half: bool,
    tile: int,
    on_progress: ProgressCallback | None,
    should_stop: StopCallback | None,
) -> tuple[Any, int]:
    """Einen Durchlauf durch das Netz. Bei Speichermangel kleinere Kacheln."""
    import torch

    net, scale = _load_net(weights, device, half)
    tensor = _to_tensor(image, device, half)
    current = max(MIN_TILE, tile) if tile > 0 else 0

    while True:
        try:
            result = _run_tiled(net, tensor, scale, current, on_progress, should_stop)
            return _to_image(result), scale
        except torch.cuda.OutOfMemoryError:  # type: ignore[attr-defined]
            torch.cuda.empty_cache()
            if current == 0 or current > MIN_TILE:
                current = MIN_TILE if current == 0 else max(MIN_TILE, current // 2)
                log.warning("Grafikspeicher knapp – Kachelgröße auf %d gesetzt.", current)
                if on_progress is not None:
                    on_progress(0.0, f"Grafikspeicher knapp – Kacheln auf {current} px")
                continue
            raise UpscaleError(
                "Der Grafikspeicher reicht selbst mit kleinsten Kacheln nicht aus. "
                "Kleineren Faktor wählen oder in den Einstellungen auf CPU umstellen."
            ) from None


def upscale_image(
    image: Any,
    factor: int = 2,
    weights: Path | None = None,
    device: str = "cpu",
    half: bool = False,
    tile: int = DEFAULT_TILE,
    on_progress: ProgressCallback | None = None,
    should_stop: StopCallback | None = None,
) -> tuple[Any, str]:
    """Bild vergrößern. Rückgabe: (Bild, benutztes Verfahren im Klartext).

    Ohne Gewichte oder ohne torch wird Lanczos benutzt – das ist kein
    Fehler, sondern die Rückfallebene, und steht so auch im Ergebnis.
    """
    ok, reason = pillow_available()
    if not ok:
        raise UpscaleError(reason)

    factor = int(factor)
    if factor <= 1:
        return image, "unverändert"

    target = (image.width * factor, image.height * factor)

    if weights is None or not Path(weights).is_file():
        return lanczos_resize(image, target=target), "Lanczos (kein Modell geladen)"

    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return lanczos_resize(image, target=target), "Lanczos (torch fehlt)"

    # Transparenz getrennt behandeln: das Netz rechnet nur RGB.
    alpha = None
    if image.mode in ("RGBA", "LA") or "transparency" in getattr(image, "info", {}):
        converted = image.convert("RGBA")
        alpha = converted.getchannel("A")
        image = converted.convert("RGB")

    def fall_back(exc: BaseException) -> tuple[Any, str]:
        log.warning("Real-ESRGAN fehlgeschlagen: %s", clean_error(exc))
        result = lanczos_resize(image, target=target)
        if alpha is not None:
            result = _attach_alpha(result, alpha)
        return result, f"Lanczos (Modell scheiterte: {clean_error(exc, 120)})"

    try:
        result, net_scale = _net_pass(
            image, Path(weights), device, half, tile, on_progress, should_stop
        )
    except UpscaleUnavailable as exc:
        # Gewichte fehlen oder passen nicht – ein weiches Bild ist besser
        # als ein abgebrochener Auftrag.
        return fall_back(exc)
    except (UpscaleError, UpscaleCancelled):
        # Speichermangel und Abbruch gehören nach oben, nicht in die Rückfallebene.
        raise
    except Exception as exc:
        return fall_back(exc)

    method = f"Real-ESRGAN x{net_scale}"
    if (result.width, result.height) != target:
        # Netzfaktor und Wunschfaktor können auseinanderliegen (nur x4 im
        # Ordner, aber x2 gewünscht) – der Rest ist eine saubere Skalierung.
        result = lanczos_resize(result, target=target)
        method += f" + Anpassung auf x{factor}"
    if alpha is not None:
        result = _attach_alpha(result, alpha)
    return result, method


def _attach_alpha(image: Any, alpha: Any) -> Any:
    """Alphakanal auf die neue Größe bringen und wieder anhängen."""
    from PIL import Image

    resized = alpha.resize((image.width, image.height), Image.LANCZOS)
    merged = image.convert("RGBA")
    merged.putalpha(resized)
    return merged


def describe() -> str:
    """Kurzer Zustandsbericht für Diagnose und Oberfläche."""
    ok, reason = pillow_available()
    if not ok:
        return reason
    import importlib.util

    lines = ["Lanczos (Pillow): verfügbar"]
    if importlib.util.find_spec("torch") is None:
        lines.append("Real-ESRGAN: torch fehlt – nur Lanczos möglich.")
    else:
        lines.append("Real-ESRGAN: torch vorhanden, Gewichte werden beim Auftrag geprüft.")
    return "\n".join(lines)
