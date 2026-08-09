"""Bild zu Diamond-Painting-Vorlage.

Aus einem beliebigen Bild wird ein Raster aus Steinen: jede Zelle ist ein
Stein einer Farbe, jede Farbe bekommt ein eigenes Symbol. Heraus kommen
zwei Dateien – die Vorlage zum Ausdrucken und eine Farbliste mit den
Stückzahlen, nach der die Steine sortiert und nachbestellt werden.

Bewusst ohne Modell und ohne torch: das ist reine Bildrechnung und muss
auch auf einem Rechner ohne Grafikkarte in Sekunden fertig sein.

Zwei Entscheidungen, die den Unterschied zwischen brauchbar und
unbrauchbar ausmachen:

  * **Kein Dithering.** Streuung sieht auf dem Bildschirm besser aus,
    erzeugt aber einzelne Fremdsteine mitten in einer Fläche. Wer klebt,
    hasst das – und in der Farbliste steht dann eine Farbe mit sieben
    Steinen.
  * **Symbole statt Farben allein.** Ausgedruckt liegen zwei Blautöne
    dicht beieinander. Das Symbol entscheidet, nicht das Auge.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Steine sind quadratisch oder rund; beides gibt es zu kaufen.
SHAPES: tuple[str, ...] = ("round", "square")
SHAPE_LABELS = {"round": "rund", "square": "eckig"}

# Übliche Kantenlängen in Millimetern. 2,8 mm ist die verbreitete
# Rundstein-Größe, 2,5 mm der eckige Standard.
STONE_SIZES_MM = {"round": 2.8, "square": 2.5}

# Grenzen. Oben, damit eine vertippte Zahl nicht ein Bild mit 40000 Pixeln
# Kantenlänge erzeugt; unten, damit überhaupt etwas zu erkennen ist.
MIN_STONES, MAX_STONES = 20, 400
MIN_COLORS, MAX_COLORS = 2, 48
MIN_CELL_PX, MAX_CELL_PX = 8, 48

# Mindestabstand zweier Farben in der Vorlage.
#
# Ohne diese Schwelle verteilt die Farbreduktion mehrere Plätze auf
# denselben Farbton: aus einem Himmel werden #96C3E6, #96C3E5 und #96C3E7.
# Auf dem Schirm ist das egal, in der Farbliste steht dann aber dreimal
# dasselbe Blau – man kauft drei Tütchen und kann die Symbole hinterher
# nicht zuordnen. Gemessen wird mit der üblichen billigen Gewichtung
# (Grün zählt am meisten, Rot mehr als Blau), 24 entspricht rund einem
# gerade noch erkennbaren Unterschied auf Papier.
MIN_COLOR_DISTANCE = 24.0

# Symbole für die Legende. Ohne die Paare, die ausgedruckt niemand
# auseinanderhält: I und 1, O und 0, S und 5.
SYMBOLS = "ABCDEFGHJKLMNPQRTUVWXYZ2346789#@%&*+=<>?$§"

# Alle zehn Steine eine kräftigere Linie – ohne die Hilfslinien verzählt
# man sich auf einer Vorlage mit 200 Spalten unweigerlich.
GRID_EVERY = 10

ProgressCallback = Callable[[float, str], None]


class DiamondError(RuntimeError):
    """Vorlage nicht erstellbar – mit Klartext-Begründung."""


@dataclass(frozen=True)
class StoneColor:
    """Eine Farbe der Vorlage samt Symbol und Stückzahl."""

    index: int
    symbol: str
    rgb: tuple[int, int, int]
    count: int
    # Leer, wenn ohne DMC-Abgleich gerechnet wurde. Dann steht in der
    # Farbliste nur der Hexwert – bestellbar ist der nicht.
    dmc_code: str = ""
    dmc_name: str = ""

    @property
    def hex_code(self) -> str:
        return "#{:02X}{:02X}{:02X}".format(*self.rgb)

    def order_label(self) -> str:
        """Bezeichnung zum Bestellen. Ohne DMC bleibt nur der Hexwert."""
        if not self.dmc_code:
            return self.hex_code
        return f"DMC {self.dmc_code}"

    def full_label(self) -> str:
        """Lange Fassung für die Farbtafel."""
        if not self.dmc_code:
            return self.hex_code
        return f"DMC {self.dmc_code} {self.dmc_name}"

    def is_dark(self) -> bool:
        """Ist die Farbe dunkel genug für ein helles Symbol darauf?"""
        red, green, blue = self.rgb
        # Helligkeit nach ITU-R BT.601 – dieselbe Gewichtung, mit der auch
        # das Einfärben in pipeline_image rechnet.
        return (0.299 * red + 0.587 * green + 0.114 * blue) < 140


@dataclass(frozen=True)
class DiamondPlan:
    """Fertige Vorlage: Raster, Farben und die Maße in Millimetern."""

    columns: int
    rows: int
    shape: str
    colors: tuple[StoneColor, ...]
    grid: tuple[tuple[int, ...], ...]  # je Zeile die Farbindizes

    @property
    def total_stones(self) -> int:
        return self.columns * self.rows

    def size_mm(self) -> tuple[float, float]:
        edge = STONE_SIZES_MM.get(self.shape, 2.8)
        return (self.columns * edge, self.rows * edge)

    def size_cm_text(self) -> str:
        width_mm, height_mm = self.size_mm()
        return f"{width_mm / 10:.1f} x {height_mm / 10:.1f} cm"

    def uses_dmc(self) -> bool:
        """Stehen bestellbare DMC-Nummern in der Vorlage?"""
        return any(color.dmc_code for color in self.colors)


# ---------------------------------------------------------------------------
# Raster und Farben
# ---------------------------------------------------------------------------
def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def target_grid(width: int, height: int, stones: int) -> tuple[int, int]:
    """Rastergröße aus der gewünschten Breite in Steinen ableiten.

    Das Seitenverhältnis bleibt erhalten – eine Vorlage, die das Motiv
    verzerrt, ist wertlos.
    """
    columns = _clamp(stones, MIN_STONES, MAX_STONES)
    if width <= 0 or height <= 0:
        raise DiamondError("Das Bild hat keine gültige Größe.")
    rows = max(MIN_STONES, round(columns * height / width))
    return columns, min(rows, MAX_STONES * 2)


def color_distance(first: tuple[int, int, int], second: tuple[int, int, int]) -> float:
    """Abstand zweier Farben, grob am Sehen ausgerichtet.

    Die Gewichte bilden nach, dass das Auge Grün am feinsten auflöst und
    Blau am gröbsten. Das reicht hier vollkommen – es geht nur um die
    Frage „sind das auf Papier zwei Farben oder eine".
    """
    delta_r = first[0] - second[0]
    delta_g = first[1] - second[1]
    delta_b = first[2] - second[2]
    return (2 * delta_r * delta_r + 4 * delta_g * delta_g + 3 * delta_b * delta_b) ** 0.5


def build_plan(
    image: Any,
    stones: int = 100,
    colors: int = 24,
    shape: str = "round",
    min_distance: float = MIN_COLOR_DISTANCE,
    use_dmc: bool = True,
    stones_only: bool = True,
) -> DiamondPlan:
    """Bild auf das Raster bringen und die Farben zusammenfassen.

    Mit ``use_dmc`` (Vorgabe) zeigt die Vorlage nicht die Bildfarben,
    sondern die nächstgelegenen **bestellbaren** DMC-Farben. Das ist der
    Punkt, an dem eine Vorlage brauchbar wird: nach Hexwert verkauft
    niemand Steine. ``stones_only`` beschränkt zusätzlich auf die
    Nummern, die es beim Diamond Painting wirklich als Stein gibt – ohne
    das stünden Garnfarben in der Liste, die man nicht kleben kann.

    Es können weniger Farben herauskommen als angefordert: zu ähnliche
    Töne werden zusammengelegt (siehe ``MIN_COLOR_DISTANCE``), und zwei
    Bildfarben können auf dieselbe DMC-Nummer fallen. Zehn wirklich
    unterscheidbare Farben sind eine brauchbare Vorlage, vierundzwanzig
    fast gleiche sind es nicht.
    """
    from PIL import Image

    from . import dmc

    if shape not in SHAPES:
        shape = "round"
    wanted_colors = _clamp(colors, MIN_COLORS, min(MAX_COLORS, len(SYMBOLS)))

    columns, rows = target_grid(image.width, image.height, stones)
    # Flächenmittel statt Lanczos: ein Stein soll den Durchschnitt seines
    # Bildbereichs zeigen. Lanczos schwingt an Kanten über und erzeugt
    # dabei Säume aus Farben, die im Motiv gar nicht vorkommen – die
    # belegen hinterher Plätze in der Farbliste.
    small = image.convert("RGB").resize((columns, rows), Image.BOX)
    reduced = small.quantize(
        colors=wanted_colors,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,  # siehe Modulkopf: keine Streuung
    )

    palette = reduced.getpalette() or []
    indices = list(reduced.getdata())
    counts: dict[int, int] = {}
    for value in indices:
        counts[value] = counts.get(value, 0) + 1

    def rgb_of(index: int) -> tuple[int, int, int]:
        base = index * 3
        return (
            int(palette[base]) if base < len(palette) else 0,
            int(palette[base + 1]) if base + 1 < len(palette) else 0,
            int(palette[base + 2]) if base + 2 < len(palette) else 0,
        )

    # Häufigste Farbe zuerst: dann trägt Symbol "A" die größte Fläche, die
    # Farbliste liest sich in der Reihenfolge, in der geklebt wird – und
    # beim Zusammenlegen gewinnt immer der größere Farbanteil.
    order = sorted(counts, key=lambda idx: (-counts[idx], idx))

    # Farbe je Palettenplatz festlegen. Mit DMC-Abgleich wird nicht der
    # Bildwert angezeigt, sondern die nächstgelegene bestellbare Farbe –
    # denn genau die klebt am Ende auf der Leinwand. Zwei Bildfarben, die
    # auf dieselbe Nummer fallen, haben danach denselben RGB-Wert und
    # verschmelzen im Schritt darunter von selbst.
    chosen: dict[int, tuple[int, int, int]] = {}
    matches: dict[int, Any] = {}
    for old_index in counts:
        raw = rgb_of(old_index)
        if not use_dmc:
            chosen[old_index] = raw
            matches[old_index] = None
            continue
        match = dmc.nearest(raw, stones_only=stones_only, distance=color_distance)
        chosen[old_index] = match.rgb
        matches[old_index] = match

    kept: list[int] = []
    absorbed: dict[int, int] = {}
    for old_index in order:
        current = chosen[old_index]
        nearest = None
        for candidate in kept:
            if color_distance(current, chosen[candidate]) < min_distance:
                nearest = candidate
                break
        if nearest is None:
            kept.append(old_index)
            absorbed[old_index] = old_index
        else:
            absorbed[old_index] = nearest

    merged_counts: dict[int, int] = {}
    for old_index, count in counts.items():
        target = absorbed[old_index]
        merged_counts[target] = merged_counts.get(target, 0) + count
    kept.sort(key=lambda idx: (-merged_counts[idx], idx))
    remap = {old: new for new, old in enumerate(kept)}

    stone_colors = tuple(
        StoneColor(
            index=new_index,
            symbol=SYMBOLS[new_index % len(SYMBOLS)],
            rgb=chosen[old_index],
            count=merged_counts[old_index],
            dmc_code=matches[old_index].code if matches[old_index] else "",
            dmc_name=matches[old_index].name if matches[old_index] else "",
        )
        for new_index, old_index in enumerate(kept)
    )

    grid = tuple(
        tuple(remap[absorbed[indices[row * columns + column]]] for column in range(columns))
        for row in range(rows)
    )
    return DiamondPlan(columns=columns, rows=rows, shape=shape, colors=stone_colors, grid=grid)


# ---------------------------------------------------------------------------
# Vorlage zeichnen
# ---------------------------------------------------------------------------
def _font(size: int):
    """Schrift für die Symbole. Fällt auf die eingebaute zurück."""
    from PIL import ImageFont

    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Pillow vor 10.1 kennt die Größe bei der eingebauten Schrift nicht.
        return ImageFont.load_default()


def render_chart(
    plan: DiamondPlan,
    cell_px: int = 18,
    symbols: bool = True,
    on_progress: ProgressCallback | None = None,
) -> Any:
    """Vorlage als Bild zeichnen: ein Kästchen je Stein, dazu ein Raster.

    Der Rand trägt die Spalten- und Zeilennummern. Ohne die findet man auf
    einer ausgedruckten Vorlage die Stelle nicht wieder, an der man
    aufgehört hat.
    """
    from PIL import Image, ImageDraw

    cell = _clamp(cell_px, MIN_CELL_PX, MAX_CELL_PX)
    margin = max(24, cell + 6)
    width = margin + plan.columns * cell + 2
    height = margin + plan.rows * cell + 2

    chart = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(chart)
    symbol_font = _font(max(7, int(cell * 0.62)))
    label_font = _font(max(7, int(min(cell, 14) * 0.72)))

    by_index = {color.index: color for color in plan.colors}

    for row in range(plan.rows):
        top = margin + row * cell
        for column in range(plan.columns):
            left = margin + column * cell
            color = by_index[plan.grid[row][column]]
            box = (left + 1, top + 1, left + cell - 1, top + cell - 1)
            if plan.shape == "round":
                draw.ellipse(box, fill=color.rgb)
            else:
                draw.rectangle(box, fill=color.rgb)
            if symbols:
                draw.text(
                    (left + cell / 2, top + cell / 2),
                    color.symbol,
                    font=symbol_font,
                    fill=(255, 255, 255) if color.is_dark() else (20, 20, 20),
                    anchor="mm",
                )
        if on_progress is not None and plan.rows:
            on_progress((row + 1) / plan.rows, f"Zeile {row + 1}/{plan.rows}")

    # Hilfslinien und Nummern zuletzt, damit sie über den Steinen liegen.
    faint, strong = (205, 205, 205), (60, 60, 60)
    for column in range(plan.columns + 1):
        x = margin + column * cell
        heavy = column % GRID_EVERY == 0
        draw.line([(x, margin), (x, margin + plan.rows * cell)], fill=strong if heavy else faint)
        if heavy and column < plan.columns:
            draw.text(
                (x + cell / 2, margin - 6),
                str(column + 1),
                font=label_font,
                fill=strong,
                anchor="mb",
            )
    for row in range(plan.rows + 1):
        y = margin + row * cell
        heavy = row % GRID_EVERY == 0
        draw.line([(margin, y), (margin + plan.columns * cell, y)], fill=strong if heavy else faint)
        if heavy and row < plan.rows:
            draw.text(
                (margin - 6, y + cell / 2), str(row + 1), font=label_font, fill=strong, anchor="rm"
            )

    return chart


def render_legend(plan: DiamondPlan, cell_px: int = 18) -> Any:
    """Farbtafel als eigenes Bild: Symbol, Farbfeld, Stückzahl."""
    from PIL import Image, ImageDraw

    cell = _clamp(cell_px, MIN_CELL_PX, MAX_CELL_PX)
    line_height = max(22, cell + 6)
    # Mit DMC-Namen wird die Zeile deutlich länger als mit einem Hexwert.
    width = 620 if plan.uses_dmc() else 420
    height = line_height * (len(plan.colors) + 3)

    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    font = _font(max(9, int(line_height * 0.5)))
    symbol_font = _font(max(8, int(cell * 0.62)))

    draw.text((12, 10), "Farbliste", font=_font(max(11, int(line_height * 0.6))), fill=(20, 20, 20))
    draw.text(
        (12, 10 + line_height),
        f"{plan.total_stones} Steine, {len(plan.colors)} Farben, {plan.size_cm_text()}",
        font=font,
        fill=(70, 70, 70),
    )

    for position, color in enumerate(plan.colors):
        top = 10 + line_height * (position + 2)
        box = (12, top, 12 + cell, top + cell)
        if plan.shape == "round":
            draw.ellipse(box, fill=color.rgb, outline=(150, 150, 150))
        else:
            draw.rectangle(box, fill=color.rgb, outline=(150, 150, 150))
        draw.text(
            (12 + cell / 2, top + cell / 2),
            color.symbol,
            font=symbol_font,
            fill=(255, 255, 255) if color.is_dark() else (20, 20, 20),
            anchor="mm",
        )
        share = 100.0 * color.count / max(1, plan.total_stones)
        draw.text(
            (24 + cell, top + cell / 2),
            f"{color.symbol}   {color.full_label()}   {color.count} Steine   {share:.1f} %",
            font=font,
            fill=(20, 20, 20),
            anchor="lm",
        )
    return sheet


def legend_text(plan: DiamondPlan, source: Path | None = None) -> str:
    """Farbliste als Textdatei – zum Nachbestellen und Sortieren."""
    width_mm, height_mm = plan.size_mm()
    with_dmc = plan.uses_dmc()
    rule = "-" * (62 if with_dmc else 40)
    lines = [
        "Diamond-Painting-Vorlage",
        "=" * len(rule),
    ]
    if source is not None:
        lines.append(f"Vorlage:      {source.name}")
    lines.extend(
        [
            f"Raster:       {plan.columns} x {plan.rows} Steine",
            f"Steine gesamt:{plan.total_stones:>8}",
            f"Steinform:    {SHAPE_LABELS.get(plan.shape, plan.shape)} "
            f"({STONE_SIZES_MM.get(plan.shape, 2.8):.1f} mm)",
            f"Fertige Größe:{width_mm / 10:>7.1f} x {height_mm / 10:.1f} cm",
            f"Farben:       {len(plan.colors)}",
            f"Farbsystem:   {'DMC' if with_dmc else 'Bildfarben (nicht bestellbar)'}",
            "",
        ]
    )
    if with_dmc:
        lines.append(f"{'Sym':<4} {'DMC':<6} {'Name':<26} {'Hex':<8} {'Steine':>7} {'Anteil':>7}")
        lines.append(rule)
        for color in plan.colors:
            share = 100.0 * color.count / max(1, plan.total_stones)
            lines.append(
                f"{color.symbol:<4} {color.dmc_code:<6} {color.dmc_name[:26]:<26} "
                f"{color.hex_code:<8} {color.count:>7} {share:>6.1f} %"
            )
    else:
        lines.append("Sym  Farbe     Steine   Anteil")
        lines.append(rule)
        for color in plan.colors:
            share = 100.0 * color.count / max(1, plan.total_stones)
            lines.append(f"{color.symbol:<4} {color.hex_code}  {color.count:>7}   {share:>5.1f} %")

    lines.append(rule)
    if with_dmc:
        lines.extend(
            [
                "Bestellung: nach DMC-Nummer, nicht nach Hexwert. Die Nummern",
                "stammen aus der beim Diamond Painting üblichen Farbliste.",
                "",
                "Die RGB-Werte sind Näherungen – ein Harzstein hat kein",
                "definiertes sRGB. Vor einer großen Bestellung die Nummer mit",
                "der Farbkarte des Anbieters abgleichen.",
            ]
        )
    else:
        lines.extend(
            [
                "Hinweis: Die Farbwerte stammen aus dem Bild, nicht aus einer",
                "Herstellerpalette – so ist nichts bestellbar. Für bestellbare",
                "Nummern die Vorlage mit DMC-Abgleich erzeugen.",
            ]
        )
    return "\n".join(lines)


def describe() -> str:
    """Kurzer Zustandsbericht für Diagnose und Oberfläche."""
    import importlib.util

    from . import dmc

    if importlib.util.find_spec("PIL") is None:
        return "Pillow fehlt – Diamond-Painting-Vorlagen nicht möglich."
    return (
        f"Verfügbar. Raster {MIN_STONES}-{MAX_STONES} Steine breit, "
        f"{MIN_COLORS}-{min(MAX_COLORS, len(SYMBOLS))} Farben, "
        f"Formen: {', '.join(SHAPE_LABELS[s] for s in SHAPES)}. "
        f"DMC-Abgleich gegen {len(dmc.STONE_COLORS)} bestellbare Steinfarben."
    )


def available_symbols() -> Sequence[str]:
    return tuple(SYMBOLS)
