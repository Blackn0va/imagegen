"""Maske malen statt in einem Bildprogramm zeichnen.

Für „Bereich ersetzen" (Inpainting) braucht die Pipeline eine Maske: weiß
wird neu gerechnet, schwarz bleibt erhalten. Bisher musste die extern
gemalt und als Datei ausgewählt werden – der einzige Schritt im ganzen
Ablauf, für den ein zweites Programm nötig war.

Zwei Auflösungen laufen parallel:

  * die **Anzeige** ist auf das Fenster verkleinert, damit auch ein
    6000-Pixel-Bild flüssig zu bemalen ist,
  * die **Maske** wird zusätzlich in voller Größe des Ausgangsbildes
    geführt und auch so gespeichert.

Beides wird bei jedem Strich gleichzeitig gezeichnet. Eine allein
verkleinerte Maske wieder hochzurechnen gäbe ausgefranste Kanten – und
die stehen am Ende genau an der Naht zwischen altem und neuem Bild.

Rückgängig arbeitet über die Striche, nicht über Kopien der Maske: eine
Kopie einer 6000x4000-Maske sind 24 MB, zehn davon wären 240 MB nur für
die Rücknahme.
"""

from __future__ import annotations

import logging
import time
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from .. import paths
from .theme import Palette
from .widgets import Card

log = logging.getLogger(__name__)

# Größte Kantenlänge der Arbeitsfläche. Darüber wird nur die Anzeige
# verkleinert – die gespeicherte Maske behält die Größe des Originals.
MAX_VIEW = 900
MIN_BRUSH, MAX_BRUSH, DEFAULT_BRUSH = 4, 400, 48

# Rot mit halber Deckkraft: die Maske muss über jedem Motiv erkennbar sein,
# das Bild darunter aber weiter beurteilbar bleiben.
OVERLAY_RGB = (255, 64, 64)
OVERLAY_ALPHA = 0.45


@dataclass
class _Stroke:
    """Ein Strich in Koordinaten des Originalbildes."""

    erase: bool
    radius: float
    points: list[tuple[float, float]] = field(default_factory=list)


class MaskEditor(tk.Toplevel):
    """Fenster zum Malen einer Maske. ``result`` hält den Pfad oder None."""

    def __init__(
        self,
        master: tk.Misc,
        palette: Palette,
        source: Path,
        existing: Path | None = None,
    ) -> None:
        super().__init__(master)
        self.palette = palette
        self.source = Path(source)
        self.result: Path | None = None
        self._strokes: list[_Stroke] = []
        self._active: _Stroke | None = None
        self._photo: Any = None

        from PIL import Image

        self._base = Image.open(self.source)
        self._base.load()
        self._base = self._base.convert("RGB")
        self.full_size = (self._base.width, self._base.height)

        ratio = min(1.0, MAX_VIEW / float(max(self.full_size)))
        self.view_size = (
            max(1, int(self.full_size[0] * ratio)),
            max(1, int(self.full_size[1] * ratio)),
        )
        self._scale = self.full_size[0] / float(self.view_size[0])
        self._view_base = self._base.resize(self.view_size, Image.LANCZOS)

        self._mask_full = Image.new("L", self.full_size, 0)
        self._mask_view = Image.new("L", self.view_size, 0)
        if existing is not None:
            self._load_existing(Path(existing))

        self.title(f"Maske malen – {self.source.name}")
        self.transient(master)  # type: ignore[arg-type]
        self.configure(background=palette.bg)
        self._build()
        self._render()

        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.bind("<Escape>", lambda _e: self._cancel())
        self.bind("<Control-z>", lambda _e: self._undo())

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------
    def _build(self) -> None:
        outer = ttk.Frame(self, padding=12)
        outer.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        card = Card(
            outer,
            self.palette,
            "Bereich malen",
            "Rot markiert, was neu gerechnet wird. Linke Maustaste malt, "
            "rechte Maustaste radiert. Mausrad ändert die Pinselgröße, "
            "Strg+Z nimmt den letzten Strich zurück.",
        )
        card.grid(row=0, column=0, sticky="nsew")
        body = card.body()

        self.canvas = tk.Canvas(
            body,
            width=self.view_size[0],
            height=self.view_size[1],
            background=self.palette.surface_alt,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, columnspan=2, sticky="nw")
        self.canvas.bind("<Button-1>", lambda e: self._start(e, erase=False))
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish)
        self.canvas.bind("<Button-3>", lambda e: self._start(e, erase=True))
        self.canvas.bind("<B3-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-3>", self._finish)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Motion>", self._hover)
        self.canvas.bind("<Leave>", lambda _e: self.canvas.delete("cursor"))

        tools = ttk.Frame(body, style="Card.TFrame")
        tools.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        ttk.Label(tools, text="Pinsel", style="Surface.TLabel").grid(row=0, column=0)
        self.brush = tk.DoubleVar(value=DEFAULT_BRUSH)
        ttk.Scale(
            tools,
            from_=MIN_BRUSH,
            to=MAX_BRUSH,
            variable=self.brush,
            orient="horizontal",
            length=220,
            command=lambda _v: self._show_brush(),
        ).grid(row=0, column=1, padx=(8, 8))
        self.brush_label = ttk.Label(tools, text="", style="SurfaceDim.TLabel", width=10)
        self.brush_label.grid(row=0, column=2)
        self._show_brush()

        ttk.Button(tools, text="Alles füllen", command=self._fill_all).grid(
            row=0, column=3, padx=(16, 4)
        )
        ttk.Button(tools, text="Leeren", command=self._clear).grid(row=0, column=4, padx=4)
        ttk.Button(tools, text="Zurück (Strg+Z)", command=self._undo).grid(row=0, column=5, padx=4)

        self.info = ttk.Label(body, text="", style="SurfaceDim.TLabel")
        self.info.grid(row=2, column=0, sticky="w", pady=(10, 0))

        actions = ttk.Frame(outer)
        actions.grid(row=1, column=0, sticky="e", pady=(12, 0))
        ttk.Button(actions, text="Abbrechen", command=self._cancel).grid(row=0, column=0, padx=6)
        ttk.Button(actions, text="Übernehmen", style="Accent.TButton", command=self._save).grid(
            row=0, column=1
        )

    # ------------------------------------------------------------------
    # Malen
    # ------------------------------------------------------------------
    def _radius_full(self) -> float:
        """Pinselgröße in Pixeln des Originals."""
        return max(1.0, float(self.brush.get()) / 2.0 * self._scale)

    def _show_brush(self) -> None:
        self.brush_label.configure(text=f"{int(self.brush.get())} px")

    def _wheel(self, event) -> None:
        step = 4 if event.delta > 0 else -4
        self.brush.set(max(MIN_BRUSH, min(MAX_BRUSH, self.brush.get() + step)))
        self._show_brush()
        self._hover(event)

    def _hover(self, event) -> None:
        """Pinselumriss unter dem Zeiger – sonst malt man auf Verdacht."""
        self.canvas.delete("cursor")
        radius = max(2.0, float(self.brush.get()) / 2.0)
        self.canvas.create_oval(
            event.x - radius,
            event.y - radius,
            event.x + radius,
            event.y + radius,
            outline="#ffffff",
            width=1,
            tags="cursor",
        )

    def _start(self, event, erase: bool) -> None:
        self._active = _Stroke(erase=erase, radius=self._radius_full())
        self._extend(event.x, event.y)

    def _drag(self, event) -> None:
        if self._active is not None:
            self._extend(event.x, event.y)

    def _finish(self, _event) -> None:
        if self._active is not None and self._active.points:
            self._strokes.append(self._active)
        self._active = None
        self._update_info()

    def _extend(self, view_x: float, view_y: float) -> None:
        """Punkt anhängen und sofort in beide Masken zeichnen."""
        if self._active is None:
            return
        full = (view_x * self._scale, view_y * self._scale)
        previous = self._active.points[-1] if self._active.points else None
        self._active.points.append(full)

        from PIL import ImageDraw

        colour = 0 if self._active.erase else 255
        radius_full = self._active.radius
        radius_view = max(1.0, radius_full / self._scale)
        draw_full = ImageDraw.Draw(self._mask_full)
        draw_view = ImageDraw.Draw(self._mask_view)

        def blob(draw, x: float, y: float, radius: float) -> None:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=colour)

        blob(draw_full, full[0], full[1], radius_full)
        blob(draw_view, view_x, view_y, radius_view)
        if previous is not None:
            # Zwischen zwei Mausmeldungen liegen Lücken – ohne die
            # Verbindungslinie entstünde eine Perlenkette statt eines Strichs.
            draw_full.line([previous, full], fill=colour, width=int(radius_full * 2))
            draw_view.line(
                [(previous[0] / self._scale, previous[1] / self._scale), (view_x, view_y)],
                fill=colour,
                width=max(1, int(radius_view * 2)),
            )
        self._render(keep_cursor=True)

    # ------------------------------------------------------------------
    # Werkzeuge
    # ------------------------------------------------------------------
    def _fill_all(self) -> None:
        from PIL import Image

        self._strokes.clear()
        self._mask_full = Image.new("L", self.full_size, 255)
        self._mask_view = Image.new("L", self.view_size, 255)
        self._render()
        self._update_info()

    def _clear(self) -> None:
        from PIL import Image

        self._strokes.clear()
        self._mask_full = Image.new("L", self.full_size, 0)
        self._mask_view = Image.new("L", self.view_size, 0)
        self._render()
        self._update_info()

    def _undo(self) -> None:
        if not self._strokes:
            return
        self._strokes.pop()
        self._rebuild()
        self._update_info()

    def _rebuild(self) -> None:
        """Masken aus den verbliebenen Strichen neu aufbauen."""
        from PIL import Image, ImageDraw

        self._mask_full = Image.new("L", self.full_size, 0)
        self._mask_view = Image.new("L", self.view_size, 0)
        draw_full = ImageDraw.Draw(self._mask_full)
        draw_view = ImageDraw.Draw(self._mask_view)
        for stroke in self._strokes:
            colour = 0 if stroke.erase else 255
            radius_view = max(1.0, stroke.radius / self._scale)
            previous: tuple[float, float] | None = None
            for point in stroke.points:
                draw_full.ellipse(
                    (
                        point[0] - stroke.radius,
                        point[1] - stroke.radius,
                        point[0] + stroke.radius,
                        point[1] + stroke.radius,
                    ),
                    fill=colour,
                )
                view_point = (point[0] / self._scale, point[1] / self._scale)
                draw_view.ellipse(
                    (
                        view_point[0] - radius_view,
                        view_point[1] - radius_view,
                        view_point[0] + radius_view,
                        view_point[1] + radius_view,
                    ),
                    fill=colour,
                )
                if previous is not None:
                    draw_full.line([previous, point], fill=colour, width=int(stroke.radius * 2))
                    draw_view.line(
                        [(previous[0] / self._scale, previous[1] / self._scale), view_point],
                        fill=colour,
                        width=max(1, int(radius_view * 2)),
                    )
                previous = point
        self._render()

    def _load_existing(self, path: Path) -> None:
        """Vorhandene Maske übernehmen, damit Nacharbeit möglich ist."""
        from PIL import Image

        try:
            with Image.open(path) as mask:
                mask.load()
                grey = mask.convert("L")
        except Exception as exc:
            log.debug("Vorhandene Maske nicht lesbar: %s", exc)
            return
        if (grey.width, grey.height) != self.full_size:
            grey = grey.resize(self.full_size, Image.NEAREST)
        self._mask_full = grey
        self._mask_view = grey.resize(self.view_size, Image.NEAREST)

    # ------------------------------------------------------------------
    # Anzeige
    # ------------------------------------------------------------------
    def _render(self, keep_cursor: bool = False) -> None:
        from PIL import Image, ImageTk

        overlay = Image.new("RGB", self.view_size, OVERLAY_RGB)
        # Die Maske dient als Deckkraft: wo sie weiß ist, liegt Rot über dem
        # Bild, wo sie schwarz ist, bleibt das Bild unberührt.
        alpha = self._mask_view.point(lambda value: int(value * OVERLAY_ALPHA))
        composed = Image.composite(
            Image.blend(self._view_base, overlay, 1.0), self._view_base, alpha
        )
        self._photo = ImageTk.PhotoImage(composed)
        self.canvas.delete("bild")
        self.canvas.create_image(0, 0, image=self._photo, anchor="nw", tags="bild")
        self.canvas.tag_lower("bild")
        if not keep_cursor:
            self.canvas.delete("cursor")

    def _coverage(self) -> float:
        """Anteil der markierten Fläche – als Warnung vor 0 % und 100 %."""
        histogram = self._mask_view.histogram()
        total = sum(histogram) or 1
        marked = sum(histogram[128:])
        return marked / float(total)

    def _update_info(self) -> None:
        anteil = self._coverage()
        self.info.configure(
            text=(
                f"{self.full_size[0]}x{self.full_size[1]} px · "
                f"markiert {anteil * 100:.1f} % · {len(self._strokes)} Striche"
            )
        )

    # ------------------------------------------------------------------
    # Abschluss
    # ------------------------------------------------------------------
    def problem(self) -> str:
        """Was einem Speichern entgegensteht. Leer = in Ordnung.

        Getrennt vom Dialog, damit die Regel ohne Fenster prüfbar ist –
        eine Meldung, auf die niemand klickt, hält sonst alles an.
        """
        if self._coverage() <= 0.0:
            return (
                "Es ist kein Bereich markiert – eine leere Maske würde nichts "
                "verändern. Male den Bereich, der neu gerechnet werden soll."
            )
        if self._coverage() >= 0.999:
            return (
                "Es ist das ganze Bild markiert. Dann bleibt nichts vom "
                "Original übrig – dafür ist 'Nach Prompt umarbeiten' der "
                "richtige Modus."
            )
        return ""

    def write_mask(self) -> Path:
        """Maske in voller Größe des Originals schreiben."""
        target = paths.ensure_dir(paths.temp_dir() / "masken") / (
            f"{self.source.stem}_maske_{time.strftime('%Y%m%d-%H%M%S')}.png"
        )
        self._mask_full.save(target)
        return target

    def _save(self) -> None:
        hindernis = self.problem()
        if hindernis:
            messagebox.showinfo("Maske", hindernis, parent=self)
            return
        try:
            self.result = self.write_mask()
        except OSError as exc:
            messagebox.showerror(
                "Nicht gespeichert", f"Maske konnte nicht geschrieben werden: {exc}", parent=self
            )
            return
        self.grab_release()
        self.destroy()

    def _cancel(self) -> None:
        self.result = None
        self.grab_release()
        self.destroy()


def paint_mask(
    master: tk.Misc, palette: Palette, source: Path, existing: Path | None = None
) -> Path | None:
    """Maskeneditor öffnen und auf das Ergebnis warten.

    Rückgabe: Pfad zur geschriebenen Maske oder None bei Abbruch.
    """
    editor = MaskEditor(master, palette, source, existing)
    master.wait_window(editor)
    return editor.result
