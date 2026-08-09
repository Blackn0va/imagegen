"""Wiederverwendbare Bausteine für die Oberfläche.

Alles auf tkinter/ttk aufgebaut. Jeder Baustein hält seinen Wert in einer
Tk-Variable, damit das Auslesen beim Speichern einheitlich läuft
(``value()``).
"""

from __future__ import annotations

import contextlib
import tkinter as tk
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any

from .theme import FONT_MONO, FONT_SUB, FONT_UI_BOLD, Palette


# ---------------------------------------------------------------------------
# Rahmen und Layout
# ---------------------------------------------------------------------------
class Card(ttk.Frame):
    """Abgesetzte Fläche mit Titel und optionaler Beschreibung."""

    def __init__(self, master, palette: Palette, title: str = "", subtitle: str = "") -> None:
        super().__init__(master, style="Card.TFrame", padding=16)
        self.palette = palette
        self.columnconfigure(0, weight=1)
        self._row = 0
        if title:
            label = ttk.Label(self, text=title, style="Surface.TLabel", font=FONT_UI_BOLD)
            label.grid(row=self._row, column=0, sticky="w")
            self._row += 1
        if subtitle:
            sub = ttk.Label(self, text=subtitle, style="SurfaceDim.TLabel", wraplength=760)
            sub.grid(row=self._row, column=0, sticky="w", pady=(2, 8))
            self._row += 1

    def body(self) -> ttk.Frame:
        """Innenbereich für den Inhalt."""
        frame = ttk.Frame(self, style="Card.TFrame")
        frame.grid(row=self._row, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)
        self.rowconfigure(self._row, weight=1)
        self._row += 1
        return frame

    def set_visible(self, visible: bool) -> None:
        """Ganze Karte ein- oder ausblenden.

        ``grid_remove()`` behält die Rasterangaben, die Karten darunter
        rücken also nach oben. Eine Karte, die zur gewählten Aufgabe nicht
        gehört, soll nicht ausgegraut herumstehen, sondern weg sein.
        """
        if visible:
            self.grid()
        else:
            self.grid_remove()


class ScrollArea(ttk.Frame):
    """Senkrecht rollbarer Bereich – Seiten mit vielen Feldern brauchen das."""

    def __init__(self, master, palette: Palette) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, bg=palette.bg, highlightthickness=0, borderwidth=0)
        self.scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.canvas.configure(yscrollcommand=self.scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scroll.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<Enter>", lambda _e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda _e: self._bind_wheel(False))

    def _on_inner(self, _event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, event) -> None:
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, active: bool) -> None:
        if active:
            self.canvas.bind_all("<MouseWheel>", self._on_wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _on_wheel(self, event) -> None:
        self.canvas.yview_scroll(int(-event.delta / 120), "units")


# ---------------------------------------------------------------------------
# Eingabezeilen
# ---------------------------------------------------------------------------
class Visible:
    """Mischklasse: eine Zeile vollständig ein- und ausblenden.

    ``grid_remove()`` merkt sich alle Rasterangaben, ``grid()`` stellt sie
    unverändert wieder her. Deshalb muss nichts neu positioniert werden,
    und die Zeilen darunter rücken von selbst nach – anders als beim
    bloßen Ausgrauen, bei dem der leere Platz stehen bleibt.

    Angemeldet wird alles, was direkt im Raster des Elternteils liegt:
    Beschriftung, Eingabe (bei manchen Zeilen ein Rahmen darum) und der
    Hinweistext.
    """

    def _register(self, *widgets: Any) -> None:
        cells = self.__dict__.setdefault("_cells", [])
        cells.extend(widget for widget in widgets if widget is not None)

    def set_visible(self, visible: bool) -> None:
        for widget in self.__dict__.get("_cells", ()):
            try:
                if visible:
                    widget.grid()
                else:
                    widget.grid_remove()
            except tk.TclError:
                continue

    def is_visible(self) -> bool:
        for widget in self.__dict__.get("_cells", ()):
            with contextlib.suppress(tk.TclError):
                return bool(widget.winfo_manager())
        return False


class Row(Visible):
    """Basis: Beschriftung links, Eingabe rechts, Hinweis darunter."""

    def __init__(self, master: ttk.Frame, row: int, label: str, hint: str = "") -> None:
        self.master = master
        self.row = row
        style = "Surface.TLabel" if "Card" in str(master.cget("style")) else "TLabel"
        self.label_widget = ttk.Label(master, text=label, style=style)
        self.label_widget.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
        self._register(self.label_widget)
        self.hint = hint
        self._hint_label: ttk.Label | None = None

    def add_hint(self, text: str, style: str = "Dim.TLabel") -> None:
        if self._hint_label is not None:
            self._hint_label.configure(text=text, style=style)
            return
        self._hint_label = ttk.Label(self.master, text=text, style=style, wraplength=560)
        self._hint_label.grid(row=self.row + 1, column=1, sticky="w", pady=(0, 6))
        self._register(self._hint_label)

    def value(self) -> Any:  # pragma: no cover – von Unterklassen ersetzt
        raise NotImplementedError


class EntryRow(Row):
    def __init__(
        self, master, row: int, label: str, value: str = "", hint: str = "", width: int = 40
    ) -> None:
        super().__init__(master, row, label, hint)
        self.var = tk.StringVar(value=value)
        self.widget = ttk.Entry(master, textvariable=self.var, width=width)
        self.widget.grid(row=row, column=1, sticky="ew", pady=4)
        self._register(self.widget)
        if hint:
            self.add_hint(hint)

    def value(self) -> str:
        return self.var.get()

    def set(self, value: str) -> None:
        self.var.set(value)


class TextRow(Visible):
    """Mehrzeiliges Feld (Prompt, Sprechtext)."""

    def __init__(
        self,
        master,
        row: int,
        label: str,
        palette: Palette,
        value: str = "",
        height: int = 4,
        hint: str = "",
    ) -> None:
        style = "Surface.TLabel" if "Card" in str(master.cget("style")) else "TLabel"
        self.label_widget = ttk.Label(master, text=label, style=style)
        self.label_widget.grid(row=row, column=0, sticky="nw", padx=(0, 12), pady=4)
        self._register(self.label_widget)
        self.widget = tk.Text(
            master,
            height=height,
            wrap="word",
            background=palette.surface_alt,
            foreground=palette.text,
            insertbackground=palette.text,
            relief="flat",
            padx=8,
            pady=6,
            font=FONT_SUB,
        )
        self.widget.grid(row=row, column=1, sticky="ew", pady=4)
        self._register(self.widget)
        if value:
            self.widget.insert("1.0", value)
        if hint:
            hint_label = ttk.Label(master, text=hint, style="Dim.TLabel", wraplength=560)
            hint_label.grid(row=row + 1, column=1, sticky="w", pady=(0, 6))
            self._register(hint_label)

    def value(self) -> str:
        return self.widget.get("1.0", "end").strip()

    def set(self, value: str) -> None:
        self.widget.delete("1.0", "end")
        self.widget.insert("1.0", value)


class ComboRow(Row):
    def __init__(
        self,
        master,
        row: int,
        label: str,
        values: Sequence[str],
        value: str = "",
        hint: str = "",
        width: int = 28,
        on_change: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master, row, label, hint)
        self.var = tk.StringVar(value=value or (values[0] if values else ""))
        self.widget = ttk.Combobox(
            master, textvariable=self.var, values=list(values), state="readonly", width=width
        )
        self.widget.grid(row=row, column=1, sticky="w", pady=4)
        self._register(self.widget)
        if on_change is not None:
            self.widget.bind("<<ComboboxSelected>>", lambda _e: on_change(self.var.get()))
        if hint:
            self.add_hint(hint)

    def value(self) -> str:
        return self.var.get()

    def set_values(self, values: Sequence[str], keep: bool = True) -> None:
        current = self.var.get()
        self.widget.configure(values=list(values))
        if keep and current in values:
            self.var.set(current)
        elif values:
            self.var.set(values[0])
        else:
            self.var.set("")


class SpinRow(Row):
    def __init__(
        self,
        master,
        row: int,
        label: str,
        value: float,
        low: float,
        high: float,
        step: float = 1.0,
        hint: str = "",
        integer: bool = True,
    ) -> None:
        super().__init__(master, row, label, hint)
        self.integer = integer
        self.var = tk.StringVar(value=str(int(value) if integer else round(value, 2)))
        self.widget = ttk.Spinbox(
            master, from_=low, to=high, increment=step, textvariable=self.var, width=12
        )
        self.widget.grid(row=row, column=1, sticky="w", pady=4)
        self._register(self.widget)
        self.low, self.high = low, high
        if hint:
            self.add_hint(hint)

    def value(self) -> float | int:
        raw = self.var.get().strip().replace(",", ".")
        try:
            number = float(raw)
        except ValueError:
            number = self.low
        number = max(self.low, min(self.high, number))
        return int(number) if self.integer else number


class SliderRow(Row):
    def __init__(
        self,
        master,
        row: int,
        label: str,
        value: float,
        low: float,
        high: float,
        hint: str = "",
        integer: bool = False,
        unit: str = "",
    ) -> None:
        super().__init__(master, row, label, hint)
        self.integer = integer
        self.unit = unit
        self.var = tk.DoubleVar(value=value)
        holder = ttk.Frame(master, style=str(master.cget("style")) or "TFrame")
        holder.grid(row=row, column=1, sticky="ew", pady=4)
        holder.columnconfigure(0, weight=1)
        self._register(holder)
        self.widget = ttk.Scale(
            holder,
            from_=low,
            to=high,
            variable=self.var,
            orient="horizontal",
            command=self._on_move,
        )
        self.widget.grid(row=0, column=0, sticky="ew")
        self.readout = ttk.Label(holder, text=self._format(value), width=10, style="Dim.TLabel")
        self.readout.grid(row=0, column=1, sticky="e", padx=(10, 0))
        if hint:
            self.add_hint(hint)

    def _format(self, value: float) -> str:
        text = f"{round(value)}" if self.integer else f"{value:.2f}"
        return f"{text}{self.unit}"

    def _on_move(self, _value: str) -> None:
        self.readout.configure(text=self._format(self.var.get()))

    def value(self) -> float | int:
        return round(self.var.get()) if self.integer else round(self.var.get(), 3)


class CheckRow(Visible):
    def __init__(self, master, row: int, label: str, value: bool, hint: str = "") -> None:
        self.var = tk.BooleanVar(value=bool(value))
        style = "Surface.TCheckbutton" if "Card" in str(master.cget("style")) else "TCheckbutton"
        self.widget = ttk.Checkbutton(master, text=label, variable=self.var, style=style)
        self.widget.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)
        self._register(self.widget)
        if hint:
            label_style = (
                "SurfaceDim.TLabel" if "Card" in str(master.cget("style")) else "Dim.TLabel"
            )
            hint_label = ttk.Label(master, text=hint, style=label_style, wraplength=620)
            hint_label.grid(
                row=row + 1, column=0, columnspan=2, sticky="w", padx=(24, 0), pady=(0, 6)
            )
            self._register(hint_label)

    def value(self) -> bool:
        return bool(self.var.get())


class PathRow(Row):
    """Pfadfeld mit Auswahlknopf (Datei oder Ordner)."""

    def __init__(
        self,
        master,
        row: int,
        label: str,
        value: str = "",
        hint: str = "",
        directory: bool = False,
        filetypes: Sequence[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(master, row, label, hint)
        self.directory = directory
        self.filetypes = list(filetypes or [("Alle Dateien", "*.*")])
        self.var = tk.StringVar(value=value)
        holder = ttk.Frame(master, style=str(master.cget("style")) or "TFrame")
        holder.grid(row=row, column=1, sticky="ew", pady=4)
        holder.columnconfigure(0, weight=1)
        self._register(holder)
        self.widget = ttk.Entry(holder, textvariable=self.var)
        self.widget.grid(row=0, column=0, sticky="ew")
        ttk.Button(holder, text="Wählen …", command=self._choose).grid(row=0, column=1, padx=(8, 0))
        if hint:
            self.add_hint(hint)

    def _choose(self) -> None:
        if self.directory:
            chosen = filedialog.askdirectory(title="Ordner wählen")
        else:
            chosen = filedialog.askopenfilename(title="Datei wählen", filetypes=self.filetypes)
        if chosen:
            self.var.set(chosen)

    def value(self) -> str:
        return self.var.get().strip()


# ---------------------------------------------------------------------------
# Anzeige
# ---------------------------------------------------------------------------
class LogView(ttk.Frame):
    """Rollender Textbereich mit Farbmarkierung je Stufe."""

    def __init__(self, master, palette: Palette, height: int = 18) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.text = tk.Text(
            master=self,
            height=height,
            wrap="word",
            background=palette.surface,
            foreground=palette.text,
            insertbackground=palette.text,
            relief="flat",
            padx=10,
            pady=8,
            font=FONT_MONO,
            state="disabled",
        )
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        self.text.tag_configure("info", foreground=palette.text)
        self.text.tag_configure("dim", foreground=palette.text_dim)
        self.text.tag_configure("ok", foreground=palette.ok)
        self.text.tag_configure("warn", foreground=palette.warn)
        self.text.tag_configure("error", foreground=palette.error)
        self._lines = 0

    def append(self, message: str, tag: str = "info") -> None:
        self.text.configure(state="normal")
        self.text.insert("end", message.rstrip() + "\n", tag)
        self._lines += 1
        if self._lines > 2000:  # Speicher begrenzen
            self.text.delete("1.0", "500.0")
            self._lines -= 500
        self.text.see("end")
        self.text.configure(state="disabled")

    def set_text(self, content: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")
        self._lines = content.count("\n")


def _thumbnail(path: Path, box: tuple[int, int]) -> tuple[Any, str]:
    """Verkleinertes Bild für die Vorschau. Rückgabe: (PhotoImage, Text).

    Erst über Pillow (kann alle Formate), sonst über die Bordmittel von Tk
    (nur PNG/GIF). Schlägt beides fehl, ist das kein Fehler – die Vorschau
    ist Beiwerk, der Auftrag läuft trotzdem.
    """
    try:
        from PIL import Image, ImageTk
    except ImportError:
        return _tk_thumbnail(path, box)
    try:
        with Image.open(path) as image:
            image.load()
            size = f"{image.width}x{image.height}"
            copy = image.copy()
            copy.thumbnail(box, Image.LANCZOS)
            return ImageTk.PhotoImage(copy), size
    except Exception as exc:
        return None, str(exc)[:120]


def _tk_thumbnail(path: Path, box: tuple[int, int]) -> tuple[Any, str]:
    """Rückfallebene ohne Pillow: Tk kann PNG und GIF, aber nur ganzzahlig teilen."""
    if path.suffix.lower() not in (".png", ".gif"):
        return None, "Ohne Pillow sind nur PNG und GIF darstellbar."
    try:
        photo = tk.PhotoImage(file=str(path))
    except tk.TclError as exc:
        return None, str(exc)[:120]
    size = f"{photo.width()}x{photo.height()}"
    factor = max(1, -(-photo.width() // box[0]), -(-photo.height() // box[1]))
    if factor > 1:
        photo = photo.subsample(factor, factor)
    return photo, size


class ImagePreview(ttk.Frame):
    """Feste Fläche mit einer Bildvorschau und einer Zeile Text darunter."""

    def __init__(
        self,
        master,
        palette: Palette,
        width: int = 320,
        height: int = 220,
        caption: str = "Keine Vorschau",
    ) -> None:
        super().__init__(master, style=str(master.cget("style")) or "TFrame")
        self.palette = palette
        self.box = (width, height)
        self.default_caption = caption
        self._photo: Any = None
        self.canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            background=palette.surface_alt,
            highlightthickness=0,
            borderwidth=0,
        )
        self.canvas.grid(row=0, column=0, sticky="nw")
        self.caption = ttk.Label(self, text=caption, style="SurfaceDim.TLabel", wraplength=width)
        self.caption.grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.clear()

    def clear(self, text: str = "") -> None:
        self._photo = None
        self.canvas.delete("all")
        self.canvas.create_text(
            self.box[0] / 2,
            self.box[1] / 2,
            text=text or self.default_caption,
            fill=self.palette.text_dim,
            width=self.box[0] - 20,
            justify="center",
        )
        self.caption.configure(text="")

    def show(self, path: Path) -> bool:
        target = Path(path)
        if not target.is_file():
            self.clear("Datei nicht gefunden")
            return False
        photo, note = _thumbnail(target, self.box)
        self.canvas.delete("all")
        # Verweis festhalten: Tk gibt das Bild sonst sofort wieder frei und
        # die Fläche bleibt leer.
        self._photo = photo
        if photo is None:
            self.canvas.create_text(
                self.box[0] / 2,
                self.box[1] / 2,
                text="Vorschau nicht möglich",
                fill=self.palette.text_dim,
                width=self.box[0] - 20,
                justify="center",
            )
            self.caption.configure(text=note)
            return False
        self.canvas.create_image(self.box[0] / 2, self.box[1] / 2, image=photo)
        self.caption.configure(text=f"{target.name} · {note}")
        return True


class Banner(ttk.Frame):
    """Hinweisstreifen für Meldungen, die stehen bleiben sollen."""

    def __init__(self, master, palette: Palette) -> None:
        super().__init__(master, style="Card.TFrame", padding=(12, 8))
        self.palette = palette
        self.label = ttk.Label(self, text="", style="SurfaceDim.TLabel", wraplength=880)
        self.label.grid(row=0, column=0, sticky="w")
        self.columnconfigure(0, weight=1)
        self._visible = False

    def show(self, message: str, level: str = "info") -> None:
        colors = {
            "info": self.palette.text_dim,
            "ok": self.palette.ok,
            "warn": self.palette.warn,
            "error": self.palette.error,
        }
        self.label.configure(text=message, foreground=colors.get(level, self.palette.text_dim))
        if not self._visible:
            self.grid()
            self._visible = True

    def hide(self) -> None:
        if self._visible:
            self.grid_remove()
            self._visible = False


def grid_rows(frame: ttk.Frame, start: int = 0) -> Iterable[int]:
    """Zeilenzähler, der Platz für Hinweiszeilen lässt."""
    row = start
    while True:
        yield row
        row += 2
