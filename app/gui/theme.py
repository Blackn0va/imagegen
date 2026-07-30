"""Farben und ttk-Stil.

Zwei Paletten (dunkel/hell). ttk braucht als Grundlage 'clam', weil nur
dort Farben zuverlässig durchgreifen – die Windows-native Engine ignoriert
viele Farbangaben.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    surface: str
    surface_alt: str
    border: str
    text: str
    text_dim: str
    accent: str
    accent_hover: str
    accent_text: str
    ok: str
    warn: str
    error: str
    track: str


DARK = Palette(
    name="dark",
    bg="#14161c",
    surface="#1c1f27",
    surface_alt="#23272f",
    border="#31363f",
    text="#e7e9ee",
    text_dim="#98a0ae",
    accent="#4f8cff",
    accent_hover="#3b78ea",
    accent_text="#ffffff",
    ok="#43c78a",
    warn="#e2b23c",
    error="#e35d6a",
    track="#2b303a",
)

LIGHT = Palette(
    name="light",
    bg="#f3f4f7",
    surface="#ffffff",
    surface_alt="#e9ebf0",
    border="#cbd0d9",
    text="#1b1e24",
    text_dim="#5c6472",
    accent="#2f6fdc",
    accent_hover="#255cb9",
    accent_text="#ffffff",
    ok="#1f9d63",
    warn="#a8761a",
    error="#c33c49",
    track="#dcdfe6",
)

FONT_UI = ("Segoe UI", 10)
FONT_UI_BOLD = ("Segoe UI Semibold", 10)
FONT_TITLE = ("Segoe UI Semibold", 15)
FONT_SUB = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)


def palette_for(name: str) -> Palette:
    return LIGHT if str(name).lower() == "light" else DARK


def apply(root, palette: Palette) -> None:
    """Stil auf ein Tk-Fenster anwenden."""
    from tkinter import ttk

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # noqa: BLE001 – Notfalls Vorgabethema
        pass

    root.configure(background=palette.bg)

    style.configure(".", background=palette.bg, foreground=palette.text,
                    fieldbackground=palette.surface, font=FONT_UI,
                    bordercolor=palette.border, focuscolor=palette.accent)

    style.configure("TFrame", background=palette.bg)
    style.configure("Surface.TFrame", background=palette.surface)
    style.configure("Sidebar.TFrame", background=palette.surface)
    style.configure("Card.TFrame", background=palette.surface, relief="flat")

    style.configure("TLabel", background=palette.bg, foreground=palette.text)
    style.configure("Surface.TLabel", background=palette.surface, foreground=palette.text)
    style.configure("Title.TLabel", background=palette.bg, foreground=palette.text, font=FONT_TITLE)
    style.configure("Dim.TLabel", background=palette.bg, foreground=palette.text_dim, font=FONT_SUB)
    style.configure("SurfaceDim.TLabel", background=palette.surface,
                    foreground=palette.text_dim, font=FONT_SUB)
    style.configure("Ok.TLabel", background=palette.bg, foreground=palette.ok, font=FONT_SUB)
    style.configure("Warn.TLabel", background=palette.bg, foreground=palette.warn, font=FONT_SUB)
    style.configure("Error.TLabel", background=palette.bg, foreground=palette.error, font=FONT_SUB)
    style.configure("Badge.TLabel", background=palette.surface_alt,
                    foreground=palette.text_dim, font=FONT_SUB, padding=(8, 3))

    style.configure("TButton", background=palette.surface_alt, foreground=palette.text,
                    borderwidth=0, padding=(12, 7), font=FONT_UI)
    style.map("TButton",
              background=[("pressed", palette.border), ("active", palette.border),
                          ("disabled", palette.surface)],
              foreground=[("disabled", palette.text_dim)])

    style.configure("Accent.TButton", background=palette.accent,
                    foreground=palette.accent_text, font=FONT_UI_BOLD, padding=(14, 8))
    style.map("Accent.TButton",
              background=[("pressed", palette.accent_hover), ("active", palette.accent_hover),
                          ("disabled", palette.surface_alt)],
              foreground=[("disabled", palette.text_dim)])

    style.configure("Danger.TButton", background=palette.surface_alt, foreground=palette.error)
    style.map("Danger.TButton", background=[("active", palette.border)])

    # Sidebar-Navigation: flache Schaltflächen, aktive Seite hervorgehoben
    style.configure("Nav.TButton", background=palette.surface, foreground=palette.text_dim,
                    borderwidth=0, padding=(16, 10), anchor="w", font=FONT_UI)
    style.map("Nav.TButton",
              background=[("active", palette.surface_alt)],
              foreground=[("active", palette.text)])
    style.configure("NavActive.TButton", background=palette.surface_alt,
                    foreground=palette.text, borderwidth=0, padding=(16, 10),
                    anchor="w", font=FONT_UI_BOLD)

    style.configure("TEntry", fieldbackground=palette.surface, foreground=palette.text,
                    bordercolor=palette.border, insertcolor=palette.text, padding=6)
    style.map("TEntry", bordercolor=[("focus", palette.accent)])

    style.configure("TCombobox", fieldbackground=palette.surface, background=palette.surface,
                    foreground=palette.text, arrowcolor=palette.text_dim, padding=5)
    style.map("TCombobox", fieldbackground=[("readonly", palette.surface)],
              bordercolor=[("focus", palette.accent)])

    style.configure("TCheckbutton", background=palette.bg, foreground=palette.text)
    style.map("TCheckbutton", background=[("active", palette.bg)])
    style.configure("Surface.TCheckbutton", background=palette.surface, foreground=palette.text)

    style.configure("TScale", background=palette.bg, troughcolor=palette.track,
                    bordercolor=palette.border)

    style.configure("TProgressbar", background=palette.accent, troughcolor=palette.track,
                    borderwidth=0, thickness=8)

    style.configure("TNotebook", background=palette.bg, borderwidth=0)
    style.configure("TNotebook.Tab", background=palette.surface, foreground=palette.text_dim,
                    padding=(14, 8), borderwidth=0)
    style.map("TNotebook.Tab",
              background=[("selected", palette.surface_alt)],
              foreground=[("selected", palette.text)])

    style.configure("Treeview", background=palette.surface, fieldbackground=palette.surface,
                    foreground=palette.text, borderwidth=0, rowheight=26)
    style.configure("Treeview.Heading", background=palette.surface_alt,
                    foreground=palette.text_dim, borderwidth=0, font=FONT_SUB)
    style.map("Treeview", background=[("selected", palette.accent)],
              foreground=[("selected", palette.accent_text)])

    style.configure("TSeparator", background=palette.border)
    style.configure("Vertical.TScrollbar", background=palette.surface_alt,
                    troughcolor=palette.bg, borderwidth=0, arrowcolor=palette.text_dim)
    style.configure("Horizontal.TScrollbar", background=palette.surface_alt,
                    troughcolor=palette.bg, borderwidth=0, arrowcolor=palette.text_dim)
