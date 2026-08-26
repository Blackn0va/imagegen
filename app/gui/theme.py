"""Farben und ttk-Stil.

Zwei Paletten (dunkel/hell). ttk braucht als Grundlage 'clam', weil nur
dort Farben zuverlässig durchgreifen – die Windows-native Engine ignoriert
viele Farbangaben.
"""

from __future__ import annotations

import contextlib
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


# Farben von streamwizard.de (assets/css/global.min.css).
#
# Die Website legt halbdurchsichtige Karten ueber einen Verlauf. Tkinter
# kann beides nicht, also sind die Flaechen ausgerechnet: Kartenfarbe
# rgba(13,27,62,0.7) ueber dem Verlaufsgrund ergibt den Wert, der hier
# als 'surface' steht. So bleibt der Eindruck derselbe.
SW_PRIMARY = "#8B5CF6"  # --primary-color
SW_SECONDARY = "#A78BFA"  # --secondary-color
SW_ACCENT = "#C084FC"  # --accent-color
SW_DARK = "#1F1B36"  # --dark-bg
SW_DARKER = "#151224"  # --darker-bg
SW_TEXT = "#F3F4F6"  # --text-light
SW_TEXT_SEC = "#D1D5DB"  # --text-secondary
SW_MUTED = "#9CA3AF"  # --text-muted
SW_OK = "#10B981"  # --success-color
SW_WARN = "#F59E0B"  # --warning-color
SW_ERROR = "#EF4444"  # --danger-color
SW_INFO = "#3B82F6"  # --info-color

DARK = Palette(
    name="dark",
    bg=SW_DARKER,
    surface="#221D3D",  # Karte ueber dem Verlauf
    surface_alt="#2C2650",
    border="#3B3163",  # rgba(139,92,246,0.22) ueber surface
    text=SW_TEXT,
    text_dim=SW_MUTED,
    accent=SW_PRIMARY,
    accent_hover=SW_SECONDARY,
    accent_text="#FFFFFF",
    ok=SW_OK,
    warn=SW_WARN,
    # Eine Stufe heller als das Rot der Website (#EF4444). Auf der
    # dunklen Kartenflaeche kam das Original nur auf 4,3:1 - unter der
    # Schwelle, ab der Text als lesbar gilt. #F87171 liegt bei 6,5:1 und
    # bleibt in derselben Farbreihe.
    error="#F87171",
    track="#2C2650",
)

# Helle Fassung: dasselbe Violett, damit die Anwendung wiedererkennbar
# bleibt. Die Website selbst ist nur dunkel; die Grau- und Weisstoene sind
# daher aus derselben Tailwind-Reihe gewaehlt wie ihre Textfarben.
LIGHT = Palette(
    name="light",
    bg="#F4F3F8",
    surface="#FFFFFF",
    surface_alt="#EDEAF6",
    border="#D6D0E8",
    text="#1F1B36",
    text_dim="#5B5470",
    accent="#7C3AED",  # eine Stufe dunkler – auf Weiss besser lesbar
    accent_hover="#6D28D9",
    accent_text="#FFFFFF",
    ok="#047857",
    warn="#B45309",
    error="#B91C1C",
    track="#DDD8EC",
)

# Wie die Website: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif.
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
    with contextlib.suppress(Exception):
        style.theme_use("clam")

    root.configure(background=palette.bg)

    style.configure(
        ".",
        background=palette.bg,
        foreground=palette.text,
        fieldbackground=palette.surface,
        font=FONT_UI,
        bordercolor=palette.border,
        focuscolor=palette.accent,
    )

    style.configure("TFrame", background=palette.bg)
    style.configure("Surface.TFrame", background=palette.surface)
    style.configure("Sidebar.TFrame", background=palette.surface)
    style.configure("Card.TFrame", background=palette.surface, relief="flat")

    style.configure("TLabel", background=palette.bg, foreground=palette.text)
    style.configure("Surface.TLabel", background=palette.surface, foreground=palette.text)
    style.configure("Title.TLabel", background=palette.bg, foreground=palette.text, font=FONT_TITLE)
    style.configure("Dim.TLabel", background=palette.bg, foreground=palette.text_dim, font=FONT_SUB)
    style.configure(
        "SurfaceDim.TLabel", background=palette.surface, foreground=palette.text_dim, font=FONT_SUB
    )
    style.configure("Ok.TLabel", background=palette.bg, foreground=palette.ok, font=FONT_SUB)
    style.configure("Warn.TLabel", background=palette.bg, foreground=palette.warn, font=FONT_SUB)
    style.configure("Error.TLabel", background=palette.bg, foreground=palette.error, font=FONT_SUB)
    style.configure(
        "Badge.TLabel",
        background=palette.surface_alt,
        foreground=palette.text_dim,
        font=FONT_SUB,
        padding=(8, 3),
    )
    # Zustandsfarben auf Kartenflächen – ohne sie sieht "bereit" genauso aus
    # wie "fehlt", und der Nutzer muss den Text lesen, um es zu merken.
    style.configure(
        "SurfaceOk.TLabel", background=palette.surface, foreground=palette.ok, font=FONT_SUB
    )
    style.configure(
        "SurfaceWarn.TLabel", background=palette.surface, foreground=palette.warn, font=FONT_SUB
    )
    style.configure(
        "SurfaceError.TLabel", background=palette.surface, foreground=palette.error, font=FONT_SUB
    )
    style.configure(
        "Hint.TLabel", background=palette.surface, foreground=palette.text_dim, font=FONT_SUB
    )

    style.configure(
        "TButton",
        background=palette.surface_alt,
        foreground=palette.text,
        borderwidth=0,
        padding=(12, 7),
        font=FONT_UI,
    )
    style.map(
        "TButton",
        background=[
            ("pressed", palette.border),
            ("active", palette.border),
            ("disabled", palette.surface),
        ],
        foreground=[("disabled", palette.text_dim)],
    )

    style.configure(
        "Accent.TButton",
        background=palette.accent,
        foreground=palette.accent_text,
        font=FONT_UI_BOLD,
        padding=(14, 8),
    )
    style.map(
        "Accent.TButton",
        background=[
            ("pressed", palette.accent_hover),
            ("active", palette.accent_hover),
            ("disabled", palette.surface_alt),
        ],
        foreground=[("disabled", palette.text_dim)],
    )

    style.configure("Danger.TButton", background=palette.surface_alt, foreground=palette.error)
    style.map("Danger.TButton", background=[("active", palette.border)])

    # Sidebar-Navigation: flache Schaltflächen, aktive Seite hervorgehoben
    style.configure(
        "Nav.TButton",
        background=palette.surface,
        foreground=palette.text_dim,
        borderwidth=0,
        padding=(16, 10),
        anchor="w",
        font=FONT_UI,
    )
    style.map(
        "Nav.TButton",
        background=[("active", palette.surface_alt)],
        foreground=[("active", palette.text)],
    )
    style.configure(
        "NavActive.TButton",
        background=palette.surface_alt,
        foreground=palette.text,
        borderwidth=0,
        padding=(16, 10),
        anchor="w",
        font=FONT_UI_BOLD,
    )

    style.configure(
        "TEntry",
        fieldbackground=palette.surface,
        foreground=palette.text,
        bordercolor=palette.border,
        insertcolor=palette.text,
        padding=6,
    )
    style.map("TEntry", bordercolor=[("focus", palette.accent)])

    style.configure(
        "TCombobox",
        fieldbackground=palette.surface,
        background=palette.surface,
        foreground=palette.text,
        arrowcolor=palette.text_dim,
        padding=5,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette.surface)],
        bordercolor=[("focus", palette.accent)],
    )

    style.configure("TCheckbutton", background=palette.bg, foreground=palette.text)
    style.map("TCheckbutton", background=[("active", palette.bg)])
    style.configure("Surface.TCheckbutton", background=palette.surface, foreground=palette.text)

    style.configure(
        "TScale", background=palette.bg, troughcolor=palette.track, bordercolor=palette.border
    )

    style.configure(
        "TProgressbar",
        background=palette.accent,
        troughcolor=palette.track,
        borderwidth=0,
        thickness=8,
    )

    style.configure("TNotebook", background=palette.bg, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=palette.surface,
        foreground=palette.text_dim,
        padding=(14, 8),
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", palette.surface_alt)],
        foreground=[("selected", palette.text)],
    )

    # Etwas mehr Zeilenhöhe: Listen wirken sonst gedrängt und sind auf
    # hochauflösenden Bildschirmen schwer zu treffen.
    style.configure(
        "Treeview",
        background=palette.surface,
        fieldbackground=palette.surface,
        foreground=palette.text,
        borderwidth=0,
        rowheight=30,
    )
    style.configure(
        "Treeview.Heading",
        background=palette.surface_alt,
        foreground=palette.text_dim,
        borderwidth=0,
        font=FONT_SUB,
    )
    style.map(
        "Treeview",
        background=[("selected", palette.accent)],
        foreground=[("selected", palette.accent_text)],
    )

    style.configure("TSeparator", background=palette.border)
    style.configure(
        "Footer.TLabel", background=palette.surface, foreground=palette.text_dim, font=FONT_SUB
    )
    style.configure(
        "Vertical.TScrollbar",
        background=palette.surface_alt,
        troughcolor=palette.bg,
        borderwidth=0,
        arrowcolor=palette.text_dim,
    )
    style.configure(
        "Horizontal.TScrollbar",
        background=palette.surface_alt,
        troughcolor=palette.bg,
        borderwidth=0,
        arrowcolor=palette.text_dim,
    )
