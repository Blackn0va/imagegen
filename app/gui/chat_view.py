"""Chat-Anzeige: Markdown, Code-Blöcke, Bilder.

tkinter bringt keine Markdown-Darstellung mit, und für ein Text-Widget
gibt es auch keine fertige. Gebraucht wird aber nur ein kleiner Teil von
Markdown – der, den ein Sprachmodell tatsächlich ausgibt:

  * ``` Code-Blöcke mit Sprachangabe  (der wichtigste Teil)
  * `Code` mitten im Satz
  * **fett**, *kursiv*
  * # Überschriften und - Listen

Alles davon lässt sich mit Text-Tags abbilden. Code-Blöcke bekommen
zusätzlich einen **Kopierknopf**, weil genau dafür ein Code-Writer da
ist: Code herausholen, nicht bewundern.

Die Anzeige wird stückweise gefüttert (``append_delta``), während das
Modell schreibt. Markdown wird dabei erst beim Abschluss einer Antwort
ausgewertet – ein halber Code-Block lässt sich nicht sinnvoll einfärben,
und ständiges Neuzeichnen bei 20 Token/s flackert.
"""

from __future__ import annotations

import re
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import ttk
from typing import Any

from .theme import FONT_MONO, FONT_SUB, FONT_UI_BOLD, Palette

# Ein Code-Block: ```sprache ... ```
_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.S)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_BOLD = re.compile(r"\*\*([^*\n]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.M)


class ChatView(ttk.Frame):
    """Rollender Verlauf mit Markdown, Code-Blöcken und Bildern."""

    def __init__(self, master, palette: Palette, on_copy: Callable[[str], None]) -> None:
        super().__init__(master)
        self.palette = palette
        self.on_copy = on_copy
        self._photos: list[Any] = []  # Tk gibt Bilder sonst sofort frei
        self._blocks = 0

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.text = tk.Text(
            self,
            wrap="word",
            background=palette.surface,
            foreground=palette.text,
            insertbackground=palette.text,
            relief="flat",
            padx=14,
            pady=12,
            font=FONT_SUB,
            state="disabled",
            spacing1=2,
            spacing3=4,
        )
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")

        self._configure_tags()

    def _configure_tags(self) -> None:
        palette = self.palette
        self.text.tag_configure("rolle_du", foreground=palette.accent, font=FONT_UI_BOLD)
        self.text.tag_configure("rolle_ai", foreground=palette.ok, font=FONT_UI_BOLD)
        self.text.tag_configure("text", foreground=palette.text)
        self.text.tag_configure("dim", foreground=palette.text_dim)
        self.text.tag_configure("fett", font=FONT_UI_BOLD)
        self.text.tag_configure("kursiv", font=(FONT_SUB[0], FONT_SUB[1], "italic"))
        self.text.tag_configure(
            "h1", font=(FONT_SUB[0], FONT_SUB[1] + 4, "bold"), spacing1=8, spacing3=4
        )
        self.text.tag_configure(
            "h2", font=(FONT_SUB[0], FONT_SUB[1] + 2, "bold"), spacing1=6, spacing3=3
        )
        self.text.tag_configure(
            "code_inline",
            font=FONT_MONO,
            background=palette.surface_alt,
            foreground=palette.accent,
        )
        self.text.tag_configure(
            "code_block",
            font=FONT_MONO,
            background=palette.surface_alt,
            foreground=palette.text,
            lmargin1=16,
            lmargin2=16,
            rmargin=16,
            spacing1=4,
            spacing3=4,
        )
        self.text.tag_configure("liste", lmargin1=18, lmargin2=32)
        self.text.tag_configure("fehler", foreground=palette.error)

    # ------------------------------------------------------------------
    # Schreiben
    # ------------------------------------------------------------------
    def _write(self, content: str, *tags: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", content, tags or ())
        self.text.configure(state="disabled")
        self.text.see("end")

    def start_message(self, role: str) -> None:
        """Kopfzeile einer Nachricht setzen."""
        beschriftung = {"user": "Du", "assistant": "Assistent"}.get(role, role)
        tag = "rolle_du" if role == "user" else "rolle_ai"
        self._write("\n" if self.text.index("end-1c") != "1.0" else "", "text")
        self._write(f"{beschriftung}\n", tag)

    def append_delta(self, stueck: str) -> None:
        """Roh anhängen, während das Modell schreibt.

        Bewusst ohne Formatierung: ein angefangener Code-Block hat noch
        kein schließendes ``` und wäre nicht auswertbar.
        """
        self._write(stueck, "text")

    def replace_last_with_markdown(self, roh: str, laenge_roh: int) -> None:
        """Den roh geschriebenen Text durch die formatierte Fassung ersetzen."""
        self.text.configure(state="normal")
        ende = self.text.index("end-1c")
        start = f"{ende} - {laenge_roh} chars"
        self.text.delete(start, ende)
        self.text.configure(state="disabled")
        self.render_markdown(roh)

    def render_markdown(self, roh: str) -> None:
        """Markdown auswerten und mit Tags schreiben."""
        position = 0
        for treffer in _FENCE.finditer(roh):
            self._render_inline(roh[position : treffer.start()])
            self._render_code_block(treffer.group(1).strip(), treffer.group(2))
            position = treffer.end()
        self._render_inline(roh[position:])
        self._write("\n")

    def _render_code_block(self, sprache: str, code: str) -> None:
        """Code-Block mit Kopfzeile und Kopierknopf."""
        self._blocks += 1
        kopf = sprache or "Code"
        self._write(f"\n{kopf}  ", "dim")

        knopf = ttk.Button(
            self.text,
            text="Kopieren",
            width=10,
            command=lambda inhalt=code: self.on_copy(inhalt),
        )
        self.text.configure(state="normal")
        self.text.window_create("end", window=knopf)
        self.text.insert("end", "\n")
        self.text.configure(state="disabled")

        self._write(code.rstrip("\n") + "\n", "code_block")

    def _render_inline(self, roh: str) -> None:
        """Absätze, Überschriften, Listen, fett/kursiv und `Code`."""
        if not roh.strip():
            return
        for zeile in roh.split("\n"):
            kopf = _HEADING.match(zeile)
            if kopf:
                stufe = "h1" if len(kopf.group(1)) == 1 else "h2"
                self._write(kopf.group(2) + "\n", stufe)
                continue
            listen_tag = ("liste",) if zeile.lstrip().startswith(("- ", "* ", "• ")) else ()
            self._render_spans(zeile, listen_tag)
            self._write("\n", *listen_tag)

    def _render_spans(self, zeile: str, grund_tags: tuple[str, ...]) -> None:
        """Eine Zeile in Stücke zerlegen und je Stück taggen."""
        muster = [(_INLINE_CODE, "code_inline"), (_BOLD, "fett"), (_ITALIC, "kursiv")]
        # Alle Treffer sammeln, nach Position sortieren, Überlappungen
        # verwerfen – sonst würde **`x`** zweimal geschrieben.
        treffer: list[tuple[int, int, str, str]] = []
        for regel, tag in muster:
            for m in regel.finditer(zeile):
                treffer.append((m.start(), m.end(), m.group(1), tag))
        treffer.sort()
        position = 0
        for start, ende, inhalt, tag in treffer:
            if start < position:
                continue
            if start > position:
                self._write(zeile[position:start], "text", *grund_tags)
            self._write(inhalt, tag, *grund_tags)
            position = ende
        if position < len(zeile):
            self._write(zeile[position:], "text", *grund_tags)

    # ------------------------------------------------------------------
    # Bilder und Sonstiges
    # ------------------------------------------------------------------
    def add_images(self, pfade: tuple[Path, ...], hoehe: int = 160) -> None:
        """Angehängte Bilder als Vorschau in den Verlauf legen."""
        if not pfade:
            return
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self._write(f"[{len(pfade)} Bild(er) angehängt]\n", "dim")
            return
        for pfad in pfade:
            try:
                with Image.open(pfad) as bild:
                    bild.load()
                    kopie = bild.convert("RGB")
                faktor = hoehe / float(kopie.height or 1)
                kopie = kopie.resize((max(1, int(kopie.width * faktor)), hoehe), Image.LANCZOS)
                foto = ImageTk.PhotoImage(kopie)
            except Exception:
                self._write(f"[Bild nicht lesbar: {pfad.name}]\n", "fehler")
                continue
            self._photos.append(foto)
            self.text.configure(state="normal")
            self.text.image_create("end", image=foto, padx=4, pady=4)
            self.text.insert("end", "\n")
            self.text.configure(state="disabled")
        self.text.see("end")

    def add_note(self, text: str, tag: str = "dim") -> None:
        self._write(text.rstrip() + "\n", tag)

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._photos.clear()
        self._blocks = 0
