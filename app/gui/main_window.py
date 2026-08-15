"""Hauptfenster.

Aufbau: Kopfzeile (Hardware/Backend), links Navigation, rechts die Seite,
unten Statuszeile mit Fortschritt und Abbruch.

Wichtig: Ereignisse der Warteschlange treffen im Arbeiter-Thread ein.
tkinter darf nur aus dem Hauptthread bedient werden, deshalb werden sie
über eine ``queue.Queue`` eingesammelt und in ``_drain_events`` (per
``after``) verarbeitet.
"""

from __future__ import annotations

import logging
import queue
import tkinter as tk
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .. import (
    __app_display_name__,
    __version__,
    accel,
    compose,
    licensing,
    models,
    paths,
    pipeline_image,
    pipeline_video,
    pipeline_voice,
    upscale,
    voice_profiles,
)
from ..config import (
    COMPUTE_CHOICES,
    DEVICE_CHOICES,
    DIAMOND_SHAPES,
    IMAGE_FORMATS,
    OPENVINO_DEVICES,
    UPSCALE_FACTORS,
    VIDEO_CONTAINERS,
    AppConfig,
)
from ..diamond import SHAPE_LABELS as DIAMOND_SHAPE_LABELS
from ..jobs import JobEvent, JobState
from . import theme
from .widgets import (
    Banner,
    ButtonRow,
    Card,
    CheckRow,
    ComboRow,
    EntryRow,
    ImagePreview,
    LogView,
    PathRow,
    ScrollArea,
    SliderRow,
    SpinRow,
    TextRow,
)

log = logging.getLogger(__name__)

PAGES: tuple[tuple[str, str], ...] = (
    ("image", "Bild"),
    ("imageedit", "Bild bearbeiten"),
    ("video", "Video"),
    ("voice", "Stimme"),
    ("voicetrain", "Stimme anlernen"),
    ("queue", "Warteschlange"),
    ("models", "Modelle"),
    ("hardware", "Hardware"),
    ("licenses", "Lizenzen"),
    ("settings", "Einstellungen"),
    ("logs", "Protokoll"),
)


class MainWindow(tk.Tk):
    def __init__(self, runtime) -> None:
        super().__init__()
        self.runtime = runtime
        self.palette = theme.palette_for(runtime.config.theme)
        theme.apply(self, self.palette)

        self.title(f"{__app_display_name__} {__version__}")
        self.geometry("1180x780")
        self.minsize(980, 640)

        self._events: queue.Queue[JobEvent] = queue.Queue()
        # Rückgaben aus run_async – tkinter darf nur aus dem Hauptthread
        # bedient werden, deshalb derselbe Weg wie bei den Auftragsereignissen.
        self._callbacks: queue.Queue = queue.Queue()
        self._pages: dict[str, ttk.Frame] = {}
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._active_page = ""
        self._tracked_job: str | None = None
        # Zuletzt erzeugte/bearbeitete Bilder – Vorlage für "weiterbearbeiten".
        self._last_images: list[Path] = []

        self._build_layout()
        self.runtime.queue.subscribe(self._on_job_event)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._drain_events)

        self.show_page("image")
        self._report_startup()
        # AGB-Abfrage nach dem Aufbau, damit das Hauptfenster dahinter steht.
        self.after(150, self._require_agb)
        # Die teuren Prüfungen (torch-Import, PowerShell-Abfragen) laufen erst
        # jetzt – der Start hat vorher bis zu 20 Sekunden darauf gewartet.
        self.after(400, self._start_background_checks)

    def _require_agb(self) -> None:
        """Beim ersten Start blockierend nach AGB-Zustimmung fragen."""
        if licensing.agb_accepted():
            return
        dialog = AgbDialog(self, self.palette, blocking=True)
        self.wait_window(dialog)
        if not licensing.agb_accepted():
            messagebox.showinfo(
                "Nutzung nicht möglich",
                "Ohne Zustimmung zu den AGB kann die Anwendung nicht genutzt werden. "
                "Sie wird jetzt beendet.",
            )
            self.runtime.queue.shutdown(wait=False)
            self.destroy()
            return
        self.log_view.append("AGB bestätigt.", "ok")

    def show_agb(self) -> None:
        """AGB jederzeit anzeigen (Knopf auf der Lizenzseite)."""
        dialog = AgbDialog(self, self.palette, blocking=False)
        self.wait_window(dialog)
        self._refresh_licenses()

    # ------------------------------------------------------------------
    # Aufbau
    # ------------------------------------------------------------------
    def _build_layout(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        # Kopfzeile
        header = ttk.Frame(self, style="Sidebar.TFrame", padding=(18, 12))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header, text=__app_display_name__, style="Surface.TLabel", font=theme.FONT_TITLE
        ).grid(row=0, column=0, sticky="w")
        self.header_info = ttk.Label(header, text="", style="SurfaceDim.TLabel")
        self.header_info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.backend_badge = ttk.Label(header, text="", style="Badge.TLabel")
        self.backend_badge.grid(row=0, column=2, sticky="e", padx=(8, 0))

        # Navigation
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(0, 10))
        sidebar.grid(row=1, column=0, sticky="nsw")
        for index, (key, label) in enumerate(PAGES):
            button = ttk.Button(
                sidebar,
                text=label,
                style="Nav.TButton",
                width=20,
                command=lambda k=key: self.show_page(k),
            )
            button.grid(row=index, column=0, sticky="ew", padx=6, pady=1)
            self._nav_buttons[key] = button

        # Inhalt
        self.content = ttk.Frame(self, padding=(18, 14))
        self.content.grid(row=1, column=1, sticky="nsew")
        self.content.columnconfigure(0, weight=1)
        self.content.rowconfigure(0, weight=1)

        # Fußzeile
        footer = ttk.Frame(self, style="Sidebar.TFrame", padding=(18, 10))
        footer.grid(row=2, column=0, columnspan=2, sticky="ew")
        footer.columnconfigure(1, weight=1)
        self.status_label = ttk.Label(footer, text="bereit", style="SurfaceDim.TLabel")
        self.status_label.grid(row=0, column=0, sticky="w")
        self.queue_badge = ttk.Label(footer, text="", style="Badge.TLabel")
        self.queue_badge.grid(row=0, column=3, sticky="e", padx=(10, 0))
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=1000)
        self.progress.grid(row=0, column=1, sticky="ew", padx=14)
        self.cancel_button = ttk.Button(
            footer,
            text="Abbrechen",
            style="Danger.TButton",
            command=self._cancel_tracked,
            state="disabled",
        )
        self.cancel_button.grid(row=0, column=2, sticky="e")

    def show_page(self, key: str) -> None:
        if key not in self._pages:
            builder = getattr(self, f"_build_{key}", None)
            if builder is None:
                return
            frame = builder()
            frame.grid(row=0, column=0, sticky="nsew")
            self._pages[key] = frame
        for name, button in self._nav_buttons.items():
            button.configure(style="NavActive.TButton" if name == key else "Nav.TButton")
        self._pages[key].tkraise()
        self._active_page = key
        refresher = getattr(self, f"_refresh_{key}", None)
        if refresher is not None:
            refresher()

    # ------------------------------------------------------------------
    # Startmeldungen
    # ------------------------------------------------------------------
    def _report_startup(self, quiet: bool = False) -> None:
        report = self.runtime.hardware
        best = report.best_gpu
        gpu_text = best.label() if best else "keine GPU erkannt"
        self.header_info.configure(
            text=f"{report.cpu.label()}  ·  {gpu_text}  ·  {report.advice.title}"
        )
        self.backend_badge.configure(text=self.runtime.plan.label)
        if quiet:
            # Nur die Anzeige nachziehen – die Meldungen standen schon beim Start
            # im Protokoll und würden sich sonst doppeln.
            return

        for note in self.runtime.config_notes:
            self.log_view.append(f"Konfiguration: {note}", "dim")
        self.log_view.append(self.runtime.plan.report(), "dim")
        for note in report.notes:
            self.log_view.append(f"Hardware: {note}", "warn")
        if self.runtime.plan.notes:
            for note in self.runtime.plan.notes:
                self.banner.show(note, "warn")
                self.log_view.append(f"Backend: {note}", "warn")
        if not compose.available():
            self.log_view.append(
                "ffmpeg fehlt – Video und Vertonung sind gesperrt, Bilder und Sprache laufen.",
                "warn",
            )

    def _start_background_checks(self) -> None:
        """Was beim Start zu teuer war, jetzt nachholen.

        Reihenfolge ist wichtig: erst die Hardware (die Backend-Wahl hängt
        davon ab), dann das Backend. Beides läuft im Hintergrund-Thread,
        die Anzeige wird über ``run_async`` im Hauptthread nachgezogen.
        """

        def after_backend(value, error) -> None:
            if error is not None:
                self.log_view.append(f"Backend-Prüfung: {accel.clean_error(error)}", "warn")
                return
            plan, changed = value
            self.backend_badge.configure(text=plan.label)
            if changed:
                self.log_view.append(f"Backend nach der Prüfung geändert: {plan.label}", "warn")
                self.log_view.append(plan.report(), "dim")
                for note in plan.notes:
                    self.banner.show(note, "warn")
            else:
                self.log_view.append(f"Backend bestätigt: {plan.label}", "dim")
            if self._active_page == "hardware":
                self._refresh_hardware()

        def after_hardware(value, error) -> None:
            if error is not None:
                self.log_view.append(f"Hardware-Erkennung: {accel.clean_error(error)}", "warn")
            else:
                self._report_startup(quiet=True)
                if self._active_page == "hardware":
                    self._refresh_hardware()
            self.run_async(self.runtime.refine_backend, after_backend)

        if accel.hardware_report_is_cached():
            self.run_async(self.runtime.refresh_hardware, after_hardware)
        else:
            self.run_async(self.runtime.refine_backend, after_backend)

    # ------------------------------------------------------------------
    # Seiten
    # ------------------------------------------------------------------
    def _stripe(self, tree: ttk.Treeview) -> None:
        """Abwechselnde Zeilenfarbe. Ohne sie verrutscht das Auge bei
        breiten Tabellen zwischen den Spalten."""
        tree.tag_configure("gerade", background=self.palette.surface)
        tree.tag_configure("ungerade", background=self.palette.surface_alt)
        for index, item in enumerate(tree.get_children()):
            vorhandene = [t for t in tree.item(item, "tags") if t not in ("gerade", "ungerade")]
            tree.item(item, tags=(*vorhandene, "gerade" if index % 2 == 0 else "ungerade"))

    def _page_frame(self, title: str, subtitle: str = "") -> tuple[ttk.Frame, ttk.Frame]:
        """Rahmen mit Titel und rollbarem Innenbereich."""
        outer = ttk.Frame(self.content)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(2, weight=1)
        ttk.Label(outer, text=title, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        if subtitle:
            ttk.Label(outer, text=subtitle, style="Dim.TLabel", wraplength=860).grid(
                row=1, column=0, sticky="w", pady=(2, 10)
            )
        area = ScrollArea(outer, self.palette)
        area.grid(row=2, column=0, sticky="nsew")
        area.inner.columnconfigure(0, weight=1)
        return outer, area.inner

    # --- Bild --------------------------------------------------------------
    def _build_image(self) -> ttk.Frame:
        config = self.runtime.config
        outer, body = self._page_frame(
            "Bild erzeugen",
            "Text zu Bild. Auflösung und Schritte bestimmen Dauer und Speicherbedarf.",
        )
        self.banner = Banner(body, self.palette)
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.banner.hide()

        card = Card(body, self.palette, "Auftrag")
        card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        form = card.body()
        self.image_prompt = TextRow(
            form,
            0,
            "Prompt",
            self.palette,
            value="",
            height=4,
            hint="Was soll zu sehen sein? Englisch liefert bei den "
            "meisten Modellen bessere Ergebnisse.",
        )
        self.image_negative = TextRow(
            form, 2, "Negativ-Prompt", self.palette, value=config.image_negative_prompt, height=2
        )
        self.image_width = SpinRow(form, 4, "Breite", config.image_width, 256, 4096, 64)
        self.image_height = SpinRow(form, 6, "Höhe", config.image_height, 256, 4096, 64)
        self.image_steps = SliderRow(form, 8, "Schritte", config.image_steps, 1, 100, integer=True)
        self.image_guidance = SliderRow(form, 10, "Führung (CFG)", config.image_guidance, 0, 20)
        self.image_sampler = ComboRow(
            form, 12, "Sampler", pipeline_image.SAMPLERS, config.image_sampler
        )
        self.image_batch = SpinRow(form, 14, "Anzahl Bilder", config.image_batch, 1, 16, 1)
        self.image_seed = SpinRow(
            form,
            16,
            "Seed",
            -1,
            -1,
            2**31 - 1,
            1,
            hint="-1 = zufällig. Gleicher Seed und gleiche Einstellungen ergeben dasselbe Bild.",
        )
        self.image_format = ComboRow(form, 18, "Dateiformat", IMAGE_FORMATS, config.image_format)

        actions = ttk.Frame(body)
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(
            actions, text="Bild erzeugen", style="Accent.TButton", command=self._submit_image
        ).grid(row=0, column=0)
        ttk.Button(actions, text="Ergebnis weiterbearbeiten", command=self._edit_last_result).grid(
            row=0, column=1, padx=8
        )
        ttk.Button(
            actions,
            text="Ausgabeordner öffnen",
            command=lambda: self._open_path(config.resolved_output_dir() / "images"),
        ).grid(row=0, column=2, padx=8)
        self.image_result = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.image_result.grid(row=3, column=0, sticky="w", pady=(10, 0))
        self.image_preview = ImagePreview(body, self.palette, 320, 220, "Noch kein Bild erzeugt")
        self.image_preview.grid(row=4, column=0, sticky="w", pady=(10, 0))
        return outer

    def _edit_last_result(self) -> None:
        """Zuletzt erzeugte Bilder auf die Bearbeiten-Seite übernehmen."""
        if not self._last_images:
            messagebox.showinfo(
                "Noch kein Ergebnis",
                "Erzeuge zuerst ein Bild – oder wähle auf der Seite "
                "'Bild bearbeiten' eine vorhandene Datei aus.",
            )
            return
        self.show_page("imageedit")
        self._set_edit_sources(self._last_images)

    def _submit_image(self) -> None:
        prompt = self.image_prompt.value()
        if not prompt:
            messagebox.showinfo("Prompt fehlt", "Bitte zuerst beschreiben, was zu sehen sein soll.")
            return
        config = self._config_with_ui()
        request = pipeline_image.ImageRequest.from_config(
            config,
            prompt,
            negative_prompt=self.image_negative.value(),
            width=int(self.image_width.value()),
            height=int(self.image_height.value()),
            steps=int(self.image_steps.value()),
            guidance=float(self.image_guidance.value()),
            sampler=self.image_sampler.value(),
            batch=int(self.image_batch.value()),
            seed=int(self.image_seed.value()),
            file_format=self.image_format.value(),
        )
        handler = pipeline_image.make_job(
            config, self.runtime.plan, request, force_dummy=self.runtime.force_dummy()
        )
        self._submit("image", f"Bild: {prompt[:40]}", handler)

    # --- Bild bearbeiten ---------------------------------------------------
    def _build_imageedit(self) -> ttk.Frame:
        config = self.runtime.config
        outer, body = self._page_frame(
            "Bild bearbeiten",
            "Vorhandene Bilder vergrößern, nach Prompt umarbeiten oder einen "
            "markierten Bereich ersetzen. Das Ausgangsbild bleibt unangetastet – "
            "es wird immer eine neue Datei geschrieben.",
        )
        self._edit_sources: list[Path] = []

        # --- Quellen -------------------------------------------------------
        source_card = Card(
            body,
            self.palette,
            "Ausgangsbilder",
            "Mehrere Dateien sind erlaubt – beim Vergrößern der Regelfall.",
        )
        source_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        source = source_card.body()
        source.columnconfigure(0, weight=1)
        self.edit_list = tk.Listbox(
            source,
            height=6,
            selectmode="extended",
            activestyle="none",
            background=self.palette.surface_alt,
            foreground=self.palette.text,
            selectbackground=self.palette.accent,
            selectforeground=self.palette.bg,
            highlightthickness=0,
            borderwidth=0,
            exportselection=False,
        )
        self.edit_list.grid(row=0, column=0, sticky="nsew")
        self.edit_list.bind("<<ListboxSelect>>", lambda _e: self._preview_edit_source())
        self.edit_preview = ImagePreview(source, self.palette, 260, 180, "Kein Bild gewählt")
        self.edit_preview.grid(row=0, column=1, sticky="ne", padx=(12, 0))

        source_buttons = ttk.Frame(source, style="Card.TFrame")
        source_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(
            source_buttons,
            text="Dateien wählen …",
            style="Accent.TButton",
            command=self._add_edit_sources,
        ).grid(row=0, column=0)
        ttk.Button(
            source_buttons,
            text="Zuletzt erzeugte übernehmen",
            command=lambda: self._set_edit_sources(self._last_images),
        ).grid(row=0, column=1, padx=6)
        ttk.Button(
            source_buttons, text="Auswahl entfernen", command=self._remove_edit_sources
        ).grid(row=0, column=2, padx=6)
        ttk.Button(
            source_buttons, text="Liste leeren", command=lambda: self._set_edit_sources([])
        ).grid(row=0, column=3, padx=6)

        # --- Bearbeitung ---------------------------------------------------
        edit_card = Card(body, self.palette, "Bearbeitung")
        edit_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        form = edit_card.body()
        self._edit_mode_labels = {
            label: key for key, label in pipeline_image.EDIT_MODE_LABELS.items()
        }
        self.edit_mode = ComboRow(
            form,
            0,
            "Was tun",
            list(self._edit_mode_labels),
            width=32,
            value=pipeline_image.EDIT_MODE_LABELS["upscale"],
            on_change=lambda _v: self._update_edit_mode(),
        )
        self.edit_prompt = TextRow(
            form,
            2,
            "Prompt",
            self.palette,
            height=3,
            hint="Beschreibt das Ziel. Beim reinen Vergrößern nicht nötig.",
        )
        self.edit_negative = TextRow(
            form, 4, "Negativ-Prompt", self.palette, value=config.image_negative_prompt, height=2
        )
        self.edit_strength = SliderRow(
            form,
            6,
            "Stärke",
            config.image_edit_strength,
            0.05,
            1.0,
            hint="Wie weit sich das Modell vom Ausgangsbild entfernen darf. "
            "0,3 = Feinschliff, 0,8 = kaum wiederzuerkennen.",
        )
        self.edit_steps = SliderRow(form, 8, "Schritte", config.image_steps, 1, 100, integer=True)
        self.edit_guidance = SliderRow(form, 10, "Führung (CFG)", config.image_guidance, 0, 20)
        self.edit_sampler = ComboRow(
            form, 12, "Sampler", pipeline_image.SAMPLERS, config.image_sampler
        )
        self.edit_seed = SpinRow(form, 14, "Seed", -1, -1, 2**31 - 1, 1, hint="-1 = zufällig.")
        self.edit_mask = PathRow(
            form,
            16,
            "Maske",
            "",
            hint="Nur für 'Bereich ersetzen': Weiß wird neu gerechnet, Schwarz "
            "bleibt. Die Maske wird auf die Bildgröße gebracht.",
            filetypes=[("Bilder", "*.png *.jpg *.jpeg *.webp")],
        )
        self.edit_mask_paint = ButtonRow(
            form,
            18,
            "Maske malen …",
            self._paint_mask,
            hint="Öffnet das erste gewählte Bild und lässt den Bereich direkt "
            "einzeichnen – kein zweites Programm nötig.",
        )
        self.edit_max_side = SpinRow(
            form,
            20,
            "Höchstkante vorher",
            0,
            0,
            8192,
            64,
            hint="Ausgangsbild vorher auf diese Kante bringen (0 = unverändert). "
            "Schützt bei großen Vorlagen vor vollem Grafikspeicher.",
        )
        self.edit_format = ComboRow(form, 22, "Dateiformat", IMAGE_FORMATS, config.image_format)

        # --- Vergrößern ----------------------------------------------------
        self.edit_up_card = Card(
            body,
            self.palette,
            "Vergrößern",
            "Real-ESRGAN rekonstruiert Kanten und Struktur. Fehlt das Modell, "
            "wird Lanczos benutzt – weicher, aber sofort und ohne Download.",
        )
        self.edit_up_card.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        up = self.edit_up_card.body()
        self.edit_factor = ComboRow(
            up, 0, "Faktor", [f"{value}x" for value in UPSCALE_FACTORS], f"{config.upscale_factor}x"
        )
        self.edit_use_model = CheckRow(
            up,
            2,
            "Real-ESRGAN benutzen",
            config.upscale_use_model,
            hint="Aus = nur Lanczos, kein Modell-Download, kein Grafikspeicher.",
        )
        self.edit_tile = SpinRow(
            up,
            4,
            "Kachelgröße",
            config.upscale_tile,
            0,
            2048,
            64,
            hint="Große Bilder werden kachelweise gerechnet. Kleiner = weniger "
            "Speicher, etwas langsamer. 0 = ohne Kacheln.",
        )
        self.edit_refine = CheckRow(
            up,
            6,
            "Danach mit dem Bildmodell nachschärfen",
            config.upscale_refine,
            hint="Braucht einen Prompt und viel Grafikspeicher – das vergrößerte "
            "Bild läuft noch einmal durch das Bildmodell.",
        )
        self.edit_refine.widget.configure(command=self._update_edit_mode)
        self.edit_refine_strength = SliderRow(
            up,
            8,
            "Stärke beim Nachschärfen",
            config.image_edit_refine_strength,
            0.05,
            0.6,
            hint="Über 0,4 erfindet das Modell neue Bildinhalte.",
        )

        # --- Einfärben -----------------------------------------------------
        self.edit_color_card = Card(
            body,
            self.palette,
            "Einfärben",
            "Das Bildmodell setzt die Farbe, die Helligkeit kommt aus der "
            "Vorlage zurück. Dadurch bleibt jedes Detail erhalten. Kein "
            "eigener Download – es rechnet das schon geladene Bildmodell.",
        )
        self.edit_color_card.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        color = self.edit_color_card.body()
        self.edit_keep_luminance = CheckRow(
            color,
            0,
            "Helligkeit aus der Vorlage behalten",
            config.image_colorize_keep_luminance,
            hint="Empfohlen. Aus = das Modell darf auch Kanten und Details "
            "ändern; das Ergebnis wird freier, aber ungenauer.",
        )

        # --- Diamond Painting ----------------------------------------------
        self.edit_diamond_card = Card(
            body,
            self.palette,
            "Diamond Painting",
            "Aus dem Bild wird ein Raster aus Steinen, jede Farbe bekommt ein "
            "Symbol. Es entstehen drei Dateien: Vorlage, Farbtafel und "
            "Farbliste zum Nachbestellen. Kein Modell nötig.",
        )
        self.edit_diamond_card.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        gem = self.edit_diamond_card.body()
        self.edit_diamond_stones = SpinRow(
            gem,
            0,
            "Breite in Steinen",
            config.diamond_stones,
            20,
            400,
            10,
            hint="Bestimmt die fertige Größe. 100 runde Steine ≈ 28 cm breit.",
        )
        self.edit_diamond_colors = SpinRow(
            gem,
            2,
            "Farben",
            config.diamond_colors,
            2,
            48,
            1,
            hint="Weniger Farben = weniger Sortieren, gröberes Bild.",
        )
        self.edit_diamond_shape = ComboRow(
            gem,
            4,
            "Steinform",
            [DIAMOND_SHAPE_LABELS[key] for key in DIAMOND_SHAPES],
            value=DIAMOND_SHAPE_LABELS.get(config.diamond_shape, "rund"),
        )
        self.edit_diamond_cell = SpinRow(
            gem,
            6,
            "Kästchengröße",
            config.diamond_cell_px,
            8,
            48,
            2,
            hint="Pixel je Stein in der Vorlage. Größer = besser lesbar "
            "beim Ausdrucken, aber größere Datei.",
        )
        self.edit_diamond_symbols = CheckRow(
            gem,
            8,
            "Symbole in die Kästchen zeichnen",
            config.diamond_symbols,
            hint="Empfohlen. Ausgedruckt sind zwei ähnliche Farben sonst "
            "nicht auseinanderzuhalten.",
        )
        self.edit_diamond_dmc = CheckRow(
            gem,
            10,
            "Farben auf DMC-Nummern abbilden",
            config.diamond_use_dmc,
            hint="Empfohlen. Steine werden nach DMC-Nummer bestellt, nicht "
            "nach Hexwert. Aus = Bildfarben, farbtreuer, aber nicht "
            "bestellbar. Es können weniger Farben herauskommen als "
            "eingestellt, wenn mehrere auf dieselbe Nummer fallen.",
        )

        actions = ttk.Frame(body)
        actions.grid(row=5, column=0, sticky="ew")
        ttk.Button(
            actions, text="Starten", style="Accent.TButton", command=self._submit_imageedit
        ).grid(row=0, column=0)
        ttk.Button(
            actions,
            text="Ausgabeordner öffnen",
            command=lambda: self._open_path(config.resolved_output_dir() / "images"),
        ).grid(row=0, column=1, padx=8)

        self.edit_hint = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.edit_hint.grid(row=6, column=0, sticky="w", pady=(10, 0))
        # Sagt vor dem Start, was herauskommt: Zielgröße, Rastermaße,
        # Steinzahl. Erspart den Durchlauf, um zu merken, dass die Zahlen
        # nicht passen.
        self.edit_estimate = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.edit_estimate.grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.edit_result = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.edit_result.grid(row=8, column=0, sticky="w", pady=(6, 0))
        self.edit_result_preview = ImagePreview(body, self.palette, 320, 220, "Noch kein Ergebnis")
        self.edit_result_preview.grid(row=9, column=0, sticky="w", pady=(8, 0))

        # Ändert sich eine Zahl, wird die Vorschau sofort neu gerechnet.
        for row in (
            self.edit_factor,
            self.edit_max_side,
            self.edit_diamond_stones,
            self.edit_diamond_colors,
            self.edit_diamond_shape,
            self.edit_diamond_cell,
        ):
            variable = getattr(row, "var", None)
            if variable is not None:
                variable.trace_add("write", lambda *_a: self._update_edit_estimate())

        self._update_edit_mode()
        return outer

    # --- Quellenliste ------------------------------------------------------
    def _set_edit_sources(self, files: Sequence[Path]) -> None:
        if not hasattr(self, "edit_list"):
            self.show_page("imageedit")
        self._edit_sources = [Path(item) for item in files]
        self._refresh_edit_list()

    def _add_edit_sources(self) -> None:
        pattern = " ".join(f"*{suffix}" for suffix in pipeline_image.IMAGE_SUFFIXES)
        chosen = filedialog.askopenfilenames(
            title="Bilder wählen", filetypes=[("Bilder", pattern), ("Alle Dateien", "*.*")]
        )
        if not chosen:
            return
        known = {str(item) for item in self._edit_sources}
        self._edit_sources.extend(Path(item) for item in chosen if item not in known)
        self._refresh_edit_list()

    def _remove_edit_sources(self) -> None:
        for index in sorted(self.edit_list.curselection(), reverse=True):
            if 0 <= index < len(self._edit_sources):
                self._edit_sources.pop(index)
        self._refresh_edit_list()

    def _refresh_edit_list(self) -> None:
        self.edit_list.delete(0, "end")
        for path in self._edit_sources:
            self.edit_list.insert("end", str(path))
        if self._edit_sources:
            self.edit_list.selection_set(0)
            self._preview_edit_source()
        else:
            self.edit_preview.clear()
        # Die Vorschau hängt am gewählten Bild – mitziehen.
        self._update_edit_estimate()

    def _preview_edit_source(self) -> None:
        selection = self.edit_list.curselection()
        index = selection[0] if selection else 0
        if 0 <= index < len(self._edit_sources):
            self.edit_preview.show(self._edit_sources[index])

    # --- Modus -------------------------------------------------------------
    def _edit_mode_key(self) -> str:
        return self._edit_mode_labels.get(self.edit_mode.value(), "img2img")

    def _diamond_shape_key(self) -> str:
        """Beschriftung der Steinform zurück auf den Schlüssel abbilden."""
        chosen = self.edit_diamond_shape.value()
        for key in DIAMOND_SHAPES:
            if DIAMOND_SHAPE_LABELS.get(key) == chosen:
                return key
        return "round"

    def _set_rows_state(self, rows: Sequence[Any], enabled: bool) -> None:
        """Eingabefelder sperren, ohne sie auszublenden.

        Bleibt für die Fälle, in denen ein Feld sichtbar sein soll, aber
        gerade nicht bedienbar – etwa die Stärke beim Nachschärfen, solange
        das Häkchen nicht gesetzt ist. Zum Ausblenden dient
        ``_set_rows_visible``.
        """
        for row in rows:
            widget = getattr(row, "widget", None)
            if widget is None:
                continue
            try:
                if isinstance(widget, ttk.Combobox):
                    widget.configure(state="readonly" if enabled else "disabled")
                else:
                    widget.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                continue

    def _set_rows_visible(self, rows: Sequence[Any], visible: bool) -> None:
        """Zeilen samt Beschriftung und Hinweis ein- oder ausblenden."""
        for row in rows:
            setter = getattr(row, "set_visible", None)
            if callable(setter):
                setter(visible)

    def _update_edit_mode(self) -> None:
        mode = self._edit_mode_key()
        is_upscale = mode == "upscale"
        is_colorize = mode == "colorize"
        is_diamond = mode == "diamond"
        is_inpaint = mode == "inpaint"
        # Diffusionsmodi teilen sich denselben Satz Regler. Vergrößern zeigt
        # sie nur, wenn nachgeschärft wird; die Vorlage kennt sie gar nicht.
        uses_model = mode in ("img2img", "inpaint", "colorize") or (
            is_upscale and self.edit_refine.value()
        )

        # --- Karten: nur die zur Aufgabe gehörende bleibt stehen -----------
        self.edit_up_card.set_visible(is_upscale)
        self.edit_color_card.set_visible(is_colorize)
        self.edit_diamond_card.set_visible(is_diamond)

        # --- Zeilen der Karte "Bearbeitung" --------------------------------
        self._set_rows_visible([self.edit_mask, self.edit_mask_paint], is_inpaint)
        self._set_rows_visible([self.edit_strength], mode in ("img2img", "inpaint", "colorize"))
        self._set_rows_visible(
            [
                self.edit_prompt,
                self.edit_negative,
                self.edit_steps,
                self.edit_guidance,
                self.edit_sampler,
                self.edit_seed,
            ],
            uses_model,
        )
        # Die Vorlage schreibt immer PNG – ein Formatwähler wäre eine Lüge.
        self._set_rows_visible([self.edit_format], not is_diamond)
        # Höchstkante schützt vor vollem Grafikspeicher; ohne Modell ist sie
        # nur eine Vorab-Verkleinerung, aber auch dort sinnvoll.
        self._set_rows_visible([self.edit_max_side], True)

        # --- Zeilen innerhalb der Vergrößern-Karte -------------------------
        self._set_rows_visible([self.edit_refine_strength], self.edit_refine.value())

        # Nachschärfen braucht einen Prompt – ohne bleibt das Feld sichtbar,
        # aber der Hinweis sagt warum.
        self._set_rows_state([self.edit_refine_strength], self.edit_refine.value())

        self._update_edit_estimate()

        texts = {
            "img2img": "Das ganze Bild wird nach dem Prompt neu gerechnet. Die Stärke "
            "entscheidet, wie viel vom Original übrig bleibt.",
            "inpaint": "Nur der weiße Bereich der Maske wird ersetzt, der Rest bleibt "
            "Pixel für Pixel erhalten. Eine Maske gilt für ein Bild.",
            "upscale": "Reine Vergrößerung – kein Prompt nötig. Erst mit "
            "'Nachschärfen' kommt das Bildmodell ins Spiel.",
            "colorize": "Schwarz-Weiß wird bunt. Prompt ist freiwillig – ohne Angabe "
            "wählt das Modell natürliche Farben, mit Angabe ('rotes Kleid, "
            "blauer Himmel') bestimmst du sie. Die Stärke entscheidet, wie "
            "kräftig gefärbt wird.",
            "diamond": "Bild zu Klebevorlage. Kein Prompt, kein Modell, kein "
            "Download – das rechnet die CPU in Sekunden. Je Bild entstehen "
            "Vorlage, Farbtafel und Farbliste mit DMC-Nummern zum Bestellen.",
        }
        self.edit_hint.configure(text=texts.get(mode, ""))

    def _paint_mask(self) -> None:
        """Maskeneditor auf dem ersten gewählten Bild öffnen.

        Inpainting rechnet ohnehin nur ein Bild je Maske – deshalb wird
        das erste genommen und das auch gesagt, statt stillschweigend eine
        Maske auf zwanzig Bilder anzuwenden.
        """
        if not self._edit_sources:
            messagebox.showinfo(
                "Kein Bild", "Bitte zuerst das Bild wählen, für das die Maske gilt."
            )
            return
        if len(self._edit_sources) > 1:
            messagebox.showinfo(
                "Mehrere Bilder",
                f"Eine Maske gilt für ein Bild. Gemalt wird auf {self._edit_sources[0].name}.",
            )

        from .mask_editor import paint_mask

        vorhanden = self.edit_mask.value()
        try:
            ziel = paint_mask(
                self,
                self.palette,
                self._edit_sources[0],
                Path(vorhanden) if vorhanden and Path(vorhanden).is_file() else None,
            )
        except Exception as exc:  # pragma: no cover – Oberfläche darf nie abstürzen
            log.exception("Maskeneditor fehlgeschlagen")
            messagebox.showerror("Maske", f"Editor nicht möglich: {accel.clean_error(exc)}")
            return
        if ziel is not None:
            self.edit_mask.var.set(str(ziel))
            self.log_view.append(f"Maske geschrieben: {ziel}", "ok")

    def _source_size(self) -> tuple[int, int] | None:
        """Maße des ersten gewählten Bildes. None, wenn nicht lesbar."""
        if not self._edit_sources:
            return None
        try:
            from PIL import Image

            with Image.open(self._edit_sources[0]) as image:
                return image.width, image.height
        except Exception:
            return None

    def _update_edit_estimate(self) -> None:
        """Vorschau auf das Ergebnis, bevor der Auftrag läuft.

        Rechnet mit dem ersten gewählten Bild. Ohne Auswahl oder bei einer
        unlesbaren Datei bleibt die Zeile leer – eine erfundene Zahl wäre
        schlimmer als keine.
        """
        if not hasattr(self, "edit_estimate"):
            return
        size = self._source_size()
        if size is None:
            self.edit_estimate.configure(
                text="Kein lesbares Bild gewählt – keine Vorschau möglich."
                if self._edit_sources
                else ""
            )
            return

        width, height = size
        count = len(self._edit_sources)
        mehrere = f" · {count} Bilder" if count > 1 else ""
        mode = self._edit_mode_key()

        try:
            limit = int(self.edit_max_side.value())
        except (tk.TclError, ValueError):
            limit = 0
        if limit > 0 and max(width, height) > limit:
            ratio = limit / float(max(width, height))
            width, height = max(1, int(width * ratio)), max(1, int(height * ratio))
            begrenzt = f" (vorher auf {limit} px begrenzt)"
        else:
            begrenzt = ""

        if mode == "diamond":
            self._estimate_diamond(width, height, mehrere)
            return

        if mode == "upscale":
            try:
                factor = int(self.edit_factor.value().rstrip("xX") or 2)
            except (tk.TclError, ValueError):
                factor = 2
            ziel = f"{width * factor}x{height * factor}"
            megapixel = (width * factor * height * factor) / 1_000_000
            self.edit_estimate.configure(
                text=f"Ergebnis: {ziel} px ({megapixel:.1f} MP){begrenzt}{mehrere}"
            )
            return

        # Diffusionsmodi rechnen auf ein Vielfaches von 8.
        snapped_w, snapped_h = (width // 8) * 8 or 8, (height // 8) * 8 or 8
        gerundet = "" if (snapped_w, snapped_h) == (width, height) else " (auf Vielfaches von 8)"
        self.edit_estimate.configure(
            text=f"Ergebnis: {snapped_w}x{snapped_h} px{gerundet}{begrenzt}{mehrere}"
        )

    def _estimate_diamond(self, width: int, height: int, mehrere: str) -> None:
        """Rastermaße, Steinzahl und fertige Größe in Zentimetern."""
        from .. import diamond

        try:
            stones = int(self.edit_diamond_stones.value())
            cell = int(self.edit_diamond_cell.value())
            farben = int(self.edit_diamond_colors.value())
        except (tk.TclError, ValueError):
            self.edit_estimate.configure(text="")
            return

        try:
            columns, rows = diamond.target_grid(width, height, stones)
        except diamond.DiamondError as exc:
            self.edit_estimate.configure(text=str(exc))
            return

        shape = self._diamond_shape_key()
        edge = diamond.STONE_SIZES_MM.get(shape, 2.8)
        breite_cm = columns * edge / 10
        hoehe_cm = rows * edge / 10
        # Die Vorlage bekommt einen Rand für die Zeilennummern – grob, aber
        # nah genug, um eine 12000-Pixel-Datei vorher zu erkennen.
        rand = max(24, cell + 6)
        datei_px = f"{rand + columns * cell}x{rand + rows * cell}"
        self.edit_estimate.configure(
            text=(
                f"Raster: {columns}x{rows} Steine = {columns * rows} Stück · "
                f"fertig {breite_cm:.1f} x {hoehe_cm:.1f} cm "
                f"({diamond.SHAPE_LABELS.get(shape, shape)}, {edge:.1f} mm) · "
                f"bis zu {farben} Farben · Vorlage {datei_px} px{mehrere}"
            )
        )

    def _refresh_imageedit(self) -> None:
        if hasattr(self, "edit_mode"):
            self._update_edit_mode()

    def _submit_imageedit(self) -> None:
        if not self._edit_sources:
            messagebox.showinfo("Kein Bild", "Bitte zuerst mindestens eine Datei wählen.")
            return
        mode = self._edit_mode_key()
        config = self._config_with_ui()
        mask = self.edit_mask.value()
        factor = int(self.edit_factor.value().rstrip("xX") or 2)

        request = pipeline_image.EditRequest.from_config(
            config,
            self._edit_sources,
            mode=mode,
            prompt=self.edit_prompt.value(),
            negative_prompt=self.edit_negative.value(),
            mask=Path(mask) if mask and mode == "inpaint" else None,
            strength=float(self.edit_strength.value()),
            steps=int(self.edit_steps.value()),
            guidance=float(self.edit_guidance.value()),
            sampler=self.edit_sampler.value(),
            seed=int(self.edit_seed.value()),
            factor=factor,
            use_model=self.edit_use_model.value(),
            tile=int(self.edit_tile.value()),
            refine=self.edit_refine.value(),
            refine_strength=float(self.edit_refine_strength.value()),
            # Einfärben benutzt denselben Regler wie das Umarbeiten – zwei
            # Stärkeregler nebeneinander würden nur verwirren.
            colorize_strength=float(self.edit_strength.value()),
            keep_luminance=self.edit_keep_luminance.value(),
            diamond_stones=int(self.edit_diamond_stones.value()),
            diamond_colors=int(self.edit_diamond_colors.value()),
            diamond_cell_px=int(self.edit_diamond_cell.value()),
            diamond_shape=self._diamond_shape_key(),
            diamond_symbols=self.edit_diamond_symbols.value(),
            diamond_use_dmc=self.edit_diamond_dmc.value(),
            max_side=int(self.edit_max_side.value()),
            file_format=self.edit_format.value(),
        )
        problems = request.validated()
        if problems:
            messagebox.showinfo("Angaben fehlen", "\n".join(problems))
            return

        handler = pipeline_image.make_edit_job(
            config, self.runtime.plan, request, force_dummy=self.runtime.force_dummy()
        )
        label = pipeline_image.EDIT_MODE_LABELS.get(mode, mode)
        title = f"{label}: {self._edit_sources[0].name}"
        if len(self._edit_sources) > 1:
            title += f" (+{len(self._edit_sources) - 1})"
        self._submit("edit", title, handler)

    # --- Video -------------------------------------------------------------
    def _build_video(self) -> ttk.Frame:
        config = self.runtime.config
        outer, body = self._page_frame(
            "Video erzeugen",
            "Text oder Startbild zu Video. Dauer = Bilder ÷ Bildrate. "
            "Video braucht deutlich mehr VRAM als Bild.",
        )
        card = Card(body, self.palette, "Auftrag")
        card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        form = card.body()
        self.video_prompt = TextRow(form, 0, "Prompt", self.palette, height=3)
        self.video_init = PathRow(
            form,
            2,
            "Startbild (optional)",
            "",
            hint="Leer = Text zu Video. Gesetzt = Bild wird animiert.",
            filetypes=[("Bilder", "*.png *.jpg *.jpeg *.webp")],
        )
        self.video_width = SpinRow(form, 4, "Breite", config.video_width, 256, 1920, 32)
        self.video_height = SpinRow(form, 6, "Höhe", config.video_height, 256, 1088, 32)
        self.video_frames = SliderRow(form, 8, "Bilder", config.video_frames, 8, 241, integer=True)
        self.video_fps = SpinRow(form, 10, "Bildrate", config.video_fps, 4, 60, 1)
        self.video_steps = SliderRow(form, 12, "Schritte", config.video_steps, 1, 80, integer=True)
        self.video_motion = SliderRow(form, 14, "Bewegungsstärke", config.video_motion, 0.1, 3.0)
        self.video_container = ComboRow(
            form, 16, "Container", VIDEO_CONTAINERS, config.video_container
        )
        self.video_audio = PathRow(
            form,
            18,
            "Tonspur (optional)",
            "",
            hint="WAV/MP3 wird direkt mit eingebettet.",
            filetypes=[("Ton", "*.wav *.mp3 *.flac *.m4a")],
        )
        self.video_keep_frames = CheckRow(form, 20, "Einzelbilder behalten", False)

        actions = ttk.Frame(body)
        actions.grid(row=1, column=0, sticky="ew")
        ttk.Button(
            actions, text="Video erzeugen", style="Accent.TButton", command=self._submit_video
        ).grid(row=0, column=0)
        ttk.Button(
            actions,
            text="Ausgabeordner öffnen",
            command=lambda: self._open_path(config.resolved_output_dir() / "videos"),
        ).grid(row=0, column=1, padx=8)
        self.video_hint = ttk.Label(body, text="", style="Warn.TLabel", wraplength=860)
        self.video_hint.grid(row=2, column=0, sticky="w", pady=(10, 0))
        return outer

    def _refresh_video(self) -> None:
        advice = self.runtime.hardware.advice
        messages: list[str] = []
        if not advice.video_ok:
            messages.append(
                f"{advice.title}: Video ist auf dieser Hardware nicht sinnvoll. {advice.text}"
            )
        if not compose.available():
            messages.append("ffmpeg fehlt – es werden nur Einzelbilder geschrieben.")
        self.video_hint.configure(text="\n".join(messages))

    def _submit_video(self) -> None:
        prompt = self.video_prompt.value()
        if not prompt and not self.video_init.value():
            messagebox.showinfo("Eingabe fehlt", "Prompt oder Startbild angeben.")
            return
        config = self._config_with_ui()
        init = self.video_init.value()
        audio = self.video_audio.value()
        request = pipeline_video.VideoRequest.from_config(
            config,
            prompt,
            init_image=Path(init) if init else None,
            width=int(self.video_width.value()),
            height=int(self.video_height.value()),
            frames=int(self.video_frames.value()),
            fps=int(self.video_fps.value()),
            steps=int(self.video_steps.value()),
            motion=float(self.video_motion.value()),
            container=self.video_container.value(),
            audio_file=Path(audio) if audio else None,
            keep_frames=self.video_keep_frames.value(),
        )
        handler = pipeline_video.make_job(
            config, self.runtime.plan, request, force_dummy=self.runtime.force_dummy()
        )
        self._submit("video", f"Video: {prompt[:40] or 'Startbild'}", handler)

    # --- Stimme ------------------------------------------------------------
    def _build_voice(self) -> ttk.Frame:
        config = self.runtime.config
        outer, body = self._page_frame(
            "Sprache erzeugen",
            "Text zu Sprache. Angelernte Stimmen erscheinen in der Auswahl, "
            "sobald ein Profil mit Einwilligung vorliegt.",
        )
        card = Card(body, self.palette, "Auftrag")
        card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        form = card.body()
        self.voice_text = TextRow(form, 0, "Text", self.palette, height=6)
        self.voice_profile = ComboRow(form, 2, "Stimme", ["Standardstimme"], "Standardstimme")
        # Bark kennt feste Sprecher-Vorgaben; bei Piper stehen hier die
        # Stimmnamen (z. B. de_DE-thorsten-medium).
        speakers = (
            [f"v2/de_speaker_{i}" for i in range(10)]
            + [f"v2/en_speaker_{i}" for i in range(4)]
            + ["de_DE-thorsten-medium"]
        )
        if config.voice_speaker not in speakers:
            speakers.insert(0, config.voice_speaker)
        self.voice_speaker = ComboRow(
            form,
            4,
            "Sprecher",
            speakers,
            config.voice_speaker,
            hint="Bark: v2/de_speaker_N. Piper: Stimmname.",
        )
        self.voice_speed = SliderRow(form, 6, "Geschwindigkeit", config.voice_speed, 0.5, 2.0)
        self.voice_pitch = SliderRow(form, 8, "Tonhöhe", config.voice_pitch, -12, 12, unit=" HT")
        self.voice_volume = SliderRow(
            form, 10, "Lautstärke", config.voice_volume_db, -20, 6, unit=" dB"
        )
        self.voice_split = CheckRow(
            form,
            12,
            "Satzweise erzeugen",
            config.voice_split_sentences,
            hint="Bei langen Texten stabiler und besser abbrechbar.",
        )

        actions = ttk.Frame(body)
        actions.grid(row=1, column=0, sticky="ew")
        ttk.Button(
            actions, text="Sprache erzeugen", style="Accent.TButton", command=self._submit_voice
        ).grid(row=0, column=0)
        ttk.Button(actions, text="Mit Video vertonen …", command=self._mux_dialog).grid(
            row=0, column=1, padx=8
        )
        ttk.Button(
            actions,
            text="Ausgabeordner öffnen",
            command=lambda: self._open_path(config.resolved_output_dir() / "audio"),
        ).grid(row=0, column=2, padx=8)
        return outer

    def _refresh_voice(self) -> None:
        names = ["Standardstimme"]
        self._voice_profile_map: dict[str, str] = {}
        for profile in voice_profiles.list_profiles():
            usable, _ = profile.usable_for_synthesis()
            label = f"{profile.display_name}" + ("" if usable else " (gesperrt)")
            names.append(label)
            self._voice_profile_map[label] = profile.slug if usable else ""
        self.voice_profile.set_values(names)

    def _submit_voice(self) -> None:
        text = self.voice_text.value()
        if not text:
            messagebox.showinfo("Text fehlt", "Bitte Text zum Sprechen eingeben.")
            return
        config = self._config_with_ui()
        chosen = self.voice_profile.value()
        slug = getattr(self, "_voice_profile_map", {}).get(chosen, "")
        if slug:
            config = config.with_values(voice_cloning_enabled=True, voice_profile=slug)
        request = pipeline_voice.VoiceRequest.from_config(
            config,
            text,
            profile_slug=slug,
            speaker=self.voice_speaker.value(),
            speed=float(self.voice_speed.value()),
            pitch=float(self.voice_pitch.value()),
            volume_db=float(self.voice_volume.value()),
            split_sentences=self.voice_split.value(),
        )
        handler = pipeline_voice.make_job(
            config, self.runtime.plan, request, force_dummy=self.runtime.force_dummy()
        )
        self._submit("voice", f"Sprache: {text[:40]}", handler)

    def _mux_dialog(self) -> None:
        if not compose.available():
            messagebox.showwarning(
                "ffmpeg fehlt",
                "Zum Vertonen wird ffmpeg gebraucht. Lege einen LGPL-Build nach "
                f"{paths.tools_dir() / 'ffmpeg'}.",
            )
            return
        video = filedialog.askopenfilename(
            title="Video wählen", filetypes=[("Video", "*.mp4 *.webm *.mov")]
        )
        if not video:
            return
        audio = filedialog.askopenfilename(
            title="Tonspur wählen", filetypes=[("Ton", "*.wav *.mp3 *.flac *.m4a")]
        )
        if not audio:
            return
        config = self.runtime.config
        target = (
            config.resolved_output_dir()
            / "videos"
            / (f"{Path(video).stem}_vertont{Path(video).suffix}")
        )

        def handler(context) -> Path:
            return compose.mux(
                Path(video),
                Path(audio),
                target,
                audio_codec=config.mux_audio_codec,
                audio_bitrate=config.mux_audio_bitrate,
                normalize=config.mux_normalize_audio,
                loop_audio=config.mux_loop_audio,
                context=context,
            )

        self._submit("compose", f"Vertonen: {Path(video).name}", handler)

    # --- Stimme anlernen ---------------------------------------------------
    def _build_voicetrain(self) -> ttk.Frame:
        outer, body = self._page_frame(
            "Stimme anlernen",
            "Aufnahmen sammeln, Einwilligung dokumentieren, Profil anlernen. "
            "Ohne Einwilligungs-Nachweis bleibt ein Profil gesperrt.",
        )
        gate_card = Card(body, self.palette, "Freigabe")
        gate_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        gate_body = gate_card.body()
        self.voicetrain_gate = ttk.Label(
            gate_body, text="", style="SurfaceDim.TLabel", wraplength=820
        )
        self.voicetrain_gate.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(
            gate_body, text="Lizenzseite öffnen", command=lambda: self.show_page("licenses")
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        # Klonstimmen brauchen eine getrennte Laufzeit (siehe voice_runtime).
        runtime_card = Card(
            body,
            self.palette,
            "Laufzeit für Klonstimmen",
            "Chatterbox läuft in einer eigenen Umgebung, weil es ältere Fassungen "
            "von torch und diffusers verlangt und sonst die GPU-Beschleunigung "
            "für Bild und Video zerstören würde.",
        )
        runtime_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        runtime_body = runtime_card.body()
        self.voice_runtime_state = ttk.Label(
            runtime_body, text="", style="SurfaceDim.TLabel", wraplength=800
        )
        self.voice_runtime_state.grid(row=0, column=0, sticky="w")
        runtime_buttons = ttk.Frame(runtime_body, style="Card.TFrame")
        runtime_buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(
            runtime_buttons, text="Laufzeit einrichten", command=self._install_voice_runtime
        ).grid(row=0, column=0)
        ttk.Button(
            runtime_buttons,
            text="Erneut prüfen",
            command=lambda: self._check_voice_runtime(force=True),
        ).grid(row=0, column=1, padx=6)

        list_card = Card(body, self.palette, "Profile")
        list_card.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        list_body = list_card.body()
        self.profile_tree = ttk.Treeview(
            list_body,
            columns=("state", "mode", "material", "speaker"),
            show="tree headings",
            height=8,
        )
        for column, title, width in (
            ("#0", "Profil", 200),
            ("state", "Zustand", 140),
            ("mode", "Verfahren", 100),
            ("material", "Material", 90),
            ("speaker", "Einwilligung von", 200),
        ):
            self.profile_tree.heading(column, text=title)
            self.profile_tree.column(column, width=width, anchor="w")
        self.profile_tree.grid(row=0, column=0, columnspan=2, sticky="nsew")
        list_body.columnconfigure(0, weight=1)

        buttons = ttk.Frame(list_body, style="Card.TFrame")
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(
            buttons,
            text="Neues Profil …",
            style="Accent.TButton",
            command=self._create_profile_dialog,
        ).grid(row=0, column=0)
        ttk.Button(buttons, text="Aufnahme hinzufügen …", command=self._add_sample).grid(
            row=0, column=1, padx=6
        )
        ttk.Button(buttons, text="Anlernen", command=self._train_profile).grid(
            row=0, column=2, padx=6
        )
        ttk.Button(buttons, text="Ordner öffnen", command=self._open_profile_dir).grid(
            row=0, column=3, padx=6
        )
        ttk.Button(buttons, text="Auf Referenz umstellen", command=self._switch_to_zero_shot).grid(
            row=0, column=4, padx=6
        )
        ttk.Button(
            buttons, text="Löschen / Widerruf", style="Danger.TButton", command=self._delete_profile
        ).grid(row=0, column=5, padx=6)

        # --- Aufnahmen der ausgewählten Stimme -----------------------------
        sample_card = Card(
            body,
            self.palette,
            "Aufnahmen",
            "Mehr Material verbessert die Stimme: verschiedene Sätze decken mehr "
            "Laute und Tonhöhen ab als eine einzelne lange Aufnahme.",
        )
        sample_card.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        sample_body = sample_card.body()
        sample_body.columnconfigure(0, weight=1)
        self.sample_tree = ttk.Treeview(
            sample_body, columns=("dauer", "rate", "zustand"), show="tree headings", height=6
        )
        for spalte, titel, breite in (
            ("#0", "Datei", 320),
            ("dauer", "Dauer", 90),
            ("rate", "Abtastrate", 110),
            ("zustand", "Zustand", 320),
        ):
            self.sample_tree.heading(spalte, text=titel)
            self.sample_tree.column(spalte, width=breite, anchor="w")
        self.sample_tree.grid(row=0, column=0, columnspan=2, sticky="nsew")
        self.sample_tree.tag_configure("schlecht", foreground=self.palette.warn)

        sample_buttons = ttk.Frame(sample_body, style="Card.TFrame")
        sample_buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(
            sample_buttons,
            text="Aufnahmen hinzufügen …",
            style="Accent.TButton",
            command=self._add_sample,
        ).grid(row=0, column=0)
        ttk.Button(
            sample_buttons,
            text="Aufnahme entfernen",
            style="Danger.TButton",
            command=self._remove_sample,
        ).grid(row=0, column=1, padx=6)
        ttk.Button(
            sample_buttons, text="Aufnahmen-Ordner öffnen", command=self._open_samples_dir
        ).grid(row=0, column=2, padx=6)
        self.sample_summary = ttk.Label(sample_body, text="", style="Hint.TLabel", wraplength=780)
        self.sample_summary.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # --- Feinschliff der ausgewählten Stimme ---------------------------
        tune_card = Card(
            body,
            self.palette,
            "Stimme verfeinern",
            "Gilt je Profil und wird gespeichert – eine einmal gut eingestellte "
            "Stimme klingt danach immer gleich.",
        )
        tune_card.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        tune = tune_card.body()
        self.tune_exaggeration = SliderRow(
            tune,
            0,
            "Ausdruck",
            0.5,
            0.25,
            1.5,
            hint="Niedrig: ruhig und sachlich. Hoch: betont, mehr Melodie.",
        )
        self.tune_cfg = SliderRow(
            tune,
            2,
            "Führung",
            0.5,
            0.1,
            1.0,
            hint="Niedrig hält sich näher an Tempo und Rhythmus der Referenz.",
        )
        self.tune_temperature = SliderRow(
            tune,
            4,
            "Streuung",
            0.8,
            0.3,
            1.3,
            hint="Niedrig: gleichmäßig und vorhersagbar. Hoch: lebendiger, aber unruhiger.",
        )
        self.tune_reference = SliderRow(
            tune,
            6,
            "Referenzlänge",
            20,
            10,
            30,
            integer=True,
            unit=" s",
            hint="Wie viel Material in die Referenz fließt. Mehrere Aufnahmen "
            "werden zusammengesetzt.",
        )

        tune_buttons = ttk.Frame(tune, style="Card.TFrame")
        tune_buttons.grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(
            tune_buttons,
            text="Speichern und verfeinern",
            style="Accent.TButton",
            command=self._refine_profile,
        ).grid(row=0, column=0)
        ttk.Button(tune_buttons, text="Hörprobe erzeugen", command=self._preview_profile).grid(
            row=0, column=1, padx=6
        )
        ttk.Button(tune_buttons, text="Auf Vorgaben zurück", command=self._reset_tuning).grid(
            row=0, column=2, padx=6
        )
        self.tune_hint = ttk.Label(tune, text="", style="Hint.TLabel", wraplength=780)
        self.tune_hint.grid(row=9, column=0, columnspan=2, sticky="w", pady=(8, 0))

        self.voicetrain_detail = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.voicetrain_detail.grid(row=5, column=0, sticky="w")
        self.profile_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_profile_detail())
        return outer

    # --- Aufnahmen ---------------------------------------------------------
    def _refresh_sample_list(self, profile) -> None:
        """Aufnahmen des Profils anzeigen – mit Grund, falls unbrauchbar."""
        if not hasattr(self, "sample_tree"):
            return
        self.sample_tree.delete(*self.sample_tree.get_children())
        if profile is None:
            self.sample_summary.configure(text="Kein Profil ausgewählt.")
            return

        proben = profile.samples()
        brauchbar = 0
        gesamt = 0.0
        for probe in proben:
            zustand = "brauchbar" if probe.usable else (probe.note or "nicht brauchbar")
            if probe.usable:
                brauchbar += 1
                gesamt += probe.seconds
            self.sample_tree.insert(
                "",
                "end",
                iid=probe.path.name,
                text=probe.path.name,
                values=(
                    f"{probe.seconds:.1f} s" if probe.seconds else "?",
                    f"{probe.sample_rate} Hz" if probe.sample_rate else "?",
                    zustand,
                ),
                tags=() if probe.usable else ("schlecht",),
            )
        self._stripe(self.sample_tree)

        ziel = max(voice_profiles.MIN_TOTAL_SECONDS_CLONE, profile.reference_seconds)
        if not proben:
            text = (
                "Noch keine Aufnahme. Ab etwa 10 Sekunden sauberer Sprache "
                "lässt sich die Stimme nachbilden."
            )
        elif gesamt >= ziel:
            text = (
                f"{brauchbar} brauchbare Aufnahme(n), {gesamt:.0f} s Material – "
                f"reicht für die Referenzlänge von {ziel:.0f} s."
            )
        else:
            text = (
                f"{brauchbar} brauchbare Aufnahme(n), {gesamt:.0f} s Material. "
                f"Für die eingestellte Referenzlänge von {ziel:.0f} s fehlen noch "
                f"{ziel - gesamt:.0f} s – weitere Aufnahmen verbessern das Ergebnis."
            )
        if not profile.mode.available:
            text = (
                "Dieses Profil steht auf 'Nachtrainieren' – das ist nicht umgesetzt "
                "und blockiert das Anlernen. Mit 'Auf Referenz umstellen' wird es "
                "sofort nutzbar.\n" + text
            )
        self.sample_summary.configure(
            text=text, style="SurfaceWarn.TLabel" if not profile.mode.available else "Hint.TLabel"
        )

    def _remove_sample(self) -> None:
        profile = self._selected_profile()
        auswahl = self.sample_tree.selection() if hasattr(self, "sample_tree") else ()
        if profile is None or not auswahl:
            messagebox.showinfo("Nichts ausgewählt", "Zuerst eine Aufnahme in der Liste wählen.")
            return
        name = auswahl[0]
        if not messagebox.askyesno("Aufnahme entfernen", f"'{name}' aus dem Profil löschen?"):
            return
        if voice_profiles.remove_sample(profile, name):
            self.log_view.append(f"Aufnahme entfernt: {name}", "warn")
        self._refresh_voicetrain()

    def _open_samples_dir(self) -> None:
        profile = self._selected_profile()
        if profile is not None:
            self._open_path(profile.samples_dir)

    # --- Feinschliff -------------------------------------------------------
    def _load_tuning(self, profile) -> None:
        """Regler auf die Werte des Profils setzen."""
        if not hasattr(self, "tune_exaggeration"):
            return
        self.tune_exaggeration.var.set(profile.exaggeration)
        self.tune_cfg.var.set(profile.cfg_weight)
        self.tune_temperature.var.set(profile.temperature)
        self.tune_reference.var.set(profile.reference_seconds)
        for row in (
            self.tune_exaggeration,
            self.tune_cfg,
            self.tune_temperature,
            self.tune_reference,
        ):
            row._on_move("")  # Anzeige nachziehen
        self.tune_hint.configure(text=f"Eingestellt für '{profile.display_name}'.")

    def _reset_tuning(self) -> None:
        for row, value in (
            (self.tune_exaggeration, 0.5),
            (self.tune_cfg, 0.5),
            (self.tune_temperature, 0.8),
            (self.tune_reference, 20),
        ):
            row.var.set(value)
            row._on_move("")
        self.tune_hint.configure(text="Vorgaben gesetzt – zum Übernehmen speichern.")

    def _apply_tuning(self, profile) -> bool:
        """Reglerwerte ins Profil schreiben. True, wenn sich etwas geändert hat."""
        neu = (
            round(float(self.tune_exaggeration.value()), 3),
            round(float(self.tune_cfg.value()), 3),
            round(float(self.tune_temperature.value()), 3),
            float(self.tune_reference.value()),
        )
        alt = (
            profile.exaggeration,
            profile.cfg_weight,
            profile.temperature,
            profile.reference_seconds,
        )
        profile.exaggeration, profile.cfg_weight, profile.temperature, profile.reference_seconds = (
            neu
        )
        profile.save()
        return neu != alt

    def _refine_profile(self) -> None:
        """Werte speichern und die Referenz neu aufbereiten."""
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("Kein Profil", "Zuerst ein Profil auswählen.")
            return
        reference_changed = (
            abs(float(self.tune_reference.value()) - profile.reference_seconds) > 0.5
        )
        self._apply_tuning(profile)
        self.log_view.append(
            f"{profile.display_name}: Ausdruck {profile.exaggeration:.2f}, "
            f"Führung {profile.cfg_weight:.2f}, Streuung {profile.temperature:.2f}",
            "ok",
        )

        if not reference_changed and profile.artifact_path is not None:
            self.tune_hint.configure(
                text="Gespeichert. Die Referenz blieb unverändert – nur die Regler wirken."
            )
            self._show_profile_detail()
            return

        # Referenzlänge geändert oder noch nie aufbereitet: neu bauen.
        handler = pipeline_voice.make_training_job(
            self.runtime.config, self.runtime.plan, profile.slug
        )
        self._submit("train", f"Referenz neu aufbereiten: {profile.display_name}", handler)
        self.tune_hint.configure(text="Referenz wird neu aufbereitet …")

    def _preview_profile(self) -> None:
        """Kurze Hörprobe mit den aktuellen Reglern."""
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("Kein Profil", "Zuerst ein Profil auswählen.")
            return
        usable, reason = profile.usable_for_synthesis()
        if not usable:
            messagebox.showwarning("Nicht nutzbar", reason)
            return
        self._apply_tuning(profile)

        config = self.runtime.config.with_values(
            voice_cloning_enabled=True, voice_profile=profile.slug
        )
        request = pipeline_voice.VoiceRequest.from_config(
            config,
            "Dies ist eine kurze Hörprobe der angelernten Stimme.",
            profile_slug=profile.slug,
            split_sentences=False,
            name_hint=f"probe-{profile.slug}",
        )
        handler = pipeline_voice.make_job(config, self.runtime.plan, request)
        self._submit("voice", f"Hörprobe: {profile.display_name}", handler)
        self.tune_hint.configure(
            text="Hörprobe läuft. Beim ersten Mal lädt das Modell – das dauert."
        )

    def run_async(
        self, work: Callable[[], Any], done: Callable[[Any, BaseException | None], None]
    ) -> None:
        """Kurze Arbeit im Hintergrund, Ergebnis zurück im Oberflächen-Thread.

        Das Ergebnis geht über dieselbe Warteschlange wie die Auftrags-
        ereignisse und wird in ``_drain_events`` abgeholt. ``after()`` aus
        einem Fremd-Thread aufzurufen ist bei tkinter NICHT threadsicher –
        es endet je nach Zeitpunkt mit "main thread is not in main loop".
        """
        import threading

        def runner() -> None:
            try:
                value, error = work(), None
            except BaseException as exc:
                value, error = None, exc
            self._callbacks.put((done, value, error))

        threading.Thread(target=runner, daemon=True).start()

    def _set_runtime_state(self, ok: bool | None, note: str) -> None:
        if not hasattr(self, "voice_runtime_state"):
            return
        if ok is None:
            self.voice_runtime_state.configure(text=note, style="SurfaceDim.TLabel")
            return
        text = (
            ("Bereit – " + note)
            if ok
            else (
                note + "\n\nOhne sie wird beim Erzeugen die Standardstimme verwendet, "
                "kein Platzhalterton."
            )
        )
        self.voice_runtime_state.configure(
            text=text, style="SurfaceOk.TLabel" if ok else "SurfaceWarn.TLabel"
        )

    def _check_voice_runtime(self, force: bool = False) -> None:
        """Laufzeit prüfen – niemals im Oberflächen-Thread.

        Vorher lief hier eine volle Prüfung: die importiert im Unterprozess
        torch und chatterbox und brauchte über eine Minute – solange stand
        das Fenster. Jetzt kommt zuerst der gespeicherte Zustand (sofort),
        die Nachprüfung läuft nebenher.
        """
        from .. import voice_runtime

        known = voice_runtime.cached_state()
        if known is not None and not force:
            self._set_runtime_state(known[0], known[1])
        else:
            self._set_runtime_state(None, "Laufzeit wird geprüft …")

        def done(value, error) -> None:
            if error is not None:
                self._set_runtime_state(False, accel.clean_error(error))
                return
            self._set_runtime_state(value[0], value[1])

        self.run_async(lambda: voice_runtime.available(refresh=True), done)

    def _remember_selection(self) -> str:
        """Aktuell gewähltes Profil merken.

        Jedes Auffrischen leert den Baum und füllt ihn neu – ohne dieses
        Merken wäre die Auswahl danach weg, und der nächste Knopfdruck
        meldete "zuerst ein Profil auswählen", obwohl gerade eines offen war.
        """
        auswahl = self.profile_tree.selection()
        if auswahl:
            self._last_profile = auswahl[0]
        return getattr(self, "_last_profile", "")

    def _restore_selection(self) -> None:
        gewuenscht = getattr(self, "_last_profile", "")
        vorhanden = self.profile_tree.get_children()
        if not vorhanden:
            return
        ziel = gewuenscht if gewuenscht in vorhanden else vorhanden[0]
        self.profile_tree.selection_set(ziel)
        self.profile_tree.focus(ziel)
        self._last_profile = ziel
        self._show_profile_detail()

    def _refresh_voicetrain(self) -> None:
        self._remember_selection()
        gate = licensing.gate("voice-cloning")
        self.voicetrain_gate.configure(
            text=(
                "Freigegeben. Für jede angelernte Stimme muss eine Einwilligung der "
                "sprechenden Person vorliegen."
                if gate.allowed
                else gate.reason
            )
        )
        self._check_voice_runtime()

        self.profile_tree.delete(*self.profile_tree.get_children())
        for profile in voice_profiles.list_profiles():
            speaker = profile.consent.speaker_name if profile.consent else "kein Nachweis"
            self.profile_tree.insert(
                "",
                "end",
                iid=profile.slug,
                text=profile.display_name,
                values=(
                    profile.state.label(),
                    profile.mode.label(),
                    f"{profile.total_seconds():.0f}s",
                    speaker,
                ),
            )
        self._stripe(self.profile_tree)
        self._restore_selection()

    def _install_voice_runtime(self) -> None:
        """Klon-Laufzeit im Hintergrund einrichten (mehrere GB Download)."""
        from .. import voice_runtime

        if paths.is_frozen():
            messagebox.showinfo(
                "Mitgeliefert",
                "In der ausgelieferten Fassung gehört die Klon-Laufzeit zum Lieferumfang. "
                "Fehlt sie, bitte beim Anbieter melden – ein Nachinstallieren würde "
                "Python auf diesem Rechner voraussetzen.",
            )
            return
        if not messagebox.askyesno(
            "Laufzeit einrichten",
            "Es wird eine getrennte Umgebung angelegt und chatterbox-tts geladen "
            "(mehrere GB, einige Minuten). Fortfahren?",
        ):
            return

        def handler(context) -> str:
            context.status("Richte Umgebung ein …")
            target = voice_runtime.install(on_status=context.status)
            return str(target)

        self._submit("setup", "Klon-Laufzeit einrichten", handler)

    def _selected_profile(self) -> voice_profiles.VoiceProfile | None:
        selection = self.profile_tree.selection()
        if not selection:
            return None
        return voice_profiles.load_profile(selection[0])

    def _show_profile_detail(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self.voicetrain_detail.configure(text="")
            self._refresh_sample_list(None)
            return
        self._last_profile = profile.slug
        self._load_tuning(profile)
        self._refresh_sample_list(profile)
        ready, problems = profile.training_ready()
        lines = [f"Ordner: {profile.root}"]
        samples = profile.samples()
        lines.append(
            f"Aufnahmen: {len(samples)}, davon brauchbar {sum(1 for s in samples if s.usable)}"
        )
        for sample in samples[:6]:
            state = "ok" if sample.usable else sample.note
            lines.append(f"  {sample.path.name}: {sample.seconds:.1f}s – {state}")
        if not ready:
            lines.append("Blockiert: " + " | ".join(problems))
        self.voicetrain_detail.configure(text="\n".join(lines))

    def _create_profile_dialog(self) -> None:
        gate = licensing.gate("voice-cloning")
        if not gate.allowed:
            messagebox.showwarning("Nicht freigegeben", gate.reason)
            return
        dialog = ConsentDialog(self, self.palette, self.runtime.config)
        self.wait_window(dialog)
        if dialog.result is None:
            return
        data = dialog.result
        try:
            consent = licensing.SpeakerConsent.create(
                speaker_name=data["speaker"],
                purpose=data["purpose"],
                granted_by=data["granted_by"],
                self_recorded=data["self_recorded"],
                evidence_note=data["evidence"],
            )
            profile = voice_profiles.create_profile(
                display_name=data["name"] or data["speaker"],
                consent=consent,
                model_key=self.runtime.config.voice_clone_model,
                mode=voice_profiles.TrainingMode(data["mode"]),
                language=self.runtime.config.language,
            )
        except (ValueError, OSError) as exc:
            messagebox.showerror("Profil nicht angelegt", accel.clean_error(exc))
            return
        self._refresh_voicetrain()
        self.log_view.append(f"Stimmprofil angelegt: {profile.slug}", "ok")
        messagebox.showinfo(
            "Profil angelegt",
            f"Aufnahmen ablegen in:\n{profile.samples_dir}\n\n"
            f"Benötigtes Material: mindestens "
            f"{voice_profiles.TrainingMode(data['mode']).min_seconds():.0f} Sekunden.",
        )

    def _add_sample(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            # Kein Grund zu meckern, wenn es nur ein Profil gibt: einfach das
            # letzte beziehungsweise erste nehmen.
            self._restore_selection()
            profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo(
                "Noch kein Stimmprofil",
                "Lege zuerst ein Profil an ('Neues Profil …') – dort wird auch "
                "die Einwilligung der sprechenden Person festgehalten.",
            )
            return
        files = filedialog.askopenfilenames(
            title="Aufnahmen wählen",
            filetypes=[("Audio", "*.wav *.flac *.mp3 *.m4a *.ogg")],
        )
        added = 0
        for file in files:
            try:
                info = voice_profiles.add_sample(profile, Path(file))
            except (OSError, ValueError) as exc:
                self.log_view.append(f"Aufnahme abgelehnt: {accel.clean_error(exc)}", "warn")
                continue
            added += 1
            tag = "ok" if info.usable else "warn"
            self.log_view.append(
                f"{info.path.name}: {info.seconds:.1f}s, {info.sample_rate} Hz – "
                f"{'brauchbar' if info.usable else info.note}",
                tag,
            )
        if not added:
            return

        self._last_profile = profile.slug
        self._refresh_voicetrain()

        # Neue Aufnahmen wirken erst, wenn die Referenz sie enthält. Das
        # ist der eigentliche "weiter anlernen"-Schritt und dauert nur
        # Sekunden (ffmpeg), deshalb läuft er gleich mit.
        frisch = voice_profiles.load_profile(profile.slug)
        bereit, hindernisse = frisch.training_ready() if frisch else (False, ["Profil weg"])
        if not bereit:
            self.log_view.append("Referenz noch nicht möglich: " + " | ".join(hindernisse), "warn")
            return
        handler = pipeline_voice.make_training_job(
            self.runtime.config, self.runtime.plan, profile.slug
        )
        self._submit("train", f"Referenz auffrischen: {profile.display_name}", handler)

    def _train_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("Kein Profil", "Zuerst ein Profil auswählen.")
            return
        ready, problems = profile.training_ready()
        if not ready:
            messagebox.showwarning("Anlernen nicht möglich", "\n".join(problems))
            return
        handler = pipeline_voice.make_training_job(
            self.runtime.config, self.runtime.plan, profile.slug
        )
        self._submit("train", f"Stimme anlernen: {profile.display_name}", handler)

    def _switch_to_zero_shot(self) -> None:
        """Profil vom nicht umgesetzten Nachtrainieren auf Referenz umstellen."""
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("Kein Profil", "Zuerst ein Profil auswählen.")
            return
        if profile.mode.available:
            messagebox.showinfo(
                "Bereits richtig", f"'{profile.display_name}' nutzt schon das Verfahren 'Referenz'."
            )
            return
        voice_profiles.set_mode(profile.slug, voice_profiles.TrainingMode.ZERO_SHOT)
        self.log_view.append(f"{profile.display_name}: Verfahren auf 'Referenz' umgestellt.", "ok")
        self._refresh_voicetrain()
        frisch = voice_profiles.load_profile(profile.slug)
        if frisch is not None and frisch.training_ready()[0]:
            handler = pipeline_voice.make_training_job(
                self.runtime.config, self.runtime.plan, profile.slug
            )
            self._submit("train", f"Referenz aufbereiten: {frisch.display_name}", handler)

    def _open_profile_dir(self) -> None:
        profile = self._selected_profile()
        if profile is not None:
            self._open_path(profile.root)

    def _delete_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            return
        if not messagebox.askyesno(
            "Profil löschen",
            f"Profil '{profile.display_name}' samt aller Aufnahmen und Artefakte löschen?\n\n"
            "Das ist auch der Weg für einen Widerruf der Einwilligung. "
            "Der Vorgang kann nicht rückgängig gemacht werden.",
        ):
            return
        if voice_profiles.delete_profile(profile.slug):
            self.log_view.append(f"Stimmprofil gelöscht: {profile.slug}", "warn")
        self._refresh_voicetrain()
        self.voicetrain_detail.configure(text="")

    # --- Warteschlange -----------------------------------------------------
    def _build_queue(self) -> ttk.Frame:
        outer, body = self._page_frame(
            "Warteschlange",
            "Aufträge laufen im Hintergrund. Ein Abbruch beendet nur den Auftrag, "
            "nicht das Programm.",
        )
        card = Card(body, self.palette)
        card.grid(row=0, column=0, sticky="nsew")
        inner = card.body()
        inner.columnconfigure(0, weight=1)
        self.queue_tree = ttk.Treeview(
            inner,
            columns=("kind", "state", "progress", "message", "time"),
            show="tree headings",
            height=14,
        )
        for column, title, width in (
            ("#0", "Auftrag", 260),
            ("kind", "Art", 80),
            ("state", "Zustand", 110),
            ("progress", "Fortschritt", 90),
            ("message", "Meldung", 320),
            ("time", "Dauer", 80),
        ):
            self.queue_tree.heading(column, text=title)
            self.queue_tree.column(column, width=width, anchor="w")
        self.queue_tree.grid(row=0, column=0, sticky="nsew")

        buttons = ttk.Frame(inner, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(
            buttons, text="Auswahl abbrechen", style="Danger.TButton", command=self._cancel_selected
        ).grid(row=0, column=0)
        ttk.Button(buttons, text="Alle abbrechen", command=self._cancel_all).grid(
            row=0, column=1, padx=6
        )
        ttk.Button(buttons, text="Erledigte entfernen", command=self._clear_finished).grid(
            row=0, column=2, padx=6
        )
        return outer

    def _refresh_queue(self) -> None:
        if not hasattr(self, "queue_tree"):
            return
        existing = set(self.queue_tree.get_children())
        for view in self.runtime.queue.snapshot():
            values = (
                view.kind,
                view.state.label(),
                f"{int(view.fraction * 100)}%",
                view.message[:120],
                f"{view.duration:.0f}s",
            )
            if view.id in existing:
                self.queue_tree.item(view.id, text=view.title, values=values)
                existing.discard(view.id)
            else:
                self.queue_tree.insert("", "end", iid=view.id, text=view.title, values=values)
        for stale in existing:
            self.queue_tree.delete(stale)
        self._stripe(self.queue_tree)

    def _cancel_selected(self) -> None:
        for job_id in self.queue_tree.selection():
            self.runtime.queue.cancel(job_id)

    def _cancel_all(self) -> None:
        count = self.runtime.queue.cancel_all()
        self.log_view.append(f"{count} Auftrag/Aufträge abgebrochen.", "warn")

    def _clear_finished(self) -> None:
        self.runtime.queue.clear_finished()
        self._refresh_queue()

    # --- Modelle -----------------------------------------------------------
    def _build_models(self) -> ttk.Frame:
        outer, body = self._page_frame(
            "Modelle",
            "Vor dem Download steht, was der Rechner schafft und unter welcher Lizenz "
            "das Modell steht. Gesperrte Modelle sind nicht kommerziell nutzbar.",
        )
        card = Card(body, self.palette)
        card.grid(row=0, column=0, sticky="nsew")
        inner = card.body()
        inner.columnconfigure(0, weight=1)
        self.models_tree = ttk.Treeview(
            inner,
            columns=("task", "size", "license", "commercial", "state", "hardware"),
            show="tree headings",
            height=14,
        )
        for column, title, width in (
            ("#0", "Modell", 210),
            ("task", "Aufgabe", 90),
            ("size", "Größe", 80),
            ("license", "Lizenz", 220),
            ("commercial", "kommerziell", 110),
            ("state", "Zustand", 110),
            ("hardware", "Hardware", 240),
        ):
            self.models_tree.heading(column, text=title)
            self.models_tree.column(column, width=width, anchor="w")
        self.models_tree.grid(row=0, column=0, sticky="nsew")
        self.models_tree.tag_configure("denied", foreground=self.palette.error)
        self.models_tree.tag_configure("conditional", foreground=self.palette.warn)
        self.models_tree.tag_configure("allowed", foreground=self.palette.text)

        buttons = ttk.Frame(inner, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(
            buttons, text="Herunterladen", style="Accent.TButton", command=self._download_selected
        ).grid(row=0, column=0)
        ttk.Button(buttons, text="Aufräumen", command=self._prune_selected).grid(
            row=0, column=1, padx=6
        )
        ttk.Button(
            buttons, text="Entfernen", style="Danger.TButton", command=self._remove_selected
        ).grid(row=0, column=2, padx=6)
        ttk.Button(
            buttons, text="Als Bildmodell setzen", command=lambda: self._set_model("image")
        ).grid(row=0, column=3, padx=6)
        ttk.Button(
            buttons, text="Als Videomodell setzen", command=lambda: self._set_model("video")
        ).grid(row=0, column=4, padx=6)
        ttk.Button(
            buttons, text="Als Stimmmodell setzen", command=lambda: self._set_model("voice")
        ).grid(row=0, column=5, padx=6)
        ttk.Button(
            buttons,
            text="Als Vergrößerungsmodell setzen",
            command=lambda: self._set_model("upscale"),
        ).grid(row=0, column=6, padx=6)
        ttk.Button(buttons, text="Modellseite öffnen", command=self._open_model_page).grid(
            row=0, column=7, padx=6
        )

        self.models_detail = ttk.Label(body, text="", style="Dim.TLabel", wraplength=880)
        self.models_detail.grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.models_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_model_detail())
        return outer

    def _refresh_models(self) -> None:
        report = self.runtime.hardware
        best = report.best_gpu
        vram = best.total_vram_mb if best else 0
        ram = report.cpu.ram_mb
        self.models_tree.delete(*self.models_tree.get_children())
        for spec in sorted(models.REGISTRY.values(), key=lambda s: (s.task.value, s.key)):
            fits, reason = models.fits_hardware(spec, vram, ram)
            state = "vorhanden" if models.is_downloaded(spec) else "nicht geladen"
            self.models_tree.insert(
                "",
                "end",
                iid=spec.key,
                text=spec.title,
                values=(
                    spec.task.value,
                    f"{spec.approx_size_gb:g} GB",
                    spec.license_id,
                    spec.commercial.label(),
                    state,
                    ("passt" if fits else "zu klein") + f" – {reason[:60]}",
                ),
                tags=(spec.commercial.value,),
            )
        self._stripe(self.models_tree)

    def _selected_model(self):
        selection = self.models_tree.selection()
        if not selection:
            return None
        return models.REGISTRY.get(selection[0])

    def _show_model_detail(self) -> None:
        spec = self._selected_model()
        if spec is None:
            return
        lines = [
            f"{spec.title} – {spec.repo_id}",
            f"Lizenz: {spec.license_id} ({spec.license_url})",
            f"Kommerziell: {spec.commercial.label()}",
        ]
        for obligation in spec.obligations:
            lines.append(f"Auflage: {obligation}")
        if spec.notes:
            lines.append(f"Hinweis: {spec.notes}")
        if models.is_downloaded(spec):
            lines.append(
                f"Belegt: {models.disk_usage_mb(spec) / 1024:.1f} GB in {models.local_dir(spec)}"
            )
        self.models_detail.configure(text="\n".join(lines))

    def _download_selected(self) -> None:
        spec = self._selected_model()
        if spec is None:
            return
        if spec.commercial is models.Commercial.DENIED:
            messagebox.showerror(
                "Gesperrt",
                f"{spec.title} steht unter '{spec.license_id}' und ist kommerziell "
                "nicht nutzbar. Der Download wird verweigert.",
            )
            return
        allow_conditional = False
        if spec.commercial is models.Commercial.CONDITIONAL:
            allow_conditional = messagebox.askyesno(
                "Lizenzbedingung",
                f"{spec.title}\n\nLizenz: {spec.license_id}\n\n"
                + "\n".join(f"– {o}" for o in spec.obligations)
                + "\n\nBedingungen geprüft und akzeptiert?",
            )
            if not allow_conditional:
                return
        if not self.runtime.config.allow_model_download:
            messagebox.showwarning(
                "Download aus",
                "Modell-Download ist in den Einstellungen ausgeschaltet "
                "(oder der Offline-Modus ist aktiv).",
            )
            return

        offline = self.runtime.config.offline_mode

        def handler(context) -> Path:
            def on_progress(done: int, total: int) -> None:
                fraction = (done / total) if total else 0.0
                context.progress(
                    fraction,
                    f"{done / (1024 * 1024):.0f} MB von {total / (1024 * 1024):.0f} MB",
                )

            try:
                return models.download(
                    spec,
                    on_progress=on_progress,
                    on_status=context.status,
                    should_stop=context.should_stop,
                    allow_conditional=allow_conditional,
                    offline=offline,
                )
            except models.DownloadCancelled as exc:
                from ..jobs import JobCancelled

                raise JobCancelled(str(exc)) from exc

        self._submit("download", f"Download: {spec.key}", handler)

    def _prune_selected(self) -> None:
        """Überflüssige Dateien eines geladenen Modells entfernen.

        Betrifft Modelle, die mit einem älteren Filter geladen wurden –
        fp32-Doppelungen, .bin neben .safetensors, ONNX/OpenVINO-Fassungen.
        """
        spec = self._selected_model()
        if spec is None or not models.is_downloaded(spec):
            messagebox.showinfo("Nichts zu tun", "Zuerst ein geladenes Modell auswählen.")
            return
        count, freed, names = models.prune_local(spec, dry_run=True)
        if not count:
            messagebox.showinfo(
                "Nichts zu tun", f"{spec.title} enthält keine überflüssigen Dateien."
            )
            return
        preview = "\n".join(f"– {name}" for name in names[:12])
        if len(names) > 12:
            preview += f"\n… und {len(names) - 12} weitere"
        if not messagebox.askyesno(
            "Aufräumen",
            f"{spec.title}\n\n{count} Datei(en) entfernen und {freed / 1024:.1f} GB freigeben?\n\n"
            f"{preview}\n\nDas Modell bleibt vollständig nutzbar.",
        ):
            return
        count, freed, _ = models.prune_local(spec)
        self.log_view.append(
            f"{spec.key}: {count} Datei(en) entfernt, {freed / 1024:.1f} GB frei.", "ok"
        )
        self._refresh_models()
        self._show_model_detail()

    def _remove_selected(self) -> None:
        spec = self._selected_model()
        if spec is None or not models.is_downloaded(spec):
            return
        if not messagebox.askyesno(
            "Modell entfernen",
            f"{spec.title} aus dem Cache löschen ({models.disk_usage_mb(spec) / 1024:.1f} GB)?",
        ):
            return
        freed = models.remove(spec)
        self.log_view.append(f"{spec.key} entfernt, {freed / 1024:.1f} GB frei.", "ok")
        self._refresh_models()

    def _set_model(self, slot: str) -> None:
        spec = self._selected_model()
        if spec is None:
            return
        if spec.commercial is models.Commercial.DENIED:
            messagebox.showerror("Gesperrt", "Dieses Modell ist kommerziell nicht nutzbar.")
            return
        field = {
            "image": "image_model",
            "video": "video_model",
            "voice": "voice_model",
            "upscale": "upscale_model",
        }[slot]
        self.runtime.config = self.runtime.config.with_values(**{field: spec.key})
        self.runtime.config.save()
        self.log_view.append(f"{slot}-Modell gesetzt: {spec.key}", "ok")
        self._replan_backend()

    def _open_model_page(self) -> None:
        spec = self._selected_model()
        if spec is not None and spec.license_url:
            webbrowser.open(spec.license_url)

    # --- Hardware ----------------------------------------------------------
    def _build_hardware(self) -> ttk.Frame:
        outer, body = self._page_frame(
            "Hardware und Backend",
            "Erkennung von GPU, NPU und CPU. Fehlt nvidia-smi, ist das kein Fehler – "
            "dann läuft es auf der CPU.",
        )
        card = Card(body, self.palette)
        card.grid(row=0, column=0, sticky="nsew")
        inner = card.body()
        inner.columnconfigure(0, weight=1)
        self.hardware_view = LogView(inner, self.palette, height=22)
        self.hardware_view.grid(row=0, column=0, sticky="nsew")

        buttons = ttk.Frame(inner, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(buttons, text="Neu erkennen", command=self._rescan_hardware).grid(
            row=0, column=0
        )
        ttk.Button(
            buttons, text="Datenordner öffnen", command=lambda: self._open_path(paths.data_dir())
        ).grid(row=0, column=1, padx=6)
        ttk.Button(buttons, text="Bericht kopieren", command=self._copy_hardware).grid(
            row=0, column=2, padx=6
        )
        return outer

    def _hardware_text(self) -> str:
        parts = [
            accel.describe_hardware(self.runtime.hardware),
            "",
            "== Backend ==",
            self.runtime.plan.report(),
            "",
            "== ffmpeg ==",
            compose.describe(),
            "",
            "== Vergrößern ==",
            upscale.describe(),
            "",
            "== TLS ==",
            f"{self.runtime.trust.label()} – {self.runtime.trust.detail}",
            "",
            "== Pfade ==",
            paths.describe(),
        ]
        dll_dirs = accel.prepared_dll_dirs()
        if dll_dirs:
            parts += ["", "== DLL-Suchpfad ==", *dll_dirs]
        return "\n".join(parts)

    def _refresh_hardware(self) -> None:
        self.hardware_view.set_text(self._hardware_text())

    def _rescan_hardware(self) -> None:
        self.runtime.hardware = accel.hardware_report(refresh=True)
        self._replan_backend()
        self._refresh_hardware()
        self._report_startup()

    def _copy_hardware(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self._hardware_text())
        self.status_label.configure(text="Bericht in die Zwischenablage kopiert.")

    def _replan_backend(self) -> None:
        spec = models.resolve(self.runtime.config.image_model)
        self.runtime.plan = accel.resolve_backend(
            self.runtime.config,
            readiness=models.readiness(spec),
            report=self.runtime.hardware,
            allow_proprietary=licensing.proprietary_gpu_allowed(),
        )
        self.backend_badge.configure(text=self.runtime.plan.label)

    # --- Lizenzen ----------------------------------------------------------
    def _build_licenses(self) -> ttk.Frame:
        outer, body = self._page_frame(
            "Lizenzen",
            "Proprietäre Laufzeiten und das Stimmklonen brauchen eine ausdrückliche "
            "Zustimmung. Ohne Zustimmung wird der freie Pfad genutzt.",
        )
        # AGB zuerst – das ist die Vertragsgrundlage, nicht bloß ein Hinweis.
        agb_card = Card(
            body,
            self.palette,
            "AGB und Endnutzer-Lizenzvertrag",
            "Vertragsgrundlage für die Nutzung dieser Anwendung.",
        )
        agb_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        agb_body = agb_card.body()
        self.agb_state = ttk.Label(agb_body, text="", style="SurfaceDim.TLabel", wraplength=800)
        self.agb_state.grid(row=0, column=0, sticky="w")
        agb_buttons = ttk.Frame(agb_body, style="Card.TFrame")
        agb_buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(
            agb_buttons, text="AGB lesen", style="Accent.TButton", command=self.show_agb
        ).grid(row=0, column=0)
        ttk.Button(
            agb_buttons,
            text="AGB-Datei öffnen",
            command=lambda: self._open_path(licensing.agb_path()),
        ).grid(row=0, column=1, padx=8)

        self.license_vars: dict[str, tk.BooleanVar] = {}
        store = licensing.store()
        row = 1
        for key, item in sorted(licensing.COMPONENTS.items()):
            if key == licensing.AGB_COMPONENT:
                continue  # steht oben als eigene Karte
            card = Card(body, self.palette, item.title, f"{item.license_id}\n{item.why}")
            card.grid(row=row, column=0, sticky="ew", pady=(0, 10))
            inner = card.body()
            var = tk.BooleanVar(value=store.is_accepted(key))
            self.license_vars[key] = var
            ttk.Checkbutton(
                inner,
                text="Bedingungen gelesen und zugestimmt",
                variable=var,
                style="Surface.TCheckbutton",
            ).grid(row=0, column=0, sticky="w")
            for index, obligation in enumerate(item.obligations, start=1):
                ttk.Label(
                    inner, text=f"– {obligation}", style="SurfaceDim.TLabel", wraplength=780
                ).grid(row=index, column=0, sticky="w", padx=(20, 0))
            if item.license_url:
                ttk.Button(
                    inner,
                    text="Lizenztext öffnen",
                    command=lambda url=item.license_url: webbrowser.open(url),
                ).grid(row=len(item.obligations) + 1, column=0, sticky="w", pady=(8, 0))
            row += 1

        actions = ttk.Frame(body)
        actions.grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Button(
            actions,
            text="Zustimmung speichern",
            style="Accent.TButton",
            command=self._save_licenses,
        ).grid(row=0, column=0)
        ttk.Button(
            actions,
            text="THIRD-PARTY-NOTICES öffnen",
            command=lambda: self._open_path(paths.notices_path()),
        ).grid(row=0, column=1, padx=8)
        self.license_status = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.license_status.grid(row=row + 1, column=0, sticky="w", pady=(10, 0))
        return outer

    def _refresh_licenses(self) -> None:
        store = licensing.store()
        for key, var in getattr(self, "license_vars", {}).items():
            var.set(store.is_accepted(key))
        if hasattr(self, "agb_state"):
            _text, version = licensing.agb_text()
            accepted = licensing.agb_accepted()
            self.agb_state.configure(
                text=(
                    f"Fassung {version} – "
                    + ("zugestimmt" if accepted else "noch nicht zugestimmt")
                    + f"\nDatei: {licensing.agb_path()}"
                )
            )
        notices = paths.notices_path()
        self.license_status.configure(
            text=(
                f"Hinweisdatei: {notices}"
                if notices.is_file()
                else f"WARNUNG: THIRD-PARTY-NOTICES.md fehlt ({notices})."
            )
        )

    def _save_licenses(self) -> None:
        store = licensing.store()
        accept = [key for key, var in self.license_vars.items() if var.get()]
        revoke = [key for key, var in self.license_vars.items() if not var.get()]
        store.accept(accept, note="über die Oberfläche bestätigt")
        store.revoke([key for key in revoke if store.is_accepted(key)])
        self._replan_backend()
        self.log_view.append(
            f"Lizenz-Zustimmung gespeichert: {len(accept)} zugestimmt, {len(revoke)} offen.", "ok"
        )
        self._refresh_licenses()

    # --- Einstellungen -----------------------------------------------------
    def _build_settings(self) -> ttk.Frame:
        config = self.runtime.config
        outer, body = self._page_frame(
            "Einstellungen",
            "Änderungen wirken auf neue Aufträge. Gerätewahl wirkt erst nach dem "
            "nächsten Modell-Laden.",
        )
        device_card = Card(body, self.palette, "Gerät und Leistung")
        device_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        form = device_card.body()
        self.set_device = ComboRow(
            form,
            0,
            "Gerät",
            DEVICE_CHOICES,
            config.device,
            hint="auto probiert CUDA, dann DirectML, dann OpenVINO, dann CPU. "
            "Ein Beschleuniger, der erst konvertiert werden müsste, "
            "wird im Auto-Modus übersprungen – die Konvertierung läuft "
            "über 'models convert' und dauert einmalig einige Minuten.",
        )
        self.set_openvino_device = ComboRow(
            form,
            2,
            "OpenVINO-Gerät",
            OPENVINO_DEVICES,
            config.openvino_device,
            hint="Nur bei Gerät = 'openvino'. Leer = beste verfügbare Wahl "
            "(NPU vor GPU vor CPU). Die NPU ist sparsamer, eine dedizierte "
            "Grafikkarte ist schneller.",
        )
        self.set_device_index = SpinRow(form, 4, "GPU-Nummer", config.device_index, 0, 15, 1)
        self.set_compute = ComboRow(
            form, 6, "Rechengenauigkeit", COMPUTE_CHOICES, config.compute_type
        )
        self.set_low_impact = CheckRow(
            form,
            8,
            "Rechner bedienbar halten",
            config.gpu_low_impact,
            hint="Weniger Durchsatz, dafür bleibt Windows flüssig.",
        )
        self.set_attention = CheckRow(form, 10, "Attention-Slicing", config.attention_slicing)
        self.set_vae_tiling = CheckRow(form, 12, "VAE-Tiling", config.vae_tiling)
        self.set_offload = CheckRow(
            form,
            14,
            "Modellteile auslagern (CPU-Offload)",
            config.cpu_offload,
            hint="Nötig bei knappem VRAM, kostet Geschwindigkeit.",
        )
        self.set_threads = SpinRow(
            form, 16, "CPU-Threads", config.cpu_threads, 0, 128, 1, hint="0 = automatisch."
        )
        self.set_workers = SpinRow(
            form,
            18,
            "Aufträge gleichzeitig",
            config.job_workers,
            1,
            4,
            1,
            hint="1 ist empfohlen – mehr Aufträge teilen sich den VRAM.",
        )

        io_card = Card(body, self.palette, "Ablage und Netz")
        io_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        io = io_card.body()
        self.set_output = PathRow(io, 0, "Ausgabeordner", config.output_dir, directory=True)
        self.set_download = CheckRow(io, 2, "Modell-Download erlauben", config.allow_model_download)
        self.set_offline = CheckRow(
            io,
            4,
            "Offline-Modus",
            config.offline_mode,
            hint="Kein Netzzugriff. Fehlende Modelle führen zu einer "
            "klaren Meldung statt zu einem Download.",
        )
        self.set_keep_loaded = CheckRow(
            io, 6, "Modell im Speicher halten", config.keep_model_loaded
        )
        self.set_theme = ComboRow(
            io, 8, "Farbschema", ("dark", "light"), config.theme, hint="Wirkt nach einem Neustart."
        )

        voice_card = Card(body, self.palette, "Stimme")
        voice_card.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        voice = voice_card.body()
        self.set_cloning = CheckRow(
            voice,
            0,
            "Angelernte Stimmen verwenden",
            config.voice_cloning_enabled,
            hint="Nur mit dokumentierter Einwilligung der sprechenden "
            "Person. Die Prüfung bleibt aktiv.",
        )
        self.set_clone_model = ComboRow(
            voice,
            2,
            "Klon-Modell",
            [s.key for s in models.by_task(models.Task.VOICE_CLONE)],
            config.voice_clone_model,
        )
        self.set_epochs = SpinRow(
            voice, 4, "Lern-Durchläufe", config.voice_training_epochs, 1, 200, 1
        )
        self.set_sample_rate = ComboRow(
            voice,
            6,
            "Abtastrate",
            ("16000", "22050", "24000", "44100", "48000"),
            str(config.voice_sample_rate),
        )

        adult_card = Card(
            body,
            self.palette,
            "Inhalte für Erwachsene",
            "An. Die Inhaltsprüfung der Modelle ist abgeschaltet – Nacktheit "
            "und erotische Darstellungen sind möglich.",
        )
        adult_card.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        adult = adult_card.body()
        self.set_nsfw = CheckRow(
            adult,
            0,
            "Nacktheit und erotische Darstellungen zulassen",
            config.nsfw_enabled,
            hint="Aus: die Inhaltsprüfung der Modelle wird wieder eingeschaltet "
            "(betrifft SD 1.5; SDXL und FLUX bringen keine mit).",
        )
        self.set_nsfw_negative = CheckRow(
            adult,
            2,
            "Schutzbegriffe an den Negativ-Prompt hängen",
            config.nsfw_protective_negative,
            hint="Hängt Begriffe wie 'child, teen, underage' an den Negativ-Prompt. "
            "Kostet nichts und hält das Modell von Mehrdeutigkeiten weg.",
        )
        ttk.Label(
            adult,
            text=(
                "Gesperrt bleibt: Aufträge, die Begriffe für Minderjährige mit "
                "sexuellen Begriffen verbinden, werden vor dem Laden des Modells "
                "abgelehnt (§ 184b StGB gilt auch für computererzeugte Bilder). "
                "Erwachsenendarstellungen sind davon nicht betroffen."
            ),
            style="SurfaceDim.TLabel",
            wraplength=780,
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        actions = ttk.Frame(body)
        actions.grid(row=4, column=0, sticky="w")
        ttk.Button(
            actions, text="Speichern", style="Accent.TButton", command=self._save_settings
        ).grid(row=0, column=0)
        ttk.Button(
            actions,
            text="Konfiguration öffnen",
            command=lambda: self._open_path(paths.config_path()),
        ).grid(row=0, column=1, padx=8)
        self.settings_status = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.settings_status.grid(row=5, column=0, sticky="w", pady=(10, 0))
        return outer

    def _config_with_ui(self) -> AppConfig:
        """Aktuelle Konfiguration – Einstellungsseite wird berücksichtigt,
        falls sie schon gebaut wurde."""
        return self.runtime.config

    def _save_settings(self) -> None:
        values = {
            "device": self.set_device.value(),
            "device_index": int(self.set_device_index.value()),
            "compute_type": self.set_compute.value(),
            "gpu_low_impact": self.set_low_impact.value(),
            "attention_slicing": self.set_attention.value(),
            "vae_tiling": self.set_vae_tiling.value(),
            "cpu_offload": self.set_offload.value(),
            "cpu_threads": int(self.set_threads.value()),
            "job_workers": int(self.set_workers.value()),
            "output_dir": self.set_output.value() or "output",
            "allow_model_download": self.set_download.value(),
            "offline_mode": self.set_offline.value(),
            "keep_model_loaded": self.set_keep_loaded.value(),
            "openvino_device": self.set_openvino_device.value(),
            "theme": self.set_theme.value(),
            "voice_cloning_enabled": self.set_cloning.value(),
            "voice_clone_model": self.set_clone_model.value(),
            "voice_training_epochs": int(self.set_epochs.value()),
            "voice_sample_rate": int(self.set_sample_rate.value()),
            "nsfw_enabled": self.set_nsfw.value(),
            "nsfw_protective_negative": self.set_nsfw_negative.value(),
        }
        config = self.runtime.config.with_values(**values)
        config, problems = config.validated()
        try:
            config.save()
        except OSError as exc:
            messagebox.showerror("Nicht gespeichert", accel.clean_error(exc))
            return
        self.runtime.config = config
        self._replan_backend()
        text = "Gespeichert." + (" " + " ".join(problems) if problems else "")
        self.settings_status.configure(text=text)
        for problem in problems:
            self.log_view.append(f"Konfiguration: {problem}", "warn")

    # --- Protokoll ---------------------------------------------------------
    def _build_logs(self) -> ttk.Frame:
        outer, body = self._page_frame(
            "Protokoll", "Meldungen dieser Sitzung. Die Datei liegt im Datenordner."
        )
        card = Card(body, self.palette)
        card.grid(row=0, column=0, sticky="nsew")
        inner = card.body()
        inner.columnconfigure(0, weight=1)
        self._log_page_view = LogView(inner, self.palette, height=24)
        self._log_page_view.grid(row=0, column=0, sticky="nsew")
        buttons = ttk.Frame(inner, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(
            buttons, text="Logordner öffnen", command=lambda: self._open_path(paths.logs_dir())
        ).grid(row=0, column=0)
        return outer

    # ------------------------------------------------------------------
    # Protokoll-Weiche: vor dem Bau der Protokollseite in einen Puffer
    # ------------------------------------------------------------------
    class _LogProxy:
        def __init__(self, window: MainWindow) -> None:
            self.window = window
            self.buffer: list[tuple[str, str]] = []

        def append(self, message: str, tag: str = "info") -> None:
            view = getattr(self.window, "_log_page_view", None)
            if view is None:
                self.buffer.append((message, tag))
                if len(self.buffer) > 500:
                    self.buffer.pop(0)
                return
            for pending, pending_tag in self.buffer:
                view.append(pending, pending_tag)
            self.buffer.clear()
            view.append(message, tag)

    @property
    def log_view(self) -> MainWindow._LogProxy:
        if not hasattr(self, "_log_proxy"):
            self._log_proxy = MainWindow._LogProxy(self)
        return self._log_proxy

    # ------------------------------------------------------------------
    # Aufträge
    # ------------------------------------------------------------------
    def _submit(self, kind: str, title: str, handler: Callable) -> str:
        job_id = self.runtime.queue.submit(kind, title, handler)
        self._tracked_job = job_id
        self.cancel_button.configure(state="normal")
        self.status_label.configure(text=f"{title} – eingereiht")
        self.log_view.append(f"Auftrag eingereiht: {title}", "dim")
        self._refresh_queue()
        return job_id

    def _update_queue_badge(self) -> None:
        """Zeigt offene Aufträge auch dann, wenn die Warteschlange zu ist."""
        offen = self.runtime.queue.active_count()
        self.queue_badge.configure(text=f"{offen} in Arbeit" if offen else "")

    def _cancel_tracked(self) -> None:
        if self._tracked_job:
            self.runtime.queue.cancel(self._tracked_job)
            self.status_label.configure(text="Abbruch angefordert …")

    def _on_job_event(self, event: JobEvent) -> None:
        """Läuft im Arbeiter-Thread – nur einsammeln, nicht anzeigen."""
        self._events.put(event)

    def _drain_events(self) -> None:
        try:
            while True:
                event = self._events.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass

        # Ergebnisse aus Hintergrundprüfungen
        try:
            while True:
                done, value, error = self._callbacks.get_nowait()
                try:
                    done(value, error)
                except Exception as exc:
                    log.debug("Rückruf fehlgeschlagen: %s", exc)
        except queue.Empty:
            pass

        self.after(120, self._drain_events)

    def _handle_event(self, event: JobEvent) -> None:
        view = event.job
        if event.event in ("progress", "status", "started"):
            if self._tracked_job in (None, view.id):
                self._tracked_job = view.id
                self.progress.configure(value=int(view.fraction * 1000))
                self.status_label.configure(text=f"{view.title}: {event.text[:90]}")
        elif event.event == "log":
            tag = {logging.WARNING: "warn", logging.ERROR: "error"}.get(event.level, "info")
            self.log_view.append(event.text, tag)
        elif event.event == "finished":
            self._on_job_finished(event)
        self._update_queue_badge()
        if self._active_page == "queue":
            self._refresh_queue()

    def _on_job_finished(self, event: JobEvent) -> None:
        view = event.job
        tag = {
            JobState.DONE: "ok",
            JobState.FAILED: "error",
            JobState.CANCELLED: "warn",
        }.get(view.state, "info")
        self.log_view.append(f"{view.title}: {view.state.label()} – {event.text}", tag)

        if view.id == self._tracked_job:
            self._tracked_job = None
            self.cancel_button.configure(state="disabled")
            self.progress.configure(value=1000 if view.state is JobState.DONE else 0)
            self.status_label.configure(text=f"{view.title}: {view.state.label()}")

        if view.state is not JobState.DONE:
            if view.state is JobState.FAILED:
                self.log_view.append(f"Fehler: {view.error}", "error")
            return

        result = view.result
        outputs: list[Path] = []
        for attribute in ("files", "video", "audio", "artifact"):
            value = getattr(result, attribute, None)
            if not value:
                continue
            items = value if isinstance(value, (list, tuple)) else [value]
            outputs.extend(Path(item) for item in items)
        for note in getattr(result, "notes", ()) or ():
            self.log_view.append(f"Hinweis: {note}", "dim")
        for path in outputs:
            self.log_view.append(f"Ausgabe: {path}", "ok")

        if view.kind in ("image", "edit") and outputs:
            self._last_images = list(outputs)
            summary = f"{len(outputs)} Datei(en): " + ", ".join(p.name for p in outputs)
            if view.kind == "image" and hasattr(self, "image_result"):
                self.image_result.configure(text=summary)
                self.image_preview.show(outputs[0])
            if view.kind == "edit" and hasattr(self, "edit_result"):
                self.edit_result.configure(text=summary)
                self.edit_result_preview.show(outputs[-1])
        if view.kind in ("train",):
            self._refresh_voicetrain()
        if view.kind == "download":
            self._refresh_models()
            self._replan_backend()
        if self.runtime.config.auto_open_output and outputs:
            self._open_path(outputs[0].parent)

    # ------------------------------------------------------------------
    # Kleinigkeiten
    # ------------------------------------------------------------------
    def _open_path(self, path: Path) -> None:
        target = Path(path)
        if not target.exists():
            if target.suffix:
                messagebox.showinfo("Nicht vorhanden", f"{target} gibt es noch nicht.")
                return
            paths.ensure_dir(target)
        try:
            import os

            if hasattr(os, "startfile"):
                os.startfile(str(target))  # type: ignore[attr-defined]  # Windows
            else:
                webbrowser.open(target.as_uri())
        except OSError as exc:
            messagebox.showwarning("Öffnen nicht möglich", accel.clean_error(exc))

    def _on_close(self) -> None:
        active = self.runtime.queue.active_count()
        if active and not messagebox.askyesno(
            "Beenden",
            f"Es laufen noch {active} Auftrag/Aufträge. Beim Beenden werden sie abgebrochen. "
            "Trotzdem beenden?",
        ):
            return
        self.status_label.configure(text="beende …")
        self.update_idletasks()
        self.runtime.queue.shutdown(wait=True, timeout=15)
        self.destroy()


class AgbDialog(tk.Toplevel):
    """AGB anzeigen und zustimmen lassen.

    Beim ersten Start blockierend: ohne Zustimmung wird die Anwendung
    beendet. Später jederzeit über den Knopf auf der Lizenzseite lesbar.
    Der Zustimmen-Knopf wird erst frei, wenn der Text bis zum Ende
    gerollt wurde – sonst ist die Bestätigung wertlos.
    """

    def __init__(self, master, palette: theme.Palette, blocking: bool = False) -> None:
        super().__init__(master)
        self.accepted = False
        self.blocking = blocking
        self.palette = palette
        text, version = licensing.agb_text()

        self.title("Allgemeine Geschäftsbedingungen")
        self.configure(background=palette.bg)
        self.transient(master)
        self.grab_set()
        self.geometry("860x680")
        self.minsize(640, 480)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        frame = ttk.Frame(self, padding=16)
        frame.grid(row=0, column=0, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ttk.Label(frame, text="Allgemeine Geschäftsbedingungen", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        already = licensing.agb_accepted()
        ttk.Label(
            frame,
            text=(
                f"Fassung {version} · "
                + ("bereits zugestimmt" if already else "Zustimmung erforderlich")
            ),
            style="Dim.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        holder = ttk.Frame(frame)
        holder.grid(row=2, column=0, sticky="nsew")
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        self.view = tk.Text(
            holder,
            wrap="word",
            background=palette.surface,
            foreground=palette.text,
            relief="flat",
            padx=14,
            pady=12,
            font=theme.FONT_SUB,
            insertbackground=palette.text,
        )
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self._on_scroll)
        self.view.configure(yscrollcommand=self._on_view_scroll)
        self.view.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self._scrollbar = scroll
        self.view.insert("1.0", text)
        self.view.configure(state="disabled")
        self.view.bind("<MouseWheel>", lambda _e: self.after(30, self._check_end))
        self.view.bind("<Key>", lambda _e: self.after(30, self._check_end))

        self.hint = ttk.Label(frame, text="", style="Warn.TLabel", wraplength=800)
        self.hint.grid(row=3, column=0, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, sticky="e", pady=(12, 0))
        ttk.Button(buttons, text="Drucken/Öffnen", command=self._open_file).grid(
            row=0, column=0, padx=6
        )
        self.reject_button = ttk.Button(
            buttons,
            text="Ablehnen und beenden" if blocking else "Schließen",
            command=self._on_close,
        )
        self.reject_button.grid(row=0, column=1, padx=6)
        self.accept_button = ttk.Button(
            buttons, text="Zustimmen", style="Accent.TButton", command=self._accept
        )
        self.accept_button.grid(row=0, column=2)
        if already and not blocking:
            self.accept_button.configure(
                text="Zustimmung widerrufen", style="Danger.TButton", command=self._revoke
            )
        else:
            self.accept_button.configure(state="disabled")
            self.hint.configure(
                text="Bitte den Text bis zum Ende lesen – danach wird die Schaltfläche frei."
            )
        self.after(300, self._check_end)

    # --- Rollen / Ende erkennen -------------------------------------------
    def _on_scroll(self, *args) -> None:
        self.view.yview(*args)
        self.after(30, self._check_end)

    def _on_view_scroll(self, first: str, last: str) -> None:
        self._scrollbar.set(first, last)
        if float(last) >= 0.999:
            self._enable_accept()

    def _check_end(self) -> None:
        try:
            _first, last = self.view.yview()
        except tk.TclError:
            return
        if last >= 0.999:
            self._enable_accept()

    def _enable_accept(self) -> None:
        if str(self.accept_button.cget("state")) == "disabled" and not licensing.agb_accepted():
            self.accept_button.configure(state="normal")
            self.hint.configure(text="")

    # --- Aktionen ----------------------------------------------------------
    def _open_file(self) -> None:
        target = licensing.agb_path()
        try:
            import os

            if hasattr(os, "startfile"):
                os.startfile(str(target))  # type: ignore[attr-defined]
            else:
                webbrowser.open(target.as_uri())
        except OSError as exc:
            messagebox.showwarning("Öffnen nicht möglich", accel.clean_error(exc), parent=self)

    def _accept(self) -> None:
        licensing.accept_agb("über die Oberfläche bestätigt")
        self.accepted = True
        self.destroy()

    def _revoke(self) -> None:
        if messagebox.askyesno(
            "Zustimmung widerrufen",
            "Ohne Zustimmung zu den AGB darf die Anwendung nicht genutzt werden. "
            "Sie wird nach dem Widerruf beendet. Fortfahren?",
            parent=self,
        ):
            licensing.revoke_agb()
            self.accepted = False
            self.destroy()
            self.master.destroy()

    def _on_close(self) -> None:
        self.accepted = licensing.agb_accepted()
        self.destroy()


class ConsentDialog(tk.Toplevel):
    """Einwilligung für eine anzulernende Stimme aufnehmen.

    Ohne Namen der sprechenden Person und ohne Bestätigung wird kein Profil
    angelegt – das ist die fachliche Sperre, nicht bloß ein Hinweis.
    """

    def __init__(self, master: MainWindow, palette: theme.Palette, config: AppConfig) -> None:
        super().__init__(master)
        self.result: dict[str, Any] | None = None
        self.palette = palette
        self.title("Stimmprofil anlegen")
        self.configure(background=palette.bg)
        self.transient(master)
        self.grab_set()
        self.resizable(False, False)

        frame = ttk.Frame(self, padding=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Stimme anlernen", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            frame,
            text=(
                "Für jede angelernte Stimme braucht es die Einwilligung der sprechenden "
                "Person. Der Nachweis wird beim Profil gespeichert und kann jederzeit "
                "widerrufen werden (Profil löschen)."
            ),
            style="Dim.TLabel",
            wraplength=560,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 12))

        self.row_name = EntryRow(frame, 2, "Profilname", "", width=44)
        self.row_speaker = EntryRow(
            frame, 4, "Sprechende Person", "", width=44, hint="Vollständiger Name – Pflichtangabe."
        )
        self.row_purpose = EntryRow(
            frame, 6, "Zweck", "Sprachausgabe in eigenen Produktionen", width=44
        )
        self.row_granted = EntryRow(
            frame, 8, "Eingeholt von", "", width=44, hint="Leer = eigene Stimme des Bedieners."
        )
        self.row_evidence = EntryRow(
            frame,
            10,
            "Nachweis/Aktenzeichen",
            "",
            width=44,
            hint="Verweis auf die schriftliche Einwilligung.",
        )
        ttk.Label(
            frame,
            text=(
                "Verfahren: Referenzstimme. Aus den Aufnahmen entsteht eine "
                "saubere Referenz, die das Modell zur Laufzeit benutzt – ab "
                "etwa 10 Sekunden. Weitere Aufnahmen lassen sich jederzeit "
                "nachlegen und verbessern das Ergebnis."
            ),
            style="Dim.TLabel",
            wraplength=560,
        ).grid(row=12, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.confirm = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Ich bestätige, dass eine Einwilligung der genannten Person vorliegt.",
            variable=self.confirm,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=15, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Abbrechen", command=self.destroy).grid(row=0, column=0, padx=6)
        ttk.Button(
            buttons, text="Profil anlegen", style="Accent.TButton", command=self._accept
        ).grid(row=0, column=1)

    def _accept(self) -> None:
        speaker = self.row_speaker.value().strip()
        if not speaker:
            messagebox.showinfo(
                "Angabe fehlt", "Name der sprechenden Person ist Pflicht.", parent=self
            )
            return
        if not self.confirm.get():
            messagebox.showwarning(
                "Einwilligung fehlt",
                "Ohne Bestätigung der Einwilligung wird kein Profil angelegt.",
                parent=self,
            )
            return
        granted_by = self.row_granted.value().strip()
        self.result = {
            "name": self.row_name.value().strip(),
            "speaker": speaker,
            "purpose": self.row_purpose.value().strip(),
            "granted_by": granted_by or "Bediener",
            "self_recorded": not granted_by,
            "evidence": self.row_evidence.value().strip(),
            "mode": "zero_shot",
        }
        self.destroy()


def run_gui(runtime) -> int:
    """Oberfläche starten. Fehlt tkinter, gibt es eine klare Meldung."""
    try:
        import tkinter  # noqa: F401
    except ImportError as exc:
        print(
            "Die Oberfläche braucht tkinter, das in dieser Python-Installation fehlt. "
            f"Nutze die Kommandozeile (--no-gui). Ursache: {exc}"
        )
        return 1
    window = MainWindow(runtime)
    window.mainloop()
    return 0
