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
import time
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .. import __app_display_name__, __version__, accel, compose, licensing, models, paths
from .. import pipeline_image, pipeline_video, pipeline_voice, voice_profiles
from ..config import AppConfig, COMPUTE_CHOICES, DEVICE_CHOICES, IMAGE_FORMATS, VIDEO_CONTAINERS
from ..jobs import JobEvent, JobState
from . import theme
from .widgets import (
    Banner, Card, CheckRow, ComboRow, EntryRow, LogView, PathRow, ScrollArea,
    SliderRow, SpinRow, TextRow,
)

log = logging.getLogger(__name__)

PAGES: tuple[tuple[str, str], ...] = (
    ("image", "Bild"),
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
        self._pages: dict[str, ttk.Frame] = {}
        self._nav_buttons: dict[str, ttk.Button] = {}
        self._active_page = ""
        self._tracked_job: str | None = None

        self._build_layout()
        self.runtime.queue.subscribe(self._on_job_event)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._drain_events)

        self.show_page("image")
        self._report_startup()
        # AGB-Abfrage nach dem Aufbau, damit das Hauptfenster dahinter steht.
        self.after(150, self._require_agb)

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
        ttk.Label(header, text=__app_display_name__, style="Surface.TLabel",
                  font=theme.FONT_TITLE).grid(row=0, column=0, sticky="w")
        self.header_info = ttk.Label(header, text="", style="SurfaceDim.TLabel")
        self.header_info.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        self.backend_badge = ttk.Label(header, text="", style="Badge.TLabel")
        self.backend_badge.grid(row=0, column=2, sticky="e", padx=(8, 0))

        # Navigation
        sidebar = ttk.Frame(self, style="Sidebar.TFrame", padding=(0, 10))
        sidebar.grid(row=1, column=0, sticky="nsw")
        for index, (key, label) in enumerate(PAGES):
            button = ttk.Button(sidebar, text=label, style="Nav.TButton", width=20,
                                command=lambda k=key: self.show_page(k))
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
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=1000)
        self.progress.grid(row=0, column=1, sticky="ew", padx=14)
        self.cancel_button = ttk.Button(footer, text="Abbrechen", style="Danger.TButton",
                                        command=self._cancel_tracked, state="disabled")
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
    def _report_startup(self) -> None:
        report = self.runtime.hardware
        best = report.best_gpu
        gpu_text = best.label() if best else "keine GPU erkannt"
        self.header_info.configure(
            text=f"{report.cpu.label()}  ·  {gpu_text}  ·  {report.advice.title}"
        )
        self.backend_badge.configure(text=self.runtime.plan.label)

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

    # ------------------------------------------------------------------
    # Seiten
    # ------------------------------------------------------------------
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
        self.image_prompt = TextRow(form, 0, "Prompt", self.palette,
                                    value="", height=4,
                                    hint="Was soll zu sehen sein? Englisch liefert bei den "
                                         "meisten Modellen bessere Ergebnisse.")
        self.image_negative = TextRow(form, 2, "Negativ-Prompt", self.palette,
                                      value=config.image_negative_prompt, height=2)
        self.image_width = SpinRow(form, 4, "Breite", config.image_width, 256, 4096, 64)
        self.image_height = SpinRow(form, 6, "Höhe", config.image_height, 256, 4096, 64)
        self.image_steps = SliderRow(form, 8, "Schritte", config.image_steps, 1, 100, integer=True)
        self.image_guidance = SliderRow(form, 10, "Führung (CFG)", config.image_guidance, 0, 20)
        self.image_sampler = ComboRow(form, 12, "Sampler", pipeline_image.SAMPLERS,
                                      config.image_sampler)
        self.image_batch = SpinRow(form, 14, "Anzahl Bilder", config.image_batch, 1, 16, 1)
        self.image_seed = SpinRow(form, 16, "Seed", -1, -1, 2**31 - 1, 1,
                                  hint="-1 = zufällig. Gleicher Seed und gleiche Einstellungen "
                                       "ergeben dasselbe Bild.")
        self.image_format = ComboRow(form, 18, "Dateiformat", IMAGE_FORMATS, config.image_format)

        actions = ttk.Frame(body)
        actions.grid(row=2, column=0, sticky="ew")
        ttk.Button(actions, text="Bild erzeugen", style="Accent.TButton",
                   command=self._submit_image).grid(row=0, column=0)
        ttk.Button(actions, text="Ausgabeordner öffnen",
                   command=lambda: self._open_path(config.resolved_output_dir() / "images")).grid(
            row=0, column=1, padx=8)
        self.image_result = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.image_result.grid(row=3, column=0, sticky="w", pady=(10, 0))
        return outer

    def _submit_image(self) -> None:
        prompt = self.image_prompt.value()
        if not prompt:
            messagebox.showinfo("Prompt fehlt", "Bitte zuerst beschreiben, was zu sehen sein soll.")
            return
        config = self._config_with_ui()
        request = pipeline_image.ImageRequest.from_config(
            config, prompt,
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
        handler = pipeline_image.make_job(config, self.runtime.plan, request,
                                          force_dummy=self.runtime.force_dummy())
        self._submit("image", f"Bild: {prompt[:40]}", handler)

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
        self.video_init = PathRow(form, 2, "Startbild (optional)", "",
                                  hint="Leer = Text zu Video. Gesetzt = Bild wird animiert.",
                                  filetypes=[("Bilder", "*.png *.jpg *.jpeg *.webp")])
        self.video_width = SpinRow(form, 4, "Breite", config.video_width, 256, 1920, 32)
        self.video_height = SpinRow(form, 6, "Höhe", config.video_height, 256, 1088, 32)
        self.video_frames = SliderRow(form, 8, "Bilder", config.video_frames, 8, 241, integer=True)
        self.video_fps = SpinRow(form, 10, "Bildrate", config.video_fps, 4, 60, 1)
        self.video_steps = SliderRow(form, 12, "Schritte", config.video_steps, 1, 80, integer=True)
        self.video_motion = SliderRow(form, 14, "Bewegungsstärke", config.video_motion, 0.1, 3.0)
        self.video_container = ComboRow(form, 16, "Container", VIDEO_CONTAINERS,
                                        config.video_container)
        self.video_audio = PathRow(form, 18, "Tonspur (optional)", "",
                                   hint="WAV/MP3 wird direkt mit eingebettet.",
                                   filetypes=[("Ton", "*.wav *.mp3 *.flac *.m4a")])
        self.video_keep_frames = CheckRow(form, 20, "Einzelbilder behalten", False)

        actions = ttk.Frame(body)
        actions.grid(row=1, column=0, sticky="ew")
        ttk.Button(actions, text="Video erzeugen", style="Accent.TButton",
                   command=self._submit_video).grid(row=0, column=0)
        ttk.Button(actions, text="Ausgabeordner öffnen",
                   command=lambda: self._open_path(config.resolved_output_dir() / "videos")).grid(
            row=0, column=1, padx=8)
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
            config, prompt,
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
        handler = pipeline_video.make_job(config, self.runtime.plan, request,
                                          force_dummy=self.runtime.force_dummy())
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
        speakers = [f"v2/de_speaker_{i}" for i in range(10)] + \
                   [f"v2/en_speaker_{i}" for i in range(4)] + ["de_DE-thorsten-medium"]
        if config.voice_speaker not in speakers:
            speakers.insert(0, config.voice_speaker)
        self.voice_speaker = ComboRow(form, 4, "Sprecher", speakers, config.voice_speaker,
                                      hint="Bark: v2/de_speaker_N. Piper: Stimmname.")
        self.voice_speed = SliderRow(form, 6, "Geschwindigkeit", config.voice_speed, 0.5, 2.0)
        self.voice_pitch = SliderRow(form, 8, "Tonhöhe", config.voice_pitch, -12, 12, unit=" HT")
        self.voice_volume = SliderRow(form, 10, "Lautstärke", config.voice_volume_db, -20, 6,
                                      unit=" dB")
        self.voice_split = CheckRow(form, 12, "Satzweise erzeugen", config.voice_split_sentences,
                                    hint="Bei langen Texten stabiler und besser abbrechbar.")

        actions = ttk.Frame(body)
        actions.grid(row=1, column=0, sticky="ew")
        ttk.Button(actions, text="Sprache erzeugen", style="Accent.TButton",
                   command=self._submit_voice).grid(row=0, column=0)
        ttk.Button(actions, text="Mit Video vertonen …",
                   command=self._mux_dialog).grid(row=0, column=1, padx=8)
        ttk.Button(actions, text="Ausgabeordner öffnen",
                   command=lambda: self._open_path(config.resolved_output_dir() / "audio")).grid(
            row=0, column=2, padx=8)
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
            config, text,
            profile_slug=slug,
            speaker=self.voice_speaker.value(),
            speed=float(self.voice_speed.value()),
            pitch=float(self.voice_pitch.value()),
            volume_db=float(self.voice_volume.value()),
            split_sentences=self.voice_split.value(),
        )
        handler = pipeline_voice.make_job(config, self.runtime.plan, request,
                                          force_dummy=self.runtime.force_dummy())
        self._submit("voice", f"Sprache: {text[:40]}", handler)

    def _mux_dialog(self) -> None:
        if not compose.available():
            messagebox.showwarning(
                "ffmpeg fehlt",
                "Zum Vertonen wird ffmpeg gebraucht. Lege einen LGPL-Build nach "
                f"{paths.tools_dir() / 'ffmpeg'}.",
            )
            return
        video = filedialog.askopenfilename(title="Video wählen",
                                          filetypes=[("Video", "*.mp4 *.webm *.mov")])
        if not video:
            return
        audio = filedialog.askopenfilename(title="Tonspur wählen",
                                          filetypes=[("Ton", "*.wav *.mp3 *.flac *.m4a")])
        if not audio:
            return
        config = self.runtime.config
        target = config.resolved_output_dir() / "videos" / (
            f"{Path(video).stem}_vertont{Path(video).suffix}"
        )

        def handler(context) -> Path:
            return compose.mux(
                Path(video), Path(audio), target,
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
        self.voicetrain_gate = ttk.Label(gate_body, text="", style="SurfaceDim.TLabel",
                                         wraplength=820)
        self.voicetrain_gate.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Button(gate_body, text="Lizenzseite öffnen",
                   command=lambda: self.show_page("licenses")).grid(row=1, column=0,
                                                                   sticky="w", pady=(8, 0))

        # Klonstimmen brauchen eine getrennte Laufzeit (siehe voice_runtime).
        runtime_card = Card(
            body, self.palette, "Laufzeit für Klonstimmen",
            "Chatterbox läuft in einer eigenen Umgebung, weil es ältere Fassungen "
            "von torch und diffusers verlangt und sonst die GPU-Beschleunigung "
            "für Bild und Video zerstören würde.",
        )
        runtime_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        runtime_body = runtime_card.body()
        self.voice_runtime_state = ttk.Label(runtime_body, text="", style="SurfaceDim.TLabel",
                                             wraplength=800)
        self.voice_runtime_state.grid(row=0, column=0, sticky="w")
        runtime_buttons = ttk.Frame(runtime_body, style="Card.TFrame")
        runtime_buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(runtime_buttons, text="Laufzeit einrichten",
                   command=self._install_voice_runtime).grid(row=0, column=0)
        ttk.Button(runtime_buttons, text="Erneut prüfen",
                   command=self._refresh_voicetrain).grid(row=0, column=1, padx=6)

        list_card = Card(body, self.palette, "Profile")
        list_card.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        list_body = list_card.body()
        self.profile_tree = ttk.Treeview(
            list_body, columns=("state", "mode", "material", "speaker"), show="tree headings",
            height=8,
        )
        for column, title, width in (
            ("#0", "Profil", 200), ("state", "Zustand", 140), ("mode", "Verfahren", 100),
            ("material", "Material", 90), ("speaker", "Einwilligung von", 200),
        ):
            self.profile_tree.heading(column, text=title)
            self.profile_tree.column(column, width=width, anchor="w")
        self.profile_tree.grid(row=0, column=0, columnspan=2, sticky="nsew")
        list_body.columnconfigure(0, weight=1)

        buttons = ttk.Frame(list_body, style="Card.TFrame")
        buttons.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        ttk.Button(buttons, text="Neues Profil …", style="Accent.TButton",
                   command=self._create_profile_dialog).grid(row=0, column=0)
        ttk.Button(buttons, text="Aufnahme hinzufügen …",
                   command=self._add_sample).grid(row=0, column=1, padx=6)
        ttk.Button(buttons, text="Anlernen",
                   command=self._train_profile).grid(row=0, column=2, padx=6)
        ttk.Button(buttons, text="Ordner öffnen",
                   command=self._open_profile_dir).grid(row=0, column=3, padx=6)
        ttk.Button(buttons, text="Löschen / Widerruf", style="Danger.TButton",
                   command=self._delete_profile).grid(row=0, column=4, padx=6)

        self.voicetrain_detail = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.voicetrain_detail.grid(row=3, column=0, sticky="w")
        self.profile_tree.bind("<<TreeviewSelect>>", lambda _e: self._show_profile_detail())
        return outer

    def _refresh_voicetrain(self) -> None:
        gate = licensing.gate("voice-cloning")
        self.voicetrain_gate.configure(
            text=(
                "Freigegeben. Für jede angelernte Stimme muss eine Einwilligung der "
                "sprechenden Person vorliegen."
                if gate.allowed
                else gate.reason
            )
        )
        if hasattr(self, "voice_runtime_state"):
            from .. import voice_runtime

            ok, note = voice_runtime.available(refresh=True)
            self.voice_runtime_state.configure(
                text=("Bereit – " + note) if ok else (
                    note
                    + "\n\nOhne sie wird beim Erzeugen die Standardstimme verwendet, "
                      "kein Platzhalterton."
                )
            )

        self.profile_tree.delete(*self.profile_tree.get_children())
        for profile in voice_profiles.list_profiles():
            speaker = profile.consent.speaker_name if profile.consent else "kein Nachweis"
            self.profile_tree.insert(
                "", "end", iid=profile.slug, text=profile.display_name,
                values=(profile.state.label(), profile.mode.value,
                        f"{profile.total_seconds():.0f}s", speaker),
            )

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
            return
        ready, problems = profile.training_ready()
        lines = [f"Ordner: {profile.root}"]
        samples = profile.samples()
        lines.append(f"Aufnahmen: {len(samples)}, davon brauchbar "
                     f"{sum(1 for s in samples if s.usable)}")
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
            messagebox.showinfo("Kein Profil", "Zuerst ein Profil auswählen.")
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
                f"{'brauchbar' if info.usable else info.note}", tag,
            )
        if added:
            self._refresh_voicetrain()
            self.profile_tree.selection_set(profile.slug)
            self._show_profile_detail()

    def _train_profile(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            messagebox.showinfo("Kein Profil", "Zuerst ein Profil auswählen.")
            return
        ready, problems = profile.training_ready()
        if not ready:
            messagebox.showwarning("Anlernen nicht möglich", "\n".join(problems))
            return
        handler = pipeline_voice.make_training_job(self.runtime.config, self.runtime.plan,
                                                   profile.slug)
        self._submit("train", f"Stimme anlernen: {profile.display_name}", handler)

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
            inner, columns=("kind", "state", "progress", "message", "time"),
            show="tree headings", height=14,
        )
        for column, title, width in (
            ("#0", "Auftrag", 260), ("kind", "Art", 80), ("state", "Zustand", 110),
            ("progress", "Fortschritt", 90), ("message", "Meldung", 320), ("time", "Dauer", 80),
        ):
            self.queue_tree.heading(column, text=title)
            self.queue_tree.column(column, width=width, anchor="w")
        self.queue_tree.grid(row=0, column=0, sticky="nsew")

        buttons = ttk.Frame(inner, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(buttons, text="Auswahl abbrechen", style="Danger.TButton",
                   command=self._cancel_selected).grid(row=0, column=0)
        ttk.Button(buttons, text="Alle abbrechen",
                   command=self._cancel_all).grid(row=0, column=1, padx=6)
        ttk.Button(buttons, text="Erledigte entfernen",
                   command=self._clear_finished).grid(row=0, column=2, padx=6)
        return outer

    def _refresh_queue(self) -> None:
        if not hasattr(self, "queue_tree"):
            return
        existing = set(self.queue_tree.get_children())
        for view in self.runtime.queue.snapshot():
            values = (view.kind, view.state.label(), f"{int(view.fraction * 100)}%",
                      view.message[:120], f"{view.duration:.0f}s")
            if view.id in existing:
                self.queue_tree.item(view.id, text=view.title, values=values)
                existing.discard(view.id)
            else:
                self.queue_tree.insert("", "end", iid=view.id, text=view.title, values=values)
        for stale in existing:
            self.queue_tree.delete(stale)

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
            inner, columns=("task", "size", "license", "commercial", "state", "hardware"),
            show="tree headings", height=14,
        )
        for column, title, width in (
            ("#0", "Modell", 210), ("task", "Aufgabe", 90), ("size", "Größe", 80),
            ("license", "Lizenz", 220), ("commercial", "kommerziell", 110),
            ("state", "Zustand", 110), ("hardware", "Hardware", 240),
        ):
            self.models_tree.heading(column, text=title)
            self.models_tree.column(column, width=width, anchor="w")
        self.models_tree.grid(row=0, column=0, sticky="nsew")
        self.models_tree.tag_configure("denied", foreground=self.palette.error)
        self.models_tree.tag_configure("conditional", foreground=self.palette.warn)
        self.models_tree.tag_configure("allowed", foreground=self.palette.text)

        buttons = ttk.Frame(inner, style="Card.TFrame")
        buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(buttons, text="Herunterladen", style="Accent.TButton",
                   command=self._download_selected).grid(row=0, column=0)
        ttk.Button(buttons, text="Aufräumen", command=self._prune_selected).grid(
            row=0, column=1, padx=6)
        ttk.Button(buttons, text="Entfernen", style="Danger.TButton",
                   command=self._remove_selected).grid(row=0, column=2, padx=6)
        ttk.Button(buttons, text="Als Bildmodell setzen",
                   command=lambda: self._set_model("image")).grid(row=0, column=3, padx=6)
        ttk.Button(buttons, text="Als Videomodell setzen",
                   command=lambda: self._set_model("video")).grid(row=0, column=4, padx=6)
        ttk.Button(buttons, text="Als Stimmmodell setzen",
                   command=lambda: self._set_model("voice")).grid(row=0, column=5, padx=6)
        ttk.Button(buttons, text="Modellseite öffnen",
                   command=self._open_model_page).grid(row=0, column=6, padx=6)

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
                "", "end", iid=spec.key, text=spec.title,
                values=(spec.task.value, f"{spec.approx_size_gb:g} GB", spec.license_id,
                        spec.commercial.label(), state,
                        ("passt" if fits else "zu klein") + f" – {reason[:60]}"),
                tags=(spec.commercial.value,),
            )

    def _selected_model(self):
        selection = self.models_tree.selection()
        if not selection:
            return None
        return models.REGISTRY.get(selection[0])

    def _show_model_detail(self) -> None:
        spec = self._selected_model()
        if spec is None:
            return
        lines = [f"{spec.title} – {spec.repo_id}",
                 f"Lizenz: {spec.license_id} ({spec.license_url})",
                 f"Kommerziell: {spec.commercial.label()}"]
        for obligation in spec.obligations:
            lines.append(f"Auflage: {obligation}")
        if spec.notes:
            lines.append(f"Hinweis: {spec.notes}")
        if models.is_downloaded(spec):
            lines.append(f"Belegt: {models.disk_usage_mb(spec) / 1024:.1f} GB in "
                         f"{models.local_dir(spec)}")
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
                    spec, on_progress=on_progress, on_status=context.status,
                    should_stop=context.should_stop, allow_conditional=allow_conditional,
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
            messagebox.showinfo("Nichts zu tun", f"{spec.title} enthält keine überflüssigen Dateien.")
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
        self.log_view.append(f"{spec.key}: {count} Datei(en) entfernt, "
                             f"{freed / 1024:.1f} GB frei.", "ok")
        self._refresh_models()
        self._show_model_detail()

    def _remove_selected(self) -> None:
        spec = self._selected_model()
        if spec is None or not models.is_downloaded(spec):
            return
        if not messagebox.askyesno(
            "Modell entfernen",
            f"{spec.title} aus dem Cache löschen "
            f"({models.disk_usage_mb(spec) / 1024:.1f} GB)?",
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
        field = {"image": "image_model", "video": "video_model", "voice": "voice_model"}[slot]
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
        ttk.Button(buttons, text="Neu erkennen", command=self._rescan_hardware).grid(row=0, column=0)
        ttk.Button(buttons, text="Datenordner öffnen",
                   command=lambda: self._open_path(paths.data_dir())).grid(row=0, column=1, padx=6)
        ttk.Button(buttons, text="Bericht kopieren",
                   command=self._copy_hardware).grid(row=0, column=2, padx=6)
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
        agb_card = Card(body, self.palette, "AGB und Endnutzer-Lizenzvertrag",
                        "Vertragsgrundlage für die Nutzung dieser Anwendung.")
        agb_card.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        agb_body = agb_card.body()
        self.agb_state = ttk.Label(agb_body, text="", style="SurfaceDim.TLabel", wraplength=800)
        self.agb_state.grid(row=0, column=0, sticky="w")
        agb_buttons = ttk.Frame(agb_body, style="Card.TFrame")
        agb_buttons.grid(row=1, column=0, sticky="w", pady=(10, 0))
        ttk.Button(agb_buttons, text="AGB lesen", style="Accent.TButton",
                   command=self.show_agb).grid(row=0, column=0)
        ttk.Button(agb_buttons, text="AGB-Datei öffnen",
                   command=lambda: self._open_path(licensing.agb_path())).grid(
            row=0, column=1, padx=8)

        self.license_vars: dict[str, tk.BooleanVar] = {}
        store = licensing.store()
        row = 1
        for key, item in sorted(licensing.COMPONENTS.items()):
            if key == licensing.AGB_COMPONENT:
                continue  # steht oben als eigene Karte
            card = Card(body, self.palette, item.title,
                        f"{item.license_id}\n{item.why}")
            card.grid(row=row, column=0, sticky="ew", pady=(0, 10))
            inner = card.body()
            var = tk.BooleanVar(value=store.is_accepted(key))
            self.license_vars[key] = var
            ttk.Checkbutton(inner, text="Bedingungen gelesen und zugestimmt",
                            variable=var, style="Surface.TCheckbutton").grid(
                row=0, column=0, sticky="w")
            for index, obligation in enumerate(item.obligations, start=1):
                ttk.Label(inner, text=f"– {obligation}", style="SurfaceDim.TLabel",
                          wraplength=780).grid(row=index, column=0, sticky="w", padx=(20, 0))
            if item.license_url:
                ttk.Button(inner, text="Lizenztext öffnen",
                           command=lambda url=item.license_url: webbrowser.open(url)).grid(
                    row=len(item.obligations) + 1, column=0, sticky="w", pady=(8, 0))
            row += 1

        actions = ttk.Frame(body)
        actions.grid(row=row, column=0, sticky="w", pady=(4, 0))
        ttk.Button(actions, text="Zustimmung speichern", style="Accent.TButton",
                   command=self._save_licenses).grid(row=0, column=0)
        ttk.Button(actions, text="THIRD-PARTY-NOTICES öffnen",
                   command=lambda: self._open_path(paths.notices_path())).grid(
            row=0, column=1, padx=8)
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
                text=(f"Fassung {version} – "
                      + ("zugestimmt" if accepted else "noch nicht zugestimmt")
                      + f"\nDatei: {licensing.agb_path()}")
            )
        notices = paths.notices_path()
        self.license_status.configure(
            text=(f"Hinweisdatei: {notices}" if notices.is_file()
                  else f"WARNUNG: THIRD-PARTY-NOTICES.md fehlt ({notices}).")
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
        self.set_device = ComboRow(form, 0, "Gerät", DEVICE_CHOICES, config.device,
                                   hint="auto probiert CUDA, dann DirectML, dann CPU. "
                                        "Ein Beschleuniger, der erst konvertiert werden müsste, "
                                        "wird im Auto-Modus übersprungen.")
        self.set_device_index = SpinRow(form, 2, "GPU-Nummer", config.device_index, 0, 15, 1)
        self.set_compute = ComboRow(form, 4, "Rechengenauigkeit", COMPUTE_CHOICES,
                                    config.compute_type)
        self.set_low_impact = CheckRow(form, 6, "Rechner bedienbar halten", config.gpu_low_impact,
                                       hint="Weniger Durchsatz, dafür bleibt Windows flüssig.")
        self.set_attention = CheckRow(form, 8, "Attention-Slicing", config.attention_slicing)
        self.set_vae_tiling = CheckRow(form, 10, "VAE-Tiling", config.vae_tiling)
        self.set_offload = CheckRow(form, 12, "Modellteile auslagern (CPU-Offload)",
                                    config.cpu_offload,
                                    hint="Nötig bei knappem VRAM, kostet Geschwindigkeit.")
        self.set_threads = SpinRow(form, 14, "CPU-Threads", config.cpu_threads, 0, 128, 1,
                                   hint="0 = automatisch.")
        self.set_workers = SpinRow(form, 16, "Aufträge gleichzeitig", config.job_workers, 1, 4, 1,
                                   hint="1 ist empfohlen – mehr Aufträge teilen sich den VRAM.")

        io_card = Card(body, self.palette, "Ablage und Netz")
        io_card.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        io = io_card.body()
        self.set_output = PathRow(io, 0, "Ausgabeordner", config.output_dir, directory=True)
        self.set_download = CheckRow(io, 2, "Modell-Download erlauben", config.allow_model_download)
        self.set_offline = CheckRow(io, 4, "Offline-Modus", config.offline_mode,
                                    hint="Kein Netzzugriff. Fehlende Modelle führen zu einer "
                                         "klaren Meldung statt zu einem Download.")
        self.set_keep_loaded = CheckRow(io, 6, "Modell im Speicher halten", config.keep_model_loaded)
        self.set_theme = ComboRow(io, 8, "Farbschema", ("dark", "light"), config.theme,
                                  hint="Wirkt nach einem Neustart.")

        voice_card = Card(body, self.palette, "Stimme")
        voice_card.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        voice = voice_card.body()
        self.set_cloning = CheckRow(voice, 0, "Angelernte Stimmen verwenden",
                                    config.voice_cloning_enabled,
                                    hint="Nur mit dokumentierter Einwilligung der sprechenden "
                                         "Person. Die Prüfung bleibt aktiv.")
        self.set_clone_model = ComboRow(
            voice, 2, "Klon-Modell",
            [s.key for s in models.by_task(models.Task.VOICE_CLONE)],
            config.voice_clone_model,
        )
        self.set_epochs = SpinRow(voice, 4, "Lern-Durchläufe", config.voice_training_epochs,
                                  1, 200, 1)
        self.set_sample_rate = ComboRow(voice, 6, "Abtastrate",
                                        ("16000", "22050", "24000", "44100", "48000"),
                                        str(config.voice_sample_rate))

        actions = ttk.Frame(body)
        actions.grid(row=3, column=0, sticky="w")
        ttk.Button(actions, text="Speichern", style="Accent.TButton",
                   command=self._save_settings).grid(row=0, column=0)
        ttk.Button(actions, text="Konfiguration öffnen",
                   command=lambda: self._open_path(paths.config_path())).grid(
            row=0, column=1, padx=8)
        self.settings_status = ttk.Label(body, text="", style="Dim.TLabel", wraplength=860)
        self.settings_status.grid(row=4, column=0, sticky="w", pady=(10, 0))
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
            "theme": self.set_theme.value(),
            "voice_cloning_enabled": self.set_cloning.value(),
            "voice_clone_model": self.set_clone_model.value(),
            "voice_training_epochs": int(self.set_epochs.value()),
            "voice_sample_rate": int(self.set_sample_rate.value()),
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
        ttk.Button(buttons, text="Logordner öffnen",
                   command=lambda: self._open_path(paths.logs_dir())).grid(row=0, column=0)
        return outer

    # ------------------------------------------------------------------
    # Protokoll-Weiche: vor dem Bau der Protokollseite in einen Puffer
    # ------------------------------------------------------------------
    class _LogProxy:
        def __init__(self, window: "MainWindow") -> None:
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
    def log_view(self) -> "MainWindow._LogProxy":
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
        finally:
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

        if view.kind == "image" and outputs and hasattr(self, "image_result"):
            self.image_result.configure(
                text=f"{len(outputs)} Datei(en): " + ", ".join(p.name for p in outputs)
            )
        if view.kind in ("train",) :
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

        ttk.Label(frame, text="Allgemeine Geschäftsbedingungen",
                  style="Title.TLabel").grid(row=0, column=0, sticky="w")
        already = licensing.agb_accepted()
        ttk.Label(
            frame,
            text=(f"Fassung {version} · "
                  + ("bereits zugestimmt" if already else "Zustimmung erforderlich")),
            style="Dim.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 10))

        holder = ttk.Frame(frame)
        holder.grid(row=2, column=0, sticky="nsew")
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        self.view = tk.Text(holder, wrap="word", background=palette.surface,
                            foreground=palette.text, relief="flat", padx=14, pady=12,
                            font=theme.FONT_SUB, insertbackground=palette.text)
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
        ttk.Button(buttons, text="Drucken/Öffnen",
                   command=self._open_file).grid(row=0, column=0, padx=6)
        self.reject_button = ttk.Button(
            buttons, text="Ablehnen und beenden" if blocking else "Schließen",
            command=self._on_close)
        self.reject_button.grid(row=0, column=1, padx=6)
        self.accept_button = ttk.Button(buttons, text="Zustimmen", style="Accent.TButton",
                                        command=self._accept)
        self.accept_button.grid(row=0, column=2)
        if already and not blocking:
            self.accept_button.configure(text="Zustimmung widerrufen",
                                         style="Danger.TButton", command=self._revoke)
        else:
            self.accept_button.configure(state="disabled")
            self.hint.configure(text="Bitte den Text bis zum Ende lesen – "
                                     "danach wird die Schaltfläche frei.")
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
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            frame,
            text=(
                "Für jede angelernte Stimme braucht es die Einwilligung der sprechenden "
                "Person. Der Nachweis wird beim Profil gespeichert und kann jederzeit "
                "widerrufen werden (Profil löschen)."
            ),
            style="Dim.TLabel", wraplength=560,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 12))

        self.row_name = EntryRow(frame, 2, "Profilname", "", width=44)
        self.row_speaker = EntryRow(frame, 4, "Sprechende Person", "", width=44,
                                    hint="Vollständiger Name – Pflichtangabe.")
        self.row_purpose = EntryRow(frame, 6, "Zweck", "Sprachausgabe in eigenen Produktionen",
                                    width=44)
        self.row_granted = EntryRow(frame, 8, "Eingeholt von", "", width=44,
                                    hint="Leer = eigene Stimme des Bedieners.")
        self.row_evidence = EntryRow(frame, 10, "Nachweis/Aktenzeichen", "", width=44,
                                     hint="Verweis auf die schriftliche Einwilligung.")
        self.row_mode = ComboRow(frame, 12, "Verfahren", ("zero_shot", "finetune"), "zero_shot",
                                 hint="zero_shot: ab ~10 s Referenz. "
                                      "finetune: ab ~10 min Material, deutlich länger.")

        self.confirm = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame,
            text="Ich bestätige, dass eine Einwilligung der genannten Person vorliegt.",
            variable=self.confirm,
        ).grid(row=14, column=0, columnspan=2, sticky="w", pady=(8, 0))

        buttons = ttk.Frame(frame)
        buttons.grid(row=15, column=0, columnspan=2, sticky="e", pady=(16, 0))
        ttk.Button(buttons, text="Abbrechen", command=self.destroy).grid(row=0, column=0, padx=6)
        ttk.Button(buttons, text="Profil anlegen", style="Accent.TButton",
                   command=self._accept).grid(row=0, column=1)

    def _accept(self) -> None:
        speaker = self.row_speaker.value().strip()
        if not speaker:
            messagebox.showinfo("Angabe fehlt", "Name der sprechenden Person ist Pflicht.",
                                parent=self)
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
            "mode": self.row_mode.value(),
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
