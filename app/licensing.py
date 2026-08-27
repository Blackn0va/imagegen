"""Zustimmung zu Drittanbieter-Lizenzen – fail-closed.

Zwei Arten von Zustimmung:

1. **Komponenten** – proprietäre Laufzeiten (NVIDIA CUDA/cuDNN) und Modelle
   mit eigenen Bedingungen. Ohne Zustimmung wird die Komponente NICHT
   geladen; die Anwendung fällt auf den freien Pfad zurück und sagt das im
   Klartext.
2. **Sprecher-Einwilligung** – für angelernte/geklonte Stimmen. Neben der
   Modell-Lizenz steht dort das Persönlichkeitsrecht der Person. Ohne
   dokumentierte Einwilligung wird ein Stimmprofil nicht verwendet.
   Diese Sperre ist nicht über die Konfiguration abschaltbar.

Zustimmungen liegen als JSON-Markerdatei neben der Konfiguration.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import __version__, paths

log = logging.getLogger(__name__)

CONSENT_FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Komponenten-Registrierung
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LicenseComponent:
    key: str
    title: str
    license_id: str
    license_url: str
    terms_version: str  # ändert sich die Fassung, ist neu zuzustimmen
    why: str  # wofür die Komponente gebraucht wird
    obligations: tuple[str, ...] = ()
    required: bool = False  # False = es gibt einen freien Rückfallpfad


COMPONENTS: dict[str, LicenseComponent] = {
    "nvidia-cuda": LicenseComponent(
        key="nvidia-cuda",
        title="NVIDIA CUDA Runtime (cudart, cublas, cufft, curand, cusparse)",
        license_id="NVIDIA Software License Agreement / CUDA EULA",
        license_url="https://docs.nvidia.com/cuda/eula/index.html",
        terms_version="cuda-12",
        why="GPU-Beschleunigung auf NVIDIA-Karten (Bild und Video in Minuten statt Stunden).",
        obligations=(
            "Weitergabe nur als Teil einer Anwendung, nicht als eigenständiges SDK.",
            "Lizenztext muss dem Endkunden beiliegen (THIRD-PARTY-NOTICES.md).",
        ),
    ),
    "nvidia-cudnn": LicenseComponent(
        key="nvidia-cudnn",
        title="NVIDIA cuDNN",
        license_id="NVIDIA cuDNN Software License Agreement",
        license_url="https://docs.nvidia.com/deeplearning/cudnn/sla/index.html",
        terms_version="cudnn-9",
        why="Beschleunigte Faltungs- und Attention-Kerne für Diffusionsmodelle.",
        obligations=(
            "Weitergabe nur eingebettet in die Anwendung.",
            "Namensnennung und Lizenztext in der Auslieferung.",
        ),
    ),
    "piper-gpl": LicenseComponent(
        key="piper-gpl",
        title="Piper-Sprachausgabe (GPL-3.0, enthält espeak-ng)",
        license_id="GPL-3.0-or-later",
        license_url="https://github.com/OHF-voice/piper1-gpl/blob/main/LICENSE.md",
        terms_version="piper-gpl-1",
        why="Schnelle deutsche Sprachausgabe auf der CPU (Stimme Thorsten).",
        obligations=(
            "ACHTUNG: Das Paket steht unter GPL-3.0 und bettet espeak-ng ein. "
            "Wird es in denselben Prozess geladen, erfasst die GPL nach "
            "verbreiteter Auslegung die gesamte Anwendung – für eine verkaufte, "
            "proprietäre Anwendung ist das nicht tragbar.",
            "Zulässiger Weg: Piper als eigenständiges Programm ausliefern und "
            "über die Kommandozeile aufrufen (getrennter Prozess), Lizenztext "
            "und Quelltextangebot beilegen.",
            "Im Zweifel die MIT-Alternative Bark verwenden – dort entfällt die Frage vollständig.",
        ),
    ),
    "private-use": LicenseComponent(
        key="private-use",
        title="Private Nutzung – Modelle ohne kommerzielle Freigabe verwenden",
        license_id="Lizenzen der jeweiligen Modelle",
        license_url="",
        terms_version="private-1",
        why=(
            "Modelle freischalten, deren Lizenz die kommerzielle Nutzung "
            "einschränkt oder ausschließt (z. B. FLUX.1-dev, "
            "CreativeML-OpenRAIL-Varianten). Für rein private Nutzung ist "
            "das nach diesen Lizenzen zulässig."
        ),
        obligations=(
            "Gilt NUR für private, nicht-kommerzielle Nutzung. Sobald mit den "
            "Ergebnissen Geld verdient wird – Verkauf, Auftragsarbeit, "
            "Werbung, Streaming mit Einnahmen – greift die Sperre wieder.",
            "Die Auflagen der einzelnen Modelle bleiben bestehen: "
            "Namensnennung, Weitergabeverbote, Nutzungsbeschränkungen. "
            "Nachzulesen unter 'models table' und in MODELS.md.",
            "Die Anwendung selbst darf mit dieser Freischaltung nicht "
            "weitergegeben oder verkauft werden – die Modelle wären dann "
            "Teil eines kommerziellen Produkts.",
            "Diese Zustimmung wird mit Zeitpunkt und Fassung protokolliert.",
        ),
    ),
    "voice-cloning": LicenseComponent(
        key="voice-cloning",
        title="Stimme anlernen / klonen",
        license_id="Nutzungsbedingungen der Anwendung + Persönlichkeitsrecht",
        license_url="",
        terms_version="voice-1",
        why="Anlernen einer eigenen Stimme aus Sprachaufnahmen.",
        obligations=(
            "Für jede angelernte Stimme muss eine Einwilligung der sprechenden "
            "Person vorliegen (Name, Datum, Zweck).",
            "Keine Stimmen realer Personen ohne deren Einwilligung.",
            "Ergebnisse dürfen nicht als echte Äußerung dieser Person ausgegeben werden.",
        ),
    ),
}


# ---------------------------------------------------------------------------
# AGB / Endnutzer-Lizenzvertrag
# ---------------------------------------------------------------------------
AGB_FILENAME = "AGB.md"
AGB_COMPONENT = "agb"


def agb_path() -> Path:
    """AGB-Datei suchen: neben der .exe, im Bundle, sonst im Projekt."""
    for candidate in (
        paths.exe_dir / AGB_FILENAME,
        paths.bundle_dir / AGB_FILENAME,
        paths.exe_dir / "_internal" / AGB_FILENAME,
    ):
        if candidate.is_file():
            return candidate
    return paths.exe_dir / AGB_FILENAME


def agb_text() -> tuple[str, str]:
    """(Text, Fassungskennung). Fehlt die Datei, wird das klar gesagt."""
    target = agb_path()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        return (
            f"Die Datei {AGB_FILENAME} fehlt in der Auslieferung ({target}).\n\n"
            "Ohne AGB darf die Anwendung nicht verkauft werden. Bitte beim "
            "Anbieter melden.",
            "fehlt",
        )
    # Fassung = Kurz-Hash des Textes: ändert sich der Wortlaut, muss erneut
    # zugestimmt werden. Ein Datum im Text wäre pflegeanfällig.
    version = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return text, version


def register_agb() -> LicenseComponent:
    """AGB als Zustimmungs-Komponente anmelden (Fassung aus dem Text)."""
    _text, version = agb_text()
    item = LicenseComponent(
        key=AGB_COMPONENT,
        title="Allgemeine Geschäftsbedingungen und Endnutzer-Lizenzvertrag",
        license_id=f"AGB Fassung {version}",
        license_url="",
        terms_version=version,
        why="Vertragsgrundlage für die Nutzung dieser Anwendung.",
        obligations=(
            "Nutzungsbeschränkungen der Modelle einhalten.",
            "Stimmen realer Personen nur mit deren Einwilligung anlernen.",
            "Erzeugte Inhalte kennzeichnen, wo das Recht es verlangt.",
        ),
        required=True,
    )
    COMPONENTS[AGB_COMPONENT] = item
    return item


def agb_accepted() -> bool:
    register_agb()
    return store().is_accepted(AGB_COMPONENT)


def accept_agb(note: str = "") -> bool:
    """AGB bestätigen – und damit alle übrigen Komponenten mit.

    Die AGB sind die Vertragsgrundlage und nennen die Auflagen bereits im
    Wortlaut. Ein zweiter Durchgang über die Lizenzseite fragt dieselbe
    Zustimmung nur ein weiteres Mal ab, ohne dass eine andere Entscheidung
    zur Wahl stünde.

    Protokolliert wird trotzdem jede Komponente einzeln, mit dem Vermerk,
    woher die Zustimmung kam. Ein Widerruf auf der Lizenzseite hält, bis
    die AGB erneut bestätigt werden – das geschieht nur auf Knopfdruck
    oder wenn sich ihr Wortlaut ändert, also nie beiläufig.

    Nicht berührt bleibt die dokumentierte Einwilligung je angelernter
    Stimme: die wird pro Profil erhoben und ist die eigentliche Schranke
    beim Stimmenklonen.
    """
    register_agb()
    consent = store()
    zugestimmt = bool(consent.accept(AGB_COMPONENT, note=note or "AGB bestätigt"))
    if not zugestimmt:
        return False

    weitere = [key for key in COMPONENTS if key != AGB_COMPONENT]
    if weitere:
        consent.accept(weitere, note="über die AGB-Zustimmung mitbestätigt")
        log.info(
            "AGB bestätigt – %d weitere Komponente(n) mitbestätigt: %s",
            len(weitere),
            ", ".join(sorted(weitere)),
        )
    return True


def sync_agb_coverage() -> list[str]:
    """Nachziehen, was die AGB-Zustimmung abdeckt.

    Wer den AGB zugestimmt hat, bevor die Sammelzustimmung eingebaut war,
    bekam sie nie – die Punkte standen weiter offen, obwohl der Vertrag
    sie nennt. Dasselbe passiert, wenn spaeter eine neue Komponente
    hinzukommt.

    Wird beim Start aufgerufen. Widerrufenes bleibt widerrufen: geprueft
    wird nur, was noch **nie** entschieden wurde – ein Widerruf ist ein
    Eintrag im Speicher, kein fehlender.

    Rueckgabe: die nachgetragenen Schluessel.
    """
    if not agb_accepted():
        return []

    consent = store()
    consent._ensure_loaded()
    nachzutragen = [
        key for key in COMPONENTS if key != AGB_COMPONENT and key not in consent._records
    ]
    if not nachzutragen:
        return []
    consent.accept(nachzutragen, note="über die AGB-Zustimmung mitbestätigt (nachgetragen)")
    log.info(
        "AGB-Zustimmung nachgezogen fuer: %s",
        ", ".join(sorted(nachzutragen)),
    )
    return sorted(nachzutragen)


def agb_covers() -> list[LicenseComponent]:
    """Komponenten, die mit der AGB-Zustimmung mitkommen."""
    return [item for key, item in sorted(COMPONENTS.items()) if key != AGB_COMPONENT]


def revoke_agb() -> bool:
    return bool(store().revoke(AGB_COMPONENT))


PRIVATE_USE_COMPONENT = "private-use"
# Merker, dass die Vorgabe einmal gesetzt wurde. Ohne ihn ließe sich ein
# Widerruf nicht von „noch nie entschieden" unterscheiden – die Freigabe
# käme beim nächsten Start einfach zurück.
PRIVATE_USE_DEFAULT_MARK = "private-use-default"


def ensure_private_use_default() -> bool:
    """Private Nutzung beim ersten Start freischalten.

    Diese Anwendung wird nicht verkauft, sondern privat betrieben – dann
    erlauben die Lizenzen der eingeschränkten Modelle die Nutzung. Die
    Freigabe ist deshalb Vorgabe statt Handarbeit.

    Zweierlei bleibt trotzdem gewahrt: sie wird **protokolliert** (mit dem
    Vermerk, dass sie aus der Vorgabe stammt und nicht angeklickt wurde),
    und ein **Widerruf hält**. Wer sie zurückzieht, bekommt sie beim
    nächsten Start nicht wieder untergeschoben.

    Rückgabe: True, wenn jetzt freigeschaltet wurde.
    """
    consent = store()
    consent._ensure_loaded()
    if PRIVATE_USE_DEFAULT_MARK in consent._records:
        return False  # Entscheidung liegt vor – Vorgabe nicht erneut anwenden
    COMPONENTS.setdefault(
        PRIVATE_USE_DEFAULT_MARK,
        LicenseComponent(
            key=PRIVATE_USE_DEFAULT_MARK,
            title="Vermerk: Vorgabe zur privaten Nutzung wurde angewendet",
            license_id="intern",
            license_url="",
            terms_version="private-1",
            why="Hält fest, dass die Vorgabe einmal gesetzt wurde.",
        ),
    )
    consent.accept(
        [PRIVATE_USE_COMPONENT, PRIVATE_USE_DEFAULT_MARK],
        note="aus der Vorgabe gesetzt, nicht vom Bediener bestätigt",
    )
    log.info(
        "Private Nutzung als Vorgabe freigeschaltet. Die Anwendung darf mit "
        "dieser Einstellung nicht weitergegeben oder verkauft werden."
    )
    return True


def private_use_accepted() -> bool:
    """Ist die private Nutzung ausdrücklich freigeschaltet?

    Fail-closed: ohne zugestimmte, aktuelle Fassung bleibt es bei der
    Sperre. Ändert sich der Wortlaut der Auflagen, wird ``terms_version``
    hochgezählt und muss neu bestätigt werden.
    """
    return store().is_accepted(PRIVATE_USE_COMPONENT)


def accept_private_use(note: str = "") -> bool:
    return bool(store().accept(PRIVATE_USE_COMPONENT, note=note or "Private Nutzung bestätigt"))


def revoke_private_use() -> bool:
    return bool(store().revoke(PRIVATE_USE_COMPONENT))


def component(key: str) -> LicenseComponent | None:
    return COMPONENTS.get(key)


def register_component(item: LicenseComponent) -> None:
    """Zusätzliche Komponente anmelden (z. B. modellspezifische Bedingungen)."""
    COMPONENTS[item.key] = item


# ---------------------------------------------------------------------------
# Zustimmungs-Speicher
# ---------------------------------------------------------------------------
@dataclass
class ConsentRecord:
    key: str
    terms_version: str
    accepted_at: float
    app_version: str
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "terms_version": self.terms_version,
            "accepted_at": self.accepted_at,
            "accepted_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.accepted_at)),
            "app_version": self.app_version,
            "note": self.note,
        }


@dataclass
class ConsentStore:
    path: Path = field(default_factory=paths.consent_path)
    _records: dict[str, ConsentRecord] = field(default_factory=dict, repr=False)
    _loaded: bool = field(default=False, repr=False)

    # --- Laden / Speichern -------------------------------------------------
    def load(self) -> ConsentStore:
        self._records.clear()
        self._loaded = True
        if not self.path.is_file():
            return self
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            from .accel import clean_error

            log.warning(
                "Zustimmungsdatei nicht lesbar (%s) – gilt als nicht zugestimmt.", clean_error(exc)
            )
            return self
        entries = raw.get("components", {}) if isinstance(raw, dict) else {}
        if isinstance(entries, dict):
            for key, value in entries.items():
                if not isinstance(value, dict):
                    continue
                self._records[key] = ConsentRecord(
                    key=key,
                    terms_version=str(value.get("terms_version", "")),
                    accepted_at=float(value.get("accepted_at", 0) or 0),
                    app_version=str(value.get("app_version", "")),
                    note=str(value.get("note", "")),
                )
        return self

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def save(self) -> Path:
        """Atomar schreiben – halbe Zustimmungsdatei wäre fatal."""
        paths.ensure_dir(self.path.parent)
        payload = {
            "format": CONSENT_FORMAT_VERSION,
            "app_version": __version__,
            "components": {k: r.to_dict() for k, r in sorted(self._records.items())},
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        return self.path

    # --- Abfragen ----------------------------------------------------------
    def is_accepted(self, key: str) -> bool:
        """True nur, wenn genau die aktuelle Fassung zugestimmt wurde."""
        self._ensure_loaded()
        item = COMPONENTS.get(key)
        record = self._records.get(key)
        if record is None:
            return False
        if item is None:
            return bool(record.accepted_at)
        return record.terms_version == item.terms_version and record.accepted_at > 0

    def pending(self, keys: Iterable[str] | None = None) -> list[LicenseComponent]:
        """Komponenten, deren Zustimmung noch fehlt."""
        self._ensure_loaded()
        wanted = list(keys) if keys is not None else list(COMPONENTS)
        return [COMPONENTS[k] for k in wanted if k in COMPONENTS and not self.is_accepted(k)]

    def accepted_keys(self) -> list[str]:
        self._ensure_loaded()
        return sorted(k for k in self._records if self.is_accepted(k))

    # --- Setzen ------------------------------------------------------------
    def accept(self, keys: str | Iterable[str], note: str = "") -> list[str]:
        self._ensure_loaded()
        if isinstance(keys, str):
            keys = [keys]
        changed: list[str] = []
        for key in keys:
            item = COMPONENTS.get(key)
            if item is None:
                log.debug("Zustimmung für unbekannte Komponente '%s' ignoriert.", key)
                continue
            self._records[key] = ConsentRecord(
                key=key,
                terms_version=item.terms_version,
                accepted_at=time.time(),
                app_version=__version__,
                note=note,
            )
            changed.append(key)
        if changed:
            self.save()
            log.info("Lizenz-Zustimmung erteilt: %s", ", ".join(changed))
        return changed

    def revoke(self, keys: str | Iterable[str]) -> list[str]:
        """Zustimmung zurückziehen – und das festhalten.

        Der Eintrag wird NICHT gelöscht, sondern mit ``accepted_at = 0``
        stehen gelassen. Sonst ist "widerrufen" von "noch nie entschieden"
        nicht zu unterscheiden, und ein Nachtrag (siehe
        ``sync_agb_coverage``) holt die Zustimmung beim nächsten Start
        einfach zurück. Genau so verhielt es sich, bis dieser Fall
        auffiel.

        ``is_accepted`` prüft ``accepted_at > 0`` – ein Eintrag mit 0
        gilt damit als nicht zugestimmt.
        """
        self._ensure_loaded()
        if isinstance(keys, str):
            keys = [keys]
        entfernt: list[str] = []
        for key in keys:
            vorhanden = self._records.get(key)
            if vorhanden is None or vorhanden.accepted_at <= 0:
                continue
            self._records[key] = ConsentRecord(
                key=key,
                terms_version=vorhanden.terms_version,
                accepted_at=0.0,
                app_version=vorhanden.app_version,
                note="widerrufen",
            )
            entfernt.append(key)
        if entfernt:
            self.save()
            log.info("Lizenz-Zustimmung zurückgezogen: %s", ", ".join(entfernt))
        return entfernt

    def was_revoked(self, key: str) -> bool:
        """Wurde diese Zustimmung ausdrücklich zurückgezogen?"""
        self._ensure_loaded()
        eintrag = self._records.get(key)
        return eintrag is not None and eintrag.accepted_at <= 0


_store: ConsentStore | None = None


def store() -> ConsentStore:
    """Prozessweiter Zustimmungs-Speicher."""
    global _store
    if _store is None:
        _store = ConsentStore().load()
    return _store


# ---------------------------------------------------------------------------
# Tore (fail-closed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateResult:
    allowed: bool
    reason: str
    missing: tuple[str, ...] = ()


def gate(keys: str | Iterable[str]) -> GateResult:
    """Darf geladen werden? Nein -> Klartext-Begründung, kein Laden."""
    if isinstance(keys, str):
        keys = [keys]
    keys = list(keys)
    missing = [k for k in keys if not store().is_accepted(k)]
    if not missing:
        return GateResult(True, "Zustimmung liegt vor.")
    titles = ", ".join(COMPONENTS[k].title if k in COMPONENTS else k for k in missing)
    return GateResult(
        False,
        f"Nicht freigegeben: {titles}. Unter Einstellungen → Lizenzen zustimmen. "
        "Bis dahin wird der freie Pfad genutzt (z. B. CPU statt CUDA).",
        tuple(missing),
    )


def proprietary_gpu_allowed() -> bool:
    """Kurzform für die Backend-Kette: NVIDIA-Laufzeit freigegeben?

    Nur relevant, wenn die Laufzeit tatsächlich mitgeliefert wird. Nutzt der
    Kunde einen selbst installierten Treiber samt CUDA, ist keine Weitergabe
    durch uns im Spiel – dann greift die Zustimmung trotzdem, weil das
    Bundle die Bibliotheken mitbringt.
    """
    if not bundles_cuda():
        return True
    return gate(["nvidia-cuda", "nvidia-cudnn"]).allowed


def bundles_cuda() -> bool:
    """Liegt eine mitgelieferte CUDA-Laufzeit neben der .exe?"""
    for candidate in (
        paths.exe_dir / "cuda",
        paths.exe_dir / "_internal" / "cuda",
        paths.bundle_dir / "cuda",
    ):
        if candidate.is_dir() and any(candidate.glob("*.dll")):
            return True
    return False


# ---------------------------------------------------------------------------
# Sprecher-Einwilligung für angelernte Stimmen
# ---------------------------------------------------------------------------
CONSENT_DECLARATION_DE = (
    "Ich, {speaker}, willige ein, dass meine Sprachaufnahmen zum Anlernen "
    "einer synthetischen Stimme in dieser Anwendung verwendet werden. "
    "Zweck: {purpose}. Die Einwilligung kann jederzeit widerrufen werden; "
    "danach ist das Stimmprofil zu löschen."
)


@dataclass(frozen=True)
class SpeakerConsent:
    """Nachweis der Einwilligung zu einer angelernten Stimme.

    Der Nachweis wird beim Profil abgelegt. ``declaration_hash`` bindet den
    Wortlaut an den Nachweis – wird der Text später verändert, fällt das auf.
    """

    speaker_name: str
    purpose: str
    granted_by: str  # wer die Einwilligung eingeholt hat (Bediener)
    granted_at: float
    declaration: str
    declaration_hash: str
    app_version: str = __version__
    self_recorded: bool = True  # eigene Stimme des Bedieners?
    evidence_note: str = ""  # Verweis auf schriftliche Einwilligung, Aktenzeichen

    @staticmethod
    def create(
        speaker_name: str,
        purpose: str,
        granted_by: str,
        self_recorded: bool = True,
        evidence_note: str = "",
    ) -> SpeakerConsent:
        speaker = speaker_name.strip()
        if not speaker:
            raise ValueError("Sprechername fehlt – ohne Namen kein Nachweis.")
        purpose_clean = purpose.strip() or "interne Nutzung"
        declaration = CONSENT_DECLARATION_DE.format(speaker=speaker, purpose=purpose_clean)
        digest = hashlib.sha256(declaration.encode("utf-8")).hexdigest()
        return SpeakerConsent(
            speaker_name=speaker,
            purpose=purpose_clean,
            granted_by=granted_by.strip() or "unbekannt",
            granted_at=time.time(),
            declaration=declaration,
            declaration_hash=digest,
            self_recorded=self_recorded,
            evidence_note=evidence_note.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "speaker_name": self.speaker_name,
            "purpose": self.purpose,
            "granted_by": self.granted_by,
            "granted_at": self.granted_at,
            "granted_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.granted_at)),
            "declaration": self.declaration,
            "declaration_hash": self.declaration_hash,
            "app_version": self.app_version,
            "self_recorded": self.self_recorded,
            "evidence_note": self.evidence_note,
        }
        return data

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> SpeakerConsent | None:
        try:
            consent = SpeakerConsent(
                speaker_name=str(data["speaker_name"]),
                purpose=str(data.get("purpose", "")),
                granted_by=str(data.get("granted_by", "")),
                granted_at=float(data.get("granted_at", 0) or 0),
                declaration=str(data.get("declaration", "")),
                declaration_hash=str(data.get("declaration_hash", "")),
                app_version=str(data.get("app_version", "")),
                self_recorded=bool(data.get("self_recorded", True)),
                evidence_note=str(data.get("evidence_note", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None
        return consent if consent.is_valid() else None

    def is_valid(self) -> bool:
        """Nachweis prüfen: Name, Zeit und unveränderter Wortlaut."""
        if not self.speaker_name or self.granted_at <= 0 or not self.declaration:
            return False
        digest = hashlib.sha256(self.declaration.encode("utf-8")).hexdigest()
        return digest == self.declaration_hash


def voice_clone_gate(consent: SpeakerConsent | None) -> GateResult:
    """Darf ein angelerntes Stimmprofil benutzt werden? Fail-closed."""
    component_gate = gate("voice-cloning")
    if not component_gate.allowed:
        return component_gate
    if consent is None:
        return GateResult(
            False,
            "Kein Einwilligungs-Nachweis für dieses Stimmprofil. Ohne "
            "Einwilligung der sprechenden Person wird die Stimme nicht genutzt.",
            ("speaker-consent",),
        )
    if not consent.is_valid():
        return GateResult(
            False,
            "Einwilligungs-Nachweis unvollständig oder nachträglich verändert – "
            "Stimmprofil wird nicht genutzt.",
            ("speaker-consent",),
        )
    return GateResult(True, f"Einwilligung von {consent.speaker_name} liegt vor.")


# ---------------------------------------------------------------------------
# Anzeige
# ---------------------------------------------------------------------------
def summary() -> str:
    """Mehrzeilige Übersicht für CLI und GUI-Lizenzseite."""
    lines: list[str] = []
    consent = store()
    for key, item in sorted(COMPONENTS.items()):
        mark = "zugestimmt" if consent.is_accepted(key) else "offen"
        lines.append(f"[{mark}] {item.title}")
        lines.append(f"    Lizenz:  {item.license_id}")
        if item.license_url:
            lines.append(f"    Text:    {item.license_url}")
        lines.append(f"    Zweck:   {item.why}")
        for obligation in item.obligations:
            lines.append(f"    Auflage: {obligation}")
    notices = paths.notices_path()
    lines.append("")
    lines.append(
        f"Vollständige Hinweise: {notices}"
        if notices.is_file()
        else f"WARNUNG: THIRD-PARTY-NOTICES.md fehlt ({notices})."
    )
    return "\n".join(lines)
