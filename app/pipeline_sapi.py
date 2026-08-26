"""Sprachausgabe über die Windows-Stimmen (SAPI).

Gebaut für das **Telefonat**, wo die Verzögerung alles entscheidet.

Gemessen auf diesem Rechner:

    Bark small   ~20 s je Satz   → erster Ton nach 21 s
    Windows-SAPI  0,6 s je Satz  → erster Ton nach unter 1 s

Bark klingt deutlich natürlicher und bleibt die richtige Wahl, wenn eine
Datei entsteht, auf die niemand wartet. Für ein Gespräch ist es
unbrauchbar: zwanzig Sekunden Stille nach jeder Frage hält niemand aus.

Die Windows-Stimmen sind bereits installiert (Hedda, Katja, Stefan für
Deutsch), brauchen keinen Download, kein Modell und werfen keine
Lizenzfrage auf – anders als Piper, das unter GPL-3.0 steht und deshalb
in dieser Anwendung nicht eingebettet werden darf.

Welche davon sichtbar sind, hängt an der aufgerufenen PowerShell: siehe
``_SHELLS``. Mit der falschen bleiben Katja und Stefan unsichtbar.

Angesprochen wird SAPI über einen kurzen PowerShell-Aufruf statt über
``win32com``: das Paket ist hier nicht installiert, PowerShell dagegen
immer vorhanden. Der Aufruf kostet rund 0,3 s Startzeit – gegenüber
zwanzig Sekunden ist das nicht der Rede wert.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .accel import clean_error

log = logging.getLogger(__name__)

# Wie lange ein einzelner Satz höchstens dauern darf. Bleibt SAPI hängen,
# soll das Gespräch weiterlaufen statt einzufrieren.
TIMEOUT_S = 30.0


class SapiUnavailable(RuntimeError):
    """Windows-Sprachausgabe nicht nutzbar."""

    expected = True


@dataclass(frozen=True)
class SapiVoice:
    """Eine installierte Windows-Stimme."""

    name: str
    culture: str

    @property
    def is_german(self) -> bool:
        return self.culture.lower().startswith("de")

    def label(self) -> str:
        return f"Windows: {self.name} ({self.culture})"


# Welche PowerShell genommen wird, entscheidet über die Zahl der Stimmen.
#
# Gemessen: 'powershell' (5.1, .NET Framework) zeigt 2 Stimmen, 'pwsh' (7,
# .NET) zeigt 5 – Katja und Stefan sind nur über die zweite sichtbar. Die
# moderneren Stimmen stehen unter Speech_OneCore\Voices, und nur die
# .NET-Portierung von System.Speech liest diesen Registry-Pfad mit.
_SHELLS = ("pwsh", "powershell")
_shell_cache: str = ""


def _shell() -> str:
    """Die beste vorhandene PowerShell. Wird einmal ermittelt."""
    global _shell_cache
    if _shell_cache:
        return _shell_cache
    import shutil as _shutil

    for name in _SHELLS:
        if _shutil.which(name):
            _shell_cache = name
            return name
    _shell_cache = "powershell"  # letzte Hoffnung; der Aufruf meldet den Fehler
    return _shell_cache


def _powershell(script: str, timeout: float = 15.0) -> tuple[bool, str]:
    """PowerShell aufrufen. Nie werfend – Rückgabe (ok, Ausgabe)."""
    try:
        fertig = subprocess.run(
            [_shell(), "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
    except FileNotFoundError:
        return False, "PowerShell nicht gefunden."
    except subprocess.TimeoutExpired:
        return False, f"PowerShell hat nach {timeout:g}s nicht geantwortet."
    except OSError as exc:
        return False, clean_error(exc)
    if fertig.returncode != 0:
        return False, (fertig.stderr or fertig.stdout or "unbekannter Fehler").strip()[:200]
    return True, (fertig.stdout or "").strip()


def available() -> tuple[bool, str]:
    """Gibt es nutzbare Windows-Stimmen?"""
    if os.name != "nt":
        return False, "Windows-Stimmen gibt es nur unter Windows."
    stimmen = voices()
    if not stimmen:
        return False, "Keine Windows-Stimme installiert (Einstellungen → Sprache)."
    deutsch = [s for s in stimmen if s.is_german]
    return True, (
        f"{len(stimmen)} Windows-Stimme(n), davon {len(deutsch)} deutsch."
        if deutsch
        else f"{len(stimmen)} Windows-Stimme(n), keine deutsche."
    )


_cache: list[SapiVoice] | None = None


def voices(refresh: bool = False) -> list[SapiVoice]:
    """Installierte Stimmen. Wird zwischengespeichert – die Abfrage kostet."""
    global _cache
    if _cache is not None and not refresh:
        return _cache
    if os.name != "nt":
        _cache = []
        return _cache
    ok, ausgabe = _powershell(
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$s.GetInstalledVoices() | ForEach-Object "
        '{ "$($_.VoiceInfo.Name)|$($_.VoiceInfo.Culture.Name)" }; '
        "$s.Dispose()"
    )
    gefunden: list[SapiVoice] = []
    if ok:
        for zeile in ausgabe.splitlines():
            name, _, kultur = zeile.partition("|")
            if name.strip():
                gefunden.append(SapiVoice(name=name.strip(), culture=kultur.strip()))
    else:
        log.debug("Stimmenabfrage fehlgeschlagen: %s", ausgabe)

    # Reihenfolge fuer die Auswahlliste: deutsche Stimmen zuerst, und die
    # alten "Desktop"-Fassungen hinter die moderneren. Wer eine deutsche
    # Stimme sucht, soll sie nicht zwischen englischen suchen muessen.
    def rang(stimme: SapiVoice) -> tuple[int, int, str]:
        return (
            0 if stimme.is_german else 1,
            1 if stimme.name.endswith("Desktop") else 0,
            stimme.name,
        )

    _cache = sorted(gefunden, key=rang)
    return _cache


def best_voice(language: str = "de") -> SapiVoice | None:
    """Passende Stimme wählen: erst die Sprache, sonst irgendeine."""
    alle = voices()
    if not alle:
        return None
    passend = [s for s in alle if s.culture.lower().startswith(language.lower()[:2])]
    return (passend or alle)[0]


def _escape(text: str) -> str:
    """Text für ein einfach gequotetes PowerShell-Literal absichern."""
    return text.replace("'", "''")


def speak_to_file(
    text: str,
    target: Path,
    voice: str = "",
    rate: int = 0,
    volume: int = 100,
) -> Path:
    """Text zu einer WAV-Datei sprechen.

    ``rate`` läuft von -10 (langsam) bis 10 (schnell), ``volume`` von 0
    bis 100 – so gibt SAPI es vor.
    """
    if os.name != "nt":
        raise SapiUnavailable("Windows-Stimmen gibt es nur unter Windows.")
    paths.ensure_dir(target.parent)

    auswahl = f"$s.SelectVoice('{_escape(voice)}'); " if voice else ""
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"{auswahl}"
        f"$s.Rate = {int(max(-10, min(10, rate)))}; "
        f"$s.Volume = {int(max(0, min(100, volume)))}; "
        f"$s.SetOutputToWaveFile('{_escape(str(target))}'); "
        f"$s.Speak('{_escape(text)}'); "
        "$s.Dispose()"
    )
    ok, meldung = _powershell(script, timeout=TIMEOUT_S)
    if not ok:
        raise SapiUnavailable(f"Windows-Sprachausgabe fehlgeschlagen: {meldung}")
    if not target.is_file() or target.stat().st_size < 64:
        raise SapiUnavailable("Windows-Sprachausgabe hat keine Datei geschrieben.")
    return target


def wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as datei:
            return datei.getnframes() / float(datei.getframerate() or 1)
    except (wave.Error, OSError):
        return 0.0


# ---------------------------------------------------------------------------
# Als VoicePipeline
# ---------------------------------------------------------------------------
def build_pipeline(config, plan):
    """Eine ``VoicePipeline`` auf Basis der Windows-Stimmen.

    Wird spät gebaut, damit ``pipeline_voice`` nicht schon beim Import
    dieses Moduls geladen werden muss.
    """
    from .pipeline_voice import VoicePipeline as _Basis
    from .pipeline_voice import VoiceRequest, VoiceResult, output_path

    class SapiVoicePipeline(_Basis):
        """Sprachausgabe über die installierten Windows-Stimmen."""

        def load(self, context) -> None:
            ok, grund = available()
            if not ok:
                raise SapiUnavailable(grund)
            self._loaded = True
            gewaehlt = best_voice(self.config.language)
            context.status(
                f"Windows-Stimme bereit: {gewaehlt.name if gewaehlt else 'Systemvorgabe'}."
            )

        def synthesize(self, request: VoiceRequest, context) -> VoiceResult:
            if not self._loaded:
                self.load(context)
            begonnen = time.time()

            # Der Sprecher kann eine Windows-Stimme benennen; sonst wird die
            # beste passende genommen.
            name = ""
            for stimme in voices():
                if request.speaker and stimme.name.lower() == request.speaker.lower():
                    name = stimme.name
                    break
            if not name:
                gewaehlt = best_voice(request.language or self.config.language)
                name = gewaehlt.name if gewaehlt else ""

            ziel = output_path(request, suffix="wav")
            # SAPI kennt keine Geschwindigkeit als Faktor, sondern eine
            # Stufe von -10 bis 10. 1,0 entspricht 0.
            stufe = round((request.speed - 1.0) * 10)
            speak_to_file(request.text, ziel, voice=name, rate=stufe)

            sekunden = wav_seconds(ziel)
            return VoiceResult(
                audio=ziel,
                seconds=sekunden,
                sample_rate=22_050,
                backend="sapi",
                model_key="windows-sapi",
                profile_slug="",
                elapsed_s=time.time() - begonnen,
                dummy=False,
                notes=(
                    f"Windows-Stimme '{name or 'Systemvorgabe'}' – schnell, "
                    "aber weniger natürlich als ein Modell.",
                ),
            )

    return SapiVoicePipeline(config, plan)


def describe() -> str:
    ok, grund = available()
    zeilen = [grund]
    if ok:
        for stimme in voices():
            zeilen.append(f"  {stimme.label()}")
    return "\n".join(zeilen)
