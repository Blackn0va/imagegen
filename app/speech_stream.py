"""Satzweise sprechen, während das Modell noch schreibt.

Der Unterschied zwischen einem benutzbaren und einem quälenden Telefonat.

Ohne dieses Modul läuft ein Zug so: das Modell schreibt die ganze Antwort
(2–5 s), danach wird die ganze Antwort gesprochen (mehrere Sekunden), erst
dann hört der Anrufer etwas. Bei vier Sätzen sind das leicht zehn Sekunden
Stille – am Telefon legt man da auf.

Mit diesem Modul beginnt die Stimme, sobald der **erste Satz** fertig ist.
Während er gesprochen wird, schreibt das Modell weiter und der nächste
Satz wird schon erzeugt. Die wahrgenommene Wartezeit sinkt auf die Dauer
des ersten Satzes.

Zwei Feinheiten, die den Unterschied machen:

  * **Code wird nicht gesprochen.** Der Strom wird auf ```-Zäune geprüft;
    was dazwischen steht, geht nie an die Sprachausgabe, sondern in eine
    Datei. Ohne diese Prüfung liest die Stimme Klammern und Einrückungen
    vor.
  * **Abkürzungen beenden keinen Satz.** „z. B." oder „Dr." enden auf
    einen Punkt, sind aber kein Satzende. Ohne diese Ausnahme zerhackt es
    die Sprachausgabe mitten im Satz.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

# Satzende: Punkt, Frage- oder Ausrufezeichen, gefolgt von Leerraum.
_SENTENCE_END = re.compile(r"([.!?…])(\s|$)")

# Zu kurze Bruchstücke lohnen keinen eigenen Sprachlauf – sie werden an
# den nächsten Satz gehängt. Sonst entstehen abgehackte Ein-Wort-Häppchen.
MIN_SENTENCE_CHARS = 12

# Abkürzungen, deren Punkt kein Satzende ist.
_ABBREV = (
    "z. b.",
    "z.b.",
    "d. h.",
    "d.h.",
    "u. a.",
    "u.a.",
    "bzw.",
    "ca.",
    "vgl.",
    "evtl.",
    "ggf.",
    "inkl.",
    "max.",
    "min.",
    "nr.",
    "abb.",
    "dr.",
    "prof.",
    "hr.",
    "fr.",
    "st.",
    "usw.",
    "etc.",
    "bspw.",
)


def looks_like_abbreviation(text: str) -> bool:
    """Endet der Text auf einer Abkürzung statt auf einem Satzende?"""
    kleiner = text.rstrip().lower()
    return any(kleiner.endswith(kurz) for kurz in _ABBREV)


@dataclass
class SentenceSplitter:
    """Zerlegt einen Token-Strom in sprechbare Sätze.

    Zustandsbehaftet: der Strom kommt in beliebig kleinen Stücken, ein
    Satzende kann mitten in einem Stück liegen oder über zwei hinweg.
    """

    puffer: str = ""
    in_code: bool = False
    code_puffer: str = ""
    code_bloecke: list[tuple[str, str]] = field(default_factory=list)
    _code_sprache: str = ""

    def feed(self, stueck: str) -> list[str]:
        """Ein Stück einspeisen. Rückgabe: fertige Sätze zum Sprechen."""
        self.puffer += stueck
        fertige: list[str] = []
        while True:
            satz = self._naechster()
            if satz is None:
                break
            if satz.strip():
                fertige.append(satz.strip())
        return fertige

    def _naechster(self) -> str | None:
        """Nächsten abgeschlossenen Satz herauslösen, sonst None."""
        # Zaun-Wechsel hat Vorrang: solange wir im Code sind, wird nichts
        # gesprochen.
        zaun = self.puffer.find("```")
        if zaun >= 0:
            if not self.in_code:
                davor = self.puffer[:zaun]
                self.puffer = self.puffer[zaun + 3 :]
                self.in_code = True
                self._code_sprache = ""
                self.code_puffer = ""
                return davor if davor.strip() else ""
            # Code-Block endet
            code = self.puffer[:zaun]
            self.puffer = self.puffer[zaun + 3 :]
            self.in_code = False
            sprache, _, rest = code.partition("\n")
            self.code_bloecke.append((sprache.strip().lower(), rest))
            return ""

        if self.in_code:
            return None  # noch mitten im Code – nichts sprechen

        # ALLE Satzenden im Puffer durchgehen, nicht nur das erste.
        #
        # Sonst blockiert ein kurzer erster Teil den Strom für immer: bei
        # "Nimm z. B. diese Lösung hier." wäre der erste Kandidat "Nimm z."
        # – zu kurz, also verworfen –, und weil die Suche jedes Mal wieder
        # vorne beginnt, käme nie ein Satz heraus.
        for treffer in _SENTENCE_END.finditer(self.puffer):
            ende = treffer.end(1)
            kandidat = self.puffer[:ende]
            if looks_like_abbreviation(kandidat):
                continue  # Abkürzung – das nächste Satzende prüfen
            if len(kandidat.strip()) < MIN_SENTENCE_CHARS:
                continue  # zu kurz für einen eigenen Sprachlauf
            self.puffer = self.puffer[ende:]
            return kandidat
        return None

    def finish(self) -> list[str]:
        """Rest herausgeben. Nach dem letzten Token aufrufen."""
        rest: list[str] = []
        if self.in_code and self.puffer.strip():
            # Unabgeschlossener Code-Block – als Block sichern, nicht sprechen.
            sprache, _, code = self.puffer.partition("\n")
            self.code_bloecke.append((sprache.strip().lower(), code))
            self.puffer = ""
            self.in_code = False
        if self.puffer.strip():
            rest.append(self.puffer.strip())
        self.puffer = ""
        return rest


class SpeechQueue:
    """Sätze der Reihe nach sprechen, in einem eigenen Faden.

    Erzeugen und Abspielen laufen getrennt vom Sprachmodell: während Satz
    eins zu hören ist, wird Satz zwei erzeugt und Satz drei geschrieben.
    """

    def __init__(
        self,
        synth: Callable[[str, int], Path | None],
        play: Callable[[Path], None],
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self._synth = synth
        self._play = play
        self._on_error = on_error
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._zaehler = 0
        self.gesprochen: list[Path] = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._lauf, daemon=True)
        self._thread.start()

    def say(self, satz: str) -> None:
        """Satz einreihen. Kehrt sofort zurück."""
        if not self._stop.is_set() and satz.strip():
            self._queue.put(satz)

    def _lauf(self) -> None:
        while not self._stop.is_set():
            try:
                satz = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if satz is None:
                break
            try:
                self._zaehler += 1
                wav = self._synth(satz, self._zaehler)
                if wav is not None and not self._stop.is_set():
                    self.gesprochen.append(wav)
                    self._play(wav)
            except Exception as exc:
                from .accel import clean_error

                log.warning("Satz nicht gesprochen: %s", clean_error(exc))
                if self._on_error is not None:
                    self._on_error(clean_error(exc))
            finally:
                self._queue.task_done()

    def wait(self, timeout: float = 120.0) -> None:
        """Warten, bis alle eingereihten Sätze gesprochen sind."""
        import time

        ende = time.time() + timeout
        while time.time() < ende and not self._stop.is_set():
            if self._queue.unfinished_tasks == 0:
                return
            time.sleep(0.05)

    def stop(self) -> None:
        """Sofort abbrechen – auch mitten im Satz."""
        self._stop.set()
        # Warteschlange leeren, damit nichts nachklingt.
        while True:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._queue.put(None)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()
