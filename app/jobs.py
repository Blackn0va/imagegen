"""Auftragswarteschlange: Hintergrund-Thread, Fortschritt, Abbruch.

Generierung dauert Minuten. Die Oberfläche darf nie blockieren, deshalb:
  * ein (konfigurierbar mehrere) Arbeiter-Thread(s), Vorgabe 1 – zwei
    Aufträge würden sich denselben VRAM streitig machen
  * Fortschritt ausschließlich als Rückruf, nie als Rückgabewert
  * ``should_stop`` wird in jede lange Schleife hineingereicht
  * gleiche Fehlermeldung wird gedrosselt geloggt, nicht hundertfach
  * Beenden schließt die Queue sauber und joint die Threads
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)


class JobCancelled(RuntimeError):
    """Auftrag wurde abgebrochen. Wird bis zum Arbeiter durchgereicht."""


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobState.DONE, JobState.FAILED, JobState.CANCELLED)

    def label(self) -> str:
        return {
            JobState.PENDING: "wartet",
            JobState.RUNNING: "läuft",
            JobState.DONE: "fertig",
            JobState.FAILED: "fehlgeschlagen",
            JobState.CANCELLED: "abgebrochen",
        }[self]


def _spent_handler(_context: Any) -> None:
    """Platzhalter für erledigte Aufträge.

    Ersetzt den echten Handler, sobald der Auftrag durch ist. Hält als
    Funktion auf Modulebene nichts fest – anders als die Closure, die er
    ablöst.
    """
    return None


@dataclass
class Job:
    """Ein Auftrag. Die Felder werden nur unter dem Queue-Lock geschrieben."""

    id: str
    kind: str  # image | video | voice | download | train | compose
    title: str
    handler: Callable[[JobContext], Any]
    payload: dict[str, Any] = field(default_factory=dict)

    state: JobState = JobState.PENDING
    fraction: float = 0.0
    message: str = "in der Warteschlange"
    result: Any = None
    error: str = ""

    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0

    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    # --- Abbruch -----------------------------------------------------------
    def request_cancel(self) -> None:
        self._cancel.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel.is_set()

    @property
    def duration(self) -> float:
        end = self.finished_at or time.time()
        if not self.started_at:
            return 0.0
        return max(0.0, end - self.started_at)

    def snapshot(self) -> JobView:
        return JobView(
            id=self.id,
            kind=self.kind,
            title=self.title,
            state=self.state,
            fraction=self.fraction,
            message=self.message,
            error=self.error,
            result=self.result,
            created_at=self.created_at,
            duration=self.duration,
            payload=dict(self.payload),
        )


@dataclass(frozen=True)
class JobView:
    """Unveränderliche Sicht für die Oberfläche – kein Zugriff auf Interna."""

    id: str
    kind: str
    title: str
    state: JobState
    fraction: float
    message: str
    error: str
    result: Any
    created_at: float
    duration: float
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class JobEvent:
    """Ereignis für Zuhörer. Wird im Arbeiter-Thread erzeugt – die GUI muss
    es in ihren eigenen Thread umziehen (siehe gui/main_window.py)."""

    event: str  # submitted | started | progress | status | log | finished
    job: JobView
    timestamp: float = field(default_factory=time.time)
    level: int = logging.INFO
    text: str = ""


Listener = Callable[[JobEvent], None]


class _ErrorThrottle:
    """Gleiche Meldung nicht hundertfach loggen.

    Erste Meldung geht sofort durch, Wiederholungen werden gezählt und
    erst nach ``interval`` Sekunden als Sammelmeldung ausgegeben.
    """

    def __init__(self, interval: float = 5.0) -> None:
        self.interval = max(0.1, interval)
        self._lock = threading.Lock()
        self._last: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            last = self._last.get(key, 0.0)
            if now - last >= self.interval:
                suppressed = self._counts.pop(key, 0)
                self._last[key] = now
                return True, suppressed
            self._counts[key] = self._counts.get(key, 0) + 1
            return False, self._counts[key]

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._last.clear()
                self._counts.clear()
            else:
                self._last.pop(key, None)
                self._counts.pop(key, None)


class JobContext:
    """Was ein Auftrags-Handler zu sehen bekommt.

    Handler-Vertrag:
      * lange Schleifen fragen ``should_stop()`` oder rufen ``raise_if_cancelled()``
      * Fortschritt über ``progress()`` melden (0.0 … 1.0)
      * Zwischenmeldungen über ``status()``
      * wiederkehrende Fehler über ``log_error()`` – die Drosselung sitzt hier
    """

    def __init__(self, queue_ref: JobQueue, job: Job) -> None:
        self._queue = queue_ref
        self._job = job

    # --- Kennung -----------------------------------------------------------
    @property
    def job_id(self) -> str:
        return self._job.id

    @property
    def kind(self) -> str:
        return self._job.kind

    @property
    def payload(self) -> dict[str, Any]:
        return self._job.payload

    # --- Abbruch -----------------------------------------------------------
    def should_stop(self) -> bool:
        return self._job.cancel_requested or self._queue.is_shutting_down

    def raise_if_cancelled(self) -> None:
        if self.should_stop():
            raise JobCancelled("Abbruch angefordert")

    # --- Meldungen ---------------------------------------------------------
    def progress(self, fraction: float, message: str | None = None) -> None:
        self._queue._update_progress(self._job, fraction, message)

    def progress_steps(self, done: int, total: int, message: str | None = None) -> None:
        fraction = (done / total) if total > 0 else 0.0
        text = message or f"Schritt {done}/{total}"
        self._queue._update_progress(self._job, fraction, text)

    def status(self, message: str) -> None:
        self._queue._emit_status(self._job, message)

    def log(self, message: str, level: int = logging.INFO) -> None:
        self._queue._emit_log(self._job, message, level)

    def log_error(self, key: str, message: str) -> None:
        """Gedrosseltes Fehlerlogging. ``key`` gruppiert gleiche Fehler."""
        allow, count = self._queue.throttle.allow(f"{self._job.id}:{key}")
        if allow:
            suffix = f" (zuvor {count}x unterdrückt)" if count else ""
            self._queue._emit_log(self._job, message + suffix, logging.WARNING)

    def sub_context(self) -> JobContext:
        """Für verschachtelte Aufrufe (Download innerhalb einer Generierung)."""
        return self


class JobQueue:
    """Warteschlange mit Arbeiter-Threads."""

    def __init__(
        self,
        workers: int = 1,
        name: str = "jobs",
        error_throttle_seconds: float = 5.0,
    ) -> None:
        self._name = name
        self._worker_count = max(1, int(workers))
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._listeners: list[Listener] = []
        self._threads: list[threading.Thread] = []
        self._shutdown = threading.Event()
        self._started = False
        self.throttle = _ErrorThrottle(error_throttle_seconds)

    # ------------------------------------------------------------------
    # Lebenszyklus
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        self._started = True
        for index in range(self._worker_count):
            thread = threading.Thread(
                target=self._worker_loop,
                name=f"{self._name}-worker-{index}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        log.debug("Warteschlange gestartet: %d Arbeiter", self._worker_count)

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown.is_set()

    def shutdown(
        self, wait: bool = True, timeout: float = 15.0, cancel_running: bool = True
    ) -> None:
        """Queue schließen, laufende Aufträge abbrechen, Threads joinen."""
        if not self._started:
            self._shutdown.set()
            return
        self._shutdown.set()
        if cancel_running:
            with self._lock:
                for job in self._jobs.values():
                    if job.state in (JobState.PENDING, JobState.RUNNING):
                        job.request_cancel()
        # Sentinel je Arbeiter
        for _ in self._threads:
            self._queue.put(None)
        if wait:
            deadline = time.time() + timeout
            for thread in self._threads:
                remaining = max(0.1, deadline - time.time())
                thread.join(remaining)
                if thread.is_alive():
                    log.warning(
                        "Arbeiter %s läuft noch – Auftrag reagiert nicht auf Abbruch.",
                        thread.name,
                    )
        self._threads.clear()
        self._started = False

    # ------------------------------------------------------------------
    # Zuhörer
    # ------------------------------------------------------------------
    def subscribe(self, listener: Listener) -> Listener:
        with self._lock:
            self._listeners.append(listener)
        return listener

    def unsubscribe(self, listener: Listener) -> None:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _emit(self, event: JobEvent) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception as exc:
                log.debug("Zuhörer hat geworfen: %s", exc)  # darf nichts kippen

    # ------------------------------------------------------------------
    # Einreichen und Abfragen
    # ------------------------------------------------------------------
    def submit(
        self,
        kind: str,
        title: str,
        handler: Callable[[JobContext], Any],
        payload: Mapping[str, Any] | None = None,
    ) -> str:
        if self._shutdown.is_set():
            raise RuntimeError("Warteschlange wird beendet – kein neuer Auftrag.")
        self.start()
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            title=title,
            handler=handler,
            payload=dict(payload or {}),
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
        self._queue.put(job.id)
        self._emit(JobEvent("submitted", job.snapshot(), text=title))
        return job.id

    def get(self, job_id: str) -> JobView | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.snapshot() if job else None

    def snapshot(self) -> list[JobView]:
        with self._lock:
            return [self._jobs[jid].snapshot() for jid in self._order if jid in self._jobs]

    def active_count(self) -> int:
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if job.state in (JobState.PENDING, JobState.RUNNING)
            )

    def cancel(self, job_id: str) -> bool:
        """Abbruch anfordern. Wartende Aufträge sofort, laufende über Flag."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.state.finished:
                return False
            job.request_cancel()
            if job.state is JobState.PENDING:
                job.state = JobState.CANCELLED
                job.message = "abgebrochen, bevor er gestartet ist"
                job.finished_at = time.time()
                view = job.snapshot()
            else:
                job.message = "Abbruch angefordert …"
                view = job.snapshot()
        self._emit(
            JobEvent("finished" if view.state.finished else "status", view, text=view.message)
        )
        return True

    def cancel_all(self) -> int:
        count = 0
        for view in self.snapshot():
            if not view.state.finished and self.cancel(view.id):
                count += 1
        return count

    def clear_finished(self) -> int:
        with self._lock:
            done = [jid for jid, job in self._jobs.items() if job.state.finished]
            for jid in done:
                self._jobs.pop(jid, None)
                if jid in self._order:
                    self._order.remove(jid)
        return len(done)

    # ------------------------------------------------------------------
    # Interne Meldewege (vom JobContext benutzt)
    # ------------------------------------------------------------------
    def _update_progress(self, job: Job, fraction: float, message: str | None) -> None:
        clamped = 0.0 if fraction < 0 else (1.0 if fraction > 1 else float(fraction))
        with self._lock:
            job.fraction = clamped
            if message:
                job.message = message
            view = job.snapshot()
        self._emit(JobEvent("progress", view, text=view.message))

    def _emit_status(self, job: Job, message: str) -> None:
        with self._lock:
            job.message = message
            view = job.snapshot()
        self._emit(JobEvent("status", view, text=message))

    def _emit_log(self, job: Job, message: str, level: int) -> None:
        log.log(level, "[%s] %s", job.id, message)
        with self._lock:
            view = job.snapshot()
        self._emit(JobEvent("log", view, level=level, text=message))

    # ------------------------------------------------------------------
    # Arbeiter
    # ------------------------------------------------------------------
    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                self._run_job(item)
            except BaseException as exc:
                # Der Arbeiter darf NIE sterben.
                #
                # Er wird nur einmal gestartet; stirbt er, bleibt jeder
                # weitere Auftrag für immer auf "wartend" und die
                # Anwendung wirkt eingefroren, ohne dass irgendwo ein
                # Fehler steht. Genau das ist über ein KeyboardInterrupt
                # aus dem gekachelten Vergrößern passiert.
                #
                # Deshalb hier BaseException statt Exception: auch
                # KeyboardInterrupt und SystemExit werden abgefangen und
                # kosten höchstens einen Auftrag, nie die Warteschlange.
                log.exception("Auftrag ist hart gescheitert: %s", exc)
                if isinstance(exc, (KeyboardInterrupt, SystemExit)) and self._shutdown.is_set():
                    # Beim Herunterfahren ist das der gewollte Weg hinaus.
                    return
            finally:
                self._queue.task_done()

    def _run_job(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if job.state is not JobState.PENDING or job.cancel_requested:
                # zwischenzeitlich abgebrochen
                if job.state is JobState.PENDING:
                    job.state = JobState.CANCELLED
                    job.finished_at = time.time()
                    job.message = "abgebrochen"
                view = job.snapshot()
                self._emit(JobEvent("finished", view, text=view.message))
                return
            job.state = JobState.RUNNING
            job.started_at = time.time()
            job.message = "gestartet"
            view = job.snapshot()
        self._emit(JobEvent("started", view, text=job.title))

        context = JobContext(self, job)
        try:
            result = job.handler(context)
            with self._lock:
                if job.cancel_requested:
                    job.state = JobState.CANCELLED
                    job.message = "abgebrochen"
                else:
                    job.state = JobState.DONE
                    job.result = result
                    job.fraction = 1.0
                    job.message = "fertig"
        except JobCancelled as exc:
            with self._lock:
                job.state = JobState.CANCELLED
                job.message = str(exc) or "abgebrochen"
        except Exception as exc:
            from .accel import clean_error

            with self._lock:
                job.state = JobState.FAILED
                job.error = clean_error(exc)
                job.message = job.error
            # Bewusste Ablehnungen (Inhaltssperre, fehlende Freigabe) sind
            # kein Programmfehler. Sie werden festgehalten, aber ohne
            # Stacktrace – der sieht sonst nach Absturz aus.
            if getattr(exc, "expected", False):
                log.warning("Auftrag %s (%s) abgelehnt: %s", job.id, job.kind, job.error)
            else:
                log.exception("Auftrag %s (%s) fehlgeschlagen", job.id, job.kind)
        finally:
            with self._lock:
                job.finished_at = time.time()
                # Handler loslassen: die Closure hält Konfiguration,
                # Backend-Plan und die vollständige Anfrage fest. Erledigte
                # Aufträge bleiben für die Oberfläche in der Liste stehen –
                # ihre Nutzlast braucht dort niemand mehr.
                job.handler = _spent_handler
                view = job.snapshot()
                self._prune_finished()
            self.throttle.reset()
            self._emit(JobEvent("finished", view, text=view.message))

    def _prune_finished(self, keep: int = 200) -> None:
        """Zahl der erledigten Aufträge begrenzen. Läuft unter dem Lock.

        Ohne Obergrenze wächst die Liste über eine lange Sitzung endlos
        weiter. Die jüngsten bleiben stehen, weil nur die jemand ansieht.
        """
        finished = [
            jid for jid in self._order if jid in self._jobs and self._jobs[jid].state.finished
        ]
        for jid in finished[: max(0, len(finished) - keep)]:
            self._jobs.pop(jid, None)
            self._order.remove(jid)


# ---------------------------------------------------------------------------
# Kleine Helfer für Handler
# ---------------------------------------------------------------------------
def stepwise(context: JobContext, total: int, label: str = "Schritt") -> Iterable[int]:
    """Zählschleife, die Fortschritt meldet und auf Abbruch reagiert."""
    for index in range(total):
        context.raise_if_cancelled()
        context.progress_steps(index, total, f"{label} {index + 1}/{total}")
        yield index
    context.progress_steps(total, total, f"{label} {total}/{total}")


def make_diffusers_callback(context: JobContext, total_steps: int, label: str = "Diffusion"):
    """Rückruf im diffusers-Format (``callback_on_step_end``).

    Bricht über eine Ausnahme ab – diffusers hat keinen Abbruch-Rückgabewert.
    """

    def _callback(pipe, step_index: int, timestep, callback_kwargs):
        context.raise_if_cancelled()
        context.progress_steps(
            step_index + 1, total_steps, f"{label} {step_index + 1}/{total_steps}"
        )
        return callback_kwargs

    return _callback
