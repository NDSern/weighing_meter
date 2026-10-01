"""Shared daemon-thread worker scaffold.

Every long-running background worker in this service repeats the same shape:
a daemon thread, a stop event, a periodic loop that guards each pass, and a
``_log`` wrapper around an optional host-supplied ``log_fn``. ``BackgroundWorker``
captures that shape once so subclasses only implement ``run_once``.
"""

import threading

DEFAULT_STOP_TIMEOUT = 3.0


class BackgroundWorker:
    """Periodic daemon-thread worker.

    Subclasses set ``worker_name`` (started/stopped log label) and
    ``error_label`` (loop-failure log prefix), implement ``run_once``, and must
    call ``super().__init__(check_interval_seconds, log_fn)``.
    """

    worker_name = "BackgroundWorker"
    error_label = "Background task failed"

    def __init__(self, check_interval_seconds, log_fn=None):
        self.check_interval_seconds = check_interval_seconds
        self.log_fn = log_fn
        self._stop_event = threading.Event()
        self._thread = None

    def _log(self, level, msg):
        if self.log_fn:
            self.log_fn(level, msg)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._announce("started")

    def stop(self, timeout=DEFAULT_STOP_TIMEOUT):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._announce("stopped")

    def _announce(self, verb):
        if self.announce_lifecycle:
            self._log("INFO", f"{self.worker_name} {verb}")

    # Subclasses that suppress lifecycle logging (e.g. storage maintenance)
    # override this to False.
    announce_lifecycle = True

    def _run_loop(self):
        while not self._stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - loop must survive any pass failure
                self._log("ERROR", f"{self.error_label}: {exc}")
            self._stop_event.wait(self.check_interval_seconds)

    def run_once(self):
        raise NotImplementedError
