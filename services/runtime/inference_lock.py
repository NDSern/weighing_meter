"""Inference serialization with live-work priority."""

import threading
from contextlib import contextmanager


class PriorityInferenceLock:
    """Non-preemptive lock that admits queued live work before deferred work."""

    def __init__(self):
        self._condition = threading.Condition()
        self._active = False
        self._live_waiters = 0
        self._deferred_waiters = 0
        self._live_burst = 0

    def __enter__(self):
        with self._condition:
            self._live_waiters += 1
            try:
                while self._active or (self._deferred_waiters and self._live_burst):
                    self._condition.wait()
                self._active = True
                self._live_burst += 1
            finally:
                self._live_waiters -= 1
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._release()

    @contextmanager
    def deferred(self):
        with self._condition:
            if not self._deferred_waiters:
                self._live_burst = 0
            self._deferred_waiters += 1
            try:
                while self._active or (self._live_waiters and not self._live_burst):
                    self._condition.wait()
                self._active = True
                self._live_burst = 0
            finally:
                self._deferred_waiters -= 1
        try:
            yield self
        finally:
            self._release()

    def _release(self):
        with self._condition:
            self._active = False
            self._condition.notify_all()
