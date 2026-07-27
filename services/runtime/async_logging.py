import os
import queue
import sys
import threading
from datetime import datetime


class AsyncLogger:
    def __init__(self, log_dir, file_prefix, stdout=None, stderr=None, queue_size=10000):
        self.log_dir = log_dir
        self.file_prefix = file_prefix
        self.stdout = stdout or sys.stdout
        self.stderr = stderr or sys.stderr
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._sentinel = object()
        self._thread = None
        self._file = None
        self._date = None
        self._closing = False

    def _path_for_date(self, date_text):
        return os.path.join(self.log_dir, f"{self.file_prefix}_{date_text}.log")

    def _ensure_file(self, now):
        date_text = now.strftime("%Y-%m-%d")
        if self._file is not None and self._date == date_text:
            return self._file
        if self._file is not None:
            self._file.close()
        os.makedirs(self.log_dir, exist_ok=True)
        self._file = open(self._path_for_date(date_text), "a", buffering=1)
        self._date = date_text
        return self._file

    def _write(self, record):
        now, rendered = record
        text = "\n".join(rendered) + "\n"
        try:
            log_file = self._ensure_file(now)
            try:
                self.stdout.write(text)
                self.stdout.flush()
            except OSError:
                pass
            log_file.write(text)
            log_file.flush()
        except OSError as exc:
            try:
                self.stderr.write(f"File logging failed: {exc}\n{text}")
                self.stderr.flush()
            except OSError:
                pass

    def _run(self):
        while True:
            record = self._queue.get()
            try:
                if record is self._sentinel:
                    return
                self._write(record)
            finally:
                self._queue.task_done()

    def _ensure_thread(self):
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name="log-writer",
                    daemon=True,
                )
                self._thread.start()

    def log(self, level, message):
        with self._lock:
            if self._closing:
                return
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
        tag = level if len(level) > 7 else f"{level:<7}"
        prefix = f"{timestamp} [{tag}] "
        lines = str(message).replace("\r", "").splitlines() or [""]
        rendered = [
            f"{prefix}{part}"
            if index == 0
            else f"{'':23} [{'':<7}] {part}"
            for index, part in enumerate(lines)
        ]
        self._ensure_thread()
        self._enqueue((now, rendered))

    def _enqueue(self, record):
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                self._dropped += 1

    def close(self, timeout=10.0):
        with self._lock:
            self._closing = True
            thread = self._thread
        if thread and thread.is_alive():
            self._enqueue(self._sentinel)
            thread.join(timeout=timeout)
            if thread.is_alive():
                return False
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                self._date = None
            self._thread = None
        return True
