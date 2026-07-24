import io
import tempfile
import threading
import time
import unittest

from services.runtime.async_logging import AsyncLogger


class AsyncLoggingTests(unittest.TestCase):
    def test_log_does_not_wait_for_slow_output(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(directory, "test", stdout=io.StringIO())
            entered = threading.Event()
            release = threading.Event()

            def slow_write(_record):
                entered.set()
                release.wait(2)

            logger._write = slow_write
            started = time.monotonic()
            logger.log("WEIGHT", "123 kg")
            elapsed = time.monotonic() - started

            self.assertTrue(entered.wait(1))
            self.assertLess(elapsed, 0.1)
            release.set()
            logger.close()

    def test_prominent_tag_is_not_padded_or_truncated(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(directory, "test", stdout=io.StringIO())
            logger._ensure_thread = lambda: None
            logger.log(">>> SENT <<<", "plate=14C-017.80")
            record = logger._queue.get_nowait()

            self.assertIn("[>>> SENT <<<] plate=14C-017.80", record[1][0])


if __name__ == "__main__":
    unittest.main()
