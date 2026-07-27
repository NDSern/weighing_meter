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

    def test_full_queue_drops_oldest_record(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(directory, "test", stdout=io.StringIO(), queue_size=2)
            logger._ensure_thread = lambda: None

            logger.log("INFO", "one")
            logger.log("INFO", "two")
            logger.log("INFO", "three")

            records = [logger._queue.get_nowait(), logger._queue.get_nowait()]
            self.assertNotIn("one", records[0][1][0])
            self.assertIn("two", records[0][1][0])
            self.assertIn("three", records[1][1][0])
            self.assertEqual(logger._dropped, 1)

    def test_close_can_enqueue_sentinel_when_queue_is_full(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(directory, "test", stdout=io.StringIO(), queue_size=1)
            logger._queue.put_nowait((None, ["old"]))
            thread = unittest.mock.Mock()
            thread.is_alive.side_effect = [True, False]
            logger._thread = thread

            self.assertTrue(logger.close())
            self.assertIs(logger._queue.get_nowait(), logger._sentinel)

    def test_log_after_close_is_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(directory, "test", stdout=io.StringIO())
            logger.close()

            logger.log("INFO", "late")

            self.assertTrue(logger._queue.empty())


if __name__ == "__main__":
    unittest.main()
