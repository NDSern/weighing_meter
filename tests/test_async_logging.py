import io
import os
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

    def test_writes_logs_inside_date_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(directory, "test", stdout=io.StringIO())
            logger._ensure_thread = lambda: None
            now = unittest.mock.Mock()
            now.strftime.return_value = "2026-07-14"

            self.assertEqual(
                logger._path_for_date("2026-07-14"),
                os.path.join(directory, "2026-07-14", "test.log"),
            )

    def test_full_queue_drops_oldest_record(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(
                directory, "test", stdout=io.StringIO(), stderr=io.StringIO(), queue_size=2
            )
            logger._ensure_thread = lambda: None

            logger.log("INFO", "one")
            logger.log("INFO", "two")
            logger.log("INFO", "three")

            records = [logger._queue.get_nowait(), logger._queue.get_nowait()]
            self.assertNotIn("one", records[0][1][0])
            self.assertIn("two", records[0][1][0])
            self.assertIn("three", records[1][1][0])
            self.assertEqual(logger._dropped, 1)

    def test_queue_overflow_warns_once_and_reports_dropped_on_close(self):
        with tempfile.TemporaryDirectory() as directory:
            stderr = io.StringIO()
            logger = AsyncLogger(
                directory, "test", stdout=io.StringIO(), stderr=stderr, queue_size=1
            )
            logger._ensure_thread = lambda: None

            logger.log("INFO", "one")
            logger.log("INFO", "two")
            logger.log("INFO", "three")

            self.assertEqual(logger._dropped, 2)
            self.assertEqual(stderr.getvalue().count("queue overflow"), 1)
            self.assertTrue(logger.close())
            self.assertIn("dropped 2 log records", stderr.getvalue())

    def test_close_clears_thread_and_prevents_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncLogger(directory, "test", stdout=io.StringIO())
            self.assertTrue(logger.close())

            logger._ensure_thread()
            self.assertIsNone(logger._thread)

            logger.log("INFO", "late")
            self.assertIsNone(logger._thread)
            self.assertTrue(logger._queue.empty())

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
