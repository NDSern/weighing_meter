import queue
import sys
import tempfile
import time
import unittest
from types import ModuleType
from unittest import mock

if "cv2" not in sys.modules:
    sys.modules["cv2"] = ModuleType("cv2")
if "minio" not in sys.modules:
    minio = ModuleType("minio")
    minio.Minio = mock.Mock
    minio_error = ModuleType("minio.error")
    minio_error.S3Error = RuntimeError
    sys.modules["minio"] = minio
    sys.modules["minio.error"] = minio_error

from services.storage import image_save_worker as module
from services.storage.image_save_worker import ImageSaveWorker


class ImageSaveWorkerResourceTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile(delete=False)
        self.pending_file = handle.name
        handle.close()
        self.queue = queue.Queue(maxsize=2)
        self.pending = {}
        self.queued = set()
        self.patches = [
            mock.patch.object(module, "_pending_file", self.pending_file),
            mock.patch.object(module, "_upload_queue", self.queue),
            mock.patch.object(module, "_pending_tasks", self.pending),
            mock.patch.object(module, "_queued_keys", self.queued),
        ]
        for patcher in self.patches:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.patches):
            patcher.stop()

    def test_ready_queue_is_bounded_while_all_tasks_remain_durable(self):
        for index in range(5):
            ImageSaveWorker._enqueue_upload("/tmp/%d.jpg" % index, "key-%d" % index)

        self.assertEqual(self.queue.qsize(), 2)
        self.assertEqual(len(self.pending), 5)
        self.assertEqual(len(self.queued), 2)

    def test_retry_backoff_is_capped_and_persisted(self):
        task = {"object_key": "key", "fpath": "/tmp/x", "attempts": 0,
                "next_attempt_at": 0.0, "created_at": "2026-01-01T00:00:00"}
        self.pending["key"] = task

        with mock.patch.object(module.time, "time", return_value=1000):
            ImageSaveWorker._schedule_retry(task)
            first_delay = task["next_attempt_at"] - 1000
            for _ in range(20):
                ImageSaveWorker._schedule_retry(task)

        self.assertEqual(first_delay, 2)
        self.assertLessEqual(task["next_attempt_at"] - 1000, 300)
        self.assertGreater(task["attempts"], 1)

    def test_refill_skips_delayed_retry_and_queues_fresh_task(self):
        delayed = {"object_key": "delayed", "fpath": "/tmp/a", "next_attempt_at": 2000}
        fresh = {"object_key": "fresh", "fpath": "/tmp/b", "next_attempt_at": 0}
        self.pending.update(delayed=delayed, fresh=fresh)

        with mock.patch.object(module.time, "time", return_value=1000):
            ImageSaveWorker._fill_upload_queue_locked()

        self.assertEqual(self.queue.get_nowait()["object_key"], "fresh")
        self.assertEqual(self.queued, {"fresh"})

    def test_load_preserves_retry_backoff(self):
        with open(self.pending_file, "w", encoding="utf-8") as handle:
            handle.write('{"fpath":"/tmp/a","object_key":"key","created_at":"2026-01-01T00:00:00","attempts":4,"next_attempt_at":2000}\n')

        with mock.patch.object(module.os.path, "exists", return_value=True), \
             mock.patch.object(module.time, "time", return_value=1000):
            ImageSaveWorker._load_pending_uploads()

        self.assertEqual(self.pending["key"]["attempts"], 4)
        self.assertEqual(self.pending["key"]["next_attempt_at"], 2000)
        self.assertTrue(self.queue.empty())


if __name__ == "__main__":
    unittest.main()
