import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from services.storage.dead_letter import is_expired
from services.storage.retention_cleaner import ImageRetentionCleaner, StorageMaintenance


class DeadLetterTests(unittest.TestCase):
    def test_pending_expires_at_retention_boundary(self):
        now = datetime(2026, 7, 14, 12, 0, 0)
        created = (now - timedelta(days=30)).isoformat()

        self.assertTrue(is_expired(created, 30, now=now))
        self.assertFalse(is_expired((now - timedelta(days=29)).isoformat(), 30, now=now))


class StorageMaintenanceTests(unittest.TestCase):
    def test_removes_only_matching_expired_files(self):
        with tempfile.TemporaryDirectory() as root:
            now = datetime(2026, 7, 14, 12, 0, 0).timestamp()
            old_dir = os.path.join(root, "2026-05-13")
            recent_dir = os.path.join(root, "2026-07-13")
            os.makedirs(old_dir)
            os.makedirs(recent_dir)
            old_log = os.path.join(old_dir, "weighing_service.log")
            recent_log = os.path.join(recent_dir, "weighing_service.log")
            unrelated = os.path.join(root, "other.log")
            for path in (old_log, recent_log, unrelated):
                open(path, "w").close()
            with patch("services.storage.retention_cleaner.LOG_DIR", root):
                result = StorageMaintenance(86400).run_once(now=now)

            self.assertEqual(result["logs_deleted"], 1)
            self.assertFalse(os.path.exists(old_log))
            self.assertTrue(os.path.exists(recent_log))
            self.assertTrue(os.path.exists(unrelated))

    def test_scale_retention_keeps_archive_and_removes_sidecars(self):
        with tempfile.TemporaryDirectory() as root:
            for name in (
                "2025-07-13.db", "2025-07-13.db-wal", "2025-07-13.db-shm",
                "2026-07-13.db", "scale_data.archive.db",
            ):
                open(os.path.join(root, name), "w").close()
            now = datetime(2026, 7, 14, 12, 0, 0).timestamp()

            with patch("services.storage.retention_cleaner.SCALE_DATA_DIR", root):
                result = StorageMaintenance(86400).run_once(now=now)

            self.assertEqual(result["scale_databases_deleted"], 3)
            self.assertTrue(os.path.exists(os.path.join(root, "2026-07-13.db")))
            self.assertTrue(os.path.exists(os.path.join(root, "scale_data.archive.db")))

    def test_retention_removes_cutoff_date(self):
        with tempfile.TemporaryDirectory() as logs, tempfile.TemporaryDirectory() as scale_data:
            os.makedirs(os.path.join(logs, "2026-05-15"))
            open(os.path.join(logs, "2026-05-15", "weighing_service.log"), "w").close()
            open(os.path.join(scale_data, "2025-07-14.db"), "w").close()
            now = datetime(2026, 7, 14, 12, 0, 0).timestamp()

            with patch("services.storage.retention_cleaner.LOG_DIR", logs), patch(
                "services.storage.retention_cleaner.SCALE_DATA_DIR", scale_data,
            ):
                result = StorageMaintenance(86400).run_once(now=now)

            self.assertEqual(result["logs_deleted"], 1)
            self.assertEqual(result["scale_databases_deleted"], 1)

    def test_diagnostic_path_date_removes_images_and_metadata_after_30_days(self):
        with tempfile.TemporaryDirectory() as root:
            old_dir = os.path.join(root, "2026", "06", "01")
            recent_dir = os.path.join(root, "2026", "07", "14")
            os.makedirs(old_dir)
            os.makedirs(recent_dir)
            old_image = os.path.join(old_dir, "attempt_cam1.jpg")
            old_metadata = os.path.join(old_dir, "attempt.json")
            recent_image = os.path.join(recent_dir, "session_cam1.jpg")
            for path in (old_image, old_metadata, recent_image):
                open(path, "w").close()
            now = datetime(2026, 7, 15, 12, 0, 0).timestamp()
            cleaner = ImageRetentionCleaner(
                [root], 30, 86400, {".jpg", ".json"}
            )

            result = cleaner.run_once(now=now)

            self.assertEqual(result["deleted"], 2)
            self.assertEqual(result["deleted_by_path_date"], 2)
            self.assertFalse(os.path.exists(old_image))
            self.assertFalse(os.path.exists(old_metadata))
            self.assertTrue(os.path.exists(recent_image))


if __name__ == "__main__":
    unittest.main()
