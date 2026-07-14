import os
import tempfile
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

from services.storage.dead_letter import is_expired
from services.storage.retention_cleaner import StorageMaintenance


class DeadLetterTests(unittest.TestCase):
    def test_pending_expires_at_retention_boundary(self):
        now = datetime(2026, 7, 14, 12, 0, 0)
        created = (now - timedelta(days=30)).isoformat()

        self.assertTrue(is_expired(created, 30, now=now))
        self.assertFalse(is_expired((now - timedelta(days=29)).isoformat(), 30, now=now))


class StorageMaintenanceTests(unittest.TestCase):
    def test_removes_only_matching_expired_files(self):
        with tempfile.TemporaryDirectory() as root:
            old_log = os.path.join(root, "weighing_service_2026-01-01.log")
            recent_log = os.path.join(root, "weighing_service_2026-07-13.log")
            unrelated = os.path.join(root, "other.log")
            for path in (old_log, recent_log, unrelated):
                open(path, "w").close()
            now = time.time()
            os.utime(old_log, (now - 61 * 86400, now - 61 * 86400))
            os.utime(recent_log, (now - 1 * 86400, now - 1 * 86400))
            os.utime(unrelated, (now - 365 * 86400, now - 365 * 86400))

            with patch("services.storage.retention_cleaner.LOG_DIR", root):
                result = StorageMaintenance(86400).run_once(now=now)

            self.assertEqual(result["logs_deleted"], 1)
            self.assertFalse(os.path.exists(old_log))
            self.assertTrue(os.path.exists(recent_log))
            self.assertTrue(os.path.exists(unrelated))


if __name__ == "__main__":
    unittest.main()
