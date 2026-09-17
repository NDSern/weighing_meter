import os
import json
import tempfile
import time
import unittest
import tarfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from services.storage.dead_letter import is_expired
from services.storage.retention_cleaner import (
    DiagnosticArchiveCleaner, ImageRetentionCleaner, StorageMaintenance, VerifiedMinioCacheCleaner,
)


class DeadLetterTests(unittest.TestCase):
    def test_pending_expires_at_retention_boundary(self):
        now = datetime(2026, 7, 14, 12, 0, 0)
        created = (now - timedelta(days=30)).isoformat()

        self.assertTrue(is_expired(created, 30, now=now))
        self.assertFalse(is_expired((now - timedelta(days=29)).isoformat(), 30, now=now))


class StorageMaintenanceTests(unittest.TestCase):
    def test_minio_cache_cleanup_deletes_only_verified_old_image(self):
        with tempfile.TemporaryDirectory() as root:
            old_dir = os.path.join(root, "2026", "07", "11")
            recent_dir = os.path.join(root, "2026", "07", "12")
            os.makedirs(old_dir)
            os.makedirs(recent_dir)
            old_image = os.path.join(old_dir, "old.jpg")
            recent_image = os.path.join(recent_dir, "recent.jpg")
            with open(old_image, "wb") as handle:
                handle.write(b"verified-image")
            with open(recent_image, "wb") as handle:
                handle.write(b"recent-image")
            client = Mock()
            client.stat_object.return_value = type(
                "Object", (), {"size": os.path.getsize(old_image),
                                 "etag": "b97e8006dad65f5e8fd1da4ba4375b05"}
            )()
            cleaner = VerifiedMinioCacheCleaner(root, 3, 86400, client)

            result = cleaner.run_once(
                now=datetime(2026, 7, 15, 12, 0, 0).timestamp(), client=client,
            )

            self.assertEqual(result["deleted"], 1)
            self.assertFalse(os.path.exists(old_image))
            self.assertTrue(os.path.exists(recent_image))

    def test_minio_cache_cleanup_keeps_pending_or_mismatched_image(self):
        with tempfile.TemporaryDirectory() as service_dir:
            root = os.path.join(service_dir, "storage", "weighbridge")
            old_dir = os.path.join(root, "2026", "07", "11")
            os.makedirs(old_dir)
            pending = os.path.join(old_dir, "pending.jpg")
            mismatched = os.path.join(old_dir, "mismatched.jpg")
            for path in (pending, mismatched):
                with open(path, "wb") as handle:
                    handle.write(b"local-image")
            pending_file = os.path.join(service_dir, "storage", "upload_pending.jsonl")
            with open(pending_file, "w") as handle:
                handle.write(json.dumps({"fpath": pending, "object_key": "pending"}) + "\n")
            client = Mock()
            client.stat_object.return_value = type(
                "Object", (), {"size": os.path.getsize(mismatched),
                                 "etag": "00000000000000000000000000000000"}
            )()
            cleaner = VerifiedMinioCacheCleaner(root, 3, 86400, client)

            with patch("services.storage.retention_cleaner.SERVICE_DIR", service_dir):
                result = cleaner.run_once(
                    now=datetime(2026, 7, 15, 12, 0, 0).timestamp(), client=client,
                )

            self.assertEqual(result["deleted"], 0)
            self.assertEqual(result["skipped_pending"], 1)
            self.assertEqual(result["remote_mismatch"], 1)
            self.assertTrue(os.path.exists(pending))
            self.assertTrue(os.path.exists(mismatched))

    def test_minio_cache_cleanup_keeps_image_when_verification_fails(self):
        with tempfile.TemporaryDirectory() as root:
            old_dir = os.path.join(root, "2026", "07", "11")
            os.makedirs(old_dir)
            image = os.path.join(old_dir, "remote-missing.jpg")
            with open(image, "wb") as handle:
                handle.write(b"local-image")
            client = Mock()
            client.stat_object.side_effect = RuntimeError("object not found")
            cleaner = VerifiedMinioCacheCleaner(root, 3, 86400, client)

            result = cleaner.run_once(
                now=datetime(2026, 7, 15, 12, 0, 0).timestamp(), client=client,
            )

            self.assertEqual(result["deleted"], 0)
            self.assertEqual(result["failed"], 1)
            self.assertTrue(os.path.exists(image))

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

    def test_removes_expired_legacy_rotated_logs(self):
        with tempfile.TemporaryDirectory() as root:
            old_service = os.path.join(root, "weighing_service_2026-05-13.log")
            old_watchdog = os.path.join(root, "resource-watchdog.20260513_235959.jsonl")
            current_watchdog = os.path.join(root, "resource-watchdog.jsonl")
            recent_service = os.path.join(root, "weighing_service_2026-07-13.log")
            for path in (old_service, old_watchdog, current_watchdog, recent_service):
                open(path, "w").close()
            now = datetime(2026, 7, 14, 12, 0, 0).timestamp()

            with patch("services.storage.retention_cleaner.LOG_DIR", root):
                result = StorageMaintenance(86400).run_once(now=now)

            self.assertEqual(result["logs_deleted"], 2)
            self.assertFalse(os.path.exists(old_service))
            self.assertFalse(os.path.exists(old_watchdog))
            self.assertTrue(os.path.exists(current_watchdog))
            self.assertTrue(os.path.exists(recent_service))

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

    def test_diagnostic_archive_replaces_old_day_and_expires_old_archive(self):
        with tempfile.TemporaryDirectory() as root:
            archive_dir = os.path.join(root, "2026", "07")
            source_day = os.path.join(archive_dir, "11")
            expired_archive = os.path.join(root, "2026", "06", "14.tar.zst")
            os.makedirs(source_day)
            os.makedirs(os.path.dirname(expired_archive))
            source_file = os.path.join(source_day, "attempt_cam1.jpg")
            with open(source_file, "wb") as handle:
                handle.write(b"diagnostic-image")
            with tarfile.open(expired_archive, "w") as archive:
                archive.add(source_file, arcname="old.jpg")
            cleaner = DiagnosticArchiveCleaner([root], 3, 30, 86400)

            result = cleaner.run_once(now=datetime(2026, 7, 15, 12, 0, 0).timestamp())

            self.assertEqual(result["archived"], 1)
            self.assertEqual(result["archive_deleted"], 1)
            self.assertFalse(os.path.exists(source_day))
            self.assertTrue(os.path.exists(source_day + ".tar.zst"))
            self.assertFalse(os.path.exists(expired_archive))

    def test_diagnostic_archive_keeps_recent_day_and_archive(self):
        with tempfile.TemporaryDirectory() as root:
            month = os.path.join(root, "2026", "07")
            recent_day = os.path.join(month, "13")
            retained_archive = os.path.join(month, "02.tar.zst")
            os.makedirs(recent_day)
            open(os.path.join(recent_day, "attempt.jpg"), "w").close()
            open(retained_archive, "w").close()
            cleaner = DiagnosticArchiveCleaner([root], 3, 30, 86400)

            result = cleaner.run_once(now=datetime(2026, 7, 15, 12, 0, 0).timestamp())

            self.assertEqual(result, {"archived": 0, "archive_deleted": 0, "failed": 0})
            self.assertTrue(os.path.exists(recent_day))
            self.assertTrue(os.path.exists(retained_archive))

    def test_diagnostic_archive_keeps_source_when_archive_already_exists(self):
        with tempfile.TemporaryDirectory() as root:
            day = os.path.join(root, "2026", "07", "11")
            os.makedirs(day)
            open(os.path.join(day, "attempt.jpg"), "w").close()
            open(day + ".tar.zst", "w").close()
            cleaner = DiagnosticArchiveCleaner([root], 3, 30, 86400)

            result = cleaner.run_once(now=datetime(2026, 7, 15, 12, 0, 0).timestamp())

            self.assertEqual(result, {"archived": 0, "archive_deleted": 0, "failed": 1})
            self.assertTrue(os.path.exists(day))

    def test_diagnostic_archive_shutdown_keeps_source(self):
        with tempfile.TemporaryDirectory() as root:
            day = os.path.join(root, "2026", "07", "11")
            os.makedirs(day)
            open(os.path.join(day, "attempt.jpg"), "w").close()
            cleaner = DiagnosticArchiveCleaner([root], 3, 30, 86400)
            cleaner._stop_event.set()

            result = cleaner.run_once(now=datetime(2026, 7, 15, 12, 0, 0).timestamp())

            self.assertEqual(result, {"archived": 0, "archive_deleted": 0, "failed": 1})
            self.assertTrue(os.path.exists(day))
            self.assertFalse(os.path.exists(day + ".tar.zst"))
            self.assertFalse(os.path.exists(day + ".tar.zst.tmp"))

    def test_pressure_cleanup_deletes_oldest_images_but_keeps_pending_images(self):
        with tempfile.TemporaryDirectory() as service_dir:
            root = os.path.join(service_dir, "storage", "weighbridge")
            old_dir = os.path.join(root, "2026", "07", "12")
            recent_dir = os.path.join(root, "2026", "07", "14")
            os.makedirs(old_dir)
            os.makedirs(recent_dir)
            pending = os.path.join(old_dir, "pending.jpg")
            pending_publish = os.path.join(old_dir, "pending-publish.jpg")
            old_image = os.path.join(old_dir, "old.jpg")
            recent_image = os.path.join(recent_dir, "recent.jpg")
            fresh_image = os.path.join(recent_dir, "fresh.jpg")
            for path in (pending, pending_publish, old_image, recent_image):
                with open(path, "wb") as handle:
                    handle.write(b"12345")
                os.utime(path, (1, 1))
            with open(fresh_image, "wb") as handle:
                handle.write(b"12345")
            pending_file = os.path.join(service_dir, "storage", "upload_pending.jsonl")
            with open(pending_file, "w") as handle:
                handle.write(json.dumps({"fpath": pending, "object_key": "pending"}) + "\n")
            publish_file = os.path.join(service_dir, "storage", "publish_pending.jsonl")
            with open(publish_file, "w") as handle:
                handle.write(json.dumps({"image_paths": [pending_publish]}) + "\n")
            cleaner = ImageRetentionCleaner(
                [root], 30, 3600, {".jpg"}, pressure_free_bytes=110,
            )
            usage = type("Usage", (), {"free": 100})()

            with patch("services.storage.retention_cleaner.SERVICE_DIR", service_dir), patch(
                "services.storage.retention_cleaner.shutil.disk_usage", return_value=usage,
            ):
                result = cleaner.run_once(
                    now=datetime(2026, 7, 15, 12, 0, 0).timestamp()
                )

            self.assertEqual(result["pressure_deleted"], 2)
            self.assertEqual(result["pressure_reclaimed"], 10)
            self.assertTrue(os.path.exists(pending))
            self.assertTrue(os.path.exists(pending_publish))
            self.assertTrue(os.path.exists(fresh_image))
            self.assertFalse(os.path.exists(old_image))
            self.assertFalse(os.path.exists(recent_image))


if __name__ == "__main__":
    unittest.main()
