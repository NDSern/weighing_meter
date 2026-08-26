import json
import os
import tempfile
import time
import unittest
from unittest import mock

import sys
from types import ModuleType

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = ModuleType("cv2")
    cv2.IMWRITE_JPEG_QUALITY = 1
    sys.modules.setdefault("cv2", cv2)
    np = None

from services.capture.session_frame_spool import SessionFrameSpool


class Grabber:
    def __init__(self, value):
        self.frame = Frame(value)
        self.reads = 0

    def peek_latest_frame(self, copy_frame=False):
        self.reads += 1
        return self.frame.copy() if copy_frame else self.frame


class TimestampedGrabber(Grabber):
    def peek_latest_frame_snapshot(self, copy_frame=False):
        frame = self.peek_latest_frame(copy_frame=copy_frame)
        return frame, 7, "2026-07-15T00:00:00.123+00:00"


class Frame:
    def __init__(self, value):
        self.value = value

    def copy(self):
        return Frame(self.value)


class Encoded:
    def __init__(self, data):
        self.data = data

    def tobytes(self):
        return self.data


class FakeCv2:
    IMWRITE_JPEG_QUALITY = 1

    @staticmethod
    def imencode(_extension, frame, _options):
        return True, Encoded(("jpeg:%s" % frame.value).encode())

    @staticmethod
    def imread(path):
        if not os.path.exists(path):
            return None
        with open(path, "rb") as handle:
            return Frame(handle.read())


class SessionFrameSpoolTests(unittest.TestCase):
    def make_spool(self, root, **kwargs):
        return SessionFrameSpool(
            root, Grabber(40), Grabber(180), interval=0.03,
            min_free_bytes=0, cv2_module=FakeCv2, **kwargs
        )

    def test_samples_active_session_as_jpegs_and_keeps_sessions_separate(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.start()
            spool.begin_session("A", {"cam1": Frame(0)})
            time.sleep(0.11)
            job_a = spool.end_session("A", {"weight": 10})
            with open(job_a, encoding="utf-8") as handle:
                manifest_a = json.load(handle)

            spool.begin_session("B")
            time.sleep(0.07)
            # Reading A while B captures must not block or mutate A.
            for name in manifest_a["files"]:
                self.assertIsNotNone(FakeCv2.imread(os.path.join(manifest_a["session_dir"], name)))
            job_b = spool.end_session("B", {"weight": 20})
            self.assertTrue(spool.stop(1))

            with open(job_b, encoding="utf-8") as handle:
                manifest_b = json.load(handle)
            self.assertGreaterEqual(manifest_a["frame_counts"]["cam1"], 2)
            self.assertGreaterEqual(manifest_b["frame_counts"]["cam3"], 1)
            self.assertNotEqual(manifest_a["session_dir"], manifest_b["session_dir"])
            self.assertEqual(manifest_a["metadata"], {"weight": 10})

    def test_samples_optional_cam2_with_session_timeline(self):
        with tempfile.TemporaryDirectory() as root:
            spool = SessionFrameSpool(
                root, Grabber(40), Grabber(180), cam2_grabber=Grabber(90),
                interval=0.03, min_free_bytes=0, cv2_module=FakeCv2,
            )
            spool.start()
            spool.begin_session("rear")
            time.sleep(0.07)
            job = spool.end_session("rear", {})
            self.assertTrue(spool.stop(1))

            with open(job, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertGreaterEqual(manifest["frame_counts"]["cam2"], 1)
            self.assertTrue(any(name.startswith("cam2-") for name in manifest["files"]))

    def test_records_timestamp_and_plate_track_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            provider = lambda camera, frame_id: [{"bbox": [1, 2, 10, 8], "track_id": 7,
                                                   "confidence": 0.9,
                                                   "frame_id": frame_id}] if camera == "cam1" else []
            spool = self.make_spool(root, metadata_provider=provider)
            spool.start()
            spool.begin_session("tracked")
            time.sleep(0.05)
            job = spool.end_session("tracked", {})
            self.assertTrue(spool.stop(1))

            with open(job, encoding="utf-8") as handle:
                manifest = json.load(handle)
            cam1 = next(name for name in manifest["files"] if name.startswith("cam1-"))
            self.assertIn("captured_at", manifest["frame_metadata"][cam1])
            self.assertEqual(manifest["frame_metadata"][cam1]["tracks"][0]["track_id"], 7)

    def test_records_camera_acquisition_timestamp(self):
        with tempfile.TemporaryDirectory() as root:
            spool = SessionFrameSpool(
                root, TimestampedGrabber(40), Grabber(180), interval=0.03,
                min_free_bytes=0, cv2_module=FakeCv2,
            )
            spool.start()
            spool.begin_session("timestamped")
            time.sleep(0.04)
            job = spool.end_session("timestamped", {})
            self.assertTrue(spool.stop(1))

            with open(job, encoding="utf-8") as handle:
                manifest = json.load(handle)
            cam1 = next(name for name in manifest["files"] if name.startswith("cam1-"))
            self.assertEqual(
                manifest["frame_metadata"][cam1]["captured_at"],
                "2026-07-15T00:00:00.123+00:00",
            )

    def test_pending_jobs_recover_fifo_after_queue_overflow_and_restart(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root, notification_queue_size=1)
            paths = []
            for session_id in ("one", "two", "three"):
                spool.begin_session(session_id)
                paths.append(spool.end_session(session_id, {}))

            restarted = self.make_spool(root, notification_queue_size=1)
            recovered = []
            for _ in paths:
                path = restarted.get_pending_job(timeout=0)
                recovered.append(path)
                restarted.acknowledge_job(path)
            self.assertEqual([os.path.basename(path) for path in recovered],
                             [os.path.basename(path) for path in paths])
            self.assertTrue(all(os.path.dirname(path) == restarted.processing_dir
                                for path in recovered))
            self.assertIsNone(restarted.get_pending_job(timeout=0))

    def test_disk_cap_marks_manifest_incomplete_without_frame_buffering(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root, disk_cap_bytes=0)
            spool.start()
            spool.begin_session("full", {"cam1": Frame(0)})
            time.sleep(0.05)
            job = spool.end_session("full", {"reason": "test"})
            spool.stop(1)

            with open(job, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertTrue(manifest["incomplete"])
            self.assertIn("disk cap reached", manifest["errors"])
            self.assertEqual(manifest["files"], [])
            self.assertFalse(any(name.endswith(".tmp") for name in os.listdir(root)))

    def test_rejects_overlapping_or_unsafe_sessions(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            with self.assertRaises(ValueError):
                spool.begin_session("../escape")
            spool.begin_session("safe")
            with self.assertRaises(RuntimeError):
                spool.begin_session("other")

    def test_manifest_failure_keeps_session_active_for_retry(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.begin_session("retry")
            original = spool._atomic_json
            spool._atomic_json = lambda *_args: (_ for _ in ()).throw(OSError("disk"))
            with self.assertRaises(OSError):
                spool.end_session("retry", {})
            spool._atomic_json = original

            path = spool.end_session("retry", {})
            self.assertTrue(os.path.exists(path))

    def test_terminal_snapshot_survives_failed_finalize_and_restart(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.begin_session("retry", {"cam1": Frame(1)}, {
                "session_id": "retry", "started_at": "2026-07-24T00:00:00+00:00",
                "stable_weight": None,
            })
            terminal = {
                "session_id": "retry", "started_at": "2026-07-24T00:00:00+00:00",
                "ended_at": "2026-07-24T00:01:00+00:00", "stable_weight": 1200,
                "duration_s": 60.0, "end_reason": "scale_empty",
                "weight_observed_at": "2026-07-24T00:00:30+00:00",
                "raw_peak_weight": 1210, "filtered_peak_weight": 1200,
            }
            spool.update_active_metadata("retry", terminal)
            original = spool._atomic_json
            spool._atomic_json = lambda path, value: (
                (_ for _ in ()).throw(OSError("disk"))
                if os.path.dirname(path) == spool.jobs_dir else original(path, value)
            )
            with self.assertRaises(OSError):
                spool.end_session("retry", terminal)

            restarted = self.make_spool(root)
            recovered = restarted.get_pending_job(timeout=0)
            with open(recovered, encoding="utf-8") as handle:
                metadata = json.load(handle)["metadata"]

            self.assertEqual(metadata["stable_weight"], 1200)
            self.assertEqual(metadata["weight_observed_at"], "2026-07-24T00:00:30+00:00")
            self.assertEqual(metadata["raw_peak_weight"], 1210)
            self.assertEqual(metadata["end_reason"], "scale_empty")
            self.assertEqual(metadata["ended_at"], "2026-07-24T00:01:00+00:00")
            self.assertEqual(metadata["duration_s"], 60.0)

    def test_acknowledge_removes_manifest_and_session_directory(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            session_dir = spool.begin_session("done", {"cam1": Frame(1)})
            path = spool.end_session("done", {})

            spool.acknowledge_job(path)

            self.assertFalse(os.path.exists(path))
            self.assertFalse(os.path.exists(session_dir))

    def test_abort_clears_partial_active_session(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            session_dir = spool.begin_session("partial", {"cam1": Frame(1)})

            self.assertTrue(spool.abort_session("partial"))
            self.assertFalse(os.path.exists(session_dir))
            spool.begin_session("next")

    def test_active_manifest_is_durable_and_metadata_updates(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.begin_session("active")
            marker = os.path.join(spool.active_dir, "active.json")
            self.assertTrue(os.path.exists(marker))
            spool.update_active_metadata("active", {"weight": 12})
            with open(marker, encoding="utf-8") as handle:
                self.assertEqual(json.load(handle)["metadata"], {"weight": 12})
            pending = spool.end_session("active", {"weight": 13})
            self.assertTrue(os.path.exists(pending))
            self.assertFalse(os.path.exists(marker))

    def test_restart_requeues_processing_job(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.begin_session("claimed", {"cam1": Frame(1)})
            pending = spool.end_session("claimed", {})
            processing = spool.get_pending_job(timeout=0)
            self.assertFalse(os.path.exists(pending))
            self.assertEqual(os.path.dirname(processing), spool.processing_dir)

            restarted = self.make_spool(root)
            recovered = restarted.get_pending_job(timeout=0)
            self.assertEqual(os.path.basename(recovered), os.path.basename(processing))

    def test_restart_finishes_cleanup_without_requeue(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            session_dir = spool.begin_session("cleanup", {"cam1": Frame(1)})
            spool.end_session("cleanup", {})
            processing = spool.get_pending_job(timeout=0)
            cleanup = os.path.join(spool.cleanup_dir, os.path.basename(processing))
            os.replace(processing, cleanup)

            restarted = self.make_spool(root)
            self.assertFalse(os.path.exists(cleanup))
            self.assertFalse(os.path.exists(session_dir))
            self.assertIsNone(restarted.get_pending_job(timeout=0))

    def test_restart_recovers_active_frames_as_incomplete_pending(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.begin_session("interrupted", {"cam1": Frame(1)})
            spool.update_active_metadata("interrupted", {
                "session_id": "interrupted",
                "started_at": "2026-07-15T00:00:00+00:00",
                "stable_weight": 9000,
                "decimal_pos": 0,
                "stability_rule": "exact_5",
            })

            restarted = self.make_spool(root)
            processing = restarted.get_pending_job(timeout=0)
            with open(processing, encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["metadata"]["end_reason"], "machine_offline")
            self.assertTrue(manifest["metadata"]["recovered_after_restart"])
            self.assertTrue(manifest["metadata"]["incomplete"])
            self.assertEqual(manifest["metadata"]["stable_weight"], 9000)
            self.assertEqual(manifest["metadata"]["end_reason"], "machine_offline")
            self.assertIn("ended_at", manifest["metadata"])
            self.assertGreaterEqual(manifest["metadata"]["duration_s"], 0)
            self.assertFalse(os.path.exists(os.path.join(restarted.active_dir,
                                                         "interrupted.json")))

    def test_orphans_and_corrupt_manifests_are_quarantined(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            orphan = os.path.join(spool.sessions_dir, "orphaned")
            os.makedirs(orphan)
            corrupt = os.path.join(spool.jobs_dir, "000-corrupt.json")
            with open(corrupt, "w", encoding="utf-8") as handle:
                handle.write("{")

            restarted = self.make_spool(root)
            self.assertTrue(os.path.isdir(os.path.join(restarted.orphan_dir, "orphaned")))
            self.assertTrue(any(name.startswith("000-corrupt.json")
                                for name in os.listdir(restarted.failed_dir)))

    def test_failed_jobs_and_job_paths_use_processing_state_safely(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.begin_session("failed")
            spool.end_session("failed", {})
            processing = spool.get_pending_job(timeout=0)
            failed = spool.fail_job(processing)
            self.assertEqual(os.path.dirname(failed), spool.failed_dir)
            with self.assertRaises(ValueError):
                spool.acknowledge_job(os.path.join(root, "outside.json"))

    def test_quarantine_counts_toward_cap_and_expires(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root, quarantine_retention_days=1)
            failed_session = os.path.join(spool.sessions_dir, "failed-session")
            os.makedirs(failed_session)
            frame = os.path.join(failed_session, "frame.jpg")
            with open(frame, "wb") as handle:
                handle.write(b"12345")
            orphan = os.path.join(spool.orphan_dir, "old")
            os.makedirs(orphan)
            orphan_frame = os.path.join(orphan, "frame.jpg")
            with open(orphan_frame, "wb") as handle:
                handle.write(b"123")
            old = time.time() - 2 * 86400
            os.utime(orphan, (old, old))

            restarted = self.make_spool(root, quarantine_retention_days=1)

            self.assertGreaterEqual(restarted._bytes_written, 5)
            self.assertFalse(os.path.exists(orphan))

    def test_quarantine_scan_skips_unreadable_entry(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            expired = os.path.join(spool.orphan_dir, "expired")
            unreadable = os.path.join(spool.orphan_dir, "unreadable")
            os.makedirs(expired)
            os.makedirs(unreadable)
            old = time.time() - 2 * 86400
            os.utime(expired, (old, old))

            original_getmtime = os.path.getmtime
            def getmtime(path):
                if path == unreadable:
                    raise OSError("unreadable")
                return original_getmtime(path)

            with mock.patch(
                "services.capture.session_frame_spool.os.path.getmtime",
                side_effect=getmtime,
            ):
                paths = spool._expired_paths(spool.orphan_dir, time.time() - 86400)

            self.assertEqual(paths, [expired])

    def test_expired_failed_manifest_removes_session_frames(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root, quarantine_retention_days=1)
            session_dir = spool.begin_session("failed", {"cam1": Frame(1)})
            spool.end_session("failed", {})
            failed = spool.fail_job(spool.get_pending_job(timeout=0))
            old = time.time() - 2 * 86400
            os.utime(failed, (old, old))

            restarted = self.make_spool(root, quarantine_retention_days=1)

            self.assertFalse(os.path.exists(failed))
            self.assertFalse(os.path.exists(session_dir))
            self.assertEqual(restarted._bytes_written, 0)

    def test_replacing_auxiliary_frame_keeps_accounting_and_manifest_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root)
            spool.begin_session("replace")

            spool.save_session_frame("replace", "rear.jpg", Frame(1))
            first_bytes = spool._bytes_written
            spool.save_session_frame("replace", "rear.jpg", Frame(2))

            self.assertEqual(spool._bytes_written, first_bytes)
            self.assertEqual(spool._active["files"].count("rear.jpg"), 1)

    def test_failed_expiry_keeps_session_referenced_by_pending_job(self):
        with tempfile.TemporaryDirectory() as root:
            spool = self.make_spool(root, quarantine_retention_days=1)
            session_dir = spool.begin_session("shared", {"cam1": Frame(1)})
            pending = spool.end_session("shared", {})
            with open(pending, encoding="utf-8") as handle:
                manifest = json.load(handle)
            failed = os.path.join(spool.failed_dir, "old.json")
            with open(failed, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            old = time.time() - 2 * 86400
            os.utime(failed, (old, old))

            restarted = self.make_spool(root, quarantine_retention_days=1)

            self.assertFalse(os.path.exists(failed))
            self.assertTrue(os.path.isdir(session_dir))
            self.assertIsNotNone(restarted.get_pending_job(timeout=0))


if __name__ == "__main__":
    unittest.main()
