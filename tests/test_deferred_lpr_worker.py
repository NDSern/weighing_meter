import json
import os
import tempfile
import threading
import time
import unittest
import sys
from datetime import datetime
from types import SimpleNamespace
from types import ModuleType
from unittest import mock

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = ModuleType("cv2")

from services.pipeline.deferred_lpr_worker import DeferredLprWorker


class FakeSpool:
    def __init__(self, paths):
        self.paths = list(paths)
        self.acknowledged = []
        self.failed = []

    def get_pending_job(self, timeout=None):
        if self.paths:
            return self.paths.pop(0)
        time.sleep(min(timeout or 0, 0.01))
        return None

    def acknowledge_job(self, path):
        self.acknowledged.append(path)

    def fail_job(self, path):
        self.failed.append(path)
        return path + ".failed"


class FakeCv2:
    def __init__(self, frames):
        self.frames = frames

    def imread(self, path):
        return self.frames.get(os.path.basename(path))


class Frame:
    shape = (12, 18, 3)

    def __getitem__(self, _key):
        return self

    def copy(self):
        return self


class Tracker:
    def __init__(self):
        self.observations = []
        self.images = []
        self.unknown = None
        self.clear_count = 0

    def add_observation(self, *args, **kwargs):
        self.observations.append((args, kwargs))

    def update_image(self, *args):
        self.images.append(args)

    def needs_undetectable(self):
        return self.unknown is None

    def save_undetectable(self, frame):
        self.unknown = frame

    def clear(self):
        self.clear_count += 1
        self.observations.clear()
        self.images.clear()
        self.unknown = None


class DeferredLprWorkerTests(unittest.TestCase):
    def make_manifest(self, root, name, files, metadata=None):
        session_dir = os.path.join(root, name)
        os.mkdir(session_dir)
        path = os.path.join(root, name + ".json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"session_id": name, "session_dir": session_dir,
                       "files": files, "metadata": metadata or {},
                       "started_at": "2026-07-15T00:00:00+00:00",
                       "capture_interval_seconds": 0.2}, handle)
        return path

    def test_tracked_bbox_skips_deferred_detector(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_manifest(root, "bbox", ["cam1-000001.jpg"])
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["frame_metadata"] = {
                "cam1-000001.jpg": {
                    "captured_at": "2026-07-15T00:00:01+00:00",
                    "tracks": [{"bbox": [1, 2, 10, 8], "track_id": 1, "confidence": 0.9}],
                }
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            detect = mock.Mock(side_effect=AssertionError("detector must not run"))
            recognize = mock.Mock(return_value=[])
            spool = FakeSpool([path])
            camera = SimpleNamespace(name="cam1", detector="d1", ocr="o1", lpr_crop="full")
            worker = DeferredLprWorker(
                spool, [camera], "chars", lambda _metadata, _tracker: True,
                detect_regions_fn=detect, recognize_regions_fn=recognize,
                tracker_factory=Tracker, cv2_module=FakeCv2({"cam1-000001.jpg": Frame()}),
            )
            worker.start()
            self.assertTrue(self.wait_for(lambda: spool.acknowledged))
            self.assertTrue(worker.stop())
            detect.assert_not_called()
            region = recognize.call_args.args[0][0]
            self.assertEqual(region["bbox"], [1, 2, 10, 8])
            self.assertEqual(region["crop_size"], "9x6")
            self.assertIn("crop_img", region)
            self.assertNotIn("crop", region)

    def test_tracked_obb_reuses_perspective_crop(self):
        frame = Frame()
        crop = Frame()
        crop.shape = (6, 9, 3)
        track = {
            "bbox": [1, 2, 10, 8],
            "obb": [[1, 2], [10, 2], [10, 8], [1, 8]],
            "confidence": 0.9,
            "class": "BSV",
            "two_row": True,
        }

        with mock.patch(
            "services.pipeline.deferred_lpr_worker.crop_obb", return_value=crop
        ) as crop_obb:
            regions = DeferredLprWorker._tracked_regions(frame, [track])

        crop_obb.assert_called_once_with(frame, track["obb"])
        self.assertIs(regions[0]["crop_img"], crop)
        self.assertEqual(regions[0]["class"], "BSV")
        self.assertTrue(regions[0]["two_row"])

    def test_long_session_selection_combines_confidence_and_timeline(self):
        files = ["cam1-%06d.jpg" % index for index in range(10)]
        metadata = {
            name: {"tracks": [{"confidence": 0.99 if index in (4, 5) else 0.1}]}
            for index, name in enumerate(files)
        }

        selected = DeferredLprWorker._select_frame_paths(files, metadata, 4)

        self.assertEqual(selected, [files[0], files[4], files[5], files[-1]])

    def test_long_session_without_tracks_still_spreads_selection(self):
        files = ["cam1-%06d.jpg" % index for index in range(10)]

        selected = DeferredLprWorker._select_frame_paths(files, {}, 4)

        indexes = [files.index(path) for path in selected]
        self.assertEqual(indexes[0], 0)
        self.assertEqual(indexes[-1], 9)
        self.assertLessEqual(max(right - left for left, right in zip(indexes, indexes[1:])), 5)

    def test_diagnostics_count_ocr_outcomes_per_camera(self):
        diagnostics = DeferredLprWorker._new_diagnostics(["cam1-000001.jpg"])
        camera = SimpleNamespace(name="cam1", detector="detector", ocr="ocr", lpr_crop="full")
        region = {
            "bbox": [1, 2, 10, 8], "obb": None, "det_conf": 0.9,
            "class": "BSD", "crop_size": "9x6", "crop_img": Frame(),
            "ocr_status": None,
        }
        worker = DeferredLprWorker(
            FakeSpool([]), [camera], "chars", lambda *_args: True,
            detect_regions_fn=mock.Mock(return_value=[region]),
            recognize_regions_fn=mock.Mock(return_value=[{
                "plate": "unknown", "crop_size": "9x6", "det_conf": 0.9,
                "valid_candidates": [], "ocr_outcome": "invalid_format",
            }]),
            tracker_factory=Tracker,
            cv2_module=FakeCv2({"cam1-000001.jpg": Frame()}),
        )

        worker._process_frame(
            "/tmp", "cam1-000001.jpg", Tracker(), 1.0, diagnostics=diagnostics
        )

        self.assertEqual(diagnostics["detector_successes"], 1)
        self.assertEqual(diagnostics["detected_regions"], 1)
        self.assertEqual(diagnostics["ocr_invalid_format"], 1)
        self.assertEqual(
            diagnostics["evidence"]["cam1"]["plate_detected_ocr_invalid_format"],
            "cam1-000001.jpg",
        )

    def test_all_detector_failures_retry_then_dead_letter_with_classification(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_manifest(root, "failure", ["cam1-000001.jpg"])
            logs = []
            spool = FakeSpool([path])
            callback = mock.Mock()
            worker = DeferredLprWorker(
                spool,
                [SimpleNamespace(name="cam1", detector=1, ocr=2, lpr_crop="full")],
                [], callback,
                detect_regions_fn=mock.Mock(side_effect=RuntimeError("detector failed")),
                tracker_factory=Tracker,
                cv2_module=FakeCv2({"cam1-000001.jpg": Frame()}),
                failed_retry_delay=0.01,
                max_retries=2,
                log_fn=lambda level, message: logs.append((level, message)),
            )

            worker.start()
            self.assertTrue(self.wait_for(lambda: spool.failed))
            self.assertTrue(worker.stop())

            callback.assert_not_called()
            self.assertEqual(spool.acknowledged, [])
            metrics = [json.loads(message) for level, message in logs if level == "METRIC"]
            dead_letter = next(
                metric for metric in metrics
                if metric["event"] == "session_processing_dead_lettered"
            )
            self.assertEqual(dead_letter["classification"], "detector_inference_error")

    @staticmethod
    def wait_for(predicate, timeout=1):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_processes_fifo_frames_with_camera_models_then_acknowledges(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_manifest(root, "one", ["cam1-000001.jpg", "cam3-000001.jpg"])
            calls = []
            trackers = []
            cameras = [
                SimpleNamespace(name="cam1", detector="d1", ocr="o1", lpr_crop="full"),
                SimpleNamespace(name="cam3", detector="d3", ocr="o3", lpr_crop="full"),
            ]

            def detect(_frame, detector):
                calls.append(("detect", detector))
                return [{"camera": detector}]

            def recognize(regions, ocr, charset):
                calls.append(("ocr", regions[0]["camera"], ocr, charset))
                return [{"plate": "30A-12345", "crop_size": "100x40",
                         "det_conf": 0.9, "valid_candidates": []}]

            spool = FakeSpool([path])
            worker = DeferredLprWorker(
                spool, cameras, "chars", lambda metadata, tracker: trackers.append(tracker),
                detect_regions_fn=detect, recognize_regions_fn=recognize,
                tracker_factory=Tracker,
                cv2_module=FakeCv2({"cam1-000001.jpg": Frame(), "cam3-000001.jpg": Frame()}),
            )
            worker.start()
            self.assertTrue(self.wait_for(lambda: spool.acknowledged))
            self.assertTrue(worker.stop())
            self.assertEqual(calls, [("detect", "d1"), ("ocr", "d1", "o1", "chars"),
                                     ("detect", "d3"), ("ocr", "d3", "o3", "chars")])
            self.assertEqual(trackers[0].observations, [])
            self.assertEqual(spool.acknowledged, [path])
            self.assertFalse(worker.status()["running"])

    def test_manifest_degradation_is_passed_to_finalizer(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_manifest(root, "degraded", [], {"errors": ["metadata-error"]})
            with open(path, encoding="utf-8") as handle:
                manifest = json.load(handle)
            manifest["incomplete"] = True
            manifest["errors"] = ["disk cap reached", "metadata-error"]
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle)
            received = []
            spool = FakeSpool([path])
            worker = DeferredLprWorker(
                spool, [], [], lambda metadata, _tracker: received.append(dict(metadata)) or True,
                tracker_factory=Tracker, cv2_module=FakeCv2({}),
            )

            worker.start()
            self.assertTrue(self.wait_for(lambda: spool.acknowledged))
            self.assertTrue(worker.stop())

            self.assertTrue(received[0]["incomplete"])
            self.assertEqual(received[0]["errors"], ["metadata-error", "disk cap reached"])

    def test_cleans_memory_and_logs_rss_after_job(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_manifest(root, "one", [])
            spool = FakeSpool([path])
            cleanup = mock.Mock()
            logs = []
            worker = DeferredLprWorker(
                spool, [], [], lambda _metadata, _tracker: True,
                tracker_factory=Tracker, cv2_module=FakeCv2({}),
                memory_cleanup_fn=cleanup, job_interval=0, gc_interval=1,
                log_fn=lambda level, message: logs.append((level, message)),
            )

            worker.start()
            self.assertTrue(self.wait_for(lambda: cleanup.called))
            self.assertTrue(worker.stop())

            cleanup.assert_called_once()
            metrics = [json.loads(message) for level, message in logs if level == "METRIC"]
            self.assertEqual(metrics[-1]["event"], "deferred_job_memory")
            self.assertTrue(metrics[-1]["finished"])

    def test_bad_frame_and_inference_do_not_abort_job(self):
        with tempfile.TemporaryDirectory() as root:
            files = ["cam1-bad.jpg", "cam1-throws.jpg", "cam1-good.jpg"]
            path = self.make_manifest(root, "frames", files)
            seen = []

            def detect(frame, detector):
                if frame is throwing:
                    raise RuntimeError("inference")
                seen.append(frame)
                return []

            bad, throwing, good = None, Frame(), Frame()
            spool = FakeSpool([path])
            callbacks = []
            worker = DeferredLprWorker(
                spool, [SimpleNamespace(name="cam1", detector=1, ocr=2, lpr_crop="full")],
                [], lambda metadata, tracker: callbacks.append(tracker),
                detect_regions_fn=detect, recognize_regions_fn=lambda *args, **kwargs: [],
                tracker_factory=Tracker,
                cv2_module=FakeCv2({"cam1-bad.jpg": bad, "cam1-throws.jpg": throwing,
                                    "cam1-good.jpg": good}),
            )
            worker.start()
            self.assertTrue(self.wait_for(lambda: spool.acknowledged))
            self.assertTrue(worker.stop())
            self.assertEqual(seen, [good])
            self.assertEqual(len(callbacks), 1)

    def test_callback_failure_blocks_later_jobs_to_preserve_fifo(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.make_manifest(root, "first", [], {"id": 1})
            second = self.make_manifest(root, "second", [], {"id": 2})
            spool = FakeSpool([first, second])
            second_called = threading.Event()

            def callback(metadata, tracker):
                if metadata["id"] == 1:
                    raise RuntimeError("publish failed")
                second_called.set()

            worker = DeferredLprWorker(spool, [], [], callback, tracker_factory=Tracker,
                                       cv2_module=FakeCv2({}), failed_retry_delay=10)
            worker.start()
            self.assertTrue(self.wait_for(lambda: worker.status()["failed_jobs"] == 1))
            self.assertTrue(worker.stop())
            self.assertFalse(second_called.is_set())
            self.assertEqual(spool.acknowledged, [])
            self.assertEqual(worker.status()["failed_jobs"], 1)
            self.assertIn("publish failed", worker.status()["last_error"])

    def test_callback_failure_clears_tracker(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_manifest(root, "first", [])
            trackers = []

            def make_tracker():
                tracker = Tracker()
                trackers.append(tracker)
                return tracker

            def fail_callback(_metadata, _tracker):
                raise RuntimeError("publish failed")

            worker = DeferredLprWorker(
                FakeSpool([path]), [], [],
                fail_callback,
                tracker_factory=make_tracker, cv2_module=FakeCv2({}), failed_retry_delay=10,
            )
            worker.start()
            self.assertTrue(self.wait_for(lambda: worker.status()["failed_jobs"] == 1))
            self.assertTrue(worker.stop())

            self.assertEqual(trackers[0].clear_count, 1)

    def test_frame_failure_clears_tracker(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.make_manifest(root, "first", ["cam1-bad.jpg"])
            trackers = []

            def make_tracker():
                tracker = Tracker()
                trackers.append(tracker)
                return tracker

            worker = DeferredLprWorker(
                FakeSpool([path]),
                [SimpleNamespace(name="cam1", detector=1, ocr=2, lpr_crop="full")],
                [], lambda _metadata, _tracker: True,
                tracker_factory=make_tracker, cv2_module=FakeCv2({}), failed_retry_delay=10,
            )
            worker.start()
            self.assertTrue(self.wait_for(lambda: worker.status()["failed_jobs"] == 1))
            self.assertTrue(worker.stop())

            self.assertEqual(trackers[0].clear_count, 1)

    def test_cleanup_failure_does_not_stop_worker(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.make_manifest(root, "first", [])
            second = self.make_manifest(root, "second", [])
            spool = FakeSpool([first, second])
            completed = []
            logs = []

            worker = DeferredLprWorker(
                spool, [], [], lambda metadata, _tracker: completed.append(metadata),
                tracker_factory=Tracker, cv2_module=FakeCv2({}),
                memory_cleanup_fn=mock.Mock(side_effect=RuntimeError("trim failed")),
                log_fn=lambda level, message: logs.append((level, message)),
            )
            worker.start()
            self.assertTrue(self.wait_for(lambda: len(completed) == 2))
            self.assertTrue(worker.stop())

            events = [json.loads(message)["event"] for level, message in logs if level == "METRIC"]
            self.assertEqual(spool.acknowledged, [first, second])
            self.assertEqual(events.count("deferred_memory_cleanup_failed"), 2)

    def test_job_interval_paces_jobs_and_stop_interrupts_wait(self):
        with tempfile.TemporaryDirectory() as root:
            first = self.make_manifest(root, "first", [])
            second = self.make_manifest(root, "second", [])
            spool = FakeSpool([first, second])
            completed_at = []
            worker = DeferredLprWorker(
                spool, [], [], lambda _metadata, _tracker: completed_at.append(time.monotonic()),
                tracker_factory=Tracker, cv2_module=FakeCv2({}), job_interval=0.15,
            )
            worker.start()
            self.assertTrue(self.wait_for(lambda: len(completed_at) == 2))
            self.assertGreaterEqual(completed_at[1] - completed_at[0], 0.14)
            self.assertTrue(worker.stop())

            third = self.make_manifest(root, "third", [])
            waiting_worker = DeferredLprWorker(
                FakeSpool([third]), [], [], lambda _metadata, _tracker: True,
                tracker_factory=Tracker, cv2_module=FakeCv2({}), job_interval=5,
            )
            waiting_worker.start()
            self.assertTrue(self.wait_for(lambda: waiting_worker.status()["current_job"] is None))
            started = time.monotonic()
            self.assertTrue(waiting_worker.stop())
            self.assertLess(time.monotonic() - started, 0.5)

    def test_malformed_job_does_not_kill_thread(self):
        with tempfile.TemporaryDirectory() as root:
            malformed = os.path.join(root, "bad.json")
            with open(malformed, "w", encoding="utf-8") as handle:
                handle.write("not-json")
            spool = FakeSpool([malformed])
            worker = DeferredLprWorker(spool, [], [], lambda metadata, tracker: None,
                                       tracker_factory=Tracker, cv2_module=FakeCv2({}),
                                       failed_retry_delay=10)
            worker.start()
            self.assertTrue(self.wait_for(lambda: worker.status()["failed_jobs"] == 1))
            self.assertTrue(worker.stop())
            self.assertEqual(spool.acknowledged, [])
            self.assertEqual(worker.status()["failed_jobs"], 1)

    def test_permanent_failure_dead_letters_then_processes_next_job(self):
        with tempfile.TemporaryDirectory() as root:
            malformed = os.path.join(root, "bad.json")
            with open(malformed, "w") as handle:
                handle.write("bad")
            valid = self.make_manifest(root, "valid", [], {"ok": True})
            spool = FakeSpool([malformed, valid])
            completed = threading.Event()
            worker = DeferredLprWorker(
                spool, [], [], lambda metadata, tracker: completed.set(),
                tracker_factory=Tracker, cv2_module=FakeCv2({}),
                failed_retry_delay=0.01, max_retries=2,
            )

            worker.start()
            self.assertTrue(completed.wait(1))
            self.assertTrue(worker.stop())
            self.assertEqual(spool.failed, [malformed])
            self.assertEqual(spool.acknowledged, [valid])


if __name__ == "__main__":
    unittest.main()
