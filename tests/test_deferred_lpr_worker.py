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

    def add_observation(self, *args, **kwargs):
        self.observations.append((args, kwargs))

    def update_image(self, *args):
        self.images.append(args)

    def needs_undetectable(self):
        return self.unknown is None

    def save_undetectable(self, frame):
        self.unknown = frame


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
            detect = unittest.mock.Mock(side_effect=AssertionError("detector must not run"))
            recognize = unittest.mock.Mock(return_value=[])
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
            self.assertEqual(recognize.call_args.args[0][0]["bbox"], [1, 2, 10, 8])

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
            self.assertEqual(len(trackers[0].observations), 2)
            self.assertEqual(spool.acknowledged, [path])
            self.assertFalse(worker.status()["running"])

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
