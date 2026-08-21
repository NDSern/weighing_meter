import unittest
import threading
import time
from unittest import mock

from services.runtime.inference_lock import PriorityInferenceLock
from services.runtime.rknn_models import RknnModelSet
from services.tracking.plate_tracker import PlateTracker


class FakeRknn:
    NPU_CORE_0 = 0
    NPU_CORE_1 = 1
    NPU_CORE_2 = 2
    instances = []
    fail_at = None
    calls = []
    release_failures = set()

    def __init__(self):
        self.index = len(self.instances)
        self.instances.append(self)

    def load_rknn(self, path):
        self.calls.append(("load", self.index, path))
        return 1 if self.fail_at == (self.index, "load") else 0

    def init_runtime(self, core_mask=None):
        self.calls.append(("init", self.index, core_mask))
        return 1 if self.fail_at == (self.index, "init") else 0

    def release(self):
        self.calls.append(("release", self.index))
        if self.index in self.release_failures:
            self.release_failures.remove(self.index)
            raise RuntimeError("release failed")


class RknnModelSetTests(unittest.TestCase):
    def setUp(self):
        FakeRknn.instances = []
        FakeRknn.fail_at = None
        FakeRknn.calls = []
        FakeRknn.release_failures = set()

    def open_models(self, vehicle=True):
        return RknnModelSet.open("detector", "ocr", "vehicle", vehicle, FakeRknn)

    def test_partial_failure_releases_current_and_prior_models(self):
        FakeRknn.fail_at = (2, "init")

        with self.assertRaises(RuntimeError):
            self.open_models()

        releases = [call for call in FakeRknn.calls if call[0] == "release"]
        self.assertEqual(releases, [("release", 2), ("release", 1), ("release", 0)])

    def test_close_releases_reverse_order_once(self):
        models = self.open_models()

        models.close()
        models.close()

        releases = [call for call in FakeRknn.calls if call[0] == "release"]
        self.assertEqual(releases, [("release", 4), ("release", 3), ("release", 2), ("release", 1), ("release", 0)])

    def test_selective_release_preserves_live_lpr_models(self):
        models = self.open_models()

        models.close(release_lpr=False, release_vehicle=True)

        releases = [call for call in FakeRknn.calls if call[0] == "release"]
        self.assertEqual(releases, [("release", 4)])

    def test_failed_release_can_retry(self):
        models = self.open_models(vehicle=False)
        FakeRknn.release_failures = {3}

        self.assertEqual(len(models.close()), 1)
        self.assertEqual(models.close(), [])

        releases = [call for call in FakeRknn.calls if call == ("release", 3)]
        self.assertEqual(len(releases), 2)


class PriorityInferenceLockTests(unittest.TestCase):
    def test_waiting_live_work_runs_before_next_deferred_call(self):
        lock = PriorityInferenceLock()
        order = []

        def run_deferred():
            with lock.deferred():
                order.append("deferred")

        def run_live():
            with lock:
                order.append("live")

        with lock.deferred():
            deferred_thread = threading.Thread(target=run_deferred)
            deferred_thread.start()
            time.sleep(0.01)
            live_thread = threading.Thread(target=run_live)
            live_thread.start()
            deadline = time.time() + 1
            while lock._live_waiters == 0 and time.time() < deadline:
                time.sleep(0.001)

        live_thread.join(1)
        deferred_thread.join(1)
        self.assertEqual(order, ["live", "deferred"])

    def test_deferred_work_runs_after_one_live_call_under_live_backlog(self):
        lock = PriorityInferenceLock()
        order = []

        def run(label, context):
            with context():
                order.append(label)
                time.sleep(0.005)

        with lock.deferred():
            threads = [
                threading.Thread(target=run, args=("deferred", lock.deferred)),
                threading.Thread(target=run, args=("live-1", lambda: lock)),
                threading.Thread(target=run, args=("live-2", lambda: lock)),
            ]
            for thread in threads:
                thread.start()
            deadline = time.time() + 1
            while (lock._deferred_waiters < 1 or lock._live_waiters < 2) and time.time() < deadline:
                time.sleep(0.001)

        for thread in threads:
            thread.join(1)
        self.assertTrue(order[0].startswith("live-"))
        self.assertEqual(order[1], "deferred")


class PlateTrackerResourceTests(unittest.TestCase):
    def test_candidate_and_global_best_share_one_frame_copy(self):
        frame = mock.Mock()
        copied = object()
        frame.copy.return_value = copied
        tracker = PlateTracker(max_plate_images=2)

        tracker.update_image("30A-12345", 0.9, frame, "cam1")

        self.assertIs(tracker._plate_images["30A-12345"][0], copied)
        self.assertIs(tracker._image_frame, copied)
        frame.copy.assert_called_once()

    def test_confirmation_diagnostics_explain_missing_time_span(self):
        tracker = PlateTracker()
        tracker.add_observation("30A-123.45", 0.9, 200, 60, observed_at=10.0)
        tracker.add_observation("30A-123.45", 0.9, 200, 60, observed_at=10.2)

        diagnostics = tracker.get_confirmation_diagnostics()

        self.assertEqual(diagnostics["best_candidate"], "30A-123.45")
        self.assertEqual(diagnostics["failure_reason"], "insufficient_observation_span")


if __name__ == "__main__":
    unittest.main()
