import unittest
import sys
from unittest.mock import Mock, patch

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("numpy", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from services.capture.detect_coordinator import CameraPlateTrack, DetectCoordinator, bbox_iou
from services.session.session_manager import SessionManager
from test_weight_stability import make_frame


class PlateTrackTests(unittest.TestCase):
    def test_iou_and_two_hit_confirmation(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [5, 0, 15, 10]), 1 / 3)
        track = CameraPlateTrack("cam1")
        track.observe([{"bbox": [0, 0, 10, 10], "det_conf": 0.8}])
        self.assertFalse(track.valid)
        track.observe([{"bbox": [1, 0, 11, 10], "det_conf": 0.9}])
        self.assertTrue(track.valid)
        old_id = track.track_id
        track.observe([{"bbox": [30, 0, 40, 10], "det_conf": 0.9}])
        self.assertFalse(track.valid)
        self.assertGreater(track.track_id, old_id)

    def test_valid_track_expires_after_camera_stalls(self):
        track = CameraPlateTrack("cam1")
        region = [{"bbox": [0, 0, 10, 10], "det_conf": 0.9}]
        track.observe(region, frame_id=1, observed_at=10.0)
        track.observe(region, frame_id=2, observed_at=10.1)
        self.assertTrue(track.valid)
        self.assertFalse(track.expire(now=13.09, stale_seconds=3.0))
        self.assertTrue(track.expire(now=13.1, stale_seconds=3.0))
        self.assertFalse(track.valid)

    def test_live_ocr_runs_on_confirmation_then_periodically(self):
        track = CameraPlateTrack("cam1")
        region = [{"bbox": [0, 0, 10, 10], "det_conf": 0.9}]

        track.observe(region, observed_at=10.0)
        self.assertFalse(track.claim_live_ocr(now=10.0, interval=0.5))
        track.observe(region, observed_at=10.1)
        self.assertTrue(track.claim_live_ocr(now=10.1, interval=0.5))
        track.observe(region, observed_at=10.4)
        self.assertFalse(track.claim_live_ocr(now=10.4, interval=0.5))
        track.observe(region, observed_at=10.6)
        self.assertTrue(track.claim_live_ocr(now=10.6, interval=0.5))

    def test_new_track_gets_ocr_after_confirmation(self):
        track = CameraPlateTrack("cam1")
        first = [{"bbox": [0, 0, 10, 10], "det_conf": 0.9}]
        second = [{"bbox": [30, 0, 40, 10], "det_conf": 0.9}]
        track.observe(first, observed_at=10.0)
        track.observe(first, observed_at=10.1)
        self.assertTrue(track.claim_live_ocr(now=10.1, interval=10.0))

        track.observe(second, observed_at=10.2)
        self.assertFalse(track.claim_live_ocr(now=10.2, interval=10.0))
        track.observe(second, observed_at=10.3)
        self.assertTrue(track.claim_live_ocr(now=10.3, interval=10.0))

    def test_track_follows_iou_match_when_other_region_has_higher_confidence(self):
        track = CameraPlateTrack("cam1")
        tracked = {"bbox": [0, 0, 10, 10], "det_conf": 0.8}
        other = {"bbox": [30, 0, 40, 10], "det_conf": 0.99}

        track.observe([tracked])
        selected = track.observe([tracked, other])

        self.assertIs(selected, tracked)
        self.assertTrue(track.valid)
        self.assertEqual(track.bbox, tracked["bbox"])

    def test_track_metadata_keeps_obb_for_deferred_crop(self):
        track = CameraPlateTrack("cam1", confirm_hits=1)
        obb = [[1, 2], [10, 2], [10, 8], [1, 8]]

        track.observe([{
            "bbox": [1, 2, 10, 8], "obb": obb, "det_conf": 0.9,
            "class": "BSV", "two_row": True,
        }], frame_id=7)

        self.assertEqual(track.metadata()[0]["obb"], obb)
        self.assertEqual(track.metadata()[0]["class"], "BSV")
        self.assertTrue(track.metadata()[0]["two_row"])

    def test_detection_submits_only_confirmed_best_region_to_ocr(self):
        camera = Mock(name="camera")
        camera.name = "cam1"
        camera.lpr_crop = "full"
        camera.inference_lock = unittest.mock.MagicMock()
        best = {"bbox": [0, 0, 10, 10], "det_conf": 0.9}
        extra = {"bbox": [30, 0, 40, 10], "det_conf": 0.5}
        coordinator = DetectCoordinator([camera], Mock())
        coordinator._enabled = True
        coordinator._detect_regions_fn = Mock(return_value=[best, extra])
        coordinator._recognize_regions_fn = Mock()
        coordinator._submit_ocr_job = Mock()
        frame = Mock()
        frame.shape = (100, 100, 3)

        coordinator._run_detection(camera, frame, frame_id=1)
        coordinator._run_detection(camera, frame, frame_id=2)

        coordinator._submit_ocr_job.assert_called_once()
        self.assertEqual(coordinator._submit_ocr_job.call_args.args[2], [best])

    def test_detect_loop_does_not_resubmit_same_frame_generation(self):
        camera = Mock(name="camera")
        camera.name = "cam1"
        camera.peek_latest_frame_with_id.return_value = (Mock(), 9)
        coordinator = DetectCoordinator([camera], Mock())
        coordinator._enabled = True
        coordinator._running = True
        coordinator._detect_events = {"cam1": Mock()}
        coordinator._detect_locks = {"cam1": unittest.mock.MagicMock()}
        coordinator._detect_jobs = {"cam1": None}

        waits = 0

        def stop_after_two(_seconds):
            nonlocal waits
            waits += 1
            if waits == 2:
                coordinator._running = False

        with patch("services.capture.detect_coordinator.time.sleep", side_effect=stop_after_two):
            coordinator._detect_loop()

        self.assertEqual(coordinator._detect_events["cam1"].set.call_count, 1)
        camera.peek_latest_frame_with_id.assert_called_with(copy_frame=False)


class SessionStrategyTests(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager(Mock(), lpr_grabbers={})
        self.manager._update_peak_candidate = Mock()
        self.log = Mock()

    @staticmethod
    def feed(manager, weights, log):
        for weight in weights:
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            manager.on_frame(frame, log)

    def test_five_falls_and_one_rise_confirm_falling_without_new_session(self):
        self.feed(self.manager, [10000, 9700, 9400, 9500, 9000, 8500], self.log)
        self.assertFalse(self.manager.session.session_active)

    def test_confirmed_rise_starts_scale_session(self):
        self.feed(self.manager, [8500, 8800, 9100, 9000, 9400, 9800], self.log)
        self.assertTrue(self.manager.session.session_active)
        self.assertFalse(self.manager._plate_owned)

    def test_plate_starts_without_weight_and_resets_idle_stable_value(self):
        self.manager.session.stable_weight = 12000
        self.manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, self.log)
        self.assertTrue(self.manager.session.session_active)
        self.assertTrue(self.manager._plate_owned)
        self.assertIsNone(self.manager.session.stable_weight)

    def test_plate_upgrades_scale_session_without_new_id(self):
        self.feed(self.manager, [1000, 1200, 1400, 1300, 1600, 1800], self.log)
        session_id = self.manager.session.session_id
        self.manager.on_plate_presence("cam3", {"cam1": False, "cam3": True}, self.log)
        self.assertEqual(self.manager.session.session_id, session_id)
        self.assertTrue(self.manager._plate_owned)

    def test_one_camera_presence_cancels_loss(self):
        self.manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, self.log)
        self.manager.on_plate_presence("cam1", {"cam1": False, "cam3": False}, self.log)
        self.manager.on_plate_presence("cam3", {"cam1": False, "cam3": True}, self.log)
        self.assertTrue(self.manager.session.session_active)
        self.assertIsNone(self.manager._plate_absent_since)

    def test_both_camera_loss_for_one_second_ends_plate_session(self):
        with patch("services.session.session_manager.time.monotonic", side_effect=[0.0, 0.0, 1.1]):
            self.manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, self.log)
            self.manager.on_plate_presence("cam1", {"cam1": False, "cam3": False}, self.log)
            self.manager.on_plate_presence("cam3", {"cam1": False, "cam3": False}, self.log)
        self.assertFalse(self.manager.session.session_active)

    def test_scale_callback_completes_plate_loss_deadline(self):
        with patch("services.session.session_manager.time.monotonic", side_effect=[0.0, 0.0, 1.1]):
            self.manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, self.log)
            self.manager.on_plate_presence("cam1", {"cam1": False, "cam3": False}, self.log)
            frame = make_frame(5000)
            frame.status = "UNSTABLE"
            self.manager.on_frame(frame, self.log)
        self.assertFalse(self.manager.session.session_active)


if __name__ == "__main__":
    unittest.main()
