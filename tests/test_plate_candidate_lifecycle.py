import sys
import unittest
from unittest.mock import Mock, patch

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from d2008_scale_reader import WeightFrame
from services.session.session_manager import SessionManager


def frame(weight):
    value = WeightFrame(b"", "+", weight, 0, True, f"{int(weight):06d}")
    value.status = "UNSTABLE"
    return value


class PlateCandidateLifecycleTests(unittest.TestCase):
    def test_candidate_defers_spool_until_scale_occupancy(self):
        spool = Mock()
        spool.begin_session.return_value = "/spool/session"
        manager = SessionManager(Mock(), frame_spool=spool)
        log = Mock()

        manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, log)

        self.assertTrue(manager.session.session_active)
        self.assertTrue(manager._plate_owned)
        self.assertFalse(manager.session.scale_owned)
        spool.begin_session.assert_not_called()

        manager._update_peak_candidate = Mock()
        manager.on_frame(frame(101), log)

        self.assertTrue(manager.session.scale_owned)
        self.assertFalse(manager._plate_owned)
        spool.begin_session.assert_called_once()

    def test_scale_owned_session_ends_empty_despite_live_plate(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        manager.session.scale_owned = True
        manager._plate_owned = True
        manager._end_session = Mock(return_value=True)
        empty = frame(0)
        log = Mock()

        with patch("services.session.session_manager.time.time", side_effect=[10.0, 12.1]):
            self.assertFalse(manager._check_scale_empty(empty, log))
            self.assertTrue(manager._check_scale_empty(empty, log))

        manager._end_session.assert_called_once_with("scale_empty", log)

    def test_scale_owned_session_ignores_falling_weight_trend(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        manager.session.scale_owned = True
        manager.session.weight_trend_window.clear()
        manager._end_session = Mock()

        for weight in range(40000, 38000, -100):
            manager.on_frame(frame(weight), Mock())

        manager._end_session.assert_not_called()
        self.assertTrue(manager.session.session_active)

    def test_loaded_plate_session_promotes_before_expired_track_loss(self):
        spool = Mock()
        spool.begin_session.return_value = "/spool/session"
        manager = SessionManager(Mock(), frame_spool=spool)
        log = Mock()

        manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, log)
        with patch("services.session.session_manager.time.monotonic", return_value=0.0):
            manager.on_plate_presence("cam1", {"cam1": False, "cam3": False}, log)
        manager._end_session = Mock()

        with patch("services.session.session_manager.time.monotonic", return_value=1.1):
            manager.on_frame(frame(5000), log)

        self.assertTrue(manager.session.session_active)
        self.assertTrue(manager.session.scale_owned)
        self.assertFalse(manager._plate_owned)
        manager._end_session.assert_not_called()

    def test_lpr_session_controls_camera_lights(self):
        lights = Mock()
        manager = SessionManager(Mock(), lpr_light_controller=lights)
        log = Mock()

        manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, log)
        manager._end_session("both_plate_tracks_lost", log)

        self.assertEqual(
            lights.set_lpr_active.call_args_list,
            [
                unittest.mock.call(True, log),
                unittest.mock.call(False, log),
            ],
        )

    def test_timeout_blocks_persistent_plate_until_track_loss(self):
        manager = SessionManager(Mock())
        log = Mock()
        manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, log)
        manager._end_session = Mock(return_value=True)
        manager.session.started_at = 1.0

        with patch("services.session.session_manager.time.time", return_value=181.0):
            self.assertTrue(manager._check_plate_only_timeout(log))

        self.assertTrue(manager._plate_rearm_blocked)
        manager.session.session_active = False
        manager.on_plate_presence("cam1", {"cam1": True, "cam3": False}, log)
        self.assertTrue(manager._plate_rearm_blocked)
        manager.on_plate_presence("cam1", {"cam1": False, "cam3": False}, log)
        self.assertFalse(manager._plate_rearm_blocked)


if __name__ == "__main__":
    unittest.main()
