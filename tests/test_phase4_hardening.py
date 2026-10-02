"""Regression tests for the Phase 4 low-priority hardening batch."""

import sys
import unittest
from unittest.mock import Mock

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())


class SafeSheetTitleTests(unittest.TestCase):
    def test_reserved_characters_are_replaced_and_truncated(self):
        from scripts.export_sessions import _safe_sheet_title

        self.assertEqual(_safe_sheet_title("hp1"), "hp1")
        self.assertEqual(_safe_sheet_title("a[b]:c*d?e/f\\g"), "a_b__c_d_e_f_g")
        self.assertEqual(len(_safe_sheet_title("x" * 50)), 31)
        self.assertEqual(_safe_sheet_title(""), "sessions")
        self.assertEqual(_safe_sheet_title(None), "sessions")


class ParseTimestampTests(unittest.TestCase):
    def test_z_and_offset_forms_compare_equal(self):
        from scripts.export_sessions import _parse_ts

        zulu = _parse_ts("2026-10-01T04:00:00Z")
        offset = _parse_ts("2026-10-01T04:00:00+00:00")
        self.assertEqual(zulu, offset)
        self.assertIsNone(_parse_ts(""))
        self.assertIsNone(_parse_ts("not-a-timestamp"))

    def test_in_window_compares_instants_not_strings(self):
        from scripts.export_sessions import _in_window

        rec = {"started_at": "2026-10-01T04:00:00+00:00"}
        self.assertTrue(_in_window(rec, None, "2026-10-01", None, None))
        self.assertFalse(_in_window(rec, None, "2026-10-02", None, None))
        # Mixed representations must still compare correctly by instant.
        self.assertTrue(
            _in_window(rec, None, None, "2026-10-01T03:00:00Z", "2026-10-01T05:00:00Z")
        )
        self.assertFalse(
            _in_window(rec, None, None, "2026-10-01T05:00:00Z", None)
        )


class UnknownCaptureFallbackTests(unittest.TestCase):
    def test_nearest_session_frame_without_started_at_returns_none(self):
        from services.session import unknown_capture

        metadata = {"session_dir": "/tmp/spool", "session_files": ["cam1-0.jpg"]}
        self.assertIsNone(unknown_capture.nearest_session_frame(metadata, "cam1", 10.0))

    def test_load_diagnostic_frames_without_started_at_uses_start_frames(self):
        from services.session import unknown_capture

        sentinel = {"cam1": object()}
        metadata = {}
        with unittest.mock.patch.object(
            unknown_capture, "load_start_frames", return_value=sentinel
        ) as start_frames:
            result = unknown_capture.load_diagnostic_frames(metadata, 1.0)

        start_frames.assert_called_once_with(metadata)
        self.assertIs(result, sentinel)


class VehicleSummaryGuardTests(unittest.TestCase):
    def test_on_weight_tolerates_missing_vehicle_summary(self):
        from test_weight_stability import make_frame
        from services.session.session_manager import SessionManager

        manager = SessionManager(Mock())
        manager.vehicle_tracker = Mock()
        manager._get_vehicle_summary = Mock(return_value=None)
        manager.plate_tracker.get_confirmed_plate = Mock(return_value=(None, 0.0, None))
        manager.session.session_active = True

        # Must not raise even though the summary is None.
        manager.on_weight(make_frame(39120), Mock())


if __name__ == "__main__":
    unittest.main()
