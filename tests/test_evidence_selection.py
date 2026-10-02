import unittest
from unittest.mock import Mock

import sys

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from services.session import evidence_selection as selection


class TargetTimestampTests(unittest.TestCase):
    def test_ocr_prefers_local_peak_then_falls_back_to_ended_at(self):
        metadata = {
            "local_peak_observed_at": "2026-07-24T00:00:10+00:00",
            "weight_observed_at": "2026-07-24T00:00:05+00:00",
            "ended_at": "2026-07-24T00:00:20+00:00",
        }
        target = selection.select_target_timestamp(metadata, "UNKNOWN_OCR")
        self.assertIsNotNone(target)

        legacy = selection.select_target_timestamp(metadata, "UNKNOWN")
        self.assertLess(legacy, target)

    def test_invalid_timestamp_returns_none(self):
        self.assertIsNone(selection.select_target_timestamp({}, "UNKNOWN_OCR"))


class RankCombinationTests(unittest.TestCase):
    def test_unknown_plate_prefers_closest_to_target(self):
        candidate_sets = {
            "cam1": [{"timestamp": 100.0}, {"timestamp": 103.0}],
            "cam3": [{"timestamp": 101.0}],
        }
        ranked = selection.rank_combinations(["cam1", "cam3"], candidate_sets, 100.5, "UNKNOWN_OCR")
        first = ranked[0]
        self.assertEqual([item["timestamp"] for item in first], [100.0, 101.0])

    def test_no_cameras_returns_empty(self):
        self.assertEqual(selection.rank_combinations([], {}, 0.0, "UNKNOWN_OCR"), [])


class SynchronizeCameraTests(unittest.TestCase):
    def test_skewed_camera_is_rejected(self):
        candidate_sets = {
            "cam1": [{"timestamp": 100.0, "captured_at": "a"}],
            "cam2": [{"timestamp": 100.2, "captured_at": "b"}],
            "cam3": [{"timestamp": 105.0, "captured_at": "c"}],
        }
        rejected = {}
        kept = selection.synchronize_cameras(
            ["cam1", "cam2", "cam3"], candidate_sets, 100.0, "local_peak", rejected,
        )
        self.assertEqual(kept, ["cam1", "cam2"])
        self.assertEqual(candidate_sets["cam3"], [])
        self.assertEqual(rejected["cam3"]["reason"], "inter_camera_skew")


class DetectorTrackTests(unittest.TestCase):
    def test_exact_path_wins(self):
        tracks = [{"confidence": 0.1}]
        resolved = selection.resolve_detector_tracks(
            "cam1", "cam1-0-frame.jpg", {"cam1-0-frame.jpg": {"tracks": tracks}}, {},
        )
        self.assertIs(resolved, tracks)

    def test_same_camera_best_confidence_fallback(self):
        metadata = {"lpr_diagnostics": {"detected_boxes": {
            "cam1-1-a.jpg": [{"confidence": 0.2}],
            "cam1-2-b.jpg": [{"confidence": 0.9}],
            "cam3-1-c.jpg": [{"confidence": 0.99}],
        }}}
        resolved = selection.resolve_detector_tracks("cam1", "cam1-9-missing.jpg", {}, metadata)
        self.assertEqual(resolved, [{"confidence": 0.9}])


class StartWindowTests(unittest.TestCase):
    def test_out_of_window_dedicated_rejected_and_timeline_added(self):
        start_candidates = {"cam1": [{"timestamp": 50.0, "captured_at": "late"}],
                            "cam2": [], "cam3": []}
        timeline = {"cam1": [{"timestamp": 10.1, "captured_at": "ok"}],
                    "cam2": [], "cam3": []}
        rejected = {}
        selection.apply_start_window(start_candidates, timeline, 10.0, rejected)
        self.assertEqual([c["captured_at"] for c in start_candidates["cam1"]], ["ok"])
        self.assertEqual(rejected["cam1"]["reason"], "late_or_invalid_timestamp")


if __name__ == "__main__":
    unittest.main()