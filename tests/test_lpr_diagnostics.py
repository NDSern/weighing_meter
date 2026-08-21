import unittest
import sys
from types import SimpleNamespace
from unittest import mock

sys.modules.setdefault("cv2", mock.Mock())
sys.modules.setdefault("minio", mock.Mock())
sys.modules.setdefault("minio.error", mock.Mock())

from services.capture.frame_source import CameraGrabber, set_log_fn
from services.session.session_manager import classify_lpr_failure


class LprFailureClassificationTests(unittest.TestCase):
    def test_failure_precedence(self):
        cases = [
            ({"available_lpr_frames": 0}, "lpr_frames_unavailable"),
            ({"available_lpr_frames": 1, "detector_errors": 1}, "detector_inference_error"),
            ({"available_lpr_frames": 1, "detector_successes": 2}, "no_plate_detection"),
            ({"available_lpr_frames": 1, "crop_failures": 1, "detector_successes": 2}, "crop_failed"),
            ({"available_lpr_frames": 1, "ocr_blank": 1, "crop_failures": 1}, "plate_detected_ocr_blank"),
            ({"available_lpr_frames": 1, "ocr_invalid_format": 1, "ocr_blank": 1}, "plate_detected_ocr_invalid_format"),
            ({"available_lpr_frames": 1, "ocr_low_confidence": 1, "ocr_invalid_format": 1}, "plate_detected_ocr_low_confidence"),
            ({"available_lpr_frames": 1, "ocr_valid_candidates": 1, "ocr_errors": 1}, "no_confirmed_plate_after_voting"),
        ]

        for diagnostics, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(classify_lpr_failure(diagnostics), expected)


class RtspResolutionTests(unittest.TestCase):
    def setUp(self):
        self.logs = []
        set_log_fn(lambda level, message: self.logs.append((level, message)))
        self.addCleanup(set_log_fn, None)

    def test_expected_resolution_accepts_matching_frame(self):
        source = CameraGrabber("rtsp://user:secret@camera/main", expected_resolution=(2880, 1624))
        frame = SimpleNamespace(shape=(1624, 2880, 3))

        resolution, accepted = source._check_resolution(frame, None)

        self.assertEqual(resolution, (2880, 1624))
        self.assertTrue(accepted)
        self.assertNotIn("secret", str(self.logs))

    def test_expected_resolution_rejects_substream(self):
        source = CameraGrabber("rtsp://camera/main", expected_resolution=(2880, 1624))
        frame = SimpleNamespace(shape=(448, 800, 3))

        resolution, accepted = source._check_resolution(frame, None)

        self.assertEqual(resolution, (800, 448))
        self.assertFalse(accepted)
        self.assertTrue(any("resolution rejected" in message for _, message in self.logs))


if __name__ == "__main__":
    unittest.main()
