import unittest
import sys
import threading
import numpy as np
from types import SimpleNamespace
from unittest import mock

sys.modules.setdefault("cv2", mock.Mock())
sys.modules.setdefault("minio", mock.Mock())
sys.modules.setdefault("minio.error", mock.Mock())

from services.capture import frame_source
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

    def test_gst_unavailable_uses_opencv_capture_backend(self):
        source = CameraGrabber("rtsp://camera/main")
        source._grab_with_opencv = mock.Mock()

        with mock.patch.object(frame_source, "_GST_AVAILABLE", False):
            source._grab_loop()

        source._grab_with_opencv.assert_called_once()

    def test_green_frame_guard_rejects_decoder_corruption(self):
        corrupt = np.zeros((128, 128, 3), dtype=np.uint8)
        corrupt[:, :, 1] = 64
        valid = np.zeros((128, 128, 3), dtype=np.uint8)
        valid[:, :, 2] = 10

        self.assertTrue(CameraGrabber._is_corrupt_green_frame(corrupt))
        self.assertFalse(CameraGrabber._is_corrupt_green_frame(valid))


class CaptureLifecycleTests(unittest.TestCase):
    """The grab loop owns the VideoCapture lifetime; stop() must never release it."""

    def setUp(self):
        self.logs = []
        set_log_fn(lambda level, message: self.logs.append((level, message)))
        self.addCleanup(set_log_fn, None)

    def test_opencv_capture_sets_finite_timeouts(self):
        cap = mock.Mock()

        with mock.patch.object(frame_source.cv2, "VideoCapture", return_value=cap):
            source = CameraGrabber("rtsp://camera/main")
            source._open_capture()

        cap.set.assert_any_call(frame_source.cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
        cap.set.assert_any_call(frame_source.cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

    def test_stop_does_not_release_an_in_flight_capture(self):
        cap = mock.Mock()
        cap.isOpened.return_value = True
        in_flight = threading.Event()
        release_block = threading.Event()

        def fake_grab():
            in_flight.set()
            release_block.wait(2.0)
            return True

        cap.grab.side_effect = fake_grab
        cap.retrieve.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))

        source = CameraGrabber("rtsp://camera/main")
        source._opencv_capture = lambda: cap
        with mock.patch.object(frame_source, "_GST_AVAILABLE", False):
            source.start()
            self.assertTrue(in_flight.wait(2.0), "grab loop never started")
            # stop() while the loop is inside cap.grab() must not release the
            # capture concurrently (undefined behaviour / FFmpeg crash).
            self.assertFalse(source.stop(timeout=0.2))
            cap.release.assert_not_called()
            release_block.set()
        source._thread.join(2.0)
        self.assertEqual(cap.release.call_count, 1)


if __name__ == "__main__":
    unittest.main()
