import unittest
import sys
from unittest import mock

import numpy as np

sys.modules.setdefault("cv2", mock.Mock())

from services.pipeline import license_plate_recognition as lpr
from services.pipeline.detector_obb_decode import FEATURE_SIZES, decode_detector_outputs


class OcrInferenceTests(unittest.TestCase):
    def setUp(self):
        original_buffers = lpr._INPUT_BUFFERS
        original_cv2 = lpr.cv2
        self.addCleanup(setattr, lpr, "_INPUT_BUFFERS", original_buffers)
        self.addCleanup(setattr, lpr, "cv2", original_cv2)
        lpr._INPUT_BUFFERS = __import__("threading").local()
        lpr.cv2 = mock.Mock()

    @staticmethod
    def fake_resize(_image, size, interpolation=None):
        result = np.empty((size[1], size[0], 3), dtype=np.uint8)
        result[:] = (10, 20, 30)
        return result

    def test_ocr_preprocessing_reuses_bucket_buffer(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        with mock.patch.object(lpr.cv2, "resize", side_effect=self.fake_resize):
            first, _ = lpr._preprocess_ocr(image)
            second, _ = lpr._preprocess_ocr(image)

        self.assertIs(first, second)
        self.assertTrue(first.flags.c_contiguous)

    def test_ocr_preprocessing_preserves_rgb_normalization_and_padding(self):
        image = np.zeros((20, 41, 3), dtype=np.uint8)
        with mock.patch.object(lpr.cv2, "resize", side_effect=self.fake_resize):
            blob, ratio = lpr._preprocess_ocr(image)

        np.testing.assert_allclose(blob[0, 0, 0], np.array([30, 20, 10]) / 127.5 - 1.0)
        np.testing.assert_allclose(blob[0, 0, -1], [-1.0, -1.0, -1.0])
        self.assertAlmostEqual(ratio, 98 / 112)

    def test_detector_preprocessing_reuses_size_buffer(self):
        image = np.zeros((20, 40, 3), dtype=np.uint8)
        with mock.patch.object(lpr.cv2, "resize", side_effect=self.fake_resize):
            first, *_ = lpr._preprocess_detector(image, 640)
            second, *_ = lpr._preprocess_detector(image, 640)

        self.assertIs(first, second)
        self.assertTrue(first.flags.c_contiguous)

    def test_raw_detector_head_decode(self):
        count = sum(size * size for size in FEATURE_SIZES)
        outputs = [
            np.full((1, 4, count), 7.5, dtype=np.float32),
            np.full((1, 2, count), 0.5, dtype=np.float32),
            np.full((1, 1, count), np.pi / 4, dtype=np.float32),
        ]

        decoded = decode_detector_outputs(outputs)

        self.assertEqual(decoded.shape, (1, 7, count))
        np.testing.assert_allclose(decoded[0, :4, 0], [4.0, 4.0, 120.0, 120.0])
        np.testing.assert_allclose(decoded[0, 4:6, 0], [0.5, 0.5])
        self.assertAlmostEqual(float(decoded[0, 6, 0]), np.pi / 4)

    def test_raw_detector_head_rejects_wrong_contract(self):
        with self.assertRaisesRegex(ValueError, "Expected distances"):
            decode_detector_outputs([
                np.zeros((1, 3, 18900), dtype=np.float32),
                np.zeros((1, 2, 18900), dtype=np.float32),
                np.zeros((1, 1, 18900), dtype=np.float32),
            ])

    def test_restricted_probabilities_are_renormalized_without_softmax(self):
        values = np.zeros((2, 38), dtype=np.float32)
        values[:, 0] = 0.2
        values[:, 1] = 0.6

        probabilities = lpr._to_probs(values)

        np.testing.assert_allclose(probabilities[:, 0], 0.25)
        np.testing.assert_allclose(probabilities[:, 1], 0.75)

    def test_restricted_probability_contract_rejects_logits(self):
        values = np.zeros((2, 38), dtype=np.float32)
        values[:, 0] = -1.0

        with self.assertRaisesRegex(ValueError, "within"):
            lpr._to_probs(values)

    def test_fine_tuned_charset_contract_is_exact(self):
        charset = ["[blank]", *list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), " "]

        lpr.validate_lpr_charset(charset)

        with self.assertRaisesRegex(ValueError, "charset"):
            lpr.validate_lpr_charset(charset[:-1])

    def test_ocr_output_contract_rejects_transposed_class_dimension(self):
        ocr = mock.Mock()
        ocr.inference.return_value = [np.zeros((1, 38, 6), dtype=np.float32)]

        with mock.patch.object(
            lpr, "_preprocess_ocr", return_value=(np.zeros((1, 48, 32, 3)), 1.0)
        ):
            with self.assertRaisesRegex(RuntimeError, "class dimension"):
                lpr._run_ocr_logits(
                    ocr,
                    np.zeros((10, 20, 3), dtype=np.uint8),
                    ["[blank]", *list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), " "],
                )

    def test_combined_one_row_decoders_share_one_inference(self):
        logits = np.zeros((6, 4), dtype=np.float32)
        ocr = mock.Mock()
        ocr.inference.return_value = [logits[None, ...]]
        seen = []

        def greedy(value, _charset):
            seen.append(value)
            return "30A12345", 0.8

        def topk(value, _charset, topk=None):
            seen.append(value)
            return [("30A12345", 0.9)]

        with mock.patch.object(lpr, "_prepare_crop_for_ocr", side_effect=lambda image: image), \
             mock.patch.object(lpr, "_preprocess_ocr", return_value=(np.zeros((1, 1, 1, 3)), 1.0)), \
             mock.patch.object(lpr, "_ctc_greedy_decode", side_effect=greedy), \
             mock.patch.object(lpr, "_ctc_decode_topk", side_effect=topk):
            plate, confidence, raw, candidates = lpr.recognize_combined(
                ocr, ["[blank]", "3", "0", "A"], np.zeros((10, 20, 3), dtype=np.uint8)
            )

        self.assertEqual(ocr.inference.call_count, 1)
        self.assertIs(seen[0], seen[1])
        self.assertEqual((plate, confidence, raw), ("30A-123.45", 0.9, "30A-123.45"))
        self.assertEqual(candidates, [("30A12345", 0.9)])

    def test_combined_two_row_runs_once_per_row(self):
        logits = np.zeros((6, 4), dtype=np.float32)
        ocr = mock.Mock()
        ocr.inference.return_value = [logits[None, ...]]

        with mock.patch.object(lpr, "_prepare_crop_for_ocr", side_effect=lambda image: image), \
             mock.patch.object(lpr, "_preprocess_ocr", return_value=(np.zeros((1, 1, 1, 3)), 1.0)), \
             mock.patch.object(lpr, "_ctc_greedy_decode", return_value=("30A12345", 0.8)), \
             mock.patch.object(lpr, "_ctc_decode_topk", return_value=[("30A12345", 0.9)]):
            lpr.recognize_combined(
                ocr, ["[blank]", "3", "0", "A"], np.zeros((20, 20, 3), dtype=np.uint8), two_row=True
            )

        self.assertEqual(ocr.inference.call_count, 2)


if __name__ == "__main__":
    unittest.main()
