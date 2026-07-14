import unittest

from services.runtime.rknn_models import RknnModelSet


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


if __name__ == "__main__":
    unittest.main()
