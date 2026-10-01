"""Smoke tests that the production entry point can be imported and constructed.

These guard against config/handler drift that module-level unit tests miss
because they never import the runtime bootstrap.
"""

import sys
import unittest
from unittest import mock

for _name in ("cv2", "minio", "minio.error", "serial", "rknnlite", "rknnlite.api"):
    sys.modules.setdefault(_name, mock.MagicMock())


class BootstrapSmokeTests(unittest.TestCase):
    def test_entry_points_import(self):
        from services.runtime import bootstrap

        self.assertTrue(callable(bootstrap.construct_service))
        self.assertTrue(callable(bootstrap.shutdown_service))
        self.assertTrue(callable(bootstrap.configure_module_logging))

    def test_service_resources_fields(self):
        from services.runtime import bootstrap

        res = bootstrap.ServiceResources()
        for attr in ("session_manager", "reader", "mqtt_svc", "frame_spool", "deferred_lpr"):
            self.assertTrue(hasattr(res, attr), attr)

    def test_lpr_bundle_paths_leave_fallback_disabled_by_default(self):
        from services.runtime import bootstrap

        self.assertNotIn("fallback_detector", bootstrap._lpr_bundle_paths(""))

    def test_lpr_bundle_paths_include_configured_fallback(self):
        from services.runtime import bootstrap

        self.assertEqual(
            bootstrap._lpr_bundle_paths("/models/fallback.rknn")["fallback_detector"],
            "/models/fallback.rknn",
        )


class MaskUrlSecretTests(unittest.TestCase):
    def _masks(self):
        from services.capture import frame_source
        from services.runtime import bootstrap

        return (("bootstrap", bootstrap.mask_url_secret), ("frame_source", frame_source.mask_url_secret))

    def test_masks_password_containing_at_sign(self):
        url = "rtsp://admin:p@ss@192.168.1.181:554/live/0/MAIN"
        for name, mask in self._masks():
            masked = mask(url)
            self.assertNotIn("ss@192", masked, name)
            self.assertNotIn("p@ss", masked, name)
            self.assertEqual(masked, "rtsp://admin:***@192.168.1.181:554/live/0/MAIN", name)

    def test_masks_simple_password(self):
        url = "rtsp://admin:123456@192.168.1.20:554/ch01/0"
        for name, mask in self._masks():
            self.assertEqual(mask(url), "rtsp://admin:***@192.168.1.20:554/ch01/0", name)

    def test_leaves_url_without_credentials_unchanged(self):
        url = "rtsp://192.168.1.20:554/ch01/0"
        for name, mask in self._masks():
            self.assertEqual(mask(url), url, name)


if __name__ == "__main__":
    unittest.main()
