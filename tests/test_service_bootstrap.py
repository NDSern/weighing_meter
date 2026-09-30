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


if __name__ == "__main__":
    unittest.main()
