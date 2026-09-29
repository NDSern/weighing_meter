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

    def test_excluded_lpr_fallback_is_absent(self):
        from services.runtime import bootstrap

        with open(bootstrap.__file__.replace(".pyc", ".py"), encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn("fallback_detector", source)
        self.assertNotIn("detect_axis_plate_regions", source)


if __name__ == "__main__":
    unittest.main()