import sys
import tempfile
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

import config

# d5310f0's session_manager imports these; provide defaults when config omits them.
for _name, _default in (
    ("UNKNOWN_PHOTO_LOCAL_PEAK_DWELL_SECONDS", 1.0),
    ("UNKNOWN_PHOTO_LOCAL_PEAK_DROP_KG", 300.0),
):
    if not hasattr(config, _name):
        setattr(config, _name, _default)

from services.session import session_manager
from services.session.session_manager import SessionManager


class DiagnosticArchiveConfigTests(unittest.TestCase):
    def test_archive_happens_before_expiry(self):
        self.assertGreaterEqual(config.DIAGNOSTIC_ARCHIVE_AFTER_DAYS, 1)
        self.assertGreater(
            config.DIAGNOSTIC_ARCHIVE_RETENTION_DAYS,
            config.DIAGNOSTIC_ARCHIVE_AFTER_DAYS,
        )
        self.assertGreater(config.DIAGNOSTIC_ARCHIVE_CHECK_INTERVAL_SECONDS, 0)


class CapturePathTests(unittest.TestCase):
    def test_capture_paths_no_longer_include_unchosen_images(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            session_manager, "CAPTURE_DIR", tmp
        ):
            manager = SessionManager(Mock())
            paths = manager._prepare_capture_paths(datetime.now(), "30A-12345", "sid")
        self.assertIn("cam1", paths)
        self.assertIn("cam3", paths)
        self.assertEqual([key for key in paths if key.startswith("unchosen")], [])


if __name__ == "__main__":
    unittest.main()