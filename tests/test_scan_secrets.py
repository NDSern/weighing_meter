import tempfile
import unittest
from pathlib import Path


from scripts import scan_secrets as module


class ScanSecretsTests(unittest.TestCase):
    def test_flags_non_empty_credential_assignment(self):
        findings = module.scan_text('MQTT_PASSWORD = "live-value"\n')

        self.assertEqual(findings, [(1, "credential assignment")])

    def test_allows_empty_credential_assignment(self):
        findings = module.scan_text('MINIO_SECRET_KEY = ""\n')

        self.assertEqual(findings, [])

    def test_flags_rtsp_embedded_credentials(self):
        findings = module.scan_text('RTSP_URL = "rtsp://admin:secret@10.0.0.5:554/live"\n')

        self.assertEqual(findings, [(1, "rtsp credentials")])

    def test_allows_rtsp_url_without_credentials(self):
        findings = module.scan_text('RTSP_URL = "rtsp://10.0.0.5:554/live"\n')

        self.assertEqual(findings, [])

    def test_scan_tree_reports_relative_paths(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "service.py"
            source.write_text('MQTT_PASSWORD = "live-value"\n', encoding="utf-8")
            module.tracked_files = lambda _root: ["service.py"]

            findings = module.scan_tree(root)

        self.assertEqual(findings, [("service.py", 1, "credential assignment")])


if __name__ == "__main__":
    unittest.main()