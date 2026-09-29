import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path


def load_cleanup_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_absorbed_peak_candidates.py"
    spec = importlib.util.spec_from_file_location("cleanup_absorbed_peak_candidates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AbsorbedPeakCleanupTests(unittest.TestCase):
    def setUp(self):
        self.cleanup = load_cleanup_script()

    @staticmethod
    def write_record(root, item_id, category="absorbed_active_session", images=None):
        root = Path(root)
        day = root / "2026" / "09" / "28"
        day.mkdir(parents=True, exist_ok=True)
        if images is None:
            image = day / f"{item_id}_cam1.jpg"
            image.write_bytes(b"image")
            images = [str(image)]
        metadata = day / f"{item_id}.json"
        metadata.write_text(json.dumps({"category": category, "images": images}))
        return metadata, [Path(path) for path in images]

    def test_dry_run_reports_exact_files_without_mutation(self):
        with tempfile.TemporaryDirectory() as root:
            metadata, images = self.write_record(root, "absorbed")

            report = self.cleanup.inventory(root)

            self.assertEqual(len(report["candidates"]), 1)
            self.assertEqual(report["files"], 2)
            self.assertEqual(report["unsafe"], [])
            self.assertTrue(metadata.exists())
            self.assertTrue(images[0].exists())

    def test_apply_removes_only_absorbed_record_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as root:
            absorbed, absorbed_images = self.write_record(root, "absorbed")
            retained, retained_images = self.write_record(root, "retained", "stable_session")

            first = self.cleanup.apply_cleanup(self.cleanup.inventory(root))
            second = self.cleanup.apply_cleanup(self.cleanup.inventory(root))

            self.assertEqual(first["deleted_records"], 1)
            self.assertEqual(first["deleted_images"], 1)
            self.assertEqual(first["failed"], 0)
            self.assertEqual(second["deleted_records"], 0)
            self.assertFalse(absorbed.exists())
            self.assertFalse(absorbed_images[0].exists())
            self.assertTrue(retained.exists())
            self.assertTrue(retained_images[0].exists())

    def test_unsafe_records_are_reported_and_untouched(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            root_path = Path(root)
            day = root_path / "2026" / "09" / "28"
            day.mkdir(parents=True)
            malformed = day / "malformed.json"
            malformed.write_text("{")
            unknown, _ = self.write_record(root, "unknown", "unexpected_category")
            outside_image = Path(outside) / "outside.jpg"
            outside_image.write_bytes(b"outside")
            escaped, _ = self.write_record(root, "escaped", images=[str(outside_image)])
            missing, _ = self.write_record(
                root, "missing", images=[str(day / "missing_cam1.jpg")],
            )
            symlink = day / "linked.jpg"
            symlink.symlink_to(outside_image)
            linked, _ = self.write_record(root, "linked", images=[str(symlink)])

            report = self.cleanup.inventory(root)
            reasons = {path.name: reason for path, reason in report["unsafe"]}

            self.assertEqual(reasons["malformed.json"], "invalid_metadata")
            self.assertEqual(reasons["unknown.json"], "unknown_category")
            self.assertEqual(reasons["escaped.json"], "image_outside_root")
            self.assertEqual(reasons["missing.json"], "image_missing")
            self.assertEqual(reasons["linked.json"], "image_symlink")
            for path in (malformed, unknown, escaped, missing, linked, outside_image, symlink):
                self.assertTrue(path.exists())

            result = self.cleanup.apply_cleanup(report)
            self.assertEqual(result["deleted_records"], 0)
            self.assertEqual(result["deleted_images"], 0)
            for path in (malformed, unknown, escaped, missing, linked, outside_image, symlink):
                self.assertTrue(path.exists())

    def test_parent_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            root_path = Path(root)
            linked_dir = root_path / "linked-dir"
            linked_dir.symlink_to(outside, target_is_directory=True)
            outside_image = Path(outside) / "outside.jpg"
            outside_image.write_bytes(b"outside")
            metadata, _ = self.write_record(
                root, "parent-link", images=[str(linked_dir / "outside.jpg")],
            )

            report = self.cleanup.inventory(root)

            self.assertIn((metadata, "image_outside_root"), report["unsafe"])
            self.assertTrue(outside_image.exists())

    def test_metadata_parent_symlink_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            root_path = Path(root)
            outside_path = Path(outside)
            linked_dir = root_path / "linked-dir"
            linked_dir.symlink_to(outside_path, target_is_directory=True)
            image = outside_path / "image.jpg"
            image.write_bytes(b"outside")
            metadata = outside_path / "record.json"
            metadata.write_text(json.dumps({
                "category": "absorbed_active_session",
                "images": [str(image)],
            }))

            record, reason = self.cleanup._load_record(
                linked_dir / "record.json", root_path,
            )

            self.assertIsNone(record)
            self.assertEqual(reason, "metadata_outside_root")
            self.assertTrue(metadata.exists())
            self.assertTrue(image.exists())


if __name__ == "__main__":
    unittest.main()
