import importlib.util
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "migrate_scale_data", ROOT / "scripts" / "migrate_scale_data.py",
)
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


class ScaleDataMigrationTests(unittest.TestCase):
    def make_legacy_database(self, root):
        source = os.path.join(root, "scale_data.db")
        with sqlite3.connect(source) as conn:
            conn.execute(
                "CREATE TABLE weight_log ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
                "weight_kg REAL NOT NULL, sign TEXT NOT NULL, decimal_pos INTEGER NOT NULL, "
                "checksum_ok INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'UNSTABLE')"
            )
            conn.executemany(
                "INSERT INTO weight_log "
                "(id, timestamp, weight_kg, sign, decimal_pos, checksum_ok, status) "
                "VALUES (?, ?, ?, '+', 0, 1, 'STABLE')",
                [
                    (1, "2026-07-14T23:59:59", 100),
                    (2, "2026-07-15T12:00:00", 200),
                    (3, "2026-07-15T00:00:01", 300),
                ],
            )
        return Path(source)

    def test_partitions_rows_and_preserves_source_values(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.make_legacy_database(root)
            destination = Path(root) / "scale_data"
            destination.mkdir()

            migration.migrate(source, destination)

            with sqlite3.connect(destination / "2026-07-14.db") as conn:
                self.assertEqual(conn.execute("SELECT id, weight_kg FROM weight_log").fetchall(), [(1, 100.0)])
            with sqlite3.connect(destination / "2026-07-15.db") as conn:
                self.assertEqual(conn.execute("SELECT id, weight_kg FROM weight_log ORDER BY id").fetchall(), [(2, 200.0), (3, 300.0)])

    def test_merges_rows_into_existing_daily_database(self):
        with tempfile.TemporaryDirectory() as root:
            source = self.make_legacy_database(root)
            destination = Path(root) / "scale_data"
            destination.mkdir()
            target = destination / "2026-07-14.db"
            with sqlite3.connect(target) as conn:
                conn.execute(migration.WEIGHT_LOG_SCHEMA)
                conn.execute(
                    "INSERT INTO weight_log (id, timestamp, weight_kg, sign, decimal_pos, checksum_ok, status) "
                    "VALUES (99, '2026-07-14T00:00:01', 400, '+', 0, 1, 'STABLE')"
                )

            migration.migrate(source, destination)

            with sqlite3.connect(target) as conn:
                self.assertEqual(conn.execute("SELECT count(*) FROM weight_log").fetchone()[0], 2)

    def test_rejects_database_owned_by_another_process(self):
        result = mock.Mock(returncode=0)
        with mock.patch("subprocess.run", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "Stop the service"):
                migration.require_exclusive_source(Path("scale_data.db"))

    def test_rejects_active_daily_scale_writer_lock(self):
        with tempfile.TemporaryDirectory() as root:
            destination = Path(root) / "scale_data"
            destination.mkdir()
            lock = migration.acquire_migration_lock(destination)
            try:
                with self.assertRaisesRegex(RuntimeError, "Stop the service"):
                    migration.acquire_migration_lock(destination)
            finally:
                migration.fcntl.flock(lock, migration.fcntl.LOCK_UN)
                lock.close()


if __name__ == "__main__":
    unittest.main()
