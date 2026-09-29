import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import timedelta
from pathlib import Path


from scripts import recover_missed_session as module


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "audited_scale_cycles.json"


class AuditedScaleFixtureTests(unittest.TestCase):
    """Sanitized reproductions of the three audited missed-weight cycles."""

    def _cycle_rows(self, case, local_start, local_end):
        return [
            (local_start - timedelta(seconds=5), 0, "STABLE"),
            (local_start + timedelta(seconds=1), 1000, "UNSTABLE"),
            (local_start + timedelta(seconds=3), case["raw_peak_weight"], "UNSTABLE"),
            (local_start + timedelta(seconds=5), case["selected_weight"], "STABLE"),
            (local_start + timedelta(seconds=6), case["selected_weight"], "STABLE"),
            (local_start + timedelta(seconds=7), case["selected_weight"], "STABLE"),
            (local_end - timedelta(seconds=3), 0, "STABLE"),
            (local_end, 0, "STABLE"),
        ]

    def _write_days(self, scale_dir, rows):
        by_day = {}
        for timestamp, weight, status in rows:
            by_day.setdefault(timestamp.date(), []).append((timestamp, weight, status))
        for day, day_rows in by_day.items():
            database = scale_dir / f"{day.isoformat()}.db"
            with closing(sqlite3.connect(database)) as connection:
                connection.execute(
                    "CREATE TABLE weight_log ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
                    "weight_kg REAL NOT NULL, sign TEXT NOT NULL, decimal_pos INTEGER NOT NULL, "
                    "checksum_ok INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'UNSTABLE')"
                )
                connection.executemany(
                    "INSERT INTO weight_log (timestamp, weight_kg, sign, decimal_pos, "
                    "checksum_ok, status) VALUES (?, ?, '+', 0, 1, ?)",
                    [(timestamp.isoformat(), weight, status) for timestamp, weight, status in day_rows],
                )
                connection.commit()

    def test_fixtures_reproduce_audited_evidence(self):
        self.assertTrue(FIXTURES.exists(), "audited scale fixtures are missing")
        for case in json.loads(FIXTURES.read_text(encoding="utf-8")):
            with self.subTest(case=case["label"]):
                with tempfile.TemporaryDirectory() as root:
                    service = Path(root)
                    scale_dir = service / "scale_data"
                    scale_dir.mkdir()
                    local_start = module._local_timestamp(case["started_at"])
                    local_end = module._local_timestamp(case["ended_at"])
                    self._write_days(
                        scale_dir, self._cycle_rows(case, local_start, local_end),
                    )

                    recovery = module.MissedSessionRecovery(service, case["host_id"])
                    evidence = recovery._scale_evidence(
                        {"started_at": case["started_at"], "ended_at": case["ended_at"]}
                    )

                    self.assertEqual(evidence["selected_weight"], case["selected_weight"])
                    self.assertEqual(evidence["raw_peak_weight"], case["raw_peak_weight"])
                    self.assertTrue(evidence["confirmed_empty_before"])
                    self.assertTrue(evidence["confirmed_empty_at_end"])


if __name__ == "__main__":
    unittest.main()