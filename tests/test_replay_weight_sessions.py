"""Replay harness tests against recorded-window shape fixtures."""

import sqlite3
import sys
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.modules.setdefault("serial", unittest.mock.Mock())
sys.modules.setdefault("cv2", unittest.mock.Mock())
sys.modules.setdefault("minio", unittest.mock.Mock())
sys.modules.setdefault("minio.error", unittest.mock.Mock())

from scripts.replay_weight_sessions import (  # noqa: E402
    match_targets,
    parse_can_hang,
    replay_db,
)

VN_TZ = timezone(timedelta(hours=7))
START = datetime(2026, 10, 24, 19, 0, 0, tzinfo=VN_TZ)
STEP = 0.21


def build_db(weights, day_start=START):
    """Write a weight_log sqlite with rows at live 0.21s spacing."""
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    with sqlite3.connect(handle.name) as conn:
        conn.execute(
            "CREATE TABLE weight_log (id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "timestamp TEXT, weight_kg REAL, sign TEXT, decimal_pos INTEGER, "
            "checksum_ok INTEGER, status TEXT DEFAULT 'UNSTABLE')"
        )
        for index, weight in enumerate(weights):
            timestamp = (day_start + timedelta(seconds=(index + 1) * STEP)).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )
            conn.execute(
                "INSERT INTO weight_log (timestamp, weight_kg, sign, decimal_pos, "
                "checksum_ok) VALUES (?, ?, '+', 0, 1)",
                (timestamp, weight),
            )
    return handle.name


class ReplayWindowTests(unittest.TestCase):
    def tearDown(self):
        for path in self._dbs:
            Path(path).unlink(missing_ok=True)

    def setUp(self):
        self._dbs = []

    def _db(self, weights):
        path = build_db(weights)
        self._dbs.append(path)
        return path

    def test_shallow_dip_then_descent_is_one_departed_visit(self):
        # HP-2 Oct24 19:35: ~610kg rocking dip (not a departure), then the
        # real descent to empty.
        weights = (
            [0, 0, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
            + [37495] * 10
            + [36890, 36910, 36910]
            + [38320, 38370, 38400, 38440, 38420]
            + [33000, 31000, 25000, 18000, 12000, 5000, 2000, 500]
            + [0] * 12
        )
        sessions = replay_db(self._db(weights))

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["end_reason"], "weight_drop_confirmed")
        self.assertGreater(sessions[0]["stable_weight"], 37000)
        self.assertEqual(sessions[0]["boundary_kind"], "departure")
        self.assertGreater(sessions[0]["boundary_reference_kg"], 37000)

    def test_fast_exit_confirms_departure_without_rise(self):
        # HP-2 Oct24 21:09: stable load, fast drop, no renewed rise.
        weights = (
            [0, 0, 3000, 6000, 9000, 12000, 15000]
            + [16960] * 10
            + [16530, 12070, 11250, 10810, 9400, 8510]
            + [-10] * 20
        )
        sessions = replay_db(self._db(weights))

        self.assertEqual(len(sessions), 1)
        self.assertEqual(sessions[0]["end_reason"], "weight_drop_confirmed")
        self.assertAlmostEqual(sessions[0]["stable_weight"], 16960, delta=20)

    def test_nonzero_valley_splits_into_two_visits(self):
        # HP-2 Oct24 07:48: load, partial exit to a 9.7t valley, second
        # vehicle climbs on. No zero crossing; a split is required.
        weights = (
            [0, 0, 5000, 10000, 16000, 22000, 28000, 34000, 40000]
            + [40610] * 10
            + [39900, 34900, 34900]
            + [9690, 9690]
            + [11320]
            + [28710, 30560, 45460]
            + [45900] * 12
            + [22000, 10000, 500]
            + [0] * 12
        )
        sessions = replay_db(self._db(weights))

        self.assertEqual(len(sessions), 2)
        first, second = sessions
        self.assertEqual(first["end_reason"], "weight_drop_confirmed")
        self.assertAlmostEqual(first["stable_weight"], 40610, delta=20)
        self.assertEqual(second["boundary_kind"], "rise")
        self.assertAlmostEqual(second["stable_weight"], 45900, delta=20)
        self.assertLess(second["started_at"], second["ended_at"])

    def test_brief_empty_dip_then_big_rise_splits_into_two_visits(self):
        # HP-2 Oct25 19:47: ~0.2s near-empty blip inside the transition; the
        # old 2s-empty rule cannot end the first visit here.
        weights = (
            [0, 0, 3000, 6000, 9000, 12000, 15000, 18000, 20000]
            + [21010] * 10
            + [20600, 10860, 9240, 30, 780]
            + [9900, 19810, 40540, 43050, 61130, 62110]
            + [62170] * 12
            + [30000, 12000, 500]
            + [0] * 12
        )
        sessions = replay_db(self._db(weights))

        self.assertEqual(len(sessions), 2)
        first, second = sessions
        self.assertEqual(first["end_reason"], "weight_drop_confirmed")
        self.assertAlmostEqual(first["stable_weight"], 21010, delta=20)
        self.assertEqual(second["boundary_kind"], "rise")
        self.assertAlmostEqual(second["stable_weight"], 62170, delta=20)


class MatchTests(unittest.TestCase):
    def test_one_to_one_match_respects_both_sides(self):
        weights = [40000, 40010, 17000]
        targets = [40005, 17005, 39990]

        self.assertEqual(match_targets(weights, targets, 50), 3)

    def test_duplicate_targets_match_only_once(self):
        weights = [40000]
        targets = [40000, 40000]

        self.assertEqual(match_targets(weights, targets, 50), 1)

    def test_out_of_tolerance_never_matches(self):
        self.assertEqual(match_targets([40000], [40200], 50), 0)


class CsvParseTests(unittest.TestCase):
    def _csv(self, tmpdir):
        lines = [
            "\ufeffCông ty,x,,,,,,,,,,,,,,,",
            "ADD,y,,,,,,,,,,,,,,,",
            "Tiêu đề,z,,,,,,,,,,,,,,,",
            "Ngày,24/10/2026,,,,,,,,,,,,,,,",
            "TT,Xe 54 Tổng,Xe 54 Bì,Xe 54 Hàng,Xe 90 Tổng,Xe 90 Bì,Xe 90 Hàng",
            "1,37.350,16.940,20.410,36.870,16.930,19.940",
            "2,,0,,37.000,16.850,20.150",
            "Tổng,73,,,73,,,",
        ]
        path = Path(tmpdir) / "can_hang.csv"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def test_pairs_require_both_gross_and_tare(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pairs = parse_can_hang(self._csv(tmpdir))

        self.assertEqual(pairs, [(37350.0, 16940.0), (36870.0, 16930.0), (37000.0, 16850.0)])


if __name__ == "__main__":
    unittest.main()
