import tempfile
import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("numpy", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from d2008_scale_reader import D2008Reader, WeightFrame
from services.session.session_manager import SessionManager


def make_frame(weight):
    return WeightFrame(b"", "+", weight, 0, True, f"{int(weight):06d}")


class WeightStabilityTests(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.NamedTemporaryFile(suffix=".db")
        self.reader = D2008Reader(db_file=self.db.name)

    def tearDown(self):
        self.reader._db.close()
        self.db.close()

    def statuses(self, weights):
        frames = [make_frame(weight) for weight in weights]
        for frame in frames:
            frame.status = self.reader._get_status(frame)
        return frames

    def test_twenty_kg_spread_is_stable_at_window_mode(self):
        frames = self.statuses([39120, 39120, 39130, 39130, 39120] * 2)

        self.assertEqual(frames[-1].status, "STABLE")
        self.assertEqual(frames[-1].stable_weight, 39120)

    def test_latest_reading_breaks_window_mode_tie(self):
        frames = self.statuses([39120, 39120, 39130, 39130, 39140] * 2)

        self.assertEqual(frames[-1].status, "STABLE")
        self.assertEqual(frames[-1].stable_weight, 39130)

    def test_more_than_twenty_kg_spread_is_unstable(self):
        frames = self.statuses([39120, 39120, 39130, 39130, 39150] * 2)

        self.assertEqual(frames[-1].status, "UNSTABLE")
        self.assertIsNone(frames[-1].stable_weight)

    def test_invalid_checksum_does_not_reach_callbacks(self):
        self.reader.on_frame = Mock()
        frame = make_frame(39120)
        frame.checksum_ok = False

        self.reader._handle_frame(frame)

        self.reader.on_frame.assert_not_called()
        self.assertEqual(list(self.reader._recent_weights), [])

    def test_four_exact_readings_do_not_reach_stability(self):
        frames = self.statuses([39120] * 4)

        self.assertTrue(all(frame.status == "UNSTABLE" for frame in frames))

    def test_five_exact_readings_use_fast_stability(self):
        frames = self.statuses([39120] * 5)

        self.assertEqual(frames[-1].status, "STABLE")
        self.assertEqual(frames[-1].stable_weight, 39120)
        self.assertEqual(frames[-1].stability_rule, "exact_5")

    def test_nine_nonexact_readings_do_not_reach_stability(self):
        frames = self.statuses([39120, 39130] * 4 + [39120])

        self.assertTrue(all(frame.status == "UNSTABLE" for frame in frames))

    def test_scale_database_retains_one_year(self):
        now = datetime(2026, 7, 14, 12, 0, 0)
        old = (now - timedelta(days=366)).isoformat()
        recent = (now - timedelta(days=364)).isoformat()
        with self.reader._db._lock:
            self.reader._db._conn.executemany(
                "INSERT INTO weight_log (timestamp, weight_kg, sign, decimal_pos, checksum_ok, status) "
                "VALUES (?, 100, '+', 0, 1, 'STABLE')",
                [(old,), (recent,)],
            )
            self.reader._db._conn.commit()
            self.reader._db._delete_expired_rows_locked(now.timestamp())
            timestamps = [row[0] for row in self.reader._db._conn.execute("SELECT timestamp FROM weight_log")]

        self.assertEqual(timestamps, [recent])

    def test_maintenance_failure_does_not_fail_weight_save(self):
        frame = make_frame(39120)
        with unittest.mock.patch.object(
            self.reader._db,
            "_maintain_locked",
            side_effect=OSError("maintenance failed"),
        ):
            self.reader._db.save(frame)

        self.assertEqual(self.reader._db.get_recent(1)[0]["weight_kg"], 39120)
        self.assertEqual(self.reader._db.last_maintenance_error, "maintenance failed")

    def test_overload_clears_stability_history(self):
        self.statuses([39120, 39120, 39120, 39120])
        overload = make_frame(999999)

        self.assertEqual(self.reader._get_status(overload), "OVERLOAD")
        self.assertEqual(list(self.reader._recent_weights), [])
        self.assertEqual(self.reader._same_weight_count, 0)

    def test_invalid_checksum_resets_exact_stability_count(self):
        self.statuses([39120] * 4)
        frame = make_frame(39120)
        frame.checksum_ok = False
        self.reader._handle_frame(frame)

        next_frame = self.statuses([39120])[0]
        self.assertEqual(next_frame.status, "UNSTABLE")
        self.assertEqual(next_frame.same_weight_count, 1)


class SessionWeightTests(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager(Mock())
        self.manager.session.session_active = True
        self.manager.session.stable_weight = 39120
        self.manager.session.latest_stable_weight = 39120
        self.manager.session.weight_departure_baseline = 39120
        self.manager.session.stable_weight_counts[39120] = 1
        self.manager.session.stable_weight_last_seen[39120] = 1
        self.manager.session.stable_weight_sequence = 1

    def stable_frame(self, stable_weight):
        frame = make_frame(stable_weight)
        frame.status = "STABLE"
        frame.stable_weight = stable_weight
        return frame

    def test_session_mode_uses_recency_to_break_ties(self):
        self.manager.on_frame(self.stable_frame(39130), Mock())
        self.assertEqual(self.manager.session.stable_weight, 39130)

        self.manager.on_frame(self.stable_frame(39120), Mock())
        self.assertEqual(self.manager.session.stable_weight, 39120)

    def test_weight_departure_ends_after_two_seconds(self):
        frame = make_frame(38500)
        frame.status = "UNSTABLE"
        self.manager._end_session = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.time.time",
            side_effect=[10.0, 11.9, 12.0],
        ):
            self.manager.on_frame(frame, Mock())
            self.manager.on_frame(frame, Mock())
            self.manager.on_frame(frame, Mock())

        self.manager._end_session.assert_called_once_with("weight_departure", unittest.mock.ANY)

    def test_stable_drop_does_not_move_departure_baseline(self):
        self.manager._end_session = Mock()
        dropped = self.stable_frame(38500)

        with unittest.mock.patch(
            "services.session.session_manager.time.time",
            side_effect=[10.0, 12.0],
        ):
            self.manager.on_frame(dropped, Mock())
            self.manager.on_frame(dropped, Mock())

        self.manager._end_session.assert_called_once_with("weight_departure", unittest.mock.ANY)

    def test_gradual_stable_drop_keeps_high_water_baseline(self):
        self.manager._end_session = Mock()
        with unittest.mock.patch(
            "services.session.session_manager.time.time",
            side_effect=[10.0, 12.0],
        ):
            self.manager.on_frame(self.stable_frame(38800), Mock())
            self.manager.on_frame(self.stable_frame(38600), Mock())
            self.manager.on_frame(self.stable_frame(38600), Mock())

        self.assertEqual(self.manager.session.weight_departure_baseline, 39120)
        self.manager._end_session.assert_called_once_with("weight_departure", unittest.mock.ANY)

    def test_rolling_mode_forgets_old_plateau(self):
        for _ in range(25):
            self.manager.on_frame(self.stable_frame(39120), Mock())
        for _ in range(25):
            self.manager.on_frame(self.stable_frame(38500), Mock())

        self.assertEqual(self.manager.session.stable_weight, 38500)
        self.assertEqual(len(self.manager.session.stable_weight_history), 25)

    def test_rearm_rejects_same_nonzero_plateau(self):
        self.manager.session.session_active = False
        self.manager.session.rearm_block_until = 11.0
        self.manager.session.rearm_reference_weight = 39120
        self.manager.session.stable_weight = 39120

        with unittest.mock.patch("services.session.session_manager.time.time", return_value=12.0):
            self.assertFalse(self.manager._can_start_session(Mock()))

        self.manager.session.stable_weight = 38500
        with unittest.mock.patch("services.session.session_manager.time.time", return_value=12.0):
            self.assertTrue(self.manager._can_start_session(Mock()))

    def test_recent_same_plate_skips_regardless_of_weight(self):
        self.manager._last_publish_plate = "15C-326.77"
        self.manager._last_publish_weight = 8500
        self.manager._last_publish_session_end = "2026-07-15T06:26:48+00:00"

        self.assertTrue(self.manager._should_skip_duplicate_publish(
            "15C-326.77", 47500, "2026-07-15T06:26:52+00:00"
        ))
        self.assertFalse(self.manager._should_skip_duplicate_publish(
            "16N-6554", 47500, "2026-07-15T06:26:52+00:00"
        ))

    def test_same_plate_at_ten_seconds_is_allowed(self):
        self.manager._last_publish_plate = "15C-326.77"
        self.manager._last_publish_weight = 8500
        self.manager._last_publish_session_end = "2026-07-15T06:26:48+00:00"

        self.assertFalse(self.manager._should_skip_duplicate_publish(
            "15C-326.77", 47500, "2026-07-15T06:26:58+00:00"
        ))

    def test_post_session_descent_does_not_start_chained_attempt(self):
        manager = SessionManager(Mock())
        manager._attempt_wait_reference = 10000
        manager._post_session_low = 10000

        for weight in (9700, 9000, 8000, 5000):
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            manager.on_frame(frame, Mock())

        self.assertIsNone(manager._attempt)
        self.assertEqual(manager._post_session_low, 5000)

    def test_post_session_rebound_starts_chained_attempt(self):
        manager = SessionManager(Mock())
        manager._attempt_wait_reference = 10000
        manager._post_session_low = 10000

        for weight in (8000, 8400):
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            manager.on_frame(frame, Mock())
        self.assertIsNone(manager._attempt)

        frame = make_frame(8500)
        frame.status = "UNSTABLE"
        manager.on_frame(frame, Mock())

        self.assertIsNotNone(manager._attempt)

    def test_direct_post_session_rise_starts_chained_attempt(self):
        manager = SessionManager(Mock())
        manager._attempt_wait_reference = 10000
        manager._post_session_low = 10000
        frame = make_frame(10500)
        frame.status = "UNSTABLE"

        manager.on_frame(frame, Mock())

        self.assertIsNotNone(manager._attempt)

    def test_empty_scale_clears_post_session_tail(self):
        manager = SessionManager(Mock())
        manager._attempt_wait_reference = 10000
        manager._post_session_low = 8000
        frame = make_frame(0)
        frame.status = "UNSTABLE"

        manager.on_frame(frame, Mock())

        self.assertIsNone(manager._attempt_wait_reference)
        self.assertIsNone(manager._post_session_low)

    def test_stable_frame_promotes_before_chained_wait_gate(self):
        manager = SessionManager(Mock())
        manager.session.rearm_block_until = 11.0
        manager.session.rearm_reference_weight = 10000
        manager.session.rearm_block_reason = "vehicle_left"
        manager._attempt_wait_reference = 10000
        frame = self.stable_frame(10600)
        frame.stability_rule = "exact_5"

        with unittest.mock.patch("services.session.session_manager.time.time", return_value=12.0):
            manager.on_frame(frame, Mock())

        self.assertTrue(manager.session.session_active)
        self.assertEqual(manager.session.stability_rule, "exact_5")

    def test_stable_same_plateau_remains_blocked_after_rearm_delay(self):
        manager = SessionManager(Mock())
        manager.session.rearm_block_until = 11.0
        manager.session.rearm_reference_weight = 10000
        manager._attempt_wait_reference = 10000
        frame = self.stable_frame(10000)
        frame.stability_rule = "exact_5"

        with unittest.mock.patch("services.session.session_manager.time.time", return_value=12.0):
            manager.on_frame(frame, Mock())

        self.assertFalse(manager.session.session_active)

    def test_stable_frame_waits_for_two_second_rearm_delay(self):
        manager = SessionManager(Mock())
        manager.session.rearm_block_until = 12.0
        manager.session.rearm_reference_weight = 10000
        manager._attempt_wait_reference = 10000
        frame = self.stable_frame(10600)
        frame.stability_rule = "exact_5"

        with unittest.mock.patch("services.session.session_manager.time.time", return_value=11.0):
            manager.on_frame(frame, Mock())

        self.assertFalse(manager.session.session_active)

    def test_stable_frame_promotes_before_attempt_wait_gate_without_rearm(self):
        self.manager.session.session_active = False
        self.manager._attempt_wait_reference = 10000
        frame = self.stable_frame(10300)
        frame.stability_rule = "exact_5"

        self.manager.on_frame(frame, Mock())

        self.assertTrue(self.manager.session.session_active)
        self.assertEqual(self.manager.session.stability_rule, "exact_5")


class AttemptArchiveTests(unittest.TestCase):
    def test_unstable_attempt_archives_maximum_weight_after_empty_dwell(self):
        manager = SessionManager(Mock(), lpr_grabbers={})
        manager._save_diagnostic_frames = Mock(return_value=0)
        unstable = make_frame(1000)
        unstable.status = "UNSTABLE"
        empty = make_frame(0)
        empty.status = "UNSTABLE"

        with unittest.mock.patch(
            "services.session.session_manager.time.time",
            side_effect=[10.0, 11.0, 12.0, 14.0],
        ):
            manager.on_frame(unstable, Mock())
            unstable.weight = 1600
            manager.on_frame(unstable, Mock())
            manager.on_frame(empty, Mock())
            manager.on_frame(empty, Mock())

        manager._save_diagnostic_frames.assert_called_once()
        metadata = manager._save_diagnostic_frames.call_args.args[3]
        self.assertEqual(metadata["maximum_weight_kg"], 1600)
        self.assertIsNone(manager._attempt)

    def test_stable_attempt_promotes_without_no_stable_archive(self):
        manager = SessionManager(Mock(), lpr_grabbers={})
        manager._archive_no_stable = Mock()
        frame = make_frame(1200)
        frame.status = "STABLE"
        frame.stable_weight = 1200

        manager.on_frame(frame, Mock())

        self.assertTrue(manager.session.session_active)
        manager._archive_no_stable.assert_not_called()

if __name__ == "__main__":
    unittest.main()
