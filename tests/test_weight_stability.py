import json
import tempfile
import unittest
import sys
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from d2008_scale_reader import D2008Reader, WeightFrame
from services.session.session_manager import SessionManager
from services.storage.image_save_worker import ImageSaveWorker


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

    def test_reader_reconnects_after_valid_frame_stall(self):
        class SilentSerial:
            is_open = True
            in_waiting = 0

            def read(self, _size):
                return b""

            def close(self):
                self.is_open = False

        health = []
        self.reader.stall_seconds = 0.01
        self.reader.reconnect_initial_seconds = 0.001
        self.reader.reconnect_max_seconds = 0.001
        def on_health(event, details):
            health.append((event, details))
            if event == "stalled":
                self.reader._running = False

        self.reader.on_health = on_health
        self.reader._running = True

        with unittest.mock.patch("d2008_scale_reader.serial.Serial", return_value=SilentSerial()):
            self.reader._run()

        self.assertEqual([event for event, _ in health], ["stalled"])
        self.assertEqual(self.reader.state, "stopped")

    def test_reader_reports_recovery_after_valid_frame(self):
        def valid_frame(weight=1000):
            body = b"+" + f"{weight:06d}".encode() + b"0"
            xor_value = 0
            for byte in body:
                xor_value ^= byte
            encode = lambda nibble: nibble + (0x30 if nibble <= 9 else 0x37)
            return b"\x02" + body + bytes((encode(xor_value >> 4), encode(xor_value & 0x0F))) + b"\x03"

        class SilentSerial:
            is_open = True
            in_waiting = 0

            def read(self, _size):
                return b""

            def close(self):
                self.is_open = False

        class ValidSerial(SilentSerial):
            def __init__(self):
                self.sent = False

            def read(self, _size):
                if not self.sent:
                    self.sent = True
                    return valid_frame()
                return b""

        health = []
        self.reader.stall_seconds = 0.01
        self.reader.reconnect_initial_seconds = 0.001
        self.reader.reconnect_max_seconds = 0.001

        def on_health(event, details):
            health.append((event, details))
            if event == "recovered":
                self.reader._running = False

        self.reader.on_health = on_health
        self.reader._running = True
        with unittest.mock.patch(
            "d2008_scale_reader.serial.Serial", side_effect=[SilentSerial(), ValidSerial()],
        ):
            self.reader._run()

        self.assertEqual([event for event, _ in health], ["stalled", "recovered"])
        self.assertIsNotNone(self.reader.latest)
        self.assertEqual(self.reader.latest.weight, 1000)
        self.assertIsNotNone(health[-1][1]["last_valid_timestamp"])

    def test_stop_signals_reconnect_backoff(self):
        self.reader._running = True
        self.reader.state = "reconnecting"
        self.reader._stop_event.clear()
        self.reader._thread = unittest.mock.Mock()
        self.reader._thread.is_alive.return_value = False

        with unittest.mock.patch.object(self.reader._stop_event, "wait") as wait:
            self.reader.stop()

        self.assertTrue(self.reader._stop_event.is_set())
        self.reader._thread.join.assert_called_once_with(timeout=3)
        wait.assert_not_called()

    def test_db_persists_at_five_hz_while_display_remains_one_hz(self):
        self.reader.log_interval = 0.2
        self.reader.on_frame = Mock()
        self.reader.on_weight = Mock()
        self.reader._db.save = Mock()
        frames = [make_frame(weight) for weight in (100, 200, 300, 400, 500, 600)]

        with unittest.mock.patch(
            "d2008_scale_reader.time.time",
            side_effect=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        ):
            for frame in frames:
                self.reader._handle_frame(frame)

        self.assertEqual(self.reader.on_frame.call_count, 6)
        self.assertEqual(self.reader._db.save.call_count, 5)
        self.assertEqual(self.reader.on_weight.call_count, 1)

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

    def test_scale_database_rotates_to_frame_date(self):
        with tempfile.TemporaryDirectory() as directory:
            reader = D2008Reader(db_file=directory)
            try:
                first = make_frame(100)
                first.timestamp = datetime(2026, 7, 14, 23, 59, 59)
                second = make_frame(200)
                second.timestamp = datetime(2026, 7, 15, 0, 0, 1)

                reader._db.save(first)
                reader._db.save(second)

                for date_text, expected_weight in (("2026-07-14", 100), ("2026-07-15", 200)):
                    path = os.path.join(directory, f"{date_text}.db")
                    self.assertTrue(os.path.exists(path))
                    with closing(sqlite3.connect(path)) as conn:
                        self.assertEqual(conn.execute(
                            "SELECT weight_kg FROM weight_log"
                        ).fetchone()[0], expected_weight)
            finally:
                reader._db.close()

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

    def test_frame_callback_error_does_not_escape_handle_frame(self):
        # A raising session callback must not propagate out of _handle_frame:
        # an escaped exception kills the reader thread permanently.
        self.reader.on_frame = Mock(side_effect=RuntimeError("session callback boom"))

        frame = make_frame(39120)
        frame.status = self.reader._get_status(frame)

        self.reader._handle_frame(frame)  # must not raise

        self.reader.on_frame.assert_called_once_with(frame)
        self.assertEqual(self.reader.latest, frame)

    def test_db_save_error_does_not_escape_handle_frame(self):
        self.reader._db.save = Mock(side_effect=RuntimeError("db locked"))
        self.reader.log_interval = 0

        frame = make_frame(39120)
        frame.status = self.reader._get_status(frame)

        self.reader._handle_frame(frame)  # must not raise


class SessionWeightTests(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager(Mock())
        self.manager.session.session_active = True
        self.manager.session.stable_weight = 39120
        self.manager.session.latest_stable_weight = 39120
        self.manager.session.weight_trend_window.extend([39120] * 15)
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

    def test_falling_trend_does_not_end_loaded_session(self):
        end_session = Mock()
        self.manager._end_session = end_session
        self.manager.session.weight_trend_window.clear()

        for weight in range(39120, 37620, -100):
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            self.manager.on_frame(frame, Mock())

        end_session.assert_not_called()
        self.assertTrue(self.manager.session.session_active)
        self.assertTrue(self.manager.session.scale_owned)

    def test_rising_trend_does_not_split_active_session(self):
        self.manager._end_session = Mock()
        self.manager.session.weight_trend_window.clear()

        for weight in range(39120, 40620, 100):
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            self.manager.on_frame(frame, Mock())

        self.manager._end_session.assert_not_called()
        self.assertTrue(self.manager.session.session_active)

    def test_rocking_does_not_confirm_weight_trend(self):
        self.manager._end_session = Mock()
        self.manager.session.weight_trend_window.clear()

        for weight in (39120, 38500, 39200, 38400, 39300) * 3:
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            self.manager.on_frame(frame, Mock())

        self.manager._end_session.assert_not_called()

    def test_vehicle_disappearance_does_not_end_session(self):
        self.manager.vehicle_tracker = Mock()
        self.manager.vehicle_tracker.get_summary.return_value = {
            "vehicle_type": "truck",
            "cam1_truck_stable": False,
            "cam3_truck_stable": False,
            "cam1_truck_unstable": True,
            "cam3_truck_unstable": True,
        }
        self.manager._end_session = Mock()

        self.manager.on_frame(make_frame(39120), Mock())

        self.manager._end_session.assert_not_called()
        self.assertEqual(self.manager.session.vehicle_type, "truck")

    def test_rolling_mode_forgets_old_plateau(self):
        for _ in range(25):
            self.manager.on_frame(self.stable_frame(39120), Mock())
        for _ in range(25):
            self.manager.on_frame(self.stable_frame(38500), Mock())

        self.assertEqual(self.manager.session.stable_weight, 38500)
        self.assertEqual(len(self.manager.session.stable_weight_history), 25)

    def test_empty_dwell_does_not_replace_loaded_stable_weight(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        started = datetime(2026, 9, 28, 9, 8, 52, tzinfo=timezone.utc)

        for index in range(25):
            frame = self.stable_frame(30630)
            frame.timestamp = started + timedelta(seconds=index * 0.2)
            manager._handle_stable_frame(frame, Mock())
        loaded_observed_at = manager.session.stable_weight_observed_at

        for index in range(25):
            frame = self.stable_frame(20)
            frame.timestamp = started + timedelta(seconds=5 + index * 0.2)
            manager._handle_stable_frame(frame, Mock())

        metadata = manager._snapshot_session("scale_empty")
        self.assertEqual(manager.session.latest_stable_weight, 20)
        self.assertEqual(manager.session.stable_weight, 30630)
        self.assertEqual(len(manager.session.stable_weight_history), 25)
        self.assertEqual(metadata["stable_weight"], 30630)
        self.assertEqual(metadata["weight_source"], "stable")
        self.assertEqual(metadata["weight_observed_at"], loaded_observed_at)

    def test_stable_empty_status_change_does_not_replace_active_weight(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        manager.session.stable_weight = 105960
        frame = self.stable_frame(50)

        manager.on_status_change(frame, "UNSTABLE", "STABLE", Mock())

        self.assertEqual(manager.session.stable_weight, 105960)

    def test_on_status_change_write_takes_the_lifecycle_lock(self):
        entries = []
        real_lock = self.manager._lifecycle_lock

        class CountingRLock:
            def __enter__(inner):
                entries.append(1)
                real_lock.acquire()
                return inner

            def __exit__(inner, *exc):
                real_lock.release()
                return False

        self.manager.session.session_active = False
        self.manager._lifecycle_lock = CountingRLock()
        frame = self.stable_frame(50)

        self.manager.on_status_change(frame, "UNSTABLE", "STABLE", Mock())

        self.assertEqual(self.manager.session.stable_weight, 50)
        self.assertGreaterEqual(len(entries), 1)

    def test_on_weight_snapshots_stable_weight_once(self):
        class FlakySession:
            session_active = True
            stable_count = 0

            def __init__(self):
                self.calls = 0

            @property
            def stable_weight(self):
                # A concurrent session reset could clear the value between two
                # reads; the callback must only read it once.
                self.calls += 1
                return 12.5 if self.calls == 1 else None

        self.manager.vehicle_tracker = None
        self.manager.plate_tracker.get_confirmed_plate = Mock(return_value=(None, 0.0, None))
        self.manager.session = FlakySession()
        frame = make_frame(39120)
        log_fn = Mock()

        self.manager.on_weight(frame, log_fn)  # must not raise

        rendered = " ".join(str(call.args) for call in log_fn.call_args_list)
        self.assertIn("stable_wt=12", rendered)
        self.assertEqual(self.manager.session.calls, 1)

    def test_snapshot_rejects_sub_threshold_stable_candidate(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        manager.session.stable_weight = 50
        manager.session.stable_weight_observed_at = "2026-09-28T10:09:35+00:00"
        manager._session_raw_peak = 106040
        manager._session_raw_peak_observed_at = "2026-09-28T10:07:03+00:00"
        manager._session_filtered_peak = 106040
        manager._session_filtered_peak_observed_at = "2026-09-28T10:08:47+00:00"

        metadata = manager._snapshot_session("scale_empty")

        self.assertEqual(metadata["stable_weight"], 106040)
        self.assertEqual(metadata["weight_source"], "filtered_peak")
        self.assertEqual(metadata["weight_observed_at"], "2026-09-28T10:08:47+00:00")

    def test_stable_empty_frame_with_no_loaded_candidate_does_not_crash(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        self.assertIsNone(manager.session.stable_weight)

        manager._handle_stable_frame(self.stable_frame(20), Mock())

        self.assertIsNone(manager.session.stable_weight)
        metadata = manager._snapshot_session("scale_empty")
        self.assertEqual(metadata["weight_source"], "none")

    def test_eviction_mode_change_keeps_selected_weight_observation_time(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        started = datetime(2026, 7, 24, tzinfo=timezone.utc)
        frames = []
        for index, weight in enumerate([1000] * 13 + [2000] * 12 + [3000]):
            frame = self.stable_frame(weight)
            frame.timestamp = started + timedelta(seconds=index)
            frames.append(frame)
            manager._handle_stable_frame(frame, Mock())

        self.assertEqual(manager.session.stable_weight, 2000)
        self.assertEqual(
            manager.session.stable_weight_observed_at,
            frames[-2].timestamp.isoformat(timespec="milliseconds"),
        )

    def test_local_peak_confirms_after_dwell_and_ignores_later_rise(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        started = datetime(2026, 7, 24, tzinfo=timezone.utc)
        weights = [1000, 20000, 20000, 20000, 20000, 20000, 20000, 45000]

        for index, weight in enumerate(weights):
            frame = self.stable_frame(weight)
            frame.timestamp = started + timedelta(seconds=index * 0.5)
            manager.on_frame(frame, Mock())

        self.assertTrue(manager._session_local_peak_confirmed)
        self.assertEqual(manager._session_local_peak, 20000)
        self.assertEqual(
            manager._session_local_peak_observed_at,
            (started + timedelta(seconds=0.5)).isoformat(timespec="milliseconds"),
        )

    def test_local_peak_confirms_on_weight_drop(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        started = datetime(2026, 7, 24, tzinfo=timezone.utc)

        for index, weight in enumerate([30000, 29000]):
            frame = self.stable_frame(weight)
            frame.timestamp = started + timedelta(seconds=index * 0.2)
            manager.on_frame(frame, Mock())

        self.assertTrue(manager._session_local_peak_confirmed)
        self.assertEqual(manager._session_local_peak, 30000)
        self.assertEqual(
            manager._session_local_peak_observed_at,
            started.isoformat(timespec="milliseconds"),
        )

    def test_unknown_lpr_photos_do_not_fall_back_after_session_start(self):
        metadata = {
            "session_id": "local-peak",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:20+00:00",
            "weight_observed_at": "2026-07-21T00:00:18+00:00",
            "local_peak_observed_at": "2026-07-21T00:00:05+00:00",
            "weight_source": "stable",
            "session_dir": "/spool/session",
            "unknown_snapshot_paths": {},
            "session_files": [
                "cam1-000001-sample.jpg", "cam1-000002-sample.jpg",
                "cam2-000001-sample.jpg",
            ],
            "capture_interval_seconds": 0.2,
        }
        frame_metadata = {
            "cam1-000001-sample.jpg": {"captured_at": "2026-07-21T00:00:01.800+00:00", "frame_id": 10},
            "cam1-000002-sample.jpg": {"captured_at": "2026-07-21T00:00:02.100+00:00", "frame_id": 11},
            "cam2-000001-sample.jpg": {"captured_at": "2026-07-21T00:00:17.900+00:00", "frame_id": 20},
        }
        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", return_value="later-frame", create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, Mock(),
            )

        self.assertEqual(selected, {})
        self.assertEqual(captured_at, {})

    def test_unknown_photos_use_only_available_session_start_frames(self):
        metadata = {
            "session_id": "weighted",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:09+00:00",
            "session_dir": "/spool/session",
            "session_files": ["cam1-000000-start.jpg"],
            "unknown_snapshot_paths": {"cam1": "/spool/cam1-2s.jpg", "cam2": "/spool/cam2-2s.jpg"},
            "unknown_snapshot_captured_at": {
                "cam1": "2026-07-21T00:00:02.030+00:00",
                "cam2": "2026-07-21T00:00:02.040+00:00",
            },
            "unknown_weight_snapshot_paths": {"cam1": "/spool/cam1-weight.jpg"},
            "unknown_weight_snapshot_captured_at": {"cam1": "2026-07-21T00:00:01.030+00:00"},
        }

        frame_metadata = {
            "cam1-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00.010+00:00", "frame_id": 1,
            },
        }
        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: {
                "/spool/session/cam1-000000-start.jpg": "cam1-start",
                "/spool/cam1-2s.jpg": "cam1-2s",
                "/spool/cam2-2s.jpg": "cam2-2s",
            }.get(path),
            create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, Mock(),
            )

        self.assertEqual(selected, {"cam1": "cam1-start"})
        self.assertEqual(captured_at, {"cam1": "2026-07-21T00:00:00.010+00:00"})

    def test_unknown_photos_choose_session_start_over_synchronized_weight_set(self):
        metadata = {
            "session_id": "synchronized-weight",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:09+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000000-start.jpg",
                "cam2-000000-start.jpg",
                "cam3-000000-start.jpg",
                "cam1-000005-sample.jpg",
                "cam2-000005-sample.jpg",
                "cam3-000005-sample.jpg",
            ],
            "unknown_snapshot_paths": {},
            "unknown_weight_snapshot_triggered_at": "2026-07-21T00:00:01+00:00",
            "unknown_weight_snapshot_paths": {
                "cam1": "/spool/cam1-weight.jpg",
                "cam2": "/spool/cam2-weight.jpg",
                "cam3": "/spool/cam3-weight.jpg",
            },
            "unknown_weight_snapshot_captured_at": {
                "cam1": "2026-07-21T00:00:01.100+00:00",
                "cam2": "2026-07-21T00:00:02.500+00:00",
                "cam3": "2026-07-21T00:00:01.120+00:00",
            },
        }
        frame_metadata = {
            "cam1-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00.010+00:00", "frame_id": 1,
            },
            "cam2-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00.030+00:00", "frame_id": 2,
            },
            "cam3-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00.020+00:00", "frame_id": 3,
            },
            "cam1-000005-sample.jpg": {
                "captured_at": "2026-07-21T00:00:01.090+00:00", "frame_id": 10,
            },
            "cam2-000005-sample.jpg": {
                "captured_at": "2026-07-21T00:00:01.130+00:00", "frame_id": 20,
            },
            "cam3-000005-sample.jpg": {
                "captured_at": "2026-07-21T00:00:01.110+00:00", "frame_id": 30,
            },
        }
        frames = {
            "/spool/session/cam1-000000-start.jpg": "cam1",
            "/spool/session/cam2-000000-start.jpg": "cam2",
            "/spool/session/cam3-000000-start.jpg": "cam3",
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", side_effect=frames.get, create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn,
            )

        self.assertEqual(selected, {"cam1": "cam1", "cam2": "cam2", "cam3": "cam3"})
        self.assertEqual(captured_at["cam2"], "2026-07-21T00:00:00.030+00:00")
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["lpr_target_source"], "session_start")
        self.assertEqual(metric["synchronized_gap_ms"], 20)
        self.assertTrue(all(
            item["source"] == "session_start" for item in metric["selected"].values()
        ))

    def test_unknown_ocr_photos_anchor_to_recorded_weight(self):
        metadata = {
            "session_id": "unknown-ocr",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:05+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000025-sample.jpg",
                "cam2-000025-sample.jpg",
                "cam3-000025-sample.jpg",
            ],
            "lpr_diagnostics": {
                "detected_regions": 1,
                "evidence": {
                    "cam1": {"plate_detected": "cam1-000025-sample.jpg"},
                },
            },
        }
        frame_metadata = {
            "cam1-000025-sample.jpg": {
                "captured_at": "2026-07-21T00:00:04.980+00:00", "frame_id": 25,
                "tracks": [{"bbox": [1, 2, 3, 4]}],
            },
            "cam2-000025-sample.jpg": {
                "captured_at": "2026-07-21T00:00:05.020+00:00", "frame_id": 25,
            },
            "cam3-000025-sample.jpg": {
                "captured_at": "2026-07-21T00:00:05.010+00:00", "frame_id": 25,
            },
        }
        frames = {
            "/spool/session/cam1-000025-sample.jpg": "cam1-weight",
            "/spool/session/cam2-000025-sample.jpg": "cam2-weight",
            "/spool/session/cam3-000025-sample.jpg": "cam3-weight",
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", side_effect=frames.get, create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_OCR",
            )

        self.assertEqual(selected, {
            "cam1": "cam1-weight", "cam2": "cam2-weight", "cam3": "cam3-weight",
        })
        self.assertEqual(captured_at["cam1"], "2026-07-21T00:00:04.980+00:00")
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["unknown_type"], "UNKNOWN_OCR")
        self.assertEqual(metric["lpr_target_source"], "weight_recorded")
        self.assertEqual(metric["target_at"], "2026-07-21T00:00:05.000+00:00")
        self.assertEqual(metric["synchronized_gap_ms"], 40)

    def test_unknown_ocr_photos_prefer_sustained_local_peak(self):
        metadata = {
            "session_id": "unknown-ocr-local-peak",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:20+00:00",
            "weight_observed_at": "2026-07-21T00:00:15+00:00",
            "local_peak_observed_at": "2026-07-21T00:00:05+00:00",
            "filtered_peak_observed_at": "2026-07-21T00:00:05.100+00:00",
            "raw_peak_observed_at": "2026-07-21T00:00:05.050+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000025-sample.jpg",
                "cam1-000075-sample.jpg",
            ],
        }
        frame_metadata = {
            "cam1-000025-sample.jpg": {
                "captured_at": "2026-07-21T00:00:05.020+00:00", "frame_id": 25,
            },
            "cam1-000075-sample.jpg": {
                "captured_at": "2026-07-21T00:00:14.980+00:00", "frame_id": 75,
            },
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: "peak" if "000025" in path else "late",
            create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_OCR",
            )

        self.assertEqual(selected, {"cam1": "peak"})
        self.assertEqual(captured_at, {"cam1": "2026-07-21T00:00:05.020+00:00"})
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["lpr_target_source"], "local_peak")
        self.assertEqual(metric["target_at"], "2026-07-21T00:00:05.000+00:00")

    def test_unknown_detection_photos_use_recorded_weight(self):
        metadata = {
            "session_id": "unknown-detection",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:07+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000013-sample.jpg",
                "cam2-000013-sample.jpg",
                "cam3-000013-sample.jpg",
            ],
            "lpr_diagnostics": {"detector_successes": 3, "detected_regions": 0},
        }
        frame_metadata = {
            "cam1-000013-sample.jpg": {
                "captured_at": "2026-07-21T00:00:06.990+00:00", "frame_id": 13,
            },
            "cam2-000013-sample.jpg": {
                "captured_at": "2026-07-21T00:00:07.020+00:00", "frame_id": 13,
            },
            "cam3-000013-sample.jpg": {
                "captured_at": "2026-07-21T00:00:07.010+00:00", "frame_id": 13,
            },
        }
        frames = {
            "/spool/session/cam1-000013-sample.jpg": "cam1-weight",
            "/spool/session/cam2-000013-sample.jpg": "cam2-weight",
            "/spool/session/cam3-000013-sample.jpg": "cam3-weight",
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", side_effect=frames.get, create=True,
        ):
            selected, _captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_DETECTION",
            )

        self.assertEqual(selected, {
            "cam1": "cam1-weight", "cam2": "cam2-weight", "cam3": "cam3-weight",
        })
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["unknown_type"], "UNKNOWN_DETECTION")
        self.assertEqual(metric["lpr_target_source"], "weight_recorded")
        self.assertEqual(metric["target_at"], "2026-07-21T00:00:07.000+00:00")
        self.assertEqual(metric["synchronized_gap_ms"], 30)

    def test_unknown_detection_photos_prefer_sustained_local_peak(self):
        metadata = {
            "session_id": "unknown-detection-local-peak",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:20+00:00",
            "weight_observed_at": "2026-07-21T00:00:15+00:00",
            "local_peak_observed_at": "2026-07-21T00:00:05+00:00",
            "filtered_peak_observed_at": "2026-07-21T00:00:05.100+00:00",
            "raw_peak_observed_at": "2026-07-21T00:00:05.050+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000025-sample.jpg",
                "cam1-000075-sample.jpg",
            ],
            "lpr_diagnostics": {"detector_successes": 2, "detected_regions": 0},
        }
        frame_metadata = {
            "cam1-000025-sample.jpg": {
                "captured_at": "2026-07-21T00:00:05.020+00:00", "frame_id": 25,
            },
            "cam1-000075-sample.jpg": {
                "captured_at": "2026-07-21T00:00:14.980+00:00", "frame_id": 75,
            },
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: "peak" if "000025" in path else "late",
            create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_DETECTION",
            )

        self.assertEqual(selected, {"cam1": "peak"})
        self.assertEqual(captured_at, {"cam1": "2026-07-21T00:00:05.020+00:00"})
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["lpr_target_source"], "local_peak")
        self.assertEqual(metric["target_at"], "2026-07-21T00:00:05.000+00:00")

    def test_unknown_detection_drops_camera_outside_target_window(self):
        metadata = {
            "session_id": "unknown-detection-stale-camera",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:40+00:00",
            "weight_observed_at": "2026-07-21T00:00:02+00:00",
            "session_dir": "/spool/session",
            "session_files": ["cam1-000035-sample.jpg", "cam2-000001-sample.jpg"],
            "lpr_diagnostics": {"detector_successes": 2, "detected_regions": 0},
        }
        frame_metadata = {
            "cam1-000035-sample.jpg": {
                "captured_at": "2026-07-21T00:00:02.050+00:00", "frame_id": 35,
            },
            "cam2-000001-sample.jpg": {
                "captured_at": "2026-07-21T00:00:37.100+00:00", "frame_id": 1,
            },
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: "cam1" if "cam1" in path else "cam2",
            create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_DETECTION",
            )

        self.assertEqual(selected, {"cam1": "cam1"})
        self.assertEqual(captured_at, {"cam1": "2026-07-21T00:00:02.050+00:00"})
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["missing_cameras"], ["cam2", "cam3"])
        self.assertEqual(metric["rejected"]["cam2"], {
            "source": "weight_recorded",
            "reason": "outside_target_window",
            "captured_at": "2026-07-21T00:00:37.100+00:00",
            "offset_ms": 35100,
        })

    def test_unknown_detection_drops_camera_outside_synchronized_group(self):
        metadata = {
            "session_id": "unknown-detection-skewed-camera",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:02+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000031-sample.jpg", "cam2-000039-sample.jpg",
                "cam3-000040-sample.jpg",
            ],
            "lpr_diagnostics": {"detector_successes": 3, "detected_regions": 0},
        }
        frame_metadata = {
            "cam1-000031-sample.jpg": {
                "captured_at": "2026-07-21T00:00:01.100+00:00", "frame_id": 31,
            },
            "cam2-000039-sample.jpg": {
                "captured_at": "2026-07-21T00:00:02.800+00:00", "frame_id": 39,
            },
            "cam3-000040-sample.jpg": {
                "captured_at": "2026-07-21T00:00:02.900+00:00", "frame_id": 40,
            },
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: path.split("/")[-1].split("-", 1)[0],
            create=True,
        ):
            selected, _captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_DETECTION",
            )

        self.assertEqual(selected, {"cam2": "cam2", "cam3": "cam3"})
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["synchronized_gap_ms"], 100)
        self.assertEqual(metric["rejected"]["cam1"]["reason"], "inter_camera_skew")

    def test_unknown_photos_fall_back_to_peak_timestamps(self):
        metadata = {
            "session_id": "unknown-peaks",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "session_dir": "/spool/session",
            "session_files": ["cam1-000025-sample.jpg", "cam1-000035-sample.jpg"],
            "filtered_peak_observed_at": "2026-07-21T00:00:05+00:00",
            "raw_peak_observed_at": "2026-07-21T00:00:07+00:00",
        }
        frame_metadata = {
            "cam1-000025-sample.jpg": {
                "captured_at": "2026-07-21T00:00:05.000+00:00", "frame_id": 25,
            },
            "cam1-000035-sample.jpg": {
                "captured_at": "2026-07-21T00:00:07.000+00:00", "frame_id": 35,
            },
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: "filtered" if "000025" in path else "raw",
            create=True,
        ):
            selected, _captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_OCR",
            )

        self.assertEqual(selected, {"cam1": "filtered"})
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["target_at"], "2026-07-21T00:00:05.000+00:00")

        metadata.pop("filtered_peak_observed_at")
        log_fn.reset_mock()
        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: "filtered" if "000025" in path else "raw",
            create=True,
        ):
            selected, _captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, unknown_plate="UNKNOWN_DETECTION",
            )

        self.assertEqual(selected, {"cam1": "raw"})
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["target_at"], "2026-07-21T00:00:07.000+00:00")

    def test_unknown_photos_ignore_late_weight_snapshot_without_start_frame(self):
        metadata = {
            "session_id": "late-weight",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:09+00:00",
            "session_dir": "/spool/session",
            "session_files": [],
            "unknown_snapshot_paths": {"cam1": "/spool/cam1-2s.jpg"},
            "unknown_snapshot_captured_at": {"cam1": "2026-07-21T00:00:02.030+00:00"},
            "unknown_weight_snapshot_paths": {"cam1": "/spool/cam1-weight.jpg"},
            "unknown_weight_snapshot_captured_at": {"cam1": "2026-07-21T00:00:05.030+00:00"},
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", return_value="later-frame", create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, {}, log_fn,
            )

        self.assertEqual(selected, {})
        self.assertEqual(captured_at, {})
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["lpr_target_source"], "session_start")
        self.assertEqual(metric["rejected"]["cam1"]["reason"], "late_or_invalid_timestamp")

    def test_unknown_two_second_snapshots_include_rear_and_are_captured_once(self):
        cam1 = Mock()
        cam1.peek_latest_frame_snapshot = None
        cam1.peek_latest_frame.return_value = "cam1-frame"
        cam2 = Mock()
        cam2.peek_latest_frame_snapshot = None
        cam2.peek_latest_frame.return_value = "cam2-frame"
        spool = Mock()
        spool.save_session_frame.side_effect = [
            "/spool/cam1-unknown-2s.jpg", "/spool/cam2-unknown-2s.jpg",
        ]
        manager = SessionManager(
            Mock(), rear_grabber=cam2, lpr_grabbers={"cam1": cam1}, frame_spool=spool,
        )
        manager.session.session_active = True
        manager.session.spool_active = True
        manager.session.session_id = "session-1"
        manager.session.unknown_snapshot_deadline = 12.0

        with unittest.mock.patch(
            "services.session.unknown_capture.time.time", side_effect=[11.9, 12.0, 13.0],
        ), unittest.mock.patch(
            "services.session.unknown_capture.datetime"
        ) as datetime_mock:
            datetime_mock.now.return_value.isoformat.return_value = "2026-07-21T00:00:02.000+00:00"
            datetime_mock.fromtimestamp.return_value.isoformat.return_value = "2026-07-21T00:00:02.000+00:00"
            self.assertFalse(manager._capture_unknown_snapshots_if_due(Mock()))
            self.assertTrue(manager._capture_unknown_snapshots_if_due(Mock()))
            self.assertFalse(manager._capture_unknown_snapshots_if_due(Mock()))

        cam1.peek_latest_frame.assert_called_once_with(copy_frame=True)
        cam2.peek_latest_frame.assert_called_once_with(copy_frame=True)
        self.assertEqual(spool.save_session_frame.call_args_list, [
            unittest.mock.call("session-1", "cam1-unknown-2s.jpg", "cam1-frame"),
            unittest.mock.call("session-1", "cam2-unknown-2s.jpg", "cam2-frame"),
        ])

    def test_unknown_weight_snapshots_capture_once_after_ten_tons(self):
        cam1 = Mock()
        cam1.peek_latest_frame_snapshot.return_value = (
            "cam1-frame", 42, "2026-07-21T00:00:01.100+00:00",
        )
        spool = Mock()
        spool.save_session_frame.return_value = "/spool/cam1-unknown-weight-10000.jpg"
        manager = SessionManager(Mock(), lpr_grabbers={"cam1": cam1}, frame_spool=spool)
        manager.session.session_active = True
        manager.session.spool_active = True
        manager.session.session_id = "session-1"
        started_at = datetime(2026, 7, 21, tzinfo=timezone.utc).timestamp()
        manager.session.started_at = started_at

        with unittest.mock.patch(
            "services.session.unknown_capture.time.time", return_value=started_at + 1.0,
        ), unittest.mock.patch(
            "services.session.unknown_capture.datetime", wraps=datetime,
        ) as datetime_mock:
            datetime_mock.now.return_value = datetime(
                2026, 7, 21, 0, 0, 1, tzinfo=timezone.utc,
            )
            self.assertFalse(manager._capture_unknown_weight_snapshots_if_due(9990, Mock()))
            self.assertFalse(manager._capture_unknown_weight_snapshots_if_due(10000, Mock()))
            self.assertTrue(manager._capture_unknown_weight_snapshots_if_due(10000, Mock()))
        self.assertFalse(manager._capture_unknown_weight_snapshots_if_due(20000, Mock()))

        cam1.peek_latest_frame_snapshot.assert_called_once_with(copy_frame=True)
        spool.save_session_frame.assert_called_once_with(
            "session-1", "cam1-unknown-weight-10000.jpg", "cam1-frame",
        )

    def test_unknown_weight_snapshot_stops_retrying_after_deadline(self):
        cam1 = Mock()
        cam1.peek_latest_frame_snapshot.return_value = (
            "stale-frame", 41, "2026-07-21T00:00:00.900+00:00",
        )
        spool = Mock()
        manager = SessionManager(Mock(), lpr_grabbers={"cam1": cam1}, frame_spool=spool)
        manager.session.session_active = True
        manager.session.spool_active = True
        manager.session.session_id = "session-1"
        started_at = datetime(2026, 7, 21, tzinfo=timezone.utc).timestamp()
        manager.session.started_at = started_at

        with unittest.mock.patch(
            "services.session.unknown_capture.time.time",
            side_effect=[started_at + 1.0, started_at + 1.1, started_at + 3.1],
        ), unittest.mock.patch(
            "services.session.unknown_capture.datetime", wraps=datetime,
        ) as datetime_mock:
            datetime_mock.now.return_value = datetime(
                2026, 7, 21, 0, 0, 1, tzinfo=timezone.utc,
            )
            self.assertFalse(manager._capture_unknown_weight_snapshots_if_due(10000, Mock()))
            self.assertFalse(manager._capture_unknown_weight_snapshots_if_due(10000, Mock()))
            self.assertFalse(manager._capture_unknown_weight_snapshots_if_due(10000, Mock()))

        self.assertTrue(manager.session.unknown_weight_snapshot_attempted)
        cam1.peek_latest_frame_snapshot.assert_called_once_with(copy_frame=True)
        spool.save_session_frame.assert_not_called()

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

    def test_recent_same_plate_is_not_a_transaction_identity(self):
        self.manager._last_publish_plate = "15C-326.77"
        self.manager._last_publish_weight = 8500
        self.manager._last_publish_session_end = "2026-07-15T06:26:48+00:00"

        self.assertFalse(hasattr(self.manager, "_should_skip_duplicate_publish"))

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

    def test_post_session_rebound_remains_blocked_until_empty(self):
        manager = SessionManager(Mock())
        manager._waiting_for_empty = True

        for weight in (8000, 8400, 8500, 12000):
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            manager.on_frame(frame, Mock())

        self.assertIsNone(manager._attempt)
        self.assertTrue(manager._waiting_for_empty)

    def test_stable_post_session_plateau_remains_blocked_until_empty(self):
        manager = SessionManager(Mock())
        manager._waiting_for_empty = True
        frame = self.stable_frame(10500)
        frame.stability_rule = "exact_5"

        manager.on_frame(frame, Mock())

        self.assertFalse(manager.session.session_active)
        self.assertIsNone(manager._attempt)

    def test_empty_dwell_rearms_scale_cycle(self):
        manager = SessionManager(Mock())
        manager._waiting_for_empty = True
        frame = make_frame(0)
        frame.status = "UNSTABLE"

        with unittest.mock.patch(
            "services.session.session_manager.time.monotonic",
            side_effect=[10.0, 11.9, 12.0],
        ):
            manager.on_frame(frame, Mock())
            manager.on_frame(frame, Mock())
            manager.on_frame(frame, Mock())

        self.assertFalse(manager._waiting_for_empty)

    def test_empty_dwell_resets_when_weight_rises(self):
        manager = SessionManager(Mock())
        manager._waiting_for_empty = True
        empty = make_frame(0)
        empty.status = "UNSTABLE"
        loaded = make_frame(1000)
        loaded.status = "UNSTABLE"

        with unittest.mock.patch(
            "services.session.session_manager.time.monotonic",
            side_effect=[10.0, 12.0, 14.0],
        ):
            manager.on_frame(empty, Mock())
            manager.on_frame(loaded, Mock())
            manager.on_frame(empty, Mock())

        self.assertTrue(manager._waiting_for_empty)

    def test_session_end_requires_empty_cycle_for_rearm(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        manager.session.session_id = "session-1"
        manager.session.started_at = 1.0
        manager.session.started_at_iso = "2026-07-20T00:00:00+00:00"
        manager.session.stable_weight = 10000
        manager.session.last_publish_weight = 10000

        manager._end_session("weight_trend_falling", Mock())

        self.assertFalse(manager._waiting_for_empty)

    def test_nearest_session_frame_uses_requested_timestamp(self):
        metadata = {
            "started_at": "2026-07-21T00:00:00+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam2-000001-sample.jpg",
                "cam2-000004-sample.jpg",
                "cam1-000004-sample.jpg",
            ],
            "capture_interval_seconds": 0.2,
        }

        observed_at = datetime.fromisoformat(metadata["started_at"]).timestamp() + 0.75
        path = self.manager._nearest_session_frame(metadata, "cam2", observed_at)

        self.assertEqual(path, "/spool/session/cam2-000004-sample.jpg")

    def test_no_plate_diagnostic_prefers_frames_one_second_after_start(self):
        metadata = {
            "started_at": "2026-07-21T00:00:00+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000000-start.jpg", "cam1-000005-sample.jpg",
                "cam3-000000-start.jpg", "cam3-000005-sample.jpg",
            ],
            "capture_interval_seconds": 0.2,
            "start_frame_paths": {"cam1": "/fallback-cam1.jpg"},
        }
        frames = {
            "/spool/session/cam1-000005-sample.jpg": "cam1+1s",
            "/spool/session/cam3-000005-sample.jpg": "cam3+1s",
        }

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", side_effect=frames.get, create=True,
        ):
            selected = self.manager._load_diagnostic_frames(metadata, offset_seconds=1.0)

        self.assertEqual(selected, {"cam1": "cam1+1s", "cam3": "cam3+1s"})

    def test_unknown_photos_require_session_start_frames_for_current_jobs(self):
        metadata = {
            "session_id": "session-1",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:12+00:00",
            "weight_observed_at": "2026-07-21T00:00:10+00:00",
            "weight_source": "stable",
            "session_dir": "/spool/session",
            "unknown_snapshot_paths": {},
            "session_files": [
                "cam1-000001-sample.jpg", "cam1-000002-sample.jpg",
                "cam2-000001-sample.jpg", "cam2-000002-sample.jpg",
                "cam3-000001-sample.jpg",
            ],
            "capture_interval_seconds": 0.2,
        }
        frame_metadata = {
            "cam1-000001-sample.jpg": {"captured_at": "2026-07-21T00:00:01.700+00:00", "frame_id": 10},
            "cam1-000002-sample.jpg": {"captured_at": "2026-07-21T00:00:02.050+00:00", "frame_id": 11},
            "cam2-000001-sample.jpg": {"captured_at": "2026-07-21T00:00:01.960+00:00", "frame_id": 20},
            "cam2-000002-sample.jpg": {"captured_at": "2026-07-21T00:00:10.400+00:00", "frame_id": 21},
            "cam3-000001-sample.jpg": {"captured_at": "2026-07-21T00:00:08.500+00:00", "frame_id": 30},
        }
        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", return_value="later-frame", create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, Mock(),
            )

        self.assertEqual(selected, {})
        self.assertEqual(captured_at, {})

    def test_unknown_photo_legacy_job_uses_session_end_timing(self):
        metadata = {
            "session_id": "legacy",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:05+00:00",
            "weight_source": "filtered_peak",
            "session_dir": "/spool/session",
            "session_files": ["cam1-000025-sample.jpg"],
            "capture_interval_seconds": 0.2,
        }

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", return_value="end-frame", create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, {}, Mock(),
            )

        self.assertEqual(selected, {"cam1": "end-frame"})
        self.assertEqual(captured_at, {"cam1": "2026-07-21T00:00:05.000+00:00"})

    def test_unknown_photo_legacy_job_uses_authoritative_spool_start(self):
        metadata = {
            "session_id": "legacy-spool-start",
            "started_at": "2026-07-20T23:59:58+00:00",
            "ended_at": "2026-07-21T00:00:05+00:00",
            "weight_source": "filtered_peak",
            "session_dir": "/spool/session",
            "session_files": ["cam1-000025-sample.jpg"],
            "capture_interval_seconds": 0.2,
        }

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", return_value="end-frame", create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, {}, Mock(), "2026-07-21T00:00:00+00:00",
            )

        self.assertEqual(selected, {"cam1": "end-frame"})
        self.assertEqual(captured_at, {"cam1": "2026-07-21T00:00:05.000+00:00"})

    def test_unknown_photos_reject_repeated_stale_camera_frame(self):
        metadata = {
            "session_id": "stale-camera",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:10+00:00",
            "weight_source": "stable",
            "session_dir": "/spool/session",
            "session_files": [
                "cam2-000001-sample.jpg", "cam2-000002-sample.jpg",
            ],
        }
        frame_metadata = {
            "cam2-000001-sample.jpg": {
                "captured_at": "2026-07-21T00:00:01+00:00", "frame_id": 20,
            },
            "cam2-000002-sample.jpg": {
                "captured_at": "2026-07-21T00:00:09.900+00:00", "frame_id": 20,
            },
        }

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", return_value="start-frame", create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, Mock(),
            )

        self.assertEqual(selected, {})
        self.assertEqual(captured_at, {})

    def test_unknown_photos_reject_frame_acquired_before_session(self):
        metadata = {
            "session_id": "pre-session-frame",
            "started_at": "2026-07-21T00:00:10+00:00",
            "ended_at": "2026-07-21T00:00:10.500+00:00",
            "weight_observed_at": "2026-07-21T00:00:10.200+00:00",
            "weight_source": "stable",
            "session_dir": "/spool/session",
            "session_files": ["cam2-000000-sample.jpg"],
            "capture_interval_seconds": 0.2,
        }
        frame_metadata = {
            "cam2-000000-sample.jpg": {
                "captured_at": "2026-07-21T00:00:05+00:00", "frame_id": 20,
            },
        }

        selected, captured_at = self.manager._load_unknown_publish_frames(
            metadata, frame_metadata, Mock(), "2026-07-21T00:00:10+00:00",
        )

        self.assertEqual(selected, {})
        self.assertEqual(captured_at, {})

    def test_unknown_photos_fall_back_to_session_start_snapshot(self):
        metadata = {
            "session_id": "start-only",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:00.100+00:00",
            "weight_observed_at": "2026-07-21T00:00:00.100+00:00",
            "weight_source": "stable",
            "session_dir": "/spool/session",
            "session_files": ["cam1-000000-start.jpg"],
        }
        frame_metadata = {
            "cam1-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00+00:00", "frame_id": None,
            },
        }

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", return_value="start-frame", create=True,
        ):
            selected, captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, Mock(),
            )

        self.assertEqual(selected, {"cam1": "start-frame"})
        self.assertEqual(captured_at, {"cam1": "2026-07-21T00:00:00+00:00"})

    def test_unknown_photos_use_complete_session_start_set_instead_of_mixed_times(self):
        metadata = {
            "session_id": "complete-start",
            "started_at": "2026-07-21T00:00:00+00:00",
            "ended_at": "2026-07-21T00:00:10+00:00",
            "weight_observed_at": "2026-07-21T00:00:09+00:00",
            "session_dir": "/spool/session",
            "session_files": [
                "cam1-000000-start.jpg",
                "cam2-000000-start.jpg",
                "cam3-000000-start.jpg",
            ],
            "unknown_snapshot_paths": {
                "cam1": "/spool/cam1-2s.jpg",
                "cam3": "/spool/cam3-2s.jpg",
            },
            "unknown_snapshot_captured_at": {
                "cam1": "2026-07-21T00:00:02.050+00:00",
                "cam3": "2026-07-21T00:00:02.100+00:00",
            },
        }
        frame_metadata = {
            "cam1-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00.050+00:00", "frame_id": 1,
            },
            "cam2-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00.070+00:00", "frame_id": 2,
            },
            "cam3-000000-start.jpg": {
                "captured_at": "2026-07-21T00:00:00.060+00:00", "frame_id": 3,
            },
        }
        frames = {
            "/spool/session/cam1-000000-start.jpg": "cam1-start",
            "/spool/session/cam2-000000-start.jpg": "cam2-start",
            "/spool/session/cam3-000000-start.jpg": "cam3-start",
        }
        log_fn = Mock()

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread", side_effect=frames.get, create=True,
        ):
            selected, _captured_at = self.manager._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn,
            )

        self.assertEqual(selected, {
            "cam1": "cam1-start", "cam2": "cam2-start", "cam3": "cam3-start",
        })
        metric = next(
            json.loads(call.args[1]) for call in log_fn.call_args_list
            if call.args[0] == "METRIC" and "unknown_photo_selection" in call.args[1]
        )
        self.assertEqual(metric["lpr_target_source"], "session_start")
        self.assertEqual(metric["synchronized_gap_ms"], 20)

    def test_unknown_cam2_uses_configured_lane_crop(self):
        class Frame:
            shape = (2, 4, 3)

            def __getitem__(self, key):
                return key

        frame = Frame()

        left = SessionManager(Mock(), cam2_result_crop="left")._crop_cam2_result_image(frame)
        right = SessionManager(Mock(), cam2_result_crop="right")._crop_cam2_result_image(frame)
        full = SessionManager(Mock(), cam2_result_crop="full")._crop_cam2_result_image(frame)

        self.assertEqual(left, (slice(None), slice(None, 2)))
        self.assertEqual(right, (slice(None), slice(2, None)))
        self.assertIs(full, frame)

    def test_peak_only_snapshot_uses_filtered_peak_timestamp(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        manager.session.session_id = "peak"
        manager.session.started_at = 1.0
        manager.session.started_at_iso = "2026-07-24T00:00:00+00:00"
        manager._session_raw_peak = 1300
        manager._session_raw_peak_observed_at = "2026-07-24T00:00:04+00:00"
        manager._session_filtered_peak = 1200
        manager._session_filtered_peak_observed_at = "2026-07-24T00:00:05+00:00"

        metadata = manager._snapshot_session("weight_trend_falling")

        self.assertEqual(metadata["weight_source"], "filtered_peak")
        self.assertEqual(metadata["weight_observed_at"], "2026-07-24T00:00:05+00:00")
        self.assertEqual(metadata["raw_peak_observed_at"], "2026-07-24T00:00:04+00:00")
        self.assertEqual(metadata["filtered_peak_observed_at"], "2026-07-24T00:00:05+00:00")

    def test_stable_frame_does_not_bypass_chained_wait_gate(self):
        manager = SessionManager(Mock())
        manager.session.rearm_block_until = 11.0
        manager.session.rearm_reference_weight = 10000
        manager.session.rearm_block_reason = "weight_departure"
        manager._attempt_wait_reference = 10000
        frame = self.stable_frame(10600)
        frame.stability_rule = "exact_5"

        with unittest.mock.patch("services.session.session_manager.time.time", return_value=12.0):
            manager.on_frame(frame, Mock())

        self.assertFalse(manager.session.session_active)

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
        self.assertIsNone(manager._attempt)

    def test_stable_same_plateau_does_not_create_attempt(self):
        manager = SessionManager(Mock())
        manager.session.rearm_block_until = 11.0
        manager.session.rearm_reference_weight = 10000
        manager._attempt_wait_reference = 10000
        manager._post_session_low = 10000
        frame = self.stable_frame(10200)
        frame.stability_rule = "exact_5"

        with unittest.mock.patch("services.session.session_manager.time.time", return_value=12.0):
            manager.on_frame(frame, Mock())

        self.assertFalse(manager.session.session_active)
        self.assertIsNone(manager._attempt)

    def test_stable_frame_alone_does_not_start_session(self):
        self.manager.session.session_active = False
        self.manager._attempt_wait_reference = 10000
        frame = self.stable_frame(10300)
        frame.stability_rule = "exact_5"

        self.manager.on_frame(frame, Mock())

        self.assertFalse(self.manager.session.session_active)

    def test_cam2_snapshot_is_saved_when_session_becomes_active(self):
        rear = Mock()
        rear.peek_latest_frame.return_value = "promotion-frame"
        spool = Mock()
        spool.begin_session.return_value = "/spool/session-1"
        spool.save_session_frame.return_value = "/spool/session-1/cam2-start.jpg"
        manager = SessionManager(Mock(), rear_grabber=rear, frame_spool=spool)
        manager.session.stable_weight = 1200
        manager.session.stability_rule = "exact_5"
        manager._attempt = {
            "id": "session-1",
            "started_at": "2026-07-24T00:00:00+00:00",
            "max_weight": 1200,
            "start_frames": {},
        }

        manager._start_session(0, Mock())

        self.assertTrue(manager.session.session_active)
        rear.peek_latest_frame.assert_called_once_with(copy_frame=True)
        spool.save_session_frame.assert_called_once_with(
            "session-1", "cam2-start.jpg", "promotion-frame"
        )
        self.assertEqual(manager.session.rear_capture_source, "promotion")
        self.assertIsNone(manager.session.rear_fallback_deadline)

    def test_spool_admission_failure_does_not_activate_or_clear_attempt(self):
        spool = Mock()
        spool.begin_session.side_effect = OSError("disk failed")
        manager = SessionManager(Mock(), frame_spool=spool)
        manager.session.stable_weight = 1200
        manager.session.stability_rule = "exact_5"
        manager._attempt = {
            "id": "session-1",
            "started_at": "2026-07-24T00:00:00+00:00",
            "max_weight": 1200,
            "start_frames": {},
        }
        log = Mock()

        result = manager._start_session(0, log)

        self.assertFalse(result)
        self.assertFalse(manager.session.session_active)
        self.assertFalse(manager.session.spool_active)
        self.assertEqual(manager._attempt["id"], "session-1")
        self.assertEqual(manager._generation, 0)
        self.assertIn("disk failed", manager.fatal_error)
        spool.abort_session.assert_called_once_with("session-1")
        self.assertTrue(any(
            call.args[0] == "METRIC" and '"event":"session_spool_admission_failed"' in call.args[1]
            for call in log.call_args_list
        ))

    def test_spool_finalize_failure_preserves_session_and_sets_fatal_error(self):
        spool = Mock()
        spool.end_session.side_effect = OSError("disk failed")
        manager = SessionManager(Mock(), frame_spool=spool)
        manager.session.session_active = True
        manager.session.spool_active = True
        manager.session.session_id = "session-1"
        manager.session.started_at_iso = "2026-07-24T00:00:00+00:00"
        manager.session.started_at = 1.0
        manager.session.stable_weight = 1200
        manager._save_diagnostic_frames = Mock()
        log = Mock()

        result = manager._end_session("scale_empty", log)

        self.assertFalse(result)
        self.assertTrue(manager.session.session_active)
        self.assertEqual(manager.session.session_id, "session-1")
        self.assertIn("disk failed", manager.fatal_error)
        manager._save_diagnostic_frames.assert_not_called()

        manager._end_session("shutdown", log)
        failure_metrics = [
            call for call in manager.frame_spool.method_calls
            if call[0] == "end_session"
        ]
        self.assertEqual(len(failure_metrics), 2)
        metric_messages = [
            call.args[1] for call in log.call_args_list
            if call.args and call.args[0] == "METRIC"
        ]
        self.assertEqual(
            sum('"event":"session_spool_finalization_failed"' in message for message in metric_messages),
            1,
        )

    def test_stable_observation_timestamp_is_in_terminal_snapshot(self):
        manager = SessionManager(Mock())
        manager.session.session_active = True
        manager.session.session_id = "session-1"
        manager.session.started_at_iso = "2026-07-24T00:00:00+00:00"
        manager.session.started_at = 1.0
        frame = self.stable_frame(1200)
        frame.timestamp = datetime.fromisoformat("2026-07-24T00:01:02.345+00:00")

        manager._handle_stable_frame(frame, Mock())
        metadata = manager._snapshot_session("scale_empty")

        self.assertEqual(metadata["weight_observed_at"], "2026-07-24T00:01:02.345+00:00")

    def test_scale_reader_stall_finalizes_loaded_session_once(self):
        spool = Mock()
        manager = SessionManager(Mock(), frame_spool=spool)
        manager.session.session_active = True
        manager.session.spool_active = True
        manager.session.scale_owned = True
        manager.session.session_id = "session-1"
        manager.session.started_at_iso = "2026-07-24T00:00:00+00:00"
        manager.session.started_at = 1.0
        manager.session.stable_weight = 30630
        manager.session.stable_weight_observed_at = "2026-07-24T00:01:00+00:00"
        log = Mock()

        manager.on_scale_reader_health("stalled", {"last_valid_age_seconds": 31, "reconnect_count": 0}, log)
        manager.on_scale_reader_health("stalled", {"last_valid_age_seconds": 32, "reconnect_count": 1}, log)

        spool.end_session.assert_called_once()
        metadata = spool.end_session.call_args.args[1]
        self.assertEqual(metadata["end_reason"], "scale_reader_stalled")
        self.assertEqual(metadata["stable_weight"], 30630)
        self.assertTrue(metadata["scale_data_gap"])
        self.assertTrue(manager._scale_recovery_blocked)

    def test_scale_reader_health_payload_event_does_not_break_metric(self):
        manager = SessionManager(Mock())
        log = Mock()

        manager.on_scale_reader_health(
            "stalled",
            {"event": "stalled", "last_valid_age_seconds": 31, "reconnect_count": 0},
            log,
        )

        self.assertTrue(any(
            call.args[0] == "METRIC" and '"event":"scale_reader_stalled"' in call.args[1]
            for call in log.call_args_list
        ))

    def test_scale_reader_stall_blocks_plate_until_fresh_empty_dwell(self):
        manager = SessionManager(Mock())
        log = Mock()
        manager.on_scale_reader_health("stalled", {"last_valid_age_seconds": 31, "reconnect_count": 0}, log)
        manager.on_plate_presence("cam3", {"cam3": True}, log, revision=1)
        self.assertFalse(manager.session.session_active)

        empty = make_frame(0)
        with unittest.mock.patch(
            "services.session.session_manager.time.monotonic", side_effect=[1.0, 3.1],
        ):
            manager.on_frame(empty, log)
            manager.on_frame(empty, log)

        self.assertFalse(manager._scale_recovery_blocked)

    def test_cam2_fallback_runs_once_two_seconds_after_failed_primary(self):
        rear = Mock()
        rear.peek_latest_frame.side_effect = [None, "fallback-frame"]
        spool = Mock()
        spool.begin_session.return_value = "/spool/session-1"
        spool.save_session_frame.return_value = "/spool/session-1/cam2-fallback.jpg"
        manager = SessionManager(Mock(), rear_grabber=rear, frame_spool=spool)
        manager.session.stable_weight = 1200
        manager.session.stability_rule = "exact_5"
        manager._attempt = {
            "id": "session-1",
            "started_at": "2026-07-24T00:00:00+00:00",
            "max_weight": 1200,
            "start_frames": {},
        }

        with unittest.mock.patch(
            "services.session.session_manager.time.time",
            side_effect=[10.0, 11.9, 12.0, 12.1],
        ):
            manager._start_session(0, Mock())
            self.assertFalse(manager._capture_rear_fallback_if_due(Mock()))
            self.assertTrue(manager._capture_rear_fallback_if_due(Mock()))
            self.assertFalse(manager._capture_rear_fallback_if_due(Mock()))

        self.assertEqual(rear.peek_latest_frame.call_count, 2)
        spool.save_session_frame.assert_called_once_with(
            "session-1", "cam2-fallback.jpg", "fallback-frame"
        )
        self.assertEqual(manager.session.rear_capture_source, "fallback_2s")

    def test_publish_uses_saved_cam2_snapshot_not_later_sample(self):
        manager = SessionManager(Mock(), rear_grabber=Mock())
        tracker = Mock()
        tracker.get_image_frame.return_value = (
            Mock(shape=(448, 800, 3)),
            "14C-017.80",
            "cam1",
            10.0,
        )
        manager._nearest_session_frame = Mock(
            return_value="/spool/session/cam2-later.jpg"
        )
        manager._prepare_capture_paths = Mock(return_value={
            key: (f"/{key}.jpg", f"key/{key}.jpg", f"/url/{key}.jpg")
            for key in ("front", "rear", "merged", "unchosen_cam1", "unchosen_cam3")
        })
        manager._build_publish_images = Mock(
            return_value=("front", "merged", "rear")
        )
        manager._crop_cam2_result_image = Mock(side_effect=lambda frame: frame)
        result = {}

        with unittest.mock.patch(
            "services.session.session_manager.cv2.imread",
            side_effect=lambda path: "saved-rear" if path == "/spool/session/cam2-start.jpg" else None,
            create=True,
        ), unittest.mock.patch.object(
            ImageSaveWorker, "save_local_only", return_value=True
        ):
            attached = manager._attach_publish_images(
                result,
                1200,
                0,
                "14C-017.80",
                [],
                Mock(),
                tracker=tracker,
                rear_start_path="/spool/session/cam2-start.jpg",
                session_dir="/spool/session",
                session_files=["cam2-000010-sample.jpg"],
                session_started_at="2026-07-24T00:00:00+00:00",
                session_id="session-1",
            )

        self.assertTrue(attached)
        manager._nearest_session_frame.assert_not_called()
        manager._build_publish_images.assert_called_once_with(
            tracker.get_image_frame.return_value[0],
            "14C-017.80",
            1200,
            0,
            "saved-rear",
        )


class PeakCandidateTests(unittest.TestCase):
    def setUp(self):
        self.manager = SessionManager(Mock(), lpr_grabbers={})
        self.manager._save_diagnostic_frames = Mock(return_value=0)
        self.log = Mock()

    @staticmethod
    def frame(weight, seconds, status="UNSTABLE"):
        frame = make_frame(weight)
        frame.status = status
        frame.timestamp = datetime(2026, 7, 20) + timedelta(seconds=seconds)
        return frame

    def finish_peak(self):
        for index, weight in enumerate([10000] * 5 + [9500] * 13):
            self.manager.on_frame(self.frame(weight, index * 0.2), self.log)

    def archived_metadata(self):
        return self.manager._save_diagnostic_frames.call_args.args[3]

    def test_unstable_departed_peak_is_archived_for_shadow_audit(self):
        self.finish_peak()

        metadata = self.archived_metadata()
        self.assertEqual(metadata["category"], "unstable_local_peak")
        self.assertEqual(metadata["peak_weight_kg"], 10000)
        self.assertTrue(metadata["shadow_only"])

    def test_waiting_for_empty_peak_records_block_reason(self):
        self.manager._waiting_for_empty = True
        self.finish_peak()

        self.assertEqual(self.archived_metadata()["category"], "blocked_waiting_for_empty")

    def test_active_session_peak_does_not_write_redundant_evidence(self):
        self.manager.session.session_active = True
        self.manager.session.session_id = "session-1"
        self.manager.session.weight_departure_baseline = 20000
        self.manager.session.latest_stable_weight = 20000
        with unittest.mock.patch(
            "services.session.session_manager.SAVE_ABSORBED_PEAK_CANDIDATE_EVIDENCE", False,
        ):
            self.finish_peak()

        self.manager._save_diagnostic_frames.assert_not_called()
        metrics = [call.args[1] for call in self.log.call_args_list if call.args[0] == "METRIC"]
        self.assertTrue(any('"category":"absorbed_active_session"' in metric for metric in metrics))
        self.assertTrue(any('"images":0' in metric for metric in metrics))

    def test_active_session_peak_keeps_shadow_evidence_when_enabled(self):
        self.manager.session.session_active = True
        self.manager.session.session_id = "session-1"
        self.manager.session.weight_departure_baseline = 20000
        self.manager.session.latest_stable_weight = 20000
        with unittest.mock.patch(
            "services.session.session_manager.SAVE_ABSORBED_PEAK_CANDIDATE_EVIDENCE", True,
        ):
            self.finish_peak()

        metadata = self.archived_metadata()
        self.assertEqual(metadata["category"], "absorbed_active_session")
        self.assertEqual(metadata["session_ids"], ["session-1"])

    def test_short_spike_does_not_start_peak_candidate(self):
        for index, weight in enumerate([0, 0, 10000, 0, 0]):
            self.manager.on_frame(self.frame(weight, index * 0.2), self.log)

        self.assertIsNone(self.manager._peak_candidate)

    def test_rocking_return_cancels_movement_without_archiving(self):
        weights = [10000] * 5 + [9400] * 4 + [10000] * 5
        for index, weight in enumerate(weights):
            self.manager.on_frame(self.frame(weight, index * 0.2), self.log)

        self.assertIsNotNone(self.manager._peak_candidate)
        self.manager._save_diagnostic_frames.assert_not_called()
        metrics = [call.args[1] for call in self.log.call_args_list if call.args[0] == "METRIC"]
        self.assertTrue(any('"event":"weight_peak_rocking_cancelled"' in metric for metric in metrics))

    def test_five_filtered_departure_frames_archive_candidate(self):
        weights = [10000] * 5 + list(range(9500, 8200, -100))
        for index, weight in enumerate(weights):
            self.manager.on_frame(self.frame(weight, index * 0.2), self.log)

        self.assertIsNone(self.manager._peak_candidate)
        self.assertEqual(self.archived_metadata()["end_reason"], "weight_departure")


class AttemptArchiveTests(unittest.TestCase):
    def test_unstable_attempt_archives_maximum_weight_after_empty_dwell(self):
        manager = SessionManager(Mock(), lpr_grabbers={})
        manager._save_diagnostic_frames = Mock(return_value=0)
        unstable = make_frame(1000)
        unstable.status = "UNSTABLE"
        empty = make_frame(0)
        empty.status = "UNSTABLE"

        with unittest.mock.patch(
            "services.session.session_manager.time.monotonic",
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

    def test_stable_attempt_waits_for_rising_trend(self):
        manager = SessionManager(Mock(), lpr_grabbers={})
        manager._archive_no_stable = Mock()
        log = Mock()
        frame = make_frame(1200)
        frame.status = "STABLE"
        frame.stable_weight = 1200
        frame.stability_rule = "exact_5"

        manager.on_frame(frame, log)

        self.assertFalse(manager.session.session_active)
        manager._archive_no_stable.assert_not_called()

    def test_no_stable_departure_arms_rearm_guard(self):
        from config import SESSION_REARM_DELAY_SECONDS

        manager = SessionManager(Mock(), lpr_grabbers={})
        manager._save_diagnostic_frames = Mock(return_value=0)
        manager._attempt = {
            "id": "att-1",
            "started_at": "2026-07-24T00:00:00+00:00",
            "start_frames": {},
            "max_weight": 1200,
        }

        with unittest.mock.patch(
            "services.session.session_manager.time.time", return_value=100.0,
        ):
            manager._archive_no_stable(Mock(), require_new_rise=True, current_weight=1200)

        self.assertEqual(manager._attempt_rearm_low, 1200)
        self.assertEqual(manager._attempt_wait_reference, 1200)
        self.assertEqual(manager.session.rearm_reference_weight, 1200)
        self.assertEqual(manager.session.rearm_block_reason, "no_stable_weight")
        self.assertEqual(
            manager.session.rearm_block_until, 100.0 + SESSION_REARM_DELAY_SECONDS,
        )

        # Same plateau within the delay must not start a chained session.
        manager.session.stable_weight = 1200
        with unittest.mock.patch(
            "services.session.session_manager.time.time", return_value=100.5,
        ):
            self.assertFalse(manager._can_start_session(Mock()))

        # Once the weight departed and the delay elapsed the guard disarms.
        manager.session.stable_weight = 500
        with unittest.mock.patch(
            "services.session.session_manager.time.time",
            return_value=100.0 + SESSION_REARM_DELAY_SECONDS + 1.0,
        ):
            self.assertTrue(manager._can_start_session(Mock()))
        self.assertEqual(manager.session.rearm_block_until, 0.0)
        self.assertIsNone(manager.session.rearm_reference_weight)

    def test_spread_stability_alone_does_not_log_session_start(self):
        manager = SessionManager(Mock(), lpr_grabbers={})
        log = Mock()
        frame = make_frame(1200)
        frame.status = "STABLE"
        frame.stable_weight = 1200
        frame.stability_rule = "spread_10"

        manager.on_frame(frame, log)

        self.assertFalse(manager.session.session_active)
        self.assertFalse(any(
            call.args[0] == "EVENT" and "Session start reason=" in call.args[1]
            for call in log.call_args_list
        ))

class ReaderFailureTests(unittest.TestCase):
    def _reader(self, max_open_attempts=3):
        db = tempfile.NamedTemporaryFile(suffix=".db")
        self.addCleanup(db.close)
        reader = D2008Reader(
            port="/dev/null",
            db_file=db.name,
            dump_file=None,
            reconnect_initial_seconds=0.0,
            reconnect_max_seconds=0.0,
            max_open_attempts=max_open_attempts,
        )
        self.addCleanup(reader._db.close)
        reader.on_health = Mock()
        reader._running = True
        return reader

    def test_unopenable_serial_port_marks_reader_failed(self):
        import d2008_scale_reader as module

        reader = self._reader(max_open_attempts=3)

        with unittest.mock.patch.object(module.serial, "SerialException", OSError), \
                unittest.mock.patch.object(module.serial, "Serial", side_effect=OSError("no port")):
            reader._run()

        self.assertEqual(reader.state, "failed")
        self.assertEqual(reader._open_failures, 3)
        self.assertFalse(reader._running)
        self.assertTrue(any(call.args[0] == "failed" for call in reader.on_health.call_args_list))
        self.assertIn("serial port unavailable", reader.last_error or "")

    def test_unexpected_error_leaves_reader_failed(self):
        import d2008_scale_reader as module

        reader = self._reader()
        fake = Mock()
        fake.is_open = True
        fake.in_waiting = 0
        fake.read.side_effect = RuntimeError("boom")

        with unittest.mock.patch.object(module.serial, "SerialException", OSError), \
                unittest.mock.patch.object(module.serial, "Serial", return_value=fake):
            reader._run()

        self.assertEqual(reader.state, "failed")
        self.assertIn("unexpected", reader.last_error or "")
        self.assertTrue(any(call.args[0] == "failed" for call in reader.on_health.call_args_list))

    def test_read_failure_on_open_port_is_not_fatal(self):
        import d2008_scale_reader as module

        reader = self._reader()
        fake = Mock()
        fake.is_open = True
        fake.in_waiting = 0

        def boom(*_args, **_kwargs):
            reader._running = False
            raise OSError("read failed")

        fake.read.side_effect = boom

        with unittest.mock.patch.object(module.serial, "SerialException", OSError), \
                unittest.mock.patch.object(module.serial, "Serial", return_value=fake):
            reader._run()

        self.assertNotEqual(reader.state, "failed")
        self.assertEqual(reader._open_failures, 0)

    def test_stop_closes_database_even_when_thread_will_not_join(self):
        reader = self._reader()
        thread = Mock()
        thread.is_alive.return_value = True
        reader._thread = thread
        reader._db = Mock()

        reader.stop()

        reader._db.close.assert_called_once_with()
        thread.join.assert_called_once_with(timeout=3)

    def test_reconnect_resets_stability_history_and_parser(self):
        import d2008_scale_reader as module

        reader = self._reader(max_open_attempts=1)
        stale_parser = reader._parser
        reader._recent_weights.append(1000)
        reader._same_weight = 1000
        reader._same_weight_count = 4
        fake = Mock()
        fake.is_open = True
        fake.in_waiting = 0

        def boom(*_args, **_kwargs):
            reader._running = False
            raise OSError("read failed")

        fake.read.side_effect = boom

        with unittest.mock.patch.object(module.serial, "SerialException", OSError), \
                unittest.mock.patch.object(module.serial, "Serial", return_value=fake):
            reader._run()

        self.assertIsNot(reader._parser, stale_parser)
        self.assertEqual(len(reader._recent_weights), 0)
        self.assertIsNone(reader._same_weight)
        self.assertEqual(reader._same_weight_count, 0)


if __name__ == "__main__":
    unittest.main()
