import tempfile
import unittest
import sys
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

sys.modules.setdefault("serial", Mock())
sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("numpy", Mock())
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

    def test_falling_trend_ends_session_and_starts_chained_attempt(self):
        end_session = Mock(side_effect=lambda *_args: setattr(self.manager.session, "session_active", False))
        self.manager._end_session = end_session
        self.manager.session.weight_trend_window.clear()

        for weight in range(39120, 37620, -100):
            frame = make_frame(weight)
            frame.status = "UNSTABLE"
            self.manager.on_frame(frame, Mock())

        end_session.assert_called_once_with("weight_trend_falling", unittest.mock.ANY)
        self.assertIsNotNone(self.manager._attempt)

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
            "services.session.session_manager.time.time",
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
            "services.session.session_manager.time.time",
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

    def test_active_session_peak_records_absorption_reason(self):
        self.manager.session.session_active = True
        self.manager.session.session_id = "session-1"
        self.manager.session.weight_departure_baseline = 20000
        self.manager.session.latest_stable_weight = 20000
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

if __name__ == "__main__":
    unittest.main()
