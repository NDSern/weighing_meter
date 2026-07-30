import json
import os
import queue
import tempfile
import unittest
import sys
from unittest.mock import patch
from unittest.mock import Mock

sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("numpy", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from services.storage import publish_outbox as module
from services.storage.publish_outbox import PublishOutbox
from services.session import session_manager as session_module


class PublishOutboxIdempotencyTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.pending = os.path.join(self.root.name, "pending.jsonl")
        self.completed = os.path.join(self.root.name, "completed.db")
        self.paths = patch.multiple(
            module, _outbox_file=self.pending, _published_db=self.completed,
        )
        self.paths.start()
        self.finalization_db = os.path.join(self.root.name, "finalization.db")
        self.finalization_patch = patch.object(
            session_module, "SESSION_FINALIZATION_DB", self.finalization_db
        )
        self.finalization_patch.start()
        module._pending_events.clear()
        while not module._publish_queue.empty():
            module._publish_queue.get_nowait()

    def tearDown(self):
        self.paths.stop()
        self.finalization_patch.stop()
        self.root.cleanup()

    def test_completed_session_id_is_not_enqueued_again(self):
        PublishOutbox._mark_completed("session-1")

        event_id = PublishOutbox.enqueue({"offline_event_id": "session-1"})

        self.assertEqual(event_id, "session-1")
        self.assertEqual(PublishOutbox.pending_count(), 0)

    def test_pending_copy_is_removed_when_completed_ledger_wins_after_crash(self):
        event = {
            "id": "session-2",
            "created_at": "2026-07-15T00:00:00",
            "image_object_keys": [],
            "image_paths": [],
            "session_result": {"offline_event_id": "session-2"},
        }
        with open(self.pending, "w") as fp:
            fp.write(json.dumps(event) + "\n")
        PublishOutbox._mark_completed("session-2")

        PublishOutbox._init_completed_db()
        PublishOutbox._load_pending()

        self.assertEqual(PublishOutbox.pending_count(), 0)
        with open(self.pending) as fp:
            self.assertEqual(fp.read(), "")

    def test_deferred_enqueue_waits_for_activation(self):
        event_id = PublishOutbox.enqueue(
            {"offline_event_id": "deferred"}, activate=False,
        )

        self.assertEqual(event_id, "deferred")
        self.assertTrue(module._publish_queue.empty())
        module._pending_events.clear()
        PublishOutbox._load_pending()
        self.assertTrue(module._publish_queue.empty())
        self.assertTrue(PublishOutbox.activate(event_id))
        self.assertEqual(module._publish_queue.get_nowait(), "deferred")

    def test_activation_survives_crash_before_queue_insertion(self):
        PublishOutbox.enqueue({"offline_event_id": "activation-crash"}, activate=False)
        with patch.object(module._publish_queue, "put", side_effect=RuntimeError("power loss")):
            with self.assertRaises(RuntimeError):
                PublishOutbox.activate("activation-crash")

        module._pending_events.clear()
        PublishOutbox._load_pending()

        self.assertEqual(module._publish_queue.get_nowait(), "activation-crash")

    def test_outbox_replace_fsyncs_parent_directory(self):
        with patch.object(module.os, "open", return_value=123) as open_dir, patch.object(
            module.os, "fsync",
        ) as fsync, patch.object(module.os, "close") as close:
            PublishOutbox.enqueue({"offline_event_id": "durable"}, activate=False)

        open_dir.assert_called_once_with(self.root.name, os.O_RDONLY | os.O_DIRECTORY)
        fsync.assert_any_call(123)
        close.assert_called_once_with(123)

    def test_finalized_retry_activates_deferred_outbox(self):
        PublishOutbox.enqueue({"offline_event_id": "retry"}, activate=False)
        session_module.markSessionFinalized("retry", "published", {
            "event": "session_publish_queued", "id": "retry", "outbox_event_id": "retry",
        })
        manager = session_module.SessionManager(Mock())

        self.assertTrue(manager.finalize_deferred_session(
            self.deferred_metadata("retry"), Mock(), Mock(),
        ))

        self.assertEqual(module._publish_queue.get_nowait(), "retry")

    def test_finalized_publish_without_outbox_evidence_is_not_acknowledged(self):
        session_module.markSessionFinalized("missing", "published", {
            "event": "session_publish_queued", "id": "missing", "outbox_event_id": "missing",
        })
        manager = session_module.SessionManager(Mock())

        with patch.object(session_module, "MQTT_ENABLED", True):
            self.assertFalse(manager.finalize_deferred_session(
                self.deferred_metadata("missing"), Mock(), Mock(),
            ))

    def test_legacy_published_finalization_requires_session_id_evidence(self):
        session_module.markSessionFinalized("legacy", "published")
        manager = session_module.SessionManager(Mock())

        with patch.object(session_module, "MQTT_ENABLED", True):
            self.assertFalse(manager.finalize_deferred_session(
                self.deferred_metadata("legacy"), Mock(), Mock(),
            ))

    def test_mqtt_disabled_finalization_needs_no_outbox_evidence(self):
        session_module.markSessionFinalized("local-only", "published")
        manager = session_module.SessionManager(Mock())

        with patch.object(session_module, "MQTT_ENABLED", False):
            self.assertTrue(manager.finalize_deferred_session(
                self.deferred_metadata("local-only"), Mock(), Mock(),
            ))

    def test_completed_ledger_satisfies_finalized_retry(self):
        PublishOutbox._mark_completed("completed")
        session_module.markSessionFinalized("completed", "published", {
            "event": "session_publish_queued", "id": "completed",
            "outbox_event_id": "completed",
        })
        manager = session_module.SessionManager(Mock())

        self.assertTrue(manager.finalize_deferred_session(
            self.deferred_metadata("completed"), Mock(), Mock(),
        ))

    def test_completed_ledger_is_not_loaded_into_memory(self):
        for index in range(1000):
            PublishOutbox._mark_completed(f"session-{index}")

        self.assertFalse(hasattr(module, "_published_ids"))
        self.assertTrue(PublishOutbox.has_event("session-999"))

    def test_finalization_ledger_survives_restart(self):
        terminal = {
            "event": "session_no_plate", "id": "session-3",
            "started_at": "2026-07-24T00:00:00+00:00",
            "ended_at": "2026-07-24T00:01:00+00:00",
        }
        session_module.markSessionFinalized("session-3", "no_plate", terminal)

        self.assertTrue(session_module.isSessionFinalized("session-3"))
        self.assertFalse(session_module.isSessionFinalized("session-4"))
        import sqlite3
        connection = sqlite3.connect(self.finalization_db)
        try:
            stored = connection.execute(
                "SELECT record_json FROM terminal_outcomes WHERE session_id = ?", ("session-3",)
            ).fetchone()
        finally:
            connection.close()
        self.assertEqual(json.loads(stored[0]), terminal)

    def test_plate_count_updates_once_per_session_id(self):
        db_path = os.path.join(self.root.name, "plates.db")
        with patch.object(session_module, "SERVICE_DIR", self.root.name):
            first = session_module.saveConfirmedLicensePlate("14C-000.01", "session-5")
            replay = session_module.saveConfirmedLicensePlate("14C-000.01", "session-5")

        self.assertEqual(first, 1)
        self.assertEqual(replay, 1)

    def terminal_outcome(self, session_id):
        import sqlite3
        connection = sqlite3.connect(self.finalization_db)
        try:
            row = connection.execute(
                "SELECT record_json FROM terminal_outcomes WHERE session_id = ?", (session_id,)
            ).fetchone()
        finally:
            connection.close()
        return json.loads(row[0])

    def deferred_metadata(self, session_id):
        return {
            "session_id": session_id,
            "stable_weight": 1000,
            "decimal_pos": 0,
            "started_at": "2026-07-24T00:00:05+00:00",
            "ended_at": "2026-07-24T00:00:06+00:00",
            "end_reason": "scale_empty",
        }

    def test_duplicate_terminal_outcome_uses_registry_corrected_plate(self):
        manager = session_module.SessionManager(Mock())
        manager._last_publish_plate = "CANON-1"
        manager._last_publish_session_end = "2026-07-24T00:00:00+00:00"
        tracker = Mock()
        tracker.get_confirmed_plate.return_value = ("RAW-1", 0.9, 4)
        tracker.get_all_plates_summary.return_value = {"RAW-1": 4}

        with patch.object(
            session_module, "correctWithRegisteredLicensePlate",
            return_value=("CANON-1", "exact"),
        ):
            self.assertTrue(manager.finalize_deferred_session(
                self.deferred_metadata("registry"), tracker, Mock(),
            ))

        self.assertEqual(self.terminal_outcome("registry")["plate"], "CANON-1")

    def test_duplicate_terminal_outcome_uses_detailed_candidate_plate(self):
        manager = session_module.SessionManager(Mock())
        manager._last_publish_plate = "15C12340"
        manager._last_publish_session_end = "2026-07-24T00:00:00+00:00"
        tracker = Mock()
        tracker.get_confirmed_plate.return_value = ("15C1234", 0.9, 4)
        tracker.get_all_plates_summary.return_value = {"15C1234": 4, "15C12340": 3}

        with patch.object(
            session_module, "correctWithRegisteredLicensePlate",
            side_effect=lambda plate: (plate, None),
        ):
            self.assertTrue(manager.finalize_deferred_session(
                self.deferred_metadata("detailed"), tracker, Mock(),
            ))

        self.assertEqual(self.terminal_outcome("detailed")["plate"], "15C12340")

    def test_publish_activates_outbox_only_after_terminal_ledger(self):
        manager = session_module.SessionManager(Mock())
        tracker = Mock()
        tracker.get_confirmed_plate.return_value = ("30A-1", 0.9, 4)
        manager.publish_result = Mock(return_value={
            "status": "published", "plate": "30A-1", "outbox_event_id": "ordered",
        })

        def assert_finalized(event_id):
            self.assertEqual(event_id, "ordered")
            self.assertTrue(session_module.isSessionFinalized("ordered"))
            return True

        with patch.object(PublishOutbox, "activate", side_effect=assert_finalized):
            self.assertTrue(manager.finalize_deferred_session(
                self.deferred_metadata("ordered"), tracker, Mock(),
            ))

    def test_publish_success_logs_one_prominent_line_and_metric(self):
        event_id = "1234567890abcdef"
        module._pending_events[event_id] = {
            "id": event_id,
            "created_at": "2026-07-23T00:00:00",
            "image_paths": [],
            "session_result": {
                "official_plate": "14C-017.80",
                "stable_weight": 34780.0,
            },
        }
        mqtt = Mock()
        mqtt.publish_weighbridge_event.return_value = True
        logs = []

        with patch.object(module, "_mqtt_svc", mqtt), patch.object(
            module, "_log_fn", lambda level, message: logs.append((level, message))
        ):
            PublishOutbox._publish_event(event_id)

        human = [entry for entry in logs if entry[0] != "METRIC"]
        self.assertEqual(human, [(">>> SENT <<<", "plate=14C-017.80 wt=34780kg id=12345678")])
        self.assertEqual(len([entry for entry in logs if entry[0] == "METRIC"]), 1)


if __name__ == "__main__":
    unittest.main()
