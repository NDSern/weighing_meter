import json
import os
import queue
import tempfile
import unittest
from unittest.mock import patch

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
        module._published_ids.clear()
        while not module._publish_queue.empty():
            module._publish_queue.get_nowait()

    def tearDown(self):
        self.paths.stop()
        self.finalization_patch.stop()
        self.root.cleanup()

    def test_completed_session_id_is_not_enqueued_again(self):
        module._published_ids.add("session-1")
        PublishOutbox._persist_published_locked()

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
        module._published_ids.add("session-2")
        PublishOutbox._persist_published_locked()

        PublishOutbox._load_published_ids()
        PublishOutbox._load_pending()

        self.assertEqual(PublishOutbox.pending_count(), 0)
        with open(self.pending) as fp:
            self.assertEqual(fp.read(), "")

    def test_finalization_ledger_survives_restart(self):
        session_module.markSessionFinalized("session-3", "no_plate")

        self.assertTrue(session_module.isSessionFinalized("session-3"))
        self.assertFalse(session_module.isSessionFinalized("session-4"))


if __name__ == "__main__":
    unittest.main()
