"""Tests for the session-finalization read/write store."""

import os
import sqlite3
import sys
import tempfile
import unittest
from unittest import mock
from unittest.mock import Mock

sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from services.session import finalization_store


class FinalizationStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = os.path.join(self.tmp.name, "session-finalization.db")

    def test_unknown_session_is_not_finalized(self):
        self.assertFalse(finalization_store.contains(self.db, "nope"))
        self.assertIsNone(finalization_store.get(self.db, "nope"))

    def test_marked_session_is_finalized_with_record(self):
        finalization_store.mark(
            self.db, "s1", "published", {"event": "session_publish_queued", "id": "s1"},
        )
        self.assertTrue(finalization_store.contains(self.db, "s1"))
        outcome, record = finalization_store.get(self.db, "s1")
        self.assertEqual(outcome, "published")
        self.assertEqual(record["id"], "s1")

    def test_contains_raises_on_read_failure(self):
        error = sqlite3.OperationalError("database is locked")
        with mock.patch.object(finalization_store.sqlite3, "connect", side_effect=error):
            with self.assertRaises(sqlite3.OperationalError):
                finalization_store.contains(self.db, "s1")

    def test_get_raises_on_read_failure(self):
        error = sqlite3.OperationalError("database is locked")
        with mock.patch.object(finalization_store.sqlite3, "connect", side_effect=error):
            with self.assertRaises(sqlite3.OperationalError):
                finalization_store.get(self.db, "s1")

    def test_read_path_commits_schema(self):
        self.assertFalse(finalization_store.contains(self.db, "nope"))
        with sqlite3.connect(self.db) as connection:
            names = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'",
                )
            }
        self.assertIn("finalized_sessions", names)


if __name__ == "__main__":
    unittest.main()
