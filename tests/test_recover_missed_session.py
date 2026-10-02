import json
import sqlite3
import sys
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock


sys.modules.setdefault("cv2", Mock())
sys.modules.setdefault("minio", Mock())
sys.modules.setdefault("minio.error", Mock())

from scripts import recover_missed_session as module


SESSION_ID = "c9f95576942341a28ee7e93812f26061"


def _local_time(utc_iso):
    return datetime.fromisoformat(utc_iso).astimezone()


def _fmt_local(value):
    return value.strftime("%Y-%m-%dT%H:%M:%S.%f")


class MissedSessionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.service = Path(self.root.name)
        for relative in ("storage/no-stable/2026/09/28", "scale_data", "storage"):
            (self.service / relative).mkdir(parents=True, exist_ok=True)
        self.started = "2026-09-28T09:08:53.950+00:00"
        self.ended = "2026-09-28T09:09:51.281+00:00"
        self.local_started = _fmt_local(_local_time(self.started))
        self.local_ended = _fmt_local(_local_time(self.ended))
        self.image = self.service / "storage/no-stable/2026/09/28" / f"{SESSION_ID}_cam3.jpg"
        self.image.write_bytes(b"audited-image")
        metadata = {
            "reason": "no_usable_weight",
            "session_id": SESSION_ID,
            "started_at": self.started,
            "ended_at": self.ended,
            "duration_s": 57.3,
            "end_reason": "scale_empty",
            "stable_weight": 20,
            "weight_source": "stable",
            "raw_peak_weight": 30640,
            "filtered_peak_weight": 30630,
            "filtered_peak_observed_at": "2026-09-28T09:09:14.532+00:00",
            "decimal_pos": 0,
            "images": [str(self.image)],
        }
        (self.image.parent / f"{SESSION_ID}.json").write_text(json.dumps(metadata))
        self._create_finalization(metadata)
        self._create_scale_db()
        self.recovery = module.MissedSessionRecovery(self.service, module.HP1_ID)

    def tearDown(self):
        self.root.cleanup()

    def _create_finalization(self, metadata):
        path = self.service / "storage/session-finalization.db"
        connection = sqlite3.connect(path)
        connection.executescript(
            "CREATE TABLE finalized_sessions (session_id TEXT PRIMARY KEY, outcome TEXT, finalized_at TEXT);"
            "CREATE TABLE terminal_outcomes (session_id TEXT PRIMARY KEY, event_type TEXT, record_json TEXT, finalized_at TEXT);"
        )
        terminal = {"event": "no_stable_attempt", "id": SESSION_ID, **metadata}
        connection.execute(
            "INSERT INTO finalized_sessions VALUES (?, ?, ?)",
            (SESSION_ID, "no_weight", self.ended),
        )
        connection.execute(
            "INSERT INTO terminal_outcomes VALUES (?, ?, ?, ?)",
            (SESSION_ID, "no_stable_attempt", json.dumps(terminal), self.ended),
        )
        connection.commit()
        connection.close()

    def _create_scale_db(self):
        path = self.service / "scale_data/2026-09-28.db"
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE weight_log (timestamp TEXT, weight_kg REAL, status TEXT, checksum_ok INTEGER)"
        )
        started = _local_time(self.started)
        ended = _local_time(self.ended)

        def at(**offset):
            return _fmt_local(started + timedelta(**offset))

        rows = [
            (at(seconds=-3.95), 0, "STABLE", 1),
            (self.local_started, 1000, "UNSTABLE", 1),
            (at(seconds=16.05), 30630, "STABLE", 1),
            (at(seconds=17.05), 30620, "STABLE", 1),
            (at(seconds=18.05), 30630, "STABLE", 1),
            (at(seconds=19.05), 30640, "UNSTABLE", 1),
            (_fmt_local(ended), 0, "STABLE", 1),
        ]
        connection.executemany("INSERT INTO weight_log VALUES (?, ?, ?, ?)", rows)
        connection.commit()
        connection.close()

    def test_plan_uses_raw_loaded_stable_mode_and_exact_image(self):
        plan = self.recovery.plan(SESSION_ID)

        self.assertEqual(plan["scale"]["selected_weight"], 30630)
        self.assertEqual(plan["scale"]["selected_weight_count"], 2)
        self.assertTrue(plan["scale"]["confirmed_empty_before"])
        self.assertTrue(plan["scale"]["confirmed_empty_at_end"])
        self.assertEqual(plan["source_image"], str(self.image.resolve()))
        self.assertEqual(plan["outbox_event"]["id"], SESSION_ID)
        self.assertEqual(
            plan["outbox_event"]["session_result"]["stable_weight"], 30630,
        )
        self.assertEqual(
            plan["outbox_event"]["image_object_keys"], [plan["object_key"]],
        )
        self.assertFalse(plan["pending"])
        self.assertFalse(plan["completed"])

    def test_wrong_host_is_blocked(self):
        recovery = module.MissedSessionRecovery(self.service, module.HP2_ID)

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            recovery.plan(SESSION_ID)

    def test_non_allowlisted_session_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "allowlist"):
            self.recovery.plan("not-allowed")

    def test_missing_or_unstable_loaded_evidence_is_blocked(self):
        path = self.service / "scale_data/2026-09-28.db"
        connection = sqlite3.connect(path)
        connection.execute("UPDATE weight_log SET status = 'UNSTABLE'")
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(RuntimeError, "no loaded STABLE"):
            self.recovery.plan(SESSION_ID)

    def test_missing_image_is_blocked(self):
        self.image.unlink()

        with self.assertRaisesRegex(RuntimeError, "image is missing"):
            self.recovery.plan(SESSION_ID)

    def test_missing_empty_dwell_is_blocked(self):
        path = self.service / "scale_data/2026-09-28.db"
        connection = sqlite3.connect(path)
        connection.execute(
            "UPDATE weight_log SET weight_kg = 500 WHERE timestamp = ?",
            (self.local_ended,),
        )
        connection.commit()
        connection.close()

        with self.assertRaisesRegex(RuntimeError, "stay empty"):
            self.recovery.plan(SESSION_ID)

    def test_apply_requires_exact_frontend_confirmation(self):
        plan = self.recovery.plan(SESSION_ID)

        with self.assertRaisesRegex(RuntimeError, "frontend absence"):
            self.recovery.apply(plan, None, Mock())

    def test_apply_uploads_then_stages_once_and_preserves_finalization(self):
        plan = self.recovery.plan(SESSION_ID)
        upload = Mock()

        self.assertEqual(self.recovery.apply(plan, SESSION_ID, upload), "staged")
        upload.assert_called_once_with(
            module.MINIO_BUCKET, plan["object_key"], plan["destination_image"],
        )
        pending = module._read_json_lines(self.recovery.pending_file)
        self.assertEqual([event["id"] for event in pending], [SESSION_ID])
        self.assertTrue(pending[0]["activated"])
        self.assertEqual(
            pending[0]["session_result"]["offline_event_id"], SESSION_ID,
        )
        audits = module._read_json_lines(self.recovery.audit_file)
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]["session_id"], SESSION_ID)
        with closing(sqlite3.connect(self.recovery.finalization_db)) as connection:
            outcome = connection.execute(
                "SELECT outcome FROM finalized_sessions WHERE session_id = ?", (SESSION_ID,)
            ).fetchone()[0]
        self.assertEqual(outcome, "no_weight")

        replay_plan = self.recovery.plan(SESSION_ID)
        second_upload = Mock()
        self.assertEqual(
            self.recovery.apply(replay_plan, SESSION_ID, second_upload), "pending",
        )
        second_upload.assert_not_called()
        self.assertEqual(len(module._read_json_lines(self.recovery.pending_file)), 1)
        self.assertEqual(len(module._read_json_lines(self.recovery.audit_file)), 1)

    def test_upload_failure_never_stages_outbox(self):
        plan = self.recovery.plan(SESSION_ID)
        upload = Mock(side_effect=OSError("offline"))

        with self.assertRaisesRegex(RuntimeError, "upload failed"):
            self.recovery.apply(plan, SESSION_ID, upload)

        self.assertFalse(self.recovery.pending_file.exists())
        self.assertFalse(self.recovery.audit_file.exists())

    def test_completed_event_is_not_staged(self):
        path = self.service / "storage/publish_completed.db"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(
                "CREATE TABLE completed_events (event_id TEXT PRIMARY KEY, completed_at TEXT)"
            )
            connection.execute(
                "INSERT INTO completed_events VALUES (?, ?)",
                (SESSION_ID, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
        plan = self.recovery.plan(SESSION_ID)
        upload = Mock()

        self.assertEqual(self.recovery.apply(plan, SESSION_ID, upload), "completed")
        upload.assert_not_called()
        self.assertFalse(self.recovery.pending_file.exists())

    def test_completed_event_ignores_stale_audit_payload(self):
        path = self.service / "storage/publish_completed.db"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute(
                "CREATE TABLE completed_events (event_id TEXT PRIMARY KEY, completed_at TEXT)"
            )
            connection.execute(
                "INSERT INTO completed_events VALUES (?, ?)",
                (SESSION_ID, datetime.now(timezone.utc).isoformat()),
            )
            connection.commit()
        self.recovery.audit_file.write_text(
            json.dumps(
                {
                    "event": "staged",
                    "session_id": SESSION_ID,
                    "payload_sha256": "0" * 64,
                }
            )
            + "\n"
        )

        plan = self.recovery.plan(SESSION_ID)

        self.assertTrue(plan["completed"])
        self.assertTrue(plan["already_staged"])

    def test_hp2_case_is_image_less_when_allowed(self):
        hp2 = "9c3dfb52707d4344a7baee4fdaf6ffed"
        original = module.RECOVERY_CASES[hp2]
        self.assertEqual(self.recovery._evidence(hp2, original, {}), (None, None, None))


class ImageLessRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.TemporaryDirectory()
        self.service = Path(self.root.name)
        for relative in ("storage", "scale_data"):
            (self.service / relative).mkdir(parents=True, exist_ok=True)
        self.started = "2026-09-27T07:40:24.276+00:00"
        self.ended = "2026-09-27T07:40:47.526+00:00"
        metadata = {
            "event": "session_duplicate",
            "id": module.HP2_CASE_ID,
            "session_id": module.HP2_CASE_ID,
            "started_at": self.started,
            "ended_at": self.ended,
            "end_reason": "scale_empty",
            "stable_weight": 17710,
            "weight_source": "stable",
            "raw_peak_weight": 17940,
        }
        path = self.service / "storage/session-finalization.db"
        connection = sqlite3.connect(path)
        connection.executescript(
            "CREATE TABLE finalized_sessions (session_id TEXT PRIMARY KEY, outcome TEXT, finalized_at TEXT);"
            "CREATE TABLE terminal_outcomes (session_id TEXT PRIMARY KEY, event_type TEXT, record_json TEXT, finalized_at TEXT);"
        )
        connection.execute(
            "INSERT INTO finalized_sessions VALUES (?, ?, ?)",
            (module.HP2_CASE_ID, "duplicate", self.ended),
        )
        connection.execute(
            "INSERT INTO terminal_outcomes VALUES (?, ?, ?, ?)",
            (module.HP2_CASE_ID, "session_duplicate", json.dumps(metadata), self.ended),
        )
        connection.commit()
        connection.close()
        path = self.service / "scale_data/2026-09-27.db"
        connection = sqlite3.connect(path)
        connection.execute(
            "CREATE TABLE weight_log (timestamp TEXT, weight_kg REAL, status TEXT, checksum_ok INTEGER)"
        )
        started = _local_time(self.started)

        def at(seconds):
            return _fmt_local(started + timedelta(seconds=seconds))

        rows = [
            (at(-14.276), 41300, "STABLE", 1),
            (at(-8.276), 0, "STABLE", 1),
            (at(2.460), 17680, "UNSTABLE", 1),
            (at(5.724), 17710, "STABLE", 1),
            (at(10.724), 17710, "STABLE", 1),
            (at(15.724), 17710, "STABLE", 1),
            (at(18.724), 17940, "UNSTABLE", 1),
            (at(20.940), 17720, "STABLE", 1),
            (at(22.224), 0, "STABLE", 1),
        ]
        connection.executemany("INSERT INTO weight_log VALUES (?, ?, ?, ?)", rows)
        connection.commit()
        connection.close()
        self.recovery = module.MissedSessionRecovery(self.service, module.HP2_ID)

    def tearDown(self):
        self.root.cleanup()

    def test_hp2_plans_image_less_without_attributable_image(self):
        plan = self.recovery.plan(module.HP2_CASE_ID)

        self.assertTrue(plan["image_less"])
        self.assertIsNone(plan["object_key"])
        self.assertIsNone(plan["source_image"])
        self.assertIsNone(plan["destination_image"])
        self.assertEqual(plan["outbox_event"]["image_object_keys"], [])
        self.assertEqual(plan["outbox_event"]["image_paths"], [])
        self.assertEqual(plan["outbox_event"]["session_result"]["photos"], [])
        self.assertTrue(plan["outbox_event"]["session_result"]["metadata"]["image_less"])
        self.assertEqual(plan["scale"]["selected_weight"], 17710)
        self.assertEqual(plan["scale"]["raw_peak_weight"], 17940)
        self.assertTrue(plan["scale"]["confirmed_empty_before"])
        self.assertTrue(plan["scale"]["confirmed_empty_at_end"])

    def test_image_less_apply_requires_explicit_approval(self):
        plan = self.recovery.plan(module.HP2_CASE_ID)

        with self.assertRaisesRegex(RuntimeError, "image-less"):
            self.recovery.apply(plan, module.HP2_CASE_ID, Mock())

    def test_image_less_apply_stages_without_upload(self):
        plan = self.recovery.plan(module.HP2_CASE_ID)
        upload = Mock()

        self.assertEqual(
            self.recovery.apply(
                plan, module.HP2_CASE_ID, upload, allow_image_less=True,
            ),
            "staged",
        )
        upload.assert_not_called()
        pending = module._read_json_lines(self.recovery.pending_file)
        self.assertEqual([event["id"] for event in pending], [module.HP2_CASE_ID])
        self.assertEqual(pending[0]["image_object_keys"], [])
        self.assertEqual(pending[0]["image_paths"], [])
        audits = module._read_json_lines(self.recovery.audit_file)
        self.assertEqual(len(audits), 1)
        self.assertTrue(audits[0]["image_less"])
        self.assertIsNone(audits[0]["object_key"])


if __name__ == "__main__":
    unittest.main()
