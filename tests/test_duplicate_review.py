import json
import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta

from services.review.duplicate_review import (
    DuplicateReviewStore,
    DuplicateReviewer,
    ScaleContinuityReader,
)
from services.review.mock_integrations import MINIO_MOCK_FILE, MQTT_MOCK_FILE

BASE = datetime(2026, 9, 23, 12, 0, 0)


def _session(session_id, start, end, weight=50000.0, plate="51A-12345",
             plate_status="confirmed", weight_source="stable",
             published=True, outbox=None, images=None):
    return {
        "session_id": session_id,
        "started_at": start.isoformat(),
        "ended_at": end.isoformat(),
        "plate": plate,
        "plate_status": plate_status,
        "stable_weight_kg": weight,
        "weight_source": weight_source,
        "published": published,
        "outbox_event_id": outbox or session_id,
        "image_object_keys": list(images or []),
    }


def _write_day(data_dir, day, readings):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "%s.db" % day.strftime("%Y-%m-%d"))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS weight_log ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "weight_kg REAL NOT NULL, sign TEXT NOT NULL, decimal_pos INTEGER NOT NULL, "
            "checksum_ok INTEGER NOT NULL, status TEXT NOT NULL DEFAULT 'UNSTABLE')"
        )
        for moment, weight, checksum in readings:
            connection.execute(
                "INSERT INTO weight_log (timestamp, weight_kg, sign, decimal_pos, checksum_ok, status) "
                "VALUES (?, ?, '+', 0, ?, 'STABLE')",
                (moment.isoformat(), weight, checksum),
            )
        connection.commit()


def _ramp(day, start, seconds, weight=50000.0, step=1.0):
    return [
        (start + timedelta(seconds=offset), weight, 1)
        for offset in [index * step for index in range(int(seconds / step) + 1)]
    ]


class DuplicateReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.scale_dir = os.path.join(self.tmp.name, "scale_data")
        self.mock_dir = os.path.join(self.tmp.name, "review-mock")
        self.store = DuplicateReviewStore(os.path.join(self.tmp.name, "review.db"))
        self.reader = ScaleContinuityReader(
            self.scale_dir, max_gap_seconds=15.0, weight_threshold=100.0,
        )
        self.reviewer = DuplicateReviewer(
            store=self.store, reader=self.reader, mock_dir=self.mock_dir,
            reconcile_hours=100000.0,
        )

    def _count_lines(self, name):
        path = os.path.join(self.mock_dir, name)
        if not os.path.exists(path):
            return 0
        with open(path) as fp:
            return len([line for line in fp if line.strip()])

    def _payloads(self, name):
        path = os.path.join(self.mock_dir, name)
        with open(path) as fp:
            return [json.loads(line) for line in fp if line.strip()]

    def test_continuous_load_emits_one_action_for_published_loser(self):
        _write_day(self.scale_dir, BASE.date(), _ramp(BASE.date(), BASE, 8))
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE,
                           weight_source="filtered_peak",
                           images=["storage/weighbridge/a-cam1.jpg"])
        later = _session("b" * 32, BASE + timedelta(seconds=8), BASE + timedelta(seconds=40),
                         images=["storage/weighbridge/b-cam1.jpg", "storage/weighbridge/b-cam3.jpg"])
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)

        self.assertTrue(self.store.has_action("%s->%s" % (earlier["session_id"], later["session_id"])))
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 1)
        self.assertEqual(self._count_lines(MINIO_MOCK_FILE), 1)
        mqtt = self._payloads(MQTT_MOCK_FILE)[0]
        self.assertEqual(mqtt["payload"]["published_event_id"], earlier["outbox_event_id"])
        self.assertEqual(mqtt["payload"]["surviving_session_id"], later["session_id"])
        removal = self._payloads(MINIO_MOCK_FILE)[0]
        self.assertEqual(removal["objects"], ["storage/weighbridge/a-cam1.jpg"])

    def test_confirmed_empty_reading_blocks_automatic_action(self):
        readings = _ramp(BASE.date(), BASE, 3) + [(BASE + timedelta(seconds=4), 60.0, 1)]
        readings += _ramp(BASE.date(), BASE + timedelta(seconds=5), 4)
        _write_day(self.scale_dir, BASE.date(), readings)
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE)
        later = _session("b" * 32, BASE + timedelta(seconds=10), BASE + timedelta(seconds=40))
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)

        self.assertFalse(self.store.has_action("%s->%s" % (earlier["session_id"], later["session_id"])))
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 0)

    def test_missing_scale_database_is_review_only(self):
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE)
        later = _session("b" * 32, BASE + timedelta(seconds=5), BASE + timedelta(seconds=40))
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 0)

    def test_large_scale_gap_is_review_only(self):
        readings = [(BASE, 50000.0, 1), (BASE + timedelta(seconds=40), 50000.0, 1)]
        _write_day(self.scale_dir, BASE.date(), readings)
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE)
        later = _session("b" * 32, BASE + timedelta(seconds=40), BASE + timedelta(seconds=80))
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 0)

    def test_corrupt_reading_is_review_only(self):
        readings = _ramp(BASE.date(), BASE, 8)
        readings[2] = (readings[2][0], readings[2][1], 0)
        _write_day(self.scale_dir, BASE.date(), readings)
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE)
        later = _session("b" * 32, BASE + timedelta(seconds=8), BASE + timedelta(seconds=40))
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 0)

    def test_confirmed_plate_beats_unknown_even_with_fewer_images(self):
        _write_day(self.scale_dir, BASE.date(), _ramp(BASE.date(), BASE, 8))
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE, plate="UNKNOWN_OCR",
                           plate_status="unreadable",
                           images=["storage/weighbridge/a1.jpg", "storage/weighbridge/a2.jpg"])
        later = _session("b" * 32, BASE + timedelta(seconds=8), BASE + timedelta(seconds=40),
                         images=["storage/weighbridge/b1.jpg"])
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)
        self.assertTrue(self.store.has_action("%s->%s" % (earlier["session_id"], later["session_id"])))
        self.assertFalse(self.store.has_action("%s->%s" % (later["session_id"], earlier["session_id"])))

    def test_unpublished_loser_gets_no_mock_action(self):
        _write_day(self.scale_dir, BASE.date(), _ramp(BASE.date(), BASE, 8))
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE, published=False)
        later = _session("b" * 32, BASE + timedelta(seconds=8), BASE + timedelta(seconds=40))
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 0)

    def test_transitive_group_and_stronger_later_survivor(self):
        _write_day(self.scale_dir, BASE.date(), _ramp(BASE.date(), BASE, 80))
        first = _session("a" * 32, BASE - timedelta(seconds=30), BASE,
                         weight_source="filtered_peak",
                         images=["storage/weighbridge/a1.jpg"])
        second = _session("b" * 32, BASE + timedelta(seconds=8), BASE + timedelta(seconds=45),
                          weight_source="filtered_peak",
                          images=["storage/weighbridge/b1.jpg", "storage/weighbridge/b2.jpg"])
        self.reviewer.register_session(first)
        self.reviewer.register_session(second)
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 1)

        third = _session("c" * 32, BASE + timedelta(seconds=55), BASE + timedelta(seconds=75),
                         weight_source="stable",
                         images=["storage/weighbridge/c1.jpg", "storage/weighbridge/c2.jpg"],
                         outbox="c-outbox")
        self.reviewer.register_session(third)

        self.assertTrue(self.store.has_action("%s->%s" % (first["session_id"], third["session_id"])))
        self.assertTrue(self.store.has_action("%s->%s" % (second["session_id"], third["session_id"])))
        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), 3)

    def test_replay_is_idempotent(self):
        _write_day(self.scale_dir, BASE.date(), _ramp(BASE.date(), BASE, 8))
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE)
        later = _session("b" * 32, BASE + timedelta(seconds=8), BASE + timedelta(seconds=40))
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)
        before_mqtt = self._count_lines(MQTT_MOCK_FILE)

        self.reviewer.reconcile()
        self.reviewer.register_session(later)
        self.reviewer.reconcile()

        self.assertEqual(self._count_lines(MQTT_MOCK_FILE), before_mqtt)
        self.assertEqual(self._count_lines(MINIO_MOCK_FILE), before_mqtt)

    def test_mocks_stay_local_and_scope_exact_keys(self):
        _write_day(self.scale_dir, BASE.date(), _ramp(BASE.date(), BASE, 8))
        earlier = _session("a" * 32, BASE - timedelta(seconds=30), BASE,
                           weight_source="filtered_peak",
                           images=["storage/weighbridge/2026/09/23/a_cam1.jpg"])
        later = _session("b" * 32, BASE + timedelta(seconds=8), BASE + timedelta(seconds=40),
                         images=["storage/weighbridge/2026/09/23/b_cam1.jpg"])
        self.reviewer.register_session(earlier)
        self.reviewer.register_session(later)

        self.assertEqual(
            sorted(os.listdir(self.mock_dir)), sorted([MINIO_MOCK_FILE, MQTT_MOCK_FILE]),
        )
        removal = self._payloads(MINIO_MOCK_FILE)[0]
        self.assertEqual(removal["objects"], ["storage/weighbridge/2026/09/23/a_cam1.jpg"])
        mqtt = self._payloads(MQTT_MOCK_FILE)[0]
        self.assertTrue(mqtt["provisional"])


if __name__ == "__main__":
    unittest.main()