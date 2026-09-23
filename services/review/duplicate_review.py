"""HP-01 duplicate session review driven by authoritative raw scale records.

MOCK-ONLY. Compares adjacent finalized sessions, groups sessions whose raw
scale evidence proves an unbroken above-threshold load, keeps the single
best-evidence survivor, and emits mock compensation intents for the published
losers. No real MQTT event or MinIO object is ever deleted here.
"""

import json
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta, timezone

from config import (
    DUPLICATE_REVIEW_DB,
    DUPLICATE_REVIEW_MAX_GAP_SECONDS,
    DUPLICATE_REVIEW_MOCK_DIR,
    DUPLICATE_REVIEW_RECONCILE_HOURS,
    SCALE_DATA_DIR,
    WEIGHT_THRESHOLD,
)
from services.review.mock_integrations import (
    build_minio_removal_intent,
    build_mqtt_compensation_intent,
    emit_mock_actions,
)

SCHEMA = (
    "CREATE TABLE IF NOT EXISTS review_sessions ("
    "session_id TEXT PRIMARY KEY,"
    "started_at TEXT NOT NULL,"
    "ended_at TEXT NOT NULL,"
    "plate TEXT,"
    "plate_status TEXT,"
    "stable_weight_kg REAL,"
    "weight_source TEXT,"
    "published INTEGER NOT NULL,"
    "outbox_event_id TEXT,"
    "image_object_keys TEXT NOT NULL,"
    "updated_at TEXT NOT NULL"
    ");"
    "CREATE TABLE IF NOT EXISTS review_actions ("
    "action_key TEXT PRIMARY KEY,"
    "losing_session_id TEXT NOT NULL,"
    "surviving_session_id TEXT NOT NULL,"
    "reason TEXT NOT NULL,"
    "mqtt_mock_path TEXT,"
    "minio_mock_path TEXT,"
    "created_at TEXT NOT NULL"
    ");"
)

COLUMNS = (
    "session_id", "started_at", "ended_at", "plate", "plate_status",
    "stable_weight_kg", "weight_source", "published", "outbox_event_id",
    "image_object_keys", "updated_at",
)


def _iso_to_epoch(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        return parsed.timestamp()
    return parsed.timestamp()


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ScaleContinuityReader:
    """Read-only access to the authoritative per-day scale-log databases."""

    def __init__(self, data_dir=SCALE_DATA_DIR, max_gap_seconds=DUPLICATE_REVIEW_MAX_GAP_SECONDS,
                 weight_threshold=WEIGHT_THRESHOLD):
        self.data_dir = data_dir
        self.max_gap_seconds = float(max_gap_seconds)
        self.weight_threshold = float(weight_threshold)

    def _db_path(self, day):
        return os.path.join(self.data_dir, "%s.db" % day.strftime("%Y-%m-%d"))

    def _query(self, path, start_epoch, end_epoch):
        iso_start = datetime.fromtimestamp(start_epoch).isoformat()
        iso_end = datetime.fromtimestamp(end_epoch).isoformat()
        with closing(sqlite3.connect(path)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(weight_log)")}
            if "timestamp" not in columns or "weight_kg" not in columns:
                raise sqlite3.Error("weight_log schema is not usable")
            has_checksum = "checksum_ok" in columns
            select = "timestamp, weight_kg" + (", checksum_ok" if has_checksum else "")
            rows = connection.execute(
                "SELECT %s FROM weight_log WHERE timestamp >= ? AND timestamp <= ?" % select,
                (iso_start, iso_end),
            ).fetchall()
        readings = []
        for row in rows:
            epoch = _iso_to_epoch(row[0])
            if epoch is None:
                continue
            if has_checksum and row[2] in (0, False):
                readings.append((epoch, None))
            else:
                readings.append((epoch, float(row[1])))
        return readings

    def readings_between(self, start_epoch, end_epoch):
        """Return (readings, complete). complete=False when data could not be read."""
        readings = []
        complete = True
        if end_epoch < start_epoch:
            return readings, complete
        day = datetime.fromtimestamp(start_epoch).date()
        last_day = datetime.fromtimestamp(end_epoch).date()
        while day <= last_day:
            path = self._db_path(day)
            if not os.path.exists(path):
                complete = False
            else:
                try:
                    readings.extend(self._query(path, start_epoch, end_epoch))
                except sqlite3.Error:
                    complete = False
            day += timedelta(days=1)
        readings.sort(key=lambda item: item[0])
        return readings, complete

    def continuous_load(self, start_epoch, end_epoch):
        """Return (is_continuous, reason) for the interval between two sessions."""
        if start_epoch is None or end_epoch is None:
            return False, "missing_session_time"
        if end_epoch <= start_epoch:
            return False, "non_positive_window"
        readings, complete = self.readings_between(start_epoch, end_epoch)
        if not complete:
            return False, "missing_scale_data"
        if not readings:
            return False, "no_scale_readings"
        if any(weight is None for _, weight in readings):
            return False, "corrupt_scale_reading"
        if any(weight <= self.weight_threshold for _, weight in readings):
            return False, "confirmed_empty"
        previous = start_epoch
        for epoch, _ in readings:
            if epoch - previous > self.max_gap_seconds:
                return False, "scale_data_gap"
            previous = epoch
        if end_epoch - previous > self.max_gap_seconds:
            return False, "scale_data_gap"
        return True, "continuous_load"


class DuplicateReviewStore:
    """Durable HP-01 review inputs and emitted mock actions."""

    def __init__(self, database_path=DUPLICATE_REVIEW_DB):
        self.database_path = database_path
        self._lock = threading.Lock()

    def _connect(self):
        directory = os.path.dirname(self.database_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        connection = sqlite3.connect(self.database_path)
        connection.execute("PRAGMA synchronous=FULL")
        connection.executescript(SCHEMA)
        return connection

    def upsert_session(self, record):
        keys = json.dumps(list(record.get("image_object_keys") or []), ensure_ascii=False)
        values = (
            record.get("session_id"), record.get("started_at"), record.get("ended_at"),
            record.get("plate"), record.get("plate_status"),
            record.get("stable_weight_kg"), record.get("weight_source"),
            1 if record.get("published") else 0, record.get("outbox_event_id"),
            keys, _now_iso(),
        )
        with self._lock, closing(self._connect()) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO review_sessions (%s) VALUES (%s)"
                % (", ".join(COLUMNS), ", ".join("?" for _ in COLUMNS)),
                values,
            )
            connection.commit()

    def recent_sessions(self, since_iso):
        with self._lock, closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT %s FROM review_sessions WHERE ended_at >= ? ORDER BY started_at, session_id"
                % ", ".join(COLUMNS),
                (since_iso,),
            ).fetchall()
        sessions = []
        for row in rows:
            session = dict(zip(COLUMNS, row))
            session["published"] = bool(session["published"])
            session["image_object_keys"] = json.loads(session["image_object_keys"] or "[]")
            sessions.append(session)
        return sessions

    def has_action(self, action_key):
        with self._lock, closing(self._connect()) as connection:
            return connection.execute(
                "SELECT 1 FROM review_actions WHERE action_key = ?", (action_key,)
            ).fetchone() is not None

    def record_action(self, action_key, losing_session_id, surviving_session_id,
                      reason, mqtt_mock_path, minio_mock_path):
        with self._lock, closing(self._connect()) as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO review_actions "
                "(action_key, losing_session_id, surviving_session_id, reason, "
                "mqtt_mock_path, minio_mock_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    action_key, losing_session_id, surviving_session_id, reason,
                    mqtt_mock_path, minio_mock_path, _now_iso(),
                ),
            )
            connection.commit()
            return cursor.rowcount > 0


class DuplicateReviewer:
    """Group connected continuous-load sessions and emit mock removal intents."""

    def __init__(self, store=None, reader=None, mock_dir=DUPLICATE_REVIEW_MOCK_DIR,
                 reconcile_hours=DUPLICATE_REVIEW_RECONCILE_HOURS, log_fn=None):
        self.store = store or DuplicateReviewStore()
        self.reader = reader or ScaleContinuityReader()
        self.mock_dir = mock_dir
        self.reconcile_hours = float(reconcile_hours)
        self.log_fn = log_fn

    def _log(self, level, message, log_fn=None):
        writer = log_fn or self.log_fn
        if writer:
            writer(level, message)

    def register_session(self, record, log_fn=None):
        self.store.upsert_session(record)
        self.review_recent(log_fn)

    def reconcile(self, log_fn=None):
        self.review_recent(log_fn)

    def review_recent(self, log_fn=None):
        since = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(hours=self.reconcile_hours)
        sessions = self.store.recent_sessions(since.isoformat())
        for group in self._groups(sessions, log_fn):
            self._review_group(group, log_fn)

    def _groups(self, sessions, log_fn):
        if len(sessions) < 2:
            return []
        parent = list(range(len(sessions)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left, right):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for index in range(len(sessions) - 1):
            earlier, later = sessions[index], sessions[index + 1]
            start = _iso_to_epoch(earlier["ended_at"])
            end = _iso_to_epoch(later["started_at"])
            if start is None or end is None or end < start:
                continue
            continuous, reason = self.reader.continuous_load(start, end)
            if continuous:
                union(index, index + 1)
            else:
                self._log(
                    "MERGE",
                    "Duplicate review pair kept separate earlier=%s later=%s reason=%s"
                    % (earlier["session_id"], later["session_id"], reason),
                    log_fn,
                )
        grouped = {}
        for index, session in enumerate(sessions):
            grouped.setdefault(find(index), []).append(session)
        return [members for members in grouped.values() if len(members) > 1]

    @staticmethod
    def _rank(session):
        plate = session.get("plate") or ""
        valid_plate = 0 if not plate or plate.startswith("UNKNOWN") else 1
        has_images = 1 if session.get("image_object_keys") else 0
        reliable_weight = 1 if session.get("weight_source") == "stable" else 0
        published = 1 if session.get("published") else 0
        return (valid_plate, has_images, reliable_weight, published)

    def _survivor(self, group):
        ranked = sorted(
            group,
            key=lambda s: (
                tuple(-value for value in self._rank(s)),
                s.get("started_at") or "",
                s.get("session_id") or "",
            ),
        )
        return ranked[0]

    def _review_group(self, group, log_fn):
        weighted = (
            all(
                isinstance(s.get("stable_weight_kg"), (int, float))
                and s["stable_weight_kg"] > self.reader.weight_threshold
                for s in group
            )
        )
        if not weighted:
            return
        survivor = self._survivor(group)
        for member in group:
            if member["session_id"] == survivor["session_id"]:
                continue
            if not member["published"]:
                continue
            action_key = "%s->%s" % (member["session_id"], survivor["session_id"])
            if self.store.has_action(action_key):
                continue
            mqtt_intent = build_mqtt_compensation_intent(member, survivor)
            minio_intent = build_minio_removal_intent(member)
            try:
                mqtt_path, minio_path = emit_mock_actions(self.mock_dir, mqtt_intent, minio_intent)
            except OSError as exc:
                self._log(
                    "ERROR",
                    "Duplicate review mock write failed key=%s error=%s" % (action_key, exc),
                    log_fn,
                )
                continue
            self.store.record_action(
                action_key, member["session_id"], survivor["session_id"],
                "continuous_load", mqtt_path, minio_path,
            )
            self._log(
                "MERGE",
                "Duplicate review mock removal losing=%s surviving=%s weight=%skg"
                % (member["session_id"], survivor["session_id"], member.get("stable_weight_kg")),
                log_fn,
            )
            if log_fn:
                log_fn("METRIC", json.dumps(
                    {
                        "event": "duplicate_review_removal_intent",
                        "losing_session_id": member["session_id"],
                        "surviving_session_id": survivor["session_id"],
                        "reason": "continuous_load",
                        "image_objects": len(member.get("image_object_keys") or []),
                    },
                    separators=(",", ":"), sort_keys=True,
                ))