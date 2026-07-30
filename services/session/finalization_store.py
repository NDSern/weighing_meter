import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone


SCHEMA = (
    "CREATE TABLE IF NOT EXISTS finalized_sessions "
    "(session_id TEXT PRIMARY KEY, outcome TEXT NOT NULL, finalized_at TEXT NOT NULL)"
)
OUTCOME_SCHEMA = (
    "CREATE TABLE IF NOT EXISTS terminal_outcomes "
    "(session_id TEXT PRIMARY KEY, event_type TEXT NOT NULL, "
    "record_json TEXT NOT NULL, finalized_at TEXT NOT NULL)"
)


def contains(database_path, session_id):
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(SCHEMA)
            return connection.execute(
                "SELECT 1 FROM finalized_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone() is not None
    except sqlite3.Error:
        return False


def get(database_path, session_id):
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            connection.execute(SCHEMA)
            connection.execute(OUTCOME_SCHEMA)
            row = connection.execute(
                "SELECT f.outcome, t.record_json "
                "FROM finalized_sessions f LEFT JOIN terminal_outcomes t "
                "ON t.session_id = f.session_id WHERE f.session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return row[0], json.loads(row[1]) if row[1] is not None else None
    except (sqlite3.Error, json.JSONDecodeError):
        return None


def mark(database_path, session_id, outcome, record=None):
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(SCHEMA)
        connection.execute(OUTCOME_SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO finalized_sessions "
            "(session_id, outcome, finalized_at) VALUES (?, ?, ?)",
            (session_id, outcome, datetime.now(timezone.utc).isoformat()),
        )
        if record is not None:
            connection.execute(
                "INSERT OR REPLACE INTO terminal_outcomes "
                "(session_id, event_type, record_json, finalized_at) VALUES (?, ?, ?, ?)",
                (
                    session_id, record["event"],
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        connection.commit()
