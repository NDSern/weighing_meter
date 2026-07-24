import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone


SCHEMA = (
    "CREATE TABLE IF NOT EXISTS finalized_sessions "
    "(session_id TEXT PRIMARY KEY, outcome TEXT NOT NULL, finalized_at TEXT NOT NULL)"
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


def mark(database_path, session_id, outcome):
    os.makedirs(os.path.dirname(database_path), exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute(SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO finalized_sessions "
            "(session_id, outcome, finalized_at) VALUES (?, ?, ?)",
            (session_id, outcome, datetime.now(timezone.utc).isoformat()),
        )
        connection.commit()
