import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime


_lock = threading.Lock()


def increment(database_path, license_plate, session_id=None):
    if not license_plate or license_plate == "none":
        return None

    now = datetime.now().isoformat(timespec="seconds")
    with _lock, closing(sqlite3.connect(database_path)) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS confirmed_license_plates (
                license_plate TEXT PRIMARY KEY,
                recognition_count INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS confirmed_plate_sessions (
                session_id TEXT PRIMARY KEY,
                license_plate TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        if session_id:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO confirmed_plate_sessions "
                "(session_id, license_plate, created_at) VALUES (?, ?, ?)",
                (session_id, license_plate, now),
            )
            if cursor.rowcount == 0:
                row = connection.execute(
                    "SELECT recognition_count FROM confirmed_license_plates "
                    "WHERE license_plate = ?",
                    (license_plate,),
                ).fetchone()
                connection.commit()
                return row[0] if row else None
        connection.execute(
            """
            INSERT INTO confirmed_license_plates (
                license_plate, recognition_count, first_seen_at, last_seen_at
            )
            VALUES (?, 1, ?, ?)
            ON CONFLICT(license_plate) DO UPDATE SET
                recognition_count = recognition_count + 1,
                last_seen_at = excluded.last_seen_at
            """,
            (license_plate, now, now),
        )
        connection.commit()
        row = connection.execute(
            "SELECT recognition_count FROM confirmed_license_plates "
            "WHERE license_plate = ?",
            (license_plate,),
        ).fetchone()
        return row[0] if row else None
