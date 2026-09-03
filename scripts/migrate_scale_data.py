#!/usr/bin/env python3
"""Partition legacy root scale_data.db into daily SQLite databases."""

import argparse
import fcntl
import os
import shutil
import sqlite3
import sys
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCE_NAME = "scale_data.db"
ARCHIVE_NAME = "scale_data.archive.db"
WEIGHT_LOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS weight_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT    NOT NULL,
    weight_kg   REAL    NOT NULL,
    sign        TEXT    NOT NULL,
    decimal_pos INTEGER NOT NULL,
    checksum_ok INTEGER NOT NULL,
    status      TEXT    NOT NULL DEFAULT 'UNSTABLE'
)
"""


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root.")
    return parser.parse_args()


def read_rows(conn):
    columns = [row[1] for row in conn.execute("PRAGMA table_info(weight_log)")]
    required = ["id", "timestamp", "weight_kg", "sign", "decimal_pos", "checksum_ok", "status"]
    if columns != required:
        raise RuntimeError(f"Unexpected weight_log schema: {columns}")
    rows = conn.execute(
        "SELECT id, timestamp, weight_kg, sign, decimal_pos, checksum_ok, status "
        "FROM weight_log ORDER BY id"
    ).fetchall()
    dates = Counter()
    for row in rows:
        if len(row[1]) < 10:
            raise RuntimeError(f"Invalid weight timestamp for id={row[0]}: {row[1]!r}")
        dates[row[1][:10]] += 1
    return rows, dates


def migrate(source, destination_dir, source_conn=None):
    if source_conn is None:
        with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as conn:
            return migrate(source, destination_dir, conn)
    rows, source_counts = read_rows(source_conn)
    for date_text in source_counts:
        target = destination_dir / f"{date_text}.db"
        if target.exists():
            raise RuntimeError(f"Target already exists: {target}")

    for date_text in source_counts:
        target = destination_dir / f"{date_text}.db"
        target_rows = [row for row in rows if row[1].startswith(date_text)]
        with sqlite3.connect(target) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(WEIGHT_LOG_SCHEMA)
            conn.executemany(
                "INSERT INTO weight_log "
                "(id, timestamp, weight_kg, sign, decimal_pos, checksum_ok, status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                target_rows,
            )
            conn.commit()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
            count, low, high = conn.execute(
                "SELECT count(*), min(timestamp), max(timestamp) FROM weight_log"
            ).fetchone()
            if integrity != "ok" or count != len(target_rows) or low != target_rows[0][1] or high != target_rows[-1][1]:
                raise RuntimeError(f"Validation failed for {target}")

    target_counts = Counter()
    for target in destination_dir.glob("????-??-??.db"):
        with sqlite3.connect(target) as conn:
            count = conn.execute("SELECT count(*) FROM weight_log").fetchone()[0]
        target_counts[target.stem] = count
    if target_counts != source_counts:
        raise RuntimeError(f"Row count mismatch source={source_counts} target={target_counts}")


def require_exclusive_source(source):
    try:
        result = subprocess.run(
            ["fuser", str(source)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("fuser is required to verify scale_data.db is not in use") from exc
    if result.returncode == 0:
        raise RuntimeError("Stop the service before migrating scale_data.db")
    if result.returncode != 1:
        raise RuntimeError("Unable to verify scale_data.db ownership")


def acquire_migration_lock(destination_dir):
    lock_path = destination_dir / ".scale_data.lock"
    lock = open(lock_path, "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("Stop the service before migrating scale_data.db") from exc
    return lock


def main():
    args = parse_args()
    source = args.root / SOURCE_NAME
    destination_dir = args.root / "scale_data"
    archive = destination_dir / ARCHIVE_NAME

    if not source.exists():
        raise SystemExit(f"Legacy database not found: {source}")
    if archive.exists():
        raise SystemExit(f"Archive already exists: {archive}")

    destination_dir.mkdir(exist_ok=True)
    existing = list(destination_dir.glob("????-??-??.db*"))
    if existing:
        raise SystemExit(f"Daily scale databases already exist: {existing[0]}")
    lock = acquire_migration_lock(destination_dir)
    try:
        try:
            require_exclusive_source(source)
            with sqlite3.connect(source) as conn:
                conn.execute("PRAGMA busy_timeout=1000")
                if conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] != 0:
                    raise RuntimeError("Stop the service before migrating scale_data.db")
                conn.execute("BEGIN EXCLUSIVE")
                migrate(source, destination_dir, conn)
            require_exclusive_source(source)
            if any(Path(str(source) + suffix).exists() for suffix in ("-wal", "-shm")):
                raise RuntimeError("Stop the service before archiving scale_data.db")
            shutil.move(source, archive)
            for suffix in ("-wal", "-shm"):
                Path(str(source) + suffix).unlink(missing_ok=True)
        except Exception:
            for path in destination_dir.glob("????-??-??.db*"):
                path.unlink(missing_ok=True)
            raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    print(f"Migrated legacy scale database to {destination_dir} and archived source at {archive}")


if __name__ == "__main__":
    main()
