"""Raw scale-log data access and shared cycle-boundary helpers.

Pure data access plus small, policy-free helpers shared by duplicate review and
missed-session recovery. Callers keep their own validation and decision policy.
"""

import os
import sqlite3
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta

from config import SCALE_DATA_DIR, SESSION_END_EMPTY_DWELL_SECONDS, WEIGHT_THRESHOLD


def iso_to_epoch(value):
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return parsed.timestamp()


def _as_local_datetime(value):
    return datetime.fromisoformat(str(value))


class ScaleLogReader:
    """Read-only access to the authoritative per-day scale-log databases."""

    def __init__(self, data_dir=SCALE_DATA_DIR, weight_threshold=WEIGHT_THRESHOLD):
        self.data_dir = data_dir
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
            epoch = iso_to_epoch(row[0])
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

    def rows_between(self, local_start, local_end):
        """Strict row read over local datetimes: (timestamp, weight_kg, status, checksum_ok)."""
        rows = []
        day = local_start.date()
        while day <= local_end.date():
            database = self._db_path(day)
            if not os.path.exists(database):
                raise RuntimeError(f"raw scale database missing for {day.isoformat()}")
            with closing(sqlite3.connect(database)) as connection:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(weight_log)")
                }
                required = {"timestamp", "weight_kg", "status"}
                if not required.issubset(columns):
                    raise RuntimeError(f"invalid raw scale schema for {day.isoformat()}")
                checksum = "checksum_ok" if "checksum_ok" in columns else "1"
                rows.extend(connection.execute(
                    f"SELECT timestamp, weight_kg, status, {checksum} "
                    "FROM weight_log WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp",
                    (local_start.isoformat(), local_end.isoformat()),
                ).fetchall())
            day += timedelta(days=1)
        if not rows:
            raise RuntimeError("raw scale interval has no readings")
        if any(row[3] not in (None, 1) for row in rows):
            raise RuntimeError("raw scale interval contains invalid checksums")
        return rows


def group_loaded_cycles(rows, weight_threshold=WEIGHT_THRESHOLD, dwell_seconds=SESSION_END_EMPTY_DWELL_SECONDS):
    """Split ordered raw rows into above-threshold cycles separated by empty dwell."""
    groups = []
    current = []
    previous_loaded_at = None
    low_since_previous = []
    for row in rows:
        observed_at = _as_local_datetime(row[0])
        if float(row[1]) <= weight_threshold:
            if current:
                low_since_previous.append(row)
            continue
        if current and low_since_previous:
            gap = (observed_at - previous_loaded_at).total_seconds()
            if gap >= dwell_seconds:
                groups.append(current)
                current = []
        current.append(row)
        previous_loaded_at = observed_at
        low_since_previous = []
    if current:
        groups.append(current)
    return groups


def select_loaded_stable(rows, weight_threshold=WEIGHT_THRESHOLD):
    """Pick the most frequent loaded STABLE weight, ties broken by latest observation."""
    stable = [
        row for row in rows
        if float(row[1]) > weight_threshold and row[2] == "STABLE"
    ]
    if not stable:
        return None
    counts = Counter(float(row[1]) for row in stable)
    latest_seen = {}
    for index, row in enumerate(stable):
        latest_seen[float(row[1])] = index
    selected = max(counts, key=lambda value: (counts[value], latest_seen[value]))
    selected_rows = [row for row in stable if float(row[1]) == selected]
    return {
        "selected_weight": selected,
        "selected_weight_count": counts[selected],
        "selected_weight_observed_at": selected_rows[-1][0],
        "stable": stable,
        "counts": counts,
        "latest_seen": latest_seen,
    }