"""Replay recorded scale DB rows through the current visit lifecycle.

Reads a weight_log SQLite database (mode=ro), re-derives per-frame stability
exactly like the live reader, and drives a real SessionManager so visit
splitting (confirmed departure + renewed rise) can be evaluated against
recorded days.

Optionally matches replayed visit weights against a CAN HANG csv export
(one-to-one, pooled hosts). Matching is weight-only: it is an upper bound on
coverage, never a recall claim.

Usage:
    python3 scripts/replay_weight_sessions.py db1.sqlite [db2.sqlite ...]
        [--csv can_hang.csv] [--tolerance 50] [--json out.json]
"""

import argparse
import csv
import json
import sqlite3
import sys
import tempfile
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.modules.setdefault("serial", unittest.mock.Mock())
sys.modules.setdefault("cv2", unittest.mock.Mock())
sys.modules.setdefault("minio", unittest.mock.Mock())
sys.modules.setdefault("minio.error", unittest.mock.Mock())

from d2008_scale_reader import D2008Reader, WeightFrame  # noqa: E402
from services.session import session_manager as session_manager_module  # noqa: E402
from services.session.session_manager import SessionManager  # noqa: E402

VN_TZ = timezone(timedelta(hours=7))


class _RecordingSpool:
    """Collects session metadata instead of writing frame spools."""

    def __init__(self):
        self.begins = {}
        self.ended = []

    def begin_session(self, session_id, lpr_start_frames, metadata=None):
        self.begins[session_id] = dict(metadata or {})
        return "/tmp/replay-unused"

    def update_active_metadata(self, session_id, metadata):
        self.begins.setdefault(session_id, {}).update(metadata)

    def end_session(self, session_id, metadata):
        merged = self.begins.setdefault(session_id, {})
        merged.update(metadata)
        self.ended.append(merged)
        return True

    def save_session_frame(self, *_args, **_kwargs):
        return "/tmp/replay-unused.jpg"

    def abort_session(self, _session_id):
        pass


class _ReplayClock:
    """Dwell time driven by recorded timestamps instead of wall clock."""

    def __init__(self):
        self.mono = 0.0
        self.wall = 0.0
        self.now_dt = None

    def set(self, timestamp):
        self.mono = timestamp.timestamp()
        self.wall = timestamp.timestamp()
        self.now_dt = timestamp

    def monotonic(self):
        return self.mono

    def time(self):
        return self.wall


def _replay_datetime_factory(clock):
    class _ReplayDateTime(datetime):
        @classmethod
        def now(cls, tz=None):  # noqa: N805
            value = clock.now_dt
            return value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)

    return _ReplayDateTime


def _parse_timestamp(raw):
    value = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S.%f")
    return value.replace(tzinfo=VN_TZ)


def read_rows(db_path):
    uri = "file:%s?mode=ro" % Path(db_path).resolve().as_posix()
    with sqlite3.connect(uri, uri=True) as conn:
        cursor = conn.execute(
            "SELECT timestamp, weight_kg, decimal_pos FROM weight_log "
            "WHERE checksum_ok = 1 ORDER BY id"
        )
        rows = cursor.fetchall()
    return [(_parse_timestamp(ts), weight, decimal_pos) for ts, weight, decimal_pos in rows]


def replay_db(db_path, host=None):
    """Replay one day DB; returns the list of ended session metadata dicts."""
    spool = _RecordingSpool()
    manager = SessionManager(
        unittest.mock.Mock(), frame_spool=spool, lpr_grabbers={},
    )
    rows = read_rows(db_path)
    clock = _ReplayClock()
    if rows:
        clock.set(rows[0][0])
    original_time = session_manager_module.time
    original_datetime = session_manager_module.datetime
    session_manager_module.time = clock
    session_manager_module.datetime = _replay_datetime_factory(clock)
    with tempfile.NamedTemporaryFile(suffix=".db") as stability_db:
        reader = D2008Reader(db_file=stability_db.name)
        try:
            for timestamp, weight, decimal_pos in rows:
                clock.set(timestamp)
                frame = WeightFrame(
                    b"", "+", weight, decimal_pos or 0, True,
                    "%06d" % int(max(0, weight)),
                    timestamp=timestamp,
                )
                frame.status = reader._get_status(frame)
                manager.on_frame(frame, lambda *_args: None)
        finally:
            reader._db.close()
            session_manager_module.time = original_time
            session_manager_module.datetime = original_datetime
    sessions = []
    for metadata in spool.ended:
        metadata = dict(metadata)
        metadata["host"] = host or str(db_path)
        sessions.append(metadata)
    return sessions


def parse_can_hang(csv_path):
    """Extract (gross_kg, tare_kg) pairs per vehicle block from a CAN HANG csv.

    Row 5 is the header naming blocks ("Xe 54 Tổng", ...); data rows follow
    until the totals section. Dots are thousands separators. A pair exists
    only when both gross and tare are populated.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header = rows[4]
    blocks = []
    for index, name in enumerate(header):
        if name and name.startswith("Xe") and "Tổng" in name:
            vehicle = name.split()[1]
            blocks.append((vehicle, index))
    pairs = []
    for row in rows[5:]:
        if not row or not str(row[0]).strip().isdigit():
            break
        for _vehicle, index in blocks:
            gross = _cell_to_kg(row, index)
            tare = _cell_to_kg(row, index + 1)
            if gross is not None and tare is not None:
                pairs.append((gross, tare))
    return pairs


def _cell_to_kg(row, index):
    if index >= len(row):
        return None
    cell = str(row[index]).strip()
    if not cell or cell == "0":
        return None
    try:
        return float(cell.replace(".", ""))
    except ValueError:
        return None


def match_targets(weights, targets, tolerance):
    """Greedy one-to-one match; each weight and target used at most once."""
    edges = []
    for weight_index, weight in enumerate(weights):
        for target_index, target in enumerate(targets):
            if abs(weight - target) <= tolerance:
                edges.append((abs(weight - target), weight_index, target_index))
    edges.sort()
    used_weights = set()
    used_targets = set()
    matches = 0
    for _delta, weight_index, target_index in edges:
        if weight_index in used_weights or target_index in used_targets:
            continue
        used_weights.add(weight_index)
        used_targets.add(target_index)
        matches += 1
    return matches


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("databases", nargs="+", help="weight_log sqlite files")
    parser.add_argument("--csv", help="CAN HANG csv to match against")
    parser.add_argument("--tolerance", type=float, default=50.0)
    parser.add_argument("--json", help="write full session list as JSON")
    args = parser.parse_args(argv)

    sessions = []
    for db_path in args.databases:
        host = Path(db_path).parent.name or str(db_path)
        sessions.extend(replay_db(db_path, host=host))

    weights = [s.get("stable_weight") or 0.0 for s in sessions]
    reasons = {}
    for session in sessions:
        reasons[session["end_reason"]] = reasons.get(session["end_reason"], 0) + 1

    summary = {
        "databases": args.databases,
        "sessions": len(sessions),
        "end_reasons": reasons,
        "loaded_sessions": sum(
            1 for w in weights if w > 10000
        ),
    }

    if args.csv:
        pairs = parse_can_hang(args.csv)
        targets = [value for pair in pairs for value in pair]
        summary["csv_pairs"] = len(pairs)
        summary["csv_targets"] = len(targets)
        summary["matched_exact"] = match_targets(weights, targets, 0.5)
        summary["matched_within_tolerance"] = match_targets(
            weights, targets, args.tolerance
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(
            json.dumps(sessions, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
