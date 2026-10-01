#!/usr/bin/env python3
"""Export finalized weighbridge sessions to an Excel (.xlsx) workbook.

Reads session-finalization.db (finalized_sessions + terminal_outcomes) and
writes one sheet per machine by default, or a single combined sheet with a
`machine` column when --combined is passed.

Usage:
  python3 scripts/export_sessions.py --db storage/session-finalization.db \
      --machine hp1 --date 2026-10-01 --out sessions_hp1_2026-10-01.xlsx

  # combine two hosts on one sheet
  python3 scripts/export_sessions.py \
      --db hp1=hp1.db --db hp2=hp2.db --combined --out sessions.xlsx

Filters use the session START timestamp (record_json.started_at), falling back
to finalized_at when a record has no started_at. All times are UTC ISO.
"""
import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone

# Minimal columns per user feedback: timestamp + plate + weight are the key
# values; the rest are kept small but useful for sorting/filtering.
COLUMNS = [
    ("machine", "Machine"),
    ("started_at", "Start (UTC)"),
    ("ended_at", "End (UTC)"),
    ("plate", "Plate"),
    ("stable_weight_kg", "Weight (kg)"),
    ("plate_status", "Plate status"),
    ("outcome", "Outcome"),
    ("unknown_type", "Unknown type"),
    ("weight_source", "Weight source"),
    ("session_id", "Session ID"),
    ("end_reason", "End reason"),
    ("duration_s", "Duration (s)"),
]

# Slim layout: start split into local day + time for shift-style review.
COMPACT_COLUMNS = [
    ("machine", "Machine"),
    ("start_day", "Day (Start)"),
    ("start_time", "Time (Start)"),
    ("plate", "Plate"),
    ("stable_weight_kg", "Weight (kg)"),
    ("duration", "Duration"),
]

LOCAL_TZ = timezone(timedelta(hours=7))  # production hosts are UTC+7


def _to_iso(value):
    """Normalize stored timestamp to a sortable ISO string."""
    if value is None:
        return ""
    return str(value)


def _rows_from_db(db_path, machine):
    con = sqlite3.connect("file:%s?mode=ro" % db_path, uri=True)
    try:
        cur = con.execute(
            "SELECT f.session_id, f.outcome, f.finalized_at, t.record_json "
            "FROM finalized_sessions f "
            "LEFT JOIN terminal_outcomes t ON t.session_id = f.session_id"
        )
        for session_id, outcome, finalized_at, record_json in cur:
            rec = {}
            if record_json:
                try:
                    rec = json.loads(record_json)
                except json.JSONDecodeError:
                    rec = {}
            yield {
                "machine": machine,
                "session_id": session_id,
                "outcome": outcome,
                "finalized_at": finalized_at,
                "record": rec,
            }
    finally:
        con.close()


def _in_window(rec, finalized_at, date, start, end):
    ts = _to_iso(rec.get("started_at") or rec.get("ended_at") or finalized_at)
    if date and not ts.startswith(date):
        return False
    if start and ts < start:
        return False
    if end and ts > end:
        return False
    return True


def _record_to_row(entry):
    rec = entry["record"]
    stable = rec.get("stable_weight_kg", rec.get("stable_weight"))
    started = _to_iso(rec.get("started_at") or rec.get("session_start"))
    duration_s = rec.get("duration_s")
    row = {
        "machine": entry["machine"],
        "started_at": started,
        "ended_at": _to_iso(rec.get("ended_at") or rec.get("session_end")),
        "plate": rec.get("plate") or "",
        "stable_weight_kg": stable if stable is not None else "",
        "plate_status": rec.get("plate_status") or "",
        "outcome": entry["outcome"],
        "unknown_type": rec.get("unknown_type") or "",
        "weight_source": rec.get("weight_source") or "",
        "session_id": rec.get("id") or rec.get("session_id") or entry["session_id"],
        "end_reason": rec.get("end_reason") or "",
        "duration_s": duration_s if duration_s is not None else "",
    }
    # compact fields
    day = ""
    clock = ""
    if started:
        try:
            ts = started.replace("Z", "+00:00")
            local = datetime.fromisoformat(ts)
            if local.tzinfo is None:
                local = local.replace(tzinfo=timezone.utc)
            local = local.astimezone(LOCAL_TZ)
            day = local.strftime("%d/%m")
            clock = local.strftime("%H:%M:%S")
        except ValueError:
            day = started[:10]
            clock = started[11:19]
    row["start_day"] = day
    row["start_time"] = clock
    row["duration"] = "" if duration_s is None else str(timedelta(seconds=int(duration_s)))
    return row


def export(dbs, out_path, combined, date, start, end, layout):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
        from openpyxl.utils import get_column_letter
    except ImportError:
        sys.exit(
            "openpyxl is required: pip install openpyxl"
        )

    wb = Workbook()
    wb.remove(wb.active)
    if layout == "compact":
        cols = COMPACT_COLUMNS
    else:
        cols = COLUMNS
        if not combined:
            cols = cols[1:]  # drop Machine column when one sheet per host
    headers = [h for _, h in cols]

    def write_sheet(title, rows):
        ws = wb.create_sheet(title)
        ws.append(headers)
        for c in ws[1]:
            c.font = Font(bold=True)
        ws.freeze_panes = "A2"
        for r in rows:
            ws.append([r.get(k, "") for k, _ in cols])
        # light autosize
        for idx, (k, _) in enumerate(cols, start=1):
            width = max(len(str(cols[idx-1][1])), max((len(str(r.get(k, ""))) for r in rows), default=10)) + 2
            ws.column_dimensions[get_column_letter(idx)].width = min(width, 60)
        ws.auto_filter.ref = "A1:%s%d" % (get_column_letter(len(cols)), len(rows) + 1)

    if combined:
        all_rows = []
        for machine, db_path in dbs:
            for e in _rows_from_db(db_path, machine):
                if _in_window(e["record"], e["finalized_at"], date, start, end):
                    all_rows.append(_record_to_row(e))
        all_rows.sort(key=lambda r: (r["started_at"], r["machine"]))
        write_sheet("sessions", all_rows)
        print("combined sheet: %d rows across %d machine(s)" % (len(all_rows), len(dbs)))
    else:
        for machine, db_path in dbs:
            rows = [
                _record_to_row(e)
                for e in _rows_from_db(db_path, machine)
                if _in_window(e["record"], e["finalized_at"], date, start, end)
            ]
            rows.sort(key=lambda r: r["started_at"])
            write_sheet(machine, rows)
            print("sheet %-8s %d rows" % (machine, len(rows)))

    wb.save(out_path)
    print("wrote %s" % out_path)


def _parse_db_arg(values):
    """Each --db is either PATH or NAME=PATH."""
    out = []
    for v in values:
        if "=" in v:
            name, path = v.split("=", 1)
            out.append((name, path))
        else:
            # derive a short machine name from the path
            base = v.rstrip("/").rsplit("/", 1)[-1]
            name = base.replace("session-finalization.db", "").strip("_-.") or "db"
            out.append((name, v))
    return out


def main():
    p = argparse.ArgumentParser(description="Export finalized weighbridge sessions to Excel.")
    p.add_argument("--db", action="append", required=True,
                   help="db path, or NAME=path. Repeat per machine.")
    p.add_argument("--combined", action="store_true",
                   help="put all machines on one sheet with a Machine column")
    p.add_argument("--date", help="keep only sessions started on this UTC date YYYY-MM-DD")
    p.add_argument("--from", dest="start", help="keep sessions started >= this ISO timestamp")
    p.add_argument("--to", dest="end", help="keep sessions started <= this ISO timestamp")
    p.add_argument("--out", required=True, help="output .xlsx path")
    p.add_argument("--layout", choices=("full", "compact"), default="full",
                   help="full keeps all columns; compact = Machine, Day(Start), "
                        "Time(Start), Plate, Weight(kg), Duration")
    a = p.parse_args()

    dbs = _parse_db_arg(a.db)
    export(dbs, a.out, a.combined, a.date, a.start, a.end, a.layout)


if __name__ == "__main__":
    main()
