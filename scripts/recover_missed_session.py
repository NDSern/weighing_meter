#!/usr/bin/env python3
"""Validate and stage audited missed weighbridge events for outbox replay."""

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CAPTURE_DIR,
    MINIO_BUCKET,
    NO_STABLE_DIR,
    SCALE_DATA_DIR,
    SERVICE_DIR,
    SESSION_END_EMPTY_DWELL_SECONDS,
    SESSION_FINALIZATION_DB,
    WEIGHBRIDGE_ID,
    WEIGHT_THRESHOLD,
)


HP1_ID = "100ecc11-dbcb-4c23-8e89-d41ccefcda37"
HP2_ID = "9aa29a10-6605-47dd-9460-970d66c3d1c3"
HP2_CASE_ID = "9c3dfb52707d4344a7baee4fdaf6ffed"

RECOVERY_CASES = {
    "c9f95576942341a28ee7e93812f26061": {
        "host_id": HP1_ID,
        "expected_outcome": "no_weight",
        "started_at": "2026-09-28T09:08:53.950+00:00",
        "ended_at": "2026-09-28T09:09:51.281+00:00",
        "stable_weight": 30630.0,
        "raw_peak_weight": 30640.0,
        "plate": "UNKNOWN_DETECTION",
        "evidence_camera": "cam3",
    },
    "7679222acf0c4bce926192bc96636c3d": {
        "host_id": HP1_ID,
        "expected_outcome": "no_weight",
        "started_at": "2026-09-28T10:06:12.700+00:00",
        "ended_at": "2026-09-28T10:09:37.655+00:00",
        "stable_weight": 105960.0,
        "raw_peak_weight": 106040.0,
        "plate": "UNKNOWN_DETECTION",
        "evidence_camera": "cam3",
    },
    "9c3dfb52707d4344a7baee4fdaf6ffed": {
        "host_id": HP2_ID,
        "expected_outcome": "duplicate",
        "started_at": "2026-09-27T07:40:24.276+00:00",
        "ended_at": "2026-09-27T07:40:47.526+00:00",
        "stable_weight": 17710.0,
        "raw_peak_weight": 17940.0,
        "plate": "14C-017.80",
        "evidence_camera": None,
        "allow_no_image": True,
    },
}


def _parse_timestamp(value):
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _local_timestamp(value):
    parsed = _parse_timestamp(value)
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone().replace(tzinfo=None)


def _json_hash(value):
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_lines(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL {path}:{line_number}: {exc}") from exc
    return records


def _fsync_directory(path):
    directory_fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_json_lines(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".recovery.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)
    _fsync_directory(path.parent)


def _append_json_line(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


class MissedSessionRecovery:
    def __init__(self, service_dir=SERVICE_DIR, host_id=WEIGHBRIDGE_ID):
        self.service_dir = Path(service_dir).resolve()
        self.host_id = host_id
        self.scale_dir = self._service_path(SCALE_DATA_DIR, "scale_data")
        self.capture_dir = self._service_path(CAPTURE_DIR, "storage/weighbridge")
        self.no_stable_dir = self._service_path(NO_STABLE_DIR, "storage/no-stable")
        self.finalization_db = self._service_path(
            SESSION_FINALIZATION_DB, "storage/session-finalization.db",
        )
        self.pending_file = self.service_dir / "storage" / "publish_pending.jsonl"
        self.completed_db = self.service_dir / "storage" / "publish_completed.db"
        self.upload_pending_file = self.service_dir / "storage" / "upload_pending.jsonl"
        self.audit_file = self.service_dir / "storage" / "recovery-audit.jsonl"

    def _service_path(self, configured, relative):
        configured_path = Path(configured)
        if configured_path.is_absolute() and self.service_dir == Path(SERVICE_DIR).resolve():
            return configured_path
        return self.service_dir / relative

    def _terminal_record(self, session_id, expected_outcome):
        if not self.finalization_db.exists():
            raise RuntimeError("session finalization database is missing")
        with closing(sqlite3.connect(self.finalization_db)) as connection:
            row = connection.execute(
                "SELECT f.outcome, t.record_json "
                "FROM finalized_sessions f "
                "LEFT JOIN terminal_outcomes t ON t.session_id = f.session_id "
                "WHERE f.session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            raise RuntimeError("session is absent from finalization ledger")
        outcome, record_json = row
        if outcome != expected_outcome:
            raise RuntimeError(
                f"unexpected final outcome {outcome!r}; expected {expected_outcome!r}"
            )
        if not record_json:
            raise RuntimeError("terminal record is missing")
        record = json.loads(record_json)
        if record.get("id") != session_id:
            raise RuntimeError("terminal record session ID mismatch")
        return record

    def _diagnostic_record(self, session_id, terminal):
        started_at = _parse_timestamp(terminal["started_at"])
        local_day = started_at.astimezone().strftime("%Y/%m/%d")
        path = self.no_stable_dir / local_day / f"{session_id}.json"
        if not path.exists():
            return terminal, None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("session_id") != session_id:
            raise RuntimeError("diagnostic metadata session ID mismatch")
        return record, path

    def _scale_rows_between(self, local_start, local_end):
        rows = []
        day = local_start.date()
        while day <= local_end.date():
            database = self.scale_dir / f"{day.isoformat()}.db"
            if not database.exists():
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

    def _scale_rows(self, started_at, ended_at):
        return self._scale_rows_between(
            _local_timestamp(started_at), _local_timestamp(ended_at),
        )

    @staticmethod
    def _as_local_datetime(value):
        return datetime.fromisoformat(str(value))

    def _confirmed_cycle_boundaries(self, metadata):
        local_start = _local_timestamp(metadata["started_at"])
        local_end = _local_timestamp(metadata["ended_at"])
        rows = self._scale_rows_between(
            local_start - timedelta(seconds=30),
            local_end + timedelta(seconds=5),
        )
        groups = []
        current = []
        previous_loaded_at = None
        low_since_previous = []
        for row in rows:
            observed_at = self._as_local_datetime(row[0])
            if float(row[1]) <= WEIGHT_THRESHOLD:
                if current:
                    low_since_previous.append(row)
                continue
            if current and low_since_previous:
                gap = (observed_at - previous_loaded_at).total_seconds()
                if gap >= SESSION_END_EMPTY_DWELL_SECONDS:
                    groups.append(current)
                    current = []
            current.append(row)
            previous_loaded_at = observed_at
            low_since_previous = []
        if current:
            groups.append(current)
        overlapping = [
            group for group in groups
            if any(
                local_start <= self._as_local_datetime(row[0]) <= local_end
                for row in group
            )
        ]
        if len(overlapping) != 1:
            raise RuntimeError(
                f"expected one raw scale cycle in session interval; found {len(overlapping)}"
            )
        cycle = overlapping[0]
        first_loaded_at = self._as_local_datetime(cycle[0][0])
        last_loaded_at = self._as_local_datetime(cycle[-1][0])
        before = [row for row in rows if self._as_local_datetime(row[0]) < first_loaded_at]
        after = [
            row for row in rows
            if last_loaded_at < self._as_local_datetime(row[0]) <= local_end
        ]
        if not before or float(before[-1][1]) > WEIGHT_THRESHOLD:
            raise RuntimeError("raw scale does not prove empty before recovered cycle")
        if not after or any(float(row[1]) > WEIGHT_THRESHOLD for row in after):
            raise RuntimeError("raw scale does not stay empty through session end")
        if (local_end - last_loaded_at).total_seconds() < SESSION_END_EMPTY_DWELL_SECONDS:
            raise RuntimeError("raw scale does not prove configured empty dwell at session end")
        previous_loaded = next(
            (row for row in reversed(before) if float(row[1]) > WEIGHT_THRESHOLD),
            None,
        )
        if previous_loaded is not None:
            previous_loaded_at = self._as_local_datetime(previous_loaded[0])
            if (
                first_loaded_at - previous_loaded_at
            ).total_seconds() < SESSION_END_EMPTY_DWELL_SECONDS:
                raise RuntimeError("raw scale does not separate recovered cycle from prior load")
        return {
            "cycle_first_loaded_at": cycle[0][0],
            "cycle_last_loaded_at": cycle[-1][0],
            "confirmed_empty_before": True,
            "confirmed_empty_at_end": True,
        }

    def _scale_evidence(self, metadata):
        rows = self._scale_rows(metadata["started_at"], metadata["ended_at"])
        loaded = [row for row in rows if float(row[1]) > WEIGHT_THRESHOLD]
        stable = [row for row in loaded if row[2] == "STABLE"]
        if not loaded:
            raise RuntimeError("raw scale interval contains no loaded readings")
        if not stable:
            raise RuntimeError("raw scale interval contains no loaded STABLE readings")
        stable_counts = Counter(float(row[1]) for row in stable)
        latest_seen = {}
        for index, row in enumerate(stable):
            latest_seen[float(row[1])] = index
        selected_weight = max(
            stable_counts,
            key=lambda value: (stable_counts[value], latest_seen[value]),
        )
        selected_rows = [row for row in stable if float(row[1]) == selected_weight]
        evidence = {
            "row_count": len(rows),
            "loaded_row_count": len(loaded),
            "loaded_stable_row_count": len(stable),
            "raw_peak_weight": max(float(row[1]) for row in loaded),
            "selected_weight": selected_weight,
            "selected_weight_count": stable_counts[selected_weight],
            "selected_weight_observed_at": selected_rows[-1][0],
            "first_loaded_at": loaded[0][0],
            "last_loaded_at": loaded[-1][0],
        }
        evidence.update(self._confirmed_cycle_boundaries(metadata))
        return evidence

    def _completed(self, session_id):
        if not self.completed_db.exists():
            return False
        with closing(sqlite3.connect(self.completed_db)) as connection:
            try:
                row = connection.execute(
                    "SELECT 1 FROM completed_events WHERE event_id = ?", (session_id,)
                ).fetchone()
            except sqlite3.Error as exc:
                raise RuntimeError(f"invalid publish completion database: {exc}") from exc
        return row is not None

    def _pending(self, session_id):
        return next(
            (record for record in _read_json_lines(self.pending_file)
             if record.get("id") == session_id),
            None,
        )

    def _upload_pending(self, object_key):
        return next(
            (record for record in _read_json_lines(self.upload_pending_file)
             if record.get("object_key") == object_key),
            None,
        )

    def _evidence(self, session_id, case, metadata):
        camera = case["evidence_camera"]
        if not camera:
            if case.get("allow_no_image"):
                return None, None, None
            raise RuntimeError("blocked: no attributable local image for this audited session")
        images = [Path(path) for path in metadata.get("images") or []]
        suffix = f"{session_id}_{camera}.jpg"
        source = next((path for path in images if path.name == suffix), None)
        if source is None:
            candidate = self.no_stable_dir / _parse_timestamp(
                metadata["started_at"]
            ).astimezone().strftime("%Y/%m/%d") / suffix
            source = candidate if candidate.exists() else None
        if source is None or not source.exists() or not source.is_file() or source.is_symlink():
            raise RuntimeError(f"attributable {camera} evidence image is missing")
        resolved = source.resolve()
        if os.path.commonpath((str(self.no_stable_dir.resolve()), str(resolved))) != str(
            self.no_stable_dir.resolve()
        ):
            raise RuntimeError("evidence image escapes diagnostic root")
        if resolved.stat().st_size <= 0:
            raise RuntimeError("evidence image is empty")
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
        return resolved, camera, digest

    def _build_event(self, session_id, case, metadata, scale, source, camera):
        event_time = _parse_timestamp(metadata["ended_at"])
        local_time = event_time.astimezone()
        date_path = local_time.strftime("%Y/%m/%d")
        image_less = source is None or camera is None
        if image_less:
            destination = None
            object_key = None
            photos = []
        else:
            filename = f"{session_id}_{case['plate']}_photo-{camera}.jpg"
            destination = self.capture_dir / local_time.strftime(
                "%Y"
            ) / local_time.strftime("%m") / local_time.strftime("%d") / filename
            object_key = f"storage/weighbridge/{date_path}/{filename}"
            photos = [{
                "url": f"/storage/weighbridge/{date_path}/{filename}",
                "type": camera,
                "captured_at": metadata.get("filtered_peak_observed_at")
                or metadata.get("raw_peak_observed_at")
                or metadata.get("ended_at"),
            }]
        session_result = {
            "start": _parse_timestamp(metadata["started_at"]).astimezone().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "end": event_time.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": metadata["ended_at"],
            "duration_s": metadata.get("duration_s", 0),
            "stable_weight": scale["selected_weight"],
            "official_plate": case["plate"],
            "official_plate_count": 0,
            "all_plates": {},
            "image_path": None,
            "offline_event_id": session_id,
            "ocr_plate_read": None if case["plate"].startswith("UNKNOWN") else case["plate"],
            "photos": photos,
            "metadata": {
                "plate_status": "unreadable" if case["plate"].startswith("UNKNOWN") else "confirmed",
                "recovery": True,
                "recovery_reason": "audited_missed_session",
                "image_less": image_less,
                "original_outcome": case["expected_outcome"],
                "weight_source": "raw_scale_stable_mode",
                "loaded_stable_samples": scale["loaded_stable_row_count"],
                "selected_weight_samples": scale["selected_weight_count"],
                "raw_peak_weight": scale["raw_peak_weight"],
            },
        }
        outbox = {
            "id": session_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_object_keys": [object_key] if object_key else [],
            "image_paths": [str(destination)] if destination else [],
            "session_result": session_result,
            "activated": True,
        }
        return destination, object_key, outbox

    def plan(self, session_id):
        case = RECOVERY_CASES.get(session_id)
        if case is None:
            raise RuntimeError("session is not in the audited recovery allowlist")
        if self.host_id != case["host_id"]:
            raise RuntimeError("session allowlist does not match this weighbridge")
        terminal = self._terminal_record(session_id, case["expected_outcome"])
        metadata, metadata_path = self._diagnostic_record(session_id, terminal)
        for field in ("started_at", "ended_at"):
            if metadata.get(field) != case[field]:
                raise RuntimeError(f"audited {field} does not match allowlist")
        if metadata.get("end_reason") != "scale_empty":
            raise RuntimeError("session did not end after confirmed scale empty")
        scale = self._scale_evidence(metadata)
        if scale["selected_weight"] != case["stable_weight"]:
            raise RuntimeError("raw stable weight does not match audited allowlist")
        if scale["raw_peak_weight"] != case["raw_peak_weight"]:
            raise RuntimeError("raw peak weight does not match audited allowlist")
        source, camera, source_sha256 = self._evidence(session_id, case, metadata)
        destination, object_key, outbox = self._build_event(
            session_id, case, metadata, scale, source, camera,
        )
        pending = self._pending(session_id)
        completed = self._completed(session_id)
        audits = [
            record for record in _read_json_lines(self.audit_file)
            if record.get("session_id") == session_id and record.get("event") == "staged"
        ]
        # A completed event is already published; payload hashes from earlier
        # script revisions must not turn an idempotent re-run into an error.
        if not completed:
            if pending and _json_hash(pending.get("session_result")) != _json_hash(
                outbox["session_result"]
            ):
                raise RuntimeError("existing pending event has different payload")
            if any(
                record.get("payload_sha256") != _json_hash(outbox["session_result"])
                for record in audits
            ):
                raise RuntimeError("existing recovery audit has different payload")
        return {
            "session_id": session_id,
            "host_id": self.host_id,
            "expected_outcome": case["expected_outcome"],
            "completed": completed,
            "pending": bool(pending),
            "already_staged": bool(audits),
            "metadata_path": str(metadata_path) if metadata_path else None,
            "source_image": str(source) if source else None,
            "source_sha256": source_sha256,
            "destination_image": str(destination) if destination else None,
            "object_key": object_key,
            "image_less": source is None,
            "scale": scale,
            "payload_sha256": _json_hash(outbox["session_result"]),
            "outbox_event": outbox,
        }

    def apply(self, plan, frontend_confirmation, uploader, allow_image_less=False):
        session_id = plan["session_id"]
        if frontend_confirmation != session_id:
            raise RuntimeError("frontend absence confirmation must equal session ID")
        if plan["completed"]:
            return "completed"
        if plan["pending"]:
            return "pending"
        if plan["image_less"]:
            if not allow_image_less:
                raise RuntimeError("image-less recovery requires explicit approval")
        else:
            destination = Path(plan["destination_image"])
            source = Path(plan["source_image"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if destination.is_symlink() or hashlib.sha256(destination.read_bytes()).hexdigest() != plan[
                    "source_sha256"
                ]:
                    raise RuntimeError("existing recovery image differs from audited evidence")
            else:
                temporary = destination.with_name(destination.name + ".recovery.tmp")
                shutil.copyfile(source, temporary)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                if hashlib.sha256(temporary.read_bytes()).hexdigest() != plan["source_sha256"]:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError("recovery image copy verification failed")
                os.replace(temporary, destination)
                _fsync_directory(destination.parent)
            try:
                uploader(MINIO_BUCKET, plan["object_key"], str(destination))
            except Exception as exc:
                raise RuntimeError(f"recovery image upload failed: {exc}") from exc
            if self._upload_pending(plan["object_key"]):
                raise RuntimeError("recovery object key already exists in upload retry queue")
        records = _read_json_lines(self.pending_file)
        if any(record.get("id") == session_id for record in records):
            return "pending"
        records.append(plan["outbox_event"])
        _write_json_lines(self.pending_file, records)
        audit = {
            "event": "staged",
            "session_id": session_id,
            "host_id": self.host_id,
            "staged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "frontend_absence_confirmed": True,
            "image_less": bool(plan["image_less"]),
            "payload_sha256": plan["payload_sha256"],
            "source_sha256": plan["source_sha256"],
            "object_key": plan["object_key"],
        }
        _append_json_line(self.audit_file, audit)
        return "staged"


def _service_running():
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace"
            )
        except (OSError, PermissionError):
            continue
        if "weighing_service.py" in command:
            return True
    return False


def _upload_image(bucket, object_key, path):
    from services.storage.image_save_worker import ImageSaveWorker

    ImageSaveWorker._get_minio().fput_object(
        bucket, object_key, path, content_type="image/jpeg",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_id", choices=sorted(RECOVERY_CASES))
    parser.add_argument("--apply", action="store_true", help="stage event; default is dry-run")
    parser.add_argument(
        "--frontend-absence-confirmed", metavar="SESSION_ID",
        help="required apply gate after checking frontend/consumer history",
    )
    parser.add_argument(
        "--service-stopped", action="store_true",
        help="confirm weighing_service was stopped at empty scale before apply",
    )
    parser.add_argument(
        "--allow-image-less", metavar="SESSION_ID",
        help="required apply gate for an audited session with no attributable image",
    )
    args = parser.parse_args(argv)
    recovery = MissedSessionRecovery()
    try:
        plan = recovery.plan(args.session_id)
        public = {key: value for key, value in plan.items() if key != "outbox_event"}
        if not args.apply:
            print(json.dumps({"status": "dry_run", **public}, indent=2, sort_keys=True))
            return 0
        if not args.service_stopped:
            raise RuntimeError("--service-stopped is required for outbox staging")
        if _service_running():
            raise RuntimeError("weighing_service.py is still running")
        status = recovery.apply(
            plan,
            args.frontend_absence_confirmed,
            _upload_image,
            allow_image_less=args.allow_image_less == args.session_id,
        )
        print(json.dumps({"status": status, **public}, indent=2, sort_keys=True))
        return 0
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
