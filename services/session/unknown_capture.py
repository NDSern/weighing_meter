"""UNKNOWN-session snapshot capture and evidence frame selection.

Extracted from :mod:`services.session.session_manager` so the lifecycle
orchestrator stays focused. Stateful helpers take the owning ``SessionManager``
explicitly; the manager keeps thin delegating methods with the original names
so call sites and tests remain unchanged.
"""

import os
import time
from datetime import datetime, timezone

import cv2

from services.session.evidence_selection import (
    CAMERAS,
    UNKNOWN_THUMBNAIL_OFFSET_SECONDS,
    UNKNOWN_THUMBNAIL_WEIGHT_KG,
    UNKNOWN_WEIGHT_SNAPSHOT_DEADLINE_SECONDS,
    apply_start_window,
    apply_threshold_window,
    build_session_start_candidates,
    build_timeline,
    dedicated_candidates,
    rank_combinations,
    resolve_detector_tracks,
    select_candidate_sets,
    select_target_timestamp,
    synchronize_cameras,
)
from services.session.metrics import log_metric


def capture_unknown_snapshots_if_due(manager, log_fn):
    """Capture the start-relative UNKNOWN snapshot set once its deadline passes."""
    deadline = manager.session.unknown_snapshot_deadline
    if (
        not manager.session.session_active
        or deadline is None
        or manager.session.unknown_snapshot_attempted
        or time.time() < deadline
    ):
        return False
    manager.session.unknown_snapshot_attempted = True
    captured = capture_unknown_snapshot_set(manager, "2s", log_fn)
    manager._update_spool_metadata(log_fn)
    log_metric(
        log_fn, "unknown_snapshot_captured", id=manager.session.session_id,
        source="start_2s",
        target_at=datetime.fromtimestamp(deadline, timezone.utc).isoformat(timespec="milliseconds"),
        captured_at=captured,
    )
    return bool(captured)


def capture_unknown_weight_snapshots_if_due(manager, weight, log_fn):
    """Capture UNKNOWN snapshots once weight passes the thumbnail threshold."""
    if (
        not manager.session.session_active
        or manager.session.started_at is None
        or manager.session.unknown_weight_snapshot_attempted
    ):
        return False
    capture_deadline = manager.session.started_at + UNKNOWN_WEIGHT_SNAPSHOT_DEADLINE_SECONDS
    if time.time() > capture_deadline:
        manager.session.unknown_weight_snapshot_attempted = True
        log_metric(
            log_fn, "unknown_weight_snapshot_expired", id=manager.session.session_id,
            deadline_at=datetime.fromtimestamp(
                capture_deadline, timezone.utc,
            ).isoformat(timespec="milliseconds"),
            captured_cameras=sorted(manager.session.unknown_weight_snapshot_paths),
        )
        return False
    if (
        manager.session.unknown_weight_snapshot_triggered_at is None
        and weight < UNKNOWN_THUMBNAIL_WEIGHT_KG
    ):
        return False
    if manager.session.unknown_weight_snapshot_triggered_at is None:
        manager.session.unknown_weight_snapshot_triggered_at = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        manager._update_spool_metadata(log_fn)
        return False
    captured = capture_unknown_snapshot_set(
        manager, "weight-10000", log_fn,
        minimum_captured_ts=datetime.fromisoformat(
            manager.session.unknown_weight_snapshot_triggered_at
        ).timestamp(),
        maximum_captured_ts=capture_deadline,
    )
    expected_cameras = set(manager.lpr_grabbers)
    if manager.rear_grabber is not None:
        expected_cameras.add("cam2")
    manager.session.unknown_weight_snapshot_attempted = expected_cameras.issubset(
        manager.session.unknown_weight_snapshot_paths
    )
    manager._update_spool_metadata(log_fn)
    if captured:
        log_metric(
            log_fn, "unknown_snapshot_captured", id=manager.session.session_id,
            source="weight_10000", weight_kg=weight,
            triggered_at=manager.session.unknown_weight_snapshot_triggered_at,
            captured_at=captured,
        )
    return bool(captured)


def capture_unknown_snapshot_set(
    manager, source, log_fn, minimum_captured_ts=None, maximum_captured_ts=None,
):
    """Save one frame per camera into the session spool under a named source."""
    captured = {}
    grabbers = dict(manager.lpr_grabbers)
    if manager.rear_grabber is not None:
        grabbers["cam2"] = manager.rear_grabber
    for camera, grabber in grabbers.items():
        if source == "weight-10000" and camera in manager.session.unknown_weight_snapshot_paths:
            continue
        frame_id = None
        try:
            snapshot = getattr(grabber, "peek_latest_frame_snapshot", None)
            if snapshot:
                frame, frame_id, captured_at = snapshot(copy_frame=True)
            else:
                frame = grabber.peek_latest_frame(copy_frame=True)
                captured_at = None
        except Exception as exc:
            log_fn("ERROR", f"UNKNOWN {source} snapshot failed camera={camera}: {exc}")
            continue
        if frame is None:
            continue
        captured_at = captured_at or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        try:
            captured_ts = datetime.fromisoformat(captured_at).timestamp()
        except (TypeError, ValueError):
            continue
        if minimum_captured_ts is not None and captured_ts < minimum_captured_ts:
            continue
        if maximum_captured_ts is not None and captured_ts > maximum_captured_ts:
            continue
        path = None
        if manager.frame_spool and manager.session.spool_active:
            try:
                args = (manager.session.session_id, f"{camera}-unknown-{source}.jpg", frame)
                if source.startswith("local-peak-"):
                    path = manager.frame_spool.save_session_frame(
                        *args, frame_id=frame_id, captured_at=captured_at,
                    )
                else:
                    path = manager.frame_spool.save_session_frame(*args)
            except Exception as exc:
                log_fn("ERROR", f"UNKNOWN {source} snapshot save failed camera={camera}: {exc}")
        if path:
            session = manager.session
            if source == "weight-10000":
                session.unknown_weight_snapshot_paths[camera] = path
                session.unknown_weight_snapshot_captured_at[camera] = captured_at
            elif source == "local-peak-dwell":
                session.local_peak_dwell_snapshot_paths[camera] = path
                session.local_peak_dwell_snapshot_captured_at[camera] = captured_at
            elif source == "local-peak-drop":
                session.local_peak_drop_snapshot_paths[camera] = path
                session.local_peak_drop_snapshot_captured_at[camera] = captured_at
            else:
                session.unknown_snapshot_paths[camera] = path
                session.unknown_snapshot_captured_at[camera] = captured_at
            captured[camera] = captured_at
    return captured


def load_start_frames(metadata):
    """Load the recorded session-start frames (front cameras plus rear)."""
    frames = {}
    for camera, path in metadata.get("start_frame_paths", {}).items():
        frame = cv2.imread(path)
        if frame is not None:
            frames[camera] = frame
    rear_path = metadata.get("rear_start_path")
    rear_frame = cv2.imread(rear_path) if rear_path else None
    if rear_frame is not None:
        frames["cam2"] = rear_frame
    return frames


def nearest_session_frame(metadata, camera, observed_at):
    """Return the spooled frame for ``camera`` closest to ``observed_at``."""
    session_dir = metadata.get("session_dir")
    files = metadata.get("session_files", [])
    started_at_raw = metadata.get("started_at")
    if not session_dir or observed_at is None or not started_at_raw:
        return None
    started_at = datetime.fromisoformat(started_at_raw).timestamp()
    interval = float(metadata.get("capture_interval_seconds", 0.2))
    candidates = []
    for relative_path in files:
        if not relative_path.startswith(camera + "-"):
            continue
        try:
            index = int(relative_path.split("-", 2)[1])
        except (ValueError, IndexError):
            continue
        candidates.append((abs(started_at + index * interval - observed_at), relative_path))
    if not candidates:
        return None
    return os.path.join(session_dir, min(candidates)[1])


def load_diagnostic_frames(metadata, offset_seconds):
    """Load front-camera frames ``offset_seconds`` after session start."""
    started_at_raw = metadata.get("started_at")
    if not started_at_raw:
        return load_start_frames(metadata)
    target = datetime.fromisoformat(started_at_raw).timestamp() + offset_seconds
    frames = {}
    for camera in ("cam1", "cam3"):
        path = nearest_session_frame(metadata, camera, target)
        frame = cv2.imread(path) if path else None
        if frame is not None:
            frames[camera] = frame
    return frames or load_start_frames(metadata)


def load_unknown_publish_frames(
    metadata, frame_metadata, log_fn, spool_started_at=None,
    unknown_plate="UNKNOWN",
):
    """Select the best-timed UNKNOWN-session frames per camera."""
    target_ts = select_target_timestamp(metadata, unknown_plate)
    if target_ts is None and unknown_plate in ("UNKNOWN", "UNKNOWN_OCR", "UNKNOWN_DETECTION"):
        log_fn("WARNING", f"Unknown photo timing unavailable id={metadata['session_id']}")
        return {}, {}

    session_dir = metadata.get("session_dir")
    started_at = spool_started_at or metadata.get("started_at")
    if not session_dir or not started_at:
        return {}, {}
    session_dir = os.path.abspath(session_dir)
    started_ts = datetime.fromisoformat(started_at).timestamp()
    interval = float(metadata.get("capture_interval_seconds", 0.2))
    start_target_ts = started_ts + UNKNOWN_THUMBNAIL_OFFSET_SECONDS
    deadline_ts = started_ts + UNKNOWN_WEIGHT_SNAPSHOT_DEADLINE_SECONDS
    cameras = CAMERAS
    rejected = {}

    timeline, start_snapshots = build_timeline(
        metadata, frame_metadata, started_ts, interval, session_dir, cameras,
    )

    start_candidates = dedicated_candidates(
        metadata, "unknown_snapshot_paths", "unknown_snapshot_captured_at",
        "start_2s", rejected, cameras,
    )
    apply_start_window(start_candidates, timeline, start_target_ts, rejected, cameras)

    threshold_candidates = dedicated_candidates(
        metadata, "unknown_weight_snapshot_paths", "unknown_weight_snapshot_captured_at",
        "weight_10000", rejected, cameras,
    )
    threshold_candidates = apply_threshold_window(
        threshold_candidates, timeline, metadata, started_ts, deadline_ts, rejected, cameras,
    )

    session_start_candidates = build_session_start_candidates(start_snapshots, metadata, cameras)

    source, target, candidate_sets = select_candidate_sets(
        metadata, timeline, target_ts, started_ts, unknown_plate,
        session_start_candidates, rejected, cameras,
    )

    available_cameras = [camera for camera in cameras if candidate_sets[camera]]
    if unknown_plate in ("UNKNOWN_OCR", "UNKNOWN_DETECTION") and len(available_cameras) > 1:
        available_cameras = synchronize_cameras(
            available_cameras, candidate_sets, target, source, rejected, cameras,
        )

    selected = {}
    captured_at = {}
    selection = {}
    synchronized_gap_ms = None
    ranked = rank_combinations(available_cameras, candidate_sets, target, unknown_plate)
    for combination in ranked:
        frames = [cv2.imread(item["path"]) for item in combination]
        if any(frame is None for frame in frames):
            continue
        timestamps = [item["timestamp"] for item in combination]
        synchronized_gap_ms = round((max(timestamps) - min(timestamps)) * 1000)
        for camera, item, frame in zip(available_cameras, combination, frames):
            if unknown_plate == "UNKNOWN_OCR":
                tracks = resolve_detector_tracks(
                    camera, item.get("relative_path"), frame_metadata, metadata,
                )
                draw_unknown_ocr_bbox(frame, tracks)
            selected[camera] = frame
            captured_at[camera] = item["captured_at"]
            selection[camera] = {
                "captured_at": item["captured_at"],
                "offset_ms": round((item["timestamp"] - target) * 1000),
                "source": source,
                "fallback": item["origin"] if item["origin"] != "dedicated" else None,
            }
        break
    missing = [camera for camera in cameras if camera not in selected]
    log_metric(
        log_fn, "unknown_photo_selection", id=metadata["session_id"],
        target_at=datetime.fromtimestamp(target, timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        weight_source=metadata.get("weight_source"), unknown_type=unknown_plate,
        lpr_target_source=source,
        camera_target_at={
            camera: datetime.fromtimestamp(target, timezone.utc).isoformat(timespec="milliseconds")
            for camera in cameras
        },
        synchronized_gap_ms=synchronized_gap_ms,
        selected=selection, rejected=rejected, missing_cameras=missing,
    )
    return selected, captured_at


def draw_unknown_ocr_bbox(frame, tracks):
    """Draw the highest-confidence detector box on an UNKNOWN_OCR frame."""
    if not tracks:
        return
    if not hasattr(frame, "shape"):
        return
    bbox = max(tracks, key=lambda track: track.get("confidence", 0.0)).get("bbox") or []
    if len(bbox) != 4:
        return
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width - 1, x2), min(height - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(
        frame, (x1, y1), (x2, y2), (0, 255, 0),
        max(2, round(height / 1080 * 3)), cv2.LINE_AA,
    )


def load_lpr_diagnostic_frames(metadata, classification):
    """Load the best available LPR diagnostic frame per camera."""
    session_dir = metadata.get("session_dir")
    evidence = (metadata.get("lpr_diagnostics") or {}).get("evidence") or {}
    if not session_dir:
        return {}
    order = (
        classification,
        "valid",
        "plate_detected_ocr_low_confidence",
        "plate_detected_ocr_invalid_format",
        "plate_detected_ocr_blank",
        "crop_failed",
        "no_plate_detection",
        "ocr_inference_error",
        "detector_inference_error",
        "lpr_frames_unavailable",
    )
    frames = {}
    for camera, paths in evidence.items():
        relative_path = next((paths.get(key) for key in order if paths.get(key)), None)
        if not relative_path:
            continue
        path = os.path.abspath(os.path.join(session_dir, relative_path))
        if os.path.commonpath((os.path.abspath(session_dir), path)) != os.path.abspath(session_dir):
            continue
        frame = cv2.imread(path)
        if frame is not None:
            frames[camera] = frame
    return frames
