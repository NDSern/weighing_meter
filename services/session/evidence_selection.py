"""Pure selection stages for unknown-plate publish evidence.

Collects and ranks candidate frames from session metadata and frame metadata.
No file decoding, bounding-box drawing, or logging lives here so the policy can
be unit tested with plain dict inputs.
"""

import os
from datetime import datetime, timezone
from itertools import combinations as candidate_combinations, product

CAMERAS = ("cam1", "cam2", "cam3")

UNKNOWN_PHOTO_MAX_OFFSET_SECONDS = 1.0
UNKNOWN_THUMBNAIL_OFFSET_SECONDS = 2.0
UNKNOWN_THUMBNAIL_WEIGHT_KG = 10000.0
UNKNOWN_WEIGHT_SNAPSHOT_DEADLINE_SECONDS = (
    UNKNOWN_THUMBNAIL_OFFSET_SECONDS + UNKNOWN_PHOTO_MAX_OFFSET_SECONDS
)


def select_target_timestamp(metadata, unknown_plate):
    """Return the epoch seconds a frame should be matched against, or None."""
    if unknown_plate in ("UNKNOWN_OCR", "UNKNOWN_DETECTION"):
        target_at = (
            metadata.get("local_peak_observed_at")
            or metadata.get("filtered_peak_observed_at")
            or metadata.get("raw_peak_observed_at")
            or metadata.get("weight_observed_at")
            or metadata.get("ended_at")
        )
    else:
        target_at = (
            metadata.get("weight_observed_at")
            or metadata.get("filtered_peak_observed_at")
            or metadata.get("raw_peak_observed_at")
            or metadata.get("ended_at")
        )
    try:
        return datetime.fromisoformat(target_at).timestamp()
    except (TypeError, ValueError):
        return None


def build_timeline(metadata, frame_metadata, started_ts, interval, session_dir, cameras=CAMERAS):
    """Return (timeline, start_snapshots) of session frames keyed by camera."""
    timeline = {camera: [] for camera in cameras}
    start_snapshots = {}
    first_seen_by_frame_id = {camera: {} for camera in cameras}

    for relative_path in metadata.get("session_files", []):
        camera = relative_path.split("-", 1)[0]
        if camera not in timeline:
            continue
        item_metadata = frame_metadata.get(relative_path) or {}
        observed_iso = item_metadata.get("captured_at")
        try:
            observed_ts = datetime.fromisoformat(observed_iso).timestamp()
        except (TypeError, ValueError):
            try:
                index = int(relative_path.split("-", 2)[1])
            except (ValueError, IndexError):
                continue
            observed_ts = started_ts + index * interval
            observed_iso = datetime.fromtimestamp(
                observed_ts, timezone.utc,
            ).isoformat(timespec="milliseconds")
        if observed_ts < started_ts - interval:
            continue
        path = os.path.abspath(os.path.join(session_dir, relative_path))
        if os.path.commonpath((session_dir, path)) != session_dir:
            continue
        candidate = {
            "path": path, "captured_at": observed_iso, "timestamp": observed_ts,
            "origin": "timeline", "camera": camera, "relative_path": relative_path,
        }
        frame_id = item_metadata.get("frame_id")
        if frame_id is not None:
            first_seen = first_seen_by_frame_id[camera]
            if frame_id in first_seen:
                continue
            first_seen[frame_id] = observed_ts
        if relative_path.endswith("-start.jpg"):
            start_snapshots[camera] = candidate
        else:
            timeline[camera].append(candidate)
    return timeline, start_snapshots


def dedicated_candidates(metadata, paths_key, times_key, source, rejected, cameras=CAMERAS):
    """Collect explicitly recorded snapshot paths for each camera."""
    candidates = {camera: [] for camera in cameras}
    paths = metadata.get(paths_key) or {}
    times = metadata.get(times_key) or {}
    for camera in cameras:
        path, observed_iso = paths.get(camera), times.get(camera)
        if not path or not observed_iso:
            continue
        try:
            observed_ts = datetime.fromisoformat(observed_iso).timestamp()
        except (TypeError, ValueError):
            rejected[camera] = {
                "source": source, "reason": "invalid_timestamp",
                "captured_at": observed_iso,
            }
            continue
        candidates[camera].append({
            "path": os.path.abspath(path), "captured_at": observed_iso,
            "timestamp": observed_ts, "origin": "dedicated", "camera": camera,
            "relative_path": os.path.basename(path),
        })
    return candidates


def apply_start_window(start_candidates, timeline, start_target_ts, rejected, cameras=CAMERAS):
    """Keep start snapshots within the thumbnail offset window, else fall back to timeline."""
    for camera in cameras:
        eligible = []
        for candidate in start_candidates[camera]:
            if abs(candidate["timestamp"] - start_target_ts) <= UNKNOWN_PHOTO_MAX_OFFSET_SECONDS:
                eligible.append(candidate)
            else:
                rejected[camera] = {
                    "source": "start_2s", "reason": "late_or_invalid_timestamp",
                    "captured_at": candidate["captured_at"],
                }
        start_candidates[camera] = eligible
        start_candidates[camera].extend(
            candidate for candidate in timeline[camera]
            if abs(candidate["timestamp"] - start_target_ts) <= UNKNOWN_PHOTO_MAX_OFFSET_SECONDS
        )


def apply_threshold_window(threshold_candidates, timeline, metadata, started_ts, deadline_ts,
                           rejected, cameras=CAMERAS):
    """Keep threshold snapshots inside the capture deadline window."""
    trigger_at = metadata.get("unknown_weight_snapshot_triggered_at")
    try:
        threshold_target_ts = datetime.fromisoformat(trigger_at).timestamp()
        threshold_target_valid = started_ts <= threshold_target_ts <= deadline_ts
    except (TypeError, ValueError):
        timely_dedicated = [
            candidate["timestamp"]
            for candidates in threshold_candidates.values()
            for candidate in candidates
            if started_ts <= candidate["timestamp"] <= deadline_ts
        ]
        threshold_target_ts = min(timely_dedicated) if timely_dedicated else None
        threshold_target_valid = threshold_target_ts is not None
    if threshold_target_valid:
        for camera in cameras:
            threshold_candidates[camera].extend(
                candidate for candidate in timeline[camera]
                if threshold_target_ts <= candidate["timestamp"] <= deadline_ts
            )
            eligible = []
            for candidate in threshold_candidates[camera]:
                if threshold_target_ts <= candidate["timestamp"] <= deadline_ts:
                    eligible.append(candidate)
                elif candidate["origin"] == "dedicated":
                    rejected[camera] = {
                        "source": "weight_10000",
                        "reason": "late_or_invalid_timestamp",
                        "captured_at": candidate["captured_at"],
                    }
            threshold_candidates[camera] = eligible
    else:
        for camera in cameras:
            for candidate in threshold_candidates[camera]:
                rejected[camera] = {
                    "source": "weight_10000",
                    "reason": "late_or_invalid_timestamp",
                    "captured_at": candidate["captured_at"],
                }
        threshold_candidates = {camera: [] for camera in cameras}
    return threshold_candidates


def build_session_start_candidates(start_snapshots, metadata, cameras=CAMERAS):
    """Start-of-session frames, including the rear camera when recorded."""
    session_start_candidates = {camera: [] for camera in cameras}
    for camera in cameras:
        if camera in start_snapshots:
            start_snapshot = dict(start_snapshots[camera])
            start_snapshot["origin"] = "session_start"
            session_start_candidates[camera].append(start_snapshot)
    if metadata.get("rear_start_path"):
        try:
            rear_ts = datetime.fromisoformat(metadata.get("rear_captured_at")).timestamp()
        except (TypeError, ValueError):
            rear_ts = None
        if rear_ts is not None:
            session_start_candidates["cam2"].append({
                "path": os.path.abspath(metadata["rear_start_path"]),
                "captured_at": metadata["rear_captured_at"],
                "timestamp": rear_ts, "origin": "session_start",
            })
    return session_start_candidates


def select_candidate_sets(metadata, timeline, target_ts, started_ts, unknown_plate,
                          session_start_candidates, rejected, cameras=CAMERAS):
    """Choose per-camera candidate lists and the target/source used for ranking."""
    if unknown_plate in ("UNKNOWN_OCR", "UNKNOWN_DETECTION"):
        dwell_candidates = dedicated_candidates(
            metadata, "local_peak_dwell_snapshot_paths",
            "local_peak_dwell_snapshot_captured_at", "local_peak_dwell", rejected, cameras,
        )
        drop_candidates = dedicated_candidates(
            metadata, "local_peak_drop_snapshot_paths",
            "local_peak_drop_snapshot_captured_at", "local_peak_drop", rejected, cameras,
        )
        source = "local_peak" if metadata.get("local_peak_observed_at") else "weight_recorded"
        target = target_ts
        candidate_sets = {}
        for camera in cameras:
            if dwell_candidates[camera]:
                candidate_sets[camera] = dwell_candidates[camera]
                continue
            if drop_candidates[camera]:
                candidate_sets[camera] = drop_candidates[camera]
                continue
            nearest = min(
                timeline[camera], key=lambda item: abs(item["timestamp"] - target),
                default=None,
            )
            if (
                nearest is not None
                and abs(nearest["timestamp"] - target) <= UNKNOWN_PHOTO_MAX_OFFSET_SECONDS
            ):
                candidate_sets[camera] = [nearest]
            else:
                candidate_sets[camera] = []
                if nearest is not None:
                    rejected[camera] = {
                        "source": source, "reason": "outside_target_window",
                        "captured_at": nearest["captured_at"],
                        "offset_ms": round((nearest["timestamp"] - target) * 1000),
                    }
    elif "unknown_snapshot_paths" in metadata or any(session_start_candidates[camera] for camera in cameras):
        source = "session_start"
        target = started_ts
        candidate_sets = session_start_candidates
    else:
        source = "legacy_weight"
        target = target_ts
        candidate_sets = {
            camera: [
                candidate for candidate in timeline[camera]
                if abs(candidate["timestamp"] - target_ts) <= UNKNOWN_PHOTO_MAX_OFFSET_SECONDS
            ]
            for camera in cameras
        }
    return source, target, candidate_sets


def synchronize_cameras(available_cameras, candidate_sets, target, source, rejected, cameras=CAMERAS):
    """Keep the largest group of cameras whose top frames are within one offset window."""
    synchronized_groups = [
        group
        for size in range(1, len(available_cameras) + 1)
        for group in candidate_combinations(available_cameras, size)
        if (
            max(candidate_sets[camera][0]["timestamp"] for camera in group)
            - min(candidate_sets[camera][0]["timestamp"] for camera in group)
            <= UNKNOWN_PHOTO_MAX_OFFSET_SECONDS
        )
    ]
    synchronized_cameras = min(
        synchronized_groups,
        key=lambda group: (
            -len(group),
            sum(abs(candidate_sets[camera][0]["timestamp"] - target) for camera in group),
            max(candidate_sets[camera][0]["timestamp"] for camera in group)
            - min(candidate_sets[camera][0]["timestamp"] for camera in group),
            group,
        ),
    )
    for camera in set(available_cameras) - set(synchronized_cameras):
        candidate = candidate_sets[camera][0]
        rejected[camera] = {
            "source": source, "reason": "inter_camera_skew",
            "captured_at": candidate["captured_at"],
            "offset_ms": round((candidate["timestamp"] - target) * 1000),
        }
        candidate_sets[camera] = []
    return [camera for camera in cameras if candidate_sets[camera]]


def rank_combinations(available_cameras, candidate_sets, target, unknown_plate):
    """Order candidate combinations, closest-to-target first for unknown plates."""
    if not available_cameras:
        return []
    combinations = product(*(candidate_sets[camera] for camera in available_cameras))
    if unknown_plate in ("UNKNOWN_OCR", "UNKNOWN_DETECTION"):
        return sorted(
            combinations,
            key=lambda items: (
                sum(abs(item["timestamp"] - target) for item in items),
                max(item["timestamp"] for item in items)
                - min(item["timestamp"] for item in items),
            ),
        )
    return sorted(
        combinations,
        key=lambda items: (
            max(item["timestamp"] for item in items)
            - min(item["timestamp"] for item in items),
            sum(abs(item["timestamp"] - target) for item in items),
        ),
    )


def resolve_detector_tracks(camera, relative_path, frame_metadata, metadata):
    """Detector boxes for a frame: exact path, then same-camera best-confidence fallback."""
    tracks = (frame_metadata.get(relative_path) or {}).get("tracks") or []
    if tracks:
        return tracks
    detected_boxes = (metadata.get("lpr_diagnostics") or {}).get("detected_boxes", {})
    tracks = detected_boxes.get(relative_path, [])
    if tracks:
        return tracks
    return max(
        (
            boxes for path, boxes in detected_boxes.items()
            if path.startswith(camera + "-") and boxes
        ),
        key=lambda boxes: max(box.get("confidence", 0.0) for box in boxes),
        default=[],
    )