"""Publication payload and result-image construction helpers."""

import os
from datetime import datetime, timezone

import cv2
import numpy as np

from config import CAPTURE_DIR


def prepare_capture_paths(now, plate, session_id=None, capture_dir=CAPTURE_DIR):
    """Build local paths plus object keys/URLs for every result image role."""
    date_path = now.strftime("%Y/%m/%d")
    day_dir = os.path.join(capture_dir, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
    os.makedirs(day_dir, exist_ok=True)
    ts = session_id or now.strftime("%Y%m%d_%H%M%S_%f")

    def _make(suffix):
        fname = f"{ts}_{plate}_{suffix}.jpg"
        fpath = os.path.join(day_dir, fname)
        key = f"storage/weighbridge/{date_path}/{fname}"
        url = f"/storage/weighbridge/{date_path}/{fname}"
        return fpath, key, url

    return {
        "front": _make("photo-front"),
        "rear": _make("photo-rear"),
        "merged": _make("photo-merged"),
        "cam1": _make("photo-cam1"),
        "cam2": _make("photo-cam2"),
        "cam3": _make("photo-cam3"),
    }


def build_publish_result(stable_weight, plate, count, all_plates, metadata=None):
    """Build the MQTT session result payload."""
    metadata = metadata or {}
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    start = metadata.get("started_at")
    end = metadata.get("ended_at")
    event_timestamp = end or datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    if start:
        start = datetime.fromisoformat(start).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    if end:
        end = datetime.fromisoformat(end).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "start": start or timestamp,
        "end": end or timestamp,
        "timestamp": event_timestamp,
        "duration_s": metadata.get("duration_s", 0),
        "stable_weight": stable_weight,
        "official_plate": plate or "none",
        "official_plate_count": count,
        "all_plates": all_plates,
        "image_path": None,
    }


def crop_cam2_result_image(frame, crop_mode):
    """Crop the rear/side result image according to the configured mode."""
    h, w = frame.shape[:2]
    if crop_mode == "left":
        return frame[:, : w // 2]
    if crop_mode == "right":
        return frame[:, w // 2 :]
    if crop_mode == "full":
        return frame
    raise ValueError(f"Invalid cam2 result crop mode: {crop_mode!r}")


def build_publish_images(frame, plate, stable_weight, decimal_pos, rear_frame):
    """Draw the weight/plate caption and optionally merge the rear frame."""
    frame_h = frame.shape[0]
    font_scale = max(0.9, frame_h / 1080)
    thickness = max(2, round(font_scale * 3))
    cv2.putText(
        frame,
        f"Bien so: {plate}    Tai trong xe: {stable_weight:.{decimal_pos}f} kg",
        (10, frame_h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    front_img = frame
    if rear_frame is None:
        return front_img, front_img, None

    rear_h = rear_frame.shape[0]
    rear_width = rear_frame.shape[1] * frame_h // rear_h
    rear_resized = cv2.resize(rear_frame, (rear_width, frame_h))
    merged_img = np.hstack([front_img, rear_resized])
    return front_img, merged_img, rear_resized