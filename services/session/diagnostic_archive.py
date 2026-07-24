import json
import os
from datetime import datetime

from services.storage.image_save_worker import ImageSaveWorker


def save_frames(root, item_id, frames, metadata, log_fn):
    started_at = metadata.get("started_at")
    try:
        now = datetime.fromisoformat(started_at).astimezone() if started_at else datetime.now()
    except (TypeError, ValueError):
        now = datetime.now()
    target = os.path.join(
        root,
        now.strftime("%Y"),
        now.strftime("%m"),
        now.strftime("%d"),
    )
    try:
        os.makedirs(target, exist_ok=True)
    except OSError as exc:
        log_fn("ERROR", f"Diagnostic directory create failed id={item_id}: {exc}")
        return False

    saved = []
    for camera, frame in frames.items():
        path = os.path.join(target, f"{item_id}_{camera}.jpg")
        if ImageSaveWorker.save_local_only(path, frame):
            saved.append(path)
        else:
            log_fn(
                "WARNING",
                f"Diagnostic image save failed camera={camera} id={item_id}",
            )

    path = os.path.join(target, f"{item_id}.json")
    temp_path = path + ".tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            json.dump(
                {**metadata, "images": saved},
                handle,
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except OSError as exc:
        log_fn("ERROR", f"Diagnostic metadata save failed id={item_id}: {exc}")
        return False
    return len(saved)
