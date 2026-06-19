"""Session manager — orchestrates weighing session lifecycle."""

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from services.storage.image_save_worker import ImageSaveWorker
from services.storage.publish_outbox import PublishOutbox, set_log_fn as set_publish_outbox_log

from config import (
    CAPTURE_DIR,
    MQTT_ENABLED,
    SESSION_END_WEIGHT_DROP_THRESHOLD,
    SERVICE_DIR,
    STABLE_COUNT_THRESHOLD,
    WEIGHT_CHANGE_THRESHOLD,
    WEIGHT_THRESHOLD,
)

_log_fn = None
_plate_db_lock = threading.Lock()
_registry_lock = threading.Lock()
_registry_loaded = False
_registry_mtime = None
_registry_exact = {}
_registry_family = {}
_registry_active_count = 0


def set_log_fn(log_fn):
    """Set the logging function to use."""
    global _log_fn
    _log_fn = log_fn
    set_publish_outbox_log(log_fn)


def log(level: str, msg: str):
    """Log using the configured log function."""
    if _log_fn:
        _log_fn(level, msg)


def saveConfirmedLicensePlate(license_plate):
    """Persist confirmed plate count once per confirmed session."""
    if not license_plate or license_plate == "none":
        return None

    db_file = os.path.join(SERVICE_DIR, "confirmed_license_plates.db")
    now = datetime.now().isoformat(timespec="seconds")
    try:
        with _plate_db_lock:
            conn = sqlite3.connect(db_file)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS confirmed_license_plates (
                        license_plate TEXT PRIMARY KEY,
                        recognition_count INTEGER NOT NULL DEFAULT 0,
                        first_seen_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    INSERT INTO confirmed_license_plates (
                        license_plate,
                        recognition_count,
                        first_seen_at,
                        last_seen_at
                    )
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(license_plate) DO UPDATE SET
                        recognition_count = recognition_count + 1,
                        last_seen_at = excluded.last_seen_at
                """, (license_plate, now, now))
                conn.commit()
                row = conn.execute(
                    "SELECT recognition_count FROM confirmed_license_plates WHERE license_plate = ?",
                    (license_plate,),
                ).fetchone()
                return row[0] if row else None
            finally:
                conn.close()
    except Exception as exc:
        print(f"[PLATE_DB] saveConfirmedLicensePlate failed: {exc}", flush=True)
        return None


def normalizeLicensePlate(license_plate):
    return "".join(ch for ch in (license_plate or "").upper() if ch.isalnum())


def licensePlatePrefix(normalized_plate):
    if len(normalized_plate) < 3:
        return normalized_plate
    return normalized_plate[:3]


def registeredFamilyKeys(normalized_plate):
    keys = {normalized_plate}
    if len(normalized_plate) >= 8:
        keys.add(normalized_plate[:-1])
    return keys


def editDistanceAtMostOne(left, right):
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left) and j < len(right):
        if left[i] == right[j]:
            i += 1
            j += 1
        else:
            edits += 1
            if edits > 1:
                return False
            j += 1
    return True


def loadRegisteredLicensePlates():
    global _registry_loaded, _registry_mtime, _registry_exact, _registry_family, _registry_active_count

    registry_file = os.path.join(SERVICE_DIR, "registered_license_plates.json")
    try:
        mtime = os.path.getmtime(registry_file)
    except OSError:
        mtime = None

    with _registry_lock:
        if _registry_loaded and _registry_mtime == mtime:
            return

        exact = {}
        family = {}
        active_count = 0
        if mtime is not None:
            try:
                with open(registry_file, "r", encoding="utf-8") as fh:
                    rows = json.load(fh)
                for row in rows:
                    if not isinstance(row, dict) or not row.get("active", True):
                        continue
                    plate = row.get("plate")
                    normalized = normalizeLicensePlate(plate)
                    if not normalized:
                        continue
                    active_count += 1
                    exact[normalized] = plate
                    for key in registeredFamilyKeys(normalized):
                        family.setdefault(key, []).append(plate)
            except Exception as exc:
                log("ERROR", f"[REGISTRY] Failed to load registered_license_plates.json: {exc}")

        _registry_loaded = True
        _registry_mtime = mtime
        _registry_exact = exact
        _registry_family = family
        _registry_active_count = active_count
        if mtime is None:
            log("REGISTRY", "No registered_license_plates.json found")
        else:
            log("REGISTRY", f"Loaded {active_count} active registered plates")


def correctWithRegisteredLicensePlate(license_plate):
    if not license_plate or license_plate == "none":
        return license_plate, None

    loadRegisteredLicensePlates()
    normalized = normalizeLicensePlate(license_plate)
    if not normalized:
        return license_plate, None

    with _registry_lock:
        exact = dict(_registry_exact)
        family = {key: list(value) for key, value in _registry_family.items()}

    if normalized in exact:
        registered = exact[normalized]
        if registered != license_plate:
            return registered, "exact"
        return license_plate, None

    family_matches = list(dict.fromkeys(family.get(normalized, [])))
    if len(family_matches) == 1:
        return family_matches[0], "family_unique"

    fuzzy_matches = []
    for registered_norm, registered_plate in exact.items():
        if editDistanceAtMostOne(normalized, registered_norm):
            fuzzy_matches.append(registered_plate)

    fuzzy_matches = list(dict.fromkeys(fuzzy_matches))
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0], "fuzzy_distance_1"
    if len(family_matches) > 1 or len(fuzzy_matches) > 1:
        log("REGISTRY", f"Kept plate {license_plate} reason=ambiguous registered_matches={family_matches + fuzzy_matches}")
    return license_plate, None


def preferDetailedLicensePlateCandidate(license_plate, all_plates):
    """Prefer 5-digit VN plate over same shortened 4-digit variant when seen in session."""
    cleaned = "".join(ch for ch in (license_plate or "").upper() if ch.isalnum())
    if len(cleaned) != 7:
        return license_plate

    detailed = []
    for plate, count in all_plates.items():
        candidate = "".join(ch for ch in plate.upper() if ch.isalnum())
        if len(candidate) == 8 and candidate.endswith("0") and candidate[:-1] == cleaned:
            detailed.append((plate, count))
    if not detailed:
        return license_plate
    return max(detailed, key=lambda item: item[1])[0]


class WeighingSessionState:
    def __init__(self):
        self.stable_weight = None
        self.stable_decimal_pos = 0
        self.last_publish_weight = None
        self.last_publish_decimal_pos = 0
        self.stable_count = 0
        self.session_active = False
        self.published_this_stop = False
        self.session_had_weight = False
        self.vehicle_type = None
        self.vehicle_was_stable = False
        self.both_unstable_since = None
        self.skipped_duplicate_publish = False
        self.rearm_block_until = 0.0
        self.rearm_block_reason = None
        self.lpr_start_frames = {}


class SessionManager:
    """Manages weighing session lifecycle, publish logic, and image capture."""

    def __init__(
        self,
        plate_tracker,
        mqtt_svc=None,
        vehicle_tracker=None,
        detect_coord=None,
        rear_grabber=None,
        lpr_grabbers=None,
        save_images_fn=None,
        undetectable_dir=None,
        cam2_result_crop="left",
    ):
        if cam2_result_crop not in ("left", "right", "full"):
            raise ValueError(f"Invalid cam2 result crop mode: {cam2_result_crop!r}")
        self.plate_tracker = plate_tracker
        self.mqtt_svc = mqtt_svc
        self.vehicle_tracker = vehicle_tracker
        self.detect_coord = detect_coord
        self.rear_grabber = rear_grabber
        self.lpr_grabbers = lpr_grabbers or {}
        self.save_images_fn = save_images_fn
        self.undetectable_dir = undetectable_dir
        self.cam2_result_crop = cam2_result_crop

        self.session = WeighingSessionState()
        self._last_publish_plate = None
        self._last_publish_weight = None
        self._publish_lock = threading.Lock()
        self._vehicle_summary_cache = None
        self._vehicle_summary_ts = 0.0
        if MQTT_ENABLED and self.mqtt_svc:
            PublishOutbox.start(self.mqtt_svc)

    def _get_vehicle_summary(self, max_age=0.25):
        if not self.vehicle_tracker:
            return None
        now = time.time()
        if self._vehicle_summary_cache is None or (now - self._vehicle_summary_ts) > max_age:
            self._vehicle_summary_cache = self.vehicle_tracker.get_summary()
            self._vehicle_summary_ts = now
        return self._vehicle_summary_cache

    def on_weight(self, frame, log_fn):
        """Logging callback (throttled ~1s by reader)."""
        vehicle_info = ""
        if self.vehicle_tracker and self.session.session_active:
            summary = self._get_vehicle_summary()
            if summary["vehicle_type"]:
                vehicle_info = f"  vehicle={summary['vehicle_type']}"
        tracker_plate, tracker_score = None, 0.0
        if self.session.session_active and frame.weight > WEIGHT_THRESHOLD:
            tracker_plate, tracker_score, _ = self.plate_tracker.get_confirmed_plate()
        plates_info = ""
        if self.session.stable_weight is not None:
            plates_info += f"  stable_wt={self.session.stable_weight:.{frame.decimal_pos}f}"
        if tracker_plate:
            plates_info += f"  plate={tracker_plate}({tracker_score:.2f})"
        if self.session.stable_count > 0:
            plates_info += f"  stable_count={self.session.stable_count}/{STABLE_COUNT_THRESHOLD}"
        log_fn(
            "WEIGHT",
            f"{frame.weight:>10.{frame.decimal_pos}f} kg  {frame.status:<10}{plates_info}{vehicle_info}",
        )

    def on_frame(self, frame, log_fn):
        """Per-frame callback (fires on every scale frame)."""
        if frame.status == "STABLE":
            self.session.stable_weight = frame.weight
            self.session.stable_decimal_pos = frame.decimal_pos
            self.session.stable_count += 1
            if self.session.stable_weight > WEIGHT_THRESHOLD:
                self.session.last_publish_weight = frame.weight
                self.session.last_publish_decimal_pos = frame.decimal_pos
                if not self.session.session_active and self._can_start_session(log_fn):
                    self._start_session(frame.decimal_pos, log_fn)
        else:
            self.session.stable_count = 0

        if self.session.session_active and frame.weight <= WEIGHT_THRESHOLD:
            self._end_session("scale_empty", log_fn)
            self.session.stable_weight = frame.weight
            return

        if self.session.session_active and self.vehicle_tracker:
            summary = self._get_vehicle_summary()
            if summary["vehicle_type"] and summary["vehicle_type"] != self.session.vehicle_type:
                self.session.vehicle_type = summary["vehicle_type"]
                log_fn("VEHICLE", f"Session vehicle_type={self.session.vehicle_type}")

            if summary["cam1_truck_stable"] or summary["cam3_truck_stable"]:
                if not self.session.vehicle_was_stable:
                    log_fn("VEHICLE", "Session truck stabilized")
                self.session.vehicle_was_stable = True

            both_unstable = summary["cam1_truck_unstable"] and summary["cam3_truck_unstable"]
            if self.session.vehicle_was_stable and both_unstable:
                if self.session.both_unstable_since is None:
                    self.session.both_unstable_since = time.time()
                elif (time.time() - self.session.both_unstable_since) > 0.5:
                    self._end_session("vehicle_left", log_fn)
            else:
                self.session.both_unstable_since = None

        if (
            self.session.session_active
            and self.session.last_publish_weight is not None
            and frame.weight <= (self.session.last_publish_weight - SESSION_END_WEIGHT_DROP_THRESHOLD)
        ):
            self._end_session("weight_drop", log_fn)

    def on_status_change(self, frame, old_status: str, new_status: str, log_fn):
        """Transition callback."""
        if frame.weight > WEIGHT_THRESHOLD:
            log_fn("SIGNAL", f"{old_status} → {new_status}  wt={frame.weight:.{frame.decimal_pos}f} kg")

        if new_status == "STABLE" and frame.weight <= WEIGHT_THRESHOLD:
            self._end_session("scale_empty", log_fn)
            self.session.stable_weight = frame.weight

    def _can_start_session(self, log_fn):
        if self.session.rearm_block_until <= 0:
            return True
        if time.time() >= self.session.rearm_block_until:
            log_fn("EVENT", f"Session rearm after {self.session.rearm_block_reason or 'unknown'} timeout")
            self.session.rearm_block_until = 0.0
            self.session.rearm_block_reason = None
            return True
        return False

    def _start_session(self, decimal_pos: int, log_fn):
        self.session.rearm_block_until = 0.0
        self.session.rearm_block_reason = None
        self.session.lpr_start_frames = self._capture_lpr_start_frames(log_fn)
        self.session.session_active = True
        self.session.session_had_weight = True
        self.session.stable_decimal_pos = decimal_pos
        self.session.last_publish_weight = self.session.stable_weight
        self.session.last_publish_decimal_pos = decimal_pos
        if self.detect_coord:
            self.detect_coord.set_enabled(True)
        log_fn("EVENT", f"===== SESSION START wt={self.session.stable_weight:.{decimal_pos}f}kg lpr=on =====")

    def _capture_lpr_start_frames(self, log_fn):
        frames = {}
        for name, grabber in self.lpr_grabbers.items():
            try:
                frame = grabber.peek_latest_frame(copy_frame=True)
            except Exception as exc:
                log_fn("ERROR", f"Session start snapshot failed camera={name}: {exc}")
                frame = None
            if frame is not None:
                frames[name] = frame
        if self.lpr_grabbers:
            status = " ".join(f"{name}={'yes' if name in frames else 'no'}" for name in sorted(self.lpr_grabbers))
            log_fn("EVENT", f"Session start snapshots {status}")
        return frames

    def _end_session(self, reason: str, log_fn):
        if not self.session.session_active:
            return

        end_reason = reason
        published_on_end = self._publish_on_session_end(reason, log_fn)
        confirmed_plate, _, _ = self.plate_tracker.get_confirmed_plate()
        saved_undetectable = False
        if self.session.session_had_weight and not self.session.published_this_stop and confirmed_plate is None:
            saved_undetectable = self._save_undetectable_frame(log_fn)

        publish_state = "duplicate" if self.session.skipped_duplicate_publish else self.session.published_this_stop
        log_fn(
            "EVENT",
            f"===== SESSION END reason={end_reason} wt={self.session.stable_weight or 0:.1f}kg "
            f"plate={confirmed_plate or 'none'} published={publish_state} "
            f"vehicle={self.session.vehicle_type or 'unknown'} undetectable={saved_undetectable} "
            f"publish_on_end={published_on_end} lpr=off =====",
        )

        if reason == "vehicle_left" and (self.session.stable_weight or 0) > WEIGHT_THRESHOLD:
            self.session.rearm_block_until = time.time() + 5.0
            self.session.rearm_block_reason = reason
            log_fn("EVENT", "Session rearm blocked for 5s after vehicle_left with weight still on scale")
        else:
            self.session.rearm_block_until = 0.0
            self.session.rearm_block_reason = None

        self.plate_tracker.clear()
        if self.detect_coord:
            self.detect_coord.set_enabled(False)
        self.session.session_active = False
        self.session.published_this_stop = False
        self.session.session_had_weight = False
        self.session.stable_count = 0
        self.session.last_publish_weight = None
        self.session.last_publish_decimal_pos = 0
        self.session.vehicle_type = None
        self.session.vehicle_was_stable = False
        self.session.both_unstable_since = None
        self.session.skipped_duplicate_publish = False
        self.session.lpr_start_frames = {}

    def _publish_on_session_end(self, reason: str, log_fn):
        if self.session.published_this_stop:
            return False
        if self.session.last_publish_weight is None or self.session.last_publish_weight <= WEIGHT_THRESHOLD:
            return False

        plate, score, count = self.plate_tracker.get_confirmed_plate()
        if not plate:
            return False

        published = self.publish_result(self.session.last_publish_weight, self.session.last_publish_decimal_pos, log_fn)
        if published == "duplicate":
            self.session.skipped_duplicate_publish = True
            log_fn("EVENT", f"Session end duplicate skip — reason={reason} plate={plate} score={score:.2f} hits={count}")
            return False
        if published:
            self.session.published_this_stop = True
            log_fn("EVENT", f"Session end publish — reason={reason} plate={plate} score={score:.2f} hits={count}")
        return bool(published)

    def _should_skip_duplicate_publish(self, plate, stable_weight):
        with self._publish_lock:
            plate_same = plate is not None and plate == self._last_publish_plate
            weight_similar = (
                self._last_publish_weight is not None
                and abs(stable_weight - self._last_publish_weight) < WEIGHT_CHANGE_THRESHOLD
            )
            if plate_same and weight_similar:
                log(
                    "MERGE",
                    f"Same plate [{plate}] + similar weight [{stable_weight} vs {self._last_publish_weight}] — same vehicle, skipping",
                )
                return True
        return False

    def publish_result(self, stable_weight, decimal_pos, log_fn):
        """Query PlateTracker and publish if plate is confirmed."""
        plate, score, count = self.plate_tracker.get_confirmed_plate()
        image_lookup_plate = plate
        all_plates = self.plate_tracker.get_all_plates_summary()
        preferred_plate = preferDetailedLicensePlateCandidate(plate, all_plates)
        if preferred_plate != plate:
            log_fn("PLATE", f"Canonicalized confirmed plate {plate} -> {preferred_plate} candidates={all_plates}")
            image_lookup_plate = plate
            plate = preferred_plate
            count = all_plates.get(plate, count)
        registered_plate, registry_reason = correctWithRegisteredLicensePlate(plate)
        if registered_plate != plate:
            log_fn("REGISTRY", f"Corrected plate {plate} -> {registered_plate} reason={registry_reason}")
            plate = registered_plate
        if self._should_skip_duplicate_publish(plate, stable_weight):
            return "duplicate"

        result = self._build_publish_result(stable_weight, plate, count, all_plates)
        self._log_publish_summary(stable_weight, decimal_pos, plate, score, count, all_plates, log_fn)
        image_aliases = [image_lookup_plate, *all_plates.keys()]
        if not self._attach_publish_images(result, stable_weight, decimal_pos, plate, image_aliases, log_fn):
            log_fn("ERROR", f"Publish skipped — image capture failed for plate={plate}")
            return False

        saved_count = saveConfirmedLicensePlate(plate)
        if saved_count is not None:
            log_fn("PLATE_DB", f"Saved confirmed plate={plate} recognition_count={saved_count}")

        image_object_keys = result.pop("_image_object_keys", [])
        image_paths = result.pop("_image_paths", [])
        save_items = result.pop("_image_save_items", [])
        if MQTT_ENABLED and self.mqtt_svc:
            PublishOutbox.enqueue(result, image_object_keys=image_object_keys, image_paths=image_paths)
        if save_items and not ImageSaveWorker.save_and_enqueue_upload(save_items):
            log_fn("WARNING", f"Publish queued offline — local image save incomplete for plate={plate}")
        else:
            log_fn("OFFLINE", f"Publish queued offline id={result.get('offline_event_id')} plate={plate}")

        with self._publish_lock:
            self._last_publish_plate = plate
            self._last_publish_weight = stable_weight
        return True

    def _build_publish_result(self, stable_weight, plate, count, all_plates):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "start": timestamp,
            "end": timestamp,
            "duration_s": 0,
            "stable_weight": stable_weight,
            "official_plate": plate or "none",
            "official_plate_count": count,
            "all_plates": all_plates,
            "image_path": None,
        }

    def _log_publish_summary(self, stable_weight, decimal_pos, plate, score, count, all_plates, log_fn):
        plate_text = plate or "none"
        candidates = ", ".join(f"{p}:{c}" for p, c in sorted(all_plates.items())) if all_plates else "-"
        log_fn(
            "EVENT",
            f"PUBLISH wt={stable_weight:.{decimal_pos}f}kg plate={plate_text} score={score:.2f} hits={count} candidates=[{candidates}]",
        )

    def _prepare_capture_paths(self, now, plate):
        date_path = now.strftime("%Y/%m/%d")
        day_dir = os.path.join(CAPTURE_DIR, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
        os.makedirs(day_dir, exist_ok=True)
        ts = now.strftime("%Y%m%d_%H%M%S_%f")

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
            "unchosen_cam1": _make("photo-unchosen-cam1"),
            "unchosen_cam3": _make("photo-unchosen-cam3"),
        }

    def _crop_cam2_result_image(self, frame):
        h, w = frame.shape[:2]
        crop_mode = self.cam2_result_crop
        if crop_mode == "left":
            return frame[:, : w // 2]
        if crop_mode == "right":
            return frame[:, w // 2 :]
        if crop_mode == "full":
            return frame
        raise ValueError(f"Invalid cam2 result crop mode: {crop_mode!r}")

    def _build_publish_images(self, frame, plate, stable_weight, decimal_pos, rear_frame):
        frame_h = frame.shape[0]
        cv2.putText(
            frame,
            f"Bien so: {plate}    Tai trong xe: {stable_weight:.{decimal_pos}f} kg",
            (10, frame_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8680625,
            (0, 255, 0),
            3,
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

    def _attach_publish_images(self, result, stable_weight, decimal_pos, plate, image_aliases, log_fn):
        import numpy as np
        attach_started_at = time.time()

        frame, img_plate, camera_name = self.plate_tracker.get_image_frame(plate, aliases=image_aliases)
        if frame is None or not plate:
            return False
        if img_plate != plate:
            log_fn("SAVE", f"Using image plate {img_plate} for final plate {plate}")

        now = datetime.now()
        captured_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        paths = self._prepare_capture_paths(now, plate)
        rear_frame = self.rear_grabber.get_latest_frame() if self.rear_grabber else None
        if rear_frame is not None:
            rear_frame = self._crop_cam2_result_image(rear_frame)
        front_img, merged_img, rear_img = self._build_publish_images(
            frame, plate, stable_weight, decimal_pos, rear_frame
        )

        photos = [
            {"url": paths["merged"][2], "type": "merged", "captured_at": captured_at},
            {"url": paths["front"][2], "type": "front", "captured_at": captured_at},
        ]
        save_items = [
            [paths["merged"][0], merged_img, paths["merged"][1]],
            [paths["front"][0], front_img, paths["front"][1]],
        ]
        if rear_img is not None:
            photos.append({"url": paths["rear"][2], "type": "rear", "captured_at": captured_at})
            save_items.append([paths["rear"][0], rear_img, paths["rear"][1]])

        unchosen_camera = None
        if camera_name == "cam1":
            unchosen_camera = "cam3"
        elif camera_name == "cam3":
            unchosen_camera = "cam1"
        if unchosen_camera:
            unchosen_frame = self.session.lpr_start_frames.get(unchosen_camera)
            unchosen_key = f"unchosen_{unchosen_camera}"
            if unchosen_frame is not None and unchosen_key in paths:
                ImageSaveWorker._encode_save_local(paths[unchosen_key][0], unchosen_frame)
                log_fn("SAVE", f"Saved local-only unchosen LPR start image camera={unchosen_camera} plate={plate}")

        result["photos"] = photos
        result["_image_object_keys"] = [item[2] for item in save_items]
        result["_image_paths"] = [item[0] for item in save_items]
        result["_image_save_items"] = save_items

        log_fn("TIMING", f"Publish images: build={(time.time() - attach_started_at) * 1000:.0f}ms")
        return True

    def _save_undetectable_frame(self, log_fn):
        """Save the first 'unknown' detection frame to /storage/undetectable/."""
        frame_data = self.plate_tracker.get_undetectable_frame()
        if frame_data is None:
            log_fn("WARNING", "No undetectable frame to save")
            return False
        os.makedirs(self.undetectable_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        fpath = os.path.join(self.undetectable_dir, f"{ts}_undetectable.jpg")
        try:
            cv2.imwrite(fpath, frame_data)
            log_fn("SAVE", f"Undetectable saved: {fpath}")
            return True
        except Exception as exc:
            log_fn("ERROR", f"Failed to save undetectable image: {exc}")
            return False
        finally:
            del frame_data
