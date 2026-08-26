"""Session manager — orchestrates weighing session lifecycle."""

import json
import os
import threading
import time
import uuid
from collections import Counter, deque
from datetime import datetime, timezone

import cv2
import numpy as np

from services.storage.image_save_worker import ImageSaveWorker
from services.storage.publish_outbox import PublishOutbox, set_log_fn as set_publish_outbox_log
from services.session import diagnostic_archive, finalization_store, plate_store

from config import (
    CAPTURE_DIR,
    MQTT_ENABLED,
    NO_PLATE_DIR,
    NO_STABLE_DIR,
    PEAK_CANDIDATE_DIR,
    PEAK_FILTER_FRAMES,
    PEAK_MOVEMENT_CANCEL_KG,
    PEAK_MOVEMENT_CONFIRM_FRAMES,
    SAME_PLATE_DUPLICATE_SECONDS,
    SESSION_DEDUP_STATE_FILE,
    SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT,
    SESSION_FINALIZATION_DB,
    SESSION_END_EMPTY_DWELL_SECONDS,
    SESSION_STABLE_WEIGHT_WINDOW,
    SESSION_WEIGHT_DEPARTURE_DWELL_SECONDS,
    SESSION_WEIGHT_DEPARTURE_KG,
    SESSION_WEIGHT_TREND_DIRECTIONAL_STEPS,
    SESSION_WEIGHT_TREND_FRAMES,
    SERVICE_DIR,
    STABLE_COUNT_THRESHOLD,
    WEIGHT_THRESHOLD,
)

_log_fn = None
_registry_lock = threading.Lock()
_registry_loaded = False
_registry_mtime = None
_registry_exact = {}
_registry_family = {}
_registry_active_count = 0
MAX_STABLE_WEIGHT_CANDIDATES = 256
REAR_CAPTURE_FALLBACK_SECONDS = 2.0
UNKNOWN_PHOTO_MAX_OFFSET_SECONDS = 1.0


def set_log_fn(log_fn):
    """Set the logging function to use."""
    global _log_fn
    _log_fn = log_fn
    set_publish_outbox_log(log_fn)


def log(level: str, msg: str):
    """Log using the configured log function."""
    if _log_fn:
        _log_fn(level, msg)


def isSessionFinalized(session_id):
    return finalization_store.contains(SESSION_FINALIZATION_DB, session_id)


def getSessionFinalization(session_id):
    return finalization_store.get(SESSION_FINALIZATION_DB, session_id)


def markSessionFinalized(session_id, outcome, record=None):
    finalization_store.mark(SESSION_FINALIZATION_DB, session_id, outcome, record)


def log_metric(log_fn, event, **fields):
    log_fn("METRIC", json.dumps({"event": event, **fields}, separators=(",", ":"), sort_keys=True))


def classify_lpr_failure(diagnostics):
    diagnostics = diagnostics or {}
    if diagnostics.get("available_lpr_frames") == 0:
        return "lpr_frames_unavailable"
    if diagnostics.get("ocr_valid_candidates", 0):
        return "no_confirmed_plate_after_voting"
    if diagnostics.get("ocr_low_confidence", 0):
        return "plate_detected_ocr_low_confidence"
    if diagnostics.get("ocr_invalid_format", 0):
        return "plate_detected_ocr_invalid_format"
    if diagnostics.get("ocr_blank", 0):
        return "plate_detected_ocr_blank"
    if diagnostics.get("crop_failures", 0) or diagnostics.get("crop_too_small", 0):
        return "crop_failed"
    if diagnostics.get("detector_successes", 0) and not diagnostics.get("detected_regions", 0):
        return "no_plate_detection"
    if diagnostics.get("ocr_errors", 0):
        return "ocr_inference_error"
    if diagnostics.get("detector_errors", 0):
        return "detector_inference_error"
    return "no_plate_detection"


def saveConfirmedLicensePlate(license_plate, session_id=None):
    """Persist confirmed plate count once per confirmed session."""
    db_file = os.path.join(SERVICE_DIR, "confirmed_license_plates.db")
    try:
        return plate_store.increment(db_file, license_plate, session_id)
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
        self.stable_weight_counts = Counter()
        self.stable_weight_last_seen = {}
        self.stable_weight_decimal_pos = {}
        self.stable_weight_observation_times = {}
        self.stable_weight_history = deque(maxlen=SESSION_STABLE_WEIGHT_WINDOW)
        self.stable_weight_sequence = 0
        self.latest_stable_weight = None
        self.stable_weight_observed_at = None
        self.weight_trend_window = deque(maxlen=SESSION_WEIGHT_TREND_FRAMES)
        self.weight_trend_observation_times = deque(maxlen=SESSION_WEIGHT_TREND_FRAMES)
        self.session_active = False
        self.vehicle_type = None
        self.empty_since = None
        self.rearm_block_until = 0.0
        self.rearm_block_reason = None
        self.rearm_reference_weight = None
        self.lpr_start_frames = {}
        self.start_frame_paths = {}
        self.rear_start_frame = None
        self.rear_start_path = None
        self.rear_captured_at = None
        self.rear_capture_source = None
        self.rear_fallback_deadline = None
        self.rear_fallback_attempted = False
        self.started_at = None
        self.started_at_iso = None
        self.session_id = None
        self.spool_active = False
        self.stability_rule = None

    def record_stable_weight(self, weight, decimal_pos, observed_at=None):
        self.latest_stable_weight = weight
        if not self.session_active:
            self.stable_weight = weight
            self.stable_decimal_pos = decimal_pos
            return

        self.stable_weight_sequence += 1
        if len(self.stable_weight_history) == self.stable_weight_history.maxlen:
            expired_weight, _expired_decimal_pos, _expired_observed_at = self.stable_weight_history.popleft()
            self.stable_weight_counts[expired_weight] -= 1
            if self.stable_weight_counts[expired_weight] <= 0:
                self.stable_weight_counts.pop(expired_weight, None)
                self.stable_weight_last_seen.pop(expired_weight, None)
                self.stable_weight_decimal_pos.pop(expired_weight, None)
                self.stable_weight_observation_times.pop(expired_weight, None)
        self.stable_weight_history.append((weight, decimal_pos, observed_at))
        self.stable_weight_counts[weight] += 1
        self.stable_weight_last_seen[weight] = self.stable_weight_sequence
        self.stable_weight_decimal_pos[weight] = decimal_pos
        self.stable_weight_observation_times[weight] = observed_at
        if len(self.stable_weight_counts) > MAX_STABLE_WEIGHT_CANDIDATES:
            oldest_weakest = min(
                self.stable_weight_counts,
                key=lambda value: (
                    self.stable_weight_counts[value],
                    self.stable_weight_last_seen[value],
                ),
            )
            self.stable_weight_counts.pop(oldest_weakest)
            self.stable_weight_last_seen.pop(oldest_weakest, None)
            self.stable_weight_decimal_pos.pop(oldest_weakest, None)
            self.stable_weight_observation_times.pop(oldest_weakest, None)
        self.stable_weight = max(
            self.stable_weight_counts,
            key=lambda value: (
                self.stable_weight_counts[value],
                self.stable_weight_last_seen[value],
            ),
        )
        self.stable_decimal_pos = self.stable_weight_decimal_pos[self.stable_weight]
        self.stable_weight_observed_at = self.stable_weight_observation_times[self.stable_weight]

    def start_stable_weight_history(self):
        self.stable_weight_counts.clear()
        self.stable_weight_last_seen.clear()
        self.stable_weight_decimal_pos.clear()
        self.stable_weight_observation_times.clear()
        self.stable_weight_history.clear()
        self.stable_weight_sequence = 1
        self.stable_weight_history.append(
            (self.stable_weight, self.stable_decimal_pos, self.stable_weight_observed_at)
        )
        self.stable_weight_counts[self.stable_weight] = 1
        self.stable_weight_last_seen[self.stable_weight] = 1
        self.stable_weight_decimal_pos[self.stable_weight] = self.stable_decimal_pos
        self.stable_weight_observation_times[self.stable_weight] = self.stable_weight_observed_at

    def clear_stable_weight_history(self):
        self.stable_weight_counts.clear()
        self.stable_weight_last_seen.clear()
        self.stable_weight_decimal_pos.clear()
        self.stable_weight_observation_times.clear()
        self.stable_weight_history.clear()
        self.stable_weight_sequence = 0
        self.latest_stable_weight = None
        self.weight_trend_window.clear()
        self.weight_trend_observation_times.clear()


class SessionManager:
    """Manages weighing session lifecycle, publish logic, and image capture."""

    def __init__(
        self,
        plate_tracker,
        mqtt_svc=None,
        vehicle_tracker=None,
        rear_grabber=None,
        lpr_grabbers=None,
        save_images_fn=None,
        undetectable_dir=None,
        cam2_result_crop="left",
        frame_spool=None,
    ):
        if cam2_result_crop not in ("left", "right", "full"):
            raise ValueError(f"Invalid cam2 result crop mode: {cam2_result_crop!r}")
        self.plate_tracker = plate_tracker
        self.mqtt_svc = mqtt_svc
        self.vehicle_tracker = vehicle_tracker
        self.rear_grabber = rear_grabber
        self.lpr_grabbers = lpr_grabbers or {}
        self.save_images_fn = save_images_fn
        self.undetectable_dir = undetectable_dir
        self.cam2_result_crop = cam2_result_crop
        self.frame_spool = frame_spool

        self.session = WeighingSessionState()
        self._last_publish_plate = None
        self._last_publish_weight = None
        self._last_publish_session_end = None
        self._publish_lock = threading.Lock()
        self._vehicle_summary_cache = None
        self._vehicle_summary_ts = 0.0
        self._attempt = None
        self._attempt_departure_since = None
        self._attempt_empty_since = None
        self._attempt_rearm_low = None
        self._attempt_wait_reference = None
        self._post_session_low = None
        self._waiting_for_empty = False
        self._post_session_empty_since = None
        self._peak_candidate = None
        self._peak_weight_window = deque(maxlen=PEAK_FILTER_FRAMES)
        self._peak_movement_frames = []
        self._peak_movement_started_at = None
        self._lifecycle_lock = threading.RLock()
        self._generation = 0
        self._plate_owned = False
        self._plate_absent_since = None
        self._plate_presence_revision = 0
        self._session_raw_peak = None
        self._session_raw_peak_observed_at = None
        self._session_filtered_peak = None
        self._session_filtered_peak_observed_at = None
        self._session_weight_window = deque(maxlen=PEAK_FILTER_FRAMES)
        self.fatal_error = None
        self._spool_failure_logged_for = None
        self._pending_terminal_snapshot = None
        self._last_spool_weight_checkpoint = 0.0
        self._load_dedup_state()

    def session_context(self):
        with self._lifecycle_lock:
            return self._generation, self.session.session_id

    def on_plate_presence(self, _camera, valid_by_camera, log_fn=None, revision=None):
        log_fn = log_fn or log
        with self._lifecycle_lock:
            if self.fatal_error:
                return
            if revision is not None:
                if revision <= self._plate_presence_revision:
                    return
                self._plate_presence_revision = revision
            any_valid = any(valid_by_camera.values())
            now = time.monotonic()
            if any_valid:
                self._plate_absent_since = None
                if not self.session.session_active:
                    self._clear_attempt()
                    self._start_session(0, log_fn, trigger="plate_detected")
                elif not self._plate_owned:
                    self._plate_owned = True
                    log_fn("EVENT", f"Session upgrade id={self.session.session_id} trigger=plate_detected")
                return
            if not self.session.session_active or not self._plate_owned:
                return
            if self._plate_absent_since is None:
                self._plate_absent_since = now

    def _complete_plate_loss(self, log_fn, current_weight=0.0):
        observed_weight = max(current_weight, self._session_raw_peak or 0.0)
        if (
            SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT
            and observed_weight > WEIGHT_THRESHOLD
        ):
            self._plate_owned = False
            self._plate_absent_since = None
            log_fn(
                "EVENT",
                f"Session continues under scale lifecycle id={self.session.session_id} "
                f"after plate tracks lost weight={observed_weight:g}kg",
            )
            return
        self._end_session("both_plate_tracks_lost", log_fn)

    def _load_dedup_state(self):
        try:
            with open(SESSION_DEDUP_STATE_FILE, encoding="utf-8") as handle:
                state = json.load(handle)
            self._last_publish_plate = state.get("plate")
            self._last_publish_weight = state.get("weight")
            self._last_publish_session_end = state.get("ended_at")
        except (OSError, ValueError, TypeError):
            pass

    def _save_dedup_state(self):
        os.makedirs(os.path.dirname(SESSION_DEDUP_STATE_FILE), exist_ok=True)
        temp = SESSION_DEDUP_STATE_FILE + ".tmp"
        with open(temp, "w", encoding="utf-8") as handle:
            json.dump({
                "plate": self._last_publish_plate,
                "weight": self._last_publish_weight,
                "ended_at": self._last_publish_session_end,
            }, handle, separators=(",", ":"), sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, SESSION_DEDUP_STATE_FILE)

    def shutdown(self, log_fn):
        """Persist active capture state before sampler and inference workers stop."""
        with self._lifecycle_lock:
            return self._shutdown_locked(log_fn)

    def _shutdown_locked(self, log_fn):
        if self._peak_candidate:
            frame = type("PeakEndFrame", (), {
                "weight": self._peak_candidate["peak_weight_kg"],
                "timestamp": datetime.now(timezone.utc),
            })()
            self._archive_peak_candidate(frame, log_fn, "shutdown")
        if self.session.session_active:
            self._end_session("shutdown", log_fn)
        elif self._attempt:
            self._archive_no_stable(log_fn)

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
        if frame.stability_rule:
            plates_info += f"  stable_rule={frame.stability_rule}"
        log_fn(
            "WEIGHT",
            f"{frame.weight:>10.{frame.decimal_pos}f} kg  {frame.status:<10}{plates_info}{vehicle_info}",
        )

    def on_frame(self, frame, log_fn):
        """Per-frame callback (fires on every scale frame)."""
        with self._lifecycle_lock:
            return self._on_frame_locked(frame, log_fn)

    def _on_frame_locked(self, frame, log_fn):
        if self.fatal_error:
            return
        if (
            self.session.session_active
            and self._plate_owned
            and self._plate_absent_since is not None
            and time.monotonic() - self._plate_absent_since >= 1.0
        ):
            self._complete_plate_loss(log_fn, frame.weight)
        self._capture_rear_fallback_if_due(log_fn)
        self._update_peak_candidate(frame, log_fn)
        if self.session.session_active and frame.weight > 0:
            observed_at = frame.timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds")
            if self._session_raw_peak is None or frame.weight > self._session_raw_peak:
                self._session_raw_peak = frame.weight
                self._session_raw_peak_observed_at = observed_at
            self._session_weight_window.append(frame.weight)
            if len(self._session_weight_window) == self._session_weight_window.maxlen:
                filtered = sorted(self._session_weight_window)[len(self._session_weight_window) // 2]
                if self._session_filtered_peak is None or filtered > self._session_filtered_peak:
                    self._session_filtered_peak = filtered
                    self._session_filtered_peak_observed_at = observed_at
            if time.monotonic() - self._last_spool_weight_checkpoint >= 1.0:
                self._update_spool_metadata(log_fn)
                self._last_spool_weight_checkpoint = time.monotonic()
        if self._wait_for_empty_cycle(frame, log_fn):
            self.session.stable_count = 0
            return
        if frame.status == "STABLE":
            stable_weight = frame.stable_weight if frame.stable_weight is not None else frame.weight
            blocked_tail = (
                not self.session.session_active
                and self._attempt_wait_reference is not None
                and (
                    time.time() < self.session.rearm_block_until
                    or abs(stable_weight - self._attempt_wait_reference) < SESSION_WEIGHT_DEPARTURE_KG
                )
            )
            if not blocked_tail:
                self._update_attempt(frame, log_fn, allow_chained_stable=True)
            self._handle_stable_frame(frame, log_fn)
        else:
            if not self.session.session_active and self._attempt_wait_reference is not None:
                if frame.weight <= WEIGHT_THRESHOLD:
                    self._attempt_wait_reference = None
                    self._post_session_low = None
                else:
                    if self._post_session_low is None:
                        self._post_session_low = self._attempt_wait_reference
                    self._post_session_low = min(self._post_session_low, frame.weight)
                    direct_rise = frame.weight - self._attempt_wait_reference
                    rebound = frame.weight - self._post_session_low
                    if max(direct_rise, rebound) < SESSION_WEIGHT_DEPARTURE_KG:
                        return
                    self._attempt_wait_reference = None
                    self._post_session_low = None
            self._update_attempt(frame, log_fn)
            self.session.stable_count = 0

        if self._check_weight_trend(frame, log_fn):
            return
        if self._check_scale_empty(frame, log_fn):
            return
        self._update_vehicle_type(log_fn)

    def _update_peak_candidate(self, frame, log_fn):
        """Record shadow peak evidence without changing session behavior."""
        self._peak_weight_window.append((frame.timestamp, frame.weight, frame.status, frame.stability_rule))
        if len(self._peak_weight_window) < PEAK_FILTER_FRAMES:
            return
        ordered_weights = sorted(item[1] for item in self._peak_weight_window)
        filtered_weight = ordered_weights[len(ordered_weights) // 2]
        if self._peak_candidate is None:
            if filtered_weight <= WEIGHT_THRESHOLD:
                return
            first = self._peak_weight_window[0]
            peak = max(self._peak_weight_window, key=lambda item: item[1])
            self._peak_candidate = {
                "id": uuid.uuid4().hex,
                "started_at": first[0].astimezone(timezone.utc).isoformat(timespec="milliseconds"),
                "peak_at": peak[0].astimezone(timezone.utc).isoformat(timespec="milliseconds"),
                "peak_weight_kg": peak[1],
                "filtered_peak_weight_kg": filtered_weight,
                "peak_status": peak[2],
                "peak_stability_rule": peak[3],
                "saw_stable": any(item[2] == "STABLE" for item in self._peak_weight_window),
                "blocked_waiting_for_empty": self._waiting_for_empty,
                "absorbed_active_session": self.session.session_active,
                "session_ids": [self.session.session_id] if self.session.session_id else [],
                "start_frames": self._capture_lpr_start_frames(log_fn),
            }
            self._peak_movement_frames = []
            self._peak_movement_started_at = None
            log_metric(
                log_fn, "weight_peak_candidate", id=self._peak_candidate["id"],
                started_at=self._peak_candidate["started_at"], start_weight_kg=filtered_weight,
                waiting_for_empty=self._waiting_for_empty,
                session_active=self.session.session_active,
            )
            return

        candidate = self._peak_candidate
        candidate["saw_stable"] = candidate["saw_stable"] or frame.status == "STABLE"
        candidate["blocked_waiting_for_empty"] = (
            candidate["blocked_waiting_for_empty"] or self._waiting_for_empty
        )
        candidate["absorbed_active_session"] = (
            candidate["absorbed_active_session"] or self.session.session_active
        )
        if self.session.session_id and self.session.session_id not in candidate["session_ids"]:
            candidate["session_ids"].append(self.session.session_id)
        if frame.weight >= candidate["peak_weight_kg"]:
            candidate["peak_weight_kg"] = frame.weight
            candidate["peak_at"] = frame.timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds")
            candidate["peak_status"] = frame.status
            candidate["peak_stability_rule"] = frame.stability_rule
        candidate["filtered_peak_weight_kg"] = max(
            candidate["filtered_peak_weight_kg"], filtered_weight,
        )

        baseline = candidate["filtered_peak_weight_kg"]
        departed = baseline - filtered_weight >= SESSION_WEIGHT_DEPARTURE_KG
        empty = filtered_weight <= WEIGHT_THRESHOLD
        if not departed and not empty:
            if self._peak_movement_frames and baseline - filtered_weight <= PEAK_MOVEMENT_CANCEL_KG:
                log_metric(
                    log_fn, "weight_peak_rocking_cancelled", id=candidate["id"],
                    peak_weight_kg=candidate["peak_weight_kg"],
                    filtered_weight_kg=filtered_weight,
                    excursion_frames=len(self._peak_movement_frames),
                )
            self._peak_movement_frames = []
            self._peak_movement_started_at = None
            return
        if self._peak_movement_started_at is None:
            self._peak_movement_started_at = frame.timestamp.timestamp()
        self._peak_movement_frames.append(filtered_weight)
        if len(self._peak_movement_frames) > PEAK_MOVEMENT_CONFIRM_FRAMES:
            self._peak_movement_frames.pop(0)
        if len(self._peak_movement_frames) < PEAK_MOVEMENT_CONFIRM_FRAMES:
            return
        if (
            frame.timestamp.timestamp() - self._peak_movement_started_at
            < SESSION_WEIGHT_DEPARTURE_DWELL_SECONDS
        ):
            return
        downward_steps = sum(
            later <= earlier
            for earlier, later in zip(self._peak_movement_frames, self._peak_movement_frames[1:])
        )
        if not empty and downward_steps < PEAK_MOVEMENT_CONFIRM_FRAMES - 1:
            return
        self._archive_peak_candidate(frame, log_fn, "scale_empty" if empty else "weight_departure")

    def _archive_peak_candidate(self, frame, log_fn, end_reason):
        candidate = self._peak_candidate
        if not candidate:
            return
        if candidate["blocked_waiting_for_empty"]:
            category = "blocked_waiting_for_empty"
        elif candidate["absorbed_active_session"]:
            category = "absorbed_active_session"
        elif candidate["saw_stable"]:
            category = "stable_session"
        else:
            category = "unstable_local_peak"
        metadata = {
            key: value for key, value in candidate.items() if key != "start_frames"
        }
        metadata.update({
            "category": category,
            "ended_at": frame.timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds"),
            "end_reason": end_reason,
            "end_weight_kg": frame.weight,
            "shadow_only": True,
        })
        saved = self._save_diagnostic_frames(
            PEAK_CANDIDATE_DIR, candidate["id"], candidate["start_frames"], metadata, log_fn,
        )
        log_metric(
            log_fn, "weight_peak_rejected" if category != "stable_session" else "weight_peak_promoted",
            id=candidate["id"], category=category,
            peak_weight_kg=candidate["peak_weight_kg"], end_reason=end_reason,
            images=saved, shadow_only=True,
        )
        self._peak_candidate = None
        self._peak_weight_window.clear()
        self._peak_movement_frames = []
        self._peak_movement_started_at = None

    def _wait_for_empty_cycle(self, frame, log_fn):
        if not self._waiting_for_empty:
            return False
        if frame.weight > WEIGHT_THRESHOLD:
            self._post_session_empty_since = None
            return True
        now = time.time()
        if self._post_session_empty_since is None:
            self._post_session_empty_since = now
            return True
        if now - self._post_session_empty_since < SESSION_END_EMPTY_DWELL_SECONDS:
            return True
        self._waiting_for_empty = False
        self._post_session_empty_since = None
        self.session.rearm_block_until = 0.0
        self.session.rearm_block_reason = None
        self.session.rearm_reference_weight = None
        self._attempt_wait_reference = None
        self._post_session_low = None
        log_fn("EVENT", "Scale cycle rearmed after empty dwell")
        log_metric(log_fn, "scale_cycle_rearmed", empty_dwell_s=SESSION_END_EMPTY_DWELL_SECONDS)
        return False

    def _handle_stable_frame(self, frame, log_fn):
        stable_weight = frame.stable_weight if frame.stable_weight is not None else frame.weight
        previous = self.session.stable_weight
        observed_at = frame.timestamp.astimezone(timezone.utc).isoformat(timespec="milliseconds")
        self.session.record_stable_weight(stable_weight, frame.decimal_pos, observed_at)
        self.session.last_publish_weight = self.session.stable_weight
        self.session.last_publish_decimal_pos = self.session.stable_decimal_pos
        self.session.stability_rule = frame.stability_rule
        if (
            self.session.session_active
            and self.session.stable_weight > WEIGHT_THRESHOLD
            and self.session.stable_weight != previous
        ):
            self._update_spool_metadata(log_fn)
        self.session.stable_count += 1
        if self.session.stable_weight <= WEIGHT_THRESHOLD:
            return
        # Stable weight alone never starts a cycle; rising trend does.

    def _update_attempt(self, frame, log_fn, allow_chained_stable=False):
        if self.session.session_active:
            if self._attempt:
                self._attempt["max_weight"] = max(self._attempt["max_weight"], frame.weight)
            return
        if self._attempt_rearm_low is not None:
            self._attempt_rearm_low = min(self._attempt_rearm_low, frame.weight)
            if frame.weight <= WEIGHT_THRESHOLD:
                self._attempt_rearm_low = None
            elif not allow_chained_stable and frame.weight - self._attempt_rearm_low < SESSION_WEIGHT_DEPARTURE_KG:
                return
            else:
                self._attempt_rearm_low = None
        if self._attempt is None and frame.weight > WEIGHT_THRESHOLD:
            self._attempt = {
                "id": uuid.uuid4().hex,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "max_weight": frame.weight,
                "start_frames": self._capture_lpr_start_frames(log_fn),
            }
            self._attempt_departure_since = None
            self._attempt_empty_since = None
            log_fn("EVENT", f"Weight attempt start id={self._attempt['id']} wt={frame.weight:.{frame.decimal_pos}f}kg")
            log_metric(
                log_fn, "weight_attempt_start", id=self._attempt["id"],
                started_at=self._attempt["started_at"], start_weight_kg=frame.weight,
            )
        if self._attempt is None:
            return

        self._attempt["max_weight"] = max(self._attempt["max_weight"], frame.weight)
        now = time.time()
        if frame.weight <= WEIGHT_THRESHOLD:
            self._attempt_empty_since = self._attempt_empty_since or now
            if now - self._attempt_empty_since >= SESSION_END_EMPTY_DWELL_SECONDS:
                self._archive_no_stable(log_fn, require_new_rise=False)
            return
        self._attempt_empty_since = None
        if self._attempt["max_weight"] - frame.weight >= SESSION_WEIGHT_DEPARTURE_KG:
            self._attempt_departure_since = self._attempt_departure_since or now
            if now - self._attempt_departure_since >= SESSION_WEIGHT_DEPARTURE_DWELL_SECONDS:
                self._archive_no_stable(log_fn, require_new_rise=True, current_weight=frame.weight)
        else:
            self._attempt_departure_since = None

    def _archive_no_stable(self, log_fn, require_new_rise=False, current_weight=None):
        attempt = self._attempt
        if not attempt:
            return
        saved = self._save_diagnostic_frames(
            NO_STABLE_DIR,
            attempt["id"],
            attempt["start_frames"],
            {"reason": "no_stable_weight", "started_at": attempt["started_at"],
             "ended_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
             "maximum_weight_kg": attempt["max_weight"]},
            log_fn,
        )
        log_fn("EVENT", f"NO STABLE id={attempt['id']} max_wt={attempt['max_weight']:.1f}kg images={saved}")
        log_metric(
            log_fn, "no_stable_weight", id=attempt["id"],
            started_at=attempt["started_at"],
            ended_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            maximum_weight_kg=attempt["max_weight"], images=saved,
            end_reason="weight_departure" if require_new_rise else "scale_empty",
        )
        self._clear_attempt()
        if require_new_rise:
            self._attempt_rearm_low = current_weight

    def _clear_attempt(self):
        self._attempt = None
        self._attempt_departure_since = None
        self._attempt_empty_since = None

    @staticmethod
    def _save_diagnostic_frames(root, item_id, frames, metadata, log_fn):
        return diagnostic_archive.save_frames(
            root,
            item_id,
            frames,
            metadata,
            log_fn,
        )

    def _check_weight_trend(self, frame, log_fn):
        if frame.weight <= WEIGHT_THRESHOLD:
            self.session.weight_trend_window.clear()
            self.session.weight_trend_observation_times.clear()
            return False
        window = self.session.weight_trend_window
        window.append(frame.weight)
        self.session.weight_trend_observation_times.append(frame.timestamp)
        if len(window) < SESSION_WEIGHT_TREND_FRAMES:
            return False
        net_movement = window[-1] - window[0]
        if abs(net_movement) < SESSION_WEIGHT_DEPARTURE_KG:
            return False
        if net_movement > 0:
            direction = "rising"
            directional_steps = sum(later > earlier for earlier, later in zip(window, list(window)[1:]))
        else:
            direction = "falling"
            directional_steps = sum(later < earlier for earlier, later in zip(window, list(window)[1:]))
        if directional_steps < SESSION_WEIGHT_TREND_DIRECTIONAL_STEPS:
            return False
        log_metric(
            log_fn, "weight_trend_confirmed", direction=direction,
            start_weight_kg=window[0], end_weight_kg=window[-1],
            net_movement_kg=net_movement, directional_steps=directional_steps,
            comparison_frames=len(window),
        )
        if not self.session.session_active:
            if direction == "rising" and self._can_start_session(log_fn):
                self._attempt = {
                    "id": uuid.uuid4().hex,
                    "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "max_weight": max(window),
                    "start_frames": self._capture_lpr_start_frames(log_fn),
                }
                self._start_session(
                    frame.decimal_pos, log_fn, trigger="scale_rising",
                    observed_at=frame.timestamp,
                )
                return True
            return False
        if self._plate_owned:
            return False
        if direction == "falling":
            self._end_session("weight_trend_falling", log_fn)
        return True

    def _check_scale_empty(self, frame, log_fn):
        if not self.session.session_active:
            return False
        if self._plate_owned or frame.weight > WEIGHT_THRESHOLD:
            self.session.empty_since = None
            return False
        if self.session.empty_since is None:
            self.session.empty_since = time.time()
            return False
        if (time.time() - self.session.empty_since) < SESSION_END_EMPTY_DWELL_SECONDS:
            return False

        if self._end_session("scale_empty", log_fn) is False:
            return True
        self.session.stable_weight = frame.weight
        return True

    def _update_vehicle_type(self, log_fn):
        if not self.session.session_active or not self.vehicle_tracker:
            return

        summary = self._get_vehicle_summary()
        if summary["vehicle_type"] and summary["vehicle_type"] != self.session.vehicle_type:
            self.session.vehicle_type = summary["vehicle_type"]
            log_fn("VEHICLE", f"Session vehicle_type={self.session.vehicle_type}")

    def on_status_change(self, frame, old_status: str, new_status: str, log_fn):
        """Transition callback."""
        if frame.weight > WEIGHT_THRESHOLD:
            log_fn("SIGNAL", f"{old_status} → {new_status}  wt={frame.weight:.{frame.decimal_pos}f} kg")

        if new_status == "STABLE" and frame.weight <= WEIGHT_THRESHOLD:
            self.session.stable_weight = frame.weight

    def _can_start_session(self, log_fn):
        if self.session.rearm_block_until <= 0:
            return True
        if time.time() < self.session.rearm_block_until:
            return False
        if (
            self.session.rearm_reference_weight is not None
            and abs(self.session.stable_weight - self.session.rearm_reference_weight) < SESSION_WEIGHT_DEPARTURE_KG
        ):
            return False
        if time.time() >= self.session.rearm_block_until:
            log_fn("EVENT", f"Session rearm after {self.session.rearm_block_reason or 'unknown'} timeout")
            self.session.rearm_block_until = 0.0
            self.session.rearm_block_reason = None
            self.session.rearm_reference_weight = None
            return True

    def _start_session(
        self, decimal_pos: int, log_fn, trigger="scale_rising", observed_at=None,
    ):
        self.plate_tracker.clear()
        self.session.stable_weight_observed_at = None
        self.session.rearm_block_until = 0.0
        self.session.rearm_block_reason = None
        self.session.rearm_reference_weight = None
        self._attempt_wait_reference = None
        self._post_session_low = None
        if self._attempt is None:
            self._attempt = {
                "id": uuid.uuid4().hex,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "max_weight": self.session.stable_weight or 0,
                "start_frames": self._capture_lpr_start_frames(log_fn),
            }
        if trigger == "plate_detected":
            # Idle stable values can belong to a departing vehicle. Plate ownership
            # starts a fresh weight window and must not inherit that plateau.
            self.session.stable_weight = None
            self.session.stable_decimal_pos = decimal_pos
            self.session.last_publish_weight = None
            self.session.last_publish_decimal_pos = decimal_pos
            self.session.clear_stable_weight_history()
        elif trigger == "scale_rising":
            rising_window = list(self.session.weight_trend_window)
            rising_observation_times = list(self.session.weight_trend_observation_times)
            self.session.stable_weight = None
            self.session.last_publish_weight = None
            self.session.clear_stable_weight_history()
            self._session_weight_window.extend(rising_window)
            if self._session_weight_window:
                self._session_raw_peak = max(self._session_weight_window)
                if rising_observation_times:
                    raw_peak_index = max(
                        range(len(rising_window)), key=rising_window.__getitem__,
                    )
                    self._session_raw_peak_observed_at = rising_observation_times[
                        raw_peak_index
                    ].astimezone(
                        timezone.utc
                    ).isoformat(timespec="milliseconds")
                if len(self._session_weight_window) == self._session_weight_window.maxlen:
                    self._session_filtered_peak = sorted(self._session_weight_window)[
                        len(self._session_weight_window) // 2
                    ]
                    if observed_at is not None:
                        self._session_filtered_peak_observed_at = observed_at.astimezone(
                            timezone.utc
                        ).isoformat(timespec="milliseconds")
        self.session.lpr_start_frames = self._attempt["start_frames"]
        self.session.started_at = time.time()
        self.session.session_id = self._attempt["id"]
        self.session.started_at_iso = self._attempt["started_at"]
        if self.session.stable_weight is not None:
            self.session.start_stable_weight_history()
        self.session.stable_decimal_pos = decimal_pos
        self.session.last_publish_weight = self.session.stable_weight
        self.session.last_publish_decimal_pos = decimal_pos
        if self.frame_spool:
            try:
                session_dir = self.frame_spool.begin_session(
                    self.session.session_id,
                    self.session.lpr_start_frames,
                    metadata={
                        "session_id": self.session.session_id,
                        "started_at": self.session.started_at_iso,
                        "stable_weight": self.session.last_publish_weight,
                        "decimal_pos": self.session.last_publish_decimal_pos,
                        "stability_rule": self.session.stability_rule,
                        "weight_observed_at": self.session.stable_weight_observed_at,
                        "raw_peak_weight": self._session_raw_peak,
                        "raw_peak_observed_at": self._session_raw_peak_observed_at,
                        "filtered_peak_weight": self._session_filtered_peak,
                        "filtered_peak_observed_at": self._session_filtered_peak_observed_at,
                        "vehicle_type": self.session.vehicle_type,
                        "rear_start_path": None,
                        "start_frame_paths": {},
                    },
                )
                self.session.spool_active = True
                self.session.start_frame_paths = {
                    camera: os.path.join(session_dir, f"{camera}-000000-start.jpg")
                    for camera in self.session.lpr_start_frames
                    if os.path.exists(os.path.join(session_dir, f"{camera}-000000-start.jpg"))
                }
            except Exception as exc:
                try:
                    self.frame_spool.abort_session(self.session.session_id)
                except Exception as abort_exc:
                    log_fn("ERROR", f"Session frame spool abort failed: {abort_exc}")
                self.session.spool_active = False
                self.fatal_error = "Session frame spool admission failed: %s" % exc
                log_fn("FATAL", self.fatal_error)
                log_metric(
                    log_fn, "session_spool_admission_failed",
                    id=self.session.session_id, started_at=self.session.started_at_iso,
                    trigger=trigger, error=str(exc),
                )
                return False
        self.session.session_active = True
        self._generation += 1
        self._plate_owned = trigger == "plate_detected"
        self._plate_absent_since = None
        if not self._capture_and_save_rear("promotion", "cam2-start.jpg", log_fn):
            self.session.rear_fallback_deadline = (
                self.session.started_at + REAR_CAPTURE_FALLBACK_SECONDS
            )
        self._update_spool_metadata(log_fn)
        self._clear_attempt()
        weight_text = f"{self.session.stable_weight:.{decimal_pos}f}" if self.session.stable_weight is not None else "none"
        log_fn("EVENT", f"===== SESSION START trigger={trigger} wt={weight_text}kg lpr=on =====")
        stability_reason = {
            "exact_5": "5 exact weight frames",
            "spread_10": "weights value within tolerance",
        }.get(self.session.stability_rule, self.session.stability_rule or "unknown stability rule")
        log_fn(
            "EVENT",
            f"Session start reason={stability_reason} rule={self.session.stability_rule or 'unknown'}",
        )
        log_metric(
            log_fn, "session_start", id=self.session.session_id,
            trigger=trigger,
            attempt_started_at=self.session.started_at_iso,
            stable_at=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            stable_weight_kg=self.session.stable_weight,
            stability_rule=self.session.stability_rule,
        )
        return True

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

    def _capture_rear_start_frame(self, log_fn):
        if not self.rear_grabber:
            return None
        try:
            frame = self.rear_grabber.peek_latest_frame(copy_frame=True)
        except Exception as exc:
            log_fn("ERROR", f"Session start rear snapshot failed camera=cam2: {exc}")
            return None
        log_fn("EVENT", f"Session start rear snapshot cam2={'yes' if frame is not None else 'no'}")
        return frame

    def _capture_and_save_rear(self, source, filename, log_fn):
        frame = self._capture_rear_start_frame(log_fn)
        if frame is None:
            return False
        self.session.rear_start_frame = frame
        self.session.rear_captured_at = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        self.session.rear_capture_source = source
        path = None
        if self.frame_spool and self.session.spool_active:
            try:
                path = self.frame_spool.save_session_frame(
                    self.session.session_id,
                    filename,
                    frame,
                )
            except Exception as exc:
                log_fn("ERROR", f"Session rear snapshot save failed source={source}: {exc}")
        if not path:
            return not (self.frame_spool and self.session.spool_active)
        self.session.rear_start_path = path
        self.session.rear_fallback_deadline = None
        log_fn("EVENT", f"Session rear snapshot saved source={source} path={path}")
        return True

    def _capture_rear_fallback_if_due(self, log_fn):
        deadline = self.session.rear_fallback_deadline
        if (
            not self.session.session_active
            or deadline is None
            or self.session.rear_fallback_attempted
            or time.time() < deadline
        ):
            return False
        self.session.rear_fallback_attempted = True
        saved = self._capture_and_save_rear("fallback_2s", "cam2-fallback.jpg", log_fn)
        self._update_spool_metadata(log_fn)
        return saved

    def _update_spool_metadata(self, log_fn):
        if not self.frame_spool or not self.session.spool_active:
            return
        try:
            self.frame_spool.update_active_metadata(
                self.session.session_id,
                {
                    "session_id": self.session.session_id,
                    "started_at": self.session.started_at_iso,
                    "stable_weight": self.session.last_publish_weight,
                    "decimal_pos": self.session.last_publish_decimal_pos,
                    "stability_rule": self.session.stability_rule,
                    "weight_observed_at": self.session.stable_weight_observed_at,
                    "raw_peak_weight": self._session_raw_peak,
                    "raw_peak_observed_at": self._session_raw_peak_observed_at,
                    "filtered_peak_weight": self._session_filtered_peak,
                    "filtered_peak_observed_at": self._session_filtered_peak_observed_at,
                    "vehicle_type": self.session.vehicle_type,
                    "rear_start_path": self.session.rear_start_path,
                    "rear_captured_at": self.session.rear_captured_at,
                    "rear_capture_source": self.session.rear_capture_source,
                    "start_frame_paths": dict(self.session.start_frame_paths),
                },
            )
        except Exception as exc:
            log_fn("ERROR", f"Session frame spool metadata update failed: {exc}")

    def _end_session(self, reason: str, log_fn):
        if not self.session.session_active:
            return

        if self._pending_terminal_snapshot is None:
            self._pending_terminal_snapshot = self._snapshot_session(reason)
        metadata = dict(self._pending_terminal_snapshot)
        end_reason = metadata["end_reason"]
        log_fn("EVENT", f"Session end id={self.session.session_id} reason={reason} weight_source={metadata['weight_source']}")
        queued = False
        if self.frame_spool and self.session.spool_active:
            try:
                self.frame_spool.update_active_metadata(self.session.session_id, metadata)
                self.frame_spool.end_session(self.session.session_id, metadata)
                queued = True
                self.fatal_error = None
                if self._spool_failure_logged_for == metadata["session_id"]:
                    log_metric(
                        log_fn, "session_spool_finalization_recovered",
                        id=metadata["session_id"], started_at=metadata["started_at"],
                        ended_at=metadata["ended_at"],
                    )
            except Exception as exc:
                self.fatal_error = "Session frame spool finalization failed: %s" % exc
                log_fn("FATAL", self.fatal_error)
                if self._spool_failure_logged_for != metadata["session_id"]:
                    log_metric(
                        log_fn, "session_spool_finalization_failed",
                        id=metadata["session_id"], started_at=metadata["started_at"],
                        ended_at=metadata["ended_at"], error=str(exc),
                    )
                    self._spool_failure_logged_for = metadata["session_id"]
                return False
        if not queued:
            self._save_diagnostic_frames(
                NO_PLATE_DIR, self.session.session_id, self.session.lpr_start_frames,
                {"reason": "lpr_spool_unavailable", **metadata}, log_fn,
            )
        log_fn(
            "EVENT",
            f"===== SESSION END reason={end_reason} wt={self.session.stable_weight or 0:.1f}kg "
            f"plate=pending published=pending vehicle={self.session.vehicle_type or 'unknown'} "
            f"deferred_lpr={queued} lpr=off =====",
        )
        log_metric(
            log_fn, "session_end", id=metadata["session_id"],
            started_at=metadata["started_at"], ended_at=metadata["ended_at"],
            end_reason=end_reason, stable_weight_kg=metadata["stable_weight"],
            weight_source=metadata["weight_source"],
            raw_peak_weight=metadata["raw_peak_weight"],
            filtered_peak_weight=metadata["filtered_peak_weight"],
            weight_observed_at=metadata["weight_observed_at"], deferred_lpr=queued,
        )

        self.session.rearm_block_until = 0.0
        self.session.rearm_block_reason = None
        self.session.rearm_reference_weight = None
        self._attempt_wait_reference = None
        self._post_session_low = None

        self.session.session_active = False
        self._generation += 1
        self._plate_owned = False
        self._plate_absent_since = None
        self._session_raw_peak = None
        self._session_raw_peak_observed_at = None
        self._session_filtered_peak = None
        self._session_filtered_peak_observed_at = None
        self._session_weight_window.clear()
        self.session.stable_count = 0
        self.session.clear_stable_weight_history()
        self.session.last_publish_weight = None
        self.session.last_publish_decimal_pos = 0
        self.session.vehicle_type = None
        self.session.empty_since = None
        self.session.lpr_start_frames = {}
        self.session.start_frame_paths = {}
        self.session.rear_start_frame = None
        self.session.rear_captured_at = None
        self.session.rear_capture_source = None
        self.session.rear_fallback_deadline = None
        self.session.rear_fallback_attempted = False
        self.session.started_at = None
        self.session.started_at_iso = None
        self.session.session_id = None
        self.session.rear_start_path = None
        self.session.spool_active = False
        self.session.stability_rule = None
        self.session.stable_weight_observed_at = None
        self._pending_terminal_snapshot = None
        return True

    def _snapshot_session(self, reason):
        ended_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        stable = self.session.stable_weight if self.session.stable_weight and self.session.stable_weight > 0 else None
        if stable is not None:
            assigned, source = stable, "stable"
            observed_at = self.session.stable_weight_observed_at
        elif self._session_filtered_peak is not None:
            assigned, source = self._session_filtered_peak, "filtered_peak"
            observed_at = self._session_filtered_peak_observed_at
        else:
            assigned, source = self._session_raw_peak, "raw_peak" if self._session_raw_peak else "none"
            observed_at = self._session_raw_peak_observed_at
        return {
            "session_id": self.session.session_id,
            "started_at": self.session.started_at_iso,
            "ended_at": ended_at,
            "duration_s": max(0.0, time.time() - (self.session.started_at or time.time())),
            "end_reason": reason,
            "stable_weight": assigned,
            "weight_source": source,
            "raw_peak_weight": self._session_raw_peak,
            "raw_peak_observed_at": self._session_raw_peak_observed_at,
            "filtered_peak_weight": self._session_filtered_peak,
            "filtered_peak_observed_at": self._session_filtered_peak_observed_at,
            "weight_observed_at": observed_at,
            "decimal_pos": self.session.last_publish_decimal_pos,
            "vehicle_type": self.session.vehicle_type,
            "rear_start_path": self.session.rear_start_path,
            "rear_captured_at": self.session.rear_captured_at,
            "rear_capture_source": self.session.rear_capture_source,
            "start_frame_paths": dict(self.session.start_frame_paths),
        }

    def _should_skip_duplicate_publish(self, plate, stable_weight, session_started_at=None):
        with self._publish_lock:
            plate_same = plate is not None and plate == self._last_publish_plate
            recent = False
            if plate_same and session_started_at and self._last_publish_session_end:
                started = datetime.fromisoformat(session_started_at)
                previous_end = datetime.fromisoformat(self._last_publish_session_end)
                recent = 0 <= (started - previous_end).total_seconds() < SAME_PLATE_DUPLICATE_SECONDS
            if recent:
                log("MERGE", f"Same plate [{plate}] within {SAME_PLATE_DUPLICATE_SECONDS:g}s — skipping")
                return True
        return False

    def finalize_deferred_session(self, metadata, tracker, log_fn=None):
        """Finalize one immutable ended session after deferred LPR completes."""
        log_fn = log_fn or log
        session_id = metadata["session_id"]
        frame_metadata = metadata.pop("_frame_metadata", {})
        spool_started_at = metadata.pop("_spool_started_at", None)
        finalization = getSessionFinalization(session_id)
        if finalization:
            outcome, record = finalization
            outbox_event_id = record.get("outbox_event_id") if record else None
            expected_outbox_id = outbox_event_id or (session_id if outcome == "published" else None)
            if MQTT_ENABLED and expected_outbox_id and not PublishOutbox.activate(expected_outbox_id):
                log_fn(
                    "ERROR",
                    f"Finalized session missing publication evidence id={session_id} "
                    f"outbox_id={expected_outbox_id}",
                )
                return False
            log_fn("EVENT", f"Deferred session already finalized id={session_id}")
            return True
        if not metadata.get("stable_weight") or metadata["stable_weight"] <= WEIGHT_THRESHOLD:
            frames = self._load_diagnostic_frames(metadata, offset_seconds=1.0)
            saved = self._save_diagnostic_frames(
                NO_STABLE_DIR,
                session_id,
                frames,
                {"reason": "no_usable_weight", **metadata},
                log_fn,
            )
            if saved is False:
                return False
            log_fn("EVENT", f"DEFERRED END id={session_id} weight=none published=False")
            markSessionFinalized(session_id, "no_weight", {
                "event": "no_stable_attempt", "id": session_id, **metadata,
            })
            return True
        plate, score, count = tracker.get_confirmed_plate()
        if not plate:
            lpr_diagnostics = metadata.get("lpr_diagnostics") or {}
            classification = classify_lpr_failure(lpr_diagnostics)
            lpr_diagnostics["classification"] = classification
            frames = self._load_lpr_diagnostic_frames(metadata, classification)
            if not frames:
                frames = self._load_diagnostic_frames(metadata, offset_seconds=1.0)
            saved = self._save_diagnostic_frames(
                NO_PLATE_DIR,
                metadata["session_id"],
                frames,
                {**metadata, "reason": classification},
                log_fn,
            )
            if saved is False:
                return False
            log_fn(
                "EVENT",
                f"WEIGHT-BACKED LPR NO RESULT id={metadata['session_id']} "
                f"class={classification} weight={metadata['stable_weight']:g}kg "
                f"selected={lpr_diagnostics.get('selected_frames', 0)} "
                f"processed={lpr_diagnostics.get('processed_frames', 0)} "
                f"regions={lpr_diagnostics.get('detected_regions', 0)}",
            )
            unknown_plate = "UNKNOWN"
            publish_result = self._build_publish_result(
                metadata["stable_weight"], unknown_plate, 0, {}, metadata,
            )
            publish_result.update({
                "offline_event_id": session_id,
                "ocr_plate_read": None,
                "metadata": {
                    "plate_status": "unreadable",
                    "lpr_classification": classification,
                    "weight_source": metadata.get("weight_source"),
                    "raw_peak_weight": metadata.get("raw_peak_weight"),
                    "filtered_peak_weight": metadata.get("filtered_peak_weight"),
                },
            })
            publish_frames, captured_at = self._load_unknown_publish_frames(
                metadata, frame_metadata, log_fn, spool_started_at,
            )
            self._attach_unknown_publish_images(
                publish_result, publish_frames, captured_at, session_id,
            )
            image_object_keys = publish_result.pop("_image_object_keys", [])
            image_paths = publish_result.pop("_image_paths", [])
            save_items = publish_result.pop("_image_save_items", [])
            saved_photos = []
            saved_object_keys = []
            saved_paths = []
            for photo, item, object_key, image_path in zip(
                publish_result["photos"], save_items, image_object_keys, image_paths,
            ):
                if ImageSaveWorker.save_and_enqueue_upload([item]):
                    saved_photos.append(photo)
                    saved_object_keys.append(object_key)
                    saved_paths.append(image_path)
                else:
                    log_fn(
                        "WARNING",
                        f"Unknown publication image dropped id={session_id} "
                        f"camera={photo['type']}",
                    )
            publish_result["photos"] = saved_photos
            image_object_keys = saved_object_keys
            image_paths = saved_paths
            outbox_event_id = None
            if MQTT_ENABLED and self.mqtt_svc:
                outbox_event_id = PublishOutbox.enqueue(
                    publish_result,
                    image_object_keys=image_object_keys,
                    image_paths=image_paths,
                    activate=False,
                )
                log_fn(
                    "OFFLINE",
                    f"Weight event queued without plate id={session_id} "
                    f"weight={metadata['stable_weight']:g}kg",
                )
            terminal_record = {
                "event": "session_publish_queued", "id": session_id,
                "plate": unknown_plate, "plate_status": "unreadable",
                "classification": classification,
                "stable_weight_kg": metadata["stable_weight"], **metadata,
            }
            if outbox_event_id:
                terminal_record["outbox_event_id"] = outbox_event_id
            markSessionFinalized(session_id, "published", terminal_record)
            if outbox_event_id and not PublishOutbox.activate(outbox_event_id):
                log_fn(
                    "ERROR",
                    f"Session publication evidence missing id={session_id} "
                    f"outbox_id={outbox_event_id}",
                )
                return False
            log_metric(
                log_fn, "session_no_plate", id=metadata["session_id"],
                started_at=metadata["started_at"], ended_at=metadata["ended_at"],
                end_reason=metadata["end_reason"],
                stable_weight_kg=metadata["stable_weight"], images=saved,
                weight_source=metadata.get("weight_source"),
                raw_peak_weight=metadata.get("raw_peak_weight"),
                filtered_peak_weight=metadata.get("filtered_peak_weight"),
                weight_observed_at=metadata.get("weight_observed_at"),
                recovered_after_restart=bool(metadata.get("recovered_after_restart")),
                incomplete=bool(metadata.get("incomplete")), errors=metadata.get("errors", []),
                classification=classification, lpr_diagnostics=lpr_diagnostics,
            )
            log_metric(
                log_fn, "weight_backed_lpr_no_result",
                id=metadata["session_id"], classification=classification,
                stable_weight_kg=metadata["stable_weight"],
                lpr_diagnostics=lpr_diagnostics,
            )
            log_metric(
                log_fn, "session_publish_queued", id=session_id,
                plate=unknown_plate, plate_status="unreadable",
                stable_weight_kg=metadata["stable_weight"],
                classification=classification,
            )
            return True
        result = self.publish_result(
            metadata["stable_weight"], metadata["decimal_pos"], log_fn,
            tracker=tracker, metadata=metadata,
        )
        if result is False:
            log_fn("EVENT", f"DEFERRED END id={metadata['session_id']} plate={plate} published=False")
            return False
        plate = result["plate"]
        duplicate = result["status"] == "duplicate"
        state = "duplicate" if duplicate else True
        log_fn("EVENT", f"DEFERRED END id={metadata['session_id']} plate={plate} published={state}")
        event = "session_duplicate" if duplicate else "session_publish_queued"
        terminal_record = {
            "event": event, "id": session_id, "plate": plate,
            "stable_weight_kg": metadata["stable_weight"], **metadata,
        }
        outbox_event_id = result.get("outbox_event_id")
        if outbox_event_id:
            terminal_record["outbox_event_id"] = outbox_event_id
        markSessionFinalized(
            session_id, "duplicate" if duplicate else "published",
            terminal_record,
        )
        if outbox_event_id:
            if not PublishOutbox.activate(outbox_event_id):
                log_fn(
                    "ERROR",
                    f"Session publication evidence missing id={session_id} outbox_id={outbox_event_id}",
                )
                return False
        log_metric(
            log_fn, event, id=metadata["session_id"], plate=plate,
            started_at=metadata["started_at"], ended_at=metadata["ended_at"],
            stable_weight_kg=metadata["stable_weight"],
            weight_source=metadata.get("weight_source"),
            raw_peak_weight=metadata.get("raw_peak_weight"),
            filtered_peak_weight=metadata.get("filtered_peak_weight"),
            weight_observed_at=metadata.get("weight_observed_at"),
            recovered_after_restart=bool(metadata.get("recovered_after_restart")),
            incomplete=bool(metadata.get("incomplete")), errors=metadata.get("errors", []),
        )
        return True

    @staticmethod
    def _load_start_frames(metadata):
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

    @staticmethod
    def _nearest_session_frame(metadata, camera, observed_at):
        session_dir = metadata.get("session_dir")
        files = metadata.get("session_files", [])
        if not session_dir or observed_at is None:
            return None
        started_at = datetime.fromisoformat(metadata["started_at"]).timestamp()
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

    def _load_diagnostic_frames(self, metadata, offset_seconds):
        target = datetime.fromisoformat(metadata["started_at"]).timestamp() + offset_seconds
        frames = {}
        for camera in ("cam1", "cam3"):
            path = self._nearest_session_frame(metadata, camera, target)
            frame = cv2.imread(path) if path else None
            if frame is not None:
                frames[camera] = frame
        return frames or self._load_start_frames(metadata)

    def _load_unknown_publish_frames(
        self, metadata, frame_metadata, log_fn, spool_started_at=None,
    ):
        target_at = (
            metadata.get("weight_observed_at")
            or metadata.get("filtered_peak_observed_at")
            or metadata.get("raw_peak_observed_at")
            or metadata.get("ended_at")
        )
        try:
            target_ts = datetime.fromisoformat(target_at).timestamp()
        except (TypeError, ValueError):
            log_fn("WARNING", f"Unknown photo timing unavailable id={metadata['session_id']}")
            return {}, {}

        session_dir = metadata.get("session_dir")
        started_at = spool_started_at or metadata.get("started_at")
        if not session_dir or not started_at:
            return {}, {}
        session_dir = os.path.abspath(session_dir)
        started_ts = datetime.fromisoformat(started_at).timestamp()
        interval = float(metadata.get("capture_interval_seconds", 0.2))
        selected = {}
        captured_at = {}
        selection = {}
        missing = []
        for camera in ("cam1", "cam2", "cam3"):
            candidates = []
            first_seen_by_frame_id = {}
            for relative_path in metadata.get("session_files", []):
                if not relative_path.startswith(camera + "-"):
                    continue
                if relative_path.endswith("-start.jpg"):
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
                frame_id = item_metadata.get("frame_id")
                if frame_id is not None:
                    observed_ts, observed_iso = first_seen_by_frame_id.setdefault(
                        frame_id, (observed_ts, observed_iso),
                    )
                candidates.append((abs(observed_ts - target_ts), relative_path, observed_iso))
            if not candidates:
                missing.append(camera)
                continue
            chosen = None
            for offset, relative_path, observed_iso in sorted(candidates):
                if offset > UNKNOWN_PHOTO_MAX_OFFSET_SECONDS:
                    break
                path = os.path.abspath(os.path.join(session_dir, relative_path))
                if os.path.commonpath((session_dir, path)) != session_dir:
                    continue
                frame = cv2.imread(path)
                if frame is not None:
                    chosen = frame, observed_iso, offset
                    break
            if chosen is None:
                missing.append(camera)
                continue
            frame, observed_iso, offset = chosen
            selected[camera] = frame
            captured_at[camera] = observed_iso
            selection[camera] = {
                "captured_at": observed_iso,
                "offset_ms": round(offset * 1000),
            }
        log_metric(
            log_fn, "unknown_photo_selection", id=metadata["session_id"],
            target_at=target_at, weight_source=metadata.get("weight_source"),
            selected=selection, missing_cameras=missing,
        )
        return selected, captured_at

    @staticmethod
    def _load_lpr_diagnostic_frames(metadata, classification):
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

    def publish_result(self, stable_weight, decimal_pos, log_fn, tracker=None, metadata=None):
        """Query PlateTracker and publish if plate is confirmed."""
        tracker = tracker or self.plate_tracker
        metadata = metadata or {}
        plate, score, count = tracker.get_confirmed_plate()
        image_lookup_plate = plate
        all_plates = tracker.get_all_plates_summary()
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
        if self._should_skip_duplicate_publish(plate, stable_weight, metadata.get("started_at")):
            return {"status": "duplicate", "plate": plate}

        result = self._build_publish_result(stable_weight, plate, count, all_plates, metadata)
        if metadata.get("session_id"):
            result["offline_event_id"] = metadata["session_id"]
        self._log_publish_summary(stable_weight, decimal_pos, plate, score, count, all_plates, log_fn)
        image_aliases = [image_lookup_plate, *all_plates.keys()]
        if not self._attach_publish_images(
            result, stable_weight, decimal_pos, plate, image_aliases, log_fn,
            tracker=tracker, rear_start_path=metadata.get("rear_start_path"),
            start_frame_paths=metadata.get("start_frame_paths", {}),
            session_dir=metadata.get("session_dir"),
            session_files=metadata.get("session_files", []),
            capture_interval_seconds=metadata.get("capture_interval_seconds", 0.2),
            session_started_at=metadata.get("started_at"),
            session_id=metadata.get("session_id"),
        ):
            log_fn("ERROR", f"Publish skipped — image capture failed for plate={plate}")
            return False

        image_object_keys = result.pop("_image_object_keys", [])
        image_paths = result.pop("_image_paths", [])
        save_items = result.pop("_image_save_items", [])
        if save_items and not ImageSaveWorker.save_and_enqueue_upload(save_items):
            log_fn("ERROR", f"Publish skipped — local image save failed for plate={plate}")
            return False

        saved_count = saveConfirmedLicensePlate(plate, metadata.get("session_id"))
        if saved_count is not None:
            log_fn("PLATE_DB", f"Saved confirmed plate={plate} recognition_count={saved_count}")

        outbox_event_id = None
        if MQTT_ENABLED and self.mqtt_svc:
            outbox_event_id = PublishOutbox.enqueue(
                result, image_object_keys=image_object_keys, image_paths=image_paths,
                activate=False,
            )
            log_fn("OFFLINE", f"Publish queued offline id={result.get('offline_event_id')} plate={plate}")

        with self._publish_lock:
            self._last_publish_plate = plate
            self._last_publish_weight = stable_weight
            self._last_publish_session_end = metadata.get("ended_at")
            try:
                self._save_dedup_state()
            except OSError as exc:
                log_fn("ERROR", f"Session dedup state save failed: {exc}")
        return {"status": "published", "plate": plate, "outbox_event_id": outbox_event_id}

    def _build_publish_result(self, stable_weight, plate, count, all_plates, metadata=None):
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

    def _log_publish_summary(self, stable_weight, decimal_pos, plate, score, count, all_plates, log_fn):
        plate_text = plate or "none"
        candidates = ", ".join(f"{p}:{c}" for p, c in sorted(all_plates.items())) if all_plates else "-"
        log_fn(
            "EVENT",
            f"PUBLISH wt={stable_weight:.{decimal_pos}f}kg plate={plate_text} score={score:.2f} hits={count} candidates=[{candidates}]",
        )

    def _prepare_capture_paths(self, now, plate, session_id=None):
        date_path = now.strftime("%Y/%m/%d")
        day_dir = os.path.join(CAPTURE_DIR, now.strftime("%Y"), now.strftime("%m"), now.strftime("%d"))
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
            "unchosen_cam1": _make("photo-unchosen-cam1"),
            "unchosen_cam3": _make("photo-unchosen-cam3"),
        }

    def _attach_unknown_publish_images(self, result, frames, captured_at, session_id):
        available = [
            camera for camera in ("cam1", "cam2", "cam3")
            if frames.get(camera) is not None
        ]
        result["photos"] = []
        result["_image_object_keys"] = []
        result["_image_paths"] = []
        result["_image_save_items"] = []
        if not available:
            return

        paths = self._prepare_capture_paths(datetime.now(), "UNKNOWN", session_id)
        for camera in available:
            frame = frames[camera]
            if camera == "cam2":
                frame = self._crop_cam2_result_image(frame)
            fpath, object_key, url = paths[camera]
            result["photos"].append({
                "url": url, "type": camera, "captured_at": captured_at.get(camera),
            })
            result["_image_object_keys"].append(object_key)
            result["_image_paths"].append(fpath)
            result["_image_save_items"].append([fpath, frame, object_key])

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

    def _attach_publish_images(
        self, result, stable_weight, decimal_pos, plate, image_aliases, log_fn,
        tracker=None, rear_start_path=None, start_frame_paths=None,
        session_dir=None, session_files=None, capture_interval_seconds=0.2,
        session_started_at=None,
        session_id=None,
    ):
        import numpy as np
        attach_started_at = time.time()

        tracker = tracker or self.plate_tracker
        frame, img_plate, camera_name, observed_at = tracker.get_image_frame(
            plate, aliases=image_aliases
        )
        if frame is None or not plate:
            return False
        if img_plate != plate:
            log_fn("SAVE", f"Using image plate {img_plate} for final plate {plate}")

        now = datetime.now()
        captured_at = now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        paths = self._prepare_capture_paths(now, plate, session_id)
        rear_frame = cv2.imread(rear_start_path) if rear_start_path else None
        if rear_frame is None and not rear_start_path:
            rear_frame = self.session.rear_start_frame
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
            unchosen_frame = None
            if start_frame_paths and start_frame_paths.get(unchosen_camera):
                unchosen_frame = cv2.imread(start_frame_paths[unchosen_camera])
            elif not start_frame_paths:
                unchosen_frame = self.session.lpr_start_frames.get(unchosen_camera)
            unchosen_key = f"unchosen_{unchosen_camera}"
            if unchosen_frame is not None and unchosen_key in paths:
                if ImageSaveWorker.save_local_only(paths[unchosen_key][0], unchosen_frame):
                    log_fn("SAVE", f"Saved local-only unchosen LPR start image camera={unchosen_camera} plate={plate}")
                else:
                    log_fn("WARNING", f"Failed local-only unchosen LPR start image camera={unchosen_camera} plate={plate}")

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
            if not cv2.imwrite(fpath, frame_data):
                raise OSError("cv2.imwrite returned false")
            log_fn("SAVE", f"Undetectable saved: {fpath}")
            return True
        except Exception as exc:
            log_fn("ERROR", f"Failed to save undetectable image: {exc}")
            return False
        finally:
            del frame_data
