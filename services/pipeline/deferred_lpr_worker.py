"""Deferred license plate recognition for finalized session frame spools."""

import json
import os
import threading
import time
from datetime import datetime

import cv2 as _cv2

from services.capture.detect_coordinator import crop_lpr_frame, remap_lpr_regions
from services.pipeline.license_plate_recognition import (
    detect_plate_regions,
    recognize_plate_regions,
)
from services.tracking.plate_tracker import PlateTracker


class DeferredLprWorker:
    """Consume finalized spool jobs in FIFO order on one background thread."""

    def __init__(
        self,
        spool,
        cameras,
        charset,
        callback,
        detect_regions_fn=detect_plate_regions,
        recognize_regions_fn=recognize_plate_regions,
        tracker_factory=PlateTracker,
        cv2_module=_cv2,
        poll_interval=0.1,
        failed_retry_delay=1.0,
        max_retries=3,
        log_fn=None,
    ):
        self._spool = spool
        self._cameras = {camera.name: camera for camera in cameras if camera.name in ("cam1", "cam3")}
        self._charset = charset
        self._callback = callback
        self._detect_regions = detect_regions_fn
        self._recognize_regions = recognize_regions_fn
        self._tracker_factory = tracker_factory
        self._cv2 = cv2_module
        self._poll_interval = poll_interval
        self._failed_retry_delay = failed_retry_delay
        self._max_retries = max_retries
        self._log_fn = log_fn
        self._stop_event = threading.Event()
        self._thread = None
        self._failed_until = {}
        self._retry_counts = {}
        self._state_lock = threading.Lock()
        self._current_job = None
        self._last_error = None

    def start(self):
        """Start the worker. Safe to call more than once."""
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run, name="deferred-lpr-worker", daemon=True
            )
            self._thread.start()

    def stop(self, timeout=5.0):
        """Request clean shutdown and report whether the thread stopped."""
        self._stop_event.set()
        thread = self._thread
        if thread:
            thread.join(timeout)
        return not thread or not thread.is_alive()

    def status(self):
        """Return a thread-safe worker status snapshot."""
        with self._state_lock:
            thread = self._thread
            return {
                "running": bool(thread and thread.is_alive()),
                "current_job": self._current_job,
                "failed_jobs": len(self._failed_until),
                "last_error": self._last_error,
            }

    def _run(self):
        current_path = None
        while not self._stop_event.is_set():
            path = current_path or self._spool.get_pending_job(timeout=self._poll_interval)
            if path is None:
                continue
            retry_at = self._failed_until.get(path, 0.0)
            if retry_at > time.monotonic():
                self._stop_event.wait(min(self._poll_interval, retry_at - time.monotonic()))
                current_path = path
                continue
            with self._state_lock:
                self._current_job = path
            try:
                self._process_job(path)
                self._spool.acknowledge_job(path)
                self._failed_until.pop(path, None)
                self._retry_counts.pop(path, None)
                current_path = None
            except Exception as exc:
                self._retry_counts[path] = self._retry_counts.get(path, 0) + 1
                self._failed_until[path] = time.monotonic() + self._failed_retry_delay
                with self._state_lock:
                    self._last_error = "%s: %s" % (type(exc).__name__, exc)
                self._log("ERROR", "Deferred LPR job failed [%s]: %s" % (path, exc))
                self._log(
                    "METRIC",
                    json.dumps(
                        {"event": "session_processing_failed", "manifest": path,
                         "error": "%s: %s" % (type(exc).__name__, exc)},
                        separators=(",", ":"), sort_keys=True,
                    ),
                )
                if self._retry_counts[path] >= self._max_retries:
                    failed_path = self._spool.fail_job(path)
                    self._log(
                        "METRIC",
                        json.dumps(
                            {"event": "session_processing_dead_lettered",
                             "manifest": failed_path, "retries": self._retry_counts[path]},
                            separators=(",", ":"), sort_keys=True,
                        ),
                    )
                    self._failed_until.pop(path, None)
                    self._retry_counts.pop(path, None)
                    current_path = None
                else:
                    current_path = path
            finally:
                with self._state_lock:
                    self._current_job = None

    def _process_job(self, manifest_path):
        with open(manifest_path, encoding="utf-8") as handle:
            job = json.load(handle)
        session_dir = job["session_dir"]
        files = job["files"]
        metadata = job["metadata"]
        if not isinstance(files, list) or not isinstance(metadata, dict):
            raise ValueError("manifest files and metadata have invalid types")

        tracker = self._tracker_factory()
        successful_frames = 0
        started_at = datetime.fromisoformat(job["started_at"]).timestamp()
        interval = float(job.get("capture_interval_seconds", 0.2))
        camera_indexes = {"cam1": 0, "cam3": 0}
        for relative_path in files:
            if self._stop_event.is_set():
                raise RuntimeError("worker stopped during job")
            try:
                camera_name = relative_path.split("-", 1)[0]
                observed_at = started_at + camera_indexes.get(camera_name, 0) * interval
                camera_indexes[camera_name] = camera_indexes.get(camera_name, 0) + 1
                if self._process_frame(session_dir, relative_path, tracker, observed_at):
                    successful_frames += 1
            except Exception as exc:
                self._log(
                    "ERROR",
                    "Deferred LPR frame failed [%s/%s]: %s"
                    % (job.get("session_id", "?"), relative_path, exc),
                )
        if files and successful_frames == 0:
            raise RuntimeError("no session frames processed successfully")
        if self._callback(metadata, tracker) is False:
            raise RuntimeError("deferred session finalization failed")

    def _process_frame(self, session_dir, relative_path, tracker, observed_at):
        if not isinstance(relative_path, str):
            raise ValueError("frame path is not a string")
        camera_name = relative_path.split("-", 1)[0]
        camera = self._cameras.get(camera_name)
        if camera is None:
            return
        path = os.path.abspath(os.path.join(session_dir, relative_path))
        if os.path.commonpath((os.path.abspath(session_dir), path)) != os.path.abspath(session_dir):
            raise ValueError("frame is outside session directory")
        frame = self._cv2.imread(path)
        if frame is None:
            raise ValueError("JPEG decode failed")

        cropped, dx, dy = crop_lpr_frame(frame, camera.lpr_crop)
        regions = self._detect_regions(cropped, detector=camera.detector)
        regions = remap_lpr_regions(regions, dx, dy)
        plates = self._recognize_regions(regions, ocr=camera.ocr, charset=self._charset)
        self._update_tracker(tracker, plates, frame, camera_name, observed_at)
        return True

    @staticmethod
    def _update_tracker(tracker, plates, frame, camera_name, observed_at):
        best_plate = None
        best_conf = 0.0
        has_unknown = False
        for plate in plates:
            text = plate["plate"]
            width, height = (int(value) for value in plate["crop_size"].split("x"))
            if text != "unknown":
                tracker.add_observation(
                    text, plate["det_conf"], width, height,
                    source="selected", observed_at=observed_at,
                )
                candidates = plate.get("valid_candidates", [])[1:]
                confidence_scale = 0.5
                if plate["det_conf"] > best_conf:
                    best_plate, best_conf = text, plate["det_conf"]
            else:
                has_unknown = True
                candidates = plate.get("valid_candidates", [])
                confidence_scale = 0.75
            for candidate, _ocr_conf in candidates:
                confidence = plate["det_conf"] * confidence_scale
                tracker.add_observation(
                    candidate, confidence, width, height,
                    source="candidate", observed_at=observed_at,
                )
                tracker.update_image(candidate, confidence, frame, camera_name)
                if best_plate is None:
                    best_plate, best_conf = candidate, confidence
        if has_unknown and best_plate is None and tracker.needs_undetectable():
            tracker.save_undetectable(frame.copy())
        if best_plate is not None:
            tracker.update_image(best_plate, best_conf, frame, camera_name)

    def _log(self, level, message):
        if self._log_fn:
            self._log_fn(level, message)
