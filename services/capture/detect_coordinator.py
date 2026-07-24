"""Detection coordinators for LPR and vehicle detection."""

import threading
import time

from config import DETECT_FPS, PLATE_TRACK_STALE_SECONDS, YOLO26_DETECT_FPS


def bbox_iou(left, right):
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = max(0, left[2] - left[0]) * max(0, left[3] - left[1]) + max(0, right[2] - right[0]) * max(0, right[3] - right[1]) - intersection
    return intersection / union if union else 0.0


class CameraPlateTrack:
    """Camera-local detector track. Missing/error observations leave state untouched."""

    def __init__(self, camera, confirm_hits=2, iou_threshold=0.3):
        self.camera = camera
        self.confirm_hits = confirm_hits
        self.iou_threshold = iou_threshold
        self.track_id = 0
        self.bbox = None
        self.confidence = 0.0
        self.hits = 0
        self.valid = False
        self.frame_id = None
        self.last_observed_at = None

    def observe(self, regions, frame_id=None, observed_at=None):
        self.last_observed_at = time.monotonic() if observed_at is None else observed_at
        best = max(regions, key=lambda item: item.get("det_conf", item.get("confidence", 0.0)), default=None)
        if best is None:
            self.bbox = None
            self.confidence = 0.0
            self.hits = 0
            self.valid = False
            self.frame_id = frame_id
            return
        bbox = list(best["bbox"])
        if self.bbox is None or bbox_iou(self.bbox, bbox) < self.iou_threshold:
            self.track_id += 1
            self.hits = 1
        else:
            self.hits += 1
        self.bbox = bbox
        self.confidence = float(best.get("det_conf", best.get("confidence", 0.0)))
        self.valid = self.hits >= self.confirm_hits
        self.frame_id = frame_id

    def expire(self, now=None, stale_seconds=PLATE_TRACK_STALE_SECONDS):
        if not self.valid or self.last_observed_at is None:
            return False
        now = time.monotonic() if now is None else now
        if now - self.last_observed_at < stale_seconds:
            return False
        self.bbox = None
        self.confidence = 0.0
        self.hits = 0
        self.valid = False
        self.frame_id = None
        return True

    def metadata(self):
        if not self.valid:
            return []
        return [{"bbox": list(self.bbox), "track_id": self.track_id,
                 "confidence": self.confidence, "frame_id": self.frame_id}]


def crop_lpr_frame(frame, mode):
    h, w = frame.shape[:2]
    if mode == "full":
        return frame, 0, 0
    x2 = (2 * w) // 3
    y_mid = h // 2
    if mode == "bottom_left_two_thirds":
        return frame[y_mid:h, :x2], 0, y_mid
    if mode == "top_left_two_thirds":
        return frame[:y_mid, :x2], 0, 0
    raise ValueError(f"Invalid LPR crop mode: {mode!r}")


def remap_lpr_regions(regions, dx, dy):
    if not dx and not dy:
        return regions
    remapped = []
    for region in regions:
        item = dict(region)
        if item.get("bbox"):
            item["bbox"] = [item["bbox"][0] + dx, item["bbox"][1] + dy,
                             item["bbox"][2] + dx, item["bbox"][3] + dy]
        if item.get("obb"):
            item["obb"] = [[point[0] + dx, point[1] + dy] for point in item["obb"]]
        remapped.append(item)
    return remapped

_log_fn = None


def set_log_fn(log_fn):
    """Set the logging function to use."""
    global _log_fn
    _log_fn = log_fn


def log(level: str, msg: str):
    """Log using the configured log function."""
    if _log_fn:
        _log_fn(level, msg)


class DetectCoordinator:
    """Run one persistent detector and OCR worker per camera."""

    def __init__(self, cameras: list, tracker, detect_plates_fn=None, presence_callback=None, session_context=None):
        self._cameras = cameras
        self._tracker = tracker
        self._detect_plates_fn = detect_plates_fn
        self._presence_callback = presence_callback
        self._session_context = session_context or (lambda: (None, None))
        self._tracks = {cam.name: CameraPlateTrack(cam.name) for cam in cameras}
        self._track_lock = threading.Lock()
        self._presence_revision = 0
        self._detect_regions_fn = None
        self._recognize_regions_fn = None
        self._charset = None
        self._running = False
        self._enabled = False
        self._state_lock = threading.Lock()
        self._ocr_jobs = {}
        self._ocr_locks = {}
        self._ocr_events = {}
        self._detect_thread = None
        self._worker_threads = []
        self._detect_events = {}
        self._detect_jobs = {}
        self._detect_locks = {}
        self._last_submitted_frame_ids = {}
        self._ocr_threads = []

    def configure_split_pipeline(self, detect_regions_fn, recognize_regions_fn, charset):
        self._detect_regions_fn = detect_regions_fn
        self._recognize_regions_fn = recognize_regions_fn
        self._charset = charset

    def start(self):
        if self._running:
            return
        self._running = True
        for cam in self._cameras:
            self._detect_events[cam.name] = threading.Event()
            self._detect_locks[cam.name] = threading.Lock()
            self._detect_jobs[cam.name] = None
            thread = threading.Thread(target=self._camera_worker_loop, args=(cam,), daemon=True)
            thread.start()
            self._worker_threads.append(thread)
        if self._detect_regions_fn and self._recognize_regions_fn:
            for cam in self._cameras:
                self._ocr_jobs[cam.name] = None
                self._ocr_locks[cam.name] = threading.Lock()
                self._ocr_events[cam.name] = threading.Event()
                thread = threading.Thread(target=self._ocr_loop, args=(cam,), daemon=True)
                thread.start()
                self._ocr_threads.append(thread)
        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._detect_thread.start()
        log("INFO", f"DetectCoordinator started with {len(self._cameras)} cameras")

    def stop(self, timeout=5.0):
        deadline = time.time() + timeout
        self._running = False
        for event in self._ocr_events.values():
            event.set()
        for event in self._detect_events.values():
            event.set()
        for name, lock in self._ocr_locks.items():
            with lock:
                self._ocr_jobs[name] = None
        for thread in [self._detect_thread, *self._worker_threads]:
            if thread and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.time()))
        for thread in self._ocr_threads:
            if thread and thread.is_alive():
                thread.join(timeout=max(0.0, deadline - time.time()))
        threads = [self._detect_thread, *self._worker_threads, *self._ocr_threads]
        return all(not thread or not thread.is_alive() for thread in threads)

    def set_enabled(self, enabled: bool):
        with self._state_lock:
            self._enabled = enabled
        if not enabled:
            for name, lock in self._ocr_locks.items():
                with lock:
                    self._ocr_jobs[name] = None

    def is_enabled(self):
        with self._state_lock:
            return self._enabled

    def _camera_worker_loop(self, cam):
        event = self._detect_events[cam.name]
        while self._running:
            event.wait(0.1)
            event.clear()
            if not self._running:
                break
            with self._detect_locks[cam.name]:
                job = self._detect_jobs[cam.name]
                self._detect_jobs[cam.name] = None
            if job is not None:
                frame, frame_id = job
                try:
                    self._run_detection(cam, frame, frame_id)
                except Exception as exc:
                    log("ERROR", f"Worker detection error [{cam.name}]: {exc}")

    def get_frame_metadata(self, camera, frame_id=None):
        with self._track_lock:
            metadata = self._tracks[camera].metadata() if camera in self._tracks else []
            if frame_id is None or not metadata or metadata[0]["frame_id"] != frame_id:
                return []
            return metadata

    def _process_plate_detections(self, cam, plates, full_frame):
        """Process plate detections: log, update tracker, capture best plate and unknown frame."""
        if not self.is_enabled():
            return
        best_conf = 0.0
        best_plate = None
        has_unknown = False
        for p in plates:
            plate_text = p["plate"]
            is_known = plate_text != "unknown"
            if is_known:
                log(
                    "PLATE",
                    f"[{cam.name}] >> {plate_text:<16} det_conf={p['det_conf']:.3f} "
                    f"crop={p['crop_size']} votes={p['votes']}",
                )
                crop_parts = p["crop_size"].split("x")
                cw, ch = int(crop_parts[0]), int(crop_parts[1])
                self._tracker.add_observation(plate_text, p["det_conf"], cw, ch, source="selected")
                for alt_plate, _ in p.get("valid_candidates", [])[1:]:
                    self._tracker.add_observation(alt_plate, p["det_conf"] * 0.5, cw, ch, source="candidate")
                    self._tracker.update_image(alt_plate, p["det_conf"] * 0.5, full_frame, cam.name)
                if p["det_conf"] > best_conf:
                    best_conf = p["det_conf"]
                    best_plate = plate_text
            else:
                has_unknown = True
                log(
                    "PLATE",
                    f"[{cam.name}] unknown status={p.get('ocr_status')} crop={p['crop_size']} "
                    f"candidates={','.join(p.get('candidates', [])[:5])}",
                )
                crop_parts = p["crop_size"].split("x")
                cw, ch = int(crop_parts[0]), int(crop_parts[1])
                for alt_plate, _ in p.get("valid_candidates", []):
                    self._tracker.add_observation(alt_plate, p["det_conf"] * 0.75, cw, ch, source="candidate")
                    self._tracker.update_image(alt_plate, p["det_conf"] * 0.75, full_frame, cam.name)
                    if best_plate is None:
                        best_plate = alt_plate
                        best_conf = p["det_conf"] * 0.75
        del plates
        if has_unknown and best_plate is None and self._tracker.needs_undetectable():
            self._tracker.save_undetectable(full_frame.copy())
        if best_plate is not None:
            self._tracker.update_image(best_plate, best_conf, full_frame, cam.name)

    def _submit_ocr_job(self, cam, full_frame, regions, detect_started_at):
        lock = self._ocr_locks.get(cam.name)
        event = self._ocr_events.get(cam.name)
        if lock is None or event is None:
            return
        with lock:
            self._ocr_jobs[cam.name] = {
                "frame": full_frame,
                "regions": regions,
                "created_at": time.time(),
                "detect_started_at": detect_started_at,
                "session_context": self._session_context(),
            }
        event.set()

    def _take_ocr_job(self, cam):
        lock = self._ocr_locks.get(cam.name)
        if lock is None:
            return None
        with lock:
            job = self._ocr_jobs.get(cam.name)
            self._ocr_jobs[cam.name] = None
        return job

    def _ocr_loop(self, cam):
        event = self._ocr_events[cam.name]
        while self._running:
            event.wait(0.1)
            event.clear()
            if not self._running:
                break
            job = self._take_ocr_job(cam)
            if job is None:
                continue
            full_frame = job["frame"]
            regions = job["regions"]
            try:
                if not self.is_enabled():
                    continue
                t0 = time.time()
                with cam.inference_lock:
                    plates = self._recognize_regions_fn(regions, ocr=cam.ocr, charset=self._charset)
                if not self.is_enabled():
                    continue
                if job["session_context"] != self._session_context():
                    log("INFO", f"[{cam.name}] stale OCR result ignored context={job['session_context']}")
                    continue
                elapsed_ms = (time.time() - t0) * 1000
                age_ms = (t0 - job["created_at"]) * 1000
                if plates:
                    if len(plates) == 1:
                        p = plates[0]
                        candidates = ",".join(p.get("candidates", [])[:5])
                        log(
                            "TIMING",
                            f"[{cam.name}] OCR: {elapsed_ms:.0f}ms  plates=1 age={age_ms:.0f}ms "
                            f"plate={p.get('plate')} status={p.get('ocr_status')} candidates={candidates}",
                        )
                    else:
                        summary = ",".join(p.get("plate", "unknown") for p in plates[:5])
                        log("TIMING", f"[{cam.name}] OCR: {elapsed_ms:.0f}ms  plates={len(plates)} age={age_ms:.0f}ms plate={summary}")
                self._process_plate_detections(cam, plates, full_frame)
            except Exception as exc:
                log("ERROR", f"OCR worker error [{cam.name}]: {exc}")
            finally:
                del full_frame
                del regions

    def _run_detection(self, cam, full_frame, frame_id=None):
        """Run detection for one camera. Feeds tracker directly."""
        t0 = time.time()
        if self._detect_regions_fn and self._recognize_regions_fn:
            lpr_frame, dx, dy = crop_lpr_frame(full_frame, cam.lpr_crop)
            with cam.inference_lock:
                regions = self._detect_regions_fn(lpr_frame, detector=cam.detector)
            regions = remap_lpr_regions(regions, dx, dy)
            with self._track_lock:
                self._tracks[cam.name].observe(regions, frame_id)
                valid = {name: track.valid for name, track in self._tracks.items()}
                self._presence_revision += 1
                revision = self._presence_revision
            if self._presence_callback:
                self._presence_callback(cam.name, valid, revision)
            elapsed_ms = (time.time() - t0) * 1000
            log("TIMING", f"[{cam.name}] Detect: {elapsed_ms:.0f}ms regions={len(regions)} "
                           f"lpr_crop={cam.lpr_crop} source={full_frame.shape[1]}x{full_frame.shape[0]} "
                           f"input={lpr_frame.shape[1]}x{lpr_frame.shape[0]}")
            self._submit_ocr_job(cam, full_frame, regions, t0)
            return
        plates = self._detect_plates_fn(full_frame, detector=cam.detector, ocr=cam.ocr)
        elapsed_ms = (time.time() - t0) * 1000
        if plates:
            log("TIMING", f"[{cam.name}] Frame inference: {elapsed_ms:.0f}ms  plates={len(plates)}")
        self._process_plate_detections(cam, plates, full_frame)

    def _expire_stale_tracks(self):
        with self._track_lock:
            changed = False
            for track in self._tracks.values():
                changed = track.expire() or changed
            if not changed:
                return None
            valid = {name: track.valid for name, track in self._tracks.items()}
            self._presence_revision += 1
            return valid, self._presence_revision

    def _detect_loop(self):
        interval = 1.0 / DETECT_FPS
        while self._running:
            if not self.is_enabled():
                time.sleep(0.05)
                continue
            expired = self._expire_stale_tracks()
            if expired and self._presence_callback:
                valid, revision = expired
                self._presence_callback(None, valid, revision)

            frames = []
            for cam in self._cameras:
                peek = getattr(cam, "peek_latest_frame_with_id", None)
                frame, frame_id = peek(copy_frame=True) if peek else (
                    cam.peek_latest_frame(copy_frame=True), None
                )
                frames.append((cam, frame, frame_id))
            if not any(frame is not None for _cam, frame, _frame_id in frames):
                time.sleep(0.01)
                continue

            t0 = time.time()
            try:
                for cam, frame, frame_id in frames:
                    if frame is None:
                        continue
                    if (
                        frame_id is not None
                        and self._last_submitted_frame_ids.get(cam.name) == frame_id
                    ):
                        continue
                    self._last_submitted_frame_ids[cam.name] = frame_id
                    with self._detect_locks[cam.name]:
                        self._detect_jobs[cam.name] = (frame, frame_id)
                    self._detect_events[cam.name].set()
            except Exception as exc:
                log("ERROR", f"DetectCoordinator pipeline error: {exc}")
            finally:
                frames = []

            elapsed_total = time.time() - t0
            time.sleep(max(0.0, interval - elapsed_total))


class VehicleDetectCoordinator:
    def __init__(self, cameras: list, tracker, detector=None, detect_vehicles_fn=None):
        self._cameras = cameras
        self._tracker = tracker
        self._detector = detector
        self._detect_vehicles_fn = detect_vehicles_fn
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._thread.start()
        log("INFO", f"VehicleDetectCoordinator started with {len(self._cameras)} cameras")

    def stop(self, timeout=3.0):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        return not self._thread or not self._thread.is_alive()

    def _detect_loop(self):
        interval = 1.0 / YOLO26_DETECT_FPS
        while self._running:
            t0 = time.time()
            for cam in self._cameras:
                full_frame = cam.peek_latest_frame(copy_frame=True)
                if full_frame is None:
                    continue
                try:
                    detections = self._detect_vehicles_fn(full_frame, detector=self._detector)
                    self._tracker.update(cam.name, detections, full_frame.shape)
                except Exception as exc:
                    log("ERROR", f"Vehicle detection error [{cam.name}]: {exc}")
                finally:
                    del full_frame
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))
