"""Detection coordinators for LPR and vehicle detection."""

import threading
import time

from config import DETECT_FPS, YOLO26_DETECT_FPS

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
    """Runs LPR on frames from multiple cameras in parallel.
    Uses a persistent worker thread for second camera to avoid per-cycle thread creation."""

    def __init__(self, cameras: list, tracker, detect_plates_fn=None):
        self._cameras = cameras
        self._tracker = tracker
        self._detect_plates_fn = detect_plates_fn
        self._detect_regions_fn = None
        self._recognize_regions_fn = None
        self._charset = None
        self._running = False
        self._enabled = False
        self._state_lock = threading.Lock()
        self._ocr_jobs = {}
        self._ocr_locks = {}
        self._ocr_events = {}
        self._worker_frame = None
        self._worker_event = threading.Event()
        self._worker_done = threading.Event()
        self._worker_done.set()
        self._detect_thread = None
        self._worker_thread = None
        self._ocr_threads = []

    def configure_split_pipeline(self, detect_regions_fn, recognize_regions_fn, charset):
        self._detect_regions_fn = detect_regions_fn
        self._recognize_regions_fn = recognize_regions_fn
        self._charset = charset

    def start(self):
        self._running = True
        if len(self._cameras) > 1:
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()
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
        self._running = False
        for event in self._ocr_events.values():
            event.set()
        self._worker_event.set()
        self._worker_done.set()
        self._worker_frame = None
        for name, lock in self._ocr_locks.items():
            with lock:
                self._ocr_jobs[name] = None
        for thread in (self._detect_thread, self._worker_thread):
            if thread and thread.is_alive():
                thread.join(timeout=timeout / 2)
        for thread in self._ocr_threads:
            if thread and thread.is_alive():
                thread.join(timeout=timeout / 2)

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

    def _worker_loop(self):
        """Persistent worker thread for second camera inference."""
        cam = self._cameras[1]
        while self._running:
            self._worker_event.wait()
            self._worker_event.clear()
            if not self._running:
                break
            frame = self._worker_frame
            self._worker_frame = None
            if frame is not None:
                try:
                    self._run_detection(cam, frame)
                except Exception as exc:
                    log("ERROR", f"Worker detection error [{cam.name}]: {exc}")
                finally:
                    del frame
            self._worker_done.set()

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
                self._tracker.add_observation(plate_text, p["det_conf"], cw, ch)
                for alt_plate, _ in p.get("valid_candidates", [])[1:]:
                    self._tracker.add_observation(alt_plate, p["det_conf"] * 0.5, cw, ch)
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
                    self._tracker.add_observation(alt_plate, p["det_conf"] * 0.75, cw, ch)
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
                plates = self._recognize_regions_fn(regions, ocr=cam.ocr, charset=self._charset)
                if not self.is_enabled():
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

    def _run_detection(self, cam, full_frame):
        """Run detection for one camera. Feeds tracker directly."""
        t0 = time.time()
        if self._detect_regions_fn and self._recognize_regions_fn:
            regions = self._detect_regions_fn(full_frame, detector=cam.detector)
            elapsed_ms = (time.time() - t0) * 1000
            if regions:
                log("TIMING", f"[{cam.name}] Detect: {elapsed_ms:.0f}ms  regions={len(regions)}")
            self._submit_ocr_job(cam, full_frame, regions, t0)
            return
        plates = self._detect_plates_fn(full_frame, detector=cam.detector, ocr=cam.ocr)
        elapsed_ms = (time.time() - t0) * 1000
        if plates:
            log("TIMING", f"[{cam.name}] Frame inference: {elapsed_ms:.0f}ms  plates={len(plates)}")
        self._process_plate_detections(cam, plates, full_frame)

    def _detect_loop(self):
        interval = 1.0 / DETECT_FPS
        _cam0 = self._cameras[0]
        _has_cam1 = len(self._cameras) > 1
        while self._running:
            if not self.is_enabled():
                time.sleep(0.05)
                continue

            frame1 = _cam0.get_latest_frame()
            frame2 = self._cameras[1].get_latest_frame() if _has_cam1 else None

            if frame1 is None and frame2 is None:
                time.sleep(0.01)
                continue

            t0 = time.time()
            try:
                if frame1 is not None and frame2 is not None:
                    self._worker_done.wait()
                    self._worker_done.clear()
                    self._worker_frame = frame2
                    frame2 = None
                    self._worker_event.set()
                    self._run_detection(_cam0, frame1)
                    self._worker_done.wait()
                elif frame1 is not None:
                    self._run_detection(_cam0, frame1)
                else:
                    self._run_detection(self._cameras[1], frame2)
            except Exception as exc:
                log("ERROR", f"DetectCoordinator pipeline error: {exc}")
            finally:
                frame1 = None
                frame2 = None

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
