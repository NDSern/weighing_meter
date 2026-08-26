"""Frame source classes for RTSP stream capture."""

import threading
import time
import re
from datetime import datetime, timezone

import cv2

from config import DETECT_FPS, FRAME_GRAB_DRAIN_MAX, FRAME_GRAB_DRAIN_SECONDS, RECONNECT_DELAY
from services.runtime.inference_lock import PriorityInferenceLock

_log_fn = None


def set_log_fn(log_fn):
    """Set the logging function to use."""
    global _log_fn
    _log_fn = log_fn


def log(level: str, msg: str):
    """Log using the configured log function."""
    if _log_fn:
        _log_fn(level, msg)


def mask_url_secret(url: str):
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", str(url))


class _LatestFrameSource:
    """Continuously grabs frames from an RTSP stream and keeps only the latest."""

    def __init__(
        self, url: str, start_log: str, open_fail_log: str, connect_log: str,
        grab_fail_log: str, source_name: str, expected_resolution=None,
    ):
        self._url = url
        self._start_log = start_log
        self._open_fail_log = open_fail_log
        self._connect_log = connect_log
        self._grab_fail_log = grab_fail_log
        self._source_name = source_name
        self._expected_resolution = (
            tuple(int(value) for value in expected_resolution)
            if expected_resolution is not None else None
        )
        if self._expected_resolution is not None and len(self._expected_resolution) != 2:
            raise ValueError("expected_resolution must contain width and height")
        self._running = False
        self._latest_frame = None
        self._latest_frame_id = 0
        self._latest_frame_captured_at = None
        self._frame_lock = threading.Lock()
        self._capture_lock = threading.Lock()
        self._capture = None
        self._thread = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        log("INFO", self._start_log)

    def stop(self, timeout=3.0):
        self._running = False
        with self._capture_lock:
            if self._capture is not None:
                self._capture.release()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        with self._frame_lock:
            self._latest_frame = None
        return not self._thread or not self._thread.is_alive()

    def get_latest_frame(self):
        with self._frame_lock:
            frame = self._latest_frame
            self._latest_frame = None
            return frame

    def peek_latest_frame(self, copy_frame=False):
        with self._frame_lock:
            frame = self._latest_frame
            if frame is None:
                return None
            return frame.copy() if copy_frame else frame

    def peek_latest_frame_with_id(self, copy_frame=False):
        """Return latest frame and capture generation from one locked snapshot."""
        with self._frame_lock:
            frame = self._latest_frame
            if frame is None:
                return None, None
            return (frame.copy() if copy_frame else frame), self._latest_frame_id

    def peek_latest_frame_snapshot(self, copy_frame=False):
        """Return latest frame, generation, and RTSP acquisition timestamp."""
        with self._frame_lock:
            frame = self._latest_frame
            if frame is None:
                return None, None, None
            return (
                frame.copy() if copy_frame else frame,
                self._latest_frame_id,
                self._latest_frame_captured_at,
            )

    def _clear_latest_frame(self):
        with self._frame_lock:
            self._latest_frame = None
            self._latest_frame_captured_at = None

    def _grab_loop(self):
        interval = 1.0 / DETECT_FPS
        cam_frame_time = 1.0 / 25
        while self._running:
            self._clear_latest_frame()
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            with self._capture_lock:
                self._capture = cap
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                log("WARNING", f"{self._open_fail_log} Retry in {RECONNECT_DELAY}s...")
                cap.release()
                with self._capture_lock:
                    if self._capture is cap:
                        self._capture = None
                time.sleep(RECONNECT_DELAY)
                continue

            log("INFO", self._connect_log)
            last_retrieve = 0.0
            decoded_resolution = None
            while self._running:
                t_grab = time.time()
                ret = cap.grab()
                if not ret:
                    log("WARNING", self._grab_fail_log)
                    break
                now = time.time()
                if now - last_retrieve >= interval:
                    drain_deadline = now + FRAME_GRAB_DRAIN_SECONDS
                    drain_count = 1
                    while drain_count < FRAME_GRAB_DRAIN_MAX and time.time() < drain_deadline:
                        if not cap.grab():
                            break
                        drain_count += 1
                    ret2, frame = cap.retrieve()
                    if not ret2:
                        log("WARNING", self._grab_fail_log)
                        break
                    decoded_resolution, accepted = self._check_resolution(
                        frame, decoded_resolution
                    )
                    if not accepted:
                        break
                    with self._frame_lock:
                        self._latest_frame = frame
                        self._latest_frame_id += 1
                        self._latest_frame_captured_at = datetime.now(timezone.utc).isoformat(
                            timespec="milliseconds"
                        )
                    last_retrieve = now
                grab_took = time.time() - t_grab
                time.sleep(max(0.0, cam_frame_time - grab_took))
            cap.release()
            with self._capture_lock:
                if self._capture is cap:
                    self._capture = None
            self._clear_latest_frame()
            if self._running:
                time.sleep(RECONNECT_DELAY)

    def _check_resolution(self, frame, previous):
        resolution = (int(frame.shape[1]), int(frame.shape[0]))
        if previous is None:
            log(
                "INFO",
                f"[{self._source_name}] RTSP decoded source="
                f"{resolution[0]}x{resolution[1]}",
            )
        elif resolution != previous:
            log(
                "WARNING",
                f"[{self._source_name}] RTSP resolution changed "
                f"{previous[0]}x{previous[1]} -> {resolution[0]}x{resolution[1]}",
            )
        if self._expected_resolution is not None and resolution != self._expected_resolution:
            log(
                "WARNING",
                f"[{self._source_name}] RTSP resolution rejected "
                f"actual={resolution[0]}x{resolution[1]} "
                f"expected={self._expected_resolution[0]}x{self._expected_resolution[1]}",
            )
            return resolution, False
        return resolution, True


class FrameGrabber(_LatestFrameSource):
    """Latest-frame RTSP source used to snapshot a second camera at publish time."""

    def __init__(self, url: str):
        super().__init__(
            url=url,
            start_log=f"FrameGrabber started. RTSP: {mask_url_secret(url)}",
            open_fail_log=f"FrameGrabber: cannot open {mask_url_secret(url)}.",
            connect_log=f"FrameGrabber: stream connected ({mask_url_secret(url)})",
            grab_fail_log="FrameGrabber: frame grab failed — reconnecting...",
            source_name="rear",
        )


class CameraGrabber(_LatestFrameSource):
    """Latest-frame RTSP source for LPR cameras. Detection is handled externally."""

    def __init__(
        self, url: str, name: str = "cam1", detector=None, ocr=None,
        lpr_crop: str = "full", expected_resolution=None,
    ):
        self.name = name
        self.detector = detector
        self.ocr = ocr
        self.lpr_crop = lpr_crop
        self.inference_lock = PriorityInferenceLock()
        super().__init__(
            url=url,
            start_log=f"CameraGrabber [{name}] started. RTSP: {mask_url_secret(url)}",
            open_fail_log=f"[{name}] Cannot open RTSP stream.",
            connect_log=f"[{name}] RTSP stream connected.",
            grab_fail_log=f"[{name}] Frame grab failed — reconnecting...",
            source_name=name,
            expected_resolution=expected_resolution,
        )
