"""Frame source classes for RTSP stream capture."""

import threading
import time
import re

import cv2

from config import DETECT_FPS, FRAME_GRAB_DRAIN_MAX, FRAME_GRAB_DRAIN_SECONDS, RECONNECT_DELAY

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

    def __init__(self, url: str, start_log: str, open_fail_log: str, connect_log: str, grab_fail_log: str):
        self._url = url
        self._start_log = start_log
        self._open_fail_log = open_fail_log
        self._connect_log = connect_log
        self._grab_fail_log = grab_fail_log
        self._running = False
        self._latest_frame = None
        self._frame_lock = threading.Lock()
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        log("INFO", self._start_log)

    def stop(self, timeout=3.0):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        with self._frame_lock:
            self._latest_frame = None

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

    def _grab_loop(self):
        interval = 1.0 / DETECT_FPS
        cam_frame_time = 1.0 / 25
        while self._running:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                log("WARNING", f"{self._open_fail_log} Retry in {RECONNECT_DELAY}s...")
                time.sleep(RECONNECT_DELAY)
                continue

            log("INFO", self._connect_log)
            last_retrieve = 0.0
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
                    if ret2:
                        with self._frame_lock:
                            self._latest_frame = frame
                    last_retrieve = now
                grab_took = time.time() - t_grab
                time.sleep(max(0.0, cam_frame_time - grab_took))
            cap.release()
            if self._running:
                time.sleep(RECONNECT_DELAY)


class FrameGrabber(_LatestFrameSource):
    """Latest-frame RTSP source used to snapshot a second camera at publish time."""

    def __init__(self, url: str):
        super().__init__(
            url=url,
            start_log=f"FrameGrabber started. RTSP: {mask_url_secret(url)}",
            open_fail_log=f"FrameGrabber: cannot open {mask_url_secret(url)}.",
            connect_log=f"FrameGrabber: stream connected ({mask_url_secret(url)})",
            grab_fail_log="FrameGrabber: frame grab failed — reconnecting...",
        )


class CameraGrabber(_LatestFrameSource):
    """Latest-frame RTSP source for LPR cameras. Detection is handled externally."""

    def __init__(self, url: str, name: str = "cam1", detector=None, ocr=None):
        self.name = name
        self.detector = detector
        self.ocr = ocr
        super().__init__(
            url=url,
            start_log=f"CameraGrabber [{name}] started. RTSP: {mask_url_secret(url)}",
            open_fail_log=f"[{name}] Cannot open RTSP stream.",
            connect_log=f"[{name}] RTSP stream connected.",
            grab_fail_log=f"[{name}] Frame grab failed — reconnecting...",
        )
