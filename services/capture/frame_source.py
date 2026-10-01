"""Frame source classes for RTSP stream capture."""

import threading
import time
import re
from datetime import datetime, timezone

import cv2
import numpy as np

try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo

    Gst.init(None)
    _GST_AVAILABLE = True
except (ImportError, ValueError):
    Gst = GstVideo = None
    _GST_AVAILABLE = False

from config import DETECT_FPS, FRAME_GRAB_DRAIN_MAX, FRAME_GRAB_DRAIN_SECONDS, RECONNECT_DELAY
from services.runtime.inference_lock import PriorityInferenceLock

_log_fn = None
_FRAME_PULL_INTERVAL_SECONDS = 0.2


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
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._latest_frame_id = 0
        self._latest_frame_captured_at = None
        self._frame_lock = threading.Lock()
        self._thread = None
        self._opencv_capture = self._open_capture

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()
        log("INFO", self._start_log)

    def stop(self, timeout=3.0):
        self._running = False
        self._stop_event.set()
        # Do not touch self._capture here: the grab loop owns the capture's
        # lifetime. Releasing a VideoCapture concurrently with an in-flight
        # cap.grab()/retrieve() is undefined behaviour and crashes the process.
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
        if _GST_AVAILABLE:
            self._grab_with_gst()
        else:
            self._grab_with_opencv()

    def _grab_with_opencv(self):
        interval = 1.0 / DETECT_FPS
        cam_frame_time = 1.0 / 25
        while self._running:
            self._clear_latest_frame()
            cap = self._opencv_capture()
            if cap is None:
                self._stop_event.wait(RECONNECT_DELAY)
                continue
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                log("WARNING", f"{self._open_fail_log} Retry in {RECONNECT_DELAY}s...")
                cap.release()
                self._stop_event.wait(RECONNECT_DELAY)
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
            self._clear_latest_frame()
            if self._running:
                self._stop_event.wait(RECONNECT_DELAY)

    def _open_capture(self):
        """Open an RTSP capture with finite connect/read timeouts.

        A stalled read must return control to the loop (which reconnects)
        instead of blocking the grab thread forever.
        """
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        except Exception:
            pass
        return cap

    def _grab_with_gst(self):
        while self._running:
            self._clear_latest_frame()
            try:
                pipeline = self._create_gst_pipeline()
            except Exception as exc:
                log("WARNING", f"{self._open_fail_log} error={exc}")
                self._stop_event.wait(RECONNECT_DELAY)
                continue
            sink = pipeline.get_by_name("sink")
            bus = pipeline.get_bus()
            try:
                state = pipeline.set_state(Gst.State.PLAYING)
                if state == Gst.StateChangeReturn.FAILURE:
                    log("WARNING", f"{self._open_fail_log} Retry in {RECONNECT_DELAY}s...")
                else:
                    decoded_resolution = None
                    connected = False
                    next_pull = time.monotonic()
                    while self._running:
                        delay = next_pull - time.monotonic()
                        if delay > 0 and self._stop_event.wait(delay):
                            break
                        sample = sink.emit("try-pull-sample", 100 * Gst.MSECOND)
                        next_pull = max(next_pull + _FRAME_PULL_INTERVAL_SECONDS, time.monotonic())
                        message = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
                        if message is not None:
                            if message.type == Gst.MessageType.ERROR:
                                error, _debug = message.parse_error()
                                log("WARNING", f"{self._grab_fail_log} error={error.message}")
                            else:
                                log("WARNING", self._grab_fail_log)
                            break
                        if sample is None:
                            continue
                        frame = self._sample_to_frame(sample)
                        if frame is None or self._is_corrupt_green_frame(frame):
                            continue
                        decoded_resolution, accepted = self._check_resolution(frame, decoded_resolution)
                        if not accepted:
                            break
                        if not connected:
                            log("INFO", self._connect_log)
                            connected = True
                        with self._frame_lock:
                            self._latest_frame = frame
                            self._latest_frame_id += 1
                            self._latest_frame_captured_at = datetime.now(timezone.utc).isoformat(
                                timespec="milliseconds"
                            )
            finally:
                pipeline.set_state(Gst.State.NULL)
            self._clear_latest_frame()
            if self._running:
                self._stop_event.wait(RECONNECT_DELAY)

    def _create_gst_pipeline(self):
        if Gst.ElementFactory.find("mppvideodec") is not None:
            decoder = "mppvideodec format=NV12 discard-corrupted-frames=true ! video/x-raw,format=NV12 ! "
        else:
            decoder = (
                "avdec_h265 max-threads=2 output-corrupt=false discard-corrupted-frames=true ! "
                "videoconvert n-threads=2 ! video/x-raw,format=BGR ! "
            )
        pipeline = Gst.parse_launch(
            "rtspsrc name=src protocols=tcp latency=500 drop-on-latency=true "
            "buffer-mode=slave tcp-timeout=5000000 ! "
            "application/x-rtp,media=video,encoding-name=H265 ! "
            "rtph265depay ! h265parse ! " + decoder +
            "appsink name=sink max-buffers=1 drop=true sync=false wait-on-eos=false"
        )
        pipeline.get_by_name("src").set_property("location", self._url)
        return pipeline

    @staticmethod
    def _sample_to_frame(sample):
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        pixel_format = caps.get_value("format")
        buffer = sample.get_buffer()
        data = buffer.extract_dup(0, buffer.get_size())
        if pixel_format == "NV12":
            metadata = GstVideo.buffer_get_video_meta(buffer)
            if metadata is None or metadata.n_planes != 2:
                return None
            y_offset, uv_offset = metadata.offset[:2]
            y_stride, uv_stride = metadata.stride[:2]
            y_end = y_offset + (height - 1) * y_stride + width
            uv_end = uv_offset + (height // 2 - 1) * uv_stride + width
            if min(y_offset, uv_offset, y_stride, uv_stride) < 0 or max(y_end, uv_end) > len(data):
                return None
            y_plane = np.ndarray((height, width), dtype=np.uint8, buffer=data, offset=y_offset, strides=(y_stride, 1))
            uv_plane = np.ndarray(
                (height // 2, width // 2, 2), dtype=np.uint8, buffer=data,
                offset=uv_offset, strides=(uv_stride, 2, 1),
            )
            return cv2.cvtColorTwoPlane(y_plane, uv_plane, cv2.COLOR_YUV2BGR_NV12)
        if len(data) != width * height * 3:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3)).copy()

    @staticmethod
    def _is_corrupt_green_frame(frame):
        sample = frame[::64, ::64]
        blue, green, red = sample.mean(axis=(0, 1))
        return blue < 2 and red < 2 and green > 20

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
        lpr_crop: str = "full", expected_resolution=None, fallback_detector=None,
    ):
        self.name = name
        self.detector = detector
        self.ocr = ocr
        self.fallback_detector = fallback_detector
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
