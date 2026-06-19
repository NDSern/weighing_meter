"""
Weighing Service — DETECT-FIRST architecture.

- Camera detection runs CONTINUOUSLY (not gated by scale)
- PlateTracker accumulates weighted votes across frames
- When scale stabilizes → pairs confirmed plate with stable weight
- When scale returns to zero → clears plate tracker for next vehicle
- Uses test3 YOLOv8-OBB RKNN detector + PP-OCR RKNN recognizer
"""

import ctypes
import os
import re
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MALLOC_ARENA_MAX", "4")
os.environ.setdefault("OPENCV_FFMPEG_THREADS", "2")

try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int

    def _malloc_trim():
        _libc.malloc_trim(0)
except (OSError, AttributeError):

    def _malloc_trim():
        pass

import cv2
import warnings
from minio import Minio
from minio.error import S3Error

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ────────────────────────────────────────────────────
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
LPR_DIR = os.path.join(SERVICE_DIR, "yolov5lpr")
sys.path.insert(0, SERVICE_DIR)
sys.path.insert(0, LPR_DIR)

from config import (
    BAUD_RATE,
    CAM2_RESULT_CROP,
    CAPTURE_DIR,
    DETECT_FPS,
    IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
    IMAGE_RETENTION_DAYS,
    IMAGE_RETENTION_ENABLED,
    IMAGE_RETENTION_EXTENSIONS,
    LOG_DIR,
    LOG_FILE_PREFIX,
    LOG_FILE_PATH,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
    MQTT_ENABLED,
    RTSP_URL,
    RTSP_URL_2,
    RTSP_URL_3,
    SERIAL_PORT,
    SERVICE_DIR,
    LPR_CHARSET,
    LPR_DETECTOR_MODEL,
    LPR_IMAGE_SIZE,
    LPR_RECOGNIZER_MODEL,
    UNDETECTABLE_DIR,
    WEIGHT_THRESHOLD,
    YOLO26_ENABLED,
    YOLO26_MODEL_PATH,
)

# ── Logging ───────────────────────────────────────────────────────
_log_lock = threading.Lock()
_log_fp = None
_log_date = None


def _log_path_for_date(date_text: str):
    return os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}_{date_text}.log")


def _ensure_log_file(now: datetime):
    global _log_fp, _log_date
    date_text = now.strftime("%Y-%m-%d")
    if _log_fp is not None and _log_date == date_text:
        return _log_fp
    if _log_fp is not None:
        _log_fp.close()
        _log_fp = None
        _log_date = None
    os.makedirs(LOG_DIR, exist_ok=True)
    _log_fp = open(_log_path_for_date(date_text), "a", buffering=1)
    _log_date = date_text
    return _log_fp


def close_log():
    global _log_fp, _log_date
    with _log_lock:
        if _log_fp is not None:
            _log_fp.close()
            _log_fp = None
            _log_date = None


def log(level: str, msg: str):
    now = datetime.now()
    ts = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:23]
    prefix = f"{ts} [{level:<7}] "
    text = str(msg).replace("\r", "")
    lines = text.splitlines() or [""]
    with _log_lock:
        fp = _ensure_log_file(now)
        for idx, part in enumerate(lines):
            line = f"{prefix}{part}" if idx == 0 else f"{'':23} [{'':<7}] {part}"
            print(line, flush=True)
            fp.write(line + "\n")
        fp.flush()


def mask_url_secret(url: str):
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", str(url))


# ── MinIO client ──────────────────────────────────────────────────
_minio = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)


# ── Load RKNN models ─────────────────────────────────────────────
def load_rknn_model(path, name, core_mask=None):
    from rknnlite.api import RKNNLite as RKNN

    log("INFO", f"Loading RKNN {name}: {path}")
    rknn = RKNN()
    ret = rknn.load_rknn(path)
    if ret != 0:
        raise RuntimeError(f"Failed to load RKNN model {path} (ret={ret})")
    ret = rknn.init_runtime(core_mask=core_mask)
    if ret != 0:
        raise RuntimeError(f"Failed to init RKNN runtime for {path} (ret={ret})")
    log("INFO", f"RKNN {name} ready (core_mask={core_mask}).")
    return rknn


log("INFO", "Loading RKNN models on NPU...")
from rknnlite.api import RKNNLite as RKNN

rknn_detector = load_rknn_model(
    LPR_DETECTOR_MODEL,
    "lpr_detector_cam1",
    core_mask=RKNN.NPU_CORE_0,
)
rknn_ocr = load_rknn_model(LPR_RECOGNIZER_MODEL, "lpr_recognizer_cam1", core_mask=RKNN.NPU_CORE_0)
rknn_detector_2 = load_rknn_model(
    LPR_DETECTOR_MODEL,
    "lpr_detector_cam3",
    core_mask=RKNN.NPU_CORE_1,
)
rknn_ocr_2 = load_rknn_model(LPR_RECOGNIZER_MODEL, "lpr_recognizer_cam3", core_mask=RKNN.NPU_CORE_1)
rknn_vehicle = None
if YOLO26_ENABLED:
    rknn_vehicle = load_rknn_model(
        YOLO26_MODEL_PATH,
        "YOLO26_vehicle",
        core_mask=RKNN.NPU_CORE_2,
    )
log("INFO", "All models loaded (2 cores).")


# ── Import pipeline and services ─────────────────────────────────
from datetime import datetime

from d2008_scale_reader import D2008Reader, WeightFrame
from mqtt_service import MqttService

from services.pipeline.inference import detect_vehicles_rknn
from services.pipeline.license_plate_recognition import detect_license_plates, detect_plate_regions, load_lpr_charset, recognize_plate_regions

from services.tracking import PlateTracker, VehicleTracker
from services.tracking.vehicle_tracker import set_log_fn as set_vehicle_tracker_log
from services.capture import FrameGrabber, CameraGrabber, DetectCoordinator, VehicleDetectCoordinator
from services.capture.detect_coordinator import set_log_fn as set_detect_coordinator_log
from services.capture.frame_source import set_log_fn as set_frame_source_log
from services.storage import ImageSaveWorker
from services.storage.image_save_worker import set_log_fn as set_image_save_log
from services.storage.retention_cleaner import ImageRetentionCleaner
from services.session import SessionManager, WeighingSessionState
from services.session.session_manager import set_log_fn as set_session_log

# Configure logging in all modules
set_vehicle_tracker_log(log)
set_detect_coordinator_log(log)
set_image_save_log(log)
set_session_log(log)
set_frame_source_log(log)

lpr_charset = load_lpr_charset(LPR_CHARSET)


def detect_plates_in_frame(frame, detector=None, ocr=None):
    return detect_license_plates(
        frame,
        detector=detector,
        ocr=ocr,
        imgsz=LPR_IMAGE_SIZE,
        charset=lpr_charset,
    )


# ── Main Service ──────────────────────────────────────────────────
def main():
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    stop_event = threading.Event()

    def request_stop(signum=None, frame=None):
        if signum is not None:
            log("INFO", f"Received signal {signum}; stopping service...")
        stop_event.set()

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    log("INFO", "=" * 60)
    log("INFO", "Weighing Service starting (DETECT-FIRST architecture)")
    log("INFO", f"Scale: {SERIAL_PORT} @ {BAUD_RATE}")
    log("INFO", f"Camera 1 (full-frame LPR/result):  {mask_url_secret(RTSP_URL)}")
    log("INFO", f"Camera 3 (full-frame LPR/result):  {mask_url_secret(RTSP_URL_3)}")
    log("INFO", f"Camera 2 (rear result crop={CAM2_RESULT_CROP}):  {mask_url_secret(RTSP_URL_2)}")
    log("INFO", f"LPR detector: {LPR_DETECTOR_MODEL}")
    log("INFO", f"LPR OCR: {LPR_RECOGNIZER_MODEL}")
    log("INFO", f"Captures: {CAPTURE_DIR}")
    log("INFO", f"Log file: {LOG_FILE_PATH}")
    log("INFO", "=" * 60)

    plate_tracker = PlateTracker()

    mqtt_svc = None
    if MQTT_ENABLED:
        mqtt_svc = MqttService(on_log=log)
        mqtt_svc.start()
    else:
        log("INFO", "MQTT disabled (MQTT_ENABLED=False)")

    cam1 = CameraGrabber(url=RTSP_URL, name="cam1", detector=rknn_detector, ocr=rknn_ocr)
    cam1.start()

    cam3 = CameraGrabber(url=RTSP_URL_3, name="cam3", detector=rknn_detector_2, ocr=rknn_ocr_2)
    cam3.start()

    detect_coord = DetectCoordinator(
        cameras=[cam1, cam3],
        tracker=plate_tracker,
        detect_plates_fn=detect_plates_in_frame,
    )
    detect_coord.configure_split_pipeline(detect_plate_regions, recognize_plate_regions, lpr_charset)
    detect_coord.start()

    vehicle_tracker = None
    vehicle_coord = None
    if YOLO26_ENABLED and rknn_vehicle is not None:
        vehicle_tracker = VehicleTracker()
        vehicle_coord = VehicleDetectCoordinator(
            cameras=[cam1, cam3],
            tracker=vehicle_tracker,
            detector=rknn_vehicle,
            detect_vehicles_fn=detect_vehicles_rknn,
        )
        vehicle_coord.start()

    grabber2 = FrameGrabber(RTSP_URL_2)
    grabber2.start()

    reader = D2008Reader(
        port=SERIAL_PORT,
        baud=BAUD_RATE,
        db_file=os.path.join(SERVICE_DIR, "scale_data.db"),
        log_interval=1.0,
    )

    retention_cleaner = None
    if IMAGE_RETENTION_ENABLED:
        retention_cleaner = ImageRetentionCleaner(
            roots=[CAPTURE_DIR, UNDETECTABLE_DIR],
            retention_days=IMAGE_RETENTION_DAYS,
            check_interval_seconds=IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
            extensions=IMAGE_RETENTION_EXTENSIONS,
            log_fn=log,
        )
        retention_cleaner.start()

    session_manager = SessionManager(
        plate_tracker=plate_tracker,
        mqtt_svc=mqtt_svc,
        vehicle_tracker=vehicle_tracker,
        detect_coord=detect_coord,
        rear_grabber=grabber2,
        lpr_grabbers={"cam1": cam1, "cam3": cam3},
        save_images_fn=ImageSaveWorker.save_and_upload_now,
        undetectable_dir=UNDETECTABLE_DIR,
        cam2_result_crop=CAM2_RESULT_CROP,
    )

    ImageSaveWorker.start_upload_worker()

    reader.on_weight = lambda frame: session_manager.on_weight(frame, log)
    reader.on_frame = lambda frame: session_manager.on_frame(frame, log)
    reader.on_status_change = lambda frame, old, new: session_manager.on_status_change(frame, old, new, log)
    reader.start()

    try:
        while not stop_event.wait(0.1):
            pass
    except KeyboardInterrupt:
        request_stop()
    finally:
        log("INFO", "Stopping service...")
        reader.stop()
        detect_coord.stop()
        if vehicle_coord:
            vehicle_coord.stop()
        cam1.stop()
        cam3.stop()
        grabber2.stop()
        if retention_cleaner:
            retention_cleaner.stop()

        plate, score, count = plate_tracker.get_confirmed_plate()
        if plate and session_manager.session.stable_weight and session_manager.session.stable_weight > WEIGHT_THRESHOLD:
            log("EVENT", f"Service stopping — final plate: {plate} (score={score:.2f})")

        ImageSaveWorker.wait_for_pending(timeout=15.0)
        log("INFO", "Weighing Service stopped.")
        if mqtt_svc:
            mqtt_svc.stop()
        rknn_detector.release()
        rknn_ocr.release()
        rknn_detector_2.release()
        rknn_ocr_2.release()
        if rknn_vehicle is not None:
            rknn_vehicle.release()
        close_log()


if __name__ == "__main__":
    main()
