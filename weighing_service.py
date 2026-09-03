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

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Path setup ────────────────────────────────────────────────────
SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
LPR_DIR = os.path.join(SERVICE_DIR, "yolov5lpr")
sys.path.insert(0, SERVICE_DIR)
sys.path.insert(0, LPR_DIR)

from config import (
    BAUD_RATE,
    CAM1_LPR_CROP,
    CAM1_EXPECTED_RESOLUTION,
    CAM2_RESULT_CROP,
    CAM3_LPR_CROP,
    CAPTURE_DIR,
    IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
    IMAGE_RETENTION_DAYS,
    IMAGE_RETENTION_ENABLED,
    IMAGE_RETENTION_EXTENSIONS,
    LOG_DIR,
    LOG_FILE_PREFIX,
    LOG_FILE_PATH,
    MQTT_ENABLED,
    NO_PLATE_DIR,
    NO_STABLE_DIR,
    PEAK_CANDIDATE_DIR,
    RTSP_URL,
    RTSP_URL_2,
    RTSP_URL_3,
    SERIAL_PORT,
    SERVICE_DIR,
    LPR_CHARSET,
    LPR_DETECTOR_MODEL,
    LPR_RECOGNIZER_MODEL,
    LPR_SPOOL_DIR,
    SESSION_FRAME_DISK_CAP_BYTES,
    SESSION_FRAME_INTERVAL_SECONDS,
    SESSION_FRAME_JPEG_QUALITY,
    SESSION_FRAME_MIN_FREE_BYTES,
    SESSION_FRAME_QUEUE_SIZE,
    SCALE_DATA_DIR,
    UNDETECTABLE_DIR,
    WEIGHT_THRESHOLD,
    validate_runtime_config,
)
from services.runtime.lpr_bundle import verify_lpr_bundle

_LPR_BUNDLE_PATHS = {
    "detector": LPR_DETECTOR_MODEL,
    "recognizer": LPR_RECOGNIZER_MODEL,
    "charset": LPR_CHARSET,
    "decoder": os.path.join(SERVICE_DIR, "services", "pipeline", "detector_obb_decode.py"),
}
verify_lpr_bundle(_LPR_BUNDLE_PATHS)

from services.runtime.async_logging import AsyncLogger

_logger = AsyncLogger(LOG_DIR, LOG_FILE_PREFIX)
log = _logger.log
close_log = _logger.close


def mask_url_secret(url: str):
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", str(url))


# ── Import pipeline and services ─────────────────────────────────
from d2008_scale_reader import D2008Reader
from mqtt_service import MqttService

from services.pipeline.license_plate_recognition import (
    detect_plate_regions,
    load_lpr_charset,
    recognize_plate_regions,
    validate_lpr_runtime,
)
from services.pipeline import detector_obb_decode

from services.tracking import PlateTracker
from services.capture import FrameGrabber, CameraGrabber, DetectCoordinator
from services.capture.detect_coordinator import set_log_fn as set_detect_coordinator_log
from services.capture.session_frame_spool import SessionFrameSpool
from services.pipeline.deferred_lpr_worker import DeferredLprWorker
from services.capture.frame_source import set_log_fn as set_frame_source_log
from services.storage.image_save_worker import ImageSaveWorker
from services.storage.image_save_worker import set_log_fn as set_image_save_log
from services.storage.publish_outbox import PublishOutbox
from services.storage.retention_cleaner import ImageRetentionCleaner, StorageMaintenance
from services.runtime import RknnModelSet
from services.session import SessionManager
from services.session.session_manager import set_log_fn as set_session_log

# Configure logging in all modules
set_detect_coordinator_log(log)
set_image_save_log(log)
set_session_log(log)
set_frame_source_log(log)

# ── Main Service ──────────────────────────────────────────────────
def main():
    stop_event = threading.Event()
    def request_stop(signum=None, frame=None):
        stop_event.set()
    previous_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    models = mqtt_svc = cam1 = cam3 = grabber2 = None
    detect_coord = reader = frame_spool = deferred_lpr = None
    retention_cleaner = storage_maintenance = session_manager = plate_tracker = None
    mqtt_started = image_worker_started = outbox_started = False
    detect_stopped = deferred_stopped = True

    def cleanup(name, callback):
        try:
            return callback()
        except Exception as exc:
            log("ERROR", f"Cleanup failed resource={name}: {exc}")
            return False

    try:
        validate_runtime_config()
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        log("INFO", "=" * 60)
        log("INFO", "Weighing Service starting (DETECT-FIRST architecture)")
        log("INFO", f"Scale: {SERIAL_PORT} @ {BAUD_RATE}")
        log("INFO", f"Camera 1 (LPR crop={CAM1_LPR_CROP}): {mask_url_secret(RTSP_URL)}")
        log("INFO", f"Camera 3 (LPR crop={CAM3_LPR_CROP}): {mask_url_secret(RTSP_URL_3)}")
        log("INFO", f"Camera 2 (rear crop={CAM2_RESULT_CROP}): {mask_url_secret(RTSP_URL_2)}")

        lpr_charset = load_lpr_charset(LPR_CHARSET)
        from rknnlite.api import RKNNLite as RKNN

        bundle_hashes = verify_lpr_bundle(_LPR_BUNDLE_PATHS)
        log("INFO", f"Fine-tuned LPR bundle hashes passed hashes={bundle_hashes}")
        models = RknnModelSet.open(
            LPR_DETECTOR_MODEL,
            LPR_RECOGNIZER_MODEL,
            None,
            False,
            RKNN,
            log_fn=log,
        )
        handles = models.handles
        validate_lpr_runtime(
            [("cam1", handles.cam1_detector), ("cam3", handles.cam3_detector)],
            [("cam1", handles.cam1_ocr), ("cam3", handles.cam3_ocr)],
            lpr_charset,
            model_paths=_LPR_BUNDLE_PATHS,
            log_fn=log,
        )

        plate_tracker = PlateTracker()
        if MQTT_ENABLED:
            mqtt_svc = MqttService(on_log=log)
            mqtt_svc.start()
            mqtt_started = True
        else:
            log("INFO", "MQTT disabled (MQTT_ENABLED=False)")

        cam1 = CameraGrabber(
            RTSP_URL, "cam1", handles.cam1_detector, handles.cam1_ocr,
            CAM1_LPR_CROP, expected_resolution=CAM1_EXPECTED_RESOLUTION,
        )
        cam1.start()
        cam3 = CameraGrabber(RTSP_URL_3, "cam3", handles.cam3_detector, handles.cam3_ocr, CAM3_LPR_CROP)
        cam3.start()
        grabber2 = FrameGrabber(RTSP_URL_2)
        grabber2.start()

        vehicle_tracker = None

        reader = D2008Reader(
            port=SERIAL_PORT,
            baud=BAUD_RATE,
            db_file=SCALE_DATA_DIR,
            log_interval=0.2,
        )
        storage_maintenance = StorageMaintenance(IMAGE_RETENTION_CHECK_INTERVAL_SECONDS, log_fn=log)
        storage_maintenance.start()
        if IMAGE_RETENTION_ENABLED:
            retention_cleaner = ImageRetentionCleaner(
                [CAPTURE_DIR, UNDETECTABLE_DIR, NO_STABLE_DIR, NO_PLATE_DIR, PEAK_CANDIDATE_DIR],
                IMAGE_RETENTION_DAYS,
                IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
                IMAGE_RETENTION_EXTENSIONS,
                log_fn=log,
            )
            retention_cleaner.start()

        session_manager = SessionManager(
            plate_tracker=plate_tracker,
            mqtt_svc=mqtt_svc,
            vehicle_tracker=vehicle_tracker,
            rear_grabber=grabber2,
            lpr_grabbers={"cam1": cam1, "cam3": cam3},
            save_images_fn=ImageSaveWorker.save_and_upload_now,
            undetectable_dir=UNDETECTABLE_DIR,
            cam2_result_crop=CAM2_RESULT_CROP,
        )
        detect_coord = DetectCoordinator(
            [cam1, cam3], plate_tracker,
            presence_callback=lambda camera, valid, revision: session_manager.on_plate_presence(
                camera, valid, log, revision=revision
            ),
            session_context=session_manager.session_context,
        )
        detect_coord.configure_split_pipeline(detect_plate_regions, recognize_plate_regions, lpr_charset)
        frame_spool = SessionFrameSpool(
            LPR_SPOOL_DIR,
            cam1,
            cam3,
            cam2_grabber=grabber2,
            interval=SESSION_FRAME_INTERVAL_SECONDS,
            jpeg_quality=SESSION_FRAME_JPEG_QUALITY,
            notification_queue_size=SESSION_FRAME_QUEUE_SIZE,
            disk_cap_bytes=SESSION_FRAME_DISK_CAP_BYTES,
            min_free_bytes=SESSION_FRAME_MIN_FREE_BYTES,
            metadata_provider=detect_coord.get_frame_metadata,
        )
        session_manager.frame_spool = frame_spool
        deferred_lpr = DeferredLprWorker(
            frame_spool,
            [cam1, cam3],
            lpr_charset,
            lambda metadata, tracker: session_manager.finalize_deferred_session(metadata, tracker, log),
            detect_regions_fn=detect_plate_regions,
            recognize_regions_fn=recognize_plate_regions,
            tracker_factory=lambda: PlateTracker(max_plate_images=2),
            job_interval=1.0,
            memory_cleanup_fn=_malloc_trim,
            log_fn=log,
        )
        ImageSaveWorker.start_upload_worker()
        image_worker_started = True
        if MQTT_ENABLED and mqtt_svc:
            PublishOutbox.start(mqtt_svc)
            outbox_started = True
        frame_spool.start()
        deferred_lpr.start()
        detect_coord.start()
        detect_coord.set_enabled(True)

        reader.on_weight = lambda frame: session_manager.on_weight(frame, log)
        reader.on_frame = lambda frame: session_manager.on_frame(frame, log)
        reader.on_status_change = lambda frame, old, new: session_manager.on_status_change(frame, old, new, log)
        reader.start()
        while not stop_event.wait(0.1):
            if reader.state == "failed":
                raise RuntimeError(f"Scale reader failed: {reader.last_error}")
            if session_manager.fatal_error:
                raise RuntimeError(session_manager.fatal_error)
    except KeyboardInterrupt:
        request_stop()
    finally:
        log("INFO", "Stopping service...")
        if detect_coord:
            detect_coord.set_enabled(False)
            detect_stopped = cleanup("detect_coordinator", detect_coord.stop) is not False
        if reader:
            reader.on_weight = reader.on_frame = reader.on_status_change = None
            cleanup("scale_reader", reader.stop)
        if session_manager:
            cleanup("session_manager", lambda: session_manager.shutdown(log))
        if deferred_lpr:
            deferred_stopped = cleanup("deferred_lpr", deferred_lpr.stop) is not False
        if frame_spool:
            cleanup("session_frame_spool", frame_spool.stop)
        for name, camera in (("cam2", grabber2), ("cam3", cam3), ("cam1", cam1)):
            if camera and cleanup(name, camera.stop) is False:
                log("ERROR", f"{name} capture thread did not stop")
        if retention_cleaner:
            cleanup("image_retention", retention_cleaner.stop)
        if storage_maintenance:
            cleanup("storage_maintenance", storage_maintenance.stop)
        if outbox_started:
            cleanup("publish_outbox", PublishOutbox.stop)
        if mqtt_started and mqtt_svc:
            cleanup("mqtt", mqtt_svc.stop)
        if image_worker_started:
            cleanup("image_upload_drain", lambda: ImageSaveWorker.wait_for_pending(timeout=15.0))
            cleanup("image_upload_worker", ImageSaveWorker.stop)
        if models:
            cleanup(
                "rknn_models",
                lambda: models.close(
                    release_lpr=detect_stopped and deferred_stopped,
                    release_vehicle=True,
                ),
            )
        for sig, handler in previous_handlers.items():
            cleanup(f"signal_{sig}", lambda sig=sig, handler=handler: signal.signal(sig, handler))
        cleanup("final_log", lambda: log("INFO", "Weighing Service stopped."))
        cleanup("log_file", close_log)


if __name__ == "__main__":
    main()
