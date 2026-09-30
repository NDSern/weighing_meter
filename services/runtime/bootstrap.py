"""Service construction and shutdown for the weighing service.

``main()`` keeps a single visible startup sequence; the long resource
construction and teardown bodies live here so the entry point stays readable.
"""

import ctypes
import os
import re
from datetime import datetime

from config import (
    BAUD_RATE,
    CAM1_LPR_CROP,
    CAM1_EXPECTED_RESOLUTION,
    CAM2_RESULT_CROP,
    CAM3_LPR_CROP,
    CAPTURE_DIR,
    DIAGNOSTIC_ARCHIVE_AFTER_DAYS,
    DIAGNOSTIC_ARCHIVE_CHECK_INTERVAL_SECONDS,
    DIAGNOSTIC_ARCHIVE_RETENTION_DAYS,
    DUPLICATE_REVIEW_ENABLED,
    IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
    IMAGE_RETENTION_DAYS,
    IMAGE_RETENTION_ENABLED,
    IMAGE_RETENTION_EXTENSIONS,
    IMAGE_STORAGE_TARGET_FREE_BYTES,
    LPR_CHARSET,
    LPR_DEFERRED_MAX_FRAMES_PER_CAMERA,
    LPR_DETECTOR_MODEL,
    LPR_RECOGNIZER_MODEL,
    LPR_SPOOL_DIR,
    MQTT_ENABLED,
    NO_PLATE_DIR,
    NO_STABLE_DIR,
    PEAK_CANDIDATE_DIR,
    RTSP_URL,
    RTSP_URL_2,
    RTSP_URL_3,
    SCALE_DATA_DIR,
    SERIAL_PORT,
    SERVICE_DIR,
    SESSION_FRAME_DISK_CAP_BYTES,
    SESSION_FRAME_INTERVAL_SECONDS,
    SESSION_FRAME_JPEG_QUALITY,
    SESSION_FRAME_MIN_FREE_BYTES,
    SESSION_FRAME_QUEUE_SIZE,
    UNDETECTABLE_DIR,
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

try:
    _libc = ctypes.CDLL("libc.so.6")
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    _libc.malloc_trim.restype = ctypes.c_int

    def _malloc_trim():
        _libc.malloc_trim(0)
except (OSError, AttributeError):

    def _malloc_trim():
        pass


def mask_url_secret(url: str):
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", str(url))


def configure_module_logging(log):
    """Install the process log function into the imported service modules."""
    from services.capture.detect_coordinator import set_log_fn as set_detect_coordinator_log
    from services.capture.frame_source import set_log_fn as set_frame_source_log
    from services.storage.image_save_worker import set_log_fn as set_image_save_log
    from services.session.session_manager import set_log_fn as set_session_log

    set_detect_coordinator_log(log)
    set_image_save_log(log)
    set_session_log(log)
    set_frame_source_log(log)


class ServiceResources:
    """Handles and start-state flags owned by one service run."""

    def __init__(self):
        self.models = None
        self.mqtt_svc = None
        self.cam1 = None
        self.cam3 = None
        self.grabber2 = None
        self.detect_coord = None
        self.reader = None
        self.frame_spool = None
        self.deferred_lpr = None
        self.retention_cleaner = None
        self.minio_cache_cleaner = None
        self.storage_maintenance = None
        self.session_manager = None
        self.plate_tracker = None
        self.diagnostic_archive_cleaner = None
        self.duplicate_reviewer = None
        self.mqtt_started = False
        self.image_worker_started = False
        self.outbox_started = False
        self.detect_stopped = True
        self.deferred_stopped = True

    def cleanup(self, name, callback, log):
        try:
            return callback()
        except Exception as exc:
            log("ERROR", f"Cleanup failed resource={name}: {exc}")
            return False


def construct_service(log, res=None) -> ServiceResources:
    """Build and start every service resource, returning the run's handles.

    ``res`` may be supplied so a partial construction can still be torn down
    by the caller when a mid-startup step raises.
    """
    from d2008_scale_reader import D2008Reader
    from mqtt_service import MqttService

    from services.pipeline.license_plate_recognition import (
        detect_plate_regions,
        load_lpr_charset,
        recognize_plate_regions,
        validate_lpr_runtime,
    )
    from services.tracking import PlateTracker
    from services.capture import FrameGrabber, CameraGrabber, DetectCoordinator
    from services.capture.camera_light_controller import CameraLightController
    from services.capture.session_frame_spool import SessionFrameSpool
    from services.pipeline.deferred_lpr_worker import DeferredLprWorker
    from services.storage.image_save_worker import ImageSaveWorker
    from services.storage.publish_outbox import PublishOutbox
    from services.storage.retention_cleaner import (
        DiagnosticArchiveCleaner, ImageRetentionCleaner, StorageMaintenance,
        VerifiedMinioCacheCleaner,
    )
    from services.review.duplicate_review import DuplicateReviewer
    from services.runtime import RknnModelSet
    from services.session import SessionManager

    if res is None:
        res = ServiceResources()

    validate_runtime_config()
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
    res.models = RknnModelSet.open(
        LPR_DETECTOR_MODEL,
        LPR_RECOGNIZER_MODEL,
        None,
        False,
        RKNN,
        log_fn=log,
    )
    handles = res.models.handles
    validate_lpr_runtime(
        [("cam1", handles.cam1_detector), ("cam3", handles.cam3_detector)],
        [("cam1", handles.cam1_ocr), ("cam3", handles.cam3_ocr)],
        lpr_charset,
        model_paths=_LPR_BUNDLE_PATHS,
        log_fn=log,
    )

    res.plate_tracker = PlateTracker()
    if MQTT_ENABLED:
        res.mqtt_svc = MqttService(on_log=log)
        res.mqtt_svc.start()
        res.mqtt_started = True
    else:
        log("INFO", "MQTT disabled (MQTT_ENABLED=False)")

    res.cam1 = CameraGrabber(
        RTSP_URL, "cam1", handles.cam1_detector, handles.cam1_ocr,
        CAM1_LPR_CROP, expected_resolution=CAM1_EXPECTED_RESOLUTION,
    )
    res.cam1.start()
    res.cam3 = CameraGrabber(
        RTSP_URL_3, "cam3", handles.cam3_detector, handles.cam3_ocr,
        CAM3_LPR_CROP,
    )
    res.cam3.start()
    res.grabber2 = FrameGrabber(RTSP_URL_2)
    res.grabber2.start()

    vehicle_tracker = None

    res.reader = D2008Reader(
        port=SERIAL_PORT,
        baud=BAUD_RATE,
        db_file=SCALE_DATA_DIR,
        log_interval=0.2,
    )
    res.storage_maintenance = StorageMaintenance(IMAGE_RETENTION_CHECK_INTERVAL_SECONDS, log_fn=log)
    res.storage_maintenance.start()
    if IMAGE_RETENTION_ENABLED:
        res.retention_cleaner = ImageRetentionCleaner(
            [UNDETECTABLE_DIR, PEAK_CANDIDATE_DIR],
            IMAGE_RETENTION_DAYS,
            IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
            IMAGE_RETENTION_EXTENSIONS,
            log_fn=log,
            pressure_free_bytes=IMAGE_STORAGE_TARGET_FREE_BYTES,
        )
        res.retention_cleaner.start()
        res.minio_cache_cleaner = VerifiedMinioCacheCleaner(
            CAPTURE_DIR,
            3,
            IMAGE_RETENTION_CHECK_INTERVAL_SECONDS,
            ImageSaveWorker._get_minio,
            log_fn=log,
        )
        res.minio_cache_cleaner.start()
        res.diagnostic_archive_cleaner = DiagnosticArchiveCleaner(
            [NO_STABLE_DIR, NO_PLATE_DIR],
            DIAGNOSTIC_ARCHIVE_AFTER_DAYS,
            DIAGNOSTIC_ARCHIVE_RETENTION_DAYS,
            DIAGNOSTIC_ARCHIVE_CHECK_INTERVAL_SECONDS,
            log_fn=log,
        )
        res.diagnostic_archive_cleaner.start()

    if DUPLICATE_REVIEW_ENABLED:
        res.duplicate_reviewer = DuplicateReviewer(log_fn=log)
        res.duplicate_reviewer.reconcile(log)
    res.session_manager = SessionManager(
        plate_tracker=res.plate_tracker,
        mqtt_svc=res.mqtt_svc,
        vehicle_tracker=vehicle_tracker,
        rear_grabber=res.grabber2,
        lpr_grabbers={"cam1": res.cam1, "cam3": res.cam3},
        save_images_fn=ImageSaveWorker.save_and_upload_now,
        undetectable_dir=UNDETECTABLE_DIR,
        cam2_result_crop=CAM2_RESULT_CROP,
        lpr_light_controller=CameraLightController(
            [RTSP_URL, RTSP_URL_2, RTSP_URL_3], datetime.now,
        ),
        duplicate_reviewer=res.duplicate_reviewer,
    )
    res.detect_coord = DetectCoordinator(
        [res.cam1, res.cam3], res.plate_tracker,
        presence_callback=lambda camera, valid, revision: res.session_manager.on_plate_presence(
            camera, valid, log, revision=revision
        ),
        session_context=res.session_manager.session_context,
    )
    res.detect_coord.configure_split_pipeline(
        detect_plate_regions,
        recognize_plate_regions,
        lpr_charset,
    )
    res.frame_spool = SessionFrameSpool(
        LPR_SPOOL_DIR,
        res.cam1,
        res.cam3,
        cam2_grabber=res.grabber2,
        interval=SESSION_FRAME_INTERVAL_SECONDS,
        jpeg_quality=SESSION_FRAME_JPEG_QUALITY,
        notification_queue_size=SESSION_FRAME_QUEUE_SIZE,
        disk_cap_bytes=SESSION_FRAME_DISK_CAP_BYTES,
        min_free_bytes=SESSION_FRAME_MIN_FREE_BYTES,
        metadata_provider=res.detect_coord.get_frame_metadata,
        max_frames_per_camera=LPR_DEFERRED_MAX_FRAMES_PER_CAMERA,
    )
    res.session_manager.frame_spool = res.frame_spool
    res.deferred_lpr = DeferredLprWorker(
        res.frame_spool,
        [res.cam1, res.cam3],
        lpr_charset,
        lambda metadata, tracker: res.session_manager.finalize_deferred_session(metadata, tracker, log),
        detect_regions_fn=detect_plate_regions,
        recognize_regions_fn=recognize_plate_regions,
        tracker_factory=lambda: PlateTracker(max_plate_images=2),
        job_interval=1.0,
        memory_cleanup_fn=_malloc_trim,
        log_fn=log,
    )
    ImageSaveWorker.start_upload_worker()
    res.image_worker_started = True
    if MQTT_ENABLED and res.mqtt_svc:
        PublishOutbox.start(res.mqtt_svc)
        res.outbox_started = True
    res.frame_spool.start()
    res.deferred_lpr.start()
    res.detect_coord.start()
    res.detect_coord.set_enabled(True)

    res.reader.on_weight = lambda frame: res.session_manager.on_weight(frame, log)
    res.reader.on_frame = lambda frame: res.session_manager.on_frame(frame, log)
    res.reader.on_status_change = lambda frame, old, new: res.session_manager.on_status_change(frame, old, new, log)
    res.reader.on_health = lambda event, details: res.session_manager.on_scale_reader_health(event, details, log)
    res.reader.start()
    return res


def shutdown_service(res: ServiceResources, log):
    """Stop every resource, preserving the original teardown order."""
    from services.storage.image_save_worker import ImageSaveWorker
    from services.storage.publish_outbox import PublishOutbox

    log("INFO", "Stopping service...")
    if res.detect_coord:
        res.detect_coord.set_enabled(False)
        res.detect_stopped = res.cleanup("detect_coordinator", res.detect_coord.stop, log) is not False
    if res.reader:
        res.reader.on_weight = res.reader.on_frame = res.reader.on_status_change = None
        res.cleanup("scale_reader", res.reader.stop, log)
    if res.session_manager:
        res.cleanup("session_manager", lambda: res.session_manager.shutdown(log), log)
    if res.deferred_lpr:
        res.deferred_stopped = res.cleanup("deferred_lpr", res.deferred_lpr.stop, log) is not False
    if res.frame_spool:
        res.cleanup("session_frame_spool", res.frame_spool.stop, log)
    for name, camera in (("cam2", res.grabber2), ("cam3", res.cam3), ("cam1", res.cam1)):
        if camera and res.cleanup(name, camera.stop, log) is False:
            log("ERROR", f"{name} capture thread did not stop")
    if res.retention_cleaner:
        res.cleanup("image_retention", res.retention_cleaner.stop, log)
    if res.minio_cache_cleaner:
        res.cleanup("minio_cache_retention", res.minio_cache_cleaner.stop, log)
    if res.storage_maintenance:
        res.cleanup("storage_maintenance", res.storage_maintenance.stop, log)
    if res.diagnostic_archive_cleaner:
        res.cleanup("diagnostic_archive", res.diagnostic_archive_cleaner.stop, log)
    if res.outbox_started:
        res.cleanup("publish_outbox", PublishOutbox.stop, log)
    if res.mqtt_started and res.mqtt_svc:
        res.cleanup("mqtt", res.mqtt_svc.stop, log)
    if res.image_worker_started:
        res.cleanup("image_upload_drain", lambda: ImageSaveWorker.wait_for_pending(timeout=15.0), log)
        res.cleanup("image_upload_worker", ImageSaveWorker.stop, log)
    if res.models:
        res.cleanup(
            "rknn_models",
            lambda: res.models.close(
                release_lpr=res.detect_stopped and res.deferred_stopped,
                release_vehicle=True,
            ),
            log,
        )
