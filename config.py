import importlib.util
import os
from urllib.parse import urlsplit


SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
LPR_DIR = os.path.join(SERVICE_DIR, "yolov5lpr")

SERIAL_PORT = "/dev/ttyS6"
BAUD_RATE = 9600
SCALE_READER_STALL_SECONDS = 30.0
SCALE_READER_RECONNECT_INITIAL_SECONDS = 1.0
SCALE_READER_RECONNECT_MAX_SECONDS = 30.0
# Consecutive serial-open failures before the reader is declared unrecoverable.
# A silent-but-open port keeps reconnecting forever; an absent port fails fast so
# the service can restart rather than run forever with no scale input.
SCALE_READER_MAX_OPEN_ATTEMPTS = 5
# Camera addressing is not secret; credentials come from the host environment
# or config.local.py and are never stored in tracked code.
RTSP_HOST_CAM1 = "192.168.1.181"
RTSP_HOST_CAM2 = "192.168.1.179"
RTSP_HOST_CAM3 = "192.168.1.177"
RTSP_PATH = "/ch01/0"
RTSP_USERNAME = ""
RTSP_PASSWORD = ""
RTSP_URL = ""
RTSP_URL_2 = ""
RTSP_URL_3 = ""
CAM1_EXPECTED_RESOLUTION = None

WEIGHT_THRESHOLD = 100.0
LOG_PRINT_INTERVAL = 1.0
RECONNECT_DELAY = 5
LOG_DIR = os.path.join(SERVICE_DIR, "logs")
LOG_FILE_PREFIX = "weighing_service"
LOG_FILE_PATH = os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}.log")
SCALE_DATA_DIR = os.path.join(SERVICE_DIR, "scale_data")
CAPTURE_DIR = os.path.join(SERVICE_DIR, "storage", "weighbridge")
UNDETECTABLE_DIR = os.path.join(SERVICE_DIR, "storage", "undetectable")
NO_STABLE_DIR = os.path.join(SERVICE_DIR, "storage", "no-stable")
PEAK_CANDIDATE_DIR = os.path.join(SERVICE_DIR, "storage", "peak-candidates")
NO_PLATE_DIR = os.path.join(SERVICE_DIR, "storage", "no-plate")
LPR_SPOOL_DIR = os.path.join(SERVICE_DIR, "storage", "lpr-spool")
SESSION_DEDUP_STATE_FILE = os.path.join(SERVICE_DIR, "storage", "session-dedup-state.json")
SESSION_FINALIZATION_DB = os.path.join(SERVICE_DIR, "storage", "session-finalization.db")
MQTT_ENABLED = True
MQTT_HOST = "103.75.184.181"
MQTT_PORT = 1883
MQTT_USERNAME = ""
MQTT_PASSWORD = ""
MQTT_QOS = 1
MQTT_KEEPALIVE = 30
WEIGHBRIDGE_ID = "9aa29a10-6605-47dd-9460-970d66c3d1c3"
MQTT_WEIGHBRIDGE_TOPIC_ID = WEIGHBRIDGE_ID
MQTT_CLIENT_ID = f"smartport-weighbridge-{MQTT_WEIGHBRIDGE_TOPIC_ID}"
MQTT_TOPIC = (
    "m/2e206e45-9c8e-4be0-a97a-b25e49cac58d/"
    "c/d3fc99f9-76ac-4047-807d-b04759f798fc/"
    "Hub/AIBOXCAN/weighbridge/"
    f"{MQTT_WEIGHBRIDGE_TOPIC_ID}/events"
)
DEFAULT_TRANSACTION_TYPE = "gate_out"
IMAGE_RETENTION_ENABLED = True
IMAGE_RETENTION_DAYS = 30
IMAGE_RETENTION_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
IMAGE_RETENTION_EXTENSIONS = {".jpg", ".jpeg", ".png", ".json"}
LOG_RETENTION_DAYS = 60
LOG_COMPRESSION_ENABLED = False
LOG_COMPRESS_AFTER_DAYS = 1
SCALE_DATA_RETENTION_DAYS = 365
PENDING_RETENTION_DAYS = 30
MQTT_DEAD_LETTER_RETENTION_DAYS = 365
IMAGE_DEAD_LETTER_RETENTION_DAYS = 90
DIAGNOSTIC_ARCHIVE_AFTER_DAYS = 2
DIAGNOSTIC_ARCHIVE_RETENTION_DAYS = 30
DIAGNOSTIC_ARCHIVE_CHECK_INTERVAL_SECONDS = 60 * 60
SQLITE_PASSIVE_CHECKPOINT_SECONDS = 60 * 60
SQLITE_TRUNCATE_CHECKPOINT_SECONDS = 24 * 60 * 60
SQLITE_WAL_TRUNCATE_BYTES = 100 * 1024 * 1024
RESULT_JPEG_QUALITY = 82

MINIO_ENDPOINT = "storage.mobifone.vn:8443"
MINIO_ACCESS_KEY = ""
MINIO_SECRET_KEY = ""
MINIO_BUCKET = "vns-camera-poc"
MINIO_SECURE = True
MINIO_REGION = "HCM-Q9"
STABLE_COUNT_THRESHOLD = 1
SESSION_END_EMPTY_DWELL_SECONDS = 2.0
SESSION_PLATE_ONLY_TIMEOUT_SECONDS = 180.0
SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT = True
SESSION_WEIGHT_DEPARTURE_KG = 500.0
SESSION_WEIGHT_DEPARTURE_DWELL_SECONDS = 2.0
SESSION_WEIGHT_TREND_FRAMES = 6
SESSION_WEIGHT_TREND_DIRECTIONAL_STEPS = 4
PEAK_FILTER_FRAMES = 5
PEAK_MOVEMENT_CONFIRM_FRAMES = 5
PEAK_MOVEMENT_CANCEL_KG = 250.0
SAVE_ABSORBED_PEAK_CANDIDATE_EVIDENCE = True
SESSION_REARM_DELAY_SECONDS = 2.0
SESSION_STABLE_WEIGHT_WINDOW = 25
SESSION_FRAME_INTERVAL_SECONDS = 0.2
SESSION_FRAME_JPEG_QUALITY = 82
SESSION_FRAME_QUEUE_SIZE = 32
SESSION_FRAME_DISK_CAP_BYTES = 20 * 1024 * 1024 * 1024
SESSION_FRAME_MIN_FREE_BYTES = 5 * 1024 * 1024 * 1024
IMAGE_STORAGE_TARGET_FREE_BYTES = 7 * 1024 * 1024 * 1024
UNKNOWN_PHOTO_LOCAL_PEAK_DWELL_SECONDS = 1.0
UNKNOWN_PHOTO_LOCAL_PEAK_DROP_KG = 300.0

# Duplicate session review (mock-only; enable per host via config.local.py)
DUPLICATE_REVIEW_ENABLED = False
DUPLICATE_REVIEW_DB = os.path.join(SERVICE_DIR, "storage", "duplicate-review.db")
DUPLICATE_REVIEW_MOCK_DIR = os.path.join(SERVICE_DIR, "storage", "review-mock")
DUPLICATE_REVIEW_MAX_GAP_SECONDS = 15.0
DUPLICATE_REVIEW_RECONCILE_HOURS = 48.0

LP_DETECTOR_RKNN = os.path.join(LPR_DIR, "model", "LP_detector.rknn")
LP_OCR_RKNN = os.path.join(LPR_DIR, "model", "LP_ocr.rknn")
LPR_DETECTOR_MODEL = os.path.join(SERVICE_DIR, "models", "lpr", "license_plate_detector.rknn")
LPR_RECOGNIZER_MODEL = os.path.join(SERVICE_DIR, "models", "lpr", "license_plate_recognizer.rknn")
LPR_CHARSET = os.path.join(SERVICE_DIR, "models", "lpr", "charset.txt")
LPR_IMAGE_SIZE = 960
LPR_FALLBACK_DETECTOR_MODEL = ""
LPR_FALLBACK_IMAGE_SIZE = 640
LPR_FALLBACK_OCR_CONFIDENCE = 0.98
LPR_OCR_TOPK = 10
LPR_OCR_BEAM_WIDTH = 50
LPR_OCR_MIN_CONFIDENCE = None
LPR_LIVE_OCR_INTERVAL_SECONDS = 0.5
LPR_DEFERRED_MAX_FRAMES_PER_CAMERA = 30
FRAME_GRAB_DRAIN_MAX = 8
FRAME_GRAB_DRAIN_SECONDS = 0.02

IMG_SIZE = 640
DET_CONF_THRES = 0.25
DET_IOU_THRES = 0.45
OCR_CONF_THRES = 0.60
OCR_IOU_THRES = 0.45
MIN_CROP_W = 70
MIN_CROP_H = 35
MAX_PLATE_OCR_CANDIDATES = 3

DETECT_FPS = 20
PLATE_TRACK_STALE_SECONDS = 3.0
PLATE_CONFIRM_THRESHOLD = 2
MIN_SELECTED_PLATE_HITS = 2
MIN_PLATE_OBSERVATION_SPAN_SECONDS = 1.0
WEIGHT_CHANGE_THRESHOLD = 500.0
SESSION_END_WEIGHT_DROP_THRESHOLD = 300.0

# Main pipeline defaults to full cam1/cam3 frames. Crop modes apply only to LPR inference.
CAM2_RESULT_CROP = "left"  # "left", "right", or "full"

# Kept for scripts that still import these names; production cam1/cam3 use full frames.
CAM1_LPR_CROP = "full"
CAM3_LPR_CROP = "full"
CAM1_RESULT_CROP = "full"
CAM3_RESULT_CROP = "full"

YOLO26_ENABLED = False
YOLO26_MODEL_PATH = os.path.join(SERVICE_DIR, "models", "yolo26n_native_rk3588_fp.rknn")
YOLO26_DETECT_FPS = 5
YOLO26_CONF_THRES = 0.25
YOLO26_IOU_THRES = 0.35
YOLO26_MIN_BOX_AREA_RATIO = 0.08
YOLO26_MIN_BOTTOM_Y_RATIO = 0.55
YOLO26_STATIONARY_SECONDS = 1.0
YOLO26_STATIONARY_PIXELS_THRESHOLD = 12.0

VEHICLE_CLASS_IDS = {2, 3, 5, 7}

VEHICLE_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

_LOCAL_CONFIG = os.path.join(SERVICE_DIR, "config.local.py")
_LOCAL_OVERRIDES = set()
if os.path.exists(_LOCAL_CONFIG):
    _spec = importlib.util.spec_from_file_location("config_local", _LOCAL_CONFIG)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    for _name in dir(_module):
        if _name.isupper():
            globals()[_name] = getattr(_module, _name)
            _LOCAL_OVERRIDES.add(_name)

# Environment values win over config.local.py for secrets and endpoints.
# systemd supplies them through /etc/weighing-meter/weighing.env.
_ENV_OVERRIDES = {
    "WEIGHBRIDGE_ID": "WEIGHING_WEIGHBRIDGE_ID",
    "RTSP_URL": "WEIGHING_RTSP_URL",
    "RTSP_URL_2": "WEIGHING_RTSP_URL_2",
    "RTSP_URL_3": "WEIGHING_RTSP_URL_3",
    "RTSP_USERNAME": "WEIGHING_RTSP_USERNAME",
    "RTSP_PASSWORD": "WEIGHING_RTSP_PASSWORD",
    "MQTT_HOST": "WEIGHING_MQTT_HOST",
    "MQTT_PORT": "WEIGHING_MQTT_PORT",
    "MQTT_USERNAME": "WEIGHING_MQTT_USERNAME",
    "MQTT_PASSWORD": "WEIGHING_MQTT_PASSWORD",
    "MINIO_ENDPOINT": "WEIGHING_MINIO_ENDPOINT",
    "MINIO_ACCESS_KEY": "WEIGHING_MINIO_ACCESS_KEY",
    "MINIO_SECRET_KEY": "WEIGHING_MINIO_SECRET_KEY",
}
for _name, _env_name in _ENV_OVERRIDES.items():
    _env_value = os.environ.get(_env_name)
    if _env_value:
        if _name == "MQTT_PORT":
            try:
                _env_value = int(_env_value)
            except ValueError:
                _env_value = -1
        globals()[_name] = _env_value
        _LOCAL_OVERRIDES.add(_name)

if "RTSP_URL" not in _LOCAL_OVERRIDES:
    RTSP_URL = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_HOST_CAM1}:554{RTSP_PATH}"
if "RTSP_URL_2" not in _LOCAL_OVERRIDES:
    RTSP_URL_2 = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_HOST_CAM2}:554{RTSP_PATH}"
if "RTSP_URL_3" not in _LOCAL_OVERRIDES:
    RTSP_URL_3 = f"rtsp://{RTSP_USERNAME}:{RTSP_PASSWORD}@{RTSP_HOST_CAM3}:554{RTSP_PATH}"

if "MQTT_WEIGHBRIDGE_TOPIC_ID" not in _LOCAL_OVERRIDES:
    MQTT_WEIGHBRIDGE_TOPIC_ID = WEIGHBRIDGE_ID
if "MQTT_CLIENT_ID" not in _LOCAL_OVERRIDES:
    MQTT_CLIENT_ID = f"smartport-weighbridge-{MQTT_WEIGHBRIDGE_TOPIC_ID}"
if "MQTT_TOPIC" not in _LOCAL_OVERRIDES:
    MQTT_TOPIC = (
        "m/2e206e45-9c8e-4be0-a97a-b25e49cac58d/"
        "c/d3fc99f9-76ac-4047-807d-b04759f798fc/"
        "Hub/AIBOXCAN/weighbridge/"
        f"{MQTT_WEIGHBRIDGE_TOPIC_ID}/events"
    )

_HOST_POLICIES = {
    "100ecc11-dbcb-4c23-8e89-d41ccefcda37": {
        "DEFAULT_TRANSACTION_TYPE": "gate_in",
        "SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT": False,
    },
    "9aa29a10-6605-47dd-9460-970d66c3d1c3": {
        "DEFAULT_TRANSACTION_TYPE": "gate_out",
        "DIAGNOSTIC_ARCHIVE_RETENTION_DAYS": 14,
        "LOG_COMPRESSION_ENABLED": True,
        "LOG_RETENTION_DAYS": 14,
        "SAVE_ABSORBED_PEAK_CANDIDATE_EVIDENCE": False,
        "SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT": True,
    },
}
_HOST_POLICY = _HOST_POLICIES.get(WEIGHBRIDGE_ID)
if _HOST_POLICY:
    for _name, _value in _HOST_POLICY.items():
        if _name not in _LOCAL_OVERRIDES:
            globals()[_name] = _value


def validate_runtime_config():
    errors = []

    def require_text(name):
        value = globals().get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} must be a non-empty string")

    for name in (
        "MINIO_ENDPOINT", "MINIO_ACCESS_KEY", "MINIO_SECRET_KEY", "MINIO_BUCKET",
    ):
        require_text(name)

    def require_rtsp(name):
        value = globals().get(name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{name} must be a non-empty string")
            return
        parts = urlsplit(value)
        if not parts.hostname:
            errors.append(f"{name} must include a camera host")
        if parts.username is not None and (not parts.username or not parts.password):
            errors.append(f"{name} credentials are incomplete")

    for name in ("RTSP_URL", "RTSP_URL_2", "RTSP_URL_3"):
        require_rtsp(name)
    if DEFAULT_TRANSACTION_TYPE not in (
        "gate_in", "gate_out", "vgm", "reweigh", "spot_check",
    ):
        errors.append("DEFAULT_TRANSACTION_TYPE is invalid")
    if not isinstance(SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT, bool):
        errors.append("SESSION_CONTINUE_AFTER_PLATE_LOSS_WITH_WEIGHT must be a boolean")
    if IMAGE_STORAGE_TARGET_FREE_BYTES < SESSION_FRAME_MIN_FREE_BYTES:
        errors.append("IMAGE_STORAGE_TARGET_FREE_BYTES must be >= SESSION_FRAME_MIN_FREE_BYTES")
    if LPR_FALLBACK_DETECTOR_MODEL and not os.path.isfile(LPR_FALLBACK_DETECTOR_MODEL):
        errors.append("LPR_FALLBACK_DETECTOR_MODEL must be an existing file when configured")
    if _HOST_POLICY:
        for name, expected in _HOST_POLICY.items():
            if globals().get(name) != expected:
                errors.append(f"{name} conflicts with canonical host policy")

    if MQTT_ENABLED:
        for name in (
            "WEIGHBRIDGE_ID", "MQTT_WEIGHBRIDGE_TOPIC_ID", "MQTT_CLIENT_ID",
            "MQTT_HOST", "MQTT_USERNAME", "MQTT_PASSWORD", "MQTT_TOPIC",
        ):
            require_text(name)
        if not isinstance(MQTT_PORT, int) or isinstance(MQTT_PORT, bool) or MQTT_PORT <= 0:
            errors.append("MQTT_PORT must be a positive integer")
        if MQTT_QOS not in (0, 1, 2):
            errors.append("MQTT_QOS must be 0, 1, or 2")
        if MQTT_WEIGHBRIDGE_TOPIC_ID != WEIGHBRIDGE_ID:
            errors.append("MQTT_WEIGHBRIDGE_TOPIC_ID must equal WEIGHBRIDGE_ID")
        if (
            isinstance(MQTT_WEIGHBRIDGE_TOPIC_ID, str)
            and isinstance(MQTT_CLIENT_ID, str)
            and MQTT_WEIGHBRIDGE_TOPIC_ID
            and MQTT_CLIENT_ID
            and MQTT_WEIGHBRIDGE_TOPIC_ID not in MQTT_CLIENT_ID
        ):
            errors.append("MQTT_CLIENT_ID does not contain MQTT_WEIGHBRIDGE_TOPIC_ID")
        if (
            isinstance(MQTT_WEIGHBRIDGE_TOPIC_ID, str)
            and isinstance(MQTT_TOPIC, str)
            and MQTT_WEIGHBRIDGE_TOPIC_ID
            and MQTT_TOPIC
            and MQTT_WEIGHBRIDGE_TOPIC_ID not in MQTT_TOPIC
        ):
            errors.append("MQTT_TOPIC does not contain MQTT_WEIGHBRIDGE_TOPIC_ID")

    if errors:
        raise ValueError("Invalid runtime configuration: " + "; ".join(errors))
