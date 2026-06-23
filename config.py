import importlib.util
import os


SERVICE_DIR = os.path.dirname(os.path.abspath(__file__))
LPR_DIR = os.path.join(SERVICE_DIR, "yolov5lpr")

SERIAL_PORT = "/dev/ttyS6"
BAUD_RATE = 9600
RTSP_URL = "rtsp://admin:123456@192.168.1.181:554/ch01/0"
RTSP_URL_2 = "rtsp://admin:123456@192.168.1.19:554/ch01/0"
RTSP_URL_3 = "rtsp://admin:123456@192.168.1.177:554/ch01/0"

WEIGHT_THRESHOLD = 100.0
LOG_PRINT_INTERVAL = 1.0
RECONNECT_DELAY = 5
LOG_DIR = os.path.join(SERVICE_DIR, "logs")
LOG_FILE_PREFIX = "weighing_service"
LOG_FILE_PATH = os.path.join(LOG_DIR, f"{LOG_FILE_PREFIX}.log")
CAPTURE_DIR = os.path.join(SERVICE_DIR, "storage", "weighbridge")
UNDETECTABLE_DIR = os.path.join(SERVICE_DIR, "storage", "undetectable")
MQTT_ENABLED = True
MQTT_HOST = "103.75.184.185"
MQTT_PORT = 1883
MQTT_USERNAME = "90157317-f4b2-48d2-8d8b-d5a9e899bad2"
MQTT_PASSWORD = "0b91a95f-7b15-4778-a700-a1994878112c"
MQTT_QOS = 1
MQTT_KEEPALIVE = 60
WEIGHBRIDGE_ID = "9aa29a10-6605-47dd-9460-970d66c3d1c3"
MQTT_WEIGHBRIDGE_TOPIC_ID = WEIGHBRIDGE_ID[:8]
MQTT_CLIENT_ID = f"smartport-weighbridge-{MQTT_WEIGHBRIDGE_TOPIC_ID}"
MQTT_TOPIC = (
    "m/2e206e45-9c8e-4be0-a97a-b25e49cac58d/"
    "c/d3fc99f9-76ac-4047-807d-b04759f798fc/"
    "Hub/AIBOXCAN/weighbridge/"
    f"{MQTT_WEIGHBRIDGE_TOPIC_ID}/events"
)
DEFAULT_TRANSACTION_TYPE = "gate_in"
IMAGE_RETENTION_ENABLED = True
IMAGE_RETENTION_DAYS = 30
IMAGE_RETENTION_CHECK_INTERVAL_SECONDS = 24 * 60 * 60
IMAGE_RETENTION_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RESULT_JPEG_QUALITY = 82

MINIO_ENDPOINT = "103.75.184.181:59000"
MINIO_ACCESS_KEY = "smartport"
MINIO_SECRET_KEY = "smartport123"
MINIO_BUCKET = "smartport"
MINIO_SECURE = False
STABLE_COUNT_THRESHOLD = 3
SESSION_END_EMPTY_DWELL_SECONDS = 2.0

LP_DETECTOR_RKNN = os.path.join(LPR_DIR, "model", "LP_detector.rknn")
LP_OCR_RKNN = os.path.join(LPR_DIR, "model", "LP_ocr.rknn")
LPR_DETECTOR_MODEL = os.path.join(SERVICE_DIR, "models", "lpr", "license_plate_detector.rknn")
LPR_RECOGNIZER_MODEL = os.path.join(SERVICE_DIR, "models", "lpr", "license_plate_recognizer.rknn")
LPR_CHARSET = os.path.join(SERVICE_DIR, "models", "lpr", "charset.txt")
LPR_IMAGE_SIZE = 960
LPR_OCR_TOPK = 10
LPR_OCR_BEAM_WIDTH = 50
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
PLATE_CONFIRM_THRESHOLD = 2
MIN_SELECTED_PLATE_HITS = 2
MIN_PLATE_OBSERVATION_SPAN_SECONDS = 1.0
WEIGHT_CHANGE_THRESHOLD = 500.0
SESSION_END_WEIGHT_DROP_THRESHOLD = 300.0

# Main pipeline uses full cam1/cam3 frames. Only cam2 rear result image is cropped.
CAM2_RESULT_CROP = "left"  # "left", "right", or "full"

# Kept for scripts that still import these names; production cam1/cam3 use full frames.
CAM1_LPR_CROP = "full"
CAM3_LPR_CROP = "full"
CAM1_RESULT_CROP = "full"
CAM3_RESULT_CROP = "full"

YOLO26_ENABLED = True
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
if os.path.exists(_LOCAL_CONFIG):
    _spec = importlib.util.spec_from_file_location("config_local", _LOCAL_CONFIG)
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    for _name in dir(_module):
        if _name.isupper():
            globals()[_name] = getattr(_module, _name)
