from .inference import (
    letterbox,
    preprocess,
    xywh2xyxy,
    nms_boxes,
    postprocess,
    yolo26_postprocess,
    detect_plates_rknn,
    detect_vehicles_rknn,
)
import importlib

_OCR_EXPORTS = {
    "detect_plates_in_frame",
    "_deskew_optimized",
    "_upscale_plate",
    "_compute_skew_angles",
    "read_plate_rknn",
    "_decode_ocr_detections",
    "_classify_plate_layout",
    "_assemble_plate_string",
    "_format_plate_number",
    "set_ocr_model",
}


def __getattr__(name):
    if name in _OCR_EXPORTS:
        ocr = importlib.import_module(f"{__name__}.ocr")
        return getattr(ocr, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "letterbox",
    "preprocess",
    "xywh2xyxy",
    "nms_boxes",
    "postprocess",
    "yolo26_postprocess",
    "detect_plates_rknn",
    "detect_vehicles_rknn",
    "detect_plates_in_frame",
    "_deskew_optimized",
    "_upscale_plate",
    "_compute_skew_angles",
    "read_plate_rknn",
    "_decode_ocr_detections",
    "_classify_plate_layout",
    "_assemble_plate_string",
    "_format_plate_number",
    "set_ocr_model",
]
