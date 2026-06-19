"""OCR pipeline for license plate recognition."""

import function.utils_rotate as utils_rotate
import math

import cv2
import numpy as np

from config import MIN_CROP_H, MIN_CROP_W, OCR_CONF_THRES, OCR_IOU_THRES, IMG_SIZE

# Import inference helpers
from .inference import preprocess, postprocess

OCR_CLASSES = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "K",
    "L",
    "M",
    "N",
    "P",
    "S",
    "T",
    "U",
    "V",
    "X",
    "Y",
    "Z",
    "0",
]

_ocr_model = None


def set_ocr_model(model):
    """Set the default OCR model for standalone usage."""
    global _ocr_model
    _ocr_model = model


def _decode_ocr_detections(dets):
    """Convert OCR detection boxes to character centers with recognized chars."""
    center_list = []
    y_sum = 0.0
    for d in dets:
        x1, y1, x2, y2, conf, cls_id = d
        x_c = (x1 + x2) / 2
        y_c = (y1 + y2) / 2
        y_sum += y_c
        char = OCR_CLASSES[cls_id] if cls_id < len(OCR_CLASSES) else "?"
        center_list.append([x_c, y_c, char])
    return center_list, y_sum


def _classify_plate_layout(center_list):
    """Determine if plate is 1-line or 2-line layout."""
    if not center_list:
        return "1"
    l_point = center_list[0]
    r_point = center_list[0]
    for cp in center_list:
        if cp[0] < l_point[0]:
            l_point = cp
        if cp[0] > r_point[0]:
            r_point = cp
    if l_point[0] == r_point[0]:
        return "1"
    x1l, y1l = l_point[0], l_point[1]
    x2l, y2l = r_point[0], r_point[1]
    if x1l == 0:
        return "1"
    b = y1l - (y2l - y1l) * x1l / (x2l - x1l)
    a = (y1l - b) / x1l
    for ct in center_list:
        if not math.isclose(a * ct[0] + b, ct[1], abs_tol=3):
            return "2"
    return "1"


def _assemble_plate_string(center_list, plate_type):
    """Assemble final plate string from character centers."""
    y_mean = int(sum(c[1] for c in center_list) / len(center_list))
    if plate_type == "2":
        line_1 = sorted([c for c in center_list if int(c[1]) <= y_mean], key=lambda x: x[0])
        line_2 = sorted([c for c in center_list if int(c[1]) > y_mean], key=lambda x: x[0])
        prefix = "".join(str(c[2]) for c in line_1)
        digits = "".join(str(c[2]) for c in line_2)
    else:
        sorted_chars = sorted(center_list, key=lambda x: x[0])
        last_letter = -1
        for i, c in enumerate(sorted_chars):
            if str(c[2]).isalpha():
                last_letter = i
        if last_letter >= 0:
            prefix = "".join(str(c[2]) for c in sorted_chars[: last_letter + 1])
            digits = "".join(str(c[2]) for c in sorted_chars[last_letter + 1 :])
        else:
            prefix = ""
            digits = "".join(str(c[2]) for c in sorted_chars)
    return prefix, digits


def _format_plate_number(prefix, digits):
    """Format digits with decimal point and combine with prefix."""
    if len(digits) > 2:
        digits = digits[:-2] + "." + digits[-2:]
    return prefix + "-" + digits if prefix else digits


def read_plate_rknn(crop_img, ocr=None):
    """Run OCR on a cropped plate image via RKNN. Returns plate string or 'unknown'."""
    if ocr is None:
        if _ocr_model is None:
            return "unknown"
        ocr = _ocr_model
    img_input, ratio, dw, dh = preprocess(crop_img)
    outputs = ocr.inference(inputs=[img_input], data_format="nhwc")
    dets = postprocess(outputs, OCR_CONF_THRES, OCR_IOU_THRES, ratio, dw, dh, crop_img.shape)
    del img_input, outputs

    if len(dets) < 7 or len(dets) > 10:
        return "unknown"

    center_list, _ = _decode_ocr_detections(dets)
    plate_type = _classify_plate_layout(center_list)
    prefix, digits = _assemble_plate_string(center_list, plate_type)
    return _format_plate_number(prefix, digits)


def _compute_skew_angles(src_img):
    """Run Canny + HoughLines once, return skew angles for ct=0 and ct=1."""
    import cv2
    import math
    import numpy as np

    if len(src_img.shape) == 3:
        h, w, _ = src_img.shape
    else:
        h, w = src_img.shape
    img = cv2.medianBlur(src_img, 3)
    edges = cv2.Canny(img, threshold1=30, threshold2=100, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, 30, minLineLength=w / 1.5, maxLineGap=h / 3.0)
    if lines is None:
        return 1.0, 1.0

    centers = []
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cy = (y1 + y2) / 2
            centers.append((i, cy))

    angles = []
    for center_thres in (0, 1):
        min_line = 100
        min_line_pos = 0
        for i, cy in centers:
            if center_thres == 1 and cy < 7:
                continue
            if cy < min_line:
                min_line = cy
                min_line_pos = i

        angle = 0.0
        cnt = 0
        for x1, y1, x2, y2 in lines[min_line_pos]:
            ang = np.arctan2(y2 - y1, x2 - x1)
            if math.fabs(ang) <= 30:
                angle += ang
                cnt += 1
        if cnt == 0:
            angles.append(0.0)
        else:
            angles.append((angle / cnt) * 180 / math.pi)

    return angles[0], angles[1]


def _deskew_optimized(crop_img, ocr=None):
    """Optimized deskew with at most 3 OCR attempts and early exit."""
    import function.utils_rotate as utils_rotate

    candidates = []
    angle_ct0, angle_ct1 = _compute_skew_angles(crop_img)

    deskewed = utils_rotate.rotate_image(crop_img, angle_ct0)
    lp = read_plate_rknn(deskewed, ocr=ocr)
    del deskewed
    if lp != "unknown":
        candidates.append(lp)
        return lp, candidates

    if angle_ct1 != angle_ct0:
        deskewed = utils_rotate.rotate_image(crop_img, angle_ct1)
        lp = read_plate_rknn(deskewed, ocr=ocr)
        del deskewed
        if lp != "unknown":
            candidates.append(lp)
            return lp, candidates

    enhanced = utils_rotate.changeContrast(crop_img)
    angle_ct0_e, angle_ct1_e = _compute_skew_angles(enhanced)
    deskewed = utils_rotate.rotate_image(enhanced, angle_ct0_e)
    del enhanced
    lp = read_plate_rknn(deskewed, ocr=ocr)
    del deskewed
    if lp != "unknown":
        candidates.append(lp)
        return lp, candidates

    return "unknown", candidates


UPSCALE_TARGET_W = 240


def _upscale_plate(crop_img):
    """Upscale small plate crops to improve OCR character detection."""
    h, w = crop_img.shape[:2]
    if w >= UPSCALE_TARGET_W:
        return crop_img
    scale = UPSCALE_TARGET_W / w
    new_w = UPSCALE_TARGET_W
    new_h = int(h * scale)
    return cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def detect_plates_in_frame(frame, detector=None, ocr=None):
    """Run full plate detection pipeline on a frame. Returns list of plate detections."""
    from .inference import detect_plates_rknn

    dets = detect_plates_rknn(frame, detector=detector)
    plates = []
    for det in dets:
        x1, y1, x2, y2 = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        det_conf = det[4]
        crop_img = frame[y1:y2, x1:x2].copy()
        if crop_img.size == 0:
            continue
        h, w = crop_img.shape[:2]
        if w < MIN_CROP_W or h < MIN_CROP_H:
            del crop_img
            continue
        crop_img = utils_rotate.perspective_correct(crop_img)
        crop_img = _upscale_plate(crop_img)
        plate_text, candidates = _deskew_optimized(crop_img, ocr=ocr)
        del crop_img
        plates.append(
            {
                "plate": plate_text,
                "bbox": [x1, y1, x2, y2],
                "det_conf": det_conf,
                "crop_size": f"{w}x{h}",
                "votes": len(candidates),
                "candidates": candidates,
            }
        )
    del dets
    return plates
