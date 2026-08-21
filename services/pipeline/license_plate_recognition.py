"""Production license plate recognition pipeline.

Uses YOLOv8-OBB detector + PP-OCR CTC recognizer.
Returns same plate dict contract consumed by DetectCoordinator.
"""

import re
import threading
from pathlib import Path

import cv2
import numpy as np

from config import (
    DET_CONF_THRES,
    DET_IOU_THRES,
    LPR_OCR_BEAM_WIDTH,
    LPR_OCR_MIN_CONFIDENCE,
    LPR_OCR_TOPK,
)
from services.pipeline.detector_obb_decode import decode_detector_outputs
from services.runtime.lpr_bundle import verify_lpr_bundle


CLASS_NAMES = ["BSD", "BSV"]
REC_HEIGHT = 48
REC_WIDTH = 320
REC_WIDTH_BUCKETS = (32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 288, 320)
_NORM_MEAN = 0.5
_NORM_STD = 0.5
_PLATE_CLEAN_RE = re.compile(r"[^A-Z0-9]")
_PLATE_DISPLAY_RE = re.compile(r"[^A-Z0-9.-]")
_PLATE_FORMAT_5_RE = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{5}$")
_PLATE_FORMAT_4_RE = re.compile(r"^\d{2}[A-Z][A-Z0-9]?\d{4}$")
_CROP_INSET_RATIO = 0.05
_TWO_ROW_SLICE_RATIO = 0.55
_MIN_TRACK_CROP_W = 30
_MIN_TRACK_CROP_H = 15
_DIGIT_FIX = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6", "T": "7"})
_LETTER_FIX = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})
_INPUT_BUFFERS = threading.local()


def load_lpr_charset(dict_path):
    chars = Path(dict_path).read_text(encoding="utf-8").splitlines()
    if not chars:
        raise ValueError(f"Empty OCR charset: {dict_path}")
    return ["[blank]"] + chars + [" "]


def validate_lpr_charset(charset):
    expected = ["[blank]", *list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"), " "]
    if charset != expected:
        raise ValueError("Fine-tuned OCR charset must be blank, 0-9, A-Z, then space")


def clean_plate(s):
    return _PLATE_CLEAN_RE.sub("", s.upper())


def clean_plate_display(s):
    return _PLATE_DISPLAY_RE.sub("", s.upper()).strip(".-")


def format_plate_display(s):
    cleaned = clean_plate_display(s)
    normalized = clean_plate(cleaned)
    if _PLATE_FORMAT_5_RE.fullmatch(normalized):
        prefix_len = len(normalized) - 5
        return f"{normalized[:prefix_len]}-{normalized[prefix_len:prefix_len + 3]}.{normalized[prefix_len + 3:]}"
    if _PLATE_FORMAT_4_RE.fullmatch(normalized):
        prefix_len = len(normalized) - 4
        return f"{normalized[:prefix_len]}-{normalized[prefix_len:]}"
    return cleaned


def is_valid_plate_text(s):
    normalized = clean_plate(s)
    return bool(_PLATE_FORMAT_5_RE.fullmatch(normalized) or _PLATE_FORMAT_4_RE.fullmatch(normalized))


def normalize_plate_candidate(s):
    cleaned = clean_plate(s)
    variants = [cleaned]
    if len(cleaned) in (7, 8, 9):
        prefix_len = 4 if len(cleaned) == 9 else 3
        chars = list(cleaned)
        for i, ch in enumerate(chars):
            if i in (0, 1) or i >= prefix_len:
                chars[i] = ch.translate(_DIGIT_FIX)
            elif i == 2:
                chars[i] = ch.translate(_LETTER_FIX)
        normalized = "".join(chars)
        if normalized != cleaned:
            variants.append(normalized)
    return variants


def is_five_digit_plate(s):
    normalized = clean_plate(s)
    return bool(_PLATE_FORMAT_5_RE.fullmatch(normalized))


def is_four_digit_plate(s):
    normalized = clean_plate(s)
    return bool(_PLATE_FORMAT_4_RE.fullmatch(normalized))


def _inset_crop(img_bgr, ratio=_CROP_INSET_RATIO):
    h, w = img_bgr.shape[:2]
    dx = int(round(w * ratio))
    dy = int(round(h * ratio))
    if dx <= 0 and dy <= 0:
        return img_bgr
    if (w - 2 * dx) < 4 or (h - 2 * dy) < 4:
        return img_bgr
    return img_bgr[dy:h - dy, dx:w - dx]


def _emphasize_black(img_bgr, strength=0.35, dark_limit=95, bright_limit=205):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_float = l.astype(np.float32)
    mask = np.clip((bright_limit - l_float) / max(bright_limit - dark_limit, 1), 0.0, 1.0)
    l_new = l_float - (l_float * strength * mask)
    enhanced = cv2.merge((np.clip(l_new, 0, 255).astype(np.uint8), a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _prepare_crop_for_ocr(img_bgr):
    return _emphasize_black(_inset_crop(img_bgr))


def _split_two_row_crop(img_bgr, ratio=_TWO_ROW_SLICE_RATIO):
    h = img_bgr.shape[0]
    split = max(1, min(h, int(round(h * ratio))))
    return img_bgr[:split], img_bgr[h - split:]


def _preprocess_ocr(img_bgr):
    h, w = img_bgr.shape[:2]
    new_w = max(1, min(REC_WIDTH, int(round(w * REC_HEIGHT / max(h, 1)))))
    bucket_w = next((bw for bw in REC_WIDTH_BUCKETS if bw >= new_w), REC_WIDTH)
    resized = cv2.resize(img_bgr, (new_w, REC_HEIGHT), interpolation=cv2.INTER_CUBIC)
    blob = _input_buffer((1, REC_HEIGHT, bucket_w, 3))
    blob.fill((0.0 - _NORM_MEAN) / _NORM_STD)
    target = blob[0, :, :new_w]
    np.multiply(resized[..., ::-1], 1.0 / (255.0 * _NORM_STD), out=target)
    target -= _NORM_MEAN / _NORM_STD
    return blob, (new_w / bucket_w)


def _input_buffer(shape):
    buffers = getattr(_INPUT_BUFFERS, "values", None)
    if buffers is None:
        buffers = {}
        _INPUT_BUFFERS.values = buffers
    buffer = buffers.get(shape)
    if buffer is None:
        buffer = np.empty(shape, dtype=np.float32)
        buffers[shape] = buffer
    return buffer


def _logsumexp(a, b):
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    m = max(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _to_probs(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 38:
        raise ValueError(f"Expected OCR output [T,38], got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("OCR output contains non-finite values")
    if np.min(x) < 0 or np.max(x) > 1.0001:
        raise ValueError("OCR restricted probabilities must be within [0,1]")
    totals = np.sum(x, axis=-1, keepdims=True)
    if np.any(totals <= 0) or np.any(totals > 1.01):
        raise ValueError("OCR restricted probability rows have invalid mass")
    return x / totals


def _ctc_greedy_decode(logits, charset, blank_idx=0):
    probs_all = _to_probs(logits)
    idx = logits.argmax(axis=-1)
    probs = probs_all.max(axis=-1)
    chars, confs = [], []
    prev = -1
    for t, k in enumerate(idx):
        if k != prev and k != blank_idx:
            if int(k) < len(charset):
                chars.append(charset[int(k)])
                confs.append(float(probs[t]))
        prev = int(k)
    text = "".join(chars)
    return text, (float(np.mean(confs)) if confs else 0.0)


def _ctc_decode_topk(logits, charset, topk=None, beam_width=None, blank_idx=0):
    topk = LPR_OCR_TOPK if topk is None else topk
    beam_width = LPR_OCR_BEAM_WIDTH if beam_width is None else beam_width
    probs = _to_probs(logits)
    log_probs = np.log(np.maximum(probs, 1e-12))
    beams = {"": (0.0, -np.inf)}
    for t in range(log_probs.shape[0]):
        next_beams = {}
        candidate_idx = np.argsort(log_probs[t])[-beam_width:]
        for prefix, (p_b, p_nb) in beams.items():
            for c in candidate_idx:
                p = float(log_probs[t, c])
                if int(c) == blank_idx:
                    nb_b, nb_nb = next_beams.get(prefix, (-np.inf, -np.inf))
                    nb_b = _logsumexp(nb_b, _logsumexp(p_b, p_nb) + p)
                    next_beams[prefix] = (nb_b, nb_nb)
                    continue
                if int(c) >= len(charset):
                    continue
                ch = charset[int(c)]
                end = prefix[-1:] if prefix else ""
                new_prefix = prefix if ch == end else prefix + ch
                nb_b, nb_nb = next_beams.get(new_prefix, (-np.inf, -np.inf))
                if ch == end:
                    nb_nb = _logsumexp(nb_nb, p_b + p)
                    old_b, old_nb = next_beams.get(prefix, (-np.inf, -np.inf))
                    old_nb = _logsumexp(old_nb, p_nb + p)
                    next_beams[prefix] = (old_b, old_nb)
                else:
                    nb_nb = _logsumexp(nb_nb, _logsumexp(p_b, p_nb) + p)
                next_beams[new_prefix] = (nb_b, nb_nb)
        beams = dict(sorted(next_beams.items(), key=lambda kv: _logsumexp(kv[1][0], kv[1][1]), reverse=True)[:beam_width])
    ranked = []
    for text, (p_b, p_nb) in beams.items():
        score = _logsumexp(p_b, p_nb)
        ranked.append((text, float(np.exp(score / max(logits.shape[0], 1)))))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:topk]


def _run_ocr_logits(ocr, img_bgr, charset):
    if ocr is None:
        raise ValueError("ocr model is required")
    if not charset:
        raise ValueError("charset is required")
    blob, valid_ratio = _preprocess_ocr(img_bgr)
    outputs = ocr.inference(inputs=[blob], data_format=["nhwc"])
    if len(outputs) != 1:
        raise RuntimeError(f"Expected one OCR output, got {len(outputs)}")
    logits = np.asarray(outputs[0])
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise RuntimeError(f"Unexpected OCR output shape: {logits.shape}")
    logits = logits[0]
    if logits.ndim != 2:
        raise RuntimeError(f"Unexpected OCR output shape: {logits.shape}")
    if logits.shape[-1] != len(charset):
        raise RuntimeError(f"Expected OCR class dimension {len(charset)}, got {logits.shape}")
    if not np.isfinite(logits).all():
        raise RuntimeError("OCR output contains non-finite values")
    valid_t = max(1, min(logits.shape[0], int(round(logits.shape[0] * valid_ratio))))
    return logits[:valid_t]


def _run_ocr(ocr, img_bgr, charset, topk=None):
    logits = _run_ocr_logits(ocr, img_bgr, charset)
    if topk is None:
        return _ctc_greedy_decode(logits, charset)
    return _ctc_decode_topk(logits, charset, topk=topk)


def recognize_old(ocr, charset, img_bgr, two_row=False):
    if two_row:
        top_crop, bottom_crop = _split_two_row_crop(img_bgr)
        top_text, top_conf = _run_ocr(ocr, top_crop, charset)
        bottom_text, bottom_conf = _run_ocr(ocr, bottom_crop, charset)
        text = f"{top_text}-{bottom_text}" if top_text and bottom_text else (top_text or bottom_text)
        conf = (top_conf + bottom_conf) / 2 if (top_conf and bottom_conf) else max(top_conf, bottom_conf)
        return text, conf
    return _run_ocr(ocr, img_bgr, charset)


def recognize_topk(ocr, charset, img_bgr, two_row=False, topk=None):
    topk = LPR_OCR_TOPK if topk is None else topk
    img_bgr = _prepare_crop_for_ocr(img_bgr)
    if two_row:
        top_crop, bottom_crop = _split_two_row_crop(img_bgr)
        top_rows = _run_ocr(ocr, top_crop, charset, topk=topk)
        bottom_rows = _run_ocr(ocr, bottom_crop, charset, topk=topk)
        merged = []
        for top_text, top_conf in top_rows:
            for bottom_text, bottom_conf in bottom_rows:
                text = f"{top_text}-{bottom_text}" if top_text and bottom_text else (top_text or bottom_text)
                conf = (top_conf + bottom_conf) / 2 if (top_conf and bottom_conf) else max(top_conf, bottom_conf)
                merged.append((text, conf))
        dedup = {}
        for text, conf in sorted(merged, key=lambda x: x[1], reverse=True):
            dedup.setdefault(text, conf)
        return list(dedup.items())[:topk]
    return _run_ocr(ocr, img_bgr, charset, topk=topk)


def select_plate_candidate_combined(candidates):
    if not candidates:
        return "", 0.0
    dedup = {}
    for text, conf in candidates:
        for cleaned in normalize_plate_candidate(text):
            if cleaned not in dedup or conf > dedup[cleaned][1]:
                dedup[cleaned] = (format_plate_display(cleaned), conf)
    candidates = list(dedup.values())
    five_digit = [c for c in candidates if is_five_digit_plate(c[0])]
    if five_digit:
        return max(five_digit, key=lambda c: c[1])
    four_digit = [c for c in candidates if is_four_digit_plate(c[0])]
    if four_digit:
        return max(four_digit, key=lambda c: c[1])
    return max(candidates, key=lambda c: c[1])


def recognize_combined(ocr, charset, img_bgr, two_row=False):
    topk = LPR_OCR_TOPK
    img_bgr = _prepare_crop_for_ocr(img_bgr)
    crops = _split_two_row_crop(img_bgr) if two_row else (img_bgr,)
    decoded = []
    for crop in crops:
        logits = _run_ocr_logits(ocr, crop, charset)
        decoded.append((
            _ctc_greedy_decode(logits, charset),
            _ctc_decode_topk(logits, charset, topk=topk),
        ))

    if two_row:
        (top_old, top_rows), (bottom_old, bottom_rows) = decoded
        top_text, top_conf = top_old
        bottom_text, bottom_conf = bottom_old
        old_raw = f"{top_text}-{bottom_text}" if top_text and bottom_text else (top_text or bottom_text)
        old_conf = (top_conf + bottom_conf) / 2 if (top_conf and bottom_conf) else max(top_conf, bottom_conf)
        merged = []
        for top_text, top_conf in top_rows:
            for bottom_text, bottom_conf in bottom_rows:
                text = f"{top_text}-{bottom_text}" if top_text and bottom_text else (top_text or bottom_text)
                conf = (top_conf + bottom_conf) / 2 if (top_conf and bottom_conf) else max(top_conf, bottom_conf)
                merged.append((text, conf))
        dedup = {}
        for text, candidate_conf in sorted(merged, key=lambda item: item[1], reverse=True):
            dedup.setdefault(text, candidate_conf)
        top5 = list(dedup.items())[:topk]
    else:
        (old_raw, old_conf), top5 = decoded[0]
    raw, conf = select_plate_candidate_combined([(old_raw, old_conf), *top5])
    plate = format_plate_display(raw)
    return plate, conf, raw, top5


def _preprocess_detector(img, size):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    blob = _input_buffer((1, size, size, 3))
    blob.fill(114.0 / 255.0)
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    target = blob[0, dy:dy + new_h, dx:dx + new_w]
    np.multiply(resized[..., ::-1], 1.0 / 255.0, out=target)
    return blob, scale, dx, dy


def _decode_obb(output, num_classes, conf_thres):
    out = output[0].transpose(1, 0)
    boxes = out[:, :4]
    cls_scores = out[:, 4:4 + num_classes]
    angles = out[:, 4 + num_classes]

    cls_id = cls_scores.argmax(axis=1)
    conf = cls_scores.max(axis=1)
    keep = conf >= conf_thres
    if not keep.any():
        return (np.empty((0, 5), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.int64))
    xywhr = np.concatenate([boxes[keep], angles[keep, None]], axis=1)
    return xywhr.astype(np.float32), conf[keep].astype(np.float32), cls_id[keep]


def _rotated_nms(xywhr, scores, iou_thres):
    if len(xywhr) == 0:
        return np.empty((0,), np.int64)
    rects = [((float(b[0]), float(b[1])), (float(b[2]), float(b[3])), float(np.degrees(b[4]))) for b in xywhr]
    keep = cv2.dnn.NMSBoxesRotated(rects, scores.tolist(), 0.0, iou_thres)
    if isinstance(keep, np.ndarray):
        return keep.flatten().astype(np.int64)
    if len(keep) == 0:
        return np.empty((0,), np.int64)
    return np.array(keep, dtype=np.int64).flatten()


def _xywhr_to_corners(xywhr):
    corners = np.zeros((len(xywhr), 4, 2), dtype=np.float32)
    for i, (cx, cy, w, h, r) in enumerate(xywhr):
        corners[i] = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(np.degrees(r))))
    return corners


def _unletterbox(pts, scale, dx, dy):
    out = pts.copy()
    out[..., 0] = (out[..., 0] - dx) / scale
    out[..., 1] = (out[..., 1] - dy) / scale
    return out


def crop_obb(img_bgr, xy, pad_ratio=0.0):
    pts = np.asarray(xy, dtype=np.float32).reshape(4, 2)
    order_y = np.argsort(pts[:, 1])
    top2 = pts[order_y[:2]]
    bot2 = pts[order_y[2:]]
    tl, tr = top2[np.argsort(top2[:, 0])]
    bl, br = bot2[np.argsort(bot2[:, 0])]

    out_w = int(round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2))
    out_h = int(round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2))
    if out_w < 2 or out_h < 2:
        return None

    pad_x = int(round(out_w * pad_ratio))
    pad_y = int(round(out_h * pad_ratio))
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([
        [pad_x, pad_y],
        [pad_x + out_w, pad_y],
        [pad_x + out_w, pad_y + out_h],
        [pad_x, pad_y + out_h],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        img_bgr,
        matrix,
        (out_w + 2 * pad_x, out_h + 2 * pad_y),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def detect_obb_plates(frame, detector, imgsz=960, conf_thres=None, iou_thres=None):
    if detector is None:
        raise ValueError("detector is required for OBB plate detection")
    conf = DET_CONF_THRES if conf_thres is None else conf_thres
    iou = DET_IOU_THRES if iou_thres is None else iou_thres
    blob, scale, dx, dy = _preprocess_detector(frame, imgsz)
    outputs = detector.inference(inputs=[blob], data_format=["nhwc"])
    raw = decode_detector_outputs(outputs)
    if not np.isfinite(raw).all():
        raise ValueError("Detector output contains non-finite values")
    num_classes = int(raw.shape[1]) - 5
    xywhr, scores, classes = _decode_obb(raw, num_classes, conf)
    keep = _rotated_nms(xywhr, scores, iou)
    xywhr, scores, classes = xywhr[keep], scores[keep], classes[keep]
    corners = _unletterbox(_xywhr_to_corners(xywhr), scale, dx, dy)
    return corners, scores, classes


def detect_plate_regions(frame, detector=None, imgsz=960, conf_thres=None, iou_thres=None, pad_ratio=0.0):
    corners, scores, classes = detect_obb_plates(
        frame,
        detector,
        imgsz=imgsz,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )
    order = np.argsort(scores)[::-1]

    regions = []
    frame_h, frame_w = frame.shape[:2]
    for idx in order:
        xy = corners[idx]
        det_conf = float(scores[idx])
        x1 = max(0, min(frame_w, int(np.floor(xy[:, 0].min()))))
        y1 = max(0, min(frame_h, int(np.floor(xy[:, 1].min()))))
        x2 = max(0, min(frame_w, int(np.ceil(xy[:, 0].max()))))
        y2 = max(0, min(frame_h, int(np.ceil(xy[:, 1].max()))))

        crop_img = crop_obb(frame, xy, pad_ratio=pad_ratio)
        if crop_img is None or crop_img.size == 0:
            h, w = 0, 0
            crop_img = None
            ocr_status = "crop_failed"
            two_row = False
        else:
            h, w = crop_img.shape[:2]
            if w < _MIN_TRACK_CROP_W or h < _MIN_TRACK_CROP_H:
                del crop_img
                crop_img = None
                ocr_status = "crop_too_small"
                two_row = False
            else:
                plate_class = CLASS_NAMES[int(classes[idx])] if int(classes[idx]) < len(CLASS_NAMES) else ""
                two_row = plate_class == "BSV" and w / max(h, 1) < 2.2
                ocr_status = None

        regions.append({
            "bbox": [x1, y1, x2, y2],
            "obb": xy.tolist(),
            "det_conf": det_conf,
            "class": CLASS_NAMES[int(classes[idx])] if int(classes[idx]) < len(CLASS_NAMES) else str(int(classes[idx])),
            "crop_size": f"{w}x{h}",
            "crop_img": crop_img,
            "two_row": two_row,
            "ocr_status": ocr_status,
        })
    return regions


def recognize_plate_regions(regions, ocr=None, charset=None):
    if ocr is None:
        raise ValueError("ocr is required for PP-OCR recognition")
    if charset is None:
        raise ValueError("charset is required for PP-OCR recognition")

    plates = []
    for region in regions:
        crop_img = region.get("crop_img")
        two_row = bool(region.get("two_row"))
        valid_candidates = []
        low_confidence_candidates = []
        ocr_conf = 0.0
        raw_text = ""
        if crop_img is None:
            plate_text, candidates = "unknown", []
            ocr_status = region.get("ocr_status") or "crop_failed"
            ocr_outcome = ocr_status
        else:
            plate_text, ocr_conf, raw_text, top5 = recognize_combined(ocr, charset, crop_img, two_row=two_row)
            candidates = [raw_text] + [text for text, _ in top5]
            for text, conf in [(raw_text, ocr_conf), *top5]:
                for normalized in normalize_plate_candidate(text):
                    display = format_plate_display(normalized)
                    if is_valid_plate_text(display) and display not in [p for p, _ in valid_candidates]:
                        valid_candidates.append((display, conf))
            if LPR_OCR_MIN_CONFIDENCE is not None:
                low_confidence_candidates = [
                    candidate for candidate in valid_candidates
                    if candidate[1] < LPR_OCR_MIN_CONFIDENCE
                ]
                valid_candidates = [
                    candidate for candidate in valid_candidates
                    if candidate[1] >= LPR_OCR_MIN_CONFIDENCE
                ]
            if valid_candidates:
                ocr_outcome = "valid"
                plate_text, ocr_conf = max(valid_candidates, key=lambda candidate: candidate[1])
            elif low_confidence_candidates:
                ocr_outcome = "low_confidence"
                plate_text = "unknown"
            elif not clean_plate(raw_text):
                ocr_outcome = "blank"
            else:
                ocr_outcome = "invalid_format"
            if ocr_outcome == "low_confidence":
                ocr_status = f"low_confidence:{ocr_conf:.3f}{':two_row' if two_row else ''}"
            elif not is_valid_plate_text(plate_text):
                plate_text = "unknown"
                ocr_status = f"invalid_plate:{ocr_conf:.3f}{':two_row' if two_row else ''}"
            else:
                ocr_status = f"lpr:{ocr_conf:.3f}{':two_row' if two_row else ''}"
        del crop_img

        plates.append({
            "plate": plate_text,
            "bbox": region["bbox"],
            "obb": region["obb"],
            "coord_space": "input_frame",
            "detector_backend": "yolov8_obb_rknn",
            "ocr_backend": "ppocr_rknn",
            "det_conf": region["det_conf"],
            "class": region["class"],
            "crop_size": region["crop_size"],
            "ocr_status": ocr_status,
            "ocr_outcome": ocr_outcome,
            "ocr_confidence": float(ocr_conf),
            "raw_text": raw_text,
            "votes": len(candidates),
            "candidates": candidates,
            "valid_candidates": valid_candidates,
            "low_confidence_candidates": low_confidence_candidates,
        })
    return plates


def validate_lpr_runtime(detectors, recognizers, charset, model_paths=None, log_fn=None):
    validate_lpr_charset(charset)
    detector_input = np.zeros((1, 960, 960, 3), dtype=np.float32)
    for name, detector in detectors:
        outputs = detector.inference(inputs=[detector_input], data_format=["nhwc"])
        if not all(np.isfinite(np.asarray(output)).all() for output in outputs):
            raise ValueError(f"{name} detector output contains non-finite values")
        decoded = decode_detector_outputs(outputs)
        if decoded.shape != (1, 7, 18900) or not np.isfinite(decoded).all():
            raise ValueError(f"{name} decoded detector shape is {decoded.shape}")
    for name, recognizer in recognizers:
        for width in REC_WIDTH_BUCKETS:
            blob = np.full((1, REC_HEIGHT, width, 3), -1.0, dtype=np.float32)
            outputs = recognizer.inference(inputs=[blob], data_format=["nhwc"])
            if len(outputs) != 1:
                raise ValueError(f"{name} expected one OCR output, got {len(outputs)}")
            output = np.asarray(outputs[0])
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != len(charset):
                raise ValueError(f"{name} unexpected OCR output shape at width {width}: {output.shape}")
            _to_probs(output[0])
    hashes = verify_lpr_bundle(model_paths) if model_paths else {}
    if log_fn:
        log_fn("INFO", f"Fine-tuned LPR runtime contract passed hashes={hashes}")
    return hashes


def detect_license_plates(frame, detector=None, ocr=None, imgsz=960, conf_thres=None, iou_thres=None, pad_ratio=0.0, charset=None):
    regions = detect_plate_regions(frame, detector=detector, imgsz=imgsz, conf_thres=conf_thres, iou_thres=iou_thres, pad_ratio=pad_ratio)
    return recognize_plate_regions(regions, ocr=ocr, charset=charset)
