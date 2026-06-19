"""RKNN inference helpers for plate and vehicle detection."""

import cv2
import math

import numpy as np

from config import (
    DET_CONF_THRES,
    DET_IOU_THRES,
    IMG_SIZE,
    OCR_CONF_THRES,
    OCR_IOU_THRES,
    YOLO26_CONF_THRES,
    YOLO26_IOU_THRES,
    VEHICLE_CLASS_IDS,
)


def letterbox(img, new_shape=640):
    """Resize image with letterbox padding, returns (padded_img, ratio, (dw, dh))."""
    h, w = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def preprocess(img):
    """Preprocess BGR image for RKNN YOLOv5: letterbox + BGR→RGB."""
    img_lb, ratio, (dw, dh) = letterbox(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    del img_lb
    inp = np.expand_dims(img_rgb, axis=0)
    del img_rgb
    return inp, ratio, dw, dh


def xywh2xyxy(x):
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def nms_boxes(boxes, scores, iou_threshold):
    """Simple NMS. boxes: (N,4) as x1y1x2y2, scores: (N,)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return np.array(keep, dtype=int)


def postprocess(output, conf_thres, iou_thres, ratio, dw, dh, orig_shape):
    """Post-process YOLOv5 RKNN output."""
    pred = output[0].squeeze(0)
    mask = pred[:, 4] > conf_thres
    pred = pred[mask]
    if len(pred) == 0:
        return []

    class_scores = pred[:, 5:]
    class_ids = class_scores.argmax(axis=1)
    class_confs = class_scores[np.arange(len(class_ids)), class_ids]
    confs = pred[:, 4] * class_confs

    mask2 = confs > conf_thres
    pred, confs, class_ids = pred[mask2], confs[mask2], class_ids[mask2]
    if len(pred) == 0:
        return []

    boxes = xywh2xyxy(pred[:, :4])
    keep = nms_boxes(boxes, confs, iou_thres)
    boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]

    boxes[:, 0] = (boxes[:, 0] - dw) / ratio
    boxes[:, 1] = (boxes[:, 1] - dh) / ratio
    boxes[:, 2] = (boxes[:, 2] - dw) / ratio
    boxes[:, 3] = (boxes[:, 3] - dh) / ratio

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    return [[boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], float(confs[i]), int(class_ids[i])] for i in range(len(boxes))]


def yolo26_postprocess(outputs, conf_thres, iou_thres, ratio, dw, dh, orig_shape):
    """Post-process YOLOv26 RKNN output for vehicle detection."""
    pred = outputs[0].squeeze(0).transpose(1, 0)
    boxes = pred[:, :4]
    class_scores = pred[:, 4:]
    class_ids = class_scores.argmax(axis=1)
    confs = class_scores[np.arange(len(class_ids)), class_ids]

    vehicle_mask = np.isin(class_ids, list(VEHICLE_CLASS_IDS))
    conf_mask = confs > conf_thres
    mask = vehicle_mask & conf_mask
    if not np.any(mask):
        return []

    boxes = boxes[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    boxes = xywh2xyxy(boxes)
    keep = nms_boxes(boxes, confs, iou_thres)
    boxes, confs, class_ids = boxes[keep], confs[keep], class_ids[keep]

    boxes[:, 0] = (boxes[:, 0] - dw) / ratio
    boxes[:, 1] = (boxes[:, 1] - dh) / ratio
    boxes[:, 2] = (boxes[:, 2] - dw) / ratio
    boxes[:, 3] = (boxes[:, 3] - dh) / ratio

    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    return [
        [boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], float(confs[i]), int(class_ids[i])]
        for i in range(len(boxes))
    ]


def detect_plates_rknn(frame, detector=None):
    """Run plate detection via RKNN. Returns list of [x1,y1,x2,y2,conf,class_id]."""
    img_input, ratio, dw, dh = preprocess(frame)
    outputs = detector.inference(inputs=[img_input], data_format="nhwc")
    result = postprocess(outputs, DET_CONF_THRES, DET_IOU_THRES, ratio, dw, dh, frame.shape)
    del img_input, outputs
    return result


def detect_vehicles_rknn(frame, detector=None):
    """Run vehicle detection via RKNN. Returns list of [x1,y1,x2,y2,conf,class_id]."""
    img_input, ratio, dw, dh = preprocess(frame)
    outputs = detector.inference(inputs=[img_input], data_format="nhwc")
    result = yolo26_postprocess(outputs, YOLO26_CONF_THRES, YOLO26_IOU_THRES, ratio, dw, dh, frame.shape)
    del img_input, outputs
    return result
