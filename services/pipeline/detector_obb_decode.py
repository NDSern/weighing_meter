from __future__ import annotations

import numpy as np


FEATURE_SIZES = (120, 60, 30)
STRIDES = (8.0, 16.0, 32.0)


def _anchors_and_strides() -> tuple[np.ndarray, np.ndarray]:
    anchors = []
    strides = []
    for size, stride in zip(FEATURE_SIZES, STRIDES):
        coordinates = np.arange(size, dtype=np.float32) + 0.5
        yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
        anchors.append(np.stack((xx.reshape(-1), yy.reshape(-1))))
        strides.append(np.full(size * size, stride, dtype=np.float32))
    return np.concatenate(anchors, axis=1)[None], np.concatenate(strides)[None]


ANCHORS, ANCHOR_STRIDES = _anchors_and_strides()


def decode_detector_outputs(outputs: list[np.ndarray] | tuple[np.ndarray, ...]) -> np.ndarray:
    if len(outputs) != 3:
        raise ValueError(f"Expected three detector head outputs, got {len(outputs)}")
    distances, class_scores, angle = (np.asarray(value, dtype=np.float32) for value in outputs)
    expected = ANCHORS.shape[-1]
    if distances.shape != (1, 4, expected):
        raise ValueError(f"Expected distances [1,4,{expected}], got {distances.shape}")
    if class_scores.shape != (1, 2, expected):
        raise ValueError(f"Expected class scores [1,2,{expected}], got {class_scores.shape}")
    if angle.shape != (1, 1, expected):
        raise ValueError(f"Expected angles [1,1,{expected}], got {angle.shape}")

    left_top, right_bottom = np.split(distances, 2, axis=1)
    half_delta = (right_bottom - left_top) / 2.0
    cosine = np.cos(angle)
    sine = np.sin(angle)
    center_offset = np.concatenate(
        (
            half_delta[:, 0:1] * cosine - half_delta[:, 1:2] * sine,
            half_delta[:, 0:1] * sine + half_delta[:, 1:2] * cosine,
        ),
        axis=1,
    )
    centers = center_offset + ANCHORS
    sizes = left_top + right_bottom
    boxes = np.concatenate((centers, sizes), axis=1) * ANCHOR_STRIDES[:, None]
    return np.concatenate((boxes, class_scores, angle), axis=1).astype(np.float32, copy=False)
