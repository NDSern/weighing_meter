"""VehicleTracker — tracks vehicle detections across cameras."""

import math
import threading
import time

from config import (
    YOLO26_STATIONARY_SECONDS,
    YOLO26_STATIONARY_PIXELS_THRESHOLD,
    VEHICLE_CLASSES,
    VEHICLE_CLASS_IDS,
)

_log_fn = None


def set_log_fn(log_fn):
    """Set the logging function to use."""
    global _log_fn
    _log_fn = log_fn


def log(level: str, msg: str):
    """Log using the configured log function."""
    if _log_fn:
        _log_fn(level, msg)


class VehicleState:
    def __init__(self, camera_name: str):
        self.camera_name = camera_name
        self.bbox = None
        self.stable_bbox = None
        self.class_id = None
        self.class_name = None
        self.conf = 0.0
        self.stable_since_ts = None
        self.is_stable = False
        self.logged_type = None
        self.logged_stable = None
        self.missing_frames = 0
        self.unstable_pending_frames = 0
        self.object_id = 0


class VehicleTracker:
    def __init__(self):
        self._states = {"cam1": VehicleState(camera_name="cam1"), "cam3": VehicleState(camera_name="cam3")}
        self._next_id = {"cam1": 1, "cam3": 1}
        self._lock = threading.Lock()
        self._distance_ratio_threshold = 0.1
        self._unstable_safety_frames = 3

    @staticmethod
    def _bbox_center(box):
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    @classmethod
    def _bbox_debug(cls, box):
        cx, cy = cls._bbox_center(box)
        return f"box=({int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}) center=({cx:.0f},{cy:.0f})"

    @staticmethod
    def _center_distance(box_a, box_b):
        ax = (box_a[0] + box_a[2]) / 2.0
        ay = (box_a[1] + box_a[3]) / 2.0
        bx = (box_b[0] + box_b[2]) / 2.0
        by = (box_b[1] + box_b[3]) / 2.0
        return math.hypot(ax - bx, ay - by)

    @staticmethod
    def _class_name(class_id: int):
        if 0 <= class_id < len(VEHICLE_CLASSES):
            return VEHICLE_CLASSES[class_id]
        return str(class_id)

    @staticmethod
    def _filter_vehicle_classes(detections):
        return [det for det in detections if int(det[5]) in VEHICLE_CLASS_IDS]

    @staticmethod
    def _filter_by_camera_side(camera_name: str, detections, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        mid_x = frame_w / 2.0
        kept = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            cx = (x1 + x2) / 2.0
            if camera_name == "cam1" and cx < mid_x:
                continue
            if camera_name == "cam3" and cx >= mid_x:
                continue
            kept.append(det)
        return kept

    @staticmethod
    def _filter_close_objects(detections, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_h * frame_w)
        kept = []
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det
            box_w = max(0.0, x2 - x1)
            box_h = max(0.0, y2 - y1)
            area_ratio = (box_w * box_h) / frame_area if frame_area > 0 else 0.0
            if area_ratio < 0.08:  # YOLO26_MIN_BOX_AREA_RATIO
                continue
            if y2 < frame_h * 0.55:  # YOLO26_MIN_BOTTOM_Y_RATIO
                continue
            kept.append(det)
        return kept

    @classmethod
    def _select_camera_object(cls, camera_name: str, detections):
        if not detections:
            return None
        if camera_name == "cam1":
            return max(detections, key=lambda det: det[2])
        return min(detections, key=lambda det: ((det[0] + det[2]) / 2.0, det[0]))

    @classmethod
    def _distance_ratio(cls, prev_box, new_box, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        frame_area = float(frame_h * frame_w)
        if frame_area <= 0:
            return float("inf")
        center_dist = cls._center_distance(prev_box, new_box)
        box_w = max(0.0, new_box[2] - new_box[0])
        box_h = max(0.0, new_box[3] - new_box[1])
        area_ratio = (box_w * box_h) / frame_area
        if area_ratio <= 0:
            return float("inf")
        scale = math.sqrt(frame_area * area_ratio)
        if scale <= 0:
            return float("inf")
        return center_dist / scale

    @staticmethod
    def _bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a[:4]
        bx1, by1, bx2, by2 = box_b[:4]
        xx1 = max(ax1, bx1)
        yy1 = max(ay1, by1)
        xx2 = min(ax2, bx2)
        yy2 = min(ay2, by2)
        inter_w = max(0.0, xx2 - xx1)
        inter_h = max(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    @staticmethod
    def _bbox_edges_moved_too_far(stable_box, new_box):
        stable_w = max(1.0, stable_box[2] - stable_box[0])
        stable_h = max(1.0, stable_box[3] - stable_box[1])
        max_dx = stable_w * 0.10
        max_dy = stable_h * 0.10
        return (
            abs(new_box[0] - stable_box[0]) > max_dx
            or abs(new_box[2] - stable_box[2]) > max_dx
            or abs(new_box[1] - stable_box[1]) > max_dy
            or abs(new_box[3] - stable_box[3]) > max_dy
        )

    def _log_track_state(self, state: VehicleState):
        if state.class_name is None:
            return
        vehicle_type = state.class_name
        if state.logged_type != vehicle_type:
            log("VEHICLE", f"[{state.camera_name}] object={state.object_id} type={vehicle_type} conf={state.conf:.2f}")
            state.logged_type = vehicle_type
        if state.logged_stable != state.is_stable:
            log("VEHICLE", f"[{state.camera_name}] object={state.object_id} type={vehicle_type} stable={state.is_stable}")
            state.logged_stable = state.is_stable

    def _handle_missing_detection(self, state, now):
        """Handle case where no detection was selected for this camera."""
        if state.bbox is not None:
            state.missing_frames += 1
            state.unstable_pending_frames += 1
            if state.unstable_pending_frames >= self._unstable_safety_frames:
                state.bbox = None
                state.stable_bbox = None
                state.class_id = None
                state.class_name = None
                state.conf = 0.0
                state.stable_since_ts = None
                state.is_stable = False
        self._log_track_state(state)

    def _is_same_object(self, state, selected, frame_shape):
        """Check if the new detection matches the currently tracked object."""
        if state.bbox is None or state.class_id != int(selected[5]):
            return False, 0.0
        distance_ratio = self._distance_ratio(state.bbox, selected, frame_shape)
        return distance_ratio <= self._distance_ratio_threshold, distance_ratio

    def _update_stability_state(self, state, selected, now):
        """Update stability state based on movement."""
        movement = self._center_distance(state.bbox, selected)
        if movement <= YOLO26_STATIONARY_PIXELS_THRESHOLD:
            if state.stable_since_ts is None:
                state.stable_since_ts = now
        else:
            state.stable_since_ts = now
            state.is_stable = False
        if state.stable_since_ts is not None and (now - state.stable_since_ts) >= YOLO26_STATIONARY_SECONDS:
            state.is_stable = True

    def _update_baseline_bbox(self, state, selected, now):
        """Update or validate the stable baseline bbox."""
        if state.stable_bbox is None:
            state.stable_bbox = selected[:4]
        else:
            baseline_iou = self._bbox_iou(state.stable_bbox, selected)
            moved_from_baseline = baseline_iou < 0.80 or self._bbox_edges_moved_too_far(state.stable_bbox, selected)
            if moved_from_baseline:
                state.unstable_pending_frames += 1
                if state.unstable_pending_frames >= self._unstable_safety_frames:
                    state.is_stable = False
                    state.stable_since_ts = now
                    state.stable_bbox = None
            else:
                state.unstable_pending_frames = 0

    def _apply_detection(self, state, selected, now):
        """Apply a selected detection to the tracker state."""
        state.bbox = selected[:4]
        state.class_id = int(selected[5])
        state.class_name = self._class_name(state.class_id)
        state.conf = selected[4]
        state.missing_frames = 0
        if not state.is_stable:
            state.unstable_pending_frames = 0

    def update(self, camera_name: str, detections, frame_shape):
        now = time.time()
        raw_detections = self._filter_vehicle_classes(list(detections))
        side_detections = self._filter_by_camera_side(camera_name, raw_detections, frame_shape)
        filtered_detections = self._filter_close_objects(side_detections, frame_shape)
        selection_pool = filtered_detections if filtered_detections else side_detections
        selected = self._select_camera_object(camera_name, selection_pool)
        final_detections = [selected] if selected is not None else []
        with self._lock:
            state = self._states[camera_name]
            if selected is None:
                self._handle_missing_detection(state, now)
                return {
                    "raw_detections": raw_detections,
                    "side_detections": side_detections,
                    "final_detections": [],
                }

            same_object, distance_ratio = self._is_same_object(state, selected, frame_shape)
            if not same_object:
                state.object_id = self._next_id[camera_name]
                self._next_id[camera_name] += 1
                state.stable_since_ts = now
                state.stable_bbox = None
                state.is_stable = False
            else:
                self._update_stability_state(state, selected, now)

            if state.is_stable:
                self._update_baseline_bbox(state, selected, now)

            self._apply_detection(state, selected, now)
            self._log_track_state(state)
        return {
            "raw_detections": raw_detections,
            "side_detections": side_detections,
            "final_detections": final_detections,
        }

    def get_summary(self):
        with self._lock:
            camera_state = {}
            best_vehicle_type = None
            best_conf = -1.0
            for camera_name, state in self._states.items():
                truck_stable = state.class_id == 7 and state.is_stable
                truck_unstable = state.class_id != 7 or (state.class_id == 7 and not state.is_stable)
                camera_state[camera_name] = {
                    "best_type": state.class_name,
                    "best_conf": state.conf,
                    "truck_stable": truck_stable,
                    "truck_unstable": truck_unstable,
                    "has_confirmed_truck": state.class_id == 7,
                    "object_id": state.object_id if state.bbox is not None else None,
                }
                if state.class_name is not None and state.conf > best_conf:
                    best_vehicle_type = state.class_name
                    best_conf = state.conf

            return {
                "vehicle_type": best_vehicle_type,
                "cam1_truck_unstable": camera_state.get("cam1", {}).get("truck_unstable", True),
                "cam3_truck_unstable": camera_state.get("cam3", {}).get("truck_unstable", True),
                "cam1_truck_stable": camera_state.get("cam1", {}).get("truck_stable", False),
                "cam3_truck_stable": camera_state.get("cam3", {}).get("truck_stable", False),
                "camera_state": camera_state,
            }
