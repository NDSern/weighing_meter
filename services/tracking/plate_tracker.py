"""PlateTracker — accumulates plate observations across a session."""

import threading


def _clean_plate(plate_text: str) -> str:
    return "".join(ch for ch in (plate_text or "").upper() if ch.isalnum())


def _prefer_detailed_variant(best, counts, scores):
    """Prefer 5-digit VN plate variant when seen in same family/session."""
    best_clean = _clean_plate(best)
    if len(best_clean) != 7:
        return best, scores[best], counts[best]

    detailed = []
    for plate, count in counts.items():
        clean = _clean_plate(plate)
        if len(clean) == 8 and clean.endswith("0") and clean[:-1] == best_clean:
            detailed.append(plate)

    if not detailed:
        return best, scores[best], counts[best]

    chosen = max(detailed, key=lambda plate: (counts[plate], scores.get(plate, 0.0)))
    return chosen, scores[best] + scores[chosen], counts[best] + counts[chosen]


class PlateTracker:
    """Accumulates plate observations across a session and returns the most frequent plate."""

    def __init__(self):
        self._observations = []  # list of (plate_text, weight)
        self._lock = threading.Lock()
        self._image_frame = None  # full frame for saving on publish
        self._image_plate = None  # plate text associated with saved frame
        self._image_camera = None  # camera name associated with saved frame
        self._image_conf = 0.0  # best det_conf for saved frame
        self._plate_images = {}  # plate_text -> (frame, camera_name, det_conf)
        self._undetectable_frame = None  # first "unknown" frame for undetectable save
        self._undetectable_saved = False  # only save once per session

    def add_observation(self, plate_text: str, det_conf: float, crop_w: int, crop_h: int):
        crop_score = min(1.0, (crop_w * crop_h) / (200.0 * 60.0))
        weight = det_conf * crop_score
        with self._lock:
            self._observations.append((plate_text, weight))

    def update_image(self, plate_text: str, det_conf: float, frame, camera_name: str):
        """Store the best-confidence frame. Caller transfers ownership (no copy made)."""
        with self._lock:
            old = self._plate_images.get(plate_text)
            if old is None or det_conf > old[2]:
                if old is None and len(self._plate_images) >= 8:
                    weakest = min(self._plate_images, key=lambda key: self._plate_images[key][2])
                    self._plate_images.pop(weakest, None)
                self._plate_images[plate_text] = (frame.copy(), camera_name, det_conf)
            if det_conf > self._image_conf:
                self._image_frame = frame.copy()
                self._image_plate = plate_text
                self._image_camera = camera_name
                self._image_conf = det_conf

    def get_confirmed_plate(self):
        """Returns the most frequent session plate if it appears at least PLATE_CONFIRM_THRESHOLD times."""
        from config import PLATE_CONFIRM_THRESHOLD

        with self._lock:
            scores = {}
            counts = {}
            for plate, weight in self._observations:
                scores[plate] = scores.get(plate, 0.0) + weight
                counts[plate] = counts.get(plate, 0) + 1

        if not scores:
            return None, 0.0, 0

        best = max(counts, key=lambda plate: (counts[plate], scores.get(plate, 0.0)))
        if counts[best] >= PLATE_CONFIRM_THRESHOLD:
            return _prefer_detailed_variant(best, counts, scores)
        return None, scores[best], counts.get(best, 0)

    def get_all_plates_summary(self) -> dict:
        """Returns {plate: count} for all plate observations in the current session."""
        with self._lock:
            counts = {}
            for plate, weight in self._observations:
                counts[plate] = counts.get(plate, 0) + 1
        return counts

    def save_undetectable(self, frame):
        """Store the first 'unknown' frame per session. Only saves once until clear().
        Caller must check needs_undetectable() first to avoid unnecessary frame copy."""
        with self._lock:
            if not self._undetectable_saved:
                self._undetectable_frame = frame
                self._undetectable_saved = True

    def needs_undetectable(self):
        """Check if an undetectable frame is still needed (no copy yet)."""
        with self._lock:
            return not self._undetectable_saved

    def get_undetectable_frame(self):
        """Returns the undetectable frame. Transfers ownership — clears internal ref."""
        with self._lock:
            frame = self._undetectable_frame
            self._undetectable_frame = None
            return frame

    def get_image_frame(self, plate_text=None, aliases=None):
        """Returns (frame, plate_text, camera_name). Transfers ownership — clears internal ref."""
        with self._lock:
            lookup_plates = []
            for candidate in [plate_text, *(aliases or [])]:
                if candidate and candidate not in lookup_plates:
                    lookup_plates.append(candidate)

            matched_plate = next((candidate for candidate in lookup_plates if candidate in self._plate_images), None)
            if matched_plate is not None:
                frame, camera_name, _conf = self._plate_images.pop(matched_plate)
                plate = matched_plate
            else:
                frame = self._image_frame
                plate = self._image_plate
                camera_name = self._image_camera
            self._image_frame = None
            self._image_plate = None
            self._image_camera = None
            self._image_conf = 0.0
            self._plate_images.clear()
            return frame, plate, camera_name

    def clear(self):
        with self._lock:
            self._observations.clear()
            self._image_frame = None
            self._image_plate = None
            self._image_camera = None
            self._image_conf = 0.0
            self._plate_images.clear()
            self._undetectable_frame = None
            self._undetectable_saved = False
