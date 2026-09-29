"""Weighted-session state and loaded stable-weight candidate selection."""

from collections import Counter, deque

from config import (
    SESSION_STABLE_WEIGHT_WINDOW,
    SESSION_WEIGHT_TREND_FRAMES,
    WEIGHT_THRESHOLD,
)

MAX_STABLE_WEIGHT_CANDIDATES = 256


class WeighingSessionState:
    """Mutable state for one weighing session lifecycle."""

    def __init__(self):
        self.stable_weight = None
        self.stable_decimal_pos = 0
        self.last_publish_weight = None
        self.last_publish_decimal_pos = 0
        self.stable_count = 0
        self.stable_weight_counts = Counter()
        self.stable_weight_last_seen = {}
        self.stable_weight_decimal_pos = {}
        self.stable_weight_observation_times = {}
        self.stable_weight_history = deque(maxlen=SESSION_STABLE_WEIGHT_WINDOW)
        self.stable_weight_sequence = 0
        self.latest_stable_weight = None
        self.stable_weight_observed_at = None
        self.weight_trend_window = deque(maxlen=SESSION_WEIGHT_TREND_FRAMES)
        self.weight_trend_observation_times = deque(maxlen=SESSION_WEIGHT_TREND_FRAMES)
        self.session_active = False
        self.vehicle_type = None
        self.empty_since = None
        self.rearm_block_until = 0.0
        self.rearm_block_reason = None
        self.rearm_reference_weight = None
        self.lpr_start_frames = {}
        self.start_frame_paths = {}
        self.rear_start_frame = None
        self.rear_start_path = None
        self.rear_captured_at = None
        self.rear_capture_source = None
        self.rear_fallback_deadline = None
        self.rear_fallback_attempted = False
        self.unknown_snapshot_paths = {}
        self.unknown_snapshot_captured_at = {}
        self.unknown_snapshot_deadline = None
        self.unknown_snapshot_attempted = False
        self.unknown_weight_snapshot_paths = {}
        self.unknown_weight_snapshot_captured_at = {}
        self.unknown_weight_snapshot_triggered_at = None
        self.unknown_weight_snapshot_attempted = False
        self.local_peak_dwell_snapshot_paths = {}
        self.local_peak_dwell_snapshot_captured_at = {}
        self.local_peak_drop_snapshot_paths = {}
        self.local_peak_drop_snapshot_captured_at = {}
        self.started_at = None
        self.started_at_iso = None
        self.session_id = None
        self.spool_active = False
        self.scale_owned = False
        self.stability_rule = None

    def record_stable_weight(self, weight, decimal_pos, observed_at=None):
        self.latest_stable_weight = weight
        if not self.session_active:
            self.stable_weight = weight
            self.stable_decimal_pos = decimal_pos
            return

        # Empty readings close the cycle; they are not loaded-weight candidates.
        if weight <= WEIGHT_THRESHOLD:
            return

        self.stable_weight_sequence += 1
        if len(self.stable_weight_history) == self.stable_weight_history.maxlen:
            expired_weight, _expired_decimal_pos, _expired_observed_at = self.stable_weight_history.popleft()
            self.stable_weight_counts[expired_weight] -= 1
            if self.stable_weight_counts[expired_weight] <= 0:
                self.stable_weight_counts.pop(expired_weight, None)
                self.stable_weight_last_seen.pop(expired_weight, None)
                self.stable_weight_decimal_pos.pop(expired_weight, None)
                self.stable_weight_observation_times.pop(expired_weight, None)
        self.stable_weight_history.append((weight, decimal_pos, observed_at))
        self.stable_weight_counts[weight] += 1
        self.stable_weight_last_seen[weight] = self.stable_weight_sequence
        self.stable_weight_decimal_pos[weight] = decimal_pos
        self.stable_weight_observation_times[weight] = observed_at
        if len(self.stable_weight_counts) > MAX_STABLE_WEIGHT_CANDIDATES:
            oldest_weakest = min(
                self.stable_weight_counts,
                key=lambda value: (
                    self.stable_weight_counts[value],
                    self.stable_weight_last_seen[value],
                ),
            )
            self.stable_weight_counts.pop(oldest_weakest)
            self.stable_weight_last_seen.pop(oldest_weakest, None)
            self.stable_weight_decimal_pos.pop(oldest_weakest, None)
            self.stable_weight_observation_times.pop(oldest_weakest, None)
        self.stable_weight = max(
            self.stable_weight_counts,
            key=lambda value: (
                self.stable_weight_counts[value],
                self.stable_weight_last_seen[value],
            ),
        )
        self.stable_decimal_pos = self.stable_weight_decimal_pos[self.stable_weight]
        self.stable_weight_observed_at = self.stable_weight_observation_times[self.stable_weight]

    def start_stable_weight_history(self):
        self.stable_weight_counts.clear()
        self.stable_weight_last_seen.clear()
        self.stable_weight_decimal_pos.clear()
        self.stable_weight_observation_times.clear()
        self.stable_weight_history.clear()
        self.stable_weight_sequence = 1
        self.stable_weight_history.append(
            (self.stable_weight, self.stable_decimal_pos, self.stable_weight_observed_at)
        )
        self.stable_weight_counts[self.stable_weight] = 1
        self.stable_weight_last_seen[self.stable_weight] = 1
        self.stable_weight_decimal_pos[self.stable_weight] = self.stable_decimal_pos
        self.stable_weight_observation_times[self.stable_weight] = self.stable_weight_observed_at

    def clear_stable_weight_history(self):
        self.stable_weight_counts.clear()
        self.stable_weight_last_seen.clear()
        self.stable_weight_decimal_pos.clear()
        self.stable_weight_observation_times.clear()
        self.stable_weight_history.clear()
        self.stable_weight_sequence = 0
        self.latest_stable_weight = None
        self.weight_trend_window.clear()
        self.weight_trend_observation_times.clear()