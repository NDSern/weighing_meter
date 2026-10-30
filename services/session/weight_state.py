"""Weighted-session state and loaded stable-weight candidate selection."""

import statistics
from collections import Counter, deque
from datetime import datetime

from config import (
    SESSION_STABLE_WEIGHT_WINDOW,
    SESSION_WEIGHT_TREND_FRAMES,
    WEIGHT_THRESHOLD,
)

# Candidate episodes share the reader's stability tolerance: candidates whose
# total spread leaves this band belong to a different plateau episode.
PLATEAU_BAND_KG = 20.0

# Sustained filtered-plateau fallback (lower confidence than a stable plateau).
SUSTAINED_PLATEAU_FILTER_FRAMES = 5
SUSTAINED_PLATEAU_MIN_READINGS = 3
SUSTAINED_PLATEAU_MIN_SPAN_SECONDS = 0.4


def _mode_newest(values, counts, last_seen):
    """Pick the most frequent value; the newest occurrence breaks ties."""
    if not counts:
        return None
    return max(counts, key=lambda value: (counts[value], last_seen[value]))


class SustainedPlateauTracker:
    """Highest sustained filtered-median plateau observed during one visit.

    Feeds every valid loaded frame of the active visit. A candidate is a full
    5-reading median (always one of the observed readings). A plateau run is a
    consecutive sequence of medians within the plateau band; a run qualifies
    once it holds at least three medians spanning at least 0.4 seconds. The
    tracker keeps the highest qualified run's modal median so terminal
    selection can fall back to a supported sustained value instead of an
    isolated raw spike.
    """

    def __init__(self):
        self._window = deque(maxlen=SUSTAINED_PLATEAU_FILTER_FRAMES)
        self._run = []
        self._best = None

    def reset(self):
        self._window.clear()
        self._run.clear()
        self._best = None

    def observe(self, weight, observed_at, observed_ts):
        if weight <= WEIGHT_THRESHOLD:
            self._window.clear()
            self._run.clear()
            return
        self._window.append((weight, observed_at))
        if len(self._window) < self._window.maxlen:
            return
        median = statistics.median(value for value, _observed_at in self._window)
        run_values = [value for value, _observed_at, _ts in self._run]
        if run_values and max(run_values + [median]) - min(run_values + [median]) > PLATEAU_BAND_KG:
            self._run = []
        self._run.append((median, observed_at, observed_ts))
        if len(self._run) < SUSTAINED_PLATEAU_MIN_READINGS:
            return
        if (self._run[-1][2] - self._run[0][2]).total_seconds() < SUSTAINED_PLATEAU_MIN_SPAN_SECONDS:
            return
        counts = Counter(value for value, _observed_at, _ts in self._run)
        last_seen = {}
        for index, (value, _observed_at, _ts) in enumerate(self._run):
            last_seen[value] = index
        candidate = _mode_newest(None, counts, last_seen)
        if self._best is None or candidate > self._best[0]:
            self._best = (candidate, self._run[last_seen[candidate]][1])

    @property
    def best(self):
        if self._best is None:
            return None
        return {"weight": self._best[0], "observed_at": self._best[1]}


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
        # Retained measurement: the validated plateau of this visit. Lower
        # plateaus (departure ramps, partial entries) never overwrite it; a
        # distinctly higher qualified episode can replace a partial entry.
        self.plateau_retained_weight = None
        self.plateau_retained_decimal_pos = 0
        self.plateau_retained_observed_at = None
        # Current plateau episode (contiguous candidates within the band).
        self._episode_counts = Counter()
        self._episode_last_seen = {}
        self._episode_decimal_pos = {}
        self._episode_observation_times = {}
        self._episode_min = None
        self._episode_max = None
        self._episode_count = 0
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
        self._record_episode_candidate(weight, decimal_pos, observed_at)
        self._publish_selection()

    def _record_episode_candidate(self, weight, decimal_pos, observed_at):
        if self._episode_count:
            spread = max(self._episode_max, weight) - min(self._episode_min, weight)
            if spread > PLATEAU_BAND_KG:
                self._finalize_plateau_episode()
        if not self._episode_count:
            self._episode_min = weight
            self._episode_max = weight
        else:
            self._episode_min = min(self._episode_min, weight)
            self._episode_max = max(self._episode_max, weight)
        self._episode_count += 1
        self._episode_counts[weight] += 1
        self._episode_last_seen[weight] = self.stable_weight_sequence
        self._episode_decimal_pos[weight] = decimal_pos
        self._episode_observation_times[weight] = observed_at

    def _finalize_plateau_episode(self):
        selection = _mode_newest(None, self._episode_counts, self._episode_last_seen)
        if (
            selection is not None
            and self._episode_count >= 2
            and (
                self.plateau_retained_weight is None
                or selection > self.plateau_retained_weight + PLATEAU_BAND_KG
            )
        ):
            self.plateau_retained_weight = selection
            self.plateau_retained_decimal_pos = self._episode_decimal_pos[selection]
            self.plateau_retained_observed_at = self._episode_observation_times[selection]
        self._reset_plateau_episode()

    def _reset_plateau_episode(self):
        self._episode_counts.clear()
        self._episode_last_seen.clear()
        self._episode_decimal_pos.clear()
        self._episode_observation_times.clear()
        self._episode_min = None
        self._episode_max = None
        self._episode_count = 0

    def end_plateau_candidate(self):
        """Close the current episode on a non-stable frame; retention stays."""
        if self._episode_count:
            self._finalize_plateau_episode()

    def _publish_selection(self):
        selection = _mode_newest(None, self.stable_weight_counts, self.stable_weight_last_seen)
        if selection is None:
            return
        retained = self.plateau_retained_weight
        if retained is not None:
            episode_selection = _mode_newest(None, self._episode_counts, self._episode_last_seen)
            if (
                episode_selection is not None
                and self._episode_count >= 2
                and episode_selection > retained + PLATEAU_BAND_KG
            ):
                self.plateau_retained_weight = episode_selection
                self.plateau_retained_decimal_pos = self._episode_decimal_pos[episode_selection]
                self.plateau_retained_observed_at = self._episode_observation_times[episode_selection]
                retained = episode_selection
                selection = episode_selection
            elif abs(selection - retained) > PLATEAU_BAND_KG:
                selection = retained
        self.stable_weight = selection
        self.stable_decimal_pos = self.stable_weight_decimal_pos[selection]
        self.stable_weight_observed_at = self.stable_weight_observation_times[selection]

    def start_stable_weight_history(self):
        self.clear_stable_weight_history()
        self.stable_weight_sequence = 1
        self.stable_weight_history.append(
            (self.stable_weight, self.stable_decimal_pos, self.stable_weight_observed_at)
        )
        self.stable_weight_counts[self.stable_weight] = 1
        self.stable_weight_last_seen[self.stable_weight] = 1
        self.stable_weight_decimal_pos[self.stable_weight] = self.stable_decimal_pos
        self.stable_weight_observation_times[self.stable_weight] = self.stable_weight_observed_at
        if self.stable_weight is not None and self.stable_weight > WEIGHT_THRESHOLD:
            self.plateau_retained_weight = self.stable_weight
            self.plateau_retained_decimal_pos = self.stable_decimal_pos
            self.plateau_retained_observed_at = self.stable_weight_observed_at
        self._episode_counts[self.stable_weight] = 1
        self._episode_last_seen[self.stable_weight] = 1
        self._episode_decimal_pos[self.stable_weight] = self.stable_decimal_pos
        self._episode_observation_times[self.stable_weight] = self.stable_weight_observed_at
        self._episode_min = self.stable_weight
        self._episode_max = self.stable_weight
        self._episode_count = 1

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
        self.plateau_retained_weight = None
        self.plateau_retained_decimal_pos = 0
        self.plateau_retained_observed_at = None
        self._reset_plateau_episode()
