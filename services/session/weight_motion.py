"""Deterministic departure and renewed-rise boundary detection.

Pure detector: every time input is passed in (monotonic seconds plus an
observation timestamp); no I/O and no direct clock access, so unit tests and
replays drive it frame by frame.

Departure: the SESSION_MOTION_FILTER_FRAMES median falls to
``reference - max(SESSION_DROP_MIN_KG, SESSION_DROP_RATIO * reference)``
(inclusive) for SESSION_DROP_MIN_READINGS consecutive readings spanning at
least SESSION_DROP_DWELL_SECONDS. The pending reference freezes when a run
starts; a reading back above the crossing level cancels the run. A departure
is emitted once until the next state transition (reset/start_visit/arm).

Renewed rise (after ``arm_reentry``): the running filtered trough plus
``max(SESSION_RISE_MIN_KG, SESSION_RISE_RATIO * previous reference)`` is held
for SESSION_RISE_DWELL_SECONDS (no minimum reading count). A new low resets
the pending rise; a below-threshold reading resets the rise timer. Bounded
rising samples are retained for new-session seeding.
"""

import statistics
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

from config import (
    SESSION_DROP_DWELL_SECONDS,
    SESSION_DROP_MIN_KG,
    SESSION_DROP_MIN_READINGS,
    SESSION_DROP_RATIO,
    SESSION_MOTION_FILTER_FRAMES,
    SESSION_RISE_DWELL_SECONDS,
    SESSION_RISE_MIN_KG,
    SESSION_RISE_RATIO,
)

DEPARTURE = "departure"
RISE = "rise"

# Float clocks drift a few ulps around the dwell boundary; treat the dwell as
# met when the elapsed span is within this epsilon of the configured dwell.
_DWELL_EPSILON_SECONDS = 1e-9


@dataclass
class MotionDecision:
    kind: str  # DEPARTURE or RISE
    crossed_at: str
    confirmed_at: str
    reference_kg: float
    threshold_kg: float  # drop/rise magnitude derived from the reference
    trough_kg: Optional[float] = None  # rise only


class WeightMotion:
    def __init__(self):
        self._filter = deque(maxlen=SESSION_MOTION_FILTER_FRAMES)
        self._drop_reference = None  # frozen while a drop run is pending
        self._drop_run = None  # [count, start_mono, crossed_at]
        self._drop_emitted = False
        self._armed_reference = None
        self._rise_threshold = None
        self._trough = None
        self._trough_at = None
        self._rise_run = None  # (start_mono, crossed_at)
        self._rise_emitted = False
        self._rise_samples = deque(maxlen=2 * SESSION_MOTION_FILTER_FRAMES)

    def reset(self):
        """Discard all continuity: pending runs, trough, samples, arming."""
        self._filter.clear()
        self._drop_reference = None
        self._drop_run = None
        self._drop_emitted = False
        self._armed_reference = None
        self._rise_threshold = None
        self._trough = None
        self._trough_at = None
        self._rise_run = None
        self._rise_emitted = False
        self._rise_samples.clear()

    def start_visit(self):
        """A new visit began; the detector starts from a clean slate."""
        self.reset()

    def arm_reentry(self, previous_reference_kg):
        """Watch for the renewed rise after a confirmed departure."""
        if previous_reference_kg is None:
            raise ValueError("arm_reentry requires a previous reference")
        self._armed_reference = previous_reference_kg
        self._rise_threshold = max(
            SESSION_RISE_MIN_KG, SESSION_RISE_RATIO * previous_reference_kg
        )
        self._filter.clear()
        self._trough = None
        self._trough_at = None
        self._rise_run = None
        self._rise_emitted = False
        self._rise_samples.clear()
        self._drop_reference = None
        self._drop_run = None
        self._drop_emitted = False

    def observe_loaded(self, weight, now_mono, observed_at, reference_kg):
        """Feed one loaded frame; return a MotionDecision on confirmed drop."""
        self._filter.append(weight)
        if self._drop_emitted or reference_kg is None:
            return None
        level = statistics.median(self._filter)
        reference = (
            self._drop_reference
            if self._drop_reference is not None
            else reference_kg
        )
        threshold = max(SESSION_DROP_MIN_KG, SESSION_DROP_RATIO * reference)
        crossing = reference - threshold
        if level <= crossing:
            if self._drop_run is None:
                self._drop_reference = reference
                self._drop_run = [1, now_mono, observed_at]
            else:
                self._drop_run[0] += 1
            count, start_mono, crossed_at = self._drop_run
            if (
                count >= SESSION_DROP_MIN_READINGS
                and (now_mono - start_mono)
                >= SESSION_DROP_DWELL_SECONDS - _DWELL_EPSILON_SECONDS
            ):
                self._drop_emitted = True
                return MotionDecision(
                    DEPARTURE, crossed_at, observed_at, reference, threshold
                )
            return None
        # Back above the crossing level: cancel the pending run (the frozen
        # reference is released so the next run re-freezes from the caller).
        self._drop_reference = None
        self._drop_run = None
        return None

    def observe_reentry(self, weight, now_mono, observed_at):
        """Feed one post-departure frame; return MotionDecision on rise."""
        if self._armed_reference is None:
            raise ValueError("observe_reentry called before arm_reentry()")
        self._filter.append(weight)
        level = statistics.median(self._filter)
        if self._trough is None or level < self._trough:
            self._trough = level
            self._trough_at = observed_at
            self._rise_run = None
            return None
        crossing = self._trough + self._rise_threshold
        if level >= crossing:
            if self._rise_run is None:
                self._rise_run = (now_mono, observed_at)
                start_mono, crossed_at = self._rise_run
            else:
                start_mono, crossed_at = self._rise_run
            self._rise_samples.append((weight, observed_at))
            if not self._rise_emitted and (
                (now_mono - start_mono)
                >= SESSION_RISE_DWELL_SECONDS - _DWELL_EPSILON_SECONDS
            ):
                self._rise_emitted = True
                return MotionDecision(
                    RISE,
                    crossed_at,
                    observed_at,
                    self._armed_reference,
                    self._rise_threshold,
                    trough_kg=self._trough,
                )
            return None
        # Below the crossing level: reset the rise timer.
        self._rise_run = None
        return None

    def rise_samples(self):
        """Bounded rising (weight, observed_at) samples for new-session seeding."""
        return list(self._rise_samples)

    def trough(self):
        return self._trough
