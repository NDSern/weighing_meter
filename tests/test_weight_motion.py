import unittest
import sys
import os

sys.modules.setdefault("serial", __import__("unittest.mock", fromlist=["Mock"]).Mock())
sys.modules.setdefault("cv2", __import__("unittest.mock", fromlist=["Mock"]).Mock())
sys.modules.setdefault("minio", __import__("unittest.mock", fromlist=["Mock"]).Mock())
sys.modules.setdefault("minio.error", __import__("unittest.mock", fromlist=["Mock"]).Mock())

from services.session.weight_motion import (
    DEPARTURE,
    RISE,
    MotionDecision,
    WeightMotion,
)


class WeightMotionDepartureTests(unittest.TestCase):
    def drop(self, motion, weights, start=0.0, step=0.2, reference=40000.0):
        """Feed departure readings; return (decisions, next_mono_index)."""
        decisions = []
        for index, weight in enumerate(weights):
            mono = start + index * step
            decision = motion.observe_loaded(
                weight, mono, "obs-%s" % mono, reference
            )
            if decision is not None:
                decisions.append(decision)
        return decisions

    def test_full_drop_at_40k_reference_confirms(self):
        motion = WeightMotion()
        decisions = self.drop(motion, [36000.0] * 3)

        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision.kind, DEPARTURE)
        self.assertEqual(decision.reference_kg, 40000.0)
        self.assertEqual(decision.threshold_kg, 4000.0)
        self.assertEqual(decision.crossed_at, "obs-0.0")
        self.assertEqual(decision.confirmed_at, "obs-0.4")
        self.assertIsNone(decision.trough_kg)

    def test_crossing_is_inclusive_and_36010_cancels(self):
        motion = WeightMotion()
        # Run starts at the first 36 000 median; the second reading pushes the
        # window median to 36 005, above the crossing, cancelling the run.
        decisions = self.drop(motion, [36000.0, 36010.0, 36010.0, 36010.0, 36010.0])
        self.assertEqual(decisions, [])

        # A fresh run needs a majority of the window back at the crossing.
        decisions = self.drop(motion, [36000.0] * 5, start=1.0)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].crossed_at, "obs-1.4")
        self.assertEqual(decisions[0].confirmed_at, "obs-1.8")

    def test_small_reference_uses_floor_threshold(self):
        motion = WeightMotion()
        decisions = self.drop(motion, [8500.0] * 3, reference=10000.0)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].threshold_kg, 1500.0)

    def test_just_above_floor_crossing_never_confirms(self):
        motion = WeightMotion()
        decisions = self.drop(motion, [8501.0] * 5, reference=10000.0)
        self.assertEqual(decisions, [])

    def test_three_readings_inside_dwell_fail_then_later_confirm(self):
        motion = WeightMotion()
        decisions = self.drop(motion, [36000.0] * 3, step=0.05)
        self.assertEqual(decisions, [])

        decisions = self.drop(motion, [36000.0, 36000.0], start=0.15)
        self.assertEqual(decisions, [])

        decisions = self.drop(motion, [36000.0], start=0.4)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].crossed_at, "obs-0.0")

    def test_no_duplicate_departure_until_transition(self):
        motion = WeightMotion()
        first = self.drop(motion, [36000.0] * 4)
        self.assertEqual(len(first), 1)

        again = self.drop(motion, [36000.0] * 3, start=1.0)
        self.assertEqual(again, [])

        motion.start_visit()
        third = self.drop(motion, [36000.0] * 3, start=2.0)
        self.assertEqual(len(third), 1)

    def test_shallow_dip_does_not_split(self):
        # HP-2 Oct24 19:35 rocking: 610 kg dip on a 37 500 kg plateau.
        motion = WeightMotion()
        decisions = self.drop(motion, [36890.0] * 5, reference=37500.0)
        self.assertEqual(decisions, [])

    def test_zero_and_negative_samples_are_valid_readings(self):
        motion = WeightMotion()
        decisions = self.drop(motion, [0.0, 0.0, -10.0], reference=10000.0)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].threshold_kg, 1500.0)

    def test_reset_discards_continuity(self):
        motion = WeightMotion()
        self.drop(motion, [36000.0, 36000.0])
        motion.reset()

        decisions = self.drop(motion, [36000.0, 36000.0], start=1.0)
        self.assertEqual(decisions, [])

        decisions = self.drop(motion, [36000.0] * 3, start=2.0)
        self.assertEqual(len(decisions), 1)


class WeightMotionRiseTests(unittest.TestCase):
    def setUp(self):
        self.motion = WeightMotion()
        # Confirmed departure from a 40 000 kg visit.
        for index in range(3):
            self.motion.observe_loaded(36000.0, index * 0.2, "d%d" % index, 40000.0)
        self.motion.arm_reentry(40000.0)

    def reenter(self, weights, start=0.0, step=0.2):
        decisions = []
        for index, weight in enumerate(weights):
            mono = start + index * step
            decision = self.motion.observe_reentry(weight, mono, "r-%s" % mono)
            if decision is not None:
                decisions.append(decision)
        return decisions

    def test_arm_reentry_requires_reference(self):
        with self.assertRaises(ValueError):
            WeightMotion().arm_reentry(None)

    def test_reentry_before_arm_raises(self):
        with self.assertRaises(ValueError):
            WeightMotion().observe_reentry(100.0, 0.0, "x")

    def test_rise_requires_trough_plus_ratio_for_dwell(self):
        # Trough settles at 10 000; crossing is 12 000 for 0.4 s.
        decisions = self.reenter([10000.0] * 5)
        self.assertEqual(decisions, [])

        decisions = self.reenter([12100.0] * 5, start=1.0)
        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision.kind, RISE)
        self.assertEqual(decision.reference_kg, 40000.0)
        self.assertEqual(decision.threshold_kg, 2000.0)
        self.assertEqual(decision.trough_kg, 10000.0)
        self.assertEqual(decision.crossed_at, "r-1.4")
        self.assertEqual(decision.confirmed_at, "r-1.8")

    def test_rise_below_crossing_never_confirms(self):
        decisions = self.reenter([10000.0] * 5 + [11500.0] * 10, start=0.0)
        self.assertEqual(decisions, [])

    def test_new_low_resets_pending_rise(self):
        self.reenter([10000.0] * 5)
        # Pending rise run starts at 1.2 s but is younger than the dwell.
        decisions = self.reenter([12100.0] * 3, start=1.0, step=0.1)
        self.assertEqual(decisions, [])

        # A sustained new low pulls the window median to 9 000 and clears
        # the pending run before the dwell could confirm.
        decisions = self.reenter([9000.0] * 3, start=1.3, step=0.1)
        self.assertEqual(decisions, [])

        # Rising from the 9 000 trough needs only 11 000.
        decisions = self.reenter([11200.0] * 5, start=2.0)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].trough_kg, 9000.0)
        self.assertEqual(decisions[0].threshold_kg, 2000.0)
        self.assertEqual(decisions[0].crossed_at, "r-2.4")
        self.assertEqual(decisions[0].confirmed_at, "r-2.8")

    def test_no_duplicate_rise_and_samples_are_retained(self):
        self.reenter([10000.0] * 5)
        first = self.reenter([12100.0] * 5, start=1.0)
        self.assertEqual(len(first), 1)

        again = self.reenter([13000.0] * 3, start=3.0)
        self.assertEqual(again, [])

        samples = self.motion.rise_samples()
        self.assertTrue(samples)
        self.assertEqual(samples[0][0], 12100.0)

    def test_start_visit_clears_reentry_state(self):
        self.motion.start_visit()
        with self.assertRaises(ValueError):
            self.motion.observe_reentry(100.0, 0.0, "x")


if __name__ == "__main__":
    unittest.main()
