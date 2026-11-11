"""Verify strict thresholds, consecutive-sample reset, and per-process accounting."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lm.resource_guard import Guard

GIB = 1024**3


class GuardChecks(unittest.TestCase):
    def test_first_sample_below_one_gib_stops_immediately(self):
        guard = Guard()
        self.assertEqual(guard.observe(GIB - 1, []), "available_system_ram_below_1_GiB_first_sample")

    def test_three_low_samples_and_reset(self):
        guard = Guard()
        for available in [GIB, GIB, 2 * GIB, GIB, GIB]:
            self.assertIsNone(guard.observe(available, []))
        self.assertEqual(guard.observe(GIB, []), "available_system_ram_below_2_GiB_three_consecutive_samples")

    def test_exact_thresholds_do_not_trigger_the_strict_inequalities(self):
        guard = Guard()
        for _ in range(4):
            self.assertIsNone(guard.observe(2 * GIB, [{"pid": 1, "resident_bytes": 4 * GIB}]))

    def test_process_limit_tracks_real_pid_and_requires_three_samples(self):
        guard = Guard()
        launcher = {"pid": 1, "resident_bytes": 4 * 1024**2}
        worker = {"pid": 2, "resident_bytes": 4 * GIB + 1}
        self.assertIsNone(guard.observe(3 * GIB, [launcher, worker]))
        self.assertIsNone(guard.observe(3 * GIB, [launcher, worker]))
        self.assertEqual(guard.observe(3 * GIB, [launcher, worker]), "process_resident_ram_above_4_GiB_three_consecutive_samples")

    def test_high_process_streak_is_not_transferred_to_another_pid(self):
        guard = Guard()
        for pid in [1, 1, 2, 2]:
            self.assertIsNone(guard.observe(3 * GIB, [{"pid": pid, "resident_bytes": 4 * GIB + 1}]))
        self.assertEqual(guard.observe(3 * GIB, [{"pid": 2, "resident_bytes": 4 * GIB + 1}]), "process_resident_ram_above_4_GiB_three_consecutive_samples")


if __name__ == "__main__":
    unittest.main()
