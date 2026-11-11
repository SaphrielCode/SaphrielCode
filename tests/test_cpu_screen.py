"""Check screen aggregation/selection without running model training."""
from pathlib import Path
import sys
import unittest
import subprocess
import time
import psutil
import json
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from cpu_screen_worker import summarize_steps, save_json
from run_cpu_screen import select_finalist, sample_owned_tree, terminate_owned_tree


class ScreenChecks(unittest.TestCase):
    def test_transient_windows_replace_failure_is_retried(self):
        original = Path.replace
        calls = []
        def transient(source, target):
            calls.append(str(source))
            if len(calls) == 1:
                raise PermissionError("simulated transient sharing violation")
            return original(source, target)
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "replace", transient), patch("cpu_screen_worker.time.sleep"):
            target = Path(directory) / "phase.json"
            save_json(target, {"phase": "finished"})
            self.assertEqual(json.loads(target.read_text()), {"phase": "finished"})
            self.assertEqual(len(calls), 2)

    def test_real_interpreter_memory_is_included_below_windows_launcher(self):
        worker = subprocess.Popen([sys.executable, "-c", "x=bytearray(32*1024**2); import time; time.sleep(3)"])
        process = psutil.Process(worker.pid)
        try:
            deadline = time.monotonic() + 2
            largest = 0
            while time.monotonic() < deadline:
                result = sample_owned_tree(process)
                largest = max(largest, result["max_process_resident_bytes"])
                if largest > 32 * 1024**2:
                    break
                time.sleep(0.1)
            self.assertGreater(largest, 32 * 1024**2)
            self.assertEqual(result["resident_bytes"], sum(p["resident_bytes"] for p in result["processes"]))
        finally:
            terminate_owned_tree(process, worker)
            worker.wait(timeout=10)

    def test_throughput_uses_valid_tokens_and_actual_elapsed_time(self):
        result = summarize_steps([0.1, 0.3], 173, 0.5)
        self.assertEqual(result["valid_target_tokens"], 173)
        self.assertEqual(result["valid_tokens_per_second"], 346)
        self.assertEqual(result["completed_updates"], 2)
        self.assertAlmostEqual(result["step_median_seconds"], 0.2)

    def test_selection_requires_all_repeats_and_prefers_small_model_in_tie_band(self):
        runs = []
        for model in ("A", "B"):
            for batch in (1, 4):
                for threads in (4, 8):
                    speed = 100 if model == "B" and batch == 4 and threads == 8 else 97 if model == "A" and batch == 4 and threads == 4 else 70
                    for _ in range(3):
                        runs.append({"candidate": model, "batch": batch, "threads": threads,
                                     "status": "PASS", "timed": {"valid_tokens_per_second": speed},
                                     "peak_process_resident_bytes": 100 if model == "A" else 200})
        winner, rankings, rule = select_finalist(runs)
        self.assertEqual((winner["candidate"], winner["batch"], winner["threads"]), ("A", 4, 4))
        self.assertEqual(len(rankings), 8)
        with self.assertRaises(RuntimeError):
            select_finalist(runs[:-1])


if __name__ == "__main__":
    unittest.main()
