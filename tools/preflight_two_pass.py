"""Audit prior low-memory telemetry and enforce the requested two-pass preflight."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.resource_guard import Guard, Policy


def main():
    out = ROOT / "results" / ("TwoPassA_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    prior = ROOT / "results/CPUScreen128_20261111T021018Z/sustained"
    raw = (prior / "telemetry.jsonl").read_bytes()
    history = [json.loads(line) for line in raw.splitlines()]
    resource = json.loads((prior / "resources.json").read_text())
    smallest = min(range(len(history)), key=lambda i: history[i]["available_system_bytes"])
    replay = Guard()
    would_stop = None
    accounting_consistent = True
    for index, sample in enumerate(history):
        processes = sample["processes"]
        accounting_consistent &= sum(p["resident_bytes"] for p in processes) == sample["resident_bytes"]
        accounting_consistent &= max(p["peak_resident_bytes"] for p in processes) == sample["peak_resident_bytes"]
        reason = replay.observe(sample["available_system_bytes"], processes)
        if reason and would_stop is None:
            would_stop = {"sample_index": index, "seconds": sample["seconds"], "phase": sample["phase"],
                          "reason": reason, "available_system_bytes": sample["available_system_bytes"]}
    if not accounting_consistent:
        raise RuntimeError("Prior process-tree memory accounting is inconsistent")
    audit = {"prior_log": str((prior / "telemetry.jsonl").relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(), "prior_reserve_bytes": resource["free_system_reserve_bytes"],
        "prior_guard_stop": resource["guard_stop"], "minimum_sample": history[smallest],
        "minimum_neighborhood": history[max(0, smallest-3):smallest+4],
        "real_interpreter_and_launcher_accounting_consistent": accounting_consistent,
        "strict_new_policy_first_stop_in_replay": would_stop,
        "interpretation": "The previous run used the explicitly approved 1.25 GiB reserve. Its 1.43 GiB minimum did not violate that policy. The requested 2 GiB reserve is stricter and would have stopped the historical run earlier. No prior guard exception is inherited."}
    (out / "prior-memory-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    guard = Guard()
    samples, reason = [], None
    process = psutil.Process()
    start = time.perf_counter()
    with (out / "preflight-memory.jsonl").open("w", encoding="utf-8") as log:
        for index in range(3):
            vm, memory = psutil.virtual_memory(), process.memory_info()
            owned = [{"pid": process.pid, "resident_bytes": memory.rss}]
            reason = guard.observe(vm.available, owned)
            sample = {"sample": index + 1, "elapsed_seconds": time.perf_counter() - start,
                      "available_system_bytes": vm.available, "processes": owned,
                      "guard_state": guard.snapshot(), "stop_reason": reason}
            samples.append(sample)
            log.write(json.dumps(sample) + "\n")
            log.flush()
            if reason:
                break
            if index < 2:
                time.sleep(guard.policy.sampling_interval_seconds)
    report = {"status": "STOPPED" if reason else "PREFLIGHT_PASSED", "phase": "before_training_launch",
        "policy": asdict(guard.policy), "stop_reason": reason, "preflight_samples": samples,
        "minimum_available_system_bytes": min(s["available_system_bytes"] for s in samples),
        "preflight_peak_process_resident_bytes": max(s["processes"][0]["resident_bytes"] for s in samples),
        "planned": {"candidate": "A", "parameters": 5115072, "batch": 4, "context": 128, "threads": 8,
                    "interop_threads": 1, "initialization_seed": 42, "data_order_seed": 43,
                    "passes": 2, "valid_targets_per_pass": 4401479, "total_valid_targets": 8802958,
                    "updates_per_pass": 11301, "total_updates": 22602,
                    "warmup": "No extra uncounted warmup exposures before the two passes."},
        "completed_passes": 0, "completed_optimizer_updates": 0, "valid_targets_processed": 0,
        "model_initialized": False, "validation": None, "checkpoints": [], "continuations": [],
        "guard_tests": "Five focused threshold/consecutive-sample/per-process checks passed before preflight.",
        "initial_validation_loss": None, "pass_throughput": None,
        "another_pass_assessment": "Not assessable: neither pass was executed.",
        "training_seconds": 0, "validation_seconds": 0, "checkpoint_seconds": 0,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [ROOT / "lm/resource_guard.py", Path(__file__), ROOT / "tests/test_resource_guard.py"]}}
    (out / "run.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["CANDIDATE A - CONTROLLED TWO-PASS LEARNING EXPERIMENT", "", f"Status: {report['status']}",
             "Prelaunch memory-log audit and guard verification completed.",
             "The earlier sustained run used an approved 1.25 GiB reserve; this experiment uses the requested 2 GiB reserve.",
             "Five focused guard checks passed: immediate <1 GiB, three consecutive <2 GiB, streak reset, strict boundary comparisons, and three high readings for the same real process (>4 GiB).",
             f"Historical minimum: {history[smallest]['available_system_bytes']:,} bytes ({history[smallest]['available_system_bytes']/2**30:.3f} GiB).",
             "The old launcher/real-interpreter accounting was checked; replaying the stricter policy would have stopped that prior run earlier.", "", "Current preflight:"]
    for sample in samples:
        lines.append(f"  Sample {sample['sample']}: available RAM {sample['available_system_bytes']:,} bytes ({sample['available_system_bytes']/2**30:.3f} GiB); consecutive low count {sample['guard_state']['low_available_count']}.")
    lines += ["", "Stop reason: " + str(reason),
              "Model training was not launched. No threshold was lowered and no automatic retry was started.",
              "Completed passes: 0/2. Optimizer updates: 0. Valid targets processed: 0/8,802,958.",
              "The target shortfall is entirely due to the prelaunch resource stop, not a token-count or padding discrepancy.",
              "Training/validation/checkpoint time: 0/0/0 seconds. Throughput: not measured.",
              "Validation loss at initialization/pass 1/pass 2: unavailable; no model was allocated.",
              "Training peak process RAM: unavailable. The recorded process RAM is only the lightweight preflight process.",
              f"Minimum sampled system RAM: {report['minimum_available_system_bytes']:,} bytes ({report['minimum_available_system_bytes']/2**30:.3f} GiB).",
              "No checkpoints or sample continuations were produced. The frozen final test split remains reserved.",
              "Whether a third pass is worthwhile cannot be assessed without pass-one and pass-two validation losses.",
              "", "Artifacts: prior-memory-audit.json, preflight-memory.jsonl, run.json.",
              "A new launch requires enough free RAM to respect the specified reserve throughout model initialization and training. Do not reuse the old 15-minute weights as the fresh starting point."]
    if reason:
        text = "\n".join(lines) + "\n"
        (out / "Results.txt").write_text(text, encoding="utf-8")
        destination = ROOT / "Results.txt"
        if destination.exists():
            raise RuntimeError("Root Results.txt already exists; artifact Results.txt preserved, no overwrite")
        destination.write_text(text + "\nRun directory: " + str(out.relative_to(ROOT)) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "status": report["status"], "stop_reason": reason,
                      "available_GiB_samples": [s["available_system_bytes"]/2**30 for s in samples],
                      "historical_new_policy_stop": would_stop}, indent=2), flush=True)
    return 2 if reason else 0


if __name__ == "__main__":
    sys.exit(main())
