"""Sequential guarded 24-run 128-token CPU screen, then one 900-second finalist."""
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from cpu_screen_worker import save_json


class ResourceStop(RuntimeError):
    pass


def sample_owned_tree(process):
    """Windows venv redirectors can put the real interpreter below Popen's PID."""
    records = []
    for owned in [process] + process.children(recursive=True):
        try:
            memory, cpu = owned.memory_info(), owned.cpu_times()
            records.append({"pid": owned.pid, "resident_bytes": memory.rss,
                "peak_resident_bytes": getattr(memory, "peak_wset", memory.rss),
                "private_bytes": getattr(memory, "private", 0), "cpu_seconds": cpu.user + cpu.system,
                "page_faults": getattr(memory, "num_page_faults", None)})
        except psutil.NoSuchProcess:
            continue
    if not records:
        raise psutil.NoSuchProcess(process.pid)
    return {"processes": records, "resident_bytes": sum(p["resident_bytes"] for p in records),
            "max_process_resident_bytes": max(p["resident_bytes"] for p in records),
            "peak_resident_bytes": max(p["peak_resident_bytes"] for p in records),
            "private_bytes": sum(p["private_bytes"] for p in records),
            "cpu_seconds": sum(p["cpu_seconds"] for p in records),
            "process_page_faults_soft_and_hard": sum(p["page_faults"] or 0 for p in records)}


def terminate_owned_tree(process, worker):
    try:
        descendants = process.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    if worker.poll() is None:
        worker.terminate()


def preflight():
    samples = []
    for _ in range(3):
        available = psutil.virtual_memory().available
        samples.append(available)
        if available < 1024**3:
            raise ResourceStop("Available RAM below immediate 1 GiB guard")
        time.sleep(1)
    if all(v < int(1.25 * 1024**3) for v in samples):
        raise ResourceStop("Available RAM below approved 1.25 GiB reserve for three consecutive preflight samples")
    return samples


def run_one(root, name, candidate, batch, threads, seconds, evaluate=False):
    out = root / name
    out.mkdir(exist_ok=False)
    preflight_samples = preflight()
    command = [sys.executable, str(ROOT / "tools/cpu_screen_worker.py"), "--candidate", candidate,
               "--batch-size", str(batch), "--threads", str(threads), "--seconds", str(seconds), "--out", str(out)]
    if evaluate:
        command.append("--evaluate")
    env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "OPENBLAS_NUM_THREADS": str(threads)}
    resource = {"preflight_available_bytes": preflight_samples, "samples": [], "guard_stop": None,
                "cpu_usage_convention": "100% means one fully occupied logical CPU; divide by logical CPU count for machine share.",
                "sampling_interval_seconds": 1, "free_system_reserve_bytes": int(1.25 * 1024**3), "immediate_stop_available_bytes": 1024**3, "process_limit_bytes": 4 * 1024**3, "scope": "Owned worker interpreter plus Windows venv launcher descendants. RSS/private/CPU are summed across this tree. Peak process RSS is the maximum OS-reported per-process peak, not a sum of nonconcurrent peaks. The 4 GiB guard checks each process. System RAM/disk counters include unrelated applications."}
    start = time.perf_counter()
    reason, stop_at = None, None
    low, high = 0, 0
    with (out / "stdout.txt").open("wb") as stdout, (out / "stderr.txt").open("wb") as stderr, (out / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry:
        worker = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        process = psutil.Process(worker.pid)
        while worker.poll() is None:
            try:
                owned, vm = sample_owned_tree(process), psutil.virtual_memory()
                disk = psutil.disk_io_counters()
                try:
                    phase = json.loads((out / "phase.json").read_text())["phase"]
                except (FileNotFoundError, json.JSONDecodeError):
                    phase = "startup"
                sample = {"seconds": time.perf_counter() - start, "phase": phase,
                    **owned, "available_system_bytes": vm.available,
                    "system_disk_read_bytes": disk.read_bytes if disk else None}
                resource["samples"].append(sample)
                telemetry.write(json.dumps(sample) + "\n")
                telemetry.flush()
                low = low + 1 if vm.available < int(1.25 * 1024**3) else 0
                high = high + 1 if owned["max_process_resident_bytes"] > 4 * 1024**3 else 0
                if reason is None:
                    if vm.available < 1024**3:
                        reason = "available_ram_below_1_GiB"
                    elif low >= 3:
                        reason = "available_ram_below_1.25_GiB_three_samples"
                    elif high >= 3:
                        reason = "process_resident_above_4_GiB_three_samples"
                    elif sample["seconds"] > seconds + 600:
                        reason = "worker_overall_watchdog"
                    if reason:
                        resource["guard_stop"] = {"reason": reason, "sample": sample}
                        save_json(out / "stop.json", {"reason": reason})
                        stop_at = time.monotonic()
                if reason and (reason == "available_ram_below_1_GiB" or time.monotonic() - stop_at >= 5):
                    terminate_owned_tree(process, worker)
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(1)
        worker.wait(timeout=30)
    resource["worker_exit_code"] = worker.returncode
    resource["parent_observed_wall_seconds"] = time.perf_counter() - start
    samples = resource["samples"]
    resource.update(peak_process_resident_bytes=max((s["peak_resident_bytes"] for s in samples), default=0),
                    minimum_sampled_available_system_bytes=min((s["available_system_bytes"] for s in samples), default=None),
                    peak_sampled_private_bytes=max((s["private_bytes"] or 0 for s in samples), default=0))
    timed_samples = [s for s in samples if s["phase"] == "timed_training"]
    if len(timed_samples) >= 2:
        first, last = timed_samples[0], timed_samples[-1]
        resource["timed_cpu_percent_one_core"] = 100 * (last["cpu_seconds"] - first["cpu_seconds"]) / (last["seconds"] - first["seconds"])
    save_json(out / "resources.json", resource)
    report_path = out / "training.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"status": "FAIL", "error": "No worker result"}
    summary = {"name": name, "candidate": candidate, "batch": batch, "threads": threads,
               "directory": out.relative_to(ROOT).as_posix(),
               "status": "PASS" if report["status"] == "PASS" and worker.returncode == 0 and not reason else "STOPPED" if reason else "FAIL",
               "guard_stop": reason, "timed": report.get("timed_training"),
               "peak_process_resident_bytes": resource["peak_process_resident_bytes"],
               "minimum_sampled_available_system_bytes": resource["minimum_sampled_available_system_bytes"],
               "initial_parameter_sha256": report.get("initial_parameter_sha256"), "error": report.get("error")}
    print(json.dumps(summary), flush=True)
    return summary


def select_finalist(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["candidate"], run["batch"], run["threads"]), []).append(run)
    rankings = []
    for key, records in groups.items():
        if len(records) != 3 or any(r["status"] != "PASS" for r in records):
            continue
        values = [r["timed"]["valid_tokens_per_second"] for r in records]
        median = statistics.median(values)
        mad = statistics.median(abs(v - median) for v in values)
        rankings.append({"candidate": key[0], "batch": key[1], "threads": key[2], "median_tokens_per_second": median,
                         "minimum_tokens_per_second": min(values), "maximum_tokens_per_second": max(values),
                         "relative_median_absolute_deviation": mad / median,
                         "peak_resident_bytes": max(r["peak_process_resident_bytes"] for r in records)})
    if len(rankings) != 8:
        raise RuntimeError("Need all eight configurations with three successful runs")
    rankings.sort(key=lambda r: r["median_tokens_per_second"], reverse=True)
    best = rankings[0]
    tolerance = max(0.05, best["relative_median_absolute_deviation"])
    close = [r for r in rankings if r["median_tokens_per_second"] >= best["median_tokens_per_second"] * (1 - tolerance)]
    close.sort(key=lambda r: (r["candidate"] != "A", r["peak_resident_bytes"], r["threads"]))
    return close[0], rankings, {"relative_tie_band": tolerance,
        "rule": "All 3 runs must pass. Rank median valid tokens/s; within max(5%, fastest relative MAD), prefer smaller model, then lower observed peak RAM, then fewer threads. No language-quality ranking."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, help="Reuse successful runs from an interrupted suite; write a separate continuation suite.")
    args = parser.parse_args()
    root = ROOT / "results" / ("CPUScreen128_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    root.mkdir(parents=True, exist_ok=False)
    battery = psutil.sensors_battery()
    suite = {"status": "incomplete", "runs": [], "machine": {"platform": platform.platform(),
        "processor_identifier": os.environ.get("PROCESSOR_IDENTIFIER"), "logical_cpus": psutil.cpu_count(),
        "physical_cpus": psutil.cpu_count(logical=False), "physical_ram_bytes": psutil.virtual_memory().total,
        "plugged_in": battery.power_plugged if battery else None},
        "scope": "128-token speed/memory screen and one 900-second sustained run. No 256-context, auxiliary objective or final-test quality evaluation.",
        "recovery_seconds": 3, "approved_guard_exception": "User approved 1.25 GiB available-RAM reserve for this screen/sustained experiment; immediate 1 GiB stop and per-process 4 GiB limit unchanged."}
    if args.resume:
        previous = args.resume.resolve()
        if not previous.is_relative_to((ROOT / "results").resolve()) or (previous / "INVALID_PROCESS_MEMORY.txt").exists():
            raise RuntimeError("Refusing invalid/out-of-workspace resume source")
        old = json.loads((previous / "suite.json").read_text())
        suite["continuation_of"] = str(previous.relative_to(ROOT))
        suite["excluded_runs"] = [r for r in old["runs"] if r["status"] != "PASS"]
        for old_run in old["runs"]:
            if old_run["status"] != "PASS":
                continue
            directory = ROOT / old_run.get("directory", str(previous.relative_to(ROOT) / old_run["name"]))
            training = json.loads((directory / "training.json").read_text())
            resources = json.loads((directory / "resources.json").read_text())
            if training["status"] != "PASS" or resources["worker_exit_code"] != 0 or resources["guard_stop"] or training["timed_training"]["elapsed_seconds"] < 20:
                raise RuntimeError("Resume source lacks successful training/resource completion")
            suite["runs"].append({**old_run, "directory": str(directory.relative_to(ROOT)).replace("\\", "/"), "reused_verified_completion": True})
    try:
        power = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True, timeout=10)
        suite["machine"]["power_scheme_read_only"] = power.stdout.strip() or power.stderr.strip()
    except Exception as error:
        suite["machine"]["power_scheme_read_error"] = repr(error)
    print("Suite:", root, flush=True)
    save_json(root / "suite.json", suite)
    configs = [(m, b, t) for m in ("A", "B") for b in (1, 4) for t in (4, 8)]
    try:
        for repetition in range(3):
            shift = (0, 3, 5)[repetition]
            rotated = configs[shift:] + configs[:shift]
            for candidate, batch, threads in rotated:
                name = f"r{repetition+1}_{candidate}_b{batch}_t{threads}"
                if any(r["name"] == name for r in suite["runs"]):
                    continue
                suite["runs"].append(run_one(root, name, candidate, batch, threads, 20))
                save_json(root / "suite.json", suite)
                latest = suite["runs"][-1]
                if latest["status"] != "PASS":
                    raise ResourceStop(f"Run {name} did not pass; no automatic guard increase/retry")
                hashes = {r["initial_parameter_sha256"] for r in suite["runs"] if r["candidate"] == candidate}
                if len(hashes) != 1:
                    raise RuntimeError("Initialization mismatch between fresh runs")
                time.sleep(3)
        winner, rankings, rule = select_finalist(suite["runs"])
        suite.update(finalist=winner, rankings=rankings, selection_rule=rule)
        save_json(root / "suite.json", suite)
        print("Finalist:", json.dumps(winner), flush=True)
        suite["sustained"] = run_one(root, "sustained", winner["candidate"], winner["batch"], winner["threads"], 900, evaluate=True)
        if suite["sustained"]["status"] != "PASS":
            raise ResourceStop("Sustained run stopped or failed")
        rate = suite["sustained"]["timed"]["valid_tokens_per_second"]
        suite["projections"] = {"valid_tokens_per_hour": rate * 3600, "valid_tokens_per_day": rate * 86400,
            "hours_per_million_tokens": 1e6 / rate / 3600, "hours_per_ten_million_tokens": 1e7 / rate / 3600,
            "caveat": "Extrapolation from 15 minutes, not a quality prediction or guarantee of unchanged day-long thermals/background load. Excludes evaluation/checkpoint downtime."}
        suite["status"] = "PASS"
    except Exception as error:
        suite.update(status="STOPPED" if isinstance(error, ResourceStop) else "FAIL", error=repr(error))
    finally:
        save_json(root / "suite.json", suite)
        print(json.dumps({"status": suite["status"], "completed_screen_runs": len(suite["runs"]),
                          "error": suite.get("error"), "projections": suite.get("projections")}), flush=True)
    return 0 if suite["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
