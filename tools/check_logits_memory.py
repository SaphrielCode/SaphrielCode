"""Fresh-process comparison of releasing an unneeded logits reference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(release):
    import psutil
    import torch
    from lm.batches import read_batch
    from lm.model import Decoder, make_optimizer, parameter_hash, token_loss
    from lm.resource_guard import Guard
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    process = psutil.Process()
    done = threading.Event()
    stopped = []
    samples = []
    def monitor():
        guard = Guard()
        while True:
            available = psutil.virtual_memory().available
            resident = process.memory_info().rss
            samples.append((available, resident))
            reason = guard.observe(available, [{"pid": process.pid, "resident_bytes": resident}])
            if reason:
                stopped.append(reason)
                return
            if done.wait(1):
                return
    watcher = threading.Thread(target=monitor)
    watcher.start()
    try:
        model = Decoder(seed=42)
        optimizer = make_optimizer(model)
        # Fixed real training windows; this is an allocation/speed diagnostic,
        # not a corpus-learning experiment or held-out quality measurement.
        x, y, _ = read_batch("train", [0, 1, 2, 3])
        tokens = int(y.ne(-100).sum())
        def step():
            if stopped:
                raise RuntimeError(stopped[0])
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, summed, valid = token_loss(logits, y)
            if release:
                del logits
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
            optimizer.step()
            return float(summed), valid
        for _ in range(10):
            step()
        # Equal-work correctness checkpoint before time-based divergence.
        correctness_hash = parameter_hash(model)
        started = time.perf_counter()
        updates = 0
        while time.perf_counter() - started < 20:
            step()
            updates += 1
        elapsed = time.perf_counter() - started
        return {"release_logits": release, "updates": updates, "seconds": elapsed,
                "valid_tokens_per_second": tokens * updates / elapsed,
                "hash_after_10_updates": correctness_hash,
                "os_peak_resident_bytes": process.memory_info().peak_wset,
                "minimum_available_bytes": min(s[0] for s in samples),
                "guard_stop": stopped, "samples": samples}
    finally:
        done.set()
        watcher.join()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("baseline", "release"))
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker == "release")))
        return
    out = ROOT / "results" / time.strftime("LogitsMemory_%Y%m%dT%H%M%SZ", time.gmtime())
    out.mkdir()
    results = []
    for mode in ("baseline", "release", "release", "baseline", "baseline", "release"):
        completed = subprocess.run([sys.executable, __file__, "--worker", mode], cwd=ROOT,
                                   capture_output=True, text=True)
        if completed.returncode:
            (out / "failure.txt").write_text(completed.stdout + completed.stderr)
            raise RuntimeError("Diagnostic stopped; see " + str(out))
        result = json.loads(completed.stdout)
        results.append(result)
        (out / "comparison.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}), flush=True)
    print(str(out), flush=True)


if __name__ == "__main__":
    main()
