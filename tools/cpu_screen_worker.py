"""One fresh CPU training screen/sustained run, controlled by run_cpu_screen.py."""
import argparse
from array import array
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    # Windows readers/OneDrive may briefly hold the destination without delete sharing.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.05 * (attempt + 1))


def summarize_steps(latencies, valid_tokens, elapsed):
    import numpy as np
    return {"completed_updates": len(latencies), "valid_target_tokens": valid_tokens,
            "elapsed_seconds": elapsed, "valid_tokens_per_second": valid_tokens / elapsed,
            "steps_per_second": len(latencies) / elapsed,
            "step_median_seconds": float(np.median(latencies)),
            "step_p95_seconds": float(np.percentile(latencies, 95)),
            "step_max_seconds": max(latencies)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=("A", "B"), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--threads", type=int, choices=(4, 8), required=True)
    parser.add_argument("--seconds", type=float, choices=(20., 900.), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import psutil
    import torch
    from lm.batches import BATCH_DIR, digest, verify_batch_lock
    from lm.model import Config, Decoder, make_optimizer, parameter_hash, token_loss

    out = args.out.resolve()
    report = {"status": "incomplete", "candidate": args.candidate, "batch_size": args.batch_size,
              "threads": args.threads, "interop_threads": 1, "context": 128, "requested_seconds": args.seconds,
              "seed": 42, "order_seed": 43, "versions": {"torch": torch.__version__, "numpy": np.__version__, "python": sys.version},
              "source_hashes": {str(p.relative_to(ROOT)): digest(p) for p in [ROOT / "lm/model.py", ROOT / "lm/batches.py", Path(__file__)]},
              "validation": {}, "final_test_evaluated": False}
    started = time.perf_counter()
    interrupted, watcher_done = threading.Event(), threading.Event()
    phase = "setup"

    def set_phase(name):
        nonlocal phase
        phase = name
        save_json(out / "phase.json", {"phase": name, "elapsed_seconds": time.perf_counter() - started})

    def watch_stop():
        while not watcher_done.wait(0.2):
            if (out / "stop.json").exists():
                interrupted.set()

    def guard():
        if interrupted.is_set():
            raise RuntimeError("Parent resource/time guard requested stop")

    class Buffers:
        def __init__(self, split):
            self.x = np.fromfile(BATCH_DIR / f"{split}.x.u16", dtype="<u2").reshape(-1, 128)
            self.y = np.fromfile(BATCH_DIR / f"{split}.y.u16", dtype="<u2").reshape(-1, 128)
            self.count = len(self.x)
            self.order = np.fromfile(BATCH_DIR / "train.order.u32", dtype="<u4").astype(np.int64) if split == "train" else np.arange(self.count, dtype=np.int64)
            self.cursor = 0
            self.epochs = 0
            self.delivered = 0

        def batch(self, size):
            if self.cursor == self.count:
                self.cursor = 0
                self.epochs += 1
            end = min(self.cursor + size, self.count)
            index = self.order[self.cursor:end]
            self.delivered += len(index)
            self.cursor = end
            x = torch.from_numpy(self.x[index].astype(np.int64))
            y = torch.from_numpy(self.y[index].astype(np.int64))
            y.masked_fill_(y.eq(0), -100)
            return x, y

        def sequential_batch(self, start, size):
            x = torch.from_numpy(self.x[start:start+size].astype(np.int64))
            y = torch.from_numpy(self.y[start:start+size].astype(np.int64))
            y.masked_fill_(y.eq(0), -100)
            return x, y

    watcher = threading.Thread(target=watch_stop, daemon=True)
    watcher.start()
    latencies, tokens, loss_sum, data_seconds = array("d"), 0, 0., 0.
    executed_positions = 0
    buckets = [{"valid_tokens": 0, "updates": 0} for _ in range(math.ceil(args.seconds / 60))]
    timed_start = None
    optimizer = model = None
    try:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)
        guard()
        batch_lock = verify_batch_lock()
        report["batch_manifest_sha256"] = digest(BATCH_DIR / "manifest.json")
        report["tokenizer_sha256"] = batch_lock["tokenizer_sha256"]
        cfg = Config() if args.candidate == "A" else Config(layers=10, width=256, heads=8, ff_width=1024)
        model = Decoder(cfg, seed=42)
        count = sum(p.numel() for p in model.parameters())
        if count != (5115072 if args.candidate == "A" else 9966848):
            raise RuntimeError("Parameter count differs from plan")
        report.update(configuration=asdict(cfg), unique_parameters=count, initial_parameter_sha256=parameter_hash(model))
        optimizer = make_optimizer(model)
        train = Buffers("train")
        report["data_mode"] = "All prepared train windows resident as uint16 NumPy buffers; only the current batch is converted to int64. Gather/conversion/masking included in step timing. No disk data reads inside timed training."
        report["environment"] = {k: __import__('os').environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
        report["torch_parallel_info"] = torch.__config__.parallel_info()
        report["setup_seconds"] = time.perf_counter() - started
        valid_data = Buffers("validation") if args.evaluate else None

        def evaluate(label):
            set_phase("validation_" + label)
            model.eval()
            total, n = 0., 0
            start = time.perf_counter()
            with torch.no_grad():
                for at in range(0, valid_data.count, 4):
                    guard()
                    x, y = valid_data.sequential_batch(at, 4)
                    _, summed, valid = token_loss(model(x), y)
                    total += float(summed)
                    n += valid
            if n != batch_lock["statistics"]["validation"]["valid_target_tokens"]:
                raise RuntimeError("Validation token coverage differs")
            report["validation"][label] = {"loss_sum": total, "valid_tokens": n, "loss_nats_per_token": total / n,
                                             "seconds": time.perf_counter() - start}

        if args.evaluate:
            evaluate("fresh_initialization")

        def step():
            nonlocal data_seconds
            guard()
            at = time.perf_counter()
            x, y = train.batch(args.batch_size)
            data_seconds += time.perf_counter() - at
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss, summed, valid = token_loss(logits, y)
            # Cross-entropy retains what backward needs. Release this extra
            # vocabulary-sized reference before backward/optimizer allocations.
            del logits
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
            optimizer.step()
            guard()
            return valid, float(summed), float(loss.detach()), float(norm), x.shape[0]

        set_phase("warmup")
        model.train()
        warm_start = time.perf_counter()
        warm_tokens, warm_sum = 0, 0.
        for _ in range(10):
            n, summed, last_loss, norm, actual_batch = step()
            warm_tokens += n
            warm_sum += summed
        report["warmup"] = {"updates": 10, "valid_tokens": warm_tokens, "loss_sum": warm_sum,
                             "seconds": time.perf_counter() - warm_start, "last_loss": last_loss}
        if args.evaluate:
            evaluate("after_warmup")
        model.train()
        set_phase("timed_training")
        data_seconds = 0.
        timed_start = time.perf_counter()
        report["first_timed_loss"] = None
        first_norm, clipping = None, 0
        while time.perf_counter() - timed_start < args.seconds:
            at = time.perf_counter()
            n, summed, last_loss, norm, actual_batch = step()
            elapsed = time.perf_counter() - at
            latencies.append(elapsed)
            tokens += n
            executed_positions += actual_batch * 128
            loss_sum += summed
            clipping += norm > 1.0
            if report["first_timed_loss"] is None:
                report["first_timed_loss"], first_norm = last_loss, norm
            bucket = min(int((at - timed_start) // 60), len(buckets) - 1)
            buckets[bucket]["valid_tokens"] += n
            buckets[bucket]["updates"] += 1
            if len(latencies) == 1 or (bucket > 0 and buckets[bucket]["updates"] == 1):
                print(json.dumps({"phase": "training_progress", "elapsed_seconds": time.perf_counter() - timed_start,
                                  "updates": len(latencies), "valid_tokens": tokens, "last_loss": last_loss}), flush=True)
        timed_elapsed = time.perf_counter() - timed_start
        report["timed_training"] = {**summarize_steps(latencies, tokens, timed_elapsed), "loss_sum": loss_sum,
            "aggregate_loss_nats_per_token": loss_sum / tokens, "last_loss": last_loss,
            "executed_positions_including_padding": executed_positions, "ignored_padding_targets": executed_positions - tokens,
            "batch_gather_seconds_included": data_seconds, "gradient_clipping_updates": clipping,
            "first_gradient_norm": first_norm}
        for index, bucket in enumerate(buckets):
            duration = min(60., max(0., timed_elapsed - index * 60))
            bucket.update(index=index + 1, duration_seconds=duration,
                          valid_tokens_per_second=bucket["valid_tokens"] / duration if duration else None)
        report["time_windows"] = buckets
        report["window_accounting"] = "Completed step work attributed to the minute in which that step started; boundary-straddling steps may slightly affect adjacent minute rates."
        report["coverage"] = {"warmup_and_timed_windows_delivered": train.delivered,
            "completed_corpus_passes": train.epochs, "cursor_in_current_pass": train.cursor,
            "distinct_windows_seen": min(train.delivered, train.count), "total_windows": train.count}
        set_phase("post_training_checks")
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Non-finite final weights")
        report["final_parameter_sha256"] = parameter_hash(model)
        if report["final_parameter_sha256"] == report["initial_parameter_sha256"]:
            raise RuntimeError("Weights did not change")
        if args.evaluate:
            evaluate("after_timed_training")
            set_phase("checkpoint")
            saving = time.perf_counter()
            path = out / "checkpoint.pt"
            torch.save({"config": asdict(cfg), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "completed_updates": 10 + len(latencies), "tokenizer_sha256": report["tokenizer_sha256"],
                        "batch_manifest_sha256": report["batch_manifest_sha256"]}, path)
            report["checkpoint"] = {"file": path.name, "bytes": path.stat().st_size,
                                    "save_seconds": time.perf_counter() - saving, "sha256": digest(path)}
        # Explicitly labeled arithmetic estimate, not elapsed-time-as-compute or measured hardware FLOPs.
        forward = cfg.layers * (24 * args.batch_size * 128 * cfg.width**2 + 4 * args.batch_size * 128**2 * cfg.width) + 2 * args.batch_size * 128 * cfg.width * cfg.vocab_size
        report["arithmetic_estimate"] = {"full_batch_forward_matrix_flops": forward,
            "scope": "Analytical dense-matmul cross-check only; no profiled backward ledger or total-compute claim in this speed/memory screen."}
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="STOPPED" if interrupted.is_set() else "FAIL", error=repr(error), stopped_phase=phase)
        if timed_start is not None and len(latencies):
            report["partial_timed_training"] = summarize_steps(latencies, tokens, time.perf_counter() - timed_start)
    finally:
        watcher_done.set()
        watcher.join(timeout=1)
        report["worker_wall_seconds_excluding_initial_import"] = time.perf_counter() - started
        if len(latencies):
            with (out / "step_times.f64").open("wb") as output:
                if sys.byteorder != "little":
                    latencies.byteswap()
                latencies.tofile(output)
        save_json(out / "training.json", report)
        set_phase("finished")
        print(json.dumps({"status": report["status"], "candidate": args.candidate, "batch": args.batch_size,
                          "threads": args.threads, "timed": report.get("timed_training"), "error": report.get("error")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
