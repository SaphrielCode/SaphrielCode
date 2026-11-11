"""One real-data Candidate A update; not a speed benchmark or sustained training."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import struct
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import psutil
import torch
from lm.batches import BATCH_DIR, digest, read_batch, verify_batch_lock
from lm.model import Config, Decoder, make_optimizer, parameter_hash, token_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, choices=(1, 4), default=1)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    out = ROOT / "results" / ("PilotA_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    report = {"status": "incomplete", "scope": "Exactly one optimizer update; no sustained-speed measurement",
              "configuration": asdict(Config()), "batch_size": args.batch_size, "context": 128,
              "dtype": "float32", "device": "cpu", "initialization_seed": 42, "training_order_seed": 43,
              "threads": 4, "interop_threads": 1, "versions": {"python": platform.python_version(), "torch": torch.__version__,
                    "numpy": np.__version__, "psutil": psutil.__version__}, "samples": []}
    process = psutil.Process()
    stop, finished = threading.Event(), threading.Event()
    started = time.perf_counter()
    low_count, high_count = 0, 0

    def sample():
        nonlocal low_count, high_count
        memory, system, cpu = process.memory_info(), psutil.virtual_memory(), process.cpu_times()
        value = {"elapsed_seconds": time.perf_counter() - started, "resident_bytes": memory.rss,
                 "peak_resident_bytes": getattr(memory, "peak_wset", memory.rss),
                 "private_bytes": getattr(memory, "private", None), "available_system_bytes": system.available,
                 "cpu_seconds": cpu.user + cpu.system}
        report["samples"].append(value)
        low_count = low_count + 1 if system.available < 2 * 1024**3 else 0
        high_count = high_count + 1 if memory.rss > 4 * 1024**3 else 0
        if system.available < 1024**3 or low_count >= 3 or high_count >= 3 or value["elapsed_seconds"] >= 120:
            report["guard_stop"] = value
            stop.set()

    def monitor():
        while not finished.wait(1):
            sample()

    def guard():
        if stop.is_set():
            raise RuntimeError("Pilot resource/time guard reached (checked at operation boundaries)")

    sample()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        guard()
        lock = verify_batch_lock()
        report["batch_manifest_sha256"] = digest(BATCH_DIR / "manifest.json")
        report["tokenizer_sha256"] = lock["tokenizer_sha256"]
        order = [r[0] for r in struct.iter_unpack("<I", (BATCH_DIR / "train.order.u32").read_bytes())]
        row_meta = (BATCH_DIR / "train.rows.u32").read_bytes()
        # Use the first full windows in the fixed order so this pilot has exactly B*128 real targets.
        chosen = [r for r in order if struct.unpack_from("<III", row_meta, r * 12)[2] == 128][:args.batch_size]
        if len(chosen) != args.batch_size:
            raise RuntimeError("Not enough full training windows")
        inputs, targets, metadata = read_batch("train", chosen)
        report.update(batch_rows=chosen, batch_provenance=[list(v) for v in metadata], input_shape=list(inputs.shape),
                      selection="First full windows from fixed seed-43 order; partial windows excluded from this pilot only.")
        model = Decoder(Config(), seed=42)
        count = sum(p.numel() for p in model.parameters())
        if count != 5115072 or model.output_weight is not model.embedding.weight:
            raise RuntimeError("Parameter count or embedding sharing differs from specification")
        if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in model.parameters()):
            raise RuntimeError("Expected CPU FP32 parameters")
        report["unique_parameters"] = count
        report["initial_parameter_sha256"] = parameter_hash(model)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        optimizer = make_optimizer(model)
        report["optimizer"] = {"type": "AdamW", "lr": 0.0003, "betas": [0.9, 0.95], "eps": 1e-8,
            "matrix_weight_decay": 0.1, "normalization_weight_decay": 0, "foreach": False, "fused": False}
        guard()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss, total_loss, valid = token_loss(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite loss")
        report.update(loss_before_update=float(loss.detach()), token_loss_sum=float(total_loss), valid_target_tokens=valid,
                      logits_shape=list(logits.shape))
        loss.backward()
        guard()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"Missing/non-finite gradient: {name}")
        report["all_gradients_present_and_finite"] = True
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True, foreach=False)
        report["gradient_norm_before_clipping"] = float(norm)
        report["gradient_norm_after_clipping"] = float(torch.stack([p.grad.norm() for p in model.parameters()]).norm())
        optimizer.step()
        guard()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError("Non-finite updated parameters")
        changed = {name: int(torch.count_nonzero(p.detach() != before[name]).item()) for name, p in model.named_parameters()}
        report["changed_parameter_elements"] = sum(changed.values())
        report["changed_parameter_tensors"] = sum(v > 0 for v in changed.values())
        report["max_absolute_parameter_change"] = max(float((p.detach() - before[name]).abs().max()) for name, p in model.named_parameters())
        report["updated_parameter_sha256"] = parameter_hash(model)
        if not report["changed_parameter_elements"] or report["initial_parameter_sha256"] == report["updated_parameter_sha256"]:
            raise RuntimeError("Optimizer did not change weights")
        model.eval()
        with torch.no_grad():
            after_loss, _, _ = token_loss(model(inputs), targets)
        if not torch.isfinite(after_loss):
            raise RuntimeError("Non-finite post-update loss")
        report["same_batch_loss_after_update"] = float(after_loss)
        report["loss_caveat"] = "Same training batch, not held-out loss; one update is not a learning-quality result."
        checkpoint = out / "step1.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": asdict(model.config),
                    "completed_updates": 1, "initial_seed": 42, "tokenizer_sha256": lock["tokenizer_sha256"],
                    "batch_manifest_sha256": report["batch_manifest_sha256"]}, checkpoint)
        report["checkpoint"] = {"file": checkpoint.name, "bytes": checkpoint.stat().st_size, "sha256": digest(checkpoint)}
        report["completed_optimizer_updates"] = 1
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        finished.set()
        watcher.join(timeout=2)
        sample()
        report["diagnostic_wall_seconds"] = time.perf_counter() - started
        report["timing_caveat"] = "Includes checks, hashes, imports already completed, file access and checkpoint I/O; not a throughput result."
        (out / "pilot.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(out), **{k: v for k, v in report.items() if k != "samples"}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
