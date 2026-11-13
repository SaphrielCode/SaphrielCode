"""Read frozen, document-isolated windows; padding is never a loss target."""
import hashlib
import json
from pathlib import Path
import struct

import torch

ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = ROOT / "data/TinyStoriesBatches128"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify_batch_lock(batch_dir=None, source_dir=None):
    batch_dir = BATCH_DIR if batch_dir is None else Path(batch_dir)
    source_dir = ROOT / "data/TinyStoriesSubset" if source_dir is None else Path(source_dir)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    if digest(source_dir / "manifest.json") != manifest["source_manifest_sha256"]:
        raise RuntimeError("Frozen source manifest changed")
    for name, entry in manifest["files"].items():
        path = batch_dir / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise RuntimeError(f"Batch file changed: {name}")
    for name, entry in manifest["source_files"].items():
        if digest(source_dir / name) != entry["sha256"]:
            raise RuntimeError(f"Frozen source changed: {name}")
    return manifest


def read_batch(split, row_indices, batch_dir=None):
    if split not in ("train", "validation", "test"):
        raise ValueError("Unknown split")
    length = 128
    batch_dir = BATCH_DIR if batch_dir is None else Path(batch_dir)
    x_rows, y_rows, metadata = [], [], []
    with (batch_dir / f"{split}.x.u16").open("rb") as xs, (batch_dir / f"{split}.y.u16").open("rb") as ys, (batch_dir / f"{split}.rows.u32").open("rb") as index:
        for row in row_indices:
            if row < 0:
                raise ValueError("Negative row")
            xs.seek(row * length * 2)
            ys.seek(row * length * 2)
            index.seek(row * 12)
            x, y, meta = xs.read(length * 2), ys.read(length * 2), index.read(12)
            if len(x) != length * 2 or len(y) != length * 2 or len(meta) != 12:
                raise ValueError("Batch row outside file")
            x_rows.append(list(struct.unpack("<128H", x)))
            y_rows.append(list(struct.unpack("<128H", y)))
            metadata.append(struct.unpack("<III", meta))
    inputs = torch.tensor(x_rows, dtype=torch.long)
    targets = torch.tensor(y_rows, dtype=torch.long)
    targets.masked_fill_(targets.eq(0), -100)
    return inputs, targets, metadata
