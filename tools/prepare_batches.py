"""Freeze tokenizer metadata and rearrange existing token IDs into isolated windows."""
from array import array
import hashlib
import json
from pathlib import Path
import random
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/TinyStoriesSubset"
OUT = ROOT / "data/TinyStoriesBatches128"
EXPECTED_TOKENIZER = "5d99b0bf97ebebccc88cfc73e657cd82379e9b1495dbf0c75fb5f93fe1dfd16f"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def window_story(ids, length=128):
    if not ids or ids[-1] != 2 or any(i < 4 or i >= 8192 for i in ids[:-1]):
        raise ValueError("Expected body tokens and one terminal EOS")
    inputs = [1] + list(ids[:-1])
    for start in range(0, len(ids), length):
        valid = min(length, len(ids) - start)
        yield (inputs[start:start + valid] + [0] * (length - valid),
               list(ids[start:start + valid]) + [0] * (length - valid), start, valid)


def u16_bytes(values):
    data = array("H", values)
    if sys.byteorder != "little":
        data.byteswap()
    return data.tobytes()


def main():
    import tokenizers
    if OUT.exists():
        raise RuntimeError("Frozen batches already exist; verify/reuse them instead of overwriting")
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    if digest(SOURCE / "tokenizer.json") != EXPECTED_TOKENIZER:
        raise RuntimeError("Tokenizer differs from the approved frozen encoding")
    verification = subprocess.check_output([sys.executable, str(ROOT / "tools/verify_tinystories.py")], cwd=ROOT)
    report = json.loads(verification)
    if report["status"] != "PASS":
        raise RuntimeError("Frozen corpus verification failed")
    config = json.loads((SOURCE / "tokenizer.json").read_text(encoding="utf-8"))
    vocabulary = config["model"]["vocab"]
    special_ids = {s: vocabulary[s] for s in ("<pad>", "<bos>", "<eos>", "<unk>")}
    if len(vocabulary) != 8192 or list(special_ids.values()) != [0, 1, 2, 3]:
        raise RuntimeError("Vocabulary/special IDs changed")
    lock = {"file": "data/TinyStoriesSubset/tokenizer.json", "sha256": EXPECTED_TOKENIZER,
            "vocabulary_size": len(vocabulary), "special_ids": special_ids,
            "configuration": {k: v for k, v in config.items() if k != "model"},
            "model_configuration": {k: v for k, v in config["model"].items() if k not in ("vocab", "merges")},
            "merge_count": len(config["model"]["merges"]), "tokenizers_version": tokenizers.__version__,
            "training_source": "Retained training stories only; original saved tokens reused, no retraining or retokenization.",
            "normalization": source_manifest["deduplication"]["normalization"],
            "verification": report}
    OUT.mkdir(parents=True)
    (OUT / "tokenizer-lock.json").write_text(json.dumps(lock, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = {}
    for split in ("train", "validation", "test"):
        ids = array("H")
        with (SOURCE / f"{split}.u16").open("rb") as source:
            ids.frombytes(source.read())
        if sys.byteorder != "little":
            ids.byteswap()
        offsets = [v[0] for v in struct.iter_unpack("<Q", (SOURCE / f"{split}.idx").read_bytes())]
        count, valid_tokens = 0, 0
        with (OUT / f"{split}.x.u16").open("wb") as xs, (OUT / f"{split}.y.u16").open("wb") as ys, (OUT / f"{split}.rows.u32").open("wb") as rows:
            for doc, (start, end) in enumerate(zip(offsets, offsets[1:])):
                for x, y, within, valid in window_story(ids[start:end]):
                    xs.write(u16_bytes(x))
                    ys.write(u16_bytes(y))
                    rows.write(struct.pack("<III", doc, within, valid))
                    count += 1
                    valid_tokens += valid
        if valid_tokens != source_manifest["statistics"][split]["tokens_including_eos"]:
            raise RuntimeError("Target tokens lost/duplicated during windowing")
        if split == "train":
            order = list(range(count))
            random.Random(43).shuffle(order)
            with (OUT / "train.order.u32").open("wb") as output:
                for index in order:
                    output.write(struct.pack("<I", index))
        names = [f"{split}.x.u16", f"{split}.y.u16", f"{split}.rows.u32"]
        if split == "train":
            names.append("train.order.u32")
        stats[split] = {"documents": len(offsets) - 1, "saved_source_tokens_including_eos": len(ids),
                        "windows": count, "valid_target_tokens": valid_tokens,
                        "eos_targets": len(offsets) - 1, "padding_targets": count * 128 - valid_tokens,
                        "window_file_bytes": sum((OUT / n).stat().st_size for n in names)}
    manifest = {"context": 128, "policy": "document-isolated-right-pad-v1", "statistics": stats,
        "boundaries": "One story per window. First input BOS=1; subsequent chunks use preceding token of the same story. Last target EOS=2. Chunk RoPE positions restart at zero. No cross-story attention or targets.",
        "padding": "Right pad ID=0. Stored target padding is 0; loader converts it to ignore_index=-100. Causal attention prevents valid queries attending later padding; padded query outputs are ignored.",
        "target_coverage": "Every saved body token and EOS is a target exactly once per corpus pass. BOS is input-only; no targets dropped at story/chunk boundaries.",
        "batching": "Rows can be grouped as batch 1 or 4 without duplicating token files. Training order is a frozen seed-43 permutation. Emit a smaller final batch; do not drop targets or mix splits.",
        "storage": "x/y: little-endian uint16 [windows,128]; rows: uint32 triples(document_index,start_target_in_story,valid_length); order: uint32 window indices.",
        "tokenizer_sha256": EXPECTED_TOKENIZER,
        "source_files": {name: value for name, value in source_manifest["files"].items()},
        "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "preparation_script_sha256": digest(Path(__file__)),
        "files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in OUT.iterdir() if p.is_file()}}
    manifest["payload_bytes"] = sum(v["bytes"] for v in manifest["files"].values())
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"statistics": stats, "payload_bytes": manifest["payload_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
