"""Prepare a frozen TinyStories benchmark corpus; does not train an LM."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["RAYON_NUM_THREADS"] = "1"

from array import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import struct
import sys
import time
import unicodedata
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPO = "roneneldan/TinyStories"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
OUT = ROOT / "data/TinyStoriesSubset"
REF = ROOT / "corpus-references/TinyStories"
CACHE = ROOT / "preparation-cache/TinyStories"
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def normalize(text):
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def choose_indices(population, count, seed):
    return random.Random(seed).sample(range(population), count)


def deduplicate(splits):
    # Protect held-out material: test has priority, then validation, then training.
    seen, retained, removed = set(), {}, {}
    for split in ("test", "validation", "train"):
        retained[split], removed[split] = [], []
        for row in sorted(splits[split], key=lambda r: r["source_row"]):
            text = normalize(row["text"])
            digest = hashlib.sha256(text.encode()).hexdigest()
            if not text or digest in seen:
                removed[split].append({"source_row": row["source_row"], "text_sha256": digest,
                                       "reason": "empty" if not text else "duplicate"})
                continue
            seen.add(digest)
            retained[split].append({**row, "text": text, "text_sha256": digest})
    return retained, removed


def download(entry):
    target = CACHE / Path(entry["rfilename"]).name
    expected = entry["lfs"]["sha256"]
    if target.exists() and target.stat().st_size == entry["size"] and sha256(target) == expected:
        return target
    size = entry["size"]
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{entry['rfilename']}?download=true"
    chunk_size = 8 * 1024 * 1024

    def fetch(start):
        end = min(start + chunk_size, size) - 1
        for attempt in range(3):
            try:
                request = urllib.request.Request(url + f"&range={start}", headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise RuntimeError("Unexpected source range")
                    data = response.read(end - start + 2)
                if len(data) != end - start + 1:
                    raise RuntimeError("Truncated source range")
                return data
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    partial = target.with_suffix(".part")
    with partial.open("wb") as output, ThreadPoolExecutor(max_workers=8) as pool:
        for group in range(0, size, 8 * chunk_size):
            for data in pool.map(fetch, range(group, min(group + 8 * chunk_size, size), chunk_size)):
                output.write(data)
            print(target.name, output.tell(), "/", size, flush=True)
    if partial.stat().st_size != size or sha256(partial) != expected:
        raise RuntimeError("Source checksum failed")
    partial.replace(target)
    return target


def read_selected(files, wanted, pq):
    selected, base = [], 0
    wanted = set(wanted)
    for entry, path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=2048, columns=["text"]):
            for local, value in enumerate(batch.column(0)):
                index = base + local
                if index in wanted:
                    selected.append({"source_row": index, "source_file": entry["rfilename"], "text": value.as_py()})
            base += batch.num_rows
    if len(selected) != len(wanted):
        raise RuntimeError("Selected row count mismatch")
    return selected


def main():
    import pyarrow
    import pyarrow.parquet as pq
    import tokenizers
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    if OUT.exists():
        raise RuntimeError("Frozen output already exists; do not overwrite it. Use the verifier.")
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        raise RuntimeError("Need 3 GiB free for source cache, subset and headroom")
    REF.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}?blobs=true", timeout=60) as response:
        source = json.load(response)
    entries = sorted([e for e in source["siblings"] if e["rfilename"].startswith("data/") and e["rfilename"].endswith(".parquet")], key=lambda e: e["rfilename"])
    if len(entries) != 5:
        raise RuntimeError("Expected four train shards and one validation shard")
    write_json(REF / "source-manifest.json", {"repo": REPO, "revision": REVISION, "files": entries})
    references = {
        "dataset-card.md": f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/README.md",
        "LICENSE-CDLA-Sharing-1.0.txt": "https://raw.githubusercontent.com/spdx/license-list-data/main/text/CDLA-Sharing-1.0.txt",
    }
    for name, url in references.items():
        with urllib.request.urlopen(url, timeout=60) as response:
            (REF / name).write_bytes(response.read())
    card = (REF / "dataset-card.md").read_bytes()
    expected_card = next(e["blobId"] for e in source["siblings"] if e["rfilename"] == "README.md")
    if hashlib.sha1(f"blob {len(card)}\0".encode() + card).hexdigest() != expected_card:
        raise RuntimeError("Dataset card verification failed")
    files = [(e, download(e)) for e in entries]
    train = [(e, p) for e, p in files if e["rfilename"].startswith("data/train-")]
    valid = [(e, p) for e, p in files if e["rfilename"].startswith("data/validation-")]
    populations = {name: sum(pq.ParquetFile(p).metadata.num_rows for _, p in group)
                   for name, group in [("train", train), ("validation", valid)]}
    train_indices = choose_indices(populations["train"], 20000, 1729)
    held_indices = choose_indices(populations["validation"], 2000, 1730)
    valid_indices, test_indices = set(held_indices[:1000]), set(held_indices[1000:])
    held = read_selected(valid, held_indices, pq)
    splits = {"train": read_selected(train, train_indices, pq),
              "validation": [r for r in held if r["source_row"] in valid_indices],
              "test": [r for r in held if r["source_row"] in test_indices]}
    splits, removed = deduplicate(splits)
    OUT.mkdir(parents=True)
    for split, rows in splits.items():
        with (OUT / (split + ".jsonl")).open("w", encoding="utf-8", newline="\n") as output:
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8192, min_frequency=2, show_progress=False,
        special_tokens=SPECIALS, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((r["text"] for r in splits["train"]), trainer=trainer, length=len(splits["train"]))
    if tokenizer.get_vocab_size() != 8192:
        raise RuntimeError("Tokenizer did not reach exactly 8192 entries")
    for index, special in enumerate(SPECIALS):
        if tokenizer.token_to_id(special) != index:
            raise RuntimeError("Special token ID mismatch")
    tokenizer.save(str(OUT / "tokenizer.json"))
    stats = {}
    for split, rows in splits.items():
        count, body_count = 0, 0
        with (OUT / (split + ".u16")).open("wb") as output, (OUT / (split + ".idx")).open("wb") as offsets:
            offsets.write(struct.pack("<Q", 0))
            for row in rows:
                ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
                if any(i < 4 for i in ids) or tokenizer.decode(ids) != row["text"]:
                    raise RuntimeError("Story failed lossless byte coverage/special-token check")
                body_count += len(ids)
                ids.append(2)  # Exactly one EOS boundary per story; no automatic BOS or padding.
                data = array("H", ids)
                if sys.byteorder != "little":
                    data.byteswap()
                data.tofile(output)
                count += len(ids)
                offsets.write(struct.pack("<Q", count))
        stats[split] = {"stories": len(rows), "body_tokens": body_count, "tokens_including_eos": count,
                        "mean_body_tokens_per_story": body_count / len(rows)}
    for text in ["Unseen: 中文, café, 🐈\n\t", "e\u0301", "".join(chr(i) for i in range(128))]:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if 3 in ids or tokenizer.decode(ids) != unicodedata.normalize("NFC", text):
            raise RuntimeError("Unseen Unicode byte-coverage check failed")
    manifest = {
        "dataset": REPO, "revision": REVISION, "license": "CDLA-Sharing-1.0",
        "attribution": "TinyStories by Ronen Eldan and Yuanzhi Li; source dataset card preserved.",
        "dataset_url": f"https://huggingface.co/datasets/{REPO}",
        "source_files": entries, "source_populations": populations,
        "sampling": {"method": "Python Random.sample without replacement from full supplied split row ranges",
                     "train_seed": 1729, "validation_pool_seed": 1730,
                     "requested": {"train": 20000, "validation": 1000, "test": 1000},
                     "heldout_partition": "First 1000 sampled validation indices for validation, next 1000 for test; source-order output."},
        "deduplication": {"normalization": "NFC, CRLF/CR -> LF, strip outer whitespace; preserve internal whitespace/case",
                          "priority": ["test", "validation", "train"], "removed": removed,
                          "refill_after_removal": False, "near_duplicates": "Not checked; exact normalized matches only."},
        "tokenizer": {"type": "ByteLevel BPE", "vocabulary": 8192, "training_split": "train only after deduplication",
                      "min_frequency": 2, "special_ids": dict(zip(SPECIALS, range(4))),
                      "normalizer": "NFC", "add_prefix_space": False, "parallelism": "disabled, Rayon threads 1",
                      "random_seed": "No tokenizer sampling/dropout; deterministic trainer inputs and pinned version; saved tokenizer hash is authoritative."},
        "token_files": "Little-endian uint16 body IDs + one EOS per story; no BOS/padding. .idx contains uint64 starts plus final end offset. No packing yet.",
        "statistics": stats, "versions": {"python": platform.python_version(), "pyarrow": pyarrow.__version__, "tokenizers": tokenizers.__version__},
        "script_sha256": sha256(Path(__file__)),
        "reference_files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in REF.iterdir() if p.is_file()},
        "reference_urls": references,
        "files": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in OUT.iterdir() if p.is_file()},
    }
    manifest["subset_payload_bytes"] = sum(f["bytes"] for f in manifest["files"].values())
    write_json(OUT / "manifest.json", manifest)
    write_json(REF / "subset-receipt.json", manifest)
    print(json.dumps({"statistics": stats, "payload_bytes": manifest["subset_payload_bytes"],
                      "removed_counts": {s: len(v) for s, v in removed.items()}}, indent=2), flush=True)


if __name__ == "__main__":
    main()
