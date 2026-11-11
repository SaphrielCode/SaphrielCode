"""Verify frozen subset integrity and token boundaries without network access."""
from array import array
import hashlib
import json
from pathlib import Path
import struct
import sys
import unicodedata

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/TinyStoriesSubset"


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name, entry in manifest["files"].items():
        path = OUT / name
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise RuntimeError(f"Frozen file changed: {name}")
    for name, entry in manifest["reference_files"].items():
        if digest(ROOT / "corpus-references/TinyStories" / name) != entry["sha256"]:
            raise RuntimeError(f"Reference changed: {name}")
    tokenizer = Tokenizer.from_file(str(OUT / "tokenizer.json"))
    if tokenizer.get_vocab_size() != 8192:
        raise RuntimeError("Vocabulary size mismatch")
    seen, totals = set(), {}
    source_indices = {}
    for split in ("test", "validation", "train"):
        tokens = array("H")
        with (OUT / (split + ".u16")).open("rb") as source:
            tokens.frombytes(source.read())
        if sys.byteorder != "little":
            tokens.byteswap()
        offsets_data = (OUT / (split + ".idx")).read_bytes()
        offsets = [item[0] for item in struct.iter_unpack("<Q", offsets_data)]
        if offsets[0] != 0 or offsets[-1] != len(tokens):
            raise RuntimeError("Invalid token-file offsets")
        count, body_count = 0, 0
        with (OUT / (split + ".jsonl")).open(encoding="utf-8") as source:
            source_indices[split] = set()
            for index, line in enumerate(source):
                row = json.loads(line)
                expected_prefix = "data/train-" if split == "train" else "data/validation-"
                if not row["source_file"].startswith(expected_prefix) or row["source_row"] in source_indices[split]:
                    raise RuntimeError("Source split/index violation")
                source_indices[split].add(row["source_row"])
                text = row["text"]
                value = hashlib.sha256(text.encode()).hexdigest()
                if value != row["text_sha256"] or value in seen:
                    raise RuntimeError("Duplicate/changed story")
                seen.add(value)
                ids = list(tokens[offsets[index]:offsets[index + 1]])
                if not ids or ids[-1] != 2 or any(i < 4 or i >= 8192 for i in ids[:-1]):
                    raise RuntimeError("Invalid story boundary or token ID")
                if tokenizer.decode(ids[:-1]) != text or tokenizer.encode(text, add_special_tokens=False).ids != ids[:-1]:
                    raise RuntimeError("Text/token mismatch")
                count += 1
                body_count += len(ids) - 1
        if len(offsets) != count + 1:
            raise RuntimeError("Unexpected document offset count")
        actual = manifest["statistics"][split]
        if count != actual["stories"] or body_count != actual["body_tokens"] or len(tokens) != actual["tokens_including_eos"]:
            raise RuntimeError("Reported count mismatch")
        totals[split] = {"stories": count, "body_tokens": body_count, "tokens_including_eos": len(tokens)}
    if source_indices["validation"] & source_indices["test"]:
        raise RuntimeError("Validation/test source-row overlap")
    for text in ["New symbols: 中文 🐈", "e\u0301", "\n\t\x00"]:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if 3 in ids or tokenizer.decode(ids) != unicodedata.normalize("NFC", text):
            raise RuntimeError("Byte coverage failure")
    print(json.dumps({"status": "PASS", "zero_cross_split_exact_overlap": True, "splits": totals}, indent=2))


if __name__ == "__main__":
    main()
