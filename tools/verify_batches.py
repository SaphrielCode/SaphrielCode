"""Independently check frozen window coverage, padding and document isolation."""
import json
import argparse
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from lm.batches import BATCH_DIR, read_batch, verify_batch_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-size',type=int,choices=(4096,8192),default=8192)
    parser.add_argument('--tokenizer-kind',choices=('bpe','unigram'),default='bpe')
    args = parser.parse_args()
    if args.tokenizer_kind=='unigram' and args.vocab_size!=8192:
        parser.error('Unigram comparison requires8192 entries')
    batch_dir = BATCH_DIR if args.vocab_size==8192 else ROOT/'data/TinyStoriesBatches128_BPE4096'
    source_dir = ROOT/'data/TinyStoriesSubset' if args.vocab_size==8192 else ROOT/'data/TinyStoriesSubset_BPE4096'
    if args.tokenizer_kind=='unigram':
        batch_dir=ROOT/'data/TinyStoriesBatches128_Unigram8192'
        source_dir=ROOT/'data/TinyStoriesSubset_Unigram8192'
    manifest = verify_batch_lock(batch_dir,source_dir)
    result = {"status": "PASS", "splits": {}}
    for split in ("train", "validation", "test"):
        source = np.fromfile(source_dir / f"{split}.u16", dtype="<u2")
        offsets = np.fromfile(source_dir / f"{split}.idx", dtype="<u8")
        rows = np.fromfile(batch_dir / f"{split}.rows.u32", dtype="<u4").reshape(-1, 3)
        xs = np.memmap(batch_dir / f"{split}.x.u16", dtype="<u2", mode="r", shape=(len(rows), 128))
        ys = np.memmap(batch_dir / f"{split}.y.u16", dtype="<u2", mode="r", shape=(len(rows), 128))
        next_target = np.zeros(len(offsets) - 1, dtype=np.uint64)
        valid_count = 0
        for row_index, (doc, start, valid) in enumerate(rows):
            doc, start, valid = int(doc), int(start), int(valid)
            if not 0 <= doc < len(next_target) or not 1 <= valid <= 128 or start != next_target[doc]:
                raise RuntimeError("Gap/overlap/invalid window descriptor")
            beginning, ending = int(offsets[doc]), int(offsets[doc + 1])
            if beginning + start + valid > ending:
                raise RuntimeError("Window crosses story boundary")
            x, y = xs[row_index], ys[row_index]
            if not np.array_equal(y[:valid], source[beginning + start:beginning + start + valid]):
                raise RuntimeError("Wrong next-token targets")
            preceding = 1 if start == 0 else source[beginning + start - 1]
            if x[0] != preceding or not np.array_equal(x[1:valid], y[:valid - 1]):
                raise RuntimeError("Wrong causal shift/BOS")
            if np.any(x[valid:] != 0) or np.any(y[valid:] != 0):
                raise RuntimeError("Wrong right padding")
            next_target[doc] += valid
            valid_count += valid
        if not np.array_equal(next_target, np.diff(offsets)) or valid_count != len(source):
            raise RuntimeError("Missing/duplicate targets")
        stats = manifest["statistics"][split]
        if stats["valid_target_tokens"] != valid_count or stats["windows"] != len(rows):
            raise RuntimeError("Incorrect manifest counts")
        if split == "train":
            order = np.fromfile(batch_dir / "train.order.u32", dtype="<u4")
            if not np.array_equal(np.sort(order), np.arange(len(rows))):
                raise RuntimeError("Training order is not a full permutation")
        partials = np.flatnonzero(rows[:, 2] < 128)
        if len(partials):
            inputs, targets, metadata = read_batch(split, [int(partials[0])],batch_dir)
            valid = metadata[0][2]
            if int(targets.ne(-100).sum()) != valid or not targets[0, valid:].eq(-100).all() or not inputs[0, valid:].eq(0).all():
                raise RuntimeError("Runtime loader failed to mask padding")
        result["splits"][split] = stats
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
