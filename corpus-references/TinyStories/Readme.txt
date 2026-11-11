Frozen TinyStories benchmark corpus

Dataset/reference: https://huggingface.co/datasets/roneneldan/TinyStories
Revision: f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
License: CDLA-Sharing-1.0. Full text and original dataset card are preserved here.
Attribution: TinyStories by Ronen Eldan and Yuanzhi Li.
Derived preparation: deterministic row selection, NFC/newline/outer-whitespace normalization,
exact deduplication, split assignment and a newly trained tokenizer. See subset-receipt.json.

Local frozen corpus: data/TinyStoriesSubset/.
Use tools/verify_tinystories.py before benchmark runs. No network is required to verify/read it.
The tokenizer was trained only on retained training stories, never validation or final test.
Large corpus files are excluded by .gitignore; this folder preserves metadata and references.

Reproduction tools:
  python -m venv .venv-data
  .venv-data\Scripts\python.exe -m pip install -r requirements-data.lock.txt
  .venv-data\Scripts\python.exe tools/prepare_tinystories.py
The preparation command refuses to replace an existing frozen output. Do not delete it just to
rerun preparation during candidate comparisons. Dependencies and tokenizer versions are pinned;
the saved tokenizer/subset hashes, not an assumption about retraining determinism, define the benchmark.

Sources are streamed to a temporary disk cache and read with 2,048-row Parquet batches.
Only selected stories are retained for tokenizer training. The full source corpus is not held in RAM.
Temporary source-cache files were removed after verification; their published hashes remain recorded.

No model training or benchmark has been run. This corpus evaluates a narrow synthetic-story domain,
not broad instruction following or general language capability. Near duplicates remain a limitation.
