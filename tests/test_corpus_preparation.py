"""Focused checks for deterministic selection and held-out protection."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from prepare_tinystories import choose_indices, deduplicate, normalize


class PreparationChecks(unittest.TestCase):
    def test_sampling_is_repeatable_without_replacement(self):
        first = choose_indices(10000, 2000, 1730)
        self.assertEqual(first, choose_indices(10000, 2000, 1730))
        self.assertEqual(len(set(first)), 2000)
        self.assertFalse(set(first[:1000]) & set(first[1000:]))
        self.assertNotEqual(first, choose_indices(10000, 2000, 1731))

    def test_deduplication_protects_heldout_and_handles_normalized_matches(self):
        def row(index, text):
            return {"source_row": index, "text": text}
        splits = {"test": [row(0, " café\r\n")],
                  "validation": [row(1, "cafe\u0301"), row(2, "Held out")],
                  "train": [row(3, "café"), row(4, "Held out"), row(5, "Training"), row(6, "Training")]}
        kept, removed = deduplicate(splits)
        self.assertEqual([r["text"] for r in kept["test"]], ["café"])
        self.assertEqual([r["text"] for r in kept["validation"]], ["Held out"])
        self.assertEqual([r["text"] for r in kept["train"]], ["Training"])
        self.assertEqual({k: len(v) for k, v in removed.items()}, {"test": 0, "validation": 1, "train": 3})

    def test_normalization_preserves_internal_whitespace_and_case(self):
        self.assertEqual(normalize(" \r\nA  b\r\nC\t \r"), "A  b\nC")
        self.assertNotEqual(normalize("A b"), normalize("a b"))


if __name__ == "__main__":
    unittest.main()
