"""Focused causal-mask, padding, loss-aggregation and boundary checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import torch
from lm.model import Config, Decoder, parameter_hash, token_loss
from prepare_batches import window_story


class DecoderChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = Config(vocab_size=64, layers=2, width=32, heads=2, ff_width=128, max_context=16)

    def test_causal_mask_prevents_future_and_padding_leakage(self):
        model = Decoder(self.cfg).eval()
        first = torch.tensor([[1, 4, 5, 6, 0, 0, 0, 0]])
        changed = torch.tensor([[1, 4, 5, 6, 9, 10, 11, 12]])
        with torch.no_grad():
            a, b = model(first), model(changed)
        torch.testing.assert_close(a[:, :4], b[:, :4], rtol=0, atol=1e-6)
        self.assertFalse(torch.allclose(a[:, 4:], b[:, 4:]))

    def test_initialization_and_embedding_sharing(self):
        a, b = Decoder(self.cfg, seed=42), Decoder(self.cfg, seed=42)
        self.assertEqual(parameter_hash(a), parameter_hash(b))
        self.assertIs(a.output_weight, a.embedding.weight)
        expected = 64 * 32 + 2 * (12 * 32**2 + 2 * 32) + 32
        self.assertEqual(sum(p.numel() for p in a.parameters()), expected)
        self.assertTrue(all(torch.equal(m.weight, torch.ones_like(m.weight)) for m in [a.final_norm, a.blocks[0].attn_norm]))

    def test_padding_and_token_weighted_loss(self):
        logits = torch.tensor([[[2., 0., -1.], [0., 1., 0.]], [[0., 0., 3.], [100., -100., 0.]]])
        targets = torch.tensor([[0, 1], [2, -100]])
        mean, total, count = token_loss(logits, targets)
        expected = -torch.log_softmax(logits, -1)[0, 0, 0] - torch.log_softmax(logits, -1)[0, 1, 1] - torch.log_softmax(logits, -1)[1, 0, 2]
        self.assertEqual(count, 3)
        self.assertAlmostEqual(float(total), float(expected), places=6)
        self.assertAlmostEqual(float(mean), float(expected / 3), places=6)
        _, first_sum, first_n = token_loss(logits[:1], targets[:1])
        _, second_sum, second_n = token_loss(logits[1:], targets[1:])
        self.assertAlmostEqual(float((first_sum + second_sum) / (first_n + second_n)), float(mean), places=6)
        with self.assertRaises(ValueError):
            token_loss(logits, torch.full_like(targets, -100))

    def test_story_windows_cover_all_targets_without_crossing_documents(self):
        story = [7, 8, 9, 10, 11, 12, 13, 14, 2]
        rows = list(window_story(story, length=4))
        self.assertEqual([v for _, y, _, n in rows for v in y[:n]], story)
        self.assertEqual(rows[0][0], [1, 7, 8, 9])
        self.assertEqual(rows[1][0], [10, 11, 12, 13])
        self.assertEqual(rows[2], ([14, 0, 0, 0], [2, 0, 0, 0], 8, 1))
        self.assertEqual(list(window_story([20, 2], length=4))[0][0], [1, 20, 0, 0])


if __name__ == "__main__":
    unittest.main()
