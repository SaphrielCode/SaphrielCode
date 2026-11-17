"""Regression checks for continuation alignment and visibility."""
from pathlib import Path
import unittest

import torch
from tokenizers import Tokenizer
from lm.efficiency import EfficientDecoder, training_loss
from lm.experimental_model import Config
from lm.stage_one import encode_example, prompt_text, response_text


class StageOneChecks(unittest.TestCase):
    def test_response_targets_include_first_token_and_eos(self):
        tokenizer = Tokenizer.from_file(str(Path(__file__).resolve().parents[1] / 'model_assets/tokenizer.json'))
        example = dict(scene='Molly put the key in his coat.', user='George reached inside.',
                       response='He found the brass key.')
        x, y = encode_example(tokenizer, example, 128)
        prompt = tokenizer.encode(prompt_text(example['scene'], example['user'])).ids
        reply = tokenizer.encode(response_text(example['response'])).ids
        self.assertEqual(x[:len(prompt)+1], [1] + prompt)
        self.assertEqual(y[:len(prompt)], [-100] * len(prompt))
        self.assertEqual([token for token in y if token != -100], reply + [2])
        with self.assertRaises(ValueError):
            encode_example(tokenizer, example, 8)

    def test_future_and_right_padding_cannot_change_valid_scores(self):
        torch.manual_seed(1)
        model = EfficientDecoder(Config(vocab_size=32, layers=2, width=16, heads=2,
                                        ff_width=24, ff_activation='swiglu', max_context=16))
        model.eval()
        original = torch.tensor([[1, 5, 6, 7, 0, 0]])
        changed = torch.tensor([[1, 5, 6, 12, 14, 15]])
        with torch.no_grad():
            torch.testing.assert_close(model(original)[:, :3], model(changed)[:, :3])
        targets = torch.tensor([[-100, -100, 7, 2, -100, -100]])
        dense = training_loss(model, original, targets)
        selected = training_loss(model, original, targets, valid_only=True, trim=True)
        torch.testing.assert_close(dense[0], selected[0])
        self.assertEqual(selected[2], 2)
        self.assertIs(model.output_weight, model.embedding.weight)


if __name__ == '__main__':
    unittest.main()
