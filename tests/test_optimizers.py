"""Hybrid optimizer assignment and exact checkpoint continuation regressions."""
import copy
import io
import unittest
from unittest.mock import patch

import torch

from lm.efficiency import EfficientDecoder
from lm.experimental_model import Config
from lm.optimizers import MuonAdamW, MuonFP32, checkpoint_optimizer_kind, make_optimizer


def model():
    return EfficientDecoder(Config(vocab_size=32, layers=2, width=16, heads=2,
                                   ff_width=24, ff_activation='swiglu', max_context=16))


def update(network, optimizer):
    optimizer.zero_grad(set_to_none=True)
    logits = network(torch.tensor([[1, 5, 6, 7]]))
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([5, 6, 7, 2]))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(network.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()


class OptimizerChecks(unittest.TestCase):
    def test_fp32_update_matches_pytorch_muon_equations(self):
        def reference(gradient, coefficients, steps, epsilon):
            matrix = gradient.clone()
            tall = matrix.shape[0] > matrix.shape[1]
            if tall:
                matrix = matrix.T
            matrix = matrix / matrix.norm().clamp(min=epsilon)
            a, b, c = coefficients
            for _ in range(steps):
                gram = matrix @ matrix.T
                matrix = a * matrix + (b * gram + c * (gram @ gram)) @ matrix
            return matrix.T if tall else matrix

        for shape in ((5, 9), (9, 5)):
            actual = torch.nn.Parameter(torch.randn(shape))
            expected = torch.nn.Parameter(actual.detach().clone())
            optimizer = MuonFP32([actual])
            native = torch.optim.Muon([expected], lr=1e-4, weight_decay=.1,
                                      adjust_lr_fn='match_rms_adamw')
            with patch('torch.optim._muon._zeropower_via_newtonschulz', reference):
                for gradient in (torch.zeros(shape), torch.randn(shape), torch.randn(shape)):
                    actual.grad = gradient.clone()
                    expected.grad = gradient.clone()
                    optimizer.step()
                    native.step()
                    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)

    def test_every_weight_is_assigned_once(self):
        network = model()
        optimizer = make_optimizer(network)
        self.assertIsInstance(optimizer, MuonAdamW)
        ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(parameter) for parameter in network.parameters()})
        self.assertNotIn(id(network.embedding.weight), {id(parameter) for parameter in optimizer.muon.param_groups[0]['params']})
        self.assertEqual(len(optimizer.parameter_names['muon']), 10)
        self.assertEqual(optimizer.parameter_names['adamw_decay'], ['embedding.weight'])
        self.assertTrue(all(parameter.ndim == 1 for parameter in optimizer.adamw.param_groups[1]['params']))
        self.assertEqual(optimizer.adamw.param_groups[1]['weight_decay'], 0.)

    def test_updates_both_groups_and_resumes_exactly(self):
        network = model()
        before = copy.deepcopy(network.state_dict())
        optimizer = make_optimizer(network)
        update(network, optimizer)
        self.assertFalse(torch.equal(before['embedding.weight'], network.embedding.weight))
        self.assertFalse(torch.equal(before['blocks.0.qkv.weight'], network.blocks[0].qkv.weight))
        self.assertTrue(all(torch.isfinite(parameter).all() for parameter in network.parameters()))
        stream = io.BytesIO()
        torch.save(dict(model=network.state_dict(), optimizer=optimizer.state_dict()), stream)
        stream.seek(0)
        saved = torch.load(stream, weights_only=True)
        restored = model()
        restored.load_state_dict(saved['model'])
        resumed = make_optimizer(restored)
        resumed.load_state_dict(saved['optimizer'])
        for instance in (optimizer, resumed):
            for group in instance.param_groups:
                group['lr'] = 2e-5
        self.assertTrue(all(group['lr'] == 2e-5 for group in resumed.muon.param_groups + resumed.adamw.param_groups))
        update(network, optimizer)
        update(restored, resumed)
        for name, parameter in network.named_parameters():
            torch.testing.assert_close(parameter, dict(restored.named_parameters())[name], rtol=0, atol=0)
        self.assertEqual(checkpoint_optimizer_kind(saved['optimizer']), 'muon-adamw')

    def test_legacy_adamw_remains_loadable(self):
        network = model()
        optimizer = make_optimizer(network, 'adamw')
        update(network, optimizer)
        state = copy.deepcopy(optimizer.state_dict())
        self.assertEqual(checkpoint_optimizer_kind(state), 'adamw')
        make_optimizer(network, 'adamw').load_state_dict(state)
        with self.assertRaises(ValueError):
            make_optimizer(network).load_state_dict(state)
        with self.assertRaises(ValueError):
            checkpoint_optimizer_kind({})

    def test_changed_parameter_assignment_is_rejected(self):
        optimizer = make_optimizer(model())
        state = copy.deepcopy(optimizer.state_dict())
        state['parameter_names']['muon'][0] = 'different.weight'
        with self.assertRaises(ValueError):
            optimizer.load_state_dict(state)


if __name__ == '__main__':
    unittest.main()
