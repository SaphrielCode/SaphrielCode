"""Muon for hidden matrices, AdamW for tied embeddings and normalization."""
import math

import torch

from lm.experimental_model import make_optimizer as make_adamw


def orthogonalize_fp32(update, steps, epsilon):
    """PyTorch Muon's quintic Newton-Schulz iteration, evaluated in FP32."""
    matrix = update.clone()
    transposed = matrix.shape[0] > matrix.shape[1]
    if transposed:
        matrix = matrix.T
    matrix.div_(matrix.norm().clamp(min=epsilon))
    for _ in range(steps):
        gram = matrix @ matrix.T
        polynomial = torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
        matrix = torch.addmm(matrix, polynomial, matrix, beta=3.4445)
    return matrix.T if transposed else matrix


class MuonFP32(torch.optim.Optimizer):
    """CPU FP32 Muon using PyTorch's momentum and match_rms_adamw equations."""

    def __init__(self, params, lr=1e-4):
        if not 0 <= lr:
            raise ValueError('Learning rate must be nonnegative')
        super().__init__(params, dict(lr=lr, weight_decay=.1, momentum=.95,
                                    nesterov=True, ns_steps=5, eps=1e-7))
        for group in self.param_groups:
            for parameter in group['params']:
                if parameter.ndim != 2 or parameter.dtype != torch.float32 or parameter.device.type != 'cpu':
                    raise ValueError('MuonFP32 requires CPU FP32 matrix parameters')

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group['params']:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise ValueError('MuonFP32 requires dense gradients')
                state = self.state[parameter]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(parameter)
                momentum = state['momentum_buffer']
                momentum.lerp_(gradient, 1 - group['momentum'])
                update = gradient.lerp(momentum, group['momentum']) if group['nesterov'] else momentum
                update = orthogonalize_fp32(update, group['ns_steps'], group['eps'])
                rate = group['lr'] * .2 * math.sqrt(max(parameter.shape))
                parameter.mul_(1 - group['lr'] * group['weight_decay'])
                parameter.add_(update, alpha=-rate)


class MuonAdamW:
    format = 'muon-adamw-fp32-v1'

    def __init__(self, model, lr=1e-4):
        matrices, embeddings, vectors = [], [], []
        matrix_names, embedding_names, vector_names = [], [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith('blocks.') and parameter.ndim == 2:
                matrices.append(parameter)
                matrix_names.append(name)
            elif parameter.ndim >= 2:
                embeddings.append(parameter)
                embedding_names.append(name)
            else:
                vectors.append(parameter)
                vector_names.append(name)
        self.parameter_names = dict(muon=matrix_names, adamw_decay=embedding_names,
                                    adamw_no_decay=vector_names)
        self.muon = MuonFP32([dict(params=matrices, param_names=matrix_names)], lr=lr)
        self.adamw = torch.optim.AdamW(
            [dict(params=embeddings, param_names=embedding_names, weight_decay=.1),
             dict(params=vectors, param_names=vector_names, weight_decay=0.)],
            lr=lr, betas=(.9, .95), eps=1e-8, foreach=False, fused=False)
        self.param_groups = self.muon.param_groups + self.adamw.param_groups

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adamw.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adamw.step()

    def state_dict(self):
        return dict(format=self.format, parameter_names=self.parameter_names,
                    muon=self.muon.state_dict(), adamw=self.adamw.state_dict())

    def load_state_dict(self, state):
        if state.get('format') != self.format:
            raise ValueError('Expected a Muon+AdamW optimizer checkpoint')
        if state.get('parameter_names') != self.parameter_names:
            raise ValueError('Optimizer parameter assignment differs from checkpoint')
        self.muon.load_state_dict(state['muon'])
        self.adamw.load_state_dict(state['adamw'])
        self.param_groups = self.muon.param_groups + self.adamw.param_groups


def make_optimizer(model, kind='muon-adamw'):
    if kind == 'muon-adamw':
        return MuonAdamW(model)
    if kind == 'adamw':
        return make_adamw(model)
    raise ValueError(f'Unknown optimizer: {kind}')


def checkpoint_optimizer_kind(state):
    if state.get('format') == MuonAdamW.format:
        return 'muon-adamw'
    if 'state' in state and 'param_groups' in state:
        return 'adamw'
    raise ValueError('Unrecognized optimizer checkpoint format')
