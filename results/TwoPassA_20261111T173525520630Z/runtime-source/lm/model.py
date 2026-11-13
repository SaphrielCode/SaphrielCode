"""Causal FP32 decoder matching the DataEval benchmark parameter specification."""
from dataclasses import asdict, dataclass
import hashlib
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab_size: int = 8192
    layers: int = 8
    width: int = 192
    heads: int = 6
    ff_width: int = 768
    max_context: int = 256
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.heads
        self.head_dim = cfg.width // cfg.heads
        self.attn_norm = RMSNorm(cfg.width, cfg.norm_eps)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width, bias=False)
        self.attn_out = nn.Linear(cfg.width, cfg.width, bias=False)
        self.ff_norm = RMSNorm(cfg.width, cfg.norm_eps)
        self.ff_in = nn.Linear(cfg.width, cfg.ff_width, bias=False)
        self.ff_out = nn.Linear(cfg.ff_width, cfg.width, bias=False)

    @staticmethod
    def rotate(x, cos, sin):
        pairs = x.reshape(*x.shape[:-1], -1, 2)
        a, b = pairs[..., 0], pairs[..., 1]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    def forward(self, x, cos, sin, mask):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).chunk(3, dim=-1)
        q, k, v = [t.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2) for t in (q, k, v)]
        q, k = self.rotate(q, cos, sin), self.rotate(k, cos, sin)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = scores.masked_fill(mask, float("-inf")).softmax(dim=-1)
        attended = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        x = x + self.attn_out(attended)
        return x + self.ff_out(F.gelu(self.ff_in(self.ff_norm(x)), approximate="none"))


class Decoder(nn.Module):
    def __init__(self, cfg=Config(), seed=42):
        super().__init__()
        if cfg.width % cfg.heads or (cfg.width // cfg.heads) % 2:
            raise ValueError("RoPE requires an even integer head dimension")
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.final_norm = RMSNorm(cfg.width, cfg.norm_eps)
        head_dim = cfg.width // cfg.heads
        inv = cfg.rope_theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        angle = torch.arange(cfg.max_context, dtype=torch.float32)[:, None] * inv[None, :]
        self.register_buffer("rope_cos", angle.cos(), persistent=False)
        self.register_buffer("rope_sin", angle.sin(), persistent=False)
        self.register_buffer("causal_mask", torch.ones(cfg.max_context, cfg.max_context, dtype=torch.bool).triu(1), persistent=False)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            nn.init.normal_(self.embedding.weight, mean=0, std=0.02, generator=generator)
            for block in self.blocks:
                for name in ("qkv", "attn_out", "ff_in", "ff_out"):
                    std = 0.02 / math.sqrt(2 * cfg.layers) if name in ("attn_out", "ff_out") else 0.02
                    nn.init.normal_(getattr(block, name).weight, mean=0, std=std, generator=generator)

    @property
    def output_weight(self):
        return self.embedding.weight

    def forward(self, ids):
        if ids.ndim != 2 or ids.shape[1] > self.config.max_context:
            raise ValueError("Expected [batch, context] within configured context limit")
        length = ids.shape[1]
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, self.rope_cos[:length], self.rope_sin[:length], self.causal_mask[:length, :length])
        return F.linear(self.final_norm(x), self.output_weight)


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode() + b"\0")
        digest.update(str(tuple(parameter.shape)).encode() + b"\0")
        digest.update(parameter.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def token_loss(logits, targets):
    valid = targets.ne(-100).sum()
    if valid.item() == 0:
        raise ValueError("No valid targets")
    losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100, reduction="none")
    return losses.sum() / valid, losses.detach().double().sum(), int(valid.item())


def make_optimizer(model):
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                            lr=0.0003, betas=(0.9, 0.95), eps=1e-8, foreach=False, fused=False)
