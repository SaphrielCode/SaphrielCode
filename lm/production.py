"""Approved Com Device production architecture: SwiGLU128 + frozen8k unigram.

Historical GELU experiments remain in lm.model. This entry point deliberately
matches the tested swiglu128 checkpoint schema and initialization.
"""
from .experimental_model import Config as ArchitectureConfig, Decoder as ArchitectureDecoder
from .experimental_model import make_optimizer, parameter_hash, token_loss


def production_config():
    # max_context256 preserves the tested checkpoint schema; working window128.
    return ArchitectureConfig(vocab_size=8192,layers=8,width=192,heads=6,
                              ff_width=512,ff_activation='swiglu',max_context=256)


class Decoder(ArchitectureDecoder):
    def __init__(self, seed=42):
        super().__init__(production_config(),seed=seed)
