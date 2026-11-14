"""Width-only SwiGLU candidates; preserve approved production geometry at192."""
from .experimental_model import Config,Decoder,make_optimizer,parameter_hash,token_loss

def width_config(width):
    if width not in (192,288,384):raise ValueError(width)
    return Config(vocab_size=8192,layers=8,width=width,heads=width//32,
                  ff_width=8*width//3,ff_activation='swiglu',max_context=256)

def expected_parameters(width):
    return 8192*width+8*(12*width*width+2*width)+width
