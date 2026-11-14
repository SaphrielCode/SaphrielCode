"""Focused architecture correctness checks without changing baseline implementation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from lm.experimental_model import Decoder,Config,variant_config,parameter_hash,token_loss,make_optimizer
from lm.model import Decoder as Original

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 assert parameter_hash(Decoder(Config(),seed=42))==parameter_hash(Original(seed=42))
 for name in ('swiglu128','context256','deep12x160'):
  model=Decoder(variant_config(name),seed=42);assert sum(p.numel() for p in model.parameters())==(5001120 if name=='deep12x160' else 5115072);assert model.output_weight is model.embedding.weight
  assert parameter_hash(model)==parameter_hash(Decoder(variant_config(name),seed=42))
  x=torch.tensor([[1,14,29,35,0,0]]);z=x.clone();z[0,3:]=torch.tensor([47,50,61]);model.eval()
  with torch.no_grad():assert torch.equal(model(x)[:,:3],model(z)[:,:3])
  y=torch.tensor([[14,29,35,2,-100,-100]]);opt=make_optimizer(model);before=parameter_hash(model);loss,total,count=token_loss(model(x),y);assert count==4 and torch.isfinite(loss);loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters());torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();assert parameter_hash(model)!=before
  print(name,'PASS')
if __name__=='__main__':main()
