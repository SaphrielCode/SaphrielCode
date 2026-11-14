import unittest
import torch
from lm.experimental_model import Config,Decoder,token_loss
from lm.efficiency import EfficientDecoder,RollingCache,training_loss

class EfficiencyTests(unittest.TestCase):
 def test_losses_gradients_and_update(self):
  torch.set_num_threads(2)
  cfg=Config(vocab_size=64,layers=2,width=32,heads=1,ff_width=80,ff_activation='swiglu',max_context=128)
  x=torch.tensor([[1,4,5,6,0,0],[1,7,8,0,0,0]])
  y=torch.tensor([[4,5,6,2,-100,-100],[7,8,2,-100,-100,-100]])
  base=Decoder(cfg);loss,_,n=token_loss(base(x),y);loss.backward()
  for attention,valid,trim in [('manual',False,False),('manual',True,False),('manual',False,True),('sdpa',False,False),('sdpa',True,True)]:
   with self.subTest(attention=attention,valid=valid,trim=trim):
    model=EfficientDecoder(cfg,attention=attention);model.load_state_dict(base.state_dict())
    actual,_,count=training_loss(model,x,y,valid,trim);actual.backward()
    torch.testing.assert_close(actual,loss,rtol=2e-6,atol=2e-6);self.assertEqual(count,n)
    for a,b in zip(base.parameters(),model.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=2e-4,atol=2e-6)
 def test_cache_rollover(self):
  cfg=Config(vocab_size=64,layers=2,width=32,heads=1,ff_width=80,ff_activation='swiglu',max_context=128)
  for attention in ('manual','sdpa'):
   model=EfficientDecoder(cfg,attention=attention).eval();cache=RollingCache(model,context=8);ids=torch.tensor([[1,4,5]])
   with torch.no_grad():
    for _ in range(10):
     full=model(ids[:,-8:])[:,-1];last=model(ids[:,-8:],last_only=True);cached=cache(ids)
     torch.testing.assert_close(full,last,rtol=2e-5,atol=2e-6);torch.testing.assert_close(full,cached,rtol=2e-5,atol=2e-6)
     ids=torch.cat((ids,full.argmax(-1).reshape(1,1)),1)
   self.assertGreater(cache.rebuilds,1)
 def test_factor_count_and_gradient(self):
  cfg=Config(ff_width=512,ff_activation='swiglu')
  m=EfficientDecoder(cfg,factor_dim=96)
  self.assertEqual(sum(p.numel() for p in m.parameters()),4347072)
  loss,_,_=training_loss(m,torch.tensor([[1,4,5]]),torch.tensor([[4,5,2]]),True);loss.backward()
  self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
if __name__=='__main__':unittest.main()
