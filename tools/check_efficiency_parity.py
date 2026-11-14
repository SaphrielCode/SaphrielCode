"""Real frozen-data, trained-checkpoint parity including tied gradients and AdamW update."""
import sys,json,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 import torch,numpy as np
 from lm.experimental_model import Decoder,Config,token_loss,make_optimizer
 from lm.efficiency import EfficientDecoder,training_loss
 from lm.batches import read_batch,digest
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 s=json.loads((ROOT/'WorkingSelection.json').read_text());path=ROOT/s['reference_checkpoint'];assert digest(path)==s['reference_checkpoint_sha256'];saved=torch.load(path,weights_only=True,mmap=True);cfg=Config(**saved['config']);bd=ROOT/s['batch_directory'];meta=np.fromfile(bd/'train.rows.u32',dtype='<u4').reshape(-1,3);partial=np.flatnonzero(meta[:,2]<100)[:4];full=np.flatnonzero(meta[:,2]==128)[:4];results=[]
 for optimizer_mode in ('reset_at_trained_weights','restored_training_state'):
  for name,indices in [('mixed',[int(full[0]),*map(int,partial[:3])]),('trimmed',list(map(int,partial)))]:
   x,y,_=read_batch('train',indices,bd);base=Decoder(cfg);base.load_state_dict(saved['model']);opt=make_optimizer(base)
   if optimizer_mode=='restored_training_state':opt.load_state_dict(copy.deepcopy(saved['optimizer']))
   loss,_,n=token_loss(base(x),y);loss.backward();grads={k:p.grad.clone() for k,p in base.named_parameters()};torch.nn.utils.clip_grad_norm_(base.parameters(),1.,error_if_nonfinite=True,foreach=False);opt.step();updated={k:p.detach().clone() for k,p in base.named_parameters()};del base,opt
   for label,att,valid,trim in [('valid','manual',True,False),('trim','manual',False,True),('sdpa','sdpa',False,False),('combined','sdpa',True,True)]:
    MONITOR.check()
    model=EfficientDecoder(cfg,attention=att);model.load_state_dict(saved['model']);opt=make_optimizer(model)
    if optimizer_mode=='restored_training_state':opt.load_state_dict(copy.deepcopy(saved['optimizer']))
    actual,_,count=training_loss(model,x,y,valid,trim);actual.backward();torch.testing.assert_close(actual,loss,rtol=2e-5,atol=2e-6);assert count==n;maxgrad=0.
    for k,p in model.named_parameters():
     torch.testing.assert_close(p.grad,grads[k],rtol=1e-3,atol=3e-6);maxgrad=max(maxgrad,float((p.grad-grads[k]).abs().max()))
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);opt.step();maxupdate=max(float((p.detach()-updated[k]).abs().max()) for k,p in model.named_parameters())
    if optimizer_mode=='restored_training_state':assert maxupdate<3e-6
    results.append(dict(optimizer_mode=optimizer_mode,batch=name,variant=label,loss_difference=float((actual-loss).detach()),gradient_max_abs_difference=maxgrad,post_update_parameter_max_abs_difference=maxupdate,valid_targets=n));del model,opt
 out=ROOT/json.loads((ROOT/'results/Efficiency_Current.json').read_text())['directory']
 (out/'trained-parity.json').write_text(json.dumps(dict(status='PASS',checkpoint_sha256=s['reference_checkpoint_sha256'],note='Loss/gradients checked in both modes. Resetting AdamW at already-trained weights can amplify tiny gradient differences near zero; those weight differences are reported, not required to be bitwise equal. Continued-training parameter check uses independent copies of saved optimizer moments and max absolute3e-6.',comparisons=results),indent=2));print('Trained parity PASS')
if __name__=='__main__':
 from lm.monitor import Monitor
 out=ROOT/json.loads((ROOT/'results/Efficiency_Current.json').read_text())['directory']/'trained-parity-resources'
 out=out.with_name('trained-parity-resources-v2');out.mkdir(exist_ok=False);MONITOR=Monitor(out)
 try:
  MONITOR.start();main()
 finally:
  (out/'resources.json').write_text(json.dumps(MONITOR.close(),indent=2))
