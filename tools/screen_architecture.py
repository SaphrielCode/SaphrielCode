"""Fresh-process 10-update warmup plus 20s CPU screen for one architecture."""
import argparse,json,os,sys,time,threading,traceback
from pathlib import Path
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lm.resource_guard import Guard,Policy

def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True)
 import psutil
 guard=Guard();process=psutil.Process();stop=threading.Event();done=threading.Event();samples=[];phase='preflight';started=time.perf_counter();report=dict(status='RUNNING',variant=a.variant,policy=asdict(Policy()))
 log=(a.out/'telemetry.jsonl').open('w')
 def sample():
  m=process.memory_info();available=psutil.virtual_memory().available;reason=guard.observe(available,[dict(pid=process.pid,resident_bytes=m.rss)])
  s=dict(seconds=time.perf_counter()-started,phase=phase,available_system_bytes=available,resident_bytes=m.rss,peak_resident_bytes=getattr(m,'peak_wset',m.rss),guard_stop=reason);samples.append(s);log.write(json.dumps(s)+'\n');log.flush()
  if reason:report['guard_stop']=s;stop.set()
 def check():
  if stop.is_set():raise RuntimeError('Resource guard stop')
 def watch():
  try:
   while not done.wait(1):sample()
  except Exception as e:report['monitor_error']=repr(e);stop.set()
 watcher=None
 try:
  for i in range(3):
   sample();check()
   if i<2:time.sleep(1)
  watcher=threading.Thread(target=watch,daemon=True);watcher.start();phase='imports'
  os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
  import numpy as np
  import torch
  from lm.experimental_model import Decoder,variant_config,parameter_hash,make_optimizer,token_loss
  from lm.batches import verify_batch_lock
  torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
  length=256 if a.variant=='context256' else 128;batchdir=ROOT/f'data/TinyStoriesBatches{length}_Unigram8192';source=ROOT/'data/TinyStoriesSubset_Unigram8192';manifest=verify_batch_lock(batchdir,source)
  x=np.fromfile(batchdir/'train.x.u16',dtype='<u2').reshape(-1,length);y=np.fromfile(batchdir/'train.y.u16',dtype='<u2').reshape(-1,length);order=np.fromfile(batchdir/'train.order.u32',dtype='<u4')
  model=Decoder(variant_config(a.variant),seed=42);opt=make_optimizer(model);initial=parameter_hash(model);cursor=0;steps=[];warmtokens=0
  report.update(model_config=asdict(model.config),unique_parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=initial,tokenizer_sha256=manifest['tokenizer_sha256'],context=length,batch=4,threads=8)
  def step():
   nonlocal cursor
   check();at=time.perf_counter();idx=order[cursor:cursor+4];cursor+=len(idx);assert len(idx)>0
   tx=torch.from_numpy(x[idx].astype(np.int64));ty=torch.from_numpy(y[idx].astype(np.int64));ty.masked_fill_(ty.eq(0),-100)
   opt.zero_grad(set_to_none=True);loss,total,n=token_loss(model(tx),ty);assert torch.isfinite(loss);check();loss.backward();assert all(p.grad is not None for p in model.parameters());norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);check();opt.step()
   return dict(seconds=time.perf_counter()-at,valid_targets=n,loss_sum=float(total),gradient_norm=float(norm))
  phase='warmup';at=time.perf_counter()
  for i in range(10):warmtokens+=step()['valid_targets']
  report.update(warmup_updates=10,warmup_valid_targets=warmtokens,warmup_seconds=time.perf_counter()-at)
  phase='timed';at=time.perf_counter()
  while time.perf_counter()-at<20:steps.append(step())
  elapsed=time.perf_counter()-at;phase='verification';check();final=parameter_hash(model);assert final!=initial;assert all(torch.isfinite(p).all() for p in model.parameters())
  report.update(status='PASS',timed_updates=len(steps),timed_seconds=elapsed,valid_targets=sum(s['valid_targets'] for s in steps),valid_targets_per_second=sum(s['valid_targets'] for s in steps)/elapsed,final_parameter_sha256=final,weights_changed=True,finite_losses_and_gradient_norms=True)
  (a.out/'steps.json').write_text(json.dumps(steps))
 except Exception as e:report.update(status='STOPPED' if stop.is_set() else 'FAIL',error=repr(e),traceback=traceback.format_exc())
 finally:
  done.set()
  if watcher:watcher.join(3)
  log.close();report.update(peak_process_resident_bytes=max((s['peak_resident_bytes'] for s in samples),default=0),minimum_available_system_bytes=min((s['available_system_bytes'] for s in samples),default=0));(a.out/'screen.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
 return 0 if report['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
