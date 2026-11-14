"""Fresh width comparison at fixed active-training time, with exact data cursor."""
from pathlib import Path
from dataclasses import asdict
import argparse,copy,json,os,sys,time,traceback,hashlib,random,math
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lm.monitor import Monitor
from cpu_screen_worker import save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--width',type=int,choices=(192,288,384),required=True);p.add_argument('--minutes',type=float,default=90);p.add_argument('--out',type=Path,required=True);p.add_argument('--resume',type=Path);p.add_argument('--marks',default='15,30,60,90');a=p.parse_args()
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);m=Monitor(out);started=time.perf_counter();model=opt=None;r=dict(status='RUNNING',width=a.width,budget_minutes=a.minutes,training_seconds=0.,validation_seconds=0.,checkpoint_seconds=0.,completed_updates=0,valid_targets=0,completed_passes=0,next_row=0,validation={},checkpoints=[],extra_warmup_updates=0,final_test_evaluated=False,initialization_seed=42,order_seed=43,batch=4,context=128,threads=8,interop_threads=1)
 sourcefiles=('lm/width_model.py','lm/experimental_model.py','lm/monitor.py','lm/resource_guard.py','tools/train_width.py')
 r['source_hashes']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sourcefiles}
 try:
  m.start();m.phase='imports_and_integrity';os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
  import numpy as np
  import torch
  from lm.width_model import Config,Decoder,width_config,expected_parameters,parameter_hash,make_optimizer,token_loss
  from lm.batches import digest,verify_batch_lock
  torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42);random.seed(42);np.random.seed(42)
  batchdir=ROOT/'data/TinyStoriesBatches128_Unigram8192';source=ROOT/'data/TinyStoriesSubset_Unigram8192';manifest=verify_batch_lock(batchdir,source)
  r.update(tokenizer_sha256=manifest['tokenizer_sha256'],batch_manifest_sha256=digest(batchdir/'manifest.json'),targets_per_pass=4388821,validation_text_bytes=884630)
  tx=np.fromfile(batchdir/'train.x.u16',dtype='<u2').reshape(-1,128);ty=np.fromfile(batchdir/'train.y.u16',dtype='<u2').reshape(-1,128);order=np.fromfile(batchdir/'train.order.u32',dtype='<u4');vx=np.fromfile(batchdir/'validation.x.u16',dtype='<u2').reshape(-1,128);vy=np.fromfile(batchdir/'validation.y.u16',dtype='<u2').reshape(-1,128)
  assert len(order)==45137 and np.array_equal(np.sort(order),np.arange(len(order))) and np.count_nonzero(ty)==4388821 and np.count_nonzero(vy)==216150
  cfg=width_config(a.width);model=Decoder(cfg,seed=42);opt=make_optimizer(model);r.update(config=asdict(cfg),unique_parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=parameter_hash(model));assert r['unique_parameters']==expected_parameters(a.width);assert model.output_weight is model.embedding.weight
  if a.width==192:assert r['initial_parameter_sha256']==json.loads((ROOT/'WorkingSelection.json').read_text())['initial_parameter_sha256']
  if a.resume:
   restored=torch.load(a.resume,map_location='cpu',weights_only=True);assert restored['config']==asdict(cfg) and restored['report']['width']==a.width;assert restored['report']['tokenizer_sha256']==r['tokenizer_sha256'] and restored['report']['batch_manifest_sha256']==r['batch_manifest_sha256'];model.load_state_dict(restored['model']);opt.load_state_dict(restored['optimizer']);assert parameter_hash(model)==restored['parameter_sha256'];r=restored['report'];r.update(status='RUNNING',resume_source=str(a.resume),budget_minutes=a.minutes);assert all(int(s['step'])==r['completed_updates'] for s in opt.state.values());assert r['valid_targets']==r['completed_passes']*4388821+int(np.count_nonzero(ty[order[:r['next_row']]]));del restored
  def batch(x,y,idx):
   ix=torch.from_numpy(x[idx].astype(np.int64));iy=torch.from_numpy(y[idx].astype(np.int64));iy.masked_fill_(iy.eq(0),-100);return ix,iy
  def evaluate(label,requested_minutes):
   m.phase='validation_'+label;model.eval();at=time.perf_counter();total=0.;count=0
   with torch.no_grad():
    for start in range(0,len(vx),4):
     m.check();x,y=batch(vx,vy,slice(start,start+4));_,summed,n=token_loss(model(x),y);total+=float(summed);count+=n
   elapsed=time.perf_counter()-at;assert count==216150;r['validation_seconds']+=elapsed;r['validation'][label]=dict(loss_sum=total,valid_targets=count,loss_nats_per_target=total/count,bits_per_utf8_text_byte_including_eos_cost=total/884630/math.log(2),seconds=elapsed,requested_training_minutes=requested_minutes,actual_training_seconds=r['training_seconds'],completed_updates=r['completed_updates'],valid_targets_processed=r['valid_targets'],completed_passes=r['completed_passes'],next_row=r['next_row']);print(json.dumps(dict(validation=label,**r['validation'][label])),flush=True);save_json(out/'run.json',r)
  def checkpoint(name,reload=False):
   nonlocal model,opt
   m.phase='checkpoint_'+name;at=time.perf_counter();h=parameter_hash(model);path=out/(name+'.pt');tmp=path.with_suffix('.pt.tmp');torch.save(dict(config=asdict(cfg),model=model.state_dict(),optimizer=opt.state_dict(),parameter_sha256=h,report=copy.deepcopy(r)),tmp);tmp.replace(path)
   if reload:
    del model,opt
    model=opt=None
    saved=torch.load(path,map_location='cpu',weights_only=True);model=Decoder(cfg,seed=42);model.load_state_dict(saved['model']);opt=make_optimizer(model);opt.load_state_dict(saved['optimizer']);assert parameter_hash(model)==h;assert all(int(s['step'])==r['completed_updates'] for s in opt.state.values());del saved
   elapsed=time.perf_counter()-at;r['checkpoint_seconds']+=elapsed;entry=dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path),parameter_sha256=h,training_seconds=r['training_seconds'],completed_updates=r['completed_updates'],valid_targets=r['valid_targets'],seconds=elapsed,reload_verified=reload)
   if name=='recovery':r['recovery_checkpoint']=entry
   else:r['checkpoints'].append(entry)
   save_json(out/'run.json',r)
  if not a.resume:evaluate('initialization',0)
  marks=sorted(set([float(v) for v in a.marks.split(',') if float(v)<=a.minutes]+[a.minutes]))
  with (out/'steps.jsonl').open('w') as log:
   for mark in marks:
    label='minute'+str(int(mark))
    if label in r['validation']:continue
    base=r['training_seconds'];at=time.perf_counter();m.phase='training';model.train()
    while base+time.perf_counter()-at<mark*60:
     m.check();step_start=time.perf_counter();cursor=r['next_row'];idx=order[cursor:cursor+4];x,y=batch(tx,ty,idx);opt.zero_grad(set_to_none=True);logits=model(x);loss,summed,n=token_loss(logits,y);del logits
     if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
     m.check();loss.backward()
     if not all(p.grad is not None for p in model.parameters()):raise RuntimeError('Missing gradient')
     norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);m.check();opt.step();r['completed_updates']+=1;r['valid_targets']+=n;r['next_row']+=len(idx)
     if r['next_row']==len(order):r['next_row']=0;r['completed_passes']+=1
     r['training_seconds']=base+time.perf_counter()-at;log.write(json.dumps(dict(update=r['completed_updates'],valid_targets=n,loss_sum=float(summed),gradient_norm=float(norm),training_seconds=r['training_seconds'],seconds=time.perf_counter()-step_start,next_row=r['next_row'],completed_passes=r['completed_passes']))+'\n');del loss,summed,norm
     if r['completed_updates']%1000==0:
      log.flush();r['training_seconds']=base+time.perf_counter()-at;checkpoint('recovery');base=r['training_seconds'];at=time.perf_counter();m.phase='training'
     if r['completed_updates']%100==0:save_json(out/'run.json',r);log.flush();print(json.dumps({k:r[k] for k in ('width','training_seconds','completed_updates','valid_targets','completed_passes')}),flush=True)
    r['training_seconds']=base+time.perf_counter()-at
    assert r['valid_targets']==r['completed_passes']*4388821+int(np.count_nonzero(ty[order[:r['next_row']]]))
    evaluate(label,mark);checkpoint(label,reload=True);m.check()
  assert all(torch.isfinite(p).all() for p in model.parameters());r['status']='PASS'
 except Exception as e:
  r.update(status='STOPPED' if m.stop.is_set() else 'FAIL',error=repr(e),traceback=traceback.format_exc())
  if model is not None and opt is not None:
   try:checkpoint('recovery')
   except Exception as saving_error:r['recovery_save_error']=repr(saving_error)
 finally:
  r.update(m.close());r['wall_seconds']=time.perf_counter()-started;save_json(out/'run.json',r)
  for name in sourcefiles:
   dest=out/'runtime-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
  (out/'Results.txt').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(status=r['status'],width=a.width,output=str(out))),flush=True)
 return 0 if r['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
