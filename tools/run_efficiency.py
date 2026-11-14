"""Reproducible sequential efficiency screens and preliminary matched-time quality tests."""
from pathlib import Path
import argparse,json,sys,os,time,subprocess,statistics,traceback,hashlib,math
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
VARIANTS=['baseline','valid','trim','sdpa','combined','factor96','factor96wide288']
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def worker(a):
 out=a.out;out.mkdir(parents=True,exist_ok=False)
 from lm.monitor import Monitor
 m=Monitor(out);r=dict(status='RUNNING',variant=a.variant,stage=a.stage,training_seconds=0.,updates=0,valid_targets=0,validation={},warmup_updates=0,seed=a.seed,order_seed=43,batch=4,context=128,threads=8,interop_threads=1,final_test_evaluated=False)
 try:
  m.start();m.phase='imports_integrity';os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
  import numpy as np
  import torch
  from lm.experimental_model import Decoder,make_optimizer,parameter_hash,token_loss
  from lm.width_model import width_config
  from lm.efficiency import EfficientDecoder,training_loss,RollingCache
  from lm.batches import verify_batch_lock,digest
  torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(a.seed)
  bd=ROOT/'data/TinyStoriesBatches128_Unigram8192';manifest=verify_batch_lock(bd,ROOT/'data/TinyStoriesSubset_Unigram8192')
  r.update(tokenizer_sha256=manifest['tokenizer_sha256'],batch_manifest_sha256=digest(bd/'manifest.json'),torch_version=str(torch.__version__))
  tx=np.fromfile(bd/'train.x.u16',dtype='<u2').reshape(-1,128);ty=np.fromfile(bd/'train.y.u16',dtype='<u2').reshape(-1,128);order=np.fromfile(bd/'train.order.u32',dtype='<u4');vx=np.fromfile(bd/'validation.x.u16',dtype='<u2').reshape(-1,128);vy=np.fromfile(bd/'validation.y.u16',dtype='<u2').reshape(-1,128)
  assert np.count_nonzero(ty)==4388821 and np.count_nonzero(vy)==216150
  v=a.variant;cfg=width_config(288 if v=='factor96wide288' else 192)
  valid=v in ('valid','combined','factor96','factor96wide288');trim=v in ('trim','combined');attention='sdpa' if v in ('sdpa','combined') else 'manual'
  # Factor screens isolate factorization plus valid-only scoring; no automatic attention change.
  model=Decoder(cfg,seed=a.seed) if v=='baseline' else EfficientDecoder(cfg,seed=a.seed,attention=attention,factor_dim=96 if v.startswith('factor') else None)
  opt=make_optimizer(model);r.update(parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=parameter_hash(model),attention=attention,valid_only=valid,trim=trim,config=vars(cfg),factor_initialization='token table normal(0,.02), tied projection normal(0,1/sqrt(96)), independent seed'+str(a.seed+1000) if v.startswith('factor') else None)
  def batch(x,y,idx):
   ix=torch.from_numpy(x[idx].astype(np.int64));iy=torch.from_numpy(y[idx].astype(np.int64));iy.masked_fill_(iy.eq(0),-100);return ix,iy
  def loss(x,y):return token_loss(model(x),y) if v=='baseline' else training_loss(model,x,y,valid,trim)
  def evaluate(label):
   m.phase='validation_'+label;model.eval();at=time.perf_counter();total=0.;count=0
   with torch.no_grad():
    for start in range(0,len(vx),4):
     m.check();x,y=batch(vx,vy,slice(start,start+4));_,s,n=loss(x,y);total+=float(s);count+=n
   assert count==216150
   r['validation'][label]=dict(nll_sum=total,valid_targets=count,nats_per_target=total/count,bits_per_utf8_text_byte_including_eos=total/884630/math.log(2),seconds=time.perf_counter()-at,training_seconds=r['training_seconds'],updates=r['updates'],training_targets=r['valid_targets'])
   save(out/'run.json',r);print(json.dumps(r['validation'][label]),flush=True)
  if a.stage=='quality':evaluate('initialization')
  cursor=0
  def step():
   nonlocal cursor
   m.check();idx=order[cursor:cursor+4];x,y=batch(tx,ty,idx);cursor+=len(idx)
   if cursor==len(order):cursor=0
   opt.zero_grad(set_to_none=True);l,s,n=loss(x,y)
   if not torch.isfinite(l):raise RuntimeError('Nonfinite loss')
   l.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);m.check();opt.step()
   return float(s),n,float(norm),int(y.numel()),int(y.ne(-100).any(0).sum())*len(idx)
  model.train();m.phase='warmup'
  if a.stage=='screen':
   at=time.perf_counter();warmtokens=0
   for _ in range(10):
    s,n,*_=step();warmtokens+=n;r['warmup_updates']+=1;r['warmup_targets']=warmtokens
   r.update(warmup_targets=warmtokens,warmup_seconds=time.perf_counter()-at)
  m.phase='training';at=time.perf_counter();latencies=[];executed=0;trimmed=0
  with (out/'steps.jsonl').open('w') as log:
   while time.perf_counter()-at<a.seconds:
    st=time.perf_counter();s,n,norm,pos,tr=step();elapsed=time.perf_counter()-st;latencies.append(elapsed);r['updates']+=1;r['valid_targets']+=n;executed+=pos;trimmed+=tr
    r['training_seconds']=time.perf_counter()-at
    log.write(json.dumps(dict(update=r['updates'],targets=n,nll_sum=s,gradient_norm=norm,seconds=elapsed,training_seconds=r['training_seconds']))+'\n')
    if r['updates']%100==0:save(out/'run.json',r)
  r.update(training_seconds=time.perf_counter()-at,executed_untrimmed_positions=executed,potential_trimmed_positions=trimmed,next_row=cursor)
  r['valid_targets_per_second']=r['valid_targets']/r['training_seconds'];r['median_step_seconds']=statistics.median(latencies);r['max_step_seconds']=max(latencies)
  if a.stage=='quality':
   evaluate('final');m.phase='checkpoint';at=time.perf_counter();h=parameter_hash(model)
   torch.save(dict(config=vars(cfg),factor_dim=96 if v.startswith('factor') else None,model=model.state_dict(),optimizer=opt.state_dict(),parameter_sha256=h,report=r),out/'final.pt')
   saved=torch.load(out/'final.pt',weights_only=True,mmap=True);model.load_state_dict(saved['model']);assert parameter_hash(model)==h
   r['checkpoint']=dict(file='final.pt',sha256=digest(out/'final.pt'),parameter_sha256=h,reload_verified=True,seconds=time.perf_counter()-at)
  r['status']='PASS'
 except Exception as e:r.update(status='STOPPED' if m.stop.is_set() else 'ERROR',error=repr(e),traceback=traceback.format_exc(),interrupted_step_work='partial or unknown; not counted as a completed update')
 finally:
  r['resources']=m.close();save(out/'run.json',r);print(json.dumps(dict(status=r['status'],variant=a.variant,stage=a.stage,targets=r['valid_targets'],seconds=r['training_seconds'])),flush=True)
 return 0 if r['status']=='PASS' else 2

def suite(a):
 out=ROOT/'results'/('Efficiency_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir();save(ROOT/'results/Efficiency_Current.json',dict(directory=str(out.relative_to(ROOT))));print(out,flush=True)
 plan=dict(status='RUNNING',screen_seconds=20,screen_repetitions=3,quality_seconds=600,quality_seeds=[42,44,46],quality_scope='Three paired initialization seeds, ten-minute matched active-training budget per architecture. Conclusions limited to this budget.',runs=[],source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'lm/efficiency.py',Path(__file__)]})
 save(out/'suite.json',plan)
 def run(v,stage,seconds,name,seed=42):
  path=out/name
  cp=subprocess.run([sys.executable,str(Path(__file__)),'--worker','--variant',v,'--stage',stage,'--seconds',str(seconds),'--out',str(path),'--seed',str(seed)],cwd=ROOT,stdout=(out/(name+'.stdout.log')).open('w'),stderr=(out/(name+'.stderr.log')).open('w'))
  r=json.loads((path/'run.json').read_text());plan['runs'].append(dict(directory=name,variant=v,stage=stage,status=r['status']));save(out/'suite.json',plan)
  print(name,r['status'],r.get('valid_targets_per_second'),flush=True)
  return cp.returncode==0
 for rep in range(3):
  variants=VARIANTS[rep:]+VARIANTS[:rep]
  for v in variants:
   if not run(v,'screen',20,f'screen{rep+1}_{v}'):plan['status']='STOPPED';save(out/'suite.json',plan);return 2
   time.sleep(2)
 for rep,seed in enumerate((42,44,46)):
  variants=['valid','factor96','factor96wide288'];variants=variants[rep:]+variants[:rep]
  for v in variants:
   if not run(v,'quality',600,f'quality_{v}_seed{seed}',seed):plan['status']='STOPPED';save(out/'suite.json',plan);return 2
 plan['status']='PASS';save(out/'suite.json',plan);return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--worker',action='store_true');p.add_argument('--variant',choices=VARIANTS,default='baseline');p.add_argument('--stage',choices=['screen','quality'],default='screen');p.add_argument('--seconds',type=float,default=20);p.add_argument('--out',type=Path);a=p.parse_args();sys.exit(worker(a) if a.worker else suite(a))
