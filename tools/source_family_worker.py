"""Controlled ordinary/family-weighted/balanced-data instruction comparison."""
from pathlib import Path
from dataclasses import asdict
import sys,json,os,time,random,hashlib,traceback,math,copy
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lm.monitor import Monitor
from lm.resource_guard import Policy
from cpu_screen_worker import save_json

def run(a):
 out=a.out.resolve();out.mkdir(exist_ok=False,parents=True);m=Monitor(out,Policy(available_reserve_bytes=512*1024**2,immediate_available_bytes=512*1024**2));r=dict(status='RUNNING',arm=a.arm,order_seed=a.seed,updates=0,valid_response_targets=0,training_seconds=0.,evaluation_seconds=0.,generation_seconds=0.,checkpoint_seconds=0.,final_test_evaluated=False,guard_authorization='User approved0.5 GiB first-sample available-RAM stop;4 GiB process guard unchanged.')
 try:
  m.start();os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8');m.phase='imports'
  import numpy as np
  import torch
  from tokenizers import Tokenizer
  from lm.experimental_model import Config,make_optimizer,parameter_hash,token_loss
  from lm.efficiency import EfficientDecoder,RollingCache
  from lm.source_learning import encode,response_loss,family_weights
  from lm.batches import digest
  torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
  data=ROOT/'data/SourceFamilies_v1';manifest=json.loads((data/'manifest.json').read_text());assert all(digest(data/name)==entry['sha256'] for name,entry in manifest['files'].items());r['dataset_manifest_sha256']=digest(data/'manifest.json')
  rows=[json.loads(s) for s in (data/('balanced.jsonl' if a.arm=='balanced' else 'train.jsonl')).read_text().splitlines()];held=[json.loads(s) for s in (data/'heldout.jsonl').read_text().splitlines()];tok=Tokenizer.from_file(str(ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json'));r['tokenizer_sha256']=digest(ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json');x,y=encode(rows,tok);hx,hy=encode(held,tok);weights=family_weights(rows) if a.arm=='family' else torch.ones(len(rows));assert abs(float(weights.mean())-1)<1e-6
  assert digest(a.checkpoint)==a.expected_sha256
  saved=torch.load(a.checkpoint,weights_only=True,mmap=True);model=EfficientDecoder(Config(**saved['config']));model.load_state_dict(saved['model']);r['initial_parameter_sha256']=parameter_hash(model);assert r['initial_parameter_sha256']==saved['parameter_sha256'];r['pretraining_checkpoint_sha256']=digest(a.checkpoint);del saved;opt=make_optimizer(model)
  def story_loss():
   at=time.perf_counter();m.phase='story_validation';model.eval();bd=ROOT/'data/TinyStoriesBatches128_Unigram8192';vx=np.fromfile(bd/'validation.x.u16',dtype='<u2').reshape(-1,128);vy=np.fromfile(bd/'validation.y.u16',dtype='<u2').reshape(-1,128);total=0.;count=0
   with torch.no_grad():
    for start in range(0,len(vx),4):
     m.check();ix=torch.from_numpy(vx[start:start+4].astype(np.int64));iy=torch.from_numpy(vy[start:start+4].astype(np.int64));iy.masked_fill_(iy.eq(0),-100);_,s,n=token_loss(model(ix),iy);total+=float(s);count+=n
   assert count==216150;r['evaluation_seconds']+=time.perf_counter()-at;return dict(loss_sum=total,valid_targets=count,nats_per_target=total/count)
  def evaluate():
   m.phase='instruction_evaluation';model.eval();at=time.perf_counter();total=0.;count=0
   with torch.no_grad():
    for i in range(0,len(held),4):m.check();_,s,n=response_loss(model,hx[i:i+4],hy[i:i+4]);total+=float(s);count+=n
   r['response_nats_per_target']=total/count;r['evaluation_seconds']+=time.perf_counter()-at;m.phase='counterfactual_generation';at=time.perf_counter();predictions=[]
   with torch.inference_mode():
    for e in held:
     ids=[1]+tok.encode('Instruction:\n'+e['instruction']+'\nResponse:\n').ids;generated=[];cache=RollingCache(model)
     for _ in range(8):
      m.check();ix=torch.tensor([ids]);scores=cache(ix)[0] if len(ids)<=128 else model(ix[:,-128:],last_only=True)[0];scores=scores.clone();scores[[0,1,3]]=-float('inf');n=int(scores.argmax());ids.append(n);generated.append(n)
      if n==2:break
     answer=tok.decode(generated,skip_special_tokens=True).strip();predictions.append(dict(id=e['id'],family=e['family'],variant=e['variant'],expected=e['response'],answer=answer,correct=answer==e['response']))
   r['generation_seconds']+=time.perf_counter()-at;families=sorted({e['family'] for e in held});pairs=[];strict=[]
   for fid in families:
    group={e['variant']:e for e in predictions if e['family']==fid};pairs.append(group['original']['correct'] and group['moved']['correct']);strict.append(all(e['correct'] for e in group.values()))
   r['metrics']=dict(exact_match=sum(e['correct'] for e in predictions)/len(predictions),both_answerable_pair_correct=sum(pairs)/len(pairs),all_four_family_correct=sum(strict)/len(strict),per_variant={v:sum(e['correct'] for e in predictions if e['variant']==v)/64 for v in ('original','moved','missing','negated')},answer_frequencies=dict(Counter(e['answer'] for e in predictions)));save_json(out/'responses.json',predictions)
  r['story_before']=story_loss();order=list(range(len(rows)));rng=random.Random(a.seed);cursor=0;clipped=0;trace=hashlib.sha256();m.phase='instruction_training';model.train();at=time.perf_counter()
  with (out/'steps.jsonl').open('w') as log:
   for step in range(a.updates):
    if cursor==0:rng.shuffle(order)
    idx=order[cursor:cursor+4];cursor+=len(idx)
    if cursor==len(order):cursor=0
    trace.update(json.dumps(idx).encode());m.check();opt.zero_grad(set_to_none=True);loss,s,n=response_loss(model,x[idx],y[idx],weights[idx]);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);clipped+=int(float(norm)>1);m.check();opt.step();r['updates']+=1;r['valid_response_targets']+=n;r['training_seconds']=time.perf_counter()-at;log.write(json.dumps(dict(update=r['updates'],loss=float(loss.detach()),response_targets=n,gradient_norm=float(norm)))+'\n')
    if step%100==0:save_json(out/'run.json',r)
  r['training_seconds']=time.perf_counter()-at;r['batch_index_trace_sha256']=trace.hexdigest();r['clipping_frequency']=clipped/a.updates;evaluate();r['story_after']=story_loss();m.phase='checkpoint';at=time.perf_counter();h=parameter_hash(model);path=out/'instruction.pt';torch.save(dict(config=asdict(model.config),model=model.state_dict(),parameter_sha256=h,tokenizer_sha256=r['tokenizer_sha256']),path);r.update(checkpoint_sha256=digest(path),final_parameter_sha256=h,checkpoint_seconds=time.perf_counter()-at,status='PASS');m.check()
 except Exception as e:r.update(status='STOPPED' if m.stop.is_set() else 'ERROR',error=repr(e),traceback=traceback.format_exc())
 finally:r['resources']=m.close();save_json(out/'run.json',r);print(json.dumps(dict(arm=a.arm,seed=a.seed,status=r['status'],metrics=r.get('metrics'))),flush=True)
 return 0 if r['status']=='PASS' else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['ordinary','family','balanced'],required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--updates',type=int,default=2400);p.add_argument('--out',type=Path,required=True);sys.exit(run(p.parse_args()))
