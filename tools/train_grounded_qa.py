"""Story-rehearsed SQuAD-derived QA candidate after five-hour pretraining."""
from pathlib import Path
from dataclasses import asdict
from collections import Counter
import sys,json,time,random,os,re,string,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lm.monitor import Monitor
from lm.resource_guard import Policy
from cpu_screen_worker import save_json

def normalize(text):return ' '.join(re.sub(r'\b(a|an|the)\b',' ',''.join(c for c in text.lower() if c not in string.punctuation)).split())
def f1(pred,gold):
 p=normalize(pred).split();g=normalize(gold).split();common=sum((Counter(p)&Counter(g)).values())
 return 2*common/(len(p)+len(g)) if common else float(p==g)
def main(a):
 out=a.out.resolve();out.mkdir(exist_ok=False,parents=True);m=Monitor(out,Policy(available_reserve_bytes=512*1024**2,immediate_available_bytes=512*1024**2));r=dict(status='RUNNING',updates=0,training_seconds=0.,validation_seconds=0.,generation_seconds=0.,qa_targets=0,story_replay_targets=0,epochs=[],final_test_evaluated=False,policy=a.policy,objective='Per-example response+EOS mean QA loss (family weighting only if source experiment qualified) +1.0 token-mean TinyStories replay loss. Four QA sequences plus one story sequence; two passes, one clipped AdamW update. Fresh AdamW moments, 100-update warmup; chosen LR and replay weight recorded separately.',seed=56,source_checkpoint=str(a.checkpoint))
 try:
  r['learning_rate']=a.lr;r['story_replay_weight']=a.story_weight;r['objective']=f'Per-example response+EOS mean QA loss, with family weights only if qualified, plus{a.story_weight} times token-mean TinyStories replay loss. Four QA sequences and one story sequence; two forward/backward branches, one clipped AdamW update. Fresh moments; LR{a.lr},100-update linear warmup.'
  m.start();os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8');m.phase='imports'
  import numpy as np
  import torch
  from tokenizers import Tokenizer
  from lm.experimental_model import Config,make_optimizer,token_loss,parameter_hash
  from lm.efficiency import EfficientDecoder,RollingCache
  from lm.source_learning import encode,response_loss,family_weights
  from lm.batches import digest
  torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(56)
  data=ROOT/'data/SQuAD2Windows128_v1';manifest=json.loads((data/'manifest.json').read_text());assert all(digest(data/name)==v['sha256'] for name,v in manifest['files'].items());r['dataset_manifest_sha256']=digest(data/'manifest.json');tok=Tokenizer.from_file(str(ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json'));r['tokenizer_sha256']=digest(ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json');rows=[json.loads(s) for s in (data/'train.jsonl').read_text(encoding='utf-8').splitlines()];rows +=[json.loads(s) for s in (ROOT/'data/SourceFamilies_v1/balanced.jsonl').read_text().splitlines()];dialogue=ROOT/'data/LocalDialogue_v1';dm=json.loads((dialogue/'manifest.json').read_text());assert digest(dialogue/'train.jsonl')==dm['files']['train.jsonl']['sha256'];rows +=[json.loads(t) for t in (dialogue/'train.jsonl').read_text().splitlines()];r['dialogue_manifest_sha256']=digest(dialogue/'manifest.json');r['training_examples']=len(rows);r['planned_epochs']=a.epochs;held=[json.loads(s) for s in (data/'dev.jsonl').read_text(encoding='utf-8').splitlines()];tx,ty=encode(rows,tok);hx,hy=encode(held,tok);weights=family_weights(rows) if a.policy=='family' else torch.ones(len(rows))
  assert digest(a.checkpoint)==a.expected_sha256;saved=torch.load(a.checkpoint,weights_only=True,mmap=True);model=EfficientDecoder(Config(**saved['config']));model.load_state_dict(saved['model']);assert parameter_hash(model)==saved['parameter_sha256'];r['initial_parameter_sha256']=saved['parameter_sha256'];r['source_checkpoint_sha256']=a.expected_sha256;del saved;opt=make_optimizer(model)
  bd=ROOT/'data/TinyStoriesBatches128_Unigram8192';sx=np.fromfile(bd/'train.x.u16',dtype='<u2').reshape(-1,128);sy=np.fromfile(bd/'train.y.u16',dtype='<u2').reshape(-1,128);sorder=np.fromfile(bd/'train.order.u32',dtype='<u4');vx=np.fromfile(bd/'validation.x.u16',dtype='<u2').reshape(-1,128);vy=np.fromfile(bd/'validation.y.u16',dtype='<u2').reshape(-1,128)
  def story_eval():
   at=time.perf_counter();m.phase='story_validation';total=0.;n=0;model.eval()
   with torch.no_grad():
    for i in range(0,len(vx),4):
     m.check();x=torch.from_numpy(vx[i:i+4].astype(np.int64));y=torch.from_numpy(vy[i:i+4].astype(np.int64));y.masked_fill_(y.eq(0),-100);_,s,c=token_loss(model(x),y);total+=float(s);n+=c
   r['validation_seconds']+=time.perf_counter()-at;assert n==216150;return dict(nll_sum=total,valid_targets=n,nats_per_target=total/n)
  def evaluate(label):
   at=time.perf_counter();m.phase='qa_validation';model.eval();summed=0.;count=0
   with torch.no_grad():
    for i in range(0,len(held),4):m.check();_,s,n=response_loss(model,hx[i:i+4],hy[i:i+4]);summed+=float(s);count+=n
   r['validation_seconds']+=time.perf_counter()-at;at=time.perf_counter();pred=[];m.phase='qa_generation'
   with torch.inference_mode():
    for e in held:
     ids=[1]+tok.encode('Instruction:\n'+e['instruction']+'\nResponse:\n').ids;gen=[];cache=RollingCache(model)
     for _ in range(12):
      m.check();x=torch.tensor([ids]);scores=cache(x)[0] if len(ids)<=128 else model(x[:,-128:],last_only=True)[0];scores=scores.clone();scores[[0,1,3]]=-float('inf');n=int(scores.argmax());gen.append(n);ids.append(n)
      if n==2:break
     answer=tok.decode(gen,skip_special_tokens=True).strip();pred.append(dict(id=e['id'],answerable=e['answerable'],answer=answer,references=e['references'],exact_match=any(normalize(answer)==normalize(g) for g in e['references']),f1=max(f1(answer,g) for g in e['references'])))
   r['generation_seconds']+=time.perf_counter()-at;metrics=dict(response_nll_sum=summed,response_targets=count,response_nats_per_target=summed/count,exact_match=sum(e['exact_match'] for e in pred)/len(pred),f1=sum(e['f1'] for e in pred)/len(pred),answerable_exact_match=sum(e['exact_match'] for e in pred if e['answerable'])/128,answerable_f1=sum(e['f1'] for e in pred if e['answerable'])/128,unanswerable_exact_match=sum(e['exact_match'] for e in pred if not e['answerable'])/128);save_json(out/(label+'-responses.json'),pred);return metrics
  def source_diagnostic(label):
   m.phase='source_diagnostic_'+label;model.eval();at=time.perf_counter();examples=[json.loads(s) for s in (ROOT/'data/SourceFamilies_v1/heldout.jsonl').read_text().splitlines()];pred=[]
   with torch.inference_mode():
    for e in examples:
     ids=[1]+tok.encode('Instruction:\n'+e['instruction']+'\nResponse:\n').ids;gen=[];cache=RollingCache(model)
     for _ in range(8):
      m.check();ix=torch.tensor([ids]);scores=cache(ix)[0] if len(ids)<=128 else model(ix[:,-128:],last_only=True)[0];scores=scores.clone();scores[[0,1,3]]=-float('inf');n=int(scores.argmax());ids.append(n);gen.append(n)
      if n==2:break
     answer=tok.decode(gen,skip_special_tokens=True).strip();pred.append(dict(id=e['id'],family=e['family'],variant=e['variant'],expected=e['response'],answer=answer,correct=answer==e['response']))
   pairs=[]
   for family in sorted({e['family'] for e in pred}):
    group={e['variant']:e for e in pred if e['family']==family};pairs.append(group['original']['correct'] and group['moved']['correct'])
   r['generation_seconds']+=time.perf_counter()-at;save_json(out/(label+'-source-responses.json'),pred);return dict(exact_match=sum(e['correct'] for e in pred)/len(pred),both_answerable_pair_correct=sum(pairs)/len(pairs),per_variant={v:sum(e['correct'] for e in pred if e['variant']==v)/64 for v in ('original','moved','missing','negated')})
  r['before']=dict(story=story_eval(),qa=evaluate('before'),source=source_diagnostic('before'));order=list(range(len(rows)));rng=random.Random(56);story_cursor=0
  with (out/'steps.jsonl').open('w') as log:
   for epoch in range(a.epochs):
    rng.shuffle(order);m.phase='qa_and_story_replay';model.train();at=time.perf_counter()
    for i in range(0,len(order),4):
     m.check();idx=order[i:i+4];opt.zero_grad(set_to_none=True);lr=a.lr*min(1.,(r['updates']+1)/100)
     for group in opt.param_groups:group['lr']=lr
     qloss,_,qn=response_loss(model,tx[idx],ty[idx],weights[idx]);row=sorder[story_cursor];story_cursor=(story_cursor+1)%len(sorder);x=torch.from_numpy(sx[row:row+1].astype(np.int64));y=torch.from_numpy(sy[row:row+1].astype(np.int64));y.masked_fill_(y.eq(0),-100);sloss,_,sn=token_loss(model(x),y);loss=qloss+a.story_weight*sloss;assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);m.check();opt.step();r['updates']+=1;r['qa_targets']+=qn;r['story_replay_targets']+=sn;log.write(json.dumps(dict(update=r['updates'],qa_loss=float(qloss.detach()),story_loss=float(sloss.detach()),gradient_norm=float(norm),qa_targets=qn,story_targets=sn))+'\n')
    r['training_seconds']+=time.perf_counter()-at;entry=dict(epoch=epoch+1,story=story_eval(),qa=evaluate('epoch'+str(epoch+1)));r['epochs'].append(entry);m.phase='checkpoint';h=parameter_hash(model);path=out/f'epoch{epoch+1}.pt';torch.save(dict(config=asdict(model.config),model=model.state_dict(),parameter_sha256=h,tokenizer_sha256=r['tokenizer_sha256']),path);entry['checkpoint']=dict(file=path.name,sha256=digest(path),parameter_sha256=h);save_json(out/'run.json',r);print(json.dumps(entry),flush=True)
  initial=r['before'];eligible=[e for e in r['epochs'] if e['qa']['answerable_exact_match']>=.25 and e['qa']['exact_match']>=.60 and e['story']['nats_per_target']<=initial['story']['nats_per_target']+.15 and e['qa']['exact_match']>=initial['qa']['exact_match']+.05 and (e['qa']['answerable_exact_match']>=initial['qa']['answerable_exact_match']+.05 or e['qa']['answerable_f1']>=initial['qa']['answerable_f1']+.10)];r['selected_epoch']=max(eligible,key=lambda e:e['qa']['exact_match'])['epoch'] if eligible else None;r['selection_rule']='Answerable EM>=25%, balanced-dev EM>=60%; story loss increase<=0.15nats; balanced-dev exact match improves>=5pp; answerable EM>=+5pp or answerable F1>=+10pp. Highest eligible exact match. Unknown-only model cannot qualify.';
  if r['selected_epoch'] is not None:
   restored=torch.load(out/f"epoch{r['selected_epoch']}.pt",weights_only=True,mmap=True);model.load_state_dict(restored['model']);assert parameter_hash(model)==restored['parameter_sha256'];del restored
  r['after_source']=source_diagnostic('selected' if r['selected_epoch'] else 'unselected_final_candidate');r['status']='PASS';m.check()
 except Exception as e:r.update(status='STOPPED' if m.stop.is_set() else 'ERROR',error=repr(e),traceback=traceback.format_exc())
 finally:r['resources']=m.close();save_json(out/'run.json',r);print(json.dumps(dict(status=r['status'],selected_epoch=r.get('selected_epoch'))),flush=True)
 return 0 if r['status']=='PASS' else 2
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=6);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-sha256',required=True);p.add_argument('--policy',default='ordinary');p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--story-weight',type=float,default=3.);p.add_argument('--out',type=Path,required=True);sys.exit(main(p.parse_args()))
