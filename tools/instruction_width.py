"""Shared response-only instruction diagnostic after matched-time pretraining."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,os,sys,time,random,unicodedata,re,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lm.monitor import Monitor
from cpu_screen_worker import save_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--width',type=int,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True);m=Monitor(out);r=dict(status='RUNNING',width=a.width,training_seconds=0.,evaluation_seconds=0.,generation_seconds=0.,checkpoint_seconds=0.,updates=0,valid_response_targets=0,epochs=5,final_test_evaluated=False,pretraining_checkpoint=str(a.checkpoint),metrics={})
 started=time.perf_counter()
 try:
  m.start();m.phase='imports';os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
  import numpy as np
  import torch
  from tokenizers import Tokenizer
  from lm.width_model import Decoder,width_config,parameter_hash,make_optimizer,token_loss
  from lm.batches import digest
  torch.set_num_threads(8);torch.set_num_interop_threads(1);torch.manual_seed(42)
  data=ROOT/'data/InstructionDiagnostic_v1';manifest=json.loads((data/'manifest.json').read_text());r['instruction_manifest_sha256']=digest(data/'manifest.json');r['rubric']=manifest['rubric']
  for name,e in manifest['files'].items():assert digest(data/name)==e['sha256']
  tokenizer=Tokenizer.from_file(str(ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json'));r['tokenizer_sha256']=digest(ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json')
  examples={split:[json.loads(x) for x in (data/f'{split}.jsonl').read_text().splitlines()] for split in ('train','heldout')}
  def encode(items):
   xs=[];ys=[]
   for e in items:
    prompt='Instruction:\n'+e['instruction']+'\nResponse:\n';pi=tokenizer.encode(prompt).ids;ri=tokenizer.encode(e['response']).ids;sequence=[1]+pi+ri+[2];x=sequence[:-1];y=sequence[1:];y[:len(pi)]=[-100]*len(pi);assert len(x)<=128 and sum(t!=-100 for t in y)==len(ri)+1
    xs.append(x+[0]*(128-len(x)));ys.append(y+[-100]*(128-len(y)))
   return torch.tensor(xs,dtype=torch.long),torch.tensor(ys,dtype=torch.long)
  train_x,train_y=encode(examples['train']);val_x,val_y=encode(examples['heldout']);r['targets_per_epoch']=int(train_y.ne(-100).sum());r['heldout_targets']=int(val_y.ne(-100).sum())
  restored=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert restored['config']==asdict(width_config(a.width));assert restored['report']['width']==a.width;model=Decoder(width_config(a.width));model.load_state_dict(restored['model']);assert parameter_hash(model)==restored['parameter_sha256'];r['pretraining_parameter_sha256']=parameter_hash(model);r['pretraining_checkpoint_sha256']=digest(a.checkpoint);del restored
  optimizer=make_optimizer(model)
  def evaluate(label):
   m.phase='heldout_loss_'+label;model.eval();at=time.perf_counter();total=0.;count=0
   with torch.no_grad():
    for i in range(0,len(val_x),4):m.check();_,summed,n=token_loss(model(val_x[i:i+4]),val_y[i:i+4]);total+=float(summed);count+=n
   elapsed=time.perf_counter()-at;r['evaluation_seconds']+=elapsed;assert count==r['heldout_targets'];r['metrics'][label]=dict(response_loss_sum=total,valid_response_targets=count,response_nats_per_target=total/count,seconds=elapsed)
  def responses(label):
   m.phase='generation_'+label;model.eval();at=time.perf_counter();rows=[]
   with torch.no_grad():
    for e in examples['heldout']:
     prompt='Instruction:\n'+e['instruction']+'\nResponse:\n';ids=[1]+tokenizer.encode(prompt).ids;generated=[]
     for i in range(32):
      m.check();n=int(model(torch.tensor([ids[-128:]],dtype=torch.long))[0,-1].argmax());generated.append(n);ids.append(n)
      if n==2:break
     answer=tokenizer.decode(generated,skip_special_tokens=True);normalized=unicodedata.normalize('NFC',answer).strip();words=normalized.split();grams=[tuple(words[i:i+4]) for i in range(max(0,len(words)-3))]
     if e['task']=='repeat':format_ok=bool(normalized) and '\n' not in normalized and len(words)==3
     elif e['task']=='yes_no':format_ok=normalized in ('yes','no')
     else:format_ok=re.fullmatch('[A-Za-z]+',normalized) is not None
     longest=current=0;last=None
     for word in words:current=current+1 if word==last else 1;last=word;longest=max(longest,current)
     rows.append(dict(**e,generated_text=answer,generated_ids=generated,exact_match=normalized==e['response'],instruction_format=bool(format_ok),repeated_fourgram_fraction=(len(grams)-len(set(grams)))/len(grams) if grams else 0.,longest_identical_word_run=longest))
   elapsed=time.perf_counter()-at;r['generation_seconds']+=elapsed;result=dict(exact_match=sum(x['exact_match'] for x in rows)/len(rows),instruction_format=sum(x['instruction_format'] for x in rows)/len(rows),mean_repeated_fourgram_fraction=sum(x['repeated_fourgram_fraction'] for x in rows)/len(rows),per_task={})
   for task in manifest['tasks']:
    subset=[x for x in rows if x['task']==task];result['per_task'][task]=dict(exact_match=sum(x['exact_match'] for x in subset)/len(subset),instruction_format=sum(x['instruction_format'] for x in subset)/len(subset),examples=len(subset))
   r['metrics'][label].update(result);save_json(out/f'{label}-responses.json',rows);save_json(out/'run.json',r)
  evaluate('before');responses('before')
  order=list(range(len(train_x)));random.Random(44).shuffle(order);at=time.perf_counter();m.phase='instruction_training';model.train()
  with (out/'steps.jsonl').open('w') as log:
   for epoch in range(5):
    for i in range(0,len(order),4):
     m.check();indices=order[i:i+4];optimizer.zero_grad(set_to_none=True);loss,summed,n=token_loss(model(train_x[indices]),train_y[indices]);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False);m.check();optimizer.step();r['updates']+=1;r['valid_response_targets']+=n;log.write(json.dumps(dict(update=r['updates'],epoch=epoch+1,valid_targets=n,loss_sum=float(summed),gradient_norm=float(norm)))+'\n')
   r['training_seconds']=time.perf_counter()-at
  assert r['updates']==625 and r['valid_response_targets']==5*r['targets_per_epoch'];evaluate('after');responses('after');m.phase='checkpoint';at=time.perf_counter();expected=parameter_hash(model);checkpoint=out/'instruction.pt';torch.save(dict(config=asdict(model.config),model=model.state_dict(),tokenizer_sha256=r['tokenizer_sha256'],parameter_sha256=expected),checkpoint);del optimizer,model,loss,summed,norm;saved=torch.load(checkpoint,map_location='cpu',weights_only=True);model=Decoder(width_config(a.width));model.load_state_dict(saved['model']);assert parameter_hash(model)==expected;del saved;r.update(checkpoint_seconds=time.perf_counter()-at,checkpoint_sha256=digest(checkpoint),checkpoint_reload_verified=True,final_parameter_sha256=expected,status='PASS');m.check()
 except Exception as e:r.update(status='STOPPED' if m.stop.is_set() else 'FAIL',error=repr(e),traceback=traceback.format_exc())
 finally:
  r.update(m.close());r['wall_seconds']=time.perf_counter()-started;save_json(out/'run.json',r);(out/'Results.txt').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(status=r['status'],width=a.width,metrics=r['metrics'])),flush=True)
 return 0 if r['status']=='PASS' else 1
if __name__=='__main__':sys.exit(main())
