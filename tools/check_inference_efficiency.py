"""Fresh-process trained-checkpoint decoding tests; fixed 32 generated steps, EOS not early-stopped."""
from pathlib import Path
import sys,os,json,time,argparse,subprocess,statistics
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['full','last','cache','hybrid']);p.add_argument('--length',type=int,default=24);a=p.parse_args()
 if not a.mode:
  a.out.mkdir(exist_ok=False);rows=[]
  for length in (24,128):
   for rep in range(3):
    for mode in ('full','last','cache'):
     target=a.out/f'{length}_{rep}_{mode}';cp=subprocess.run([sys.executable,__file__,'--out',str(target),'--mode',mode,'--length',str(length)],cwd=ROOT,capture_output=True,text=True)
     if cp.returncode:raise RuntimeError(cp.stdout+cp.stderr)
     rows.append(json.loads((target/'run.json').read_text()))
   group=[r for r in rows if r['prompt_tokens']==length];assert all(r['generated_ids']==group[0]['generated_ids'] for r in group)
  (a.out/'summary.json').write_text(json.dumps(dict(status='PASS',runs=rows,eos_policy='32 fixed decoding steps; EOS does not end diagnostic timing',cache_policy='Full rebuild at every rolling-window overflow to preserve reset positions and conditioning'),indent=2));return
 a.out.mkdir();from lm.monitor import Monitor
 m=Monitor(a.out);r=dict(status='RUNNING',mode=a.mode,prompt_tokens=a.length)
 try:
  m.start();m.phase='load';os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
  import torch
  from tokenizers import Tokenizer
  from lm.experimental_model import Config,Decoder,parameter_hash
  from lm.efficiency import EfficientDecoder,RollingCache
  from lm.batches import digest
  torch.set_num_threads(8);torch.set_num_interop_threads(1)
  s=json.loads((ROOT/'WorkingSelection.json').read_text());path=ROOT/s['reference_checkpoint'];assert digest(path)==s['reference_checkpoint_sha256'];assert digest(ROOT/s['tokenizer_file'])==s['tokenizer_sha256']
  saved=torch.load(path,weights_only=True,mmap=True);cfg=Config(**saved['config']);model=(Decoder(cfg) if a.mode=='full' else EfficientDecoder(cfg));model.load_state_dict(saved['model']);assert parameter_hash(model)==s['reference_parameter_sha256'];del saved;model.eval().requires_grad_(False)
  tok=Tokenizer.from_file(str(ROOT/s['tokenizer_file']));prompt='Once upon a time, a little girl found a small red box in the garden. ';encoded=tok.encode(prompt*12,add_special_tokens=False).ids;ids=[1]+encoded[:a.length-1];r['prompt_text']=tok.decode(ids,skip_special_tokens=True);r['checkpoint_sha256']=s['reference_checkpoint_sha256'];r['parameter_sha256']=s['reference_parameter_sha256']
  cache=RollingCache(model);generated=[];m.phase='generation';latencies=[]
  with torch.inference_mode():
   # Untimed warm execution creates runtime kernels without modifying weights.
   model(torch.tensor([ids]),**({'last_only':True} if a.mode!='full' else {}))
   at=time.perf_counter()
   for i in range(32):
    m.check();st=time.perf_counter();x=torch.tensor([ids],dtype=torch.long)
    scores=model(x[:,-128:])[0,-1] if a.mode=='full' else model(x[:,-128:],last_only=True)[0] if a.mode=='last' else model(x[:,-128:],last_only=True)[0] if a.mode=='hybrid' and len(ids)>128 else cache(x)[0]
    scores=scores.clone();scores[[0,1,3]]=-float('inf');n=int(scores.argmax());ids.append(n);generated.append(n);latencies.append(time.perf_counter()-st)
   elapsed=time.perf_counter()-at
  r.update(status='PASS',seconds=elapsed,first_token_seconds=latencies[0],decode_tokens_per_second=31/sum(latencies[1:]),generated_ids=generated,generated_text=tok.decode(generated,skip_special_tokens=True),cache_rebuilds=cache.rebuilds if a.mode in ('cache','hybrid') else None)
 except Exception as e:r.update(status='ERROR',error=repr(e));raise
 finally:r['resources']=m.close();(a.out/'run.json').write_text(json.dumps(r,indent=2))
if __name__=='__main__':main()
