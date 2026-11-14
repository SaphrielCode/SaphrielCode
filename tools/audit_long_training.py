"""Independent exposure, telemetry and checkpoint audit for long training."""
from pathlib import Path
import sys,json,math,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 import torch,numpy as np
 from lm.experimental_model import Config,parameter_hash
 from lm.efficiency import EfficientDecoder
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 out=ROOT/json.loads((ROOT/'results/SourceAndFiveHour_Current.json').read_text())['directory'];d=out/'five_hours';r=json.loads((d/'run.json').read_text());assert r['status']=='PASS' and r['training_seconds']>=18000
 bd=ROOT/'data/TinyStoriesBatches128_Unigram8192';y=np.fromfile(bd/'train.y.u16',dtype='<u2').reshape(-1,128);order=np.fromfile(bd/'train.order.u32',dtype='<u4');lengths=(y!=0).sum(1);assert len(order)==45137 and int(lengths.sum())==4388821
 steps=[json.loads(s) for s in (d/'steps.jsonl').read_text().splitlines()];cursor=0;passes=0;targets=0
 for index,e in enumerate(steps,1):
  rows=order[cursor:cursor+4];n=int(lengths[rows].sum());assert e['update']==index and e['valid_targets']==n;targets+=n;cursor+=len(rows)
  if cursor==len(order):cursor=0;passes+=1
  assert e['next_row']==cursor and e['completed_passes']==passes and math.isfinite(e['loss_sum']) and math.isfinite(e['gradient_norm'])
 assert targets==r['valid_targets'] and len(steps)==r['completed_updates'] and cursor==r['next_row'] and passes==r['completed_passes']
 for v in r['validation'].values():assert v['valid_targets']==216150 and abs(v['loss_sum']/216150-v['loss_nats_per_target'])<1e-12
 tele=[json.loads(s) for s in (d/'telemetry.jsonl').read_text().splitlines()];assert all(e['available_system_bytes']>=512*1024**2 and not e['guard_stop'] for e in tele);assert min(e['available_system_bytes'] for e in tele)==r['minimum_available_system_bytes']
 records=[]
 for path in sorted(out.rglob('*.pt')):
  saved=torch.load(path,weights_only=True,mmap=True);model=EfficientDecoder(Config(**saved['config']));model.load_state_dict(saved['model']);actual=parameter_hash(model);assert actual==saved['parameter_sha256'];assert sum(p.numel() for p in model.parameters())==5115072 and all(torch.isfinite(p).all() for p in model.parameters());record=dict(file=path.relative_to(ROOT).as_posix(),sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),parameter_sha256=actual)
  matching=next((e for e in r['checkpoints'] if d/e['file']==path),None)
  if matching:assert record['sha256']==matching['sha256']
  if 'optimizer' in saved:
   counts={int(e['step']) for e in saved['optimizer']['state'].values() if 'step' in e};record['optimizer_step_counts']=sorted(counts);assert counts=={saved['report']['completed_updates']}
  records.append(record);del model,saved
 for seed in (44,45,46):
  a=json.loads((out/f'source_ordinary_{seed}/run.json').read_text());b=json.loads((out/f'source_family_{seed}/run.json').read_text());assert a['batch_index_trace_sha256']==b['batch_index_trace_sha256'] and a['updates']==b['updates']==2400 and a['valid_response_targets']==b['valid_response_targets'] and a['initial_parameter_sha256']==b['initial_parameter_sha256']
 audit=dict(status='PASS',updates=len(steps),valid_targets=targets,completed_passes=passes,next_row=cursor,maximum_step_seconds=max(e['seconds'] for e in steps),maximum_monitor_gap_seconds=max(b['seconds']-a['seconds'] for a,b in zip(tele,tele[1:])),minimum_available_system_bytes=r['minimum_available_system_bytes'],checkpoints=records,final_test_evaluated=False)
 (out/'independent-audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='checkpoints'}))
if __name__=='__main__':main()
