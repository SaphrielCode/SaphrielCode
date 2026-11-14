"""Approved width screens, matched training-time runs, shared instruction diagnostics."""
from pathlib import Path
from datetime import datetime,timezone
import json,subprocess,sys,time,statistics,hashlib,struct,math,shutil
ROOT=Path(__file__).resolve().parents[1]
def save(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')
def read(path):return json.loads(path.read_text())
def call(cmd,path):
 with path.open('w') as log:
  child=subprocess.Popen([sys.executable]+cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
  for line in child.stdout:log.write(line);log.flush();print(line,end='',flush=True)
  return child.wait()
def verify_run(out):
 r=read(out/'run.json');assert r['status']=='PASS' and not r['final_test_evaluated'];assert r['training_seconds']>=r['budget_minutes']*60
 data=ROOT/'data/TinyStoriesBatches128_Unigram8192';valid=[x[2] for x in struct.iter_unpack('<III',(data/'train.rows.u32').read_bytes())];order=[x[0] for x in struct.iter_unpack('<I',(data/'train.order.u32').read_bytes())];cursor=epochs=updates=targets=0;last=0.
 with (out/'steps.jsonl').open() as f:
  for line in f:
   s=json.loads(line);updates+=1;indices=order[cursor:cursor+4];n=sum(valid[i] for i in indices);assert n==s['valid_targets'];targets+=n;cursor+=len(indices)
   if cursor==len(order):cursor=0;epochs+=1
   assert s['update']==updates and s['completed_passes']==epochs and s['next_row']==cursor and s['training_seconds']>=last;last=s['training_seconds'];assert math.isfinite(s['loss_sum']) and math.isfinite(s['gradient_norm'])
 assert updates==r['completed_updates'] and targets==r['valid_targets'] and epochs==r['completed_passes'] and cursor==r['next_row']
 for c in r['checkpoints']:
  p=out/c['file'];assert c['reload_verified'] and p.stat().st_size==c['bytes']
  with p.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==c['sha256']
 for label,v in r['validation'].items():assert v['valid_targets']==216150 and math.isfinite(v['loss_nats_per_target']);assert v['actual_training_seconds']>=v['requested_training_minutes']*60
 low=high=0
 with (out/'telemetry.jsonl').open() as f:
  for line in f:
   s=json.loads(line);assert s['guard_stop'] is None and s['available_system_bytes']>=2**30;low=low+1 if s['available_system_bytes']<2*2**30 else 0;high=high+1 if s['resident_bytes']>4*2**30 else 0;assert low<3 and high<3
 save(out/'verification.json',dict(status='PASS',update_and_target_counts=True,finite_step_losses_norms=True,checkpoint_hashes_and_reloads=True,guard_telemetry=True,fixed_validation_targets=True));return r

def summaries(out,suite):
 rows=[]
 for item in suite['training']:
  if item.get('status')!='PASS':continue
  r=read(out/item['directory']/'run.json')
  for label,v in r['validation'].items():rows.append(dict(width=r['width'],parameters=r['unique_parameters'],label=label,training_minutes=v['actual_training_seconds']/60,bits_per_byte=v['bits_per_utf8_text_byte_including_eos_cost'],valid_targets=v['valid_targets_processed'],updates=v['completed_updates'],passes=v['completed_passes']))
 summary=dict(status=suite['status'],learning_curves=rows,screening=suite['screens'],training=suite['training'],instruction=suite['instruction'],final_test_reserved=True,production_baseline_unchanged=True,limits='One initialization seed. Equal active training time is not equal arithmetic compute. Fixed corpus repetitions are not extra unique data. Evaluation and checkpoint time excluded from training clock and reported separately. Sequential order/background workloads may influence speed. Instruction diagnostic is synthetic with shared templates and disjoint subject words/names, not general conversational evaluation.')
 save(out/'summary.json',summary);save(ROOT/'results/WidthStudy_Summary.json',summary)
 lines=['WIDTH SCALING STUDY','Status: '+suite['status'],'','Width | Parameters | Checkpoint | Active train min | Validation bits/byte | Valid targets | Updates | Full passes']
 for x in rows:lines.append(f"{x['width']} | {x['parameters']} | {x['label']} | {x['training_minutes']:.4f} | {x['bits_per_byte']:.9f} | {x['valid_targets']} | {x['updates']} | {x['passes']}")
 lines+=['',summary['limits'],'',f'Artifacts: {out.relative_to(ROOT)}','Instruction results:']
 for item in suite['instruction']:
  if item.get('status')=='PASS':
   r=read(out/item['directory']/'run.json');lines += [f"Width {item['width']}: before {json.dumps(r['metrics']['before'])}; after {json.dumps(r['metrics']['after'])}; instruction training{r['training_seconds']:.3f}s, evaluation{r['evaluation_seconds']:.3f}s, generation{r['generation_seconds']:.3f}s; coherence review recorded separately."]
 if suite.get('error'):lines.append(suite['error'])
 text='\n'.join(lines)+'\n';(out/'Results.txt').write_text(text);(ROOT/'results/WidthStudy_Results.txt').write_text(text)

def main():
 out=ROOT/'results'/('WidthStudy_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir();print(str(out),flush=True)
 suite=dict(status='RUNNING',screens={},training=[],instruction=[],screen_384_rule='If median>=4388821/5400t/s use90min; if median>=4388821/10800t/s use180min with matching90min checkpoint; otherwise screen-only. Guards must pass. No automatic resource-stop retry.',training_order=[192,288,384],time_definition='Complete training loop work including gather, zero-grad, forward, loss, backward, clipping, optimizer and identical logging; excludes validation/checkpoint/startup. First training updates count; no extra warmup.',instruction_protocol_sha256=hashlib.sha256((ROOT/'data/InstructionDiagnostic_v1/manifest.json').read_bytes()).hexdigest(),instruction_pretrain_checkpoint='minute90 for every included width; fresh instruction AdamW, same500examples x5epochs, heldout60fixed',production_baseline_unchanged=True)
 save(out/'suite.json',suite);save(ROOT/'results/WidthStudy_Current.json',dict(directory=str(out.relative_to(ROOT))))
 try:
  for width in (288,384):
   reports=[];suite['active']=f'screen_width{width}';save(out/'suite.json',suite)
   for rep in range(1,4):
    directory=out/f'screen{width}_{rep}';code=call(['tools/screen_width.py','--width',str(width),'--out',str(directory)],out/f'screen{width}_{rep}.stdout.txt');r=read(directory/'screen.json');reports.append(r);suite['screens'][str(width)]=reports;save(out/'suite.json',suite)
    if code or r['status']!='PASS':raise RuntimeError(f'Width{width} screen stopped/failed; no training launched')
    if rep<3:time.sleep(5)
   assert len({r['initial_parameter_sha256'] for r in reports})==1
  median384=statistics.median(r['valid_targets_per_second'] for r in suite['screens']['384']);minutes384=90 if median384>=4388821/5400 else 180 if median384>=4388821/10800 else None;suite['width384_budget_minutes']=minutes384;save(out/'suite.json',suite)
  for width,minutes in ((192,90),(288,90),(384,minutes384)):
   if minutes is None:continue
   directory=out/f'train{width}';item=dict(width=width,budget_minutes=minutes,directory=directory.name,status='RUNNING');suite['training'].append(item);suite['active']=f'training_width{width}';save(out/'suite.json',suite);marks='15,30,60,90' if minutes==90 else '15,30,60,90,120,180';code=call(['tools/train_width.py','--width',str(width),'--minutes',str(minutes),'--marks',marks,'--out',str(directory)],out/f'train{width}.stdout.txt')
   if code:raise RuntimeError(f'Width{width} training stopped/failed; partial artifacts retained')
   r=verify_run(directory)
   if width!=192:assert r['initial_parameter_sha256']==suite['screens'][str(width)][0]['initial_parameter_sha256']
   item['status']='PASS';save(out/'suite.json',suite);summaries(out,suite)
  for item in suite['training']:
   width=item['width'];directory=out/f'instruction{width}';entry=dict(width=width,directory=directory.name,status='RUNNING');suite['instruction'].append(entry);suite['active']=f'instruction_width{width}';save(out/'suite.json',suite)
   code=call(['tools/instruction_width.py','--width',str(width),'--checkpoint',str(out/item['directory']/'minute90.pt'),'--out',str(directory)],out/f'instruction{width}.stdout.txt')
   if code:raise RuntimeError(f'Width{width} instruction stage stopped/failed')
   r=read(directory/'run.json');assert r['status']=='PASS' and r['checkpoint_reload_verified'] and r['updates']==625 and r['valid_response_targets']==r['targets_per_epoch']*5 and not r['final_test_evaluated'];assert r['instruction_manifest_sha256']==suite['instruction_protocol_sha256'];entry['status']='PASS';save(out/'suite.json',suite);summaries(out,suite)
  suite.update(status='PASS',active='complete')
 except Exception as e:suite.update(status='STOPPED_OR_FAILED',error=repr(e));raise
 finally:
  save(out/'suite.json',suite);summaries(out,suite)
  for name in ('tools/run_width_study.py','tools/train_width.py','tools/instruction_width.py','tools/screen_width.py','tools/prepare_instruction_diagnostic.py','lm/width_model.py','lm/monitor.py'):
   p=out/'runtime-source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
if __name__=='__main__':main()
