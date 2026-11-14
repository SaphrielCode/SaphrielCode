"""Execute approved architecture experiments sequentially, stopping on guards/failures."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,subprocess,sys,json,time,hashlib,shutil,math,statistics,struct
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
def verify_training(directory,variant):
 r=json.loads((directory/'run.json').read_text());assert r['status']=='PASS' and r['completed_passes']==2 and not r['final_test_evaluated'];assert r['valid_targets_processed']==8777642
 batch=ROOT/r['batch_directory'];metadata=[x[2] for x in struct.iter_unpack('<III',(batch/'train.rows.u32').read_bytes())];order=[x[0] for x in struct.iter_unpack('<I',(batch/'train.order.u32').read_bytes())];expected=[sum(metadata[k] for k in order[i:i+4]) for i in range(0,len(order),4)]
 assert len(expected)==r['expected_updates_per_pass'] and sum(expected)==4388821
 for n in (1,2):
  with (directory/f'pass{n}-steps.jsonl').open() as f:steps=[json.loads(s) for s in f]
  assert [s['update'] for s in steps]==list(range((n-1)*len(expected)+1,n*len(expected)+1));assert [s['valid_targets'] for s in steps]==expected;assert all(math.isfinite(s['loss_sum']) and math.isfinite(s['gradient_norm']) for s in steps)
 assert r['completed_updates']==2*len(expected)
 for c in r['checkpoints']:
  p=directory/c['file'];assert c['reload_verified'] and sha(p)==c['sha256'] and p.stat().st_size==c['bytes']
 for name,h in r['source_hashes'].items():
  assert sha(ROOT/name)==h;p=directory/'runtime-source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
 with (directory/'telemetry.jsonl').open() as f:
  low=high=0
  for line in f:
   s=json.loads(line);assert s['guard_stop'] is None and s['available_system_bytes']>=2**30;low=low+1 if s['available_system_bytes']<2*2**30 else 0;high=high+1 if s['resident_bytes']>4*2**30 else 0;assert low<3 and high<3
 baseline=json.loads((ROOT/'results/Unigram8192_TwoPass_Summary.json').read_text());textbytes=baseline['validation_utf8_text_bytes'];assert textbytes==884630
 for label,v in r['validation'].items():assert v['valid_targets']==216150;v['bits_per_utf8_text_byte_including_eos_cost']=v['loss_sum']/textbytes/math.log(2)
 r['training_seconds']=sum(p['seconds'] for p in r['passes']);r['validation_utf8_text_bytes']=textbytes
 primary='common128_pass_2' if variant=='context256' else 'pass_2'
 r['common128_final_change_vs_working_baseline_percent']=(r['validation'][primary]['bits_per_utf8_text_byte_including_eos_cost']/baseline['validation']['pass_2']['bits_per_utf8_text_byte_including_eos_cost']-1)*100
 r['verification']='PASS: exact update/target coverage; finite step losses and norms; checkpoint hashes and runtime reloads; source snapshots; guard telemetry. Test split reserved.'
 save(directory/'verified-summary.json',r)
 lines=[f'ARCHITECTURE EXPERIMENT: {variant}',r['verification'],'',json.dumps(r['model_config']),f"Parameters: {r['unique_parameters']}; updates: {r['completed_updates']}; valid targets: {r['valid_targets_processed']}; fresh seed42, order43; two complete passes; no extra warmup.",f"Training: {r['training_seconds']:.3f}s; validation: {r['validation_seconds']:.3f}s; checkpoint: {r['checkpoint_seconds']:.3f}s. Periodic recovery I/O is already included in pass wall times; do not add that I/O again.",f"Peak process resident: {r['peak_process_resident_bytes']/2**20:.2f} MiB; minimum available: {r['minimum_available_system_bytes']/2**30:.3f} GiB.",'']
 for label,v in r['validation'].items():lines.append(f"{label}: {v['loss_nats_per_target']:.9f} nats/target; {v['bits_per_utf8_text_byte_including_eos_cost']:.9f} bits/byte; NLL sum {v['loss_sum']:.9f}; count {v['valid_targets']}.")
 for p in r['passes']:lines.append(f"Pass {p['pass_number']}: {p['seconds']:.3f}s; {p['valid_targets_per_second']:.3f} valid targets/s.")
 lines += ['',f"Common128 final bits/byte change versus working unigram GELU baseline: {r['common128_final_change_vs_working_baseline_percent']:+.3f}% (lower is better).",'For context256, native256 and common128 predict identical target tokens with different histories/chunk boundaries. Use common128 for matching evaluation conditioning. Training windows/order still differ. Other variants use unchanged128 windows. One seed; background workloads differ; neither elapsed time nor parameter count establishes equal total compute. Shared architectural tensor values are not assumed identical across changed shapes. Baseline working selection is unchanged.']
 for c in r['continuations']:lines+=['',f"Pass {c['pass_number']} prompt: {c['prompt']}",c['decoding'],c['continuation']]
 lines+=['',f'Run artifacts: {directory.relative_to(ROOT)}']
 text='\n'.join(lines)+'\n';(directory/'VerifiedFindings.txt').write_text(text,encoding='utf-8');(ROOT/'results'/f'Architecture_{variant}_Results.txt').write_text(text,encoding='utf-8');save(ROOT/'results'/f'Architecture_{variant}_Summary.json',r)
 return r

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 suite=dict(status='RUNNING',order=['swiglu128','context256','deep12x160'],experiments=[],working_selection_unchanged=True,policy='2GiB reserve for3 samples; first sample below1GiB; actual trainer RSS above4GiB for3; no retries or relaxation',validation='same frozen216150 targets; context256 also evaluated on common128 windows',initialization_seed=42,order_seed=43)
 save(out/'suite.json',suite)
 try:
  for variant in suite['order']:
   item=dict(variant=variant,screens=[],status='SCREENING');suite['experiments'].append(item);save(out/'suite.json',suite)
   for repeat in range(1,4):
    screen=out/f'{variant}_screen{repeat}';cmd=[sys.executable,str(ROOT/'tools/screen_architecture.py'),'--variant',variant,'--out',str(screen)]
    with (out/f'{variant}_screen{repeat}.stdout.txt').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    report=json.loads((screen/'screen.json').read_text());item['screens'].append(report);save(out/'suite.json',suite)
    if result.returncode or report['status']!='PASS':raise RuntimeError(f'{variant} screen stopped/failed; no further run launched')
    print(json.dumps(dict(variant=variant,screen=repeat,tokens_per_second=report['valid_targets_per_second'])),flush=True)
    if repeat<3:time.sleep(5)
   assert len({s['initial_parameter_sha256'] for s in item['screens']})==1
   item['median_screen_tokens_per_second']=statistics.median(s['valid_targets_per_second'] for s in item['screens']);item['status']='TRAINING';save(out/'suite.json',suite)
   cmd=[sys.executable,str(ROOT/'tools/train_architecture.py'),'--variant',variant,'--reserve-gib','2','--tokenizer-kind','unigram','--vocab-size','8192','--track-app-memory']
   with (out/f'{variant}_training.stdout.txt').open('w') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
    directory=None
    for line in child.stdout:
     log.write(line);log.flush();print(line,end='',flush=True)
     possible=Path(line.strip())
     if line.strip().startswith(str(ROOT/'results')):directory=possible;item['run_directory']=str(directory.relative_to(ROOT));save(out/'suite.json',suite)
    code=child.wait()
   if code or directory is None:raise RuntimeError(f'{variant} training stopped/failed; no further run launched')
   r=verify_training(directory,variant);assert r['initial_parameter_sha256']==item['screens'][0]['initial_parameter_sha256'];item.update(status='PASS',final_common128_change_percent=r['common128_final_change_vs_working_baseline_percent']);save(out/'suite.json',suite)
  suite['status']='PASS'
 except Exception as error:suite.update(status='STOPPED_OR_FAILED',error=repr(error));raise
 finally:
  save(out/'suite.json',suite)
  (ROOT/'results/ArchitectureSuite_Status.txt').write_text(json.dumps(suite,indent=2)+'\n')
if __name__=='__main__':main()
