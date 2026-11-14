"""Continue the accepted efficiency plan, reusing completed measurements only."""
from pathlib import Path
import sys,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/json.loads((ROOT/'results/Efficiency_Current.json').read_text())['directory']
plan=json.loads((out/'suite.json').read_text());plan['status']='RUNNING';plan['continuation_note']='Reuse21 completed screens and seed42 valid training after metadata-only checkpoint repair. No learned weights reused as initialization for other runs.';plan['continuation_source_sha256']=hashlib.sha256((ROOT/'tools/run_efficiency.py').read_bytes()).hexdigest()
for row in plan['runs']:
 row['status']=json.loads((out/row['directory']/'run.json').read_text())['status']
def save():(out/'suite.json').write_text(json.dumps(plan,indent=2))
save()
for rep,seed in enumerate((42,44,46)):
 variants=['valid','factor96','factor96wide288'];variants=variants[rep:]+variants[:rep]
 for v in variants:
  name=f'quality_{v}_seed{seed}';path=out/name
  if path.exists():
   r=json.loads((path/'run.json').read_text())
   if r['status']=='PASS':print('Reused verified completed measurement:',name,flush=True);continue
   raise RuntimeError('Refusing to overwrite incomplete run '+name)
  with (out/(name+'.stdout.log')).open('w') as stdout,(out/(name+'.stderr.log')).open('w') as stderr:
   cp=subprocess.run([sys.executable,str(ROOT/'tools/run_efficiency.py'),'--worker','--variant',v,'--stage','quality','--seconds','600','--seed',str(seed),'--out',str(path)],cwd=ROOT,stdout=stdout,stderr=stderr)
  r=json.loads((path/'run.json').read_text());plan['runs'].append(dict(directory=name,variant=v,stage='quality',status=r['status']));save();print(name,r['status'],r.get('validation',{}).get('final',{}).get('bits_per_utf8_text_byte_including_eos'),flush=True)
  if cp.returncode:plan['status']='STOPPED' if r['status']=='STOPPED' else 'ERROR';save();sys.exit(2)
plan['status']='PASS';save()
