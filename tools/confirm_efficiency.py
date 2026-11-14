"""Longer rotated confirmation of same-architecture training implementations."""
from pathlib import Path
import sys,json,subprocess,hashlib,time,statistics
ROOT=Path(__file__).resolve().parents[1];out=ROOT/json.loads((ROOT/'results/Efficiency_Current.json').read_text())['directory'];plan=dict(status='RUNNING',seconds=60,repetitions=3,variants=['baseline','valid','trim','sdpa','combined'],runs=[],source_sha256=hashlib.sha256((ROOT/'tools/run_efficiency.py').read_bytes()).hexdigest(),selection_rule='Fastest repeated median; gains under5% are treated as practically close. For close eligible paths prefer lower peak RAM, with uncertainty disclosed. No automatic production architecture change.')
def save():(out/'confirmation.json').write_text(json.dumps(plan,indent=2))
save()
for rep in range(3):
 variants=plan['variants'][rep:]+plan['variants'][:rep]
 for v in variants:
  name=f'confirm{rep+1}_{v}';path=out/name
  with (out/(name+'.stdout.log')).open('w') as stdout,(out/(name+'.stderr.log')).open('w') as stderr:
   cp=subprocess.run([sys.executable,str(ROOT/'tools/run_efficiency.py'),'--worker','--variant',v,'--stage','screen','--seconds','60','--out',str(path)],cwd=ROOT,stdout=stdout,stderr=stderr)
  r=json.loads((path/'run.json').read_text());plan['runs'].append(dict(directory=name,variant=v,status=r['status']));save();print(name,r['status'],r.get('valid_targets_per_second'),flush=True)
  if cp.returncode:plan['status']=r['status'];save();sys.exit(2)
  time.sleep(2)
plan['status']='PASS';save()
