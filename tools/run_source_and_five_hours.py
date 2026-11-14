"""Authorized sequential source-family experiment followed by five active training hours."""
from pathlib import Path
from datetime import datetime,timezone
import subprocess,json,sys,statistics,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'results'/('SourceAndFiveHour_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir();(ROOT/'results/SourceAndFiveHour_Current.json').write_text(json.dumps(dict(directory=str(out.relative_to(ROOT))),indent=2));print(out,flush=True)
 checkpoint=ROOT/'results/WidthStudy_20261112T133901595745Z/train192/minute90.pt';expected='1bcc403374cc5045f66fe99dce31f6030829df127a2ab63210a5d18a0774ad09';plan=dict(status='RUNNING',source_runs=[],source_order_seeds=[44,45,46],source_updates=2400,source_initial_checkpoint_sha256=expected,selection_rule='Family weighting useful only if mean both-answerable-pair accuracy gain>=5pp over ordinary, >=2 seeds nonnegative, and mean story-loss increase <=ordinary+0.05 nats. Prefer balanced data if within2pp of best qualifying approach. No statistical-independence claim.',five_hour_minutes=300,five_hour_initialization='Fresh seed42 approved5.1M model; no source instruction weights reused',available_ram_immediate_stop_gib=.5,process_resident_limit_gib=4,final_test_reserved=True)
 def save():(out/'suite.json').write_text(json.dumps(plan,indent=2))
 def run(name,args):
  with (out/(name+'.stdout.log')).open('w') as stdout,(out/(name+'.stderr.log')).open('w') as stderr:cp=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,stdout=stdout,stderr=stderr)
  print(name,'exit',cp.returncode,flush=True);return cp.returncode
 save()
 for rep,seed in enumerate((44,45,46)):
  arms=['ordinary','family','balanced'];arms=arms[rep:]+arms[:rep]
  for arm in arms:
   name=f'source_{arm}_{seed}';code=run(name,[ROOT/'tools/source_family_worker.py','--arm',arm,'--seed',seed,'--checkpoint',checkpoint,'--expected-sha256',expected,'--out',out/name]);r=json.loads((out/name/'run.json').read_text());plan['source_runs'].append(dict(directory=name,arm=arm,seed=seed,status=r['status']));save()
   if code:plan['status']=r['status'];save();return code
 rows={arm:[json.loads((out/f'source_{arm}_{seed}/run.json').read_text()) for seed in (44,45,46)] for arm in ('ordinary','family','balanced')};means={arm:statistics.mean(r['metrics']['both_answerable_pair_correct'] for r in rr) for arm,rr in rows.items()};regress={arm:statistics.mean(r['story_after']['nats_per_target']-r['story_before']['nats_per_target'] for r in rr) for arm,rr in rows.items()};consistent=sum(a['metrics']['both_answerable_pair_correct']>=b['metrics']['both_answerable_pair_correct'] for a,b in zip(rows['family'],rows['ordinary']))
 selected='family' if means['family']>=means['ordinary']+.05 and consistent>=2 and regress['family']<=regress['ordinary']+.05 else 'ordinary'
 if means['balanced']>=means[selected]-.02 and regress['balanced']<=regress[selected]+.05:selected='balanced'
 decision=dict(selected=selected,mean_both_answerable_pair_correct=means,mean_story_loss_increase=regress,family_nonnegative_seeds=consistent,family_benefit_meets_rule=selected=='family',raw_runs=plan['source_runs']);(out/'source-decision.json').write_text(json.dumps(decision,indent=2));(ROOT/'results/SourceFamily_Decision.json').write_text(json.dumps(decision,indent=2));print('SOURCE DECISION',json.dumps(decision),flush=True);plan['source_decision']=decision;save()
 code=run('five_hours',[ROOT/'tools/train_five_hours.py','--width','192','--minutes','300','--out',out/'five_hours']);plan['five_hour_status']=json.loads((out/'five_hours/run.json').read_text())['status'];plan['status']='PRETRAINING_COMPLETE' if code==0 else plan['five_hour_status'];save();return code
if __name__=='__main__':sys.exit(main())
