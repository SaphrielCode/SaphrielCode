"""Export current best story and eligible QA weights for verified local deployment."""
from pathlib import Path
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
 import torch
 from lm.experimental_model import Config,parameter_hash
 from lm.efficiency import EfficientDecoder,RollingCache
 from tokenizers import Tokenizer
 torch.set_num_threads(8);torch.set_num_interop_threads(1)
 from lm.batches import digest
 out=ROOT/json.loads((ROOT/'results/SourceAndFiveHour_Current.json').read_text())['directory'];r=json.loads((out/'five_hours/run.json').read_text());assert r['status']=='PASS' and r['training_seconds']>=18000
 best=min((key for key in r['validation'] if key!='initialization'),key=lambda key:r['validation'][key]['loss_nats_per_target']);entry=next(e for e in r['checkpoints'] if e['file']==best+'.pt');assert entry['reload_verified'];candidates=[('latest-story',out/'five_hours'/entry['file'],'story','5.1M - five-hour story model',entry['sha256'])]
 qa=out/'grounded_qa/run.json'
 if qa.exists():
  q=json.loads(qa.read_text())
  if q['status']=='PASS' and q.get('selected_epoch'):
   e=next(e for e in q['epochs'] if e['epoch']==q['selected_epoch']);candidates.append(('latest-instruction',qa.parent/e['checkpoint']['file'],'chat','5.1M - grounded passage QA',e['checkpoint']['sha256']))
 assets=ROOT/'model_assets';assets.mkdir(exist_ok=True);profiles=[];selection=json.loads((ROOT/'WorkingSelection.json').read_text())
 for identifier,source,style,title,expected in candidates:
  assert digest(source)==expected;saved=torch.load(source,weights_only=True,mmap=True);cfg=Config(**saved['config']);model=EfficientDecoder(cfg);model.load_state_dict(saved['model']);h=parameter_hash(model);assert h==saved['parameter_sha256'] and sum(p.numel() for p in model.parameters())==5115072;assert all(torch.isfinite(p).all() for p in model.parameters());name='story.pt' if style=='story' else 'grounded_qa.pt';dest=assets/name
  if dest.exists():raise FileExistsError('Preserve existing model asset: '+str(dest))
  torch.save(dict(config=saved['config'],model=model.state_dict(),parameter_sha256=h,tokenizer_sha256=selection['tokenizer_sha256']),dest);restored=torch.load(dest,weights_only=True,mmap=True);fresh=EfficientDecoder(cfg);fresh.load_state_dict(restored['model']);assert parameter_hash(fresh)==h
  tokenizer=Tokenizer.from_file(str(ROOT/selection['tokenizer_file']));prompt='Once upon a time, a little girl found a' if style=='story' else 'Read the passage. Answer briefly, or unknown if not stated.\nPassage: Lily left the red ball in the garden.\nQuestion: Where is the ball?';text=prompt if style=='story' else 'Instruction:\n'+prompt+'\nResponse:\n';ids=[1]+tokenizer.encode(text).ids;generated=[];model.eval();cache=RollingCache(model)
  with torch.inference_mode():
   for _ in range(16):
    ix=torch.tensor([ids]);scores=cache(ix)[0] if len(ids)<=128 else model(ix[:,-128:],last_only=True)[0];scores=scores.clone();scores[[0,1,3]]=-float('inf');n=int(scores.argmax());ids.append(n);generated.append(n)
    if n==2:break
  regression=dict(prompt=prompt,mode=style,temperature=0,max_tokens=16,expected_text=tokenizer.decode(generated,skip_special_tokens=True),expected_ids=generated)
  profiles.append(dict(id=identifier,name=title,detail='TinyStories continuation; experimental limited model.' if style=='story' else 'Answers short passage questions; experimental, with heldout and story-preservation checks.',checkpoint=dest.relative_to(ROOT).as_posix(),file_hash=digest(dest),parameter_hash=h,parameters=5115072,style=style,regression=regression,training_checkpoint=source.relative_to(ROOT).as_posix(),training_checkpoint_sha256=expected));del saved,restored,model,fresh
 codec=assets/'tokenizer.json'
 if codec.exists() and digest(codec)!=selection['tokenizer_sha256']:raise FileExistsError('Existing published tokenizer differs')
 if not codec.exists():codec.write_bytes((ROOT/selection['tokenizer_file']).read_bytes())
 assert digest(codec)==selection['tokenizer_sha256'];deploy=dict(status='PASS',profiles=profiles,default='latest-instruction' if len(profiles)>1 else 'latest-story',tokenizer_file=codec.relative_to(ROOT).as_posix(),tokenizer_sha256=selection['tokenizer_sha256'],best_story_validation_label=best,study_directory=out.relative_to(ROOT).as_posix(),model_quality_limits='Tiny story model and bounded passage QA, not a general conversational benchmark.')
 (assets/'deployment.json').write_text(json.dumps(deploy,indent=2));selection.update(reference_checkpoint=profiles[0]['checkpoint'],reference_checkpoint_sha256=profiles[0]['file_hash'],reference_parameter_sha256=profiles[0]['parameter_hash'],reference_checkpoint_usage='Verified best-validation model from five active hours of fresh TinyStories training. Raw optimizer checkpoints retained separately; fresh experiments still start seed42.',training_resume_checkpoint=profiles[0]['training_checkpoint'],deployment_manifest='model_assets/deployment.json');(ROOT/'WorkingSelection.json').write_text(json.dumps(selection,indent=2));(out/'deployment.json').write_text(json.dumps(deploy,indent=2));print(json.dumps(deploy,indent=2))
if __name__=='__main__':main()
