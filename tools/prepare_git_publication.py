"""Build a byte-exact GitHub snapshot plan without embedding large tool arguments."""
from pathlib import Path
import json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1];CHUNK=4*1024**2

def main():
 study=ROOT/json.loads((ROOT/'results/SourceAndFiveHour_Current.json').read_text())['directory'];deployment=json.loads((ROOT/'model_assets/deployment.json').read_text());assert deployment['status']=='PASS'
 attrs=ROOT/'.gitattributes';text=attrs.read_text() if attrs.exists() else ''
 for line in ('artifacts/checkpoint-parts/** -text -diff','artifacts/log-parts/** -text -diff','model_assets/tokenizer.json -text'):
  if line not in text:text+=line+'\n'
 attrs.write_text(text);blobs={};tree={};specs=[]
 def digest(p):
  with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
 def blob(p,offset,size):
  with p.open('rb') as f:f.seek(offset);data=f.read(size)
  sha=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest();entry=dict(sha=sha,local_file=p.relative_to(ROOT).as_posix(),offset=offset,bytes=len(data),sha256=hashlib.sha256(data).hexdigest());blobs.setdefault(sha,entry);return entry
 def add_file(p):
  if p.is_symlink() or not p.resolve().is_relative_to(ROOT):raise ValueError('Refusing external/symlink publication path')
  entry=blob(p,0,p.stat().st_size)
  if entry['bytes']>CHUNK:raise ValueError('Unexpected large ordinary publication file: '+str(p))
  tree[p.relative_to(ROOT).as_posix()]=dict(path=p.relative_to(ROOT).as_posix(),mode='100644',type='blob',sha=entry['sha'])
 def add_parts(p,kind,needed=False):
  parts=[]
  for offset in range(0,p.stat().st_size,CHUNK):
   entry=blob(p,offset,min(CHUNK,p.stat().st_size-offset));target=f'artifacts/{kind}-parts/{entry["sha"]}.bin';tree[target]=dict(path=target,mode='100644',type='blob',sha=entry['sha']);parts.append(dict(path=target,bytes=entry['bytes'],sha256=entry['sha256']))
  specs.append(dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=digest(p),parts=parts,needed_for_chat=needed))
 for name in sorted({deployment['best_story_validation_label']+'.pt','minute300.pt'}):add_parts(study/'five_hours'/name,'checkpoint')
 for p in sorted((ROOT/'model_assets').glob('*.pt')):add_parts(p,'checkpoint',True)
 add_parts(ROOT/'results/ExperimentArtifacts.zip','log')
 if (ROOT/'results/LocalHistory.bundle').exists():add_parts(ROOT/'results/LocalHistory.bundle','log')
 artifacts=ROOT/'artifacts';artifacts.mkdir(exist_ok=True);selection=json.loads((ROOT/'WorkingSelection.json').read_text());manifest=dict(format='saphriel-artifact-parts-v1',chunk_bytes=CHUNK,files=specs,tokenizer=dict(published_file='model_assets/tokenizer.json',runtime_file=selection['tokenizer_file'].replace('\\','/'),sha256=selection['tokenizer_sha256']),scope='Selected and final five-hour full optimizer checkpoints plus deployed inference weights; all study and prior efficiency reports/logs packaged. All intermediate source/QA/training checkpoints remain local with hashes in the inventory. Published weights are byte-exact.',restore='RunChat.cmd restores deployment weights and tokenizer automatically. tools/restore_published_artifacts.py --all restores all listed checkpoints and the log archive.')
 (artifacts/'ArtifactManifest.json').write_text(json.dumps(manifest,indent=2));(artifacts/'README.txt').write_text('Published binary results are byte-exact4MiB parts, streamed through the authenticated GitHub app.\nRun .venv-data\\Scripts\\python.exe tools\\restore_published_artifacts.py --all to restore checkpoints and results/ExperimentArtifacts.zip.\nUse python -m zipfile -e results/ExperimentArtifacts.zip . to unpack raw reports/logs.\nRunChat.cmd restores only the selected inference weights and tokenizer automatically.\nEvery part and reassembled file is SHA-256 checked; existing different files are never overwritten.\nDataset source corpora remain local; source URLs, hashes, licenses and selected synthetic data are included in the results archive.\n')
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0');others=subprocess.check_output(['git','ls-files','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0');paths=set(tracked)
 for name in others:
  if name.startswith(('lm/','tools/','tests/','chat/','corpus-references/','model_assets/','artifacts/')) or name in ('.gitattributes','RunChat.cmd','ChatInstructions.txt','WorkingSelection.json'):paths.add(name)
  elif name.startswith('results/') and '/' not in name[len('results/'):] and name.endswith(('.txt','.json','.csv')) and not any(x in name for x in ('GitPublication','GitBlob','Prepared','Receipt','LocalEnvironment')):paths.add(name)
 for name in sorted(paths):
  p=ROOT/name
  if not name or not p.is_file() or p.suffix in ('.pt','.zip','.pyc') or '__pycache__' in p.parts:continue
  add_file(p)
 base=subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT).decode().strip();deleted=subprocess.check_output(['git','diff','--name-only','--diff-filter=D','origin/main','--'],cwd=ROOT).decode().splitlines()
 for name in deleted:tree[name]=dict(path=name,mode='100644',type='blob',sha=None)
 plan=dict(expected_remote_base=base,repository='SaphrielCode/SaphrielCode',branch='main',root=str(ROOT),blobs=list(blobs.values()),tree_elements=list(tree.values()),local_commit_paths=sorted(p for p in tree if not p.startswith(('artifacts/checkpoint-parts/','artifacts/log-parts/'))),artifact_manifest='artifacts/ArtifactManifest.json',binary_files=len(specs),binary_bytes=sum(x['bytes'] for x in specs))
 (ROOT/'results/GitPublicationPlan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(dict(unique_blobs=len(blobs),tree_entries=len(tree),binary_files=len(specs),binary_bytes=plan['binary_bytes'])))
if __name__=='__main__':main()
