"""Restore published binary artifact parts, checking every hash before replacement."""
from pathlib import Path
import json,hashlib,shutil,argparse
ROOT=Path(__file__).resolve().parents[1]
def inside(path):
 p=(ROOT/path).resolve()
 if not p.is_relative_to(ROOT):raise ValueError('Artifact path escapes project')
 return p
def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def restore(entry):
 dest=inside(entry['file'])
 if dest.exists():
  if sha(dest)==entry['sha256']:return False
  raise FileExistsError('Refusing to replace existing different artifact: '+entry['file'])
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_name(dest.name+'.restore.tmp')
 if tmp.exists():raise FileExistsError('Previous restoration temporary exists: '+str(tmp))
 try:
  with tmp.open('wb') as target:
   for part in entry['parts']:
    source=inside(part['path'])
    if source.stat().st_size!=part['bytes'] or sha(source)!=part['sha256']:raise ValueError('Artifact part hash/size mismatch: '+part['path'])
    with source.open('rb') as f:shutil.copyfileobj(f,target,1024*1024)
  if tmp.stat().st_size!=entry['bytes'] or sha(tmp)!=entry['sha256']:raise ValueError('Restored artifact hash mismatch')
  tmp.replace(dest);return True
 except Exception:
  if tmp.exists():tmp.unlink()
  raise
def main():
 p=argparse.ArgumentParser();p.add_argument('--all',action='store_true');a=p.parse_args()
 if any(x.lower()=='onedrive' for x in ROOT.parts):raise SystemExit('Restore in a local project outside OneDrive')
 manifest=ROOT/'artifacts/ArtifactManifest.json'
 if not manifest.exists():
  deployment=ROOT/'model_assets/deployment.json'
  if not deployment.exists():return
  d=json.loads(deployment.read_text());selection=json.loads((ROOT/'WorkingSelection.json').read_text());r=dict(files=[],tokenizer=dict(published_file=d['tokenizer_file'],runtime_file=selection['tokenizer_file'],sha256=d['tokenizer_sha256']))
 else:r=json.loads(manifest.read_text(encoding='utf-8'))
 count=0
 for entry in r['files']:
  if a.all or entry.get('needed_for_chat'):count+=int(restore(entry))
 codec=r.get('tokenizer')
 if codec:
  source=inside(codec['published_file']);dest=inside(codec['runtime_file'])
  if sha(source)!=codec['sha256']:raise ValueError('Published tokenizer hash mismatch')
  if dest.exists():
   if sha(dest)!=codec['sha256']:raise FileExistsError('Existing tokenizer differs; refusing overwrite')
  else:dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
 print('Verified/restored',count,'artifacts. Use --all for experiment checkpoints and log archive.')
if __name__=='__main__':main()
