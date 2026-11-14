"""Package all new experiment reports/logs; list checkpoint blobs separately for LFS."""
from pathlib import Path
import json,zipfile,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 current=ROOT/json.loads((ROOT/'results/SourceAndFiveHour_Current.json').read_text())['directory'];targets=[ROOT/'results/Efficiency_20261113T111156Z',ROOT/'results/Efficiency_20261113T111809Z',current];archive=ROOT/'results/ExperimentArtifacts.zip';entries=[];checkpoints=[]
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for folder in targets:
   for p in sorted(folder.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.suffix=='.tmp':continue
    relative=p.relative_to(ROOT).as_posix()
    if p.suffix=='.pt':checkpoints.append(dict(file=relative,bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest()));continue
    z.write(p,relative);entries.append(dict(file=relative,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  for name in ('SourceFamilies_v1','LocalDialogue_v1','SQuAD2Windows128_v1','SQuAD2Short128_v1'):
   folder=ROOT/'data'/name
   files=list(folder.glob('*.jsonl'))+[folder/'manifest.json'] if name in ('SourceFamilies_v1','LocalDialogue_v1') else [folder/'manifest.json']
   for p in files:
    if p.exists():z.write(p,p.relative_to(ROOT).as_posix());entries.append(dict(file=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 manifest=dict(status='PACKAGED',archive=archive.name,archive_bytes=archive.stat().st_size,archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),entries=entries,checkpoints=checkpoints,corpus_note='Licensed source corpora remain local; manifests, hashes, source URLs and attribution are published. Authored synthetic data included. Checkpoint files are enumerated for Git LFS rather than embedded in this log archive.')
 (ROOT/'results/ExperimentArtifacts_Manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(archive_bytes=manifest['archive_bytes'],report_files=len(entries),checkpoint_files=len(checkpoints),checkpoint_bytes=sum(e['bytes'] for e in checkpoints))))
if __name__=='__main__':main()
