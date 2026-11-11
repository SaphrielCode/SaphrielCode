"""Copy this workspace outside OneDrive with streamed content verification."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).resolve().parents[1]
    target=args.destination.resolve()
    if target.exists() or target.is_relative_to(source) or source.is_relative_to(target):
        raise RuntimeError('Destination must be a new, separate directory')
    if any(part.lower().startswith('onedrive') for part in target.parts):
        raise RuntimeError('Destination must be outside OneDrive')
    files=[]; total=0
    for current,dirs,names in os.walk(source,followlinks=False):
        for name in dirs+names:
            path=Path(current)/name
            if path.is_symlink() or path.is_junction():
                raise RuntimeError('Refusing to follow filesystem link: '+str(path))
        for name in names:
            path=Path(current)/name
            files.append(path.relative_to(source)); total+=path.stat().st_size
    if shutil.disk_usage(target.parent).free<total+512*1024**2:
        raise RuntimeError('Insufficient disk space for complete verified copy')
    target.mkdir()
    receipt=dict(source=str(source),destination=str(target),started_utc=datetime.now(timezone.utc).isoformat(),
                 source_unchanged=True,planned_files=len(files),planned_bytes=total,files={})
    copied=0
    for number,relative in enumerate(files,1):
        src=source/relative; dst=target/relative; dst.parent.mkdir(parents=True,exist_ok=True)
        before=src.stat(); digest=hashlib.sha256()
        with src.open('rb') as inp,dst.open('xb') as output:
            while data:=inp.read(1024**2):
                output.write(data); digest.update(data)
        after=src.stat()
        if before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns:
            raise RuntimeError('Source changed during copy: '+str(relative))
        with dst.open('rb') as check:
            if hashlib.file_digest(check,'sha256').digest()!=digest.digest():
                raise RuntimeError('Destination hash mismatch: '+str(relative))
        shutil.copystat(src,dst)
        receipt['files'][relative.as_posix()]=dict(bytes=before.st_size,sha256=digest.hexdigest())
        copied+=before.st_size
        if number%1000==0:
            print(json.dumps(dict(verified_files=number,total_files=len(files),copied_bytes=copied)),flush=True)
    # Rebuild entry-point launchers instead of retaining embedded old venv paths.
    scripts=target/'.venv-data/Scripts'
    python=scripts/'python.exe'
    for path in [target/'.venv-data/pyvenv.cfg',scripts/'activate',scripts/'activate.bat',scripts/'activate.fish',scripts/'Activate.ps1']:
        if path.exists():
            text=path.read_text(encoding='utf-8')
            text=text.replace(str(source),str(target)).replace(source.as_posix(),target.as_posix())
            path.write_text(text,encoding='utf-8')
    repair="""import importlib.metadata,sys,json
from pip._vendor.distlib.scripts import ScriptMaker
from pathlib import Path
scripts=Path(sys.executable).parent
maker=ScriptMaker(None,str(scripts)); maker.executable=sys.executable; maker.clobber=True
created=[]
for dist in importlib.metadata.distributions():
 for entry in dist.entry_points:
  if entry.group=='console_scripts': created+=maker.make(entry.name+' = '+entry.value)
created+=maker.make('pip3 = pip._internal.cli.main:main')
created+=maker.make('pip3.13 = pip._internal.cli.main:main')
print(json.dumps({'prefix':sys.prefix,'generated_entry_points':len(created)}))
"""
    completed=subprocess.run([str(python),'-c',repair],cwd=target,capture_output=True,text=True)
    if completed.returncode:
        raise RuntimeError('Venv launcher repair failed: '+completed.stderr)
    receipt['venv_repair']=json.loads(completed.stdout)
    receipt['intentional_post_copy_changes']='Activation/config paths and installed console-script launchers regenerated for the new local environment; archived historical experiment paths left unchanged.'
    (target/'SaphrielComDevice.code-workspace').write_text(json.dumps(dict(folders=[dict(path='.')],settings={}),indent=2)+'\n')
    receipt.update(status='VERIFIED_COPY',verified_files=len(files),verified_bytes=copied,finished_utc=datetime.now(timezone.utc).isoformat())
    path=target/'results/Relocation_Receipt.json'
    path.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (target/'results/Relocation_Receipt.txt').write_text('WORKSPACE RELOCATION\n\nSource: '+str(source)+'\nDestination: '+str(target)+f'\nVerified files: {len(files)}\nVerified bytes: {copied}\nAll copied file contents matched SHA256 before environment path repairs. Original source preserved.\nPython environment entry points and activation paths repaired for the local destination.\nHistorical report/checkpoint provenance paths remain unchanged.\nNo training launched as part of relocation.\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2),flush=True)


if __name__=='__main__':
    main()
