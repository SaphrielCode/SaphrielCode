"""Exercise actual local chat HTTP streaming, checkpoint and stop behavior."""
import json,subprocess,sys,time,urllib.request,urllib.error
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[1]
def main():
 process=subprocess.Popen([sys.executable,str(ROOT/'tools/chat_server.py'),'--port','0'],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 checks=[]
 try:
  url=process.stdout.readline().strip();assert url.startswith('http://127.0.0.1:');url=url.rstrip('/')
  def request(path,payload=None,origin=None):
   headers={'Content-Type':'application/json'}
   if origin:headers['Origin']=origin
   return urllib.request.urlopen(urllib.request.Request(url+path,data=None if payload is None else json.dumps(payload).encode(),headers=headers),timeout=60)
  def read(path):return json.load(request(path))
  for i in range(180):
   status=read('/api/status')
   if status['state']=='ready':break
   if status['state']=='error':raise RuntimeError(status['error'])
   time.sleep(.25)
  models=read('/api/models');assert status['state']=='ready' and status['selected']==models['default'];default_id=status['selected'];checks.append('Default verified checkpoint loads');assert models['models'];checks.append('Known trained checkpoints listed')
  deployment=ROOT/'model_assets/deployment.json'
  if deployment.exists():
   entry=next(p for p in json.loads(deployment.read_text())['profiles'] if p['id']==default_id);reference=entry['regression']
   with request('/api/generate',dict(messages=[dict(role='user',content=reference['prompt'])],mode=reference['mode'],temperature=0,max_tokens=reference['max_tokens'])) as response:new_events=[json.loads(line) for line in response if line.strip()]
   assert new_events[-1]['text']==reference['expected_text'];checks.append('New deployed checkpoint streaming matches its independently recorded greedy continuation')
  if 'instruction-192' in {p['id'] for p in models['models']}:
   json.load(request('/api/load',dict(model='instruction-192')))
   for _ in range(180):
    status=read('/api/status')
    if status['state']=='ready':break
    time.sleep(.25)
  if 'instruction-192' in {p['id'] for p in models['models']}:
   study=ROOT/json.loads((ROOT/'results/WidthStudy_Current.json').read_text())['directory'];example=json.loads((study/'instruction192/after-responses.json').read_text())[38]
   payload=dict(messages=[dict(role='user',content=example['instruction'])],mode='chat',temperature=0,max_tokens=16)
   with request('/api/generate',payload) as response:events=[json.loads(line) for line in response if line.strip()]
   assert events[0]['type']=='start' and events[-1]['type']=='done';assert events[-1]['text']==example['generated_text'] and events[-1]['text']=='garden';checks.append('Streamed greedy answer exactly matches recorded checkpoint response')
  with request('/api/generate',dict(messages=[dict(role='user',content='Once upon a time. '*100)],mode='story',temperature=0,max_tokens=128)) as response:
   first=json.loads(response.readline());assert first['context_trimmed'] and first['used_prompt_tokens']==128
   json.load(request('/api/stop',{}));rest=[json.loads(line) for line in response if line.strip()]
  assert rest[-1]['type']=='done' and rest[-1]['stopped'] and rest[-1]['tokens']<128;checks.append('Long context cropped and stop generation honored')
  try:request('/api/load',dict(model='../../outside.pt'))
  except urllib.error.HTTPError as e:assert e.code==400
  else:raise AssertionError('Invalid checkpoint accepted')
  try:request('/api/stop',{},origin='https://example.com')
  except urllib.error.HTTPError as e:assert e.code==403
  else:raise AssertionError('Cross-origin request accepted')
  checks.append('Invalid checkpoint selection and cross-origin requests rejected')
  result=dict(status='PASS',checks=checks,default_model=default_id,reference_response=events[-1]['text'] if 'events' in locals() else new_events[-1]['text'],messages_persisted=False)
  (ROOT/'results/Chat_Verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 finally:
  try:
   parent=psutil.Process(process.pid);children=parent.children(recursive=True)
   for child in children:child.terminate()
   parent.terminate();psutil.wait_procs(children+[parent],timeout=5)
  except psutil.NoSuchProcess:pass
if __name__=='__main__':main()
