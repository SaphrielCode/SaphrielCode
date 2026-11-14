"""Local-only Saphriel checkpoint chat. No training and no message persistence."""
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from dataclasses import dataclass,asdict
from urllib.parse import urlparse
import argparse,gc,hashlib,json,math,os,sys,threading,time,webbrowser
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
ASSETS=ROOT/'chat'
APP_ID='saphriel-local-chat-v1'

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def inside(path,parent):
 p=path.resolve()
 if not p.is_relative_to(parent.resolve()):raise ValueError('Checkpoint must remain inside local results')
 return p
@dataclass
class Profile:
 id:str
 name:str
 detail:str
 checkpoint:Path
 file_hash:str
 parameter_hash:str
 parameters:int
 style:str
 def public(self):return dict(id=self.id,name=self.name,detail=self.detail,parameters=self.parameters,style=self.style)

def default_profile(profiles):
 return next((p for p in ('latest-instruction','latest-story','instruction-192') if p in profiles),next(iter(profiles),None))

def discover(root=ROOT):
 selection=read(root/'WorkingSelection.json');profiles=[]
 deployment=root/'model_assets/deployment.json'
 if deployment.exists():
  deployed=read(deployment)
  if deployed.get('status')!='PASS' or deployed['tokenizer_sha256']!=selection['tokenizer_sha256']:raise ValueError('Unverified deployment manifest')
  for p in deployed['profiles']:
   checkpoint=inside(root/p['checkpoint'],root/'model_assets')
   if checkpoint.exists():profiles.append(Profile(p['id'],p['name'],p['detail'],checkpoint,p['file_hash'],p['parameter_hash'],p['parameters'],p['style']))
 current=root/'results/WidthStudy_Current.json'
 if current.exists():
  study=inside(root/read(current)['directory'],root/'results')
  for width,label in ((192,'5.1M'),(288,'10.3M'),(384,'17.3M')):
   for kind,suffix,title,detail in (('instruction','instruction.pt','instruction test','Simple instruction practice; can still give incorrect answers.'),('train','minute90.pt','story model','Trained to continue TinyStories text.')):
    folder=study/f'{kind}{width}';report=folder/'run.json';checkpoint=folder/suffix
    if not report.exists() or not checkpoint.exists():continue
    r=read(report)
    if r.get('status')!='PASS':continue
    if kind=='instruction':file_hash=r['checkpoint_sha256'];parameter_hash=r['final_parameter_sha256']
    else:
     entry=next((x for x in r['checkpoints'] if x['file']==suffix and x['reload_verified']),None)
     if not entry:continue
     file_hash=entry['sha256'];parameter_hash=entry['parameter_sha256']
    profiles.append(Profile(f'{kind}-{width}',f'{label} · {title}',detail,inside(checkpoint,root/'results'),file_hash,parameter_hash,8192*width+8*(12*width*width+2*width)+width,'chat' if kind=='instruction' else 'story'))
 reference=(root/selection['reference_checkpoint']).resolve()
 checkpoint=inside(reference,root/'model_assets') if reference.is_relative_to((root/'model_assets').resolve()) else inside(reference,root/'results')
 if checkpoint.exists():profiles.append(Profile('approved-reference','5.1M · approved reference','Original two-pass SwiGLU checkpoint.',checkpoint,selection['reference_checkpoint_sha256'],selection['reference_parameter_sha256'],selection['unique_parameters'],'story'))
 return selection,{p.id:p for p in profiles}

def prompt_text(messages,mode):
 if not isinstance(messages,list) or not messages or len(messages)>40:raise ValueError('Send between1 and40 messages')
 clean=[]
 for x in messages:
  if not isinstance(x,dict) or x.get('role') not in ('user','assistant') or not isinstance(x.get('content'),str):raise ValueError('Invalid conversation')
  if len(x['content'])>6000:raise ValueError('Please shorten the message to6000 characters')
  clean.append(dict(role=x['role'],content=x['content']))
 if clean[-1]['role']!='user' or not clean[-1]['content'].strip():raise ValueError('Enter a message first')
 if mode=='story':return clean[-1]['content']
 if mode!='chat':raise ValueError('Choose chat or story mode')
 text=''
 for x in clean:
  text+=('Instruction:\n'+x['content']+'\nResponse:\n') if x['role']=='user' else x['content']+'\n'
 return text

def settings(body):
 maximum=body.get('max_tokens',64);temperature=body.get('temperature',0.7)
 if type(maximum)!=int or not 1<=maximum<=128:raise ValueError('Reply length must be1 to128 tokens')
 if type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0<=temperature<=1.5:raise ValueError('Temperature must be0 to1.5')
 return maximum,float(temperature)

class Engine:
 def __init__(self):
  self.selection,self.profiles=discover();self.lock=threading.Lock();self.stop=threading.Event();self.state='empty';self.error=None;self.selected=None;self.model=None;self.tokenizer=None;self.torch=None;self.context=128;self.load_seconds=None
 def status(self):
  with self.lock:return dict(app=APP_ID,state=self.state,error=self.error,selected=self.selected,context=self.context,threads=8,load_seconds=self.load_seconds)
 def load(self,identifier):
  if identifier not in self.profiles:raise ValueError('Unknown model')
  with self.lock:
   if self.state in ('loading','generating'):raise RuntimeError('Wait for the current operation or stop generation first')
   if self.state=='ready' and self.selected==identifier:return
   self.state='loading';self.error=None
  def work():
   at=time.perf_counter()
   try:
    profile=self.profiles[identifier]
    if sha(profile.checkpoint)!=profile.file_hash:raise ValueError('Checkpoint hash does not match its verified report')
    tokenizer_path=(ROOT/self.selection['tokenizer_file']).resolve()
    if not tokenizer_path.is_relative_to((ROOT/'data').resolve()):raise ValueError('Invalid local tokenizer path')
    if sha(tokenizer_path)!=self.selection['tokenizer_sha256']:raise ValueError('Tokenizer hash does not match the approved encoding')
    self.model=None;gc.collect()
    if self.torch is None:
     os.environ.update(OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',OPENBLAS_NUM_THREADS='8')
     import torch
     torch.set_num_threads(8);torch.set_num_interop_threads(1);self.torch=torch
    from tokenizers import Tokenizer
    from lm.experimental_model import Config,parameter_hash
    from lm.efficiency import EfficientDecoder as Decoder
    # Map checkpoint storages; optimizer buffers are never materialized for chat.
    saved=self.torch.load(profile.checkpoint,map_location='cpu',weights_only=True,mmap=True);cfg=Config(**saved['config'])
    if cfg.vocab_size!=8192 or cfg.layers!=8 or cfg.ff_activation!='swiglu' or cfg.width not in (192,288,384) or cfg.heads!=cfg.width//32 or cfg.ff_width!=cfg.width*8//3:raise ValueError('Unsupported checkpoint architecture')
    model=Decoder(cfg,seed=42);model.load_state_dict(saved['model'],strict=True)
    if parameter_hash(model)!=profile.parameter_hash or saved.get('parameter_sha256')!=profile.parameter_hash:raise ValueError('Loaded weights do not match the recorded parameter hash')
    if sum(p.numel() for p in model.parameters())!=profile.parameters:raise ValueError('Unexpected parameter count')
    if saved.get('tokenizer_sha256',saved.get('report',{}).get('tokenizer_sha256'))!=self.selection['tokenizer_sha256']:raise ValueError('Checkpoint/tokenizer mismatch')
    model.eval();model.requires_grad_(False);tokenizer=Tokenizer.from_file(str(tokenizer_path));del saved
    with self.lock:self.model=model;self.tokenizer=tokenizer;self.selected=identifier;self.state='ready';self.load_seconds=time.perf_counter()-at
   except Exception as e:
    self.model=None;gc.collect()
    with self.lock:self.state='error';self.error=str(e);self.selected=None
  threading.Thread(target=work,daemon=True).start()
 def begin(self,body):
  text=prompt_text(body.get('messages'),body.get('mode','chat'));maximum,temperature=settings(body)
  with self.lock:
   if self.state!='ready':raise RuntimeError('Wait until a model is ready')
   self.state='generating';self.error=None;self.stop.clear()
  return text,maximum,temperature
 def generate(self,text,maximum,temperature):
  torch=self.torch;encoded=[1]+self.tokenizer.encode(text,add_special_tokens=False).ids;trimmed=len(encoded)>self.context;ids=encoded[-self.context:];generated=[];at=time.perf_counter();first=None
  try:
   yield dict(type='start',prompt_tokens=len(encoded),used_prompt_tokens=len(ids),context_trimmed=trimmed)
   generator=torch.Generator(device='cpu').manual_seed(45)
   from lm.efficiency import RollingCache
   cache=RollingCache(self.model,self.context)
   with torch.inference_mode():
    for i in range(maximum):
     if self.stop.is_set():break
     input_ids=torch.tensor([ids],dtype=torch.long)
     logits=(cache(input_ids)[0] if len(ids)<=self.context else self.model(input_ids[:,-self.context:],last_only=True)[0]).clone();logits[[0,1,3]]=float('-inf')
     if not torch.isfinite(logits[4:]).all() or not torch.isfinite(logits[2]):raise RuntimeError('Model produced non-finite scores')
     if temperature==0:n=int(logits.argmax())
     else:
      probabilities=(logits/temperature).softmax(-1);values,indices=probabilities.sort(descending=True);mask=values.cumsum(0)-values>0.9;values[mask]=0;values/=values.sum();n=int(indices[torch.multinomial(values,1,generator=generator)])
     generated.append(n);ids.append(n)
     if first is None:first=time.perf_counter()-at
     if n==2:break
     yield dict(type='text',text=self.tokenizer.decode(generated,skip_special_tokens=True),tokens=len(generated))
   elapsed=time.perf_counter()-at
   yield dict(type='done',text=self.tokenizer.decode(generated,skip_special_tokens=True),tokens=len(generated),seconds=elapsed,first_token_seconds=first,stopped=self.stop.is_set(),eos=bool(generated and generated[-1]==2))
  finally:
   with self.lock:self.state='ready'

class Handler(BaseHTTPRequestHandler):
 server_version='SaphrielLocal/1.0'
 def log_message(self,*args):pass
 def allowed(self):
  allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
  if self.headers.get('Host') not in allowed:return False
  origin=self.headers.get('Origin');return not origin or origin in {'http://'+x for x in allowed}
 def reply(self,status,value):
  payload=json.dumps(value).encode();self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(payload)
 def do_GET(self):
  if not self.allowed():return self.reply(403,dict(error='Local requests only'))
  path=urlparse(self.path).path;engine=self.server.engine
  if path=='/api/status':return self.reply(200,engine.status())
  if path=='/api/models':return self.reply(200,dict(models=[p.public() for p in engine.profiles.values()],default=default_profile(engine.profiles)))
  assets={'/':('index.html','text/html'),'/app.js':('app.js','text/javascript'),'/style.css':('style.css','text/css')}
  if path not in assets:return self.reply(404,dict(error='Not found'))
  filename,mime=assets[path];data=(ASSETS/filename).read_bytes();self.send_response(200);self.send_header('Content-Type',mime+'; charset=utf-8');self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'");self.end_headers();self.wfile.write(data)
 def do_POST(self):
  if not self.allowed():return self.reply(403,dict(error='Local requests only'))
  path=urlparse(self.path).path;engine=self.server.engine
  try:
   n=int(self.headers.get('Content-Length','0'))
   if not 0<=n<=65536:raise ValueError('Request is too large')
   body=json.loads(self.rfile.read(n) or b'{}')
   if not isinstance(body,dict):raise ValueError('Expected an object')
   if path=='/api/load':engine.load(body.get('model'));return self.reply(202,engine.status())
   if path=='/api/stop':engine.stop.set();return self.reply(200,dict(ok=True))
   if path!='/api/generate':return self.reply(404,dict(error='Not found'))
   text,maximum,temperature=engine.begin(body)
  except (ValueError,TypeError,json.JSONDecodeError) as e:return self.reply(400,dict(error=str(e)))
  except RuntimeError as e:return self.reply(409,dict(error=str(e)))
  try:
   self.send_response(200);self.send_header('Content-Type','application/x-ndjson; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Connection','close');self.end_headers()
   for event in engine.generate(text,maximum,temperature):self.wfile.write((json.dumps(event)+'\n').encode());self.wfile.flush()
  except (BrokenPipeError,ConnectionResetError):engine.stop.set()
  except Exception as e:
   try:self.wfile.write((json.dumps(dict(type='error',error=str(e)))+'\n').encode());self.wfile.flush()
   except OSError:pass
  finally:
   with engine.lock:
    if engine.state=='generating':engine.state='ready'
   self.close_connection=True

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=8765);p.add_argument('--open-browser',action='store_true');p.add_argument('--no-autoload',action='store_true');args=p.parse_args()
 if 'onedrive' in [x.lower() for x in ROOT.parts]:raise SystemExit('Run the local project outside OneDrive')
 engine=Engine()
 if not engine.profiles:raise SystemExit('No verified trained checkpoints were found in the local results folder')
 try:server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
 except OSError:
  import urllib.request
  try:
   status=json.load(urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/status',timeout=2))
   if status.get('app')!=APP_ID:raise RuntimeError('Port is used by another application')
  except Exception:raise SystemExit(f'Port {args.port} is in use. Choose another with --port.')
  if args.open_browser:webbrowser.open(f'http://127.0.0.1:{args.port}/')
  return
 server.engine=engine;server.daemon_threads=True;url=f'http://127.0.0.1:{server.server_port}/';print(url,flush=True)
 if not args.no_autoload:engine.load(default_profile(engine.profiles))
 if args.open_browser:webbrowser.open(url)
 try:server.serve_forever(poll_interval=0.25)
 except KeyboardInterrupt:pass
 finally:engine.stop.set();server.server_close()
if __name__=='__main__':main()
