"""Shared strict guard monitor; samples actual trainer process every second."""
from pathlib import Path
from dataclasses import asdict
import json,time,threading
import psutil
from .resource_guard import Policy,Guard
class Monitor:
 def __init__(self,out,policy=None):
  self.out=Path(out);self.phase='preflight';self.started=time.perf_counter();self.process=psutil.Process();self.guard=Guard(policy or Policy());self.stop=threading.Event();self.done=threading.Event();self.samples=[];self.file=(self.out/'telemetry.jsonl').open('w');self.thread=None;self.error=None
 def sample(self):
  m=self.process.memory_info();available=psutil.virtual_memory().available;reason=self.guard.observe(available,[dict(pid=self.process.pid,resident_bytes=m.rss)])
  row=dict(seconds=time.perf_counter()-self.started,phase=self.phase,available_system_bytes=available,resident_bytes=m.rss,peak_resident_bytes=getattr(m,'peak_wset',m.rss),private_bytes=getattr(m,'private',None),cpu_seconds=sum(self.process.cpu_times()[:2]),guard_stop=reason);self.samples.append(row);self.file.write(json.dumps(row)+'\n');self.file.flush()
  if reason:self.stop.set()
 def check(self):
  if self.stop.is_set():raise RuntimeError('Resource monitor requested stop')
 def start(self):
  for i in range(3):
   self.sample();self.check()
   if i<2:time.sleep(1)
  def watch():
   try:
    while not self.done.wait(1):
     self.sample()
     if self.stop.is_set():return
   except Exception as e:self.error=repr(e);self.stop.set()
  self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()
 def close(self):
  self.done.set()
  if self.thread:self.thread.join(3)
  self.file.close()
  return dict(guard_policy=asdict(self.guard.policy),peak_process_resident_bytes=max((s['peak_resident_bytes'] for s in self.samples),default=0),minimum_available_system_bytes=min((s['available_system_bytes'] for s in self.samples),default=0),guard_stop=next((s for s in self.samples if s['guard_stop']),None),monitor_error=self.error)
