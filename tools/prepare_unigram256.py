"""Prepare and independently verify 256-position windows from frozen unigram IDs."""
from pathlib import Path
import json,random,struct,shutil
import numpy as np
from prepare_batches import window_story,u16_bytes,digest
ROOT=Path(__file__).resolve().parents[1]
def main():
 source=ROOT/'data/TinyStoriesSubset_Unigram8192'; out=ROOT/'data/TinyStoriesBatches256_Unigram8192'
 original=json.loads((ROOT/'data/TinyStoriesBatches128_Unigram8192/manifest.json').read_text())
 if out.exists():raise RuntimeError('Refuse overwrite')
 for n,e in original['source_files'].items():assert digest(source/n)==e['sha256']
 out.mkdir();shutil.copy2(ROOT/'data/TinyStoriesBatches128_Unigram8192/tokenizer-lock.json',out/'tokenizer-lock.json')
 manifest=dict(original);manifest['context']=256;manifest['statistics']={}
 for split in ('train','validation','test'):
  ids=np.fromfile(source/f'{split}.u16',dtype='<u2'); offsets=np.fromfile(source/f'{split}.idx',dtype='<u8'); count=total=0
  with (out/f'{split}.x.u16').open('wb') as xf,(out/f'{split}.y.u16').open('wb') as yf,(out/f'{split}.rows.u32').open('wb') as rf:
   for doc,(a,b) in enumerate(zip(offsets,offsets[1:])):
    for x,y,start,n in window_story(ids[int(a):int(b)].tolist(),256):
     xf.write(u16_bytes(x));yf.write(u16_bytes(y));rf.write(struct.pack('<III',doc,start,n));count+=1;total+=n
  assert total==len(ids)==original['statistics'][split]['valid_target_tokens']
  if split=='train':
   order=list(range(count));random.Random(43).shuffle(order);np.array(order,dtype='<u4').tofile(out/'train.order.u32')
  stats=dict(original['statistics'][split]);stats.update(windows=count,padding_targets=count*256-total,window_file_bytes=sum(p.stat().st_size for p in out.glob(split+'.*')))
  manifest['statistics'][split]=stats
  x=np.memmap(out/f'{split}.x.u16',dtype='<u2',mode='r',shape=(count,256));y=np.memmap(out/f'{split}.y.u16',dtype='<u2',mode='r',shape=(count,256));rows=np.fromfile(out/f'{split}.rows.u32',dtype='<u4').reshape(-1,3);covered=np.zeros(len(offsets)-1,dtype=np.uint64)
  for i,(doc,start,n) in enumerate(rows):
   doc,start,n=map(int,(doc,start,n));a=int(offsets[doc]);b=int(offsets[doc+1]);assert start==covered[doc] and a+start+n<=b
   assert np.array_equal(y[i,:n],ids[a+start:a+start+n]);assert x[i,0]==(1 if start==0 else ids[a+start-1]);assert np.array_equal(x[i,1:n],y[i,:n-1]);assert not np.any(x[i,n:]) and not np.any(y[i,n:]);covered[doc]+=n
  assert np.array_equal(covered,np.diff(offsets));del x,y
 manifest['preparation_script_sha256']=digest(Path(__file__));manifest['storage']='x/y uint16 [windows,256]; rows uint32 triples; order uint32'
 manifest['files']={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir() if p.is_file()};manifest['payload_bytes']=sum(e['bytes'] for e in manifest['files'].values())
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(ROOT/'results/Unigram256_WindowPreparation.json').write_text(json.dumps(dict(status='PASS',statistics=manifest['statistics'],context=256),indent=2)+'\n')
 print(json.dumps(manifest['statistics']))
if __name__=='__main__':main()
