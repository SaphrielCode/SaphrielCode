"""Frozen provenance-controlled location-reading experiment; locally authored data."""
from pathlib import Path
from collections import Counter
import json,random,hashlib,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'data/SourceFamilies_v1';out.mkdir(exist_ok=False)
 names='Amy Ben Cara Dan Ella Finn Gina Hank Ivy Jack Kate Leo Mia Noah Owen Pia'.split();objects='key ball book toy hat cup bag doll coin box shoe bell'.split();places='desk garden kitchen park bedroom shed school beach shelf'.split()
 templates=['{n} left the {o} at the {p}.','The {o} is at the {p}. {n} put it there.','{n} put the {o} at the {p}.','At the {p}, {n} left the {o}.','{n} has a {o}. It is at the {p}.','The {o} that belongs to {n} is at the {p}.','{n} keeps the {o} at the {p}.','Look at the {p}. The {o} is there, where {n} left it.','{n} says the {o} is at the {p}.','The {o} was left at the {p} by {n}.','{n} went to the {p} and left the {o} there.','You can find the {o} at the {p}, where {n} put it.']
 questions=['Where is the {o}?','Where did {n} leave the {o}?','What is the location of the {o}?','Where can the {o} be found?']
 def make(fid,n,o,p,t,q,variant=None):
  context=(f'{n} has a {o}. It is a nice day.' if p=='unknown' else t.format(n=n,o=o,p=p))
  instruction='Read the passage and answer using only its facts. If the location is not given, answer unknown. Answer with one word.\nPassage: '+context+'\nQuestion: '+q.format(n=n,o=o)
  return dict(id=f'{fid}_{variant or t[:8]}_{q[:8]}',family=fid,instruction=instruction,response=p,variant=variant,source_facts=dict(person=n,object=o,location=None if p=='unknown' else p))
 train=[]
 for i in range(64):
  n=names[i%16];o=objects[(i//16+i)%len(objects)];p='desk' if i<16 else 'unknown' if i>=52 else places[1+(i-16)%8];fid=f'train_source{i:03}'
  combos=[(t,q) for t in templates for q in questions];random.Random(4410+i).shuffle(combos)
  for j,(t,q) in enumerate(combos[:48 if i<16 else 4]):
   e=make(fid,n,o,p,t,q);e['id']=f'{fid}_paraphrase{j:02}';train.append(e)
 held=[]
 for i in range(64):
  n=['Rose','Sam','Tina','Uma','Vera','Will','Yara','Zoe'][i%8];o=['ring','pen','sock','brush','kite','spoon','drum','apple'][(i//8+i)%8];a=places[i%9];b=places[(i%9+1+i//9)%9]
  if a==b:b=places[(places.index(a)+1)%9]
  fid=f'heldout_source{i:03}';q='Where is the {o}?'
  contexts=[('original',f'{n} set the {o} at the {a}.',a),('moved',f'{n} set the {o} at the {b}.',b),('missing',f'{n} picked up the {o}. The day was sunny.','unknown'),('negated',f'The {o} is not at the {a}. {n} picked it up.','unknown')]
  for variant,context,p in contexts:
   e=make(fid,n,o,p,templates[0],q,variant);e['instruction']=e['instruction'].split('Passage:')[0]+'Passage: '+context+'\nQuestion: '+q.format(o=o);e['id']=fid+'_'+variant;held.append(e)
 counts=Counter(e['family'] for e in train);balanced=[e for e in train if int(e['id'].split('paraphrase')[1])<4]
 for split,rows in [('train',train),('balanced',balanced),('heldout',held)]:
  assert len({e['id'] for e in rows})==len(rows)
  (out/(split+'.jsonl')).write_text(''.join(json.dumps(e)+'\n' for e in rows),encoding='utf-8')
 manifest=dict(name='SourceFamilies_v1',provenance='Locally authored deterministic synthetic controlled dependency test; no model responses used.',train_examples=len(train),train_families=len(counts),family_counts=dict(counts),balanced_examples=len(balanced),heldout_examples=len(held),heldout_families=64,weighting='Per-example response+EOS mean NLL; weights N/(G*n_family); mean weighted losses with fixed batch denominator, never divide by within-batch weight sum.',control='Same64 source families, first4 paraphrases each; fixed identical optimizer update count.',limitations='Deliberately skewed paraphrase frequency. Distinct synthetic scenarios are provenance groups, not a claim of statistical independence. Whole heldout families, names, objects and passage wording excluded from instruction training.',files={})
 for p in out.glob('*.jsonl'):manifest['files'][p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({k:manifest[k] for k in ('train_examples','train_families','balanced_examples','heldout_examples')}))
if __name__=='__main__':main()
