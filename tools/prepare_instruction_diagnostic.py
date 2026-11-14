"""Freeze an authored, synthetic instruction diagnostic before inspecting responses."""
from pathlib import Path
import json,random,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/'data/InstructionDiagnostic_v1'
 if out.exists():raise RuntimeError('Refuse overwrite')
 out.mkdir();rng=random.Random(4401)
 train_words='cat dog ball boat tree flower cup book chair apple rabbit horse bear fox duck frog kite cake star spoon hat coat shoe drum bell box toy bag pear fish desk lamp'.split()
 held_words='cloud stone shell leaf whale tiger orange lemon truck train sheep turtle'.split()
 colors=['red','blue','green','yellow'];places=['garden','kitchen','bedroom','school','park','yard'];train_names=['Mia','Tom','Ben','Anna','Lily','Sam'];held_names=['Nora','Owen','Iris','Theo']
 def examples(split,n):
  words=train_words if split=='train' else held_words;names=train_names if split=='train' else held_names;items=[]
  for task in ('extract','repeat','mapping','context_qa','yes_no'):
   seen=set();i=0
   while len(seen)<n:
    i+=1;a,b,c=rng.choices(words,k=3);color=rng.choice(colors);place=rng.choice(places);name=rng.choice(names)
    if task=='extract':instruction=f'Return only the final word in this list: {a}, {b}, {c}.';answer=c
    elif task=='repeat':instruction=f'Repeat exactly these words, with no explanation: {color} {a} {b}.';answer=f'{color} {a} {b}'
    elif task=='mapping':
     k1,k2=rng.sample(colors,2);target=rng.choice([k1,k2]);instruction=f'Use this code: {k1} means {a}; {k2} means {b}. What does {target} mean? Return only its meaning.';answer=a if target==k1 else b
    elif task=='context_qa':instruction=f'Read: {name} left the {color} {a} in the {place}. Where did {name} leave it? Answer only with the place.';answer=place
    else:
     ask=color if len(seen)%2==0 else rng.choice([x for x in colors if x!=color]);instruction=f'Read: The {a} is {color}. Is the {a} {ask}? Answer only yes or no.';answer='yes' if ask==color else 'no'
    if instruction in seen:continue
    seen.add(instruction);items.append(dict(id=f'{split}_{task}_{len(seen):03d}',task=task,instruction=instruction,response=answer))
  return items
 splits={'train':examples('train',100),'heldout':examples('heldout',12)}
 assert not set(x['instruction'] for x in splits['train'])&set(x['instruction'] for x in splits['heldout'])
 for split,items in splits.items():(out/f'{split}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in items))
 manifest=dict(name='InstructionDiagnostic_v1',provenance='Locally authored deterministic synthetic templates; no external dataset. No model responses used during construction.',selection_seed=4401,training_documents=500,heldout_documents=60,tasks=['extract','repeat','mapping','context_qa','yes_no'],split_rule='Distinct subject words and names across train/heldout; shared task templates. Measures template/compositional transfer, not broad instruction generalization.',prompt_format='Instruction:\n{instruction}\nResponse:\n',training_epochs=5,batch_size=4,order_seed=44,optimizer='Fresh AdamW moments; lr0.0003 betas0.9/0.95 eps1e-8 decay0.1 matrices/0norm; clipping1; FP32; no accumulation',supervision='Response tokens and EOS only; prompt, BOS and padding ignored. Reject sequences exceeding128; no truncation.',decoding='Greedy argmax; max32 new tokens; EOS2; BOS1; rolling context128; no stop strings',rubric={'correctness':'Exact match after NFC and outer whitespace stripping, case/punctuation preserved; report per task and overall.','instruction_format':'Nonempty single line; repeat exactly3 whitespace words; yes_no exactlyyes/no; other tasks a single alphabetic word. This is format compliance, not semantic correctness.','coherence':'Human review:2 clear task-appropriate expression;1 readable but irrelevant/fragmented;0 unintelligible. Report separately from exact correctness; retain all responses.','repetition':'Report repeated4gram fraction for responses with>=4tokens, and longest identical whitespace-word run. Short outputs have no repeated4grams.'},limits='Small synthetic diagnostic; one seed; shared templates across splits. Not a general conversation, reasoning, factuality or safety benchmark.',files={})
 for p in out.glob('*.jsonl'):manifest['files'][p.name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(ROOT/'results/WidthStudy_InstructionProtocol.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
