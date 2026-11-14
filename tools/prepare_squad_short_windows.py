"""Freeze short, untruncated SQuAD2 examples with explicit paragraph provenance."""
from pathlib import Path
import urllib.request,json,hashlib,random,sys,html,re
ROOT=Path(__file__).resolve().parents[1]
def fetch(url,path):
 if path.exists():return
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 Saphriel data preparation'}),timeout=60) as source,path.open('wb') as target:
  while chunk:=source.read(1024*1024):target.write(chunk)
def main():
 out=ROOT/'data/SQuAD2Windows128_v1';out.mkdir(exist_ok=True)
 if (out/'manifest.json').exists():raise FileExistsError('Frozen SQuAD output exists')
 cache=ROOT/'preparation-cache/SQuAD2';cache.mkdir(parents=True,exist_ok=True);refs=ROOT/'corpus-references/SQuAD2';refs.mkdir(exist_ok=True)
 urls={s:f'https://rajpurkar.github.io/SQuAD-explorer/dataset/{s}-v2.0.json' for s in ('train','dev')}
 fetch('https://raw.githubusercontent.com/spdx/license-list-data/main/text/CC-BY-SA-4.0.txt',refs/'CC-BY-SA-4.0.txt')
 fetch('https://rajpurkar.github.io/SQuAD-explorer/',refs/'dataset-page.html')
 from tokenizers import Tokenizer
 tok=Tokenizer.from_file(str(ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json'));records={};manifest=dict(dataset='SQuAD2.0',license='CC-BY-SA-4.0',source_files={},selection='Question-independent paragraph prefix of complete regex-segmented sentences, at most72 tokenizer tokens; no answer-aware crop. Answerable questions retained only if an original gold span is wholly in the prefix. Drop coreferential questions. Adapted local diagnostic, not official SQuAD evaluation; prompt+response+EOS<=128; answer<=8 tokens; at most2 answerable and2 unanswerable questions per paragraph; deterministic seed5501 train/5502 dev; balanced answerability cap4000 train/256 dev. Whole supplied train/dev article split preserved.',files={})
 for split in ('train','dev'):
  path=cache/(split+'-v2.0.json');fetch(urls[split],path);manifest['source_files'][split]=dict(url=urls[split],bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest());raw=json.loads(path.read_text(encoding='utf-8'));pool={True:[],False:[]}
  for article in raw['data']:
   for index,para in enumerate(article['paragraphs']):
    family=hashlib.sha256((article['title']+'\0'+para['context']).encode()).hexdigest();counts={True:0,False:0};context=''
    for end in [m.end() for m in re.finditer(r'[.!?](?:\s+|$)',para['context'])]:
     prefix=para['context'][:end].rstrip()
     if len(tok.encode(prefix).ids)>72:break
     context=prefix
    if not context:continue
    for q in para['qas']:
     answerable=not q['is_impossible']
     if re.search(r'\b(he|she|his|her|they|their|those|these|this|that|it|its)\b',q['question'],re.I):continue
     answers=[x['text'] for x in q['answers'] if x['answer_start']+len(x['text'])<=len(context)] if answerable else ['unknown']
     if not answers:continue
     answer=answers[0]
     if counts[answerable]>=2 or len(tok.encode(answer).ids)>8:continue
     if answerable and any(para['context'][x['answer_start']:x['answer_start']+len(x['text'])]!=x['text'] for x in q['answers']):raise ValueError('Source answer offset mismatch')
     instruction='Read the passage. Answer briefly, or unknown if not stated.\nPassage: '+context+'\nQuestion: '+q['question'];prompt='Instruction:\n'+instruction+'\nResponse:\n'
     if 1+len(tok.encode(prompt).ids)+len(tok.encode(answer).ids)>128:continue
     row=dict(id=q['id'],family=family,title=article['title'],paragraph=index,context_characters=len(context),full_context_characters=len(para['context']),instruction=instruction,response=answer,references=answers,answerable=answerable,source_split=split);pool[answerable].append(row);counts[answerable]+=1
  rng=random.Random(5501 if split=='train' else 5502);cap=2000 if split=='train' else 128
  for rows in pool.values():rng.shuffle(rows)
  cap=min(cap,len(pool[True]),len(pool[False]));selected=pool[True][:cap]+pool[False][:cap];rng.shuffle(selected);records[split]=selected;manifest[split]=dict(examples=len(selected),answerable=sum(e['answerable'] for e in selected),unanswerable=sum(not e['answerable'] for e in selected),families=len({e['family'] for e in selected}),eligible_answerable=len(pool[True]),eligible_unanswerable=len(pool[False]))
  del raw,pool
 assert not ({e['family'] for e in records['train']}&{e['family'] for e in records['dev']})
 for split,rows in records.items():
  p=out/(split+'.jsonl');p.write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in rows),encoding='utf-8');manifest['files'][p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
 manifest['tokenizer_sha256']=hashlib.sha256((ROOT/'data/TinyStoriesSubset_Unigram8192/tokenizer.json').read_bytes()).hexdigest();(out/'manifest.json').write_text(json.dumps(manifest,indent=2));(refs/'Attribution.txt').write_text('SQuAD2.0: Pranav Rajpurkar, Robin Jia and Percy Liang (2018), Know What You Don\'t Know: Unanswerable Questions for SQuAD. Stanford NLP. Source: https://rajpurkar.github.io/SQuAD-explorer/ . Distributed CC BY-SA4.0. Underlying passages from Wikipedia; article titles and paragraph/source IDs retained. Local subset and prompt formatting are adaptations, with source unchanged. Full source URL/hash inventory is in the corpus manifest. No CoQA or FineWeb-EDU used.\n');print(json.dumps({s:manifest[s] for s in ('train','dev')}))
if __name__=='__main__':main()
