"""Verify and summarize the completed local unigram experiment."""
from pathlib import Path
import json, hashlib, math, shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
old=ROOT/'results/TwoPassA_Unigram8192_20261111T225845533672Z'
run=ROOT/'results/TwoPassA_Unigram8192_20261111T233317210713Z'
r=json.loads((run/'run.json').read_text())
assert r['status']=='PASS' and r['completed_passes']==2
assert r['completed_updates']==22570 and r['valid_targets_processed']==8777642
assert not r['final_test_evaluated']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def lines(p):
 with p.open(encoding='utf-8') as f:
  for line in f:
   if line.strip():yield json.loads(line)
rows=np.fromfile(ROOT/'data/TinyStoriesBatches128_Unigram8192/train.rows.u32',dtype='<u4').reshape(-1,3)
order=np.fromfile(ROOT/'data/TinyStoriesBatches128_Unigram8192/train.order.u32',dtype='<u4')
expected=[int(rows[order[i:i+4],2].sum()) for i in range(0,len(order),4)]
assert sum(expected)==4388821 and len(expected)==11285
for n in (1,2):
 steps=list(lines(run/f'pass{n}-steps.jsonl'))
 if n==1:steps=[x for x in lines(old/'pass1-steps.jsonl') if x['update']<=8000]+steps
 assert [x['update'] for x in steps]==list(range((n-1)*11285+1,n*11285+1))
 assert [x['valid_targets'] for x in steps]==expected
 assert all(math.isfinite(x['loss_sum']) and math.isfinite(x['gradient_norm']) for x in steps)
 assert sum(x['valid_targets'] for x in steps)==r['passes'][n-1]['valid_targets']
 assert math.isclose(sum(x['loss_sum'] for x in steps),r['passes'][n-1]['loss_sum'],rel_tol=1e-12)
for c in r['checkpoints']:
 p=Path(c['file']);p=p if p.is_absolute() else run/p
 assert c['reload_verified'] and p.stat().st_size==c['bytes'] and sha(p)==c['sha256']
for name,h in r['source_hashes'].items():
 assert sha(ROOT/name)==h
 p=run/'runtime-source'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,p)
telemetry=[]
for d in (old,run):
 samples=list(lines(d/'telemetry.jsonl')); low=high=0
 for s in samples:
  assert s.get('guard_stop') is None and s['available_system_bytes']>=2**30
  low=low+1 if s['available_system_bytes']<2*2**30 else 0
  high=high+1 if s['resident_bytes']>4*2**30 else 0
  assert low<3 and high<3
 telemetry+=samples
raw_training_seconds=r['training_seconds']
r['training_seconds']=sum(p['seconds'] for p in r['passes'])
bpe=json.loads((ROOT/'results/BPE8192_TwoPass_Summary.json').read_text())
bpe4=json.loads((ROOT/'results/BPE4096_TwoPass_Summary.json').read_text())
source=ROOT/'data/TinyStoriesSubset_Unigram8192'
for split in ('train','validation','test'):assert sha(source/f'{split}.jsonl')==sha(ROOT/'data/TinyStoriesSubset'/f'{split}.jsonl')
byte_count=sum(len(x['text'].encode('utf-8')) for x in lines(source/'validation.jsonl'))
assert byte_count==bpe['validation_utf8_text_bytes']==884630
for v in r['validation'].values():
 assert v['valid_targets']==216150 and math.isclose(v['loss_sum']/v['valid_targets'],v['loss_nats_per_target'])
 v['bits_per_utf8_text_byte_including_eos_cost']=v['loss_sum']/byte_count/math.log(2)
summary=dict(status='PASS',tokenizer_kind='unigram',vocabulary=8192,unique_parameters=r['unique_parameters'],completed_updates=r['completed_updates'],valid_targets=r['valid_targets_processed'],validation=r['validation'],validation_utf8_text_bytes=byte_count,passes=r['passes'],training_seconds_successful_trajectory=r['training_seconds'],validation_seconds_successful_trajectory=r['validation_seconds'],checkpoint_seconds=r['checkpoint_seconds'],trajectory_peak_process_bytes=max(s['peak_resident_bytes'] for s in telemetry),trajectory_minimum_available_bytes=min(s['available_system_bytes'] for s in telemetry),run_directory=str(run.relative_to(ROOT)),recovery=r['recovery_resume'],checkpoints=r['checkpoints'],continuations=r['continuations'],final_test_reserved=True,checks='Exact update sequence, every batch valid-target count, finite step losses/norms, checkpoint size/hash and runtime reload, unchanged model source, identical source documents and guard telemetry verified.')
u=r['validation']['pass_2']['bits_per_utf8_text_byte_including_eos_cost']; b=bpe['validation']['pass_2']['bits_per_utf8_text_byte_including_eos_cost']
summary['raw_runner_training_seconds']=raw_training_seconds
summary['timing_correction']='Raw runner aggregate omits inherited partial-pass seconds after recovery; verified trajectory time is the sum of complete per-pass wall times.'
summary['final_bits_per_byte_change_vs_bpe8192_percent']=(u/b-1)*100
summary['validation_improvement_pass1_to_pass2_percent']=(1-r['validation']['pass_2']['loss_sum']/r['validation']['pass_1']['loss_sum'])*100
summary['limitations']='One seed and one explicitly seeded byte-level unigram recipe with mandatory-byte reservation; not an exhaustive algorithm comparison. Fixed token contexts span different text. Background workloads differ. No equal-total-compute claim. Recovery I/O is included in pass times and separately disclosed in checkpoint time; do not add as disjoint intervals. Interrupted unsaved work is discarded experiment overhead.'
(ROOT/'results/Unigram8192_TwoPass_Summary.json').write_text(json.dumps(summary,indent=2)+'\n')
text=['UNIGRAM 8192 TWO-PASS RESULTS','',summary['checks'],'',f"Updates: {r['completed_updates']}; valid targets: {r['valid_targets_processed']} (4,388,821/pass); expected exposure exactly met.",'Same 19,966 training stories, same 1,000 validation stories, 8,192 entries and 5,115,072 parameters. Seed42 numerical initialization matches BPE8192; data seed43; batch4/context128/8 threads/1 inter-op; FP32 AdamW unchanged. No extra warmup. Padding excluded; stories isolated.','']
for label,v in r['validation'].items():text.append(f"Validation {label}: {v['loss_nats_per_target']:.9f} nats/target; sum {v['loss_sum']:.9f}; count {v['valid_targets']}; {v['bits_per_utf8_text_byte_including_eos_cost']:.9f} bits/UTF8 text byte including EOS cost.")
for p in r['passes']:text.append(f"Pass {p['pass_number']}: {p['completed_updates']} updates, {p['valid_targets']} targets, {p['seconds']:.3f}s, {p['valid_targets_per_second']:.3f} valid targets/s.")
text += ['',f"Training trajectory: {r['training_seconds']:.3f}s; validation: {r['validation_seconds']:.3f}s; checkpoint save/reload: {r['checkpoint_seconds']:.3f}s.",f"Whole-trajectory process peak: {summary['trajectory_peak_process_bytes']/2**20:.2f} MiB; minimum available RAM: {summary['trajectory_minimum_available_bytes']/2**30:.3f} GiB. 2GiB reserve, immediate1GiB and process4GiB guards respected.",'',f"Final BPE8192: {b:.9f} bits/byte. Unigram change: {(u/b-1)*100:+.3f}% (lower is better).",f"Final BPE4096: {bpe4['validation']['pass_2']['bits_per_utf8_text_byte_including_eos_cost']:.9f} bits/byte.",f"Pass1-to-pass2 held-out improvement: {summary['validation_improvement_pass1_to_pass2_percent']:.3f}%. Another pass is worth a bounded test if improvement remains useful, but none launched.",'','Tokenizer: SentencePiece unigram EM over training-only ByteLevel chunks with explicit frequency-weighted substring seeds, then deterministic reservation of all256 byte symbols in the fixed8192 vocabulary; final HF tokenizer.json is authoritative. Missing byte symbols replaced the same count of lowest-score non-required multi-character pieces. No sampling during encoding. Earlier preparation diagnostics were not LM runs. See Unigram8192_Preparation.txt and local construction artifacts.','',summary['limitations'],'','Previous process ended without final report; checkpoint8000 restored weights/Adam and row32000. Later unsaved work was repeated and is excluded from trajectory exposure. Both pass checkpoints reloaded and hashed successfully. Final test remains reserved.']
for c in r['continuations']:text+=['',f"Pass {c['pass_number']} prompt: {c['prompt']}",c['decoding'],c['continuation']]
text+=['',f"Artifacts: {run.relative_to(ROOT)}; source snapshots in runtime-source. No DataEval changes."]
text += ['',summary['timing_correction'],f'Uncorrected runner aggregate: {raw_training_seconds:.3f}s; verified total: {r["training_seconds"]:.3f}s.']
report='\n'.join(text)+'\n'
(ROOT/'results/Unigram8192_TwoPass_Results.txt').write_text(report,encoding='utf-8')
(run/'VerifiedFindings.txt').write_text(report,encoding='utf-8')
(run/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
