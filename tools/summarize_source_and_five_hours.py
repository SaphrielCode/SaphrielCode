"""Publish provenance experiment and long-training results, without editing DataEval."""
from pathlib import Path
import json,statistics,math
ROOT=Path(__file__).resolve().parents[1];out=ROOT/json.loads((ROOT/'results/SourceAndFiveHour_Current.json').read_text())['directory'];plan=json.loads((out/'suite.json').read_text());lines=['SOURCE-AWARE LEARNING AND FIVE-HOUR TINYSTORIES TRAINING','',f'Status: {plan["status"]}','Approved unchanged model:5,115,072 parameters,8 layers,width192,6heads,SwiGLU512,context128,frozen8192 unigram.','User-authorized immediate stop below0.5 GiB available system RAM; process4 GiB consecutive guard retained.','Final TinyStories test split reserved. All numbers below are measured local results, not general conversation claims.','','SOURCE-FAMILY EXPERIMENT','960 examples from64 authored source families, including48 paraphrases each for16 desk families and4 for48 other families.','Balanced-data control:4 paraphrases per family,256 examples. Heldout64 distinct families, four factual variants each.','Three training-order seeds44/45/46; same90-minute pretrained checkpoint and2400 updates per arm. Ordinary/family batch-index traces must match.','Per-example response+EOS mean NLL; family weights N/(G*n_family), fixed batch denominator. Not self-normalized inside batches.','This tests a deliberately skewed synthetic dependency scenario. Families are provenance groups, not statistically independent observations.','Arm       Seed  Pair-correct  Overall-EM  Missing  Negated  Story-before  Story-after  Updates   Seconds']
rows=[]
for seed in (44,45,46):
 for arm in ('ordinary','family','balanced'):
  p=out/f'source_{arm}_{seed}/run.json'
  if not p.exists():continue
  r=json.loads(p.read_text());rows.append(r)
  if r['status']!='PASS':lines.append(f'{arm} seed{seed}: {r["status"]}, completed{r["updates"]} updates');continue
  e=r['metrics'];lines.append(f'{arm:10} {seed:3} {e["both_answerable_pair_correct"]:11.4f} {e["exact_match"]:11.4f} {e["per_variant"]["missing"]:8.4f} {e["per_variant"]["negated"]:8.4f} {r["story_before"]["nats_per_target"]:12.6f} {r["story_after"]["nats_per_target"]:12.6f} {r["updates"]:8} {r["training_seconds"]:9.2f}')
 for a in ('ordinary','family'):
  p=out/f'source_{a}_{seed}/run.json'
  if not p.exists() or json.loads(p.read_text())['status']!='PASS':break
 else:
  a=json.loads((out/f'source_ordinary_{seed}/run.json').read_text());b=json.loads((out/f'source_family_{seed}/run.json').read_text());assert a['batch_index_trace_sha256']==b['batch_index_trace_sha256'] and a['initial_parameter_sha256']==b['initial_parameter_sha256'] and a['updates']==b['updates'] and a['valid_response_targets']==b['valid_response_targets']
if (out/'source-decision.json').exists():lines+=['', 'Source-method decision: '+json.dumps(json.loads((out/'source-decision.json').read_text()),indent=2), 'Pure instruction fine-tuning caused major forgetting. These instruction weights were NOT promoted. Any useful weighting/data-selection principle is provisional and is applied to a separate story-rehearsed candidate after pretraining.']
long=out/'five_hours/run.json'
if long.exists():
 r=json.loads(long.read_text());lines+=['','FIVE-HOUR PRETRAINING','Fresh seed42, frozen data order43, CPUFP32 AdamW, batch4,8intra/1inter-op threads.','Counted200-update LR warmup; cosine3e-4 to3e-5 over18000 active seconds; no extra warmup.','Checkpoint/evaluation/generation time excluded from active-training budget and reported separately.',f'Status:{r["status"]}; active seconds:{r["training_seconds"]:.6f}; updates:{r["completed_updates"]}; valid targets:{r["valid_targets"]}; complete corpus passes:{r["completed_passes"]}; next frozen row:{r["next_row"]}.',f'Training targets per complete pass:{r["targets_per_pass"]}; repeat exposure is not new unique data.',f'Validation seconds:{r["validation_seconds"]:.3f}; checkpoint seconds:{r["checkpoint_seconds"]:.3f}.', 'Validation numerator = float64 sum of per-target NLL; denominator =216150 eligible validation targets; padding excluded.']
 for label,e in r['validation'].items():lines.append(f'{label}: sum{e["loss_sum"]:.8f} / {e["valid_targets"]} = {e["loss_nats_per_target"]:.8f} nats/target; {e["bits_per_utf8_text_byte_including_eos_cost"]:.8f} bits/UTF-8 text byte including EOS NLL. Active seconds{e["actual_training_seconds"]:.3f}, updates{e["completed_updates"]}, targets{e["valid_targets_processed"]}.')
 if r['validation']:
  best=min(r['validation'],key=lambda k:r['validation'][k]['loss_nats_per_target']);lines.append('Lowest validation loss checkpoint:'+best+'. Last checkpoint retained separately.')
 lines+=['Learning curve selection uses validation only. No final-test selection or score.']
 for label,g in r.get('continuations',{}).items():
  lines.append(f'Fixed continuations at{label}: temperature0.7,top-p0.9,seed45,max64new tokens,rolling128 context; generation seconds{g["seconds"]:.3f}.')
  for e in g['samples']:lines+=['PROMPT: '+e['prompt'],'CONTINUATION: '+e['continuation']]
 if 'peak_process_resident_bytes' in r:lines+=[f'Whole-process peak resident:{r["peak_process_resident_bytes"]} bytes; minimum available system RAM:{r["minimum_available_system_bytes"]} bytes; stop:{r["guard_stop"]}.']
qa=out/'grounded_qa/run.json'
if qa.exists():
 r=json.loads(qa.read_text());lines+=['','STORY-REHEARSED GROUNDED QA','SQuAD2.0-derived short-prefix diagnostic (NOT official SQuAD evaluation).4000train examples/2994paragraph families;256dev examples/201families; balanced answerability.', 'Question-independent complete-sentence prefixes,<=72unigram tokens; answerable gold span must remain. Source train/dev split retained. Corpus sources/hashes/license CC-BY-SA4.0 and attribution preserved.', 'Balanced synthetic source examples and40 locally authored greeting/scope examples supplement training; greetings do not count as reading-comprehension evidence. No CoQA or FineWeb-EDU used in this run.',f'Status:{r["status"]}; updates:{r["updates"]}; training seconds:{r["training_seconds"]:.3f}; QA targets:{r["qa_targets"]}; replay story targets:{r["story_replay_targets"]}.',r['objective']]
 if 'timing_limitation' in r:lines+=['QA timing limitation: '+json.dumps(r['timing_limitation'])]
 if 'before' in r:lines+=['Before: '+json.dumps(r['before'])]
 for e in r['epochs']:lines+=['Epoch: '+json.dumps(e)]
 lines+=['Selected epoch: '+str(r.get('selected_epoch')),r.get('selection_rule',''), 'After source diagnostic:'+json.dumps(r.get('after_source'))]
 if r.get('selected_epoch') is None:lines+=['No QA candidate qualified for deployment. Preserve the best TinyStories model and the QA diagnostic artifacts.']
lines+=['','Provenance and outcome records are not calibrated confidence and do not establish weight-level evidence independence.','Source prep was run while early source workers were active; their wall time is not a controlled speed comparison. Primary comparison controls optimizer updates, batch traces, starting weights and targets.']
text='\n'.join(lines)+'\n';(out/'Results.txt').write_text(text,encoding='utf-8');(ROOT/'results/SourceAndFiveHour_Results.txt').write_text(text,encoding='utf-8');print(text)
