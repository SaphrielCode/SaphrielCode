"""Final human-readable width study report and independent instruction checks."""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[1]
def main():
 out=ROOT/json.loads((ROOT/'results/WidthStudy_Current.json').read_text())['directory'];suite=json.loads((out/'suite.json').read_text());assert suite['status']=='PASS'
 report=['WIDTH SCALING STUDY - VERIFIED RESULTS','','All three candidates: frozen8192 unigram,8layers,SwiGLU intermediate8/3width,head dimension32,context128,batch4,CPU FP32,8threads/1inter-op,seed42,order43,unchanged AdamW. Fresh weights; no uncounted warmup. All validation uses216150 targets and884630 UTF8 text bytes; bits/byte includesEOS cost.','Equal ACTIVE training time excludes evaluation, checkpointing and startup. Complete updates stop at budget boundaries, with actual overshoot recorded. This is not equal arithmetic compute.','', 'Validation bits per byte (lower is better)','Width | Parameters | Initialization |15min |30min |60min |90min']
 results={};instructions={}
 for width in (192,288,384):
  r=json.loads((out/f'train{width}/run.json').read_text());assert r['status']=='PASS';results[width]=r;v=r['validation'];report.append(f"{width} |{r['unique_parameters']} |"+' |'.join(f"{v[k]['bits_per_utf8_text_byte_including_eos_cost']:.9f}" for k in ('initialization','minute15','minute30','minute60','minute90')))
 report+=['','90-minute resources and exposure','Width |Targets |Updates |Full passes |Training sec |Eval sec |Checkpoint sec |Peak MiB |Min free GiB']
 for width,r in results.items():report.append(f"{width} |{r['valid_targets']} |{r['completed_updates']} |{r['completed_passes']} |{r['training_seconds']:.3f} |{r['validation_seconds']:.3f} |{r['checkpoint_seconds']:.3f} |{r['peak_process_resident_bytes']/2**20:.2f} |{r['minimum_available_system_bytes']/2**30:.3f}")
 report+=['','Instruction diagnostic: same500 synthetic examples x5epochs,625updates; fresh AdamW moments; response+EOS loss only;60heldout prompts; identical greedy32-token decoding. Distinct subject words/names across splits, shared templates. Not a broad instruction or conversation benchmark.','Width |Before exact |After exact |After format |Before response NLL |After response NLL |Instruction train sec |Evaluation sec |Generation sec']
 for width in (192,288,384):
  d=out/f'instruction{width}';r=json.loads((d/'run.json').read_text());assert r['status']=='PASS' and r['checkpoint_reload_verified'];assert r['updates']==625;assert r['valid_response_targets']==r['targets_per_epoch']*5
  counts=0;updates=0
  with (d/'steps.jsonl').open() as f:
   for line in f:
    step=json.loads(line);updates+=1;assert step['update']==updates and math.isfinite(step['loss_sum']) and math.isfinite(step['gradient_norm']);counts+=step['valid_targets']
  assert counts==r['valid_response_targets'] and updates==625
  for phase in ('before','after'):
   rows=json.loads((d/f'{phase}-responses.json').read_text());assert len(rows)==60;assert sum(x['exact_match'] for x in rows)/60==r['metrics'][phase]['exact_match'];assert sum(x['instruction_format'] for x in rows)/60==r['metrics'][phase]['instruction_format']
  low=high=0
  with (d/'telemetry.jsonl').open() as f:
   for line in f:
    s=json.loads(line);assert not s['guard_stop'] and s['available_system_bytes']>=2**30;low=low+1 if s['available_system_bytes']<2*2**30 else 0;high=high+1 if s['resident_bytes']>4*2**30 else 0;assert low<3 and high<3
  b=r['metrics']['before'];a=r['metrics']['after'];instructions[width]=r;report.append(f"{width} |{round(b['exact_match']*60)}/60 |{round(a['exact_match']*60)}/60 |{round(a['instruction_format']*60)}/60 |{b['response_nats_per_target']:.6f} |{a['response_nats_per_target']:.6f} |{r['training_seconds']:.3f} |{r['evaluation_seconds']:.3f} |{r['generation_seconds']:.3f}")
 report+=['','Instruction exact matches by task (each12 cases):']
 for width,r in instructions.items():report.append(str(width)+': '+', '.join(f"{task}={round(m['exact_match']*12)}/12" for task,m in r['metrics']['after']['per_task'].items()))
 advantage=(1-results[288]['validation']['minute90']['bits_per_utf8_text_byte_including_eos_cost']/results[192]['validation']['minute90']['bits_per_utf8_text_byte_including_eos_cost'])*100
 report+=['',f'Interpretation:5.1M led at15/30/60min;10.3M caught up and led by{advantage:.3f}% at90min on heldout story prediction.17.3M remained behind at90min. This is a narrow single-seed result and does not establish a universal width optimum.','The shared instruction diagnostic tests a separate, limited behavior. Per-task failures and subjective coherence review must be considered alongside the aggregate. No automatic production-baseline change.','All pretraining checkpoints hash/reload and exact step/target coverage checks passed. All instruction checkpoints reloaded and step counts/loss finiteness/guard logs passed independent checks. Guard reserve2GiB/3samples, immediate1GiB and process4GiB/3samples unchanged. Final TinyStories test split never evaluated for quality.','Screening and smoke-test updates were independent diagnostic trajectories; their learned weights were discarded. Failed initial timer smoke is retained as engineering diagnostic overhead. No screening/smoke updates are included in90min trajectories.','Limits: one seed; sequential runs with differing background loads; fixed learning rate was not tuned by width; no equal-token or equal-compute control; repeated corpus exposure is not more unique data; syntheticinstruction test is small and shares templates between splits. Story-loss advantage is not proof of reasoning or general conversation.','Production selection and DataEval were not changed. Full run logs, curves, source hashes, data locks and checkpoints are in '+str(out.relative_to(ROOT))+'.']
 if (out/'coherence-review.json').exists():
  review=json.loads((out/'coherence-review.json').read_text());report+=['','COHERENCE AND REPETITION REVIEW',review['evaluator'],review['interpretation']]
  for width,metrics in review['summary'].items():report.append(str(width)+': '+json.dumps(metrics))
  report+=['','CONCLUSION','Retain the approved 5.1M production baseline for now. The 10.3M model is a promising research candidate, with a narrow 90-minute story-prediction advantage after trailing at shorter budgets. The 17.3M model requires more RAM and remains behind at 90 minutes. Instruction differences are only 3-4 answers out of 60; copying, extraction and mapping all failed. Confirm with another initialization seed before promotion. No further run was launched.','All 360 pre/post responses: WidthStudy_InstructionResponses.json. Per-example qualitative ratings: WidthStudy_CoherenceReview.json.']
 text='\n'.join(report)+'\n';(out/'VerifiedFindings.txt').write_text(text);(ROOT/'results/WidthStudy_Results.txt').write_text(text)
 combined=dict(status='PASS',pretraining=results,instruction=instructions,story_90min_advantage_288_vs192_percent=advantage,production_baseline_unchanged=True,final_test_reserved=True,artifacts=str(out.relative_to(ROOT)))
 if (out/'coherence-review.json').exists():
  combined.update(coherence_review=review['summary'],qualitative_evaluator=review['evaluator'],qualitative_limits=review['interpretation'])
 (ROOT/'results/WidthStudy_VerifiedSummary.json').write_text(json.dumps(combined,indent=2)+'\n');(out/'final-verification.json').write_text(json.dumps(dict(status='PASS',instructions='step sequence, counts, finite values, responses and guard telemetry verified',pretraining='per-run verification.json'),indent=2)+'\n');print(text)
if __name__=='__main__':main()
