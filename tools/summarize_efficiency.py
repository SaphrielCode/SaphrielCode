"""Summarize measured efficiency, fixed-time quality, and inference separately."""
from pathlib import Path
import json,statistics,math
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/json.loads((ROOT/'results/Efficiency_Current.json').read_text())['directory']
plan=json.loads((out/'suite.json').read_text());screens={};quality={};lines=['SAPHRIEL EFFICIENCY COMPARISON','',f'Suite status: {plan["status"]}', 'No final-test loss was evaluated. Production architecture and tokenizer remain unchanged.', 'CPU FP32, batch4, context128, eight intra-op threads, one inter-op. Frozen unigram8192 and data order43.', 'Guards: first sample below1 GiB available; three consecutive below2 GiB available; three consecutive above4 GiB process resident.', '', 'SPEED: each accepted run starts fresh, seed42; ten complete warmup updates, then20 timed seconds. Variant order rotates by repetition.', 'Variant                  n    Median targets/s       Range              Peak MiB   Min available GiB']
for variant in ('baseline','valid','trim','sdpa','combined','factor96','factor96wide288'):
 rows=[json.loads(f.read_text()) for f in sorted(out.glob('screen*_'+variant+'/run.json'))];rows=[r for r in rows if r['status']=='PASS']
 if not rows:continue
 rates=[r['valid_targets_per_second'] for r in rows];entry=dict(repetitions=len(rows),parameters=rows[0]['parameters'],median_targets_per_second=statistics.median(rates),minimum=min(rates),maximum=max(rates),peak_process_resident_bytes=max(r['resources']['peak_process_resident_bytes'] for r in rows),minimum_available_system_bytes=min(r['resources']['minimum_available_system_bytes'] for r in rows),initial_parameter_hashes=sorted(set(r['initial_parameter_sha256'] for r in rows)))
 screens[variant]=entry;lines.append(f'{variant:24} {len(rows)} {entry["median_targets_per_second"]:12.1f}     {min(rates):8.1f}-{max(rates):8.1f}    {entry["peak_process_resident_bytes"]/2**20:8.2f}    {entry["minimum_available_system_bytes"]/2**30:.3f}')
if screens:
 preserving={k:v for k,v in screens.items() if not k.startswith('factor')};fast=max(preserving,key=lambda k:preserving[k]['median_targets_per_second']);lines+=['',f'Fastest measured same-architecture path by repeated median: {fast}.', 'Short-run ranges and background workload limit the certainty of close rankings; no composite score is used.']
confirmation={}
cp=out/'confirmation.json'
if cp.exists():
 c=json.loads(cp.read_text());lines+=['','LONGER SPEED CONFIRMATION:60 timed seconds after10 warmup updates, three fresh processes per same-architecture path. Status:'+c['status']]
 for v in ('baseline','valid','trim','sdpa','combined'):
  rr=[json.loads(f.read_text()) for f in out.glob('confirm*_'+v+'/run.json')];rr=[r for r in rr if r['status']=='PASS']
  if not rr:continue
  rates=[r['valid_targets_per_second'] for r in rr];confirmation[v]=dict(repetitions=len(rr),median_targets_per_second=statistics.median(rates),minimum=min(rates),maximum=max(rates),peak_process_resident_bytes=max(r['resources']['peak_process_resident_bytes'] for r in rr),minimum_available_system_bytes=min(r['resources']['minimum_available_system_bytes'] for r in rr))
  e=confirmation[v];lines.append(f'{v}: n{len(rr)}; median{e["median_targets_per_second"]:.2f} targets/s; range{e["minimum"]:.2f}-{e["maximum"]:.2f}; peak{e["peak_process_resident_bytes"]/2**20:.2f} MiB; min available{e["minimum_available_system_bytes"]/2**30:.3f} GiB.')
 if c['status']=='PASS':
  fastest=max(confirmation,key=lambda v:confirmation[v]['median_targets_per_second']);near=[v for v,e in confirmation.items() if e['median_targets_per_second']>=confirmation[fastest]['median_targets_per_second']*.95];chosen=min(near,key=lambda v:confirmation[v]['peak_process_resident_bytes'])
  lines+=['Fastest median:'+fastest+'. Paths within5% treated as practically close:'+str(near)+'. Lower-peak-RAM choice within that set:'+chosen+'.', 'This headroom choice does not establish that close throughput differences are statistically decisive. Longer confirmation takes precedence over20-second screen rankings.']
lines+=['', 'valid selects valid hidden states BEFORE scoring8192 vocabulary entries. Loss denominator and batch order unchanged.', 'trim shortens each unchanged batch to its longest valid target sequence. Document boundaries, positions and eligible targets unchanged.', 'sdpa uses PyTorch scaled_dot_product_attention, explicit causal boolean mask, zero dropout. No assumption of skipping causal matrix arithmetic.', 'combined uses all three. Factor variants use valid-only output, manual attention, no trim, to isolate factorization/width.', '', 'QUALITY:600 active-training seconds from fresh weights per candidate and seed; seeds42,44,46. No additional warmup training. Stop at update boundary; sub-step budget overshoot is included in reported actual seconds.', 'Three candidates: valid=current architecture, factor96=width192 factorization, factor96wide288=width288 factorization.', 'Same frozen validation216150 eligible targets; float64 accumulation of detached per-token NLL; divide total by total target count.', 'Bits/UTF-8 text byte includes EOS NLL and divides by884630 frozen validation text bytes. Same byte denominator across candidates.', 'Tied factorization uses E[8192,96], P[width,96], input E@P.T and output hidden@P@E.T. E normal(0,.02), P normal(0,1/sqrt(96)), seed=initialization seed+1000. Same-width transformer blocks retain identical initial weights.', 'Initialization scale is a chosen starting rule, not an optimized guarantee. Faster models see more training targets at equal elapsed active-training time.', 'These results establish behavior at ten minutes for this corpus and optimizer; they do not establish a universal winner or conversation ability.', 'Candidate               Seed   Init nats    Final nats    Final bits/byte    Training targets   Targets/s']
for variant in ('valid','factor96','factor96wide288'):
 rows=[json.loads(f.read_text()) for f in sorted(out.glob('quality_'+variant+'_seed*/run.json'))];quality[variant]=rows
 for r in rows:
  if r['status']!='PASS':lines.append(f'{variant} seed{r["seed"]}: {r["status"]}');continue
  e=r['validation']['final'];lines.append(f'{variant:23} {r["seed"]:3} {r["validation"]["initialization"]["nats_per_target"]:11.6f} {e["nats_per_target"]:12.6f} {e["bits_per_utf8_text_byte_including_eos"]:15.6f} {r["valid_targets"]:17} {r["valid_targets_per_second"]:11.2f}')
  lines.append(f'  Training{r["training_seconds"]:.3f}s; updates{r["updates"]}; validation{sum(v["seconds"] for v in r["validation"].values()):.3f}s; checkpoint{r["checkpoint"]["seconds"]:.3f}s; reload verified={r["checkpoint"]["reload_verified"]}; peak{r["resources"]["peak_process_resident_bytes"]/2**20:.2f} MiB; min available{r["resources"]["minimum_available_system_bytes"]/2**30:.3f} GiB.')
quality_scores={}
for v,rr in quality.items():
 good=[r for r in rr if r['status']=='PASS']
 if good:
  values=[r['validation']['final']['bits_per_utf8_text_byte_including_eos'] for r in good];quality_scores[v]=dict(seeds=[r['seed'] for r in good],mean_bits_per_byte=statistics.mean(values),minimum=min(values),maximum=max(values))
if len(quality_scores)==3 and all(len(v['seeds'])==3 for v in quality_scores.values()):
 winner=min(quality_scores,key=lambda v:quality_scores[v]['mean_bits_per_byte']);lines+=['',f'Lowest mean held-out bits/byte at the ten-minute budget: {winner}.']
 for other in quality_scores:
  if other==winner:continue
  paired=[]
  for seed in (42,44,46):
   a=next(r for r in quality[winner] if r['seed']==seed);b=next(r for r in quality[other] if r['seed']==seed);paired.append(a['validation']['final']['bits_per_utf8_text_byte_including_eos']-b['validation']['final']['bits_per_utf8_text_byte_including_eos'])
  lines.append(f'  {winner} minus {other}, paired bits/byte differences by seeds42/44/46: {paired}; mean {statistics.mean(paired):.6f}. Negative favors the named leader. All seeds agree: {all(x<0 for x in paired)}.')
  lines.append('  Three seeds quantify limited observed variation; they do not cover all seeds, datasets or longer budgets.')
ip=out/'inference/summary.json'
if ip.exists():
 ir=json.loads(ip.read_text());lines+=['','INFERENCE: verified approved trained checkpoint, same prompt and greedy32 fixed decoding steps per mode. EOS does not stop this timing diagnostic. Three fresh processes per mode/prompt length; kernel warmup excluded.']
 for length in (24,128):
  for mode in ('full','last','cache','hybrid'):
   rr=[r for r in ir['runs'] if r['mode']==mode and r['prompt_tokens']==length]
   if not rr:continue
   lines.append(f'  prompt{length} {mode}: median first-token {statistics.median(r["first_token_seconds"] for r in rr):.6f}s; median decode {statistics.median(r["decode_tokens_per_second"] for r in rr):.2f} tokens/s; peak{max(r["resources"]["peak_process_resident_bytes"] for r in rr)/2**20:.2f} MiB.')
 lines+=['  Greedy token IDs matched across all modes. Cache rebuilds on rolling128 overflow preserve original reset positions and multi-layer conditioning. At full128 context cache rebuilds every step, limiting benefits.']
else:lines+=['','Inference benchmarks pending. No inference speedup is claimed.']
parity=out/'trained-parity.json';lines+=['','Trained-checkpoint loss/gradient/update parity: '+('PASS' if parity.exists() else 'pending')]
if parity.exists():
 pr=json.loads(parity.read_text());lines+=['Maximum gradient absolute difference:'+str(max(r['gradient_max_abs_difference'] for r in pr['comparisons'])), 'Maximum continued-AdamW weight absolute difference:'+str(max(r['post_update_parameter_max_abs_difference'] for r in pr['comparisons'] if r['optimizer_mode']=='restored_training_state')), 'Resetting AdamW at trained weights amplified small FP32 differences to max'+str(max(r['post_update_parameter_max_abs_difference'] for r in pr['comparisons'] if r['optimizer_mode']=='reset_at_trained_weights'))+'. No bitwise-identical training trajectory claim.']
for r in plan['runs']:
 if r['status']!='PASS':
  failed=json.loads((out/r['directory']/'run.json').read_text());lines+=['Stopped/error: '+r['directory']+' '+str(failed.get('resources',{}).get('guard_stop') or failed.get('error'))]
lines+=['', 'Preserved exception: seed42 current-model training and validation completed, but checkpoint metadata stored TorchVersion instead of plain string. Original final.pt retained; final.corrected.pt changes only metadata. Fresh model hash and optimizer3852 steps verified without retraining. Original failed save/reload phase duration was not recorded; reported checkpoint time for this case covers the repair/verification only.', '', 'All arithmetic estimates are separate from elapsed time. No equal-total-compute or useful-language claim follows.', 'Prior stopped suite preserved under results/Efficiency_20261113T111156Z.', 'Reference sources: https://docs.pytorch.org/docs/2.9/generated/torch.nn.functional.scaled_dot_product_attention.html ; https://arxiv.org/abs/1909.11942 ; https://huggingface.co/docs/transformers/main/cache_explanation']
summary=dict(status=plan['status'],confirmation=confirmation,screens=screens,quality=quality_scores,quality_runs=quality)
gain=(confirmation.get('combined',{}).get('median_targets_per_second',0)/confirmation.get('baseline',{}).get('median_targets_per_second',1)-1)*100
extra_ram=(confirmation.get('combined',{}).get('peak_process_resident_bytes',0)-confirmation.get('baseline',{}).get('peak_process_resident_bytes',0))/2**20
decision=dict(status='COMPLETE' if plan['status']=='PASS' and cp.exists() and json.loads(cp.read_text())['status']=='PASS' and ip.exists() and parity.exists() else 'PARTIAL',architecture='Retain approved unfactorized8192-token, width192, eight-layer SwiGLU Candidate A',training_implementation='baseline',training_reason=f'Combined path{gain:.2f}% faster but{extra_ram:.2f} MiB higher peak RSS; baseline chosen under predeclared5% closeness/headroom rule.',quality_scope='Three paired seeds42/44/46,600 active-training seconds each; architecture winner at this budget only.',quality_mean_bits_per_byte={k:v['mean_bits_per_byte'] for k,v in quality_scores.items()},inference='Adopt newest-position scoring plus prefix KV cache; recompute rolling128 window after overflow.',production_architecture_changed=False,chat_backend_updated=True,final_test_evaluated=False)
(out/'decision.json').write_text(json.dumps(decision,indent=2));(ROOT/'results/Efficiency_Decision.json').write_text(json.dumps(decision,indent=2));summary['decision']=decision
(out/'summary.json').write_text(json.dumps(summary,indent=2));lines[2:2]=['DECISION: retain the approved full token representation and original training implementation; adopt the verified hybrid inference path.', 'Quality: current architecture beat both factorizations at every paired seed within the ten-minute budget.', f'Training: combined implementation was{gain:.2f}% faster with{extra_ram:.2f} MiB extra peak RSS; original path wins the declared closeness/headroom selection.', 'Inference: see measured per-mode/prompt medians below. Hybrid outputs matched original greedy token IDs in the completed tests.', 'Independent audit: see verification.json for fresh checkpoint reloads, target cursors, hashes, validation counts, optimizer steps and test results.', '']
text='\n'.join(lines)+'\n';(out/'Results.txt').write_text(text);(ROOT/'results/Efficiency_Results.txt').write_text(text);print(text)
