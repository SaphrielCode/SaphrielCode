"""Fresh, exactly two-pass Candidate A training with strict sampled guards."""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lm.resource_guard import Guard, Policy
from cpu_screen_worker import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-pass-one', type=Path)
    parser.add_argument('--reserve-gib', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    policy = Policy(available_reserve_bytes=args.reserve_gib * 1024**3)
    import psutil
    out = ROOT / 'results' / ('TwoPassA_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    print(str(out), flush=True)
    report = dict(status='RUNNING', completed_updates=0, valid_targets_processed=0,
                  completed_passes=0, passes=[], validation={}, checkpoints=[], continuations=[],
                  training_seconds=0., validation_seconds=0., checkpoint_seconds=0.,
                  final_test_evaluated=False, extra_warmup_updates=0,
                  configuration=dict(candidate='A', batch=4, context=128, threads=8,
                                     interop_threads=1, initialization_seed=42, order_seed=43),
                  expected_targets_per_pass=4401479, expected_total_targets=8802958,
                  guard_policy=asdict(policy))
    if args.reserve_gib == 1:
        report['guard_authorization'] = 'User explicitly requested 1 GiB stopping threshold for finishing this experiment; first sample below 1 GiB stops training. Per-process 4 GiB consecutive-sample guard unchanged.'
    for name in ('lm/model.py', 'lm/batches.py', 'lm/resource_guard.py', 'tools/train_two_pass.py'):
        report.setdefault('source_hashes', {})[name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    phase = 'preflight'
    stop = threading.Event()
    done = threading.Event()
    guard = Guard(policy)
    process = psutil.Process()
    samples = []
    started = time.perf_counter()
    telemetry = (out/'telemetry.jsonl').open('w', encoding='utf-8')

    def sample():
        memory = process.memory_info()
        available = psutil.virtual_memory().available
        reason = guard.observe(available, [{'pid': process.pid, 'resident_bytes': memory.rss}])
        record = dict(seconds=time.perf_counter()-started, phase=phase, available_system_bytes=available,
                      resident_bytes=memory.rss, peak_resident_bytes=getattr(memory, 'peak_wset', memory.rss),
                      private_bytes=getattr(memory, 'private', None), cpu_seconds=sum(process.cpu_times()[:2]),
                      guard_stop=reason)
        samples.append(record)
        telemetry.write(json.dumps(record)+'\n')
        telemetry.flush()
        if reason:
            report['guard_stop'] = record
            stop.set()

    def watch():
        try:
            while not done.wait(1):
                sample()
                if stop.is_set():
                    return
        except Exception as error:
            report['monitor_error'] = repr(error)
            stop.set()

    def check():
        if stop.is_set():
            raise RuntimeError('Resource monitor requested stop')

    watcher = None
    model = optimizer = None
    torch = None
    def recovery_checkpoint():
        if model is None or optimizer is None or torch is None:
            return
        # State dictionaries reference existing storage; do not duplicate model
        # or Adam buffers while saving under memory pressure.
        before = time.perf_counter()
        path = out/'recovery.pt'
        temporary = out/'recovery.pt.tmp'
        torch.save(dict(config=asdict(model.config),model=model.state_dict(),optimizer=optimizer.state_dict(),
                        completed_updates=report['completed_updates'],valid_targets=report['valid_targets_processed'],
                        completed_passes=report['completed_passes'],passes=copy.deepcopy(report['passes']),
                        tokenizer_sha256=report['tokenizer_sha256'],batch_manifest_sha256=report['batch_manifest_sha256'],
                        validation=copy.deepcopy(report['validation']),
                        next_row_in_current_pass=(report['passes'][-1]['completed_updates']*4 if report['passes'] and not report['passes'][-1]['completed'] else 0)),temporary)
        temporary.replace(path)
        elapsed = time.perf_counter()-before
        report['recovery_checkpoint'] = dict(file=path.name,completed_updates=report['completed_updates'],
                                             valid_targets=report['valid_targets_processed'],seconds=elapsed)
        report['checkpoint_seconds'] += elapsed
    try:
        for i in range(3):
            sample()
            check()
            if i < 2:
                time.sleep(1)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        phase = 'imports_and_integrity'
        os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
        import numpy as np
        import torch
        from tokenizers import Tokenizer
        from lm.model import Config, Decoder, make_optimizer, parameter_hash, token_loss
        from lm.batches import BATCH_DIR, digest, verify_batch_lock
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        lock = verify_batch_lock()
        check()
        report.update(torch_version=torch.__version__, numpy_version=np.__version__, python=sys.version,
                      tokenizer_sha256=lock['tokenizer_sha256'], batch_manifest_sha256=digest(BATCH_DIR/'manifest.json'))
        train_x = np.fromfile(BATCH_DIR/'train.x.u16', dtype='<u2').reshape(-1,128)
        train_y = np.fromfile(BATCH_DIR/'train.y.u16', dtype='<u2').reshape(-1,128)
        order = np.fromfile(BATCH_DIR/'train.order.u32', dtype='<u4')
        val_x = np.fromfile(BATCH_DIR/'validation.x.u16', dtype='<u2').reshape(-1,128)
        val_y = np.fromfile(BATCH_DIR/'validation.y.u16', dtype='<u2').reshape(-1,128)
        assert len(order)==45204 and len(np.unique(order))==45204
        assert np.count_nonzero(train_y)==4401479
        model = Decoder(Config(), seed=42)
        report['unique_parameters'] = sum(p.numel() for p in model.parameters())
        assert report['unique_parameters']==5115072
        assert model.output_weight is model.embedding.weight
        report['initial_parameter_sha256'] = parameter_hash(model)
        assert report['initial_parameter_sha256']=='d9eb1198a6f3f166db7de535d96f0fa3b0dd3dcc92c03b1344031c70333027bb'
        optimizer = make_optimizer(model)
        tokenizer = Tokenizer.from_file(str(ROOT/'data/TinyStoriesSubset/tokenizer.json'))
        if args.resume_pass_one:
            source = args.resume_pass_one.resolve()
            if not source.is_relative_to((ROOT/'results').resolve()):
                raise RuntimeError('Resume checkpoint must be in workspace results')
            prior = json.loads((source.parent/'run.json').read_text())
            saved = next(c for c in prior['checkpoints'] if c['file']==source.name)
            assert saved['reload_verified'] and digest(source)==saved['sha256']
            for name in ('lm/model.py','lm/batches.py'):
                assert digest(ROOT/name)==prior['source_hashes'][name]
            restored = torch.load(source,map_location='cpu',weights_only=True)
            assert restored['completed_passes']==1 and restored['completed_updates']==11301
            assert restored['valid_targets']==4401479
            assert restored['tokenizer_sha256']==report['tokenizer_sha256']
            assert restored['batch_manifest_sha256']==report['batch_manifest_sha256']
            assert restored['config']==asdict(model.config)
            model.load_state_dict(restored['model'])
            optimizer.load_state_dict(restored['optimizer'])
            assert parameter_hash(model)==saved['parameter_sha256']==restored['parameter_sha256']
            assert all(int(s['step'])==11301 for s in optimizer.state.values())
            del restored
            report.update(completed_updates=11301,valid_targets_processed=4401479,completed_passes=1,
                          passes=[copy.deepcopy(prior['passes'][0])],validation=copy.deepcopy(prior['validation']),
                          continuations=copy.deepcopy(prior['continuations']),
                          checkpoints=[dict(saved,file=str(source),inherited=True)],
                          training_seconds=prior['passes'][0]['seconds'],
                          validation_seconds=prior['validation_seconds'],checkpoint_seconds=prior['checkpoint_seconds'])
            report['resume'] = dict(source=str(source.relative_to(ROOT)),verified_parameter_sha256=saved['parameter_sha256'],
                                   discarded_previous_partial_updates=prior['completed_updates']-11301,
                                   discarded_previous_partial_targets=prior['valid_targets_processed']-4401479,
                                   semantics='Continue the original fresh-seed42 trajectory from completed pass1, including Adam moments/step. Repeat all of pass2; old unsaved partial pass2 is not counted in this trajectory.')
            save_json(out/'run.json',report)
            print(json.dumps(dict(resume='PASS',source=report['resume'],guard_policy=report['guard_policy'])),flush=True)

        def batch(xs, ys, index):
            x = torch.from_numpy(xs[index].astype(np.int64))
            y = torch.from_numpy(ys[index].astype(np.int64))
            y.masked_fill_(y.eq(0), -100)
            return x,y

        def evaluate(label):
            nonlocal phase
            phase = 'validation_'+label
            model.eval()
            total,n = 0.,0
            at = time.perf_counter()
            with torch.no_grad():
                for start in range(0,len(val_x),4):
                    check()
                    x,y = batch(val_x,val_y,slice(start,start+4))
                    _,summed,count = token_loss(model(x), y)
                    total += float(summed)
                    n += count
            assert n==216320
            elapsed = time.perf_counter()-at
            report['validation'][label] = dict(loss_sum=total, valid_targets=n, loss_nats_per_target=total/n, seconds=elapsed)
            report['validation_seconds'] += elapsed
            print(json.dumps({'validation': label, **report['validation'][label]}), flush=True)

        if not args.resume_pass_one:
            evaluate('initialization')
        for pass_number in ((2,) if args.resume_pass_one else (1,2)):
            check()
            phase = f'training_pass_{pass_number}'
            model.train()
            pass_info = dict(pass_number=pass_number, completed_updates=0, valid_targets=0, loss_sum=0., seconds=0., completed=False)
            report['passes'].append(pass_info)
            pass_start = time.perf_counter()
            try:
                with (out/f'pass{pass_number}-steps.jsonl').open('w') as log:
                    for start in range(0,len(order),4):
                        check()
                        step_start = time.perf_counter()
                        x,y = batch(train_x,train_y,order[start:start+4])
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(x)
                        loss,summed,count = token_loss(logits,y)
                        del logits
                        if not torch.isfinite(loss):
                            raise RuntimeError('Non-finite training loss')
                        check()
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True,foreach=False)
                        check()
                        optimizer.step()
                        pass_info['completed_updates'] += 1
                        pass_info['valid_targets'] += count
                        pass_info['loss_sum'] += float(summed)
                        report['completed_updates'] += 1
                        report['valid_targets_processed'] += count
                        log.write(json.dumps(dict(update=report['completed_updates'], valid_targets=count,
                                                  loss_sum=float(summed), gradient_norm=float(norm),
                                                  seconds=time.perf_counter()-step_start))+'\n')
                        if pass_info['completed_updates']%1000==0:
                            pass_info['seconds'] = time.perf_counter()-pass_start
                            recovery_checkpoint()
                        if pass_info['completed_updates']%100==0:
                            pass_info['seconds'] = time.perf_counter()-pass_start
                            save_json(out/'run.json',report)
                            log.flush()
                            print(json.dumps(dict(pass_number=pass_number, updates=pass_info['completed_updates'],
                                                  valid_targets=pass_info['valid_targets'], seconds=pass_info['seconds'])),flush=True)
            finally:
                pass_info['seconds'] = time.perf_counter()-pass_start
                report['training_seconds'] += pass_info['seconds']
            assert pass_info['valid_targets']==4401479 and pass_info['completed_updates']==11301
            assert all(torch.isfinite(p).all() for p in model.parameters())
            pass_info.update(completed=True, valid_targets_per_second=pass_info['valid_targets']/pass_info['seconds'])
            report['completed_passes'] += 1
            evaluate(f'pass_{pass_number}')
            phase = f'checkpoint_pass_{pass_number}'
            check()
            checkpoint_start = time.perf_counter()
            path = out/f'pass{pass_number}.pt'
            expected_hash = parameter_hash(model)
            torch.save(dict(config=asdict(model.config), model=model.state_dict(), optimizer=optimizer.state_dict(),
                            completed_updates=report['completed_updates'], valid_targets=report['valid_targets_processed'],
                            completed_passes=pass_number, tokenizer_sha256=report['tokenizer_sha256'],
                            batch_manifest_sha256=report['batch_manifest_sha256'], parameter_sha256=expected_hash),path)
            # Drop live allocations before reload to avoid duplicating optimizer buffers.
            del optimizer,model,loss,summed,norm
            restored = torch.load(path,map_location='cpu',weights_only=True)
            model = Decoder(Config(**restored['config']),seed=42)
            model.load_state_dict(restored['model'])
            optimizer = make_optimizer(model)
            optimizer.load_state_dict(restored['optimizer'])
            assert parameter_hash(model)==expected_hash
            assert all(int(state['step'])==report['completed_updates'] for state in optimizer.state.values())
            del restored
            elapsed = time.perf_counter()-checkpoint_start
            report['checkpoint_seconds'] += elapsed
            report['checkpoints'].append(dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path),
                                              parameter_sha256=expected_hash,reload_verified=True,save_and_reload_seconds=elapsed))
            phase = f'continuation_pass_{pass_number}'
            model.eval()
            prompt = 'Once upon a time, a little girl found a'
            ids = [1]+tokenizer.encode(prompt).ids
            generated=[]
            with torch.no_grad():
                for _ in range(64):
                    check()
                    next_id=int(model(torch.tensor([ids[-128:]],dtype=torch.long))[0,-1].argmax())
                    generated.append(next_id)
                    ids.append(next_id)
                    if next_id==2:
                        break
            sample_info=dict(pass_number=pass_number,prompt=prompt,decoding='greedy argmax; max 64 new tokens; EOS=2; rolling context128; BOS=1',
                             generated_ids=generated,continuation=tokenizer.decode(generated,skip_special_tokens=True))
            report['continuations'].append(sample_info)
            save_json(out/f'pass{pass_number}-continuation.json',sample_info)
            save_json(out/'run.json',report)
        assert report['valid_targets_processed']==8802958 and report['completed_updates']==22602
        check()
        report['status']='PASS'
        first=report['validation']['pass_1']['loss_nats_per_target']
        second=report['validation']['pass_2']['loss_nats_per_target']
        report['pass_two_validation_loss_change']=second-first
        report['another_pass_assessment']=('A third pass is worth a bounded test because validation loss improved in pass two; this single seed does not establish an optimum.' if second<first else 'Do not extend this configuration automatically: pass two did not improve validation loss. Review the learning schedule and overfitting first.')
    except Exception as error:
        report.update(status='STOPPED' if stop.is_set() else 'FAIL', stopped_phase=phase,error=repr(error),traceback=traceback.format_exc())
        if stop.is_set():
            try:
                recovery_checkpoint()
            except Exception as saving_error:
                report['recovery_save_error'] = repr(saving_error)
    finally:
        done.set()
        if watcher:
            watcher.join(timeout=3)
        telemetry.close()
        report['wall_seconds']=time.perf_counter()-started
        report['peak_process_resident_bytes']=max((s['peak_resident_bytes'] for s in samples),default=0)
        report['minimum_available_system_bytes']=min((s['available_system_bytes'] for s in samples),default=0)
        report['stop_semantics']='Monitor latches the first guard violation. Training checks it before forward/backward/update and between validation/generation batches; an already executing operator cannot be preempted. No new optimizer update is started after the stop check detects the latch.'
        save_json(out/'run.json',report)
        lines=['CONTROLLED TWO-PASS CANDIDATE A LEARNING RUN','',f'Status: {report["status"]}',
               'Fresh seed42 weights; batch4/context128/8 CPU threads/1 inter-op; FP32 AdamW unchanged; next-token objective only.',
               'Same seed43 frozen row permutation reused on each pass; no extra warmup updates. Padding ignored; stories isolated.',
               f'Completed passes: {report["completed_passes"]}/2; updates: {report["completed_updates"]}; valid targets: {report["valid_targets_processed"]}/8,802,958.',
               f'Training time: {report["training_seconds"]:.3f}s; validation: {report["validation_seconds"]:.3f}s; checkpoint save/reload: {report["checkpoint_seconds"]:.3f}s.',
               f'Peak process RAM: {report["peak_process_resident_bytes"]/2**20:.2f} MiB; minimum available RAM: {report["minimum_available_system_bytes"]/2**30:.3f} GiB.',
               f'Available-RAM reserve: {args.reserve_gib} GiB; first sample below1GiB stops; three consecutive below the reserve stop; three consecutive above4GiB process RSS stop. Sampling interval1s.',
               'Final test remains reserved. No instruction training or replacement detection. No DataEval edits.']
        for label,value in report['validation'].items():
            lines.append(f'Validation {label}: {value["loss_nats_per_target"]:.8f} nats/valid target; sum={value["loss_sum"]:.8f}; count={value["valid_targets"]}.')
        if args.resume_pass_one:
            lines.extend(['Resumed from verified pass1 weights AND AdamW state. Prior unsaved partial pass2 work was discarded; this session repeats pass2 from its beginning.',
                          'Reported trajectory training/validation/checkpoint totals include inherited pass1 work. Session wall time is recorded separately in run.json; interrupted discarded work is additional experiment overhead.',
                          'Resume provenance: '+json.dumps(report.get('resume'))])
        for value in report['passes']:
            lines.append('Pass work: '+json.dumps(value))
        for value in report['continuations']:
            lines.extend(['',f'Pass {value["pass_number"]} prompt: '+value['prompt'],'Decoding: '+value['decoding'],'Continuation: '+value['continuation']])
        lines.extend(['',report.get('another_pass_assessment','Third-pass value cannot be assessed without both pass validation losses.'),
                      'Stop/error: '+str(report.get('guard_stop',report.get('error'))),
                      'Any exposure shortfall is explained by the recorded interruption; partial runs are not two-pass results.',
                      'Details, source hashes, checkpoint verification and per-step/resource logs are stored beside this file.'])
        text='\n'.join(lines)+'\n'
        (out/'Results.txt').write_text(text,encoding='utf-8')
        (ROOT/'results'/('TwoPassA_'+out.name.removeprefix('TwoPassA_')+'_Results.txt')).write_text(text,encoding='utf-8')
        print(json.dumps(dict(status=report['status'],directory=str(out),updates=report['completed_updates'],error=report.get('error'))),flush=True)
    return 0 if report['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(main())
