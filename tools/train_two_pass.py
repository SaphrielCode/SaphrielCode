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
    parser.add_argument('--resume-recovery', type=Path)
    parser.add_argument('--track-app-memory', action='store_true')
    parser.add_argument('--reserve-gib', type=int, choices=(1, 2), default=2)
    parser.add_argument('--vocab-size', type=int, choices=(4096, 8192), default=8192)
    parser.add_argument('--tokenizer-kind', choices=('bpe','unigram'), default='bpe')
    args = parser.parse_args()
    if args.tokenizer_kind=='unigram' and args.vocab_size!=8192:
        parser.error('The current unigram comparison uses 8192 entries')
    if args.resume_pass_one and args.resume_recovery:
        parser.error('Choose only one resume mode')
    policy = Policy(available_reserve_bytes=args.reserve_gib * 1024**3)
    import psutil
    prefix = ('TwoPassA_Unigram8192_' if args.tokenizer_kind=='unigram' else 'TwoPassA_' if args.vocab_size==8192 else 'TwoPassA_BPE4096_')
    out = ROOT / 'results' / (prefix + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    print(str(out), flush=True)
    report = dict(status='RUNNING', completed_updates=0, valid_targets_processed=0,
                  completed_passes=0, passes=[], validation={}, checkpoints=[], continuations=[],
                  training_seconds=0., validation_seconds=0., checkpoint_seconds=0.,
                  final_test_evaluated=False, extra_warmup_updates=0,
                  configuration=dict(candidate='A', batch=4, context=128, threads=8,
                                     interop_threads=1, initialization_seed=42, order_seed=43,vocabulary_size=args.vocab_size),
                  expected_targets_per_pass=4401479, expected_total_targets=8802958,
                  guard_policy=asdict(policy))
    report['tokenizer_kind']=args.tokenizer_kind
    if args.tokenizer_kind=='unigram':
        report['configuration']['tokenizer_kind']='unigram'
    if args.reserve_gib == 1:
        report['guard_authorization'] = 'User explicitly requested 1 GiB stopping threshold for finishing this experiment; first sample below 1 GiB stops training. Per-process 4 GiB consecutive-sample guard unchanged.'
    planned_batches=ROOT/('data/TinyStoriesBatches128' if args.vocab_size==8192 else 'data/TinyStoriesBatches128_BPE4096')
    if args.tokenizer_kind=='unigram':
        planned_batches=ROOT/'data/TinyStoriesBatches128_Unigram8192'
    planned_manifest=json.loads((planned_batches/'manifest.json').read_text(encoding='utf-8'))
    report['expected_targets_per_pass']=planned_manifest['statistics']['train']['valid_target_tokens']
    report['expected_total_targets']=2*report['expected_targets_per_pass']
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
    app_log = (out/'application-memory.jsonl').open('w',encoding='utf-8') if args.track_app_memory else None
    last_app_sample = -10.
    report['application_memory_tracking'] = 'Every five seconds: grouped resident/private bytes, CPU time and I/O counters; group RSS can double-count shared pages and must not be summed as machine RAM. Private bytes are committed allocation, not additional resident RAM.' if app_log else None

    def sample():
        nonlocal last_app_sample
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
        if app_log and (record['seconds']-last_app_sample>=5 or reason):
            last_app_sample = record['seconds']
            groups={}
            for app in psutil.process_iter(['pid','name','memory_info','cpu_times']):
                try:
                    entry=app.info
                    m=entry['memory_info']; cpu=entry['cpu_times']
                    if m is None or cpu is None:
                        continue
                    g=groups.setdefault(entry['name'] or '?',dict(processes=0,resident_bytes=0,private_bytes=0,cpu_seconds=0.,read_bytes=0,write_bytes=0,io_unavailable=0))
                    g['processes']+=1; g['resident_bytes']+=m.rss
                    g['private_bytes']+=getattr(m,'private',0)
                    g['cpu_seconds']+=cpu.user+cpu.system
                    try:
                        io=app.io_counters(); g['read_bytes']+=io.read_bytes;g['write_bytes']+=io.write_bytes
                    except (psutil.AccessDenied,psutil.NoSuchProcess):
                        g['io_unavailable']+=1
                except (psutil.AccessDenied,psutil.NoSuchProcess,TypeError):
                    continue
            app_log.write(json.dumps(dict(seconds=record['seconds'],phase=phase,available_system_bytes=available,groups=groups))+'\n')
            app_log.flush()

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
                        parameter_sha256=parameter_hash(model),
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
        source_dir = ROOT/'data/TinyStoriesSubset'
        if args.vocab_size==4096:
            BATCH_DIR = ROOT/'data/TinyStoriesBatches128_BPE4096'
            source_dir = ROOT/'data/TinyStoriesSubset_BPE4096'
        if args.tokenizer_kind=='unigram':
            BATCH_DIR=ROOT/'data/TinyStoriesBatches128_Unigram8192'
            source_dir=ROOT/'data/TinyStoriesSubset_Unigram8192'
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        lock = verify_batch_lock(BATCH_DIR,source_dir)
        token_config = json.loads((BATCH_DIR/'tokenizer-lock.json').read_text(encoding='utf-8'))
        assert token_config['vocabulary_size']==args.vocab_size
        encoding_config=json.loads((source_dir/'tokenizer.json').read_text(encoding='utf-8'))
        assert encoding_config['model']['type']==('Unigram' if args.tokenizer_kind=='unigram' else 'BPE')
        assert digest(source_dir/'tokenizer.json')==lock['tokenizer_sha256']
        train_targets = lock['statistics']['train']['valid_target_tokens']
        val_targets = lock['statistics']['validation']['valid_target_tokens']
        train_windows = lock['statistics']['train']['windows']
        updates_per_pass = (train_windows+3)//4
        report.update(expected_targets_per_pass=train_targets,expected_total_targets=2*train_targets,
                      expected_updates_per_pass=updates_per_pass,batch_directory=str(BATCH_DIR.relative_to(ROOT)),
                      source_directory=str(source_dir.relative_to(ROOT)))
        check()
        report.update(torch_version=torch.__version__, numpy_version=np.__version__, python=sys.version,
                      tokenizer_sha256=lock['tokenizer_sha256'], batch_manifest_sha256=digest(BATCH_DIR/'manifest.json'))
        train_x = np.fromfile(BATCH_DIR/'train.x.u16', dtype='<u2').reshape(-1,128)
        train_y = np.fromfile(BATCH_DIR/'train.y.u16', dtype='<u2').reshape(-1,128)
        order = np.fromfile(BATCH_DIR/'train.order.u32', dtype='<u4')
        val_x = np.fromfile(BATCH_DIR/'validation.x.u16', dtype='<u2').reshape(-1,128)
        val_y = np.fromfile(BATCH_DIR/'validation.y.u16', dtype='<u2').reshape(-1,128)
        assert len(order)==train_windows and np.array_equal(np.sort(order),np.arange(train_windows))
        assert np.count_nonzero(train_y)==train_targets
        model = Decoder(Config(vocab_size=args.vocab_size), seed=42)
        report['unique_parameters'] = sum(p.numel() for p in model.parameters())
        assert report['unique_parameters']==5115072+(args.vocab_size-8192)*192
        assert model.output_weight is model.embedding.weight
        report['initial_parameter_sha256'] = parameter_hash(model)
        if args.vocab_size==8192:
            assert report['initial_parameter_sha256']=='d9eb1198a6f3f166db7de535d96f0fa3b0dd3dcc92c03b1344031c70333027bb'
        optimizer = make_optimizer(model)
        tokenizer = Tokenizer.from_file(str(source_dir/'tokenizer.json'))
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
            assert restored['completed_passes']==1 and restored['completed_updates']==updates_per_pass
            assert restored['valid_targets']==train_targets
            assert restored['tokenizer_sha256']==report['tokenizer_sha256']
            assert restored['batch_manifest_sha256']==report['batch_manifest_sha256']
            assert restored['config']==asdict(model.config)
            model.load_state_dict(restored['model'])
            optimizer.load_state_dict(restored['optimizer'])
            assert parameter_hash(model)==saved['parameter_sha256']==restored['parameter_sha256']
            assert all(int(s['step'])==updates_per_pass for s in optimizer.state.values())
            del restored
            report.update(completed_updates=updates_per_pass,valid_targets_processed=train_targets,completed_passes=1,
                          passes=[copy.deepcopy(prior['passes'][0])],validation=copy.deepcopy(prior['validation']),
                          continuations=copy.deepcopy(prior['continuations']),
                          checkpoints=[dict(saved,file=str(source),inherited=True)],
                          training_seconds=prior['passes'][0]['seconds'],
                          validation_seconds=prior['validation_seconds'],checkpoint_seconds=prior['checkpoint_seconds'])
            report['resume'] = dict(source=str(source.relative_to(ROOT)),verified_parameter_sha256=saved['parameter_sha256'],
                                   discarded_previous_partial_updates=prior['completed_updates']-updates_per_pass,
                                   discarded_previous_partial_targets=prior['valid_targets_processed']-train_targets,
                                   semantics='Continue the original fresh-seed42 trajectory from completed pass1, including Adam moments/step. Repeat all of pass2; old unsaved partial pass2 is not counted in this trajectory.')
            save_json(out/'run.json',report)
            print(json.dumps(dict(resume='PASS',source=report['resume'],guard_policy=report['guard_policy'])),flush=True)
        if args.resume_recovery:
            source=args.resume_recovery.resolve()
            if not source.is_relative_to((ROOT/'results').resolve()):
                raise RuntimeError('Recovery source must be inside workspace results')
            prior=json.loads((source.parent/'run.json').read_text(encoding='utf-8'))
            assert source.name==prior['recovery_checkpoint']['file']
            assert prior['configuration']==report['configuration']
            for name in ('lm/model.py','lm/batches.py'):
                assert digest(ROOT/name)==prior['source_hashes'][name]
            restored=torch.load(source,map_location='cpu',weights_only=True)
            assert restored['tokenizer_sha256']==report['tokenizer_sha256']
            assert restored['batch_manifest_sha256']==report['batch_manifest_sha256']
            assert restored['config']==asdict(model.config)
            assert restored['completed_updates']==prior['recovery_checkpoint']['completed_updates']
            assert restored['valid_targets']==prior['recovery_checkpoint']['valid_targets']
            assert restored['completed_passes'] in (0,1)
            current=restored['passes'][-1]
            assert not current['completed'] and current['pass_number']==restored['completed_passes']+1
            assert restored['next_row_in_current_pass']==current['completed_updates']*4
            assert restored['next_row_in_current_pass']<len(order)
            # Verify exact data position and valid-target coverage before resuming.
            prefix_rows=order[:restored['next_row_in_current_pass']]
            assert sum(int(np.count_nonzero(train_y[i])) for i in prefix_rows)==current['valid_targets']
            assert restored['valid_targets']==restored['completed_passes']*train_targets+current['valid_targets']
            model.load_state_dict(restored['model'])
            optimizer.load_state_dict(restored['optimizer'])
            assert all(int(s['step'])==restored['completed_updates'] for s in optimizer.state.values())
            resume_hash=parameter_hash(model)
            if 'parameter_sha256' in restored:
                assert resume_hash==restored['parameter_sha256']
            report.update(completed_updates=restored['completed_updates'],valid_targets_processed=restored['valid_targets'],
                          completed_passes=restored['completed_passes'],passes=copy.deepcopy(restored['passes']),
                          validation=copy.deepcopy(restored['validation']),checkpoints=copy.deepcopy(prior['checkpoints']),
                          continuations=copy.deepcopy(prior['continuations']),training_seconds=prior['training_seconds'],
                          validation_seconds=prior['validation_seconds'],checkpoint_seconds=prior['checkpoint_seconds'])
            for checkpoint in report['checkpoints']:
                saved_path=Path(checkpoint['file'])
                if not saved_path.is_absolute():
                    checkpoint['file']=str(source.parent/saved_path)
            report['recovery_resume']=dict(source=str(source.relative_to(ROOT)),sha256=digest(source),
                                           restored_parameter_sha256=resume_hash,
                                           restored_updates=restored['completed_updates'],
                                           restored_valid_targets=restored['valid_targets'],
                                           next_row=restored['next_row_in_current_pass'],
                                           semantics='Restore weights, AdamW moments/step and exact frozen data cursor; no previous updates or targets repeated.')
            del restored
            save_json(out/'run.json',report)
            print(json.dumps(dict(recovery_resume='PASS',**report['recovery_resume'])),flush=True)

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
            assert n==val_targets
            elapsed = time.perf_counter()-at
            report['validation'][label] = dict(loss_sum=total, valid_targets=n, loss_nats_per_target=total/n, seconds=elapsed)
            report['validation_seconds'] += elapsed
            print(json.dumps({'validation': label, **report['validation'][label]}), flush=True)

        if not args.resume_pass_one and not args.resume_recovery:
            evaluate('initialization')
        next_pass = report['completed_passes']+1
        for pass_number in range(next_pass,3):
            check()
            phase = f'training_pass_{pass_number}'
            model.train()
            resuming_partial=bool(args.resume_recovery and report['passes'] and report['passes'][-1]['pass_number']==pass_number and not report['passes'][-1]['completed'])
            if resuming_partial:
                pass_info=report['passes'][-1]
            else:
                pass_info = dict(pass_number=pass_number, completed_updates=0, valid_targets=0, loss_sum=0., seconds=0., completed=False)
                report['passes'].append(pass_info)
            inherited_seconds=pass_info['seconds']
            first_row=pass_info['completed_updates']*4
            pass_start = time.perf_counter()
            try:
                with (out/f'pass{pass_number}-steps.jsonl').open('w') as log:
                    for start in range(first_row,len(order),4):
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
                            pass_info['seconds'] = inherited_seconds+time.perf_counter()-pass_start
                            recovery_checkpoint()
                        if pass_info['completed_updates']%100==0:
                            pass_info['seconds'] = inherited_seconds+time.perf_counter()-pass_start
                            save_json(out/'run.json',report)
                            log.flush()
                            print(json.dumps(dict(pass_number=pass_number, updates=pass_info['completed_updates'],
                                                  valid_targets=pass_info['valid_targets'], seconds=pass_info['seconds'])),flush=True)
            finally:
                session_pass_seconds=time.perf_counter()-pass_start
                pass_info['seconds'] = inherited_seconds+session_pass_seconds
                report['training_seconds'] += session_pass_seconds
            assert pass_info['valid_targets']==train_targets and pass_info['completed_updates']==updates_per_pass
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
        assert report['valid_targets_processed']==2*train_targets and report['completed_updates']==2*updates_per_pass
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
        if app_log:
            app_log.close()
        report['wall_seconds']=time.perf_counter()-started
        report['peak_process_resident_bytes']=max((s['peak_resident_bytes'] for s in samples),default=0)
        report['minimum_available_system_bytes']=min((s['available_system_bytes'] for s in samples),default=0)
        report['stop_semantics']='Monitor latches the first guard violation. Training checks it before forward/backward/update and between validation/generation batches; an already executing operator cannot be preempted. No new optimizer update is started after the stop check detects the latch.'
        save_json(out/'run.json',report)
        lines=['CONTROLLED TWO-PASS CANDIDATE A LEARNING RUN','',f'Status: {report["status"]}',f'Tokenizer: {args.tokenizer_kind}; vocabulary: {args.vocab_size}.',
               'Fresh seed42 weights; batch4/context128/8 CPU threads/1 inter-op; FP32 AdamW unchanged; next-token objective only.',
               'Same seed43 frozen row permutation reused on each pass; no extra warmup updates. Padding ignored; stories isolated.',
               f'Completed passes: {report["completed_passes"]}/2; updates: {report["completed_updates"]}; valid targets: {report["valid_targets_processed"]}/{report["expected_total_targets"]}.',
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
        if args.resume_recovery:
            lines.extend(['Resumed saved weights, AdamW state and exact data cursor. Prior completed work is included once in trajectory totals.',
                          'Recovery provenance: '+json.dumps(report.get('recovery_resume'))])
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
