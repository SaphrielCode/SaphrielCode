"""Four-hour checkpoint adaptation with continuation-only loss and recovery."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8',
                  TOKENIZERS_PARALLELISM='false')

import torch
from tokenizers import Tokenizer
from lm.efficiency import EfficientDecoder, training_loss
from lm.experimental_model import Config, make_optimizer, parameter_hash
from lm.monitor import Monitor
from lm.resource_guard import Policy
from lm.stage_one import encode_example, generate


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def save_checkpoint(path, model, tokenizer_hash, **extra):
    payload = dict(config=asdict(model.config), model=model.state_dict(),
                   parameter_sha256=parameter_hash(model), tokenizer_sha256=tokenizer_hash,
                   task='stage-one', prompt_format='Scene: {scene}\nUser: {user}\nReply:',
                   context=extra.pop('context'), reply_limit=48, **extra)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload['parameter_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=14400)
    parser.add_argument('--reserve-gib', type=float, default=.5)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if args.seconds <= 0 or args.reserve_gib <= 0:
        parser.error('Duration and RAM reserve must be positive')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    out = args.out or ROOT / 'results' / ('StageOne_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    if args.resume:
        out = args.resume.resolve().parent
    out = out.resolve()
    if not out.is_relative_to(ROOT / 'results'):
        parser.error('Run output must be inside results')
    out.mkdir(parents=True, exist_ok=bool(args.resume))
    policy = Policy(available_reserve_bytes=int(args.reserve_gib * 2**30),
                    immediate_available_bytes=int(args.reserve_gib * 2**30))
    monitor = Monitor(out, policy)
    report = dict(status='PREPARING', run_directory=str(out.relative_to(ROOT)),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  target_phase_seconds=args.seconds, phase_seconds=0., optimizer_seconds=0.,
                  updates=0, valid_targets=0, epochs=0, validation=[],
                  timing='Four hours from first training-step start, including periodic validation and checkpoint saves. Preparation and initial/final evaluation excluded. Stop at an update boundary.',
                  hyperparameters=dict(batch=4, threads=8, learning_rate=1e-4,
                                       minimum_learning_rate=1e-5, warmup_updates=100,
                                       schedule='100-update warmup then elapsed-phase cosine',
                                       optimizer='Fresh AdamW moments; betas .9/.95, eps1e-8, matrix decay .1, norm decay0, clip1',
                                       seed=42, order_seed=5301),
                  guard_authorization='User requested .5 GiB available RAM guard; immediate stop below .5 GiB. Existing 4 GiB process guard retained.')
    elapsed_before = 0.
    phase_start = None
    optimizer = model = None
    order, position, epoch = [], 0, 0
    rng = random.Random(5301)
    best_loss = float('inf')
    try:
        monitor.start()
        selection = json.loads((ROOT / 'WorkingSelection.json').read_text())
        source = ROOT / selection['reference_checkpoint']
        if digest(source) != selection['reference_checkpoint_sha256']:
            raise ValueError('Starting checkpoint hash mismatch')
        data = ROOT / 'data/StageOne_v1'
        manifest = json.loads((data / 'manifest.json').read_text())
        tokenizer_path = ROOT / 'model_assets/tokenizer.json'
        tokenizer_hash = digest(tokenizer_path)
        if tokenizer_hash != selection['tokenizer_sha256'] or tokenizer_hash != manifest['tokenizer_sha256']:
            raise ValueError('Tokenizer mismatch')
        context = manifest['context']
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        examples, tensors = {}, {}
        for split in ('train', 'validation'):
            path = data / f'{split}.jsonl'
            if digest(path) != manifest['files'][path.name]:
                raise ValueError('Frozen dataset changed')
            examples[split] = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
            encoded = [encode_example(tokenizer, row, context) for row in examples[split]]
            tensors[split] = (torch.tensor([row[0] for row in encoded]),
                              torch.tensor([row[1] for row in encoded]))
        checkpoint = torch.load(args.resume or source, map_location='cpu', weights_only=True)
        config = Config(**checkpoint['config'])
        if (config.vocab_size, config.layers, config.width, config.heads, config.ff_width, config.ff_activation) != (8192, 8, 192, 6, 512, 'swiglu'):
            raise ValueError('Incompatible checkpoint architecture')
        model = EfficientDecoder(config)
        model.load_state_dict(checkpoint['model'], strict=True)
        if parameter_hash(model) != checkpoint['parameter_sha256']:
            raise ValueError('Checkpoint parameter hash mismatch')
        if checkpoint['tokenizer_sha256'] != tokenizer_hash:
            raise ValueError('Checkpoint tokenizer mismatch')
        optimizer = make_optimizer(model)
        report.update(source_checkpoint=selection['reference_checkpoint'],
                      source_checkpoint_sha256=digest(source),
                      source_parameter_sha256=selection['reference_parameter_sha256'],
                      tokenizer_sha256=tokenizer_hash, dataset_manifest_sha256=digest(data / 'manifest.json'),
                      dataset=manifest, context=context, parameters=sum(p.numel() for p in model.parameters()))
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            report = checkpoint['report']
            elapsed_before = report['phase_seconds']
            order, position, epoch = checkpoint['order'], checkpoint['position'], checkpoint['epoch']
            rng.setstate(checkpoint['order_rng'])
            best_loss = checkpoint['best_loss']
        del checkpoint

        def evaluate(split, label):
            monitor.phase = 'evaluation_' + label
            model.eval()
            x, y = tensors[split]
            total, count = 0., 0
            with torch.inference_mode():
                for index in range(0, len(x), 4):
                    monitor.check()
                    _, summed, valid = training_loss(model, x[index:index+4], y[index:index+4], valid_only=True, trim=True)
                    total += float(summed)
                    count += valid
            row = dict(label=label, split=split, nll_sum=total, valid_targets=count,
                       nats_per_target=total/count, updates=report['updates'],
                       phase_seconds=report['phase_seconds'])
            report['validation'].append(row)
            return total / count

        def probes(label):
            model.eval()
            rows = []
            for row in examples['validation'][-10:]:
                monitor.check()
                result = generate(model, tokenizer, row['scene'], row['user'], context=context,
                                  limit=min(48, context - 1 - len(tokenizer.encode('Scene: '+row['scene']+'\nUser: '+row['user']+'\nReply:').ids)))
                rows.append(dict(**row, generated=result))
            write_json(out / f'{label}-responses.json', rows)

        if not args.resume:
            best_loss = evaluate('validation', 'before')
            probes('before')
            save_checkpoint(out / 'best.pt', model, tokenizer_hash, context=context)
            report['best_update'] = 0
        report['status'] = 'TRAINING'
        report['training_started_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(out / 'run.json', report)
        write_json(ROOT / 'results/StageOne_Current.json', dict(directory=out.relative_to(ROOT).as_posix()))
        x, y = tensors['train']
        phase_start = time.perf_counter()
        next_evaluation = (int(elapsed_before // 900) + 1) * 900
        next_log = elapsed_before
        print(json.dumps(dict(status='TRAINING',directory=str(out),context=context)), flush=True)

        def recovery():
            save_checkpoint(out / 'latest.pt', model, tokenizer_hash, context=context,
                            optimizer=optimizer.state_dict(), report=report, order=order,
                            position=position, epoch=epoch, order_rng=rng.getstate(), best_loss=best_loss)

        with (out / 'steps.jsonl').open('a', encoding='utf-8') as log:
            while elapsed_before + time.perf_counter() - phase_start < args.seconds:
                monitor.check()
                if position >= len(order):
                    order = list(range(len(x)))
                    rng.shuffle(order)
                    position = 0
                    epoch += 1
                indices = order[position:position+4]
                elapsed = elapsed_before + time.perf_counter() - phase_start
                progress = min(1., elapsed / args.seconds)
                rate = 1e-5 + .5 * (1e-4 - 1e-5) * (1 + math.cos(math.pi * progress))
                rate *= min(1., (report['updates'] + 1) / 100)
                for group in optimizer.param_groups:
                    group['lr'] = rate
                model.train()
                monitor.phase = 'training'
                step_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss, summed, valid = training_loss(model, x[indices], y[indices], valid_only=True, trim=True)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
                monitor.check()
                optimizer.step()
                report['optimizer_seconds'] += time.perf_counter() - step_start
                position += len(indices)
                report['updates'] += 1
                report['valid_targets'] += valid
                report['epochs'] = epoch - 1 + position / len(order)
                report['phase_seconds'] = elapsed_before + time.perf_counter() - phase_start
                if report['phase_seconds'] >= next_log:
                    entry = dict(update=report['updates'], phase_seconds=report['phase_seconds'],
                                 loss=float(loss.detach()), valid_targets=valid, lr=rate, gradient_norm=float(norm))
                    log.write(json.dumps(entry) + '\n')
                    log.flush()
                    write_json(out / 'run.json', report)
                    print(json.dumps(entry), flush=True)
                    next_log = report['phase_seconds'] + 60
                if report['phase_seconds'] >= next_evaluation:
                    current_loss = evaluate('validation', f'minute{int(next_evaluation/60)}')
                    if current_loss < best_loss:
                        best_loss = current_loss
                        report['best_update'] = report['updates']
                        save_checkpoint(out / 'best.pt', model, tokenizer_hash, context=context)
                    report['phase_seconds'] = elapsed_before + time.perf_counter() - phase_start
                    recovery()
                    next_evaluation += 900
                    write_json(out / 'run.json', report)
            report['phase_seconds'] = elapsed_before + time.perf_counter() - phase_start
            phase_start = None
            recovery()
        final_loss = evaluate('validation', 'final')
        save_checkpoint(out / 'final.pt', model, tokenizer_hash, context=context)
        if final_loss < best_loss:
            best_loss = final_loss
            report['best_update'] = report['updates']
            save_checkpoint(out / 'best.pt', model, tokenizer_hash, context=context)
        best = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(best['model'], strict=True)
        del best
        probes('best')
        # Final test is opened only after validation selection is finished.
        test_path = data / 'test.jsonl'
        if digest(test_path) != manifest['files']['test.jsonl']:
            raise ValueError('Test file changed')
        test = [encode_example(tokenizer, json.loads(line), context)
                for line in test_path.read_text(encoding='utf-8').splitlines()]
        tensors['test'] = (torch.tensor([row[0] for row in test]), torch.tensor([row[1] for row in test]))
        evaluate('test', 'selected_best_test')
        restored = torch.load(out / 'best.pt', map_location='cpu', weights_only=True)
        check = EfficientDecoder(Config(**restored['config']))
        check.load_state_dict(restored['model'], strict=True)
        if parameter_hash(check) != restored['parameter_sha256']:
            raise ValueError('Checkpoint reload failed')
        report.update(status='PASS', best_validation_loss=best_loss,
                      best_checkpoint_sha256=digest(out / 'best.pt'),
                      best_parameter_sha256=restored['parameter_sha256'],
                      checkpoint_reload_verified=True,
                      finished_utc=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        if phase_start is not None:
            report['phase_seconds'] = elapsed_before + time.perf_counter() - phase_start
        report.update(status='STOPPED' if monitor.stop.is_set() else 'FAIL',
                      error=repr(error), traceback=traceback.format_exc())
        if model is not None and optimizer is not None:
            try:
                recovery()
            except Exception as save_error:
                report['recovery_error'] = repr(save_error)
    finally:
        report.update(monitor.close())
        write_json(out / 'run.json', report)
        print(json.dumps(dict(status=report['status'], phase_seconds=report['phase_seconds'],
                              updates=report['updates'], error=report.get('error'))), flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
