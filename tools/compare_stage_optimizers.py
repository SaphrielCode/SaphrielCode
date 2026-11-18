"""Split a fixed training-time budget between AdamW and CPU FP32 Muon+AdamW."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total-seconds', type=float, default=900)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'model_assets/stage-one.pt')
    args = parser.parse_args()
    if args.total_seconds <= 0:
        parser.error('Training budget must be positive')
    out = ROOT / 'results' / ('MuonAdamW_Compare_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir()
    pointer = ROOT / 'results/StageOne_Current.json'
    previous_pointer = pointer.read_bytes() if pointer.exists() else None
    result = dict(status='RUNNING', directory=out.relative_to(ROOT).as_posix(),
                  total_requested_training_seconds=args.total_seconds,
                  seconds_per_optimizer=args.total_seconds / 2,
                  comparison='Equal elapsed training-phase budgets; same initial weights, data, order seed, batch, context, and base learning-rate schedule. Preparation and evaluation are outside the budget.',
                  final_test_evaluated=False, runs={})
    summary = ROOT / 'results/MuonAdamW_Comparison.json'
    summary.write_text(json.dumps(result, indent=2) + '\n')
    try:
        for kind in ('adamw', 'muon-adamw'):
            command = [sys.executable, str(ROOT / 'tools/train_stage_one.py'),
                       '--seconds', str(args.total_seconds / 2), '--reserve-gib', '.5',
                       '--checkpoint', str(args.checkpoint.resolve()), '--optimizer', kind,
                       '--validation-only', '--out', str(out / kind)]
            code = subprocess.call(command, cwd=ROOT)
            report = json.loads((out / kind / 'run.json').read_text())
            before = next(row for row in report['validation'] if row['label'] == 'before')
            final = next((row for row in report['validation'] if row['label'] == 'final'), None)
            result['runs'][kind] = dict(status=report['status'], updates=report['updates'],
                                        valid_targets=report['valid_targets'],
                                        phase_seconds=report['phase_seconds'],
                                        valid_targets_per_phase_second=report['valid_targets'] / max(report['phase_seconds'], 1e-9),
                                        before_loss=before['nats_per_target'],
                                        final_loss=final['nats_per_target'] if final else None,
                                        best_loss=report.get('best_validation_loss'),
                                        best_update=report.get('best_update'),
                                        source_checkpoint_sha256=report['source_checkpoint_sha256'],
                                        dataset_manifest_sha256=report['dataset_manifest_sha256'],
                                        final_test_evaluated=report.get('final_test_evaluated', False),
                                        report=(out / kind / 'run.json').relative_to(ROOT).as_posix())
            summary.write_text(json.dumps(result, indent=2) + '\n')
            if code != 0 or report['status'] != 'PASS':
                raise RuntimeError(f'{kind} comparison stopped: {report.get("error")}')
        baseline, hybrid = result['runs']['adamw'], result['runs']['muon-adamw']
        for key in ('source_checkpoint_sha256', 'dataset_manifest_sha256', 'before_loss'):
            if baseline[key] != hybrid[key]:
                raise ValueError(f'Unmatched comparison starting condition: {key}')
        result.update(status='PASS',
                      loss_difference_muon_minus_adamw=hybrid['final_loss'] - baseline['final_loss'],
                      hybrid_to_adamw_target_throughput=hybrid['valid_targets_per_phase_second'] / baseline['valid_targets_per_phase_second'],
                      limitations='One sequential run per optimizer, fixed shared starting rates, short adaptation horizon. Shared-template validation and next-token loss do not establish factual story continuity. No sustained quality conclusion from this comparison alone.')
    except Exception as error:
        result.update(status='FAIL', error=repr(error))
    finally:
        if previous_pointer is not None:
            pointer.write_bytes(previous_pointer)
        summary.write_text(json.dumps(result, indent=2) + '\n')
        (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
