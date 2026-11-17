"""Package Stage 1 weights, recovery state, frozen examples, and run evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CHUNK = 4 * 1024**2


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve()
    if not run.is_relative_to(ROOT / 'results'):
        raise ValueError('Run must be inside results')
    report = json.loads((run / 'run.json').read_text())
    if report['status'] != 'PASS':
        raise ValueError('Training has not passed')
    summary = dict(report)
    before = next(row for row in report['validation'] if row['label'] == 'before')
    summary['validation_loss_relative_reduction'] = 1 - report['best_validation_loss'] / before['nats_per_target']
    summary['publication'] = 'Selected updated inference weights, final recovery checkpoint, and frozen data/run-evidence archive; binary parts verified by SHA-256.'
    summary_path = ROOT / 'results/StageOne_Results.json'
    summary_path.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    archive = ROOT / 'results/StageOne_Artifacts.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for directory in (run, ROOT / 'data/StageOne_v1'):
            for path in sorted(directory.rglob('*')):
                if path.is_file() and path.suffix not in ('.pt', '.tmp'):
                    output.write(path, path.relative_to(ROOT))
        for name in ('results/StageOne_Protocol.json', 'results/StageOne_Results.json', 'results/StageOne_GenerationComparison.json', 'results/Chat_Verification.json'):
            path = ROOT / name
            if path.exists():
                output.write(path, name)
    manifest_path = ROOT / 'artifacts/ArtifactManifest.json'
    manifest = json.loads(manifest_path.read_text())
    specs, paths = [], []
    for source, kind, needed in ((ROOT / 'model_assets/stage-one.pt', 'checkpoint', True),
                                  (run / 'latest.pt', 'checkpoint', False),
                                  (archive, 'log', False)):
        parts = []
        with source.open('rb') as file:
            while block := file.read(CHUNK):
                git_sha = hashlib.sha1(f'blob {len(block)}\0'.encode() + block).hexdigest()
                path = ROOT / f'artifacts/{kind}-parts/{git_sha}.bin'
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.read_bytes() != block:
                    raise ValueError('Existing artifact part differs')
                path.write_bytes(block)
                parts.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=len(block),
                                  sha256=hashlib.sha256(block).hexdigest()))
                paths.append(path.relative_to(ROOT).as_posix())
        specs.append(dict(file=source.relative_to(ROOT).as_posix(), bytes=source.stat().st_size,
                          sha256=digest(source), parts=parts, needed_for_chat=needed))
    # Recovery also needs the selected best weights at the original run path.
    # Reuse the exact same artifact parts as the deployed inference checkpoint.
    best_alias = dict(specs[0], file=(run / 'best.pt').relative_to(ROOT).as_posix(),
                      needed_for_chat=False)
    specs.append(best_alias)
    replaced = {entry['file'] for entry in specs}
    manifest['files'] = [entry for entry in manifest['files'] if entry['file'] not in replaced] + specs
    manifest['stage_one'] = dict(run=run.relative_to(ROOT).as_posix(),
                                restore_data='Restore --all, then extract results/StageOne_Artifacts.zip at the project root.',
                                summary='results/StageOne_Results.json')
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (ROOT / 'results/StageOne_PublicationFiles.json').write_text(json.dumps(sorted(set(paths)), indent=2) + '\n')
    print(json.dumps(dict(parts=len(set(paths)), binary_files=len(specs),
                          binary_bytes=sum(entry['bytes'] for entry in specs))))


if __name__ == '__main__':
    main()
