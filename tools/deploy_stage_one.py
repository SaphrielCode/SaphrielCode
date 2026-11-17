"""Validate a completed Stage 1 run and select its weights for local chat."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tokenizers import Tokenizer
from lm.efficiency import EfficientDecoder
from lm.experimental_model import Config, parameter_hash
from lm.stage_one import generate


def digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    directory = args.run.resolve()
    if not directory.is_relative_to(ROOT / 'results'):
        raise ValueError('Run must be inside results')
    report = json.loads((directory / 'run.json').read_text())
    if report['status'] != 'PASS' or report['phase_seconds'] < 14400:
        raise ValueError('Four-hour adaptation has not completed successfully')
    if not report['checkpoint_reload_verified'] or report['best_update'] == 0:
        raise ValueError('Deployment requires verified updated weights')
    source = directory / 'best.pt'
    if digest(source) != report['best_checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    saved = torch.load(source, weights_only=True, map_location='cpu')
    model = EfficientDecoder(Config(**saved['config']))
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    if parameter_hash(model) != report['best_parameter_sha256']:
        raise ValueError('Selected parameter hash mismatch')
    tokenizer = Tokenizer.from_file(str(ROOT / 'model_assets/tokenizer.json'))
    if digest(ROOT / 'model_assets/tokenizer.json') != saved['tokenizer_sha256']:
        raise ValueError('Tokenizer mismatch')
    scene = "Molly put the brass key in George's coat pocket."
    user = 'George reached into his pocket.'
    result = generate(model, tokenizer, scene, user, context=saved['context'])
    profile = dict(id='stage-one', name='5.1M - Stage 1 story continuation',
                   detail='Four-hour continuation adaptation. Write the next sentence using the scene.',
                   checkpoint='model_assets/stage-one.pt', file_hash=digest(source),
                   parameter_hash=report['best_parameter_sha256'], parameters=report['parameters'],
                   style='stage-one', context=saved['context'], reply_limit=48,
                   training_directory=directory.relative_to(ROOT).as_posix(),
                   regression=dict(prompt=f'Scene: {scene}\nUser: {user}', mode='chat',
                                   temperature=0, max_tokens=48,
                                   expected_text=tokenizer.decode(result['ids'], skip_special_tokens=True),
                                   expected_ids=result['ids']))
    target = ROOT / profile['checkpoint']
    if target.exists() and digest(target) != profile['file_hash']:
        raise FileExistsError('Existing Stage 1 deployment differs')
    shutil.copy2(source, target)
    manifest_path = ROOT / 'model_assets/deployment.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['profiles'] = [profile] + [entry for entry in manifest['profiles'] if entry['id'] != 'stage-one']
    manifest['default'] = 'stage-one'
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temporary.replace(manifest_path)
    print(json.dumps(dict(status='PASS', profile=profile, regression_eos=result['eos'])))


if __name__ == '__main__':
    main()
