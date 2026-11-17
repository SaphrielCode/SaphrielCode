"""Freeze natural next-sentence examples and authored fact-sensitive scenes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tokenizers import Tokenizer
from lm.stage_one import prompt_text, response_text


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authored(split, count):
    rng = random.Random({'train': 5101, 'validation': 5102, 'test': 5103}[split])
    names = {'train': ['Molly', 'George', 'Lily', 'Tom', 'Mia', 'Ben', 'Anna', 'Sam'],
             'validation': ['Nora', 'Owen', 'Iris', 'Theo'],
             'test': ['Ada', 'Felix', 'Ruth', 'Eli']}[split]
    objects = ['brass key', 'silver coin', 'small stone', 'red ribbon', 'blue marble',
               'gold ring', 'wooden whistle', 'paper flower']
    places = ['coat pocket', 'bag', 'drawer', 'wooden box', 'basket', 'cupboard']
    seen, rows = set(), []
    while len(rows) < count:
        a, b = rng.sample(names, 2)
        obj, other = rng.sample(objects, 2)
        place, elsewhere = rng.sample(places, 2)
        color = rng.choice(['red', 'blue', 'green', 'yellow', 'white', 'black'])
        task = len(rows) % 10
        if task == 0:
            scene = f'{a} put the {obj} in the {place}. {b} watched carefully.'
            user = f'{b} opened the {place}.'
            reply = f'{b} found the {obj} inside.'
        elif task == 1:
            scene = f'{a} put the {obj} in the {place}, then moved it to the {elsewhere}.'
            user = f'{b} looked in the {place}.'
            reply = f'The {obj} was gone because it was now in the {elsewhere}.'
        elif task == 2:
            scene = f'{a} held the {obj}. {b} held the {other}.'
            user = f'{b} gave the {other} to {a}.'
            reply = f'{a} now held both the {obj} and the {other}.'
        elif task == 3:
            scene = f'{a} was holding the {obj} when {b} came into the room.'
            user = f'{b} went to hug {a}.'
            reply = f'{a} carefully held the {obj} in one hand and hugged {b}.'
        elif task == 4:
            scene = f'The door was locked. {a} had left the {obj} in the {place}.'
            user = f'{a} went to fetch the {obj}.'
            reply = f'{a} opened the {place} and took out the {obj}.'
        elif task == 5:
            scene = f'{a} had a {color} car parked outside the shop.'
            user = f'{a} finished the lemonade and went outside.'
            reply = f'{a} climbed into the {color} car.'
        elif task == 6:
            scene = f'{a} and {b} were playing outside when rain began to fall.'
            user = f'{a} picked up a {color} umbrella.'
            reply = f'{a} opened the {color} umbrella and invited {b} to stand underneath.'
        elif task == 7:
            scene = f'{a} hid the {obj} in the {place}. The {other} was in the {elsewhere}.'
            user = f'{b} wanted to find the {other}.'
            reply = f'{b} searched the {elsewhere} and found the {other}.'
        elif task == 8:
            scene = f'{a} put a {color} hat on the table. {b} was cold.'
            user = f'{a} offered the hat to {b}.'
            reply = f'{b} put on the {color} hat and smiled.'
        else:
            scene = f'{a} had promised to bring the {obj} to {b}.'
            user = f'{a} knocked on the door with the {obj} in hand.'
            reply = f'{b} opened the door and thanked {a} for bringing the {obj}.'
        scene = f'Earlier, {a} had seen a {other} near the {elsewhere}. ' + scene
        identity = (scene, user)
        if identity in seen:
            continue
        seen.add(identity)
        rows.append(dict(scene=scene, user=user, response=reply,
                         source='authored', family=task, split=split))
    return rows


def natural(split, count):
    rows = []
    source = ROOT / 'data/TinyStoriesSubset' / f'{split}.jsonl'
    for line in source.read_text(encoding='utf-8').splitlines():
        story = json.loads(line)
        sentences = re.split(r'(?<=[.!?])\s+', story['text'].strip())
        for index in range(1, len(sentences) - 1):
            user, reply = sentences[index:index + 2]
            if not (4 <= len(user.split()) <= 28 and 4 <= len(reply.split()) <= 28):
                continue
            if any(mark in user + reply for mark in ['"', '\u201c', '\u201d']):
                continue
            if reply[-1:] not in '.!?':
                continue
            scene = ' '.join(sentences[max(0, index - 2):index])
            rows.append(dict(scene=scene, user=user, response=reply,
                             source='TinyStories', source_text_sha256=story['text_sha256'],
                             sentence_index=index, split=split))
    random.Random(5201).shuffle(rows)
    return rows[:count]


def main():
    out = ROOT / 'data/StageOne_v1'
    if out.exists():
        raise FileExistsError('StageOne_v1 is frozen and already exists')
    tokenizer_path = ROOT / 'model_assets/tokenizer.json'
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    candidates = {split: natural(split, count) + authored(split, synthetic)
                  for split, count, synthetic in [('train', 24000, 12000),
                                                   ('validation', 700, 300),
                                                   ('test', 200, 100)]}
    def length(row):
        return 2 + len(tokenizer.encode(prompt_text(row['scene'], row['user'])).ids) + len(tokenizer.encode(response_text(row['response'])).ids)
    lengths = [length(row) for row in candidates['train']]
    over128 = sum(n > 128 for n in lengths) / len(lengths)
    context = 256 if over128 > .05 else 128
    out.mkdir(parents=True)
    files, counts, dropped, seen = {}, {}, {}, set()
    # Reserve evaluation identities before selecting training examples.
    for split in ('test', 'validation', 'train'):
        rows = []
        for row in candidates[split]:
            identity = (row['scene'], row['user'])
            if identity in seen or length(row) > context:
                continue
            seen.add(identity)
            rows.append(row)
        dropped[split] = len(candidates[split]) - len(rows)
        path = out / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')
        counts[split] = dict(total=len(rows), by_source=dict(Counter(row['source'] for row in rows)))
        files[path.name] = digest(path)
    manifest = dict(format='stage-one-v1', tokenizer_sha256=digest(tokenizer_path),
                    context=context, fraction_training_candidates_over128=over128,
                    counts=counts, dropped=dropped, files=files,
                    prompt_format='Scene: {scene}\nUser: {user}\nReply:',
                    target='Leading space, one demonstrated continuation sentence, EOS2; prompt/BOS/padding ignored',
                    packing='One independent example per right-padded row; continuous positions; no cross-example attention',
                    provenance='TinyStories sentences from existing frozen document splits plus locally authored deterministic templates. Natural training prose was seen during checkpoint pretraining. Synthetic evaluation names are disjoint, templates shared.',
                    limitations='Template transfer and short TinyStories continuation do not establish broad co-writing quality. Test is reserved until final evaluation.',
                    seed=5101)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
