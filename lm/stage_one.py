"""Shared Stage 1 formatting, response supervision, and bounded greedy decoding."""
import unicodedata

import torch

PROMPT_FORMAT = "Scene: {scene}\nUser: {user}\nReply:"
BOS, EOS, PAD = 1, 2, 0
REPLY_LIMIT = 48


def prompt_text(scene, user):
    return PROMPT_FORMAT.format(scene=scene.strip(), user=user.strip())


def response_text(text):
    return " " + unicodedata.normalize("NFC", text).strip()


def encode_example(tokenizer, example, context):
    prompt = tokenizer.encode(prompt_text(example['scene'], example['user'])).ids
    reply = tokenizer.encode(response_text(example['response'])).ids
    sequence = [BOS] + prompt + reply + [EOS]
    if len(sequence) > context:
        raise ValueError('Complete example exceeds context budget')
    inputs, targets = sequence[:-1], sequence[1:]
    targets[:len(prompt)] = [-100] * len(prompt)
    padding = context - len(inputs)
    return inputs + [PAD] * padding, targets + [-100] * padding


@torch.inference_mode()
def generate(model, tokenizer, scene, user, context=128, limit=REPLY_LIMIT):
    ids = [BOS] + tokenizer.encode(prompt_text(scene, user)).ids
    if len(ids) + limit > context:
        raise ValueError('Shorten the scene or user turn to leave room for the reply')
    generated = []
    for _ in range(limit):
        scores = model(torch.tensor([ids], dtype=torch.long), last_only=True)[0]
        token = int(scores.argmax())
        generated.append(token)
        ids.append(token)
        if token == EOS:
            break
    return dict(text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
                ids=generated, eos=bool(generated and generated[-1] == EOS))
