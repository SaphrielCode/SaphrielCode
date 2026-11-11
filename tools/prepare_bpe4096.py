"""Freeze 4096-entry BPE on exactly the original TinyStories documents."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['RAYON_NUM_THREADS']='1'
from array import array
import hashlib
import json
from pathlib import Path
import random
import shutil
import struct
import sys
import time
import unicodedata

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_batches import window_story,u16_bytes


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def main():
    import tokenizers
    from tokenizers import Tokenizer,models,normalizers,pre_tokenizers,decoders,trainers
    original=ROOT/'data/TinyStoriesSubset'
    source=ROOT/'data/TinyStoriesSubset_BPE4096'
    out=ROOT/'data/TinyStoriesBatches128_BPE4096'
    if source.exists() or out.exists():
        raise RuntimeError('Frozen output exists; refusing overwrite')
    start=time.perf_counter()
    old=json.loads((original/'manifest.json').read_text(encoding='utf-8'))
    for name,item in old['files'].items():
        assert digest(original/name)==item['sha256']
    documents={s:[json.loads(line) for line in (original/f'{s}.jsonl').read_text(encoding='utf-8').splitlines()] for s in ('train','validation','test')}
    assert [len(documents[s]) for s in ('train','validation','test')]==[19966,1000,1000]
    sets=[{r['text_sha256'] for r in documents[s]} for s in ('train','validation','test')]
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i))
    tokenizer=Tokenizer(models.BPE(unk_token='<unk>'))
    tokenizer.normalizer=normalizers.NFC()
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True)
    tokenizer.decoder=decoders.ByteLevel()
    specials=['<pad>','<bos>','<eos>','<unk>']
    trainer=trainers.BpeTrainer(vocab_size=4096,min_frequency=2,show_progress=False,
                               special_tokens=specials,initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tokenizer.train_from_iterator((r['text'] for r in documents['train']),trainer=trainer,length=19966)
    assert tokenizer.get_vocab_size()==4096
    assert [tokenizer.token_to_id(s) for s in specials]==[0,1,2,3]
    source.mkdir(); out.mkdir()
    tokenizer.save(str(source/'tokenizer.json'))
    token_hash=digest(source/'tokenizer.json')
    stats={}; window_stats={}
    for split,rows in documents.items():
        shutil.copyfile(original/f'{split}.jsonl',source/f'{split}.jsonl')
        assert digest(source/f'{split}.jsonl')==digest(original/f'{split}.jsonl')
        count=windows=padding=0
        offsets=[0]
        with (source/f'{split}.u16').open('wb') as ids_file, (out/f'{split}.x.u16').open('wb') as xs, (out/f'{split}.y.u16').open('wb') as ys, (out/f'{split}.rows.u32').open('wb') as metadata:
            for doc,row in enumerate(rows):
                text=row['text']
                assert hashlib.sha256(text.encode('utf-8')).hexdigest()==row['text_sha256']
                ids=tokenizer.encode(text,add_special_tokens=False).ids
                assert all(4<=i<4096 for i in ids) and tokenizer.decode(ids)==text
                ids.append(2)
                ids_file.write(u16_bytes(ids)); count+=len(ids); offsets.append(count)
                for x,y,at,valid in window_story(ids):
                    # Independent reconstruction catches shifting/coverage errors.
                    assert y[:valid]==ids[at:at+valid] and all(v==0 for v in y[valid:])
                    assert x[0]==(1 if at==0 else ids[at-1]) and x[1:valid]==y[:valid-1]
                    xs.write(u16_bytes(x)); ys.write(u16_bytes(y)); metadata.write(struct.pack('<III',doc,at,valid))
                    windows+=1; padding+=128-valid
        (source/f'{split}.idx').write_bytes(b''.join(struct.pack('<Q',i) for i in offsets))
        assert windows*128-padding==count
        stats[split]=dict(stories=len(rows),body_tokens=count-len(rows),tokens_including_eos=count)
        if split=='train':
            order=list(range(windows)); random.Random(43).shuffle(order)
            (out/'train.order.u32').write_bytes(b''.join(struct.pack('<I',i) for i in order))
        names=[f'{split}.x.u16',f'{split}.y.u16',f'{split}.rows.u32']+(['train.order.u32'] if split=='train' else [])
        window_stats[split]=dict(documents=len(rows),saved_source_tokens_including_eos=count,windows=windows,
                                 valid_target_tokens=count,eos_targets=len(rows),padding_targets=padding,
                                 window_file_bytes=sum((out/n).stat().st_size for n in names))
    for text in ['Unseen: \u4e2d\u6587, caf\u00e9, \U0001f408\n\t','e\u0301',''.join(chr(i) for i in range(128))]:
        ids=tokenizer.encode(text,add_special_tokens=False).ids
        assert 3 not in ids and tokenizer.decode(ids)==unicodedata.normalize('NFC',text)
    manifest=dict(old)
    manifest.update(statistics=stats,original_subset_manifest_sha256=digest(original/'manifest.json'),
                    tokenizer=dict(old['tokenizer'],vocabulary=4096),script_sha256=digest(Path(__file__)),
                    versions=dict(old['versions'],tokenizers=tokenizers.__version__),
                    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in source.iterdir()})
    save(source/'manifest.json',manifest)
    config=json.loads((source/'tokenizer.json').read_text(encoding='utf-8'))
    save(out/'tokenizer-lock.json',dict(file=str((source/'tokenizer.json').relative_to(ROOT)),sha256=token_hash,
                                      vocabulary_size=4096,special_ids=dict(zip(specials,range(4))),
                                      configuration={k:v for k,v in config.items() if k!='model'},
                                      model_configuration={k:v for k,v in config['model'].items() if k not in ('vocab','merges')},
                                      tokenizers_version=tokenizers.__version__,training_source='Same retained training documents only; fixed source order.'))
    batch_manifest=dict(context=128,policy='document-isolated-right-pad-v1',statistics=window_stats,
                        tokenizer_sha256=token_hash,source_manifest_sha256=digest(source/'manifest.json'),
                        source_files=manifest['files'],vocabulary_size=4096,
                        source_directory=str(source.relative_to(ROOT)),
                        preparation_script_sha256=digest(Path(__file__)),
                        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()})
    save(out/'manifest.json',batch_manifest)
    report=dict(status='PASS',tokenizer_sha256=token_hash,vocabulary_size=4096,statistics=stats,windows=window_stats,
                identical_document_splits=True,zero_exact_cross_split_overlap=True,all_stories_roundtrip=True,
                unseen_unicode_checks=True,all_window_shifts_and_coverage_checked=True,
                preparation_seconds=time.perf_counter()-start,
                source_storage_bytes=sum(p.stat().st_size for p in source.iterdir()),
                window_storage_bytes=sum(p.stat().st_size for p in out.iterdir()),
                final_test_quality_evaluated=False)
    save(ROOT/'results/BPE4096_Preparation.json',report)
    (ROOT/'results/BPE4096_Preparation.txt').write_text('BPE4096 PREPARATION RESULTS\n\n'+json.dumps(report,indent=2)+'\nOriginal 8192-entry corpus/tokenizer/batches preserved. Same text documents and licenses. Test split preparation is an integrity operation; no test loss used.\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
