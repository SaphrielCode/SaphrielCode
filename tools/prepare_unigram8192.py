"""Freeze deterministic 8192-entry byte-level Unigram on the original documents."""
import os
import collections
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
    import sentencepiece as spm
    original=ROOT/'data/TinyStoriesSubset'
    source=ROOT/'data/TinyStoriesSubset_Unigram8192'
    out=ROOT/'data/TinyStoriesBatches128_Unigram8192'
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
    construction=ROOT/'results/Unigram8192_TokenizerConstruction'
    construction.mkdir(exist_ok=False)
    specials=['<pad>','<bos>','<eos>','<unk>']
    pretok=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True)
    words=collections.Counter()
    for row in documents['train']:
        words.update(piece for piece,_ in pretok.pre_tokenize_str(row['text']))
    seeds=collections.Counter()
    for word,count in words.items():
        for at in range(len(word)):
            for end in range(at+1,min(len(word),at+16)+1):
                seeds[word[at:end]]+=count
    with (construction/'words.tsv').open('w',encoding='utf-8') as f:
        for word,count in sorted(words.items()): f.write(f'{word}\t{count}\n')
    with (construction/'seeds.tsv').open('w',encoding='utf-8') as f:
        for piece,count in sorted(seeds.items(),key=lambda x:(-x[1],x[0])):
            if count>=2: f.write(f'{piece}\t{count}\n')
    settings=dict(input=str(construction/'words.tsv'),input_format='tsv',model_prefix=str(construction/'unigram'),
                  model_type='unigram',vocab_size=8192,hard_vocab_limit=True,
                  seed_sentencepieces_file=str(construction/'seeds.tsv'),character_coverage=1.0,
                  required_chars=''.join(sorted(pre_tokenizers.ByteLevel.alphabet())),
                  normalization_rule_name='identity',add_dummy_prefix=False,remove_extra_whitespaces=False,
                  escape_whitespaces=True,split_by_unicode_script=False,split_by_number=False,
                  split_by_whitespace=False,max_sentencepiece_length=16,num_threads=1,
                  shuffle_input_sentence=False,shrinking_factor=0.75,num_sub_iterations=2,
                  unk_id=3,bos_id=1,eos_id=2,pad_id=0,unk_piece='<unk>',bos_piece='<bos>',eos_piece='<eos>',pad_piece='<pad>',minloglevel=1)
    spm.SentencePieceTrainer.train(**settings)
    sp=spm.SentencePieceProcessor(model_file=str(construction/'unigram.model'))
    pieces=[(sp.id_to_piece(i),sp.get_score(i)) for i in range(sp.vocab_size())]
    alphabet=set(pre_tokenizers.ByteLevel.alphabet())
    assert [piece for piece,_ in pieces[:4]]==specials
    assert all(set(piece)<=alphabet for piece,_ in pieces[4:])
    missing=sorted(alphabet-{piece for piece,_ in pieces})
    # Reserve universal byte coverage just as the BPE alphabet does. Drop only
    # lowest-score multi-character pieces; never replace required byte symbols.
    removable=sorted([(score,piece) for piece,score in pieces[4:] if piece not in alphabet])
    assert len(removable)>=len(missing)
    removed={piece for _,piece in removable[:len(missing)]}
    fallback_score=min(score for _,score in pieces[4:])-10.0
    pieces=[(piece,score) for piece,score in pieces if piece not in removed]+[(piece,fallback_score) for piece in missing]
    assert len(pieces)==8192 and alphabet<={piece for piece,_ in pieces}
    save(construction/'byte-alphabet-reservation.json',dict(required_bytes=256,appended_missing_bytes=missing,
         removed_lowest_score_pieces=sorted(removed),fallback_score=fallback_score,
         method='Reserve all256 byte symbols within8192 entries; retain remaining SentencePiece EM scores; final HF tokenizer is authoritative.'))
    tokenizer=Tokenizer(models.Unigram(pieces,3,False))
    tokenizer.normalizer=normalizers.NFC()
    tokenizer.pre_tokenizer=pretok
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.add_special_tokens(specials)
    save(construction/'construction.json',dict(settings=settings,sentencepiece_version=spm.__version__,
         unique_training_chunks=len(words),candidate_count=len(seeds),seed_selection='All training-only substrings up to16 mapped characters with frequency>=2;256 byte alphabet required. Actual frequencies preserved; no frequency multiplier.',
         inference='Converted trained piece scores to HuggingFace Unigram; deterministic Viterbi, no sampling. Same ByteLevel regex pretokenization and NFC as BPE.',
         trainer_sources=['https://github.com/google/sentencepiece/blob/master/doc/options.md','https://github.com/huggingface/tokenizers/blob/v0.22.1/tokenizers/src/models/unigram/trainer.rs']))
    assert tokenizer.get_vocab_size()==8192
    assert [tokenizer.token_to_id(s) for s in specials]==[0,1,2,3]
    source.mkdir(); out.mkdir()
    shutil.copyfile(construction/'unigram.model',source/'sentencepiece.model')
    shutil.copyfile(construction/'construction.json',source/'construction.json')
    shutil.copyfile(construction/'byte-alphabet-reservation.json',source/'byte-alphabet-reservation.json')
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
                assert all(4<=i<8192 for i in ids) and tokenizer.decode(ids)==text
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
                    tokenizer=dict(old['tokenizer'],vocabulary=8192,type='ByteLevel Unigram',backend='SentencePiece0.2.1 EM with explicit training-only seeds and mandatory256-byte reservation, converted to HF Unigram',min_frequency=2,shrinking_factor=0.75,max_piece_length=16,n_sub_iterations=2,segmentation='Deterministic Viterbi; no sampling or subword regularization'),script_sha256=digest(Path(__file__)),
                    versions=dict(old['versions'],tokenizers=tokenizers.__version__,sentencepiece=spm.__version__),
                    files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in source.iterdir()})
    save(source/'manifest.json',manifest)
    config=json.loads((source/'tokenizer.json').read_text(encoding='utf-8'))
    save(out/'tokenizer-lock.json',dict(file=str((source/'tokenizer.json').relative_to(ROOT)),sha256=token_hash,
                                      vocabulary_size=8192,special_ids=dict(zip(specials,range(4))),
                                      configuration={k:v for k,v in config.items() if k!='model'},
                                      model_configuration={k:v for k,v in config['model'].items() if k not in ('vocab','merges')},
                                      tokenizers_version=tokenizers.__version__,training_source='Same retained training documents only; fixed source order.'))
    batch_manifest=dict(context=128,policy='document-isolated-right-pad-v1',statistics=window_stats,
                        tokenizer_sha256=token_hash,source_manifest_sha256=digest(source/'manifest.json'),
                        source_files=manifest['files'],vocabulary_size=8192,tokenizer_algorithm='unigram',
                        source_directory=str(source.relative_to(ROOT)),
                        preparation_script_sha256=digest(Path(__file__)),
                        files={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()})
    save(out/'manifest.json',batch_manifest)
    report=dict(status='PASS',tokenizer_sha256=token_hash,vocabulary_size=8192,tokenizer_algorithm='unigram',statistics=stats,windows=window_stats,
                identical_document_splits=True,zero_exact_cross_split_overlap=True,all_stories_roundtrip=True,
                unseen_unicode_checks=True,all_window_shifts_and_coverage_checked=True,
                preparation_seconds=time.perf_counter()-start,
                source_storage_bytes=sum(p.stat().st_size for p in source.iterdir()),
                window_storage_bytes=sum(p.stat().st_size for p in out.iterdir()),
                final_test_quality_evaluated=False,sentencepiece_version=spm.__version__,construction_directory=str(construction.relative_to(ROOT)))
    save(ROOT/'results/Unigram8192_Preparation.json',report)
    (ROOT/'results/Unigram8192_Preparation.txt').write_text('Unigram8192 PREPARATION RESULTS\n\n'+json.dumps(report,indent=2)+'\nOriginal 8192-entry corpus/tokenizer/batches preserved. Same text documents and licenses. Test split preparation is an integrity operation; no test loss used.\n',encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
