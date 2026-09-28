import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest
import soundfile as sf

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/clean_dataset.py'
spec = importlib.util.spec_from_file_location('clean_dataset', SCRIPT)
cleaner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleaner)


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')


def fixture(root):
    source = root/'source'; source.mkdir()
    (source/'labels.json').write_text(json.dumps({'next':0,'unknown':1,'background':2}))
    for split_index, split in enumerate(cleaner.SPLITS):
        rows = []
        for i,label in enumerate(['next','unknown','background']):
            name = f'{split}_{label}.wav'
            audio = np.random.default_rng(split_index*3+i).uniform(-.1,.1,16000).astype('float32')
            sf.write(source/name, audio, 16000, subtype='PCM_16')
            rows.append(dict(id=f'{split}_{label}', path=name, label=label, split=split,
                             speaker_group=f'voice:{split}', sha256=cleaner.file_hash(source/name)))
        jsonl(source/'splits'/f'{split}.jsonl', rows)
    vanya = root/'vanya'; vanya.mkdir()
    sf.write(vanya/'vanya.wav', np.random.default_rng(100).uniform(-.1,.1,(44100,2)),44100)
    jsonl(vanya/'manifest.jsonl', [dict(wav='vanya.wav',label='next',accepted=True,speaker='vanya'),
                                  dict(wav='unused.wav',label='next',accepted=False)])
    return source,vanya


def test_clean_dataset_removes_bad_audio_keeps_vanya_separate_and_keeps_assets(tmp_path):
    source,vanya = fixture(tmp_path)
    before = cleaner.file_hash(vanya/'vanya.wav')
    rows = cleaner.read_jsonl(source/'splits/train.jsonl')
    rows.extend([dict(id='missing',path='missing.wav',label='next'),
                 dict(id='broken',path='broken.wav',label='next'),
                 dict(id='mismatch',path='train_next.wav',label='next',sha256='wrong'),
                 dict(id='long',path='long.wav',label='next')])
    (source/'broken.wav').write_bytes(b'invalid wav')
    sf.write(source/'long.wav',np.ones(48001)*.1,16000)
    jsonl(source/'splits/train.jsonl',rows)
    (source/'augmentation/mit_rirs').mkdir(parents=True)
    (source/'augmentation/mit_rirs/LICENSE').write_text('license')
    sf.write(source/'augmentation/mit_rirs/rir.wav',np.array([1.,.1,.01]),16000)
    jsonl(source/'augmentation/mit_rirs/manifest.jsonl',[dict(path='augmentation/mit_rirs/rir.wav')])
    jsonl(source/'sources/background_manifest.jsonl',[])
    out = tmp_path/'clean'
    report = cleaner.main(['--dataset',str(source),'--vanya-recordings',str(vanya),'--output-dir',str(out),'--zip'])
    assert report['complete'] and report['vanya_test']==1 and report['accepted']==10
    assert report['excluded']==5
    assert cleaner.file_hash(vanya/'vanya.wav')==before
    assert report['vanya_separate_test'] and report['dataset_accepted']==9
    for split in ['train','val','test']:
        assert not any(r.get('evaluation_group')=='vanya' for r in cleaner.read_jsonl(out/'splits'/f'{split}.jsonl'))
    test = cleaner.read_jsonl(out/'splits/test.jsonl')
    assert len(test)==3
    view = cleaner.read_jsonl(out/'evaluation/vanya_test.jsonl')
    assert view[0] not in test and view[0]['sample_rate']==16000 and view[0]['split']=='test'
    assert view[0] not in cleaner.read_jsonl(out/'manifest.jsonl')
    assert len(cleaner.read_jsonl(out/'evaluation/synthetic_test.jsonl'))==3
    assert (out/'augmentation/mit_rirs/rir.wav').is_file()
    assert (out.with_suffix('.zip')).is_file()
    from ru_kws.data.validate import validate_dataset
    assert validate_dataset(out)['test']['next']==1
    with pytest.raises(FileExistsError): cleaner.clean_dataset(source,vanya,out)


def test_zip_inputs_and_path_escape(tmp_path):
    source,vanya = fixture(tmp_path)
    archives = []
    for folder in [source,vanya]:
        archive = tmp_path/f'{folder.name}.zip'
        with zipfile.ZipFile(archive,'w') as z:
            for file in folder.rglob('*'):
                if file.is_file():z.write(file, folder.name+'/'+file.relative_to(folder).as_posix())
        archives.append(archive)
    report=cleaner.clean_dataset(*archives,tmp_path/'out')
    assert report['vanya_test']==1
    bad=tmp_path/'bad.zip'
    with zipfile.ZipFile(bad,'w') as z:z.writestr('../outside.wav',b'bad')
    with pytest.raises(ValueError,match='outside'): cleaner.clean_dataset(bad,vanya,tmp_path/'escaped')


def test_cross_split_audio_leakage_is_rejected(tmp_path):
    source,vanya=fixture(tmp_path)
    rows=cleaner.read_jsonl(source/'splits/test.jsonl')
    rows[0].update(path='train_next.wav',sha256=cleaner.file_hash(source/'train_next.wav'),speaker_group='different')
    jsonl(source/'splits/test.jsonl',rows)
    with pytest.raises(RuntimeError,match='overlaps'): cleaner.clean_dataset(source,vanya,tmp_path/'out')
    assert not json.loads((tmp_path/'out/cleaning_report.json').read_text())['complete']


def test_cleaner_never_overwrites_existing_archive_or_trains_on_vanya(tmp_path):
    source,vanya=fixture(tmp_path)
    archive=tmp_path/'out.zip'
    archive.write_bytes(b'original archive')
    with pytest.raises(FileExistsError,match='Archive'):
        cleaner.clean_dataset(source,vanya,tmp_path/'out',archive=True)
    assert archive.read_bytes()==b'original archive'
    assert not (tmp_path/'out').exists()
    rows=cleaner.read_jsonl(source/'splits/train.jsonl')
    rows[0]['speaker']='vanya'
    jsonl(source/'splits/train.jsonl',rows)
    with pytest.raises(RuntimeError,match='held out'):
        cleaner.clean_dataset(source,vanya,tmp_path/'out')


def test_vanya_already_in_source_test_is_moved_to_separate_view(tmp_path):
    source,vanya=fixture(tmp_path)
    rows=cleaner.read_jsonl(source/'splits/test.jsonl')
    rows[0]['speaker']='vanya'
    jsonl(source/'splits/test.jsonl',rows)
    out=tmp_path/'out'
    report=cleaner.clean_dataset(source,vanya,out)
    assert report['vanya_test']==2
    assert len(cleaner.read_jsonl(out/'splits/test.jsonl'))==2
    assert len(cleaner.read_jsonl(out/'evaluation/synthetic_test.jsonl'))==2
    assert len(cleaner.read_jsonl(out/'evaluation/vanya_test.jsonl'))==2


def test_vanya_audio_overlapping_dataset_test_is_rejected(tmp_path):
    source,vanya=fixture(tmp_path)
    # These are distinct tests; an identical clip cannot appear in both.
    import shutil
    shutil.copyfile(source/'test_next.wav',vanya/'vanya.wav')
    with pytest.raises(RuntimeError,match='overlaps test and vanya_test'):
        cleaner.clean_dataset(source,vanya,tmp_path/'out')
