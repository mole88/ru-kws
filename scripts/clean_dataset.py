"""Clean a local dataset, add Vanya's held-out clips, and write a fresh version.

Inputs may be directories or ZIP files. Requires numpy, scipy and soundfile.
The input dataset is never modified. Exclusions belong here, not in evaluation.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import tempfile
import zipfile

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SPLITS = ('train', 'val', 'test')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows), encoding='utf-8')


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / str(relative).replace('\\', '/')).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f'Path outside input root: {relative}')
    return path


def unpack(source, stack):
    source = Path(source).resolve()
    if source.is_dir(): return source
    if not source.is_file(): raise FileNotFoundError(source)
    folder = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='ru-kws-clean-')))
    with zipfile.ZipFile(source) as archive:
        for member in archive.infolist():
            inside(folder, member.filename)
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f'Symlink in ZIP: {member.filename}')
        archive.extractall(folder)
    return folder


def dataset_root(parent):
    matches = [p.parent for p in parent.rglob('labels.json')
               if all((p.parent/'splits'/f'{s}.jsonl').is_file() for s in SPLITS)]
    if len(matches) != 1: raise ValueError(f'Expected one dataset root: {matches}')
    return matches[0]


def clean_audio(path, label, minimum, maximum, margin_ms):
    audio, rate = sf.read(path, dtype='float32', always_2d=True)
    if not audio.size or rate <= 0 or not np.isfinite(audio).all():
        raise ValueError('empty_or_nonfinite_audio')
    audio = audio.mean(axis=1)
    original_seconds = len(audio) / rate
    if rate != 16000:
        factor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // factor, rate // factor).astype('float32')
    start, end = 0, len(audio)
    if label != 'background':
        frame, hop = 320, 160
        padded = np.pad(audio, (0, max(0, frame-len(audio))))
        frames = np.lib.stride_tricks.sliding_window_view(padded, frame)[::hop]
        rms = np.sqrt(np.mean(frames.astype('float64')**2, axis=1))
        active = np.flatnonzero(rms > max(float(rms.max()) * 10**(-35/20), 10**(-55/20)))
        if not len(active): raise ValueError('no_speech_activity')
        margin = round(margin_ms * 16)
        start = max(0, int(active[0])*hop-margin)
        end = min(len(audio), int(active[-1])*hop+frame+margin)
        seconds = (end-start)/16000
        if not minimum <= seconds <= maximum:
            raise ValueError(f'speech_duration_outside_limits:{seconds:.5f}')
    peak = float(np.max(np.abs(audio[start:end])))
    gain = min(1., .98/max(peak, 1e-12))
    return audio[start:end]*gain, dict(original_duration_s=original_seconds,
        trimmed_left_s=start/16000, trimmed_right_s=(len(audio)-end)/16000,
        sample_rate=16000, duration_s=(end-start)/16000, peak_limiter_gain=gain)


def clean_dataset(dataset, vanya_recordings, output, minimum=.35, maximum=2.6, margin_ms=150, archive=False):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError(f'Choose a new output directory: {output}')
    if archive and Path(str(output)+'.zip').exists():
        raise FileExistsError(f'Archive already exists: {output}.zip')
    if not 0 < minimum <= maximum <= 3 or margin_ms < 0:
        raise ValueError('Require 0 < min <= max <= 3 seconds and a nonnegative margin')
    # Check before creating anything: an output nested in an input would be copied recursively.
    for source in [dataset, vanya_recordings]:
        source = Path(source).resolve()
        if source.is_dir() and output.is_relative_to(source):
            raise ValueError('Output must be outside both input directories')
    with ExitStack() as stack:
        source = dataset_root(unpack(dataset, stack))
        vanya = unpack(vanya_recordings, stack)
        labels = json.loads((source/'labels.json').read_text(encoding='utf-8-sig'))
        if not labels or sorted(labels.values()) != list(range(len(labels))):
            raise ValueError('Invalid labels.json')
        output.mkdir(parents=True)
        write_json(output/'cleaning_report.json', dict(complete=False))
        accepted, excluded, vanya_test = [], [], []
        seen_ids, seen_hashes, groups = {}, {}, {}

        def accept(row, source_root, kind):
            if row['label'] not in labels or row['split'] not in SPLITS:
                raise ValueError(f'Invalid label/split: {row}')
            if (row.get('evaluation_group') == 'vanya' or str(row.get('speaker', '')).lower() == 'vanya') and row['split'] != 'test':
                raise RuntimeError('Vanya must be held out of train and val')
            path = inside(source_root, row['path'])
            ident = str(row.get('id') or hashlib.sha256(str(path).encode()).hexdigest()[:24])
            if not re.fullmatch(r'[A-Za-z0-9_-]+', ident):
                raise ValueError(f'Unsafe record id: {ident}')
            try:
                if not path.is_file(): raise FileNotFoundError(f'missing_audio:{row["path"]}')
                if row.get('sha256') and file_hash(path) != row['sha256']:
                    raise ValueError('source_sha256_mismatch')
                audio, meta = clean_audio(path, row['label'], minimum, maximum, margin_ms)
                partition = 'vanya_test' if kind == 'vanya' else row['split']
                if ident in seen_ids:
                    if seen_ids[ident] != partition:
                        raise RuntimeError(f'Record ID overlaps splits: {ident}')
                    raise ValueError('duplicate_id')
                # Hash decoded PCM, so a changed WAV header cannot hide leakage.
                pcm = np.rint(np.clip(audio, -1, 1) * 32767).astype('<i2').tobytes()
                content_hash = hashlib.sha256(pcm).hexdigest()
                if content_hash in seen_hashes:
                    previous = seen_hashes[content_hash]
                    if previous != partition:
                        raise RuntimeError(f'Audio overlaps {previous} and {partition}: {row["path"]}')
                    raise ValueError('duplicate_audio_in_split')
                for key in ['speaker_group', 'parent_id', 'source_group']:
                    group = row.get(key)
                    if group and (key, group) in groups and groups[key, group] != partition:
                        raise RuntimeError(f'{key} leaks across splits: {group}')
                dest = output/'audio'/kind/f'{ident}.wav'
                dest.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dest, audio, 16000, subtype='PCM_16')
                updated = {**row, **meta, 'id': ident, 'path': dest.relative_to(output).as_posix(),
                    'sha256': file_hash(dest), 'source_path': row['path'],
                    'qc': 'local_decode_checksum_trim_duration'}
                (vanya_test if kind == 'vanya' else accepted).append(updated)
                seen_ids[ident] = partition; seen_hashes[content_hash] = partition
                for key in ['speaker_group', 'parent_id', 'source_group']:
                    if row.get(key): groups[key, row[key]] = partition
                return updated
            except (OSError, sf.LibsndfileError, ValueError) as exc:
                excluded.append(dict(kind=kind, id=ident, path=row['path'], label=row['label'],
                                     split=row['split'], reason=str(exc)))
                return None

        for split in SPLITS:
            rows = read_jsonl(source/'splits'/f'{split}.jsonl')
            for index, row in enumerate(rows):
                if row.get('split', split) != split: raise ValueError(f'Wrong split: {row}')
                record = dict(row, split=split)
                is_vanya = row.get('evaluation_group') == 'vanya' or str(row.get('speaker', '')).lower() == 'vanya'
                if is_vanya:
                    record.update(evaluation_group='vanya', engine='real', sample_kind='human_recording',
                                  speaker_group='human:vanya')
                accept(record, source, 'vanya' if is_vanya else 'dataset')
                if (index+1)%1000 == 0:
                    print(f'{split}: checked {index+1}/{len(rows)}, excluded total={len(excluded)}', flush=True)
        manifests = sorted(vanya.rglob('manifest.jsonl'))
        if not manifests: raise FileNotFoundError('Vanya archive contains no manifest.jsonl')
        for manifest in manifests:
            for index, row in enumerate(read_jsonl(manifest)):
                if 'wav' not in row: continue
                if row.get('accepted') is not True or row.get('error'):
                    excluded.append(dict(kind='vanya', path=row.get('wav'), reason='recorder_not_accepted'))
                    continue
                relative = inside(vanya, (manifest.parent/row['wav']).relative_to(vanya)).relative_to(vanya).as_posix()
                ident = 'vanya_'+hashlib.sha256(relative.encode()).hexdigest()[:24]
                record = dict(id=ident, path=relative, label=row['label'], text=row.get('text'),
                    split='test', engine='real', sample_kind='human_recording',
                    evaluation_group='vanya', speaker_group='human:vanya', parent_id=ident,
                    speaker=row.get('speaker', 'vanya'), condition=row.get('condition'))
                accept(record, vanya, 'vanya')
        write_jsonl(output/'excluded.jsonl', excluded)
        for split in SPLITS:
            subset = [r for r in accepted if r['split']==split]
            if not subset: raise ValueError(f'No usable records in {split}; inspect excluded.jsonl')
            write_jsonl(output/'splits'/f'{split}.jsonl', subset)
        if not vanya_test: raise ValueError('No usable Vanya test clips; inspect excluded.jsonl')
        # Training augmentation assets and their relative paths are preserved.
        for name in ['sources', 'augmentation']:
            if (source/name).exists(): shutil.copytree(source/name, output/name)
        for filename in ['speaker_groups.json', 'background_groups.json', 'negative_texts.json']:
            if (source/'splits'/filename).is_file(): shutil.copy2(source/'splits'/filename, output/'splits'/filename)
        write_json(output/'labels.json', labels)
        write_jsonl(output/'manifest.jsonl', accepted)
        write_jsonl(output/'evaluation'/'vanya_test.jsonl', vanya_test)
        write_jsonl(output/'evaluation'/'synthetic_test.jsonl', [r for r in accepted
            if r['split']=='test' and r.get('sample_kind')!='human_recording' and r.get('engine')!='real'])
        report = dict(complete=True, source=str(Path(dataset).resolve()), vanya_source=str(Path(vanya_recordings).resolve()),
            accepted=len(accepted)+len(vanya_test), dataset_accepted=len(accepted),
            excluded=len(excluded), vanya_test=len(vanya_test), vanya_separate_test=True,
            counts={s:dict(Counter(r['label'] for r in accepted if r['split']==s)) for s in SPLITS},
            exclusions=dict(Counter(r['reason'].split(':')[0] for r in excluded)),
            policy=dict(min_seconds=minimum, max_seconds=maximum, margin_ms=margin_ms, sample_rate=16000))
        write_json(output/'cleaning_report.json', report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if archive:
            archive_path = shutil.make_archive(str(output), 'zip', root_dir=output.parent, base_dir=output.name)
            print('Archive:', archive_path)
        return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, type=Path)
    p.add_argument('--vanya-recordings', required=True, type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--min-seconds', type=float, default=.35)
    p.add_argument('--max-seconds', type=float, default=2.6)
    p.add_argument('--margin-ms', type=float, default=150)
    p.add_argument('--zip', action='store_true')
    a = p.parse_args(argv)
    return clean_dataset(a.dataset, a.vanya_recordings, a.output_dir,
                         a.min_seconds, a.max_seconds, a.margin_ms, a.zip)


if __name__ == '__main__': main()
