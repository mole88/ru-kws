"""Copy and validate local MIT 16 kHz RIRs into a dataset (no speech changes)."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    if not (args.data_root / 'labels.json').is_file():
        raise ValueError('data-root must contain labels.json')
    files = sorted((args.source / '16khz').glob('*.wav'))
    if not files:
        raise ValueError('No WAV files in source/16khz')
    destination = args.data_root / 'augmentation' / 'mit_rirs'
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in files:
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        if sr != 16000 or audio.shape[1] != 1 or not audio.size or not np.isfinite(audio).all() or not np.any(audio):
            raise ValueError(f'Invalid RIR: {path}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        target = destination / path.name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f'Different existing file: {target}')
        if not target.exists():
            shutil.copy2(path, target)
        rows.append(dict(path=target.relative_to(args.data_root).as_posix(),
                         sample_rate=sr, frames=len(audio), sha256=digest,
                         usage='train_augmentation_only'))
    if (args.source / 'README.md').exists():
        shutil.copy2(args.source / 'README.md', destination / 'SOURCE_README.md')
    (destination / 'manifest.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    print(f'Prepared {len(rows)} RIRs: {destination}')


if __name__ == '__main__':
    main()
