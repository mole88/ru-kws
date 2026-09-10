"""Validate WAVs and split separation without importing a neural network."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path

import soundfile as sf

from ru_kws.data.filtering import filter_long_commands
from ru_kws.data.manifest import read_labels, read_manifest, audio_path


def validate_dataset(root, window_seconds=3.0):
    labels = read_labels(root)
    seen = {}
    counts = {}
    for split in ("train", "val", "test"):
        rows, _ = filter_long_commands(root, read_manifest(root, split, labels), split, window_seconds)
        counts[split] = dict(Counter(row["label"] for row in rows))
        for row in rows:
            path = audio_path(root, row["path"])
            info = sf.info(path)
            if info.samplerate != 16000 or info.channels != 1 or info.frames == 0:
                raise ValueError(f"Expected nonempty mono 16 kHz WAV: {path}")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if row.get("sha256") and row["sha256"] != sha:
                raise ValueError(f"WAV checksum mismatch: {path}")
            groups = {"sha256": sha, **{k: row.get(k) for k in ("speaker_group", "parent_id", "source_group")}}
            for kind, value in groups.items():
                if value is None or value == "":
                    continue
                key = (kind, value)
                if key in seen and seen[key] != split:
                    raise ValueError(f"Split leakage: {kind}={value} in {seen[key]} and {split}")
                seen[key] = split
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--window-seconds", type=float, default=3.0)
    args = parser.parse_args()
    for split, counts in validate_dataset(args.data_root, args.window_seconds).items():
        print(split, counts)
    print("Dataset validation passed")


if __name__ == "__main__":
    main()
