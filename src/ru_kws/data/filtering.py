"""Apply the same duration policy to validation, training and evaluation."""
from collections import Counter
import soundfile as sf
from ru_kws.data.manifest import audio_path


def filter_long_commands(root, records, split, window_seconds, sample_rate=16000):
    kept, dropped = [], []
    for row in records:
        info = sf.info(audio_path(root, row["path"]))
        if info.samplerate != sample_rate or info.channels != 1 or info.frames == 0:
            raise ValueError(f"Expected nonempty mono {sample_rate} Hz WAV: {row['path']}")
        if row["label"] not in {"unknown", "background"} and info.frames > round(window_seconds * sample_rate):
            dropped.append({"path": row["path"], "label": row["label"],
                            "duration_s": info.frames / sample_rate})
        else:
            kept.append(row)
    print(f"{split}: kept {len(kept)}, dropped {len(dropped)} commands longer than {window_seconds}s; "
          f"by label: {dict(Counter(row['label'] for row in dropped))}", flush=True)
    if not kept:
        raise ValueError(f"No records remain in {split} after duration filtering")
    return kept, dropped
