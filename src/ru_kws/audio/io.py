from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def load_audio(path, sample_rate=16000):
    """Read mono float32 without changing the recording's gain."""
    samples, rate = sf.read(Path(path), dtype="float32", always_2d=True)
    if rate != sample_rate or samples.shape[1] != 1:
        raise ValueError(f"Expected mono {sample_rate} Hz audio: {path}")
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"Empty or nonfinite audio: {path}")
    return torch.from_numpy(samples[:, 0].copy())
