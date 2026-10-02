"""Training-only RIR convolution. Uses torch RNG (including worker seeding).

Reference: https://docs.pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html
RIRs are convolution kernels, not additive background recordings.
"""
import json
import math
from collections import OrderedDict
from pathlib import Path

import torch

from ru_kws.audio.io import load_audio


class RandomGain:
    """Change whole-mixture level, retaining SNR and avoiding hard clipping."""
    def __init__(self, probability=0.8, gain_db=(-6.0, 6.0)):
        if not 0 <= probability <= 1 or len(gain_db) != 2 or not all(math.isfinite(v) for v in gain_db) or gain_db[0] > gain_db[1]:
            raise ValueError('Invalid gain probability/range')
        self.probability, self.gain_db = probability, gain_db

    def __call__(self, waveform, label):
        if torch.rand(()).item() >= self.probability:
            return waveform
        db = self.gain_db[0] + torch.rand(()).item() * (self.gain_db[1] - self.gain_db[0])
        gain = 10 ** (db / 20)
        peak = waveform.abs().max().item()
        if peak > 0:
            gain = min(gain, 1.0 / peak)
        return waveform * gain


class RandomSpeed:
    """Bandlimited speed perturbation (also changes pitch), in 1% steps."""
    def __init__(self, num_samples, command_ids, probability=0.5, rate_range=(0.9, 1.1)):
        if not 0 <= probability <= 1 or len(rate_range) != 2 or not all(math.isfinite(v) and v > 0 for v in rate_range):
            raise ValueError('Invalid speed probability/range')
        self.low = math.ceil(rate_range[0] * 100 - 1e-9)
        self.high = math.floor(rate_range[1] * 100 + 1e-9)
        if self.low > self.high:
            raise ValueError('Speed range must contain a 1% step')
        self.num_samples, self.command_ids = num_samples, set(command_ids)
        self.probability = probability

    def __call__(self, waveform, label):
        if torch.rand(()).item() >= self.probability:
            return waveform
        low = self.low
        if label in self.command_ids:
            if waveform.numel() > self.num_samples:
                raise ValueError('Command exceeds window before speed augmentation')
            low = max(low, math.ceil(waveform.numel() * 100 / self.num_samples))
        if low > self.high:
            return waveform
        rate = int(torch.randint(low, self.high + 1, ()))
        if rate == 100:
            return waveform
        from torchaudio.functional import resample
        return resample(waveform, orig_freq=rate, new_freq=100)


def build_speed_augmentation(split, command_ids, config):
    options = config.get('augmentation', {}).get('speed', {})
    if split != 'train' or not options.get('enabled', False):
        return None
    return RandomSpeed(round(config['audio']['sample_rate'] * config['audio']['window_seconds']),
                       command_ids, probability=options.get('probability', 0.5),
                       rate_range=options.get('rate_range', [0.9, 1.1]))


class RandomReverb:
    def __init__(self, root, manifest, sample_rate=16000, probability=0.5,
                 max_rir_seconds=1.0, excluded_ids=()):
        if not 0 <= probability <= 1 or max_rir_seconds <= 0:
            raise ValueError('Invalid reverb probability or maximum duration')
        self.probability = probability
        self.excluded_ids = set(excluded_ids)
        root = Path(root)
        manifest = root / manifest
        rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'Empty RIR manifest: {manifest}')
        self.kernels = []
        for row in rows:
            rir = load_audio(root / row['path'], sample_rate)
            # Align strongest arrival to zero: do not shift commands out of window.
            start = int(rir.abs().argmax())
            rir = rir[start:start + max(1, round(max_rir_seconds * sample_rate))].clone()
            norm = rir.norm()
            if norm <= 1e-12:
                raise ValueError(f'Zero-energy RIR: {row["path"]}')
            self.kernels.append(rir / norm)

    def __call__(self, waveform, label):
        if label in self.excluded_ids or torch.rand(()).item() >= self.probability:
            return waveform
        energy = waveform.square().sum()
        if energy <= 1e-12:
            return waveform
        rir = self.kernels[int(torch.randint(len(self.kernels), ()))]
        size = waveform.numel() + rir.numel() - 1
        fft_size = 1 << (size - 1).bit_length()
        wet = torch.fft.irfft(
            torch.fft.rfft(waveform, n=fft_size) * torch.fft.rfft(rir, n=fft_size),
            n=fft_size,
        )[:waveform.numel()]
        # Preserve overall input energy; avoid waveform clipping without saturation.
        wet = wet * torch.sqrt(energy / wet.square().sum().clamp_min(1e-12))
        return wet / wet.abs().max().clamp_min(1.0)


class ComposeAudio:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, waveform, label):
        for transform in self.transforms:
            waveform = transform(waveform, label)
        return waveform


class BackgroundMix:
    """Train-only background pool; SNR measured over the complete padded window."""
    def __init__(self, root, sample_rate=16000, probability=0.8,
                 snr_db=(5.0, 20.0), excluded_ids=(), cache_size=32):
        from ru_kws.data.manifest import audio_path
        if not 0 <= probability <= 1 or len(snr_db) != 2 or not all(torch.isfinite(torch.tensor(float(v))) for v in snr_db) or snr_db[0] > snr_db[1]:
            raise ValueError('Invalid background probability/SNR')
        self.root, self.sample_rate = Path(root), sample_rate
        self.probability, self.snr_db = probability, snr_db
        self.excluded_ids = set(excluded_ids)
        self.cache, self.cache_size = OrderedDict(), cache_size
        train = [json.loads(s) for s in (self.root / 'splits/train.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        source_manifest = self.root / 'sources/background_manifest.jsonl'
        sources = {}
        if source_manifest.exists():
            for line in source_manifest.read_text(encoding='utf-8').splitlines():
                if line.strip():
                    source = json.loads(line)
                    if source['id'] in sources:
                        raise ValueError(f"Duplicate background source: {source['id']}")
                    sources[source['id']] = source

        def allowed(row):
            # Explicit exclusions win. Legacy/procedural rows without a parent
            # remain eligible; source crops require explicit source permission.
            if row.get('use_for_mixing', True) is not True:
                return False
            parent = row.get('parent_id')
            if parent in sources:
                return sources[parent].get('use_for_mixing') is True
            if row.get('sample_kind') == 'procedural_noise':
                return True
            if parent:
                return False
            return row.get('sample_kind') != 'source_crop'

        rows = [r for r in train if r['label'] == 'background' and allowed(r)]
        if not rows or any(r.get('split', 'train') != 'train' for r in rows):
            raise ValueError('Missing or invalid train background pool')
        self.paths = [audio_path(root, r['path']) for r in rows]
        groups = {r['source_group'] for r in rows if r.get('source_group')}
        for split in ('val', 'test'):
            manifest = self.root / 'splits' / f'{split}.jsonl'
            if not manifest.exists():
                continue
            heldout = [json.loads(s) for s in manifest.read_text(encoding='utf-8').splitlines() if s.strip()]
            if any(audio_path(root, r['path']) in self.paths or r.get('source_group') in groups for r in heldout):
                raise ValueError(f'Train background source overlaps {split}; fix dataset split')

    def __call__(self, waveform, label):
        if label in self.excluded_ids or torch.rand(()).item() >= self.probability:
            return waveform
        power = waveform.square().mean()
        if power <= 1e-12:
            return waveform
        path = self.paths[int(torch.randint(len(self.paths), ()))]
        if path not in self.cache:
            self.cache[path] = load_audio(path, self.sample_rate)
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        noise = self.cache[path]
        n = waveform.numel()
        if noise.numel() < n:
            # Circular offset avoids always placing repetition at the same point.
            offset = int(torch.randint(noise.numel(), ()))
            noise = noise.repeat((n + offset) // noise.numel() + 1)[offset:offset + n]
        else:
            offset = int(torch.randint(noise.numel() - n + 1, ()))
            noise = noise[offset:offset + n]
        noise_power = noise.square().mean()
        if noise_power <= 1e-12:
            return waveform
        snr = self.snr_db[0] + torch.rand(()).item() * (self.snr_db[1] - self.snr_db[0])
        mixed = waveform + noise * torch.sqrt(power / (noise_power * 10 ** (snr / 10)))
        return mixed / mixed.abs().max().clamp_min(1.0)


def build_augmentation(root, split, labels, config):
    if split != 'train':
        return None
    transforms = []
    options = config.get('augmentation', {}).get('reverb', {})
    if options.get('enabled', False):
        transforms.append(RandomReverb(
        root, options.get('manifest', 'augmentation/mit_rirs/manifest.jsonl'),
        sample_rate=config['audio']['sample_rate'],
        probability=options.get('probability', 0.5),
        max_rir_seconds=options.get('max_rir_seconds', 1.0),
        excluded_ids=[labels[name] for name in options.get('exclude_labels', ['background'])],
        ))
    options = config.get('augmentation', {}).get('background', {})
    if options.get('enabled', False):
        transforms.append(BackgroundMix(
            root, sample_rate=config['audio']['sample_rate'],
            probability=options.get('probability', 0.8),
            snr_db=options.get('snr_db', [5.0, 20.0]),
            excluded_ids=[labels[name] for name in options.get('exclude_labels', ['background'])],
        ))
    options = config.get('augmentation', {}).get('gain', {})
    if options.get('enabled', False):
        transforms.append(RandomGain(probability=options.get('probability', 0.8),
                                     gain_db=options.get('gain_db', [-6.0, 6.0])))
    return ComposeAudio(transforms) if transforms else None


class SpecAugment(torch.nn.Module):
    """Qualcomm mask policy, independently per item, on [B,1,F,T] log-mel.

    https://github.com/Qualcomm-AI-research/bcresnet/blob/master/utils.py
    Width bounds are exclusive; fill is zero; no time warping.
    """
    def __init__(self, frequency_masking_para=7, time_masking_para=20,
                 frequency_mask_num=2, time_mask_num=2):
        super().__init__()
        self.settings = ((2, frequency_masking_para, frequency_mask_num),
                         (3, time_masking_para, time_mask_num))
        if any(type(v) is not int or v < 0 for _, bound, count in self.settings for v in (bound, count)):
            raise ValueError('Mask parameters must be nonnegative integers')

    def forward(self, features):
        if not self.training:
            return features
        if features.ndim != 4 or features.shape[1] != 1:
            raise ValueError('Expected [B,1,F,T] log-mel')
        result = features.clone()
        for axis, bound, count in self.settings:
            length = features.shape[axis]
            positions = torch.arange(length, device=features.device)[None, :]
            for _ in range(count):
                if bound == 0:
                    continue
                widths = torch.randint(min(bound, length + 1), (features.shape[0], 1), device=features.device)
                starts = (torch.rand(features.shape[0], 1, device=features.device) * (length - widths + 1)).long()
                mask = (positions >= starts) & (positions < starts + widths)
                shape = [features.shape[0], 1, 1, 1]
                shape[axis] = length
                result = result.masked_fill(mask.reshape(shape), 0.0)
        return result


def build_specaugment(config):
    options = dict(config.get('augmentation', {}).get('specaugment', {}))
    if not options.pop('enabled', False):
        return None
    # Frequency policy of original BC-ResNet indexed by base channels.
    options.setdefault('frequency_masking_para', {8: 0, 12: 1, 16: 3, 24: 5, 48: 7, 64: 7}[config['model']['base_c']])
    return SpecAugment(**options)
