import json
import math

import numpy as np
import soundfile as sf
import torch

from ru_kws.data.augmentation import RandomReverb, build_augmentation
from ru_kws.data.collate import PadCropCollate
from ru_kws.data.augmentation import BackgroundMix, SpecAugment
from ru_kws.data.augmentation import RandomGain, RandomSpeed, build_speed_augmentation
import pytest


def make_reverb(tmp_path, kernel, **kwargs):
    sf.write(tmp_path / 'rir.wav', np.asarray(kernel, dtype=np.float32), 16000, subtype='FLOAT')
    (tmp_path / 'manifest.jsonl').write_text(json.dumps({'path': 'rir.wav'}) + '\n')
    return RandomReverb(tmp_path, 'manifest.jsonl', probability=1, **kwargs)


def test_delta_preserves_audio(tmp_path):
    aug = make_reverb(tmp_path, [0, 1, 0])
    x = torch.randn(1600) * 0.03
    torch.testing.assert_close(aug(x, 0), x, atol=1e-6, rtol=1e-5)


def test_echo_matches_direct_convolution(tmp_path):
    aug = make_reverb(tmp_path, [1, 0, 0.5])
    x = torch.zeros(32)
    x[3] = 0.1
    expected = torch.from_numpy(np.convolve(x.numpy(), [1, 0, 0.5])[:32]).float()
    expected *= x.norm() / expected.norm()
    torch.testing.assert_close(aug(x, 0), expected, atol=1e-6, rtol=1e-5)


def test_silence_and_excluded_class(tmp_path):
    aug = make_reverb(tmp_path, [1, 0.5], excluded_ids=[7])
    x = torch.randn(32)
    assert aug(x, 7) is x
    assert torch.equal(aug(torch.zeros(32), 0), torch.zeros(32))


def test_eval_never_loads_rirs_or_augments(tmp_path):
    config = {'augmentation': {'reverb': {'enabled': True}}}
    assert build_augmentation(tmp_path, 'val', {}, config) is None
    assert build_augmentation(tmp_path, 'test', {}, config) is None
    def fail(*args):
        raise AssertionError('Evaluation must not augment')
    collate = PadCropCollate(8, [0], training=False, augmentation=fail)
    x, _ = collate([(torch.ones(4), 0)])
    torch.testing.assert_close(x[0], torch.tensor([0., 0., 1., 1., 1., 1., 0., 0.]))


def test_seed_and_length(tmp_path):
    aug = make_reverb(tmp_path, [1, 0.5])
    collate = PadCropCollate(160, [0], training=True, augmentation=aug)
    batch = [(torch.ones(60) * 0.1, 0)]
    torch.manual_seed(123)
    first, _ = collate(batch)
    torch.manual_seed(123)
    second, _ = collate(batch)
    assert first.shape == (1, 160)
    assert torch.isfinite(first).all() and first.abs().max() <= 1
    torch.testing.assert_close(first, second)


def noise_pool(tmp_path):
    (tmp_path / 'splits').mkdir()
    sf.write(tmp_path / 'noise.wav', np.ones(400, dtype=np.float32) * 0.1, 16000, subtype='FLOAT')
    row = {'path': 'noise.wav', 'label': 'background', 'source_group': 'room1'}
    (tmp_path / 'splits/train.jsonl').write_text(json.dumps(row) + '\n')
    return row


def test_gain_db_and_peak():
    x = torch.ones(100) * 0.1
    y = RandomGain(1, [-6, -6])(x, 0)
    torch.testing.assert_close(y, x * 10 ** (-6 / 20))
    assert RandomGain(1, [6, 6])(torch.ones(100) * 0.9, 0).max() <= 1
    assert torch.equal(RandomGain(1)(torch.zeros(20), 0), torch.zeros(20))


def test_speed_duration_and_command_protection():
    x = torch.ones(1600) * 0.1
    fast = RandomSpeed(48000, [0], 1, [1.1, 1.1])(x, 0)
    slow = RandomSpeed(48000, [0], 1, [0.9, 0.9])(x, 0)
    assert fast.numel() == math.ceil(1600 / 1.1)
    assert slow.numel() == math.ceil(1600 / 0.9)
    full = torch.ones(48000) * 0.1
    assert RandomSpeed(48000, [0], 1, [0.9, 0.9])(full, 0) is full
    aug = RandomSpeed(48000, [0], 1)
    for _ in range(5):
        assert aug(full, 0).numel() <= 48000
    assert build_speed_augmentation('val', [0], {'augmentation': {'speed': {'enabled': True}}}) is None


def test_speed_gain_seed_and_eval():
    speed = RandomSpeed(1600, [0], 1)
    gain = RandomGain(1)
    collate = PadCropCollate(1600, [0], True, gain, speed)
    batch = [(torch.ones(800) * 0.1, 0)]
    torch.manual_seed(9)
    first, _ = collate(batch)
    torch.manual_seed(9)
    second, _ = collate(batch)
    torch.testing.assert_close(first, second)
    evaluation = PadCropCollate(1600, [0], False, gain, speed)
    x, _ = evaluation(batch)
    assert x[0, 400:1200].eq(0.1).all()


def test_noise_snr_and_label_exclusion(tmp_path):
    noise_pool(tmp_path)
    aug = BackgroundMix(tmp_path, probability=1, snr_db=[10, 10], excluded_ids=[7])
    x = torch.ones(1600) * 0.1
    mixed = aug(x, 0)
    snr = 10 * torch.log10(x.square().mean() / (mixed - x).square().mean())
    assert snr.item() == pytest.approx(10, abs=1e-4)
    assert aug(x, 7) is x
    assert torch.equal(aug(torch.zeros(1600), 0), torch.zeros(1600))


def test_noise_rejects_heldout_source(tmp_path):
    row = noise_pool(tmp_path)
    row['path'] = 'other.wav'
    (tmp_path / 'splits/val.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='overlaps val'):
        BackgroundMix(tmp_path)


def test_specaugment_masks_and_eval():
    aug = SpecAugment()
    x = torch.ones(8, 1, 40, 301) * -3
    torch.manual_seed(1)
    y = aug(x)
    assert torch.equal(x, torch.full_like(x, -3))
    assert y.shape == x.shape and (y == 0).any()
    assert ((y == 0) | (y == -3)).all()
    # Two frequency widths <7 and two temporal widths <20.
    assert (y[0, 0] == 0).all(dim=1).sum() <= 12
    assert (y[0, 0] == 0).all(dim=0).sum() <= 38
    torch.manual_seed(1)
    torch.testing.assert_close(y, aug(x))
    aug.eval()
    assert aug(x) is x


def test_engine_does_not_augment_validation():
    from ru_kws.training.engine import run_epoch
    class Forbidden(torch.nn.Module):
        def forward(self, x):
            raise AssertionError('SpecAugment in validation')
    model = torch.nn.Linear(4, 2)
    result = run_epoch(model, torch.nn.Identity(), [(torch.ones(2, 4), torch.zeros(2, dtype=torch.long))],
                       torch.nn.CrossEntropyLoss(), 'cpu', specaugment=Forbidden())
    assert result['count'] == 2


def test_background_respects_source_and_crop_permissions(tmp_path):
    row = noise_pool(tmp_path)
    (tmp_path / 'sources').mkdir()
    sources = [{'id': 'allowed', 'use_for_mixing': True}, {'id': 'banned', 'use_for_mixing': False}]
    (tmp_path / 'sources/background_manifest.jsonl').write_text('\n'.join(map(json.dumps, sources)))
    rows = [row]
    for parent in ['allowed', 'banned', 'missing']:
        rows.append({**row, 'path': f'{parent}.wav', 'parent_id': parent, 'sample_kind': 'source_crop'})
    rows.append({**row, 'path': 'crop_banned.wav', 'parent_id': 'allowed', 'use_for_mixing': False})
    rows.append({**row, 'path': 'orphan.wav', 'sample_kind': 'source_crop'})
    rows.append({**row, 'path': 'procedural.wav', 'sample_kind': 'procedural_noise', 'parent_id': 'self'})
    (tmp_path / 'splits/train.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    assert {p.name for p in BackgroundMix(tmp_path).paths} == {'noise.wav', 'allowed.wav', 'procedural.wav'}
    (tmp_path / 'splits/train.jsonl').write_text(json.dumps(rows[2]))
    with pytest.raises(ValueError, match='background pool'):
        BackgroundMix(tmp_path)
