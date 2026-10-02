import json

import numpy as np
import pytest
import soundfile as sf
import torch

from ru_kws.audio.frontend import LogMelFrontend
from ru_kws.data.collate import PadCropCollate
from ru_kws.data.validate import validate_dataset
from ru_kws.training.engine import run_epoch
from ru_kws.evaluation.clips import classification_metrics


def test_collate_preserves_command_and_eval_alignment():
    collate = PadCropCollate(10, {0}, training=False)
    x, y = collate([(torch.tensor([1., 2., 3., 4.]), 0)])
    assert x.tolist() == [[0., 0., 0., 1., 2., 3., 4., 0., 0., 0.]]
    with pytest.raises(ValueError, match="Command exceeds"):
        collate([(torch.ones(11), 0)])
    x, _ = collate([(torch.arange(14).float(), 1)])
    assert x.tolist() == [list(range(2, 12))]


def test_frontend_baseline_shape_and_finite_silence():
    features = LogMelFrontend()(torch.zeros(1, 48000))
    assert features.shape == (1, 1, 40, 301)
    assert torch.isfinite(features).all()


def test_epoch_weights_last_batch_by_sample_count():
    logits = torch.tensor([[8., 0.]] * 32 + [[0., 8.]] * 2)
    targets = torch.zeros(34, dtype=torch.long)
    loader = [(logits[:32], targets[:32]), (logits[32:], targets[32:])]
    loss = torch.nn.CrossEntropyLoss()
    result = run_epoch(torch.nn.Identity(), torch.nn.Identity(), loader, loss, "cpu")
    assert result["accuracy"] == pytest.approx(32 / 34)
    assert result["loss"] == pytest.approx(loss(logits, targets).item())


def test_confusion_orientation_and_missing_class():
    cm = torch.tensor([[2, 1, 0], [0, 1, 0], [0, 0, 0]])
    result = classification_metrics(cm, {"next": 0, "unknown": 1, "background": 2})
    assert result["accuracy"] == 0.75
    assert result["per_class"]["next"]["recall"] == pytest.approx(2 / 3)
    assert result["per_class"]["unknown"]["precision"] == 0.5
    assert result["per_class"]["background"]["f1"] == 0


def test_validator_rejects_related_sources_across_splits(tmp_path):
    (tmp_path / "splits").mkdir()
    (tmp_path / "labels.json").write_text('{"next": 0}')
    for i, split in enumerate(("train", "val", "test")):
        name = f"{split}.wav"
        sf.write(tmp_path / name, np.full(1600, 0.1 * (i + 1)), 16000)
        row = {"path": name, "label": "next", "speaker_group": "same-speaker"}
        (tmp_path / "splits" / f"{split}.jsonl").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="Split leakage: speaker_group"):
        validate_dataset(tmp_path)

@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_evaluation_rejects_nonfinite_logits(value):
    from ru_kws.evaluation.clips import evaluate_clips
    loader = [(torch.full((3, 2), value), torch.zeros(3, dtype=torch.long))]
    with pytest.raises(ValueError, match='Nonfinite logits'):
        evaluate_clips(torch.nn.Identity(), torch.nn.Identity(), loader, {'a': 0, 'b': 1}, 'cpu')


def test_evaluation_rejects_wrong_output_shape():
    from ru_kws.evaluation.clips import evaluate_clips
    with pytest.raises(ValueError, match='Expected logits'):
        evaluate_clips(torch.nn.Identity(), torch.nn.Identity(),
                       [(torch.zeros(3, 1), torch.zeros(3, dtype=torch.long))], {'a': 0, 'b': 1}, 'cpu')


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_validator_rejects_nonfinite_wav_with_valid_checksum(tmp_path, value):
    import hashlib
    (tmp_path / 'splits').mkdir()
    (tmp_path / 'labels.json').write_text('{"next": 0}')
    audio = np.zeros(70000, dtype=np.float32)
    audio[-1] = value
    path = tmp_path / 'bad.wav'
    sf.write(path, audio, 16000, subtype='FLOAT')
    row = {'path': path.name, 'label': 'next', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'splits/train.jsonl').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='Nonfinite WAV'):
        validate_dataset(tmp_path, window_seconds=5, splits=('train',))
