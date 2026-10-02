"""Exercise the notebook frontend without Colab or a TFLite converter."""
import ast
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from ru_kws.audio.frontend import LogMelFrontend


ROOT = Path(__file__).resolve().parents[1]


def export_frontend_class():
    notebook = json.loads((ROOT / 'notebooks/ru_kws_training.ipynb').read_text(encoding='utf-8'))
    source = next(c['source'] for c in notebook['cells']
                  if 'class ExportableLogMelFrontend' in c['source'])
    tree = ast.parse(source)
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    namespace = dict(torch=torch, nn=nn, F=F)
    exec(compile(ast.Module(body=[definition], type_ignores=[]), 'frontend', 'exec'), namespace)
    return namespace['ExportableLogMelFrontend']


@pytest.mark.parametrize('n_fft,win_length,hop_length', [(512, 480, 160), (256, 200, 80)])
def test_export_frontend_matches_training(n_fft, win_length, hop_length):
    reference = LogMelFrontend(n_fft=n_fft, win_length=win_length, hop_length=hop_length).eval()
    exported = export_frontend_class()(reference).eval()
    generator = torch.Generator().manual_seed(42)
    time = torch.arange(48000) / 16000
    signals = torch.cat([
        torch.zeros(1, 48000),
        torch.randn(1, 48000, generator=generator) * .01,
        (.1 * torch.sin(2 * torch.pi * 440 * time)).unsqueeze(0),
    ])
    with torch.inference_mode():
        expected, actual = reference(signals), exported(signals)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


def test_export_frontend_rejects_unsupported_centering():
    with pytest.raises(AssertionError):
        export_frontend_class()(LogMelFrontend(center=False))


def test_generator_preserves_current_notebook_cells(tmp_path):
    script = tmp_path / 'scripts/build_colab.py'
    script.parent.mkdir()
    script.write_bytes((ROOT / 'scripts/build_colab.py').read_bytes())
    subprocess.run([sys.executable, str(script)], check=True, capture_output=True)
    generated = json.loads((tmp_path / 'notebooks/ru_kws_training.ipynb').read_text(encoding='utf-8'))
    actual = json.loads((ROOT / 'notebooks/ru_kws_training.ipynb').read_text(encoding='utf-8'))
    assert generated['cells'] == actual['cells']
