"""Verify the launcher without accessing Drive or installing dependencies."""
import ast
import hashlib
import json
from pathlib import Path
import stat
import zipfile

import pytest


def notebook():
    path = Path(__file__).resolve().parents[1] / "notebooks/ru_kws_training.ipynb"
    return json.loads(path.read_text(encoding="utf-8"))


def test_notebook_code_compiles_and_has_no_outputs():
    nb = notebook()
    assert nb["nbformat"] == 4
    for index, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"cell_{index}", "exec")
            assert cell["outputs"] == [] and cell["execution_count"] is None


def helpers():
    source = next(c["source"] for c in notebook()["cells"] if "def extract_cached(" in c["source"])
    tree = ast.parse(source)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"file_sha256", "extract_cached", "find_root"}]
    namespace = {"Path": Path, "hashlib": hashlib, "zipfile": zipfile, "stat": stat}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "helpers", "exec"), namespace)
    return namespace


def test_zip_helpers_find_nested_dataset_and_reuse_cache(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("inner/labels.json", '{"next": 0}')
        for split in ("train", "val", "test"):
            zf.writestr(f"inner/splits/{split}.jsonl", "")
    ns = helpers()
    extracted, digest = ns["extract_cached"](archive, tmp_path / "cache")
    assert digest == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert ns["find_root"](extracted, "labels.json", ["splits/train.jsonl"]) == extracted / "inner"
    assert ns["extract_cached"](archive, tmp_path / "cache") == (extracted, digest)


def test_zip_helpers_reject_path_escape(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../../escaped.txt", "bad")
    with pytest.raises(ValueError, match="ZIP"):
        helpers()["extract_cached"](archive, tmp_path / "cache")
    assert not (tmp_path / "escaped.txt").exists()


def test_notebook_measurements_are_unique_and_read_legacy_runs(tmp_path):
    from datetime import datetime, timezone
    from uuid import uuid4
    source = next(c['source'] for c in notebook()['cells'] if 'def measurement_dir(' in c['source'])
    definitions = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                   and n.name in {'unique_id', 'saved_path', 'measurement_dir', 'save_measurement'}]
    ns = dict(Path=Path, datetime=datetime, timezone=timezone, uuid4=uuid4, json=json,
              RUN_DIR=tmp_path, source_info={'revision': 'test'},
              file_sha256=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest())
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'storage', 'exec'), ns)
    legacy = tmp_path / 'best.pt'
    legacy.write_bytes(b'legacy')
    assert ns['saved_path']('checkpoints', 'best.pt') == legacy
    (tmp_path / 'checkpoints').mkdir()
    current = tmp_path / 'checkpoints/best.pt'
    current.write_bytes(b'current')
    assert ns['saved_path']('checkpoints', 'best.pt') == current
    first, second = [ns['measurement_dir']('continuous') for _ in range(2)]
    assert first != second and first.parent == tmp_path / 'measurements/continuous'
    for path in (first, second):
        ns['save_measurement'](path, 'continuous', {'checkpoint': current}, threshold=.5)
        record = json.loads((path / 'measurement.json').read_text())
        assert record['settings']['threshold'] == .5
        assert record['inputs']['checkpoint']['sha256'] == hashlib.sha256(b'current').hexdigest()


def test_notebook_headings_only_and_continuous_stage_present():
    cells = notebook()['cells']
    for cell in cells:
        if cell['cell_type'] == 'markdown':
            lines = cell['source'].strip().splitlines()
            assert len(lines) == 1 and lines[0].startswith('#')
    assert any('scripts/evaluate_continuous_tflite.py' in c['source'] for c in cells)
    assert any("'timeline.png'" in c['source'] for c in cells)
