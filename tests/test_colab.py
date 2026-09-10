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
