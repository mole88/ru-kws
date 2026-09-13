"""Read-only quality audit and small failure probes; no training data is modified.

Run from the repository root with .venv/Scripts/python.exe -X utf8
docs/quality-audit-2026-09-13/reproduce.py.
"""
import collections
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np
import soundfile as sf
import torch

from ru_kws.data.augmentation import BackgroundMix
from ru_kws.data.dataset import make_loader
from ru_kws.data.validate import validate_dataset
from ru_kws.config import load_config
from ru_kws.evaluation.clips import evaluate_clips

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nonfinite_probes():
    class BrokenModel(torch.nn.Module):
        def forward(self, waveforms):
            return torch.full((len(waveforms), 2), float("nan"))

    report = evaluate_clips(BrokenModel(), torch.nn.Identity(),
                            [(torch.zeros(3, 100), torch.zeros(3, dtype=torch.long))],
                            {"next": 0, "unknown": 1}, "cpu")
    assert report["accuracy"] == 1.0
    with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
        root = Path(folder)
        (root / "splits").mkdir()
        (root / "labels.json").write_text('{"next": 0}', encoding="utf-8")
        for i, split in enumerate(("train", "val", "test")):
            audio = np.full(1600, (i + 1) * .1, dtype=np.float32)
            audio[0] = np.nan
            sf.write(root / f"{split}.wav", audio, 16000, subtype="FLOAT")
            row = {"path": f"{split}.wav", "label": "next"}
            (root / "splits" / f"{split}.jsonl").write_text(json.dumps(row), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            validation = validate_dataset(root)
    return {"nan_logits_accuracy": report["accuracy"], "nan_wav_validation_accepted": validation}


def loader_leakage_probe():
    with tempfile.TemporaryDirectory(dir=OUTPUT) as folder:
        root = Path(folder)
        (root / "splits").mkdir()
        (root / "labels.json").write_text('{"next": 0}', encoding="utf-8")
        sf.write(root / "shared.wav", np.ones(1600, np.float32) * .1, 16000)
        row = {"path": "shared.wav", "label": "next", "speaker_group": "same-speaker"}
        for split in ("train", "val", "test"):
            (root / "splits" / f"{split}.jsonl").write_text(json.dumps(row), encoding="utf-8")
        config = load_config(ROOT / "configs/baseline.yaml")
        with contextlib.redirect_stdout(io.StringIO()):
            counts = {split: len(make_loader(root, split, {"next": 0}, config).dataset)
                      for split in ("train", "val")}
            try:
                validate_dataset(root)
            except ValueError as exc:
                reason = str(exc)
            else:
                raise AssertionError("The explicit validator should reject this fixture")
    return {"training_loaders_accept_identical_train_val": counts, "explicit_validator_rejects": reason}


def dataset_audit():
    roots = [path.parent for path in (ROOT / "datasets").rglob("labels.json")]
    if len(roots) != 1:
        raise ValueError(f"Select an unambiguous dataset root: {roots}")
    root = roots[0]
    split_rows = {split: read_jsonl(root / "splits" / f"{split}.jsonl")
                  for split in ("train", "val", "test")}
    sources = read_jsonl(root / "sources/background_manifest.jsonl")
    banned = {row["id"] for row in sources if row.get("use_for_mixing") is False}
    pool = set(BackgroundMix(root).paths)
    forbidden = [row for row in split_rows["train"] if row.get("parent_id") in banned
                 and (root / row["path"]).resolve() in pool]
    metadata_groups = {key: collections.defaultdict(set) for key in
                       ("speaker_group", "parent_id", "source_group", "text_group")}
    pcm_groups = collections.defaultdict(set)
    checks = collections.Counter()
    invalid_examples = []
    dropped = {split: collections.Counter() for split in split_rows}
    for split, rows in split_rows.items():
        for row in rows:
            path = root / row["path"]
            raw = path.read_bytes()
            audio, rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
            checks["wav_files"] += 1
            checks["nonfinite_files"] += int(not np.isfinite(audio).all())
            checks["checksum_mismatches"] += int(bool(row.get("sha256")) and hashlib.sha256(raw).hexdigest() != row["sha256"])
            checks["invalid_format"] += int(rate != 16000 or audio.shape[1] != 1 or not len(audio))
            checks["duration_metadata_mismatches"] += int(abs(len(audio) / rate - row["duration_s"]) > 1 / rate)
            if not np.isfinite(audio).all():
                invalid_examples.append(row["path"])
            if row["label"] not in {"unknown", "background"} and len(audio) > 48000:
                dropped[split][row["label"]] += 1
            pcm_groups[hashlib.sha256(audio.tobytes()).hexdigest()].add(split)
            for key, groups in metadata_groups.items():
                if row.get(key) and (key != "text_group" or row["label"] == "unknown"):
                    groups[row[key]].add(split)
    return {
        "root": str(root), "checks": dict(checks), "invalid_examples": invalid_examples,
        "counts": {split: dict(collections.Counter(row["label"] for row in rows)) for split, rows in split_rows.items()},
        "dropped_over_3_seconds": {split: dict(values) for split, values in dropped.items()},
        "cross_split_group_overlaps": {key: sum(len(splits) > 1 for splits in groups.values())
                                       for key, groups in metadata_groups.items()},
        "cross_split_pcm_duplicates": sum(len(splits) > 1 for splits in pcm_groups.values()),
        "background_pool_size": len(pool),
        "forbidden_mixing_crops": len(forbidden),
        "forbidden_source_ids": sorted({row["parent_id"] for row in forbidden}),
        "forbidden_example": forbidden[0]["path"] if forbidden else None,
        "manifest_sha256": {split: hashlib.sha256((root / "splits" / f"{split}.jsonl").read_bytes()).hexdigest()
                            for split in split_rows},
    }


def source_artifacts():
    archives = {}
    for relative, prefix in (("dist/continuous-evaluation-tools.zip", ""),
                             ("dist/ru-kws-source.zip", "ru-kws/")):
        path = ROOT / relative
        if not path.exists():
            continue
        differences = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                local = ROOT / ("scripts" if not prefix else "") / member[len(prefix):]
                if local.is_file() and archive.read(member) != local.read_bytes():
                    differences.append(str(local.relative_to(ROOT)))
            missing = [name for name in ("src/ru_kws/data/augmentation.py", "configs/augmented.yaml",
                                        "src/ru_kws/data/filtering.py")
                       if prefix and prefix + name not in archive.namelist()]
        archives[relative] = {"changed_files": differences, "missing_features": missing,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    source = {}
    for folder in ("src", "scripts", "tests", "configs", "notebooks", "voice_recorder", "local_streaming_test"):
        paths = (ROOT / folder).glob("*") if folder in {"voice_recorder", "local_streaming_test"} else (ROOT / folder).rglob("*")
        for path in paths:
            if path.is_file() and path.suffix in {".py", ".yaml", ".ipynb"} and "__pycache__" not in path.parts:
                source[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"archives": archives, "source_sha256": source}


if __name__ == "__main__":
    result = {"nonfinite": nonfinite_probes(), "loader_integrity": loader_leakage_probe(),
              "dataset": dataset_audit(), "artifacts": source_artifacts()}
    destination = OUTPUT / "checks.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
