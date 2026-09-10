import hashlib
import json
from pathlib import Path


def read_labels(root):
    labels = json.loads((Path(root) / "labels.json").read_text(encoding="utf-8"))
    if not labels or any(type(v) is not int for v in labels.values()):
        raise ValueError("labels.json must map class names to integer IDs")
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("Label IDs must be unique and contiguous, starting at zero")
    return labels


def audio_path(root, relative_path):
    root = Path(root).resolve()
    path = (root / relative_path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Audio path must stay inside dataset root: {relative_path}")
    return path


def read_manifest(root, split, labels):
    path = Path(root) / "splits" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    for row in rows:
        if row["label"] not in labels:
            raise ValueError(f"Unknown label in {path}: {row['label']}")
        if row.get("split", split) != split:
            raise ValueError(f"Wrong split in {path}: {row}")
        if not audio_path(root, row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    return rows


def manifest_hash(root, split):
    return hashlib.sha256((Path(root) / "splits" / f"{split}.jsonl").read_bytes()).hexdigest()
