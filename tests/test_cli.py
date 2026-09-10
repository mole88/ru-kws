"""Exercise the actual model and train/evaluate CLIs on tiny artificial WAVs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import yaml


def test_train_checkpoint_evaluate_roundtrip(tmp_path):
    root = tmp_path / "dataset"
    (root / "splits").mkdir(parents=True)
    labels = {"next": 0, "unknown": 1, "background": 2}
    (root / "labels.json").write_text(json.dumps(labels))
    for split_index, split in enumerate(("train", "val", "test")):
        rows = []
        for i, label in enumerate(labels):
            name = f"{split}_{i}.wav"
            t = np.arange(1600) / 16000
            sf.write(root / name, 0.1 * np.sin(2 * np.pi * (200 + 90 * i + split_index) * t), 16000)
            rows.append({"path": name, "label": label})
        (root / "splits" / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    repository = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((repository / "configs/baseline.yaml").read_text())
    config["audio"]["window_seconds"] = 0.4
    config["model"]["base_c"] = 8
    config["training"].update(max_epochs=1, batch_size=2)
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(yaml.safe_dump(config))
    run = tmp_path / "run"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    def cli(*args):
        return subprocess.run([sys.executable, "-m", *map(str, args)], env=env,
                              capture_output=True, text=True, timeout=120)
    result = cli("ru_kws.train", "--config", config_path, "--data-root", root,
                 "--run-dir", run, "--device", "cpu")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (run / "best.pt").is_file() and (run / "last.pt").is_file()
    report_path = tmp_path / "test.json"
    result = cli("ru_kws.evaluate", "--checkpoint", run / "best.pt", "--data-root", root,
                 "--split", "test", "--output", report_path, "--device", "cpu")
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report["count"] == 3 and report["epoch"] == 1
    assert sum(map(sum, report["confusion_matrix"])) == 3
    assert report["class_order"] == list(labels)
