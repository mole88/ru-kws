"""Evaluate the local FP32 model on val using existing runtime and padding code.

Run with local_streaming_test/.venv/Scripts/python.exe -X utf8.
No microphone is opened. The final test split is not evaluated.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import wave

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "local_streaming_test"))
sys.path.insert(0, str(ROOT / "scripts"))
from engine import Model
from evaluate_tflite import fit_audio

root = next((ROOT / "datasets").rglob("labels.json")).parent
model_path = ROOT / "local_streaming_test/models/bcresnet_with_frontend_fp32.tflite"
model_labels = ROOT / "local_streaming_test/models/labels.json"
labels = json.loads((root / "labels.json").read_text(encoding="utf-8"))
assert labels == json.loads(model_labels.read_text(encoding="utf-8"))
model = Model(model_path, model_labels)
rows = [json.loads(line) for line in (root / "splits/val.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
skipped = []
args = SimpleNamespace(pad_position="center", crop_labels=["unknown", "background"], long_commands="skip")
for index, row in enumerate(rows):
    with wave.open(str(root / row["path"]), "rb") as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == model.rate
        audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768
    fitted, action = fit_audio(audio, model.size, row["label"], args)
    if fitted is None:
        skipped.append(row["path"])
        continue
    probs = model.predict(fitted)
    cm[labels[row["label"]], int(probs.argmax())] += 1
    if index % 250 == 0:
        print(f"val {index + 1}/{len(rows)}", flush=True)
result = {
    "note": "Existing model, provenance not verified against a checkpoint; clip validation only.",
    "model": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "manifest_sha256": hashlib.sha256((root / "splits/val.jsonl").read_bytes()).hexdigest(),
    "class_order": model.names, "confusion_matrix": cm.tolist(),
    "evaluated": int(cm.sum()), "accuracy": float(cm.trace() / cm.sum()), "skipped": skipped,
    "recall": {name: float(cm[i, i] / cm[i].sum()) if cm[i].sum() else None for i, name in enumerate(model.names)},
    "silence_probabilities": dict(zip(model.names, model.predict(np.zeros(model.size, np.float32)).tolist())),
}
output = Path(__file__).resolve().parent / "local-model-val.json"
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: result[key] for key in ("accuracy", "evaluated", "recall")}, indent=2))
