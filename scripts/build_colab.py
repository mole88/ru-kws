"""Build the main Colab notebook. Edit cell sources here, then run this script."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source, metadata=None):
    cell = dict(cell_type=kind, metadata=metadata or {}, id=f"cell-{len(cells):02d}",
                source=dedent(source).strip() + "\n")
    if kind == "code":
        if "class ExportableLogMelFrontend" in source:
            cell["metadata"]["jupyter"] = {"source_hidden": True}
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

add('markdown', r'''
# Russian KWS
''', {})

add('markdown', '## 1. Setup')
add('markdown', '### 1.1 Configuration')

add('code', r'''
from pathlib import Path
import sys, os, json, subprocess, hashlib, zipfile, tempfile, stat, csv
import torch
from datetime import datetime, timezone
from uuid import uuid4

REPO_URL = "https://github.com/mole88/ru-kws.git"
REPO_REF = "dev"
DATASET_ZIP = Path("/content/drive/MyDrive/russian_commands/russian_commands_v003_clean.zip")
RUN_NAME = "bcresnet_clean_v003_001"
RUNS_ROOT = Path("/content/drive/MyDrive/russian_commands/runs")
BASE_CONFIG = "augmented.yaml"
WINDOW_SECONDS = 3.0
BASE_C = 64
BATCH_SIZE = 64
MAX_EPOCHS = 120
LEARNING_RATE = 3e-4
REQUIRE_GPU = False

# Evaluation-only sessions: set RUN_NAME to the candidate's saved run and supply its export.
EXISTING_TFLITE_PATH = None
RUN_FINAL_TEST = False
RUN_VANYA_TEST = False
RUN_CONTINUOUS_TESTS = False
RUN_PODCAST_TEST = False
RUN_BASELINE_COMPARISON = False

if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
    raise ValueError("RUN_NAME must be a single directory name")
RUN_DIR = RUNS_ROOT / RUN_NAME

os.environ["PYTHONUNBUFFERED"] = "1" # for realtime logs output
os.environ["MPLBACKEND"] = "Agg" # for stable working with graphics libs
print("Run path:", RUN_DIR)
''', {})

add('markdown', r'''
### 1.2 Mount Google Drive
''', {})

add('code', r'''
from google.colab import drive
drive.mount("/content/drive")
''', {})

add('markdown', r'''
### 1.3 Project and evaluation tools
''', {})

add('code', r'''
def unique_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + "_" + uuid4().hex[:8]

def saved_path(folder, name):
    path = RUN_DIR / folder / name
    return path if path.exists() else RUN_DIR / name  # Existing flat runs.

def measurement_dir(kind):
    return RUN_DIR / "measurements" / kind / unique_id()

def save_measurement(directory, kind, inputs, **settings):
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"schema_version": 1, "kind": kind, "created_at": datetime.now(timezone.utc).isoformat(),
               "run": str(RUN_DIR), "source": source_info, "settings": settings,
               "inputs": {name: {"path": str(path), "sha256": file_sha256(path)}
                          for name, path in inputs.items()}}
    (directory / "measurement.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
def run(command, cwd=None):
    print("Run:", " ".join(map(str, command)), flush=True)
    with subprocess.Popen(list(map(str, command)), cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Error {process.returncode}")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract_cached(archive, parent):
    if not archive.is_file():
        raise FileNotFoundError(f"Fix archive path: {archive}")
    digest = file_sha256(archive)
    target = parent / digest
    marker = target / ".extraction_complete"
    if marker.exists():
        print("Use unziped archive:", target)
        return target, digest
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Invalid path in ZIP: {member.filename}")
        print(f"Unpacking {len(members)} files...", flush=True)
        zf.extractall(target)
    marker.write_text(digest)
    return target, digest

def find_root(parent, marker, required):
    candidates = [p.parent for p in parent.rglob(marker)
                  if "__MACOSX" not in p.parts and all((p.parent / item).exists() for item in required)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one directory containing {marker}, found: {candidates}")
    return candidates[0]

REPO = Path(tempfile.mkdtemp(prefix="ru-kws-", dir="/content"))
run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, REPO])
run(["git", "fetch", "--depth", "1", "origin", REPO_REF], cwd=REPO)
run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
for script_name in ["evaluate_tflite.py", "compare_tflite_models.py", "evaluate_continuous_tflite.py"]:
    if not (REPO / "scripts" / script_name).is_file():
        raise FileNotFoundError(f"Missing evaluation script in {REPO_REF}: {script_name}")
source_info = {"mode": "github", "url": REPO_URL, "requested_ref": REPO_REF, "revision": revision}
print("Project:", REPO)
print(json.dumps(source_info, ensure_ascii=False, indent=2))
''', {})


add('markdown', r'''
### 1.4 Environment
''', {})

add('code', r'''
run([sys.executable, "-m", "pip", "install", "numpy>=1.24", "soundfile>=0.12", "PyYAML>=6", "matplotlib>=3.6", "packaging", "pandas", "scipy", "scikit-learn", "tqdm", "ai-edge-litert"])
run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", REPO])
project_src = str(REPO / "src")
if project_src not in sys.path:
    sys.path.insert(0, project_src)
probe = """
import sys, json, torch, torchaudio
from packaging.version import Version
from ru_kws.audio.frontend import LogMelFrontend
from ru_kws.models.bcresnet import BCResNets
assert Version(torch.__version__.split('+')[0]) >= Version('2.5')
assert Version(torchaudio.__version__.split('+')[0]) >= Version('2.5')
assert not REQUIRE_GPU or torch.cuda.is_available(), 'Enable a GPU in the Colab settings'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
with torch.inference_mode():
    x = LogMelFrontend().to(device)(torch.zeros(1, 48000, device=device))
    model = BCResNets(8, num_classes=8).to(device).eval()
    y = model(x)
assert x.shape == (1, 1, 40, 301) and y.shape == (1, 8)
print(json.dumps({'python': sys.version, 'torch': str(torch.__version__),
    'torchaudio': str(torchaudio.__version__), 'device': device,
    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, indent=2))
"""
run([sys.executable, "-c", "REQUIRE_GPU = " + repr(REQUIRE_GPU) + "\n" + probe])
''', {})

add('markdown', r'''
### 1.5 Load the locally cleaned dataset
''', {})

add('code', r'''
unpacked, dataset_archive_hash = extract_cached(DATASET_ZIP, Path("/content/ru_kws_data"))
DATA_ROOT = find_root(unpacked, "labels.json", ["splits/train.jsonl", "splits/val.jsonl", "splits/test.jsonl"])
print("Dataset:", DATA_ROOT)
print("Labels:", json.loads((DATA_ROOT / "labels.json").read_text(encoding="utf-8")))
cleaning_path = DATA_ROOT / "cleaning_report.json"
if not cleaning_path.is_file():
    raise ValueError("Prepare a clean dataset locally with scripts/clean_dataset.py before evaluation")
cleaning_report = json.loads(cleaning_path.read_text(encoding="utf-8"))
if not cleaning_report.get("complete") or not cleaning_report.get("vanya_separate_test"):
    raise ValueError("Rebuild the dataset with scripts/clean_dataset.py: Vanya must be a separate test")
for manifest_name in ["synthetic_test.jsonl", "vanya_test.jsonl"]:
    if not (DATA_ROOT / "evaluation" / manifest_name).is_file():
        raise FileNotFoundError(manifest_name)
''', {})

add('markdown', r'''
## 2. Training
''', {})
add('markdown', '### 2.1 Configuration and strict data validation')

add('code', r'''
import yaml
config = yaml.safe_load((REPO / "configs" / BASE_CONFIG).read_text(encoding="utf-8"))
config["audio"]["window_seconds"] = WINDOW_SECONDS
config["model"]["base_c"] = BASE_C
config["training"].update(batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                          learning_rate=LEARNING_RATE, num_workers=0)
CONFIG_PATH = Path("/content/ru_kws_colab_config.yaml")
CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(CONFIG_PATH.read_text())
run([sys.executable, "-m", "ru_kws.data.validate", "--data-root", DATA_ROOT,
     "--window-seconds", WINDOW_SECONDS], cwd=REPO)
''', {})

add('markdown', r'''
### 2.2 Train a new model
''', {})

add('code', r'''
if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"Run directory already exists: {RUN_DIR}.")
run([sys.executable, "-m", "ru_kws.train", "--config", CONFIG_PATH,
     "--data-root", DATA_ROOT, "--run-dir", RUN_DIR, "--device", "auto"], cwd=REPO)
(RUN_DIR / "metadata" / "colab_source.json").write_text(json.dumps({
    "source": source_info, "dataset_archive": str(DATASET_ZIP),
    "dataset_archive_sha256": dataset_archive_hash,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Best checkpoint:", saved_path("checkpoints", "best.pt"))
''', {})

add('markdown', r'''
### 2.3 Learning curves
''', {})

add('code', r'''
import matplotlib.pyplot as plt
with saved_path("training", "history.csv").open(encoding="utf-8") as stream:
    history = list(csv.DictReader(stream))
epochs = [int(row["epoch"]) for row in history]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for key in ("train_loss", "val_loss"):
    axes[0].plot(epochs, [float(row[key]) for row in history], label=key)
for key in ("train_accuracy", "val_accuracy"):
    axes[1].plot(epochs, [float(row[key]) for row in history], label=key)
for ax, title in zip(axes, ("Loss", "Accuracy")):
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
(RUN_DIR / "training").mkdir(exist_ok=True)
fig.savefig(RUN_DIR / "training" / "curves.png", dpi=150)
plt.show()
''', {})

add('markdown', r'''
## 3. PyTorch evaluation
''', {})
add('markdown', '### 3.1 Evaluation and reporting helpers')

add('code', r'''
def evaluate_split(split):
    checkpoint = saved_path("checkpoints", "best.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    report_dir = measurement_dir(f"pytorch/{split}")
    output_path = report_dir / "metrics.json"
    run([sys.executable, "-m", "ru_kws.evaluate", "--checkpoint", checkpoint,
         "--data-root", DATA_ROOT, "--split", split, "--output", output_path], cwd=REPO)
    save_measurement(report_dir, "pytorch_clips", {"checkpoint": checkpoint,
        "manifest": DATA_ROOT / "splits" / f"{split}.jsonl"}, split=split,
        dataset_archive_sha256=dataset_archive_hash)
    report = json.loads(output_path.read_text(encoding="utf-8"))
    report["report_dir"] = str(report_dir)
    return report

def show_report(report):
    import numpy as np
    import matplotlib.pyplot as plt
    print(f"Epoch: {report['epoch']} | split: {report['split']} | N={report['count']}")
    print(f"Accuracy: {report['accuracy']:.4f} | macro F1: {report['macro_f1']:.4f}")
    print(f"{'Class':20s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>8s}")
    for name in report["class_order"]:
        row = report["per_class"][name]
        print(f"{name:20s} {row['precision']:10.4f} {row['recall']:10.4f} {row['f1']:10.4f} {row['support']:8d}")
    cm = np.array(report["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    names = report["class_order"]
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, str(value), ha="center", va="center",
                color="white" if value > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted class"); ax.set_ylabel("True class")
    ax.set_title(f"{report['split']} — best checkpoint, epoch {report['epoch']}")
    fig.colorbar(im, ax=ax); fig.tight_layout()
    fig.savefig(Path(report["report_dir"]) / "confusion.png", dpi=150)
    plt.show()

''', {})
add('markdown', '### 3.2 Validation split')
add('code', r'''
val_report = evaluate_split("val")
show_report(val_report)
''', {})

add('markdown', r'''
### 3.3 Dataset final test
''', {})

add('code', r'''
if RUN_FINAL_TEST:
    test_report = evaluate_split("test")
    show_report(test_report)
else:
    print("Test skipped.")
print("All results on Drive:", RUN_DIR)
''', {})

add('markdown', r'''
## 4. TFLite export
''', {})
add('markdown', '### 4.1 Export dependencies')

add('code', r'''
run([sys.executable, "-m", "pip", "install", "litert-torch", "ai-edge-litert",
     "scipy", "scikit-learn", "tqdm"])
''', {})

add('markdown', r'''
### 4.2 Checkpoint and export paths
''', {})

add('code', r'''
from matplotlib import pyplot as plt

import torch
from torch import nn

# Make the freshly installed project visible in the current notebook kernel.
project_src = str(REPO / "src")
if project_src not in sys.path:
    sys.path.insert(0, project_src)

from ru_kws.audio.frontend import build_frontend
from ru_kws.checkpoint import load_checkpoint
from ru_kws.models.factory import build_model

CHECKPOINT_PATH = saved_path("checkpoints", "best.pt")
EXPORT_DIR = RUN_DIR / "exports" / unique_id()
TFLITE_PATH = (Path(EXISTING_TFLITE_PATH) if EXISTING_TFLITE_PATH else
               EXPORT_DIR / "bcresnet_with_frontend_fp32.tflite")
EXPORT_DIR = TFLITE_PATH.parent

checkpoint = load_checkpoint(CHECKPOINT_PATH)
export_config = checkpoint["config"]
export_labels = checkpoint["labels"]
sample_rate = int(export_config["audio"]["sample_rate"])
window_seconds = float(export_config["audio"]["window_seconds"])
num_samples = round(sample_rate * window_seconds)
num_classes = len(export_labels)

classifier = build_model(export_config, num_classes).cpu().float().eval()
classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
reference_frontend = build_frontend(export_config).cpu().float().eval()
print("Checkpoint:", CHECKPOINT_PATH)
print("Epoch:", checkpoint.get("epoch"))
print("Input shape:", (1, num_samples), "sample rate:", sample_rate)
print("Class order:", sorted(export_labels, key=export_labels.get))
''', {})

add('markdown', r'''
### 4.3 Exportable frontend
''', {})

add('code', r'''
import torch.nn.functional as F

class ExportableLogMelFrontend(nn.Module):
    def __init__(self, source_frontend):
        super().__init__()

        mel = source_frontend.mel
        spec = mel.spectrogram

        # Reject settings that this export implementation does not reproduce.
        assert spec.power == 2.0, "Expected a power spectrogram"
        assert spec.normalized is False
        assert spec.onesided is True
        assert spec.center is True
        assert spec.pad_mode == "reflect"
        assert spec.pad == 0

        self.n_fft = spec.n_fft
        self.hop_length = spec.hop_length
        self.n_freqs = self.n_fft // 2 + 1
        self.log_eps = float(getattr(source_frontend, "log_eps", 1e-6))

        # Compute constants in float64, then store float32 filters.
        window = spec.window.detach().cpu().double()
        missing = self.n_fft - window.numel()
        window = F.pad(
            window,
            (missing // 2, missing - missing // 2),
        )

        frequencies = torch.arange(self.n_freqs, dtype=torch.float64)
        samples = torch.arange(self.n_fft, dtype=torch.float64)
        angles = (
            2 * torch.pi
            * frequencies[:, None]
            * samples[None, :]
            / self.n_fft
        )

        real_filters = torch.cos(angles) * window
        imag_filters = -torch.sin(angles) * window

        filters = torch.cat(
            [real_filters, imag_filters], dim=0
        ).unsqueeze(1).float()

        self.register_buffer("dft_filters", filters)

        # [frequency, mel] -> [mel, frequency, 1]
        mel_filters = (
            mel.mel_scale.fb.detach().cpu().float()
            .T.contiguous().unsqueeze(-1)
        )
        self.register_buffer("mel_filters", mel_filters)

    def forward(self, waveforms):
        x = waveforms.unsqueeze(1)  # [B, 1, N]
        x = F.pad(
            x,
            (self.n_fft // 2, self.n_fft // 2),
            mode="reflect",
        )

        spectrum = F.conv1d(
            x, self.dft_filters, stride=self.hop_length
        )
        real = spectrum[:, :self.n_freqs, :]
        imag = spectrum[:, self.n_freqs:, :]

        power = real.square() + imag.square()
        mel = F.conv1d(power, self.mel_filters)

        return torch.log(mel + self.log_eps).unsqueeze(1)



export_frontend = ExportableLogMelFrontend(reference_frontend).eval()
''', {})

add('markdown', r'''
### 4.4 Frontend equivalence
''', {})

add('code', r'''
generator = torch.Generator().manual_seed(42)
time_axis = torch.arange(num_samples, dtype=torch.float32) / sample_rate
checks = {
    "silence": torch.zeros(1, num_samples),
    "noise": torch.randn(2, num_samples, generator=generator) * 0.01,
    "tone": (0.1 * torch.sin(2 * torch.pi * 440 * time_axis)).unsqueeze(0),
}

with torch.inference_mode():
    for name, audio in checks.items():
        expected_features = reference_frontend(audio)
        actual_features = export_frontend(audio)
        torch.testing.assert_close(actual_features, expected_features, atol=1e-3, rtol=1e-3)
        expected_logits = classifier(expected_features)
        actual_logits = classifier(actual_features)
        assert expected_logits.shape == (len(audio), num_classes)
        assert torch.isfinite(actual_logits).all()
        torch.testing.assert_close(actual_logits, expected_logits, atol=1e-3, rtol=1e-3)
        print(name, "max feature difference:", (actual_features - expected_features).abs().max().item())
print("Frontend equivalence checks passed.")
''', {})

add('markdown', r'''
### 4.5 Export the waveform model
''', {})

add('code', r'''
import litert_torch

class AudioCommandModel(nn.Module):
    def __init__(self, frontend, classifier):
        super().__init__()
        self.frontend = frontend
        self.classifier = classifier

    def forward(self, waveforms):
        return self.classifier(self.frontend(waveforms))

if TFLITE_PATH.exists():
    raise FileExistsError(f"Choose a new TFLITE_PATH: {TFLITE_PATH}")
export_model = AudioCommandModel(export_frontend, classifier).cpu().float().eval()
example_audio = checks["noise"][:1].contiguous()
with torch.no_grad():
    edge_model = litert_torch.convert(export_model, (example_audio,))
EXPORT_DIR.mkdir(parents=True, exist_ok=True)
edge_model.export(str(TFLITE_PATH))
save_measurement(EXPORT_DIR, "tflite_export", {"checkpoint": CHECKPOINT_PATH, "model": TFLITE_PATH},
                 sample_rate=sample_rate, window_seconds=window_seconds, labels=export_labels)
print("Saved:", TFLITE_PATH)
''', {})

add('markdown', r'''
### 4.6 Verify exported inference
''', {})

add('code', r'''
import numpy as np
from ai_edge_litert.interpreter import Interpreter

interpreter = Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()
inputs = interpreter.get_input_details()
outputs = interpreter.get_output_details()
assert len(inputs) == len(outputs) == 1
input_info, output_info = inputs[0], outputs[0]
assert tuple(input_info["shape"]) == (1, num_samples)
assert tuple(output_info["shape"]) == (1, num_classes)
assert input_info["dtype"] == output_info["dtype"] == np.float32

equivalence = []
with torch.inference_mode():
    for name, batch in checks.items():
        for i in range(len(batch)):
            audio = batch[i:i + 1].contiguous()
            expected = classifier(reference_frontend(audio)).numpy()
            interpreter.set_tensor(input_info["index"], audio.numpy())
            interpreter.invoke()
            actual = interpreter.get_tensor(output_info["index"])
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual, expected, atol=1e-3, rtol=1e-3)
            difference = float(np.max(np.abs(expected - actual)))
            equivalence.append({"input": name, "item": i, "max_logit_difference": difference})
            print(f"{name}[{i}]: max logit difference={difference:.6g}")
EQUIVALENCE_DIR = measurement_dir("equivalence")
save_measurement(EQUIVALENCE_DIR, "tflite_equivalence",
    {"checkpoint": CHECKPOINT_PATH, "model": TFLITE_PATH}, atol=1e-3, rtol=1e-3)
(EQUIVALENCE_DIR / "metrics.json").write_text(
    json.dumps({"passed": True, "checks": equivalence}, indent=2, allow_nan=False), encoding="utf-8")
print("Saved TFLite model equivalence checks passed.")
''', {})

add('markdown', '## 5. Clip evaluation')
add('markdown', '### 5.1 Evaluation model and report helpers')
add('code', r'''
import pandas as pd
from ru_kws.checkpoint import load_checkpoint

CHECKPOINT_PATH = saved_path("checkpoints", "best.pt")
evaluation_checkpoint = load_checkpoint(CHECKPOINT_PATH)
evaluation_labels = evaluation_checkpoint["labels"]
evaluation_audio = evaluation_checkpoint["config"]["audio"]
sample_rate = int(evaluation_audio["sample_rate"])
window_seconds = float(evaluation_audio["window_seconds"])
if evaluation_labels != json.loads((DATA_ROOT / "labels.json").read_text(encoding="utf-8")):
    raise ValueError("Dataset labels differ from checkpoint labels")

if EXISTING_TFLITE_PATH:
    ACTIVE_MODEL = Path(EXISTING_TFLITE_PATH)
elif "TFLITE_PATH" in globals() and Path(TFLITE_PATH).is_file():
    ACTIVE_MODEL = Path(TFLITE_PATH)
else:
    exports = sorted((RUN_DIR / "exports").glob("*/bcresnet_with_frontend_fp32.tflite"))
    if len(exports) != 1:
        raise ValueError(f"Set EXISTING_TFLITE_PATH; available exports: {exports}")
    ACTIVE_MODEL = exports[0]
if not ACTIVE_MODEL.is_file():
    raise FileNotFoundError(ACTIVE_MODEL)
EVAL_LABELS = Path("/content/evaluation_labels.json")
EVAL_LABELS.write_text(json.dumps(evaluation_labels), encoding="utf-8")

def evaluate_clip_manifest(name, manifest):
    output = measurement_dir("clips/" + name)
    run([sys.executable, REPO / "scripts/evaluate_tflite.py",
         "--model", ACTIVE_MODEL, "--dataset-root", DATA_ROOT,
         "--manifest", manifest, "--labels", EVAL_LABELS,
         "--split", "val" if name == "validation" else "test",
         "--sample-rate", sample_rate, "--window-seconds", window_seconds,
         "--long-commands", "error", "--label-smoothing", "0", "--output-dir", output], cwd=REPO)
    save_measurement(output, "clip_evaluation", {"model": ACTIVE_MODEL,
        "checkpoint": CHECKPOINT_PATH, "manifest": manifest}, view=name)
    return output

def show_clip_report(output):
    report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    print("Evaluated:", report["evaluated"], "of", report["manifest_records"])
    print("Accuracy:", round(report["accuracy"], 4))
    display(pd.DataFrame(report["classification_report"]).T)
    display(pd.read_csv(output / "confusion_matrix.csv", index_col=0))
    print("Predictions:", output / "predictions.csv")

print("Model:", ACTIVE_MODEL)
''')
add('markdown', '### 5.2 Validation clips')
add('code', r'''
VALIDATION_REPORT = evaluate_clip_manifest("validation", DATA_ROOT / "splits" / "val.jsonl")
show_clip_report(VALIDATION_REPORT)
''')
add('markdown', '### 5.3 Synthetic test clips')
add('code', r'''
if RUN_FINAL_TEST:
    SYNTHETIC_REPORT = evaluate_clip_manifest("synthetic_test", DATA_ROOT / "evaluation" / "synthetic_test.jsonl")
    show_clip_report(SYNTHETIC_REPORT)
else:
    print("Final tests disabled; set RUN_FINAL_TEST=True after fixing model settings.")
''')
add('markdown', '### 5.4 Vanya held-out test clips')
add('code', r'''
if RUN_VANYA_TEST:
    VANYA_REPORT = evaluate_clip_manifest("vanya_test", DATA_ROOT / "evaluation" / "vanya_test.jsonl")
    show_clip_report(VANYA_REPORT)
else:
    print("Vanya test disabled; set RUN_VANYA_TEST=True.")
''')
add('markdown', '## 6. Continuous command tests')
add('markdown', '### 6.1 Shared decoder and test configuration')
add('code', r'''
# WAV and annotation paths refer directly to Drive files. Add any number of sessions.
# Each annotation JSON must use the original WAV frame clock.
CONTINUOUS_CASES = [
    # {"name": "sergey_session", "wav": Path("/content/drive/MyDrive/.../continuous.wav"),
    #  "annotations": Path("/content/drive/MyDrive/.../continuous.reviewed.json")},
    # {"name": "vanya_session", "wav": Path("/content/drive/MyDrive/.../continuous.wav"),
    #  "annotations": Path("/content/drive/MyDrive/.../continuous.reviewed.json")},
]
THRESHOLD = 0.5
HOP_SECONDS = 0.1
MIN_CONSECUTIVE = 2
RELEASE_SECONDS = 0.3
COOLDOWN_SECONDS = 1.0
EARLY_TOLERANCE = 0.0
LATE_TOLERANCE = 1.0
ALLOW_DRAFT = False

def continuous_command(model, wav, output, annotations=None, negative=False):
    command = [sys.executable, REPO / "scripts/evaluate_continuous_tflite.py",
        "--model", model, "--wav", wav, "--labels", EVAL_LABELS,
        "--sample-rate", sample_rate, "--window-seconds", window_seconds,
        "--hop-seconds", HOP_SECONDS, "--threshold", THRESHOLD,
        "--min-consecutive", MIN_CONSECUTIVE, "--release-seconds", RELEASE_SECONDS,
        "--cooldown-seconds", COOLDOWN_SECONDS, "--early-tolerance", EARLY_TOLERANCE,
        "--late-tolerance", LATE_TOLERANCE, "--resample", "--mono", "mean",
        "--output-dir", output]
    if negative:
        command.append("--negative-only")
    else:
        command += ["--annotations", annotations]
        if ALLOW_DRAFT: command.append("--allow-draft")
    return command

def run_continuous_case(case, model, output, negative=False):
    name = case["name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"Invalid case name: {name!r}")
    wav = Path(case["wav"])
    annotations = None if negative else Path(case["annotations"])
    if not wav.is_file(): raise FileNotFoundError(wav)
    if annotations is not None and not annotations.is_file(): raise FileNotFoundError(annotations)
    run(continuous_command(model, wav, output, annotations, negative), cwd=REPO)
    inputs = {"model": model, "wav": wav, "labels": EVAL_LABELS}
    if annotations is not None: inputs["annotations"] = annotations
    save_measurement(output, "negative_audio" if negative else "continuous_commands", inputs,
        threshold=THRESHOLD, hop_seconds=HOP_SECONDS, min_consecutive=MIN_CONSECUTIVE,
        release_seconds=RELEASE_SECONDS, cooldown_seconds=COOLDOWN_SECONDS)
    return json.loads((output / "metrics.json").read_text(encoding="utf-8"))

def continuous_summary(reports):
    rows = []
    for name, directory in reports.items():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"case": name, **{k: metrics[k] for k in ["duration_seconds", "tp", "fp", "fn",
            "precision", "recall", "f1", "false_positives_per_hour", "median_latency_seconds", "provisional"]}})
    return pd.DataFrame(rows)
''')
add('markdown', '### 6.2 Run annotated sessions')
add('code', r'''
CONTINUOUS_REPORTS = {}
if RUN_CONTINUOUS_TESTS:
    if not CONTINUOUS_CASES: raise ValueError("Configure CONTINUOUS_CASES before running tests")
    if len({c["name"] for c in CONTINUOUS_CASES}) != len(CONTINUOUS_CASES):
        raise ValueError("Continuous case names must be unique")
    for case in CONTINUOUS_CASES:
        output = measurement_dir("continuous/" + case["name"])
        run_continuous_case(case, ACTIVE_MODEL, output)
        CONTINUOUS_REPORTS[case["name"]] = output
else:
    print("Continuous tests disabled; set RUN_CONTINUOUS_TESTS=True.")
''')
add('markdown', '### 6.3 Event metrics and detections')
add('code', r'''
if CONTINUOUS_REPORTS:
    display(continuous_summary(CONTINUOUS_REPORTS))
    for name, output in CONTINUOUS_REPORTS.items():
        print(name, output)
        display(pd.read_csv(output / "events.csv"))
        display(pd.read_csv(output / "detections.csv"))
else:
    print("No continuous reports yet.")
''')
add('markdown', '### 6.4 Probability timelines')
add('code', r'''
import matplotlib.pyplot as plt
for name, output in CONTINUOUS_REPORTS.items():
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    windows = pd.read_csv(output / "windows.csv")
    events = pd.read_csv(output / "events.csv")
    detections = pd.read_csv(output / "detections.csv")
    commands = list(metrics["per_class"])
    fig, axes = plt.subplots(len(commands), 1, figsize=(16, 2.2*len(commands)), sharex=True, squeeze=False)
    for ax, label in zip(axes[:, 0], commands):
        ax.plot(windows.end_seconds, windows["p_" + label], label="Probability")
        ax.axhline(metrics["config"]["threshold"], color="gray", linestyle="--")
        for event in events[events.label == label].itertuples():
            ax.axvspan(event.start_seconds, event.end_seconds, color="green", alpha=.2)
        selected = detections[detections.label == label]
        ax.scatter(selected.time_seconds, selected.confidence, color="red", marker="x")
        ax.set(title=label, ylim=(0, 1))
    axes[-1, 0].set_xlabel("Recording time (seconds)")
    fig.suptitle(name + (" [provisional]" if metrics["provisional"] else ""))
    fig.tight_layout(); fig.savefig(output / 'timeline.png', dpi=150); plt.show()
''')
add('markdown', '## 7. Podcast false-positive test')
add('markdown', '### 7.1 Negative-audio input')
add('code', r'''
PODCAST_WAV = None  # Path("/content/drive/MyDrive/russian_commands/podcast.wav")
PODCAST_IS_NEGATIVE = False  # Set True only when treating this recording as command-free.
PODCAST_REPORT = None
if RUN_PODCAST_TEST and (PODCAST_WAV is None or not PODCAST_IS_NEGATIVE):
    raise ValueError("Set PODCAST_WAV and PODCAST_IS_NEGATIVE=True. All emitted commands will count as FP.")
''')
add('markdown', '### 7.2 Run the negative recording')
add('code', r'''
if RUN_PODCAST_TEST:
    PODCAST_REPORT = measurement_dir("podcast")
    run_continuous_case({"name": "podcast", "wav": PODCAST_WAV}, ACTIVE_MODEL, PODCAST_REPORT, negative=True)
else:
    print("Podcast test disabled; set RUN_PODCAST_TEST=True.")
''')
add('markdown', '### 7.3 False positives per hour and event list')
add('code', r'''
if PODCAST_REPORT:
    metrics = json.loads((PODCAST_REPORT / "metrics.json").read_text(encoding="utf-8"))
    print("Command-free assumption:", metrics["annotation_status"])
    print("Hours:", metrics["duration_seconds"] / 3600, "FP:", metrics["fp"],
          "FP/hour:", metrics["false_positives_per_hour"])
    display(pd.DataFrame(metrics["per_class"]).T[["fp", "unmatched_detections_per_hour_full_recording"]])
    display(pd.read_csv(PODCAST_REPORT / "detections.csv"))
else:
    print("No podcast report yet.")
''')
add('markdown', '## 8. Baseline comparison')
add('markdown', '### 8.1 Baseline model and checkpoint')
add('code', r'''
BASELINE_RUN_NAME = "bcresnet_run_005"
BASELINE_MODEL = Path("/content/drive/MyDrive/russian_commands/runs/bcresnet_run_005/exports/20260913T160929_482538Z_9cf8c22a/bcresnet_with_frontend_fp32.tflite")
BASELINE_RUN = RUNS_ROOT / BASELINE_RUN_NAME
BASELINE_CHECKPOINT = BASELINE_RUN / "checkpoints" / "best.pt"
if not BASELINE_CHECKPOINT.is_file(): BASELINE_CHECKPOINT = BASELINE_RUN / "best.pt"
if RUN_BASELINE_COMPARISON:
    baseline = load_checkpoint(BASELINE_CHECKPOINT)
    if baseline["labels"] != evaluation_labels or baseline["config"]["audio"] != evaluation_audio:
        raise ValueError("Baseline and candidate label/audio configurations differ")
    if not BASELINE_MODEL.is_file(): raise FileNotFoundError(BASELINE_MODEL)
    if file_sha256(BASELINE_MODEL) == file_sha256(ACTIVE_MODEL):
        raise ValueError("Baseline and candidate models have identical contents")
''')
add('markdown', '### 8.2 Paired synthetic and Vanya clip tests')
add('code', r'''
COMPARISON_DIR = None
if RUN_BASELINE_COMPARISON:
    COMPARISON_DIR = RUNS_ROOT / "model_comparisons" / unique_id()
    run([sys.executable, REPO / "scripts/compare_tflite_models.py",
         "--old-model", BASELINE_MODEL, "--new-model", ACTIVE_MODEL,
         "--old-checkpoint", BASELINE_CHECKPOINT, "--new-checkpoint", CHECKPOINT_PATH,
         "--new-dataset-root", DATA_ROOT, "--window-seconds", window_seconds,
         "--output-dir", COMPARISON_DIR], cwd=REPO)
else:
    print("Baseline comparison disabled; set RUN_BASELINE_COMPARISON=True.")
''')
add('markdown', '### 8.3 Paired accuracy and class changes')
add('code', r'''
if COMPARISON_DIR:
    comparison = json.loads((COMPARISON_DIR / "comparison.json").read_text(encoding="utf-8"))
    for view, info in comparison["datasets"].items():
        print(view, "N:", info["compared_count"], "supported-class macro F1:", info["supported_macro_f1"])
        display(pd.read_csv(COMPARISON_DIR / view / "comparison_by_class.csv"))
    print("Paired predictions:", COMPARISON_DIR)
''')
add('markdown', '### 8.4 Continuous sessions versus baseline')
add('code', r'''
BASELINE_CONTINUOUS_ROWS = []
if RUN_BASELINE_COMPARISON and RUN_CONTINUOUS_TESTS:
    for case in CONTINUOUS_CASES:
        for role, model in [("baseline", BASELINE_MODEL), ("candidate", ACTIVE_MODEL)]:
            output = measurement_dir("continuous_comparison/" + case["name"] + "/" + role)
            metrics = run_continuous_case(case, model, output)
            BASELINE_CONTINUOUS_ROWS.append({"case": case["name"], "model": role, "report": str(output),
                **{k: metrics[k] for k in ["tp", "fp", "fn", "recall", "f1", "false_positives_per_hour", "median_latency_seconds"]}})
    display(pd.DataFrame(BASELINE_CONTINUOUS_ROWS))
''')
add('markdown', '### 8.5 Podcast versus baseline')
add('code', r'''
if RUN_BASELINE_COMPARISON and RUN_PODCAST_TEST:
    rows = []
    for role, model in [("baseline", BASELINE_MODEL), ("candidate", ACTIVE_MODEL)]:
        output = measurement_dir("podcast_comparison/" + role)
        metrics = run_continuous_case({"name": "podcast", "wav": PODCAST_WAV}, model, output, negative=True)
        rows.append({"model": role, "hours": metrics["duration_seconds"]/3600,
                     "fp": metrics["fp"], "fp_per_hour": metrics["false_positives_per_hour"], "report": str(output)})
    display(pd.DataFrame(rows))
''')

notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
    'language_info': {'name': 'python'}, 'colab': {'name': 'ru_kws_training.ipynb', 'provenance': []}},
    nbformat=4, nbformat_minor=5)
destination = ROOT / "notebooks/ru_kws_training.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
