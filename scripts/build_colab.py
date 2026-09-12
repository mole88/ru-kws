"""Build the Colab notebook for training, TFLite export, and evaluation."""
import json
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
cells = []

def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "id": f"cell-{len(cells):02d}",
            "source": dedent(source).strip() + "\n"}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)

add('markdown', r'''
# Russian KWS: training, TFLite export, and evaluation

Run the setup and dataset cells first. Training, PyTorch evaluation, TFLite export,
and TFLite dataset validation are separate stages. To export an existing checkpoint,
skip the training and PyTorch report cells and start at section 10 after setup.
All export settings come from the checkpoint; the exported model accepts raw audio.
''')

add('code', r'''
from pathlib import Path
import sys, os, json, subprocess, hashlib, zipfile, tempfile, stat, csv
import torch

REPO_URL = "https://github.com/mole88/ru-kws.git"
REPO_REF = "dev"  # Use a commit SHA for reproducibility.
DATASET_ZIP = Path("/content/drive/MyDrive/russian_commands/russian_commands_v001_clean_manual.zip")

RUN_NAME = "bcresnet_run_003"
RUNS_ROOT = Path("/content/drive/MyDrive/russian_commands/runs")
WINDOW_SECONDS = 3.0
BASE_C = 64                   # 8/12/16/24/48/64; 64 - BC-ResNet-8
BATCH_SIZE = 64
MAX_EPOCHS = 60
LEARNING_RATE = 3e-4
REQUIRE_GPU = torch.cuda.is_available()

RUN_FINAL_TEST = True

RUN_DIR = RUNS_ROOT / RUN_NAME
os.environ["PYTHONUNBUFFERED"] = "1" # for realtime logs output
os.environ["MPLBACKEND"] = "Agg" # for stable working with graphics libs
print("Run path:", RUN_DIR)
''')

add('markdown', r'''
## 1. Connect Google Drive
''')

add('code', r'''
from google.colab import drive
drive.mount("/content/drive")
''')

add('markdown', r'''
## 2. Download project code
Clone the GitHub repository and record the downloaded revision SHA.
''')

add('code', r'''
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
source_info = {"mode": "github", "url": REPO_URL, "requested_ref": REPO_REF, "revision": revision}
print("Project:", REPO)
print(json.dumps(source_info, ensure_ascii=False, indent=2))
''')

add('markdown', r'''
## 3. Install and verify the environment

Use the PyTorch/torchaudio packages provided by Colab; this cell does not reinstall them.
WAV loading uses SoundFile and does not require `torchcodec`.
If the GPU check fails, enable a GPU and restart the runtime.
''')

add('code', r'''
run([sys.executable, "-m", "pip", "install", "numpy>=1.24", "soundfile>=0.12", "PyYAML>=6", "matplotlib>=3.6", "packaging"])
run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", REPO])
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
''')

add('markdown', r'''
## 4. Local dataset copy

The root containing `labels.json` is detected automatically, including nested archive directories.
Repeated extraction of the same archive in the same runtime is skipped.
''')

add('code', r'''
unpacked, dataset_archive_hash = extract_cached(DATASET_ZIP, Path("/content/ru_kws_data"))
DATA_ROOT = find_root(unpacked, "labels.json", ["splits/train.jsonl", "splits/val.jsonl", "splits/test.jsonl"])
print("Dataset:", DATA_ROOT)
print("Labels:", json.loads((DATA_ROOT / "labels.json").read_text(encoding="utf-8")))
''')

add('markdown', r'''
## 5. Configuration and data validation

Commands longer than `WINDOW_SECONDS` are automatically excluded from train/val/test.
Source WAV files and manifests remain unchanged. The number of excluded recordings is reported
during validation and loading; metrics use the remaining recordings.
Long unknown/background recordings are retained and cropped to the window during loading.
The split validator checks format and split overlap; it does not evaluate model quality.
''')

add('code', r'''
import yaml
config = yaml.safe_load((REPO / "configs/baseline.yaml").read_text(encoding="utf-8"))
config["audio"]["window_seconds"] = WINDOW_SECONDS
config["model"]["base_c"] = BASE_C
config["training"].update(batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS,
                          learning_rate=LEARNING_RATE, num_workers=0)
CONFIG_PATH = Path("/content/ru_kws_colab_config.yaml")
CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(CONFIG_PATH.read_text())
run([sys.executable, "-m", "ru_kws.data.validate", "--data-root", DATA_ROOT,
     "--window-seconds", WINDOW_SECONDS], cwd=REPO)
''')

add('markdown', r'''
## 6. Training

Starting an epoch can take time; loss/accuracy are reported after each epoch.
`best.pt` is selected by validation loss; `last.pt` is saved after each epoch.
Runs cannot reuse a nonempty directory. Change `RUN_NAME` for a new experiment.
Skip this cell to evaluate an existing checkpoint without training.
''')

add('code', r'''
if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"Run directory already exists: {RUN_DIR}.")
run([sys.executable, "-m", "ru_kws.train", "--config", CONFIG_PATH,
     "--data-root", DATA_ROOT, "--run-dir", RUN_DIR, "--device", "auto"], cwd=REPO)
(RUN_DIR / "colab_source.json").write_text(json.dumps({
    "source": source_info, "dataset_archive": str(DATASET_ZIP),
    "dataset_archive_sha256": dataset_archive_hash,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Best checkpoint:", RUN_DIR / "best.pt")
''')

add('markdown', r'''
## 7. Training plots
Generated from the saved `history.csv`, without retraining.
''')

add('code', r'''
import matplotlib.pyplot as plt
with (RUN_DIR / "history.csv").open(encoding="utf-8") as stream:
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
(RUN_DIR / "reports").mkdir(exist_ok=True)
fig.savefig(RUN_DIR / "reports" / "training.png", dpi=150)
plt.show()
''')

add('markdown', r'''
## 8. Evaluate the best checkpoint on validation

The CLI loads `best.pt`, its frontend configuration, and its label map.
These are clip classification metrics; they do not measure false positives per hour or streaming latency.
''')

add('code', r'''
def evaluate_split(split):
    checkpoint = RUN_DIR / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    output_path = RUN_DIR / "reports" / f"{split}.json"
    run([sys.executable, "-m", "ru_kws.evaluate", "--checkpoint", checkpoint,
         "--data-root", DATA_ROOT, "--split", split, "--output", output_path], cwd=REPO)
    return json.loads(output_path.read_text(encoding="utf-8"))

def show_report(report):
    import numpy as np
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
    fig.savefig(RUN_DIR / "reports" / f"{report['split']}_confusion.png", dpi=150)
    plt.show()

val_report = evaluate_split("val")
show_report(val_report)
''')

add('markdown', r'''
## 9. Final PyTorch test (optional)

Run this only after fixing the model and tuning settings on validation.
The configuration above currently enables `RUN_FINAL_TEST`; set it to False to skip this stage.
''')

add('code', r'''
if RUN_FINAL_TEST:
    test_report = evaluate_split("test")
    show_report(test_report)
else:
    print("Test skipped.")
print("All results on Drive:", RUN_DIR)
''')

add('markdown', r'''
## 10. Install TFLite export and evaluation dependencies

Run once in the Colab runtime before export. If pip replaces packages already imported
(especially PyTorch), restart the runtime and rerun setup, dataset loading, and sections
11 onward. An existing checkpoint can be exported without retraining.
Conversion follows the [LiteRT Torch API](https://github.com/google-ai-edge/litert-torch).
''')

add('code', r'''
run([sys.executable, "-m", "pip", "install", "litert-torch", "ai-edge-litert",
     "scipy", "scikit-learn", "tqdm"])
''')

add('markdown', r'''
## 11. Configure export and load the checkpoint

Use the model and frontend from this project, with architecture, class order, sample
rate, and window length restored from the checkpoint. Change `CHECKPOINT_PATH` here
to export a different run. Export artifacts and evaluation reports are saved to Drive.
''')

add('code', r'''
import torch
from torch import nn

# Make the freshly installed project visible in the current notebook kernel.
project_src = str(REPO / "src")
if project_src not in sys.path:
    sys.path.insert(0, project_src)

from ru_kws.audio.frontend import build_frontend
from ru_kws.checkpoint import load_checkpoint
from ru_kws.models.factory import build_model

CHECKPOINT_PATH = RUN_DIR / "best.pt"
EXPORT_DIR = CHECKPOINT_PATH.parent / "export"
TFLITE_PATH = EXPORT_DIR / "bcresnet_with_frontend_fp32.tflite"
TFLITE_SPLIT = "val"
TFLITE_REPORT_DIR = EXPORT_DIR / f"{TFLITE_SPLIT}_001"

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
''')

add('markdown', r'''
## 12. Build the export-compatible log-mel frontend

Replace the complex STFT with fixed real-valued convolution filters. Windowing,
reflection padding, mel filters, and the logarithm use the training frontend's
parameters. This implementation supports the settings checked below; unsupported
settings fail explicitly. This is a full-window frontend, not a streaming frontend.
''')

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
''')

add('markdown', r'''
## 13. Check frontend and classifier equivalence

Use deterministic silence, noise, and sine-wave inputs. These numerical checks
detect conversion regressions; real recording quality is measured in section 16.
Do not export if the comparisons fail.
''')

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
''')

add('markdown', r'''
## 14. Export one waveform-to-logits TFLite model

The model includes the frontend and classifier. Input: float32 `[1, num_samples]`;
output: float32 logits `[1, num_classes]`. This is FP32 export, not quantization.
An existing export is not overwritten; choose a new path to keep multiple versions.
''')

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
print("Saved:", TFLITE_PATH)
''')

add('markdown', r'''
## 15. Verify the saved TFLite model against PyTorch

Load the actual exported file and compare its logits against the original training
frontend and classifier. Check both input and output contracts before evaluation.
''')

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
            print(f"{name}[{i}]: max logit difference={np.max(np.abs(expected - actual)):.6g}")
print("Saved TFLite model equivalence checks passed.")
''')

add('markdown', r'''
## 16. Evaluate TFLite on the dataset

Use the dataset extracted in section 4. Check its class mapping against the checkpoint.
The frontend is already inside the model, so no external frontend is passed to the
CLI. Use a new report directory for each evaluation. These are clip metrics, not
false positives per hour or streaming latency.
''')

add('code', r'''
dataset_labels = json.loads((DATA_ROOT / "labels.json").read_text(encoding="utf-8"))
if dataset_labels != export_labels:
    raise ValueError("Dataset label mapping differs from the checkpoint")
if TFLITE_SPLIT not in {"val", "test"}:
    raise ValueError("Choose val or test for TFLite evaluation")
run([sys.executable, REPO / "scripts/evaluate_tflite.py",
     "--model", TFLITE_PATH, "--dataset-root", DATA_ROOT,
     "--split", TFLITE_SPLIT, "--sample-rate", sample_rate,
     "--window-seconds", window_seconds, "--long-commands", "skip",
     "--label-smoothing", export_config["training"].get("label_smoothing", 0.0),
     "--output-dir", TFLITE_REPORT_DIR], cwd=REPO)
print("TFLite reports:", TFLITE_REPORT_DIR)
''')

notebook = dict(cells=cells, metadata={'kernelspec': {'display_name': 'Python 3', 'name': 'python3'}, 'language_info': {'name': 'python'}, 'accelerator': 'GPU', 'colab': {'name': 'ru_kws_training.ipynb', 'provenance': [], 'gpuType': 'T4'}}, nbformat=4, nbformat_minor=5)
destination = ROOT / "notebooks/ru_kws_training.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
