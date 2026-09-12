"""Build the thin Colab launcher (training stays in ru_kws)."""
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


add("markdown", '''
# ru-kws: training BC-ResNet in Colab

The notebook runs project modules; training code is not duplicated in cells.

1. Enable a GPU in the Colab runtime settings.
2. Check the paths in the configuration below.
3. Run the cells through the validation report.

Code is downloaded from [mole88/ru-kws](https://github.com/mole88/ru-kws), branch `master`.
Upload `russian_commands_v001_clean_manual.zip` to the root of My Drive.
Set `REPO_REF` to a commit SHA to reproduce an experiment.

WAV files are extracted to local Colab storage; checkpoints and reports are saved to Drive.
Baseline: 3 seconds, BC-ResNet-8, random audio placement, no noise mixing or
SpecAugment. Completed epochs are saved if Colab disconnects, but **resume is not supported yet**.
Do not treat a new training run as a continuation of a previous run.
''')
add("code", '''
from pathlib import Path
import sys, os, json, subprocess, hashlib, zipfile, tempfile, stat, csv

REPO_URL = "https://github.com/mole88/ru-kws.git"
REPO_REF = "master"       # Branch, tag, or commit SHA; a SHA improves reproducibility
DATASET_ZIP = Path("/content/drive/MyDrive/russian_commands_v001_clean_manual.zip")

RUN_NAME = "bcresnet_run_002"  # Use a new name for each training run
RUNS_ROOT = Path("/content/drive/MyDrive/russian_commands/runs")
WINDOW_SECONDS = 3.0
BASE_C = 64                   # 8/12/16/24/48/64; 64 corresponds to BC-ResNet-8
BATCH_SIZE = 32
MAX_EPOCHS = 30
LEARNING_RATE = 3e-4
REQUIRE_GPU = True            # Set False only for an intentional CPU run

# Final test evaluation is disabled to avoid using it for tuning.
RUN_FINAL_TEST = False

if not RUN_NAME or Path(RUN_NAME).name != RUN_NAME or RUN_NAME in {".", ".."}:
    raise ValueError("RUN_NAME must be a directory name without a path")
RUN_DIR = RUNS_ROOT / RUN_NAME
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["MPLBACKEND"] = "Agg"
print("Results:", RUN_DIR)
''')
add("markdown", "## 1. Connect Google Drive")
add("code", '''
from google.colab import drive
drive.mount("/content/drive")
''')
add("markdown", "## 2. Download project code\nClone the GitHub repository and record the downloaded revision SHA.")
add("code", '''
def run(command, cwd=None):
    print("Running:", " ".join(map(str, command)), flush=True)
    with subprocess.Popen(list(map(str, command)), cwd=cwd,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(f"Command exited with code {process.returncode}; see the error above")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def extract_cached(archive, parent):
    if not archive.is_file():
        raise FileNotFoundError(f"Upload the archive to Drive or correct the path: {archive}")
    digest = file_sha256(archive)
    target = parent / digest
    marker = target / ".extraction_complete"
    if marker.exists():
        print("Reusing the extracted archive:", target)
        return target, digest
    target.mkdir(parents=True, exist_ok=True)
    root = target.resolve()
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for member in members:
            destination = (root / member.filename).resolve()
            if not destination.is_relative_to(root) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Invalid path in ZIP: {member.filename}")
        print(f"Extracting {len(members)} files...", flush=True)
        zf.extractall(target)
    marker.write_text(digest)
    return target, digest

def find_root(parent, marker, required):
    candidates = [p.parent for p in parent.rglob(marker)
                  if "__MACOSX" not in p.parts and all((p.parent / item).exists() for item in required)]
    if len(candidates) != 1:
        raise ValueError(f"Expected one directory containing {marker}, found: {candidates}")
    return candidates[0]

if not REPO_URL.startswith("https://github.com/"):
    raise ValueError("Set REPO_URL to an HTTPS GitHub repository URL")
REPO = Path(tempfile.mkdtemp(prefix="ru-kws-", dir="/content"))
run(["git", "clone", "--filter=blob:none", "--no-checkout", REPO_URL, REPO])
run(["git", "fetch", "--depth", "1", "origin", REPO_REF], cwd=REPO)
run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO)
revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
source_info = {"mode": "github", "url": REPO_URL, "requested_ref": REPO_REF, "revision": revision}
print("Project:", REPO)
print(json.dumps(source_info, ensure_ascii=False, indent=2))
''')
add("markdown", '''
## 3. Install and verify the environment

Use the PyTorch/torchaudio packages provided by Colab; this cell does not reinstall them.
WAV loading uses SoundFile and does not require `torchcodec`.
If the GPU check fails, enable a GPU and restart the runtime.
''')
add("code", '''
run([sys.executable, "-m", "pip", "install", "numpy>=1.24", "soundfile>=0.12", "PyYAML>=6", "matplotlib>=3.6", "packaging"])
run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", REPO])
probe = """
import sys, json, torch, torchaudio
import ru_kws.data.validate, ru_kws.train, ru_kws.evaluate
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
run([sys.executable, "-c", "REQUIRE_GPU = " + repr(REQUIRE_GPU) + "\\n" + probe])
''')
add("markdown", '''
## 4. Local dataset copy

The root containing `labels.json` is detected automatically, including nested archive directories.
Repeated extraction of the same archive in the same runtime is skipped.
''')
add("code", '''
unpacked, dataset_archive_hash = extract_cached(DATASET_ZIP, Path("/content/ru_kws_data"))
DATA_ROOT = find_root(unpacked, "labels.json", ["splits/train.jsonl", "splits/val.jsonl", "splits/test.jsonl"])
print("Dataset:", DATA_ROOT)
print("Labels:", json.loads((DATA_ROOT / "labels.json").read_text(encoding="utf-8")))
''')
add("markdown", '''
## 5. Configuration and data validation

Commands longer than `WINDOW_SECONDS` are automatically excluded from train/val/test.
Source WAV files and manifests remain unchanged. The number of excluded recordings is reported
during validation and loading; metrics use the remaining recordings.
Long unknown/background recordings are retained and cropped to the window during loading.
The split validator checks format and split overlap; it does not evaluate model quality.
''')
add("code", '''
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
add("markdown", '''
## 6. Training

Starting an epoch can take time; loss/accuracy are reported after each epoch.
`best.pt` is selected by validation loss; `last.pt` is saved after each epoch.
Runs cannot reuse a nonempty directory. Change `RUN_NAME` for a new experiment.
Skip this cell to evaluate an existing checkpoint without training.
''')
add("code", '''
if RUN_DIR.exists() and any(RUN_DIR.iterdir()):
    raise FileExistsError(f"The directory already contains a run: {RUN_DIR}. Choose a new RUN_NAME or proceed to evaluation.")
run([sys.executable, "-m", "ru_kws.train", "--config", CONFIG_PATH,
     "--data-root", DATA_ROOT, "--run-dir", RUN_DIR, "--device", "auto"], cwd=REPO)
(RUN_DIR / "colab_source.json").write_text(json.dumps({
    "source": source_info, "dataset_archive": str(DATASET_ZIP),
    "dataset_archive_sha256": dataset_archive_hash,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print("Best checkpoint:", RUN_DIR / "best.pt")
''')
add("markdown", "## 7. Training plots\nGenerated from the saved `history.csv`, without retraining.")
add("code", '''
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
add("markdown", '''
## 8. Evaluate the best checkpoint on validation

The CLI loads `best.pt`, its frontend configuration, and its label map.
These are clip classification metrics; they do not measure false positives per hour or streaming latency.
''')
add("code", '''
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
add("markdown", '''
## 9. Final test (optional)

After fixing the model and settings, set `RUN_FINAL_TEST=True` in the configuration
or immediately before the condition below. Check its current value before using Run all.
''')
add("code", '''
if RUN_FINAL_TEST:
    test_report = evaluate_split("test")
    show_report(test_report)
else:
    print("Test skipped. Use validation for tuning.")
print("All results on Drive:", RUN_DIR)
''')

notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "accelerator": "GPU",
    "colab": {"name": "ru_kws_training.ipynb", "provenance": []}},
    "nbformat": 4, "nbformat_minor": 5}
destination = ROOT / "notebooks" / "ru_kws_training.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(destination)
