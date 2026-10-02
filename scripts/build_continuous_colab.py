"""Generate the standalone recorder event-evaluation notebook and tool archive."""
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
cells = []


def add(kind, source):
    cell = dict(cell_type=kind, metadata={}, source=source.strip() + '\n')
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add('markdown', '''
# Continuous recording: TFLite event evaluation

Run from top to bottom in Colab. No retraining or GPU is required.
Use the same checkpoint that produced the TFLite file: its labels, sample rate
and window length define the model contract. The model must include its frontend.

Upload `continuous-evaluation-tools.zip` when prompted in section 2. This bundle
contains only two Python scripts; recordings and model files stay on your Drive.
The notebook does not depend on whether the new scripts have been pushed to GitHub.
''')
add('code', '''
from google.colab import drive
from pathlib import Path
drive.mount('/content/drive')

ARCHIVE = Path('/content/drive/MyDrive/russian_commands/recordings.zip')
RUN_DIR = Path('/content/drive/MyDrive/russian_commands/runs/bcresnet_run_003')
MODEL = RUN_DIR / 'export/bcresnet_with_frontend_fp32.tflite'
CHECKPOINT = RUN_DIR / 'best.pt'
REPORT_PARENT = Path('/content/drive/MyDrive/russian_commands/continuous_evaluations')

# Decoder settings are starting values, not tuned operating thresholds.
THRESHOLD = 0.5
HOP_SECONDS = 0.1
MIN_CONSECUTIVE = 2
RELEASE_SECONDS = 0.3
COOLDOWN_SECONDS = 1.0
EARLY_TOLERANCE = 0.0
LATE_TOLERANCE = 1.0
ALLOW_DRAFT = True  # Provisional metrics; use False for a reviewed final evaluation.
''')
add('markdown', '## 2. Load evaluation tools and dependencies')
add('code', '''
import subprocess
import sys
import io
import zipfile
from google.colab import files

CODE_DIR = Path('/content/continuous_eval_code')
CODE_DIR.mkdir(exist_ok=True)
uploaded = files.upload()  # Select continuous-evaluation-tools.zip from your PC.
bundles = [data for name, data in uploaded.items() if name.endswith('.zip')]
if len(bundles) != 1:
    raise ValueError('Upload exactly one continuous-evaluation-tools.zip')
with zipfile.ZipFile(io.BytesIO(bundles[0])) as bundle:
    for name in ['evaluate_tflite.py', 'evaluate_continuous_tflite.py']:
        (CODE_DIR / name).write_bytes(bundle.read(name))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'ai-edge-litert', 'numpy', 'soundfile', 'scipy', 'tqdm',
                'pandas', 'matplotlib'], check=True)
''')
add('markdown', '## 3. Extract recordings and read the checkpoint contract')
add('code', '''
import hashlib
import json
import stat
import torch

def extract_cached(archive, parent):
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    digest = digest.hexdigest()
    target = parent / digest
    marker = target / '.extraction_complete'
    if not marker.exists():
        target.mkdir(parents=True, exist_ok=True)
        root = target.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                if (not (root / member.filename).resolve().is_relative_to(root)
                        or stat.S_ISLNK(member.external_attr >> 16)):
                    raise ValueError(f'Invalid ZIP member: {member.filename}')
            bundle.extractall(target)
        marker.write_text(digest)
    return target, digest

RECORDINGS, archive_hash = extract_cached(ARCHIVE, Path('/content/real_recordings'))
candidates = sorted(RECORDINGS.rglob('continuous.wav'))
if len(candidates) != 1:
    raise ValueError(f'Expected one continuous.wav; select WAV explicitly from: {candidates}')
WAV = candidates[0]
reviewed = WAV.with_name(WAV.stem + '.reviewed.json')
ANNOTATIONS = reviewed if reviewed.exists() else WAV.with_suffix('.json')

checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
if checkpoint.get('format_version') != 1:
    raise ValueError('Unsupported checkpoint format')
labels = checkpoint['labels']
sample_rate = int(checkpoint['config']['audio']['sample_rate'])
window_seconds = float(checkpoint['config']['audio']['window_seconds'])
LABELS = CODE_DIR / 'labels.json'
LABELS.write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding='utf-8')
print('WAV:', WAV)
print('Annotations:', ANNOTATIONS)
print('Class order:', sorted(labels, key=labels.get))
print('Sample rate / window:', sample_rate, window_seconds)
''')
add('markdown', '''
## 4. Evaluate the full recording

Every window ends at its decision timestamp. Prefix windows are left-padded;
the final partial hop is included, with no post-recording padding. The decoder
uses top-1 probabilities over ALL model classes. `unknown` and `background`
never emit commands. A class must remain top-1 above threshold for the configured
number of windows. After an event it must become ineligible for the release time
before it can emit again; cooldown also applies per class.

A detection matches one same-class annotation between its start minus early
tolerance and its end plus late tolerance. Matching is chronological, choosing
the earliest-ending unmatched eligible annotation. Duplicates and wrong-class
detections are false positives; unmatched annotations are misses.
''')
add('code', '''
from datetime import datetime
REPORT_DIR = REPORT_PARENT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
command = [sys.executable, str(CODE_DIR / 'evaluate_continuous_tflite.py'),
    '--model', str(MODEL), '--wav', str(WAV), '--annotations', str(ANNOTATIONS),
    '--labels', str(LABELS), '--output-dir', str(REPORT_DIR),
    '--sample-rate', str(sample_rate), '--window-seconds', str(window_seconds),
    '--hop-seconds', str(HOP_SECONDS), '--resample', '--output-kind', 'logits',
    '--threshold', str(THRESHOLD), '--min-consecutive', str(MIN_CONSECUTIVE),
    '--release-seconds', str(RELEASE_SECONDS), '--cooldown-seconds', str(COOLDOWN_SECONDS),
    '--early-tolerance', str(EARLY_TOLERANCE), '--late-tolerance', str(LATE_TOLERANCE),
    '--print-predictions']
if ALLOW_DRAFT:
    command.append('--allow-draft')
subprocess.run(command, check=True)
print('Saved to Drive:', REPORT_DIR)
''')
add('markdown', '''
## 5. Inspect metrics, confidence and the timeline

`events.csv`: one row per annotated command; missing detection_id means a miss.
`detections.csv`: emitted commands; missing event_id means an unmatched detection.
`windows.csv`: top prediction and all class probabilities for every window.

Latency is decision time minus annotation end; a negative value means detection
during the phrase. It excludes runtime/device latency. FP/hour uses the ENTIRE
recording duration, including commands, not separately verified negative audio.
Draft metrics are provisional. Threshold tuning on this session makes it
development data; use other sessions for a final held-out evaluation.
''')
add('code', '''
import pandas as pd
import matplotlib.pyplot as plt

metrics = json.loads((REPORT_DIR / 'metrics.json').read_text())
print('PROVISIONAL:', metrics['provisional'])
display(pd.DataFrame(metrics['per_class']).T)
events = pd.read_csv(REPORT_DIR / 'events.csv')
detections = pd.read_csv(REPORT_DIR / 'detections.csv')
windows = pd.read_csv(REPORT_DIR / 'windows.csv')
display(events.style.format({'confidence': '{:.1%}', 'latency_seconds': '{:.3f}'}, na_rep='missed'))
display(detections.style.format({'confidence': '{:.1%}', 'latency_seconds': '{:.3f}'}, na_rep='unmatched'))

commands = list(metrics['per_class'])
fig, axes = plt.subplots(len(commands), 1, figsize=(16, 2.3 * len(commands)), sharex=True, squeeze=False)
for ax, label in zip(axes[:, 0], commands):
    ax.plot(windows.end_seconds, windows['p_' + label], label='Window probability')
    ax.axhline(THRESHOLD, color='gray', linestyle='--', label='Threshold')
    for event in events[events.label == label].itertuples():
        ax.axvspan(event.start_seconds, event.end_seconds, color='green', alpha=.18)
    selected = detections[detections.label == label]
    ax.scatter(selected.time_seconds, selected.confidence, marker='x', color='red', label='Emitted event')
    ax.set(title=label, ylim=(0, 1), ylabel='Probability')
axes[0, 0].legend(loc='upper right')
axes[-1, 0].set_xlabel('Recording time, seconds (green = annotation)')
fig.suptitle('PROVISIONAL: draft annotations' if metrics['provisional'] else 'Continuous evaluation')
fig.tight_layout()
fig.savefig(REPORT_DIR / 'timeline.png', dpi=150)
plt.show()
''')

notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                          'language_info': {'name': 'python'}, 'colab': {'name': 'continuous_tflite_evaluation.ipynb'}})
(ROOT / 'notebooks' / 'continuous_tflite_evaluation.ipynb').write_text(
    json.dumps(notebook, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(ROOT / 'dist').mkdir(exist_ok=True)
with zipfile.ZipFile(ROOT / 'dist/continuous-evaluation-tools.zip', 'w', zipfile.ZIP_DEFLATED) as bundle:
    for name in ['evaluate_tflite.py', 'evaluate_continuous_tflite.py']:
        bundle.write(ROOT / 'scripts' / name, name)
print('Created notebook and continuous-evaluation-tools.zip')
