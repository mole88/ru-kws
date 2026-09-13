"""Evaluate a single-input, single-output TFLite audio classifier.

Dataset: labels.json {label: id}, splits/val.jsonl with path and label fields.
Paths in manifests are relative to --dataset-root. No training imports needed.
Optional --frontend module:factory: zero-argument factory returning a callable
that takes float32 numpy waveform [1,N] and returns the model input (numpy or
CPU torch tensor). Factory must configure eval mode and training-time features.
This is clip validation, not event-based streaming evaluation.
"""
import argparse
import csv
import hashlib
import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def quant_params(detail):
    q = detail['quantization_parameters']
    scale = np.asarray(q['scales'], dtype=np.float64)
    zero = np.asarray(q['zero_points'], dtype=np.float64)
    if scale.size == 0 or np.any(scale <= 0):
        raise ValueError('Integer tensor has no valid quantization parameters')
    if scale.size > 1:
        shape = [1] * len(detail['shape'])
        shape[q['quantized_dimension']] = scale.size
        scale, zero = scale.reshape(shape), zero.reshape(shape)
    return scale, zero


def encode(x, detail):
    dtype = detail['dtype']
    if np.issubdtype(dtype, np.integer):
        scale, zero = quant_params(detail)
        limits = np.iinfo(dtype)
        x = np.clip(np.rint(x / scale + zero), limits.min, limits.max)
    return np.ascontiguousarray(x, dtype=dtype)


def decode(x, detail):
    if np.issubdtype(detail['dtype'], np.integer):
        scale, zero = quant_params(detail)
        return (x.astype(np.float64) - zero) * scale
    return x.astype(np.float64)


def fit_audio(x, n, label, args):
    if len(x) < n:
        extra = n - len(x)
        left = extra // 2 if args.pad_position == 'center' else 0
        return np.pad(x, (left, extra - left)), 'padded'
    if len(x) > n:
        if label not in args.crop_labels and args.long_commands != 'crop':
            if args.long_commands == 'skip':
                return None, 'skipped_long'
            raise ValueError('Command exceeds window; use --long-commands skip or crop')
        start = (len(x) - n) // 2
        return x[start:start + n], 'cropped'
    return x, 'unchanged'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--dataset-root', type=Path, required=True)
    p.add_argument('--split', default='val', choices=['train', 'val', 'test'])
    p.add_argument('--manifest', type=Path, help='Override split JSONL path')
    p.add_argument('--labels', type=Path, help='Override labels.json path')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--sample-rate', type=int, default=16000)
    p.add_argument('--window-seconds', type=float, default=3.0)
    p.add_argument('--pad-position', choices=['center', 'right'], default='center')
    p.add_argument('--long-commands', choices=['error', 'skip', 'crop'], default='error')
    p.add_argument('--crop-labels', nargs='*', default=['unknown', 'background'])
    p.add_argument('--resample', action='store_true', help='Allow nonmatching sample rates')
    p.add_argument('--mono', choices=['error', 'mean'], default='error')
    p.add_argument('--output-kind', choices=['logits', 'probabilities'], default='logits')
    p.add_argument('--label-smoothing', type=float, default=0.1)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--print-predictions', action='store_true',
                   help='Print each prediction with predicted and true class probabilities')
    p.add_argument('--frontend', help='module:zero_argument_factory (omit for waveform model)')
    p.add_argument('--python-path', type=Path, help='Project src directory for frontend import')
    return p


def main():
    args = parser().parse_args()
    if not 0 <= args.label_smoothing <= 1:
        raise ValueError('label-smoothing must be in [0,1]')
    n = round(args.sample_rate * args.window_seconds)
    if n <= 0 or args.sample_rate <= 0 or args.threads <= 0:
        raise ValueError('Window, sample rate and threads must be positive')

    import soundfile as sf
    from tqdm.auto import tqdm
    from sklearn.metrics import classification_report, confusion_matrix
    from ai_edge_litert.interpreter import Interpreter

    manifest = args.manifest or args.dataset_root / 'splits' / f'{args.split}.jsonl'
    labels_path = args.labels or args.dataset_root / 'labels.json'
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    if not isinstance(labels, dict) or not labels or any(type(v) is not int for v in labels.values()):
        raise ValueError('labels.json must contain {class_name: integer_id}')
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError('Label IDs must be unique and contiguous from zero')
    names = sorted(labels, key=labels.get)
    records = [json.loads(line) for line in manifest.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not records:
        raise ValueError('Empty manifest')
    for r in records:
        if r['label'] not in labels:
            raise ValueError(f'Unknown label in manifest: {r}')

    frontend = None
    if args.python_path:
        sys.path.insert(0, str(args.python_path.resolve()))
    if args.frontend:
        module, factory = args.frontend.split(':', 1)
        frontend = getattr(importlib.import_module(module), factory)()

    runtime = Interpreter(model_path=str(args.model), num_threads=args.threads)
    runtime.allocate_tensors()
    inputs, outputs = runtime.get_input_details(), runtime.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('Only one input and one output are supported; stateful models need a separate evaluator')
    inp, out = inputs[0], outputs[0]
    print('Input:', inp['shape'], inp['dtype'], flush=True)
    print('Output:', out['shape'], out['dtype'], flush=True)
    if tuple(out['shape']) != (1, len(names)):
        raise ValueError('Expected output [1, number_of_classes]')
    if frontend is None and tuple(inp['shape']) != (1, n):
        raise ValueError('Expected waveform input [1,N]. Check window or supply --frontend')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Refuse accidental overwriting of an existing evaluation.
    files = ['metrics.json', 'predictions.csv', 'skipped.csv', 'confusion_matrix.csv', 'confusion_matrix.png']
    if any((args.output_dir / f).exists() for f in files):
        raise FileExistsError('Output directory already contains evaluation results; choose another')
    truth, predictions, losses = [], [], []
    counts = Counter()
    with (args.output_dir / 'predictions.csv').open('w', newline='', encoding='utf-8') as pf, \
         (args.output_dir / 'skipped.csv').open('w', newline='', encoding='utf-8') as sfout:
        writer = csv.writer(pf)
        writer.writerow(['record_index', 'path', 'true_label', 'predicted_label', 'correct', 'loss', 'length_action',
                         'confidence', 'true_class_probability'] + [f'p_{v}' for v in names])
        skipped = csv.writer(sfout)
        skipped.writerow(['record_index', 'path', 'label', 'reason'])
        for idx, r in enumerate(tqdm(records, desc='TFLite validation')):
            path = args.dataset_root / r['path'].replace('\\', '/')
            try:
                audio, sr = sf.read(path, dtype='float32', always_2d=True)
                if len(audio) == 0 or not np.isfinite(audio).all():
                    raise ValueError('Empty or nonfinite audio')
                if audio.shape[1] != 1 and args.mono == 'error':
                    raise ValueError('Non-mono audio; use --mono mean to downmix')
                audio = audio.mean(axis=1)
                if sr != args.sample_rate:
                    if not args.resample:
                        raise ValueError(f'Sample rate {sr}; expected {args.sample_rate}')
                    from math import gcd
                    from scipy.signal import resample_poly
                    g = gcd(sr, args.sample_rate)
                    audio = resample_poly(audio, args.sample_rate // g, sr // g).astype(np.float32)
                    counts['resampled'] += 1
                audio, action = fit_audio(audio, n, r['label'], args)
                counts[action] += 1
                if audio is None:
                    skipped.writerow([idx, r['path'], r['label'], action])
                    continue
                x = audio[None, :]
                if frontend is not None:
                    x = frontend(x)
                    if hasattr(x, 'detach'):
                        x = x.detach().cpu().numpy()
                x = np.asarray(x, dtype=np.float32)
                if tuple(x.shape) != tuple(inp['shape']) or not np.isfinite(x).all():
                    raise ValueError(f'Invalid prepared input: {x.shape}; expected {inp["shape"]}')
                runtime.set_tensor(inp['index'], encode(x, inp))
                runtime.invoke()
                scores = decode(runtime.get_tensor(out['index']), out)[0]
                if not np.isfinite(scores).all():
                    raise ValueError('Nonfinite model output')
                if args.output_kind == 'logits':
                    shifted = scores - scores.max()
                    log_probs = shifted - np.log(np.exp(shifted).sum())
                    probs = np.exp(log_probs)
                else:
                    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(scores.sum(), 1, atol=0.02):
                        raise ValueError('Output does not look like multiclass probabilities')
                    probs = scores / scores.sum()
                    log_probs = np.log(np.maximum(probs, 1e-12))
                target = labels[r['label']]
                pred = int(probs.argmax())
                confidence = float(probs[pred])
                true_class_probability = float(probs[target])
                loss = float(-(1 - args.label_smoothing) * log_probs[target] - args.label_smoothing * log_probs.mean())
                truth.append(target)
                predictions.append(pred)
                losses.append(loss)
                writer.writerow([idx, r['path'], r['label'], names[pred], int(target == pred), loss, action,
                                 confidence, true_class_probability] + probs.tolist())
                if args.print_predictions:
                    tqdm.write(f"[{idx + 1}/{len(records)}] {r['path']} | "
                               f"predicted={names[pred]} confidence={confidence:.2%} | "
                               f"true={r['label']} p_true={true_class_probability:.2%}")
            except Exception as exc:
                raise RuntimeError(f'Record {idx}, {path}: {exc}') from exc

    if not truth:
        raise ValueError('No records evaluated')
    ids = list(range(len(names)))
    report = classification_report(truth, predictions, labels=ids, target_names=names, output_dict=True, zero_division=0)
    accuracy = float(np.mean(np.asarray(truth) == predictions))
    metrics = {
        'accuracy': accuracy, 'loss': float(np.mean(losses)),
        'evaluated': len(truth), 'manifest_records': len(records),
        'processing_counts': dict(counts), 'classification_report': report,
        'label_to_id': labels,
        'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
    }
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    cm = confusion_matrix(truth, predictions, labels=ids)
    with (args.output_dir / 'confusion_matrix.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true / predicted'] + names)
        for name, row in zip(names, cm):
            writer.writerow([name] + row.tolist())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm, display_labels=names).plot(ax=ax, xticks_rotation=45, colorbar=False, values_format='d')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    print(f'\nEvaluated: {len(truth)}/{len(records)}; processing: {dict(counts)}')
    print(f'Loss: {np.mean(losses):.6f}; Accuracy: {accuracy:.6f}')
    print(classification_report(truth, predictions, labels=ids, target_names=names, digits=4, zero_division=0))
    print('Results:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
