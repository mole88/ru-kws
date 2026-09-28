"""Strict paired evaluation on locally cleaned synthetic and Vanya test manifests."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
def read_predictions(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def pair_predictions(old_csv, new_csv, expected_rows):
    old, new = read_predictions(old_csv), read_predictions(new_csv)
    if len(old) != expected_rows or len(new) != expected_rows:
        raise ValueError(f'Models evaluated different record counts: old={len(old)}, new={len(new)}, manifest={expected_rows}')
    paired = []
    for a, b in zip(old, new):
        if (a['record_index'], a['path'], a['true_label']) != (b['record_index'], b['path'], b['true_label']):
            raise ValueError('Prediction rows do not align; comparison would be invalid')
        paired.append(dict(record_index=int(a['record_index']), path=a['path'], label=a['true_label'],
                           old_prediction=a['predicted_label'], new_prediction=b['predicted_label'],
                           old_correct=int(a['correct']), new_correct=int(b['correct']),
                           old_confidence=float(a['confidence']), new_confidence=float(b['confidence'])))
    return paired


def skipped_ids(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        return {(r['record_index'], r['path'], r['label'], r['reason']) for r in csv.DictReader(stream)}


def summarize(paired, labels):
    result = {}
    for label in ['all', *sorted(labels, key=labels.get)]:
        rows = paired if label == 'all' else [r for r in paired if r['label'] == label]
        if not rows:
            result[label] = dict(count=0, old_accuracy=None, new_accuracy=None, delta_accuracy=None, improved=0, regressed=0)
            continue
        n = len(rows)
        old = sum(r['old_correct'] for r in rows)
        new = sum(r['new_correct'] for r in rows)
        result[label] = dict(count=n, old_accuracy=old/n, new_accuracy=new/n,
            delta_accuracy=(new-old)/n,
            improved=sum(not r['old_correct'] and r['new_correct'] for r in rows),
            regressed=sum(r['old_correct'] and not r['new_correct'] for r in rows))
    return result


def supported_macro_f1(summary, labels):
    classes = [summary['classification_report'][label] for label in labels
               if summary['classification_report'][label]['support'] > 0]
    if not classes:
        raise ValueError('Evaluation contains no supported classes')
    return sum(row['f1-score'] for row in classes) / len(classes)


def run_evaluator(script, model, dataset_root, manifest, labels_path, output_dir, window_seconds, threads):
    command = [sys.executable, str(script), '--model', str(model), '--dataset-root', str(dataset_root),
               '--manifest', str(manifest), '--labels', str(labels_path), '--split', 'test',
               '--window-seconds', str(window_seconds), '--long-commands', 'error',
               '--sample-rate', '16000',
               '--label-smoothing', '0', '--threads', str(threads), '--output-dir', str(output_dir)]
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old-model', required=True, type=Path)
    p.add_argument('--new-model', required=True, type=Path)
    p.add_argument('--new-dataset-root', required=True, type=Path)
    p.add_argument('--synthetic-manifest', type=Path)
    p.add_argument('--vanya-manifest', type=Path)
    p.add_argument('--output-dir', required=True, type=Path)
    p.add_argument('--evaluator', type=Path, default=Path(__file__).with_name('evaluate_tflite.py'))
    p.add_argument('--old-checkpoint', type=Path)
    p.add_argument('--new-checkpoint', type=Path)
    p.add_argument('--window-seconds', type=float, default=3.0)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args(argv)
    if args.old_model.resolve() == args.new_model.resolve():
        p.error('Old and new models must be different files')
    if args.output_dir.exists():
        p.error('Output directory already exists; choose a fresh directory')
    if args.window_seconds <= 0 or args.threads < 1:
        p.error('Window seconds and threads must be positive')
    for path in [args.old_model, args.new_model, args.evaluator]:
        if not path.is_file(): raise FileNotFoundError(path)
    if sha256(args.old_model) == sha256(args.new_model):
        p.error('Old and new TFLite files have identical contents')
    dataset = args.new_dataset_root
    labels_path = dataset/'labels.json'
    labels = json.loads(labels_path.read_text(encoding='utf-8'))
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError('Dataset label IDs must be contiguous')
    for name, path in [('old', args.old_checkpoint), ('new', args.new_checkpoint)]:
        if path:
            from ru_kws.checkpoint import load_checkpoint
            checkpoint = load_checkpoint(path)
            if checkpoint['labels'] != labels:
                raise ValueError(f'{name} checkpoint label mapping differs from new dataset')
            audio_cfg = checkpoint['config']['audio']
            if audio_cfg['sample_rate'] != 16000 or audio_cfg['window_seconds'] != args.window_seconds:
                raise ValueError(f'{name} checkpoint audio settings differ from comparison settings')
    test_manifest = args.synthetic_manifest or dataset/'evaluation'/'synthetic_test.jsonl'
    vanya_manifest = args.vanya_manifest or dataset/'evaluation'/'vanya_test.jsonl'
    cases = []
    for name, manifest in [('new_test', test_manifest), ('vanya', vanya_manifest)]:
        rows = read_jsonl(manifest)
        if not rows: raise ValueError(f'Empty evaluation manifest: {manifest}')
        if any(r['label'] not in labels or r.get('split', 'test') != 'test' for r in rows):
            raise ValueError(f'Invalid evaluation labels/splits: {manifest}')
        cases.append((name, dataset, manifest, labels_path, len(rows)))
    args.output_dir.mkdir(parents=True)
    report = dict(models={'old': dict(path=str(args.old_model), sha256=sha256(args.old_model)),
                          'new': dict(path=str(args.new_model), sha256=sha256(args.new_model))},
                  datasets={}, window_seconds=args.window_seconds)
    for case, root, manifest, case_labels, expected in cases:
        summaries = {}
        for side, model in [('old', args.old_model), ('new', args.new_model)]:
            out = args.output_dir/case/side
            run_evaluator(args.evaluator, model, root, manifest, case_labels, out,
                          args.window_seconds, args.threads)
            summaries[side] = json.loads((out/'metrics.json').read_text(encoding='utf-8'))
        paired = pair_predictions(args.output_dir/case/'old'/'predictions.csv',
                                  args.output_dir/case/'new'/'predictions.csv', expected)
        old_skipped = skipped_ids(args.output_dir/case/'old'/'skipped.csv')
        new_skipped = skipped_ids(args.output_dir/case/'new'/'skipped.csv')
        if old_skipped or new_skipped or len(paired) != expected:
            raise ValueError(f'{case}: models did not evaluate exactly the same clips')
        out_csv = args.output_dir/case/'paired_predictions.csv'
        with out_csv.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
            writer.writeheader(); writer.writerows(paired)
        per_class = summarize(paired, labels)
        with (args.output_dir/case/'comparison_by_class.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['label', *next(iter(per_class.values())).keys()])
            writer.writeheader(); writer.writerows(dict(label=k, **v) for k, v in per_class.items())
        report['datasets'][case] = dict(manifest=str(manifest), manifest_sha256=sha256(manifest),
            manifest_count=expected,
            compared_count=len(paired), skipped_count=len(old_skipped),
            metrics=per_class,
            old_macro_f1=summaries['old']['classification_report']['macro avg']['f1-score'],
            new_macro_f1=summaries['new']['classification_report']['macro avg']['f1-score'],
            supported_macro_f1={side: supported_macro_f1(summary, labels)
                                for side, summary in summaries.items()},
            old_processing=summaries['old']['processing_counts'],
            new_processing=summaries['new']['processing_counts'])
        print(case, per_class['all'], flush=True)
    write_json(args.output_dir/'comparison.json', report)
    print('Comparison saved:', args.output_dir, flush=True)
    return report


if __name__ == '__main__':
    main()
