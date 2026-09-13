"""Offline sliding-window event evaluation for a TFLite model with built-in frontend.

This is an explicitly configured decoder simulation, not a device latency benchmark.
Annotation frame positions always use the ORIGINAL WAV sample rate.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from evaluate_tflite import decode, encode


def load_annotations(path, wav_name, rate, frames, commands, allow_draft=False):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    status = data.get('annotation_status')
    if status not in {'draft', 'reviewed'} or (status == 'draft' and not allow_draft):
        raise ValueError('Use reviewed annotations, or --allow-draft for provisional metrics')
    if data.get('error'):
        raise ValueError('Recording metadata reports an audio capture error')
    if data.get('wav') != wav_name or data.get('sample_rate') != rate or data.get('frames') != frames:
        raise ValueError('Annotation WAV name, original sample rate or frame count does not match')
    regions = data.get('regions')
    if not isinstance(regions, list):
        raise ValueError('Annotations must contain a regions list (empty is allowed)')
    events = []
    for region in regions:
        start, end = region['start_frame'], region['end_frame']
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= frames:
            raise ValueError(f'Invalid annotation frame interval: {region}')
        if region['label'] not in commands:
            raise ValueError(f'Unknown or non-command annotation label: {region["label"]}')
        events.append(dict(label=region['label'], start_seconds=start / rate, end_seconds=end / rate))
    events.sort(key=lambda event: (event['start_seconds'], event['end_seconds']))
    for index, event in enumerate(events):
        event['event_id'] = index
    return events, status


def window_ends(total, hop):
    """Evaluate every hop and include a possibly shorter final hop at EOF."""
    yield from range(hop, total, hop)
    yield total


def trailing_window(audio, end, size):
    part = audio[max(0, end - size):end]
    return np.pad(part, (size - len(part), 0))[None, :].astype(np.float32)


def probabilities(scores, kind):
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite model output')
    if kind == 'logits':
        values = np.exp(scores - scores.max())
        return values / values.sum()
    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(scores.sum(), 1, atol=.02):
        raise ValueError('Output does not look like multiclass probabilities')
    return scores / scores.sum()


class EventDecoder:
    """Top-1 threshold, consecutive windows, per-class release and cooldown."""
    def __init__(self, names, ignored, threshold, consecutive, release, cooldown):
        self.names, self.ignored = names, set(ignored)
        self.threshold, self.consecutive = threshold, consecutive
        self.release, self.cooldown = release, cooldown
        self.state = {name: dict(count=0, armed=True, inactive=None, last=-math.inf)
                      for name in names if name not in self.ignored}

    def step(self, time_seconds, probs):
        index = int(np.argmax(probs))
        winner = self.names[index]
        candidate = winner if probs[index] >= self.threshold and winner not in self.ignored else None
        emitted = None
        for name, state in self.state.items():
            if name != candidate:
                state['count'] = 0
                if state['inactive'] is None:
                    state['inactive'] = time_seconds
                if time_seconds - state['inactive'] + 1e-9 >= self.release:
                    state['armed'] = True
                continue
            state['inactive'] = None
            state['count'] += 1
            if (state['armed'] and state['count'] >= self.consecutive
                    and time_seconds - state['last'] + 1e-9 >= self.cooldown):
                emitted = dict(label=name, time_seconds=time_seconds, confidence=float(probs[index]))
                state.update(armed=False, last=time_seconds)
        return emitted


def score_events(events, detections, commands, duration, early, late):
    """Chronological one-to-one matching; choose the eligible event ending first."""
    truth = [dict(event, detection_id=None, confidence=None, latency_seconds=None) for event in events]
    found = [dict(detection, detection_id=i, event_id=None, latency_seconds=None)
             for i, detection in enumerate(detections)]
    for detection in sorted(found, key=lambda row: row['time_seconds']):
        candidates = [event for event in truth if event['detection_id'] is None
                      and event['label'] == detection['label']
                      and event['start_seconds'] - early <= detection['time_seconds'] <= event['end_seconds'] + late]
        if candidates:
            event = min(candidates, key=lambda row: (row['end_seconds'], row['event_id']))
            latency = detection['time_seconds'] - event['end_seconds']
            event.update(detection_id=detection['detection_id'], confidence=detection['confidence'], latency_seconds=latency)
            detection.update(event_id=event['event_id'], latency_seconds=latency)

    def summarize(selected_truth, selected_found):
        tp = sum(row['event_id'] is not None for row in selected_found)
        fp, fn = len(selected_found) - tp, len(selected_truth) - tp
        latencies = [row['latency_seconds'] for row in selected_found if row['event_id'] is not None]
        return dict(true_events=len(selected_truth), detections=len(selected_found), tp=tp, fp=fp, fn=fn,
                    precision=tp / (tp + fp) if tp + fp else None,
                    recall=tp / (tp + fn) if tp + fn else None,
                    f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
                    unmatched_detections_per_hour_full_recording=fp * 3600 / duration,
                    median_latency_seconds=float(np.median(latencies)) if latencies else None,
                    p95_latency_seconds=float(np.percentile(latencies, 95)) if latencies else None)

    metrics = summarize(truth, found)
    metrics['per_class'] = {name: summarize([r for r in truth if r['label'] == name],
                                           [r for r in found if r['label'] == name]) for name in commands}
    return metrics, truth, found


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model', 'wav', 'annotations', 'labels', 'output-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--sample-rate', type=int, default=16000)
    p.add_argument('--window-seconds', type=float, default=3)
    p.add_argument('--hop-seconds', type=float, default=.1)
    p.add_argument('--resample', action='store_true')
    p.add_argument('--output-kind', choices=['logits', 'probabilities'], default='logits')
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--min-consecutive', type=int, default=2)
    p.add_argument('--release-seconds', type=float, default=.3)
    p.add_argument('--cooldown-seconds', type=float, default=1)
    p.add_argument('--early-tolerance', type=float, default=0)
    p.add_argument('--late-tolerance', type=float, default=1)
    p.add_argument('--ignore-labels', nargs='*', default=['unknown', 'background'])
    p.add_argument('--allow-draft', action='store_true')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--print-predictions', action='store_true', help='Print emitted command events')
    return p


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parser().parse_args(argv)
    numeric = [args.window_seconds, args.hop_seconds, args.threshold, args.release_seconds,
               args.cooldown_seconds, args.early_tolerance, args.late_tolerance]
    if (not all(math.isfinite(value) for value in numeric) or args.sample_rate <= 0
            or not 0 < args.hop_seconds <= args.window_seconds or not 0 <= args.threshold <= 1
            or min(numeric[3:]) < 0 or args.min_consecutive < 1 or args.threads < 1):
        raise ValueError('Invalid window, decoder or matching settings')
    size, hop = round(args.sample_rate * args.window_seconds), round(args.sample_rate * args.hop_seconds)
    if min(size, hop) < 1:
        raise ValueError('Window and hop must contain at least one sample')
    labels = json.loads(args.labels.read_text(encoding='utf-8-sig'))
    if (not isinstance(labels, dict) or not labels or any(type(v) is not int for v in labels.values())
            or sorted(labels.values()) != list(range(len(labels)))):
        raise ValueError('Labels must map class names to unique contiguous integer IDs')
    names = sorted(labels, key=labels.get)
    commands = [name for name in names if name not in args.ignore_labels]
    if not commands:
        raise ValueError('No command classes remain')

    import soundfile as sf
    from ai_edge_litert.interpreter import Interpreter
    from tqdm.auto import tqdm

    audio, original_rate = sf.read(args.wav, dtype='float32', always_2d=True)
    if not len(audio) or audio.shape[1] != 1 or not np.isfinite(audio).all():
        raise ValueError('Expected nonempty finite mono WAV')
    original_frames = len(audio)
    duration = original_frames / original_rate
    events, status = load_annotations(args.annotations, args.wav.name, original_rate, original_frames,
                                      commands, args.allow_draft)
    # A reviewed annotation export omits capture error details. Check the recorder sidecar too.
    sidecar = args.wav.with_suffix('.json')
    if sidecar.exists() and json.loads(sidecar.read_text(encoding='utf-8-sig')).get('error'):
        raise ValueError('Recorder sidecar reports a capture error; continuous timing is unreliable')
    audio = audio[:, 0]
    if original_rate != args.sample_rate:
        if not args.resample:
            raise ValueError(f'WAV is {original_rate} Hz; use --resample')
        from scipy.signal import resample_poly
        factor = math.gcd(original_rate, args.sample_rate)
        audio = resample_poly(audio, args.sample_rate // factor, original_rate // factor).astype(np.float32)
    runtime = Interpreter(model_path=str(args.model), num_threads=args.threads)
    runtime.allocate_tensors()
    inputs, outputs = runtime.get_input_details(), runtime.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('Only stateless single-input, single-output models are supported')
    inp, out = inputs[0], outputs[0]
    if tuple(inp['shape']) != (1, size) or tuple(out['shape']) != (1, len(names)):
        raise ValueError(f'Expected waveform [1,{size}] and output [1,{len(names)}]; use a model with built-in frontend')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if status == 'draft':
        print('PROVISIONAL: draft keyboard annotations, not reviewed speech boundaries.', flush=True)
    print(f'Duration: {duration:.3f}s; events: {len(events)}; original rate: {original_rate} Hz', flush=True)
    decoder = EventDecoder(names, args.ignore_labels, args.threshold, args.min_consecutive,
                           args.release_seconds, args.cooldown_seconds)
    detections = []
    windows_count = 0
    with (args.output_dir / 'windows.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['window_index', 'start_seconds', 'end_seconds', 'left_padding_seconds',
                         'predicted_label', 'confidence', 'emitted_label'] + [f'p_{name}' for name in names])
        for index, end in enumerate(tqdm(window_ends(len(audio), hop), total=math.ceil(len(audio) / hop), desc='Continuous evaluation')):
            x = trailing_window(audio, end, size)
            runtime.set_tensor(inp['index'], encode(x, inp))
            runtime.invoke()
            probs = probabilities(decode(runtime.get_tensor(out['index']), out)[0], args.output_kind)
            predicted = int(np.argmax(probs))
            timestamp = min(end / args.sample_rate, duration)
            event = decoder.step(timestamp, probs)
            if event is not None:
                detections.append(event)
                if args.print_predictions:
                    tqdm.write(f'{timestamp:.3f}s: {event["label"]} confidence={event["confidence"]:.2%}')
            writer.writerow([index, max(0, end - size) / args.sample_rate, timestamp,
                             max(0, size - end) / args.sample_rate, names[predicted], float(probs[predicted]),
                             event['label'] if event else ''] + probs.tolist())
            windows_count += 1
    metrics, matched_events, matched_detections = score_events(events, detections, commands, duration,
                                                             args.early_tolerance, args.late_tolerance)
    metrics.update(annotation_status=status, provisional=status == 'draft', duration_seconds=duration,
                   original_sample_rate=original_rate, windows=windows_count, label_to_id=labels,
                   config={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                   hashes={key: hashlib.sha256(getattr(args, key).read_bytes()).hexdigest()
                           for key in ['model', 'wav', 'annotations', 'labels']})
    write_csv(args.output_dir / 'events.csv', ['label', 'start_seconds', 'end_seconds', 'event_id',
              'detection_id', 'confidence', 'latency_seconds'], matched_events)
    write_csv(args.output_dir / 'detections.csv', ['label', 'time_seconds', 'confidence', 'detection_id',
              'event_id', 'latency_seconds'], matched_detections)
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: metrics[key] for key in ['provisional', 'tp', 'fp', 'fn', 'precision', 'recall',
          'f1', 'unmatched_detections_per_hour_full_recording', 'median_latency_seconds']}, indent=2))
    print('Results:', args.output_dir.resolve())


if __name__ == '__main__':
    main()
