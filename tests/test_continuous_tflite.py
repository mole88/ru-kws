"""Event matching, decoder timing and complete CLI with a deterministic runtime."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_continuous_tflite as evaluator


class ContinuousTests(unittest.TestCase):
    def test_prefix_windows_and_final_partial_hop(self):
        self.assertEqual(list(evaluator.window_ends(11, 4)), [4, 8, 11])
        self.assertEqual(list(evaluator.window_ends(8, 4)), [4, 8])
        self.assertEqual(list(evaluator.window_ends(2, 4)), [2])
        audio = np.arange(11, dtype=np.float32)
        np.testing.assert_array_equal(evaluator.trailing_window(audio, 4, 6), [[0, 0, 0, 1, 2, 3]])
        np.testing.assert_array_equal(evaluator.trailing_window(audio, 11, 6), [[5, 6, 7, 8, 9, 10]])

    def test_decoder_hysteresis_repeat_and_ignored_winner(self):
        decoder = evaluator.EventDecoder(['next', 'unknown'], ['unknown'], .5, 2, .2, .5)
        command, other = np.array([.8, .2]), np.array([.1, .9])
        self.assertIsNone(decoder.step(.1, command))
        self.assertEqual(decoder.step(.2, command)['label'], 'next')
        self.assertIsNone(decoder.step(.8, command))  # Cooldown alone must not repeat.
        for time in [.9, 1.0, 1.1]:
            self.assertIsNone(decoder.step(time, other))
        self.assertIsNone(decoder.step(1.2, command))
        self.assertIsNotNone(decoder.step(1.3, command))

    def test_one_to_one_matching_duplicates_wrong_labels_and_latency(self):
        events = [dict(event_id=0, label='next', start_seconds=1, end_seconds=2),
                  dict(event_id=1, label='next', start_seconds=4, end_seconds=5)]
        detections = [dict(label='next', time_seconds=1.5, confidence=.8),
                      dict(label='next', time_seconds=2.1, confidence=.9),
                      dict(label='back', time_seconds=4.5, confidence=.8)]
        metrics, truth, found = evaluator.score_events(events, detections, ['next', 'back'], 10, 0, 1)
        self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn']), (1, 2, 1))
        self.assertEqual(truth[0]['latency_seconds'], -.5)
        self.assertIsNone(found[1]['event_id'])
        self.assertEqual(metrics['unmatched_detections_per_hour_full_recording'], 720)

    def test_overlapping_intervals_match_earliest_ending_first(self):
        events = [dict(event_id=0, label='next', start_seconds=0, end_seconds=5),
                  dict(event_id=1, label='next', start_seconds=1, end_seconds=2)]
        detections = [dict(label='next', time_seconds=1.5, confidence=.8),
                      dict(label='next', time_seconds=4, confidence=.8)]
        metrics, _, found = evaluator.score_events(events, detections, ['next'], 6, 0, 0)
        self.assertEqual(metrics['tp'], 2)
        self.assertEqual(found[0]['event_id'], 1)

    def test_empty_negative_recording_metrics_are_defined(self):
        metrics, _, _ = evaluator.score_events([], [], ['next'], 10, 0, 1)
        self.assertIsNone(metrics['recall'])
        self.assertIsNone(metrics['precision'])
        self.assertEqual(metrics['fp'], 0)
        json.dumps(metrics, allow_nan=False)

    def test_annotation_clock_and_draft_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'continuous.json'
            data = dict(wav='continuous.wav', sample_rate=44100, frames=88200,
                        annotation_status='draft', regions=[dict(label='next', start_frame=22050, end_frame=44100)])
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                evaluator.load_annotations(path, 'continuous.wav', 44100, 88200, ['next'])
            events, status = evaluator.load_annotations(path, 'continuous.wav', 44100, 88200, ['next'], True)
            self.assertEqual((events[0]['start_seconds'], events[0]['end_seconds']), (.5, 1))
            with self.assertRaises(ValueError):
                evaluator.load_annotations(path, 'continuous.wav', 16000, 32000, ['next'], True)
            data['error'] = 'input overflow'
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                evaluator.load_annotations(path, 'continuous.wav', 44100, 88200, ['next'], True)

    def test_full_run_resamples_and_writes_confidence_without_overwriting(self):
        captured = []

        class Runtime:
            def __init__(self, **kwargs):
                self.count = 0
            def allocate_tensors(self):
                pass
            def get_input_details(self):
                return [dict(shape=np.array([1, 48000]), dtype=np.float32, index=0)]
            def get_output_details(self):
                return [dict(shape=np.array([1, 2]), dtype=np.float32, index=1)]
            def set_tensor(self, index, data):
                captured.append(data.copy())
            def invoke(self):
                self.count += 1
            def get_tensor(self, index):
                return np.array([[.9, .1] if 6 <= self.count <= 10 else [.1, .9]], dtype=np.float32)

        module = types.ModuleType('ai_edge_litert.interpreter')
        module.Interpreter = Runtime
        with tempfile.TemporaryDirectory() as temp, patch.dict(sys.modules, {'ai_edge_litert.interpreter': module}):
            root = Path(temp)
            sf.write(root / 'continuous.wav', np.ones(88200, dtype=np.float32) * .1, 44100)
            (root / 'model.tflite').write_bytes(b'deterministic test runtime')
            (root / 'labels.json').write_text(json.dumps(dict(next=0, unknown=1)))
            (root / 'continuous.json').write_text(json.dumps(dict(wav='continuous.wav', sample_rate=44100,
                frames=88200, annotation_status='reviewed', regions=[dict(label='next', start_frame=22050, end_frame=44100)])))
            args = ['--model', str(root / 'model.tflite'), '--wav', str(root / 'continuous.wav'),
                    '--annotations', str(root / 'continuous.json'), '--labels', str(root / 'labels.json'),
                    '--output-dir', str(root / 'report'), '--resample', '--output-kind', 'probabilities']
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                evaluator.main(args)
            metrics = json.loads((root / 'report/metrics.json').read_text())
            self.assertEqual((metrics['tp'], metrics['fp'], metrics['fn'], metrics['windows']), (1, 0, 0, 20))
            self.assertAlmostEqual(metrics['median_latency_seconds'], -.3)
            self.assertEqual(captured[0].shape, (1, 48000))
            np.testing.assert_array_equal(captured[0][0, :46400], np.zeros(46400))
            self.assertAlmostEqual(float(captured[-1][0, -1000:].mean()), .1, places=3)
            self.assertIn('confidence', (root / 'report/detections.csv').read_text())
            self.assertIn('p_unknown', (root / 'report/windows.csv').read_text())
            with self.assertRaises(FileExistsError):
                evaluator.main(args)


if __name__ == '__main__':
    unittest.main()
