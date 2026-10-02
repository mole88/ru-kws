import argparse
import contextlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import record


class FakeKeyboard:
    def __init__(self, keys):
        self.keys = iter(keys)

    def clear(self):
        pass

    def poll(self):
        key = next(self.keys, None)
        if isinstance(key, BaseException):
            raise key
        return key


class FakeAudio:
    def __init__(self, overflow=False):
        self.overflow = overflow

    def RawInputStream(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self, count):
        return struct.pack('<h', 1000) * count, self.overflow


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'audio.wav'

    def capture(self, keys, limit=1, overflow=False):
        with contextlib.redirect_stdout(io.StringIO()):
            return record.record(FakeAudio(overflow), FakeKeyboard(keys), self.path,
                                 16000, None, limit, [{'label': 'next', 'text': 'Дальше'}])

    def test_continuous_audio_and_sample_regions(self):
        result = self.capture(['1', None, ' ', '\r'])
        self.assertEqual(result['frames'], 1280)
        self.assertEqual(result['regions'][0]['start_frame'], 320)
        self.assertEqual(result['regions'][0]['end_frame'], 960)
        with wave.open(str(self.path)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()),
                             (1, 2, 16000, 1280))
            self.assertEqual(len(wav.readframes(9999)), 2560)

    def test_limit_closes_region_at_exact_sample(self):
        result = self.capture(['1'], limit=.051)
        self.assertEqual(result['frames'], 816)
        self.assertEqual(result['regions'][0]['end_frame'], 816)

    def test_overflow_is_not_silently_accepted(self):
        result = self.capture([], overflow=True)
        self.assertIn('overflow', result['error'])
        self.assertEqual(result['frames'], 0)
        with wave.open(str(self.path)) as wav:
            self.assertEqual(wav.getnframes(), 0)

    def test_interrupt_preserves_partial_wav_and_closes_region(self):
        result = self.capture(['1', KeyboardInterrupt()])
        self.assertTrue(result['error'])
        self.assertEqual(result['regions'][0]['end_frame'], 640)
        with wave.open(str(self.path)) as wav:
            self.assertEqual(wav.getnframes(), 640)

    def test_reviewed_label_import_and_validation(self):
        self.capture([], limit=1)
        labels = self.path.with_suffix('.txt')
        args = argparse.Namespace(wav=self.path, labels=labels,
                                  commands=Path(record.__file__).with_name('commands.json'))
        for bad in ['0\t2\tnext', 'nan\t0.5\tnext', '0\t0.5\tbogus', '0.4\t0.2\tnext']:
            labels.write_text(bad, encoding='utf-8')
            with self.assertRaises(ValueError):
                record.import_labels(args)
        labels.write_text('0.1\t0.5\tnext\n', encoding='utf-8')
        with contextlib.redirect_stdout(io.StringIO()):
            record.import_labels(args)
        result = json.loads(self.path.with_name('audio.reviewed.json').read_text())
        self.assertEqual(result['regions'], [dict(label='next', start_frame=1600, end_frame=8000)])
        with self.assertRaises(ValueError):
            record.import_labels(args)

    def test_audacity_export(self):
        target = self.path.with_suffix('.txt')
        record.export_regions(target, [dict(start_frame=1600, end_frame=8000, label='next')], 16000)
        self.assertEqual(target.read_text(), '0.100000\t0.500000\tnext\n')

    def test_frozen_output_is_beside_exe_not_extraction_folder(self):
        executable = Path(self.tmp.name) / 'portable' / 'recorder.exe'
        with patch.object(record.sys, 'frozen', True, create=True), patch.object(record.sys, 'executable', str(executable)):
            self.assertEqual(record.default_output(), executable.parent / 'recordings')
            self.assertTrue(record.default_commands().is_file())
            executable.parent.mkdir()
            external = executable.parent / 'commands.json'
            external.write_text('[]')
            self.assertEqual(record.default_commands(), external)

    def test_no_arguments_opens_wizard(self):
        with patch.object(record, 'wizard') as wizard:
            record.main([])
            wizard.assert_called_once_with()

    def test_russian_keyboard_shortcuts(self):
        for key, expected in [('з', 'p'), ('К', 'r'), ('ы', 's')]:
            keyboard = record.Keyboard.__new__(record.Keyboard)
            keyboard.api = unittest.mock.Mock()
            keyboard.api.kbhit.return_value = True
            keyboard.api.getwch.return_value = key
            self.assertEqual(keyboard.poll(), expected)


if __name__ == '__main__':
    unittest.main()
