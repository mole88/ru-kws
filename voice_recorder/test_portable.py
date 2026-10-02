"""Smoke-test the built artifact with Python removed from child search paths."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import wave
import zipfile


def main():
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='recorder-portable-') as temporary:
        folder = Path(temporary) / 'Portable recorder Кириллица'
        folder.mkdir()
        with zipfile.ZipFile(root / 'dist' / 'RU-KWS-Recorder-Windows-x64.zip') as archive:
            assert not any('recordings/' in name for name in archive.namelist())
            archive.extractall(folder)
        env = dict(os.environ)
        for name in list(env):
            if name.upper().startswith(('PYTHON', 'VIRTUAL_ENV', 'CONDA', '_PYI')):
                env.pop(name)
        env['PATH'] = str(Path(os.environ['SYSTEMROOT']) / 'System32')
        env['PYTHONHOME'] = str(folder / 'no-python-installed')
        env['PYTHONPATH'] = str(folder / 'no-python-installed')
        exe = folder / 'RU-KWS-Recorder.exe'

        def invoke(*args, input=None):
            result = subprocess.run([str(exe), *map(str, args)], cwd=temporary,
                                    env=env, input=input, capture_output=True,
                                    text=True, encoding='utf-8', timeout=30)
            if result.returncode:
                raise AssertionError(result.stdout + result.stderr)
            return result.stdout

        info = json.loads(invoke('self-test'))
        assert info['frozen'] and info['commands'] == 6
        assert Path(info['output']) == folder / 'recordings'
        (folder / 'commands.json').unlink()
        assert json.loads(invoke('self-test'))['commands'] == 6
        assert 'Record command clips' in invoke(input='0\n')
        assert '--speaker' in invoke('--help')
        audio = folder / 'sample.wav'
        with wave.open(str(audio), 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\0\0' * 16000)
        labels = folder / 'labels.txt'
        labels.write_text('0.1\t0.5\tnext\n', encoding='utf-8')
        invoke('import-labels', '--wav', audio, '--labels', labels)
        reviewed = json.loads(audio.with_name('sample.reviewed.json').read_text(encoding='utf-8'))
        assert reviewed['regions'] == [dict(label='next', start_frame=1600, end_frame=8000)]
        print(json.dumps(dict(status='PASS', checks=[
            'ZIP contains no personal recordings', 'Unicode and spaces in install path',
            'Unrelated working directory', 'No Python in child PATH; invalid PYTHONHOME',
            'Bundled PortAudio and WAV I/O', 'External and embedded command resources',
            'Double-click menu entry and exit', 'CLI help', 'Audacity label import'
        ], runtime=info), ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
