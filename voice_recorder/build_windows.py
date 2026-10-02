"""Build with the isolated environment described in README.md."""
import importlib.metadata
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


def main():
    if sys.platform != 'win32':
        raise SystemExit('Build on Windows x64 with Python 3.12 x64.')
    source = Path(__file__).resolve().parent
    root = source.parent
    dist = root / 'dist'
    build = root / 'build' / 'recorder'
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(source),
                    '-p', 'test_record.py', '-v'], check=True)
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                    '--onefile', '--console', '--noupx', '--name', 'RU-KWS-Recorder',
                    '--distpath', str(dist), '--workpath', str(build / 'work'),
                    '--specpath', str(build), '--add-data', f'{source / "commands.json"};.',
                    str(source / 'record.py')], check=True)
    # Preserve a user-customized external command list on subsequent builds.
    if not (dist / 'commands.json').exists():
        shutil.copy2(source / 'commands.json', dist / 'commands.json')
    shutil.copy2(source / 'QUICKSTART.txt', dist / 'RECORDER-README.txt')
    notices = dist / 'recorder-licenses'
    notices.mkdir(exist_ok=True)
    for package in ['sounddevice', 'cffi', 'pycparser', 'pyinstaller']:
        distribution = importlib.metadata.distribution(package)
        for entry in distribution.files or []:
            if any(word in entry.name.lower() for word in ['license', 'copying', 'copyright']):
                target = notices / package / str(entry)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(distribution.locate_file(entry), target)
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.exists():
        shutil.copy2(python_license, notices / 'Python-LICENSE.txt')
    portaudio = Path(importlib.metadata.distribution('sounddevice').locate_file(
        '_sounddevice_data/portaudio-binaries/README.md'))
    if portaudio.exists():
        shutil.copy2(portaudio, notices / 'PortAudio-README.md')
    subprocess.run([str(dist / 'RU-KWS-Recorder.exe'), 'self-test'], check=True)
    with zipfile.ZipFile(dist / 'RU-KWS-Recorder-Windows-x64.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ['RU-KWS-Recorder.exe', 'commands.json', 'RECORDER-README.txt']:
            archive.write(dist / name, name)
        for file in notices.rglob('*'):
            if file.is_file():
                archive.write(file, file.relative_to(dist))
    print('Portable ZIP ready. Personal recordings are not included.')


if __name__ == '__main__':
    main()
