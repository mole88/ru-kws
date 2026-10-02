"""Windows console recorder. Python 3.10+, pip install sounddevice."""
import argparse
import array
import json
import math
import os
import random
import re
import sys
import time
import uuid
import wave
from datetime import datetime, timezone
from pathlib import Path


def app_directory():
    """Keep user files outside the temporary one-file extraction directory."""
    return Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else Path(__file__).resolve().parent


def default_commands():
    external = app_directory() / 'commands.json'
    return external if external.exists() else Path(__file__).with_name('commands.json')


def default_output():
    if getattr(sys, 'frozen', False):
        return app_directory() / 'recordings'
    return app_directory().parent / 'dist' / 'recordings'


def ask_number(prompt, default, minimum, maximum):
    while True:
        try:
            value = int(input(f'{prompt} [{default}]: ').strip() or str(default))
            if minimum <= value <= maximum:
                return value
        except ValueError:
            pass
        print(f'Enter a number from {minimum} to {maximum}.')


def wizard():
    """Double-click entry point; recording still requires an explicit Enter."""
    import sounddevice as sd
    print('RU KWS Voice Recorder\n')
    while True:
        print('\n1. Record command clips\n2. Record continuous audio\n3. List microphones\n4. Open recordings folder\n0. Exit')
        choice = ask_number('Select', 1, 0, 4)
        if choice == 0:
            return
        try:
            if choice == 4:
                default_output().mkdir(parents=True, exist_ok=True)
                os.startfile(str(default_output()))
                continue
            devices = sd.query_devices()
            inputs = {i: d for i, d in enumerate(devices) if d['max_input_channels'] > 0}
            for i, device in inputs.items():
                print(f'{i}: {device["name"]}')
            if choice == 3:
                continue
            if not inputs:
                raise RuntimeError('No microphone found. Connect one and check Windows microphone permissions.')
            while True:
                selected = input('Microphone number [Enter: Windows default]: ').strip()
                if not selected or (selected.isdigit() and int(selected) in inputs):
                    break
                print('Select a microphone number from the list.')
            speaker = input('Speaker ID [speaker01]: ').strip() or 'speaker01'
            condition = input('Recording condition [quiet_near]: ').strip() or 'quiet_near'
            folder = input(f'Recordings folder [{default_output()}]: ').strip().strip('"')
            output = Path(folder).expanduser().resolve() if folder else default_output()
            output.mkdir(parents=True, exist_ok=True)
            import tempfile
            with tempfile.TemporaryFile(dir=output):
                pass
            argv = ['clips' if choice == 1 else 'long', '--speaker', speaker,
                    '--condition', condition, '--out', str(output)]
            if selected:
                argv += ['--device', selected]
            if choice == 1:
                argv += ['--repeats', str(ask_number('Repeats per command', 10, 1, 1000))]
            else:
                argv += ['--max-seconds', str(ask_number('Maximum seconds', 1800, 1, 3600))]
            main(argv)
            print(f'Recordings: {output}')
        except Exception as exc:
            print(f'Error: {exc}\nCheck the microphone, Windows privacy permissions and a writable output folder.')


def self_test():
    """Exercise bundled imports, PCM writing and resources without recording."""
    import tempfile
    import sounddevice as sd
    import winsound
    commands = load_commands(default_commands())
    with tempfile.TemporaryDirectory(prefix='ru-kws-') as directory:
        path = Path(directory) / 'audio.wav'
        with wave.open(str(path), 'wb') as wav:
            wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            wav.writeframes(b'\x00\x00' * 1600)
        with wave.open(str(path), 'rb') as wav:
            assert wav.getnframes() == 1600
    print(json.dumps(dict(frozen=bool(getattr(sys, 'frozen', False)),
                          commands=len(commands), output=str(default_output()),
                          portaudio=sd.get_portaudio_version(), devices=len(sd.query_devices())), ensure_ascii=False))


def save_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def load_commands(path):
    rows = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Файл команд должен содержать непустой JSON-массив.')
    seen = set()
    for row in rows:
        label, phrase = row['label'], row['text']
        if not re.fullmatch(r'[a-z][a-z0-9_]*', label) or label in seen:
            raise ValueError('Метки должны быть уникальны: латинские буквы, цифры, _.')
        if not isinstance(phrase, str) or not phrase.strip():
            raise ValueError('Текст фразы не может быть пустым.')
        seen.add(label)
    return rows


class Keyboard:
    def __init__(self):
        import msvcrt
        self.api = msvcrt

    def poll(self):
        if not self.api.kbhit():
            return None
        key = self.api.getwch()
        if key in ('\x00', '\xe0'):
            self.api.getwch()
            return None
        if key == '\x03':
            raise KeyboardInterrupt
        return {'з': 'p', 'к': 'r', 'ы': 's'}.get(key.lower(), key.lower())

    def clear(self):
        while self.api.kbhit():
            self.api.getwch()

    def wait(self, allowed):
        self.clear()
        while True:
            key = self.poll()
            if key in allowed:
                return key
            time.sleep(.02)


class Regions:
    """Draft regions on the saved audio sample clock, not wall-clock time."""
    def __init__(self, commands):
        self.commands = commands
        self.active = None
        self.rows = []

    def close(self, frame):
        if self.active is not None:
            if frame > self.active['start_frame']:
                self.rows.append(dict(self.active, end_frame=frame))
            self.active = None

    def key(self, key, frame):
        if key == ' ':
            self.close(frame)
        elif key and key in '123456789' and int(key) <= len(self.commands):
            self.close(frame)
            command = self.commands[int(key) - 1]
            self.active = dict(command, start_frame=frame)


def record(sd, keyboard, path, rate, device, limit, commands=None):
    frames = peak = clipped = squares = 0
    regions = Regions(commands or [])
    problem = None
    block = max(1, round(rate * .02))
    max_frames = round(limit * rate)
    keyboard.clear()
    try:
        with wave.open(str(path), 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            with sd.RawInputStream(samplerate=rate, channels=1, dtype='int16',
                                   device=device, blocksize=block) as stream:
                print('ЗАПИСЬ — говорите. Enter: закончить.', flush=True)
                while frames < max_frames:
                    data, overflow = stream.read(min(block, max_frames - frames))
                    if overflow:
                        raise RuntimeError('Потеря аудиоданных (input overflow). Повторите запись.')
                    data = bytes(data)
                    wav.writeframes(data)  # Update the WAV header after every block.
                    samples = array.array('h', data)
                    frames += len(samples)
                    peak = max(peak, max((abs(x) for x in samples), default=0))
                    clipped += sum(abs(x) >= 32760 for x in samples)
                    squares += sum(x * x for x in samples)
                    key = keyboard.poll()
                    if key in ('\r', '\x1b'):
                        break
                    regions.key(key, frames)
    except (Exception, KeyboardInterrupt) as exc:
        problem = str(exc) or 'Прервано Ctrl+C'
    regions.close(frames)
    rms = math.sqrt(squares / frames) / 32768 if frames else 0
    return dict(frames=frames, duration_seconds=frames / rate, sample_rate=rate,
                peak_dbfs=20 * math.log10(peak / 32768) if peak else None,
                rms_dbfs=20 * math.log10(rms) if rms else None,
                clipped_fraction=clipped / frames if frames else 0,
                error=problem, annotation_status='draft', regions=regions.rows)


def export_regions(path, rows, rate):
    path.write_text(''.join(f"{r['start_frame']/rate:.6f}\t{r['end_frame']/rate:.6f}\t{r['label']}\n"
                            for r in rows), encoding='utf-8')


def import_labels(args):
    """Validate an Audacity label export without overwriting draft annotations."""
    with wave.open(str(args.wav), 'rb') as wav:
        rate, total = wav.getframerate(), wav.getnframes()
    labels = {r['label'] for r in load_commands(args.commands)}
    rows = []
    for number, line in enumerate(args.labels.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != 3:
            raise ValueError(f'Строка {number}: нужны начало, конец, метка через TAB.')
        start, end = map(float, fields[:2])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= total/rate + .000001):
            raise ValueError(f'Строка {number}: интервал вне WAV или нулевой длины.')
        if fields[2] not in labels:
            raise ValueError(f'Строка {number}: неизвестная метка {fields[2]!r}.')
        first, last = round(start * rate), min(total, round(end * rate))
        if first >= last:
            raise ValueError(f'Строка {number}: интервал короче одного отсчёта.')
        rows.append(dict(label=fields[2], start_frame=first, end_frame=last))
    rows.sort(key=lambda r: r['start_frame'])
    output = args.wav.with_name(args.wav.stem + '.reviewed.json')
    if output.exists():
        raise ValueError(f'Файл уже существует: {output}. Сохраните прежнюю версию под другим именем.')
    save_json(output, dict(wav=args.wav.name, sample_rate=rate, frames=total,
                          annotation_status='reviewed', regions=rows))
    print(f'Сохранено: {output} ({len(rows)} интервалов)')


def run(args, sd):
    if args.mode == 'devices':
        print(sd.query_devices())
        return
    if sys.platform != 'win32' or not sys.stdin.isatty():
        raise ValueError('Запустите из обычной консоли Windows (PowerShell / Windows Terminal).')
    commands = load_commands(args.commands)
    if args.mode == 'long' and len(commands) > 9:
        raise ValueError('В длинном режиме поддерживается до 9 команд.')
    info = sd.query_devices(args.device, 'input')
    rate = args.rate or round(info['default_samplerate'])
    sd.check_input_settings(device=args.device, channels=1, dtype='int16', samplerate=rate)
    session = args.out / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    session.mkdir(parents=True, exist_ok=False)
    save_json(session / 'session.json', dict(speaker=args.speaker, condition=args.condition,
              created_utc=datetime.now(timezone.utc).isoformat(), device=dict(info),
              sample_rate=rate, commands=commands, mode=args.mode, seed=args.seed))
    keyboard = Keyboard()
    print(f'Микрофон: {info["name"]}; mono PCM16, {rate} Hz\nПапка: {session.resolve()}')
    if args.mode == 'long':
        for i, command in enumerate(commands, 1):
            print(f'{i}: {command["text"]}')
        print('Enter: начать; Esc: выход. При записи цифра: начало команды, пробел: конец.')
        print('Метки черновые. Между командами запись продолжается.')
        if keyboard.wait(('\r', '\x1b')) == '\x1b':
            return
        path = session / 'continuous.wav'
        result = record(sd, keyboard, path, rate, args.device, args.max_seconds, commands)
        save_json(path.with_suffix('.json'), dict(result, wav=path.name, speaker=args.speaker))
        export_regions(session / 'labels.draft.txt', result['regions'], rate)
        print(f'Сохранено {result["duration_seconds"]:.1f} с; интервалов: {len(result["regions"])}.')
        if result['error']:
            raise RuntimeError(f'Запись непригодна для оценки: {result["error"]}. Частичный WAV сохранён.')
        return
    plan = commands * args.repeats
    random.Random(args.seed).shuffle(plan)
    save_json(session / 'plan.json', plan)
    for index, command in enumerate(plan, 1):
        while True:
            print(f'\n[{index}/{len(plan)}] Скажите: «{command["text"]}»')
            print('Enter: начать; S: пропустить; Esc: выход.')
            key = keyboard.wait(('\r', 's', '\x1b'))
            if key == '\x1b':
                return
            if key == 's':
                break
            path = session / f'{index:04d}_{command["label"]}_{uuid.uuid4().hex[:8]}.wav'
            result = record(sd, keyboard, path, rate, args.device, args.max_seconds)
            row = dict(result, **command, wav=path.name, speaker=args.speaker,
                       condition=args.condition, accepted=False)
            save_json(path.with_suffix('.json'), row)
            print(f'Длительность {result["duration_seconds"]:.2f} с; пик {result["peak_dbfs"]} dBFS.')
            if result['error']:
                raise RuntimeError(f'{result["error"]}. Частичная запись сохранена как accepted=false.')
            if result['peak_dbfs'] is None or result['peak_dbfs'] < -35:
                print('Очень тихо: проверьте микрофон и прослушайте запись.')
            if result['clipped_fraction'] > .001:
                print('Есть клиппинг: уменьшите усиление микрофона.')
            print('Enter: сохранить; P: прослушать; R: повторить; S: отклонить; Esc: выход.')
            while True:
                key = keyboard.wait(('\r', 'p', 'r', 's', '\x1b'))
                if key != 'p':
                    break
                import winsound
                winsound.PlaySound(str(path.resolve()), winsound.SND_FILENAME)
            if key == '\r':
                row['accepted'] = True
                save_json(path.with_suffix('.json'), row)
                with (session / 'manifest.jsonl').open('a', encoding='utf-8') as file:
                    file.write(json.dumps(row, ensure_ascii=False) + '\n')
            if key == '\x1b':
                return
            if key != 'r':
                break
    print(f'Готово: {session.resolve()}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Сбор голосового тестового датасета в Windows')
    parser.add_argument('mode', nargs='?', choices=['devices', 'clips', 'long', 'import-labels', 'self-test'])
    parser.add_argument('--commands', type=Path, default=default_commands())
    parser.add_argument('--out', type=Path, default=default_output())
    parser.add_argument('--speaker', default='speaker01')
    parser.add_argument('--condition', default='quiet_near')
    parser.add_argument('--device', type=int, help='Индекс из режима devices')
    parser.add_argument('--rate', type=int, help='По умолчанию родная частота микрофона')
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seconds', type=float, help='По умолчанию 8 с для clips, 1800 с для long')
    parser.add_argument('--wav', type=Path)
    parser.add_argument('--labels', type=Path)
    args = parser.parse_args(argv)
    if args.mode is None:
        wizard()
        return
    if args.mode == 'self-test':
        self_test()
        return
    if args.mode == 'import-labels':
        if not args.wav or not args.labels:
            parser.error('Укажите --wav и --labels')
        import_labels(args)
        return
    args.max_seconds = args.max_seconds if args.max_seconds is not None else (1800 if args.mode == 'long' else 8)
    if (not math.isfinite(args.max_seconds) or not .1 <= args.max_seconds <= 3600
            or args.repeats < 1 or (args.rate is not None and not 8000 <= args.rate <= 192000)):
        parser.error('Нужны 0.1 <= max-seconds <= 3600, repeats >= 1, 8000 <= rate <= 192000.')
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError('Установите зависимость: py -m pip install sounddevice') from None
    run(args, sd)


if __name__ == '__main__':
    for output in (sys.stdout, sys.stderr):
        if hasattr(output, 'reconfigure'):
            output.reconfigure(encoding='utf-8', errors='replace')
    try:
        main()
    except KeyboardInterrupt:
        print('\nОстановлено пользователем.')
        sys.exit(130)
    except Exception as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        if len(sys.argv) == 1:
            input('Press Enter to close...')
        sys.exit(1)
