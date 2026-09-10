# Russian KWS

Минимальный проект обучения BC-ResNet для русских голосовых команд.
Основной код — Python-пакет; из Colab запускаются обычные CLI-команды.

В первой версии: JSONL dataset, проверка splits, padding/crop, log-mel,
BC-ResNet, обучение с early stopping, best/last checkpoints и оценка клипов.
Online noise mixing, SpecAugment, потоковый decoder, resume и TFLite export
пока не реализованы. Модель обучается с нуля, предобученные веса не скачиваются.

## Установка

Python 3.10+. Используйте совместимые сборки PyTorch и torchaudio.
В Colab пакет использует уже установленный PyTorch, если он удовлетворяет требованиям.

```bash
git clone <URL-вашего-репозитория> ru-kws
cd ru-kws
python -m pip install -e .
```

Для разработки: `python -m pip install -e ".[dev]"`, затем `python -m pytest`.
Данные, виртуальное окружение и результаты исключены из Git через `.gitignore`.

## Датасет

Передавайте корень **внутренней** папки датасета, где находится `labels.json`:

```text
russian_commands_v001_clean_manual/
  labels.json
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
  audio/...
```

`labels.json` — отображение названия класса в уникальный ID от 0 до N−1:

```json
{"next": 0, "back": 1, "repeat": 2, "start_timer": 3,
 "stop_timer": 4, "time_left": 5, "unknown": 6, "background": 7}
```

Каждая строка manifest:

```json
{"path": "audio/xtts/example.wav", "label": "next", "split": "train", "speaker_group": "xtts:voice1", "parent_id": "source1"}
```

Обязательны `path` (относительно корня) и `label`. Дополнительные поля сохраняются
совместимыми с генератором датасета. Валидатор проверяет предоставленные group IDs,
контрольную сумму, если указана, и точные дубликаты между splits. Отсутствующие
group IDs не позволяют установить независимость дикторов/источников.

WAV: mono, 16 kHz. Громкость не нормализуется при чтении. Длинные `unknown` и
`background` обрезаются; команды **не обрезаются и не исключаются автоматически**.
Если команда длиннее окна, выберите большее окно или явно исправьте датасет.
Train размещает короткую запись случайно, val/test — по центру.

## Запуск

Из корня репозитория:

```bash
python -m ru_kws.data.validate --data-root /content/data/russian_commands_v001_clean_manual

python -m ru_kws.train --config configs/baseline.yaml --data-root /content/data/russian_commands_v001_clean_manual --run-dir /content/drive/MyDrive/russian_commands/runs/run_001

python -m ru_kws.evaluate --checkpoint /content/drive/MyDrive/russian_commands/runs/run_001/best.pt --data-root /content/data/russian_commands_v001_clean_manual --split val --output /content/drive/MyDrive/russian_commands/runs/run_001/reports/val.json
```

`--device cpu` / `--device cuda` позволяют выбрать устройство. По умолчанию — auto.
Каждому запуску нужна новая или пустая `run-dir`; старые результаты не перезаписываются.
Для другого окна измените config и передайте такое же `--window-seconds` валидатору.

Для финальной проверки замените `--split val` на `--split test` и имя отчёта.
Подбирайте настройки только по validation; test оставляйте для финальной оценки.
Это **оценка отдельных клипов**, она не измеряет FP/hour и задержку событий.

## Результат запуска

```text
run_001/
  config.yaml
  labels.json
  environment.json
  dataset_summary.json
  history.csv
  best.pt
  last.pt
```

`best.pt` выбирается по минимальному val loss. `last.pt` сохраняет последнюю
завершённую эпоху. Checkpoint содержит архитектуру, frontend config, labels,
веса модели, optimizer/scheduler и hashes train/val manifests.
Автоматического продолжения прерванного обучения пока нет.

Evaluate загружает именно переданный checkpoint и его config. Отчёт содержит
accuracy, macro F1, per-class precision/recall/F1/support и confusion matrix
(строки — истинные классы, столбцы — предсказанные). Macro F1 включает все
классы из labels.json; для отсутствующего класса метрики принимаются равными нулю.

## Colab

Готовый блокнот: [notebooks/ru_kws_training.ipynb](notebooks/ru_kws_training.ipynb).
Он поддерживает GitHub и ZIP исходников на Drive, распаковывает датасет локально,
проверяет данные, запускает обучение и строит графики/отчёт лучшего checkpoint.
Для запуска без GitHub загрузите `ru-kws-source.zip` в корень «Мой диск» и оставьте
`SOURCE_MODE = "drive_zip"`. Test выключен по умолчанию.

Пересобрать блокнот после изменения его шаблона: `python scripts/build_colab.py`.

В начале подключите Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Скачайте репозиторий, выполните `%cd /content/ru-kws` и `%pip install -e .`.
Распакуйте архив WAV на локальный диск `/content/data`, а run-dir укажите на Drive.
Запускайте команды выше через `!python -m ...`. Checkpoints сохраняются после
каждой эпохи; данные читать из множества мелких файлов на Drive не требуется.

## Модули

- `data`: manifest, Dataset, collate, validation.
- `audio`: WAV I/O и единый frontend.
- `models`: исходная Qualcomm BC-ResNet с относительным импортом.
- `training`: эпоха обучения/валидации с усреднением по примерам.
- `evaluation`: метрики классификации клипов.
- `train.py`, `evaluate.py`: CLI-точки входа.

Происхождение модели и лицензия: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
