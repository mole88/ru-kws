# Russian KWS

Минимальный проект обучения BC-ResNet для русских голосовых команд.
Основной код — Python-пакет; из Colab запускаются обычные CLI-команды.

В первой версии: JSONL dataset, проверка splits, padding/crop, log-mel,
BC-ResNet, обучение с early stopping, best/last checkpoints и оценка клипов.
Online noise mixing, SpecAugment, потоковый decoder, resume и TFLite export
пока не реализованы. Модель обучается с нуля, предобученные веса не скачиваются.

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


## Модули

- `data`: manifest, Dataset, collate, validation.
- `audio`: WAV I/O и единый frontend.
- `models`: исходная Qualcomm BC-ResNet с относительным импортом.
- `training`: эпоха обучения/валидации с усреднением по примерам.
- `evaluation`: метрики классификации клипов.
- `train.py`, `evaluate.py`: CLI-точки входа.


## Обучение в Colab

[Открыть блокнот в Colab](https://colab.research.google.com/github/mole88/ru-kws/blob/master/notebooks/ru_kws_training.ipynb).

Включите GPU, загрузите архив датасета на Google Drive и выполните ячейки по порядку.
Код автоматически загружается из GitHub (ветка `master`); ZIP исходников не нужен.
В настройках укажите путь к датасету и новое имя запуска `RUN_NAME`.
Checkpoints и отчёты сохраняются на Drive. Финальный test выключен по умолчанию.
Для воспроизводимости можно заменить `REPO_REF` на commit SHA.

После изменения генератора обновите блокнот командой `python scripts/build_colab.py`.
