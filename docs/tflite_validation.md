# Проверка TFLite в Colab

`scripts/evaluate_tflite.py` самостоятельно читает `labels.json` и
`splits/val.jsonl` (поля `path`, `label`). Переменные тренировочного блокнота
и импорт тренировочного скрипта не нужны.

```python
%pip install -q ai-edge-litert numpy soundfile scipy scikit-learn matplotlib tqdm
```

После клонирования/обновления репозитория и распаковки датасета:

```python
!python /content/ru-kws/scripts/evaluate_tflite.py \
    --model /content/bcresnet_with_frontend_fp32.tflite \
    --dataset-root /content/russian_commands_v001/russian_commands_v001_clean_manual \
    --split val \
    --window-seconds 3 \
    --long-commands skip \
    --label-smoothing 0.1 \
    --output-dir /content/evaluation/bcresnet_val_001
```

Замените `/content/ru-kws` на фактический путь проекта. Для теста используйте
`--split test` и новую папку результатов. Существующие результаты не перезаписываются.

Короткие записи дополняются нулями по центру. Длинные `unknown`/`background`
обрезаются по центру, длинные команды с указанным флагом исключаются.
Это соответствует детерминированному выравниванию validation в проекте.
Частота по умолчанию 16000 Гц; несовпадение частоты и многоканальное аудио
вызывают ошибку. `--resample` и `--mono mean` явно разрешают преобразование.

Результаты: `metrics.json`, `predictions.csv`, `skipped.csv`,
`confusion_matrix.csv`, `confusion_matrix.png`. Loss усредняется по записям.
В текущем PyTorch evaluator проекта loss не вычисляется; остальные метрики
можно сравнить при совпадении данных, весов и подготовки аудио.

Поддерживается один вход и один выход `[1, C]`. Модель по умолчанию получает
waveform `[1, N]` и возвращает логиты; для softmax-выхода добавьте
`--output-kind probabilities`. FP32 и квантованные входы/выходы поддерживаются.
Порядок классов в `labels.json` обязан соответствовать модели.
Это оценка отдельных записей, а не событий в стриминге.

Для внешнего frontend используйте `--python-path /content/ru-kws/src`
и `--frontend module:factory`. Фабрика без аргументов должна вернуть функцию,
принимающую NumPy float32 `[1,N]` и возвращающую подготовленный вход модели
(NumPy или torch tensor). Параметры признаков и eval-режим задаются фабрикой;
автоматически определить их из TFLite невозможно.

Полная справка: `python scripts/evaluate_tflite.py --help`.
