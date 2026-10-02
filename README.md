# Russian KWS

BC-ResNet for six Russian voice commands, plus `unknown` and `background`.
Training from scratch, audio augmentation, FP32 TFLite export with frontend,
clip evaluation and continuous command detection.

Commands: Дальше (`next`), Назад (`back`), Повторить (`repeat`),
Запустить таймер (`start_timer`), Остановить таймер (`stop_timer`),
Сколько осталось (`time_left`).

## Model and training

| Parameter | Archived run settings |
| --- | --- |
| Architecture | BC-ResNet-8, `base_c=64`; 320,040 trainable parameters |
| Input | Mono, 16 kHz, 3 s; 8 classes |
| Frontend | 40 log-mel bins; FFT 512; window 30 ms; hop 10 ms |
| Optimizer | Adam; initial learning rate `3e-4` |
| Label smoothing / seed | 0.1 / 42 |
| Batch size | 32 in run 002; 64 in other runs |
| Epoch limit | 30 in run 002; 60 in run 003; 120 thereafter |
| Checkpoint selection | Lowest validation loss; early stopping patience 8 |
| Export | FP32 TFLite with frontend; 2.57 MiB |

## Augmentation

Enabled only during training in runs 004, 005 and v002.

| Transform | Probability | Settings |
| --- | ---: | --- |
| Speed | 50% | 0.90–1.10×; also changes pitch |
| Gain | 80% | −6 to +6 dB, limited to avoid clipping |
| Room reverb | 50% | MIT RIR convolution; impulse responses up to 1 s |
| Background mixing | 80% | Training backgrounds; SNR 5–20 dB |
| SpecAugment | Every batch | 2 frequency masks of 0–6 bins; 2 time masks of 0–19 frames |

Reverb and background mixing exclude the `background` class.
[Configuration](configs/augmented.yaml)

## Results

Archived runs, September 2026. Values below are **accuracy / macro F1 (%)**.
The original dataset uses 6,959 training and 1,483 validation clips;
the expanded dataset uses 27,367 and 4,818.

| Version | Run | Best / total epochs | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| Before augmentation | `bcresnet_run_002` | 26 / 30 | 98.65 / 98.91 | 96.19 / 96.89 |
| Before augmentation | `bcresnet_run_003` | 29 / 37 | 98.45 / 98.74 | 95.05 / 95.88 |
| With augmentation | `bcresnet_run_004` | 53 / 61 | 97.71 / 98.13 | 97.66 / 98.10 |
| With augmentation | `bcresnet_run_005` | 46 / 54 | 97.91 / 98.29 | — |
| Expanded dataset + augmentation | `bcresnet_v002_001` | 26 / 34 | 98.07 / 98.51 | 98.30 / 98.74 |

Validation uses PyTorch `best.pt`. Original tests contain 1,496 clips;
the expanded test score uses FP32 TFLite on 4,817 usable clips.
`—` means no original-test report was archived. Different datasets are not a paired comparison.

### Same-set comparison

| Evaluation set | Clips | Augmented run 005 accuracy | Expanded v002 accuracy |
| --- | ---: | ---: | ---: |
| New synthetic test | 4,817 | 98.67% | 98.30% |
| Its `unknown` subset | 1,310 | 95.27% | 93.74% |
| Vanya, separate real-speaker test | 60 | 66.67% (40/60) | 71.67% (43/60) |

The archived comparison excluded 1,333 invalid files from 6,150 synthetic test rows
before evaluating both models. Reference recordings used for dataset expansion
are not presented as an independent test.

### Continuous recording

Same 109.06 s recording, 29 command events; 0.1 s hop, threshold 0.5,
two confirming windows, 0.3 s release, 1 s cooldown; matching tolerance 0 s early / 1 s late.

| TFLite run | TP | FP | FN | Event F1 |
| --- | ---: | ---: | ---: | ---: |
| Before augmentation: 003 | 0 | 1 | 29 | 0.00% |
| With augmentation: 005 | 26 | 3 | 3 | 89.66% |
| Expanded dataset: v002 | 26 | 3 | 3 | 89.66% |

These are provisional results from **draft annotations**, not a command-free false-alarm benchmark.
Sources: saved configs, histories and reports in the supplied September 28 run archives.

## Use

1. Upload the dataset ZIP to Drive.
2. [Open the Colab notebook](https://colab.research.google.com/github/mole88/ru-kws/blob/master/notebooks/ru_kws_training.ipynb), enable GPU and set `DATASET_ZIP` / `RUN_NAME`.
3. Train, export and evaluate. Synthetic, Vanya, continuous and podcast tests have separate stages; reports stay on Drive.

Dataset roots contain `labels.json`, `splits/{train,val,test}.jsonl` and audio files.
Vanya uses a separate `evaluation/vanya_test.jsonl`, outside all training/dataset splits.

[Voice recorder](voice_recorder/README.md)
