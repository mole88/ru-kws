# Russian KWS

A minimal BC-ResNet training project for Russian voice commands.
The core code is a Python package; Colab runs standard CLI commands.

Implemented on `master`: JSONL datasets, split validation, padding/cropping, log-mel features,
BC-ResNet, training with early stopping, best/last checkpoints, clip evaluation,
and standalone validation of externally exported TFLite models.
Online noise mixing, SpecAugment, streaming decoding, training resume, and TFLite
export are not implemented yet. The model trains from scratch; pretrained weights
are not downloaded.

## Latest training and evaluation

The latest archived run is **`bcresnet_run_005` (2026-09-13)**, trained from scratch
using the `dev` revision
[`d3f1e3b`](https://github.com/mole88/ru-kws/tree/d3f1e3b5195a9a9b1464ebe35b5255d483ce76d5).
These results describe that revision, not the more limited `master` implementation
documented below.

BC-ResNet (`base_c=64`) uses 16 kHz audio, a 3 s input window and 40 log-mel bins
(512-point FFT, 30 ms analysis window, 10 ms hop). Training used batch size 64,
initial learning rate 0.0003, label smoothing 0.1 and seed 42.
Early stopping ended training after 54 epochs; **`best.pt` is epoch 46**, selected
by validation loss.

| Training checkpoint | Train loss | Train accuracy | Validation loss | Validation accuracy |
| --- | ---: | ---: | ---: | ---: |
| Best, epoch 46 | 0.52320 | 99.11% | 0.52906 | 97.91% |
| Last, epoch 54 | 0.52067 | 98.98% | 0.53483 | 97.64% |

Training accuracy is measured with augmentation and is not directly comparable to
clean validation accuracy. The saved configuration enables speed perturbation
(0.9–1.1×, probability 0.5), gain (±6 dB, 0.8), MIT room impulse responses
(up to 1 s, 0.5), background mixing (SNR 5–20 dB, 0.8) and SpecAugment
(two frequency masks up to 7 bins and two time masks up to 20 frames).

### Clip evaluation

| Model / evaluation set | Clips | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| PyTorch `best.pt`, synthetic validation | 1,483 | **97.91%** | **98.29%** |
| TFLite FP32 with frontend, same validation | 1,483 | **97.91%** | **98.29%** |
| Same TFLite model, real recorded commands | 60 | **98.33% (59/60)** | — |

PyTorch and TFLite have identical per-class metrics on this validation set.
The real-recording check contains 10 examples of each of the six commands, with
no `unknown` or `background` examples; it does not measure rejection quality.
All 60 clips were resampled to 16 kHz and centrally padded to 3 s.
One `next` example was missed. The synthetic figures are **validation results**,
not a held-out test score; no synthetic test report is present for this run.

### Continuous audio evaluation — provisional

On a 109.06 s recording containing 29 annotated command events, the same FP32
TFLite model produced **26 true positives, 3 false positives and 3 false negatives**:
event precision, recall and F1 are all **89.66%**.

Settings: 3 s windows, 0.1 s hop, threshold 0.5, two consecutive positive windows,
0.3 s release and 1 s cooldown; matching tolerances are 0 s early and 1 s late.
The report explicitly marks annotations as `draft` and results as `provisional`.
Its 3 unmatched detections correspond to 99.03 per hour when normalized by the
full short recording; this is not a background-only false-alarm benchmark.
These event metrics must not be confused with clip classification accuracy.

All three TFLite evaluations reference the same model SHA-256:
`bd11a4a0d1f631f4bea414ffabc7a79a8dd0d4935d3df581d57f33e3d9cf0ee7`.

Sources: [training history](https://drive.google.com/file/d/1tZS3Zf71e9V_npNSccL4SuUdzBiIcXek/view),
[PyTorch validation](https://drive.google.com/file/d/1eDXBQFtIdz1ReBqm0TZor_jRAE1PEX8M/view),
[TFLite validation](https://drive.google.com/file/d/1hCTzByoXXjBkgLsxa4iTG6qWWhkW4YlI/view),
[real clips](https://drive.google.com/file/d/1uWZkgwPqGPTwB6GeCnfgXZKZAn-5myRj/view),
[continuous evaluation](https://drive.google.com/file/d/1vH9HI3k8rb6OymbgDxOEeccbZz-7mcSa/view).

## Dataset

### Composition and versions

The task has six Russian command classes plus two rejection classes:

| ID | Label | Phrase / meaning | Run 005 train | Run 005 validation |
| --- | --- | --- | ---: | ---: |
| 0 | `next` | Дальше | 697 | 150 |
| 1 | `back` | Назад | 685 | 143 |
| 2 | `repeat` | Повторить | 698 | 149 |
| 3 | `start_timer` | Запустить таймер | 689 | 149 |
| 4 | `stop_timer` | Остановить таймер | 693 | 142 |
| 5 | `time_left` | Сколько осталось | 697 | 150 |
| 6 | `unknown` | Non-target speech | 2,100 | 450 |
| 7 | `background` | Background audio | 700 | 150 |
| | **Total used** | | **6,959** | **1,483** |

The original `v001` quality report lists **10,000 synthetic/background clips**:
1,000 per command, 3,000 unknown and 1,000 background, split 70/15/15.
Speech generation combines Silero and XTTS (configured Silero fraction: 20%);
background clips are 2.5 s long. The configured voices are separated by split:
Silero has 3/1/1 train/validation/test voices and XTTS has 14/3/3.
The synthetic test set measures generalization to held-out synthesis voices,
not necessarily to real microphones or speakers.

Run 005 actually uses **`russian_commands_v001_clean_manual`**, packaged in
`russian_commands_v001_clean_manual_with_aug.zip`. Its cleaned manifests have
6,987 training and 1,493 validation records; excluding command recordings longer
than 3 s removes another 28 and 10 respectively, giving the counts above.
Short clips are padded; long unknown/background clips can be cropped.
The original 10,000-clip inventory must therefore not be reported as this run's
effective training size. Real recordings are evaluated separately in the results above.

Manifests retain `speaker_group` and `parent_id` for split validation and source
traceability. Preserve these groups when rebuilding splits so related voices and
derived examples do not leak across partitions.

A newer **`v002_expanded`** generation configuration targets 5,000 examples per
command, 10,000 unknown and 1,000 background clips (41,000 planned in total),
and includes real-recording inputs and three voice-reference recordings.
It sets a 0.35–2.6 s speech duration range and a 150 ms margin.
These are **generation targets**, not verified completed dataset counts.
Run 005 predates this expansion and was not trained on it.

Sources: [original dataset quality report](https://drive.google.com/file/d/1ITevszdsbYSDk0CvA7uQBOltARnT9rp1/view),
[original generation configuration](https://drive.google.com/file/d/1HrjghcRLYdjOzxKH0BUF3n-JvD8_42Wf/view),
[run dataset summary](https://drive.google.com/file/d/1Keiru-rZeYxpdynoblCZ6OK970u-w78R/view),
[effective class counts](https://drive.google.com/file/d/1Q18goqfc4pJyTcrI3sRGOyCJl12ENMrM/view),
[v002 generation configuration](https://drive.google.com/file/d/1_Dkx7T5RHixsBX2-ckdnblkYLwtY1nLG/view).

### Dataset layout

Pass the root of the **inner** dataset directory containing `labels.json`:

```text
russian_commands_v001_clean_manual/
  labels.json
  splits/
    train.jsonl
    val.jsonl
    test.jsonl
  audio/...
```

`labels.json` maps class names to unique IDs from 0 to N-1:

```json
{"next": 0, "back": 1, "repeat": 2, "start_timer": 3,
 "stop_timer": 4, "time_left": 5, "unknown": 6, "background": 7}
```

Each manifest line has the following structure:

```json
{"path": "audio/xtts/example.wav", "label": "next", "split": "train", "speaker_group": "xtts:voice1", "parent_id": "source1"}
```

## Modules

- `data`: manifests, Dataset, collation, and validation.
- `audio`: WAV I/O and a shared frontend.
- `models`: the original Qualcomm BC-ResNet with relative imports.
- `training`: training/validation epochs with metrics averaged across examples.
- `evaluation`: clip classification metrics.
- `train.py`, `evaluate.py`: CLI entry points.
- `scripts/evaluate_tflite.py`: standalone TFLite dataset evaluation.

## Training in Colab

[Open the notebook in Colab](https://colab.research.google.com/github/mole88/ru-kws/blob/master/notebooks/ru_kws_training.ipynb).

Enable a GPU, upload the dataset archive to Google Drive, and run the cells in order.
The code is downloaded from GitHub (`master`); a source ZIP is not required.
Set the dataset path and a new `RUN_NAME` in the configuration.
Checkpoints and reports are saved to Drive. Check `RUN_FINAL_TEST` before running
all cells: the current notebook enables it, while the generator defaults to False.
Use validation for tuning and reserve test evaluation for fixed model settings.
For reproducibility, set `REPO_REF` to a commit SHA.

`python scripts/build_colab.py` rebuilds the notebook from the generator template.
This replaces notebook edits, settings, and saved outputs; back up local changes
before rebuilding.

## TFLite validation

`scripts/evaluate_tflite.py` reads `labels.json` and `splits/val.jsonl` independently
of the training notebook. Install its dependencies in the evaluation environment:

```shell
python -m pip install ai-edge-litert numpy soundfile scipy scikit-learn matplotlib tqdm
```

After cloning the repository and extracting the dataset, run:

```shell
python scripts/evaluate_tflite.py --model /path/to/model.tflite --dataset-root /path/to/dataset --split val --window-seconds 3 --long-commands skip --label-smoothing 0.1 --output-dir /path/to/evaluation/val_001
```

Replace the example paths with your actual paths. Use `--split test` and a new
output directory for test evaluation. Existing results are not overwritten.
Outputs include metrics, predictions, skipped recordings, and a confusion matrix
in CSV and PNG formats. See `python scripts/evaluate_tflite.py --help` for options.

Short recordings are padded centrally. Long unknown/background recordings are
cropped centrally; `--long-commands skip` excludes overlong commands. The default
sample rate is 16000 Hz. Rate mismatches and multichannel input are rejected unless
`--resample` and `--mono mean` explicitly enable conversion.

The model must have one input and one `[1, C]` output. The default input is waveform
`[1, N]`, and the default output is logits; use `--output-kind probabilities` for
softmax output. FP32 and quantized tensors are supported. The class order in
`labels.json` must match the model. This evaluates clips, not streaming events.

For an external frontend, use `--python-path /path/to/ru-kws/src` and
`--frontend module:factory`. The factory takes no arguments and returns a callable
that accepts NumPy float32 `[1, N]` audio and returns the model input as a NumPy
array or torch tensor. Configure feature parameters and evaluation mode in the
factory; the script cannot infer them from the TFLite model.

## Project language

Write documentation, code comments, docstrings, notebook explanations, and
diagnostic messages in English. Russian command phrases remain Russian when
they are dataset content.
