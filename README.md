# Russian KWS

A minimal BC-ResNet training project for Russian voice commands.
The core code is a Python package; Colab runs standard CLI commands.

Implemented: JSONL datasets, split validation, padding/cropping, log-mel features,
BC-ResNet, training with early stopping, best/last checkpoints, clip evaluation,
and standalone validation of externally exported TFLite models.
Online noise mixing, SpecAugment, streaming decoding, training resume, and TFLite
export are not implemented yet. The model trains from scratch; pretrained weights
are not downloaded.

## Dataset

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
