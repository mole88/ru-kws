# Russian KWS

A minimal BC-ResNet training project for Russian voice commands.
The core code is a Python package; Colab runs standard CLI commands.

Implemented: JSONL datasets, split validation, padding/cropping, log-mel features,
BC-ResNet, training with early stopping, best/last checkpoints, clip evaluation,
and standalone validation of externally exported TFLite models.
Online noise mixing, SpecAugment, TFLite export and continuous event evaluation are supported.
Training resume is not implemented yet. The model trains from scratch; pretrained weights
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
- `scripts/clean_dataset.py`: local cleanup into a fresh dataset with Vanya held out for testing.
- `scripts/compare_tflite_models.py`: strict paired old/new comparison on clean synthetic and Vanya tests.

## Training in Colab

[Open the notebook in Colab](https://colab.research.google.com/github/mole88/ru-kws/blob/dev/notebooks/ru_kws_training.ipynb).

Prepare the dataset locally with `scripts/clean_dataset.py`, then upload its new
archive to Drive. See [the cleanup and evaluation guide](docs/clean_dataset.md)
for a Windows command and output details. Enable a GPU for training.
The code is downloaded from GitHub (`REPO_REF`, currently `dev`); a source ZIP is not required.
Evaluation scripts are loaded from the same checkout; the notebook does not
embed or overwrite their source code.
Set the dataset path and a new `RUN_NAME` in the configuration.
Checkpoints and reports are saved to Drive. Check `RUN_FINAL_TEST` before running
all cells: final test evaluation is disabled by default.
Use validation for tuning and reserve test evaluation for fixed model settings.
For reproducibility, set `REPO_REF` to a commit SHA.

`python scripts/build_colab.py` rebuilds the notebook from the generator template.
This replaces notebook edits, settings, and saved outputs; back up local changes
before rebuilding.

Sections 5–8 separate validation clips, synthetic final tests, Vanya held-out
clips, annotated continuous sessions, podcast negatives and baseline comparison.
Vanya is stored in its own evaluation manifest, outside all dataset splits.
`RUN_FINAL_TEST` enables dataset tests; `RUN_VANYA_TEST` independently enables
Vanya's test. Evaluation
raises on missing or damaged files; it does not filter the test set.
For saved-model evaluation, run setup, set the existing `RUN_NAME` and exact
`EXISTING_TFLITE_PATH`, then run section 5.1 and the desired evaluation sections.
Enable `RUN_BASELINE_COMPARISON` for paired final clip comparison. It checks
checkpoint labels/audio settings and saves per-class accuracy, all-class and
supported-class macro F1, paired predictions and both confusion matrices under
`runs/model_comparisons/`. Continuous and podcast comparisons use identical
decoder settings for both models.

## TFLite validation

`scripts/evaluate_tflite.py` reads `labels.json` and `splits/val.jsonl` independently
of the training notebook. Install its dependencies in the evaluation environment:

```shell
python -m pip install ai-edge-litert numpy soundfile scipy scikit-learn matplotlib tqdm
```

After cloning the repository and extracting the dataset, run:

```shell
python scripts/evaluate_tflite.py --model /path/to/model.tflite --dataset-root /path/to/dataset --split val --window-seconds 3 --long-commands error --label-smoothing 0 --output-dir /path/to/evaluation/val_001
```

Replace the example paths with your actual paths. Use `--split test` and a new
output directory for test evaluation. Existing results are not overwritten.
Outputs include metrics, predictions, skipped recordings, and a confusion matrix
in CSV and PNG formats. See `python scripts/evaluate_tflite.py --help` for options.

`predictions.csv` includes `confidence` (the predicted class probability),
`true_class_probability`, and `p_<label>` for every class, all on a 0-1 scale.
Add `--print-predictions` to also print per-clip predictions and probabilities as
percentages. These model probabilities are not calibrated correctness guarantees.

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

## Continuous TFLite evaluation

Use section 6 of the main notebook for recorder `continuous.wav` sessions.
Configure `CONTINUOUS_CASES` with WAV/annotation pairs and enable
`RUN_CONTINUOUS_TESTS`. The notebook reads the matching checkpoint's label/audio
settings and saves event reports and probability timelines to Drive.
Section 7 evaluates a command-free podcast with `RUN_PODCAST_TEST`,
`PODCAST_WAV` and `PODCAST_IS_NEGATIVE=True`. Every emitted command then counts
as a false positive. No retraining is required.

Alternatively run `scripts/evaluate_continuous_tflite.py` alongside
`scripts/evaluate_tflite.py` with dependencies `ai-edge-litert numpy soundfile
scipy tqdm`. Example for a model with built-in frontend:

```shell
python scripts/evaluate_continuous_tflite.py --model model.tflite --wav recordings/SESSION/continuous.wav --annotations recordings/SESSION/continuous.reviewed.json --labels labels.json --sample-rate 16000 --window-seconds 3 --hop-seconds 0.1 --resample --threshold 0.5 --min-consecutive 2 --release-seconds 0.3 --cooldown-seconds 1 --late-tolerance 1 --output-dir reports/continuous_001 --print-predictions
```

For command-free negative audio, replace `--annotations ...` with
`--negative-only`. The output records this assumption and reports FP/hour.
Use `--mono mean` to downmix stereo input; the default rejects stereo.

Use `continuous.json` and `--allow-draft` for a provisional run with keyboard
annotations. Reviewed annotations come from the recorder's `import-labels`
command. The original WAV sample rate and frame count must match annotations;
times are converted to seconds before resampling. Existing reports are never
overwritten. Models with separate features or streaming state are not supported.

The decoder evaluates trailing windows, left-padding at the start, including the
last partial hop but adding no post-recording tail. Only a top-1 command above
threshold can fire; `unknown` and `background` do not emit events. Consecutive
windows confirm a candidate. After firing, that class must be ineligible for the
release time and satisfy its cooldown before firing again. These defaults are
an explicit starting policy, not tuned thresholds or a reproduction of firmware.

Detections are matched chronologically, once each, to the earliest-ending
unmatched same-class interval containing the detection time after applying
`--early-tolerance` before its start and `--late-tolerance` after its end.
Wrong labels and duplicate detections count as false positives. An undetected
annotation counts as a false negative. Latency is detection time minus phrase
end (negative during the phrase); runtime and audio-device latency are excluded.
FP/hour uses the entire recording, not a separately verified negative subset.
This offline whole-file resampling/window simulation is not a real-time benchmark.

Outputs: `metrics.json` with overall/per-class precision, recall, F1 and latency;
`events.csv` with annotation matches/misses; `detections.csv` with matched/extra
events and confidence; `windows.csv` with all class probabilities. No per-window
accuracy is reported, since overlapping windows are not independent events.
The notebook also creates `timeline.png`. A 109-second pilot cannot establish
a reliable rare-false-alarm rate. Keep final evaluation sessions separate from
decoder tuning. Rebuild the main notebook with `python scripts/build_colab.py`.

## Voice recorder

For live microphone inference with per-class confidence bars and an audio/prediction
log, launch `local_streaming_test/RU-KWS-Live.exe`. The application includes the
Python runtime and supports waveform-input TFLite models with a built-in frontend.
See [the live monitor guide](local_streaming_test/README.md).

Double-click `dist/RU-KWS-Recorder.exe` for the recording setup menu on Windows
x64. Python is bundled. Share `dist/RU-KWS-Recorder-Windows-x64.zip` with other
recorders; extract it to a writable folder first. Personal sessions are stored
separately in `dist/recordings`, and are excluded from the application ZIP.
See [the recorder guide](voice_recorder/README.md) for recording, annotation,
command-line options and rebuilding the portable executable.

## Project language

Write documentation, code comments, docstrings, notebook explanations, and
diagnostic messages in English. Russian command phrases remain Russian when
they are dataset content.

## Run storage and notebook stages

The main notebook has eight numbered stages: setup (1), training (2), PyTorch
evaluation (3), TFLite export (4), clip evaluation (5), continuous commands (6),
podcast negatives (7), and baseline comparison (8). For existing checkpoint
export, run setup then section 4. For saved-model tests, run setup, section 5.1,
then the desired evaluation sections. Edit `scripts/build_colab.py`, then
regenerate the notebook. Final evaluation flags are disabled by default.

```text
runs/<RUN_NAME>/
  metadata/                 # config, labels, environment, dataset and validation
  checkpoints/best.pt
  checkpoints/last.pt
  training/history.csv
  training/curves.png
  exports/<UTC-id>/         # model and export provenance
  measurements/
    pytorch/<split>/<UTC-id>/
    tflite/<split>/<UTC-id>/
    equivalence/<UTC-id>/
    clips/<view>/<UTC-id>/
    continuous/<UTC-id>/    # metrics, windows, events, detections, timeline
    podcast/<UTC-id>/       # false positives and probabilities
    continuous_comparison/...
    podcast_comparison/...
```

Each notebook measurement gets a fresh UTC timestamp plus random suffix and a
`measurement.json` with input hashes, settings and code revision. Repeating a
measurement preserves earlier results. Existing flat runs remain readable through
the notebook's checkpoint/history fallback; old artifacts are not moved.
The training CLI uses this same checkpoint layout. Pass the full checkpoint path
to `python -m ru_kws.evaluate --checkpoint ...`.

Direct training validates train/val WAV contents, checksums and split separation
before creating a run. The standalone validator also checks test. WAV NaN/Inf and
nonfinite PyTorch logits are rejected. Background source crops require explicit
`use_for_mixing=true` on their parent in `sources/background_manifest.jsonl`;
missing source permissions exclude the crop. Procedural noise and legacy rows
without source parents remain eligible unless explicitly disabled. These rules
affect mixing only, leaving standalone negative examples available for training.
