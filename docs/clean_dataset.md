# Local dataset cleanup and model evaluation

The main notebook expects a dataset prepared by `scripts/clean_dataset.py`.
Cleanup runs locally once. Evaluation uses the resulting manifests without
excluding missing or damaged files during scoring.

## Prepare a new version on Windows

From `D:\proj\ru-kws`, install the standalone cleaner dependencies:

```powershell
python -m pip install numpy scipy soundfile
```

Run this command after setting the two input paths to your local copies:

```powershell
python scripts/clean_dataset.py `
  --dataset "D:\proj\ru-kws\datasets\russian_commands_v002_expanded-20260927T131042Z-1-001.zip" `
  --vanya-recordings "D:\proj\ru-kws\datasets\recordings_vanya.zip" `
  --output-dir "D:\proj\ru-kws\datasets\russian_commands_v003_clean" `
  --zip
```

Both inputs may be ZIP files or directories. The output directory and ZIP must
be new. Inputs are never modified. You need enough disk space for temporary
extraction, the cleaned WAVs and, with `--zip`, the output archive.

The cleaner checks every WAV referenced by train/val/test, verifies a source
checksum when present, averages channels, converts to mono 16 kHz PCM16, and
trims speech edges with a 150 ms margin. Speech shorter than 0.35 seconds or
longer than 2.6 seconds after trimming is excluded. Background recordings are
not speech-trimmed or rejected by speech duration. Adjust `--min-seconds`,
`--max-seconds` and `--margin-ms` if needed; the maximum cannot exceed the
three-second model window. This activity check does not verify spoken content.

Missing, unreadable, silent speech, checksum mismatches and same-split duplicates
are listed in `excluded.jsonl`. The script stops on cross-split audio, ID or group
leakage. An interrupted or unsuccessful output has `complete: false` in
`cleaning_report.json` and must not be used; choose a fresh directory for a rerun.

Vanya's recorder archive must contain `manifest.jsonl` entries with `wav`,
`label`, and `accepted: true`. Rejected or capture-error entries are excluded.
Usable recordings are included only in `evaluation/vanya_test.jsonl`.
They are absent from train, val, the dataset test split, and its main manifest.
They are stored under `audio/vanya/` in the same archive for convenience. Do not use
Vanya's test voice as an XTTS training reference. Training augmentation assets
and source metadata are copied from the original dataset.

Outputs include:

```text
russian_commands_v003_clean/
  labels.json
  manifest.jsonl
  cleaning_report.json
  excluded.jsonl
  audio/dataset/...
  audio/vanya/...
  splits/train.jsonl
  splits/val.jsonl
  splits/test.jsonl
  evaluation/synthetic_test.jsonl
  evaluation/vanya_test.jsonl
```

The synthetic view excludes human recordings from the dataset test split.
Vanya is an independent test collection with separate counts and metrics.
The cleaner also moves any Vanya test rows already present in the input dataset
to that collection, and rejects audio shared by the two tests. Review the
exclusion counts before uploading `russian_commands_v003_clean.zip` to Drive.
Archives from an older cleaner with a combined test must be rebuilt in a new
output directory; the notebook checks `vanya_separate_test` in the report.

## Main notebook stages

Set `DATASET_ZIP` to the cleaned archive. The default Drive path is
`/content/drive/MyDrive/russian_commands/russian_commands_v003_clean.zip`.

The notebook has eight stages: setup, training, PyTorch evaluation, TFLite export,
clip evaluation, continuous command tests, podcast false positives, and baseline
comparison. Run setup first. For a saved model, set `RUN_NAME` to its existing run
and `EXISTING_TFLITE_PATH` to its exact export, then run section 5.1 and the desired
evaluation sections. Skip training and export when evaluating a saved model.

Section 5 scores validation separately from the synthetic and Vanya final tests.
`RUN_FINAL_TEST` enables the dataset final tests; `RUN_VANYA_TEST` independently
enables Vanya's test. Section 8 separately enables the
paired final clip comparison with `RUN_BASELINE_COMPARISON`. Keep both disabled
while tuning. Comparison requires the matching baseline and candidate checkpoints
so labels and audio settings can be checked. The baseline defaults to
`bcresnet_run_005` and its supplied September 13 export.

Every clean clip must be evaluated. A missing or invalid file raises an error.
Class reports, confusion matrices and prediction CSVs are saved to new report
directories. Paired comparison includes counts of improved/regressed clips and
macro F1 over classes with nonzero support. The ordinary all-class macro F1 is
also retained in `comparison.json`; on Vanya it includes classes with no examples.

## Continuous sessions and podcast negatives

In section 6, populate `CONTINUOUS_CASES` with any number of WAV/annotation pairs
and enable `RUN_CONTINUOUS_TESTS`. Annotation regions must use the original WAV
frame clock. Reviewed annotations are required by default; `ALLOW_DRAFT=True`
explicitly marks keyboard annotations as provisional. Reports include matched
commands, misses, extra detections, latency, per-class event metrics, and timelines.

In section 7, set `PODCAST_WAV`, `PODCAST_IS_NEGATIVE=True`, and
`RUN_PODCAST_TEST=True`. This assumes the audio is command-free: every emitted
command is counted as a false positive, with total counts and false positives
per hour. If the podcast contains intentional commands, use reviewed annotations
in section 6 instead. WAV input is supported; convert other formats beforehand.
Stereo and non-16-kHz continuous recordings are explicitly downmixed/resampled.

With `RUN_BASELINE_COMPARISON=True`, section 8 also compares both models on the
configured continuous sessions and podcast when their flags are enabled. Both
models use the same threshold, hop, confirmation, release, cooldown and matching
tolerances. Keep final sessions and podcasts separate from threshold tuning.
Longer recordings give more useful false-alarm evidence than isolated clips.
