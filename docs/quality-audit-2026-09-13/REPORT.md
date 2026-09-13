# Model quality and evaluation audit

Reviewed on 2026-09-13. Scope: `D:\proj\ru-kws`, including uncommitted source,
notebooks, the ignored local dataset, live inference code, and distribution archives.
Repository HEAD: `f13d01b`. Existing implementation files were not modified.
This report does not apply to the unrelated projects in the initial workspace.

## Findings

### 1. [P2] Background mixing disregards the curated source exclusion

Location: [augmentation.py:134](D:/proj/ru-kws/src/ru_kws/data/augmentation.py:134).

`BackgroundMix` selects every training row whose label is `background`. It does
not consult `sources/background_manifest.jsonl` or propagate the source's
`use_for_mixing` decision through `parent_id`. Classification as a standalone
negative and permission to mix under speech are different dataset decisions.

This is present in the actual data: source `local_b85eac0c023a5aff` is reviewed,
has `use_as_negative=true`, and `use_for_mixing=false`. Nevertheless, all 46 of
its training crops are in the 700-file mixing pool. With mixing probability 0.8,
the probability of selecting this source for an eligible, non-silent example is
approximately `0.8 * 46 / 700 = 5.26%`.

Impact: augmented training uses audio explicitly excluded from speech mixtures.
Whether it reduces recognition accuracy needs an experiment; the selection
policy violation itself is confirmed. This affects `configs/augmented.yaml`,
not a baseline run where background augmentation is disabled.

Fix: resolve a crop's parent to the source manifest and honor `use_for_mixing`;
define an explicit policy for procedural noise and sources with missing metadata.
Keep these crops available as standalone background training examples.

### 2. [P2] PyTorch evaluation turns invalid model output into valid-looking metrics

Location: [clips.py:29](D:/proj/ru-kws/src/ru_kws/evaluation/clips.py:29).

`evaluate_clips` calls `argmax` immediately on model output without checking its
shape or finiteness. An all-NaN output selects class zero in the tested runtime.
The reproduction returns **accuracy 1.0** for three class-zero examples even
though the model returned no valid scores. A corrupted checkpoint or numerical
failure can therefore produce an ordinary confusion matrix and report instead
of an error.

The training epoch rejects nonfinite loss and the TFLite evaluators reject
nonfinite scores; the separate PyTorch report path lacks the equivalent guard.
This was reproduced with an intentionally broken fixture model, not observed in
the supplied TFLite model.

Fix: keep logits in a variable, require `[batch_size, number_of_classes]` and
`torch.isfinite(logits).all()`, and fail before counting predictions.

### 3. [P2] Direct training can bypass the split-integrity checks

Location: [train.py:35](D:/proj/ru-kws/src/ru_kws/train.py:35),
[dataset.py:11](D:/proj/ru-kws/src/ru_kws/data/dataset.py:11).

The documented standalone training CLI constructs its train and validation
loaders directly. This path checks paths, labels and durations, but does not
check cross-split identities or WAV checksums. The manifest hashes saved in a
run identify the manifests; they do not validate the audio or split separation.

Reproduction: the same WAV and speaker in train and val are accepted by the
exact loader path used by training. Calling the standalone validator on that
fixture correctly rejects the overlap. Consequently, the separate validation
command is a required but unenforced prerequisite for direct CLI users.

Impact: accidentally reused training material can influence both early stopping
and reported validation quality. **No such leakage was found in the current
dataset.** The normal Colab flow explicitly invokes validation and is protected
when run in order on unchanged data.

Fix: invoke a shared integrity preflight for the relevant splits before direct
training, or require a verified validation result bound to the current manifests
and audio. Avoid making training depend on the presence of a final test split.

### 4. [P2] Dataset validation can declare a nonfinite waveform valid

Location: [validate.py:22](D:/proj/ru-kws/src/ru_kws/data/validate.py:22).

The validator reads the WAV header and hashes file bytes without decoding the
samples. Three valid mono 16 kHz float WAV containers containing NaN samples pass
`validate_dataset`. Hash verification only proves that bytes match a recorded
hash; it does not prove that the signal is usable.

Impact: a dataset can pass the advertised preflight and later stop training in
`load_audio`, potentially after substantial work. This is not evidence of NaNs
in the current dataset: the additional audit decoded all 9,980 files and found
none.

Fix: decode audio during validation and apply the same nonempty/finite checks as
the loader. Report format, checksum, sample integrity, and duration exclusion
separately. A finite all-zero background is valid and should not be rejected.

### 5. [P2, distribution artifact] The source ZIP contains an older training pipeline

Artifact: [ru-kws-source.zip](D:/proj/ru-kws/dist/ru-kws-source.zip).
Packaging entry point: [package_source.py:7](D:/proj/ru-kws/scripts/package_source.py:7).

Compared 33 archived files against corresponding current files; 10 differ.
The ZIP has no `src/ru_kws/data/augmentation.py`, `configs/augmented.yaml`, or
`src/ru_kws/data/filtering.py`. Its `train.py` and `training/engine.py` predate
the current augmentation integration. It also lacks the new continuous evaluator.

Impact: uploading this existing ZIP runs a different training/evaluation recipe
from the working tree. This is an artifact freshness problem, not a defect in
the current packaging script. The current GitHub-cloning notebook does not use
this ZIP. The separate `continuous-evaluation-tools.zip` matches both current
evaluation scripts exactly.

Fix: regenerate the source ZIP when distributing this revision and include a
source revision/file-hash manifest. Do not overwrite a distributed artifact
without identifying the replacement version. The audit did not rebuild binaries
or archives.

## Evaluation limits and configuration observations

These are not counted as additional confirmed defects:

- **Default Colab training is still baseline.**
  [build_colab.py:180](D:/proj/ru-kws/scripts/build_colab.py:180) loads
  `configs/baseline.yaml`. Adding RIR files to the dataset ZIP does not activate
  RIR, background mixing, speed, gain or SpecAugment. Choose the augmented
  configuration explicitly for the corresponding experiment. There is no local
  checkpoint with which to establish which configuration produced the bundled
  TFLite model.
- **The evaluation population depends on window length.**
  [filtering.py:13](D:/proj/ru-kws/src/ru_kws/data/filtering.py:13) deliberately
  removes long commands from every split. At 3 seconds this removes 28 train,
  10 val and 4 test examples. Seven of the 10 excluded val examples are
  `stop_timer`. Counts and dropped records are reported, so this is not a hidden
  implementation error. Compare different window lengths on an explicitly fixed
  evaluation population, or report coverage alongside conditional accuracy.
  Do not silently describe skipped examples as measured misses.
- **Export contract is manually associated with the TFLite file.** The standalone
  evaluators and live app validate shapes and class counts, but cannot establish
  that a same-sized label map or sample-rate setting belongs to that model.
  The continuous notebook loads labels from a separately selected checkpoint;
  it does not prove that checkpoint produced the selected TFLite. Export a
  model-bound sidecar containing class order, audio/frontend settings, output
  kind, checkpoint hash and TFLite hash. For this audit the dataset and bundled
  label maps matched exactly, but the model's checkpoint provenance remains
  unverified.
- **Live and offline inference are not equivalent device evaluations.** The live
  app resamples each captured trailing window and may skip inference updates
  when computation is slow. Offline evaluation resamples the complete recording
  and evaluates every configured hop. The live app displays raw class scores,
  while offline event metrics use threshold/release/cooldown decoding. These
  behaviors are documented; offline recall and latency must not be presented as
  measured live-device performance.
- **No reviewed continuous ground truth was available in the recorder session.**
  The local `dist/recordings` continuous session has 29 draft keyboard regions,
  4,809,546 samples at 44,100 Hz, and no reported capture error. Draft annotations
  and a single short session cannot establish final event accuracy or a rare
  false-alarm rate. The scorer correctly distinguishes draft metrics and uses
  original-rate annotation timestamps.
- **README configuration claims are stale.** The opening section says that noise
  mixing, SpecAugment and TFLite export are unimplemented, although source now
  implements them. It also describes a `master` code reference and a false final
  test default, while the current generator uses `dev` and `RUN_FINAL_TEST=True`.
  Align the README with the chosen training recipe to avoid mistaken runs.

## Actual local model result

Evaluated the existing waveform-input FP32 TFLite model through the live app's
runtime, using the clip evaluator's `fit_audio` padding/cropping policy. No
microphone was opened and no final-test model predictions were computed.

Model SHA256: `398ec1f633d22f7d31795652f4faee7779e0816dab0d1916d1bdb0caf00115bd`.

| Measurement | Result |
| --- | ---: |
| Validation manifest | 1,493 clips |
| Evaluated after duration filtering | 1,483 clips |
| Skipped long commands | 10 |
| Correct predictions | 1,460 |
| Clip accuracy | 98.4491% |
| `unknown` recall | 430/450 = 95.5556% |
| `unknown` predicted as `start_timer` | 17 |
| `start_timer` precision | 149/166 = 89.7590% |

The remaining errors are one `back` and two `stop_timer` clips predicted as
`unknown`; one `unknown` each predicted as `repeat`, `time_left` and `background`.
This validates execution and identifies a concrete confusion pattern. It does
not establish accuracy on real speakers or show that any of the code findings
caused these errors. The bundled model did not produce nonfinite outputs during
this run. Full confusion matrix, skipped paths and hashes:
[local-model-val.json](D:/proj/ru-kws/docs/quality-audit-2026-09-13/local-model-val.json).

## Coverage and reproducibility

- Main test suite: **33 passed**, including train/checkpoint/evaluate roundtrip,
  per-sample loss aggregation, confusion orientation, duration policy, audio
  augmentation, frontend equivalence, notebook consistency and event matching.
- Recorder unit tests: **9 passed**.
- Live engine unit tests: **9 passed**, including inference with the actual model
  and loading it from a Unicode path. Hardware/UI smoke tests were not run.
- Current dataset: **9,980 WAV files decoded**. No nonfinite samples, invalid
  formats, checksum mismatches, duration metadata mismatches or cross-split PCM
  duplicates. No cross-split speaker, parent, background-source, or unknown-text
  group overlaps were found. Missing/unreliable identity metadata cannot prove
  that acoustically similar recordings are independent.
- Reviewed source modules, both notebooks and generators, RIR preparation,
  package builders, recorder and ignored live-monitor sources. Numerical STFT
  replacement tests passed. A new TFLite conversion was not run; the local
  checkpoint needed for full PyTorch-to-TFLite model equivalence was unavailable.
- No mass retraining, audio relabeling, dataset changes, microphone capture,
  deployment or changes to pre-existing code were performed.

Run from `D:\proj\ru-kws`:

```powershell
.venv/Scripts/python.exe -X utf8 -m pytest -q
.venv/Scripts/python.exe -X utf8 -m unittest discover -s voice_recorder -p test_record.py -v
local_streaming_test/.venv/Scripts/python.exe -X utf8 -m unittest discover -s local_streaming_test -p test_engine.py -v
.venv/Scripts/python.exe -X utf8 docs/quality-audit-2026-09-13/reproduce.py
local_streaming_test/.venv/Scripts/python.exe -X utf8 docs/quality-audit-2026-09-13/model_probe.py
```

[checks.json](D:/proj/ru-kws/docs/quality-audit-2026-09-13/checks.json) includes
failure-probe results, dataset checks, forbidden mixing crops, archive comparisons
and SHA256 fingerprints of reviewed source files. The scripts intentionally
demonstrate current failure behavior; they are not regression tests asserting
the desired fixed behavior. Audit outputs may be replaced by rerunning these
scripts; source data and model files are read-only.
