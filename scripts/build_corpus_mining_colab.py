"""Build the thin Colab interface for scripts/mine_corpus_commands.py."""

import json
from pathlib import Path
from textwrap import dedent


def build():
    cells = []

    def add(kind, source):
        cell = dict(cell_type=kind, id=f"mining-{len(cells):02d}", metadata={},
                    source=dedent(source).strip() + "\n")
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", """
    # Mine real Russian commands from a speech corpus

    Transcript search → listening and crop review → training additions.
    **The default workflow runs entirely on CPU. No GPU quota or ASR model is needed.**
    Nothing is accepted automatically. Exact timer phrases may be rare: the tool does
    not synthesize missing commands or count partial phrases as positive examples.
    Existing synthetic tests and Vanya remain separate and are never modified.

    Default source: [Golos](https://github.com/salute-developers/golos/tree/master/golos).
    Check the [corpus license](https://github.com/salute-developers/golos/blob/master/license/en_us.pdf)
    for your intended use. The OPUS archive is about 20.5 GB on **local Colab disk**;
    it is not unpacked. Only matching contexts, accepted crops and review state go to Drive.
    Audio is decoded one utterance at a time (maximum 60 seconds).
    """)
    add("markdown", "## 1. Setup and persistent configuration")
    add("code", """
    from pathlib import Path
    import os, sys, json, subprocess, shutil
    from google.colab import drive, output

    drive.mount('/content/drive')
    output.enable_custom_widget_manager()
    REPO = Path('/content/ru-kws')
    ALIGNMENT_MODE = 'manual'  # 'manual': CPU, no ASR; 'whisper': optional word timestamps
    MODEL = 'small'           # used only with ALIGNMENT_MODE = 'whisper'
    DEVICE = 'cpu'            # optional Whisper: 'cpu' or 'cuda' if you have GPU quota
    if ALIGNMENT_MODE not in {'manual', 'whisper'}:
        raise ValueError("ALIGNMENT_MODE must be 'manual' or 'whisper'")
    if not (REPO / '.git').exists():
        subprocess.run(['git', 'clone', '--branch', 'dev',
                        'https://github.com/mole88/ru-kws.git', str(REPO)], check=True)
    else:
        subprocess.run(['git', '-C', str(REPO), 'pull', '--ff-only'], check=True)
    packages = ['soundfile>=0.12', 'scipy', 'ipywidgets']
    if ALIGNMENT_MODE == 'whisper':
        packages += ['faster-whisper>=1.1,<2']
        if DEVICE == 'cuda':
            packages += ['nvidia-cublas-cu12', 'nvidia-cudnn-cu12>=9,<10']
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True)

    # Child ASR processes receive CUDA library paths before Python starts.
    if ALIGNMENT_MODE == 'whisper' and DEVICE == 'cuda':
        import nvidia.cublas.lib, nvidia.cudnn.lib
        cuda_dirs = [str(Path(next(iter(module.__path__))))
                     for module in (nvidia.cublas.lib, nvidia.cudnn.lib)]
        os.environ['LD_LIBRARY_PATH'] = ':'.join(cuda_dirs + [os.environ.get('LD_LIBRARY_PATH', '')])
    sys.path.insert(0, str(REPO / 'scripts'))
    import mine_corpus_commands as mining

    ROOT = Path('/content/drive/MyDrive/russian_commands/corpus_mining/golos_v001')
    CORPUS = 'golos'
    SOURCE_MODE = 'archive'  # 'archive' or 'manifest'
    ARCHIVE_URL = 'https://cdn.chatwm.opensmodel.sberdevices.ru/golos/golos_opus.tar'
    ARCHIVE = Path('/content/corpus/golos_opus.tar')
    # Alternative: supply a TRAIN JSONL with audio_filepath/path, text, optional
    # duration and speaker_id/client_id. Audio paths are relative to AUDIO_ROOT.
    MANIFEST = Path('/content/drive/MyDrive/corpus/train.jsonl')
    AUDIO_ROOT = MANIFEST.parent
    MAX_PER_LABEL = 1000  # shortlisted sources, not an accepted-clip quota
    MAX_PER_SPEAKER = 30  # applies only when the corpus supplies a speaker ID
    PREPARE_LIMIT = 200   # sources per execution; 0 = all remaining
    RETRY_ERRORS = False
    EXPORT_ROOT = ROOT.parent / 'golos_train_additions_v001'  # must be a new directory

    mining.initialize(ROOT, corpus=CORPUS)
    print('Persistent state:', ROOT)
    print(json.dumps(mining.summary(ROOT), ensure_ascii=False, indent=2))
    """)
    add("markdown", """
    ## 2. Download the archive (archive mode only)

    This needs free disk, not 20 GB of RAM. Interrupted downloads resume with `wget -c`.
    A new Colab runtime loses `/content/corpus`; redownload only when you need more
    source audio. Listening review and export use saved Drive contexts and need no archive.
    For another corpus with extracted audio, use `SOURCE_MODE = 'manifest'` and skip this cell.
    """)
    add("code", """
    if SOURCE_MODE == 'archive':
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        complete = ARCHIVE.with_suffix(ARCHIVE.suffix + '.complete')
        if not complete.exists():
            # Reserve enough space for the official 20.5 GB archive, with resume allowance.
            already = ARCHIVE.stat().st_size if ARCHIVE.exists() else 0
            required = max(0, 21_000_000_000 - already) + 2_000_000_000
            if shutil.disk_usage(ARCHIVE.parent).free < required:
                raise RuntimeError('Not enough local Colab disk for the Golos OPUS archive')
            subprocess.run(['wget', '-c', '--tries=3', '--timeout=60', '--progress=dot:giga',
                            '-O', str(ARCHIVE), ARCHIVE_URL], check=True)
            complete.write_text(ARCHIVE_URL, encoding='utf-8')
        print('Archive GB:', round(ARCHIVE.stat().st_size / 1e9, 2))
    """)
    add("markdown", """
    ## 3. Search transcripts (CPU)

    Archive scanning can take several minutes; progress is printed every 10 seconds.
    Only manifests with a `train` path component are used automatically. The tar member
    cache is cleared while scanning, and audio files are not extracted or decoded.
    Repeating this stage adds missing candidates and preserves decisions. You can raise
    `MAX_PER_LABEL` later; repeating the scan does not produce duplicate source IDs.
    """)
    add("code", """
    if SOURCE_MODE == 'archive':
        result = mining.scan_archive(ROOT, ARCHIVE, MAX_PER_LABEL, MAX_PER_SPEAKER)
    elif SOURCE_MODE == 'manifest':
        result = mining.scan_manifest(ROOT, MANIFEST, AUDIO_ROOT,
                                      max_per_label=MAX_PER_LABEL, max_per_speaker=MAX_PER_SPEAKER)
    else:
        raise ValueError("SOURCE_MODE must be 'archive' or 'manifest'")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    """)
    add("markdown", """
    ## 4. Prepare candidates (CPU by default, no ASR)

    Short utterances whose full transcript is exactly a command need no ASR.
    With `ALIGNMENT_MODE = 'manual'`, longer sentences containing a target phrase
    are saved for listening; you select crop start/end in stage 5. Corpus transcripts
    identify the phrase but do not provide word boundaries. No model is downloaded
    or loaded. Short whole-command utterances already have full-recording bounds.

    Optional `ALIGNMENT_MODE = 'whisper'` estimates boundaries with
    [faster-whisper word timestamps](https://github.com/SYSTRAN/faster-whisper#word-level-timestamps).
    This also works with `DEVICE = 'cpu'`, but runs more slowly. These are estimates,
    not forced alignment or a guarantee of the correct phrase. Nothing is auto-accepted.

    **Resume:** reconnect Drive, rerun setup and this stage with the same `ROOT`.
    Saved candidates and decisions are skipped. You can stop a running cell; the
    current source may be repeated, while completed sources stay saved. Changing
    alignment mode affects unprocessed sources; existing clips and decisions remain.
    Optional ASR runs in a child process so CUDA library paths are set before Python starts.
    """)
    add("code", """
    command = [sys.executable, '-u', str(REPO / 'scripts/mine_corpus_commands.py'),
               '--root', str(ROOT), 'prepare', '--model', MODEL, '--device', DEVICE,
               '--alignment', ALIGNMENT_MODE,
               '--limit', str(PREPARE_LIMIT)]
    if SOURCE_MODE == 'archive' and ARCHIVE.exists():
        command += ['--archive', str(ARCHIVE)]
    if RETRY_ERRORS:
        command += ['--retry-errors']
    worker = subprocess.Popen(command, env=os.environ.copy(), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in worker.stdout:
            print(line, end='', flush=True)
        return_code = worker.wait()
    except KeyboardInterrupt:
        worker.terminate()
        try:
            worker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.wait()
        print('Stopped. Completed sources and review decisions remain on Drive.')
    else:
        if return_code:
            raise RuntimeError(f'Preparation failed (exit {return_code}); inspect the log above')
    finally:
        worker.stdout.close()
    print(json.dumps(mining.summary(ROOT), ensure_ascii=False, indent=2))
    """)
    add("markdown", """
    ## 5. Listen and review (CPU; repeat as needed)

    Listen to the source, edit start/end seconds if necessary, click **Preview crop**,
    listen to that crop, then **Accept** or **Reject**. **Later** leaves the clip pending.
    Accept only a complete target phrase with no neighbouring speech. Reject incorrect
    transcripts, cut words, unnatural fragments, music, silence or poor recording quality.
    A word occurring inside a sentence is a candidate, not automatically a valid command.
    Each decision is saved immediately. Rerun the cell to revisit postponed candidates.
    """)
    add("code", "mining.review_widget(ROOT)")
    add("markdown", """
    ## 6. Export accepted training additions

    Exports mono 16 kHz PCM16 audio, `manifest.jsonl`, `splits/train.jsonl`, `labels.json`
    and a provenance/count report. Crop duration must be 0.35–2.6 seconds. Duplicate
    WAV hashes are removed. Choose a new `EXPORT_ROOT` for each snapshot.

    This is a **training addition bundle**, not a full train/validation/test dataset.
    Merge it into training later, retaining source/parent/speaker groups and checking
    overlaps against existing splits. If speaker IDs are unavailable, the export marks
    them as unknown and keeps the corpus in a single group; it cannot claim speaker-disjoint
    validation. Do not move these reviewed training clips to Vanya or synthetic tests.
    """)
    add("code", """
    print(json.dumps(mining.summary(ROOT), ensure_ascii=False, indent=2))
    report = mining.export_dataset(ROOT, EXPORT_ROOT)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print('Training additions:', EXPORT_ROOT)
    """)
    notebook = dict(cells=cells, metadata=dict(
        colab=dict(name="corpus_command_mining.ipynb", provenance=[]),
        kernelspec=dict(display_name="Python 3", language="python", name="python3"),
        language_info=dict(name="python")), nbformat=4, nbformat_minor=5)
    path = Path(__file__).resolve().parents[1] / "notebooks/corpus_command_mining.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    print(build())
