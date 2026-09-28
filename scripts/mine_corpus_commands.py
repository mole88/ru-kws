"""Stream transcript manifests, locate command crops, review them and export train additions.

Colab entry point: notebooks/corpus_command_mining.ipynb. No corpus-wide audio decoding
or automatic acceptance. The SQLite database and selected context WAVs belong on Drive.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import tarfile
import tempfile
import time

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

COMMANDS = {
    "next": "Дальше", "back": "Назад", "repeat": "Повторить",
    "start_timer": "Запустить таймер", "stop_timer": "Остановить таймер",
    "time_left": "Сколько осталось",
}
LABELS = {name: i for i, name in enumerate([*COMMANDS, "unknown", "background"])}
SAMPLE_RATE = 16000


def tokens(text):
    return re.findall(r"[а-яa-z0-9]+", str(text).casefold().replace("ё", "е"))


def matches(text):
    words = tokens(text)
    return [label for label, phrase in COMMANDS.items()
            if any(words[i:i + len(tokens(phrase))] == tokens(phrase)
                   for i in range(len(words)))]


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


@contextmanager
def database(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "state.sqlite3", timeout=60)
    db.row_factory = sqlite3.Row
    # Rollback journals work on Drive; WAL requires filesystem locking/shared memory.
    db.executescript("""
        CREATE TABLE IF NOT EXISTS sources (
            id TEXT PRIMARY KEY, audio_key TEXT, basename TEXT, status TEXT, data TEXT);
        CREATE INDEX IF NOT EXISTS source_basename ON sources(basename);
        CREATE TABLE IF NOT EXISTS targets (
            source_id TEXT, label TEXT, PRIMARY KEY(source_id, label));
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY, source_id TEXT, label TEXT, status TEXT,
            start REAL, end REAL, method TEXT, note TEXT);
    """)
    try:
        yield db
        db.commit()
    finally:
        db.close()


def initialize(root, corpus="golos", source_split="train", min_seconds=0.35,
               max_seconds=2.6, margin=0.15, max_source_seconds=60.0):
    if source_split != "train":
        raise ValueError("This tool exports training additions. Supply corpus TRAIN data only.")
    if not corpus.strip() or not 0 < min_seconds <= max_seconds <= max_source_seconds:
        raise ValueError("Invalid corpus name or duration limits")
    if not 0 <= margin < max_seconds / 2:
        raise ValueError("Invalid crop margin")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    config = dict(version=1, corpus=corpus, source_split=source_split,
                  min_seconds=min_seconds, max_seconds=max_seconds, margin=margin,
                  max_source_seconds=max_source_seconds, commands=COMMANDS)
    path = root / "config.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != config:
        raise ValueError("ROOT has different mining settings. Restore its settings or use a new ROOT.")
    if not path.exists():
        atomic_json(path, config)
    with database(root):
        pass
    return config


def settings(root):
    return json.loads((Path(root) / "config.json").read_text(encoding="utf-8"))


def safe_key(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError(f"Expected a relative audio path: {value}")
    return str(path)


def scan_lines(root, lines, manifest_name, audio_root=None, max_per_label=1000,
               max_per_speaker=30):
    """Keep only matching metadata; the million-row manifest is never materialized."""
    config = settings(root)
    if max_per_label < 1 or max_per_speaker < 1:
        raise ValueError("Candidate limits must be positive")
    with database(root) as db:
        counts = Counter(dict(db.execute("SELECT label, COUNT(*) FROM targets GROUP BY label")))
        speakers = Counter()
        for row in db.execute("SELECT t.label, s.data FROM targets t JOIN sources s ON s.id=t.source_id"):
            speaker = json.loads(row["data"]).get("speaker_group")
            if speaker:
                speakers[(row["label"], speaker)] += 1
        added = invalid = 0
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                text = row.get("text", row.get("sentence", row.get("transcript", "")))
                found = matches(text)
                if not found:
                    continue
                if str(row.get("split", "train")).casefold() != "train":
                    continue
                relative = safe_key(row.get("audio_filepath", row.get("path", "")))
                if relative == ".":
                    raise ValueError("Missing audio_filepath/path")
                # Golos paths are relative to the manifest's directory.
                key = safe_key(str(PurePosixPath(manifest_name).parent / relative))
                source_id = "corpus_" + digest(f"{config['corpus']}\n{key}\n{text}")
                speaker_id = row.get("speaker_id", row.get("client_id", row.get("speaker")))
                speaker = f"{config['corpus']}:speaker:{speaker_id}" if speaker_id is not None else None
                duration = float(row["duration"]) if row.get("duration") is not None else None
                if duration is not None and (not math.isfinite(duration) or duration <= 0
                                            or duration > config["max_source_seconds"]):
                    continue
                found = [label for label in found if counts[label] < max_per_label
                         and (not speaker or speakers[(label, speaker)] < max_per_speaker)
                         and not db.execute("SELECT 1 FROM targets WHERE source_id=? AND label=?",
                                            (source_id, label)).fetchone()]
                if not found:
                    continue
                data = dict(id=source_id, text=text, audio_key=key, corpus=config["corpus"],
                            corpus_split="train", manifest=manifest_name, speaker_group=speaker,
                            duration_hint=duration,
                            local_path=str((Path(audio_root) / relative).resolve()) if audio_root else None)
                db.execute("INSERT OR IGNORE INTO sources VALUES (?, ?, ?, 'new', ?)",
                           (source_id, key, PurePosixPath(relative).stem, json.dumps(data, ensure_ascii=False)))
                for label in found:
                    db.execute("INSERT INTO targets VALUES (?, ?)", (source_id, label))
                    counts[label] += 1
                    if speaker:
                        speakers[(label, speaker)] += 1
                added += 1
            except (ValueError, TypeError, KeyError, AttributeError) as error:
                invalid += 1
                if invalid <= 3:
                    print(f"Skipping malformed row {line_number}: {error}", flush=True)
            finally:
                if line_number % 10000 == 0:
                    db.commit()
                    print(f"Transcripts: {line_number:,}; targets: {dict(counts)}", flush=True)
        print(f"Scanned {manifest_name}: added {added} sources; malformed {invalid}", flush=True)
    return summary(root)


def tar_members(path):
    """Skip local TAR payloads by seeking, without tarfile's unbounded member cache.

    Compressed TAR also works, but seeking requires decompression of skipped data.
    """
    last_log = time.monotonic()
    with tarfile.open(path, "r:*") as archive:
        count = 0
        while True:
            member = archive.next()
            if member is None:
                break
            count += 1
            if time.monotonic() - last_log > 10:
                print(f"Archive: {count:,} members, {archive.offset / 1e9:.2f} GB scanned", flush=True)
                last_log = time.monotonic()
            yield archive, member
            archive.members.clear()


def scan_archive(root, archive_path, max_per_label=1000, max_per_speaker=30):
    """Discover TRAIN JSONL manifests in Golos TAR; never extract the corpus."""
    found = 0
    for archive, member in tar_members(archive_path):
        name = member.name.casefold()
        if not member.isfile() or not (name.endswith(".jsonl") or
                                      ("manifest" in name and name.endswith(".json"))):
            continue
        if not any(part.startswith("train") for part in PurePosixPath(name).parts):
            print(f"Skipping non-training manifest: {member.name}", flush=True)
            continue
        safe_key(member.name)
        # Decode lines independently; do not buffer a whole training manifest.
        with archive.extractfile(member) as stream:
            scan_lines(root, (line.decode("utf-8-sig") for line in stream), member.name, max_per_label=max_per_label,
                       max_per_speaker=max_per_speaker)
        found += 1
    if not found:
        raise ValueError("No TRAIN JSONL manifests found. Use scan --manifest with an explicit train manifest.")
    return summary(root)


def scan_manifest(root, manifest_path, audio_root=None, **kwargs):
    path = Path(manifest_path)
    with path.open(encoding="utf-8-sig") as lines:
        return scan_lines(root, lines, path.name, audio_root=audio_root or path.parent, **kwargs)


def decode_audio(path, maximum):
    """Bound decoded RAM to one source utterance; FFmpeg covers OPUS on Colab."""
    try:
        stream = sf.SoundFile(path)
    except RuntimeError:
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-t", str(maximum + 1),
                                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
                                check=True, capture_output=True, timeout=120)
        audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    else:
        with stream:
            if stream.channels > 8 or stream.samplerate > 192000:
                raise ValueError("Unsupported channel count or sample rate")
            if stream.frames / stream.samplerate > maximum:
                raise ValueError("Source exceeds max_source_seconds")
            audio = stream.read(dtype="float32", always_2d=True).mean(axis=1)
            divisor = math.gcd(stream.samplerate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // divisor, stream.samplerate // divisor)
    if not len(audio) or len(audio) > round(maximum * SAMPLE_RATE):
        raise ValueError("Empty or too long source")
    if not np.isfinite(audio).all() or float(np.max(np.abs(audio))) < 1e-5:
        raise ValueError("Non-finite or silent source")
    return audio.astype("float32")


def atomic_wav(path, audio):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.wav")
    sf.write(temporary, audio, SAMPLE_RATE, subtype="PCM_16")
    temporary.replace(path)


class WhisperWords:
    """Load the model lazily, only when a matching source needs timestamps."""
    def __init__(self, model="small", device="cuda", compute_type="int8_float16"):
        self.options = dict(model_size_or_path=model, device=device, compute_type=compute_type)
        self.model = None

    def __call__(self, audio):
        try:
            if self.model is None:
                from faster_whisper import WhisperModel
                print(f"Loading timestamp model: {self.options}", flush=True)
                self.model = WhisperModel(**self.options)
            segments, _ = self.model.transcribe(audio, language="ru", word_timestamps=True,
                                               beam_size=5, vad_filter=False,
                                               condition_on_previous_text=False)
            return [dict(text=word.word, start=word.start, end=word.end)
                    for segment in segments for word in segment.words or []]
        except Exception as error:
            raise BackendError(f"Timestamp backend failed; source remains resumable: {error}") from error


class BackendError(RuntimeError):
    pass


def word_crops(words, label, duration, config):
    aligned = [(token, float(word["start"]), float(word["end"]))
               for word in words for token in tokens(word["text"])]
    target = tokens(COMMANDS[label])
    for i in range(len(aligned)):
        last = i + len(target) - 1
        if [item[0] for item in aligned[i:i + len(target)]] != target:
            continue
        start = max(0.0, aligned[i][1] - config["margin"])
        end = min(duration, aligned[last][2] + config["margin"])
        # Keep margins from eating a neighbouring recognized word.
        if i:
            start = max(start, aligned[i - 1][2])
        if last + 1 < len(aligned):
            end = min(end, aligned[last + 1][1])
        if math.isfinite(start) and math.isfinite(end) and (
                config["min_seconds"] <= end - start <= config["max_seconds"]):
            yield start, end


def process_source(root, source_id, path, aligner):
    config = settings(root)
    with database(root) as db:
        row = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
        data = json.loads(row["data"])
        labels = [r[0] for r in db.execute("SELECT label FROM targets WHERE source_id=?", (source_id,))]
    print(f"Preparing {source_id}: {labels}", flush=True)
    audio = decode_audio(path, config["max_source_seconds"])
    duration = len(audio) / SAMPLE_RATE
    context = Path(root) / "contexts" / f"{source_id}.wav"
    atomic_wav(context, audio)
    whole_label = next((label for label in labels if tokens(data["text"]) == tokens(COMMANDS[label])), None)
    use_whole = whole_label and config["min_seconds"] <= duration <= config["max_seconds"]
    print(f"Source {duration:.2f}s; {'whole command' if use_whole else 'estimating word timestamps'}", flush=True)
    words = [] if use_whole else aligner(audio)
    data.update(duration=duration, context_sha256=file_hash(context))
    with database(root) as db:
        for label in labels:
            ranges = [(0.0, duration)] if use_whole and label == whole_label else list(
                word_crops(words, label, duration, config))
            # No timestamp match is kept for manual alignment, never auto-accepted.
            for occurrence, bounds in enumerate(ranges or [(None, None)]):
                clip_id = f"{source_id}_{label}_{occurrence}"
                method = "whole_utterance" if use_whole else "whisper_words" if ranges else "manual_required"
                db.execute("INSERT OR IGNORE INTO clips VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
                           (clip_id, source_id, label, *bounds, method,
                            "" if ranges else "No valid ASR crop; set bounds manually or reject"))
        db.execute("UPDATE sources SET status='ready', data=? WHERE id=?",
                   (json.dumps(data, ensure_ascii=False), source_id))


def prepare(root, archive_path=None, aligner=None, limit=0, retry_errors=False):
    """One source at a time. Each completed source is committed before the next."""
    settings(root)
    if limit < 0:
        raise ValueError("limit must be zero or positive")
    aligner = aligner if aligner is not None else WhisperWords()
    with database(root) as db:
        if retry_errors:
            db.execute("UPDATE sources SET status='new' WHERE status='error'")
        pending = {r["id"]: dict(r) for r in db.execute("SELECT * FROM sources WHERE status='new'")}
        # A raised scan quota can add targets to an already processed source.
        for row in db.execute("SELECT s.* FROM sources s WHERE s.status='ready' AND EXISTS "
                              "(SELECT 1 FROM targets t WHERE t.source_id=s.id AND NOT EXISTS "
                              "(SELECT 1 FROM clips c WHERE c.source_id=t.source_id AND c.label=t.label))"):
            pending[row["id"]] = dict(row)
    print(f"Prepare: {len(pending)} remaining sources; existing review decisions preserved", flush=True)
    processed = 0

    def run(source, path):
        nonlocal processed
        try:
            process_source(root, source["id"], path, aligner)
        except BackendError:
            # A missing CUDA library/model is not evidence of a bad corpus recording.
            raise
        except Exception as error:
            with database(root) as db:
                data = json.loads(source["data"])
                data["error"] = f"{type(error).__name__}: {error}"
                db.execute("UPDATE sources SET status='error', data=? WHERE id=?",
                           (json.dumps(data, ensure_ascii=False), source["id"]))
            print(f"Failed {source['id']}: {error}", flush=True)
        processed += 1
        pending.pop(source["id"], None)
        print(f"Prepare: processed {processed}; remaining {len(pending)}", flush=True)

    # Contexts persist on Drive, so manual review and resume need no corpus re-download.
    for source in list(pending.values()):
        context = Path(root) / "contexts" / f"{source['id']}.wav"
        local = json.loads(source["data"]).get("local_path")
        path = context if context.is_file() else Path(local) if local else None
        if path and path.is_file():
            run(source, path)
            if limit and processed >= limit:
                return summary(root)
    if archive_path and pending:
        by_key = {source["audio_key"]: source for source in pending.values()}
        by_stem = {}
        for source in pending.values():
            by_stem.setdefault(source["basename"], []).append(source)
        for archive, member in tar_members(archive_path):
            if not member.isfile() or PurePosixPath(member.name).suffix.casefold() not in {".wav", ".opus", ".ogg", ".mp3", ".flac"}:
                continue
            source = by_key.get(str(PurePosixPath(member.name)))
            if source is None:
                candidates = by_stem.get(PurePosixPath(member.name).stem, [])
                # Golos OPUS can keep .wav paths in manifests. Unique stem fallback only.
                source = candidates[0] if len(candidates) == 1 else None
            if source is None or source["id"] not in pending:
                continue
            if member.size > 64 * 1024 * 1024:
                print(f"Skipping oversized audio member: {member.name}", flush=True)
                continue
            with tempfile.TemporaryDirectory(prefix="kws-corpus-") as temporary:
                path = Path(temporary) / ("source" + PurePosixPath(member.name).suffix)
                with archive.extractfile(member) as src, path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)
                run(source, path)
            if not pending or (limit and processed >= limit):
                break
    if pending:
        print(f"{len(pending)} sources still unavailable; supply the matching archive/audio directory.", flush=True)
    return summary(root)


def crop_audio(root, clip_id, start=None, end=None):
    with database(root) as db:
        row = db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        if row is None:
            raise KeyError(clip_id)
        clip = dict(row)
        source = json.loads(db.execute("SELECT data FROM sources WHERE id=?", (clip["source_id"],)).fetchone()[0])
    start = clip["start"] if start is None else float(start)
    end = clip["end"] if end is None else float(end)
    if start is None or end is None:
        raise ValueError("Set crop bounds before accepting this clip")
    config = settings(root)
    context = Path(root) / "contexts" / f"{clip['source_id']}.wav"
    if file_hash(context) != source["context_sha256"]:
        raise ValueError("Saved context checksum changed; do not use stale review decisions")
    audio, rate = sf.read(context, dtype="float32")
    if rate != SAMPLE_RATE or not (math.isfinite(start) and math.isfinite(end)
            and 0 <= start < end <= len(audio) / rate
            and config["min_seconds"] <= end - start <= config["max_seconds"]):
        raise ValueError(f"Crop must be inside source and {config['min_seconds']}–{config['max_seconds']} seconds")
    crop = audio[round(start * rate):round(end * rate)]
    if not np.isfinite(crop).all() or np.max(np.abs(crop)) < 1e-5:
        raise ValueError("Crop is silent or non-finite")
    return clip, crop, start, end


def decide(root, clip_id, decision, start=None, end=None):
    if decision not in {"accepted", "rejected", "pending"}:
        raise ValueError("Unknown review decision")
    if decision == "accepted":
        clip, audio, start, end = crop_audio(root, clip_id, start, end)
        atomic_wav(Path(root) / "clips" / f"{clip_id}.wav", audio)
        with database(root) as db:
            db.execute("UPDATE clips SET status=?, start=?, end=?, method=? WHERE id=?",
                       (decision, start, end, "human_reviewed", clip_id))
    else:
        with database(root) as db:
            if not db.execute("SELECT 1 FROM clips WHERE id=?", (clip_id,)).fetchone():
                raise KeyError(clip_id)
            db.execute("UPDATE clips SET status=? WHERE id=?", (decision, clip_id))


def summary(root):
    with database(root) as db:
        result = dict(sources=dict(db.execute("SELECT status, COUNT(*) FROM sources GROUP BY status")),
                      targets=dict(db.execute("SELECT label, COUNT(*) FROM targets GROUP BY label")),
                      clips=[dict(row) for row in db.execute(
                          "SELECT label, status, COUNT(*) AS count FROM clips GROUP BY label, status")])
    return result


def review_widget(root):
    """Colab manual review with context, adjustable bounds, and immediate Drive saves."""
    import ipywidgets as widgets
    from IPython.display import Audio, display, clear_output
    root = Path(root)
    with database(root) as db:
        ids = [r[0] for r in db.execute("SELECT id FROM clips WHERE status='pending' ORDER BY label, id")]
    if not ids:
        print("No pending clips. See summary(ROOT), or prepare more sources.")
        return
    output = widgets.Output()
    start_box = widgets.FloatText(description="Start (s)")
    end_box = widgets.FloatText(description="End (s)")
    buttons = [widgets.Button(description=text) for text in ["Preview crop", "Accept", "Reject", "Later"]]
    position = [0]
    previewed = [None]

    def show():
        previewed[0] = None
        with output:
            clear_output(wait=True)
            if position[0] >= len(ids):
                print("Review finished for this queue. Re-run this cell to revisit postponed clips.")
                for button in buttons:
                    button.disabled = True
                return
            clip_id = ids[position[0]]
            with database(root) as db:
                clip = dict(db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone())
                source = json.loads(db.execute("SELECT data FROM sources WHERE id=?", (clip["source_id"],)).fetchone()[0])
            print(f"{position[0] + 1}/{len(ids)} | {clip['label']}: {COMMANDS[clip['label']]} | {clip['method']}")
            print("Transcript:", source["text"])
            print("Note:", clip["note"] or "Check complete phrase, boundaries, neighbouring words and speaker quality.")
            print("Accept only the target phrase, without other spoken words. Preview the crop first.")
            display(Audio(filename=str(root / "contexts" / f"{clip['source_id']}.wav")))
            start_box.value = clip["start"] if clip["start"] is not None else 0.0
            end_box.value = clip["end"] if clip["end"] is not None else min(source["duration"], 2.6)

    def action(button):
        clip_id = ids[position[0]]
        try:
            if button is buttons[0]:
                _, audio, start, end = crop_audio(root, clip_id, start_box.value, end_box.value)
                with output:
                    display(Audio(audio, rate=SAMPLE_RATE))
                previewed[0] = (clip_id, start, end)
                return
            if button is buttons[1]:
                if previewed[0] != (clip_id, start_box.value, end_box.value):
                    raise ValueError("Preview these crop bounds before accepting")
                decide(root, clip_id, "accepted", start_box.value, end_box.value)
            elif button is buttons[2]:
                decide(root, clip_id, "rejected")
            position[0] += 1
            show()
        except Exception as error:
            with output:
                print(f"Not saved: {error}")
    for button in buttons:
        button.on_click(action)
    display(widgets.VBox([output, widgets.HBox([start_box, end_box]), widgets.HBox(buttons)]))
    show()


def export_dataset(root, output_dir):
    """Export accepted TRAIN additions only; never merge or alter existing test data."""
    root, output = Path(root), Path(output_dir)
    if output.exists():
        raise FileExistsError("Choose a new export directory; existing exports are not overwritten")
    config = settings(root)
    with database(root) as db:
        accepted = [dict(row) for row in db.execute("SELECT * FROM clips WHERE status='accepted' ORDER BY id")]
    if not accepted:
        raise ValueError("No accepted clips to export")
    output.mkdir(parents=True)
    rows, seen_hashes, counts = [], set(), Counter()
    try:
        for clip in accepted:
            _, audio, start, end = crop_audio(root, clip["id"])
            with database(root) as db:
                source = json.loads(db.execute("SELECT data FROM sources WHERE id=?", (clip["source_id"],)).fetchone()[0])
            if source["corpus_split"] != "train":
                raise ValueError("Non-training corpus source in training export")
            path = output / "audio" / clip["label"] / f"{clip['id']}.wav"
            atomic_wav(path, audio)
            checksum = file_hash(path)
            if checksum in seen_hashes:
                path.unlink()
                continue
            seen_hashes.add(checksum)
            # If no speaker metadata is supplied, keep the entire corpus in one group.
            # Do not fabricate per-file speaker IDs and claim speaker-disjoint evaluation.
            row = dict(id=clip["id"], path=path.relative_to(output).as_posix(), label=clip["label"],
                       split="train", text=COMMANDS[clip["label"]], engine="real", sample_kind="corpus_crop",
                       source_group=f"corpus:{config['corpus']}", parent_id=clip["source_id"],
                       speaker_group=source.get("speaker_group") or f"{config['corpus']}:speaker_unknown",
                       sha256=checksum, sample_rate=SAMPLE_RATE, duration=len(audio) / SAMPLE_RATE,
                       corpus=config["corpus"], corpus_split="train", source_audio=source["audio_key"],
                       source_text=source["text"], crop_start=start, crop_end=end, review="accepted")
            rows.append(row)
            counts[clip["label"]] += 1
        (output / "splits").mkdir()
        manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        (output / "manifest.jsonl").write_text(manifest, encoding="utf-8")
        (output / "splits" / "train.jsonl").write_text(manifest, encoding="utf-8")
        atomic_json(output / "labels.json", LABELS)
        report = dict(complete=True, counts=dict(counts), clips=len(rows), training_additions_only=True,
                      duplicate_crops_removed=len(accepted) - len(rows), settings=config,
                      source_url="https://github.com/salute-developers/golos" if config["corpus"] == "golos" else None,
                      license_url="https://github.com/salute-developers/golos/blob/master/license/en_us.pdf"
                      if config["corpus"] == "golos" else None)
        atomic_json(output / "report.json", report)
        return report
    except BaseException:
        atomic_json(output / "report.json", dict(complete=False, error="Export interrupted; use a new directory"))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Persistent mining directory (Drive)")
    sub = parser.add_subparsers(dest="stage", required=True)
    init = sub.add_parser("init")
    init.add_argument("--corpus", default="golos")
    scan = sub.add_parser("scan")
    source = scan.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path)
    source.add_argument("--manifest", type=Path)
    scan.add_argument("--audio-root", type=Path)
    scan.add_argument("--max-per-label", type=int, default=1000)
    scan.add_argument("--max-per-speaker", type=int, default=30)
    prep = sub.add_parser("prepare")
    prep.add_argument("--archive", type=Path)
    prep.add_argument("--model", default="small")
    prep.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    prep.add_argument("--limit", type=int, default=0, help="0 means all remaining sources")
    prep.add_argument("--retry-errors", action="store_true")
    sub.add_parser("status")
    export = sub.add_parser("export")
    export.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.stage == "init":
        result = initialize(args.root, corpus=args.corpus)
    elif args.stage == "scan":
        kwargs = dict(max_per_label=args.max_per_label, max_per_speaker=args.max_per_speaker)
        result = scan_archive(args.root, args.archive, **kwargs) if args.archive else scan_manifest(
            args.root, args.manifest, audio_root=args.audio_root, **kwargs)
    elif args.stage == "prepare":
        aligner = WhisperWords(args.model, args.device, "int8_float16" if args.device == "cuda" else "int8")
        result = prepare(args.root, args.archive, aligner, args.limit, args.retry_errors)
    elif args.stage == "export":
        result = export_dataset(args.root, args.output_dir)
    else:
        result = summary(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
