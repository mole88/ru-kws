import importlib.util
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest
import soundfile as sf

from ru_kws.data.validate import validate_dataset


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/mine_corpus_commands.py"
spec = importlib.util.spec_from_file_location("corpus_mining", SCRIPT)
mining = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mining)


def audio_bytes(seconds=1, rate=16000):
    stream = io.BytesIO()
    samples = np.arange(round(seconds * rate)) / rate
    sf.write(stream, (.15 * np.sin(2 * np.pi * 220 * samples)).astype("float32"),
             rate, format="WAV", subtype="PCM_16")
    return stream.getvalue()


def manifest(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def add_member(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


def clips(root):
    with mining.database(root) as db:
        return [dict(row) for row in db.execute("SELECT * FROM clips ORDER BY id")]


def test_exact_phrase_boundaries():
    assert mining.matches("Ну, ДАЛЬШЕ! Назад; повторить. Запустить таймер!") == [
        "next", "back", "repeat", "start_timer"]
    assert mining.matches("Остановить — таймер, сколько осталось?") == ["stop_timer", "time_left"]
    assert mining.matches("дальнейший назадний повторитель запустить осталось таймер") == []
    assert mining.matches("Сколько ещё осталось? Запустить новый таймер") == []


def test_streaming_tar_scan_training_only_and_resume(tmp_path):
    root = tmp_path / "state"
    mining.initialize(root)
    archive_path = tmp_path / "corpus.tar"
    rows = [dict(audio_filepath="files/a.wav", text="Дальше", speaker_id="alice"),
            dict(audio_filepath="files/b.wav", text="Дальше", speaker_id="alice"),
            dict(audio_filepath="files/c.wav", text="Назад", speaker_id="bob"),
            dict(audio_filepath="../../outside.wav", text="Повторить"),
            dict(audio_filepath="files/test.wav", text="Повторить", split="test")]
    with tarfile.open(archive_path, "w") as archive:
        for i in range(4000):
            add_member(archive, f"unrelated/{i}.txt", b"x")
        add_member(archive, "golos/train/crowd/manifest.jsonl",
                   "".join(json.dumps(row) + "\n" for row in rows).encode())
        add_member(archive, "golos/test/manifest.jsonl",
                   (json.dumps(dict(audio_filepath="files/test.wav", text="Повторить")) + "\n").encode())
        add_member(archive, "golos/train/crowd/files/a.opus", audio_bytes())
        add_member(archive, "golos/train/crowd/files/c.opus", audio_bytes(.9))
    for archive, _ in mining.tar_members(archive_path):
        assert len(archive.members) <= 1
    result = mining.scan_archive(root, archive_path, max_per_speaker=1)
    assert result["targets"] == {"back": 1, "next": 1}
    mining.scan_archive(root, archive_path, max_per_speaker=1)
    assert mining.summary(root)["sources"] == {"new": 2}
    # Stem fallback covers OPUS audio whose manifests still name WAV files.
    mining.prepare(root, archive_path, aligner=lambda _: pytest.fail("Whole command needs no ASR"))
    assert mining.summary(root)["sources"] == {"ready": 2}
    assert all(clip["status"] == "pending" for clip in clips(root))
    archive_path.unlink()
    for clip in clips(root):
        mining.decide(root, clip["id"], "accepted")
    report = mining.export_dataset(root, tmp_path / "export")
    assert report["complete"] and report["clips"] == 2
    exported = [json.loads(line) for line in (tmp_path / "export/manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["split"] == row["corpus_split"] == "train" for row in exported)
    assert {row["speaker_group"] for row in exported} == {"golos:speaker:alice", "golos:speaker:bob"}
    assert not (tmp_path / "export/splits/test.jsonl").exists()
    assert validate_dataset(tmp_path / "export", splits=("train",)) == {"train": {"back": 1, "next": 1}}


def test_word_crops_manual_alignment_and_export_integrity(tmp_path):
    root = tmp_path / "state"
    config = mining.initialize(root)
    source = tmp_path / "audio"
    source.mkdir()
    (source / "sentence.wav").write_bytes(audio_bytes(4, 44100))
    (source / "missing_match.wav").write_bytes(audio_bytes(4))
    path = source / "train.jsonl"
    manifest(path, [dict(audio_filepath="sentence.wav", text="Можно запустить таймер сейчас"),
                    dict(audio_filepath="missing_match.wav", text="Вернись назад")])
    mining.scan_manifest(root, path)
    words = [dict(text="Можно", start=.1, end=.95),
             dict(text="запустить", start=1, end=1.45),
             dict(text="таймер", start=1.5, end=1.9),
             dict(text="сейчас", start=2, end=2.3)]
    assert list(mining.word_crops(words, "start_timer", 4, config)) == [(.95, 2)]
    mining.prepare(root, aligner=lambda _: words)
    pending = clips(root)
    timer = next(row for row in pending if row["label"] == "start_timer")
    back = next(row for row in pending if row["label"] == "back")
    assert back["method"] == "manual_required" and back["start"] is None
    with pytest.raises(ValueError, match="No accepted"):
        mining.export_dataset(root, tmp_path / "empty_export")
    with pytest.raises(ValueError, match="bounds"):
        mining.decide(root, back["id"], "accepted")
    with pytest.raises(ValueError, match="inside source"):
        mining.decide(root, back["id"], "accepted", .1, 3.5)
    mining.decide(root, back["id"], "accepted", .5, 1.5)
    mining.decide(root, timer["id"], "rejected")
    mining.prepare(root, aligner=lambda _: pytest.fail("Resume must skip prepared sources"))
    report = mining.export_dataset(root, tmp_path / "accepted")
    assert report["counts"] == {"back": 1}
    row = json.loads((tmp_path / "accepted/manifest.jsonl").read_text(encoding="utf-8"))
    assert row["speaker_group"] == "golos:speaker_unknown"
    assert row["sha256"] == mining.file_hash(tmp_path / "accepted" / row["path"])
    with pytest.raises(FileExistsError):
        mining.export_dataset(root, tmp_path / "accepted")
    context = root / "contexts" / f"{back['source_id']}.wav"
    context.write_bytes(audio_bytes(4.1))
    with pytest.raises(ValueError, match="checksum"):
        mining.decide(root, back["id"], "accepted", .5, 1.5)


def test_interrupt_and_backend_failure_do_not_restart_completed_sources(tmp_path):
    root = tmp_path / "state"
    mining.initialize(root)
    path = tmp_path / "train.jsonl"
    for name in ["a", "b", "c"]:
        (tmp_path / f"{name}.wav").write_bytes(audio_bytes(4))
    manifest(path, [dict(audio_filepath=f"{name}.wav", text="Это дальше") for name in ["a", "b", "c"]])
    mining.scan_manifest(root, path)
    calls = []

    def interrupted(audio):
        calls.append(len(audio))
        if len(calls) == 2:
            raise KeyboardInterrupt()
        return [dict(text="дальше", start=1, end=1.5)]

    with pytest.raises(KeyboardInterrupt):
        mining.prepare(root, aligner=interrupted)
    assert mining.summary(root)["sources"] == {"new": 2, "ready": 1}
    first = clips(root)[0]
    mining.decide(root, first["id"], "accepted")

    def broken_backend(_):
        raise mining.BackendError("Missing CUDA")

    with pytest.raises(mining.BackendError):
        mining.prepare(root, aligner=broken_backend)
    assert mining.summary(root)["sources"] == {"new": 2, "ready": 1}
    calls.clear()
    mining.prepare(root, aligner=lambda audio: calls.append(len(audio)) or [dict(text="дальше", start=1, end=1.5)])
    assert len(calls) == 2
    assert next(row for row in clips(root) if row["id"] == first["id"])["status"] == "accepted"
    mining.scan_manifest(root, path)
    assert mining.summary(root)["targets"] == {"next": 3}
    with pytest.raises(ValueError, match="different mining settings"):
        mining.initialize(root, corpus="other")
    with pytest.raises(ValueError, match="TRAIN"):
        mining.initialize(tmp_path / "test_state", source_split="test")


def test_bad_audio_and_expanding_quota_preserve_reviews(tmp_path):
    root = tmp_path / "state"
    mining.initialize(root, max_source_seconds=5)
    path = tmp_path / "train.jsonl"
    (tmp_path / "a.wav").write_bytes(audio_bytes())
    (tmp_path / "b.wav").write_bytes(audio_bytes(.9))
    (tmp_path / "bad.wav").write_bytes(b"invalid")
    (tmp_path / "long.wav").write_bytes(audio_bytes(6))
    manifest(path, [dict(audio_filepath=name, text="Назад")
                    for name in ["a.wav", "b.wav", "bad.wav", "long.wav"]])
    mining.scan_manifest(root, path, max_per_label=1)
    mining.prepare(root, aligner=lambda _: [])
    accepted = clips(root)[0]["id"]
    mining.decide(root, accepted, "accepted")
    mining.scan_manifest(root, path, max_per_label=4)
    mining.prepare(root, aligner=lambda _: [])
    assert mining.summary(root)["sources"] == {"error": 2, "ready": 2}
    assert next(row for row in clips(root) if row["id"] == accepted)["status"] == "accepted"


def test_default_cli_uses_transcripts_without_loading_asr(tmp_path, monkeypatch):
    root = tmp_path / "state"
    mining.initialize(root)
    path = tmp_path / "train.jsonl"
    (tmp_path / "short.wav").write_bytes(audio_bytes())
    (tmp_path / "sentence.wav").write_bytes(audio_bytes(4))
    manifest(path, [dict(audio_filepath="short.wav", text="Назад"),
                    dict(audio_filepath="sentence.wav", text="Теперь запустить таймер")])
    mining.scan_manifest(root, path)
    monkeypatch.setattr(mining, "WhisperWords", lambda *args, **kwargs: pytest.fail("No ASR model allowed"))
    result = mining.main(["--root", str(root), "prepare"])
    assert result["sources"] == {"ready": 2}
    rows = clips(root)
    whole = next(row for row in rows if row["label"] == "back")
    manual = next(row for row in rows if row["label"] == "start_timer")
    assert whole["method"] == "whole_utterance" and whole["start"] == 0
    assert manual["method"] == "manual_required" and manual["start"] is None
    assert "ASR disabled" in manual["note"]
    mining.decide(root, manual["id"], "accepted", 1, 2)
    assert mining.export_dataset(root, tmp_path / "export")["counts"] == {"start_timer": 1}


def test_notebook_has_compilable_thin_cells():
    notebook = Path(__file__).resolve().parents[1] / "notebooks/corpus_command_mining.ipynb"
    data = json.loads(notebook.read_text(encoding="utf-8"))
    for cell in data["cells"]:
        if cell["cell_type"] == "code":
            compile(cell["source"], str(notebook), "exec")
            assert not cell["outputs"]
            assert "embedded_tools" not in cell["source"]
