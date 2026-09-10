import json
import numpy as np
import pytest
import soundfile as sf
from ru_kws.data.dataset import AudioCommandsDataset
from ru_kws.data.validate import validate_dataset


def test_duration_filter_shared_by_validation_and_all_splits(tmp_path):
    labels = {"next": 0, "unknown": 1, "background": 2}
    (tmp_path / "labels.json").write_text(json.dumps(labels))
    (tmp_path / "splits").mkdir()
    for split_index, split in enumerate(("train", "val", "test")):
        rows = []
        for i, (label, frames) in enumerate((("next", 48000), ("next", 48001),
                                             ("unknown", 48001), ("background", 48001))):
            name = f"{split}_{i}.wav"
            sf.write(tmp_path / name, np.full(frames, (split_index * 4 + i + 1) / 20), 16000)
            rows.append({"path": name, "label": label})
        (tmp_path / "splits" / f"{split}.jsonl").write_text("\n".join(map(json.dumps, rows)))
        dataset = AudioCommandsDataset(tmp_path, split, labels)
        assert [r["path"] for r in dataset.records] == [rows[i]["path"] for i in (0, 2, 3)]
        assert dataset.dropped_records[0]["path"] == rows[1]["path"]
        assert len(dataset.dropped_records) == 1
        assert (tmp_path / rows[1]["path"]).exists()
    assert validate_dataset(tmp_path) == {s: {"next": 1, "unknown": 1, "background": 1}
                                          for s in ("train", "val", "test")}
