from torch.utils.data import Dataset, DataLoader

from ru_kws.data.filtering import filter_long_commands
from ru_kws.audio.io import load_audio
from ru_kws.data.manifest import audio_path, read_manifest
from ru_kws.data.collate import PadCropCollate
from ru_kws.data.augmentation import build_augmentation, build_speed_augmentation


class AudioCommandsDataset(Dataset):
    def __init__(self, root, split, labels, sample_rate=16000, window_seconds=3.0):
        self.root, self.labels, self.sample_rate = root, labels, sample_rate
        self.records, self.dropped_records = filter_long_commands(
            root, read_manifest(root, split, labels), split, window_seconds, sample_rate)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        waveform = load_audio(audio_path(self.root, row["path"]), self.sample_rate)
        return waveform, self.labels[row["label"]]


def make_loader(root, split, labels, config):
    dataset = AudioCommandsDataset(root, split, labels, config["audio"]["sample_rate"],
                                   config["audio"]["window_seconds"])
    command_ids = {idx for name, idx in labels.items() if name not in {"unknown", "background"}}
    return DataLoader(
        dataset, batch_size=config["training"]["batch_size"], shuffle=split == "train",
        num_workers=config["training"].get("num_workers", 0),
        collate_fn=PadCropCollate(
            round(config["audio"]["sample_rate"] * config["audio"]["window_seconds"]),
            command_ids, training=split == "train",
            augmentation=build_augmentation(root, split, labels, config),
            speed_augmentation=build_speed_augmentation(split, command_ids, config),
        ),
    )
