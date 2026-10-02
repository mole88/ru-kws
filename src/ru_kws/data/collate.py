import torch
import torch.nn.functional as F


class PadCropCollate:
    """Random train alignment; deterministic evaluation; never truncate commands."""
    def __init__(self, num_samples, command_ids, training=False, augmentation=None, speed_augmentation=None):
        self.num_samples = num_samples
        self.command_ids = set(command_ids)
        self.training = training
        self.augmentation = augmentation if training else None
        self.speed_augmentation = speed_augmentation if training else None

    def __call__(self, batch):
        audio, labels = [], []
        for waveform, label in batch:
            if waveform.ndim != 1 or waveform.numel() == 0 or not torch.isfinite(waveform).all():
                raise ValueError("Expected nonempty, finite, one-dimensional waveform")
            waveform = waveform.float()
            if self.speed_augmentation is not None:
                waveform = self.speed_augmentation(waveform, label)
            difference = self.num_samples - waveform.numel()
            if difference < 0:
                if label in self.command_ids:
                    raise ValueError("Command exceeds configured window; increase window or curate the dataset explicitly")
                start = torch.randint(-difference + 1, (1,)).item() if self.training else -difference // 2
                waveform = waveform[start:start + self.num_samples]
            elif difference > 0:
                left = torch.randint(difference + 1, (1,)).item() if self.training else difference // 2
                waveform = F.pad(waveform, (left, difference - left))
            if self.augmentation is not None:
                waveform = self.augmentation(waveform, label)
            audio.append(waveform)
            labels.append(label)
        return torch.stack(audio), torch.tensor(labels, dtype=torch.long)
