import torch
from torch import nn
import torchaudio


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, win_length=480,
                 hop_length=160, n_mels=40, power=2.0, center=True,
                 log_eps=1e-6):
        super().__init__()
        self.log_eps = log_eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, win_length=win_length,
            hop_length=hop_length, n_mels=n_mels, power=power, center=center,
            pad_mode="reflect", normalized=False, norm=None, mel_scale="htk",
        )

    def forward(self, waveforms):
        return torch.log(self.mel(waveforms) + self.log_eps).unsqueeze(1)


def build_frontend(config):
    return LogMelFrontend(config["audio"]["sample_rate"], **config["frontend"])
