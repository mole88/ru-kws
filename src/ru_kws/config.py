"""Configuration shared by training and evaluation."""
from pathlib import Path
import random

import numpy as np
import torch
import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if config["audio"]["sample_rate"] != 16000:
        raise ValueError("The baseline expects 16 kHz audio")
    if config["frontend"]["n_mels"] != 40:
        raise ValueError("This Qualcomm BC-ResNet expects 40 mel bins")
    if config["audio"]["window_seconds"] <= 0:
        raise ValueError("window_seconds must be positive")
    if config["model"]["name"] != "bcresnet":
        raise ValueError("Only bcresnet is implemented")
    if config["model"]["base_c"] not in (8, 12, 16, 24, 48, 64):
        raise ValueError("base_c must be one of 8, 12, 16, 24, 48, 64")
    for key in ("batch_size", "max_epochs", "early_stopping_patience"):
        if config["training"][key] < 1:
            raise ValueError(f"training.{key} must be positive")
    return config


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)
