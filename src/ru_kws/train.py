"""Run: python -m ru_kws.train --help."""
import argparse
import csv
import platform
from pathlib import Path
import subprocess

import torch
import torchaudio
import yaml

from ru_kws.audio.frontend import build_frontend
from ru_kws.checkpoint import save_checkpoint, write_json
from ru_kws.config import load_config, seed_everything, choose_device
from ru_kws.data.dataset import make_loader
from ru_kws.data.validate import validate_dataset
from ru_kws.data.manifest import read_labels, manifest_hash
from ru_kws.models.factory import build_model
from ru_kws.training.engine import run_epoch
from ru_kws.data.augmentation import build_specaugment


def main():
    parser = argparse.ArgumentParser(description="Train BC-ResNet on a JSONL command dataset")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    args = parser.parse_args()
    config = load_config(args.config)
    labels = read_labels(args.data_root)
    device = choose_device(args.device)
    seed_everything(config["seed"])
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error("run-dir must be empty; choose a new directory")
    validation = validate_dataset(args.data_root, config["audio"]["window_seconds"],
                                  splits=("train", "val"))
    train_loader = make_loader(args.data_root, "train", labels, config)
    val_loader = make_loader(args.data_root, "val", labels, config)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    for folder in ("metadata", "checkpoints", "training"):
        (args.run_dir / folder).mkdir()
    write_json(args.run_dir / "metadata" / "validation.json", validation)
    (args.run_dir / "metadata" / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(args.run_dir / "metadata" / "labels.json", labels)
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    write_json(args.run_dir / "metadata" / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchaudio": str(torchaudio.__version__), "device": str(device), "git_revision": revision,
    })
    fingerprints = {s: manifest_hash(args.data_root, s) for s in ("train", "val")}
    write_json(args.run_dir / "metadata" / "dataset_summary.json", {
        "data_root": str(args.data_root.resolve()), "manifest_sha256": fingerprints,
        "train_count": len(train_loader.dataset), "val_count": len(val_loader.dataset),
        "dropped_records": {"train": train_loader.dataset.dropped_records,
                            "val": val_loader.dataset.dropped_records},
    })
    model = build_model(config, len(labels)).to(device)
    frontend = build_frontend(config).to(device)
    specaugment = build_specaugment(config)
    if specaugment is not None:
        specaugment = specaugment.to(device)
    settings = config["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=settings["label_smoothing"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=settings["scheduler_patience"], min_lr=1e-6)
    best_loss, stale = float("inf"), 0
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    with (args.run_dir / "training" / "history.csv").open("w", newline="", encoding="utf-8") as history:
        writer = csv.DictWriter(history, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy", "next_lr"])
        writer.writeheader()
        for epoch in range(1, settings["max_epochs"] + 1):
            train_metrics = run_epoch(model, frontend, train_loader, criterion, device, optimizer, specaugment)
            val_metrics = run_epoch(model, frontend, val_loader, criterion, device)
            scheduler.step(val_metrics["loss"])
            improved = val_metrics["loss"] < best_loss
            if improved:
                best_loss, stale = val_metrics["loss"], 0
            else:
                stale += 1
            payload = {
                "format_version": 1, "epoch": epoch, "config": config, "labels": labels,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(), "best_val_loss": best_loss,
                "val_metrics": val_metrics, "manifest_sha256": fingerprints,
            }
            save_checkpoint(args.run_dir / "checkpoints" / "last.pt", payload)
            if improved:
                save_checkpoint(args.run_dir / "checkpoints" / "best.pt", payload)
            writer.writerow({"epoch": epoch, "train_loss": train_metrics["loss"],
                             "train_accuracy": train_metrics["accuracy"], "val_loss": val_metrics["loss"],
                             "val_accuracy": val_metrics["accuracy"], "next_lr": optimizer.param_groups[0]["lr"]})
            history.flush()
            print(f"Epoch {epoch}: train loss={train_metrics['loss']:.4f}; val loss={val_metrics['loss']:.4f}; val accuracy={val_metrics['accuracy']:.4f}" + (" [best]" if improved else ""), flush=True)
            if stale >= settings["early_stopping_patience"]:
                print("Early stopping", flush=True)
                break
    print(f"Best checkpoint: {args.run_dir / 'checkpoints' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
