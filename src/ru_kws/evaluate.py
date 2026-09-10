"""Evaluate a saved checkpoint on val or test clips."""
import argparse
import json
from pathlib import Path

from ru_kws.audio.frontend import build_frontend
from ru_kws.checkpoint import load_checkpoint, write_json
from ru_kws.config import choose_device
from ru_kws.data.dataset import make_loader
from ru_kws.data.manifest import read_labels, manifest_hash
from ru_kws.evaluation.clips import evaluate_clips
from ru_kws.models.factory import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config, labels = checkpoint["config"], checkpoint["labels"]
    if labels != read_labels(args.data_root):
        parser.error("Dataset label mapping differs from checkpoint")
    device = choose_device(args.device)
    model = build_model(config, len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    frontend = build_frontend(config).to(device)
    loader = make_loader(args.data_root, args.split, labels, config)
    report = evaluate_clips(model, frontend, loader, labels, device)
    report.update(checkpoint=str(args.checkpoint.resolve()), epoch=checkpoint["epoch"],
                  split=args.split, manifest_sha256=manifest_hash(args.data_root, args.split))
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
