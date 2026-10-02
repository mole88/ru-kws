import torch


def classification_metrics(confusion, labels):
    """Rows are true classes; columns are predicted classes."""
    total = int(confusion.sum())
    classes = {}
    for name, index in labels.items():
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        classes[name] = {"precision": precision, "recall": recall,
                         "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                         "support": support}
    return {"count": total, "accuracy": int(confusion.trace()) / total if total else 0.0,
            "macro_f1": sum(v["f1"] for v in classes.values()) / len(classes),
            "per_class": classes,
            "class_order": sorted(labels, key=labels.get), "confusion_matrix": confusion.tolist()}


@torch.inference_mode()
def evaluate_clips(model, frontend, loader, labels, device):
    model.eval()
    frontend.eval()
    confusion = torch.zeros(len(labels), len(labels), dtype=torch.int64)
    for waveforms, targets in loader:
        logits = model(frontend(waveforms.to(device)))
        if logits.shape != (len(targets), len(labels)):
            raise ValueError(f"Expected logits [batch, {len(labels)}], got {tuple(logits.shape)}")
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite logits (NaN or Inf) during evaluation")
        predictions = logits.argmax(1).cpu()
        indices = targets * len(labels) + predictions
        confusion += torch.bincount(indices, minlength=len(labels) ** 2).reshape(len(labels), len(labels))
    return classification_metrics(confusion, labels)
