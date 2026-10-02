import torch


def run_epoch(model, frontend, loader, criterion, device, optimizer=None, specaugment=None):
    training = optimizer is not None
    model.train(training)
    frontend.train(training)
    if specaugment is not None:
        specaugment.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for waveforms, targets in loader:
            waveforms, targets = waveforms.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            features = frontend(waveforms)
            if training and specaugment is not None:
                features = specaugment(features)
            logits = model(features)
            loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite loss; inspect audio and training parameters")
            if training:
                loss.backward()
                optimizer.step()
            size = targets.numel()
            total_loss += loss.item() * size
            correct += (logits.argmax(1) == targets).sum().item()
            count += size
    if not count:
        raise ValueError("Empty data loader")
    return {"loss": total_loss / count, "accuracy": correct / count, "count": count}
