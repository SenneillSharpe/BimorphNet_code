from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    epoch: int,
    metrics: dict,
    scheduler=None,
    scaler=None,
    ema=None,
    best_metric: float | None = None,
    config=None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "best_metric": best_metric,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "ema_state_dict": ema.state_dict() if ema is not None else None,
        "config": config.to_checkpoint_dict() if hasattr(config, "to_checkpoint_dict") else config,
    }
    torch.save(payload, path)


def load_model_weights(path: Path, model, device, use_ema: bool = False):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and use_ema and checkpoint.get("ema_state_dict"):
        state_dict = checkpoint["ema_state_dict"]["shadow"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=not use_ema)
    return checkpoint
