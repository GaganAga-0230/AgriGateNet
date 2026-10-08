"""
train.py
────────
Unified training loop for all 6 models in PRJ-37.

Features:
  • Cosine annealing LR schedule with linear warmup
  • Label smoothing (ε=0.1) for regularisation
  • Mixed precision (AMP / FP16) for faster GPU training
  • MixUp / CutMix augmentation (applied at batch level)
  • Automatic checkpointing (best val accuracy)
  • TensorBoard / console logging

Usage:
    # Train a single model:
    python training/train.py --model agrigatenet --epochs 50

    # Train all models:
    python training/train.py --model all --epochs 30

    # Quick smoke-test (2 epochs):
    python training/train.py --model baseline --epochs 2 --smoke_test
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# ── Project imports ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloader import get_dataloaders, NUM_CLASSES
from data.preprocess_pipeline import mixup_data, cutmix_data, mixup_criterion
from models.baseline_cnn import build_baseline_cnn
from models.mobilenetv3 import build_mobilenetv3
from models.efficientnet import build_efficientnet
from models.shufflenetv2 import build_shufflenetv2
from models.squeezenet import build_squeezenet
from models.agrigatenet import build_agrigatenet


# ─────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "baseline":     lambda nc: build_baseline_cnn(num_classes=nc),
    "mobilenetv3":  lambda nc: build_mobilenetv3(num_classes=nc, pretrained=True),
    "efficientnet": lambda nc: build_efficientnet(num_classes=nc, pretrained=True),
    "shufflenetv2": lambda nc: build_shufflenetv2(num_classes=nc, pretrained=True),
    "squeezenet":   lambda nc: build_squeezenet(num_classes=nc, pretrained=True),
    "agrigatenet":  lambda nc: build_agrigatenet(num_classes=nc),
}


# ─────────────────────────────────────────────────────────────────
# LR Schedule: Warmup + Cosine Annealing
# ─────────────────────────────────────────────────────────────────

def get_scheduler(optimizer, num_epochs: int, warmup_epochs: int = 5):
    """Cosine annealing LR scheduler with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────
# One Training Epoch
# ─────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device,
    use_mixup: bool = True,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    cutmix_prob: float = 0.5,
):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # MixUp / CutMix augmentation
        apply_mix = use_mixup
        use_cutmix = apply_mix and (torch.rand(1).item() < cutmix_prob)

        if apply_mix:
            if use_cutmix:
                images, labels_a, labels_b, lam = cutmix_data(images, labels, cutmix_alpha)
            else:
                images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)

        optimizer.zero_grad()

        with autocast(enabled=(device.type == "cuda")):
            outputs = model(images)
            if apply_mix:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        # For mixed data, use original labels for accuracy tracking
        target = labels if not apply_mix else labels_a
        total_correct += preds.eq(target).sum().item()
        total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


# ─────────────────────────────────────────────────────────────────
# Validation / Test Epoch
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(labels).sum().item()
        total_samples += images.size(0)

    return total_loss / total_samples, total_correct / total_samples


# ─────────────────────────────────────────────────────────────────
# Full Training Loop
# ─────────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    data_root: str = "./data/organized",
    checkpoint_dir: str = "./checkpoints",
    log_dir: str = "./runs",
    num_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.1,
    num_workers: int = 4,
    use_mixup: bool = True,
    smoke_test: bool = False,
    device_str: str = "auto",
) -> dict:
    """
    Train a single model end-to-end.
    Returns a dict with training history and best metrics.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"\n{'='*60}")
    print(f" Training: {model_name.upper()}")
    print(f" Device  : {device}  |  Epochs: {num_epochs}")
    print(f"{'='*60}")

    if smoke_test:
        num_epochs = 2
        batch_size = 8

    # ── Data ───────────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = MODEL_REGISTRY[model_name](NUM_CLASSES)
    model = model.to(device)

    # ── Optimiser + Loss ───────────────────────────────────────────────────
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scheduler = get_scheduler(optimizer, num_epochs, warmup_epochs=5)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # ── Logging ────────────────────────────────────────────────────────────
    log_path = Path(log_dir) / model_name
    writer = SummaryWriter(str(log_path))
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training ───────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], optimizer, criterion, scaler, device,
            use_mixup=(use_mixup and model_name != "baseline"),
        )
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, epoch)
        writer.add_scalar("LR", lr_now, epoch)

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
            }, ckpt_dir / f"{model_name}_best.pth")

        print(
            f"  Epoch [{epoch:3d}/{num_epochs}]  "
            f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  Val Acc: {val_acc:.4f}  |  "
            f"LR: {lr_now:.6f}  [{elapsed:.1f}s]"
        )

    # ── Save final checkpoint ───────────────────────────────────────────────
    torch.save(model.state_dict(), ckpt_dir / f"{model_name}_final.pth")

    # Save history
    history_path = ckpt_dir / f"{model_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    writer.close()
    print(f"\n✅  {model_name} — Best Val Acc: {best_val_acc:.4f}")
    return {"model_name": model_name, "best_val_acc": best_val_acc, "history": history}


# ─────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="PRJ-37 Model Training Script")
    parser.add_argument(
        "--model",
        type=str,
        default="agrigatenet",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        help="Which model to train. Use 'all' to train all 6 models.",
    )
    parser.add_argument("--data_root", type=str, default="./data/organized")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--no_mixup", action="store_true")
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run 2 epochs with batch_size=8 to verify code correctness.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    models_to_train = (
        list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    )

    all_results = []
    for model_name in models_to_train:
        result = train_model(
            model_name=model_name,
            data_root=args.data_root,
            checkpoint_dir=args.checkpoint_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            label_smoothing=args.label_smoothing,
            num_workers=args.num_workers,
            use_mixup=not args.no_mixup,
            smoke_test=args.smoke_test,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("  Final Summary")
        print("=" * 60)
        for r in all_results:
            print(f"  {r['model_name']:15s}  Val Acc: {r['best_val_acc']:.4f}")
