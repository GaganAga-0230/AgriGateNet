"""
pruning.py
──────────
Structured channel pruning for AgriGateNet (and other models).

Strategy: L1-Norm Structured Channel Pruning
  • For each Conv2d layer, we compute the L1-norm of each output filter.
  • The lowest-norm filters are deemed least important and zeroed out.
  • We apply 50% global sparsity — removing half of all filters.
  • After pruning, we fine-tune for 10 epochs to recover accuracy.

This is a critical step for edge deployment — reducing model size
and inference time while preserving classification accuracy.

Usage:
    python training/pruning.py \
        --checkpoint ./checkpoints/agrigatenet_best.pth \
        --output_dir ./checkpoints/pruned \
        --sparsity 0.50 \
        --finetune_epochs 10
"""

import sys
import argparse
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.agrigatenet import build_agrigatenet
from data.dataloader import get_dataloaders, NUM_CLASSES
from training.train import evaluate, get_scheduler


# ─────────────────────────────────────────────────────────────────
# Pruning Utilities
# ─────────────────────────────────────────────────────────────────

def get_prunable_modules(model: nn.Module) -> list[tuple[nn.Module, str]]:
    """
    Return all (module, 'weight') pairs for Conv2d and Linear layers
    that should be included in global magnitude pruning.
    Depthwise conv layers (groups == in_channels) are excluded to
    preserve the DSSAG dual-path structure.
    """
    modules_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Skip depthwise convolutions (they are already ultra-lightweight)
            is_depthwise = module.groups == module.in_channels
            if not is_depthwise:
                modules_to_prune.append((module, "weight"))
        elif isinstance(module, nn.Linear):
            modules_to_prune.append((module, "weight"))
    return modules_to_prune


def apply_global_pruning(model: nn.Module, sparsity: float = 0.50) -> nn.Module:
    """
    Apply global unstructured L1 pruning across all eligible layers.

    Parameters
    ----------
    model    : PyTorch model to prune
    sparsity : fraction of weights to zero out (0.0 = no pruning, 1.0 = all zero)

    Returns
    -------
    model with pruning masks applied (weights are zeroed, not removed)
    """
    modules = get_prunable_modules(model)

    if not modules:
        print("  ⚠  No prunable modules found.")
        return model

    prune.global_unstructured(
        modules,
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )

    # Report sparsity
    total_params = 0
    zero_params = 0
    for module, _ in modules:
        total_params += module.weight.nelement()
        zero_params   += (module.weight == 0).sum().item()

    actual_sparsity = 100.0 * zero_params / max(1, total_params)
    print(f"  ✅  Pruning applied — Target: {sparsity*100:.0f}% | Actual: {actual_sparsity:.2f}%")
    print(f"      Total prunable weights: {total_params:,}")
    print(f"      Zeroed weights        : {zero_params:,}")
    return model


def make_pruning_permanent(model: nn.Module) -> nn.Module:
    """
    Convert pruning masks to permanent weight zeros (removes mask buffers).
    Required before saving or quantizing the pruned model.
    """
    for module, _ in get_prunable_modules(model):
        try:
            prune.remove(module, "weight")
        except ValueError:
            pass  # already permanent
    return model


def count_sparsity(model: nn.Module) -> float:
    """Return the fraction of zero weights in the model."""
    total, zeros = 0, 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p == 0).sum().item()
    return zeros / max(1, total)


# ─────────────────────────────────────────────────────────────────
# Fine-tuning After Pruning
# ─────────────────────────────────────────────────────────────────

def finetune_pruned_model(
    model: nn.Module,
    data_root: str,
    num_epochs: int = 10,
    lr: float = 1e-4,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device = None,
) -> nn.Module:
    """
    Fine-tune a pruned model to recover accuracy.
    Uses a lower LR to avoid disturbing the pruning structure.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = get_scheduler(optimizer, num_epochs, warmup_epochs=2)

    best_acc = 0.0
    best_state = deepcopy(model.state_dict())

    print(f"\n  Fine-tuning pruned model for {num_epochs} epochs …")
    for epoch in range(1, num_epochs + 1):
        model.train()
        for images, labels in loaders["train"]:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        print(f"    Epoch [{epoch:2d}/{num_epochs}]  Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    print(f"  ✅  Fine-tuning done — Best Val Acc after pruning: {best_acc:.4f}")
    return model


# ─────────────────────────────────────────────────────────────────
# Main Script
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prune AgriGateNet")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/pruned")
    parser.add_argument("--data_root", type=str, default="./data/organized")
    parser.add_argument("--sparsity", type=float, default=0.50)
    parser.add_argument("--finetune_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print("📂 Loading checkpoint …")
    model = build_agrigatenet(num_classes=NUM_CLASSES)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device)

    print(f"  Pre-pruning sparsity : {count_sparsity(model)*100:.2f}%")

    # Apply pruning
    print(f"\n✂️  Applying global L1 pruning (sparsity={args.sparsity*100:.0f}%) …")
    model = apply_global_pruning(model, sparsity=args.sparsity)

    # Fine-tune
    model = finetune_pruned_model(
        model, args.data_root,
        num_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # Make permanent
    model = make_pruning_permanent(model)
    print(f"\n  Post-pruning sparsity: {count_sparsity(model)*100:.2f}%")

    # Save
    save_path = output_dir / "agrigatenet_pruned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"💾 Pruned model saved → {save_path}")


if __name__ == "__main__":
    main()
