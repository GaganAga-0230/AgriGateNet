"""
evaluate.py
───────────
Comprehensive evaluation suite for all 6 PRJ-37 models.

Produces:
  • Per-class and macro/weighted F1-score
  • Top-1 accuracy
  • Multi-class confusion matrix (saved as PNG heatmap)
  • Per-class precision and recall

Usage:
    # Evaluate a single model:
    python evaluation/evaluate.py --model agrigatenet \
        --checkpoint ./checkpoints/agrigatenet_best.pth

    # Evaluate all models:
    python evaluation/evaluate.py --model all \
        --checkpoint_dir ./checkpoints
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloader import get_dataloaders, NUM_CLASSES, CLASS_NAMES
from models.baseline_cnn   import build_baseline_cnn
from models.mobilenetv3    import build_mobilenetv3
from models.efficientnet   import build_efficientnet
from models.shufflenetv2   import build_shufflenetv2
from models.squeezenet     import build_squeezenet
from models.agrigatenet    import build_agrigatenet


# ─────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────

MODEL_BUILDERS = {
    "baseline":     lambda: build_baseline_cnn(num_classes=NUM_CLASSES),
    "mobilenetv3":  lambda: build_mobilenetv3(num_classes=NUM_CLASSES, pretrained=False),
    "efficientnet": lambda: build_efficientnet(num_classes=NUM_CLASSES, pretrained=False),
    "shufflenetv2": lambda: build_shufflenetv2(num_classes=NUM_CLASSES, pretrained=False),
    "squeezenet":   lambda: build_squeezenet(num_classes=NUM_CLASSES, pretrained=False),
    "agrigatenet":  lambda: build_agrigatenet(num_classes=NUM_CLASSES),
}


# ─────────────────────────────────────────────────────────────────
# Inference → Collect All Predictions
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run full inference on a DataLoader.
    Returns (all_labels, all_preds) as numpy arrays.
    """
    model.eval()
    all_labels, all_preds = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────────────────────────────
# Confusion Matrix Plot
# ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    save_dir: Path,
    normalize: bool = True,
) -> None:
    """
    Plot and save a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = f"Normalised Confusion Matrix — {model_name}"
    else:
        cm_plot = cm
        fmt = "d"
        title = f"Confusion Matrix — {model_name}"

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
    )
    ax.set_xlabel("Predicted Label", fontsize=13, labelpad=10)
    ax.set_ylabel("True Label",      fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0,  fontsize=9)
    plt.tight_layout()

    save_path = save_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📊  Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────
# Full Evaluation for One Model
# ─────────────────────────────────────────────────────────────────

def evaluate_model(
    model_name: str,
    checkpoint_path: str,
    data_root: str,
    results_dir: str,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """
    Load a checkpoint and run comprehensive evaluation.
    Returns a metrics dict.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f" Evaluating: {model_name.upper()}  (split={split})")
    print(f"{'='*60}")

    # Load model
    model = MODEL_BUILDERS[model_name]()
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model = model.to(device)

    # Data
    loaders = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Predictions
    y_true, y_pred = get_predictions(model, loaders[split], device)

    # Metrics
    accuracy = (y_true == y_pred).mean()
    f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred,    average="macro", zero_division=0)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "model_name":   model_name,
        "split":        split,
        "accuracy":     float(accuracy),
        "f1_macro":     float(f1_macro),
        "f1_weighted":  float(f1_weighted),
        "precision":    float(precision),
        "recall":       float(recall),
        "per_class_f1": {CLASS_NAMES[i]: float(per_class_f1[i])
                         for i in range(len(CLASS_NAMES))},
    }

    # Print
    print(f"\n  Accuracy      : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"  F1 (macro)    : {f1_macro:.4f}")
    print(f"  F1 (weighted) : {f1_weighted:.4f}")
    print(f"  Precision     : {precision:.4f}")
    print(f"  Recall        : {recall:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    ))

    # Confusion matrix
    plot_confusion_matrix(
        y_true, y_pred,
        class_names=CLASS_NAMES,
        model_name=model_name,
        save_dir=results_dir,
        normalize=True,
    )

    # Save metrics JSON
    metrics_path = results_dir / f"metrics_{model_name}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  💾  Metrics saved → {metrics_path}")

    return metrics


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate PRJ-37 models")
    parser.add_argument("--model", type=str, default="agrigatenet",
                        choices=list(MODEL_BUILDERS.keys()) + ["all"])
    parser.add_argument("--checkpoint",     type=str, default=None,
                        help="Path to checkpoint (for single-model eval)")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Directory with all checkpoints (for --model all)")
    parser.add_argument("--data_root",  type=str, default="./data/organized")
    parser.add_argument("--results_dir",type=str, default="./results/eval")
    parser.add_argument("--split",      type=str, default="test",
                        choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers",type=int, default=4)
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    all_metrics = []

    if args.model == "all":
        for model_name in MODEL_BUILDERS:
            ckpt = ckpt_dir / f"{model_name}_best.pth"
            if not ckpt.exists():
                print(f"  ⚠  Checkpoint not found for {model_name}, skipping.")
                continue
            m = evaluate_model(
                model_name=model_name,
                checkpoint_path=str(ckpt),
                data_root=args.data_root,
                results_dir=args.results_dir,
                split=args.split,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            all_metrics.append(m)
    else:
        ckpt = args.checkpoint or str(ckpt_dir / f"{args.model}_best.pth")
        m = evaluate_model(
            model_name=args.model,
            checkpoint_path=ckpt,
            data_root=args.data_root,
            results_dir=args.results_dir,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        all_metrics.append(m)

    # Summary table
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print(f"  {'Model':<16} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}")
        print("  " + "-" * 50)
        for m in all_metrics:
            print(f"  {m['model_name']:<16} {m['accuracy']:>10.4f} "
                  f"{m['f1_macro']:>10.4f} {m['f1_weighted']:>12.4f}")


if __name__ == "__main__":
    main()
