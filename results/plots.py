"""
plots.py
────────
Publication-quality training curves and analysis plots for PRJ-37.

Generates:
  1. Training & validation loss curves (all 6 models, side-by-side)
  2. Training & validation accuracy curves
  3. F1-Score bar chart comparison
  4. FLOPs vs Accuracy scatter plot (the money shot)
  5. Latency vs Accuracy scatter plot

Usage:
    python results/plots.py \
        --checkpoint_dir ./checkpoints \
        --results_dir    ./results
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


MODEL_ORDER = [
    "baseline",
    "mobilenetv3",
    "efficientnet",
    "shufflenetv2",
    "squeezenet",
    "agrigatenet",
]

MODEL_LABELS = {
    "baseline":     "Baseline CNN",
    "mobilenetv3":  "MobileNetV3-Small",
    "efficientnet": "EfficientNet-B0",
    "shufflenetv2": "ShuffleNetV2-0.5x",
    "squeezenet":   "SqueezeNet1.1",
    "agrigatenet":  "AgriGateNet ⭐ (Ours)",
}

# Distinct colour palette
COLOURS = {
    "baseline":     "#9E9E9E",   # grey
    "mobilenetv3":  "#42A5F5",   # blue
    "efficientnet": "#66BB6A",   # green
    "shufflenetv2": "#FFA726",   # orange
    "squeezenet":   "#AB47BC",   # purple
    "agrigatenet":  "#EF5350",   # red (ours — stands out)
}


def load_history(checkpoint_dir: Path, model_name: str) -> dict | None:
    path = checkpoint_dir / f"{model_name}_history.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def plot_training_curves(checkpoint_dir: Path, save_dir: Path):
    """Loss and accuracy curves for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model_name in MODEL_ORDER:
        hist = load_history(checkpoint_dir, model_name)
        if hist is None:
            continue
        c     = COLOURS[model_name]
        label = MODEL_LABELS[model_name]
        epochs = range(1, len(hist["val_loss"]) + 1)

        # Loss
        axes[0].plot(epochs, hist["train_loss"], color=c, linestyle="--", alpha=0.5, linewidth=1.2)
        axes[0].plot(epochs, hist["val_loss"],   color=c, linestyle="-",  linewidth=2.0, label=label)

        # Accuracy
        axes[1].plot(epochs, hist["train_acc"], color=c, linestyle="--", alpha=0.5, linewidth=1.2)
        axes[1].plot(epochs, hist["val_acc"],   color=c, linestyle="-",  linewidth=2.0, label=label)

    for ax, ylabel, title in zip(
        axes,
        ["Cross-Entropy Loss", "Accuracy"],
        ["Training & Validation Loss", "Training & Validation Accuracy"],
    ):
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].text(0.02, 0.02, "Dashed = train | Solid = val",
                 transform=axes[0].transAxes, fontsize=8, color="gray")

    plt.suptitle("PRJ-37 — Model Training Curves (All 6 Models)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = save_dir / "training_curves.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"📈  Training curves saved → {save_path}")


def plot_f1_bar_chart(results_dir: Path, save_dir: Path):
    """F1-score bar chart comparison."""
    eval_dir = results_dir / "eval"
    names, f1s = [], []

    for model_name in MODEL_ORDER:
        path = eval_dir / f"metrics_{model_name}.json"
        if not path.exists():
            continue
        with open(path) as f:
            m = json.load(f)
        names.append(MODEL_LABELS[model_name])
        f1s.append(m.get("f1_macro", 0.0))

    if not names:
        return

    colours = [COLOURS[n] for n in MODEL_ORDER if
               (results_dir / "eval" / f"metrics_{n}.json").exists()]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(names, f1s, color=colours, edgecolor="white", linewidth=0.5)

    # Value labels
    for bar, val in zip(bars, f1s):
        ax.text(
            bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=10, fontweight="bold",
        )

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Macro F1-Score", fontsize=12)
    ax.set_title("PRJ-37 — Macro F1-Score Comparison (Test Set)", fontsize=13, fontweight="bold")
    ax.axvline(x=0.90, color="red", linestyle="--", linewidth=1, alpha=0.7, label="0.90 target")
    ax.legend(fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path = save_dir / "f1_comparison.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"📊  F1 bar chart saved → {save_path}")


def plot_flops_vs_accuracy(results_dir: Path, save_dir: Path):
    """The 'money shot' — FLOPs vs accuracy scatter plot."""
    eval_dir  = results_dir / "eval"
    flops_path = results_dir / "flops_results.json"
    if not flops_path.exists():
        return

    flops_data = {e["model_name"]: e for e in json.load(open(flops_path))}

    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name in MODEL_ORDER:
        metrics_path = eval_dir / f"metrics_{model_name}.json"
        if not metrics_path.exists() or model_name not in flops_data:
            continue

        acc   = json.load(open(metrics_path)).get("accuracy", 0) * 100
        flops = flops_data[model_name]["flops_M"]
        c     = COLOURS[model_name]
        label = MODEL_LABELS[model_name]
        size  = 300 if model_name == "agrigatenet" else 180

        ax.scatter(flops, acc, s=size, c=c, edgecolors="white",
                   linewidths=1.5, zorder=5, label=label)
        ax.annotate(
            label, (flops, acc),
            textcoords="offset points", xytext=(10, 5),
            fontsize=8.5, color=c, fontweight="bold" if model_name == "agrigatenet" else "normal",
        )

    ax.set_xlabel("FLOPs (Millions) — Lower is Better for Edge Deployment", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("PRJ-37 — FLOPs vs Accuracy\n(Ideal model: top-left corner)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path = save_dir / "flops_vs_accuracy.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"🎯  FLOPs-vs-Accuracy plot saved → {save_path}")


def plot_latency_vs_accuracy(results_dir: Path, save_dir: Path):
    """Latency vs accuracy trade-off plot."""
    eval_dir  = results_dir / "eval"
    lat_path  = results_dir / "latency_results.json"
    if not lat_path.exists():
        return

    lat_data = {e["model_name"]: e for e in json.load(open(lat_path))}

    fig, ax = plt.subplots(figsize=(10, 7))

    for model_name in MODEL_ORDER:
        metrics_path = eval_dir / f"metrics_{model_name}.json"
        if not metrics_path.exists() or model_name not in lat_data:
            continue

        acc     = json.load(open(metrics_path)).get("accuracy", 0) * 100
        latency = lat_data[model_name]["mean_ms"]
        size_mb = lat_data[model_name].get("size_mb", 5)
        c       = COLOURS[model_name]
        label   = MODEL_LABELS[model_name]
        dot_sz  = size_mb * 30  # Bubble size proportional to model size

        ax.scatter(latency, acc, s=dot_sz, c=c, alpha=0.85,
                   edgecolors="white", linewidths=1.5, zorder=5, label=label)
        ax.annotate(
            label, (latency, acc),
            textcoords="offset points", xytext=(8, 5),
            fontsize=8.5, color=c, fontweight="bold" if model_name == "agrigatenet" else "normal",
        )

    ax.set_xlabel("Mean Inference Latency (ms, CPU, batch=1) — Lower is Better", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title(
        "PRJ-37 — Latency vs Accuracy Trade-off\n(Bubble size = model file size)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    save_path = save_dir / "latency_vs_accuracy.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"⚡  Latency-vs-Accuracy plot saved → {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate PRJ-37 result plots")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir",    type=str, default="./results")
    args = parser.parse_args()

    ckpt_dir    = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)
    save_dir    = results_dir / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_training_curves(ckpt_dir, save_dir)
    plot_f1_bar_chart(results_dir, save_dir)
    plot_flops_vs_accuracy(results_dir, save_dir)
    plot_latency_vs_accuracy(results_dir, save_dir)

    print(f"\n✅  All plots saved to {save_dir}")


if __name__ == "__main__":
    main()
