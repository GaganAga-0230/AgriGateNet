"""
comparison_table.py
───────────────────
Aggregates all evaluation results into the final publication-quality
comparison table for PRJ-37.

Reads from:
  • results/eval/metrics_<model>.json     (accuracy, F1)
  • results/latency_results.json          (latency, size)
  • results/flops_results.json            (FLOPs, params)

Produces:
  • results/final_comparison_table.csv
  • results/final_comparison_table.png   (styled table image)

Usage:
    python results/comparison_table.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


MODEL_ORDER = [
    "baseline",
    "mobilenetv3",
    "efficientnet",
    "shufflenetv2",
    "squeezenet",
    "agrigatenet",
]

MODEL_DISPLAY = {
    "baseline":     "Baseline CNN (WeedWatch-style)",
    "mobilenetv3":  "MobileNetV3-Small",
    "efficientnet": "EfficientNet-B0",
    "shufflenetv2": "ShuffleNetV2-0.5x",
    "squeezenet":   "SqueezeNet1.1",
    "agrigatenet":  "AgriGateNet ⭐ (Ours)",
}


def load_json_safe(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def build_comparison_table(results_dir: str = "./results") -> pd.DataFrame:
    results_dir = Path(results_dir)
    eval_dir    = results_dir / "eval"

    # ── Load metric files ─────────────────────────────────────────────────
    # Latency
    latency_data = {}
    lat_path = results_dir / "latency_results.json"
    if lat_path.exists():
        for entry in json.load(open(lat_path)):
            latency_data[entry["model_name"]] = entry

    # FLOPs
    flops_data = {}
    flops_path = results_dir / "flops_results.json"
    if flops_path.exists():
        for entry in json.load(open(flops_path)):
            flops_data[entry["model_name"]] = entry

    # Per-model eval metrics
    rows = []
    for model_name in MODEL_ORDER:
        metrics_path = eval_dir / f"metrics_{model_name}.json"
        metrics = load_json_safe(metrics_path)
        lat     = latency_data.get(model_name, {})
        flops   = flops_data.get(model_name, {})

        rows.append({
            "Model":        MODEL_DISPLAY.get(model_name, model_name),
            "Params (M)":   round(flops.get("params_M",   0.0), 2),
            "FLOPs (M)":    round(flops.get("flops_M",    0.0), 1),
            "Accuracy (%)": round(metrics.get("accuracy", 0.0) * 100, 2),
            "F1 Macro":     round(metrics.get("f1_macro", 0.0), 4),
            "F1 Weighted":  round(metrics.get("f1_weighted", 0.0), 4),
            "Latency (ms)": round(lat.get("mean_ms", 0.0), 2),
            "Size (MB)":    round(lat.get("size_mb", 0.0), 2),
        })

    df = pd.DataFrame(rows)
    return df


def style_and_save_table(df: pd.DataFrame, results_dir: str = "./results"):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = results_dir / "final_comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"💾  CSV table saved → {csv_path}")

    # ── Matplotlib publication-quality table ──────────────────────────────
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.axis("off")

    # Colour rows: highlight AgriGateNet in gold
    row_colours = []
    for i, model in enumerate(df["Model"]):
        if "AgriGateNet" in model or "Ours" in model:
            row_colours.append(["#FFF9C4"] * len(df.columns))  # light gold
        elif i == 0:
            row_colours.append(["#FFEBEE"] * len(df.columns))  # light red (worst)
        else:
            row_colours.append(["#F5F5F5"] * len(df.columns))

    col_colours = [["#37474F"] * len(df.columns)]  # dark header

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellColours=row_colours,
        colColours=col_colours,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.0)

    # Header font colour white
    for j in range(len(df.columns)):
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Bold AgriGateNet row
    agri_row = None
    for i, model in enumerate(df["Model"]):
        if "AgriGateNet" in str(model):
            agri_row = i + 1  # +1 for header row
    if agri_row:
        for j in range(len(df.columns)):
            table[agri_row, j].set_text_props(fontweight="bold", color="#E65100")

    gold_patch = mpatches.Patch(color="#FFF9C4", label="Our Novel Model (AgriGateNet)")
    red_patch  = mpatches.Patch(color="#FFEBEE", label="Baseline CNN (reference floor)")
    ax.legend(handles=[gold_patch, red_patch], loc="lower center",
              bbox_to_anchor=(0.5, -0.05), fontsize=9, ncol=2)

    ax.set_title(
        "PRJ-37 — Model Comparison Table: Accuracy, FLOPs, Latency & Size",
        fontsize=13, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    img_path = results_dir / "final_comparison_table.png"
    plt.savefig(img_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"🖼️   Table image saved → {img_path}")


def main():
    df = build_comparison_table(results_dir="./results")

    print("\n" + "=" * 90)
    print("  PRJ-37 Final Comparison Table")
    print("=" * 90)
    print(df.to_string(index=False))

    style_and_save_table(df, results_dir="./results")


if __name__ == "__main__":
    main()
