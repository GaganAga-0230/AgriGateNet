"""
presentation_demo.py
====================
PRJ-37 AgriGateNet — Live Academic Presentation Tool

Two modules:
  Module 1 — Dataset Explorer
      Randomly samples 9 images from the DeepWeeds dataset and
      displays them in a grid with their ground-truth botanical labels.

  Module 2 — Live Inference Engine
      Takes any unseen image, runs it through the EXACT 3-stage pipeline
      (Laplacian blur check → CLAHE green enhancement → ImageNet norm),
      then predicts the weed class with softmax confidence.

Run with Gradio UI (recommended for presentation):
    python presentation_demo.py --ui

Run in plain CLI mode:
    python presentation_demo.py --cli

Run dataset explorer only:
    python presentation_demo.py --explore --data_root ./data/organized

Run inference on a single image:
    python presentation_demo.py --infer --image path/to/your/image.jpg
"""

import os
import sys
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.agrigatenet import build_agrigatenet
from data.preprocess_pipeline import QualityFilter, CLAHEGreenEnhance

# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee apple",
    "Lantana",
    "Parkinsonia",
    "Parthenium",
    "Prickly acacia",
    "Rubber vine",
    "Siam weed",
    "Snake weed",
    "Negative (non-weed)",
]

# Rich colour per class (for visual variety in the grid)
CLASS_COLORS = [
    "#E53935", "#8E24AA", "#1E88E5", "#00897B",
    "#F4511E", "#FFB300", "#43A047", "#00ACC1", "#546E7A",
]

CHECKPOINT_PATH = "./checkpoints/agrigatenet_best.pth"
DATA_ROOT       = "./data/organized"

# ── Detect device ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_agrigatenet(checkpoint_path: str = CHECKPOINT_PATH) -> torch.nn.Module:
    """Load AgriGateNet with best saved weights (or random weights if no checkpoint)."""
    model = build_agrigatenet(num_classes=NUM_CLASSES)

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=DEVICE)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded weights from: {ckpt_path}")
    else:
        print(f"  WARNING: No checkpoint found at {ckpt_path}")
        print("  Using random (untrained) weights for demo purposes.")
        print("  Run training/train.py --model agrigatenet first for real predictions.")

    model = model.to(DEVICE)
    model.eval()
    return model


def build_inference_transform(use_clahe: bool = True):
    """Build the exact 3-stage inference transform pipeline."""
    transform_list = []

    # Stage 1 is handled separately (Laplacian check before loading)
    # Stage 2: CLAHE green enhancement
    if use_clahe:
        transform_list.append(CLAHEGreenEnhance(clip_limit=2.0, tile_grid_size=(8, 8)))

    # Resize + Stage 3: ImageNet normalisation
    transform_list += [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    return transforms.Compose(transform_list)


def run_quality_check(img_path: str) -> dict:
    """
    Stage 1 — Run QualityFilter checks and return a detailed report dict.
    Does NOT reject the image (we still run inference either way for demo).
    """
    qf = QualityFilter(blur_threshold=100, min_brightness=20.0, max_brightness=235.0)

    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return {"error": "Cannot read image file"}

    lap_var    = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    brightness = float(img_gray.mean())
    is_sharp   = lap_var >= 100
    is_exposed = 20.0 <= brightness <= 235.0
    passes     = is_sharp and is_exposed

    return {
        "laplacian_variance": round(lap_var, 2),
        "brightness":         round(brightness, 2),
        "is_sharp":           is_sharp,
        "is_well_exposed":    is_exposed,
        "passes_filter":      passes,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def get_dataset_image_map(data_root: str) -> dict[str, list[Path]]:
    """
    Scan the organised dataset directory and return a dict:
        {class_name: [list of image Paths]}
    Handles both flat and nested structures.
    """
    data_root = Path(data_root)
    class_map = {}

    if not data_root.exists():
        print(f"  WARNING: Data root not found: {data_root}")
        print("  Run data/download_deepweeds.py first to download the dataset.")
        return {}

    # Walk subdirectories — each subdir name is a class
    for class_dir in sorted(data_root.iterdir()):
        if not class_dir.is_dir():
            continue
        # Match directory name to class names (flexible matching)
        dir_name = class_dir.name.lower().replace("_", " ").replace("-", " ")
        matched_class = None
        for cn in CLASS_NAMES:
            if cn.lower().replace("(non-weed)", "").strip() in dir_name or \
               dir_name in cn.lower():
                matched_class = cn
                break
        if matched_class is None:
            # Use directory name as-is
            matched_class = class_dir.name

        images = (
            list(class_dir.rglob("*.jpg")) +
            list(class_dir.rglob("*.jpeg")) +
            list(class_dir.rglob("*.png"))
        )
        if images:
            class_map[matched_class] = images

    return class_map


def create_dataset_explorer_figure(
    data_root: str = DATA_ROOT,
    n_images: int = 9,
    seed: int = None,
) -> tuple[plt.Figure, str]:
    """
    Module 1 — Dataset Explorer.

    Randomly samples n_images from the dataset (across all classes) and
    displays them in a grid with ground-truth labels overlaid.

    Returns: (matplotlib Figure, status text)
    """
    if seed is not None:
        random.seed(seed)

    class_map = get_dataset_image_map(data_root)

    if not class_map:
        # Generate synthetic placeholder figure when dataset not downloaded
        fig = _synthetic_explorer_figure(n_images)
        status = (
            "Dataset not found at data/organized.\n"
            "Showing synthetic placeholder — run data/download_deepweeds.py first."
        )
        return fig, status

    # Collect all (image_path, class_name) pairs
    all_pairs = []
    for class_name, paths in class_map.items():
        for p in paths:
            all_pairs.append((p, class_name))

    # Sample n_images, trying to ensure class diversity
    classes = list(class_map.keys())
    sampled = []
    if len(classes) >= n_images:
        # One image per class (first n_images classes)
        selected_classes = random.sample(classes, min(n_images, len(classes)))
        for cls in selected_classes:
            sampled.append((random.choice(class_map[cls]), cls))
    else:
        # Sample randomly across all
        sampled = random.sample(all_pairs, min(n_images, len(all_pairs)))

    # Pad to n_images if fewer found
    while len(sampled) < n_images:
        sampled.append(random.choice(all_pairs))

    # ── Build figure ──────────────────────────────────────────────────────────
    cols = 3
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    fig.patch.set_facecolor("#1A1A2E")
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (img_path, class_name) in enumerate(sampled[:n_images]):
        ax = axes[i]
        try:
            img = Image.open(str(img_path)).convert("RGB")
            img_np = np.array(img)

            # Run quality check for display
            qf_result = run_quality_check(str(img_path))
            lap_str = f"Lap={qf_result.get('laplacian_variance', '?'):.0f}"
            brt_str = f"Brt={qf_result.get('brightness', '?'):.0f}"
            qf_ok   = qf_result.get("passes_filter", True)

            ax.imshow(img_np)
        except Exception:
            ax.set_facecolor("#333")
            ax.text(0.5, 0.5, "Load Error", ha="center", va="center",
                    color="red", transform=ax.transAxes, fontsize=12)
            class_name = "Error"
            qf_ok, lap_str, brt_str = True, "", ""

        # Colour-code by class
        cls_idx  = CLASS_NAMES.index(class_name) if class_name in CLASS_NAMES else i % 9
        box_color = CLASS_COLORS[cls_idx]

        # Coloured border rectangle
        for spine in ax.spines.values():
            spine.set_edgecolor(box_color)
            spine.set_linewidth(4)

        # Ground-truth label overlay (top banner)
        ax.text(
            0.5, 0.97, class_name.upper(),
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.3", fc=box_color, ec="white", lw=1.5, alpha=0.92),
        )

        # Quality filter status (bottom right)
        qf_color = "#4CAF50" if qf_ok else "#F44336"
        qf_label = "PASS" if qf_ok else "REJECT"
        ax.text(
            0.98, 0.03,
            f"QF: {qf_label}\n{lap_str}  {brt_str}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="white",
            bbox=dict(boxstyle="round,pad=0.2", fc=qf_color, alpha=0.85),
        )

        # Image filename (bottom left)
        ax.text(
            0.02, 0.03,
            Path(str(img_path)).name[:20],
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color="#CCCCCC",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
        )
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused axes
    for j in range(len(sampled), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "DeepWeeds Dataset Explorer — Ground-Truth Labels & Quality Filter Status",
        fontsize=14, fontweight="bold", color="white", y=1.01,
    )
    plt.tight_layout(pad=0.5)

    status = (
        f"Showing {min(n_images, len(sampled))} random images from {len(all_pairs):,} total.\n"
        f"Dataset classes found: {len(class_map)} | "
        f"QF: Green border=PASS, Red border=REJECT (too blurry/dark/overexposed)"
    )
    return fig, status


def _synthetic_explorer_figure(n_images: int = 9) -> plt.Figure:
    """Generate a placeholder grid when dataset is not downloaded yet."""
    cols = 3
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    fig.patch.set_facecolor("#1A1A2E")
    axes = axes.flatten()

    np.random.seed(42)
    for i in range(n_images):
        ax = axes[i]
        cls = CLASS_NAMES[i % len(CLASS_NAMES)]
        # Generate synthetic colourful noise as placeholder
        synthetic = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        # Add class-specific tint
        tint = np.array([int(CLASS_COLORS[i % 9][j:j+2], 16) for j in (1, 3, 5)])
        synthetic = np.clip(synthetic // 2 + tint[None, None, :] // 2, 0, 255).astype(np.uint8)
        ax.imshow(synthetic)
        ax.text(0.5, 0.97, cls.upper(), transform=ax.transAxes,
                ha="center", va="top", fontsize=10, fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.3", fc=CLASS_COLORS[i % 9], alpha=0.9))
        ax.text(0.5, 0.5, "[SYNTHETIC\nPLACEHOLDER]", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="white",
                fontweight="bold", alpha=0.7)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(CLASS_COLORS[i % 9]); spine.set_linewidth(3)

    for j in range(n_images, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("DeepWeeds Dataset Explorer (PLACEHOLDER — Download Dataset First)",
                 fontsize=13, fontweight="bold", color="white", y=1.01)
    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — LIVE INFERENCE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def infer_single_image(
    image_input,                       # str path OR PIL.Image OR np.ndarray
    checkpoint_path: str = CHECKPOINT_PATH,
    use_clahe: bool = True,
    model: torch.nn.Module = None,     # pass pre-loaded model to avoid reloading
) -> tuple[plt.Figure, str]:
    """
    Module 2 — Live Inference Engine.

    Runs the exact 3-stage preprocessing pipeline on one image, then
    predicts the weed class with full confidence breakdown.

    Returns: (matplotlib Figure showing pipeline + results, text summary)
    """
    # ── 1. Load image ──────────────────────────────────────────────────────────
    img_path = None
    if isinstance(image_input, str):
        img_path = image_input
        try:
            pil_img = Image.open(image_input).convert("RGB")
        except Exception as e:
            return None, f"ERROR: Cannot open image: {e}"

    elif isinstance(image_input, np.ndarray):
        # From Gradio (HWC uint8 RGB)
        pil_img = Image.fromarray(image_input.astype(np.uint8))
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        return None, "ERROR: Unsupported image format."

    raw_img_np = np.array(pil_img)

    # ── 2. Stage 1: Quality Filter ────────────────────────────────────────────
    qf_report = {}
    if img_path:
        qf_report = run_quality_check(img_path)
    else:
        # Compute from array directly (no file path)
        gray = cv2.cvtColor(raw_img_np, cv2.COLOR_RGB2GRAY)
        lap_var    = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        qf_report  = {
            "laplacian_variance": round(lap_var, 2),
            "brightness":         round(brightness, 2),
            "is_sharp":           lap_var >= 100,
            "is_well_exposed":    20.0 <= brightness <= 235.0,
            "passes_filter":      (lap_var >= 100) and (20.0 <= brightness <= 235.0),
        }

    qf_pass = qf_report.get("passes_filter", True)

    # ── 3. Stage 2: CLAHE Green-Channel Enhancement ───────────────────────────
    clahe_enhancer = CLAHEGreenEnhance(clip_limit=2.0, tile_grid_size=(8, 8))
    clahe_img = clahe_enhancer(pil_img)
    clahe_img_np = np.array(clahe_img)

    # ── 4. Stage 3: Resize + ImageNet Normalise → Tensor ─────────────────────
    inference_tf = build_inference_transform(use_clahe=False)  # CLAHE already done
    input_tensor = inference_tf(clahe_img).unsqueeze(0).to(DEVICE)   # (1,3,224,224)

    # ── 5. Load model & run forward pass ─────────────────────────────────────
    if model is None:
        model = load_agrigatenet(checkpoint_path)

    with torch.no_grad():
        logits = model(input_tensor)              # (1, 9)
        probs  = F.softmax(logits, dim=1)[0]      # (9,)
        top_k  = torch.topk(probs, k=5)

    pred_idx  = top_k.indices[0].item()
    pred_conf = top_k.values[0].item() * 100.0
    pred_name = CLASS_NAMES[pred_idx]

    top5_indices = top_k.indices.tolist()
    top5_probs   = top_k.values.tolist()

    # ── 6. Build multi-panel figure ───────────────────────────────────────────
    fig = plt.figure(figsize=(18, 10), facecolor="#0D1117")

    # Panel layout: [original | CLAHE | green-diff] [confidence bar] [top-5]
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3,
                          left=0.04, right=0.97, top=0.92, bottom=0.06)

    # --- Panel A: Original Image ---
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_orig.imshow(raw_img_np)
    ax_orig.set_title("STAGE 1\nOriginal Image", color="white", fontsize=11, fontweight="bold")
    ax_orig.set_xticks([]); ax_orig.set_yticks([])

    # Quality filter badge
    badge_col = "#4CAF50" if qf_pass else "#F44336"
    badge_txt = "QF: PASS" if qf_pass else "QF: REJECT"
    ax_orig.text(0.02, 0.02, badge_txt, transform=ax_orig.transAxes,
                 fontsize=9, fontweight="bold", color="white", va="bottom",
                 bbox=dict(fc=badge_col, ec="white", lw=1, boxstyle="round,pad=0.3"))
    ax_orig.text(0.02, 0.15,
                 f"Lapvar={qf_report.get('laplacian_variance','?')}\n"
                 f"Brt={qf_report.get('brightness','?')}",
                 transform=ax_orig.transAxes, fontsize=7.5, color="#CCCCCC",
                 bbox=dict(fc="#333", alpha=0.8, boxstyle="round,pad=0.2"))
    for sp in ax_orig.spines.values():
        sp.set_edgecolor(badge_col); sp.set_linewidth(3)

    # --- Panel B: After CLAHE ---
    ax_clahe = fig.add_subplot(gs[0, 1])
    ax_clahe.imshow(clahe_img_np)
    ax_clahe.set_title("STAGE 2\nCLAHE Green Enhancement", color="white",
                        fontsize=11, fontweight="bold")
    ax_clahe.set_xticks([]); ax_clahe.set_yticks([])
    for sp in ax_clahe.spines.values():
        sp.set_edgecolor("#42A5F5"); sp.set_linewidth(3)

    # --- Panel C: Green Channel Difference ---
    ax_diff = fig.add_subplot(gs[0, 2])
    diff = (clahe_img_np[:, :, 1].astype(int) - raw_img_np[:, :, 1].astype(int))
    diff_norm = ((diff - diff.min()) / max(diff.max() - diff.min(), 1) * 255).astype(np.uint8)
    ax_diff.imshow(diff_norm, cmap="RdYlGn")
    ax_diff.set_title("STAGE 2 — Effect\nCLAHE Green-Channel Delta", color="white",
                       fontsize=11, fontweight="bold")
    ax_diff.set_xticks([]); ax_diff.set_yticks([])
    for sp in ax_diff.spines.values():
        sp.set_edgecolor("#66BB6A"); sp.set_linewidth(3)

    # --- Panel D: Stage 3 (Normalised tensor visualised) ---
    ax_norm = fig.add_subplot(gs[0, 3])
    # De-normalise for display only
    denorm_mean = torch.tensor([0.485, 0.456, 0.406])
    denorm_std  = torch.tensor([0.229, 0.224, 0.225])
    vis_tensor  = input_tensor[0].cpu() * denorm_std[:, None, None] + denorm_mean[:, None, None]
    vis_np      = (vis_tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
    ax_norm.imshow(vis_np)
    ax_norm.set_title("STAGE 3\nResized & Normalised (224x224)", color="white",
                       fontsize=11, fontweight="bold")
    ax_norm.set_xticks([]); ax_norm.set_yticks([])
    for sp in ax_norm.spines.values():
        sp.set_edgecolor("#FFA726"); sp.set_linewidth(3)

    # --- Panel E: Main prediction result ---
    ax_pred = fig.add_subplot(gs[1, :2])
    ax_pred.set_facecolor("#161B22")
    ax_pred.set_xlim(0, 1); ax_pred.set_ylim(0, 1)
    ax_pred.set_xticks([]); ax_pred.set_yticks([])

    pred_class_idx = CLASS_NAMES.index(pred_name) if pred_name in CLASS_NAMES else 0
    pred_color = CLASS_COLORS[pred_class_idx]

    # Large prediction text
    ax_pred.text(0.5, 0.75, "AgriGateNet Prediction",
                 ha="center", va="center", fontsize=13, color="#8B949E",
                 fontweight="bold", transform=ax_pred.transAxes)
    ax_pred.text(0.5, 0.52, pred_name.upper(),
                 ha="center", va="center", fontsize=20, color=pred_color,
                 fontweight="bold", transform=ax_pred.transAxes,
                 bbox=dict(fc="#0D1117", ec=pred_color, lw=2.5,
                           boxstyle="round,pad=0.4", alpha=0.95))
    ax_pred.text(0.5, 0.25, f"Confidence: {pred_conf:.1f}%",
                 ha="center", va="center", fontsize=16, color="white",
                 fontweight="bold", transform=ax_pred.transAxes)

    # Confidence meter (horizontal bar)
    bar_bg = patches.FancyBboxPatch((0.08, 0.08), 0.84, 0.12,
                                     boxstyle="round,pad=0.01",
                                     fc="#21262D", ec="#30363D", lw=1.5,
                                     transform=ax_pred.transAxes, zorder=1)
    ax_pred.add_patch(bar_bg)
    bar_fill = patches.FancyBboxPatch((0.08, 0.08), 0.84 * (pred_conf / 100), 0.12,
                                       boxstyle="round,pad=0.01",
                                       fc=pred_color, alpha=0.85,
                                       transform=ax_pred.transAxes, zorder=2)
    ax_pred.add_patch(bar_fill)
    ax_pred.text(0.5, 0.13, f"{pred_conf:.1f}%",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color="white", transform=ax_pred.transAxes, zorder=3)

    ax_pred.set_title("PREDICTION RESULT", color="white", fontsize=12, fontweight="bold")
    for sp in ax_pred.spines.values():
        sp.set_edgecolor(pred_color); sp.set_linewidth(2.5)

    # --- Panel F: Top-5 confidence bar chart ---
    ax_top5 = fig.add_subplot(gs[1, 2:])
    ax_top5.set_facecolor("#161B22")

    top5_names  = [CLASS_NAMES[i][:18] for i in top5_indices]
    top5_vals   = [v * 100 for v in top5_probs]
    top5_cols   = [CLASS_COLORS[i] for i in top5_indices]

    bars = ax_top5.barh(top5_names[::-1], top5_vals[::-1],
                         color=top5_cols[::-1], edgecolor="#30363D", linewidth=0.8)
    for bar, val in zip(bars, top5_vals[::-1]):
        ax_top5.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}%", va="center", ha="left",
                     fontsize=9.5, color="white", fontweight="bold")

    ax_top5.set_xlim(0, 105)
    ax_top5.set_xlabel("Softmax Confidence (%)", color="white", fontsize=10)
    ax_top5.set_title("TOP-5 CLASS PROBABILITIES", color="white",
                       fontsize=12, fontweight="bold")
    ax_top5.tick_params(colors="white"); ax_top5.xaxis.label.set_color("white")
    ax_top5.set_facecolor("#161B22")
    for sp in ax_top5.spines.values():
        sp.set_edgecolor("#30363D")
    ax_top5.tick_params(axis="y", colors="white", labelsize=9)
    ax_top5.tick_params(axis="x", colors="#8B949E")
    ax_top5.grid(axis="x", alpha=0.2, color="#30363D")

    # ── Figure title ──────────────────────────────────────────────────────────
    device_str = "GPU (CUDA)" if DEVICE.type == "cuda" else "CPU"
    ckpt_exists = Path(checkpoint_path).exists()
    weights_str = "Trained Weights" if ckpt_exists else "RANDOM WEIGHTS (no checkpoint)"

    fig.suptitle(
        f"PRJ-37 AgriGateNet — Live Inference Pipeline | "
        f"Device: {device_str} | Weights: {weights_str}",
        fontsize=13, fontweight="bold", color="white",
    )

    # ── Text summary ──────────────────────────────────────────────────────────
    qf_status = "PASS" if qf_pass else "REJECT (but inference continued for demo)"
    summary = (
        f"PIPELINE REPORT\n"
        f"{'='*45}\n"
        f"Stage 1 — Quality Filter:\n"
        f"  Laplacian Variance : {qf_report.get('laplacian_variance', 'N/A')}"
        f"  (threshold: 100) -> {'SHARP' if qf_report.get('is_sharp') else 'BLURRY'}\n"
        f"  Mean Brightness    : {qf_report.get('brightness', 'N/A')}"
        f"  (range: 20-235) -> {'OK' if qf_report.get('is_well_exposed') else 'OUT OF RANGE'}\n"
        f"  Overall QF Status  : {qf_status}\n\n"
        f"Stage 2 — CLAHE Green Enhancement:\n"
        f"  clip_limit=2.0, tile_grid=(8,8)\n"
        f"  Applied to green channel only (vegetation discriminator)\n\n"
        f"Stage 3 — Normalisation:\n"
        f"  Resize to 224x224, ImageNet mean/std normalisation\n\n"
        f"PREDICTION\n"
        f"{'='*45}\n"
        f"  Predicted Class : {pred_name}\n"
        f"  Confidence      : {pred_conf:.2f}%\n\n"
        f"Top-5 Predictions:\n"
    )
    for idx, (ci, cv) in enumerate(zip(top5_indices, top5_probs)):
        summary += f"  {idx+1}. {CLASS_NAMES[ci]:<30} {cv*100:6.2f}%\n"
    summary += f"\nDevice: {device_str} | Weights: {weights_str}"

    return fig, summary


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

def launch_gradio_ui():
    """Launch the interactive Gradio presentation interface."""
    try:
        import gradio as gr
    except ImportError:
        print("Installing Gradio...")
        os.system(f"{sys.executable} -m pip install gradio -q")
        import gradio as gr

    print(f"\n  Loading AgriGateNet on {DEVICE}...")
    _model = load_agrigatenet()   # Load once, reuse across calls

    # ── Module 1 wrapper ──────────────────────────────────────────────────────
    def explore_dataset(data_root_input, seed_val, n_images):
        seed = int(seed_val) if str(seed_val).strip() else None
        n    = int(n_images)
        fig, status = create_dataset_explorer_figure(
            data_root=data_root_input or DATA_ROOT,
            n_images=n,
            seed=seed,
        )
        return fig, status

    # ── Module 2 wrapper ──────────────────────────────────────────────────────
    def run_inference(image_input, use_clahe_cb):
        if image_input is None:
            return None, "Please upload or provide an image."
        fig, summary = infer_single_image(
            image_input=image_input,
            checkpoint_path=CHECKPOINT_PATH,
            use_clahe=use_clahe_cb,
            model=_model,
        )
        return fig, summary

    # ── Gradio Layout ─────────────────────────────────────────────────────────
    custom_css = """
    .gradio-container { background: #0D1117; font-family: 'Segoe UI', sans-serif; }
    .gr-panel { background: #161B22; border: 1px solid #30363D; border-radius: 8px; }
    h1, h2, h3 { color: #E6EDF3 !important; }
    label { color: #8B949E !important; }
    .gr-button-primary { background: linear-gradient(135deg, #EF5350, #B71C1C); border: none; }
    .gr-button { border-radius: 6px; font-weight: bold; }
    """

    with gr.Blocks(
        title="PRJ-37 AgriGateNet — Live Demo",
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="orange",
            neutral_hue="slate",
        ),
        css=custom_css,
    ) as demo:

        gr.HTML("""
        <div style="text-align:center; padding:20px; background: linear-gradient(135deg,#1A1A2E,#16213E);
                    border-radius:12px; margin-bottom:16px; border:1px solid #EF5350;">
          <h1 style="color:#EF5350; font-size:2em; margin:0;">
            AgriGateNet — Live Academic Demo
          </h1>
          <p style="color:#8B949E; margin:8px 0 0 0; font-size:1.1em;">
            PRJ-37 &nbsp;|&nbsp; Lightweight CNN for Edge-Based Weed Detection
            &nbsp;|&nbsp; DSSAG Block (Patentable Novelty)
          </p>
          <p style="color:#42A5F5; margin:4px 0 0 0; font-size:0.95em;">
            CO6 — Evaluation of CNN Algorithms on Agricultural Imagery
          </p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Dataset Explorer ────────────────────────────────────────
            with gr.Tab("Module 1 — Dataset Explorer"):
                gr.Markdown("""
                ### DeepWeeds Dataset Explorer
                Randomly samples images from the dataset and displays ground-truth labels.
                The **Quality Filter badge** shows whether each image passes our Stage 1 pipeline check.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        data_root_inp = gr.Textbox(
                            value=DATA_ROOT,
                            label="Dataset Root Directory",
                            placeholder="./data/organized",
                        )
                        n_images_sl = gr.Slider(
                            minimum=3, maximum=12, value=9, step=3,
                            label="Number of Images to Display",
                        )
                        seed_inp = gr.Textbox(
                            value="",
                            label="Random Seed (leave blank for random)",
                            placeholder="e.g. 42",
                        )
                        explore_btn = gr.Button(
                            "Explore Dataset", variant="primary", size="lg"
                        )
                        explore_status = gr.Textbox(
                            label="Status", lines=3, interactive=False
                        )
                    with gr.Column(scale=3):
                        explore_output = gr.Plot(label="Dataset Sample Grid")

                explore_btn.click(
                    fn=explore_dataset,
                    inputs=[data_root_inp, seed_inp, n_images_sl],
                    outputs=[explore_output, explore_status],
                )

            # ── Tab 2: Live Inference Engine ───────────────────────────────────
            with gr.Tab("Module 2 — Live Inference Engine"):
                gr.Markdown("""
                ### Live Inference — 3-Stage Pipeline + AgriGateNet
                Upload **any unseen image** (weed photo, crop photo, or your own).
                The system will run the exact preprocessing pipeline and predict the weed class.
                """)
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            label="Upload Test Image",
                            type="numpy",
                            height=280,
                        )
                        clahe_cb = gr.Checkbox(
                            value=True,
                            label="Apply CLAHE Green Enhancement (Stage 2)",
                        )
                        infer_btn = gr.Button(
                            "Run Inference", variant="primary", size="lg"
                        )
                        gr.Markdown("""
                        **What happens:**
                        1. Stage 1 — Laplacian blur + brightness check
                        2. Stage 2 — CLAHE green channel enhancement
                        3. Stage 3 — Resize 224×224 + ImageNet normalisation
                        4. Forward pass through AgriGateNet (DSSAG blocks)
                        5. Softmax → top-5 class probabilities
                        """)
                        infer_status = gr.Textbox(
                            label="Pipeline Report", lines=18, interactive=False
                        )
                    with gr.Column(scale=3):
                        infer_output = gr.Plot(label="Inference Result")

                infer_btn.click(
                    fn=run_inference,
                    inputs=[image_input, clahe_cb],
                    outputs=[infer_output, infer_status],
                )

            # ── Tab 3: Project Summary ─────────────────────────────────────────
            with gr.Tab("Project Summary"):
                gr.Markdown(f"""
                ## PRJ-37 — AgriGateNet Results Summary

                ### 6-Model Comparison (CO6 Requirement)

                | Model | Params (M) | FLOPs (M) | Accuracy | F1 Macro | Size (MB) |
                |---|---|---|---|---|---|
                | Baseline CNN | 0.39 | 1498 | 54.7% | 0.545 | 1.49 |
                | MobileNetV3-Small | 1.53 | 123 | 88.2% | 0.884 | 5.83 |
                | EfficientNet-B0 | 4.02 | 828 | 91.4% | 0.907 | 15.33 |
                | ShuffleNetV2-0.5x | 0.35 | 87 | 85.1% | 0.852 | 1.34 |
                | SqueezeNet1.1 | 0.73 | 527 | 82.1% | 0.818 | 2.77 |
                | **AgriGateNet (Ours)** | **0.43** | **594** | **92.7%** | **0.926** | **1.63** |

                ### Key Results
                - **+38% accuracy** over WeedWatch baseline (54.7% → 92.7%)
                - **Highest F1-score** (0.9256) of all 6 models, including pretrained EfficientNet
                - **89% smaller** than EfficientNet-B0 (1.63 MB vs 15.33 MB)
                - Deployed on: `{DEVICE}` | Checkpoint: `{'Found' if Path(CHECKPOINT_PATH).exists() else 'Not found (demo mode)'}`

                ### Novel DSSAG Block
                - **Dual-path DW Conv**: 3×3 (texture) + 5×5 (shape) in parallel
                - **DCT Spectral Gate**: frequency-domain channel attention (0 extra params)
                - **MAX Fusion**: element-wise maximum of dual paths (0 extra params)
                - **Spatial Bottleneck Mask**: suppresses soil/sky background
                """)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CLI MODE
# ══════════════════════════════════════════════════════════════════════════════

def cli_explore(data_root: str, seed: int = 42, save_path: str = "dataset_explorer.png"):
    """CLI: Run dataset explorer and save figure."""
    print(f"\n[Module 1] Dataset Explorer")
    print(f"  Root: {data_root} | Seed: {seed}")
    fig, status = create_dataset_explorer_figure(data_root=data_root, n_images=9, seed=seed)
    print(f"  {status}")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#1A1A2E")
    print(f"  Saved -> {save_path}")
    plt.close(fig)


def cli_infer(image_path: str, checkpoint: str = CHECKPOINT_PATH,
              save_path: str = "inference_result.png"):
    """CLI: Run inference on a single image and save figure."""
    print(f"\n[Module 2] Live Inference Engine")
    print(f"  Image     : {image_path}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Device    : {DEVICE}")

    model = load_agrigatenet(checkpoint)
    fig, summary = infer_single_image(
        image_input=image_path,
        checkpoint_path=checkpoint,
        model=model,
    )
    if fig is None:
        print(f"  ERROR: {summary}")
        return

    print(f"\n{summary}")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0D1117")
    print(f"\n  Saved -> {save_path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PRJ-37 AgriGateNet — Live Presentation Demo",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--ui", action="store_true",
        help="Launch Gradio web UI (recommended for live presentation)",
    )
    parser.add_argument(
        "--cli", action="store_true",
        help="Run both modules in CLI mode (saves PNG outputs)",
    )
    parser.add_argument(
        "--explore", action="store_true",
        help="Run Module 1 (Dataset Explorer) only",
    )
    parser.add_argument(
        "--infer", action="store_true",
        help="Run Module 2 (Live Inference) only",
    )
    parser.add_argument(
        "--image", type=str, default=None,
        help="Path to image for inference (Module 2)",
    )
    parser.add_argument(
        "--data_root", type=str, default=DATA_ROOT,
        help=f"Dataset root directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=CHECKPOINT_PATH,
        help=f"Model checkpoint path (default: {CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for dataset explorer (default: 42)",
    )
    args = parser.parse_args()

    # Default: launch UI if no flags specified
    if not any([args.ui, args.cli, args.explore, args.infer]):
        args.ui = True

    if args.ui:
        print("\nLaunching Gradio UI at http://localhost:7860 ...")
        launch_gradio_ui()

    elif args.cli:
        cli_explore(args.data_root, seed=args.seed)
        if args.image:
            cli_infer(args.image, checkpoint=args.checkpoint)
        else:
            print("\n[Module 2] Skipped — provide --image path/to/image.jpg for inference")

    elif args.explore:
        cli_explore(args.data_root, seed=args.seed)

    elif args.infer:
        if not args.image:
            print("ERROR: --infer requires --image path/to/image.jpg")
            sys.exit(1)
        cli_infer(args.image, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
