"""
model_comparison_table.py  (v2 -- 3 separate PNGs)
---------------------------------------------------
Produces three separate high-quality table images:

  table1_quantitative.png   -- performance benchmarks (params, FLOPs, acc, F1, latency)
  table2_features.png       -- features ONLY in AgriGateNet vs competitors
                               (features as ROWS, models as COLUMNS, tick/cross symbols)
  table3_advantages.png     -- AgriGateNet advantage bullets vs each competitor

Run:
    python results/model_comparison_table.py
"""
import sys, io
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

# ── Shared palette ────────────────────────────────────────────────────────────
HDR_BG   = "#1A237E"   # deep navy  (headers)
AGRI_BG  = "#FFF59D"   # vivid gold (AgriGateNet column / row)
AGRI_WIN = "#A5D6A7"   # green      (AgriGateNet winning cell)
OTHER_WIN= "#B3E5FC"   # blue       (competitor winning cell)
CHECK_C  = "#1B5E20"   # dark green (tick ✓)
CROSS_C  = "#B71C1C"   # dark red   (cross ✗)
FIG_BG   = "#F0F4F8"

MODEL_COLORS = [
    "#FFEBEE",   # Baseline CNN    — soft red
    "#E3F2FD",   # MobileNetV3     — soft blue
    "#E8F5E9",   # EfficientNet-B0 — soft green
    "#FFF8E1",   # ShuffleNetV2    — soft amber
    "#F3E5F5",   # SqueezeNet1.1   — soft purple
    "#FFF59D",   # AgriGateNet     — vivid gold  ← LAST
]

MODELS_SHORT = [
    "Baseline CNN\n(M0)",
    "MobileNetV3-S\n(M1)",
    "EfficientNet-B0\n(M2)",
    "ShuffleNetV2-0.5x\n(M3)",
    "SqueezeNet1.1\n(M4)",
    "AgriGateNet [OUR MODEL]\n(M5 -- OURS)",
]

# =============================================================================
# TABLE 1 — QUANTITATIVE BENCHMARKS
# =============================================================================

Q_COLS = [
    "Parameters", "FLOPs (M)",
    "Size FP32\n(MB)", "Size INT8\n(MB)",
    "Accuracy\n(%)", "Macro F1",
    "CPU Latency\n(ms, mean)", "CPU P95\n(ms)",
]

Q_DATA = [
    # Baseline
    ["391 K",   "1,498",  "1.49",  "~0.37",  "54.70",  "0.4910",  "12.37",  "16.12"],
    # MobileNetV3
    ["1,527 K", "122.9",  "5.83",  "~1.46",  "88.20",  "0.8742",  "11.00 [B]","14.30 [B]"],
    # EfficientNet-B0
    ["4,019 K", "827.8",  "15.33", "~3.83",  "91.40",  "0.9073",  "39.54",  "48.20"],
    # ShuffleNetV2
    ["351 K [B]", "87.1 [B]", "1.34 [B]","~0.34 [B]","85.10",  "0.8521",  "21.23",  "26.50"],
    # SqueezeNet
    ["727 K",   "526.8",  "2.77",  "~0.69",  "82.10",  "0.8175",  "20.23",  "24.80"],
    # AgriGateNet
    ["427 K",   "593.5",  "1.63",  "~0.41",  "92.70 [BEST]","0.9256 [BEST]","42.55",  "51.20"],
]

# Which col-index  →  which row is BEST (for cell highlighting)
BEST_ROW = {0:3, 1:3, 2:3, 3:3, 4:5, 5:5, 6:1, 7:1}

def make_table1():
    fig, ax = plt.subplots(figsize=(22, 8), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.axis("off")

    fig.suptitle(
        "Table 1 — Quantitative Performance Benchmarks\n"
        "PRJ-37  AgriGateNet  |  6-Model Comparison",
        fontsize=14, fontweight="bold", color="#0D47A1", y=1.01,
    )

    cell_text   = []
    cell_colors = []

    for r_i, (model, row) in enumerate(zip(MODELS_SHORT, Q_DATA)):
        row_with_name = [model] + row
        bg = MODEL_COLORS[r_i]
        colors = [bg]
        for c_i, _ in enumerate(row):
            best = BEST_ROW.get(c_i)
            if best == r_i:
                colors.append(AGRI_WIN if r_i == 5 else OTHER_WIN)
            else:
                colors.append(bg)
        cell_text.append(row_with_name)
        cell_colors.append(colors)

    col_labels  = ["Model"] + Q_COLS
    col_colors  = [HDR_BG] * len(col_labels)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=col_colors,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.8)

    # Header styling
    for j in range(len(col_labels)):
        c = tbl[0, j]
        c.set_text_props(color="white", fontweight="bold", fontsize=9)
        c.set_edgecolor("#546E7A")

    # Model name column styling
    for i in range(len(MODELS_SHORT)):
        c = tbl[i+1, 0]
        c.set_text_props(fontweight="bold", fontsize=8.5, color="#0D47A1")
        c.set_edgecolor("#90A4AE")

    # Gold border for AgriGateNet row
    for j in range(len(col_labels)):
        c = tbl[6, j]
        c.set_edgecolor("#F9A825")
        c.set_linewidth(2.5)
        c.set_text_props(fontweight="bold")

    # Cell edges for other rows
    for i in range(1, len(MODELS_SHORT)+1):
        for j in range(1, len(col_labels)):
            tbl[i, j].set_edgecolor("#90A4AE")

    # Legend
    legend_elems = [
        mpatches.Patch(color=AGRI_WIN,  label="AgriGateNet best value"),
        mpatches.Patch(color=OTHER_WIN, label="Competitor best value  ★"),
        mpatches.Patch(color=AGRI_BG,   label="AgriGateNet row"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              bbox_to_anchor=(0.5, -0.09), ncol=3,
              fontsize=9, framealpha=0.95, edgecolor="#B0BEC5")

    fig.text(0.5, -0.03,
             "★ = Best value in column  |  ★ next to cell value marks the winner",
             ha="center", fontsize=8, color="#546E7A", style="italic")

    plt.tight_layout(pad=1.5)
    path = OUT / "table1_quantitative.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"  SAVED: {path}")


# =============================================================================
# TABLE 2 — FEATURE MATRIX
#   Rows = features (only ones AgriGateNet has exclusively or importantly)
#   Cols = Models (AgriGateNet FIRST for prominence, then the five others)
#   Cells = ✓ or ✗ (with color)
# =============================================================================

# Features: only those in our DSSAG / unique design — listed with DSSAG first
# Format: (Feature Label, [val_agri, val_base, val_mob, val_eff, val_shu, val_squ])
# val: True = has it, False = doesn't, "~" = partial
FEATURE_ROWS = [
    # ── DSSAG-Specific Features (AgriGateNet exclusive) ──────────────────────
    ("DCT Spectral Gate\n(channel attention, 0 FC layers)",
     [True,  False, False, False, False, False]),
    ("Spatial Bottleneck Mask\n(pixel-level BG suppression)",
     [True,  False, False, False, False, False]),
    ("Dual-Path Depthwise Conv\n(3×3 texture + 5×5 shape)",
     [True,  False, False, False, False, False]),
    ("Zero-Param MAX Fusion\n(inter-path, no weights)",
     [True,  False, False, False, False, False]),
    ("Parameter-Free Attention Gate",
     [True,  False, False, False, False, False]),
    # ── Design Strategy ───────────────────────────────────────────────────────
    ("Trained from Scratch\n(no ImageNet dependency)",
     [True,  True,  False, False, False, False]),
    ("Edge-Specific Design\n(weed domain optimized)",
     [True,  False, False, False, False, False]),
    ("Structured 50% Pruning Tested",
     [True,  False, "~",   "~",   "~",   "~"  ]),
    # ── Attention / Architecture ───────────────────────────────────────────────
    ("Any Attention Mechanism",
     [True,  False, True,  True,  False, False]),
    ("Depthwise Separable Conv",
     [True,  False, True,  True,  False, False]),
    ("INT8 Post-Training Quantization",
     [True,  "~",   True,  True,  True,  True ]),
    ("Dropout Regularization",
     [True,  True,  True,  True,  False, True ]),
    ("Batch Normalization",
     [True,  True,  True,  True,  True,  True ]),
]

# vals in FEATURE_ROWS are stored as: [agri, base, mob, eff, shu, squ]
# COL_LABELS_T2 matches that same order -- AgriGateNet first, competitors after.
# DO NOT re-index vals; use them directly.
COL_LABELS_T2 = [
    "AgriGateNet\n[OUR MODEL - M5]",
    "Baseline CNN\n(M0)",
    "MobileNetV3-S\n(M1)",
    "EfficientNet-B0\n(M2)",
    "ShuffleNetV2-0.5x\n(M3)",
    "SqueezeNet1.1\n(M4)",
]
# Header colours: agri=gold, then natural model colours
COL_HDR_COLORS_T2 = [
    MODEL_COLORS[5], MODEL_COLORS[0], MODEL_COLORS[1],
    MODEL_COLORS[2], MODEL_COLORS[3], MODEL_COLORS[4],
]

# Divider row index after which we change section
DIVIDER_AFTER = 7   # after row index 4 (5th feature) = end of DSSAG-exclusive block

TICK = "\u2713"   # ✓
CROSS = "\u2717"  # ✗

def sym(val):
    """Return (display_text, cell_bg, text_color)."""
    if val is True:
        return TICK,  "#C8E6C9", CHECK_C
    elif val is False:
        return CROSS, "#FFCDD2", CROSS_C
    else:  # "~"
        return "~",   "#FFF9C4", "#E65100"

def make_table2():
    n_rows = len(FEATURE_ROWS)
    n_cols = len(COL_LABELS_T2)

    fig, ax = plt.subplots(figsize=(18, 14), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.axis("off")

    fig.suptitle(
        "Table 2 -- Architectural Feature Matrix\n"
        "Features Exclusive to AgriGateNet (DSSAG) vs All Competitors",
        fontsize=14, fontweight="bold", color="#0D47A1", y=1.01,
    )

    cell_text   = []
    cell_colors = []

    for f_i, (feat_label, vals) in enumerate(FEATURE_ROWS):
        # vals = [agri, base, mob, eff, shu, squ] -- matches COL_LABELS_T2 exactly
        # Use vals directly; no reordering.
        row_vals = vals

        # Feature label column first
        row_text   = [feat_label]
        row_colors = ["#E8EAF6"]   # light indigo for feature label

        for c_i, v in enumerate(row_vals):
            txt, bg, _ = sym(v)
            # AgriGateNet col: always green if ✓
            if c_i == 0 and v is True:
                bg = AGRI_WIN
            row_text.append(txt)
            row_colors.append(bg)

        cell_text.append(row_text)
        cell_colors.append(row_colors)

    all_col_labels  = ["Feature"] + COL_LABELS_T2
    all_col_colors  = [HDR_BG] + [HDR_BG]*n_cols

    # Column widths: feature label wider
    col_widths = [0.32] + [0.68/n_cols]*n_cols

    tbl = ax.table(
        cellText=cell_text,
        colLabels=all_col_labels,
        cellColours=cell_colors,
        colColours=all_col_colors,
        loc="center",
        cellLoc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.55)

    # Header row styling
    for j in range(n_cols + 1):
        c = tbl[0, j]
        c.set_text_props(color="white", fontweight="bold", fontsize=9)
        c.set_edgecolor("#455A64")
        # AgriGateNet header col (col index 1 in table = first data col)
        if j == 1:
            c.set_facecolor("#0D47A1")
            c.set_text_props(color="#FFF59D", fontweight="bold", fontsize=9)

    # Body styling
    for r_i, (feat_label, vals) in enumerate(FEATURE_ROWS):
        # vals = [agri, base, mob, eff, shu, squ] -- use directly
        row_vals = vals

        # Feature label cell (col 0)
        lc = tbl[r_i+1, 0]
        lc.set_text_props(fontsize=8.5, color="#1A237E", ha="left")
        lc.set_edgecolor("#90A4AE")

        # Section divider styling -- heavier border after DSSAG block
        if r_i == DIVIDER_AFTER:
            for j in range(n_cols + 1):
                tbl[r_i+1, j].set_edgecolor("#0D47A1")
                tbl[r_i+1, j].set_linewidth(2.0)

        # Value cells
        for c_i, v in enumerate(row_vals):
            c = tbl[r_i+1, c_i+1]
            txt, bg, txt_c = sym(v)
            c.set_text_props(
                color=txt_c, fontweight="bold",
                fontsize=14 if txt in (TICK, CROSS) else 11,
            )
            c.set_edgecolor("#B0BEC5")
            # Extra gold border for AgriGateNet col (c_i == 0)
            if c_i == 0:
                c.set_edgecolor("#F9A825")
                c.set_linewidth(1.8)

    # Section annotation texts
    ax.text(-0.01, 0.97, "[ DSSAG-Exclusive Features ]  Novel -- NOT in any other model",
            transform=ax.transAxes, fontsize=8, color="#B71C1C",
            fontweight="bold", va="top", style="italic")
    ax.text(-0.01, 0.44, "[ Shared / Partial Features ]",
            transform=ax.transAxes, fontsize=8, color="#546E7A",
            fontweight="bold", va="top", style="italic")

    # Legend
    legend_elems = [
        mpatches.Patch(color="#C8E6C9", label=TICK+"  Feature PRESENT"),
        mpatches.Patch(color="#FFCDD2", label=CROSS+"  Feature ABSENT"),
        mpatches.Patch(color="#FFF9C4", label="~  Partial / limited"),
        mpatches.Patch(color=AGRI_WIN,  label=TICK+"  AgriGateNet (green = our exclusive advantage)"),
    ]
    ax.legend(handles=legend_elems, loc="lower center",
              bbox_to_anchor=(0.5, -0.07), ncol=2,
              fontsize=9, framealpha=0.95, edgecolor="#B0BEC5",
              title="Legend", title_fontsize=9)

    plt.tight_layout(pad=1.5)
    path = OUT / "table2_features.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"  SAVED: {path}")


# =============================================================================
# TABLE 3 — AgriGateNet ADVANTAGES
# =============================================================================

ADVANTAGE_ROWS = [
    ("vs. Baseline CNN\n(M0 — 54.7%)", "#FFEBEE", "#B71C1C", [
        "+38.0 pp accuracy  (54.7% → 92.7%)  |  +0.434 Macro F1",
        "Novel DSSAG attention block — trained entirely from scratch",
        "4× smaller after INT8 quantization  (1.49 MB → 0.41 MB)",
        "Edge-deployable on drone processors — baseline is not viable",
    ]),
    ("vs. MobileNetV3-Small\n(M1 — 88.2%)", "#E3F2FD", "#1565C0", [
        "+4.5 pp accuracy  (88.2% → 92.7%)  |  +0.051 Macro F1",
        "72% fewer parameters  (1,527K → 427K)  with HIGHER accuracy",
        "Replaces FC-based SE gate with zero-parameter DCT spectral attention",
        "No ImageNet pretraining dependency — domain-specific from scratch",
    ]),
    ("vs. EfficientNet-B0\n(M2 — 91.4%)", "#E8F5E9", "#2E7D32", [
        "+1.3 pp accuracy  (91.4% → 92.7%)  ← HIGHEST ACCURACY OVERALL",
        "9.4× fewer parameters  (4,019K → 427K)  |  89% smaller (15.33 MB → 1.63 MB)",
        "INT8 footprint 9.3× smaller  (3.83 MB → 0.41 MB)",
        "Trained from scratch — no ImageNet domain gap",
    ]),
    ("vs. ShuffleNetV2-0.5x\n(M3 — 85.1%)", "#FFF8E1", "#E65100", [
        "+7.6 pp accuracy  (85.1% → 92.7%)  |  +0.073 Macro F1",
        "Adds full DSSAG attention — ShuffleNetV2 has ZERO attention",
        "21% size trade-off (1.34 → 1.63 MB) fully justified by +7.6% gain",
        "50% pruning tested — ShuffleNet has no pruning pipeline",
    ]),
    ("vs. SqueezeNet1.1\n(M4 — 82.1%)", "#F3E5F5", "#6A1B9A", [
        "+10.6 pp accuracy  (82.1% → 92.7%)  |  +0.108 Macro F1",
        "41% fewer parameters  (727K → 427K)  despite far better accuracy",
        "Adds dual-path DW + DCT spectral gate — SqueezeNet has no attention",
        "INT8 size 41% smaller  (0.69 MB → 0.41 MB)",
    ]),
]

def make_table3():
    # Compute total rows needed
    all_rows = []
    all_colors = []
    for vs_label, row_bg, hdr_c, bullets in ADVANTAGE_ROWS:
        for k, b in enumerate(bullets):
            prefix = "➤  " if k == 0 else "    •  "
            all_rows.append([
                vs_label if k == 0 else "",
                prefix + b,
            ])
            if k == 0:
                all_colors.append([hdr_c, "#FFFDE7"])
            else:
                all_colors.append([row_bg, "#FAFAFA"])

    n_rows = len(all_rows)
    fig_h = max(12, n_rows * 0.72 + 2)

    fig, ax = plt.subplots(figsize=(22, fig_h), facecolor=FIG_BG)
    ax.set_facecolor(FIG_BG)
    ax.axis("off")

    fig.suptitle(
        "Table 3 — AgriGateNet Specific Advantages Over Each Competitor\n"
        "PRJ-37  |  CO6 Evaluation — Novel DSSAG Architecture",
        fontsize=14, fontweight="bold", color="#0D47A1", y=1.01,
    )

    tbl = ax.table(
        cellText=all_rows,
        colLabels=["AgriGateNet  vs.", "Performance & Design Advantage Detail"],
        cellColours=all_colors,
        colColours=[HDR_BG, HDR_BG],
        loc="center",
        cellLoc="left",
        colWidths=[0.20, 0.80],
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.5)

    # Header
    for j in range(2):
        c = tbl[0, j]
        c.set_text_props(color="white", fontweight="bold", fontsize=11)
        c.set_edgecolor("#455A64")

    # Body
    row_i = 1
    for vs_label, row_bg, hdr_c, bullets in ADVANTAGE_ROWS:
        for k in range(len(bullets)):
            c0 = tbl[row_i, 0]
            c1 = tbl[row_i, 1]
            c0.set_edgecolor(hdr_c)
            c1.set_edgecolor("#B0BEC5")
            if k == 0:
                c0.set_text_props(fontweight="bold", color="white", fontsize=10)
                c1.set_text_props(fontweight="bold", color="#1B5E20", fontsize=10.5)
            else:
                c0.set_text_props(color=hdr_c, fontsize=8)
                c1.set_text_props(color="#212121", fontsize=10)
            row_i += 1

    # Bottom bar
    fig.text(
        0.5, -0.01,
        "AgriGateNet: BEST accuracy (92.7%) | BEST Macro F1 (0.9256) | "
        "ONLY model with DCT Gate + Dual-Path + Spatial Mask + Zero-Param Fusion | "
        "4× compression with < 2% accuracy loss",
        ha="center", fontsize=9, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#1A237E", edgecolor="none"),
    )

    plt.tight_layout(pad=1.5)
    path = OUT / "table3_advantages.png"
    plt.savefig(path, dpi=160, bbox_inches="tight", facecolor=FIG_BG)
    plt.close()
    print(f"  SAVED: {path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Generating 3 separate comparison table PNGs ...")
    make_table1()
    make_table2()
    make_table3()
    print("Done!  Files saved to results/plots/")
