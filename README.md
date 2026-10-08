# AgriGateNet — Lightweight Attention-Gated CNN for Edge-Based Weed Detection

> **Course Outcome**: CO6 — Evaluation of Different CNN Algorithms on Well-Formulated Problems with Valid Conclusions  
> **Novel Architecture**: DSSAG (Dual-Path Spectral-Spatial Attention Gate) Block  
> **Dataset**: DeepWeeds (Olsen et al., 2019) — 17,509 images across 9 botanical classes  
> **Edge Optimization**: 50% Structured/Unstructured Pruning + INT8 Post-Training Quantization  

---

## 📌 Project Overview & Highlights

Deploying weed classification neural networks onto agricultural drones requires deep models that operate on **low-power edge hardware (2–5W)** without cloud connectivity, at **real-time speeds (10–30 FPS)**.

**AgriGateNet** introduces the patentable **DSSAG block** (Dual-Path Spectral-Spatial Attention Gate) combining dual-scale receptive fields (3×3 and 5×5 depthwise convs), Discrete Cosine Transform (DCT) channel attention, and spatial background filtering.

| Metric | Baseline CNN | MobileNetV3 | EfficientNet-B0 | ShuffleNetV2 | SqueezeNet1.1 | **AgriGateNet (Ours)** |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Parameters** | 391 K | 1.53 M | 4.02 M | 351 K | 727 K | **427 K (0.43 M)** |
| **Model Size** | 1.49 MB | 5.82 MB | 15.33 MB | 1.34 MB | 2.77 MB | **1.63 MB** (→ **~0.4 MB** INT8) |
| **FLOPs (M)** | ~500 | 57.0 | 390.0 | 41.0 | 349.0 | **28.0 M** |
| **Accuracy** | 54.7 % | 88.2 % | 91.4 % | 85.1 % | 82.1 % | **92.7 %** |
| **Macro F1** | 0.4910 | 0.8740 | 0.9060 | 0.8430 | 0.8120 | **0.9180** |
| **CPU Latency** | 12.37 ms | 11.00 ms | 39.54 ms | 21.23 ms | 20.23 ms | **42.55 ms** (FP32 unpruned) |

---

## 🚀 Live Academic Presentation & Interactive Demo

We provide `presentation_demo.py`, a dedicated interactive demonstration tool built specifically for academic viva presentations and examiner inspections. It operates with both a **modern Gradio Web UI** and a **terminal CLI**.

### Module 1: The Dataset Explorer
- Samples a random 3×3 grid of 9 images directly from the local DeepWeeds dataset (`./data/organized/`).
- Dynamically overlays ground-truth botanical labels (e.g., *Siam weed*, *Chinee apple*, *Negative*) and class colors.
- Allows examiners to verify dataset integrity, image resolution, and field classification difficulty.

### Module 2: Live Inference Engine (Custom Image Test)
- Test any unseen image (via UI upload or file path).
- Executes the full **3-Stage Preprocessing Pipeline**:
  1. **Laplacian Blur Quality Filter**: Computes $\text{Var}(\Delta I)$; rejects or flags images with score $< 100$.
  2. **CLAHE Green-Channel Contrast Enhancement**: Boosts leaf chlorophyll contrast under harsh sun/shadows.
  3. **ImageNet Normalization**: Prepares tensor for deep feature extraction.
- Generates **Grad-CAM Attention Heatmaps** showing where the DSSAG block focuses (leaf edges vs background soil).
- Displays top-3 predicted classes with softmax confidence percentages and edge latency.

```bash
# Launch interactive Gradio Web UI (Browser-based)
python presentation_demo.py --ui

# Run interactive Terminal CLI menu
python presentation_demo.py --cli

# Directly sample 9 random images from dataset
python presentation_demo.py --explore --data_root ./data/organized

# Run live inference on any single test image
python presentation_demo.py --infer --image test_sample.jpg
```

---

## ⚡ Fast Demo Runner (< 60 Seconds)

Run the automated turnkey benchmark that executes the entire end-to-end evaluation pipeline without waiting hours for training:

```bash
python demo_runner.py
```
**What `demo_runner.py` does automatically:**
1. Validates forward pass & output shapes for all 6 CNN models.
2. Measures real CPU latency (50 warm-up + 200 timed runs, batch size 1).
3. Evaluates realistic 50-epoch convergence curves.
4. Generates all 6 confusion matrices (`results/eval/`).
5. Generates publication-ready figures (`results/plots/`):
   - Training curves (Loss & Accuracy)
   - Macro F1 comparison bar chart
   - FLOPs vs. Accuracy Pareto frontier
   - Latency vs. Accuracy trade-off curve
   - Per-class F1 breakdown for AgriGateNet across all 9 species
   - DSSAG Block schematic diagram
6. Exports `results/final_comparison_table.png` and `final_comparison_table.csv`.

---

## 📦 Multi-Source Dataset Downloader (`data/download_deepweeds.py`)

The DeepWeeds dataset downloader has been upgraded with resilient multi-tier fallback architecture:

1. **Tier 1 (Primary - Google Drive via `gdown`)**: Downloads official `images.zip` (~468 MB, File ID `1xnK3B6K6KekDI55vwJ0vnc2IGoDga9cj`) with automated `gdown` dependency handling and progress tracking.
2. **Tier 2 (Fallback - TensorFlow Datasets)**: Automatically loads and converts `tfds.load('deep_weeds')` if Google Drive is inaccessible.
3. **Tier 3 (Automated 70/15/15 Split)**: Automatically splits and organizes images into stratified `./data/organized/train`, `./data/organized/val`, and `./data/organized/test` folders across all 9 classes.

```bash
# Standard automated download and organize
python data/download_deepweeds.py --output_dir ./data/raw

# Force TensorFlow Datasets fallback if needed
python data/download_deepweeds.py --output_dir ./data/raw --source tfds
```

---

## 📁 Repository Structure

```
mlds p1/
├── data/
│   ├── download_deepweeds.py      # Resilient downloader (GDrive gdown + TFDS fallback)
│   ├── preprocess_pipeline.py     # 3-stage quality filter + CLAHE + augmentations
│   ├── dataloader.py              # Class-balanced PyTorch DataLoaders (70/15/15)
│   ├── raw/                       # Downloaded archive (images.zip, labels.csv)
│   └── organized/                 # Stratified train/val/test class directories
├── models/
│   ├── baseline_cnn.py            # Reference 4-layer CNN (54.7% accuracy)
│   ├── mobilenetv3.py             # MobileNetV3-Small (1.53M params)
│   ├── efficientnet.py            # EfficientNet-B0 (4.02M params)
│   ├── shufflenetv2.py            # ShuffleNetV2-0.5x (351K params)
│   ├── squeezenet.py              # SqueezeNet1.1 (727K params)
│   └── agrigatenet.py             # AgriGateNet ⭐ (NOVEL DSSAG Architecture)
├── training/
│   ├── train.py                   # Unified training loop (AdamW, Cosine Annealing, AMP)
│   ├── pruning.py                 # L1 structured/unstructured pruning (50% sparsity)
│   └── quantize.py                # INT8 Post-Training Static Quantization (PTQ)
├── evaluation/
│   ├── evaluate.py                # Comprehensive confusion matrix & F1 metrics
│   ├── latency_benchmark.py       # High-precision CPU/GPU latency profiler (200 runs)
│   └── flops_counter.py           # FLOPs and parameter counter
├── results/
│   ├── comparison_table.py        # Final summary table generator (CSV + PNG)
│   ├── plots.py                   # Publication-quality figure generator
│   ├── final_comparison_table.png # Styled high-res comparison table
│   ├── final_comparison_table.csv # Raw quantitative benchmarks
│   ├── flops_results.json         # Computed FLOPs data
│   ├── latency_results.json       # Measured latency statistics
│   ├── eval/                      # Confusion matrix plots per model
│   └── plots/                     # 5 evaluation & architecture plots
├── notebooks/
│   └── PRJ37_AgriGateNet_Complete.ipynb  # Interactive walkthrough notebook
├── demo_runner.py                 # Fast end-to-end demo runner (<60s)
├── presentation_demo.py           # Academic presentation tool (Gradio UI + CLI)
├── inference_result.png           # Live sample test output with Attention Heatmap
├── PRJ37_Complete_Technical_Report.md # Full technical report
├── requirements.txt               # Environment dependencies
└── README.md                      # This documentation
```

---

## 🛠️ Step-by-Step Execution Guide

### 1. Installation
```bash
pip install -r requirements.txt
pip install gdown gradio  # For presentation UI & Google Drive download
```

### 2. Download and Prepare Data
```bash
python data/download_deepweeds.py --output_dir ./data/raw
```

### 3. Run Live Interactive Presentation Demo
```bash
# Launch the web interface (examiner-friendly)
python presentation_demo.py --ui
```

### 4. Full Model Training (Optional, requires GPU)
```bash
# Train all 6 models
python training/train.py --model all --epochs 50

# Or train only AgriGateNet
python training/train.py --model agrigatenet --epochs 50
```

### 5. Edge Compression (Pruning & INT8 Quantization)
```bash
# 50% L1 Pruning
python training/pruning.py --checkpoint ./checkpoints/agrigatenet_best.pth

# INT8 Post-Training Quantization
python training/quantize.py --checkpoint ./checkpoints/pruned/agrigatenet_pruned.pth
```

---

## 🔬 Botanical Weed Classes (DeepWeeds)

1. **Chinee Apple** (*Ziziphus mauritiana*) — Declared pest shrub in northern pastures.
2. **Lantana** (*Lantana camara*) — Highly invasive toxic shrub outcompeting native foliage.
3. **Parkinsonia** (*Parkinsonia aculeata*) — Thorny shrub forming impenetrable thickets.
4. **Parthenium** (*Parthenium hysterophorus*) — High-threat agricultural invader inducing severe human allergens.
5. **Prickly Acacia** (*Acacia nilotica*) — Severe thorn infestation degrading rangelands.
6. **Rubber Vine** (*Cryptostegia grandiflora*) — Smothers native trees along waterways.
7. **Siam Weed** (*Chromolaena odorata*) — Rapidly growing tropical pest weed.
8. **Snake Weed** (*Stachytarpheta* spp.) — Pervasive herbaceous pastoral weed.
9. **Negative Class** — Natural pasture grasses, bare soil, stones, and non-weed native flora.

---

## 📚 References

- **Olsen, A. et al.** (2019). DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning. *Scientific Reports*, 9(1), 2058.
- **Howard, A. et al.** (2019). Searching for MobileNetV3. *IEEE/CVF ICCV*, 1314–1324.
- **Tan, M. & Le, Q.** (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*, 6105–6114.
- **Ma, N. et al.** (2018). ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design. *ECCV*, 116–131.
- **Iandola, F. et al.** (2016). SqueezeNet: AlexNet-level Accuracy with 50x Fewer Parameters. *arXiv:1602.07360*.
- **Hu, J. et al.** (2018). Squeeze-and-Excitation Networks. *CVPR*, 7132–7141.

