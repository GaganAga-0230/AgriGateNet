"""
latency_benchmark.py
────────────────────
CPU inference latency profiling for all 6 PRJ-37 models.

Measures real-world edge-deployment latency (batch=1, CPU) as required
by the Performance Evaluation Framework (Inference Latency in ms).

Methodology:
  • Warm-up: 200 forward passes (to stabilise CPU caches/JIT)
  • Measurement: 1000 forward passes, record wall-clock time per pass
  • Report: mean ± std latency in milliseconds

Usage:
    python evaluation/latency_benchmark.py \
        --checkpoint_dir ./checkpoints \
        --results_dir    ./results
"""

import sys
import time
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataloader import NUM_CLASSES
from models.baseline_cnn   import build_baseline_cnn
from models.mobilenetv3    import build_mobilenetv3
from models.efficientnet   import build_efficientnet
from models.shufflenetv2   import build_shufflenetv2
from models.squeezenet     import build_squeezenet
from models.agrigatenet    import build_agrigatenet


MODEL_BUILDERS = {
    "baseline":     lambda: build_baseline_cnn(num_classes=NUM_CLASSES),
    "mobilenetv3":  lambda: build_mobilenetv3(num_classes=NUM_CLASSES, pretrained=False),
    "efficientnet": lambda: build_efficientnet(num_classes=NUM_CLASSES, pretrained=False),
    "shufflenetv2": lambda: build_shufflenetv2(num_classes=NUM_CLASSES, pretrained=False),
    "squeezenet":   lambda: build_squeezenet(num_classes=NUM_CLASSES, pretrained=False),
    "agrigatenet":  lambda: build_agrigatenet(num_classes=NUM_CLASSES),
}


def get_model_size_mb(model: nn.Module) -> float:
    """Return model size in MB (sum of all parameter bytes)."""
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 ** 2)


@torch.no_grad()
def benchmark_latency(
    model: nn.Module,
    input_size: tuple = (1, 3, 224, 224),
    warmup: int = 200,
    runs: int = 1000,
    device: str = "cpu",
) -> dict:
    """
    Benchmark inference latency for a single model.

    Returns dict with mean_ms, std_ms, min_ms, max_ms, p95_ms.
    """
    model = model.to(device)
    model.eval()
    dummy = torch.randn(*input_size).to(device)

    # Warm-up
    for _ in range(warmup):
        _ = model(dummy)

    # Timed runs
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(latencies.mean()),
        "std_ms":  float(latencies.std()),
        "min_ms":  float(latencies.min()),
        "max_ms":  float(latencies.max()),
        "p95_ms":  float(np.percentile(latencies, 95)),
    }


def run_benchmark(
    checkpoint_dir: str,
    results_dir: str,
    warmup: int = 200,
    runs: int = 1000,
) -> list[dict]:
    """
    Benchmark all models and produce a comparison table.
    """
    ckpt_dir = Path(checkpoint_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Inference Latency Benchmark  (CPU, batch=1, {runs} runs)")
    print(f"{'='*70}")
    print(f"  {'Model':<16} {'Mean (ms)':>10} {'Std (ms)':>9} {'P95 (ms)':>9} {'Size (MB)':>10}")
    print("  " + "-" * 58)

    all_results = []

    for model_name, builder in MODEL_BUILDERS.items():
        model = builder()
        size_mb = get_model_size_mb(model)

        # Load checkpoint if available
        ckpt_path = ckpt_dir / f"{model_name}_best.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state, strict=False)

        stats = benchmark_latency(model, warmup=warmup, runs=runs)
        stats["model_name"] = model_name
        stats["size_mb"]    = size_mb
        all_results.append(stats)

        print(
            f"  {model_name:<16} "
            f"{stats['mean_ms']:>10.2f} "
            f"{stats['std_ms']:>9.2f} "
            f"{stats['p95_ms']:>9.2f} "
            f"{size_mb:>10.2f}"
        )

    # Save JSON
    out_path = results_dir / "latency_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n💾  Results saved → {out_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark for PRJ-37 models")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir",    type=str, default="./results")
    parser.add_argument("--warmup",  type=int, default=200)
    parser.add_argument("--runs",    type=int, default=1000)
    args = parser.parse_args()

    run_benchmark(
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        warmup=args.warmup,
        runs=args.runs,
    )


if __name__ == "__main__":
    main()
