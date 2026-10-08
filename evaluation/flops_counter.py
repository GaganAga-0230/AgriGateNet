"""
flops_counter.py
────────────────
FLOPs (Floating-Point Operations) and parameter count profiler.

Uses the `thop` library to count multiply-accumulate operations (MACs)
and total parameters for all 6 PRJ-37 models.

Why FLOPs matter for edge deployment:
  • FLOPs directly predict inference speed on CPU/edge hardware
  • Our AgriGateNet targets < 30 MFLOPs (vs ~390 for EfficientNet-B0)
  • This constitutes the patentable 70% FLOPs reduction claim

Usage:
    python evaluation/flops_counter.py
"""

import sys
import json
from pathlib import Path

import torch

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

INPUT_SIZE = (1, 3, 224, 224)


def count_flops_and_params(model: torch.nn.Module) -> tuple[float, float]:
    """
    Count FLOPs (MACs × 2) and total parameters.

    Returns (flops_M, params_M) — both in millions.
    """
    try:
        from thop import profile, clever_format
        dummy = torch.randn(*INPUT_SIZE)
        macs, params = profile(model, inputs=(dummy,), verbose=False)
        flops_M  = macs  * 2 / 1e6   # MACs → FLOPs, then to Millions
        params_M = params / 1e6
    except ImportError:
        # Fallback: manual parameter count only, FLOPs set to -1
        params_M = sum(p.numel() for p in model.parameters()) / 1e6
        flops_M  = -1.0
    return flops_M, params_M


def run_flops_analysis(
    results_dir: str = "./results",
    save_json: bool = True,
) -> list[dict]:
    """
    Count FLOPs and parameters for all models and display a table.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  FLOPs & Parameter Analysis  (input 224×224, batch=1)")
    print(f"{'='*65}")
    print(f"  {'Model':<16} {'FLOPs (M)':>12} {'Params (M)':>12} {'FLOPs vs EfficientNet':>22}")
    print("  " + "-" * 64)

    all_results = []
    efficientnet_flops = None

    for model_name, builder in MODEL_BUILDERS.items():
        model = builder()
        model.eval()
        flops_M, params_M = count_flops_and_params(model)

        if model_name == "efficientnet":
            efficientnet_flops = flops_M

        all_results.append({
            "model_name": model_name,
            "flops_M":    round(flops_M,  2),
            "params_M":   round(params_M, 3),
        })

    # Add relative FLOPs column
    for r in all_results:
        if efficientnet_flops and efficientnet_flops > 0:
            ratio = r["flops_M"] / efficientnet_flops
            r["flops_vs_efficientnet"] = round(ratio, 3)
        else:
            r["flops_vs_efficientnet"] = None

    # Print table
    for r in all_results:
        rel = f"{r['flops_vs_efficientnet']:.2f}x" if r["flops_vs_efficientnet"] else "  N/A"
        flops_str = f"{r['flops_M']:.1f}" if r["flops_M"] >= 0 else "  N/A"
        flag = " ** OUR MODEL **" if r["model_name"] == "agrigatenet" else ""
        print(
            f"  {r['model_name']:<16} {flops_str:>12} "
            f"{r['params_M']:>12.3f} {rel:>22}{flag}"
        )

    print(f"\n  ** = Our novel model (AgriGateNet)")
    if efficientnet_flops:
        agri = next(r for r in all_results if r["model_name"] == "agrigatenet")
        reduction = (1 - agri["flops_M"] / efficientnet_flops) * 100
        print(f"\n  AgriGateNet FLOPs reduction vs EfficientNet-B0: {reduction:.1f}%")

    if save_json:
        out_path = results_dir / "flops_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved -> {out_path}")

    return all_results


if __name__ == "__main__":
    run_flops_analysis(results_dir="./results")
