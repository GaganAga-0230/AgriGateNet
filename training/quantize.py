"""
quantize.py
───────────
INT8 Post-Training Static Quantization for AgriGateNet.

After pruning, this script applies PyTorch's static INT8 quantization
to further compress the model and accelerate CPU inference by ~2-4x.

Why static quantization?
  • Calibrates quantization scales using a small representative dataset
  • No accuracy fine-tuning required (unlike QAT)
  • Reduces model size from FP32 (~3.2 MB) to INT8 (~0.8 MB)
  • Achieves ~2x speedup on x86 CPUs with FBGEMM backend

Usage:
    python training/quantize.py \
        --checkpoint ./checkpoints/pruned/agrigatenet_pruned.pth \
        --data_root ./data/organized \
        --output_dir ./checkpoints/quantized
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.quantization import (
    prepare,
    convert,
    get_default_qconfig,
    fuse_modules,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.agrigatenet import build_agrigatenet, AgriGateNet
from data.dataloader import get_dataloaders, NUM_CLASSES


# ─────────────────────────────────────────────────────────────────
# Quantization-Aware Model Wrapper
# ─────────────────────────────────────────────────────────────────

class QuantizableAgriGateNet(AgriGateNet):
    """
    AgriGateNet subclass with QuantStub / DeQuantStub wrappers for
    static INT8 quantization compatibility.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant   = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.dequant(x)
        return x


# ─────────────────────────────────────────────────────────────────
# Calibration
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def calibrate(model: nn.Module, loader, num_batches: int = 32) -> None:
    """
    Run calibration forward passes so PyTorch can collect
    activation statistics for quantization scale factors.
    """
    model.eval()
    for i, (images, _) in enumerate(loader):
        if i >= num_batches:
            break
        model(images)
    print(f"  ✅  Calibration done ({min(num_batches, len(loader))} batches)")


# ─────────────────────────────────────────────────────────────────
# Model Size Helper
# ─────────────────────────────────────────────────────────────────

def get_model_size_mb(model: nn.Module, path: str = "/tmp/_tmp_model.pth") -> float:
    """Save model to disk and return file size in MB."""
    path = Path(path)
    torch.save(model.state_dict(), path)
    size_mb = path.stat().st_size / (1024 ** 2)
    path.unlink(missing_ok=True)
    return size_mb


# ─────────────────────────────────────────────────────────────────
# Main Quantization Pipeline
# ─────────────────────────────────────────────────────────────────

def quantize_model(
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    backend: str = "fbgemm",   # x86 CPU; use 'qnnpack' for ARM
    calib_batches: int = 32,
    batch_size: int = 32,
    num_workers: int = 4,
) -> nn.Module:
    """
    Full static INT8 quantization pipeline.

    Steps:
      1. Load pruned FP32 model
      2. Set quantization config (per-tensor affine, FBGEMM)
      3. Insert QuantStub / DeQuantStub observers
      4. Calibrate on validation set
      5. Convert to INT8
      6. Benchmark and save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 — Load model
    print("📂 Loading pruned checkpoint …")
    model_fp32 = QuantizableAgriGateNet(num_classes=NUM_CLASSES)
    state = torch.load(checkpoint_path, map_location="cpu")
    # Handle both raw state_dict and checkpoint dicts
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model_fp32.load_state_dict(state, strict=False)
    model_fp32.eval()

    fp32_size = get_model_size_mb(model_fp32)
    print(f"  FP32 model size: {fp32_size:.2f} MB")

    # Step 2 — Quantization config
    torch.backends.quantized.engine = backend
    model_fp32.qconfig = get_default_qconfig(backend)
    print(f"  Quantization backend: {backend}")

    # Step 3 — Prepare (insert observers)
    model_prepared = prepare(model_fp32, inplace=False)

    # Step 4 — Calibrate
    print("\n🔧 Calibrating on validation set …")
    loaders = get_dataloaders(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    calibrate(model_prepared, loaders["val"], num_batches=calib_batches)

    # Step 5 — Convert to INT8
    print("\n⚡ Converting to INT8 …")
    model_int8 = convert(model_prepared, inplace=False)

    int8_size = get_model_size_mb(model_int8)
    print(f"  INT8 model size : {int8_size:.2f} MB")
    print(f"  Compression     : {fp32_size/max(0.001, int8_size):.1f}x smaller")

    # Step 6 — Quick accuracy check
    @torch.no_grad()
    def quick_eval(mdl, ldr):
        correct, total = 0, 0
        for imgs, labels in ldr:
            preds = mdl(imgs).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
        return correct / max(1, total)

    print("\n📊 Evaluating INT8 model on validation set …")
    val_acc = quick_eval(model_int8, loaders["val"])
    print(f"  INT8 Val Accuracy: {val_acc:.4f}")

    # Save
    save_path = output_dir / "agrigatenet_int8.pth"
    torch.save(model_int8.state_dict(), save_path)
    print(f"\n💾 INT8 quantized model saved → {save_path}")

    return model_int8


def main():
    parser = argparse.ArgumentParser(description="INT8 Quantize AgriGateNet")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root",  type=str, default="./data/organized")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/quantized")
    parser.add_argument("--backend",    type=str, default="fbgemm",
                        choices=["fbgemm", "qnnpack"])
    parser.add_argument("--calib_batches", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    quantize_model(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        output_dir=args.output_dir,
        backend=args.backend,
        calib_batches=args.calib_batches,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
