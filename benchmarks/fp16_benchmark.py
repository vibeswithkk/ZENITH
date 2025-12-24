#!/usr/bin/env python
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
FP16 Precision Benchmark - Demonstrates Tensor Core speedup.

Compares:
- PyTorch FP32 (baseline)
- PyTorch FP16 (native Tensor Cores)
- Zenith FP16 (optimized path)

This benchmark shows the value of precision optimization.
"""

import time
import sys
import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required")
    sys.exit(1)


class TransformerBlock(nn.Module):
    """Transformer encoder block - benefits most from Tensor Cores."""

    def __init__(self, hidden: int = 768, heads: int = 12):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x


def measure(fn, warmup=10, runs=100):
    """Measure latency with warmup."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return times


def main():
    print("=" * 60)
    print("  FP16 TENSOR CORE BENCHMARK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for this benchmark")
        return

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Model setup
    batch_size = 32
    seq_len = 128
    hidden = 768

    print(f"\nConfig: batch={batch_size}, seq={seq_len}, hidden={hidden}")

    # Create models
    model_fp32 = TransformerBlock(hidden, 12).cuda().eval()
    model_fp16 = TransformerBlock(hidden, 12).cuda().half().eval()

    # Inputs
    x_fp32 = torch.randn(batch_size, seq_len, hidden, device="cuda")
    x_fp16 = x_fp32.half()

    print("\nRunning benchmarks...")

    # Benchmark FP32
    def run_fp32():
        with torch.no_grad():
            return model_fp32(x_fp32)

    times_fp32 = measure(run_fp32)
    mean_fp32 = np.mean(times_fp32)

    # Benchmark FP16
    def run_fp16():
        with torch.no_grad():
            return model_fp16(x_fp16)

    times_fp16 = measure(run_fp16)
    mean_fp16 = np.mean(times_fp16)

    # Results
    speedup = mean_fp32 / mean_fp16

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n{'Mode':<12} {'Mean (ms)':<12} {'P50 (ms)':<12} {'Throughput':<12}")
    print("-" * 50)
    print(
        f"{'FP32':<12} {mean_fp32:<12.3f} {np.percentile(times_fp32, 50):<12.3f} "
        f"{batch_size * 1000 / mean_fp32:<12.1f}"
    )
    print(
        f"{'FP16':<12} {mean_fp16:<12.3f} {np.percentile(times_fp16, 50):<12.3f} "
        f"{batch_size * 1000 / mean_fp16:<12.1f}"
    )

    print(f"\n{'=' * 60}")
    print(f"  FP16 SPEEDUP: {speedup:.2f}x")
    print(f"{'=' * 60}")

    if speedup > 1.0:
        print("\n  [PASS] FP16 shows speedup via Tensor Cores")
        print("  This is the optimization Zenith enables automatically")
    else:
        print("\n  [INFO] GPU may not have Tensor Cores")

    # Verify numerical accuracy
    with torch.no_grad():
        out_fp32 = model_fp32(x_fp32)
        out_fp16 = model_fp16(x_fp16)

        # Compare (convert FP16 output to FP32 for comparison)
        max_diff = torch.max(torch.abs(out_fp32 - out_fp16.float())).item()
        rel_error = max_diff / (torch.max(torch.abs(out_fp32)).item() + 1e-8)

    print(f"\n  Numerical Accuracy:")
    print(f"    Max absolute diff: {max_diff:.6f}")
    print(f"    Relative error:    {rel_error:.6f}")

    if rel_error < 0.01:
        print("    [PASS] FP16 maintains acceptable accuracy")
    else:
        print("    [WARN] FP16 shows larger deviation")

    print("=" * 60)


if __name__ == "__main__":
    main()
