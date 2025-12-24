#!/usr/bin/env python
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Optimized Backend Benchmark - Demonstrates speedup from direct tensor execution.

Compares:
- PyTorch native (baseline)
- Zenith + torch.autocast (optimized)

This benchmark validates that the OptimizedExecutor provides real speedup.
"""

import time
import sys

try:
    import numpy as np
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required")
    sys.exit(1)


class TransformerBlock(nn.Module):
    """Transformer encoder block for benchmarking."""

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
    print("  OPTIMIZED BACKEND BENCHMARK")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        return

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    # Model setup
    batch_size = 32
    seq_len = 128
    hidden = 768

    print(f"\nConfig: batch={batch_size}, seq={seq_len}, hidden={hidden}")

    # Create model
    model = TransformerBlock(hidden, 12).cuda().eval()
    x = torch.randn(batch_size, seq_len, hidden, device="cuda")

    print("\nRunning benchmarks...")

    # 1. PyTorch Native FP32
    def run_native_fp32():
        with torch.no_grad():
            return model(x)

    times_fp32 = measure(run_native_fp32)
    mean_fp32 = np.mean(times_fp32)

    # 2. PyTorch + torch.autocast FP16
    def run_autocast_fp16():
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                return model(x)

    times_fp16 = measure(run_autocast_fp16)
    mean_fp16 = np.mean(times_fp16)

    # 3. Zenith Optimized (if available)
    try:
        from zenith.runtime.cuda_optimized import create_optimized_wrapper

        optimized_fp16 = create_optimized_wrapper(
            model, precision="fp16", device="cuda"
        )

        def run_zenith_fp16():
            return optimized_fp16(x)

        times_zenith = measure(run_zenith_fp16)
        mean_zenith = np.mean(times_zenith)
        has_zenith = True
    except Exception as e:
        print(f"  (Zenith not available: {e})")
        mean_zenith = mean_fp16
        has_zenith = False

    # Results
    speedup_autocast = mean_fp32 / mean_fp16
    speedup_zenith = mean_fp32 / mean_zenith if has_zenith else 0

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"\n{'Method':<25} {'Mean (ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    print(f"{'PyTorch FP32':<25} {mean_fp32:<12.3f} {'1.00x':<10}")
    print(f"{'PyTorch + autocast FP16':<25} {mean_fp16:<12.3f} {speedup_autocast:.2f}x")
    if has_zenith:
        print(
            f"{'Zenith Optimized FP16':<25} {mean_zenith:<12.3f} {speedup_zenith:.2f}x"
        )

    print(f"\n{'=' * 60}")
    if speedup_autocast > 1.2:
        print(f"  [PASS] Tensor Cores active: {speedup_autocast:.2f}x speedup")
    else:
        print(f"  [INFO] Speedup: {speedup_autocast:.2f}x")

    if has_zenith and speedup_zenith > 1.0:
        print(f"  [PASS] Zenith optimization verified: {speedup_zenith:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
