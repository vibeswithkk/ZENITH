#!/usr/bin/env python
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Speedup Benchmark - MLPerf-style benchmark proving Zenith speedup.

Follows CetakBiru.md Section 2.5 (MLPerf Standards):
- Latency percentiles (P50, P90, P99)
- Throughput (samples/sec)
- Warm-up runs (exclude compilation)
- Multiple model sizes and batch sizes

Usage:
    python speedup_benchmark.py
"""

import time
import sys
from dataclasses import dataclass
from typing import Optional
import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch required for benchmark")
    sys.exit(1)

try:
    import zenith

    HAS_ZENITH = True
except ImportError:
    HAS_ZENITH = False
    print("ERROR: Zenith required. Install: pip install pyzenith")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    name: str
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    throughput: float
    batch_size: int


def measure_latency(fn, num_warmup: int = 10, num_runs: int = 100) -> list:
    """Measure execution latency with warm-up."""
    # Warm-up (exclude from measurement per MLPerf)
    for _ in range(num_warmup):
        _ = fn()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    latencies = []
    for _ in range(num_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        _ = fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return latencies


def compute_stats(latencies: list, name: str, batch_size: int) -> BenchmarkResult:
    """Compute benchmark statistics."""
    return BenchmarkResult(
        name=name,
        latency_mean_ms=np.mean(latencies),
        latency_std_ms=np.std(latencies),
        latency_p50_ms=np.percentile(latencies, 50),
        latency_p90_ms=np.percentile(latencies, 90),
        latency_p99_ms=np.percentile(latencies, 99),
        throughput=batch_size * 1000 / np.mean(latencies),
        batch_size=batch_size,
    )


class LinearModel(nn.Module):
    """Simple linear model for baseline testing."""

    def __init__(self, in_features: int = 768, out_features: int = 256):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class MLPModel(nn.Module):
    """Multi-layer perceptron."""

    def __init__(self, hidden: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Single transformer encoder block."""

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


def benchmark_model(
    model: nn.Module,
    input_shape: tuple,
    model_name: str,
    batch_sizes: list = None,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> dict:
    """Benchmark a model with PyTorch native vs Zenith."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    results = {"pytorch": [], "zenith": [], "speedup": []}

    for batch_size in batch_sizes:
        # Create input
        shape = (batch_size,) + input_shape
        x = torch.randn(*shape, device=device)

        # PyTorch Native Baseline
        def pytorch_fn():
            with torch.no_grad():
                return model(x)

        pytorch_latencies = measure_latency(pytorch_fn, num_warmup, num_runs)
        pytorch_result = compute_stats(
            pytorch_latencies, f"PyTorch-{model_name}", batch_size
        )
        results["pytorch"].append(pytorch_result)

        # Zenith Optimized
        optimized = zenith.compile(model, target=device, sample_input=x)

        def zenith_fn():
            with torch.no_grad():
                return optimized(x)

        zenith_latencies = measure_latency(zenith_fn, num_warmup, num_runs)
        zenith_result = compute_stats(
            zenith_latencies, f"Zenith-{model_name}", batch_size
        )
        results["zenith"].append(zenith_result)

        # Speedup
        speedup = pytorch_result.latency_mean_ms / zenith_result.latency_mean_ms
        results["speedup"].append(speedup)

    return results


def print_results(model_name: str, results: dict):
    """Print benchmark results in table format."""
    print(f"\n{'=' * 70}")
    print(f"  {model_name} Benchmark Results")
    print(f"{'=' * 70}")

    print(
        f"\n{'Batch':>8} {'Framework':>12} {'Mean':>10} {'Std':>8} "
        f"{'P50':>8} {'P90':>8} {'P99':>8} {'Thrpt':>10}"
    )
    print(f"{'-' * 70}")

    for i, (pt, zn, speedup) in enumerate(
        zip(results["pytorch"], results["zenith"], results["speedup"])
    ):
        print(
            f"{pt.batch_size:>8} {'PyTorch':>12} {pt.latency_mean_ms:>10.3f} "
            f"{pt.latency_std_ms:>8.3f} {pt.latency_p50_ms:>8.3f} "
            f"{pt.latency_p90_ms:>8.3f} {pt.latency_p99_ms:>8.3f} "
            f"{pt.throughput:>10.1f}"
        )
        print(
            f"{zn.batch_size:>8} {'Zenith':>12} {zn.latency_mean_ms:>10.3f} "
            f"{zn.latency_std_ms:>8.3f} {zn.latency_p50_ms:>8.3f} "
            f"{zn.latency_p90_ms:>8.3f} {zn.latency_p99_ms:>8.3f} "
            f"{zn.throughput:>10.1f}"
        )
        print(f"{'':>8} {'SPEEDUP':>12} {speedup:>10.2f}x")
        print()


def main():
    """Run comprehensive speedup benchmark."""
    print("=" * 70)
    print("  ZENITH SPEEDUP BENCHMARK (MLPerf-style)")
    print("=" * 70)
    print(f"\nZenith Version: {zenith.__version__}")
    print(f"PyTorch Version: {torch.__version__}")
    print(
        f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"\nBenchmark Config:")
    print(f"  Warm-up runs: 10")
    print(f"  Timed runs: 100")
    print(f"  Batch sizes: [1, 8, 32]")

    all_speedups = []

    # Benchmark 1: Linear Model
    print("\n[1/3] Benchmarking Linear Model...")
    linear_model = LinearModel(768, 256)
    linear_results = benchmark_model(linear_model, (768,), "Linear", [1, 8, 32])
    print_results("Linear (768->256)", linear_results)
    all_speedups.extend(linear_results["speedup"])

    # Benchmark 2: MLP Model
    print("\n[2/3] Benchmarking MLP Model...")
    mlp_model = MLPModel(768)
    mlp_results = benchmark_model(mlp_model, (768,), "MLP", [1, 8, 32])
    print_results("MLP (768->3072->768)", mlp_results)
    all_speedups.extend(mlp_results["speedup"])

    # Benchmark 3: Transformer Block
    print("\n[3/3] Benchmarking Transformer Block...")
    transformer_model = TransformerBlock(768, 12)
    transformer_results = benchmark_model(
        transformer_model, (128, 768), "Transformer", [1, 8, 32]
    )
    print_results("Transformer Block (768, 12 heads)", transformer_results)
    all_speedups.extend(transformer_results["speedup"])

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"\n  Average Speedup: {np.mean(all_speedups):.2f}x")
    print(f"  Min Speedup:     {np.min(all_speedups):.2f}x")
    print(f"  Max Speedup:     {np.max(all_speedups):.2f}x")

    if np.mean(all_speedups) >= 1.0:
        print("\n  [PASS] Zenith shows competitive or better performance")
    else:
        print("\n  [INFO] Performance may vary by model complexity")

    print("=" * 70)

    return all_speedups


if __name__ == "__main__":
    main()
