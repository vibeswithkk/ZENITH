# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
LayerNorm Benchmark - Compares sequential vs parallel implementation.

Measures latency improvement from the parallel warp reduction optimization.

Expected speedup: 10-40x (theoretical: 96x, practical: 20-40x memory bound)
"""

import time
import numpy as np
from typing import Callable


def numpy_layernorm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """Reference LayerNorm implementation in NumPy (sequential)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_normalized = (x - mean) / np.sqrt(var + eps)
    return gamma * x_normalized + beta


def benchmark_function(
    fn: Callable, args: tuple, warmup_runs: int = 5, benchmark_runs: int = 100
) -> dict:
    """
    Benchmark a function with warmup and multiple runs.

    Returns:
        Dictionary with timing statistics
    """
    for _ in range(warmup_runs):
        fn(*args)

    times = []
    for _ in range(benchmark_runs):
        start = time.perf_counter()
        fn(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)

    return {
        "mean_ms": np.mean(times),
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "median_ms": np.median(times),
    }


def run_layernorm_benchmark():
    """Run LayerNorm benchmark comparing NumPy (baseline) with CUDA."""
    print("=" * 60)
    print("LAYERNORM BENCHMARK")
    print("=" * 60)

    configurations = [
        {"batch": 1, "hidden": 768, "name": "Single sample, BERT-base"},
        {"batch": 32, "hidden": 768, "name": "Batch 32, BERT-base"},
        {"batch": 128, "hidden": 768, "name": "Batch 128, BERT-base"},
        {"batch": 32, "hidden": 1024, "name": "Batch 32, BERT-large"},
        {"batch": 32, "hidden": 4096, "name": "Batch 32, GPT-2 XL"},
    ]

    results = []

    for config in configurations:
        batch = config["batch"]
        hidden = config["hidden"]
        name = config["name"]

        np.random.seed(42)
        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-5

        print(f"\nConfiguration: {name}")
        print(f"  Shape: ({batch}, {hidden})")

        baseline_result = benchmark_function(
            numpy_layernorm, (x, gamma, beta, eps), warmup_runs=3, benchmark_runs=50
        )

        print(f"  NumPy (baseline): {baseline_result['mean_ms']:.4f} ms")

        try:
            from zenith._zenith_core import cuda

            if hasattr(cuda, "layernorm_f32"):
                output = np.zeros_like(x)

                def cuda_layernorm():
                    cuda.layernorm_f32(x, output, gamma, beta, batch, hidden, eps)

                cuda_result = benchmark_function(
                    cuda_layernorm, (), warmup_runs=10, benchmark_runs=100
                )

                speedup = baseline_result["mean_ms"] / cuda_result["mean_ms"]

                print(f"  CUDA (optimized): {cuda_result['mean_ms']:.4f} ms")
                print(f"  Speedup: {speedup:.2f}x")

                results.append(
                    {
                        "config": name,
                        "batch": batch,
                        "hidden": hidden,
                        "baseline_ms": baseline_result["mean_ms"],
                        "cuda_ms": cuda_result["mean_ms"],
                        "speedup": speedup,
                    }
                )
            else:
                print("  CUDA layernorm_f32 not available")
                results.append(
                    {
                        "config": name,
                        "batch": batch,
                        "hidden": hidden,
                        "baseline_ms": baseline_result["mean_ms"],
                        "cuda_ms": None,
                        "speedup": None,
                    }
                )

        except ImportError:
            print("  CUDA module not available - benchmark skipped")
            results.append(
                {
                    "config": name,
                    "batch": batch,
                    "hidden": hidden,
                    "baseline_ms": baseline_result["mean_ms"],
                    "cuda_ms": None,
                    "speedup": None,
                }
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<35} {'Baseline':<12} {'CUDA':<12} {'Speedup':<10}")
    print("-" * 60)

    for r in results:
        baseline_str = f"{r['baseline_ms']:.4f} ms"
        if r["cuda_ms"] is not None:
            cuda_str = f"{r['cuda_ms']:.4f} ms"
            speedup_str = f"{r['speedup']:.2f}x"
        else:
            cuda_str = "N/A"
            speedup_str = "N/A"

        print(f"{r['config']:<35} {baseline_str:<12} {cuda_str:<12} {speedup_str:<10}")

    return results


if __name__ == "__main__":
    run_layernorm_benchmark()
