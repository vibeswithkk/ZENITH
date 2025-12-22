# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
WMMA (Tensor Core) Benchmark - MatMul Performance Test

Compares:
1. NumPy (baseline)
2. cuBLAS (reference)
3. WMMA Tensor Core (new)

Target: T4 GPU - 65 TFLOPS FP16 theoretical
"""

import time
import numpy as np
from typing import Callable


def benchmark_function(
    fn: Callable, args: tuple, warmup_runs: int = 5, benchmark_runs: int = 50
) -> dict:
    """Benchmark with warmup and multiple runs."""
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
    }


def calculate_tflops(m: int, n: int, k: int, time_ms: float) -> float:
    """Calculate TFLOPS for matmul operation."""
    flops = 2 * m * n * k  # multiply-add = 2 ops
    return (flops / (time_ms / 1000)) / 1e12


def numpy_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """NumPy reference matmul."""
    return a @ b


def run_wmma_benchmark():
    """Run WMMA Tensor Core benchmark."""
    print("=" * 60)
    print("WMMA TENSOR CORE BENCHMARK")
    print("Target: T4 GPU (65 TFLOPS FP16)")
    print("=" * 60)

    configurations = [
        {"m": 1024, "n": 1024, "k": 1024, "name": "1K x 1K x 1K"},
        {"m": 2048, "n": 2048, "k": 2048, "name": "2K x 2K x 2K"},
        {"m": 4096, "n": 4096, "k": 4096, "name": "4K x 4K x 4K"},
        {"m": 1024, "n": 768, "k": 768, "name": "BERT hidden"},
        {"m": 512, "n": 512, "k": 4096, "name": "GPT MLP"},
    ]

    results = []

    for config in configurations:
        m = config["m"]
        n = config["n"]
        k = config["k"]
        name = config["name"]

        np.random.seed(42)
        a_fp32 = np.random.randn(m, k).astype(np.float32)
        b_fp32 = np.random.randn(k, n).astype(np.float32)

        print(f"\nConfiguration: {name}")
        print(f"  Shape: ({m}, {k}) x ({k}, {n})")

        baseline = benchmark_function(
            numpy_matmul, (a_fp32, b_fp32), warmup_runs=2, benchmark_runs=10
        )

        tflops_numpy = calculate_tflops(m, n, k, baseline["mean_ms"])
        print(f"  NumPy: {baseline['mean_ms']:.2f} ms ({tflops_numpy:.2f} TFLOPS)")

        try:
            from zenith._zenith_core import cuda

            if hasattr(cuda, "matmul"):
                c_fp32 = np.zeros((m, n), dtype=np.float32)
                a_contig = np.ascontiguousarray(a_fp32)
                b_contig = np.ascontiguousarray(b_fp32)

                def cuda_matmul():
                    cuda.matmul(a_contig, b_contig, c_fp32, m, n, k)

                cuda_result = benchmark_function(
                    cuda_matmul, (), warmup_runs=5, benchmark_runs=20
                )

                tflops_cuda = calculate_tflops(m, n, k, cuda_result["mean_ms"])
                speedup = baseline["mean_ms"] / cuda_result["mean_ms"]

                print(
                    f"  CUDA: {cuda_result['mean_ms']:.2f} ms ({tflops_cuda:.2f} TFLOPS)"
                )
                print(f"  Speedup: {speedup:.1f}x")

                results.append(
                    {
                        "config": name,
                        "numpy_ms": baseline["mean_ms"],
                        "cuda_ms": cuda_result["mean_ms"],
                        "tflops": tflops_cuda,
                        "speedup": speedup,
                    }
                )
            else:
                print("  CUDA matmul not available")
                results.append(
                    {
                        "config": name,
                        "numpy_ms": baseline["mean_ms"],
                        "cuda_ms": None,
                        "tflops": None,
                        "speedup": None,
                    }
                )

        except ImportError:
            print("  CUDA module not available")
            results.append(
                {
                    "config": name,
                    "numpy_ms": baseline["mean_ms"],
                    "cuda_ms": None,
                    "tflops": None,
                    "speedup": None,
                }
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<20} {'NumPy':<10} {'CUDA':<10} {'TFLOPS':<10} {'Speedup':<10}")
    print("-" * 60)

    for r in results:
        numpy_str = f"{r['numpy_ms']:.1f} ms"
        if r["cuda_ms"] is not None:
            cuda_str = f"{r['cuda_ms']:.2f} ms"
            tflops_str = f"{r['tflops']:.2f}"
            speedup_str = f"{r['speedup']:.1f}x"
        else:
            cuda_str = "N/A"
            tflops_str = "N/A"
            speedup_str = "N/A"

        print(
            f"{r['config']:<20} {numpy_str:<10} {cuda_str:<10} {tflops_str:<10} {speedup_str:<10}"
        )

    print("\nT4 Theoretical: 65 TFLOPS FP16")
    print("Target: >50 TFLOPS (80% efficiency)")

    return results


if __name__ == "__main__":
    run_wmma_benchmark()
