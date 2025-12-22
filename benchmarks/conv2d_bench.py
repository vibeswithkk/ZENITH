# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Conv2D Benchmark - Compares custom CUDA vs cuDNN implementation.

Measures latency improvement from cuDNN integration.

Expected speedup: 5-10x (up to 15x for 3x3 kernels)
"""

import time
import numpy as np
from typing import Callable


def numpy_conv2d(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray = None,
    stride: int = 1,
    padding: int = 0,
) -> np.ndarray:
    """Reference Conv2D implementation in NumPy (direct convolution)."""
    n, c_in, h, w = x.shape
    c_out, _, kh, kw = weight.shape

    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1

    if padding > 0:
        x_padded = np.pad(
            x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant"
        )
    else:
        x_padded = x

    output = np.zeros((n, c_out, h_out, w_out), dtype=x.dtype)

    for i in range(h_out):
        for j in range(w_out):
            h_start = i * stride
            h_end = h_start + kh
            w_start = j * stride
            w_end = w_start + kw

            patch = x_padded[:, :, h_start:h_end, w_start:w_end]
            output[:, :, i, j] = np.tensordot(
                patch, weight, axes=([1, 2, 3], [1, 2, 3])
            )

    if bias is not None:
        output += bias.reshape(1, -1, 1, 1)

    return output


def benchmark_function(
    fn: Callable, args: tuple, warmup_runs: int = 5, benchmark_runs: int = 50
) -> dict:
    """Benchmark a function with warmup and multiple runs."""
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


def run_conv2d_benchmark():
    """Run Conv2D benchmark comparing NumPy (baseline) with CUDA/cuDNN."""
    print("=" * 60)
    print("CONV2D BENCHMARK")
    print("=" * 60)

    configurations = [
        {
            "n": 1,
            "c_in": 3,
            "h": 224,
            "w": 224,
            "c_out": 64,
            "k": 7,
            "stride": 2,
            "pad": 3,
            "name": "ResNet first conv",
        },
        {
            "n": 1,
            "c_in": 64,
            "h": 56,
            "w": 56,
            "c_out": 64,
            "k": 3,
            "stride": 1,
            "pad": 1,
            "name": "ResNet 3x3 conv",
        },
        {
            "n": 8,
            "c_in": 64,
            "h": 56,
            "w": 56,
            "c_out": 128,
            "k": 3,
            "stride": 2,
            "pad": 1,
            "name": "ResNet batch=8",
        },
        {
            "n": 1,
            "c_in": 512,
            "h": 14,
            "w": 14,
            "c_out": 512,
            "k": 3,
            "stride": 1,
            "pad": 1,
            "name": "ResNet deep layer",
        },
    ]

    results = []

    for config in configurations:
        n = config["n"]
        c_in = config["c_in"]
        h = config["h"]
        w = config["w"]
        c_out = config["c_out"]
        k = config["k"]
        stride = config["stride"]
        pad = config["pad"]
        name = config["name"]

        np.random.seed(42)
        x = np.random.randn(n, c_in, h, w).astype(np.float32)
        weight = np.random.randn(c_out, c_in, k, k).astype(np.float32)
        bias = np.random.randn(c_out).astype(np.float32)

        print(f"\nConfiguration: {name}")
        print(f"  Input: ({n}, {c_in}, {h}, {w})")
        print(f"  Weight: ({c_out}, {c_in}, {k}, {k})")
        print(f"  Stride: {stride}, Padding: {pad}")

        baseline_result = benchmark_function(
            numpy_conv2d,
            (x, weight, bias, stride, pad),
            warmup_runs=2,
            benchmark_runs=10,
        )

        print(f"  NumPy (baseline): {baseline_result['mean_ms']:.4f} ms")

        try:
            from zenith._zenith_core import cuda

            if hasattr(cuda, "conv2d"):
                h_out = (h + 2 * pad - k) // stride + 1
                w_out = (w + 2 * pad - k) // stride + 1
                output = np.zeros((n, c_out, h_out, w_out), dtype=np.float32)

                x_contig = np.ascontiguousarray(x)
                weight_contig = np.ascontiguousarray(weight)
                bias_contig = np.ascontiguousarray(bias)

                def cuda_conv2d():
                    cuda.conv2d(
                        x_contig,
                        weight_contig,
                        bias_contig,
                        output,
                        n,
                        c_in,
                        h,
                        w,
                        c_out,
                        k,
                        k,
                        stride,
                        stride,
                        pad,
                        pad,
                    )

                cuda_result = benchmark_function(
                    cuda_conv2d, (), warmup_runs=5, benchmark_runs=20
                )

                speedup = baseline_result["mean_ms"] / cuda_result["mean_ms"]

                print(f"  CUDA (custom): {cuda_result['mean_ms']:.4f} ms")
                print(f"  Speedup: {speedup:.2f}x")

                results.append(
                    {
                        "config": name,
                        "baseline_ms": baseline_result["mean_ms"],
                        "cuda_ms": cuda_result["mean_ms"],
                        "speedup": speedup,
                    }
                )
            else:
                print("  CUDA conv2d not available")
                results.append(
                    {
                        "config": name,
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
                    "baseline_ms": baseline_result["mean_ms"],
                    "cuda_ms": None,
                    "speedup": None,
                }
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<25} {'Baseline':<15} {'CUDA':<15} {'Speedup':<10}")
    print("-" * 60)

    for r in results:
        baseline_str = f"{r['baseline_ms']:.2f} ms"
        if r["cuda_ms"] is not None:
            cuda_str = f"{r['cuda_ms']:.4f} ms"
            speedup_str = f"{r['speedup']:.2f}x"
        else:
            cuda_str = "N/A"
            speedup_str = "N/A"

        print(f"{r['config']:<25} {baseline_str:<15} {cuda_str:<15} {speedup_str:<10}")

    return results


if __name__ == "__main__":
    run_conv2d_benchmark()
