# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Phase 7 INT8 Quantization Benchmark

Tests:
1. Quantization/Dequantization accuracy
2. INT8 vs FP32 throughput comparison
3. Memory savings

Target: T4 GPU - 130 TOPS INT8
"""

import numpy as np
import time


def quantize_to_int8(tensor: np.ndarray, symmetric: bool = True):
    """Quantize FP32 tensor to INT8 with scale."""
    if symmetric:
        abs_max = np.max(np.abs(tensor))
        scale = abs_max / 127.0 if abs_max > 0 else 1.0
        quantized = np.clip(np.round(tensor / scale), -128, 127).astype(np.int8)
        return quantized, scale, 0
    else:
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
        zero_point = int(-min_val / scale)
        quantized = np.clip(np.round(tensor / scale) + zero_point, -128, 127).astype(
            np.int8
        )
        return quantized, scale, zero_point


def dequantize_from_int8(tensor: np.ndarray, scale: float, zero_point: int = 0):
    """Dequantize INT8 tensor back to FP32."""
    return (tensor.astype(np.float32) - zero_point) * scale


def int8_matmul_reference(a_int8, b_int8, a_scale, b_scale):
    """Reference INT8 matmul with INT32 accumulation."""
    c_int32 = a_int8.astype(np.int32) @ b_int8.astype(np.int32)
    return c_int32.astype(np.float32) * a_scale * b_scale


def measure_quantization_error(original, dequantized):
    """Measure quantization error metrics."""
    mse = np.mean((original - dequantized) ** 2)
    max_error = np.max(np.abs(original - dequantized))
    signal_power = np.mean(original**2)
    snr = 10 * np.log10(signal_power / (mse + 1e-10))
    return {"mse": mse, "max_error": max_error, "snr_db": snr}


def benchmark_function(fn, args, warmup=5, runs=50):
    """Benchmark with warmup and statistics."""
    for _ in range(warmup):
        fn(*args)

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn(*args)
        times.append((time.perf_counter() - start) * 1000)

    return {"mean_ms": np.mean(times), "std_ms": np.std(times), "result": result}


def run_int8_benchmark():
    """Run Phase 7 INT8 quantization benchmark."""
    print("=" * 60)
    print("PHASE 7: INT8 QUANTIZATION BENCHMARK")
    print("=" * 60)

    # Test configurations
    configurations = [
        {"m": 1024, "n": 1024, "k": 1024, "name": "1K x 1K"},
        {"m": 2048, "n": 2048, "k": 2048, "name": "2K x 2K"},
        {"m": 4096, "n": 4096, "k": 4096, "name": "4K x 4K"},
    ]

    results = []

    for config in configurations:
        m, n, k = config["m"], config["n"], config["k"]
        name = config["name"]

        np.random.seed(42)
        a_fp32 = np.random.randn(m, k).astype(np.float32)
        b_fp32 = np.random.randn(k, n).astype(np.float32)

        print(f"\nConfiguration: {name}")

        # Quantize
        a_int8, a_scale, a_zp = quantize_to_int8(a_fp32)
        b_int8, b_scale, b_zp = quantize_to_int8(b_fp32)

        # FP32 baseline
        fp32_result = benchmark_function(np.matmul, (a_fp32, b_fp32), warmup=2, runs=10)

        # INT8 matmul (NumPy simulation)
        int8_result = benchmark_function(
            int8_matmul_reference, (a_int8, b_int8, a_scale, b_scale), warmup=2, runs=10
        )

        # Accuracy comparison
        c_fp32 = a_fp32 @ b_fp32
        c_int8 = int8_result["result"]
        error_metrics = measure_quantization_error(c_fp32, c_int8)

        # Memory comparison
        mem_fp32 = (m * k + k * n) * 4  # FP32 = 4 bytes
        mem_int8 = (m * k + k * n) * 1  # INT8 = 1 byte
        mem_savings = (1 - mem_int8 / mem_fp32) * 100

        speedup = fp32_result["mean_ms"] / int8_result["mean_ms"]

        print(f"  FP32: {fp32_result['mean_ms']:.2f} ms")
        print(f"  INT8: {int8_result['mean_ms']:.2f} ms")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  SNR: {error_metrics['snr_db']:.1f} dB")
        print(f"  Memory savings: {mem_savings:.0f}%")

        results.append(
            {
                "config": name,
                "fp32_ms": fp32_result["mean_ms"],
                "int8_ms": int8_result["mean_ms"],
                "speedup": speedup,
                "snr_db": error_metrics["snr_db"],
                "mem_savings": mem_savings,
            }
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Config':<12} {'FP32':<10} {'INT8':<10} {'Speedup':<10} {'SNR':<10}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['config']:<12} {r['fp32_ms']:.2f} ms   {r['int8_ms']:.2f} ms   "
            f"{r['speedup']:.2f}x      {r['snr_db']:.1f} dB"
        )

    print("\nTarget on T4 GPU:")
    print("  INT8 Tensor Core: 130 TOPS (2x FP16, 16x FP32)")
    print("  Memory savings: 75%")
    print("  Accuracy: SNR > 20 dB")

    return results


if __name__ == "__main__":
    run_int8_benchmark()
