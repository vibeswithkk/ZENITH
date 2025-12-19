#!/usr/bin/env python3
"""
INT8 Quantization Benchmark
Demonstrates speedup and accuracy of INT8 quantization vs FP32.
"""

import sys

sys.path.insert(0, "build/python")
sys.path.insert(0, ".")

import numpy as np
import time

print("=" * 70)
print("ZENITH INT8 QUANTIZATION BENCHMARK")
print("FP32 vs INT8 Inference Comparison")
print("=" * 70)

from zenith.optimization.quantization import (
    Quantizer,
    QuantizationMode,
    CalibrationMethod,
    measure_quantization_error,
)

# Config
BATCH_SIZE = 32
INPUT_SIZE = 768
HIDDEN_SIZE = 3072
OUTPUT_SIZE = 768
NUM_LAYERS = 6
CALIBRATION_BATCHES = 50
BENCHMARK_RUNS = 100

print(f"\n[Config]")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Input Size: {INPUT_SIZE}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  Layers: {NUM_LAYERS}")
print(f"  Calibration Batches: {CALIBRATION_BATCHES}")
print(f"  Benchmark Runs: {BENCHMARK_RUNS}")

# ========================================
# Create Test Model (Simple MLP)
# ========================================
print("\n[1/5] Creating test model weights...")

np.random.seed(42)
weights = {}
for i in range(NUM_LAYERS):
    if i == 0:
        in_size = INPUT_SIZE
    else:
        in_size = HIDDEN_SIZE if i % 2 == 1 else OUTPUT_SIZE

    out_size = HIDDEN_SIZE if i % 2 == 0 else OUTPUT_SIZE

    weights[f"layer{i}.weight"] = (
        np.random.randn(out_size, in_size).astype(np.float32) * 0.02
    )
    weights[f"layer{i}.bias"] = np.zeros(out_size, dtype=np.float32)

total_params = sum(w.size for w in weights.values())
print(f"  Model: {NUM_LAYERS} layers, {total_params:,} parameters")

# ========================================
# Calibration
# ========================================
print("\n[2/5] Calibrating for INT8 quantization...")

quantizer = Quantizer(
    mode=QuantizationMode.STATIC,
    calibration_method=CalibrationMethod.PERCENTILE,
    symmetric=True,
)

# Collect calibration statistics
for batch_idx in range(CALIBRATION_BATCHES):
    x = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
    quantizer.collect_stats(x, "input")

    # Simulate forward pass for activation stats
    for i in range(NUM_LAYERS):
        w = weights[f"layer{i}.weight"]
        b = weights[f"layer{i}.bias"]
        x = np.matmul(x, w.T) + b
        if i < NUM_LAYERS - 1:
            x = np.maximum(x, 0)  # ReLU
        quantizer.collect_stats(x, f"layer{i}_output")

print(f"  Calibrated with {CALIBRATION_BATCHES} batches")

# ========================================
# Quantize Model
# ========================================
print("\n[3/5] Quantizing model to INT8...")

quantized_model = quantizer.quantize_weights(weights)

# Measure quantization error
total_mse = 0
total_snr = 0
for name in weights:
    error = measure_quantization_error(weights[name], quantized_model, name)
    total_mse += error["mse"]
    total_snr += error["snr_db"]

avg_mse = total_mse / len(weights)
avg_snr = total_snr / len(weights)

print(f"  Average MSE: {avg_mse:.6e}")
print(f"  Average SNR: {avg_snr:.1f} dB")

# ========================================
# Benchmark FP32 vs INT8
# ========================================
print("\n[4/5] Benchmarking FP32 vs INT8...")


def forward_fp32(x, weights):
    """FP32 forward pass."""
    for i in range(NUM_LAYERS):
        w = weights[f"layer{i}.weight"]
        b = weights[f"layer{i}.bias"]
        x = np.matmul(x, w.T) + b
        if i < NUM_LAYERS - 1:
            x = np.maximum(x, 0)
    return x


def forward_int8(x, quantized_model, quantizer):
    """INT8 forward pass (simulated with dequantization)."""
    for i in range(NUM_LAYERS):
        # Get dequantized weight (INT8 -> FP32)
        w = quantized_model.get_weight_dequantized(f"layer{i}.weight")
        b = quantized_model.get_weight_dequantized(f"layer{i}.bias")
        if b is None:
            b = np.zeros(w.shape[0], dtype=np.float32)
        x = np.matmul(x, w.T) + b
        if i < NUM_LAYERS - 1:
            x = np.maximum(x, 0)
    return x


# Warmup
x_test = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
for _ in range(10):
    _ = forward_fp32(x_test, weights)
    _ = forward_int8(x_test, quantized_model, quantizer)

# Benchmark FP32
fp32_times = []
for _ in range(BENCHMARK_RUNS):
    x = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
    t0 = time.perf_counter()
    out_fp32 = forward_fp32(x, weights)
    fp32_times.append((time.perf_counter() - t0) * 1000)

# Benchmark INT8
int8_times = []
for _ in range(BENCHMARK_RUNS):
    x = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
    t0 = time.perf_counter()
    out_int8 = forward_int8(x, quantized_model, quantizer)
    int8_times.append((time.perf_counter() - t0) * 1000)

# ========================================
# Accuracy Check
# ========================================
print("\n[5/5] Checking accuracy...")

x_check = np.random.randn(BATCH_SIZE, INPUT_SIZE).astype(np.float32)
out_fp32 = forward_fp32(x_check, weights)
out_int8 = forward_int8(x_check, quantized_model, quantizer)

max_diff = np.max(np.abs(out_fp32 - out_int8))
mean_diff = np.mean(np.abs(out_fp32 - out_int8))
relative_error = np.mean(np.abs(out_fp32 - out_int8) / (np.abs(out_fp32) + 1e-8))

print(f"  Max diff: {max_diff:.4e}")
print(f"  Mean diff: {mean_diff:.4e}")
print(f"  Relative error: {relative_error * 100:.2f}%")

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("QUANTIZATION BENCHMARK RESULTS")
print("=" * 70)

fp32_mean = np.mean(fp32_times)
fp32_std = np.std(fp32_times)
int8_mean = np.mean(int8_times)
int8_std = np.std(int8_times)

print(f"\n{'Mode':<20} {'Time (ms)':<20} {'Memory (est.)':<15}")
print("-" * 55)
print(
    f"{'FP32':<20} {fp32_mean:.3f} ± {fp32_std:.3f}       {total_params * 4 / 1e6:.1f} MB"
)
print(
    f"{'INT8':<20} {int8_mean:.3f} ± {int8_std:.3f}       {total_params * 1 / 1e6:.1f} MB"
)

# Note: In CPU simulation, INT8 may not be faster due to dequantization overhead
# Real speedup comes from hardware INT8 support (Tensor Cores, VNNI)
print("\n" + "-" * 55)
print(f"Memory Reduction: 4x (FP32: 4 bytes -> INT8: 1 byte)")
print(f"Accuracy Loss: {relative_error * 100:.2f}% relative error")
print(f"Output SNR: {avg_snr:.1f} dB")

if relative_error < 0.01:
    accuracy_status = "EXCELLENT"
elif relative_error < 0.05:
    accuracy_status = "GOOD"
else:
    accuracy_status = "ACCEPTABLE"

print(f"\nAccuracy Status: {accuracy_status}")
print("=" * 70)

print("\n[Note] For real INT8 speedup, use GPU with Tensor Cores or CPU with VNNI.")
print("       This benchmark demonstrates quantization accuracy and workflow.")
