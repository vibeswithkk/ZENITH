# FP16 Tensor Core Benchmark
# Compare FP32 vs FP16 attention performance

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("FP16 TENSOR CORE BENCHMARK")
print("=" * 70)

from zenith._zenith_core import cuda

# Test parameters
BATCH = 1
HEADS = 12
SEQ = 128  # Larger for better Tensor Core utilization
DIM = 64
NUM_RUNS = 100

print(f"\nConfig: batch={BATCH}, heads={HEADS}, seq={SEQ}, dim={DIM}")

# Create random tensors
np.random.seed(42)
Q_np = np.random.randn(BATCH, HEADS, SEQ, DIM).astype(np.float32)
K_np = np.random.randn(BATCH, HEADS, SEQ, DIM).astype(np.float32)
V_np = np.random.randn(BATCH, HEADS, SEQ, DIM).astype(np.float32)

Q_gpu = cuda.to_gpu(np.ascontiguousarray(Q_np))
K_gpu = cuda.to_gpu(np.ascontiguousarray(K_np))
V_gpu = cuda.to_gpu(np.ascontiguousarray(V_np))

# Warmup
for _ in range(5):
    _ = cuda.cublas_attention_gpu(Q_gpu, K_gpu, V_gpu)

print("\n[1/3] FP32 cuBLAS Attention...")
fp32_times = []
for _ in range(NUM_RUNS):
    t0 = time.perf_counter()
    out_fp32 = cuda.cublas_attention_gpu(Q_gpu, K_gpu, V_gpu)
    fp32_times.append((time.perf_counter() - t0) * 1000)

print(f"  FP32: {np.mean(fp32_times):.3f} ± {np.std(fp32_times):.3f} ms")

# Test FP16
print("\n[2/3] FP16 Tensor Core Attention...")

try:
    # Warmup
    for _ in range(5):
        _ = cuda.attention_fp16_gpu(Q_gpu, K_gpu, V_gpu)

    fp16_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        out_fp16 = cuda.attention_fp16_gpu(Q_gpu, K_gpu, V_gpu)
        fp16_times.append((time.perf_counter() - t0) * 1000)

    print(f"  FP16: {np.mean(fp16_times):.3f} ± {np.std(fp16_times):.3f} ms")

    # Accuracy check
    print("\n[3/3] Accuracy check (FP16 vs FP32)...")
    out_fp32_np = out_fp32.to_numpy()
    out_fp16_np = out_fp16.to_numpy()

    max_diff = np.max(np.abs(out_fp32_np - out_fp16_np))
    mean_diff = np.mean(np.abs(out_fp32_np - out_fp16_np))

    print(f"  Max diff: {max_diff:.2e}")
    print(f"  Mean diff: {mean_diff:.2e}")
    print(f"  Accuracy: {'ACCEPTABLE' if max_diff < 1e-2 else 'NEEDS CHECK'}")

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    speedup = np.mean(fp32_times) / np.mean(fp16_times)
    print(f"\n  FP32: {np.mean(fp32_times):.3f} ms")
    print(f"  FP16: {np.mean(fp16_times):.3f} ms")
    print(f"  Speedup: {speedup:.2f}x {'FASTER' if speedup > 1 else 'slower'}")

except Exception as e:
    print(f"  FP16 error: {e}")
    print("  Check if GPU supports FP16 Tensor Cores")

print("\n" + "=" * 70)
