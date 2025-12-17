# Zenith GPU Memory Optimization Test
# ====================================
# Tests the new GpuTensor API for zero-copy GPU operations

import numpy as np
import time

print("=" * 50)
print("ZENITH GPU MEMORY OPTIMIZATION TEST")
print("=" * 50)

try:
    from zenith._zenith_core import cuda

    print(f"\nCUDA available: {cuda.is_available()}")
except ImportError as e:
    print(f"Zenith CUDA not available: {e}")
    exit(1)

# Test 1: GpuTensor API
print("\n" + "=" * 50)
print("TEST 1: GpuTensor API")
print("=" * 50)

A_np = np.random.randn(1024, 1024).astype(np.float32)
B_np = np.random.randn(1024, 1024).astype(np.float32)

# Create GPU tensors (H2D copy happens here, once)
A_gpu = cuda.to_gpu(A_np)
B_gpu = cuda.to_gpu(B_np)

print(f"A_gpu: {A_gpu}")
print(f"B_gpu: {B_gpu}")

# Test 2: Zero-copy matmul benchmark
print("\n" + "=" * 50)
print("TEST 2: ZERO-COPY MATMUL BENCHMARK")
print("=" * 50)


def benchmark_old(A_np, B_np, iterations=50):
    """Old API - copies every time"""
    times = []
    for _ in range(10):
        cuda.matmul(A_np, B_np)  # Warmup

    for _ in range(iterations):
        t0 = time.perf_counter()
        C = cuda.matmul(A_np, B_np)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times)


def benchmark_new(A_gpu, B_gpu, iterations=50):
    """New API - zero copy compute"""
    times = []
    for _ in range(10):
        cuda.matmul_gpu(A_gpu, B_gpu)  # Warmup

    for _ in range(iterations):
        t0 = time.perf_counter()
        C_gpu = cuda.matmul_gpu(A_gpu, B_gpu)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times)


t_old = benchmark_old(A_np, B_np)
t_new = benchmark_new(A_gpu, B_gpu)

print(f"Old API (copy each time): {t_old:.3f}ms")
print(f"New API (zero copy):      {t_new:.3f}ms")
print(f"Speedup:                  {t_old / t_new:.2f}x")

# Test 3: Compare with PyTorch
print("\n" + "=" * 50)
print("TEST 3: COMPARISON WITH PYTORCH")
print("=" * 50)

try:
    import torch

    A_torch = torch.from_numpy(A_np).cuda()
    B_torch = torch.from_numpy(B_np).cuda()

    for _ in range(10):
        _ = torch.mm(A_torch, B_torch)
    torch.cuda.synchronize()

    times_pt = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = torch.mm(A_torch, B_torch)
        torch.cuda.synchronize()
        times_pt.append((time.perf_counter() - t0) * 1000)

    t_pt = np.mean(times_pt)
    print(f"PyTorch:        {t_pt:.3f}ms")
    print(f"Zenith (new):   {t_new:.3f}ms")
    print(f"Ratio:          {t_new / t_pt:.2f}x (1.0 = same as PyTorch)")

except ImportError:
    print("PyTorch not available")

# Test 4: Memory pool stats
print("\n" + "=" * 50)
print("TEST 4: MEMORY POOL STATS")
print("=" * 50)

stats = cuda.memory_stats()
print(f"Allocations:   {stats['allocations']}")
print(f"Cache hits:    {stats['cache_hits']}")
print(f"Cache returns: {stats['cache_returns']}")
print(f"Total alloc:   {stats['total_allocated'] / 1024**2:.1f} MB")

# Test 5: Verify accuracy
print("\n" + "=" * 50)
print("TEST 5: ACCURACY CHECK")
print("=" * 50)

C_gpu = cuda.matmul_gpu(A_gpu, B_gpu)
C_zenith = C_gpu.to_numpy()
C_numpy = A_np @ B_np

max_diff = np.max(np.abs(C_zenith - C_numpy))
print(f"Max difference vs NumPy: {max_diff:.2e}")
print(f"Accuracy: {'PASS' if max_diff < 1e-4 else 'FAIL'}")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)
