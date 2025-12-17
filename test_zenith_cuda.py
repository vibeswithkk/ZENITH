# Zenith CUDA Benchmark - Real Execution Test
# ============================================
# Run this in Colab after building with build_cuda.sh
# This tests ACTUAL Zenith execution via cuBLAS/cuDNN

import numpy as np
import time

# Test 1: Check backends
print("=" * 50)
print("ZENITH CUDA BACKEND TEST")
print("=" * 50)

try:
    from zenith._zenith_core import backends, cuda, kernels

    print(f"\nAvailable backends: {backends.list_available()}")
    print(f"CUDA available: {backends.is_cuda_available()}")
    print(f"cuDNN available: {backends.is_cudnn_available()}")

    if backends.is_cuda_available():
        print(f"cuDNN version: {backends.get_cudnn_version()}")
except ImportError as e:
    print(f"Zenith CUDA module not built: {e}")
    print("Run: !bash build_cuda.sh")
    exit(1)

# Test 2: cuBLAS MatMul
print("\n" + "=" * 50)
print("cuBLAS MATMUL BENCHMARK")
print("=" * 50)


def benchmark_matmul(M, N, K, iterations=100):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)

    # Warmup
    for _ in range(10):
        C = cuda.matmul(A, B)

    # Benchmark
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        C = cuda.matmul(A, B)
        times.append((time.perf_counter() - t0) * 1000)

    t = np.mean(times)
    gflops = (2 * M * N * K) / (t / 1000) / 1e9
    print(f"MatMul [{M}x{K}] @ [{K}x{N}] | {t:.3f}ms | {gflops:.1f} GFLOPS")
    return t


# Test various sizes
for size in [128, 256, 512, 1024, 2048]:
    benchmark_matmul(size, size, size, iterations=50)

# Test 3: Numerical accuracy
print("\n" + "=" * 50)
print("NUMERICAL ACCURACY CHECK")
print("=" * 50)

A = np.random.randn(64, 128).astype(np.float32)
B = np.random.randn(128, 64).astype(np.float32)

# Zenith cuBLAS
C_zenith = cuda.matmul(A, B)

# NumPy reference
C_numpy = A @ B

# Check
max_diff = np.max(np.abs(C_zenith - C_numpy))
print(f"Max difference vs NumPy: {max_diff:.2e}")
print(f"Accuracy: {'PASS' if max_diff < 1e-5 else 'FAIL'}")

# Test 4: Compare with PyTorch
print("\n" + "=" * 50)
print("COMPARISON: ZENITH vs PYTORCH")
print("=" * 50)

try:
    import torch

    M, N, K = 1024, 1024, 1024
    A_np = np.random.randn(M, K).astype(np.float32)
    B_np = np.random.randn(K, N).astype(np.float32)

    A_torch = torch.from_numpy(A_np).cuda()
    B_torch = torch.from_numpy(B_np).cuda()

    # PyTorch warmup
    for _ in range(10):
        _ = torch.mm(A_torch, B_torch)
    torch.cuda.synchronize()

    # PyTorch timing
    times_pt = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = torch.mm(A_torch, B_torch)
        torch.cuda.synchronize()
        times_pt.append((time.perf_counter() - t0) * 1000)

    # Zenith timing (already warmed up)
    times_z = []
    for _ in range(50):
        t0 = time.perf_counter()
        _ = cuda.matmul(A_np, B_np)
        times_z.append((time.perf_counter() - t0) * 1000)

    t_pt = np.mean(times_pt)
    t_z = np.mean(times_z)

    print(f"PyTorch: {t_pt:.3f}ms")
    print(f"Zenith:  {t_z:.3f}ms")
    print(f"Speedup: {t_pt / t_z:.2f}x")

except ImportError:
    print("PyTorch not available for comparison")

print("\n" + "=" * 50)
print("TEST COMPLETE")
print("=" * 50)
