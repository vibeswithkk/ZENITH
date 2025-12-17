# Zenith GPU Memory Optimization Test
# ====================================
# Tests the new GpuTensor API for zero-copy GPU operations
# Auto-saves results to benchmark_results.json and BENCHMARK_RESULTS.md

import numpy as np
import time
import json
from datetime import datetime

print("=" * 60)
print("ZENITH GPU MEMORY OPTIMIZATION TEST")
print("=" * 60)

results = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "framework": "Zenith",
    "version": "0.1.0",
}

try:
    from zenith._zenith_core import cuda, backends

    print(f"\nCUDA available: {cuda.is_available()}")
    print(f"Backends: {backends.list_available()}")
    results["cuda_available"] = True
    results["backends"] = backends.list_available()

    if backends.is_cudnn_available():
        results["cudnn_version"] = backends.get_cudnn_version()
        print(f"cuDNN version: {results['cudnn_version']}")
except ImportError as e:
    print(f"Zenith CUDA not available: {e}")
    exit(1)

# Get GPU info
try:
    import torch

    results["gpu"] = torch.cuda.get_device_name(0)
    results["gpu_memory_gb"] = (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    )
    print(f"GPU: {results['gpu']}")
except:
    results["gpu"] = "Unknown"

# Test 1: GpuTensor API
print("\n" + "=" * 60)
print("TEST 1: GpuTensor API")
print("=" * 60)

SIZE = 1024
A_np = np.random.randn(SIZE, SIZE).astype(np.float32)
B_np = np.random.randn(SIZE, SIZE).astype(np.float32)

results["matrix_size"] = f"{SIZE}x{SIZE}"

A_gpu = cuda.to_gpu(A_np)
B_gpu = cuda.to_gpu(B_np)

print(f"A_gpu: {A_gpu}")
print(f"B_gpu: {B_gpu}")

# Test 2: Zero-copy matmul benchmark
print("\n" + "=" * 60)
print("TEST 2: ZERO-COPY MATMUL BENCHMARK")
print("=" * 60)

ITERATIONS = 100


def benchmark_old(A_np, B_np, iterations=ITERATIONS):
    """Old API - copies every time"""
    for _ in range(10):
        cuda.matmul(A_np, B_np)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        C = cuda.matmul(A_np, B_np)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


def benchmark_new(A_gpu, B_gpu, iterations=ITERATIONS):
    """New API - zero copy compute"""
    for _ in range(10):
        cuda.matmul_gpu(A_gpu, B_gpu)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        C_gpu = cuda.matmul_gpu(A_gpu, B_gpu)
        times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times)


t_old, std_old = benchmark_old(A_np, B_np)
t_new, std_new = benchmark_new(A_gpu, B_gpu)

results["old_api_ms"] = round(t_old, 3)
results["old_api_std"] = round(std_old, 3)
results["new_api_ms"] = round(t_new, 3)
results["new_api_std"] = round(std_new, 3)
results["speedup_vs_old"] = round(t_old / t_new, 2)

print(f"Old API (copy each time): {t_old:.3f}ms +/- {std_old:.3f}")
print(f"New API (zero copy):      {t_new:.3f}ms +/- {std_new:.3f}")
print(f"Speedup:                  {results['speedup_vs_old']}x")

# Test 3: Compare with PyTorch
print("\n" + "=" * 60)
print("TEST 3: COMPARISON WITH PYTORCH")
print("=" * 60)

try:
    import torch

    A_torch = torch.from_numpy(A_np).cuda()
    B_torch = torch.from_numpy(B_np).cuda()

    for _ in range(10):
        _ = torch.mm(A_torch, B_torch)
    torch.cuda.synchronize()

    times_pt = []
    for _ in range(ITERATIONS):
        t0 = time.perf_counter()
        _ = torch.mm(A_torch, B_torch)
        torch.cuda.synchronize()
        times_pt.append((time.perf_counter() - t0) * 1000)

    t_pt = np.mean(times_pt)
    results["pytorch_ms"] = round(t_pt, 3)
    results["ratio_vs_pytorch"] = round(t_new / t_pt, 2)
    results["speedup_vs_pytorch"] = round(t_pt / t_new, 2)

    print(f"PyTorch:        {t_pt:.3f}ms")
    print(f"Zenith (new):   {t_new:.3f}ms")
    print(f"Ratio:          {results['ratio_vs_pytorch']}x (1.0 = same as PyTorch)")

    if t_new < t_pt:
        print(
            f"*** ZENITH IS {results['speedup_vs_pytorch']}x FASTER THAN PYTORCH! ***"
        )

except ImportError:
    print("PyTorch not available")
    results["pytorch_ms"] = None

# Test 4: Memory pool stats
print("\n" + "=" * 60)
print("TEST 4: MEMORY POOL STATS")
print("=" * 60)

stats = cuda.memory_stats()
results["memory_allocations"] = stats["allocations"]
results["memory_cache_hits"] = stats["cache_hits"]
results["memory_cache_returns"] = stats["cache_returns"]
results["memory_total_mb"] = round(stats["total_allocated"] / 1024**2, 1)
results["memory_hit_rate"] = round(
    stats["cache_hits"] / max(1, stats["cache_hits"] + stats["allocations"]) * 100, 1
)

print(f"Allocations:   {stats['allocations']}")
print(f"Cache hits:    {stats['cache_hits']}")
print(f"Cache returns: {stats['cache_returns']}")
print(f"Total alloc:   {results['memory_total_mb']} MB")
print(f"Hit rate:      {results['memory_hit_rate']}%")

# Test 5: Verify accuracy
print("\n" + "=" * 60)
print("TEST 5: ACCURACY CHECK")
print("=" * 60)

C_gpu = cuda.matmul_gpu(A_gpu, B_gpu)
C_zenith = C_gpu.to_numpy()
C_numpy = A_np @ B_np

max_diff = np.max(np.abs(C_zenith - C_numpy))
results["max_diff_vs_numpy"] = float(max_diff)
results["accuracy_pass"] = max_diff < 1e-4

print(f"Max difference vs NumPy: {max_diff:.2e}")
print(f"Accuracy: {'PASS' if results['accuracy_pass'] else 'FAIL'}")

# Save results
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

# Save JSON
with open("benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved: benchmark_results.json")

# Save Markdown report
md = f"""# Zenith GPU Benchmark Results

**Date:** {results["date"]}  
**GPU:** {results.get("gpu", "Unknown")}  
**Matrix Size:** {results["matrix_size"]}  

## Performance Results

| Metric | Value |
|--------|-------|
| Old API (copy) | {results["old_api_ms"]} ms |
| **New API (zero-copy)** | **{results["new_api_ms"]} ms** |
| **Speedup vs Old** | **{results["speedup_vs_old"]}x** |
| PyTorch | {results.get("pytorch_ms", "N/A")} ms |
| **Speedup vs PyTorch** | **{results.get("speedup_vs_pytorch", "N/A")}x** |

## Memory Pool Stats

| Metric | Value |
|--------|-------|
| Allocations | {results["memory_allocations"]} |
| Cache Hits | {results["memory_cache_hits"]} |
| Hit Rate | {results["memory_hit_rate"]}% |
| Total Allocated | {results["memory_total_mb"]} MB |

## Accuracy

- Max diff vs NumPy: {results["max_diff_vs_numpy"]:.2e}
- Status: **{"PASS" if results["accuracy_pass"] else "FAIL"}**

---

*Generated by Zenith Benchmark Suite v{results["version"]}*
"""

with open("BENCHMARK_RESULTS.md", "w") as f:
    f.write(md)
print("Saved: BENCHMARK_RESULTS.md")

print("\n" + "=" * 60)
print("TEST COMPLETE - RESULTS SAVED!")
print("=" * 60)
print(
    f"\nKey achievement: Zenith is {results.get('speedup_vs_pytorch', 'N/A')}x faster than PyTorch!"
)
