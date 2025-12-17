# FlashAttention Kernel Benchmark (Isolated)
# Tests ONLY the attention kernel, not the full encoder
# Now includes cuBLAS-based attention for comparison

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH ATTENTION KERNEL BENCHMARK")
print("=" * 70)

from zenith._zenith_core import cuda

# Config
BATCH = 1
HEADS = 12
SEQ_LEN = 32
HEAD_DIM = 64
NUM_WARMUP = 10
NUM_RUNS = 100

print(f"\nConfig: batch={BATCH}, heads={HEADS}, seq={SEQ_LEN}, dim={HEAD_DIM}")

# Prepare data
np.random.seed(42)
Q = np.random.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)
K = np.random.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)
V = np.random.randn(BATCH, HEADS, SEQ_LEN, HEAD_DIM).astype(np.float32)

# Upload to GPU once
Q_gpu = cuda.to_gpu(np.ascontiguousarray(Q))
K_gpu = cuda.to_gpu(np.ascontiguousarray(K))
V_gpu = cuda.to_gpu(np.ascontiguousarray(V))

# ============================================
# Test 1: Zenith Custom FlashAttention
# ============================================
print("\n[1/4] Testing Zenith Custom FlashAttention...")
for _ in range(NUM_WARMUP):
    _ = cuda.flash_attention_gpu(Q_gpu, K_gpu, V_gpu)

custom_times = []
for _ in range(NUM_RUNS):
    t0 = time.perf_counter()
    output_custom = cuda.flash_attention_gpu(Q_gpu, K_gpu, V_gpu)
    custom_times.append((time.perf_counter() - t0) * 1000)

custom_mean = np.mean(custom_times)
print(f"  Zenith Custom: {custom_mean:.4f} ms")

# ============================================
# Test 2: Zenith cuBLAS Attention
# ============================================
print("\n[2/4] Testing Zenith cuBLAS Attention (TF32 Tensor Cores)...")
try:
    for _ in range(NUM_WARMUP):
        _ = cuda.cublas_attention_gpu(Q_gpu, K_gpu, V_gpu)

    cublas_times = []
    for _ in range(NUM_RUNS):
        t0 = time.perf_counter()
        output_cublas = cuda.cublas_attention_gpu(Q_gpu, K_gpu, V_gpu)
        cublas_times.append((time.perf_counter() - t0) * 1000)

    cublas_mean = np.mean(cublas_times)
    print(f"  Zenith cuBLAS: {cublas_mean:.4f} ms")
    has_cublas = True
except Exception as e:
    print(f"  cuBLAS attention not available: {e}")
    has_cublas = False
    cublas_mean = float("inf")

# ============================================
# Test 3: PyTorch Reference
# ============================================
print("\n[3/4] Benchmarking PyTorch attention...")
import torch
import torch.nn.functional as F

Q_torch = torch.from_numpy(Q).cuda()
K_torch = torch.from_numpy(K).cuda()
V_torch = torch.from_numpy(V).cuda()

# PyTorch manual attention
scale = 1.0 / np.sqrt(HEAD_DIM)
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        scores = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1)
        _ = torch.matmul(probs, V_torch)
    torch.cuda.synchronize()

pytorch_times = []
for _ in range(NUM_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        scores = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1)
        _ = torch.matmul(probs, V_torch)
    torch.cuda.synchronize()
    pytorch_times.append((time.perf_counter() - t0) * 1000)

pytorch_mean = np.mean(pytorch_times)
print(f"  PyTorch Manual: {pytorch_mean:.4f} ms")

# PyTorch SDPA
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        _ = F.scaled_dot_product_attention(Q_torch, K_torch, V_torch)
    torch.cuda.synchronize()

sdpa_times = []
for _ in range(NUM_RUNS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = F.scaled_dot_product_attention(Q_torch, K_torch, V_torch)
    torch.cuda.synchronize()
    sdpa_times.append((time.perf_counter() - t0) * 1000)

sdpa_mean = np.mean(sdpa_times)
print(f"  PyTorch SDPA: {sdpa_mean:.4f} ms")

# ============================================
# Accuracy Check
# ============================================
print("\n[4/4] Accuracy check...")
with torch.no_grad():
    scores = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    torch_out = torch.matmul(probs, V_torch)
torch.cuda.synchronize()
torch_out_np = torch_out.cpu().numpy()

# Check custom kernel
custom_out = output_custom.to_numpy()
custom_diff = np.max(np.abs(custom_out - torch_out_np))
print(
    f"  Custom max diff: {custom_diff:.2e} - {'PASS' if custom_diff < 1e-4 else 'FAIL'}"
)

# Check cuBLAS kernel
if has_cublas:
    cublas_out = output_cublas.to_numpy()
    cublas_diff = np.max(np.abs(cublas_out - torch_out_np))
    print(
        f"  cuBLAS max diff: {cublas_diff:.2e} - {'PASS' if cublas_diff < 1e-4 else 'FAIL'}"
    )

# ============================================
# Results Summary
# ============================================
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\n  Zenith Custom:  {custom_mean:.4f} ms")
if has_cublas:
    print(f"  Zenith cuBLAS:  {cublas_mean:.4f} ms")
print(f"  PyTorch Manual: {pytorch_mean:.4f} ms")
print(f"  PyTorch SDPA:   {sdpa_mean:.4f} ms")

# Comparisons
print("\n  Comparisons vs PyTorch Manual:")
ratio_custom = pytorch_mean / custom_mean
print(f"    Custom: {ratio_custom:.2f}x {'FASTER' if ratio_custom > 1 else 'slower'}")
if has_cublas:
    ratio_cublas = pytorch_mean / cublas_mean
    print(
        f"    cuBLAS: {ratio_cublas:.2f}x {'FASTER' if ratio_cublas > 1 else 'slower'}"
    )

print("\n" + "=" * 70)
