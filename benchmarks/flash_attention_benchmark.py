# FlashAttention Kernel Benchmark (Isolated)
# Tests ONLY the attention kernel, not the full encoder

import sys

sys.path.insert(0, "build/python")

import numpy as np
import time

print("=" * 70)
print("ZENITH FLASHATTENTION KERNEL BENCHMARK (ISOLATED)")
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

print("\n[1/3] Testing Zenith FlashAttention kernel...")

# Warmup
for _ in range(NUM_WARMUP):
    _ = cuda.flash_attention_gpu(Q_gpu, K_gpu, V_gpu)

# Benchmark Zenith
zenith_times = []
for _ in range(NUM_RUNS):
    t0 = time.perf_counter()
    output_gpu = cuda.flash_attention_gpu(Q_gpu, K_gpu, V_gpu)
    # Sync is included in the kernel
    zenith_times.append((time.perf_counter() - t0) * 1000)

zenith_mean = np.mean(zenith_times)
zenith_std = np.std(zenith_times)
print(f"  Zenith FlashAttention: {zenith_mean:.4f} ± {zenith_std:.4f} ms")

# Test accuracy against PyTorch
print("\n[2/3] Testing accuracy vs PyTorch...")
import torch
import torch.nn.functional as F

Q_torch = torch.from_numpy(Q).cuda()
K_torch = torch.from_numpy(K).cuda()
V_torch = torch.from_numpy(V).cuda()

# PyTorch scaled dot product attention
scale = 1.0 / np.sqrt(HEAD_DIM)
with torch.no_grad():
    # Manual attention for comparison
    scores = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * scale
    probs = F.softmax(scores, dim=-1)
    torch_out = torch.matmul(probs, V_torch)

torch.cuda.synchronize()
torch_out_np = torch_out.cpu().numpy()
zenith_out = output_gpu.to_numpy()

max_diff = np.max(np.abs(zenith_out - torch_out_np))
mean_diff = np.mean(np.abs(zenith_out - torch_out_np))
print(f"  Max diff: {max_diff:.2e}")
print(f"  Mean diff: {mean_diff:.2e}")
print(f"  Accuracy: {'PASS' if max_diff < 1e-4 else 'FAIL'}")

print("\n[3/3] Benchmarking PyTorch attention...")

# Warmup PyTorch
for _ in range(NUM_WARMUP):
    with torch.no_grad():
        scores = torch.matmul(Q_torch, K_torch.transpose(-2, -1)) * scale
        probs = F.softmax(scores, dim=-1)
        _ = torch.matmul(probs, V_torch)
    torch.cuda.synchronize()

# Benchmark PyTorch
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
pytorch_std = np.std(pytorch_times)
print(f"  PyTorch attention: {pytorch_mean:.4f} ± {pytorch_std:.4f} ms")

# Also test PyTorch's scaled_dot_product_attention
print("\n  Testing PyTorch SDPA (optimized)...")
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
sdpa_std = np.std(sdpa_times)
print(f"  PyTorch SDPA: {sdpa_mean:.4f} ± {sdpa_std:.4f} ms")

# Results
print("\n" + "=" * 70)
print("RESULTS (ATTENTION KERNEL ONLY)")
print("=" * 70)
print(f"\n  Zenith FlashAttention: {zenith_mean:.4f} ms")
print(f"  PyTorch Manual:        {pytorch_mean:.4f} ms")
print(f"  PyTorch SDPA:          {sdpa_mean:.4f} ms")

ratio_manual = pytorch_mean / zenith_mean
ratio_sdpa = sdpa_mean / zenith_mean

if ratio_manual > 1:
    print(f"\n  vs Manual: Zenith is {ratio_manual:.2f}x FASTER!")
else:
    print(f"\n  vs Manual: Zenith is {1 / ratio_manual:.2f}x slower")

if ratio_sdpa > 1:
    print(f"  vs SDPA:   Zenith is {ratio_sdpa:.2f}x FASTER!")
else:
    print(f"  vs SDPA:   Zenith is {1 / ratio_sdpa:.2f}x slower")

print("\n" + "=" * 70)
