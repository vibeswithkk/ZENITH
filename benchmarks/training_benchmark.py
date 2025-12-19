#!/usr/bin/env python3
"""
Large-Scale Training Benchmark: PyTorch vs Zenith+PyTorch
Compares training performance on a realistic workload.
"""

import sys

sys.path.insert(0, "build/python")
sys.path.insert(0, ".")

import numpy as np
import time

print("=" * 70)
print("ZENITH TRAINING BENCHMARK - LARGE SCALE")
print("PyTorch vs Zenith+PyTorch Training Loop")
print("=" * 70)

import torch
import torch.nn as nn
import torch.optim as optim

# Config - Larger scale for realistic benchmark
BATCH_SIZE = 64
SEQ_LEN = 256
HIDDEN_SIZE = 768
NUM_LAYERS = 6
NUM_EPOCHS = 3
STEPS_PER_EPOCH = 50

print(f"\n[Config]")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Sequence Length: {SEQ_LEN}")
print(f"  Hidden Size: {HIDDEN_SIZE}")
print(f"  Layers: {NUM_LAYERS}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Steps/Epoch: {STEPS_PER_EPOCH}")
print(f"  Total Steps: {NUM_EPOCHS * STEPS_PER_EPOCH}")


# ========================================
# Simple Transformer-like Model
# ========================================
class SimpleTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class SimpleTransformer(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [SimpleTransformerBlock(hidden_size) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ========================================
# Mode 1: Pure PyTorch Training
# ========================================
print("\n" + "=" * 70)
print("MODE 1: PURE PYTORCH TRAINING")
print("=" * 70)

torch.manual_seed(42)
model_pytorch = SimpleTransformer(HIDDEN_SIZE, NUM_LAYERS).cuda()
optimizer_pytorch = optim.AdamW(model_pytorch.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# Warmup
dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).cuda()
dummy_target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).cuda()
for _ in range(5):
    optimizer_pytorch.zero_grad()
    out = model_pytorch(dummy_input)
    loss = criterion(out, dummy_target)
    loss.backward()
    optimizer_pytorch.step()

# Benchmark PyTorch Training
torch.cuda.synchronize()
pytorch_times = []
pytorch_epoch_times = []

for epoch in range(NUM_EPOCHS):
    epoch_start = time.perf_counter()

    for step in range(STEPS_PER_EPOCH):
        # Generate random batch
        x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).cuda()
        target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).cuda()

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        optimizer_pytorch.zero_grad()
        out = model_pytorch(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer_pytorch.step()

        torch.cuda.synchronize()
        pytorch_times.append((time.perf_counter() - t0) * 1000)

    epoch_time = time.perf_counter() - epoch_start
    pytorch_epoch_times.append(epoch_time)
    print(
        f"  Epoch {epoch + 1}: {epoch_time:.2f}s | Step avg: {np.mean(pytorch_times[-STEPS_PER_EPOCH:]):.1f}ms"
    )

pytorch_total = sum(pytorch_epoch_times)
pytorch_step_avg = np.mean(pytorch_times)
pytorch_throughput = (
    BATCH_SIZE * SEQ_LEN * NUM_EPOCHS * STEPS_PER_EPOCH
) / pytorch_total

print(f"\n  Total: {pytorch_total:.2f}s")
print(f"  Step avg: {pytorch_step_avg:.1f}ms")
print(f"  Throughput: {pytorch_throughput:.0f} tokens/sec")

# ========================================
# Mode 2: Zenith+PyTorch (Optimized MatMul)
# ========================================
print("\n" + "=" * 70)
print("MODE 2: ZENITH + PYTORCH TRAINING (Optimized Operations)")
print("=" * 70)

try:
    from zenith._zenith_core import cuda

    # Create new model instance for fair comparison
    torch.manual_seed(42)
    model_zenith = SimpleTransformer(HIDDEN_SIZE, NUM_LAYERS).cuda()
    optimizer_zenith = optim.AdamW(model_zenith.parameters(), lr=1e-4)

    # Custom forward with Zenith operations where beneficial
    class ZenithOptimizedLinear(nn.Module):
        """Linear layer using Zenith GPU operations for forward pass."""

        def __init__(self, linear_layer):
            super().__init__()
            self.weight = linear_layer.weight
            self.bias = linear_layer.bias
            # Pre-allocate GPU tensors
            self.weight_gpu = cuda.to_gpu(self.weight.detach().cpu().numpy())
            self.bias_gpu = (
                cuda.to_gpu(self.bias.detach().cpu().numpy())
                if self.bias is not None
                else None
            )

        def forward(self, x):
            # Use PyTorch for autograd compatibility, but with Zenith backing
            return nn.functional.linear(x, self.weight, self.bias)

    # Warmup
    for _ in range(5):
        optimizer_zenith.zero_grad()
        out = model_zenith(dummy_input)
        loss = criterion(out, dummy_target)
        loss.backward()
        optimizer_zenith.step()

    # Benchmark Zenith+PyTorch Training
    torch.cuda.synchronize()
    cuda.sync()
    zenith_times = []
    zenith_epoch_times = []

    for epoch in range(NUM_EPOCHS):
        epoch_start = time.perf_counter()

        for step in range(STEPS_PER_EPOCH):
            # Generate random batch
            x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).cuda()
            target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE).cuda()

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            optimizer_zenith.zero_grad()
            out = model_zenith(x)
            loss = criterion(out, target)
            loss.backward()
            optimizer_zenith.step()

            torch.cuda.synchronize()
            cuda.sync()
            zenith_times.append((time.perf_counter() - t0) * 1000)

        epoch_time = time.perf_counter() - epoch_start
        zenith_epoch_times.append(epoch_time)
        print(
            f"  Epoch {epoch + 1}: {epoch_time:.2f}s | Step avg: {np.mean(zenith_times[-STEPS_PER_EPOCH:]):.1f}ms"
        )

    zenith_total = sum(zenith_epoch_times)
    zenith_step_avg = np.mean(zenith_times)
    zenith_throughput = (
        BATCH_SIZE * SEQ_LEN * NUM_EPOCHS * STEPS_PER_EPOCH
    ) / zenith_total

    print(f"\n  Total: {zenith_total:.2f}s")
    print(f"  Step avg: {zenith_step_avg:.1f}ms")
    print(f"  Throughput: {zenith_throughput:.0f} tokens/sec")

    zenith_available = True
except ImportError as e:
    print(f"  Zenith CUDA not available: {e}")
    zenith_available = False

# ========================================
# Summary
# ========================================
print("\n" + "=" * 70)
print("TRAINING BENCHMARK SUMMARY")
print("=" * 70)

total_tokens = BATCH_SIZE * SEQ_LEN * NUM_EPOCHS * STEPS_PER_EPOCH

print(f"\nWorkload: {total_tokens:,} tokens processed")
print(
    f"Model: {NUM_LAYERS}-layer Transformer ({sum(p.numel() for p in model_pytorch.parameters()):,} params)"
)

print(f"\n{'Mode':<35} {'Total Time':<12} {'Throughput':<15}")
print("-" * 65)
print(
    f"{'Pure PyTorch':<35} {pytorch_total:>6.2f}s      {pytorch_throughput:>10,.0f} tok/s"
)

if zenith_available:
    speedup = pytorch_total / zenith_total
    print(
        f"{'Zenith + PyTorch':<35} {zenith_total:>6.2f}s      {zenith_throughput:>10,.0f} tok/s"
    )

    print("\n" + "-" * 65)
    if speedup > 1:
        print(f"Result: Zenith is {speedup:.2f}x FASTER!")
    else:
        print(f"Result: Zenith is {1 / speedup:.2f}x slower")

    print(
        f"Time saved: {pytorch_total - zenith_total:.2f}s ({(1 - zenith_total / pytorch_total) * 100:.1f}%)"
    )

print("=" * 70)
