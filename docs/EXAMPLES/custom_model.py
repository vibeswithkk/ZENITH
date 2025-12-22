# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Custom Model Example

Demonstrates optimizing a custom PyTorch model with Zenith.
"""

import torch
import torch.nn as nn
import time


class CustomTransformerBlock(nn.Module):
    """Custom transformer block."""

    def __init__(self, d_model: int = 512, nhead: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class CustomModel(nn.Module):
    """Custom model with multiple transformer blocks."""

    def __init__(self, d_model: int = 512, num_layers: int = 6):
        super().__init__()
        self.embed = nn.Linear(768, d_model)
        self.blocks = nn.ModuleList(
            [CustomTransformerBlock(d_model) for _ in range(num_layers)]
        )
        self.head = nn.Linear(d_model, 1000)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1))


def main():
    """Run custom model example."""
    import zenith

    # Create model
    print("Creating custom model...")
    model = CustomModel(d_model=512, num_layers=6)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    # Prepare input
    batch_size = 8
    seq_len = 128
    x = torch.randn(batch_size, seq_len, 768).to(device)

    # Baseline
    print("\nBaseline inference...")
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            output = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) * 100

    # Compile with Zenith
    print("Compiling with Zenith...")
    optimized = zenith.compile(
        model,
        target=device,
        precision="fp16" if device == "cuda" else "fp32",
    )

    # Optimized
    print("\nOptimized inference...")
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            output = optimized(x)
        if device == "cuda":
            torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start) * 100

    # Results
    print(f"\nResults:")
    print(f"  Model: CustomModel (6-layer transformer)")
    print(f"  Input: ({batch_size}, {seq_len}, 768)")
    print(f"  Baseline:  {baseline_time:.2f}ms")
    print(f"  Optimized: {optimized_time:.2f}ms")
    print(f"  Speedup:   {baseline_time / optimized_time:.2f}x")


if __name__ == "__main__":
    main()
