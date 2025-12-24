#!/usr/bin/env python
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
BERT Benchmark - MLPerf-style benchmarking for transformer models.

This script benchmarks Zenith's performance on BERT-style transformer models,
measuring latency, throughput, and comparing against native PyTorch.

Usage:
    python bert_benchmark.py
"""

import time
import numpy as np
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Benchmark requires PyTorch.")

try:
    import zenith

    HAS_ZENITH = True
except ImportError:
    HAS_ZENITH = False
    print("Zenith not available. Install with: pip install pyzenith")


@dataclass
class BenchmarkConfig:
    """Configuration for BERT benchmark."""

    batch_size: int = 8
    seq_length: int = 128
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 4
    num_warmup: int = 10
    num_runs: int = 100
    precision: str = "fp32"


@dataclass
class BenchmarkResult:
    """Results from benchmark run."""

    name: str
    latency_mean_ms: float
    latency_std_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float
    throughput_samples_per_sec: float
    memory_mb: Optional[float] = None


class TransformerBlock(nn.Module):
    """Single transformer encoder block (BERT-style)."""

    def __init__(self, hidden_size: int, num_heads: int):
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
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class BERTEncoder(nn.Module):
    """BERT-style transformer encoder."""

    def __init__(self, config: BenchmarkConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerBlock(config.hidden_size, config.num_heads)
                for _ in range(config.num_layers)
            ]
        )
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Pool first token
        return torch.tanh(self.pooler(x[:, 0]))


def measure_latency(
    model, input_tensor: torch.Tensor, num_warmup: int, num_runs: int
) -> list:
    """Measure inference latency."""
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(input_tensor)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    return latencies


def run_pytorch_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run benchmark with native PyTorch."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if config.precision == "fp16" else torch.float32

    model = BERTEncoder(config).to(device).to(dtype).eval()
    x = torch.randn(
        config.batch_size,
        config.seq_length,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )

    latencies = measure_latency(model, x, config.num_warmup, config.num_runs)

    return BenchmarkResult(
        name="PyTorch Native",
        latency_mean_ms=np.mean(latencies),
        latency_std_ms=np.std(latencies),
        latency_p50_ms=np.percentile(latencies, 50),
        latency_p90_ms=np.percentile(latencies, 90),
        latency_p99_ms=np.percentile(latencies, 99),
        throughput_samples_per_sec=config.batch_size * 1000 / np.mean(latencies),
    )


def run_zenith_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run benchmark with Zenith optimization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if config.precision == "fp16" else torch.float32

    model = BERTEncoder(config).to(device).to(dtype).eval()
    x = torch.randn(
        config.batch_size,
        config.seq_length,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )

    # Compile with Zenith
    optimized = zenith.compile(
        model, target=device, precision=config.precision, sample_input=x
    )

    latencies = measure_latency(optimized, x, config.num_warmup, config.num_runs)

    return BenchmarkResult(
        name="Zenith Optimized",
        latency_mean_ms=np.mean(latencies),
        latency_std_ms=np.std(latencies),
        latency_p50_ms=np.percentile(latencies, 50),
        latency_p90_ms=np.percentile(latencies, 90),
        latency_p99_ms=np.percentile(latencies, 99),
        throughput_samples_per_sec=config.batch_size * 1000 / np.mean(latencies),
    )


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"\n{result.name}:")
    print(f"  Latency (mean): {result.latency_mean_ms:.2f} ms")
    print(f"  Latency (std):  {result.latency_std_ms:.2f} ms")
    print(f"  Latency (P50):  {result.latency_p50_ms:.2f} ms")
    print(f"  Latency (P90):  {result.latency_p90_ms:.2f} ms")
    print(f"  Latency (P99):  {result.latency_p99_ms:.2f} ms")
    print(f"  Throughput:     {result.throughput_samples_per_sec:.1f} samples/sec")


def main():
    """Run BERT benchmark suite."""
    if not HAS_TORCH:
        print("PyTorch required for benchmark.")
        return

    print("=" * 60)
    print("BERT-style Transformer Benchmark")
    print("=" * 60)

    # Configuration
    config = BenchmarkConfig(
        batch_size=8,
        seq_length=128,
        hidden_size=768,
        num_heads=12,
        num_layers=4,
        num_warmup=10,
        num_runs=100,
        precision="fp32",
    )

    print(f"\nConfiguration:")
    print(f"  Batch Size:   {config.batch_size}")
    print(f"  Seq Length:   {config.seq_length}")
    print(f"  Hidden Size:  {config.hidden_size}")
    print(f"  Num Heads:    {config.num_heads}")
    print(f"  Num Layers:   {config.num_layers}")
    print(f"  Precision:    {config.precision}")
    print(f"  Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    if torch.cuda.is_available():
        print(f"  GPU:          {torch.cuda.get_device_name(0)}")

    # Run benchmarks
    print("\n" + "-" * 60)
    print("Running PyTorch Native...")
    pytorch_result = run_pytorch_benchmark(config)
    print_result(pytorch_result)

    if HAS_ZENITH:
        print("\n" + "-" * 60)
        print("Running Zenith Optimized...")
        zenith_result = run_zenith_benchmark(config)
        print_result(zenith_result)

        # Calculate speedup
        speedup = pytorch_result.latency_mean_ms / zenith_result.latency_mean_ms
        print("\n" + "=" * 60)
        print(f"SPEEDUP: {speedup:.2f}x")
        print("=" * 60)
    else:
        print("\nZenith not available. Skipping Zenith benchmark.")

    # FP16 benchmark
    if torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("FP16 Benchmark (Tensor Core)")
        print("=" * 60)

        config_fp16 = BenchmarkConfig(
            batch_size=8,
            seq_length=128,
            hidden_size=768,
            num_heads=12,
            num_layers=4,
            precision="fp16",
        )

        print("\nRunning PyTorch FP16...")
        pytorch_fp16_result = run_pytorch_benchmark(config_fp16)
        print_result(pytorch_fp16_result)

        fp16_speedup = (
            pytorch_result.latency_mean_ms / pytorch_fp16_result.latency_mean_ms
        )
        print(f"\nFP32 -> FP16 Speedup: {fp16_speedup:.2f}x")


if __name__ == "__main__":
    main()
