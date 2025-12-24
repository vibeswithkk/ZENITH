#!/usr/bin/env python
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
ResNet Benchmark - MLPerf-style benchmarking for CNN models.

This script benchmarks Zenith's performance on ResNet-style CNN models,
measuring latency, throughput, and comparing against native PyTorch.

Usage:
    python resnet_benchmark.py
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
    """Configuration for ResNet benchmark."""

    batch_size: int = 32
    image_size: int = 224
    num_classes: int = 1000
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
    throughput_images_per_sec: float
    memory_mb: Optional[float] = None


class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18/34."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    """Simplified ResNet-18 architecture."""

    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, in_channels: int, out_channels: int, blocks: int, stride: int = 1
    ):
        layers = [BasicBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


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

    model = ResNet18(config.num_classes).to(device).to(dtype).eval()
    x = torch.randn(
        config.batch_size,
        3,
        config.image_size,
        config.image_size,
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
        throughput_images_per_sec=config.batch_size * 1000 / np.mean(latencies),
    )


def run_zenith_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run benchmark with Zenith optimization."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if config.precision == "fp16" else torch.float32

    model = ResNet18(config.num_classes).to(device).to(dtype).eval()
    x = torch.randn(
        config.batch_size,
        3,
        config.image_size,
        config.image_size,
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
        throughput_images_per_sec=config.batch_size * 1000 / np.mean(latencies),
    )


def print_result(result: BenchmarkResult):
    """Print benchmark result."""
    print(f"\n{result.name}:")
    print(f"  Latency (mean): {result.latency_mean_ms:.2f} ms")
    print(f"  Latency (std):  {result.latency_std_ms:.2f} ms")
    print(f"  Latency (P50):  {result.latency_p50_ms:.2f} ms")
    print(f"  Latency (P90):  {result.latency_p90_ms:.2f} ms")
    print(f"  Latency (P99):  {result.latency_p99_ms:.2f} ms")
    print(f"  Throughput:     {result.throughput_images_per_sec:.1f} images/sec")


def main():
    """Run ResNet benchmark suite."""
    if not HAS_TORCH:
        print("PyTorch required for benchmark.")
        return

    print("=" * 60)
    print("ResNet-18 Image Classification Benchmark")
    print("=" * 60)

    # Configuration
    config = BenchmarkConfig(
        batch_size=32,
        image_size=224,
        num_classes=1000,
        num_warmup=10,
        num_runs=100,
        precision="fp32",
    )

    print(f"\nConfiguration:")
    print(f"  Batch Size:   {config.batch_size}")
    print(f"  Image Size:   {config.image_size}x{config.image_size}")
    print(f"  Num Classes:  {config.num_classes}")
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

    # Batch size scaling
    print("\n" + "=" * 60)
    print("Batch Size Scaling")
    print("=" * 60)

    batch_sizes = [1, 8, 16, 32, 64]
    print(f"\n{'Batch':>8} {'Latency (ms)':>14} {'Throughput':>14}")
    print("-" * 40)

    for bs in batch_sizes:
        config_bs = BenchmarkConfig(
            batch_size=bs, image_size=224, num_classes=1000, num_warmup=5, num_runs=50
        )
        try:
            result = run_pytorch_benchmark(config_bs)
            print(
                f"{bs:>8} {result.latency_mean_ms:>14.2f} {result.throughput_images_per_sec:>14.1f}"
            )
        except RuntimeError as e:
            print(f"{bs:>8} {'OOM':>14} {'-':>14}")
            break


if __name__ == "__main__":
    main()
