#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Zenith JAX Benchmark Suite

Benchmarks for JAX integration including:
- Gradient checkpointing memory reduction
- Mixed precision speedup
- Custom primitives vs baseline
"""

import time
import argparse
from dataclasses import dataclass
from typing import Callable

# Check JAX availability
try:
    import jax
    import jax.numpy as jnp
    from jax import random

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("JAX not available. Install with: pip install jax jaxlib")


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    time_ms: float
    memory_mb: float
    iterations: int
    throughput: float = 0.0


def benchmark_fn(
    fn: Callable,
    args: tuple,
    warmup: int = 5,
    iterations: int = 100,
    name: str = "benchmark",
) -> BenchmarkResult:
    """
    Benchmark a JAX function.

    Args:
        fn: Function to benchmark
        args: Arguments to pass to function
        warmup: Number of warmup iterations
        iterations: Number of timed iterations
        name: Name for the benchmark

    Returns:
        BenchmarkResult with timing information
    """
    if not HAS_JAX:
        return BenchmarkResult(name=name, time_ms=0, memory_mb=0, iterations=0)

    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()

    # Timed runs
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn(*args)
        if hasattr(result, "block_until_ready"):
            result.block_until_ready()
    end = time.perf_counter()

    total_time_ms = (end - start) * 1000
    avg_time_ms = total_time_ms / iterations

    return BenchmarkResult(
        name=name,
        time_ms=avg_time_ms,
        memory_mb=0,  # JAX doesn't expose this easily
        iterations=iterations,
        throughput=1000 / avg_time_ms,  # ops/sec
    )


def benchmark_attention_primitives():
    """Benchmark Zenith attention primitives vs baseline."""
    if not HAS_JAX:
        return []

    print("\n=== Attention Primitives Benchmark ===")

    results = []
    batch, heads, seq, dim = 2, 8, 512, 64

    key = random.PRNGKey(0)
    q = random.normal(key, (batch, heads, seq, dim))
    k = random.normal(random.PRNGKey(1), (batch, heads, seq, dim))
    v = random.normal(random.PRNGKey(2), (batch, heads, seq, dim))

    # Baseline attention
    def baseline_attention(q, k, v):
        scale = q.shape[-1] ** -0.5
        attn = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        return jnp.matmul(attn, v)

    baseline_jit = jax.jit(baseline_attention)
    result = benchmark_fn(baseline_jit, (q, k, v), name="baseline_attention")
    results.append(result)
    print(f"  Baseline: {result.time_ms:.3f} ms")

    # Zenith attention
    try:
        from zenith.jax.primitives import fused_attention

        zenith_jit = jax.jit(fused_attention)
        result = benchmark_fn(zenith_jit, (q, k, v), name="zenith_attention")
        results.append(result)
        print(f"  Zenith:   {result.time_ms:.3f} ms")

        if results[0].time_ms > 0:
            speedup = results[0].time_ms / results[1].time_ms
            print(f"  Speedup:  {speedup:.2f}x")
    except ImportError:
        print("  Zenith primitives not available")

    return results


def benchmark_gelu_primitives():
    """Benchmark Zenith GELU vs baseline."""
    if not HAS_JAX:
        return []

    print("\n=== GELU Primitives Benchmark ===")

    results = []
    shape = (16, 512, 4096)

    key = random.PRNGKey(0)
    x = random.normal(key, shape)

    # Baseline GELU
    baseline_jit = jax.jit(jax.nn.gelu)
    result = benchmark_fn(baseline_jit, (x,), name="baseline_gelu")
    results.append(result)
    print(f"  Baseline: {result.time_ms:.3f} ms")

    # Zenith GELU
    try:
        from zenith.jax.primitives import fused_gelu

        zenith_jit = jax.jit(fused_gelu)
        result = benchmark_fn(zenith_jit, (x,), name="zenith_gelu")
        results.append(result)
        print(f"  Zenith:   {result.time_ms:.3f} ms")

        if results[0].time_ms > 0:
            speedup = results[0].time_ms / results[1].time_ms
            print(f"  Speedup:  {speedup:.2f}x")
    except ImportError:
        print("  Zenith primitives not available")

    return results


def benchmark_mixed_precision():
    """Benchmark FP32 vs FP16/BF16 performance."""
    if not HAS_JAX:
        return []

    print("\n=== Mixed Precision Benchmark ===")

    results = []
    batch, hidden = 64, 4096

    key = random.PRNGKey(0)

    # FP32
    x32 = random.normal(key, (batch, hidden), dtype=jnp.float32)
    w32 = random.normal(random.PRNGKey(1), (hidden, hidden), dtype=jnp.float32)

    def matmul_fn(x, w):
        return jnp.matmul(x, w)

    fp32_jit = jax.jit(matmul_fn)
    result = benchmark_fn(fp32_jit, (x32, w32), name="matmul_fp32")
    results.append(result)
    print(f"  FP32:   {result.time_ms:.3f} ms")

    # FP16
    x16 = x32.astype(jnp.float16)
    w16 = w32.astype(jnp.float16)

    result = benchmark_fn(fp32_jit, (x16, w16), name="matmul_fp16")
    results.append(result)
    print(f"  FP16:   {result.time_ms:.3f} ms")

    # BF16
    try:
        x_bf16 = x32.astype(jnp.bfloat16)
        w_bf16 = w32.astype(jnp.bfloat16)

        result = benchmark_fn(fp32_jit, (x_bf16, w_bf16), name="matmul_bf16")
        results.append(result)
        print(f"  BF16:   {result.time_ms:.3f} ms")
    except TypeError:
        print("  BF16 not supported on this device")

    return results


def run_all_benchmarks():
    """Run all JAX benchmarks."""
    print("=" * 60)
    print("  ZENITH JAX BENCHMARK SUITE")
    print("=" * 60)

    if not HAS_JAX:
        print("JAX not available. Exiting.")
        return

    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")

    all_results = []

    all_results.extend(benchmark_attention_primitives())
    all_results.extend(benchmark_gelu_primitives())
    all_results.extend(benchmark_mixed_precision())

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<25} {'Time (ms)':<12} {'Throughput (ops/s)':<15}")
    print("-" * 60)
    for r in all_results:
        print(f"{r.name:<25} {r.time_ms:<12.3f} {r.throughput:<15.1f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zenith JAX Benchmarks")
    parser.add_argument(
        "--attention", action="store_true", help="Run attention benchmarks"
    )
    parser.add_argument("--gelu", action="store_true", help="Run GELU benchmarks")
    parser.add_argument(
        "--precision", action="store_true", help="Run precision benchmarks"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")

    args = parser.parse_args()

    if args.all or not any([args.attention, args.gelu, args.precision]):
        run_all_benchmarks()
    else:
        if args.attention:
            benchmark_attention_primitives()
        if args.gelu:
            benchmark_gelu_primitives()
        if args.precision:
            benchmark_mixed_precision()
