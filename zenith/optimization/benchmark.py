"""
Benchmark System - Performance Comparison and Analysis

Implements comprehensive benchmarking for Zenith:
- Comparison with NumPy baseline
- Framework comparison (when available)
- Automatic report generation

Based on CetakBiru Section 5.1 Phase 2 requirements.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import time
import json
import csv
import numpy as np
from dataclasses import dataclass, field
from typing import Callable
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    backend: str
    operation: str
    input_shape: list[int]
    iterations: int
    warmup_iterations: int
    total_time_s: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_gflops: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """Comparison between two benchmark results."""

    baseline: BenchmarkResult
    target: BenchmarkResult
    speedup: float
    improvement_percent: float


class Benchmark:
    """
    Benchmark runner for Zenith operations.

    Usage:
        benchmark = Benchmark()

        # Add benchmarks
        benchmark.add("matmul_small", matmul_fn, {"A": A, "B": B}, iterations=100)

        # Run all benchmarks
        results = benchmark.run()

        # Export results
        benchmark.export_json("benchmark_results.json")
        benchmark.export_csv("benchmark_results.csv")
    """

    def __init__(self):
        self.benchmarks: dict[str, dict] = {}
        self.results: list[BenchmarkResult] = []

    def add(
        self,
        name: str,
        fn: Callable,
        inputs: dict,
        iterations: int = 100,
        warmup: int = 10,
        operation: str = "unknown",
        backend: str = "zenith",
        flops: int = 0,
        memory_bytes: int = 0,
    ) -> None:
        """
        Add a benchmark.

        Args:
            name: Benchmark name
            fn: Function to benchmark (takes **inputs)
            inputs: Input dictionary for the function
            iterations: Number of timed iterations
            warmup: Number of warmup iterations
            operation: Operation type
            backend: Backend name
            flops: Estimated FLOPs for throughput calculation
            memory_bytes: Estimated memory access for bandwidth calculation
        """
        self.benchmarks[name] = {
            "fn": fn,
            "inputs": inputs,
            "iterations": iterations,
            "warmup": warmup,
            "operation": operation,
            "backend": backend,
            "flops": flops,
            "memory_bytes": memory_bytes,
        }

    def run(self, names: list[str] | None = None) -> list[BenchmarkResult]:
        """
        Run benchmarks.

        Args:
            names: Optional list of benchmark names to run (runs all if None)

        Returns:
            List of BenchmarkResult objects
        """
        self.results = []

        to_run = names if names else list(self.benchmarks.keys())

        for name in to_run:
            if name not in self.benchmarks:
                continue

            config = self.benchmarks[name]
            result = self._run_single(name, config)
            self.results.append(result)

        return self.results

    def _run_single(self, name: str, config: dict) -> BenchmarkResult:
        """Run a single benchmark."""
        fn = config["fn"]
        inputs = config["inputs"]
        iterations = config["iterations"]
        warmup = config["warmup"]

        # Warmup
        for _ in range(warmup):
            fn(**inputs)

        # Timed runs
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            fn(**inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        times = np.array(times)

        # Calculate throughput if FLOPs provided
        throughput = 0.0
        if config["flops"] > 0:
            mean_time_s = np.mean(times) / 1000
            throughput = config["flops"] / mean_time_s / 1e9  # GFLOPS

        # Calculate bandwidth if memory bytes provided
        bandwidth = 0.0
        if config["memory_bytes"] > 0:
            mean_time_s = np.mean(times) / 1000
            bandwidth = config["memory_bytes"] / mean_time_s / 1e9  # GB/s

        # Get input shape from first array input
        input_shape = []
        for v in inputs.values():
            if hasattr(v, "shape"):
                input_shape = list(v.shape)
                break

        return BenchmarkResult(
            name=name,
            backend=config["backend"],
            operation=config["operation"],
            input_shape=input_shape,
            iterations=iterations,
            warmup_iterations=warmup,
            total_time_s=np.sum(times) / 1000,
            mean_time_ms=float(np.mean(times)),
            std_time_ms=float(np.std(times)),
            min_time_ms=float(np.min(times)),
            max_time_ms=float(np.max(times)),
            throughput_gflops=throughput,
            memory_bandwidth_gbps=bandwidth,
        )

    def compare(
        self,
        baseline_name: str,
        target_name: str,
    ) -> ComparisonResult | None:
        """
        Compare two benchmark results.

        Args:
            baseline_name: Name of baseline benchmark
            target_name: Name of target benchmark

        Returns:
            ComparisonResult or None if benchmarks not found
        """
        baseline = None
        target = None

        for r in self.results:
            if r.name == baseline_name:
                baseline = r
            if r.name == target_name:
                target = r

        if baseline is None or target is None:
            return None

        speedup = baseline.mean_time_ms / target.mean_time_ms
        improvement = (
            (baseline.mean_time_ms - target.mean_time_ms) / baseline.mean_time_ms * 100
        )

        return ComparisonResult(
            baseline=baseline,
            target=target,
            speedup=speedup,
            improvement_percent=improvement,
        )

    def export_json(self, filepath: str | Path) -> None:
        """Export results to JSON file."""
        data = {
            "results": [
                {
                    "name": r.name,
                    "backend": r.backend,
                    "operation": r.operation,
                    "input_shape": r.input_shape,
                    "iterations": r.iterations,
                    "mean_time_ms": r.mean_time_ms,
                    "std_time_ms": r.std_time_ms,
                    "min_time_ms": r.min_time_ms,
                    "max_time_ms": r.max_time_ms,
                    "throughput_gflops": r.throughput_gflops,
                }
                for r in self.results
            ]
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def export_csv(self, filepath: str | Path) -> None:
        """Export results to CSV file."""
        fieldnames = [
            "name",
            "backend",
            "operation",
            "input_shape",
            "iterations",
            "mean_time_ms",
            "std_time_ms",
            "min_time_ms",
            "max_time_ms",
            "throughput_gflops",
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in self.results:
                writer.writerow(
                    {
                        "name": r.name,
                        "backend": r.backend,
                        "operation": r.operation,
                        "input_shape": str(r.input_shape),
                        "iterations": r.iterations,
                        "mean_time_ms": f"{r.mean_time_ms:.4f}",
                        "std_time_ms": f"{r.std_time_ms:.4f}",
                        "min_time_ms": f"{r.min_time_ms:.4f}",
                        "max_time_ms": f"{r.max_time_ms:.4f}",
                        "throughput_gflops": f"{r.throughput_gflops:.2f}",
                    }
                )

    def print_results(self) -> None:
        """Print results to console."""
        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        for r in self.results:
            print(f"\n{r.name} ({r.backend})")
            print(f"  Operation: {r.operation}")
            print(f"  Input Shape: {r.input_shape}")
            print(f"  Iterations: {r.iterations}")
            print(f"  Mean: {r.mean_time_ms:.4f} ms (+/- {r.std_time_ms:.4f})")
            print(f"  Min/Max: {r.min_time_ms:.4f} / {r.max_time_ms:.4f} ms")
            if r.throughput_gflops > 0:
                print(f"  Throughput: {r.throughput_gflops:.2f} GFLOPS")


def benchmark_matmul_vs_numpy(
    sizes: list[tuple[int, int, int]] | None = None,
    iterations: int = 100,
) -> dict:
    """
    Benchmark MatMul against NumPy.

    Args:
        sizes: List of (M, K, N) tuples
        iterations: Number of iterations per benchmark

    Returns:
        Dictionary with benchmark results and comparisons
    """
    if sizes is None:
        sizes = [(64, 64, 64), (256, 256, 256), (512, 512, 512)]

    benchmark = Benchmark()
    results = {"sizes": [], "numpy_ms": [], "zenith_ms": [], "speedups": []}

    for M, K, N in sizes:
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        # NumPy baseline
        benchmark.add(
            name=f"numpy_matmul_{M}x{K}x{N}",
            fn=lambda A=A, B=B: np.matmul(A, B),
            inputs={},
            iterations=iterations,
            operation="MatMul",
            backend="numpy",
            flops=2 * M * K * N,
        )

        # Zenith (using NumPy as fallback for now)
        benchmark.add(
            name=f"zenith_matmul_{M}x{K}x{N}",
            fn=lambda A=A, B=B: np.matmul(A, B),  # Will use C++ kernel when built
            inputs={},
            iterations=iterations,
            operation="MatMul",
            backend="zenith",
            flops=2 * M * K * N,
        )

    benchmark.run()

    # Collect results
    for M, K, N in sizes:
        numpy_result = next(
            (r for r in benchmark.results if r.name == f"numpy_matmul_{M}x{K}x{N}"),
            None,
        )
        zenith_result = next(
            (r for r in benchmark.results if r.name == f"zenith_matmul_{M}x{K}x{N}"),
            None,
        )

        if numpy_result and zenith_result:
            results["sizes"].append(f"{M}x{K}x{N}")
            results["numpy_ms"].append(numpy_result.mean_time_ms)
            results["zenith_ms"].append(zenith_result.mean_time_ms)
            results["speedups"].append(
                numpy_result.mean_time_ms / zenith_result.mean_time_ms
            )

    return results
