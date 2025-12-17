# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Benchmark Runner

Provides end-to-end benchmarking for Zenith performance testing.
Per CetakBiru Section 5.3 - Performance Regression Testing.
"""

import time
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
from pathlib import Path


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    model_name: str
    backend: str
    precision: str
    batch_size: int

    # Timing metrics (in milliseconds)
    latency_mean: float
    latency_p50: float
    latency_p95: float
    latency_p99: float

    # Throughput
    throughput: float  # samples/second

    # Memory (in MB)
    peak_memory: float

    # Metadata
    warmup_iterations: int
    benchmark_iterations: int
    timestamp: str


def benchmark_model(
    model_name: str,
    backend: str = "cpu",
    precision: str = "fp32",
    batch_size: int = 1,
    warmup_iterations: int = 10,
    benchmark_iterations: int = 100,
    use_zenith: bool = True,
) -> Optional[BenchmarkResult]:
    """
    Benchmark a model on specified backend.

    Args:
        model_name: Name of model from models.py registry
        backend: Target backend (cpu, cuda)
        precision: Precision level (fp32, fp16, int8)
        batch_size: Batch size for inference
        warmup_iterations: Number of warmup runs
        benchmark_iterations: Number of timed runs
        use_zenith: If True, run through Zenith; otherwise native

    Returns:
        BenchmarkResult with timing and memory metrics
    """
    from .models import get_model, get_sample_input, get_model_config

    config = get_model_config(model_name)
    if config is None:
        print(f"Model not found: {model_name}")
        return None

    # Determine device
    device = "cuda" if backend == "cuda" else "cpu"

    # Load model
    model = get_model(model_name, device)
    if model is None:
        print(f"Failed to load model: {model_name}")
        return None

    # Get sample input
    sample_input = get_sample_input(model_name, batch_size, device)
    if sample_input is None:
        print(f"Failed to create sample input for: {model_name}")
        return None

    # Apply Zenith optimization if requested
    if use_zenith:
        model = _apply_zenith_optimization(model, sample_input, backend, precision)

    # Apply precision conversion
    model = _apply_precision(model, precision, device)

    # Warmup
    print(f"Warming up ({warmup_iterations} iterations)...")
    for _ in range(warmup_iterations):
        _run_inference(model, sample_input)

    # Synchronize before timing
    _synchronize(device)

    # Benchmark
    print(f"Benchmarking ({benchmark_iterations} iterations)...")
    latencies = []

    for _ in range(benchmark_iterations):
        start = time.perf_counter()
        _run_inference(model, sample_input)
        _synchronize(device)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # Convert to ms

    # Collect memory info
    peak_memory = _get_peak_memory(device)

    # Calculate statistics
    latencies.sort()
    n = len(latencies)

    result = BenchmarkResult(
        model_name=model_name,
        backend=backend,
        precision=precision,
        batch_size=batch_size,
        latency_mean=sum(latencies) / n,
        latency_p50=latencies[n // 2],
        latency_p95=latencies[int(n * 0.95)],
        latency_p99=latencies[int(n * 0.99)],
        throughput=batch_size * 1000 / (sum(latencies) / n),  # samples/sec
        peak_memory=peak_memory,
        warmup_iterations=warmup_iterations,
        benchmark_iterations=benchmark_iterations,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    # Cleanup
    del model
    del sample_input
    gc.collect()

    return result


def run_all_benchmarks(
    models: Optional[List[str]] = None,
    backends: Optional[List[str]] = None,
    precisions: Optional[List[str]] = None,
    batch_sizes: Optional[List[int]] = None,
    output_file: Optional[str] = None,
) -> List[BenchmarkResult]:
    """
    Run benchmarks across multiple configurations.

    Args:
        models: List of model names (default: all available)
        backends: List of backends (default: ["cpu", "cuda"])
        precisions: List of precisions (default: ["fp32"])
        batch_sizes: List of batch sizes (default: [1, 8])
        output_file: Optional JSON file to save results

    Returns:
        List of BenchmarkResult objects
    """
    from .models import list_models

    if models is None:
        models = ["resnet50", "mobilenet_v2"]  # Subset for quick testing

    if backends is None:
        backends = _get_available_backends()

    if precisions is None:
        precisions = ["fp32"]

    if batch_sizes is None:
        batch_sizes = [1, 8]

    results = []
    total_configs = len(models) * len(backends) * len(precisions) * len(batch_sizes)
    current = 0

    for model_name in models:
        for backend in backends:
            for precision in precisions:
                for batch_size in batch_sizes:
                    current += 1
                    print(
                        f"\n[{current}/{total_configs}] {model_name} | {backend} | {precision} | bs={batch_size}"
                    )

                    try:
                        result = benchmark_model(
                            model_name=model_name,
                            backend=backend,
                            precision=precision,
                            batch_size=batch_size,
                        )
                        if result:
                            results.append(result)
                            _print_result(result)
                    except Exception as e:
                        print(f"  Error: {e}")

    # Save results
    if output_file and results:
        _save_results(results, output_file)

    return results


def compare_with_baseline(
    zenith_result: BenchmarkResult,
    baseline_result: BenchmarkResult,
) -> Dict[str, float]:
    """
    Compare Zenith result with baseline (native framework).

    Returns:
        Dictionary with speedup factors
    """
    return {
        "latency_speedup": baseline_result.latency_mean / zenith_result.latency_mean,
        "throughput_speedup": zenith_result.throughput / baseline_result.throughput,
        "memory_ratio": zenith_result.peak_memory
        / max(baseline_result.peak_memory, 1.0),
    }


# ============================================================================
# Helper Functions
# ============================================================================


def _get_available_backends() -> List[str]:
    """Get list of available backends."""
    backends = ["cpu"]

    try:
        import torch

        if torch.cuda.is_available():
            backends.append("cuda")
    except ImportError:
        pass

    return backends


def _apply_zenith_optimization(
    model: Any, sample_input: Any, backend: str, precision: str
) -> Any:
    """Apply Zenith optimization to model."""
    try:
        import zenith

        # Use Zenith compile
        optimized = zenith.compile(
            model,
            target=backend,
            precision=precision,
            sample_input=sample_input,
        )
        return optimized

    except ImportError:
        print("  Zenith not available, using native model")
        return model
    except Exception as e:
        print(f"  Zenith optimization failed: {e}")
        return model


def _apply_precision(model: Any, precision: str, device: str) -> Any:
    """Convert model to specified precision."""
    try:
        import torch

        if precision == "fp16" and device == "cuda":
            return model.half()
        elif precision == "bf16" and device == "cuda":
            return model.to(dtype=torch.bfloat16)

    except (ImportError, AttributeError):
        pass

    return model


def _run_inference(model: Any, inputs: Any) -> Any:
    """Run a single inference pass."""
    try:
        import torch

        with torch.no_grad():
            if isinstance(inputs, dict):
                return model(**inputs)
            else:
                return model(inputs)

    except ImportError:
        # Non-PyTorch model
        if callable(model):
            if isinstance(inputs, dict):
                return model(**inputs)
            return model(inputs)

    return None


def _synchronize(device: str) -> None:
    """Synchronize device (for accurate timing)."""
    if device == "cuda":
        try:
            import torch

            torch.cuda.synchronize()
        except ImportError:
            pass


def _get_peak_memory(device: str) -> float:
    """Get peak memory usage in MB."""
    if device == "cuda":
        try:
            import torch

            return torch.cuda.max_memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

    # CPU memory (rough estimate)
    try:
        import psutil

        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    return 0.0


def _print_result(result: BenchmarkResult) -> None:
    """Print benchmark result."""
    print(
        f"  Latency: {result.latency_mean:.2f}ms (p50={result.latency_p50:.2f}, p95={result.latency_p95:.2f})"
    )
    print(f"  Throughput: {result.throughput:.1f} samples/sec")
    print(f"  Memory: {result.peak_memory:.1f} MB")


def _save_results(results: List[BenchmarkResult], output_file: str) -> None:
    """Save results to JSON file."""
    data = [
        {
            "model_name": r.model_name,
            "backend": r.backend,
            "precision": r.precision,
            "batch_size": r.batch_size,
            "latency_mean": r.latency_mean,
            "latency_p50": r.latency_p50,
            "latency_p95": r.latency_p95,
            "latency_p99": r.latency_p99,
            "throughput": r.throughput,
            "peak_memory": r.peak_memory,
            "timestamp": r.timestamp,
        }
        for r in results
    ]

    Path(output_file).write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to: {output_file}")


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Zenith Benchmarks")
    parser.add_argument("--model", type=str, default="resnet50", help="Model name")
    parser.add_argument(
        "--backend", type=str, default="cpu", help="Backend (cpu, cuda)"
    )
    parser.add_argument("--precision", type=str, default="fp32", help="Precision")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--iterations", type=int, default=100, help="Benchmark iterations"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--output", type=str, help="Output JSON file")

    args = parser.parse_args()

    if args.all:
        results = run_all_benchmarks(output_file=args.output)
    else:
        result = benchmark_model(
            model_name=args.model,
            backend=args.backend,
            precision=args.precision,
            batch_size=args.batch_size,
            benchmark_iterations=args.iterations,
        )
        if result:
            _print_result(result)
