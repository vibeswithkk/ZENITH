"""
GPU Benchmark Utilities for Zenith.

Provides tools for measuring:
- CUDA kernel timing
- Memory bandwidth
- Throughput calculations
- Model performance metrics

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Optional CUDA support
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False


# ============================================================================
# Benchmark Result Classes
# ============================================================================


@dataclass
class LatencyStats:
    """Latency statistics from benchmark runs."""

    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    num_samples: int = 0

    @staticmethod
    def compute(samples_ms: list[float]) -> "LatencyStats":
        """Compute statistics from timing samples."""
        if not samples_ms:
            return LatencyStats()

        arr = np.array(samples_ms)
        return LatencyStats(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p50_ms=float(np.percentile(arr, 50)),
            p90_ms=float(np.percentile(arr, 90)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            num_samples=len(samples_ms),
        )

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Latency: {self.mean_ms:.3f} +/- {self.std_ms:.3f} ms "
            f"(min={self.min_ms:.3f}, max={self.max_ms:.3f}, "
            f"p50={self.p50_ms:.3f}, p99={self.p99_ms:.3f}, n={self.num_samples})"
        )


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    samples_per_sec: float = 0.0
    batches_per_sec: float = 0.0
    batch_size: int = 1
    total_samples: int = 0
    total_time_sec: float = 0.0

    def summary(self) -> str:
        """Return summary string."""
        return (
            f"Throughput: {self.samples_per_sec:.2f} samples/sec, "
            f"{self.batches_per_sec:.2f} batches/sec (batch={self.batch_size})"
        )


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    peak_bytes: int = 0
    allocated_bytes: int = 0
    reserved_bytes: int = 0

    @property
    def peak_mb(self) -> float:
        """Peak memory in MB."""
        return self.peak_bytes / (1024 * 1024)

    @property
    def allocated_mb(self) -> float:
        """Allocated memory in MB."""
        return self.allocated_bytes / (1024 * 1024)

    def summary(self) -> str:
        """Return summary string."""
        return f"Memory: peak={self.peak_mb:.2f} MB, alloc={self.allocated_mb:.2f} MB"


@dataclass
class BenchmarkResult:
    """Complete benchmark result."""

    name: str = ""
    latency: LatencyStats = field(default_factory=LatencyStats)
    throughput: ThroughputStats = field(default_factory=ThroughputStats)
    memory: MemoryStats = field(default_factory=MemoryStats)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return full summary string."""
        lines = [
            f"Benchmark: {self.name}",
            f"  {self.latency.summary()}",
            f"  {self.throughput.summary()}",
            f"  {self.memory.summary()}",
        ]
        return "\n".join(lines)


# ============================================================================
# Timer Classes
# ============================================================================


class CPUTimer:
    """High-resolution CPU timer."""

    def __init__(self):
        self._start = 0.0
        self._end = 0.0

    def start(self):
        """Start timing."""
        self._start = time.perf_counter()

    def stop(self):
        """Stop timing."""
        self._end = time.perf_counter()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self._end - self._start) * 1000.0

    def elapsed_us(self) -> float:
        """Get elapsed time in microseconds."""
        return (self._end - self._start) * 1_000_000.0

    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self._end - self._start


class GPUTimer:
    """CUDA event-based GPU timer using CuPy."""

    def __init__(self):
        if not HAS_CUPY:
            raise RuntimeError("CuPy required for GPU timing")
        self._start_event = cp.cuda.Event()
        self._end_event = cp.cuda.Event()

    def start(self, stream=None):
        """Start timing on given stream."""
        self._start_event.record(stream)

    def stop(self, stream=None):
        """Stop timing on given stream."""
        self._end_event.record(stream)

    def synchronize(self):
        """Wait for GPU operations to complete."""
        self._end_event.synchronize()

    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        self.synchronize()
        return cp.cuda.get_elapsed_time(self._start_event, self._end_event)


# ============================================================================
# Benchmark Utilities
# ============================================================================


def benchmark_function(
    func: Callable,
    args: tuple = (),
    kwargs: dict | None = None,
    warmup_runs: int = 3,
    benchmark_runs: int = 10,
    use_gpu_timer: bool = False,
) -> BenchmarkResult:
    """
    Benchmark a function with proper warmup and statistics.

    Args:
        func: Function to benchmark
        args: Positional arguments
        kwargs: Keyword arguments
        warmup_runs: Number of warmup iterations
        benchmark_runs: Number of benchmark iterations
        use_gpu_timer: Use CUDA events for GPU timing

    Returns:
        BenchmarkResult with timing and throughput stats
    """
    kwargs = kwargs or {}

    # Warmup
    for _ in range(warmup_runs):
        func(*args, **kwargs)

    # Synchronize GPU if available
    if HAS_CUPY:
        cp.cuda.Stream.null.synchronize()

    # Create timer
    if use_gpu_timer and HAS_CUPY:
        timer = GPUTimer()
    else:
        timer = CPUTimer()

    samples_ms = []

    for _ in range(benchmark_runs):
        timer.start()
        func(*args, **kwargs)

        if use_gpu_timer and HAS_CUPY:
            cp.cuda.Stream.null.synchronize()

        timer.stop()
        samples_ms.append(timer.elapsed_ms())

    latency = LatencyStats.compute(samples_ms)

    # Compute throughput
    total_time_sec = sum(samples_ms) / 1000.0
    throughput = ThroughputStats(
        batches_per_sec=benchmark_runs / total_time_sec if total_time_sec > 0 else 0,
        samples_per_sec=benchmark_runs / total_time_sec if total_time_sec > 0 else 0,
        batch_size=1,
        total_samples=benchmark_runs,
        total_time_sec=total_time_sec,
    )

    return BenchmarkResult(
        name=func.__name__ if hasattr(func, "__name__") else str(func),
        latency=latency,
        throughput=throughput,
    )


def measure_memory_bandwidth(
    size_bytes: int = 256 * 1024 * 1024,  # 256 MB
    iterations: int = 10,
) -> dict[str, float]:
    """
    Measure GPU memory bandwidth.

    Args:
        size_bytes: Size of data to transfer
        iterations: Number of iterations

    Returns:
        Dictionary with bandwidth measurements in GB/s
    """
    if not HAS_CUPY:
        return {"error": "CuPy not available"}

    results = {}

    # Host to Device
    host_data = np.random.randn(size_bytes // 4).astype(np.float32)

    timer = GPUTimer()
    timer.start()
    for _ in range(iterations):
        device_data = cp.asarray(host_data)
        cp.cuda.Stream.null.synchronize()
    timer.stop()

    h2d_time_ms = timer.elapsed_ms()
    h2d_bandwidth = (size_bytes * iterations) / (h2d_time_ms / 1000.0) / 1e9
    results["h2d_bandwidth_gbps"] = h2d_bandwidth

    # Device to Device
    src = cp.random.randn(size_bytes // 4, dtype=cp.float32)
    dst = cp.empty_like(src)

    timer.start()
    for _ in range(iterations):
        cp.copyto(dst, src)
        cp.cuda.Stream.null.synchronize()
    timer.stop()

    d2d_time_ms = timer.elapsed_ms()
    d2d_bandwidth = (2 * size_bytes * iterations) / (d2d_time_ms / 1000.0) / 1e9
    results["d2d_bandwidth_gbps"] = d2d_bandwidth

    # Device to Host
    device_data = cp.random.randn(size_bytes // 4, dtype=cp.float32)

    timer.start()
    for _ in range(iterations):
        host_result = cp.asnumpy(device_data)
        cp.cuda.Stream.null.synchronize()
    timer.stop()

    d2h_time_ms = timer.elapsed_ms()
    d2h_bandwidth = (size_bytes * iterations) / (d2h_time_ms / 1000.0) / 1e9
    results["d2h_bandwidth_gbps"] = d2h_bandwidth

    return results


def get_gpu_memory_usage() -> MemoryStats:
    """Get current GPU memory usage."""
    if not HAS_CUPY:
        return MemoryStats()

    mempool = cp.get_default_memory_pool()
    return MemoryStats(
        peak_bytes=mempool.total_bytes(),
        allocated_bytes=mempool.used_bytes(),
        reserved_bytes=mempool.total_bytes(),
    )


# ============================================================================
# Model Benchmark Utilities
# ============================================================================


@dataclass
class ModelBenchmarkConfig:
    """Configuration for model benchmarking."""

    batch_size: int = 1
    warmup_runs: int = 5
    benchmark_runs: int = 100
    use_gpu: bool = True
    precision: str = "fp32"  # fp32, fp16, int8


def benchmark_model_inference(
    model_fn: Callable,
    input_data: Any,
    config: ModelBenchmarkConfig | None = None,
) -> BenchmarkResult:
    """
    Benchmark model inference.

    Args:
        model_fn: Model inference function
        input_data: Input data for the model
        config: Benchmark configuration

    Returns:
        BenchmarkResult with detailed metrics
    """
    config = config or ModelBenchmarkConfig()

    result = benchmark_function(
        model_fn,
        args=(input_data,),
        warmup_runs=config.warmup_runs,
        benchmark_runs=config.benchmark_runs,
        use_gpu_timer=config.use_gpu and HAS_CUPY,
    )

    # Update throughput with batch info
    result.throughput.batch_size = config.batch_size
    result.throughput.samples_per_sec = (
        result.throughput.batches_per_sec * config.batch_size
    )

    # Get memory stats
    result.memory = get_gpu_memory_usage()

    # Add metadata
    result.metadata = {
        "batch_size": config.batch_size,
        "precision": config.precision,
        "use_gpu": config.use_gpu,
    }

    return result


def compare_models(
    models: dict[str, Callable],
    input_data: Any,
    config: ModelBenchmarkConfig | None = None,
) -> dict[str, BenchmarkResult]:
    """
    Compare multiple models on same input.

    Args:
        models: Dictionary of model name -> model function
        input_data: Input data for models
        config: Benchmark configuration

    Returns:
        Dictionary of model name -> BenchmarkResult
    """
    results = {}
    for name, model_fn in models.items():
        result = benchmark_model_inference(model_fn, input_data, config)
        result.name = name
        results[name] = result
    return results


def format_comparison_table(results: dict[str, BenchmarkResult]) -> str:
    """Format comparison results as a table."""
    if not results:
        return "No results to display"

    lines = [
        "| Model | Latency (ms) | Throughput (samples/s) | Memory (MB) |",
        "|-------|--------------|------------------------|-------------|",
    ]

    for name, result in results.items():
        lines.append(
            f"| {name} | {result.latency.mean_ms:.2f} "
            f"| {result.throughput.samples_per_sec:.1f} "
            f"| {result.memory.peak_mb:.1f} |"
        )

    return "\n".join(lines)
