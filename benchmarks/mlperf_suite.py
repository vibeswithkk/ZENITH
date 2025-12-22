# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
MLPerf-Style Benchmark Suite for Zenith

Implements industry-standard benchmarking methodology inspired by MLPerf.
Provides consistent, reproducible performance measurements across scenarios.

References:
- MLPerf Inference Rules v3.1 (mlcommons.org)
- MLPerf Inference Paper (MLSys 2020)

Scenarios:
- single-stream: Measure per-query latency (interactive workloads)
- offline: Measure throughput (batch processing)
- server: Measure latency under load (production serving)
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum
import time
import logging
import numpy as np

logger = logging.getLogger("zenith.benchmarks")


class BenchmarkScenario(Enum):
    """MLPerf benchmark scenarios."""

    SINGLE_STREAM = "single-stream"
    OFFLINE = "offline"
    SERVER = "server"


class PrecisionMode(Enum):
    """Precision modes for benchmarking."""

    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"


@dataclass
class BenchmarkConfig:
    """
    MLPerf-inspired benchmark configuration.

    Attributes:
        model_name: Name of the model being benchmarked.
        batch_sizes: List of batch sizes to test.
        sequence_lengths: List of sequence lengths (for transformers).
        num_warmup: Number of warmup iterations (excluded from timing).
        num_runs: Number of timed iterations.
        quality_target: Minimum accuracy relative to reference (0.0-1.0).
        scenario: Benchmark scenario type.
        precision: Precision mode for the benchmark.
        target_latency_ms: Target latency for server scenario (ms).
        target_qps: Target queries per second for server scenario.
    """

    model_name: str
    batch_sizes: list = field(default_factory=lambda: [1, 4, 8, 16])
    sequence_lengths: list = field(default_factory=lambda: [128])
    num_warmup: int = 10
    num_runs: int = 100
    quality_target: float = 0.99
    scenario: str = "single-stream"
    precision: str = "fp32"
    target_latency_ms: float = 100.0
    target_qps: float = 10.0

    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.model_name:
            raise ValueError("model_name is required")
        if self.num_warmup < 0:
            raise ValueError("num_warmup must be non-negative")
        if self.num_runs < 1:
            raise ValueError("num_runs must be at least 1")
        if not 0.0 <= self.quality_target <= 1.0:
            raise ValueError("quality_target must be between 0.0 and 1.0")
        if self.scenario not in ["single-stream", "offline", "server"]:
            raise ValueError(
                f"Invalid scenario: {self.scenario}. "
                "Must be one of: single-stream, offline, server"
            )
        if self.precision not in ["fp32", "fp16", "int8"]:
            raise ValueError(
                f"Invalid precision: {self.precision}. Must be one of: fp32, fp16, int8"
            )


@dataclass
class BenchmarkResult:
    """
    Benchmark result with MLPerf-style metrics.

    Attributes:
        model_name: Name of the benchmarked model.
        scenario: Benchmark scenario used.
        batch_size: Batch size used.
        sequence_length: Sequence length (for transformers).
        precision: Precision mode used.
        latency_mean_ms: Mean latency in milliseconds.
        latency_p50_ms: 50th percentile latency (median).
        latency_p90_ms: 90th percentile latency.
        latency_p99_ms: 99th percentile latency.
        latency_min_ms: Minimum latency.
        latency_max_ms: Maximum latency.
        latency_std_ms: Standard deviation of latency.
        throughput_qps: Throughput in queries per second.
        throughput_samples_per_sec: Throughput in samples per second.
        quality_score: Accuracy relative to reference.
        quality_passed: Whether quality target was met.
        total_samples: Total number of samples processed.
        total_time_sec: Total benchmark time in seconds.
        warmup_time_sec: Warmup time in seconds.
    """

    model_name: str
    scenario: str
    batch_size: int
    sequence_length: int = 0
    precision: str = "fp32"
    latency_mean_ms: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p90_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_min_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_std_ms: float = 0.0
    throughput_qps: float = 0.0
    throughput_samples_per_sec: float = 0.0
    quality_score: float = 0.0
    quality_passed: bool = False
    total_samples: int = 0
    total_time_sec: float = 0.0
    warmup_time_sec: float = 0.0

    def to_dict(self) -> dict:
        """Convert result to dictionary for serialization."""
        return {
            "model_name": self.model_name,
            "scenario": self.scenario,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "precision": self.precision,
            "latency": {
                "mean_ms": round(self.latency_mean_ms, 3),
                "p50_ms": round(self.latency_p50_ms, 3),
                "p90_ms": round(self.latency_p90_ms, 3),
                "p99_ms": round(self.latency_p99_ms, 3),
                "min_ms": round(self.latency_min_ms, 3),
                "max_ms": round(self.latency_max_ms, 3),
                "std_ms": round(self.latency_std_ms, 3),
            },
            "throughput": {
                "qps": round(self.throughput_qps, 2),
                "samples_per_sec": round(self.throughput_samples_per_sec, 2),
            },
            "quality": {
                "score": round(self.quality_score, 4),
                "passed": self.quality_passed,
            },
            "timing": {
                "total_samples": self.total_samples,
                "total_time_sec": round(self.total_time_sec, 3),
                "warmup_time_sec": round(self.warmup_time_sec, 3),
            },
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Benchmark Result: {self.model_name}",
            f"  Scenario: {self.scenario}",
            f"  Batch Size: {self.batch_size}",
            f"  Precision: {self.precision}",
            f"  Latency (ms): mean={self.latency_mean_ms:.2f}, "
            f"p50={self.latency_p50_ms:.2f}, "
            f"p90={self.latency_p90_ms:.2f}, "
            f"p99={self.latency_p99_ms:.2f}",
            f"  Throughput: {self.throughput_qps:.2f} QPS "
            f"({self.throughput_samples_per_sec:.2f} samples/sec)",
            f"  Quality: {self.quality_score:.4f} "
            f"({'PASS' if self.quality_passed else 'FAIL'})",
        ]
        return "\n".join(lines)


class ZenithBenchmark:
    """
    MLPerf-inspired benchmark suite for Zenith.

    Provides standardized benchmarking methodology for measuring:
    - Latency (P50, P90, P99)
    - Throughput (samples/second, queries/second)
    - Quality (accuracy vs reference)

    Example:
        benchmark = ZenithBenchmark()
        config = BenchmarkConfig(
            model_name="bert-base",
            batch_sizes=[1, 8],
            scenario="single-stream"
        )
        results = benchmark.run(config, model_fn, input_generator, reference_fn)
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize benchmark suite.

        Args:
            device: Device to run benchmarks on ("cpu", "cuda").
        """
        self._device = device
        self._cuda_available = self._check_cuda()

    def _check_cuda(self) -> bool:
        """Check if CUDA is available for synchronization."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def _synchronize(self) -> None:
        """Synchronize CUDA if available."""
        if self._cuda_available and self._device == "cuda":
            try:
                import torch

                torch.cuda.synchronize()
            except Exception:
                pass

    def run(
        self,
        config: BenchmarkConfig,
        model_fn: Callable,
        input_generator: Callable,
        reference_fn: Optional[Callable] = None,
    ) -> list:
        """
        Run benchmark suite with given configuration.

        Args:
            config: Benchmark configuration.
            model_fn: Model function to benchmark (input -> output).
            input_generator: Function that generates input (batch_size, seq_len -> input).
            reference_fn: Optional reference function for quality verification.

        Returns:
            List of BenchmarkResult for each batch_size/seq_len combination.
        """
        config.validate()
        results = []

        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                logger.info(
                    f"Running {config.scenario} benchmark: "
                    f"batch={batch_size}, seq={seq_len}"
                )

                if config.scenario == "single-stream":
                    result = self._run_single_stream(
                        config,
                        model_fn,
                        input_generator,
                        batch_size,
                        seq_len,
                        reference_fn,
                    )
                elif config.scenario == "offline":
                    result = self._run_offline(
                        config,
                        model_fn,
                        input_generator,
                        batch_size,
                        seq_len,
                        reference_fn,
                    )
                elif config.scenario == "server":
                    result = self._run_server(
                        config,
                        model_fn,
                        input_generator,
                        batch_size,
                        seq_len,
                        reference_fn,
                    )
                else:
                    raise ValueError(f"Unknown scenario: {config.scenario}")

                results.append(result)
                logger.info(f"  Latency P50: {result.latency_p50_ms:.2f} ms")
                logger.info(f"  Throughput: {result.throughput_qps:.2f} QPS")

        return results

    def _run_single_stream(
        self,
        config: BenchmarkConfig,
        model_fn: Callable,
        input_generator: Callable,
        batch_size: int,
        seq_len: int,
        reference_fn: Optional[Callable],
    ) -> BenchmarkResult:
        """
        Run single-stream scenario.

        Measures per-query latency for interactive workloads.
        Each query is processed sequentially (one at a time).
        """
        sample_input = input_generator(batch_size, seq_len)

        # Warmup phase
        warmup_start = time.perf_counter()
        for _ in range(config.num_warmup):
            self._synchronize()
            _ = model_fn(sample_input)
            self._synchronize()
        warmup_time = time.perf_counter() - warmup_start

        # Timed runs
        latencies = []
        benchmark_start = time.perf_counter()

        for _ in range(config.num_runs):
            input_data = input_generator(batch_size, seq_len)

            self._synchronize()
            start = time.perf_counter()
            output = model_fn(input_data)
            self._synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        total_time = time.perf_counter() - benchmark_start

        # Quality verification
        quality_score = 1.0
        quality_passed = True
        if reference_fn is not None:
            quality_score = self._verify_quality(
                model_fn, reference_fn, input_generator, batch_size, seq_len
            )
            quality_passed = quality_score >= config.quality_target

        # Calculate statistics
        latencies_array = np.array(latencies)

        return BenchmarkResult(
            model_name=config.model_name,
            scenario="single-stream",
            batch_size=batch_size,
            sequence_length=seq_len,
            precision=config.precision,
            latency_mean_ms=float(np.mean(latencies_array)),
            latency_p50_ms=float(np.percentile(latencies_array, 50)),
            latency_p90_ms=float(np.percentile(latencies_array, 90)),
            latency_p99_ms=float(np.percentile(latencies_array, 99)),
            latency_min_ms=float(np.min(latencies_array)),
            latency_max_ms=float(np.max(latencies_array)),
            latency_std_ms=float(np.std(latencies_array)),
            throughput_qps=1000.0 / float(np.mean(latencies_array)),
            throughput_samples_per_sec=(
                batch_size * 1000.0 / float(np.mean(latencies_array))
            ),
            quality_score=quality_score,
            quality_passed=quality_passed,
            total_samples=config.num_runs * batch_size,
            total_time_sec=total_time,
            warmup_time_sec=warmup_time,
        )

    def _run_offline(
        self,
        config: BenchmarkConfig,
        model_fn: Callable,
        input_generator: Callable,
        batch_size: int,
        seq_len: int,
        reference_fn: Optional[Callable],
    ) -> BenchmarkResult:
        """
        Run offline scenario.

        Measures maximum throughput for batch processing.
        All samples are available at once; optimize for throughput.
        """
        # Generate all inputs upfront
        all_inputs = [
            input_generator(batch_size, seq_len) for _ in range(config.num_runs)
        ]

        # Warmup
        warmup_start = time.perf_counter()
        for i in range(min(config.num_warmup, len(all_inputs))):
            self._synchronize()
            _ = model_fn(all_inputs[i])
            self._synchronize()
        warmup_time = time.perf_counter() - warmup_start

        # Timed batch processing
        self._synchronize()
        benchmark_start = time.perf_counter()

        for input_data in all_inputs:
            _ = model_fn(input_data)

        self._synchronize()
        total_time = time.perf_counter() - benchmark_start

        # Calculate throughput
        total_samples = config.num_runs * batch_size
        throughput_samples_per_sec = total_samples / total_time
        throughput_qps = config.num_runs / total_time

        # Quality verification
        quality_score = 1.0
        quality_passed = True
        if reference_fn is not None:
            quality_score = self._verify_quality(
                model_fn, reference_fn, input_generator, batch_size, seq_len
            )
            quality_passed = quality_score >= config.quality_target

        # For offline, latency is total time / num queries
        avg_latency_ms = (total_time / config.num_runs) * 1000

        return BenchmarkResult(
            model_name=config.model_name,
            scenario="offline",
            batch_size=batch_size,
            sequence_length=seq_len,
            precision=config.precision,
            latency_mean_ms=avg_latency_ms,
            latency_p50_ms=avg_latency_ms,
            latency_p90_ms=avg_latency_ms,
            latency_p99_ms=avg_latency_ms,
            latency_min_ms=avg_latency_ms,
            latency_max_ms=avg_latency_ms,
            latency_std_ms=0.0,
            throughput_qps=throughput_qps,
            throughput_samples_per_sec=throughput_samples_per_sec,
            quality_score=quality_score,
            quality_passed=quality_passed,
            total_samples=total_samples,
            total_time_sec=total_time,
            warmup_time_sec=warmup_time,
        )

    def _run_server(
        self,
        config: BenchmarkConfig,
        model_fn: Callable,
        input_generator: Callable,
        batch_size: int,
        seq_len: int,
        reference_fn: Optional[Callable],
    ) -> BenchmarkResult:
        """
        Run server scenario.

        Measures latency under sustained load.
        Simulates production serving with arrival rate.
        """
        # Calculate inter-arrival time for target QPS
        inter_arrival_ms = 1000.0 / config.target_qps

        # Warmup
        sample_input = input_generator(batch_size, seq_len)
        warmup_start = time.perf_counter()
        for _ in range(config.num_warmup):
            self._synchronize()
            _ = model_fn(sample_input)
            self._synchronize()
        warmup_time = time.perf_counter() - warmup_start

        # Simulate server workload
        latencies = []
        benchmark_start = time.perf_counter()
        next_arrival = benchmark_start

        for _ in range(config.num_runs):
            # Wait until next arrival time (simulating request arrival)
            now = time.perf_counter()
            if now < next_arrival:
                time.sleep(next_arrival - now)

            input_data = input_generator(batch_size, seq_len)

            self._synchronize()
            start = time.perf_counter()
            _ = model_fn(input_data)
            self._synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)
            next_arrival = start + (inter_arrival_ms / 1000.0)

        total_time = time.perf_counter() - benchmark_start

        # Quality verification
        quality_score = 1.0
        quality_passed = True
        if reference_fn is not None:
            quality_score = self._verify_quality(
                model_fn, reference_fn, input_generator, batch_size, seq_len
            )
            quality_passed = quality_score >= config.quality_target

        # Calculate statistics
        latencies_array = np.array(latencies)
        achieved_qps = config.num_runs / total_time

        # Check if latency target was met
        latency_p99 = float(np.percentile(latencies_array, 99))
        latency_target_met = latency_p99 <= config.target_latency_ms

        return BenchmarkResult(
            model_name=config.model_name,
            scenario="server",
            batch_size=batch_size,
            sequence_length=seq_len,
            precision=config.precision,
            latency_mean_ms=float(np.mean(latencies_array)),
            latency_p50_ms=float(np.percentile(latencies_array, 50)),
            latency_p90_ms=float(np.percentile(latencies_array, 90)),
            latency_p99_ms=latency_p99,
            latency_min_ms=float(np.min(latencies_array)),
            latency_max_ms=float(np.max(latencies_array)),
            latency_std_ms=float(np.std(latencies_array)),
            throughput_qps=achieved_qps,
            throughput_samples_per_sec=achieved_qps * batch_size,
            quality_score=quality_score,
            quality_passed=quality_passed and latency_target_met,
            total_samples=config.num_runs * batch_size,
            total_time_sec=total_time,
            warmup_time_sec=warmup_time,
        )

    def _verify_quality(
        self,
        model_fn: Callable,
        reference_fn: Callable,
        input_generator: Callable,
        batch_size: int,
        seq_len: int,
        num_samples: int = 10,
    ) -> float:
        """
        Verify output quality against reference.

        Returns accuracy score between 0.0 and 1.0.
        """
        matches = 0
        total = 0

        for _ in range(num_samples):
            input_data = input_generator(batch_size, seq_len)

            model_output = model_fn(input_data)
            reference_output = reference_fn(input_data)

            # Convert to numpy for comparison
            if hasattr(model_output, "numpy"):
                model_output = model_output.numpy()
            if hasattr(model_output, "cpu"):
                model_output = model_output.cpu().numpy()
            if hasattr(reference_output, "numpy"):
                reference_output = reference_output.numpy()
            if hasattr(reference_output, "cpu"):
                reference_output = reference_output.cpu().numpy()

            model_np = np.asarray(model_output)
            ref_np = np.asarray(reference_output)

            # Check if outputs are close
            if np.allclose(model_np, ref_np, rtol=1e-3, atol=1e-5):
                matches += 1
            total += 1

        return matches / total if total > 0 else 0.0


def generate_results_table(results: list) -> str:
    """
    Generate markdown table from benchmark results.

    Args:
        results: List of BenchmarkResult objects.

    Returns:
        Markdown formatted table string.
    """
    lines = [
        "| Model | Scenario | Batch | Seq | Precision | "
        "P50 (ms) | P90 (ms) | P99 (ms) | QPS | Quality |",
        "|-------|----------|-------|-----|-----------|"
        "---------|---------|---------|-----|---------|",
    ]

    for r in results:
        quality_str = "PASS" if r.quality_passed else "FAIL"
        line = (
            f"| {r.model_name} | {r.scenario} | {r.batch_size} | "
            f"{r.sequence_length} | {r.precision} | "
            f"{r.latency_p50_ms:.2f} | {r.latency_p90_ms:.2f} | "
            f"{r.latency_p99_ms:.2f} | {r.throughput_qps:.1f} | "
            f"{quality_str} |"
        )
        lines.append(line)

    return "\n".join(lines)


def compare_results(
    zenith_results: list,
    baseline_results: list,
) -> str:
    """
    Generate comparison table between Zenith and baseline.

    Args:
        zenith_results: Results from Zenith benchmark.
        baseline_results: Results from baseline (e.g., PyTorch).

    Returns:
        Markdown formatted comparison table.
    """
    lines = [
        "| Model | Batch | Zenith P50 | Baseline P50 | Speedup | "
        "Zenith QPS | Baseline QPS |",
        "|-------|-------|------------|--------------|---------|"
        "-----------|--------------|",
    ]

    for z, b in zip(zenith_results, baseline_results):
        speedup = b.latency_p50_ms / z.latency_p50_ms if z.latency_p50_ms > 0 else 0
        line = (
            f"| {z.model_name} | {z.batch_size} | "
            f"{z.latency_p50_ms:.2f} ms | {b.latency_p50_ms:.2f} ms | "
            f"{speedup:.2f}x | {z.throughput_qps:.1f} | {b.throughput_qps:.1f} |"
        )
        lines.append(line)

    return "\n".join(lines)
