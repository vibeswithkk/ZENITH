# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Metrics Collector for Zenith

Collects and exports inference metrics for monitoring and analysis.

Features:
- Latency histogram
- Summary statistics (p50, p90, p99)
- Optional Prometheus export

Example:
    from zenith.observability import MetricsCollector, InferenceMetrics

    collector = MetricsCollector()
    collector.record_inference(InferenceMetrics(
        latency_ms=10.5,
        memory_mb=256.0,
        kernel_calls=12
    ))
    print(collector.get_summary())
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class InferenceMetrics:
    """
    Metrics for a single inference execution.

    Attributes:
        latency_ms: Inference latency in milliseconds
        memory_mb: Memory usage in megabytes
        kernel_calls: Number of kernel invocations
        batch_size: Batch size used
        model_name: Optional model identifier
    """

    latency_ms: float
    memory_mb: float = 0.0
    kernel_calls: int = 0
    batch_size: int = 1
    model_name: Optional[str] = None


class MetricsCollector:
    """
    Collects and exports metrics for Zenith inference.

    Thread-safe collection of inference metrics with statistical analysis.

    Example:
        collector = MetricsCollector()

        # Record metrics
        collector.record_inference(InferenceMetrics(latency_ms=10.5))

        # Get summary
        summary = collector.get_summary()
        print(f"P99 latency: {summary['latency_p99_ms']:.2f}ms")
    """

    def __init__(self):
        """Initialize metrics collector."""
        self._latency_histogram: list[float] = []
        self._memory_histogram: list[float] = []
        self._kernel_calls_histogram: list[int] = []
        self._inference_count: int = 0
        self._error_count: int = 0
        self._total_samples: int = 0

    def record_inference(self, metrics: InferenceMetrics) -> None:
        """
        Record metrics from a single inference.

        Args:
            metrics: InferenceMetrics for the inference
        """
        self._latency_histogram.append(metrics.latency_ms)
        self._memory_histogram.append(metrics.memory_mb)
        self._kernel_calls_histogram.append(metrics.kernel_calls)
        self._inference_count += 1
        self._total_samples += metrics.batch_size

    def record_error(self) -> None:
        """Record an inference error."""
        self._error_count += 1

    def get_summary(self) -> dict:
        """
        Get summary statistics.

        Returns:
            Dictionary with latency percentiles, throughput, and counts
        """
        if not self._latency_histogram:
            return {
                "total_inferences": 0,
                "total_samples": 0,
                "error_count": 0,
            }

        latencies = np.array(self._latency_histogram)
        memories = np.array(self._memory_histogram)

        return {
            "total_inferences": self._inference_count,
            "total_samples": self._total_samples,
            "error_count": self._error_count,
            "latency_mean_ms": float(np.mean(latencies)),
            "latency_std_ms": float(np.std(latencies)),
            "latency_min_ms": float(np.min(latencies)),
            "latency_max_ms": float(np.max(latencies)),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p90_ms": float(np.percentile(latencies, 90)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "memory_mean_mb": float(np.mean(memories)),
            "memory_max_mb": float(np.max(memories)),
            "throughput_qps": (
                1000.0 / float(np.mean(latencies)) if np.mean(latencies) > 0 else 0.0
            ),
        }

    def get_latency_histogram(
        self, num_buckets: int = 10
    ) -> tuple[list[float], list[int]]:
        """
        Get latency histogram data.

        Args:
            num_buckets: Number of histogram buckets

        Returns:
            Tuple of (bucket_edges, counts)
        """
        if not self._latency_histogram:
            return [], []

        counts, edges = np.histogram(self._latency_histogram, bins=num_buckets)
        return edges.tolist(), counts.tolist()

    def reset(self) -> None:
        """Reset all collected metrics."""
        self._latency_histogram.clear()
        self._memory_histogram.clear()
        self._kernel_calls_histogram.clear()
        self._inference_count = 0
        self._error_count = 0
        self._total_samples = 0

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-formatted metrics string
        """
        summary = self.get_summary()
        if not summary.get("total_inferences"):
            return ""

        lines = [
            "# HELP zenith_inference_total Total number of inferences",
            "# TYPE zenith_inference_total counter",
            f"zenith_inference_total {summary['total_inferences']}",
            "",
            "# HELP zenith_inference_errors_total Total inference errors",
            "# TYPE zenith_inference_errors_total counter",
            f"zenith_inference_errors_total {summary['error_count']}",
            "",
            "# HELP zenith_inference_latency_ms Inference latency",
            "# TYPE zenith_inference_latency_ms summary",
            f'zenith_inference_latency_ms{{quantile="0.5"}} '
            f"{summary['latency_p50_ms']:.3f}",
            f'zenith_inference_latency_ms{{quantile="0.9"}} '
            f"{summary['latency_p90_ms']:.3f}",
            f'zenith_inference_latency_ms{{quantile="0.99"}} '
            f"{summary['latency_p99_ms']:.3f}",
            "",
            "# HELP zenith_memory_mb Memory usage in MB",
            "# TYPE zenith_memory_mb gauge",
            f"zenith_memory_mb {summary['memory_mean_mb']:.3f}",
        ]

        return "\n".join(lines)


_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def reset_metrics_collector() -> None:
    """Reset the global metrics collector."""
    global _global_collector
    _global_collector = None
