# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Prometheus Metrics Exporter for Zenith

Exports metrics in Prometheus format with proper labels and types.
"""

from typing import Optional
from zenith.observability import get_metrics_collector, ZenithLogger

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
    )

    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False


class PrometheusExporter:
    """
    Prometheus metrics exporter.

    Exports Zenith metrics in Prometheus format.
    """

    def __init__(self, registry: Optional["CollectorRegistry"] = None):
        """Initialize exporter."""
        if not HAS_PROMETHEUS:
            raise ImportError(
                "prometheus_client not installed. "
                "Install with: pip install prometheus-client"
            )

        self._registry = registry or CollectorRegistry()
        self._setup_metrics()
        self._logger = ZenithLogger.get()

    def _setup_metrics(self) -> None:
        """Setup Prometheus metrics."""
        self.inference_total = Counter(
            "zenith_inference_total",
            "Total number of inferences",
            ["model", "precision"],
            registry=self._registry,
        )

        self.inference_errors = Counter(
            "zenith_inference_errors_total",
            "Total inference errors",
            ["model", "error_type"],
            registry=self._registry,
        )

        self.inference_latency = Histogram(
            "zenith_inference_latency_seconds",
            "Inference latency in seconds",
            ["model", "precision"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=self._registry,
        )

        self.memory_usage = Gauge(
            "zenith_memory_mb",
            "Memory usage in megabytes",
            ["device"],
            registry=self._registry,
        )

        self.active_models = Gauge(
            "zenith_active_models",
            "Number of active compiled models",
            registry=self._registry,
        )

        self.compilation_time = Summary(
            "zenith_compilation_seconds",
            "Model compilation time",
            ["model"],
            registry=self._registry,
        )

    def record_inference(
        self,
        model: str = "default",
        precision: str = "fp32",
        latency_ms: float = 0.0,
    ) -> None:
        """Record an inference."""
        self.inference_total.labels(model=model, precision=precision).inc()
        self.inference_latency.labels(model=model, precision=precision).observe(
            latency_ms / 1000.0
        )

    def record_error(
        self,
        model: str = "default",
        error_type: str = "unknown",
    ) -> None:
        """Record an error."""
        self.inference_errors.labels(model=model, error_type=error_type).inc()

    def set_memory(self, device: str, memory_mb: float) -> None:
        """Set memory usage."""
        self.memory_usage.labels(device=device).set(memory_mb)

    def record_compilation(self, model: str, time_seconds: float) -> None:
        """Record compilation time."""
        self.compilation_time.labels(model=model).observe(time_seconds)

    def generate(self) -> bytes:
        """Generate Prometheus metrics output."""
        return generate_latest(self._registry)

    def content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


_global_exporter: Optional[PrometheusExporter] = None


def get_exporter() -> PrometheusExporter:
    """Get global Prometheus exporter."""
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = PrometheusExporter()
    return _global_exporter


def check_prometheus_available() -> bool:
    """Check if prometheus_client is available."""
    return HAS_PROMETHEUS
