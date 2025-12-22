# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Tests for Zenith Observability Module

Validates:
- Verbosity enum
- LogEntry serialization
- ZenithLogger singleton
- MetricsCollector statistics
- Verbosity control functions
"""

import io
import json
import pytest
import numpy as np

from zenith.observability import (
    Verbosity,
    LogEntry,
    ZenithLogger,
    get_logger,
    set_verbosity,
    InferenceMetrics,
    MetricsCollector,
    get_metrics_collector,
)


class TestVerbosity:
    """Tests for Verbosity enum."""

    def test_verbosity_values(self):
        """Test verbosity level values."""
        assert Verbosity.SILENT == 0
        assert Verbosity.ERROR == 1
        assert Verbosity.WARNING == 2
        assert Verbosity.INFO == 3
        assert Verbosity.DEBUG == 4

    def test_verbosity_comparison(self):
        """Test verbosity level comparison."""
        assert Verbosity.DEBUG > Verbosity.INFO
        assert Verbosity.INFO > Verbosity.WARNING
        assert Verbosity.WARNING > Verbosity.ERROR
        assert Verbosity.ERROR > Verbosity.SILENT


class TestLogEntry:
    """Tests for LogEntry dataclass."""

    def test_log_entry_creation(self):
        """Test LogEntry creation."""
        entry = LogEntry(
            level="INFO",
            message="Test message",
            timestamp="2024-12-22T00:00:00",
            component="test",
        )
        assert entry.level == "INFO"
        assert entry.message == "Test message"
        assert entry.component == "test"

    def test_log_entry_to_json(self):
        """Test JSON serialization."""
        entry = LogEntry(
            level="DEBUG",
            message="Debug message",
            timestamp="2024-12-22T00:00:00",
            component="runtime",
            duration_ms=10.5,
        )
        json_str = entry.to_json()
        data = json.loads(json_str)

        assert data["level"] == "DEBUG"
        assert data["message"] == "Debug message"
        assert data["duration_ms"] == 10.5

    def test_log_entry_to_text(self):
        """Test text format."""
        entry = LogEntry(
            level="INFO",
            message="Processing",
            timestamp="2024-12-22T00:00:00",
            component="compiler",
            duration_ms=25.3,
        )
        text = entry.to_text()

        assert "[INFO]" in text
        assert "[compiler]" in text
        assert "Processing" in text
        assert "25.30ms" in text


class TestZenithLogger:
    """Tests for ZenithLogger class."""

    def setup_method(self):
        """Reset logger before each test."""
        ZenithLogger.reset()

    def test_singleton_pattern(self):
        """Test singleton pattern."""
        logger1 = ZenithLogger.get()
        logger2 = ZenithLogger.get()
        assert logger1 is logger2

    def test_default_verbosity(self):
        """Test default verbosity level."""
        logger = ZenithLogger.get()
        assert logger.get_verbosity() == Verbosity.INFO

    def test_set_verbosity(self):
        """Test setting verbosity."""
        logger = ZenithLogger.get()
        logger.set_verbosity(Verbosity.DEBUG)
        assert logger.get_verbosity() == Verbosity.DEBUG

        logger.set_verbosity(0)
        assert logger.get_verbosity() == Verbosity.SILENT

    def test_info_logging(self):
        """Test info level logging."""
        logger = ZenithLogger.get()
        output = io.StringIO()
        logger.set_output(output)
        logger.set_verbosity(Verbosity.INFO)

        logger.info("Test info", component="test")

        content = output.getvalue()
        assert "[INFO]" in content
        assert "[test]" in content
        assert "Test info" in content

    def test_debug_logging_when_enabled(self):
        """Test debug logging when enabled."""
        logger = ZenithLogger.get()
        output = io.StringIO()
        logger.set_output(output)
        logger.set_verbosity(Verbosity.DEBUG)

        logger.debug("Debug message", component="runtime")

        content = output.getvalue()
        assert "[DEBUG]" in content

    def test_debug_logging_when_disabled(self):
        """Test debug logging is suppressed at INFO level."""
        logger = ZenithLogger.get()
        output = io.StringIO()
        logger.set_output(output)
        logger.set_verbosity(Verbosity.INFO)

        logger.debug("Should be suppressed", component="test")

        content = output.getvalue()
        assert content == ""

    def test_json_format(self):
        """Test JSON output format."""
        logger = ZenithLogger.get()
        output = io.StringIO()
        logger.set_output(output)
        logger.set_json_format(True)
        logger.set_verbosity(Verbosity.INFO)

        logger.info("JSON test", component="test")

        content = output.getvalue().strip()
        data = json.loads(content)
        assert data["level"] == "INFO"
        assert data["message"] == "JSON test"


class TestGetLogger:
    """Tests for get_logger function."""

    def setup_method(self):
        """Reset logger before each test."""
        ZenithLogger.reset()

    def test_get_logger(self):
        """Test get_logger returns singleton."""
        logger = get_logger()
        assert isinstance(logger, ZenithLogger)
        assert logger is ZenithLogger.get()


class TestSetVerbosity:
    """Tests for set_verbosity function."""

    def setup_method(self):
        """Reset logger before each test."""
        ZenithLogger.reset()

    def test_set_verbosity_function(self):
        """Test global set_verbosity function."""
        set_verbosity(4)
        assert ZenithLogger.get().get_verbosity() == Verbosity.DEBUG

        set_verbosity(0)
        assert ZenithLogger.get().get_verbosity() == Verbosity.SILENT


class TestInferenceMetrics:
    """Tests for InferenceMetrics dataclass."""

    def test_metrics_creation(self):
        """Test InferenceMetrics creation."""
        metrics = InferenceMetrics(
            latency_ms=10.5,
            memory_mb=256.0,
            kernel_calls=12,
            batch_size=8,
        )
        assert metrics.latency_ms == 10.5
        assert metrics.memory_mb == 256.0
        assert metrics.kernel_calls == 12
        assert metrics.batch_size == 8

    def test_default_values(self):
        """Test default values."""
        metrics = InferenceMetrics(latency_ms=5.0)
        assert metrics.memory_mb == 0.0
        assert metrics.kernel_calls == 0
        assert metrics.batch_size == 1


class TestMetricsCollector:
    """Tests for MetricsCollector class."""

    def test_empty_collector(self):
        """Test empty collector."""
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["total_inferences"] == 0

    def test_record_inference(self):
        """Test recording inferences."""
        collector = MetricsCollector()
        collector.record_inference(InferenceMetrics(latency_ms=10.0))
        collector.record_inference(InferenceMetrics(latency_ms=20.0))

        summary = collector.get_summary()
        assert summary["total_inferences"] == 2
        assert summary["latency_mean_ms"] == 15.0

    def test_latency_percentiles(self):
        """Test latency percentile calculation."""
        collector = MetricsCollector()

        for i in range(100):
            collector.record_inference(InferenceMetrics(latency_ms=float(i)))

        summary = collector.get_summary()
        assert 45 <= summary["latency_p50_ms"] <= 55
        assert 85 <= summary["latency_p90_ms"] <= 95
        assert 95 <= summary["latency_p99_ms"] <= 100

    def test_error_counting(self):
        """Test error counting."""
        collector = MetricsCollector()
        collector.record_inference(InferenceMetrics(latency_ms=10.0))
        collector.record_error()
        collector.record_error()

        summary = collector.get_summary()
        assert summary["error_count"] == 2

    def test_reset(self):
        """Test reset."""
        collector = MetricsCollector()
        collector.record_inference(InferenceMetrics(latency_ms=10.0))
        collector.reset()

        summary = collector.get_summary()
        assert summary["total_inferences"] == 0

    def test_prometheus_export(self):
        """Test Prometheus export format."""
        collector = MetricsCollector()
        collector.record_inference(InferenceMetrics(latency_ms=10.0))

        prom = collector.export_prometheus()
        assert "zenith_inference_total" in prom
        assert "zenith_inference_latency_ms" in prom

    def test_histogram(self):
        """Test latency histogram."""
        collector = MetricsCollector()
        for i in range(50):
            collector.record_inference(InferenceMetrics(latency_ms=float(i)))

        edges, counts = collector.get_latency_histogram(num_buckets=5)
        assert len(edges) == 6
        assert len(counts) == 5
        assert sum(counts) == 50


class TestGlobalMetricsCollector:
    """Tests for global metrics collector."""

    def test_get_metrics_collector(self):
        """Test global collector singleton."""
        from zenith.observability.metrics import reset_metrics_collector

        reset_metrics_collector()
        collector1 = get_metrics_collector()
        collector2 = get_metrics_collector()
        assert collector1 is collector2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
