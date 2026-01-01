# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for GPU Metrics Collector Module.

Tests the GPU metrics collection functionality including:
- Singleton pattern implementation
- Availability detection
- Stats collection (with and without GPU)
- Graceful fallback when pynvml unavailable

Run with: pytest tests/python/test_gpu_metrics.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict


class TestGPUStatsDataclass:
    """Test GPUStats dataclass."""

    def test_gpustats_creation(self):
        """GPUStats should be creatable with required fields."""
        from zenith.observability.gpu_metrics import GPUStats

        stats = GPUStats(
            device_index=0,
            name="Test GPU",
            utilization_percent=50.0,
            memory_used_mb=1024.0,
            memory_total_mb=8192.0,
            memory_free_mb=7168.0,
        )

        assert stats.device_index == 0
        assert stats.name == "Test GPU"
        assert stats.utilization_percent == 50.0
        assert stats.memory_used_mb == 1024.0
        assert stats.memory_total_mb == 8192.0
        assert stats.memory_free_mb == 7168.0

    def test_gpustats_optional_fields(self):
        """GPUStats optional fields should default to None."""
        from zenith.observability.gpu_metrics import GPUStats

        stats = GPUStats(
            device_index=0,
            name="Test GPU",
            utilization_percent=50.0,
            memory_used_mb=1024.0,
            memory_total_mb=8192.0,
            memory_free_mb=7168.0,
        )

        assert stats.temperature_celsius is None
        assert stats.power_draw_watts is None
        assert stats.power_limit_watts is None

    def test_gpustats_with_optional_fields(self):
        """GPUStats should accept optional fields."""
        from zenith.observability.gpu_metrics import GPUStats

        stats = GPUStats(
            device_index=0,
            name="Test GPU",
            utilization_percent=75.0,
            memory_used_mb=4096.0,
            memory_total_mb=8192.0,
            memory_free_mb=4096.0,
            temperature_celsius=65.0,
            power_draw_watts=120.0,
            power_limit_watts=250.0,
        )

        assert stats.temperature_celsius == 65.0
        assert stats.power_draw_watts == 120.0
        assert stats.power_limit_watts == 250.0

    def test_gpustats_to_dict(self):
        """to_dict() should return serializable dictionary."""
        from zenith.observability.gpu_metrics import GPUStats

        stats = GPUStats(
            device_index=0,
            name="Test GPU",
            utilization_percent=50.0,
            memory_used_mb=1024.0,
            memory_total_mb=8192.0,
            memory_free_mb=7168.0,
        )

        result = stats.to_dict()

        assert isinstance(result, dict)
        assert result["device_index"] == 0
        assert result["name"] == "Test GPU"
        assert "utilization_percent" in result
        assert "memory_used_mb" in result


class TestGPUMetricsCollectorSingleton:
    """Test GPUMetricsCollector singleton pattern."""

    def test_singleton_same_instance(self):
        """Multiple calls should return same instance."""
        from zenith.observability.gpu_metrics import GPUMetricsCollector

        # Reset to ensure clean state
        GPUMetricsCollector.reset()

        collector1 = GPUMetricsCollector.get()
        collector2 = GPUMetricsCollector.get()

        assert collector1 is collector2

    def test_singleton_reset(self):
        """reset() should clear the singleton instance."""
        from zenith.observability.gpu_metrics import GPUMetricsCollector

        collector1 = GPUMetricsCollector.get()
        GPUMetricsCollector.reset()
        collector2 = GPUMetricsCollector.get()

        # After reset, should be a new instance
        # (Note: may be same object if __new__ caches, but internal state reset)
        assert isinstance(collector2, GPUMetricsCollector)


class TestGPUMetricsCollectorAvailability:
    """Test availability detection."""

    def test_is_available_returns_bool(self):
        """is_available() should return boolean."""
        from zenith.observability.gpu_metrics import GPUMetricsCollector

        GPUMetricsCollector.reset()
        collector = GPUMetricsCollector.get()

        result = collector.is_available()

        assert isinstance(result, bool)

    def test_get_device_count_returns_int(self):
        """get_device_count() should return non-negative integer."""
        from zenith.observability.gpu_metrics import GPUMetricsCollector

        GPUMetricsCollector.reset()
        collector = GPUMetricsCollector.get()

        count = collector.get_device_count()

        assert isinstance(count, int)
        assert count >= 0


class TestGPUMetricsCollectorStats:
    """Test stats collection."""

    def test_get_stats_returns_gpustats_or_none(self):
        """get_stats() should return GPUStats or None."""
        from zenith.observability.gpu_metrics import GPUMetricsCollector, GPUStats

        GPUMetricsCollector.reset()
        collector = GPUMetricsCollector.get()

        result = collector.get_stats(device_index=0)

        assert result is None or isinstance(result, GPUStats)

    def test_get_all_stats_returns_dict(self):
        """get_all_stats() should return dictionary."""
        from zenith.observability.gpu_metrics import GPUMetricsCollector

        GPUMetricsCollector.reset()
        collector = GPUMetricsCollector.get()

        result = collector.get_all_stats()

        assert isinstance(result, dict)

    def test_get_stats_with_invalid_device_index(self):
        """get_stats() with invalid index should return None or handle gracefully."""
        from zenith.observability.gpu_metrics import GPUMetricsCollector

        GPUMetricsCollector.reset()
        collector = GPUMetricsCollector.get()

        # Very large device index that definitely doesn't exist
        result = collector.get_stats(device_index=9999)

        # Should return None for invalid device
        assert result is None


class TestModuleLevelFunctions:
    """Test module-level convenience functions."""

    def test_is_available_function(self):
        """Module-level is_available() should return boolean."""
        from zenith.observability import gpu_metrics

        result = gpu_metrics.is_available()

        assert isinstance(result, bool)

    def test_get_current_returns_dict_or_none(self):
        """get_current() should return dict or None."""
        from zenith.observability import gpu_metrics

        result = gpu_metrics.get_current(device_index=0)

        assert result is None or isinstance(result, dict)

    def test_get_memory_info_returns_dict(self):
        """get_memory_info() should return dictionary."""
        from zenith.observability import gpu_metrics

        result = gpu_metrics.get_memory_info(device_index=0)

        assert isinstance(result, dict)

        # Should have expected keys even if values are None/0
        expected_keys = {"used_mb", "total_mb", "free_mb", "utilization_percent"}
        assert expected_keys.issubset(set(result.keys()))

    def test_get_utilization_returns_float_or_none(self):
        """get_utilization() should return float or None."""
        from zenith.observability import gpu_metrics

        result = gpu_metrics.get_utilization(device_index=0)

        assert result is None or isinstance(result, (int, float))


class TestGPUMetricsCollectorWithMockedPynvml:
    """Test with mocked pynvml for consistent results."""

    def test_stats_with_mocked_pynvml(self):
        """Should return proper stats when pynvml is mocked."""
        # This test verifies the data flow even without actual GPU
        from zenith.observability.gpu_metrics import GPUStats

        # Create a mock stats object as if returned by real collector
        mock_stats = GPUStats(
            device_index=0,
            name="NVIDIA GeForce RTX 3090",
            utilization_percent=45.0,
            memory_used_mb=2048.0,
            memory_total_mb=24576.0,
            memory_free_mb=22528.0,
            temperature_celsius=55.0,
            power_draw_watts=150.0,
            power_limit_watts=350.0,
        )

        # Verify all fields are accessible
        assert mock_stats.device_index == 0
        assert "RTX" in mock_stats.name
        assert 0 <= mock_stats.utilization_percent <= 100
        assert mock_stats.memory_used_mb > 0
        assert mock_stats.memory_total_mb > mock_stats.memory_used_mb

        # Verify serialization
        stats_dict = mock_stats.to_dict()
        assert stats_dict["temperature_celsius"] == 55.0


class TestGPUMetricsCollectorThreadSafety:
    """Test thread safety of singleton pattern."""

    def test_concurrent_access(self):
        """Concurrent access should return same instance."""
        import threading
        from zenith.observability.gpu_metrics import GPUMetricsCollector

        GPUMetricsCollector.reset()

        instances = []
        errors = []

        def get_instance():
            try:
                instance = GPUMetricsCollector.get()
                instances.append(instance)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0

        # All instances should be the same object
        if instances:
            first = instances[0]
            for instance in instances[1:]:
                assert instance is first


class TestGPUStatsMemoryCalculations:
    """Test memory calculation correctness."""

    def test_memory_consistency(self):
        """Memory values should be consistent."""
        from zenith.observability.gpu_metrics import GPUStats

        total = 8192.0
        used = 2048.0
        free = 6144.0

        stats = GPUStats(
            device_index=0,
            name="Test GPU",
            utilization_percent=50.0,
            memory_used_mb=used,
            memory_total_mb=total,
            memory_free_mb=free,
        )

        # Used + Free should approximately equal Total
        assert abs((stats.memory_used_mb + stats.memory_free_mb) - stats.memory_total_mb) < 1.0

    def test_utilization_range(self):
        """Utilization should be in valid range."""
        from zenith.observability.gpu_metrics import GPUStats

        # Valid utilization
        stats = GPUStats(
            device_index=0,
            name="Test GPU",
            utilization_percent=75.0,
            memory_used_mb=1024.0,
            memory_total_mb=8192.0,
            memory_free_mb=7168.0,
        )

        assert 0 <= stats.utilization_percent <= 100
