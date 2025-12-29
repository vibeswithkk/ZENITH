#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Memory Chaos Testing Module

Simulates memory pressure to test Zenith's fault tolerance:
- Memory allocation stress
- Garbage collection pressure
- Memory limit enforcement
- Leak detection under pressure

Reference:
    Google DiRT methodology
    AWS Fault Injection patterns

Usage:
    pytest tests/chaos/memory_chaos.py -v
"""

import gc
import sys
import threading
import time
import unittest
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MemoryChaosConfig:
    """Configuration for memory chaos injection."""

    allocation_size_mb: float = 100.0
    allocation_count: int = 10
    hold_duration_seconds: float = 1.0
    force_gc: bool = True

    def __post_init__(self):
        if self.allocation_size_mb <= 0:
            raise ValueError("allocation_size_mb must be positive")
        if self.allocation_count <= 0:
            raise ValueError("allocation_count must be positive")
        if self.hold_duration_seconds < 0:
            raise ValueError("hold_duration_seconds must be non-negative")


class MemoryPressureInjector:
    """
    Injects memory pressure for chaos testing.

    Allocates and holds memory to stress the system.
    Thread-safe implementation.
    """

    def __init__(self, config: Optional[MemoryChaosConfig] = None):
        self.config = config or MemoryChaosConfig()
        self._lock = threading.Lock()
        self._allocations: List[bytes] = []
        self._peak_allocation_mb: float = 0.0
        self._allocation_failures: int = 0

    @property
    def stats(self) -> dict:
        """Get injection statistics."""
        with self._lock:
            current_mb = sum(len(a) for a in self._allocations) / (1024 * 1024)
            return {
                "current_allocation_mb": current_mb,
                "peak_allocation_mb": self._peak_allocation_mb,
                "allocation_failures": self._allocation_failures,
                "active_allocations": len(self._allocations),
            }

    def allocate(self, size_mb: Optional[float] = None) -> bool:
        """
        Allocate memory block.

        Args:
            size_mb: Size in MB to allocate (uses config default if None)

        Returns:
            True if allocation succeeded, False otherwise
        """
        size = size_mb or self.config.allocation_size_mb
        size_bytes = int(size * 1024 * 1024)

        try:
            # Allocate memory (using bytes to ensure actual allocation)
            block = bytes(size_bytes)

            with self._lock:
                self._allocations.append(block)
                current_mb = sum(len(a) for a in self._allocations) / (1024 * 1024)
                self._peak_allocation_mb = max(self._peak_allocation_mb, current_mb)

            return True

        except MemoryError:
            with self._lock:
                self._allocation_failures += 1
            return False

    def release_all(self) -> int:
        """
        Release all allocated memory.

        Returns:
            Number of blocks released
        """
        with self._lock:
            count = len(self._allocations)
            self._allocations.clear()

        if self.config.force_gc:
            gc.collect()

        return count

    def release_one(self) -> bool:
        """
        Release one memory block.

        Returns:
            True if a block was released, False if none available
        """
        with self._lock:
            if self._allocations:
                self._allocations.pop()
                return True
            return False

    def apply_pressure(self) -> dict:
        """
        Apply memory pressure according to configuration.

        Allocates multiple blocks and holds them for the configured duration.

        Returns:
            Statistics about the pressure test
        """
        successes = 0
        failures = 0

        # Allocate memory blocks
        for _ in range(self.config.allocation_count):
            if self.allocate():
                successes += 1
            else:
                failures += 1

        # Hold for duration
        if self.config.hold_duration_seconds > 0:
            time.sleep(self.config.hold_duration_seconds)

        # Get stats before release
        peak = self.stats["peak_allocation_mb"]

        # Release all
        released = self.release_all()

        return {
            "allocations_succeeded": successes,
            "allocations_failed": failures,
            "peak_mb": peak,
            "blocks_released": released,
        }


def get_process_memory_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024  # KB to MB on Linux
    except ImportError:
        return 0.0


class TestMemoryChaosConfig(unittest.TestCase):
    """Test memory chaos configuration."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = MemoryChaosConfig(
            allocation_size_mb=50.0,
            allocation_count=5,
            hold_duration_seconds=0.5,
        )
        self.assertEqual(config.allocation_size_mb, 50.0)
        self.assertEqual(config.allocation_count, 5)

    def test_invalid_config(self):
        """Test invalid configuration rejection."""
        with self.assertRaises(ValueError):
            MemoryChaosConfig(allocation_size_mb=-10)

        with self.assertRaises(ValueError):
            MemoryChaosConfig(allocation_count=0)

        with self.assertRaises(ValueError):
            MemoryChaosConfig(hold_duration_seconds=-1)


class TestMemoryPressureInjector(unittest.TestCase):
    """Test memory pressure injector."""

    def test_basic_allocation(self):
        """Test basic memory allocation."""
        config = MemoryChaosConfig(
            allocation_size_mb=1.0,
            allocation_count=1,
        )
        injector = MemoryPressureInjector(config)

        # Allocate
        success = injector.allocate()
        self.assertTrue(success)

        stats = injector.stats
        self.assertEqual(stats["active_allocations"], 1)
        self.assertGreaterEqual(stats["current_allocation_mb"], 0.9)

        # Release
        released = injector.release_all()
        self.assertEqual(released, 1)

        stats = injector.stats
        self.assertEqual(stats["active_allocations"], 0)

    def test_pressure_application(self):
        """Test full pressure application cycle."""
        config = MemoryChaosConfig(
            allocation_size_mb=1.0,
            allocation_count=5,
            hold_duration_seconds=0.1,
        )
        injector = MemoryPressureInjector(config)

        result = injector.apply_pressure()

        self.assertEqual(result["allocations_succeeded"], 5)
        self.assertEqual(result["allocations_failed"], 0)
        self.assertGreaterEqual(result["peak_mb"], 4.5)
        self.assertEqual(result["blocks_released"], 5)

        # Should be clean after
        stats = injector.stats
        self.assertEqual(stats["active_allocations"], 0)

    def test_no_memory_leak(self):
        """Test that no memory leaks occur after release."""
        gc.collect()
        before_objects = len(gc.get_objects())

        config = MemoryChaosConfig(
            allocation_size_mb=1.0,
            allocation_count=10,
            hold_duration_seconds=0.0,
            force_gc=True,
        )
        injector = MemoryPressureInjector(config)

        # Apply pressure multiple times
        for _ in range(3):
            injector.apply_pressure()

        gc.collect()
        after_objects = len(gc.get_objects())

        # Object count should not grow significantly
        delta = after_objects - before_objects
        self.assertLess(delta, 100, f"Possible memory leak: {delta} new objects")


class TestZenithMemoryResilience(unittest.TestCase):
    """Test Zenith's resilience under memory pressure."""

    def test_metrics_collector_under_pressure(self):
        """Test MetricsCollector works under memory pressure."""
        from zenith.observability import MetricsCollector, InferenceMetrics

        config = MemoryChaosConfig(
            allocation_size_mb=10.0,
            allocation_count=5,
            hold_duration_seconds=0.0,
        )
        injector = MemoryPressureInjector(config)

        collector = MetricsCollector()

        # Apply pressure while recording metrics
        for i in range(100):
            # Periodically apply pressure
            if i % 20 == 0:
                injector.apply_pressure()

            collector.record_inference(
                InferenceMetrics(
                    latency_ms=float(i),
                    memory_mb=float(i * 10),
                )
            )

        # Should still be able to get summary
        summary = collector.get_summary()
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary.get("total_inferences", 0), 100)

    def test_recovery_after_pressure(self):
        """Test system recovers after memory pressure."""
        gc.collect()
        baseline_mem = get_process_memory_mb()

        config = MemoryChaosConfig(
            allocation_size_mb=50.0,
            allocation_count=3,
            hold_duration_seconds=0.1,
            force_gc=True,
        )
        injector = MemoryPressureInjector(config)

        # Apply heavy pressure
        injector.apply_pressure()

        # Force cleanup
        gc.collect()
        gc.collect()

        # Memory should return close to baseline
        final_mem = get_process_memory_mb()
        delta = final_mem - baseline_mem

        # Allow some variance but no major leak
        self.assertLess(delta, 100, f"Memory not recovered: +{delta}MB")


def run_memory_chaos_tests():
    """Run all memory chaos tests."""
    print("=" * 60)
    print("  MEMORY CHAOS TESTS")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestMemoryChaosConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryPressureInjector))
    suite.addTests(loader.loadTestsFromTestCase(TestZenithMemoryResilience))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_memory_chaos_tests())
