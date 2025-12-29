#!/usr/bin/env python3
# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
GPU Out-of-Memory Chaos Testing Module

Simulates GPU memory exhaustion to test Zenith's fault tolerance:
- GPU OOM simulation
- Recovery verification
- CPU fallback validation
- Memory cleanup verification

Reference:
    NVIDIA CUDA Best Practices
    PyTorch Memory Management

Usage:
    pytest tests/chaos/gpu_chaos.py -v

Note:
    Tests are skipped if CUDA is not available.
"""

import gc
import sys
import unittest
from dataclasses import dataclass
from typing import Optional, List, Any


# Check for PyTorch CUDA availability
HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass


@dataclass
class GPUChaosConfig:
    """Configuration for GPU chaos injection."""

    allocation_fraction: float = 0.9
    max_allocation_attempts: int = 100
    cleanup_after_oom: bool = True
    test_cpu_fallback: bool = True

    def __post_init__(self):
        if not 0.0 < self.allocation_fraction <= 1.0:
            raise ValueError("allocation_fraction must be between 0 and 1")
        if self.max_allocation_attempts <= 0:
            raise ValueError("max_allocation_attempts must be positive")


class GPUMemoryPressureInjector:
    """
    Injects GPU memory pressure for chaos testing.

    Allocates GPU memory until OOM to test recovery.
    """

    def __init__(self, config: Optional[GPUChaosConfig] = None):
        self.config = config or GPUChaosConfig()
        self._allocations: List[Any] = []
        self._oom_triggered: bool = False
        self._peak_memory_mb: float = 0.0
        self._device: Optional[Any] = None

    @property
    def cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return HAS_CUDA

    @property
    def stats(self) -> dict:
        """Get injection statistics."""
        return {
            "cuda_available": self.cuda_available,
            "oom_triggered": self._oom_triggered,
            "peak_memory_mb": self._peak_memory_mb,
            "active_allocations": len(self._allocations),
        }

    def get_gpu_memory_info(self) -> dict:
        """
        Get current GPU memory information.

        Returns:
            Dictionary with total, allocated, and free memory in MB
        """
        if not self.cuda_available:
            return {
                "total_mb": 0,
                "allocated_mb": 0,
                "free_mb": 0,
            }

        import torch

        total = torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(0)
        free = total - allocated

        return {
            "total_mb": total / (1024 * 1024),
            "allocated_mb": allocated / (1024 * 1024),
            "free_mb": free / (1024 * 1024),
        }

    def allocate_gpu_memory(self, size_mb: float) -> bool:
        """
        Allocate GPU memory.

        Args:
            size_mb: Size in MB to allocate

        Returns:
            True if allocation succeeded, False otherwise
        """
        if not self.cuda_available:
            return False

        import torch

        try:
            size_elements = int(size_mb * 1024 * 1024 / 4)  # float32 = 4 bytes
            tensor = torch.zeros(size_elements, device="cuda", dtype=torch.float32)
            self._allocations.append(tensor)

            # Update peak memory
            allocated = torch.cuda.memory_allocated(0) / (1024 * 1024)
            self._peak_memory_mb = max(self._peak_memory_mb, allocated)

            return True

        except (RuntimeError, torch.cuda.OutOfMemoryError):
            self._oom_triggered = True
            return False

    def trigger_oom(self) -> dict:
        """
        Intentionally trigger GPU OOM.

        Allocates memory until failure.

        Returns:
            Statistics about the OOM event
        """
        if not self.cuda_available:
            return {
                "success": False,
                "reason": "CUDA not available",
            }

        import torch

        # Clear any existing allocations
        self.release_all()

        # Get total memory
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        target_mb = total_mb * self.config.allocation_fraction

        # Allocate in chunks
        chunk_mb = 100  # 100MB chunks
        allocated_mb = 0
        attempts = 0

        while attempts < self.config.max_allocation_attempts:
            if self.allocate_gpu_memory(chunk_mb):
                allocated_mb += chunk_mb
            else:
                break
            attempts += 1

        return {
            "success": self._oom_triggered,
            "peak_memory_mb": self._peak_memory_mb,
            "total_gpu_mb": total_mb,
            "attempts": attempts,
        }

    def release_all(self) -> int:
        """
        Release all GPU allocations.

        Returns:
            Number of allocations released
        """
        count = len(self._allocations)
        self._allocations.clear()

        if self.cuda_available and self.config.cleanup_after_oom:
            import torch

            torch.cuda.empty_cache()
            gc.collect()

        return count

    def test_cpu_fallback(self, operation: callable) -> dict:
        """
        Test that an operation falls back to CPU on GPU OOM.

        Args:
            operation: Callable that may use GPU

        Returns:
            Dictionary with fallback test results
        """
        if not self.cuda_available:
            return {
                "fallback_occurred": False,
                "reason": "CUDA not available",
            }

        # Fill GPU memory
        self.trigger_oom()

        try:
            # Try to run operation
            result = operation()

            # If no error, check if it's on CPU
            if hasattr(result, "device"):
                import torch

                is_cpu = result.device.type == "cpu"
                return {
                    "fallback_occurred": is_cpu,
                    "device": str(result.device),
                }

            return {
                "fallback_occurred": True,
                "result_type": type(result).__name__,
            }

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {
                    "fallback_occurred": False,
                    "error": "OOM not handled - no fallback",
                }
            raise
        finally:
            self.release_all()


class TestGPUChaosConfig(unittest.TestCase):
    """Test GPU chaos configuration."""

    def test_valid_config(self):
        """Test valid configuration creation."""
        config = GPUChaosConfig(
            allocation_fraction=0.8,
            max_allocation_attempts=50,
        )
        self.assertEqual(config.allocation_fraction, 0.8)

    def test_invalid_config(self):
        """Test invalid configuration rejection."""
        with self.assertRaises(ValueError):
            GPUChaosConfig(allocation_fraction=0.0)

        with self.assertRaises(ValueError):
            GPUChaosConfig(allocation_fraction=1.5)

        with self.assertRaises(ValueError):
            GPUChaosConfig(max_allocation_attempts=0)


class TestGPUMemoryPressureInjector(unittest.TestCase):
    """Test GPU memory pressure injector."""

    def test_cuda_detection(self):
        """Test CUDA availability detection."""
        injector = GPUMemoryPressureInjector()

        # Should match module-level detection
        self.assertEqual(injector.cuda_available, HAS_CUDA)

    def test_stats_without_cuda(self):
        """Test stats work without CUDA."""
        injector = GPUMemoryPressureInjector()

        stats = injector.stats
        self.assertIn("cuda_available", stats)
        self.assertIn("oom_triggered", stats)
        self.assertIn("peak_memory_mb", stats)

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_gpu_memory_info(self):
        """Test GPU memory info retrieval."""
        injector = GPUMemoryPressureInjector()

        info = injector.get_gpu_memory_info()

        self.assertGreater(info["total_mb"], 0)
        self.assertGreaterEqual(info["allocated_mb"], 0)
        self.assertGreater(info["free_mb"], 0)

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_allocation_and_release(self):
        """Test GPU memory allocation and release."""
        injector = GPUMemoryPressureInjector()

        # Allocate 10MB
        success = injector.allocate_gpu_memory(10)
        self.assertTrue(success)

        stats = injector.stats
        self.assertEqual(stats["active_allocations"], 1)

        # Release
        released = injector.release_all()
        self.assertEqual(released, 1)

        stats = injector.stats
        self.assertEqual(stats["active_allocations"], 0)


class TestGPURecovery(unittest.TestCase):
    """Test GPU OOM recovery."""

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_oom_recovery(self):
        """Test recovery after OOM."""
        import torch

        config = GPUChaosConfig(
            allocation_fraction=0.95,
            cleanup_after_oom=True,
        )
        injector = GPUMemoryPressureInjector(config)

        # Get baseline
        baseline = injector.get_gpu_memory_info()

        # Trigger OOM
        result = injector.trigger_oom()

        # Release and recover
        injector.release_all()
        torch.cuda.empty_cache()
        gc.collect()

        # Memory should be recovered
        after = injector.get_gpu_memory_info()

        # Should be close to baseline (within 100MB)
        delta = after["allocated_mb"] - baseline["allocated_mb"]
        self.assertLess(delta, 100, f"GPU memory not recovered: +{delta}MB")

    @unittest.skipUnless(HAS_CUDA, "CUDA not available")
    def test_operation_after_oom(self):
        """Test that GPU operations work after OOM recovery."""
        import torch

        injector = GPUMemoryPressureInjector()

        # Trigger OOM
        injector.trigger_oom()

        # Release
        injector.release_all()
        torch.cuda.empty_cache()

        # Should be able to allocate again
        try:
            tensor = torch.zeros(1000, device="cuda")
            self.assertEqual(tensor.device.type, "cuda")
            del tensor
        except RuntimeError:
            self.fail("Could not allocate after OOM recovery")


def run_gpu_chaos_tests():
    """Run all GPU chaos tests."""
    print("=" * 60)
    print("  GPU CHAOS TESTS")
    print("=" * 60)
    print(f"  CUDA Available: {HAS_CUDA}")
    print("=" * 60)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestGPUChaosConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestGPUMemoryPressureInjector))
    suite.addTests(loader.loadTestsFromTestCase(TestGPURecovery))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_gpu_chaos_tests())
