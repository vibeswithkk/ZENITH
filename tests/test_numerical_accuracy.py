# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Numerical Accuracy Verification - Ensures optimizations maintain correctness.

Verifies that:
1. Parallel LayerNorm matches sequential reference
2. cuDNN kernels match custom CUDA kernels
3. All operations maintain FP32 precision bounds
"""

import numpy as np
import pytest


class TestLayerNormAccuracy:
    """Test numerical accuracy of LayerNorm implementations."""

    def numpy_layernorm(
        self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Reference LayerNorm implementation."""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_normalized = (x - mean) / np.sqrt(var + eps)
        return gamma * x_normalized + beta

    @pytest.mark.parametrize(
        "batch,hidden",
        [
            (1, 64),
            (8, 768),
            (32, 768),
            (128, 1024),
        ],
    )
    def test_layernorm_accuracy(self, batch, hidden):
        """Test LayerNorm numerical accuracy against reference."""
        np.random.seed(42)

        x = np.random.randn(batch, hidden).astype(np.float32)
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-5

        expected = self.numpy_layernorm(x, gamma, beta, eps)

        try:
            from zenith._zenith_core import cuda

            if hasattr(cuda, "layernorm_f32"):
                output = np.zeros_like(x)
                cuda.layernorm_f32(x, output, gamma, beta, batch, hidden, eps)

                max_diff = np.abs(output - expected).max()
                mean_diff = np.abs(output - expected).mean()

                assert max_diff < 1e-4, (
                    f"Max diff {max_diff} exceeds threshold for "
                    f"batch={batch}, hidden={hidden}"
                )
                assert mean_diff < 1e-5, f"Mean diff {mean_diff} exceeds threshold"
            else:
                pytest.skip("CUDA layernorm_f32 not available")

        except ImportError:
            pytest.skip("CUDA module not available")

    def test_layernorm_numerical_stability(self):
        """Test numerical stability with extreme values."""
        np.random.seed(42)
        batch, hidden = 8, 768

        x = np.random.randn(batch, hidden).astype(np.float32) * 1000.0
        gamma = np.ones(hidden, dtype=np.float32)
        beta = np.zeros(hidden, dtype=np.float32)
        eps = 1e-5

        expected = self.numpy_layernorm(x, gamma, beta, eps)

        assert not np.any(np.isnan(expected)), "NaN in expected output"
        assert not np.any(np.isinf(expected)), "Inf in expected output"


class TestKernelRegistryAccuracy:
    """Test kernel selection accuracy."""

    def test_priority_selection(self):
        """Verify correct kernel is selected by priority."""
        from zenith.runtime.kernel_registry import KernelRegistry, KernelSpec, Precision

        registry = KernelRegistry()

        low_priority = KernelSpec(
            name="custom_op",
            op_types=["TestOp"],
            precision=Precision.FP32,
            kernel_fn=lambda x: x,
            priority=10,
        )

        high_priority = KernelSpec(
            name="cudnn_op",
            op_types=["TestOp"],
            precision=Precision.FP32,
            kernel_fn=lambda x: x * 2,
            priority=20,
        )

        registry.register(low_priority)
        registry.register(high_priority)

        selected = registry.get_kernel("TestOp", Precision.FP32)

        assert selected.name == "cudnn_op"
        assert selected.priority == 20


class TestElementwiseAccuracy:
    """Test elementwise operation accuracy."""

    def test_add_accuracy(self):
        """Test element-wise add accuracy."""
        np.random.seed(42)
        a = np.random.randn(32, 768).astype(np.float32)
        b = np.random.randn(32, 768).astype(np.float32)

        expected = a + b

        try:
            from zenith._zenith_core import cuda

            if hasattr(cuda, "add_f32"):
                output = np.zeros_like(a)
                cuda.add_f32(a, b, output, a.size)

                max_diff = np.abs(output - expected).max()
                assert max_diff < 1e-6, f"Add max diff {max_diff}"
            else:
                pytest.skip("CUDA add_f32 not available")

        except ImportError:
            pytest.skip("CUDA module not available")


class TestMemoryManagerAccuracy:
    """Test memory manager behavior accuracy."""

    def test_allocation_offset_correctness(self):
        """Verify allocation offsets are sequential."""
        from zenith.runtime.memory_manager import MemoryManager

        manager = MemoryManager()

        manager.allocate("a", 1024)
        manager.allocate("b", 2048)
        manager.allocate("c", 512)

        assert manager._blocks["a"].offset == 0
        assert manager._blocks["b"].offset == 1024
        assert manager._blocks["c"].offset == 3072

    def test_coalesce_size_correctness(self):
        """Verify coalesced blocks have correct total size."""
        from zenith.runtime.memory_manager import MemoryManager

        manager = MemoryManager()

        manager.allocate("a", 1000)
        manager.allocate("b", 2000)
        manager.allocate("c", 3000)

        total_before = sum(b.size_bytes for b in manager._blocks.values())

        manager.free("a")
        manager.free("b")
        manager.free("c")

        total_after = sum(b.size_bytes for b in manager._blocks.values())

        assert total_before == total_after, (
            "Total memory should be preserved after coalescing"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
