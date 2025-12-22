# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for cuDNN Integration.

Tests the kernel registry's cuDNN kernel registration and priority system.
"""

import pytest


class TestCuDNNKernelRegistration:
    """Test cuDNN kernel registration and priority selection."""

    def test_registry_initializes_without_error(self):
        """Registry should initialize without cuDNN errors."""
        from zenith.runtime.kernel_registry import KernelRegistry

        registry = KernelRegistry()
        result = registry.initialize()
        assert isinstance(result, bool)

    def test_kernel_priority_ordering(self):
        """cuDNN kernels should have higher priority than custom CUDA."""
        from zenith.runtime.kernel_registry import KernelRegistry, KernelSpec, Precision

        registry = KernelRegistry()

        custom_kernel = KernelSpec(
            name="custom_conv2d",
            op_types=["Conv2d"],
            precision=Precision.FP32,
            kernel_fn=lambda x: x,
            priority=10,
        )

        cudnn_kernel = KernelSpec(
            name="cudnn_conv2d",
            op_types=["Conv2d"],
            precision=Precision.FP32,
            kernel_fn=lambda x: x,
            priority=20,
        )

        registry.register(custom_kernel)
        registry.register(cudnn_kernel)

        selected = registry.get_kernel("Conv2d", Precision.FP32)

        assert selected is not None
        assert selected.name == "cudnn_conv2d"
        assert selected.priority == 20

    def test_fallback_when_cudnn_not_available(self):
        """Should fallback to custom CUDA kernel when cuDNN unavailable."""
        from zenith.runtime.kernel_registry import KernelRegistry, KernelSpec, Precision

        registry = KernelRegistry()

        custom_kernel = KernelSpec(
            name="custom_conv2d",
            op_types=["Conv2d"],
            precision=Precision.FP32,
            kernel_fn=lambda x: x,
            priority=10,
        )

        registry.register(custom_kernel)

        selected = registry.get_kernel("Conv2d", Precision.FP32)

        assert selected is not None
        assert selected.name == "custom_conv2d"

    def test_supported_ops_listing(self):
        """Should list all supported operations correctly."""
        from zenith.runtime.kernel_registry import KernelRegistry, KernelSpec, Precision

        registry = KernelRegistry()

        registry.register(
            KernelSpec(
                name="test_conv",
                op_types=["Conv2d", "Convolution"],
                precision=Precision.FP32,
                kernel_fn=lambda x: x,
                priority=10,
            )
        )

        ops = registry.list_supported_ops()
        assert "Conv2d" in ops
        assert "Convolution" in ops

    def test_precision_fallback(self):
        """Should fallback to FP32 when FP16 not available."""
        from zenith.runtime.kernel_registry import KernelRegistry, KernelSpec, Precision

        registry = KernelRegistry()

        fp32_kernel = KernelSpec(
            name="conv2d_fp32",
            op_types=["Conv2d"],
            precision=Precision.FP32,
            kernel_fn=lambda x: x,
            priority=10,
        )

        registry.register(fp32_kernel)

        selected = registry.get_kernel("Conv2d", Precision.FP16)

        assert selected is not None
        assert selected.precision == Precision.FP32


class TestKernelDispatcher:
    """Test kernel dispatcher with cuDNN integration."""

    def test_dispatcher_initializes(self):
        """Dispatcher should initialize without errors."""
        from zenith.runtime.dispatcher import KernelDispatcher
        from zenith.runtime.kernel_registry import Precision

        dispatcher = KernelDispatcher(precision=Precision.FP32)
        assert dispatcher is not None

    def test_dispatch_stats_tracking(self):
        """Dispatcher should track dispatch statistics."""
        from zenith.runtime.dispatcher import KernelDispatcher
        from zenith.runtime.kernel_registry import Precision

        dispatcher = KernelDispatcher(precision=Precision.FP32)
        stats = dispatcher.get_dispatch_stats()

        assert "total_dispatches" in stats
        assert stats["total_dispatches"] == 0

    def test_reset_stats(self):
        """Dispatcher stats should reset correctly."""
        from zenith.runtime.dispatcher import KernelDispatcher
        from zenith.runtime.kernel_registry import Precision

        dispatcher = KernelDispatcher(precision=Precision.FP32)
        dispatcher._dispatch_count = 10
        dispatcher.reset_stats()

        assert dispatcher._dispatch_count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
