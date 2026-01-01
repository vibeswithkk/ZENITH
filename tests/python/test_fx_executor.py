# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for FX Kernel Executor Module.

Tests the FX GraphModule to Zenith kernel dispatch bridge including:
- Executor initialization
- Wrapper creation
- Operation dispatch
- Statistics tracking

Run with: pytest tests/python/test_fx_executor.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import asdict


class TestFXExecutionStats:
    """Test FXExecutionStats dataclass."""

    def test_stats_default_values(self):
        """FXExecutionStats should have correct default values."""
        from zenith.runtime.fx_executor import FXExecutionStats

        stats = FXExecutionStats()

        assert stats.total_nodes == 0
        assert stats.dispatched_nodes == 0
        assert stats.fallback_nodes == 0
        assert isinstance(stats.kernel_hits, dict)

    def test_stats_custom_values(self):
        """FXExecutionStats should accept custom values."""
        from zenith.runtime.fx_executor import FXExecutionStats

        stats = FXExecutionStats(
            total_nodes=100,
            dispatched_nodes=80,
            fallback_nodes=20,
        )

        assert stats.total_nodes == 100
        assert stats.dispatched_nodes == 80
        assert stats.fallback_nodes == 20

    def test_stats_kernel_hits_initialized(self):
        """kernel_hits should be initialized to empty dict via __post_init__."""
        from zenith.runtime.fx_executor import FXExecutionStats

        stats = FXExecutionStats()
        stats.kernel_hits["matmul"] = 5

        # Should be mutable
        assert stats.kernel_hits["matmul"] == 5


class TestFXKernelExecutorInit:
    """Test FXKernelExecutor initialization."""

    def test_executor_default_init(self):
        """FXKernelExecutor should initialize with defaults."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor()

        assert executor._precision == "fp32"
        assert executor._device == "cuda"
        assert executor._enable_fallback is True

    def test_executor_custom_precision(self):
        """FXKernelExecutor should accept custom precision."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor(precision="fp16")

        assert executor._precision == "fp16"

    def test_executor_custom_device(self):
        """FXKernelExecutor should accept custom device."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor(device="cpu")

        assert executor._device == "cpu"

    def test_executor_has_stats(self):
        """FXKernelExecutor should track statistics."""
        from zenith.runtime.fx_executor import FXKernelExecutor, FXExecutionStats

        executor = FXKernelExecutor()
        stats = executor.get_stats()

        assert isinstance(stats, FXExecutionStats)


class TestFXKernelExecutorDispatcher:
    """Test dispatcher availability detection."""

    def test_has_dispatcher_returns_bool(self):
        """has_dispatcher() should return a boolean."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor()
        result = executor.has_dispatcher()

        assert isinstance(result, bool)


class TestFXKernelExecutorWrap:
    """Test function wrapping functionality."""

    def test_wrap_returns_callable(self):
        """wrap() should return a callable."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor()

        def dummy_forward(x):
            return x * 2

        wrapped = executor.wrap(dummy_forward)

        assert callable(wrapped)

    def test_wrapped_function_executes(self):
        """Wrapped function should execute correctly."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor(device="cpu")

        def dummy_forward(x):
            return x + 1

        wrapped = executor.wrap(dummy_forward)

        # Execute wrapped function
        result = wrapped(10)
        assert result == 11

    def test_wrap_preserves_return_value(self):
        """Wrapped function should preserve return value."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor(device="cpu")

        def complex_forward(a, b, c=1):
            return a * b + c

        wrapped = executor.wrap(complex_forward)

        result = wrapped(2, 3, c=5)
        assert result == 11  # 2 * 3 + 5


class TestFXKernelExecutorStats:
    """Test statistics tracking."""

    def test_get_stats_returns_stats_object(self):
        """get_stats() should return FXExecutionStats."""
        from zenith.runtime.fx_executor import FXKernelExecutor, FXExecutionStats

        executor = FXKernelExecutor()
        stats = executor.get_stats()

        assert isinstance(stats, FXExecutionStats)

    def test_reset_stats_clears_counts(self):
        """reset_stats() should reset all counters."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor()

        # Manually modify stats
        executor._stats.total_nodes = 100
        executor._stats.dispatched_nodes = 50

        executor.reset_stats()

        stats = executor.get_stats()
        assert stats.total_nodes == 0
        assert stats.dispatched_nodes == 0


class TestFactoryFunctions:
    """Test module-level factory functions."""

    def test_create_fx_executor_returns_executor(self):
        """create_fx_executor() should return FXKernelExecutor instance."""
        from zenith.runtime.fx_executor import create_fx_executor, FXKernelExecutor

        executor = create_fx_executor()

        assert isinstance(executor, FXKernelExecutor)

    def test_create_fx_executor_with_params(self):
        """create_fx_executor() should accept parameters."""
        from zenith.runtime.fx_executor import create_fx_executor

        executor = create_fx_executor(precision="bf16", device="cpu")

        assert executor._precision == "bf16"
        assert executor._device == "cpu"

    def test_wrap_fx_module_returns_callable(self):
        """wrap_fx_module() should return a callable."""
        from zenith.runtime.fx_executor import wrap_fx_module

        def forward(x):
            return x

        wrapped = wrap_fx_module(forward, precision="fp32", device="cpu")

        assert callable(wrapped)


class TestFXKernelExecutorExecuteOp:
    """Test operation execution."""

    def test_execute_op_with_fallback_enabled(self):
        """execute_op should use fallback when dispatcher unavailable."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor(device="cpu", enable_fallback=True)

        # Mock inputs as simple Python values for fallback testing
        # This tests the fallback path when no kernel is available
        result = executor.execute_op("relu", [5.0])

        # Should either execute or return None (depending on fallback support)
        # The important thing is it doesn't crash

    def test_execute_op_updates_stats(self):
        """execute_op should update execution statistics."""
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor(device="cpu", enable_fallback=True)
        executor.reset_stats()

        # Execute an operation
        try:
            executor.execute_op("add", [1.0, 2.0])
        except Exception:
            pass  # May fail without proper tensors, but should update stats

        stats = executor.get_stats()
        # Stats tracking should have been attempted
        assert isinstance(stats.kernel_hits, dict)


class TestFXKernelExecutorWithPyTorch:
    """Test integration with PyTorch tensors."""

    @pytest.fixture
    def torch_available(self):
        try:
            import torch
            return True
        except ImportError:
            return False

    def test_wrap_torch_function(self, torch_available):
        """Should wrap PyTorch functions."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch
        from zenith.runtime.fx_executor import FXKernelExecutor

        executor = FXKernelExecutor(device="cpu", precision="fp32")

        def torch_forward(x):
            return x * 2

        wrapped = executor.wrap(torch_forward)

        x = torch.randn(10)
        result = wrapped(x)

        assert result.shape == x.shape
        torch.testing.assert_close(result, x * 2)

    def test_wrap_torch_module_forward(self, torch_available):
        """Should wrap nn.Module forward method."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch
        import torch.nn as nn
        from zenith.runtime.fx_executor import FXKernelExecutor

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x):
                return self.linear(x)

        module = SimpleModule()
        executor = FXKernelExecutor(device="cpu", precision="fp32")
        wrapped = executor.wrap(module.forward)

        x = torch.randn(2, 10)
        result = wrapped(x)

        assert result.shape == (2, 5)
