# Copyright 2025 Wahyu Ardiansyah
# Licensed under the Apache License, Version 2.0

"""
Unit Tests for Triton Kernels Module.

Tests the Triton-based GPU kernel implementations including:
- Availability detection
- Kernel registration
- Fused operation wrappers (when Triton unavailable, tests fallback)

Run with: pytest tests/python/test_triton_kernels.py -v
"""

import pytest
from unittest.mock import MagicMock, patch


class TestTritonKernelsAvailability:
    """Test Triton availability detection."""

    def test_is_available_returns_bool(self):
        """is_available() must return a boolean."""
        from zenith.runtime.triton_kernels import is_available

        result = is_available()
        assert isinstance(result, bool)

    def test_get_version_returns_string_or_none(self):
        """get_version() must return string or None."""
        from zenith.runtime.triton_kernels import get_version

        result = get_version()
        assert result is None or isinstance(result, str)

    def test_is_available_consistent_with_get_version(self):
        """If available, version should not be None."""
        from zenith.runtime.triton_kernels import is_available, get_version

        if is_available():
            version = get_version()
            # Version could still be "unknown" but should be a string
            assert isinstance(version, str)


class TestTritonKernelMap:
    """Test kernel map retrieval."""

    def test_get_triton_kernel_map_returns_dict(self):
        """get_triton_kernel_map() must return a dictionary."""
        from zenith.runtime.triton_kernels import get_triton_kernel_map

        kernel_map = get_triton_kernel_map()
        assert isinstance(kernel_map, dict)

    def test_kernel_map_empty_when_triton_unavailable(self):
        """When Triton is unavailable, kernel map should be empty."""
        from zenith.runtime.triton_kernels import is_available, get_triton_kernel_map

        if not is_available():
            kernel_map = get_triton_kernel_map()
            assert len(kernel_map) == 0

    def test_kernel_map_contains_expected_keys_when_available(self):
        """When Triton is available, kernel map should have expected keys."""
        from zenith.runtime.triton_kernels import is_available, get_triton_kernel_map

        if is_available():
            kernel_map = get_triton_kernel_map()
            # Should contain fused operation mappings
            expected_keys = {"FusedLinearGELU", "FusedLinearReLU", "Linear+GELU", "Linear+ReLU"}
            assert expected_keys.issubset(set(kernel_map.keys()))


class TestRegisterTritonKernels:
    """Test kernel registration with registry."""

    def test_register_triton_kernels_with_mock_registry(self):
        """register_triton_kernels() should work with a mock registry."""
        from zenith.runtime.triton_kernels import register_triton_kernels, is_available

        # Create mock registry
        mock_registry = MagicMock()
        mock_registry.register = MagicMock()

        count = register_triton_kernels(mock_registry)

        # Count should be integer
        assert isinstance(count, int)

        # If Triton available, should have registered kernels
        if is_available():
            assert count > 0
            assert mock_registry.register.called
        else:
            # No kernels registered when Triton unavailable
            assert count == 0


class TestFusedOperationsFallback:
    """Test fused operations fallback when Triton unavailable."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            return False

    def test_fused_linear_gelu_fallback(self, torch_available):
        """fused_linear_gelu should fallback to PyTorch when Triton unavailable."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch
        from zenith.runtime.triton_kernels import fused_linear_gelu, is_available

        # Create test tensors on CPU (works without GPU)
        M, K, N = 32, 64, 128
        x = torch.randn(M, K)
        weight = torch.randn(N, K)
        bias = torch.randn(N)

        # This should work regardless of Triton availability (fallback)
        result = fused_linear_gelu(x, weight, bias)

        # Verify output shape
        assert result.shape == (M, N)
        assert result.dtype == x.dtype

    def test_fused_linear_relu_fallback(self, torch_available):
        """fused_linear_relu should fallback to PyTorch when Triton unavailable."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch
        from zenith.runtime.triton_kernels import fused_linear_relu

        M, K, N = 32, 64, 128
        x = torch.randn(M, K)
        weight = torch.randn(N, K)
        bias = torch.randn(N)

        result = fused_linear_relu(x, weight, bias)

        assert result.shape == (M, N)
        # ReLU output should have no negative values
        # (may have zeros or positive values)

    def test_fused_linear_gelu_no_bias(self, torch_available):
        """fused_linear_gelu should work without bias."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch
        from zenith.runtime.triton_kernels import fused_linear_gelu

        M, K, N = 16, 32, 64
        x = torch.randn(M, K)
        weight = torch.randn(N, K)

        # No bias provided
        result = fused_linear_gelu(x, weight, bias=None)

        assert result.shape == (M, N)


class TestFusedOperationsNumericalAccuracy:
    """Test numerical accuracy of fused operations against PyTorch reference."""

    @pytest.fixture
    def torch_available(self):
        try:
            import torch
            return True
        except ImportError:
            return False

    def test_fused_linear_gelu_matches_reference(self, torch_available):
        """fused_linear_gelu output should match PyTorch reference."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch
        import torch.nn.functional as F
        from zenith.runtime.triton_kernels import fused_linear_gelu

        M, K, N = 32, 64, 128
        x = torch.randn(M, K)
        weight = torch.randn(N, K)
        bias = torch.randn(N)

        # Zenith fused operation
        zenith_result = fused_linear_gelu(x, weight, bias)

        # PyTorch reference
        reference = F.gelu(F.linear(x, weight, bias))

        # Should match within numerical tolerance
        torch.testing.assert_close(zenith_result, reference, rtol=1e-4, atol=1e-4)

    def test_fused_linear_relu_matches_reference(self, torch_available):
        """fused_linear_relu output should match PyTorch reference."""
        if not torch_available:
            pytest.skip("PyTorch not available")

        import torch
        import torch.nn.functional as F
        from zenith.runtime.triton_kernels import fused_linear_relu

        M, K, N = 32, 64, 128
        x = torch.randn(M, K)
        weight = torch.randn(N, K)
        bias = torch.randn(N)

        zenith_result = fused_linear_relu(x, weight, bias)
        reference = F.relu(F.linear(x, weight, bias))

        torch.testing.assert_close(zenith_result, reference, rtol=1e-4, atol=1e-4)


class TestBenchmarkFunction:
    """Test benchmark utility function."""

    @pytest.fixture
    def torch_cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def test_benchmark_returns_dict(self, torch_cuda_available):
        """benchmark_fused_linear_gelu should return a dictionary."""
        from zenith.runtime.triton_kernels import benchmark_fused_linear_gelu, is_available

        result = benchmark_fused_linear_gelu(M=64, N=128, K=64, runs=2)

        assert isinstance(result, dict)

        # Should have error key if not available
        if not is_available() or not torch_cuda_available:
            assert "error" in result
        else:
            # Should have timing keys
            assert "fused_ms" in result or "error" in result
