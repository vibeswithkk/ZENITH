"""
Tests for ROCm Backend

Tests the ROCm/HIP backend functionality when available.
All tests are skipped if ROCm is not available.

Copyright 2025 Wahyu Ardiansyah
Licensed under the Apache License, Version 2.0
"""

import pytest
import numpy as np

# Try to import zenith core
try:
    from zenith._zenith_core import backends, rocm

    ZENITH_CORE_AVAILABLE = True
except ImportError:
    ZENITH_CORE_AVAILABLE = False


# Skip all tests if zenith._zenith_core is not available
pytestmark = pytest.mark.skipif(
    not ZENITH_CORE_AVAILABLE, reason="zenith._zenith_core not available (needs build)"
)


class TestROCmBackendAvailability:
    """Test ROCm backend availability checks."""

    def test_rocm_module_exists(self):
        """Test that rocm module exists in _zenith_core."""
        assert hasattr(rocm, "is_available")
        assert hasattr(rocm, "get_version")

    def test_is_rocm_available_returns_bool(self):
        """Test that is_rocm_available returns boolean."""
        result = backends.is_rocm_available()
        assert isinstance(result, bool)

    def test_rocm_is_available_returns_bool(self):
        """Test that rocm.is_available returns boolean."""
        result = rocm.is_available()
        assert isinstance(result, bool)

    def test_rocm_version_returns_int(self):
        """Test that get_version returns integer."""
        result = rocm.get_version()
        assert isinstance(result, int)
        assert result >= 0

    def test_miopen_availability_check(self):
        """Test MIOpen availability check."""
        result = rocm.is_miopen_available()
        assert isinstance(result, bool)

    def test_miopen_version_returns_int(self):
        """Test MIOpen version returns integer."""
        result = rocm.get_miopen_version()
        assert isinstance(result, int)
        assert result >= 0


class TestROCmBackendInRegistry:
    """Test ROCm backend registration in backends module."""

    def test_list_available_includes_cpu(self):
        """Test that CPU is always in available backends."""
        available = backends.list_available()
        assert "cpu" in available

    def test_list_available_returns_list(self):
        """Test that list_available returns list."""
        available = backends.list_available()
        assert isinstance(available, list)

    @pytest.mark.skipif(
        ZENITH_CORE_AVAILABLE and not backends.is_rocm_available(),
        reason="ROCm not available on this system",
    )
    def test_rocm_in_available_when_present(self):
        """Test that ROCm is in list when available."""
        available = backends.list_available()
        assert "rocm" in available


@pytest.mark.skipif(
    not ZENITH_CORE_AVAILABLE or not backends.is_rocm_available(),
    reason="ROCm not available",
)
class TestROCmMatMul:
    """Test ROCm matrix multiplication operations."""

    def test_matmul_basic(self):
        """Test basic matmul on ROCm."""
        A = np.array([[1, 2], [3, 4]], dtype=np.float32)
        B = np.array([[5, 6], [7, 8]], dtype=np.float32)

        result = rocm.matmul(A, B)
        expected = np.matmul(A, B)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_matmul_larger(self):
        """Test larger matmul on ROCm."""
        M, K, N = 64, 32, 48
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        result = rocm.matmul(A, B)
        expected = np.matmul(A, B)

        np.testing.assert_allclose(result, expected, rtol=1e-4, atol=1e-5)

    def test_matmul_dimension_mismatch_raises(self):
        """Test that dimension mismatch raises error."""
        A = np.random.randn(10, 20).astype(np.float32)
        B = np.random.randn(30, 40).astype(np.float32)

        with pytest.raises(RuntimeError):
            rocm.matmul(A, B)

    def test_matmul_non_2d_raises(self):
        """Test that non-2D arrays raise error."""
        A = np.random.randn(10).astype(np.float32)
        B = np.random.randn(10, 20).astype(np.float32)

        with pytest.raises(RuntimeError):
            rocm.matmul(A, B)


@pytest.mark.skipif(
    not ZENITH_CORE_AVAILABLE or not backends.is_rocm_available(),
    reason="ROCm not available",
)
class TestROCmDeviceSync:
    """Test ROCm device synchronization."""

    def test_sync_callable(self):
        """Test that sync is callable without error."""
        # Should not raise
        rocm.sync()


class TestROCmGracefulDegradation:
    """Test graceful degradation when ROCm is not available."""

    def test_rocm_module_always_exists(self):
        """Test rocm module exists even without ROCm hardware."""
        # The rocm module should always be importable
        # Functions should return safe defaults
        assert rocm.is_available() in (True, False)
        assert isinstance(rocm.get_version(), int)

    def test_no_crash_without_rocm(self):
        """Test no crashes when calling ROCm functions without hardware."""
        # These should all work without crashing
        _ = rocm.is_available()
        _ = rocm.get_version()
        _ = rocm.is_miopen_available()
        _ = rocm.get_miopen_version()
